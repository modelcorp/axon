# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Manages distributed Lean 4 verification via Kimina Docker servers.

Spawns one Kimina lean-server Docker container per Ray node and provides
an async client interface for proof checking.

Prerequisites:
    - Docker installed and accessible
    - Docker network 'formal_math' created: docker network create formal_math
    - pip install kimina-client
"""

import datetime
import logging
import os
import random
import socket
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("AXON_KILL_PREVIOUS_KIMINA_DOCKER", "1")))
_KIMINA_IMAGE = os.environ.get("AXON_KIMINA_IMAGE", "projectnumina/kimina-lean-server:2.0.0")
_KIMINA_NETWORK = os.environ.get("AXON_KIMINA_NETWORK", "formal_math")


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _exec_command(cmd: str):
    logger.info(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False, capture_output=True)


class KiminaServerAndClientCluster:
    """Manages Kimina verification servers across Ray nodes.

    When running inside Docker on the 'formal_math' network, set
    AXON_KIMINA_USE_DOCKER_HOSTNAME=1 to connect via container names.
    Otherwise (default), connects via localhost with mapped ports.
    """

    def __init__(self):
        self._use_docker_hostname = bool(int(os.environ.get("AXON_KIMINA_USE_DOCKER_HOSTNAME", "0")))

        try:
            import ray

            self._servers = _create_ray_actors()
            from kimina_client import AsyncKiminaClient

            # Get (docker_name, host_port) from each actor and pick the right URL
            server_infos = [ray.get(server.get_connection_info.remote()) for server in self._servers]
            urls = []
            for docker_name, host_port in server_infos:
                if self._use_docker_hostname:
                    url = f"http://{docker_name}:8000"
                else:
                    url = f"http://localhost:{host_port}"
                urls.append(url)

            # Wait for all servers to be reachable from this process
            for url in urls:
                _wait_server_ready(url)

            self._clients = [AsyncKiminaClient(api_url=url) for url in urls]
        except ImportError:
            # Fallback: single local server when Ray is not available
            logger.info("Ray not available, starting single local Kimina server")
            self._servers = []
            port = _get_free_port()
            if _KILL_PREVIOUS_KIMINA_DOCKER:
                _docker_stop_all()
            self._docker_name = _docker_start(port=port)
            api_url = f"http://localhost:{port}"
            _wait_server_ready(api_url)
            from kimina_client import AsyncKiminaClient

            self._clients = [AsyncKiminaClient(api_url=api_url)]

        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_ray_actors() -> list:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    assert len(nodes) > 0, "No alive Ray nodes found"

    actors = []
    for node in nodes:
        actors.append(
            _KiminaServerActor.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                num_cpus=0.001,
            ).remote()
        )
    return actors


try:
    import ray

    @ray.remote
    class _KiminaServerActor:
        def __init__(self):
            self.port = _get_free_port()
            if _KILL_PREVIOUS_KIMINA_DOCKER:
                _docker_stop_all()
            self.docker_name = _docker_start(port=self.port)
            # Verify the server is ready from within this node (via localhost)
            _wait_server_ready(f"http://localhost:{self.port}")

        def get_connection_info(self) -> tuple[str, int]:
            """Return (docker_container_name, host_port) for the client to choose."""
            return self.docker_name, self.port

except ImportError:
    pass


def _docker_start(port: int) -> str:
    docker_name = (
        f"kimina_lean_server_auto_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    _exec_command(
        f"docker run -d "
        f"--name {docker_name} "
        f"--restart unless-stopped "
        f"--network {_KIMINA_NETWORK} "
        f"-p {port}:8000 "
        f"{_KIMINA_IMAGE}"
    )
    return docker_name


def _wait_server_ready(base_url: str, timeout: int = 300):
    start = time.time()
    with requests.Session() as session:
        while time.time() - start < timeout:
            try:
                response = session.get(f"{base_url}/health")
                if response.status_code == 200:
                    logger.info(f"Kimina server ready at {base_url}")
                    return
            except requests.RequestException:
                pass
            logger.info(f"Waiting for Kimina server at {base_url}...")
            time.sleep(2)
    raise TimeoutError(f"Kimina server at {base_url} did not start within {timeout}s")


def _docker_stop_all():
    _exec_command(
        'ids=$(docker ps -a --filter "name=kimina_lean_server_auto" -q); '
        '[ -n "$ids" ] && docker stop $ids && docker rm $ids; '
        "true"
    )
