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
import asyncio
import ipaddress
import logging
import os
import signal
import socket
import subprocess
import time

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def ensure_port_available(port: int, force: bool = False) -> None:
    """Check if port is in use. Optionally kill the holder."""

    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    pids = result.stdout.strip().split("\n")
    pids = [p for p in pids if p]

    if not pids:
        return

    if force:
        logger.warning(
            "Port %d held by PID(s) %s — killing stale process(es)",
            port,
            pids,
        )
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)  # wait for OS to release
    else:
        raise RuntimeError(f"Port {port} already in use by PID(s) {pids}. Run: kill -9 {' '.join(pids)}")


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
