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

import logging

import ray


def kill_ray_actors():
    """List all actors and kill those that are alive or pending."""
    actor_infos = ray._private.state.actors()
    for actor_id, info in actor_infos.items():
        if info.get("State") in ("ALIVE", "PENDING_CREATION"):
            try:
                name = info.get("Name")
                actor = ray.get_actor(name=name) if name else ray.get_actor(actor_id=actor_id)
                ray.kill(actor, no_restart=True)
                logging.info(f"Killed actor: {name or actor_id}")
            except Exception as e:
                logging.error(f"Failed to kill actor {actor_id}: {e}")
