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
import importlib
import multiprocessing as mp

import browsergym.miniwob
import gymnasium as gym

from axon.core import BaseEnv, register_env


@register_env("browsergym")
class BrowserGymEnv(BaseEnv):
    def __init__(self, env_id="browsergym/openended", task=None, max_turns=None, **env_kwargs):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=self._worker, args=(self.child_conn, env_id, task, env_kwargs))
        self.timeout = None  # in seconds
        self.max_turns = max_turns
        self._step_count = 0
        self.process.start()

    def _worker(self, conn, env_id, task, env_kwargs):
        # force re-execute registration code
        importlib.reload(browsergym.miniwob)

        env = (
            gym.make(
                env_id,
                task_kwargs=task,
                **env_kwargs,
                browser_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-application-cache",
                    "--disk-cache-size=1",
                    "--media-cache-size=1",
                    "--disable-cache",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--incognito",
                ],
                user_data_dir=None,  # Forces incognito
            )
            if task
            else gym.make(env_id, **env_kwargs)
        )
        try:
            while True:
                cmd, data = conn.recv()
                if cmd == "reset":
                    obs = env.reset()
                    conn.send(obs)
                elif cmd == "step":
                    action = data
                    obs, reward, terminated, truncated, extra_info = env.step(action)
                    conn.send((obs, reward, terminated or truncated, extra_info))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    break
        except EOFError:
            env.close()

    def reset(self):
        self._step_count = 0
        self.parent_conn.send(("reset", None))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        return self.parent_conn.recv()

    def step(self, action):
        self.parent_conn.send(("step", action))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        obs, reward, done, info = self.parent_conn.recv()
        self._step_count += 1
        if self.max_turns is not None and self._step_count >= self.max_turns:
            done = True
        return obs, reward, done, info

    def close(self):
        self.parent_conn.send(("close", None))
        self.process.join(60 * 2)
        if self.process.is_alive():
            print(f"Process still alive after {self.timeout} seconds. Killing it.")
            self.process.terminate()
            self.process.join()

    @staticmethod
    def from_dict(extra_info: dict) -> "BrowserGymEnv":
        headless = extra_info.get("headless", True)
        timeout_ms = extra_info.get("timeout", 5000)
        max_turns = extra_info.get("max_turns")
        return BrowserGymEnv(env_id=extra_info["env_id"], headless=headless, timeout=timeout_ms, max_turns=max_turns)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
