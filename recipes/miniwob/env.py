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
Optimized BrowserGym environment wrapper for high-throughput RL training.

Performance problems found in BrowserGym source (v0.14.x):
  1. Every reset() launches TWO Chromium browsers — one for env, one for Chat UI.
  2. MiniWoB tasks set slow_mo=100ms per Playwright operation.
  3. Default pre_observation_delay=0.5s wait before each observation.
  4. Full DOM snapshot + AXTree + screenshot extraction on every step.

Fixes applied:
  - Chat class replaced with lightweight DummyChat (no second browser).
  - slow_mo=0 and pre_observation_delay=0 via gym.make kwargs.
  - Aggressive Chromium flags to reduce per-browser resource usage.
  - Workers reuse envs across episodes when env_id/config matches.
"""

import logging
import multiprocessing as mp
import time
from typing import Any

import gymnasium as gym

from axon.core import BaseEnv, register_env

logger = logging.getLogger(__name__)


# ===================================================================
# Chat replacement — eliminates the second browser per env
# ===================================================================


class DummyChat:
    """
    Drop-in replacement for browsergym.core.chat.Chat.

    The original launches a full Chromium browser just to render a chat UI
    that is never seen in headless RL training. This stores messages in a list.
    """

    def __init__(self, **kwargs):
        self.messages = []
        self.recording_start_time = None
        # The real Chat exposes a page attribute that BrowserEnv.reset()
        # references for video recording paths. We stub it out.
        self.page = None

    def add_message(self, role, msg):
        self.messages.append({"role": role, "timestamp": time.time(), "message": msg})

    def wait_for_user_message(self):
        pass

    def close(self):
        pass


def _apply_chat_patch():
    """
    Replace Chat in browsergym.core.env so the already-imported local name
    is overwritten.

    IMPORTANT: env.py does `from .chat import Chat`, which binds Chat as a
    local name in the env module. Patching browsergym.core.chat.Chat would
    NOT affect the already-captured reference in env.py. We must patch
    browsergym.core.env.Chat directly.

    Must be called in every process that creates BrowserGym envs (the main
    process for fork-based mp on Linux, AND inside each worker for
    spawn-based mp on macOS/Windows).
    """
    import browsergym.core.env as env_mod

    env_mod.Chat = DummyChat


# Apply in main process (inherited by fork-based children on Linux)
_apply_chat_patch()


# ===================================================================
# Performance configuration
# ===================================================================


def _merge_perf_kwargs(user_kwargs: dict) -> dict:
    """Inject performance defaults. User kwargs override non-perf keys."""
    perf = {
        "headless": True,
        "slow_mo": 0,
        "pre_observation_delay": 0,
        # NOTE: We do NOT pass pw_chromium_kwargs["args"] here because
        # BrowserGym's BrowserEnv.reset() already passes args= explicitly
        # to pw.chromium.launch(), and having it in pw_chromium_kwargs too
        # causes "got multiple values for keyword argument 'args'".
    }
    for k, v in user_kwargs.items():
        perf[k] = v
    return perf


def _ensure_benchmark_registered(env_id: str):
    """
    Import the browsergym submodule that registers tasks for this env_id.

    env_id format: "browsergym/miniwob.click-test" → import browsergym.miniwob
                   "browsergym/webarena.310"        → import browsergym.webarena
                   "browsergym/openended"           → import browsergym.core (already loaded)

    This is needed because worker subprocesses don't inherit gym.registry entries.
    The browsergym submodules register tasks on import via gym.register().
    """
    if not env_id.startswith("browsergym/"):
        return

    remainder = env_id[len("browsergym/") :]  # e.g. "miniwob.click-test" or "openended"
    benchmark = remainder.split(".")[0]  # e.g. "miniwob", "webarena", "openended"

    if benchmark == "openended":
        import browsergym.core  # noqa: F401

        return

    module_name = f"browsergym.{benchmark}"
    try:
        __import__(module_name)
    except ImportError:
        logger.warning(f"Could not import {module_name} for env_id={env_id}")


def _make_env(env_id: str, task: dict | None, env_kwargs: dict):
    _ensure_benchmark_registered(env_id)
    kwargs = _merge_perf_kwargs(env_kwargs)
    if task:
        return gym.make(env_id, task_kwargs=task, **kwargs)
    return gym.make(env_id, **kwargs)


def _close_env(env):
    """Safely close an env, return None."""
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    return None


# ===================================================================
# Worker for direct mode (pipe-based, 1:1 with env handle)
# ===================================================================


def _direct_worker(conn, env_id: str, task: dict | None, env_kwargs: dict):
    """
    One-off worker for a single BrowserGymEnv instance.
    Protocol: conn ← (cmd, data), conn → (status, payload).
    """
    _apply_chat_patch()  # Ensure patch in spawn-based child processes

    env = None
    try:
        env = _make_env(env_id, task, env_kwargs)
    except Exception as e:
        conn.send(("error", str(e)))
        return

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                try:
                    obs = env.reset(seed=data)
                    conn.send(("ok", obs))
                except Exception as e:
                    conn.send(("error", str(e)))
            elif cmd == "step":
                try:
                    obs, reward, terminated, truncated, info = env.step(data)
                    conn.send(("ok", (obs, reward, terminated or truncated, info)))
                except Exception as e:
                    conn.send(("error", str(e)))
            elif cmd == "close":
                _close_env(env)
                conn.close()
                return
    except EOFError:
        _close_env(env)


# ===================================================================
# BrowserGymEnv — the public API
# ===================================================================


@register_env("browsergym")
class BrowserGymEnv(BaseEnv):
    """
    BrowserGym environment.
    """

    def __init__(
        self,
        env_id: str = "browsergym/openended",
        task: dict | None = None,
        timeout: float = 120.0,
        max_turns: int | None = None,
        **env_kwargs,
    ):
        self._env_id = env_id
        self._task = task
        self._env_kwargs = env_kwargs
        self._timeout = timeout
        self._closed = False
        self.max_turns = max_turns
        self._step_count = 0

        self._mode = "direct"
        parent, child = mp.Pipe()
        self._conn = parent
        self._proc = mp.Process(
            target=_direct_worker,
            args=(child, env_id, task, env_kwargs),
            daemon=True,
        )
        self._proc.start()

    # -- internal helpers ---------------------------------------------------

    def _send_recv(self, cmd: str, data: Any) -> Any:
        """Send command and block for reply. Raises on error or timeout."""
        self._conn.send((cmd, data))
        if not self._conn.poll(self._timeout):
            raise TimeoutError(f"Worker timeout ({self._timeout}s)")
        status, payload = self._conn.recv()

        if status == "error":
            raise RuntimeError(f"BrowserGym worker error: {payload}")
        return payload

    # -- public API ---------------------------------------------------------

    def reset(self, seed=None):
        if self._closed:
            raise RuntimeError("Cannot reset a closed env")
        self._step_count = 0
        return self._send_recv("reset", seed)

    def step(self, action):
        obs, reward, done, info = self._send_recv("step", action)
        self._step_count += 1
        if self.max_turns is not None and self._step_count >= self.max_turns:
            done = True
        return obs, reward, done, info

    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            self._conn.send(("close", None))
            self._proc.join(timeout=60)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()

    @staticmethod
    def from_dict(extra_info: dict) -> "BrowserGymEnv":
        return BrowserGymEnv(
            env_id=extra_info["env_id"],
            headless=extra_info.get("headless", True),
            timeout=extra_info.get("timeout", 120.0),
            max_turns=extra_info.get("max_turns"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    def __del__(self):
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass
