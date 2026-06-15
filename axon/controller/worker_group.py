# Copyright 2025 Model AI Corp.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
#
# Adapted from verl single_controller/base/worker_group.py (github.com/volcengine/verl), Apache-2.0.
"""
Base class for WorkerGroup
"""

import logging
import signal
import threading
import time
from collections.abc import Callable

from axon.controller.decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn
from axon.core import ResourcePool


class WorkerGroup:
    """
    Base class for managing a group of workers in a distributed system.
    The class provides methods for worker management, aliveness checking, and method binding.
    """

    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        self.resource_pool = resource_pool
        self._workers = []
        self._worker_names = []

        self._dispatch_info = {}
        self._collect_info = {}

        self._master_addr = None
        self._master_port = None

        self._checker_thread: threading.Thread = None

    def _is_worker_alive(self, worker):
        """Check if a worker is alive. Must be implemented by derived classes."""
        raise NotImplementedError("WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        """Starts a background thread to monitor worker aliveness.

        Args:
            every_n_seconds (int): Interval between aliveness checks
        """
        # before starting checking worker aliveness, make sure all workers are already alive
        while True:
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            if False in all_state:
                time.sleep(1)
            else:
                break

        def check_workers_alive_fn(workers: list, is_alive: Callable, gap_time: float = 1) -> None:
            """Continuously monitors worker processes and raises SIGABRT if any worker dies.

            Args:
                workers (List):
                    List of worker objects to monitor
                is_alive (Callable):
                    Function to check if a worker is alive
                gap_time (float):
                    Time interval between checks
            """
            import time

            while True:
                for worker in workers:
                    if not is_alive(worker):
                        logging.warning(f"worker {worker} is not alive sending signal to main thread")
                        signal.raise_signal(signal.SIGABRT)
                time.sleep(gap_time)

        self._checker_thread = threading.Thread(
            target=check_workers_alive_fn,
            args=(self._workers, self._is_worker_alive, every_n_seconds),
            daemon=True,
        )
        self._checker_thread.start()

    @property
    def world_size(self):
        """Number of workers in the group."""
        return len(self._workers)

    def _bind_worker_method(self, user_defined_cls, func_generator):
        """Binds worker methods to the WorkerGroup based on registered attributes.

        Args:
            user_defined_cls (type): The class containing methods to bind
            func_generator (Callable): Function that generates the bound method

        Returns:
            List[str]: List of method names that were successfully bound
        """
        method_names = []
        for method_name in dir(user_defined_cls):
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                # if it is a property, it will fail because Class doesn't have instance property
                continue

            if hasattr(method, MAGIC_ATTR):
                # this method is decorated by register
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(attribute, dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]
                disable_collective = attribute["disable_collective"]

                # get dispatch fn
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                # get execute_fn_name
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # get execute_fn from string
                try:
                    execute_fn = getattr(self, wg_execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # bind a new method to the RayWorkerGroup
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                    disable_collective=disable_collective,
                )

                try:
                    setattr(self, method_name, func)
                    method_names.append(method_name)
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        return method_names
