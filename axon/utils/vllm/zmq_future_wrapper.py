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
"""ZMQ Future Wrapper that wraps multiple futures and returns a single result."""

import threading
from concurrent.futures import Future
from typing import Any


class ZMQFutureWrapper:
    """A wrapper around a list of concurrent.futures.Future objects.

    This class simulates a single Future object that waits for all futures
    in the list to complete and returns the result from the first future
    when .result() is called.

    This is useful for distributed execution where multiple workers execute
    in parallel, but only one result (typically from the first or a specific worker)
    needs to be returned.
    """

    def __init__(self, futures: list[Future], result_index: int = 0):
        """Initialize the ZMQ Future Wrapper.

        Args:
            futures: List of concurrent.futures.Future objects to wrap.
            result_index: Index of the future whose result should be returned (default: 0).
        """
        self._futures = futures
        self._result_index = result_index
        self._result_cache: Any | None = None
        self._exception_cache: Exception | None = None
        self._done = False

    def result(self, timeout: float | None = None) -> Any:
        """Wait for all futures to complete and return the result from the specified index.

        Args:
            timeout: Maximum time to wait for the futures to complete.

        Returns:
            The result from the future at result_index.

        Raises:
            Exception: If any of the futures raised an exception.
        """
        if self._done:
            if self._exception_cache is not None:
                raise self._exception_cache
            return self._result_cache

        # Wait for all futures to complete
        results = []
        for i, future in enumerate(self._futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                # If any future fails, cache the exception and raise it
                self._exception_cache = e
                self._done = True
                raise

        # Cache and return the result from the specified index
        self._result_cache = results[self._result_index]
        self._done = True
        return self._result_cache

    def done(self) -> bool:
        """Return True if all futures are done.

        Returns:
            True if all futures have completed, False otherwise.
        """
        if self._done:
            return True
        return all(future.done() for future in self._futures)

    def cancel(self) -> bool:
        """Attempt to cancel all futures.

        Returns:
            True if all futures were successfully cancelled, False otherwise.
        """
        results = [future.cancel() for future in self._futures]
        return all(results)

    def cancelled(self) -> bool:
        """Return True if all futures were cancelled.

        Returns:
            True if all futures are cancelled, False otherwise.
        """
        return all(future.cancelled() for future in self._futures)

    def running(self) -> bool:
        """Return True if any future is currently running.

        Returns:
            True if any future is running, False otherwise.
        """
        return any(future.running() for future in self._futures)

    def add_done_callback(self, fn):
        """Add a callback to be called when all futures complete.

        Args:
            fn: Callback function that takes this wrapper as an argument.
        """
        # Track how many futures have completed
        lock = threading.Lock()
        completed_count = [0]
        total = len(self._futures)

        def wrapper_callback(future):
            with lock:
                completed_count[0] += 1
                if completed_count[0] == total:
                    fn(self)

        # Add the wrapper callback to all futures
        for future in self._futures:
            future.add_done_callback(wrapper_callback)

    def exception(self, timeout: float | None = None) -> Exception | None:
        """Return the exception raised by any future, or None if no exception was raised.

        Args:
            timeout: Maximum time to wait for the futures to complete.

        Returns:
            The first exception encountered, or None if all futures completed successfully.
        """
        for future in self._futures:
            exc = future.exception(timeout=timeout)
            if exc is not None:
                return exc
        return None

    def __repr__(self) -> str:
        return f"ZMQFutureWrapper(futures={len(self._futures)}, result_index={self._result_index}, done={self.done()})"
