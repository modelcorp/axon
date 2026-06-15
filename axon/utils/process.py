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
"""Process management utilities."""

import multiprocessing as mp

from axon.utils.print_utils import colorful_print


def cleanup_subprocesses(process_list: list[mp.Process]):
    """
    Clean up zombie subprocesses by joining or terminating them.
    """
    if not process_list:
        return

    colorful_print(f"Cleaning up {len(process_list)} subprocesses...", "cyan")

    cleaned = 0
    terminated = 0

    for process in process_list:
        if process.is_alive():
            # Give process a chance to exit gracefully
            process.join(timeout=5.0)

            # If still alive after timeout, terminate it
            if process.is_alive():
                colorful_print(f"Terminating subprocess {process.pid}", "yellow")
                process.terminate()
                process.join(timeout=2.0)
                terminated += 1

                # Last resort: kill if still alive
                if process.is_alive():
                    process.kill()
                    process.join()
        else:
            # Process already exited, just join to clean up
            process.join()

        cleaned += 1

    # Clear the list after cleanup
    process_list.clear()

    colorful_print(f"Cleaned up {cleaned} subprocesses ({terminated} terminated)", "green")
