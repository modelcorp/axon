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
import importlib.util
import os
import random
import string


def get_random_string(length: int) -> str:
    """Generate a random string of specified length using letters and digits.

    Args:
        length: The length of the random string to generate

    Returns:
        A random string containing letters and digits
    """
    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


PKG_PATH_PREFIX = "pkg://"
FILE_PATH_PREFIX = "file://"


def load_module(module_path: str, module_name: str | None = None) -> object:
    """Load a module from a path.

    Args:
        module_path (str):
            The path to the module. Either
                - `pkg_path`, e.g.,
                    - "pkg://axon.utils.dataset.rl_dataset"
                    - "pkg://axon/utils/dataset/rl_dataset"
                - or `file_path` (absolute or relative), e.g.,
                    - "file://axon/utils/dataset/rl_dataset.py"
                    - "/path/to/axon/utils/dataset/rl_dataset.py"
        module_name (str, optional):
            The name of the module to added to ``sys.modules``. If not provided, the module will not be added,
                thus will not be cached and directly ``import``able.

    Returns:
        The loaded module object, or None if module_path is empty.
    """
    if not module_path:
        return None

    if module_path.startswith(PKG_PATH_PREFIX):
        module_name = module_path[len(PKG_PATH_PREFIX) :].replace("/", ".")
        module = importlib.import_module(module_name)

    else:
        if module_path.startswith(FILE_PATH_PREFIX):
            module_path = module_path[len(FILE_PATH_PREFIX) :]

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Custom module file not found: {module_path=}")

        # Use the provided module_name for the spec, or derive a unique name to avoid collisions.
        spec_name = module_name or f"custom_module_{hash(os.path.abspath(module_path))}"
        spec = importlib.util.spec_from_file_location(spec_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path=}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from {module_path=}") from e

        if module_name is not None:
            import sys

            # Avoid overwriting an existing module with a different object.
            if module_name in sys.modules and sys.modules[module_name] is not module:
                raise RuntimeError(
                    f"Module name '{module_name}' already in `sys.modules` and points to a different module."
                )
            sys.modules[module_name] = module

    return module
