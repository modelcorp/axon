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
from typing import Any


class ClassWithInitArgs:
    """
    Wrapper class that stores constructor arguments for deferred instantiation.
    This class is particularly useful for remote class instantiation where
    the actual construction needs to happen at a different time or location.

    Example:
        >>> class MyClass:
        ...     def __init__(self, x, y=10):
        ...         self.x = x
        ...         self.y = y
        >>>
        >>> # Store the class and arguments for later instantiation
        >>> wrapper = ClassWithInitArgs(MyClass, 5, y=20)
        >>>
        >>> # Later, instantiate the class
        >>> instance = wrapper()
        >>> print(instance.x, instance.y)  # Output: 5 20
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        """Initialize the ClassWithInitArgs instance.

        Args:
            cls: The class to be instantiated later
            *args: Positional arguments for the class constructor
            **kwargs: Keyword arguments for the class constructor
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        """Instantiate the stored class with the stored arguments."""
        return self.cls(*self.args, **self.kwargs)
