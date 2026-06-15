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
# ruff: noqa: E402
# Side-effectful module: temporarily sets SGLANG_USE_CPU_ENGINE=1 and mocks vllm if absent.
import os
import sys
from unittest.mock import Mock

_had_cpu_engine = "SGLANG_USE_CPU_ENGINE" in os.environ
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

try:
    import vllm  # noqa: F401
except ImportError:
    mock_vllm = Mock()
    mock_vllm._custom_ops = Mock()
    mock_vllm._custom_ops.scaled_fp8_quant = Mock()
    sys.modules.setdefault("vllm", mock_vllm)
    sys.modules.setdefault("vllm._custom_ops", mock_vllm._custom_ops)

import sglang  # noqa: F401
import sglang.srt.entrypoints.engine  # noqa: F401

if not _had_cpu_engine:
    del os.environ["SGLANG_USE_CPU_ENGINE"]
