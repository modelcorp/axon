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
Global variables for the Axon project.
"""

import os

# Debug Flags
DEBUG_MOE_REPLAY = os.environ.get("DEBUG_MOE_REPLAY", "0") == "1"


# Gemini Vertex AI Config (for dataset preprocessing).
GCP_PROJECT_ID = "cloud-llm-test"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"
