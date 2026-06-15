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
from __future__ import annotations

import os
from typing import Any

import httpx

from axon.tools.tools import Tool
from axon.tools.types import ToolOutput

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
DEFAULT_TIMEOUT = 30
REFERENCE_COUNT = 8


class GoogleSearchTool(Tool):
    """Search using Google Custom Search API."""

    def __init__(self, reference_count: int = REFERENCE_COUNT, timeout: float = DEFAULT_TIMEOUT):
        self.reference_count = reference_count
        self.timeout = timeout
        self._client = httpx.Client()
        super().__init__(
            name="google_search",
            description=f"Search Google, returning top {reference_count} results with snippets",
        )

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query for Google search"},
                    },
                    "required": ["query"],
                },
            },
        }

    def forward(self, query: str, **kwargs) -> ToolOutput:
        secret_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
        engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not secret_key or not engine_id:
            return ToolOutput(name=self.name, error="GOOGLE_SEARCH_SECRET_KEY or GOOGLE_SEARCH_ENGINE_ID not set")

        try:
            resp = self._client.get(
                GOOGLE_SEARCH_ENDPOINT,
                params={"key": secret_key, "cx": engine_id, "q": query, "num": self.reference_count},
                timeout=self.timeout,
            )
            if not resp.is_success:
                return ToolOutput(name=self.name, error=f"HTTP {resp.status_code}: {resp.text}")
            contexts = resp.json().get("items", [])[: self.reference_count]
            results = {c["link"]: c["snippet"] for c in contexts}
            return ToolOutput(name=self.name, output=results)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass


if __name__ == "__main__":
    search = GoogleSearchTool()
    print(search.forward(query="Give me current time right now in PST"))
