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
Tavily web search and extraction tools.

Requires ``TAVILY_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from axon.tools.tools import Tool
from axon.tools.types import ToolOutput

TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"


class TavilySearchTool(Tool):
    """Search the web using Tavily API."""

    def __init__(self):
        self._client: httpx.Client | None = httpx.Client()
        super().__init__(name="tavily_search", description="Search the web for information on a specific query")

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
                        "query": {"type": "string", "description": "The search query"},
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "The depth of search",
                        },
                        "max_results": {"type": "integer", "description": "Maximum number of results"},
                    },
                    "required": ["query"],
                },
            },
        }

    def forward(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        **kwargs,
    ) -> ToolOutput:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return ToolOutput(name=self.name, error="TAVILY_API_KEY is not set")
        if self._client is None:
            return ToolOutput(name=self.name, error="HTTP client not initialized")

        try:
            params: dict[str, Any] = {"query": query, "search_depth": search_depth, "max_results": max_results}
            if include_domains:
                params["include_domains"] = include_domains
            if exclude_domains:
                params["exclude_domains"] = exclude_domains

            resp = self._client.post(
                TAVILY_SEARCH_ENDPOINT,
                json=params,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            if not resp.is_success:
                return ToolOutput(name=self.name, error=f"HTTP {resp.status_code}: {resp.text}")
            return ToolOutput(name=self.name, output=resp.json())
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")

    def __del__(self):
        if self._client:
            self._client.close()


class TavilyExtractTool(Tool):
    """Extract web page content from URLs using Tavily API."""

    def __init__(self):
        self._client: httpx.Client | None = httpx.Client()
        super().__init__(name="tavily_extract", description="Extract web page content from one or more specified URLs")

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
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to extract content from",
                        },
                    },
                    "required": ["urls"],
                },
            },
        }

    def forward(self, urls: list[str], **kwargs) -> ToolOutput:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return ToolOutput(name=self.name, error="TAVILY_API_KEY is not set")
        if self._client is None:
            return ToolOutput(name=self.name, error="HTTP client not initialized")

        try:
            resp = self._client.post(
                TAVILY_EXTRACT_ENDPOINT,
                json={"urls": urls, "include_images": False, "extract_depth": "basic"},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            if not resp.is_success:
                return ToolOutput(name=self.name, error=f"HTTP {resp.status_code}: {resp.text}")
            results = resp.json()["results"]
            return ToolOutput(name=self.name, output={r["url"]: r["raw_content"] for r in results})
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")

    def __del__(self):
        if self._client:
            self._client.close()


if __name__ == "__main__":
    # Test extract tool
    extract_tool = TavilyExtractTool()
    extract_result = extract_tool.forward(urls=["https://example.com/"])
    print("Extract Tool Result:")
    print(extract_result)

    # Test search tool
    search_tool = TavilySearchTool()
    search_result = search_tool.forward(query="Latest developments in AI research")
    print("\nSearch Tool Result:")
    print(search_result)

    import asyncio

    async def test_async():
        print("\nStarting async requests...")

        # Extract async
        extract_coro = extract_tool.async_forward(urls=["https://example.com/"], use_async=True)
        extract_result = await extract_coro
        print("Async extract completed!")
        print(extract_result)

        # Search async
        search_coro = search_tool.async_forward(query="Python programming best practices", use_async=True)
        search_result = await search_coro
        print("Async search completed!")
        print(search_result)

    asyncio.run(test_async())
