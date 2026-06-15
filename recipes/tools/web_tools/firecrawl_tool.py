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
import time
from typing import Any

from axon.tools.tools import Tool
from axon.tools.types import ToolOutput

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", None)
DEFAULT_TIMEOUT = 10


class FirecrawlTool(Tool):
    """Scrape a URL and return content as markdown using Firecrawl."""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str | None = FIRECRAWL_API_KEY,
        api_url: str | None = None,
    ):
        if FirecrawlApp is None:
            raise ImportError("firecrawl not installed. Install with: pip install firecrawl")
        if not api_key and not api_url:
            raise ValueError("Either api_key or api_url must be provided")

        self.timeout = timeout
        self.app: Any = FirecrawlApp(api_key=api_key) if api_url is None else FirecrawlApp(api_url=api_url)
        super().__init__(
            name="firecrawl",
            description="Scrape a URL and return content as markdown with links",
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
                        "url": {"type": "string", "description": "Web URL to scrape"},
                    },
                    "required": ["url"],
                },
            },
        }

    def forward(self, url: str, **kwargs) -> ToolOutput:
        try:
            job = self.app.async_batch_scrape_urls(
                [url],
                params={"formats": ["markdown", "links"], "onlyMainContent": True},
            )
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Job start failed: {e}")

        if not job.get("success"):
            return ToolOutput(name=self.name, error="Firecrawl job failed to start")

        job_id = job["id"]
        start_time = time.monotonic()
        while True:
            status = self.app.check_batch_scrape_status(job_id)
            if status.get("completed"):
                break
            if time.monotonic() - start_time > self.timeout:
                return ToolOutput(name=self.name, error="Firecrawl request timed out")
            time.sleep(1)

        if status.get("success"):
            results = {page["metadata"]["url"]: page["markdown"] for page in status["data"]}
            return ToolOutput(name=self.name, output=results)
        return ToolOutput(name=self.name, error=f"Firecrawl error: {status.get('error')}")

    async def async_forward(self, url: str, **kwargs) -> ToolOutput:
        return self.forward(url=url)


if __name__ == "__main__":
    search = FirecrawlTool()

    start_time = time.monotonic()
    print(search.forward(url="https://example.com/"))
    end_time = time.monotonic()
    print(f"Time taken for sync: {end_time - start_time} seconds")

    # Test Async
    import asyncio

    async def test_async():
        coro = search.async_forward(url="https://example.com/")

        start_time = time.monotonic()
        result = await coro
        end_time = time.monotonic()
        print("Async result:", result)
        print(f"Time taken for async: {end_time - start_time} seconds")

    # Run the async test
    asyncio.run(test_async())
