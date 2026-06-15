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
import asyncio
import base64
import traceback
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from openai.types.completion import Completion
from PIL import Image


def pil_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def serialize_images_for_http(images: list[Any]) -> list[dict]:
    out: list[dict] = []
    for x in images:
        if isinstance(x, Image.Image):
            out.append(
                {
                    "type": "image_base64",
                    "mime_type": "image/png",
                    "data": pil_to_b64_png(x),
                }
            )
        else:
            raise TypeError(f"Unsupported image payload type: {type(x)}")
    return out


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids


async def poll_completions_openai(
    address: str, timeout: int = 2700, max_retries: int = 3, **completions_request
) -> Completion:
    """
    Poll a single OpenAI-compatible completions endpoint with retry logic.

    This function sends a POST request to the /v1/completions endpoint of a server
    and handles retries for failed requests, including special handling for aborted
    requests which don't count towards the retry limit.

    Args:
        address: Server address in format "host:port" (e.g., "127.0.0.1:8000")
        **completions_request: Keyword arguments for the completions request,
                             including model, prompt, temperature, etc.

    Returns:
        Dict containing the completion response from the server

    Raises:
        Exception: If all retries are exhausted or if the server returns an error
    """
    # Use httpx for async HTTP requests
    base_url = f"http://{address}/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }

    # Remove meta_info if present
    if "meta_info" in completions_request:
        completions_request.pop("meta_info")
    # Remove extra_headers from the payload
    if "extra_headers" in completions_request:
        completions_request.pop("extra_headers")

    # Serialize image data since http doesn't support it
    serialized_image_data = {}
    if "image_data" in completions_request:
        image_data = completions_request.pop("image_data")
        if image_data:
            serialized_image_data = serialize_images_for_http(image_data)
            completions_request["image_data"] = serialized_image_data

    retry_delay = 1  # Initial delay in seconds
    retry = 0
    while retry < max_retries:
        try:
            # Create a new client for each request to avoid blocking
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                response = await client.post(base_url, json=completions_request, headers=headers)
                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"API request failed with status {response.status_code}: {error_text}")
                result = response.json()

                # If server says aborted, retry without counting as "attempt"
                aborted = False
                choices = result.get("choices", [])
                for c in choices:
                    fr = (c.get("finish_reason") or "").lower()
                    if "abort" in fr:
                        aborted = True
                        break
                if aborted:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                # Convert the raw JSON response to an OpenAI Completion object
                return result
        except Exception as e:
            traceback.print_exc()
            # If this is the last retry, raise the exception
            if retry == max_retries - 1:
                raise e
            # Exponential backoff
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
            retry += 1

    # This should never be reached due to the raise in the loop, but mypy requires it
    raise Exception("All retries failed")


async def _call_api(
    address: str, endpoint: str, payload: dict[str, Any], timeout: int = 300, max_retries: int = 3
) -> dict[str, Any]:
    """
    Make an HTTP POST request to a single address with retry logic.

    Args:
        address: Server address (e.g., "127.0.0.1:8000")
        endpoint: API endpoint path (e.g., "/pause_generation")
        payload: JSON payload for the request body
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Dict with keys: "address", "success", and either "result" or "error"/"status"/"body"
    """
    url = f"http://{address}{endpoint}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return {"address": address, "success": True, "result": resp.json()}

        except Exception as e:
            # On last attempt → return full error info
            if attempt == max_retries - 1:
                status = getattr(getattr(e, "response", None), "status_code", None)
                body = None
                try:
                    if hasattr(e, "response") and e.response is not None:
                        body = e.response.text
                except Exception:
                    pass

                return {
                    "address": address,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "status": status,
                    "body": body,
                }

            await asyncio.sleep(2**attempt)  # backoff


async def fetch_responses_from_addresses(
    addresses: list[str], endpoint: str, payload: dict[str, Any], **kwargs
) -> list[dict[str, Any]]:
    """
    Send parallel HTTP requests to multiple server addresses.

    Args:
        addresses: List of server addresses (e.g., ["127.0.0.1:8000", "127.0.0.1:8001"])
        endpoint: API endpoint to call (e.g., "/pause_generation")
        payload: JSON payload to send in the request body
        **kwargs: Additional arguments passed to _call_api (timeout, max_retries)

    Returns:
        List of response dictionaries, one per address. Each contains:
        - "address": The server address
        - "success": Boolean indicating if the request succeeded
        - "result": Response JSON (if success=True)
        - "error"/"status"/"body": Error details (if success=False)
    """
    tasks = [_call_api(addr, endpoint, payload, **kwargs) for addr in addresses]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
