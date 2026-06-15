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
"""LLM utility functions for calling various LLM APIs."""

import logging
import time

import openai
import vertexai
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory

from axon.globals import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL, OAI_RM_MODEL

logger = logging.getLogger(__name__)


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    """
    Call OpenAI LLM to generate n responses at a given temperature.

    Args:
        prompt: The text prompt to send to the LLM.
        system_prompt: System instruction or system prompt to send to the model.
        n: Number of responses to generate.
        temperature: Sampling temperature.
        model_id: The specific OpenAI model to use.
        retry_count: Number of times to retry on rate-limit errors.

    Returns:
        List[str]: A list of response texts from the OpenAI model.
    """
    client = openai.OpenAI()

    backoff = 1
    retry_count = int(retry_count)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as e:
            if "429" in str(e):
                logger.warning("Retry due to rate limit: %s", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                logger.exception("Exception: %s", e)
                return []
    else:
        logger.error("All %d retries exhausted due to rate limiting.", retry_count)
        return []

    if n == 1:
        content = response.choices[0].message.content
        return [content] if content is not None else []
    return [choice.message.content for choice in response.choices if choice.message.content is not None]


def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = GCP_PROJECT_ID,
    location: str = GCP_LOCATION,
    model_id: str = GEMINI_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    """
    Call Gemini LLM on Vertex AI to generate n responses at a given temperature.

    Args:
        prompt: The text prompt to send to the LLM.
        system_prompt: System instruction or system prompt to send to the model.
        n: Number of responses to generate.
        temperature: Sampling temperature.
        project_id: Your GCP project ID.
        location: The region to use (e.g., us-central1).
        model_id: The specific Gemini model resource name.
        retry_count: Number of times to retry on rate-limit errors.

    Returns:
        List[str]: A list of response texts from the Gemini model.
    """
    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)

    # Define which harm categories to allow (or set thresholds).
    HARM_CATEGORIES = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    # Instantiate the GenerativeModel
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    # Add an exponential backoff for rate limit errors
    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    for attempt in range(retry_count):
        try:
            # Request multiple candidates by specifying n (candidate_count)
            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=[
                    SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE) for h in HARM_CATEGORIES
                ],
            )
            # Once successful, break out of the retry loop
            break
        except Exception as e:
            # Retry if there's a rate-limit error (HTTP 429)
            if "429" in str(e):
                logger.warning("Retry due to rate limit: %s", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            elif "403" in str(e):
                logger.error("NO ACCESS TO ENDPOINT: %s", e)
                raise NotImplementedError from None
            else:
                logger.exception("Exception: %s", e)
                return []  # or raise an exception if desired
    else:
        logger.error("All %d retries exhausted due to rate limiting.", retry_count)
        return []

    # Collect the texts from all returned candidates
    try:
        # Keep this to check for errors in indexing.
        [candidate.text for candidate in response.candidates]
        if len(response.candidates) == 1:
            return [response.candidates[0].text]
        return [candidate.text for candidate in response.candidates]
    except Exception as e:
        logger.error("Error extracting text from response: %s", e)
        return []


__all__ = ["call_oai_rm_llm", "call_gemini_llm"]
