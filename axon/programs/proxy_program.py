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
import json
import random
from typing import Any

import requests

from axon.programs.base_program import BaseProgram, ProgramResult, register_program


@register_program("proxy")
class ProxyProgram(BaseProgram):
    """
    A program that acts as a proxy to external services, forwarding requests and managing sessions.

    This program initiates a session with an external service via HTTP POST request,
    registers the session with the engine, and monitors the program status until completion.
    """

    def __init__(
        self,
        proxy_url: str,
        proxy_token: str,
        server_url: str,
        env_args: dict[str, Any] | None = None,  # contains payload for the init call to customer api
        group_id: str = "",
        sample_params: dict[str, Any] | None = None,
        endpoint_url: str = "",
        retry_limit: int = 1,
        program_timeout: int = 10800,
    ):
        """
        Initialize the ProxyProgram.

        Args:
            proxy_url: The URL of the external service to proxy requests to
            proxy_token: Authentication token for the external service
            server_url: The callback endpoint URL for the external service
            env_args: Additional payload data for the initialization call to the external API
            group_id: Identifier for grouping related programs
            sample_params: Parameters for sampling/generation
            endpoint_url: URL for the program endpoint
            retry_limit: Maximum number of retry attempts for failed requests
            program_timeout: Maximum time (in seconds) before program times out
        """
        super().__init__(
            group_id=group_id,
            sample_params=sample_params,
            endpoint_url=endpoint_url,
            retry_limit=retry_limit,
            program_timeout=program_timeout,
        )

        self.proxy_url = proxy_url
        self.proxy_token = proxy_token
        self.server_url = server_url
        self.env_args = env_args or {}

    async def run(self) -> ProgramResult:
        """
        Execute the proxy program workflow.

        This method:
        1. Sends a POST request to the external service to start a program
        2. Registers the external session ID with the engine
        3. Monitors the program status until completion

        Returns:
            ProgramResult: A result object indicating the program completed successfully

        Raises:
            requests.RequestException: If the HTTP request to the external service fails
            ValueError: If the response cannot be parsed as JSON
            Exception: If no sessionId is returned in the response
        """
        # Prepare the payload for the external service
        reset_payload = self._prepare_reset_payload()

        # Attempt to start the program with retry logic
        response = await self._start_program_with_retry(reset_payload)

        # Parse the response and extract session ID
        user_session_id = self._extract_session_id(response)

        # Register the session and update metadata
        await self._register_session_and_metadata(user_session_id, reset_payload)

        # Monitor program status until completion
        await self._monitor_program_status()

        # Return a dummy result indicating successful completion
        return ProgramResult(reward=0, done=True)

    def _prepare_reset_payload(self) -> dict[str, Any]:
        """
        Prepare the payload for the reset/initialization request.

        Returns:
            dict: The payload with callback endpoint and auth token added
        """
        reset_payload = self.env_args.copy()
        reset_payload["callbackEndpoint"] = self.server_url
        reset_payload["callbackAuthToken"] = self.proxy_token
        return reset_payload

    async def _start_program_with_retry(self, reset_payload: dict[str, Any]) -> requests.Response:
        """
        Start the program with retry logic for robustness.

        Args:
            reset_payload: The payload to send in the POST request

        Returns:
            requests.Response: The successful response from the external service

        Raises:
            requests.RequestException: If all retry attempts fail
        """
        reset_retry_limit = 3
        attempts = 0

        while attempts < reset_retry_limit:
            try:
                attempts += 1
                # Add random delay to avoid overwhelming the external service
                delay = random.uniform(0.5, 10)
                await asyncio.sleep(delay)

                response = await asyncio.to_thread(
                    requests.post,
                    self.proxy_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.proxy_token}",
                    },
                    data=json.dumps(reset_payload),
                    timeout=30,
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                print(f"Request failed (attempt {attempts}/{reset_retry_limit}): {e}")
                if attempts < reset_retry_limit:
                    continue
                raise e

    def _extract_session_id(self, response: requests.Response) -> str:
        """
        Extract the session ID from the response.

        Args:
            response: The HTTP response from the external service

        Returns:
            str: The extracted session ID

        Raises:
            ValueError: If the response cannot be parsed as JSON
            Exception: If no sessionId is found in the response
        """
        try:
            resp_json = response.json()
        except ValueError as e:
            print("Failed to parse JSON response")
            raise e

        user_session_id = resp_json.get("sessionId")
        if not user_session_id:
            raise Exception("No sessionId in response")

        return user_session_id

    async def _register_session_and_metadata(self, user_session_id: str, reset_payload: dict[str, Any]) -> None:
        """
        Register the external session ID with the engine and update metadata.

        Args:
            user_session_id: The session ID from the external service
            reset_payload: The payload used for initialization (stored as metadata)
        """
        print(f"Successfully started agent for session: {self.session_id}, {user_session_id}, {reset_payload['orgId']}")

        await self.engine.register_external_session_id(user_session_id, self.session_id)
        await self.engine.run_in_engine_loop_async(
            self.engine.add_to_program_metadata(
                session_id=self.session_id, metadata_key="reset_payload", metadata_val=reset_payload
            )
        )

    async def _monitor_program_status(self) -> None:
        """
        Monitor the program status until completion.

        This method polls the program status every 15 seconds until the program
        is marked as done.
        """
        # Initial sleep period
        await asyncio.sleep(15)

        # Check if the program is done
        done = await self.engine.run_in_engine_loop_async(self.engine.check_program_status(session_id=self.session_id))

        # Continue polling until the program is complete
        while not done:
            await asyncio.sleep(15)
            done = await self.engine.run_in_engine_loop_async(
                self.engine.check_program_status(session_id=self.session_id)
            )
