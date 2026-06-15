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
import logging
import unicodedata
from typing import Any

import requests
from search_r1.reward import SearchR1RewardConfig, SearchR1RewardFn

from axon.core import MultiTurnEnvironment, register_env
from axon.utils.rewards.base import RewardInput

logger = logging.getLogger(__name__)


@register_env("search_r1")
class SearchR1Env(MultiTurnEnvironment):
    """
    The environment:
    1. Accepts search queries and returns formatted results
    2. Tracks search history for reward calculation
    3. Evaluates final answers using EM (Exact Match)
    4. Provides intermediate feedback for invalid actions
    """

    def __init__(
        self,
        task: dict,
        retrieval_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        max_turns: int = 5,
        structure_format_score: float = 0.0,
        final_format_score: float = 0.0,
        retrieval_score: float = 0.0,
        terminate_on_incorrect_action: bool = True,
        **kwargs,
    ):
        """
        Initialize Search-R1 environment.

        Args:
            task: Task dictionary with question and ground truth
            retrieval_url: URL of the retrieval service API
            topk: Number of top documents to retrieve
            max_turns: Maximum number of turns before termination
            structure_format_score: Bonus for valid tag structure
            final_format_score: Bonus for proper final format
            retrieval_score: Bonus for retrieving the answer
            terminate_on_incorrect_action: If True, terminate episode on invalid action
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.retrieval_url = retrieval_url
        self.topk = topk
        self.terminate_on_incorrect_action = terminate_on_incorrect_action

        reward_config = SearchR1RewardConfig(
            correct_reward=1.0,
            incorrect_reward=0.0,
            unk_error_reward=0.0,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
        )
        self.reward_fn = SearchR1RewardFn(config=reward_config)

        self.full_program = ""  # Complete conversation for reward calculation

    def reset(self):
        """Reset environment state."""
        super().reset()
        self.full_program = ""
        return self.task, {}

    def step(self, action: Any):
        # Store the action in history
        self.history.append(action)

        # Calculate reward and get next observation
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Check if episode should end
        # Episode ends when agent provides answer OR max turns reached
        if isinstance(action, dict) and action.get("type") == "answer":
            self.done = True

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True

        # If done, return empty observation
        if self.done:
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        Args:
            task: Task dictionary with question and ground truth
            action: Parsed action dict from agent

        Returns:
            Tuple of (reward, next_observation_dict)
        """
        if not isinstance(action, dict):
            return 0.0, {}

        action_type = action.get("type")
        content = action.get("content", "")
        full_response = action.get("full_response", "")

        # Accumulate program for final reward calculation
        self.full_program += full_response

        if action_type == "search":
            search_results = self._perform_search(content)

            # Normalize the search results to handle unicode differences
            search_results = unicodedata.normalize("NFC", search_results)

            # Add to program for reward calculation
            self.full_program += f"\n\n<information>{search_results}</information>\n\n"

            # Return results as next observation (no reward yet)
            return 0.0, {"search_results": search_results}

        elif action_type == "answer":
            reward_output = self.reward_fn(
                input=RewardInput(task_info=task, action=self.full_program),
            )
            # Episode ends with answer
            return reward_output.reward, {}
        elif action_type == "invalid":
            if self.terminate_on_incorrect_action:
                # Terminate episode with zero reward
                self.done = True
                return 0.0, {}
            else:
                # Continue episode with error message
                error_msg = (
                    "\nYour previous action is invalid. "
                    "If you want to search, you should put the query between <search> and </search>. "
                    "If you want to give the final answer, you should put the answer between "
                    "<answer> and </answer>. Please try again.\n"
                )
                return 0.0, {"error": error_msg}

        return 0.0, {}

    def _perform_search(self, query: str) -> str:
        """
        Call retrieval API and format results.

        Args:
            query: Search query string

        Returns:
            Formatted search results string
        """
        try:
            # Prepare payload (line 452-456)
            payload = {"queries": [query], "topk": self.topk, "return_scores": True}

            # Call API with timeout
            response = requests.post(self.retrieval_url, json=payload, timeout=30)
            response.raise_for_status()

            results = response.json()["result"]

            if results and len(results) > 0:
                return self._passages2string(results[0])
            else:
                return "No results found."

        except requests.exceptions.Timeout:
            logger.error(f"Search timeout for query: {query}")
            return "Search timed out. Please try again with a different query."
        except requests.exceptions.RequestException as e:
            logger.error(f"Search error for query '{query}': {e}")
            return f"Search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return "An unexpected error occurred during search."

    def _passages2string(self, retrieval_result: list) -> str:
        """
        Args:
            retrieval_result: List of retrieved documents

        Returns:
            Formatted string with document titles and text
        """
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    @staticmethod
    def from_dict(env_args: dict) -> "SearchR1Env":
        retrieval_url = env_args.pop("retrieval_url", "http://127.0.0.1:8000/retrieve")
        topk = env_args.pop("topk", 3)
        max_turns = env_args.pop("max_turns", 5)
        structure_format_score = env_args.pop("structure_format_score", 0.0)
        final_format_score = env_args.pop("final_format_score", 0.0)
        retrieval_score = env_args.pop("retrieval_score", 0.0)
        terminate_on_incorrect_action = env_args.pop("terminate_on_incorrect_action", True)

        return SearchR1Env(
            task=env_args,
            retrieval_url=retrieval_url,
            topk=topk,
            max_turns=max_turns,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            terminate_on_incorrect_action=terminate_on_incorrect_action,
        )
