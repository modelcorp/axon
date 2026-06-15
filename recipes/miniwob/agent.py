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
# DISCLAIMER:
# This MiniWoB agent implementation is adapted from the BrowserGym demo agent:
# https://github.com/ServiceNow/BrowserGym/blob/main/demo_agent/agent.py
# Some parts have been modified or extended for custom use.

import base64
import io
import logging
import re
from typing import Any

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet  # type: ignore[import-untyped]
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html  # type: ignore[import-untyped]
from PIL import Image

from recipes.miniwob.prompts import SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT
from axon.core import Action, BaseAgent, register_agent

logger = logging.getLogger(__name__)

ACTION_SPACE_COT_DESCRIPTION = """\
# Action Space (This is the list of valid actions you are allowed to output after your chain-of-thought reasoning,
{action_set_description}
Here are examples of actions with chain-of-thought reasoning:
Thought: I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
Action: ```click("12")```
Thought: I found the information requested by the user, I will send it to the chat.
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""

ACTION_SPACE_DESCRIPTION = """\
# Action Space (This is the list of valid actions you are allowed to output,
{action_set_description}
Here are examples of actions that can be returned:
Action: ```click("12")```
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""


def image_to_jpg_base64_url(image: np.ndarray | Image.Image) -> str:
    """Convert a numpy array to a base64 encoded image url."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


@register_agent("miniwob")
class MiniWobAgent(BaseAgent):
    def __init__(
        self,
        use_html: bool = False,
        use_axtree: bool = True,
        use_screenshot: bool = False,
        use_cot_prompt: bool = True,
    ):
        self.use_html: bool = use_html
        self.use_axtree: bool = use_axtree
        self.use_screenshot: bool = use_screenshot
        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode="off",  # add visual effects
        )
        self.action_history: list[str] = []  # all are in string
        self.use_cot_prompt: bool = use_cot_prompt  # for interface compliance
        self.reset()

    @property
    def system_prompt(self) -> str:
        # Requires running process_observation first.
        return self.format_msgs_as_str(self.get_system_msgs(self.current_observation))

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict) -> None:
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        obs = self.parse_observation(observation)
        # Base message for the user
        user_prompt_content = self.format_msgs_as_str(self.get_user_msgs(obs))
        self.current_observation = obs
        return user_prompt_content

    def process_action(self, response: str) -> Action:
        action_str = self.parse_model_response(response)
        self.step += 1
        action_history_str = action_str if action_str != response else "Response is missing ``` ```"
        self.action_history.append(action_history_str)
        return Action(action=action_str)

    def reset(self):
        self.action_history = []
        self.current_observation = None
        self.step = 0

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def parse_model_response(self, response: str) -> str:
        """
        Extracts the last content enclosed within triple backticks (``` ```) from the response.

        If the response contains multiple segments wrapped in triple backticks,
        this function returns the content of the **last** occurrence.
        If no such formatting is found, it returns the entire response unmodified.

        Args:
            response (str): The raw text response to be processed.

        Returns:
            action (str): The extracted action (content from the last occurrence of triple backticks
                  or the full response if no match is found)
        """
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)  # Find all occurrences
        if matches:
            return matches[-1]
        return response

    def get_system_msgs(self, obs: dict[str, Any]) -> list[dict[str, str]]:
        system_msgs = []
        system_msgs.append({"type": "text", "text": SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT})
        system_msgs.append({"type": "text", "text": "\n# Goal (Below is the goal you want to accomplish):\n\n"})
        system_msgs.extend(obs["goal_object"])
        return system_msgs

    def get_user_msgs(self, obs: dict[str, Any]) -> list[dict[str, str]]:
        user_msgs = []

        # Add open tabs information
        user_msgs.extend(
            self._format_open_tabs(obs["open_pages_urls"], obs["open_pages_titles"], obs["active_page_index"])
        )

        # Add page information based on settings
        if self.use_axtree:
            user_msgs.append({"type": "text", "text": f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}\n\n"})

        if self.use_html:
            user_msgs.append({"type": "text", "text": f"# Current page DOM\n\n{obs['pruned_html']}\n\n"})

        if self.use_screenshot:
            user_msgs.extend(self._format_screenshot(obs["screenshot"]))

        # Add action history if available
        if self.action_history:
            user_msgs.append({"type": "text", "text": "# History of past actions\n"})

            for i, action in enumerate(self.action_history):
                action_label = "Last Action" if i == len(self.action_history) - 1 else f"Action {i}"
                user_msgs.append({"type": "text", "text": f"{action_label}:\n{action}\n"})

        # Add error message if present
        if obs["last_action_error"]:
            user_msgs.append(
                {"type": "text", "text": f"# Error message from last action\n\n{obs['last_action_error']}\n\n"}
            )

        # Add action space description
        user_msgs.append({"type": "text", "text": self._get_action_space_description()})

        # Add next action prompt
        user_msgs.append(
            {
                "type": "text",
                "text": (
                    "# Next action\nThe task has not been completed yet. You will now think step by step "
                    "and produce your next best action. Reflect on your past actions, any resulting "
                    "error message, and the current state of the page before deciding on your next "
                    "action. The content must be in the same format as shown before in the Action "
                    "Space. You can plan ahead but only 1 immediate action is needed."
                ),
            }
        )
        return user_msgs

    def format_msgs_as_str(self, msgs: list[dict[str, Any]]) -> str:
        prompt_text_strings = []
        for message in msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append("image_url: " + image_url[:30] + "... (truncated)")
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(f"Unknown message type {repr(message['type'])} in the task goal.")
        return " ".join(prompt_text_strings)

    def _format_open_tabs(self, urls: list, titles: list, active_index: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"type": "text", "text": "# Currently open tabs (This is the current active tabs)\n"}
        ]

        for idx, (url, title) in enumerate(zip(urls, titles, strict=False)):
            active_marker = " (active tab)" if idx == active_index else ""
            messages.append({"type": "text", "text": f"Tab {idx}{active_marker}\n  Title: {title}\n  URL: {url}\n"})
        return messages

    def _format_screenshot(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        messages.append(
            {
                "type": "text",
                "text": "# Current page Screenshot\n",
            }
        )
        messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(screenshot),
                    "detail": "auto",
                },
            }
        )
        return messages

    def _get_action_space_description(self) -> str:
        action_set_description = self.action_set.describe(with_long_description=False, with_examples=False)
        if self.use_cot_prompt:
            return ACTION_SPACE_COT_DESCRIPTION.format(action_set_description=action_set_description)
        return ACTION_SPACE_DESCRIPTION.format(action_set_description=action_set_description)
