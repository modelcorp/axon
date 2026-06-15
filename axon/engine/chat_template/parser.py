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

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoProcessor

from axon.engine.state.program_state import MultiModalData
from axon.tools.parsers.base_parser import ToolCallParser, get_tool_call_parser
from axon.tools.types import ToolCall
from axon.utils.tokenizer_pool import TokenizerPool
from axon.utils.vision_utils import process_image, process_video

PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {
        "role": "assistant",
        "content": "I'll search for that.",
        "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}],
    },
    # {"role": "tool", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {
        "role": "assistant",
        "content": "Let me search for Java information.",
        "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}],
    },
]


@dataclass
class TokenizeContext:
    """
    Optional context for tokenization. Text-only callers pass None.
    """

    messages: list[dict[str, Any]] | None = None
    images: list[Any] = field(default_factory=list)
    videos: list[Any] = field(default_factory=list)
    allow_legacy_placeholders: bool = True
    image_patch_size: int = 14
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TokenizeOutput:
    token_ids: list[int]
    token_strs: list[str] = field(default_factory=list)  # per-token strings (same length as token_ids)
    multi_modal_data: MultiModalData | None = None


class ChatTemplateParser:
    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        self.tokenizer_pool = tokenizer_pool
        self.tokenizer = tokenizer_pool.tokenizer
        self.assistant_token = ""
        self.generation_prompt = ""
        self.tool_parser = tool_parser

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        """
        Converts messages to String. Assumes contains first message in implementation.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    async def tokenize(self, input_text: str, ctx: TokenizeContext | None = None) -> TokenizeOutput:
        """
        Converts rendered prompt text -> token ids and per-token strings.

        Base behavior: text-only.
        Subclasses may use ctx to do multimodal binding + processor tokenization.
        """
        ids, strs = await self.tokenizer_pool.encode_with_strs(input_text, add_special_tokens=False)
        return TokenizeOutput(token_ids=ids, token_strs=strs)

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        stop_texts = []
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos_token, str) and eos_token:
            stop_texts.append(eos_token)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_ids = [eos_token_id]
        elif isinstance(eos_token_id, list | tuple | set):
            eos_token_ids = [int(token_id) for token_id in eos_token_id if token_id is not None]
        else:
            eos_token_ids = []

        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            for token_id in eos_token_ids:
                try:
                    token_text = decode([token_id], skip_special_tokens=False)
                except Exception:
                    continue
                if isinstance(token_text, str) and token_text:
                    stop_texts.append(token_text)

        return tuple(dict.fromkeys(stop_texts))

    def assistant_message_content(
        self,
        response: str,
    ) -> str:
        """Project raw sampled output into ``message["content"]``.

        The sampled token stream is kept intact for training.  This method
        only removes terminal chat-control tokens from the text stored in
        chat history and returned to programs.
        """
        stop_texts = sorted(
            {stop_text for stop_text in self._assistant_message_stop_texts() if stop_text},
            key=len,
            reverse=True,
        )
        if not stop_texts:
            return response

        text = response
        while True:
            stripped = text.rstrip()
            for stop_text in stop_texts:
                if stripped.endswith(stop_text):
                    text = stripped.removesuffix(stop_text).rstrip()
                    break
            else:
                return text

    def get_mm_regions(self, input_key: str, messages: list[dict]) -> list[tuple]:
        """Find multimodal placeholder regions in rendered ``input_key``.

        Returns a list of ``(text_start, text_end, modality, content_hash, data)``
        tuples describing where MM placeholders appear and what content they
        represent.  Base class returns ``[]`` (text-only).

        Subclasses that support multimodal content should override this to
        detect model-specific placeholder patterns (e.g.
        ``<|vision_start|><|image_pad|><|vision_end|>`` for Qwen).
        """
        return []

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(
        cls, tokenizer_pool: TokenizerPool, processor: AutoProcessor | None = None, disable_thinking: bool = False
    ) -> ChatTemplateParser:
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """

        # Determine parser type based on tokenizer name or path
        tokenizer = tokenizer_pool.tokenizer
        tool_parser: ToolCallParser | None = None
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
            elif "qwen" in model_name and "vl" in model_name:
                print(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                assert processor
                return QwenVLChatTemplateParser(
                    tokenizer_pool, processor=processor, disable_thinking=disable_thinking, tool_parser=tool_parser
                )
            elif "qwen" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                is_qwen3_next = "qwen3-next" in model_name or "qwen3_next" in model_name
                tool_parser = get_tool_call_parser("qwen")
                return QwenChatTemplateParser(
                    tokenizer_pool,
                    disable_thinking=disable_thinking,
                    is_qwen3_next=is_qwen3_next,
                    tool_parser=tool_parser,
                )
            elif "llama" in model_name:
                print(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
            elif "gpt" in model_name and "pretrainedtokenizerfast" in tokenizer_cls:
                print(f"Using OpenAIChatTemplateParser for {tokenizer.name_or_path}")
                tool_parser = get_tool_call_parser("openai_harmony")
                return OpenAIHarmonyChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
            elif "gemma" in model_name:
                if "gemma-4" in model_name or "gemma4" in model_name:
                    print(f"Using Gemma4ChatTemplateParser for {tokenizer.name_or_path}")
                    tool_parser = get_tool_call_parser("gemma4")
                    return Gemma4ChatTemplateParser(
                        tokenizer_pool, disable_thinking=disable_thinking, tool_parser=tool_parser
                    )
                print(f"Using GemmaChatTemplateParser for {tokenizer.name_or_path}")
                return GemmaChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
            elif "moonlight" in model_name:
                print(f"Using MoonlightChatTemplateParser for {tokenizer.name_or_path}")
                return MoonlightChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
            elif "glm" in model_name or "chatglm" in model_name:
                print(f"Using GlmChatTemplateParser for {tokenizer.name_or_path}")
                tool_parser = get_tool_call_parser("glm")
                return GlmChatTemplateParser(tokenizer_pool, disable_thinking=disable_thinking, tool_parser=tool_parser)

        # Default to the standard parser if no specific match
        print(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        parser = ChatTemplateParser(tokenizer_pool, tool_parser=tool_parser)
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        tokenizer = tokenizer_pool.tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.assistant_token + "<think>\n"

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eos_token


def _init_qwen_tokens(parser, tokenizer_pool: TokenizerPool, disable_thinking: bool = True):
    """Shared Qwen chat template token initialization for text and VL parsers."""
    tokenizer = tokenizer_pool.tokenizer
    parser.bos_token = tokenizer.bos_token
    parser.eos_token = tokenizer.eos_token
    parser.eot_token = "<|im_end|>"
    parser.system_token = "<|im_start|>system\n"
    parser.user_token = "<|im_start|>user\n"
    parser.assistant_token = "<|im_start|>assistant\n"
    if disable_thinking:
        parser.assistant_token += "<think>\n\n</think>\n\n"
    parser.generation_prompt = parser.assistant_token


class QwenChatTemplateParser(ChatTemplateParser):
    """
    Parser for Qwen models including Qwen3 and Qwen3-Next variants.

    When is_qwen3_next=True, handles:
    - Tool calls in assistant messages with <tool_call>...</tool_call> format
    - Tool response grouping under a single <|im_start|>user block
    - Tools system message with <tools>...</tools> XML format
    """

    def __init__(
        self,
        tokenizer_pool: TokenizerPool,
        disable_thinking=True,
        is_qwen3_next=False,
        tool_parser: ToolCallParser | None = None,
    ):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        self.is_qwen3_next = is_qwen3_next
        self.disable_thinking = disable_thinking

        # For Qwen3-Next, thinking is handled differently in the generation prompt
        _init_qwen_tokens(self, tokenizer_pool, disable_thinking and not is_qwen3_next)

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        return self._parse_qwen3(messages, add_generation_prompt, **kwargs)

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.eot_token, *super()._assistant_message_stop_texts())

    def _parse_qwen3(self, messages, add_generation_prompt, tools=None) -> str:
        """Original Qwen parsing logic."""
        result = ""
        if len(messages) == 0:
            return result

        # Make a copy to avoid modifying original
        messages = list(messages)

        # Handle system message and tools
        if tools:
            # When tools are present, create a combined system message
            result += self._format_tools_system_base(messages, tools)
            # Skip the system message in the main loop if it was first
            if messages and messages[0]["role"] == "system":
                messages = messages[1:]
        elif messages[0]["role"] != "system" and messages[0]["role"] != "skip":
            # if the first message is not a system message, add the default system message
            result += (
                self.system_token
                + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                + self.eot_token
            )
        elif messages[0]["role"] == "system":
            result += self.parse_system(messages[0])
            messages = messages[1:]

        for i, message in enumerate(messages):
            if result:
                # add newline to non-empty result
                result += "\n"
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message, i, messages)
            elif message["role"] == "skip":
                result += message["content"]
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt and messages[-1]["role"] != "skip":
            result += "\n" + self.generation_prompt
        return result

    def _format_tools_system_base(self, messages, tools):
        """Format system message with tools definition for Qwen3 base."""
        assert self.tool_parser, (
            f"Tool parser shouldn't be None when tools exist: messages: {messages}\n  tools: {tools}\n"
        )
        tool_section = self.tool_parser.get_tool_system_prompt(tools)

        result = self.system_token
        if messages and messages[0]["role"] == "system":
            result += messages[0]["content"] + "\n\n"
        else:
            result += "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n\n"
        result += tool_section
        result += self.eot_token
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        content = message.get("content", "")
        if not isinstance(content, str):
            content = ""

        result = self.assistant_token + content

        # Handle tool calls
        raw_tool_calls = message.get("tool_calls", [])
        if raw_tool_calls:
            assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: message: {message}\n"
            tool_calls = [ToolCall.from_raw_tool_call(raw) for raw in raw_tool_calls]
            if content:
                result += "\n"
            result += self.tool_parser.format_tool_calls(tool_calls)

        result += self.eot_token
        return result

    def parse_tool(self, message, current_index, messages):
        """
        Parse tool message with grouping logic.
        Consecutive tool messages are grouped under a single <|im_start|>user block.
        """
        assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: message: {message}\n"

        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        result = ""

        # Check if this is the first tool message in a sequence
        is_first_tool = current_index == 0 or messages[current_index - 1]["role"] != "tool"
        if is_first_tool:
            result += self.user_token.rstrip("\n")  # <|im_start|>user (no trailing newline)

        result += "\n" + self.tool_parser.format_tool_result(content=content, name=message.get("name", ""))

        # Check if this is the last tool message in a sequence
        is_last_tool = current_index == len(messages) - 1 or messages[current_index + 1]["role"] != "tool"
        if is_last_tool:
            result += self.eot_token

        return result


class GemmaChatTemplateParser(ChatTemplateParser):
    """
    A chat template parser specifically for Google's Gemma models.

    This parser formats a list of messages into a single string according to
    the Gemma instruction-tuned format. The format is turn-based, using
    <start_of_turn> and <end_of_turn> tokens to delineate messages.

    The roles are 'user' and 'model'. The 'assistant' role is mapped to 'model'.
    Since Gemma does not have a dedicated 'system' role, system messages are
    formatted as a user turn, which is a common convention. Tool responses
    are also treated as user turns.

    Example format:
    <bos><start_of_turn>user
    What are the most popular LLMs?<end_of_turn>
    <start_of_turn>model
    Gemma, Llama, and Mistral are popular choices.<end_of_turn>
    """

    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        tokenizer = tokenizer_pool.tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token  # Note: Typically not used between turns

        # Define Gemma-specific tokens and prefixes
        self.start_turn_token = "<start_of_turn>"
        self.end_turn_token = "<end_of_turn>"
        self.user_prefix = "user\n"
        self.model_prefix = "model\n"

        # The prompt to signal the model to start generating a response
        self.generation_prompt = f"{self.start_turn_token}{self.model_prefix}"

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.end_turn_token, *super()._assistant_message_stop_texts())

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        """
        Parses a list of messages into a single string for the Gemma model.
        """
        # Always start the conversation with the beginning-of-sequence token
        result = self.bos_token

        for message in messages:
            role = message.get("role")

            if role in ("system", "user", "tool"):
                # System and tool response messages are formatted as a user turn
                result += self.parse_user(message)
            elif role == "assistant":
                # The 'assistant' role corresponds to the 'model' turn
                result += self.parse_model(message)
            else:
                raise ValueError(f"Unsupported message role: {role}")

        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def parse_user(self, message):
        """Formats a user, system, or tool message."""
        content = message.get("content", "")
        return f"{self.start_turn_token}{self.user_prefix}{content}{self.end_turn_token}\n"

    def parse_model(self, message):
        """Formats an assistant (model) message."""
        content = message.get("content", "")
        return f"{self.start_turn_token}{self.model_prefix}{content}{self.end_turn_token}\n"


class Gemma4ChatTemplateParser(ChatTemplateParser):
    """
    Chat template parser for Google's Gemma 4 models.

    Matches the Jinja chat template shipped in
    ``google/gemma-4-{31B,26B-A4B}-it/chat_template.jinja``.
    The message framing/tool grammar follows that template; historical model
    thinking is intentionally preserved below for Axon prefix composition.

    Turn format:
        ``<|turn>role\\ncontent<turn|>\\n``
    Roles: ``system``, ``user``, ``model`` (mapped from ``assistant``).

    Tool calls:  ``<|tool_call>call:{name}{args}<tool_call|>``
    Tool responses: ``<|tool_response>response:{name}{...}<tool_response|>``
    (rendered inside the model turn, NOT as standalone tool turns.)

    Thinking (enabled):
        ``<bos><|turn>system\\n<|think|>\\n{system}<turn|>\\n<|turn>user\\n...``
    Thinking (disabled, default for RL):
        ``<bos><|turn>user\\nHi<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>``
    """

    THINK_TOKEN = "<|think|>"
    _TEXT_PART_TYPES = frozenset(("text", "input_text"))
    _MEDIA_PART_TYPES = frozenset(
        (
            "image",
            "input_image",
            "image_url",
            "audio",
            "input_audio",
            "audio_url",
            "video",
            "input_video",
            "video_url",
        )
    )

    def __init__(
        self, tokenizer_pool: TokenizerPool, disable_thinking: bool = True, tool_parser: ToolCallParser | None = None
    ):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        tokenizer = tokenizer_pool.tokenizer
        self.bos_token = tokenizer.bos_token
        self.disable_thinking = disable_thinking

        self.start_turn = "<|turn>"
        self.end_turn = "<turn|>"

        self.model_turn_start = f"{self.start_turn}model\n"
        self.generation_prompt = self.model_turn_start
        if disable_thinking:
            self.generation_prompt += "<|channel>thought\n<channel|>"

    # --- helpers ---

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.end_turn, *super()._assistant_message_stop_texts())

    @staticmethod
    def _render_system_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list | tuple):
            rendered = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in Gemma4ChatTemplateParser._TEXT_PART_TYPES:
                    rendered.append((part.get("text", "") or "").strip() + " ")
                elif part_type in Gemma4ChatTemplateParser._MEDIA_PART_TYPES:
                    raise ValueError("Gemma4ChatTemplateParser is text-only; use a multimodal parser for media content.")
            return "".join(rendered)
        return ""

    @staticmethod
    def _render_tool_content(content: Any):
        """Return text-only role:tool content."""
        if isinstance(content, list | tuple):
            text = ""
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in Gemma4ChatTemplateParser._TEXT_PART_TYPES:
                    text += part.get("text") or ""
                elif part_type in Gemma4ChatTemplateParser._MEDIA_PART_TYPES:
                    raise ValueError("Gemma4ChatTemplateParser is text-only; use a multimodal parser for media content.")
            return text
        return content

    @staticmethod
    def _render_content(content: Any, role: str) -> str:
        """Render message content and reject media until Gemma4 has a multimodal parser."""
        if isinstance(content, str):
            return content if role == "model" else content.strip()
        if isinstance(content, list | tuple):
            rendered = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in Gemma4ChatTemplateParser._TEXT_PART_TYPES:
                    text = part.get("text") or ""
                    rendered.append(text if role == "model" else text.strip())
                elif part_type in Gemma4ChatTemplateParser._MEDIA_PART_TYPES:
                    raise ValueError("Gemma4ChatTemplateParser is text-only; use a multimodal parser for media content.")
            return "".join(rendered)
        return ""

    def _format_turn(self, role: str, content: str) -> str:
        return f"{self.start_turn}{role}\n{content}{self.end_turn}\n"

    # --- main parse ---

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = self.bos_token

        # --- System turn (may combine: think token + system content + tool defs) ---
        enable_thinking = not self.disable_thinking
        tools = kwargs.get("tools", None)
        first_is_system = messages and messages[0].get("role") in ("system", "developer")

        loop_messages = list(messages)

        if enable_thinking or tools or first_is_system:
            result += f"{self.start_turn}system\n"
            if enable_thinking:
                result += f"{self.THINK_TOKEN}\n"
            if first_is_system:
                result += self._render_system_content(loop_messages[0].get("content", ""))
                loop_messages = loop_messages[1:]
            if tools:
                assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: tools: {tools}\n"
                result += self.tool_parser.get_tool_system_prompt(tools)
            result += f"{self.end_turn}\n"

        last_user_idx = -1
        for idx, message in enumerate(loop_messages):
            if message.get("role") == "user":
                last_user_idx = idx

        # Track previous non-tool role for model turn continuation.
        prev_non_tool_role = None
        # Mirrors the Gemma4 Jinja namespace used by the close-turn and
        # generation-prompt guards.
        prev_message_type = None

        for idx, message in enumerate(loop_messages):
            role = message.get("role")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")

            # Tool messages are consumed by the preceding assistant block.
            if role == "tool":
                continue

            prev_message_type = None
            mapped_role = "model" if role == "assistant" else role

            # --- Model turn continuation ---
            # When a model message follows another model message (only tool
            # messages in between), suppress the duplicate <|turn>model\n
            # header. Matches Jinja's continue_same_model_turn logic.
            is_continuation = mapped_role == "model" and prev_non_tool_role == "assistant"
            if not is_continuation:
                result += f"{self.start_turn}{mapped_role}\n"

            thinking_text = message.get("reasoning") or message.get("reasoning_content")
            if thinking_text and idx > last_user_idx and tool_calls:
                result += f"<|channel>thought\n{thinking_text}\n<channel|>"

            # --- Tool calls ---
            rendered_tool_response = False
            if tool_calls:
                assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: tool_calls: {tool_calls}\n"
                tool_call_objs = [ToolCall.from_raw_tool_call(tc) for tc in tool_calls]
                result += self.tool_parser.format_tool_calls(tool_call_objs)
                prev_message_type = "tool_call"

                # --- Inline tool responses ---
                # Forward-scan consecutive role:tool messages and render them
                # as <|tool_response> blocks inside this model turn.
                for j in range(idx + 1, len(loop_messages)):
                    if loop_messages[j].get("role") != "tool":
                        break
                    follow = loop_messages[j]
                    # Resolve tool name from tool_call_id if possible.
                    t_name = follow.get("name", "unknown")
                    tool_call_id = follow.get("tool_call_id")
                    if tool_call_id:
                        for tc in tool_call_objs:
                            if tc.id == tool_call_id:
                                t_name = tc.name
                                break
                    t_content = follow.get("content", "")
                    response = self._render_tool_content(t_content)
                    result += self.tool_parser.format_tool_result(content=response, name=t_name)
                    rendered_tool_response = True
                    prev_message_type = "tool_response"

            # --- Native Gemma tool responses ---
            for tool_response in message.get("tool_responses") or []:
                assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: {tool_response}\n"
                result += self.tool_parser.format_tool_result(
                    content=tool_response.get("response"),
                    name=tool_response.get("name", "unknown"),
                )
                rendered_tool_response = True
                prev_message_type = "tool_response"

            # --- Content ---
            # For RL we preserve thinking channels in prior model turns
            # verbatim — the model's own reasoning is part of the training
            # signal.  (The Jinja template's strip_thinking is designed for
            # inference-time chat where you don't want to show the user the
            # model's internal reasoning; that doesn't apply here.)
            #
            # When thinking is DISABLED, the generation prompt appends
            # ``<|channel>thought\n<channel|>`` after ``<|turn>model\n``.
            # That prefix becomes part of the KV cache (prefix tree) but is
            # NOT included in the assistant's saved ``content``. To keep
            # historical turns prefix-cumulative with the tree, we re-inject
            # the empty thinking channel before the content of every prior
            # model turn.  We skip it for the generation-prompt turn itself
            # (that's handled by ``self.generation_prompt``).
            rendered_content = self._render_content(content, mapped_role)
            if rendered_content:
                if mapped_role == "model":
                    if self.disable_thinking:
                        result += "<|channel>thought\n<channel|>"
                    result += rendered_content
                else:
                    result += rendered_content

            has_content = bool(rendered_content.strip())

            # --- Close turn ---
            # Jinja logic: emit <turn|>\n UNLESS:
            #   - tool_call pending response (emit bare <|tool_response> instead), OR
            #   - tool response was rendered but there's no content (turn stays
            #     open for the next continuation assistant message)
            if prev_message_type == "tool_call" and not rendered_tool_response:
                result += "<|tool_response>"
            elif rendered_tool_response and not has_content:
                pass  # turn left open for model continuation
            else:
                result += f"{self.end_turn}\n"

            prev_non_tool_role = role

        # --- Generation prompt ---
        if add_generation_prompt:
            if prev_message_type not in ("tool_response", "tool_call"):
                result += self.generation_prompt

        return result


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.eot_token, *super()._assistant_message_stop_texts())

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        raise Exception("Tool calling on llama chat template not supported yet.")


class OpenAIHarmonyChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        self.system_token = "<|start|>system<|message|>"
        self.user_token = "<|start|>user<|message|>"
        self.assistant_token = "<|start|>assistant"
        self.developer_token = "<|start|>developer<|message|>"
        self.end_token = "<|end|>"
        self.return_token = "<|return|>"
        self.generation_prompt = self.assistant_token

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.return_token, self.end_token, *super()._assistant_message_stop_texts())

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = ""

        # Validate messages list is not empty
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Extract kwargs
        tools = kwargs.get("tools", None)
        builtin_tools = kwargs.get("builtin_tools", None)
        model_identity = kwargs.get("model_identity", None)
        reasoning_effort = kwargs.get("reasoning_effort", "medium")

        # ALWAYS add system message at the start
        system_message = self._build_system_message(
            builtin_tools=builtin_tools, tools=tools, model_identity=model_identity, reasoning_effort=reasoning_effort
        )
        result += self.system_token + system_message + self.end_token

        # Extract developer/system message if present and render with tools
        loop_messages = messages
        developer_message = None
        if messages and messages[0]["role"] in ["developer", "system"]:
            developer_message = messages[0]["content"]
            loop_messages = messages[1:]

        # Render developer block if we have developer message or tools
        if developer_message or tools:
            result += self.developer_token
            if developer_message:
                result += "# Instructions\n\n" + developer_message + "\n\n"
            if tools:
                assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: messages: {messages}\n"
                result += "# Tools\n\n" + self.tool_parser.get_tool_system_prompt(tools)
            result += self.end_token

        # Process remaining messages
        last_tool_call_name = None
        for i, message in enumerate(loop_messages):
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                if "tool_calls" in message and message["tool_calls"]:
                    # Handle tool calls – the template assumes max 1 tool call per message
                    tool_calls = message["tool_calls"]
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]

                    # Check if there's a future final message (inference scenario)
                    future_final_message = False
                    for future_msg in loop_messages[i + 1 :]:
                        if future_msg.get("role") == "assistant" and "tool_calls" not in future_msg:
                            future_final_message = True
                            break

                    # Insert optional analysis / thinking content before the tool call
                    # Only render if there's no future final message
                    if message.get("content") and message.get("thinking"):
                        raise ValueError(
                            "Cannot pass both content and thinking in an assistant message with tool calls!"
                        )

                    if not future_final_message:
                        if message.get("thinking"):
                            result += (
                                self.assistant_token
                                + "<|channel|>analysis<|message|>"
                                + message["thinking"]
                                + self.end_token
                            )
                        elif message.get("content"):
                            result += (
                                self.assistant_token
                                + "<|channel|>analysis<|message|>"
                                + message["content"]
                                + self.end_token
                            )

                    # Only process the first tool call (template uses tool_calls[0])
                    call = tool_calls[0]
                    tool_call = ToolCall.from_raw_tool_call(call)
                    # Harmony format_tool_call returns " to=functions.{name}...<|call|>"
                    assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: messages: {messages}\n"
                    result += self.assistant_token + self.tool_parser.format_tool_calls([tool_call])
                    last_tool_call_name = tool_call.name

                else:
                    last_tool_call_name = None
                    # Regular assistant message
                    message_content = message["content"]
                    if (
                        "<|channel|>analysis<|message|>" in message_content
                        or "<|channel|>final<|message|>" in message_content
                    ):
                        result += self.assistant_token + message_content + self.end_token
                        continue
                    is_last = i == len(loop_messages) - 1
                    if is_last and not add_generation_prompt:
                        # Training case - include thinking and use return token
                        if message.get("thinking"):
                            result += (
                                self.assistant_token
                                + "<|channel|>analysis<|message|>"
                                + message["thinking"]
                                + self.end_token
                            )
                        result += (
                            self.assistant_token
                            + "<|channel|>final<|message|>"
                            + message["content"]
                            + self.return_token
                        )
                    else:
                        # Inference case - only final message, no thinking
                        result += (
                            self.assistant_token + "<|channel|>final<|message|>" + message["content"] + self.end_token
                        )
            elif message["role"] == "tool":
                assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: messages: {messages}\n"
                if last_tool_call_name is None:
                    raise ValueError("Tool message without preceding tool call")
                # Tool responses should be JSON-encoded
                formatted = self.tool_parser.format_tool_result(content=message["content"], name=last_tool_call_name)
                result += "<|start|>" + formatted + self.end_token
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.end_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.end_token

    def parse_assistant(self, message):
        return self.assistant_token + "<|channel|>final<|message|>" + message["content"] + self.end_token

    def _build_system_message(self, builtin_tools=None, tools=None, model_identity=None, reasoning_effort="medium"):
        """Build the system message for OpenAI Harmony format."""
        if model_identity is None:
            model_identity = "You are ChatGPT, a large language model trained by OpenAI."

        system_parts = [
            model_identity,
            "Knowledge cutoff: 2024-06",
            f"Current date: {self._get_current_date()}",
            "",
            f"Reasoning: {reasoning_effort}",
            "",
        ]

        # Add builtin tools documentation if provided
        if builtin_tools:
            assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: builtin_tools: {builtin_tools}\n"
            system_parts.append(self.tool_parser.get_tool_system_prompt([{"_builtin": builtin_tools}]))

        # Add channel instructions
        channel_instruction = (
            "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        )
        if tools:
            channel_instruction += "\nCalls to these tools must go to the commentary channel: 'functions'."
        system_parts.append(channel_instruction)

        return "\n".join(system_parts)

    def _get_current_date(self):
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")


class MoonlightChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer_pool: TokenizerPool, tool_parser: ToolCallParser | None = None):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        tokenizer = tokenizer_pool.tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>"
        self.system_token = "<|im_system|>system<|im_middle|>"
        self.user_token = "<|im_user|>user<|im_middle|>"
        self.assistant_token = "<|im_assistant|>assistant<|im_middle|>"
        self.generation_prompt = self.assistant_token

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.eot_token, *super()._assistant_message_stop_texts())

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = ""
        if not messages:
            return result

        # if the first message is not a system message, add the system message
        if messages[0]["role"] != "system":
            result += self.system_token + "You are a helpful assistant provided by Moonshot-AI." + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message):
        raise Exception("Moonlight does not support tools")


class GlmChatTemplateParser(ChatTemplateParser):
    """
    A chat template parser for GLM (ChatGLM) models.

    This parser formats messages according to the GLM chat template format,
    which includes support for thinking blocks, tool calls, and tool responses.
    """

    def __init__(
        self, tokenizer_pool: TokenizerPool, disable_thinking=False, tool_parser: ToolCallParser | None = None
    ):
        super().__init__(tokenizer_pool, tool_parser=tool_parser)
        self.bos_token = "[gMASK]<sop>"
        self.system_token = "<|system|>\n"
        self.user_token = "<|user|>\n"
        self.assistant_token = "<|assistant|>"
        self.observation_token = "<|observation|>"
        self.disable_thinking = disable_thinking

        # Generation prompt - includes empty think block if thinking is disabled
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "\n<think></think>"
        else:
            self.generation_prompt = self.assistant_token

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = self.bos_token

        # Check if tools are provided
        tools = kwargs.get("tools", None)

        # Add tool system message if tools are present
        if tools:
            assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: tools: {tools}\n"
            result += self.system_token + self.tool_parser.get_tool_system_prompt(tools)

        # Track the last user message index for reasoning content handling
        last_user_index = -1
        for i, message in enumerate(messages):
            if message["role"] == "user":
                last_user_index = i

        # Process messages
        for i, message in enumerate(messages):
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, i, last_user_index)
            elif message["role"] == "tool":
                result += self.parse_tool(message, i, messages)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def _extract_visible_text(self, content):
        """Extract visible text from content (handle both string and structured content)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        else:
            return str(content)

    def parse_system(self, message):
        content = self._extract_visible_text(message["content"])
        return self.system_token + content

    def parse_user(self, message):
        content = self._extract_visible_text(message["content"])
        # Add /nothink if thinking is disabled and not already present
        if self.disable_thinking and not content.endswith("/nothink"):
            content += "/nothink"
        return self.user_token + content

    def parse_assistant(self, message, current_index, last_user_index):
        result = self.assistant_token

        # Extract reasoning content and regular content
        content = self._extract_visible_text(message["content"])
        reasoning_content = ""

        # Check if reasoning_content is explicitly provided
        if "reasoning_content" in message and isinstance(message["reasoning_content"], str):
            reasoning_content = message["reasoning_content"]
        else:
            # Try to extract from content with <think> tags
            if "</think>" in content:
                parts = content.split("</think>")
                think_parts = parts[0].rstrip("\n").split("<think>")
                if len(think_parts) > 1:
                    reasoning_content = think_parts[-1].lstrip("\n")
                    content = parts[-1].lstrip("\n")

        # Add reasoning content if we're past the last user message
        if current_index > last_user_index and reasoning_content:
            result += "\n<think>" + reasoning_content.strip() + "</think>"
        else:
            result += "\n<think></think>"

        # Add content if present
        if content.strip():
            result += "\n" + content.strip()

        # Handle tool calls
        if "tool_calls" in message and message["tool_calls"]:
            raw_tool_calls = message["tool_calls"]
            assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: tool_calls: {raw_tool_calls}\n"

            tool_calls = [ToolCall.from_raw_tool_call(raw) for raw in raw_tool_calls]
            result += "\n" + self.tool_parser.format_tool_calls(tool_calls)

        return result

    def parse_tool(self, message, current_index, messages):
        """Parse tool/observation messages with grouping."""
        assert self.tool_parser, f"Tool parser shouldn't be None when tools exist: message: {message}\n"

        content = message.get("content", "")

        result = ""

        # Handle string content
        if isinstance(content, str):
            # Check if this is the first tool message or if the previous message is not a tool
            is_first_tool = current_index == 0 or messages[current_index - 1]["role"] != "tool"
            if is_first_tool:
                result += self.observation_token
            result += "\n" + self.tool_parser.format_tool_result(content=content)
        # Handle list of tool responses
        elif isinstance(content, list):
            # For list content, ALWAYS add observation token (matches Jinja template behavior)
            result += self.observation_token
            for tr in content:
                if isinstance(tr, dict):
                    output = tr.get("output", tr)
                else:
                    output = tr
                result += "\n" + self.tool_parser.format_tool_result(content=str(output))
        else:
            # Fallback for other content types - treat as string
            is_first_tool = current_index == 0 or messages[current_index - 1]["role"] != "tool"
            if is_first_tool:
                result += self.observation_token
            result += "\n" + self.tool_parser.format_tool_result(content=str(content))

        return result


########################
###### MultiModal ######
########################
_IMAGE_TAG = "<image>"
_VIDEO_TAG = "<video>"

_LEGACY_SPLIT_RE = re.compile(r"(<image>|<video>)")

_TEXT_TYPES = {"text", "input_text"}
_IMAGE_TYPES = {"image", "input_image", "image_url"}
_VIDEO_TYPES = {"video", "input_video", "video_url"}
MM_PART_TYPES = frozenset(_IMAGE_TYPES | _VIDEO_TYPES)


def _hash_mm_content(data) -> str:
    """Fast content hash for MM change detection.  NOT cryptographic.

    Samples a few pixels from different positions to detect content changes
    without hashing the entire image.  Falls back to ``id()`` for non-image types.
    """
    try:
        # PIL.Image → sample corner + center pixels
        if hasattr(data, "size") and hasattr(data, "mode"):
            import numpy as np

            h = hashlib.md5(usedforsecurity=False)
            h.update(f"{data.size}:{data.mode}".encode())
            arr = np.asarray(data)
            if arr.size > 0:
                for pos in [(0, 0), (-1, -1), (arr.shape[0] // 2, arr.shape[1] // 2)]:
                    h.update(arr[pos].tobytes())
            return h.hexdigest()
        # String path / URL
        if isinstance(data, str):
            return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()
    except Exception:
        pass
    return str(id(data))


class MultiModalChatTemplateParser(ChatTemplateParser):
    # Tokenization can be called multiple times per request; higher layers may
    # add caching or offload this work to a dedicated preprocessing service.
    """
    Users pass `messages` only.
    Inline multimodal is preferred:
        {"type":"image","image": ...}
        {"type":"image_url","image_url": ...}
        {"type":"video","video": ...}
        {"type":"video_url","video_url": ...}
    Legacy text placeholders supported:
        ".... <image> ...."
      but requires explicit images=/videos= kwargs.
    """

    def __init__(
        self,
        tokenizer_pool: TokenizerPool,
        processor: AutoProcessor,
        tool_parser: ToolCallParser | None = None,
    ):
        super().__init__(tokenizer_pool=tokenizer_pool, tool_parser=tool_parser)
        self.processor = processor

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        norm = self._normalize_messages(
            messages, allow_legacy_placeholders=kwargs.get("allow_legacy_placeholders", True)
        )
        # Prefer processor chat template if available (often correct for VLMs)
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                norm,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

        return self.tokenizer.apply_chat_template(
            norm,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _normalize_messages(
        self, messages: list[dict[str, Any]], allow_legacy_placeholders=True
    ) -> list[dict[str, Any]]:
        """
        Normalization:
        - Validates basic schema.
        - If content is a string with <image>/<video> and legacy enabled -> convert to list[parts].
        - If content is list[parts] -> validate parts and lightly normalize text parts.
        - If content is a string without tags -> keep string.

        Output guarantees:
        - Each message has: {"role": str, "content": str | list[dict]}
        - If content is list, each part is a dict with at least {"type": str}
        - Text parts are canonicalized to {"type":"text","text": str}
        """
        if not isinstance(messages, list):
            raise TypeError(f"messages must be a list, got {type(messages)}")

        out: list[dict[str, Any]] = []
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                raise TypeError(f"messages[{i}] must be a dict, got {type(m)}")

            role = m.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"messages[{i}]['role'] must be a non-empty str, got {role!r}")

            content = m.get("content", "")

            # string content
            if isinstance(content, str):
                if (_IMAGE_TAG in content) or (_VIDEO_TAG in content):
                    if not allow_legacy_placeholders:
                        raise ValueError("Legacy <image>/<video> placeholders are disabled.")
                    segs = [s for s in _LEGACY_SPLIT_RE.split(content) if s]
                    parts: list[dict[str, Any]] = []
                    for s in segs:
                        if s == _IMAGE_TAG:
                            parts.append({"type": "image"})
                        elif s == _VIDEO_TAG:
                            parts.append({"type": "video"})
                        else:
                            parts.append({"type": "text", "text": s})
                    out.append({"role": role, "content": parts})
                else:
                    out.append({"role": role, "content": content})
                continue

            # list-of-parts content
            if isinstance(content, list):
                norm_parts: list[dict[str, Any]] = []
                for j, part in enumerate(content):
                    if not isinstance(part, dict):
                        raise TypeError(f"messages[{i}]['content'][{j}] must be a dict part, got {type(part)}")
                    t = part.get("type")
                    if not isinstance(t, str) or not t:
                        raise ValueError(f"messages[{i}]['content'][{j}] missing/invalid 'type': {t!r}")

                    if t in _TEXT_TYPES:
                        txt = part.get("text", "")
                        if txt is None:
                            txt = ""
                        if not isinstance(txt, str):
                            raise TypeError(f"messages[{i}]['content'][{j}]['text'] must be str, got {type(txt)}")
                        norm_parts.append({"type": "text", "text": txt})
                        continue

                    if t in MM_PART_TYPES:
                        # keep payload inline for later binding
                        norm_parts.append(part)
                        continue

                    raise ValueError(f"Unknown content part type: {t!r} at messages[{i}]['content'][{j}]")

                out.append({"role": role, "content": norm_parts})
                continue

            raise TypeError(f"Unsupported content type at messages[{i}]['content']: {type(content)}")

        return out

    def _count_placeholders(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Counts image/video placeholders across the conversation.

        IMPORTANT: counts 'image_url' as an image placeholder and 'video_url' as a video placeholder.
        """
        need_images = 0
        need_videos = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise TypeError(f"Expected str or list for content, got {type(content)}")
            for part in content:
                t = part.get("type")
                if t in _IMAGE_TYPES:
                    need_images += 1
                elif t in _VIDEO_TYPES:
                    need_videos += 1
        return need_images, need_videos

    def get_mm_regions(self, input_key: str, messages: list[dict]) -> list[tuple]:
        """Find multimodal placeholder regions in the rendered ``input_key``.

        Walks through normalized messages to match placeholders with their
        bound image/video data, then locates the placeholder text spans in
        ``input_key``.

        Returns:
            List of ``(text_start, text_end, modality, content_hash, data)``
            where ``text_start``/``text_end`` are character offsets into
            ``input_key`` bounding the placeholder string (e.g.
            ``<|vision_start|><|image_pad|><|vision_end|>``).
        """
        norm = self._normalize_messages(messages, allow_legacy_placeholders=True)
        need_images, need_videos = self._count_placeholders(norm)
        if need_images == 0 and need_videos == 0:
            return []

        # Collect MM payloads in placeholder order (same logic as tokenize())
        mm_payloads: list[tuple[str, Any]] = []  # (modality, raw_data)
        for m in norm:
            content = m.get("content", "")
            if isinstance(content, str):
                continue
            for part in content:
                t = part.get("type")
                if t in _IMAGE_TYPES:
                    raw = part.get("image") or part.get("image_url") or part.get("input_image")
                    mm_payloads.append(("image", self._unwrap_url_field(raw) if raw else None))
                elif t in _VIDEO_TYPES:
                    raw = part.get("video") or part.get("video_url") or part.get("input_video")
                    mm_payloads.append(("video", self._unwrap_url_field(raw) if raw else None))

        # Find placeholder positions in input_key.
        # Subclasses define the placeholder patterns via _get_mm_placeholder_patterns().
        regions = []
        search_from = 0
        for modality, data in mm_payloads:
            patterns = self._get_mm_placeholder_patterns(modality)
            found = False
            for pattern in patterns:
                idx = input_key.find(pattern, search_from)
                if idx != -1:
                    chash = _hash_mm_content(data) if data is not None else ""
                    regions.append((idx, idx + len(pattern), modality, chash, data))
                    search_from = idx + len(pattern)
                    found = True
                    break
            if not found:
                # Placeholder not found in input_key — stop matching.
                # Remaining payloads can't be located.
                break

        return regions

    def _get_mm_placeholder_patterns(self, modality: str) -> list[str]:
        """Return placeholder text patterns for a given modality.

        Subclasses should override for model-specific patterns.
        Returns a list of patterns to try in order.
        """
        # Generic fallback — won't match any real template.
        if modality == "image":
            return ["<image>"]
        elif modality == "video":
            return ["<video>"]
        return []

    @staticmethod
    def _unwrap_url_field(x: Any) -> Any:
        """
        Unwrap OpenAI-ish shapes:
          {"url": "..."} -> "..."
        Or return the value as-is (string/path/etc).
        """
        if isinstance(x, dict):
            return x.get("url") or x.get("path") or x.get("uri")
        return x

    def _resolve_image_payloads(self, payloads: list[Any], image_patch_size: int) -> list[Any]:
        # Keep multimodal payload resolution centralized so faster processors can
        # replace this path without changing template parsing semantics.
        return [process_image(x, image_patch_size=image_patch_size) for x in payloads]

    def _resolve_video_payloads(self, payloads: list[Any], image_patch_size: int) -> tuple[list[Any], Any]:
        videos, video_metadata = zip(
            *[
                process_video(video, image_patch_size=image_patch_size, return_video_metadata=True)
                for video in payloads
            ],
            strict=True,
        )
        videos = list(videos)
        video_metadata = list(video_metadata)
        videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

        # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
        # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
        return [
            (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
        ], videos_kwargs

    async def tokenize(
        self,
        input_text: str,
        ctx: TokenizeContext | None = None,
    ) -> TokenizeOutput:
        """
        Returns: (token_ids, multi_modal_data)

        Binding rules
        - Walk placeholders in the exact order they appear across `messages`.
        - For each placeholder:
            1) Prefer inline payload fields:
               - image: part["image"]
               - image_url: part["image_url"] (string or {"url": ...})
               - video: part["video"]
               - video_url: part["video_url"] (string or {"url": ...})
            2) If none, consume from legacy `images=` / `videos=` iterators, in order.
        - Resolve each bound payload via process_image/process_video into processor-ready objects.
        - Call processor(text=[input_text], images=..., videos=...).

        Notes:
        - If messages is None -> text-only.
        - If no placeholders -> text-only.
        """

        if not isinstance(input_text, str):
            raise TypeError(f"input_text must be a str, got {type(input_text)}")

        if ctx is None or ctx.messages is None:
            # fallback: treat as text-only
            return super().tokenize(input_text, ctx=None)

        messages = ctx.messages
        images = ctx.images
        videos = ctx.videos

        norm = self._normalize_messages(messages, ctx.allow_legacy_placeholders)
        need_images, need_videos = self._count_placeholders(norm)
        if need_images == 0 and need_videos == 0:
            ids, strs = await self.tokenizer_pool.encode_with_strs(input_text, add_special_tokens=False)
            return TokenizeOutput(token_ids=ids, token_strs=strs)

        if self.processor is None:
            raise ValueError("Multimodal placeholders found but processor is None.")

        img_it = iter(images or [])
        vid_it = iter(videos or [])

        bound_images: list[Any] = []
        bound_videos: list[Any] = []

        # 1) Bind raw payload refs in placeholder order
        for m in norm:
            content = m.get("content", "")
            if isinstance(content, str):
                continue

            for part in content:
                t = part["type"]

                if t in _IMAGE_TYPES:
                    payload = None
                    if part.get("image") is not None:
                        payload = part["image"]
                    elif part.get("image_url") is not None:
                        payload = self._unwrap_url_field(part["image_url"])
                    else:
                        try:
                            payload = next(img_it)
                        except StopIteration:
                            raise ValueError(
                                f"Missing image payload: need {need_images} image(s) total, but ran out of images=. messages: {messages}"
                            ) from None
                    bound_images.append(payload)

                elif t in _VIDEO_TYPES:
                    payload = None
                    if part.get("video") is not None:
                        payload = part["video"]
                    elif part.get("video_url") is not None:
                        payload = self._unwrap_url_field(part["video_url"])
                    else:
                        try:
                            payload = next(vid_it)
                        except StopIteration:
                            raise ValueError(
                                f"Missing video payload: need {need_videos} video(s) total, but ran out of videos=. messages: {messages}"
                            ) from None
                    bound_videos.append(payload)

        if len(bound_images) != need_images:
            raise ValueError(
                f"Image binding mismatch: need_images={need_images}, bound_images={len(bound_images)}. messages: {messages}"
            )
        if len(bound_videos) != need_videos:
            raise ValueError(
                f"Video binding mismatch: need_videos={need_videos}, bound_videos={len(bound_videos)}. messages: {messages}"
            )

        try:
            next(img_it)
            raise ValueError(f"Unused images= payload: more images provided than placeholders. messages: {messages}")
        except StopIteration:
            pass
        try:
            next(vid_it)
            raise ValueError(f"Unused videos= payload: more videos provided than placeholders. messages: {messages}")
        except StopIteration:
            pass

        # Resolve into processor-ready objects
        resolved_images = self._resolve_image_payloads(bound_images, ctx.image_patch_size) if need_images > 0 else None
        resolved_videos, videos_kwargs = None, {}
        if need_videos > 0:
            resolved_videos, videos_kwargs = self._resolve_video_payloads(bound_videos, ctx.image_patch_size)

        call_kwargs = dict(ctx.processor_kwargs) if ctx.processor_kwargs else {}
        if resolved_videos is not None and videos_kwargs:
            call_kwargs["videos_kwargs"] = videos_kwargs

        # Multimodal tokenization is via processor
        model_inputs = self.processor(
            text=[input_text],
            images=resolved_images,
            videos=resolved_videos,
            return_tensors="pt",
            **call_kwargs,
        )
        if "input_ids" not in model_inputs:
            raise RuntimeError("Processor did not return input_ids; incompatible processor/model.")

        token_ids = model_inputs["input_ids"].squeeze(0).tolist()

        # Build per-token strings via decode.  The processor expands image/video
        # placeholders into pad tokens so offset-based slicing cannot work here.
        # These strings won't join back to input_text (token count differs from
        # character count due to expansion), but they are still useful as trie
        # node metadata for any text tokens in the sequence.
        tokenizer = self.tokenizer_pool.tokenizer
        token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]

        mm: MultiModalData = MultiModalData()
        if need_images > 0:
            mm.image = resolved_images
        if need_videos > 0:
            mm.video = resolved_videos

        # Make sure we have enough context to recreate model_inputs later
        mm.processor_kwargs = call_kwargs

        return TokenizeOutput(token_ids=token_ids, token_strs=token_strs, multi_modal_data=mm)


class QwenVLChatTemplateParser(MultiModalChatTemplateParser):
    def __init__(
        self,
        tokenizer_pool: TokenizerPool,
        processor,
        disable_thinking=True,
        add_vision_id=False,
        tool_parser: ToolCallParser | None = None,
    ):
        super().__init__(tokenizer_pool, processor, tool_parser)
        _init_qwen_tokens(self, tokenizer_pool, disable_thinking)
        self.add_vision_id = add_vision_id

    def _assistant_message_stop_texts(self) -> tuple[str, ...]:
        return (self.eot_token, *super()._assistant_message_stop_texts())

    def parse(self, messages, add_generation_prompt=False, **kwargs) -> str:
        result = ""
        if len(messages) == 0:
            return result

        messages = self._normalize_messages(
            messages, allow_legacy_placeholders=kwargs.get("allow_legacy_placeholders", True)
        )

        # if the first message is not a system message, add the system message
        if messages[0]["role"] != "system" and messages[0]["role"] != "skip":
            result += (
                self.system_token
                + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                + self.eot_token
            )

        image_counter = 0
        video_counter = 0

        for message in messages:
            if result:
                # add newline to non-empty result
                result += "\n"
            content, image_counter, video_counter = self.parse_content(message["content"], image_counter, video_counter)
            if message["role"] == "system":
                result += self.system_token + content + self.eot_token
            elif message["role"] == "user":
                result += self.user_token + content + self.eot_token
            elif message["role"] == "assistant":
                result += self.assistant_token + content + self.eot_token
            elif message["role"] == "skip":
                result += message["content"]
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt and messages[-1]["role"] != "skip":
            result += "\n" + self.generation_prompt
        return result

    def parse_content(self, content, image_counter, video_counter):
        if isinstance(content, str):
            return content, image_counter, video_counter

        if not isinstance(content, list):
            raise TypeError(f"Expected str or list for content, got {type(content)}")

        result = ""

        for part in content:
            t = part.get("type")
            if t in _TEXT_TYPES:
                result += part.get("text", "")

            elif t in _IMAGE_TYPES:
                image_counter += 1
                if self.add_vision_id:
                    result += f"Picture {image_counter}"
                result += "<|vision_start|><|image_pad|><|vision_end|>"

            elif t in _VIDEO_TYPES:
                video_counter += 1
                if self.add_vision_id:
                    result += f"Video {video_counter}"
                result += "<|vision_start|><|video_pad|><|vision_end|>"

            else:
                raise ValueError(f"Unknown content part type: {t!r}")
        return result, image_counter, video_counter

    def _get_mm_placeholder_patterns(self, modality: str) -> list[str]:
        """Qwen VL pad-only placeholder patterns.

        Returns only the pad token text (not vision_start/end wrappers)
        because the tree walk handles start/end as normal text tokens
        and only needs the pad region for expansion skipping.
        """
        if modality == "image":
            return ["<|image_pad|>"]
        elif modality == "video":
            return ["<|video_pad|>"]
        return []
