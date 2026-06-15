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
from unittest.mock import MagicMock

import pytest

from axon.engine.chat_template.parser import (
    DeepseekQwenChatTemplateParser,
    Gemma4ChatTemplateParser,
    GemmaChatTemplateParser,
    GlmChatTemplateParser,
    LlamaChatTemplateParser,
    MoonlightChatTemplateParser,
    MultiModalChatTemplateParser,
    QwenChatTemplateParser,
)
from axon.tools.parsers.base_parser import get_tool_call_parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer_pool(bos_token="<bos>", eos_token="<eos>", name_or_path="mock-model"):
    tokenizer = MagicMock()
    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token
    tokenizer.eos_token_id = 2
    tokenizer.name_or_path = name_or_path
    tokenizer.__class__.__name__ = "MockTokenizer"

    pool = MagicMock()
    pool.tokenizer = tokenizer
    return pool


# ---------------------------------------------------------------------------
# DeepseekQwenChatTemplateParser
# ---------------------------------------------------------------------------


class TestDeepseekQwenChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool(bos_token="<bos>", eos_token="<eos>")
        return DeepseekQwenChatTemplateParser(pool)

    def test_simple_user_message(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        assert result == "<bos><\uff5cUser\uff5c>Hello"

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = parser.parse(messages)
        assert result == "<bos>Be helpful.<\uff5cUser\uff5c>Hi<\uff5cAssistant\uff5c>Hello!<eos>"

    def test_add_generation_prompt_true(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<\uff5cAssistant\uff5c><think>\n")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert not result.endswith("<\uff5cAssistant\uff5c><think>\n")
        assert result == "<bos><\uff5cUser\uff5c>Hi"

    def test_starts_with_bos(self, parser):
        messages = [{"role": "user", "content": "test"}]
        result = parser.parse(messages)
        assert result.startswith("<bos>")

    def test_assistant_message_ends_with_eos(self, parser):
        messages = [{"role": "assistant", "content": "bye"}]
        result = parser.parse(messages)
        assert "<\uff5cAssistant\uff5c>bye<eos>" in result

    def test_unsupported_role_raises(self, parser):
        messages = [{"role": "tool", "content": "data"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            parser.parse(messages)

    def test_multiple_turns(self, parser):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert "<\uff5cUser\uff5c>Q1" in result
        assert "<\uff5cAssistant\uff5c>A1<eos>" in result
        assert "<\uff5cUser\uff5c>Q2" in result
        assert result.endswith("<\uff5cAssistant\uff5c><think>\n")


# ---------------------------------------------------------------------------
# QwenChatTemplateParser
# ---------------------------------------------------------------------------


class TestQwenChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool(bos_token=None, eos_token="<|endoftext|>")
        return QwenChatTemplateParser(pool, disable_thinking=False)

    @pytest.fixture()
    def parser_thinking_disabled(self):
        pool = _make_tokenizer_pool(bos_token=None, eos_token="<|endoftext|>")
        return QwenChatTemplateParser(pool, disable_thinking=True)

    def test_simple_user_message_inserts_default_system(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        assert "<|im_start|>system\n" in result
        assert "You are Qwen" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result

    def test_explicit_system_message(self, parser):
        messages = [
            {"role": "system", "content": "Custom system."},
            {"role": "user", "content": "Hi"},
        ]
        result = parser.parse(messages)
        assert "<|im_start|>system\nCustom system.<|im_end|>" in result
        assert "You are Qwen" not in result

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = parser.parse(messages)
        assert "<|im_start|>system\nSys<|im_end|>" in result
        assert "<|im_start|>user\nQ<|im_end|>" in result
        assert "<|im_start|>assistant\nA<|im_end|>" in result

    def test_add_generation_prompt_true(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<|im_start|>assistant\n")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert not result.endswith("<|im_start|>assistant\n")

    def test_disable_thinking_adds_think_block(self, parser_thinking_disabled):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser_thinking_disabled.parse(messages, add_generation_prompt=True)
        assert "<think>\n\n</think>\n\n" in result

    def test_thinking_enabled_no_think_block(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert "<think>" not in result

    def test_empty_messages_returns_empty(self, parser):
        result = parser.parse([])
        assert result == ""

    def test_unsupported_role_raises(self, parser):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "unknown", "content": "bad"},
        ]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            parser.parse(messages)

    def test_newlines_between_messages(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = parser.parse(messages)
        # Consecutive messages are separated by newline
        assert "<|im_end|>\n<|im_start|>user" in result
        assert "<|im_end|>\n<|im_start|>assistant" in result

    def test_skip_role_passes_content_through(self, parser):
        messages = [{"role": "skip", "content": "RAW_TEXT"}]
        result = parser.parse(messages)
        assert "RAW_TEXT" in result

    def test_assistant_message_content_strips_trailing_eos_token(self, parser):
        assert parser.assistant_message_content("Answer<|im_end|>") == "Answer"

    def test_assistant_non_string_content_treated_as_empty(self, parser):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "assistant", "content": None},
        ]
        result = parser.parse(messages)
        assert "<|im_start|>assistant\n<|im_end|>" in result


# ---------------------------------------------------------------------------
# GemmaChatTemplateParser
# ---------------------------------------------------------------------------


class TestGemmaChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool(bos_token="<bos>", eos_token="<eos>")
        return GemmaChatTemplateParser(pool)

    def test_simple_user_message(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        assert result == "<bos><start_of_turn>user\nHello<end_of_turn>\n"

    def test_system_treated_as_user(self, parser):
        messages = [{"role": "system", "content": "Be helpful."}]
        result = parser.parse(messages)
        assert "<start_of_turn>user\nBe helpful.<end_of_turn>\n" in result

    def test_tool_treated_as_user(self, parser):
        messages = [{"role": "tool", "content": "result data"}]
        result = parser.parse(messages)
        assert "<start_of_turn>user\nresult data<end_of_turn>\n" in result

    def test_assistant_is_model(self, parser):
        messages = [{"role": "assistant", "content": "Sure!"}]
        result = parser.parse(messages)
        assert "<start_of_turn>model\nSure!<end_of_turn>\n" in result

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = parser.parse(messages)
        expected = (
            "<bos>"
            "<start_of_turn>user\nSystem msg<end_of_turn>\n"
            "<start_of_turn>user\nQuestion<end_of_turn>\n"
            "<start_of_turn>model\nAnswer<end_of_turn>\n"
        )
        assert result == expected

    def test_add_generation_prompt_true(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<start_of_turn>model\n")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert not result.endswith("<start_of_turn>model\n")
        assert result.endswith("<end_of_turn>\n")

    def test_starts_with_bos(self, parser):
        messages = [{"role": "user", "content": "test"}]
        result = parser.parse(messages)
        assert result.startswith("<bos>")

    def test_unsupported_role_raises(self, parser):
        messages = [{"role": "developer", "content": "bad"}]
        with pytest.raises(ValueError, match="Unsupported message role"):
            parser.parse(messages)

    def test_empty_content(self, parser):
        messages = [{"role": "user", "content": ""}]
        result = parser.parse(messages)
        assert "<start_of_turn>user\n<end_of_turn>\n" in result


# ---------------------------------------------------------------------------
# Gemma4ChatTemplateParser
# ---------------------------------------------------------------------------


class TestGemma4ChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool(bos_token="<bos>", eos_token="<eos>")
        return Gemma4ChatTemplateParser(pool, disable_thinking=True, tool_parser=get_tool_call_parser("gemma4"))

    def test_simple_generation_prompt_disables_thinking(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

    def test_assistant_message_content_strips_trailing_turn_markers(self, parser):
        assert parser.assistant_message_content("Action: ```Up```<turn|>") == "Action: ```Up```"
        assert parser.assistant_message_content("Action: ```Up```<turn|><turn|>\n") == "Action: ```Up```"
        assert parser.assistant_message_content("Action: <turn|> inside") == "Action: <turn|> inside"

    def test_cleaned_history_renders_single_turn_boundary_before_next_user(self, parser):
        cleaned = parser.assistant_message_content("Action: ```Up```<turn|>")
        messages = [
            {"role": "user", "content": "obs 1"},
            {"role": "assistant", "content": cleaned},
            {"role": "user", "content": "obs 2"},
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert "Action: ```Up```<turn|><turn|>" not in result
        assert "Action: ```Up```<turn|>\n<|turn>user\nobs 2<turn|>" in result

    def test_tool_response_no_content_keeps_model_turn_open(self, parser):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert "<|tool_call>call:get_weather{city:<|\"|>SF<|\"|>}<tool_call|>" in result
        assert "<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>" in result
        assert result.endswith("<tool_response|>")
        assert not result.endswith("<tool_response|><|turn>model\n<|channel>thought\n<channel|>")

    def test_tool_response_no_content_continues_model_turn(self, parser):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]
        result = parser.parse(messages)
        assert result.count("<|turn>model\n") == 1
        assert "<tool_response|><|channel>thought\n<channel|>It is sunny.<turn|>\n" in result

    def test_pending_tool_call_with_content_emits_response_marker(self, parser):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                    }
                ],
            },
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("Checking.<|tool_response>")
        assert "<turn|>\n<|turn>model" not in result[result.index("<|tool_call>") :]

    def test_tool_parser_extracts_gemma_tool_calls(self, parser):
        response = "Let me check.<|tool_call>call:get_weather{city:<|\"|>SF<|\"|>,count:2}<tool_call|>"
        tool_calls, remaining = parser.tool_parser.parse(response)
        assert remaining == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"city": "SF", "count": 2}

    def test_tool_parser_drops_pending_tool_response_marker(self, parser):
        response = "<|tool_call>call:get_weather{city:<|\"|>SF<|\"|>}<tool_call|><|tool_response>"
        tool_calls, remaining = parser.tool_parser.parse(response)
        assert remaining == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"city": "SF"}

    def test_native_tool_responses_are_rendered(self, parser):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_responses": [{"name": "search", "response": {"result": "ok"}}],
            }
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result == "<bos><|turn>model\n<|tool_response>response:search{result:<|\"|>ok<|\"|>}<tool_response|>"

    def test_text_content_parts_render_as_text(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                ],
            }
        ]
        result = parser.parse(messages)
        assert result == "<bos><|turn>user\nLook<turn|>\n"

    def test_media_content_requires_multimodal_parser(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                    {"type": "image", "image": "image-data"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="text-only"):
            parser.parse(messages)

    def test_tools_render_gemma_declarations(self, parser):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name."},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = parser.parse([{"role": "user", "content": "Hi"}], tools=tools)
        assert "<|turn>system\n<|tool>declaration:get_weather" in result
        assert 'description:<|"|>Get weather.<|"|>' in result
        assert "parameters:{properties:{city:" in result
        assert 'city:{description:<|"|>City name.<|"|>,type:<|"|>STRING<|"|>}' in result
        assert 'required:[<|"|>city<|"|>]' in result
        assert result.endswith("<turn|>\n<|turn>user\nHi<turn|>\n")


# ---------------------------------------------------------------------------
# LlamaChatTemplateParser
# ---------------------------------------------------------------------------


class TestLlamaChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool()
        return LlamaChatTemplateParser(pool)

    def test_simple_user_message(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        expected = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        assert result == expected

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey!"},
        ]
        result = parser.parse(messages)
        assert "<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>" in result
        assert "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>\n\nHey!<|eot_id|>" in result

    def test_add_generation_prompt_true(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert result.endswith("<|eot_id|>")

    def test_starts_with_bos(self, parser):
        messages = [{"role": "user", "content": "test"}]
        result = parser.parse(messages)
        assert result.startswith("<|begin_of_text|>")

    def test_tool_role_raises(self, parser):
        messages = [{"role": "tool", "content": "data"}]
        with pytest.raises(Exception, match="Tool calling on llama chat template not supported"):
            parser.parse(messages)

    def test_unsupported_role_raises(self, parser):
        messages = [{"role": "developer", "content": "bad"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            parser.parse(messages)

    def test_multiple_turns(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        # Verify all messages are present in order
        sys_pos = result.index("Sys")
        q1_pos = result.index("Q1")
        a1_pos = result.index("A1")
        q2_pos = result.index("Q2")
        assert sys_pos < q1_pos < a1_pos < q2_pos
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


# ---------------------------------------------------------------------------
# MoonlightChatTemplateParser
# ---------------------------------------------------------------------------


class TestMoonlightChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool(bos_token="<bos>", eos_token="<eos>")
        return MoonlightChatTemplateParser(pool)

    def test_simple_user_message_adds_default_system(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        assert "<|im_system|>system<|im_middle|>" in result
        assert "You are a helpful assistant provided by Moonshot-AI." in result
        assert "<|im_user|>user<|im_middle|>Hello<|im_end|>" in result

    def test_explicit_system_message(self, parser):
        messages = [
            {"role": "system", "content": "My system."},
            {"role": "user", "content": "Hi"},
        ]
        result = parser.parse(messages)
        assert "<|im_system|>system<|im_middle|>My system.<|im_end|>" in result
        assert "Moonshot-AI" not in result

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = parser.parse(messages)
        assert "<|im_system|>system<|im_middle|>Sys<|im_end|>" in result
        assert "<|im_user|>user<|im_middle|>Q<|im_end|>" in result
        assert "<|im_assistant|>assistant<|im_middle|>A<|im_end|>" in result

    def test_add_generation_prompt_true(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<|im_assistant|>assistant<|im_middle|>")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert not result.endswith("<|im_assistant|>assistant<|im_middle|>")

    def test_empty_messages_returns_empty(self, parser):
        result = parser.parse([])
        assert result == ""

    def test_tool_role_raises(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "tool", "content": "data"},
        ]
        with pytest.raises(Exception, match="Moonlight does not support tools"):
            parser.parse(messages)

    def test_unsupported_role_raises(self, parser):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "developer", "content": "bad"},
        ]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            parser.parse(messages)

    def test_no_default_system_when_system_first(self, parser):
        messages = [
            {"role": "system", "content": "Custom"},
            {"role": "user", "content": "Q"},
        ]
        result = parser.parse(messages)
        # Should only have one system block
        assert result.count("<|im_system|>") == 1


# ---------------------------------------------------------------------------
# GlmChatTemplateParser
# ---------------------------------------------------------------------------


class TestGlmChatTemplateParser:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool()
        return GlmChatTemplateParser(pool, disable_thinking=False)

    @pytest.fixture()
    def parser_no_think(self):
        pool = _make_tokenizer_pool()
        return GlmChatTemplateParser(pool, disable_thinking=True)

    def test_simple_user_message(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser.parse(messages)
        assert result.startswith("[gMASK]<sop>")
        assert "<|user|>\nHello" in result

    def test_system_user_assistant_conversation(self, parser):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = parser.parse(messages)
        assert "<|system|>\nBe brief." in result
        assert "<|user|>\nQ" in result
        assert "<|assistant|>" in result
        assert "A" in result

    def test_starts_with_bos(self, parser):
        messages = [{"role": "user", "content": "test"}]
        result = parser.parse(messages)
        assert result.startswith("[gMASK]<sop>")

    def test_add_generation_prompt_true_thinking_enabled(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=True)
        assert result.endswith("<|assistant|>")

    def test_add_generation_prompt_true_thinking_disabled(self, parser_no_think):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser_no_think.parse(messages, add_generation_prompt=True)
        assert result.endswith("<|assistant|>\n<think></think>")

    def test_add_generation_prompt_false(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages, add_generation_prompt=False)
        assert not result.endswith("<|assistant|>")

    def test_disable_thinking_appends_nothink_to_user(self, parser_no_think):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser_no_think.parse(messages)
        assert "<|user|>\nHi/nothink" in result

    def test_thinking_enabled_no_nothink(self, parser):
        messages = [{"role": "user", "content": "Hi"}]
        result = parser.parse(messages)
        assert "/nothink" not in result

    def test_nothink_not_double_appended(self, parser_no_think):
        messages = [{"role": "user", "content": "Hi/nothink"}]
        result = parser_no_think.parse(messages)
        # Should not have "/nothink/nothink"
        assert "/nothink/nothink" not in result

    def test_unsupported_role_raises(self, parser):
        messages = [{"role": "developer", "content": "bad"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            parser.parse(messages)

    def test_assistant_with_think_tags_in_content(self, parser):
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "<think>reasoning</think>final answer"},
        ]
        result = parser.parse(messages)
        assert "<|assistant|>" in result
        # The parser extracts thinking and re-formats it
        assert "final answer" in result

    def test_assistant_with_reasoning_content_field(self, parser):
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "final", "reasoning_content": "my reasoning"},
        ]
        result = parser.parse(messages)
        assert "my reasoning" in result
        assert "final" in result

    def test_extract_visible_text_string(self, parser):
        assert parser._extract_visible_text("hello") == "hello"

    def test_extract_visible_text_list(self, parser):
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert parser._extract_visible_text(content) == "hello world"

    def test_extract_visible_text_non_text_parts_ignored(self, parser):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "img.png"},
        ]
        assert parser._extract_visible_text(content) == "hello"


# ---------------------------------------------------------------------------
# MultiModalChatTemplateParser._normalize_messages
# ---------------------------------------------------------------------------


class TestMultiModalNormalizeMessages:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool()
        processor = MagicMock()
        return MultiModalChatTemplateParser(pool, processor=processor)

    def test_plain_string_content_passthrough(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        result = parser._normalize_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_legacy_image_placeholder_converted(self, parser):
        messages = [{"role": "user", "content": "Look at this <image> please"}]
        result = parser._normalize_messages(messages)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "image" in types
        assert "text" in types

    def test_legacy_video_placeholder_converted(self, parser):
        messages = [{"role": "user", "content": "Watch <video> now"}]
        result = parser._normalize_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "video" in types

    def test_legacy_placeholders_disabled_raises(self, parser):
        messages = [{"role": "user", "content": "See <image>"}]
        with pytest.raises(ValueError, match="Legacy.*placeholders are disabled"):
            parser._normalize_messages(messages, allow_legacy_placeholders=False)

    def test_list_content_text_normalized(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                ],
            }
        ]
        result = parser._normalize_messages(messages)
        # input_text should be normalized to text
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "hello"

    def test_list_content_image_kept(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data:..."},
                ],
            }
        ]
        result = parser._normalize_messages(messages)
        assert result[0]["content"][0]["type"] == "image"

    def test_list_content_image_url_kept(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        result = parser._normalize_messages(messages)
        assert result[0]["content"][0]["type"] == "image_url"

    def test_list_content_video_types_kept(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "file.mp4"},
                    {"type": "video_url", "video_url": "http://example.com/v.mp4"},
                    {"type": "input_video", "video": "input.mp4"},
                ],
            }
        ]
        result = parser._normalize_messages(messages)
        types = [p["type"] for p in result[0]["content"]]
        assert "video" in types
        assert "video_url" in types
        assert "input_video" in types

    def test_unknown_part_type_raises(self, parser):
        messages = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": "data"}],
            }
        ]
        with pytest.raises(ValueError, match="Unknown content part type"):
            parser._normalize_messages(messages)

    def test_non_dict_part_raises(self, parser):
        messages = [
            {
                "role": "user",
                "content": ["just a string"],
            }
        ]
        with pytest.raises(TypeError, match="must be a dict part"):
            parser._normalize_messages(messages)

    def test_missing_type_in_part_raises(self, parser):
        messages = [
            {
                "role": "user",
                "content": [{"text": "no type key"}],
            }
        ]
        with pytest.raises(ValueError, match="missing/invalid 'type'"):
            parser._normalize_messages(messages)

    def test_non_list_messages_raises(self, parser):
        with pytest.raises(TypeError, match="messages must be a list"):
            parser._normalize_messages("not a list")

    def test_non_dict_message_raises(self, parser):
        with pytest.raises(TypeError, match="must be a dict"):
            parser._normalize_messages(["not a dict"])

    def test_missing_role_raises(self, parser):
        with pytest.raises(ValueError, match="must be a non-empty str"):
            parser._normalize_messages([{"content": "no role"}])

    def test_empty_role_raises(self, parser):
        with pytest.raises(ValueError, match="must be a non-empty str"):
            parser._normalize_messages([{"role": "", "content": "empty role"}])

    def test_unsupported_content_type_raises(self, parser):
        with pytest.raises(TypeError, match="Unsupported content type"):
            parser._normalize_messages([{"role": "user", "content": 12345}])

    def test_text_none_becomes_empty_string(self, parser):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": None}],
            }
        ]
        result = parser._normalize_messages(messages)
        assert result[0]["content"][0]["text"] == ""

    def test_text_non_string_raises(self, parser):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": 42}],
            }
        ]
        with pytest.raises(TypeError, match="must be str"):
            parser._normalize_messages(messages)

    def test_multiple_images_and_text(self, parser):
        messages = [{"role": "user", "content": "<image> describe this and <image> this too"}]
        result = parser._normalize_messages(messages)
        content = result[0]["content"]
        image_count = sum(1 for p in content if p["type"] == "image")
        assert image_count == 2

    def test_preserves_role(self, parser):
        messages = [
            {"role": "assistant", "content": "response"},
            {"role": "system", "content": "instructions"},
        ]
        result = parser._normalize_messages(messages)
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "system"

    def test_mixed_content_string_no_tags(self, parser):
        messages = [{"role": "user", "content": "just plain text"}]
        result = parser._normalize_messages(messages)
        # String without tags stays as string
        assert result[0]["content"] == "just plain text"


# ---------------------------------------------------------------------------
# MultiModalChatTemplateParser._count_placeholders
# ---------------------------------------------------------------------------


class TestMultiModalCountPlaceholders:
    @pytest.fixture()
    def parser(self):
        pool = _make_tokenizer_pool()
        processor = MagicMock()
        return MultiModalChatTemplateParser(pool, processor=processor)

    def test_no_placeholders(self, parser):
        messages = [{"role": "user", "content": "plain text"}]
        norm = parser._normalize_messages(messages)
        img, vid = parser._count_placeholders(norm)
        assert img == 0
        assert vid == 0

    def test_count_images(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "a"},
                    {"type": "image_url", "image_url": "b"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        norm = parser._normalize_messages(messages)
        img, vid = parser._count_placeholders(norm)
        assert img == 2
        assert vid == 0

    def test_count_videos(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "a"},
                    {"type": "video_url", "video_url": "b"},
                    {"type": "input_video", "video": "c"},
                ],
            }
        ]
        norm = parser._normalize_messages(messages)
        img, vid = parser._count_placeholders(norm)
        assert img == 0
        assert vid == 3

    def test_mixed_image_video(self, parser):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "x"},
                    {"type": "video", "video": "y"},
                ],
            }
        ]
        norm = parser._normalize_messages(messages)
        img, vid = parser._count_placeholders(norm)
        assert img == 1
        assert vid == 1


# ---------------------------------------------------------------------------
# MultiModalChatTemplateParser._unwrap_url_field
# ---------------------------------------------------------------------------


class TestUnwrapUrlField:
    def test_dict_with_url(self):
        assert MultiModalChatTemplateParser._unwrap_url_field({"url": "http://ex.com"}) == "http://ex.com"

    def test_dict_with_path(self):
        assert MultiModalChatTemplateParser._unwrap_url_field({"path": "/tmp/img.png"}) == "/tmp/img.png"

    def test_dict_with_uri(self):
        assert MultiModalChatTemplateParser._unwrap_url_field({"uri": "s3://bucket/key"}) == "s3://bucket/key"

    def test_string_passthrough(self):
        assert MultiModalChatTemplateParser._unwrap_url_field("http://ex.com") == "http://ex.com"

    def test_none_passthrough(self):
        assert MultiModalChatTemplateParser._unwrap_url_field(None) is None

    def test_empty_dict_returns_none(self):
        assert MultiModalChatTemplateParser._unwrap_url_field({}) is None


# ---------------------------------------------------------------------------
# ChatTemplateParser.get_parser factory method
# ---------------------------------------------------------------------------


class TestGetParser:
    def test_deepseek_model_returns_deepseek_parser(self):
        pool = _make_tokenizer_pool(name_or_path="deepseek-ai/DeepSeek-R1")
        pool.tokenizer.__class__.__name__ = "LlamaTokenizerFast"
        _ = DeepseekQwenChatTemplateParser.__class__
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, DeepseekQwenChatTemplateParser)

    def test_qwen_model_returns_qwen_parser(self):
        pool = _make_tokenizer_pool(name_or_path="Qwen/Qwen2.5-72B-Instruct")
        pool.tokenizer.__class__.__name__ = "Qwen2Tokenizer"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, QwenChatTemplateParser)

    def test_llama_model_returns_llama_parser(self):
        pool = _make_tokenizer_pool(name_or_path="meta-llama/Llama-3.1-8B")
        pool.tokenizer.__class__.__name__ = "LlamaTokenizer"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, LlamaChatTemplateParser)

    def test_gemma_model_returns_gemma_parser(self):
        pool = _make_tokenizer_pool(name_or_path="google/gemma-2-27b-it")
        pool.tokenizer.__class__.__name__ = "GemmaTokenizer"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, GemmaChatTemplateParser)

    def test_moonlight_model_returns_moonlight_parser(self):
        pool = _make_tokenizer_pool(name_or_path="moonshot/moonlight-16b")
        pool.tokenizer.__class__.__name__ = "SomeTokenizer"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, MoonlightChatTemplateParser)

    def test_glm_model_returns_glm_parser(self):
        pool = _make_tokenizer_pool(name_or_path="THUDM/glm-4-32b")
        pool.tokenizer.__class__.__name__ = "ChatGLMTokenizer"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, GlmChatTemplateParser)

    def test_deepcoder_returns_deepseek_parser(self):
        pool = _make_tokenizer_pool(name_or_path="agentica/deepcoder-14b")
        pool.tokenizer.__class__.__name__ = "LlamaTokenizerFast"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, DeepseekQwenChatTemplateParser)

    def test_deepscaler_returns_deepseek_parser(self):
        pool = _make_tokenizer_pool(name_or_path="org/deepscaler-1.5b")
        pool.tokenizer.__class__.__name__ = "LlamaTokenizerFast"
        from axon.engine.chat_template.parser import ChatTemplateParser

        result = ChatTemplateParser.get_parser(pool)
        assert isinstance(result, DeepseekQwenChatTemplateParser)


# ---------------------------------------------------------------------------
# _strip_chain_of_thought (from gpqa_reward, tested here for coverage)
# ---------------------------------------------------------------------------


class TestStripChainOfThought:
    @pytest.fixture(autouse=True)
    def _import(self):
        from axon.utils.rewards.gpqa_reward import _strip_chain_of_thought

        self.strip_cot = _strip_chain_of_thought

    def test_empty_string(self):
        assert self.strip_cot("") == ""

    def test_no_think_tags(self):
        assert self.strip_cot("plain answer") == "plain answer"

    def test_with_think_tags(self):
        assert self.strip_cot("prefix<think>reasoning</think>final answer") == "final answer"

    def test_only_think_tags(self):
        result = self.strip_cot("<think>only thinking</think>")
        assert result == ""

    def test_multiple_think_tags(self):
        text = "<think>first</think>middle<think>second</think>last"
        result = self.strip_cot(text)
        # rsplit with maxsplit=1 takes text after last </think>
        assert result == "last"

    def test_none_input(self):
        assert self.strip_cot(None) == ""
