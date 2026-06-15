"""Tests for helper functions in axon.programs.external.nemo_gym_program."""

from axon.programs.external.nemo_gym_program import (
    _clean_tool_params,
    _clean_tools,
    _extract_text_content,
    build_nemo_gym_response,
    nemo_gym_task_to_messages,
)

# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


class TestExtractTextContent:
    def test_string_input_returned_as_is(self):
        assert _extract_text_content("hello world") == "hello world"

    def test_empty_string(self):
        assert _extract_text_content("") == ""

    def test_list_of_strings_joined_with_newline(self):
        result = _extract_text_content(["line one", "line two", "line three"])
        assert result == "line one\nline two\nline three"

    def test_list_of_dicts_with_text_key(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        result = _extract_text_content(content)
        assert result == "Hello\nWorld"

    def test_list_of_dicts_missing_text_key_defaults_empty(self):
        content = [{"type": "image_url", "url": "http://example.com/img.png"}]
        result = _extract_text_content(content)
        assert result == ""

    def test_mixed_list_strings_and_dicts(self):
        content = ["plain text", {"text": "dict text"}]
        result = _extract_text_content(content)
        assert result == "plain text\ndict text"

    def test_empty_list(self):
        assert _extract_text_content([]) == ""

    def test_other_type_uses_str(self):
        assert _extract_text_content(42) == "42"

    def test_dict_input_uses_str(self):
        d = {"key": "value"}
        result = _extract_text_content(d)
        assert result == str(d)

    def test_none_uses_str(self):
        assert _extract_text_content(None) == "None"

    def test_single_item_list_no_trailing_newline(self):
        result = _extract_text_content(["only one"])
        assert result == "only one"


# ---------------------------------------------------------------------------
# _clean_tool_params
# ---------------------------------------------------------------------------


class TestCleanToolParams:
    def test_removes_none_valued_properties(self):
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "board": None,
                "visitor_id": None,
            },
            "required": ["name", "board"],
        }
        result = _clean_tool_params(params)
        assert "board" not in result["properties"]
        assert "visitor_id" not in result["properties"]
        assert "name" in result["properties"]

    def test_updates_required_list_to_match_remaining_props(self):
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "board": None,
            },
            "required": ["name", "board"],
        }
        result = _clean_tool_params(params)
        assert result["required"] == ["name"]

    def test_no_required_key(self):
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "junk": None,
            },
        }
        result = _clean_tool_params(params)
        assert "required" not in result
        assert "name" in result["properties"]
        assert "junk" not in result["properties"]

    def test_all_properties_none_yields_empty_props(self):
        params = {
            "type": "object",
            "properties": {"a": None, "b": None},
            "required": ["a"],
        }
        result = _clean_tool_params(params)
        assert result["properties"] == {}
        assert result["required"] == []

    def test_no_properties_key_returns_as_is(self):
        params = {"type": "object"}
        result = _clean_tool_params(params)
        assert result == {"type": "object"}

    def test_non_dict_input_returned_as_is(self):
        assert _clean_tool_params("not a dict") == "not a dict"
        assert _clean_tool_params(None) is None

    def test_preserves_other_fields(self):
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "additionalProperties": False,
            "description": "Some schema",
        }
        result = _clean_tool_params(params)
        assert result["additionalProperties"] is False
        assert result["description"] == "Some schema"
        assert result["type"] == "object"

    def test_does_not_mutate_original(self):
        original_props = {"name": {"type": "string"}, "junk": None}
        params = {"type": "object", "properties": original_props, "required": ["name", "junk"]}
        _clean_tool_params(params)
        # Original should still have the None-valued property
        assert "junk" in params["properties"]
        assert params["required"] == ["name", "junk"]


# ---------------------------------------------------------------------------
# _clean_tools
# ---------------------------------------------------------------------------


class TestCleanTools:
    def test_cleans_function_wrapper_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string"},
                            "junk": None,
                        },
                        "required": ["arg1", "junk"],
                    },
                },
            }
        ]
        result = _clean_tools(tools)
        assert len(result) == 1
        fn = result[0]["function"]
        assert "junk" not in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["arg1"]

    def test_cleans_flat_format(self):
        """Tools without a 'function' wrapper (just {name, parameters})."""
        tools = [
            {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "unused": None,
                    },
                },
            }
        ]
        result = _clean_tools(tools)
        assert "unused" not in result[0]["parameters"]["properties"]

    def test_tool_without_parameters_unchanged(self):
        tools = [{"type": "function", "function": {"name": "noop"}}]
        result = _clean_tools(tools)
        assert result[0]["function"]["name"] == "noop"
        assert "parameters" not in result[0]["function"]

    def test_empty_tools_list(self):
        assert _clean_tools([]) == []

    def test_multiple_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "int"}, "y": None},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "parameters": {
                        "type": "object",
                        "properties": {"z": {"type": "string"}},
                    },
                },
            },
        ]
        result = _clean_tools(tools)
        assert len(result) == 2
        assert "y" not in result[0]["function"]["parameters"]["properties"]
        assert "z" in result[1]["function"]["parameters"]["properties"]

    def test_does_not_mutate_original(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}, "b": None},
                    },
                },
            }
        ]
        _clean_tools(tools)
        # Original should still have the None property
        assert "b" in tools[0]["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# nemo_gym_task_to_messages
# ---------------------------------------------------------------------------


class TestNemoGymTaskToMessages:
    def test_instructions_become_system_message(self):
        task = {
            "responses_create_params": {
                "instructions": "You are a helpful assistant.",
                "input": [],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}

    def test_string_input_becomes_user_message(self):
        task = {
            "responses_create_params": {
                "input": "What is 2+2?",
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What is 2+2?"}

    def test_string_input_with_instructions(self):
        task = {
            "responses_create_params": {
                "instructions": "Be concise.",
                "input": "Hello!",
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Hello!"}

    def test_list_input_with_message_type(self):
        task = {
            "responses_create_params": {
                "input": [
                    {"type": "message", "role": "user", "content": "Hi there"},
                ],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages[0] == {"role": "user", "content": "Hi there"}

    def test_developer_role_mapped_to_system(self):
        task = {
            "responses_create_params": {
                "input": [
                    {"type": "message", "role": "developer", "content": "System prompt here"},
                ],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt here"

    def test_tools_returned_from_params(self):
        tool_def = {"type": "function", "function": {"name": "search"}}
        task = {
            "responses_create_params": {
                "input": "query",
                "tools": [tool_def],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert tools == [tool_def]

    def test_no_instructions_no_system_message(self):
        task = {
            "responses_create_params": {
                "input": [{"type": "message", "role": "user", "content": "Hello"}],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert all(m["role"] != "system" for m in messages)

    def test_function_call_items_are_skipped(self):
        task = {
            "responses_create_params": {
                "input": [
                    {"type": "message", "role": "user", "content": "Do something"},
                    {"type": "function_call", "name": "search", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": "result"},
                    {"type": "message", "role": "user", "content": "Thanks"},
                ],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert len(messages) == 2
        assert messages[0]["content"] == "Do something"
        assert messages[1]["content"] == "Thanks"

    def test_unknown_types_silently_skipped(self):
        task = {
            "responses_create_params": {
                "input": [
                    {"type": "reasoning", "content": "thinking..."},
                    {"type": "message", "role": "user", "content": "Hello"},
                ],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_empty_input_list(self):
        task = {"responses_create_params": {"input": [], "tools": []}}
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages == []
        assert tools == []

    def test_fallback_to_task_when_no_responses_create_params(self):
        """When 'responses_create_params' is missing, task itself is used."""
        task = {
            "input": "direct input",
            "tools": [],
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages[0] == {"role": "user", "content": "direct input"}

    def test_content_list_extracted_via_helper(self):
        task = {
            "responses_create_params": {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Part A"},
                            {"type": "text", "text": "Part B"},
                        ],
                    }
                ],
                "tools": [],
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        assert messages[0]["content"] == "Part A\nPart B"

    def test_default_role_is_user(self):
        """Items without explicit role should default to 'user'."""
        task = {
            "responses_create_params": {
                "input": [{"type": "message", "content": "no role specified"}],
                "tools": [],
            }
        }
        messages, _ = nemo_gym_task_to_messages(task)
        assert messages[0]["role"] == "user"

    def test_string_items_in_list_become_user_messages(self):
        task = {
            "responses_create_params": {
                "input": ["first string", "second string"],
                "tools": [],
            }
        }
        messages, _ = nemo_gym_task_to_messages(task)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "first string"}
        assert messages[1] == {"role": "user", "content": "second string"}

    def test_multi_turn_conversation(self):
        task = {
            "responses_create_params": {
                "instructions": "You are helpful.",
                "input": [
                    {"type": "message", "role": "user", "content": "Hello"},
                    {"type": "message", "role": "assistant", "content": "Hi!"},
                    {"type": "message", "role": "user", "content": "How are you?"},
                ],
                "tools": [],
            }
        }
        messages, _ = nemo_gym_task_to_messages(task)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"


# ---------------------------------------------------------------------------
# build_nemo_gym_response
# ---------------------------------------------------------------------------


class TestBuildNemoGymResponse:
    def test_required_fields_present(self):
        task = {"responses_create_params": {"model": "test-model"}}
        resp = build_nemo_gym_response([], task)
        assert resp["id"].startswith("resp_")
        assert isinstance(resp["created_at"], int)
        assert resp["model"] == "test-model"
        assert resp["object"] == "response"
        assert resp["status"] == "completed"

    def test_output_items_carried_through(self):
        items = [
            {"type": "message", "role": "assistant", "content": "Hello"},
        ]
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response(items, task)
        assert resp["output"] == items

    def test_tools_from_argument_override(self):
        explicit_tools = [{"type": "function", "function": {"name": "a"}}]
        task_tools = [{"type": "function", "function": {"name": "b"}}]
        task = {"responses_create_params": {"tools": task_tools}}
        resp = build_nemo_gym_response([], task, tools=explicit_tools)
        assert resp["tools"] == explicit_tools

    def test_tools_from_task_when_none(self):
        task_tools = [{"type": "function", "function": {"name": "b"}}]
        task = {"responses_create_params": {"tools": task_tools}}
        resp = build_nemo_gym_response([], task, tools=None)
        assert resp["tools"] == task_tools

    def test_tools_default_empty_list(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["tools"] == []

    def test_tool_choice_from_task(self):
        task = {"responses_create_params": {"tool_choice": "required"}}
        resp = build_nemo_gym_response([], task)
        assert resp["tool_choice"] == "required"

    def test_tool_choice_defaults_to_auto(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["tool_choice"] == "auto"

    def test_model_defaults_to_axon(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["model"] == "axon"

    def test_parallel_tool_calls_default_true(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["parallel_tool_calls"] is True

    def test_instructions_carried_through(self):
        task = {"responses_create_params": {"instructions": "Be helpful."}}
        resp = build_nemo_gym_response([], task)
        assert resp["instructions"] == "Be helpful."

    def test_temperature_and_top_p_defaults(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["temperature"] == 1.0
        assert resp["top_p"] == 1.0

    def test_temperature_and_top_p_from_task(self):
        task = {"responses_create_params": {"temperature": 0.5, "top_p": 0.8}}
        resp = build_nemo_gym_response([], task)
        assert resp["temperature"] == 0.5
        assert resp["top_p"] == 0.8

    def test_missing_responses_create_params(self):
        """When task has no responses_create_params, defaults should apply."""
        task = {}
        resp = build_nemo_gym_response([], task)
        assert resp["model"] == "axon"
        assert resp["tools"] == []
        assert resp["tool_choice"] == "auto"

    def test_unique_ids(self):
        task = {"responses_create_params": {}}
        r1 = build_nemo_gym_response([], task)
        r2 = build_nemo_gym_response([], task)
        assert r1["id"] != r2["id"]

    def test_created_at_is_recent_timestamp(self):
        import time

        before = int(time.time())
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        after = int(time.time())
        assert before <= resp["created_at"] <= after

    def test_optional_fields_present(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["error"] is None
        assert resp["incomplete_details"] is None
        assert resp["previous_response_id"] is None
        assert resp["usage"] is None
        assert resp["metadata"] == {}
        assert resp["truncation"] == "disabled"

    def test_user_field_carried_through(self):
        task = {"responses_create_params": {"user": "axon:session_123"}}
        resp = build_nemo_gym_response([], task)
        assert resp["user"] == "axon:session_123"

    def test_text_format_default(self):
        task = {"responses_create_params": {}}
        resp = build_nemo_gym_response([], task)
        assert resp["text"] == {"format": {"type": "text"}}


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------


class TestNemoGymHelperEdgeCases:
    """Cross-cutting integration and edge cases."""

    def test_clean_tools_then_task_to_messages_integration(self):
        """Real-world flow: NeMo Gym sends dirty tools, we clean then convert."""
        dirty_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "board": None,
                            "visitor_id": None,
                            "session": None,
                        },
                        "required": ["query", "board"],
                    },
                },
            }
        ]
        task = {
            "responses_create_params": {
                "instructions": "You are a search assistant.",
                "input": [{"type": "message", "role": "user", "content": "Find me a hotel"}],
                "tools": dirty_tools,
            }
        }
        messages, tools = nemo_gym_task_to_messages(task)
        # Tools come through raw from task
        cleaned = _clean_tools(tools)
        assert len(cleaned) == 1
        props = cleaned[0]["function"]["parameters"]["properties"]
        assert "query" in props
        assert "board" not in props
        assert cleaned[0]["function"]["parameters"]["required"] == ["query"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Find me a hotel"

    def test_extract_text_content_with_deeply_mixed_content(self):
        """Content list with strings, dicts with text, dicts without text."""
        content = [
            "Plain text",
            {"type": "text", "text": "Dict text"},
            {"type": "image_url", "url": "http://example.com"},
            "Another plain",
        ]
        result = _extract_text_content(content)
        assert "Plain text" in result
        assert "Dict text" in result
        assert "Another plain" in result
        # image_url entry should contribute empty string, not crash
        lines = result.split("\n")
        assert len(lines) == 4
        assert lines[2] == ""  # from image_url dict without 'text'

    def test_clean_tool_params_preserves_deeply_nested_schemas(self):
        """Cleaning should only affect top-level properties, not nested ones."""
        params = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                        "nested_null": None,  # this is inside a real prop, not null itself
                    },
                },
                "junk": None,
            },
            "required": ["config"],
        }
        result = _clean_tool_params(params)
        assert "config" in result["properties"]
        assert "junk" not in result["properties"]
        # Inner schema should be preserved as-is (cleaning is shallow)
        inner = result["properties"]["config"]
        assert inner["properties"]["nested_null"] is None

    def test_nemo_gym_task_to_messages_large_conversation(self):
        """20-turn conversation should all come through."""
        inputs = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            inputs.append({"type": "message", "role": role, "content": f"Turn {i}"})

        task = {"responses_create_params": {"input": inputs, "tools": []}}
        messages, _ = nemo_gym_task_to_messages(task)
        assert len(messages) == 20
        assert messages[0]["content"] == "Turn 0"
        assert messages[19]["content"] == "Turn 19"
        # Check alternating roles
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_build_response_with_complex_output_items(self):
        """Output items with tool calls and nested content."""
        items = [
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Thinking..."}]},
            {"type": "function_call", "name": "search", "arguments": '{"q": "test"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": '{"results": [1, 2, 3]}'},
            {"type": "message", "role": "assistant", "content": "Found results."},
        ]
        task = {"responses_create_params": {"model": "test-model", "tools": [{"name": "search"}]}}
        resp = build_nemo_gym_response(items, task)
        assert resp["output"] == items
        assert resp["model"] == "test-model"
        assert len(resp["tools"]) == 1
