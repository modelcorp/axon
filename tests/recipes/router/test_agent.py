"""Tests for recipes/router/agent.py — parse_xml_response and RouterAgent."""

from recipes.router.agent import RouterAgent, parse_xml_response
from axon.core import Action
from axon.tools.types import ToolCall

# ---------------------------------------------------------------------------
# parse_xml_response
# ---------------------------------------------------------------------------


class TestParseXmlResponse:
    """Tests for the standalone parse_xml_response helper."""

    def test_with_function_call(self):
        """A well-formed function block should be parsed into a ToolCall."""
        text = (
            "I will call the expert.\n<function=call_expert>\n<parameter=question>What is 2+2?</parameter>\n</function>"
        )
        thought, tool_call = parse_xml_response(text)
        assert thought == "I will call the expert."
        assert isinstance(tool_call, ToolCall)
        assert tool_call.name == "call_expert"
        assert tool_call.arguments["question"] == "What is 2+2?"

    def test_without_function_call(self):
        """When there is no <function> block the entire text is the thought."""
        text = "Just some reasoning with no action."
        thought, tool_call = parse_xml_response(text)
        assert thought == "Just some reasoning with no action."
        assert tool_call is None

    def test_thought_text_before_function(self):
        """Everything before the <function> block is captured as thought."""
        text = (
            "Let me think about this problem carefully.\n"
            "I need expert help.\n"
            "<function=finish>\n"
            "<parameter=answer>\\boxed{42}</parameter>\n"
            "</function>"
        )
        thought, tool_call = parse_xml_response(text)
        assert "Let me think about this problem carefully." in thought
        assert "I need expert help." in thought
        assert tool_call is not None
        assert tool_call.name == "finish"
        assert tool_call.arguments["answer"] == "\\boxed{42}"

    def test_function_with_no_params(self):
        """A function block with no parameters should still parse correctly."""
        text = "<function=call_expert>\n</function>"
        thought, tool_call = parse_xml_response(text)
        assert thought == ""
        assert tool_call is not None
        assert tool_call.name == "call_expert"
        assert tool_call.arguments == {}

    def test_only_function_block_no_thought_text(self):
        """A response that is ONLY a function block (no preceding text) -> empty thought."""
        text = "<function=finish>\n<parameter=answer>42</parameter>\n</function>"
        thought, tool_call = parse_xml_response(text)
        assert thought == ""
        assert tool_call is not None
        assert tool_call.name == "finish"
        assert tool_call.arguments["answer"] == "42"

    # -- edge cases for parse_xml_response ----------------------------------

    def test_multiple_function_blocks_parses_first(self):
        """When multiple function blocks exist, only the FIRST should be parsed (non-greedy regex)."""
        text = (
            "Thought text.\n"
            "<function=call_expert>\n"
            "<parameter=question>help</parameter>\n"
            "</function>\n"
            "More text.\n"
            "<function=finish>\n"
            "<parameter=answer>42</parameter>\n"
            "</function>"
        )
        thought, tool_call = parse_xml_response(text)
        assert thought == "Thought text."
        assert tool_call is not None
        assert tool_call.name == "call_expert"
        assert tool_call.arguments["question"] == "help"

    def test_nested_function_text_inside_parameter(self):
        """The literal string '<function>' inside a parameter value should be handled."""
        text = "Reasoning.\n<function=finish>\n<parameter=answer>use <function> tag carefully</parameter>\n</function>"
        thought, tool_call = parse_xml_response(text)
        assert thought == "Reasoning."
        assert tool_call is not None
        assert tool_call.name == "finish"

    def test_whitespace_heavy_parameter_values(self):
        """Parameter values with leading/trailing whitespace and newlines."""
        text = "Think.\n<function=finish>\n<parameter=answer>\n   \\boxed{42}   \n</parameter>\n</function>"
        thought, tool_call = parse_xml_response(text)
        assert thought == "Think."
        assert tool_call is not None
        assert tool_call.name == "finish"
        # The value may contain surrounding whitespace depending on parser
        assert "\\boxed{42}" in tool_call.arguments["answer"]

    def test_very_long_thought_before_function(self):
        """A very long thought section (1000+ chars) before function block."""
        long_thought = "A" * 2000
        text = f"{long_thought}\n<function=call_expert>\n</function>"
        thought, tool_call = parse_xml_response(text)
        assert thought == long_thought
        assert tool_call is not None
        assert tool_call.name == "call_expert"

    def test_close_function_tag_in_thought_before_real_block(self):
        """'</function>' appearing in thought text before any real function block."""
        text = (
            "I saw a </function> tag earlier but it was not inside a real block.\n"
            "<function=finish>\n"
            "<parameter=answer>done</parameter>\n"
            "</function>"
        )
        thought, tool_call = parse_xml_response(text)
        # The non-greedy regex matches from the first <function= to the first </function>.
        # Since </function> appears in the thought, the regex will match from
        # the text "I saw a </function>" — but that is NOT a <function=...> block.
        # The regex pattern is r"(<function=.*?</function>)" which requires <function=
        # so the literal </function> in the thought won't start a match.
        assert tool_call is not None
        assert tool_call.name == "finish"


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class TestRouterAgent:
    """Tests for the RouterAgent class."""

    def test_reset(self):
        """reset() restores initial state after mutations."""
        agent = RouterAgent()
        agent.step = 5
        agent.first_time = False
        agent.reset()
        assert agent.step == 0
        assert agent.first_time is True

    # -- process_observation ------------------------------------------------

    def test_process_observation_first_time(self):
        """First observation appends the user_prompt and flips the flag."""
        agent = RouterAgent()
        result = agent.process_observation("What is 1+1?", reward=0.0, done=False, info={})
        assert result == f"What is 1+1? {agent.user_prompt}"
        assert agent.first_time is False

    def test_process_observation_subsequent(self):
        """Subsequent observations are returned as-is (no user_prompt)."""
        agent = RouterAgent()
        # First call flips the flag
        agent.process_observation("first", reward=0.0, done=False, info={})
        # Second call
        result = agent.process_observation("Expert says 2", reward=0.0, done=False, info={})
        assert result == "Expert says 2"

    def test_process_observation_with_max_turns_remaining(self):
        """When max_turns is in info and steps remain, a remaining-steps note is appended."""
        agent = RouterAgent()
        agent.first_time = False  # skip user_prompt logic
        agent.step = 2
        result = agent.process_observation("obs", reward=0.0, done=False, info={"max_turns": 5})
        assert "Steps Remaining: 3." in result

    def test_process_observation_max_turns_exhausted(self):
        """When step >= max_turns, a submit-now warning is appended."""
        agent = RouterAgent()
        agent.first_time = False
        agent.step = 5
        result = agent.process_observation("obs", reward=0.0, done=False, info={"max_turns": 5})
        assert "submit your answer NOW" in result

    def test_process_observation_special_characters_passthrough(self):
        """Observation with newlines, quotes, backslashes should pass through unchanged."""
        agent = RouterAgent()
        agent.first_time = False
        special = 'line1\nline2\t"quoted"\\backslash'
        result = agent.process_observation(special, reward=0.0, done=False, info={})
        assert result == special

    # -- process_action -----------------------------------------------------

    def test_process_action_increments_step(self):
        """Each process_action call increments step by 1."""
        agent = RouterAgent()
        assert agent.step == 0

        action_text = "Thinking...\n<function=call_expert>\n</function>"
        result = agent.process_action(action_text)
        assert agent.step == 1
        assert isinstance(result, Action)
        assert result.action == action_text
        assert "Thinking..." in result.thought

        agent.process_action("More reasoning, no function call.")
        assert agent.step == 2

    # -- edge cases for RouterAgent -----------------------------------------

    def test_multiple_episodes(self):
        """Process several observations, reset, process again -- verify clean state."""
        agent = RouterAgent()

        # Episode 1
        obs1 = agent.process_observation("Q1", reward=0.0, done=False, info={})
        assert agent.user_prompt in obs1
        assert agent.first_time is False

        agent.process_action("Thinking.\n<function=call_expert>\n</function>")
        assert agent.step == 1

        obs2 = agent.process_observation("Expert says 7", reward=0.0, done=False, info={})
        assert obs2 == "Expert says 7"

        agent.process_action("<function=finish>\n<parameter=answer>7</parameter>\n</function>")
        assert agent.step == 2

        # Reset for episode 2
        agent.reset()
        assert agent.step == 0
        assert agent.first_time is True

        # Episode 2 - should behave like a fresh agent
        obs3 = agent.process_observation("Q2", reward=0.0, done=False, info={})
        assert agent.user_prompt in obs3
        assert agent.first_time is False
        assert agent.step == 0  # step only increments via process_action

    def test_process_observation_max_turns_1_step_0(self):
        """With max_turns=1 and step=0, remaining should be 1."""
        agent = RouterAgent()
        agent.first_time = False
        agent.step = 0
        result = agent.process_observation("obs", reward=0.0, done=False, info={"max_turns": 1})
        assert "Steps Remaining: 1." in result

    def test_process_action_plain_text_no_function(self):
        """process_action with plain text (no function block) -- thought should be the full text."""
        agent = RouterAgent()
        raw = "I am just reasoning here, no function call at all."
        result = agent.process_action(raw)
        assert isinstance(result, Action)
        assert result.thought == raw
        assert result.action == raw
        assert agent.step == 1
