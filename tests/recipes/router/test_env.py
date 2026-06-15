"""Tests for recipes/router/env.py — _validate_action_format."""

from recipes.router.env import _validate_action_format


class TestValidateActionFormat:
    """Thorough tests for the _validate_action_format helper."""

    # -- valid cases --------------------------------------------------------

    def test_valid_with_think_function_and_params(self):
        """Standard valid action: think block followed by function with parameters."""
        action = (
            "<think>I need to call the expert for help.</think>"
            "<function=call_expert>"
            "<parameter=question>What is 2+2?</parameter>"
            "</function>"
        )
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_valid_with_think_function_no_params(self):
        """Valid action: think block + function with no parameters inside."""
        action = "<think>Let me call the expert.</think><function=call_expert></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_valid_multiline_think(self):
        """Think block spanning multiple lines is valid."""
        action = (
            "<think>\nLine one.\nLine two.\n</think>\n<function=finish>\n<parameter=answer>yes</parameter>\n</function>"
        )
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    # -- missing think tags -------------------------------------------------

    def test_missing_think_open(self):
        """No <think> tag at all."""
        action = "Some text</think><function=finish></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "think" in err.lower()

    def test_missing_think_close(self):
        """No </think> tag."""
        action = "<think>reasoning<function=finish></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "think" in err.lower()

    def test_no_think_at_all(self):
        """No think tags whatsoever."""
        action = "<function=finish><parameter=answer>42</parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "think" in err.lower()

    # -- multiple think blocks ----------------------------------------------

    def test_multiple_think_open_tags(self):
        """Two <think> open tags should fail."""
        action = "<think>first</think><think>second</think><function=finish></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "one" in err.lower()

    def test_multiple_think_close_tags(self):
        """Two </think> close tags should fail."""
        action = "<think>block</think></think><function=finish></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "one" in err.lower()

    # -- think ordering -----------------------------------------------------

    def test_think_after_end_think(self):
        """</think> appearing before <think> should fail."""
        action = "</think><think>backwards</think><function=finish></function>"
        # This has 1 open and 2 close, so it will fail on count first.
        # Let's use the exact case the code checks:
        valid, err = _validate_action_format(action)
        assert valid is False

    def test_think_close_before_open_exact(self):
        """Exactly one of each but </think> before <think>."""
        # We need to craft a string with exactly one <think> and one </think>
        # but </think> appearing first in the string.
        action = "prefix</think>middle<think>content<function=finish></function>"
        # counts: opens=1, closes=1 -- passes count check
        # start_think > end_think -- triggers ordering check
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "before" in err.lower()

    # -- missing function call ----------------------------------------------

    def test_missing_function_call(self):
        """Think block with no function call after it."""
        action = "<think>I am thinking.</think>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "missing" in err.lower()

    # -- multiple function calls --------------------------------------------

    def test_multiple_function_calls(self):
        """Two function calls after think should fail."""
        action = (
            "<think>plan</think>"
            "<function=call_expert></function>"
            "<function=finish><parameter=answer>42</parameter></function>"
        )
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "one" in err.lower() or "only" in err.lower()

    # -- trailing text after </function> ------------------------------------

    def test_trailing_text_after_function(self):
        """Non-whitespace text after </function> should fail."""
        action = "<think>ok</think><function=finish><parameter=answer>42</parameter></function>some trailing text"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "unexpected" in err.lower() or "after" in err.lower()

    def test_trailing_whitespace_after_function_is_ok(self):
        """Trailing whitespace only after </function> should be fine."""
        action = "<think>ok</think><function=finish><parameter=answer>42</parameter></function>   \n  "
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_stray_close_function_tag_after_real_close(self):
        """Extra </function> after the real closing tag hits func_closings > 1 branch."""
        action = "<think>ok</think><function=f></function></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "one" in err.lower() or "only" in err.lower()

    # -- mismatched parameter tags ------------------------------------------

    def test_mismatched_parameter_tags_more_opens(self):
        """More parameter opens than closes."""
        action = "<think>ok</think><function=tool><parameter=a>val<parameter=b>val</parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "parameter" in err.lower()

    def test_mismatched_parameter_tags_more_closes(self):
        """More parameter closes than opens."""
        action = "<think>ok</think><function=tool><parameter=a>val</parameter></parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "parameter" in err.lower()

    # -- non-parameter tags inside function ---------------------------------

    def test_non_parameter_tag_inside_function(self):
        """Arbitrary tags like <div> inside the function block should fail."""
        action = "<think>ok</think><function=tool><div>not allowed</div></function>"
        valid, err = _validate_action_format(action)
        assert valid is False
        assert "parameter" in err.lower() or "only" in err.lower()

    def test_nested_think_inside_function(self):
        """<think> inside the function block is a non-parameter tag, should fail."""
        action = "<think>ok</think><function=tool><think>nested</think></function>"
        valid, err = _validate_action_format(action)
        assert valid is False

    def test_random_tag_inside_function_with_valid_params(self):
        """Valid params plus a stray tag should still fail."""
        action = "<think>ok</think><function=tool><parameter=a>1</parameter><span>bad</span></function>"
        valid, err = _validate_action_format(action)
        assert valid is False

    # -- function inner content that is all whitespace (no parameter tags) --

    def test_function_with_whitespace_only_inner_content(self):
        """Function block whose inner content is only whitespace (no param tags) is valid.

        The code checks ``if inner.strip():`` — all-whitespace inner content
        skips parameter validation entirely.
        """
        action = "<think>ok</think><function=call_expert>   \n  \t  </function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    # -- edge cases ---------------------------------------------------------

    def test_content_before_think_tag(self):
        """Content before <think> tag should be valid since code only checks count/order."""
        action = "prefix text <think>some reasoning</think><function=finish><parameter=answer>42</parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_empty_think_block(self):
        """Empty think block should be valid."""
        action = "<think></think><function=finish><parameter=answer>42</parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_close_function_text_inside_think_block(self):
        """'</function>' text inside the think block should be valid.

        The code extracts the suffix after </think> and only looks for
        function tags there, so stray </function> inside think is irrelevant.
        """
        action = (
            "<think>I noticed a </function> tag in the problem statement.</think>"
            "<function=finish>"
            "<parameter=answer>42</parameter>"
            "</function>"
        )
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_function_name_with_special_characters(self):
        """Function name with a space -- regex <function=[^>]+> will match it."""
        action = "<think>ok</think><function=my tool></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_parameter_with_empty_value(self):
        """Parameter with empty value should be valid."""
        action = "<think>ok</think><function=finish><parameter=x></parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None

    def test_extremely_long_think_content(self):
        """Very long content (1000+ chars) inside think block."""
        long_content = "A" * 2000
        action = f"<think>{long_content}</think><function=finish><parameter=answer>42</parameter></function>"
        valid, err = _validate_action_format(action)
        assert valid is True
        assert err is None
