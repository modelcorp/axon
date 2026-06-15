"""Tests for axon.utils.rewards.math_utils.utils module."""

import pytest

from axon.utils.rewards.math_utils.utils import (
    _is_float,
    _is_frac,
    _is_int,
    _normalize,
    _parse_latex,
    _str_is_int,
    _str_to_int,
    _strip_string,
    are_equal_under_sympy,
    extract_answer,
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    last_boxed_only_string,
    remove_boxed,
    should_allow_eval,
    split_tuple,
)


# ---------------------------------------------------------------------------
# Numeric type checks – parametrized
# ---------------------------------------------------------------------------
class TestNumericChecks:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("42", True),
            ("3.14", True),
            ("-2.5", True),
            ("1e10", True),
            ("-3.14e-2", True),
            ("", False),
            ("   ", False),
            ("abc", False),
        ],
    )
    def test_is_float(self, s, expected):
        assert _is_float(s) is expected

    @pytest.mark.parametrize(
        "x,expected",
        [
            (5.0, True),
            (5.00000001, True),
            (5.001, False),
            (0.0, True),
            (-3.0, True),
        ],
    )
    def test_is_int(self, x, expected):
        assert _is_int(x) is expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("1/2", True),
            ("-3/4", True),
            ("12/7", True),
            ("1/0", False),
            ("0/0", False),
            ("1/-2", False),
            ("abc", False),
        ],
    )
    def test_is_frac(self, s, expected):
        assert _is_frac(s) is expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("42", True),
            ("5.0", True),
            ("1,000,000", True),
            ("3.5", False),
            ("abc", False),
            ("", False),
        ],
    )
    def test_str_is_int(self, s, expected):
        assert _str_is_int(s) is expected

    def test_str_to_int_basic(self):
        assert _str_to_int("42") == 42

    def test_str_to_int_with_commas(self):
        assert _str_to_int("1,000") == 1000

    def test_str_to_int_from_float_string(self):
        assert _str_to_int("5.0") == 5


# ---------------------------------------------------------------------------
# _strip_string – LaTeX normalization
# ---------------------------------------------------------------------------
class TestStripString:
    @pytest.mark.parametrize(
        "input_,expected_contains",
        [
            ("\\frac12", "\\frac{1}{2}"),
            ("\\frac1{2}", "{1}"),
            ("\\sqrt2", "\\sqrt{2}"),
            ("1/2", "\\frac{1}{2}"),
        ],
    )
    def test_shorthand_expansion(self, input_, expected_contains):
        assert expected_contains in _strip_string(input_)

    def test_half_to_frac_exact(self):
        assert _strip_string("0.5") == "\\frac{1}{2}"

    def test_leading_dot_normalized(self):
        assert _strip_string(".5") == "\\frac{1}{2}"

    def test_equation_strips_short_lhs(self):
        """'x = 42' strips 'x = ', but 'abc = 42' keeps lhs (len > 2)."""
        assert _strip_string("x = 42") == "42"
        result = _strip_string("abc = 42")
        assert "abc" in result  # lhs too long to strip

    def test_removes_left_right_and_spaces(self):
        result = _strip_string("\\left( 1 + 2 \\right)")
        assert "\\left" not in result
        assert "\\right" not in result
        assert " " not in result

    def test_tfrac_dfrac_replaced(self):
        assert "tfrac" not in _strip_string("\\tfrac{a}{b}")
        assert "dfrac" not in _strip_string("\\dfrac{1}{2}")

    def test_empty_string(self):
        assert _strip_string("") == ""

    def test_incomplete_frac_does_not_crash(self):
        """Bare \\frac should not raise IndexError."""
        result = _strip_string("\\frac")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _normalize – computed output verification
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_none_returns_none(self):
        assert _normalize(None) is None

    def test_integer_normalization(self):
        assert _normalize("42.0") == "42"

    def test_text_wrapper_stripped(self):
        assert _normalize("\\text{42}") == "42"

    def test_curly_braces_stripped(self):
        assert _normalize("{42}") == "42"

    def test_million_conversion(self):
        result = _normalize("2 million")
        assert "2*10^6" in result

    def test_comma_in_number(self):
        assert _normalize("1,000") == "1000"

    def test_unit_removal(self):
        for unit in ["meters", "feet", "inches", "degrees", "seconds"]:
            assert unit not in _normalize(f"5 {unit}")

    def test_or_and_replaced_with_comma(self):
        assert "," in _normalize("1 or 2")
        assert "," in _normalize("3 and 4")

    def test_percent_and_dollar_removed(self):
        result = _normalize("50\\% of \\$100")
        assert "%" not in result
        assert "$" not in result

    def test_latex_parsing_triggered(self):
        """If \\ present after initial cleanup, _parse_latex is called."""
        result = _normalize("\\frac{1}{2}")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# should_allow_eval
# ---------------------------------------------------------------------------
class TestShouldAllowEval:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("2+3*4-1/5", True),  # pure numbers
            ("x+y", True),  # 2 unknowns ok
            ("a+b+c+d", False),  # 3+ unknowns
            ("x^{2}", False),  # bad substring ^{
            ("x^(2)", False),  # bad substring ^(
            ("2^34^5", False),  # bad regex
            ("sqrt(4)", True),  # sqrt not counted as unknown
        ],
    )
    def test_cases(self, expr, expected):
        assert should_allow_eval(expr) is expected


# ---------------------------------------------------------------------------
# are_equal_under_sympy
# ---------------------------------------------------------------------------
class TestAreEqualUnderSympy:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("5", "2+3", True),
            ("x+x", "2*x", True),
            ("2+3", "6", False),
            ("???", "42", False),  # unparseable → False, not crash
            ("", "0", False),  # empty → False
            ("1/3", "1/3", True),
        ],
    )
    def test_cases(self, a, b, expected):
        assert are_equal_under_sympy(a, b) is expected


# ---------------------------------------------------------------------------
# split_tuple
# ---------------------------------------------------------------------------
class TestSplitTuple:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("", []),
            ("42", ["42"]),
            ("(1, 2)", ["1", "2"]),
            ("[3, 4]", ["3", "4"]),
            ("(1, 2, 3)", ["1", "2", "3"]),
            ("1,000", ["1000"]),  # comma in number stripped
            ("[1,(2,3)]", ["[1,(2,3)]"]),  # nested → no split
            ("(x)", ["x"]),
        ],
    )
    def test_cases(self, expr, expected):
        assert split_tuple(expr) == expected


# ---------------------------------------------------------------------------
# last_boxed_only_string / remove_boxed / extract_boxed_answer
# ---------------------------------------------------------------------------
class TestExtractBoxed:
    def test_simple(self):
        assert last_boxed_only_string("\\boxed{42}") == "\\boxed{42}"

    def test_multiple_takes_last(self):
        assert last_boxed_only_string("\\boxed{1} \\boxed{2}") == "\\boxed{2}"

    def test_no_boxed(self):
        assert last_boxed_only_string("no box") is None

    def test_fbox_fallback(self):
        assert last_boxed_only_string("\\fbox{42}") == "\\fbox{42}"

    def test_nested_braces_captured_fully(self):
        s = "Answer: \\boxed{\\frac{1}{2}}"
        assert last_boxed_only_string(s) == "\\boxed{\\frac{1}{2}}"

    def test_deeply_nested_braces(self):
        s = "\\boxed{\\sqrt{\\frac{a}{b}}}"
        result = last_boxed_only_string(s)
        assert result == s

    def test_unbalanced_braces_returns_none(self):
        assert last_boxed_only_string("\\boxed{abc") is None

    def test_remove_boxed(self):
        assert remove_boxed("\\boxed{42}") == "42"
        assert remove_boxed("not boxed") is None

    def test_extract_boxed_answer_nested(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_extract_answer_dispatches(self):
        assert extract_answer("\\boxed{7}") == "7"
        assert extract_answer("no box") is None


# ---------------------------------------------------------------------------
# grade_answer_sympy – the main grading function
# ---------------------------------------------------------------------------
class TestGradeAnswerSympy:
    @pytest.mark.parametrize(
        "given,truth,expected",
        [
            ("42", "42", True),
            ("0.5", "1/2", True),
            ("5", "42", False),
            ("", "42", False),
            ("2.0", "2", True),  # int-like float matches int
            ("-3", "-3", True),
            ("-3", "3", False),
            ("\\frac{1}{3}", "\\frac{1}{3}", True),
            ("(1, 2)", "(1, 2)", True),
            ("(1, 2)", "(2, 1)", False),  # order matters
            ("(1, 2)", "(1, 2, 3)", False),  # length mismatch
            ("(1, 2)", "[1, 2]", False),  # bracket type mismatch
        ],
    )
    def test_cases(self, given, truth, expected):
        assert grade_answer_sympy(given, truth) is expected

    def test_none_ground_truth(self):
        assert grade_answer_sympy("42", None) is False

    def test_unreduced_fraction_rejected(self):
        """Unreduced 2/4 should NOT match 1/2 (frac comparison is exact)."""
        assert grade_answer_sympy("2/4", "1/2") is False


# ---------------------------------------------------------------------------
# grade_answer_mathd
# ---------------------------------------------------------------------------
class TestGradeAnswerMathd:
    @pytest.mark.parametrize(
        "given,truth,expected",
        [
            ("42", "42", True),
            ("\\frac{1}{2}", "\\frac{1}{2}", True),
            ("1", "2", False),
            (" 42 ", "42", True),  # whitespace stripped
        ],
    )
    def test_cases(self, given, truth, expected):
        assert grade_answer_mathd(given, truth) is expected


# ---------------------------------------------------------------------------
# _parse_latex – symbol replacement
# ---------------------------------------------------------------------------
class TestParseLatex:
    def test_sqrt_pi_times_replaced(self):
        result = _parse_latex("\\sqrt{\\pi \\times 2}")
        assert "√" not in result
        assert "π" not in result
        assert "*" in result
        assert "pi" in result

    def test_infinity_replaced(self):
        result = _parse_latex("\\infty")
        assert "inf" in result

    def test_union_replaced(self):
        result = _parse_latex("A \\cup B")
        assert "U" in result
