"""Tests for axon.utils.system_prompts and per-recipe prompt modules.

Recipe-specific prompts now live in ``recipes/<name>/prompts.py``; this module
keeps the cross-recipe data-processing prompts (ORM, LCB, Codeforces, etc.).
The tests cover both surfaces.
"""

import inspect

import axon.utils.system_prompts as system_prompts
from recipes.swe import prompts as swe_prompts


def _get_prompt_constants():
    """Return all ALL_CAPS module-level attribute names (excluding private ones)."""
    return [name for name in dir(system_prompts) if not name.startswith("_") and name == name.upper()]


class TestPromptIntegrity:
    """Comprehensive integrity checks for all prompt constants."""

    # ------------------------------------------------------------------
    # Basic invariants
    # ------------------------------------------------------------------

    def test_at_least_eight_prompt_constants(self):
        """Sanity check that we haven't accidentally deleted prompt constants.

        After splitting recipe-specific prompts into ``recipes/<name>/prompts.py``,
        ``axon.utils.system_prompts`` keeps cross-recipe ones (ORM, LCB, Codeforces,
        AMC, proof / solution extraction).
        """
        count = len(_get_prompt_constants())
        assert count >= 8, f"Expected at least 8 prompt constants, found {count}"

    def test_all_constants_are_non_empty_strings(self):
        """Every ALL_CAPS constant in the module must be a non-empty string."""
        for name in _get_prompt_constants():
            value = getattr(system_prompts, name)
            assert isinstance(value, str), f"{name} is not a str"
            assert len(value) > 0, f"{name} is empty"

    def test_each_prompt_is_at_least_50_characters(self):
        """Real prompts should not be trivially short one-liners."""
        for name in _get_prompt_constants():
            value = getattr(system_prompts, name)
            assert len(value) >= 50, f"{name} is only {len(value)} chars — too short for a real prompt"

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def test_no_duplicate_prompt_content(self):
        """No two prompt constants should have identical content (detect copy-paste errors)."""
        seen: dict[str, str] = {}  # content -> first constant name
        for name in _get_prompt_constants():
            value = getattr(system_prompts, name)
            if value in seen:
                raise AssertionError(f"{name} has identical content to {seen[value]}")
            seen[value] = name

    # ------------------------------------------------------------------
    # Template-variable checks (based on actual module content)
    # ------------------------------------------------------------------

    def test_swe_user_prompt_contains_problem_statement_placeholder(self):
        """SWE_USER_PROMPT (in recipes/swe/prompts.py) must include {problem_statement}."""
        assert "{problem_statement}" in swe_prompts.SWE_USER_PROMPT, (
            "SWE_USER_PROMPT is missing the {problem_statement} placeholder"
        )

    def test_swe_user_prompt_fn_call_contains_problem_statement_placeholder(self):
        """SWE_USER_PROMPT_FN_CALL (in recipes/swe/prompts.py) must include {problem_statement}."""
        assert "{problem_statement}" in swe_prompts.SWE_USER_PROMPT_FN_CALL, (
            "SWE_USER_PROMPT_FN_CALL is missing the {problem_statement} placeholder"
        )

    def test_sweagent_user_prompt_contains_problem_statement_placeholder(self):
        """SWEAGENT_USER_PROMPT (in recipes/swe/prompts.py) must include {problem_statement}."""
        assert "{problem_statement}" in swe_prompts.SWEAGENT_USER_PROMPT, (
            "SWEAGENT_USER_PROMPT is missing the {problem_statement} placeholder"
        )

    # ------------------------------------------------------------------
    # Module-purity check
    # ------------------------------------------------------------------

    def test_no_callable_attributes_that_look_like_constants(self):
        """A constants-only module should not contain functions or classes
        whose names follow the ALL_CAPS naming convention.  This detects
        accidental function definitions in the module."""
        for name in _get_prompt_constants():
            value = getattr(system_prompts, name)
            assert not callable(value), (
                f"{name} is callable ({type(value).__name__}) — "
                "expected a plain string constant, not a function or class"
            )

    def test_no_unexpected_public_callables(self):
        """The module should not define public functions or classes at all
        (it is meant to hold only string constants)."""
        for name in dir(system_prompts):
            if name.startswith("_"):
                continue
            value = getattr(system_prompts, name)
            # Allow imported modules (e.g. if someone did `import re` at the top)
            if inspect.ismodule(value):
                continue
            assert not callable(value), (
                f"Unexpected callable '{name}' ({type(value).__name__}) found in a constants-only module"
            )
