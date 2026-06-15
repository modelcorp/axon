"""Tests for axon.core.role module."""

import pytest

from axon.core.role import Role


# ---------------------------------------------------------------------------
# Role enum members and __str__
# ---------------------------------------------------------------------------
class TestRoleEnum:
    def test_str_returns_value_for_all_members(self):
        for member in Role:
            assert str(member) == member.value

    def test_role_usable_as_dict_key(self):
        d = {Role.Actor: "a", Role.Critic: "c"}
        assert d[Role.Actor] == "a"

    def test_role_identity_across_lookups(self):
        assert Role("actor") is Role.Actor

    def test_role_not_equal_to_raw_string(self):
        assert Role.Actor != "actor"


# ---------------------------------------------------------------------------
# Role.from_string – valid lowercase inputs
# ---------------------------------------------------------------------------
class TestRoleFromString:
    def test_actor(self):
        assert Role.from_string("actor") is Role.Actor


# ---------------------------------------------------------------------------
# Role.from_string – case-insensitive
# ---------------------------------------------------------------------------
class TestRoleFromStringCaseInsensitive:
    def test_uppercase_actor(self):
        assert Role.from_string("ACTOR") is Role.Actor

    def test_titlecase_actor(self):
        assert Role.from_string("Actor") is Role.Actor

    def test_mixed_case_sampler(self):
        assert Role.from_string("SaMpLeR") is Role.Sampler


# ---------------------------------------------------------------------------
# Role.from_string – invalid inputs
# ---------------------------------------------------------------------------
class TestRoleFromStringInvalid:
    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="No Role found for string: unknown"):
            Role.from_string("unknown")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="No Role found for string:"):
            Role.from_string("")

    def test_whitespace_raises_value_error(self):
        with pytest.raises(ValueError):
            Role.from_string("  ")

    def test_partial_name_raises_value_error(self):
        with pytest.raises(ValueError):
            Role.from_string("act")

    def test_numeric_string_raises_value_error(self):
        with pytest.raises(ValueError):
            Role.from_string("123")

    def test_none_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            Role.from_string(None)

    def test_leading_trailing_whitespace_not_stripped(self):
        with pytest.raises(ValueError):
            Role.from_string(" actor ")

    def test_value_name_mismatch(self):
        """'ref' maps to RefPolicy, not 'refpolicy'."""
        with pytest.raises(ValueError):
            Role.from_string("refpolicy")
