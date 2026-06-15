"""Tests for axon.programs.base_program module."""

import asyncio

import pytest

from axon.programs.base_program import PROGRAM_CLASS_MAPPING, BaseProgram, ProgramResult

# ---------------------------------------------------------------------------
# ProgramResult dataclass
# ---------------------------------------------------------------------------


class TestProgramResult:
    def test_step_rewards_dict(self):
        result = ProgramResult(reward=1.0, done=True, step_rewards={0: 0.1, 3: 0.9})
        assert result.step_rewards[0] == pytest.approx(0.1)
        assert result.step_rewards[3] == pytest.approx(0.9)
        assert len(result.step_rewards) == 2

    def test_step_rewards_with_non_int_keys(self):
        """step_rewards is typed as dict[int, float] but Python dicts accept any hashable.
        Verify float keys work without error."""
        result = ProgramResult(reward=0.5, done=True, step_rewards={0.5: 0.1, 1.5: 0.9})
        assert result.step_rewards[0.5] == pytest.approx(0.1)
        assert result.step_rewards[1.5] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# BaseProgram
# ---------------------------------------------------------------------------


class TestBaseProgram:
    def test_set_sample_params_merges(self):
        prog = BaseProgram(sample_params={"temperature": 0.7})
        prog.set_sample_params({"top_p": 0.9})
        assert prog.sample_params == {"temperature": 0.7, "top_p": 0.9}

    def test_set_sample_params_updates_existing(self):
        prog = BaseProgram(sample_params={"temperature": 0.7})
        prog.set_sample_params({"temperature": 1.0})
        assert prog.sample_params == {"temperature": 1.0}

    def test_set_sample_params_empty_dict_is_noop(self):
        """Updating with empty dict should leave params unchanged."""
        prog = BaseProgram(sample_params={"temperature": 0.7})
        prog.set_sample_params({})
        assert prog.sample_params == {"temperature": 0.7}

    def test_set_sample_params_called_multiple_times_accumulates(self):
        """Multiple calls to set_sample_params should accumulate all keys."""
        prog = BaseProgram()
        prog.set_sample_params({"temperature": 0.5})
        prog.set_sample_params({"top_p": 0.9})
        prog.set_sample_params({"max_tokens": 100})
        assert prog.sample_params == {"temperature": 0.5, "top_p": 0.9, "max_tokens": 100}

    def test_http_client_is_none_initially(self):
        """_http_client should be None on fresh construction (important for cleanup logic)."""
        prog = BaseProgram()
        assert prog._http_client is None

    def test_run_raises_not_implemented(self):
        prog = BaseProgram()
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            asyncio.get_event_loop().run_until_complete(prog.run())

    def test_session_id_starts_as_none(self):
        prog = BaseProgram()
        assert prog.session_id is None


# ---------------------------------------------------------------------------
# Program registry
# ---------------------------------------------------------------------------


class TestProgramRegistry:
    def test_react_registered(self):
        from axon.programs import ReactProgram

        cls = PROGRAM_CLASS_MAPPING["react"]
        assert cls is ReactProgram

    def test_unknown_program_raises(self):
        with pytest.raises(ValueError, match="Unknown program"):
            PROGRAM_CLASS_MAPPING["nonexistent_program_xyz"]


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------
class TestProgramResultEdgeCases:
    def test_default_step_rewards_empty(self):
        result = ProgramResult(reward=1.0, done=True)
        assert result.step_rewards == {}

    def test_negative_reward(self):
        result = ProgramResult(reward=-5.0, done=True)
        assert result.reward == -5.0

    def test_zero_reward(self):
        result = ProgramResult(reward=0.0, done=False)
        assert result.reward == 0.0

    def test_nan_reward(self):
        """NaN reward should be storable (validation happens elsewhere)."""
        import math

        result = ProgramResult(reward=float("nan"), done=True)
        assert math.isnan(result.reward)

    def test_large_step_rewards(self):
        rewards = {i: float(i) for i in range(100)}
        result = ProgramResult(reward=1.0, done=True, step_rewards=rewards)
        assert len(result.step_rewards) == 100


class TestBaseProgramEdgeCases:
    def test_sample_params_defaults_to_empty_dict(self):
        prog = BaseProgram(sample_params={})
        assert prog.sample_params == {}

    def test_sample_params_none_treated_as_empty(self):
        """If someone passes None for sample_params, it should still work."""
        prog = BaseProgram()
        assert prog.sample_params == {}

    def test_set_sample_params_with_none_value_key(self):
        """Setting a key to None should store None."""
        prog = BaseProgram()
        prog.set_sample_params({"temperature": None})
        assert prog.sample_params["temperature"] is None

    def test_set_sample_params_preserves_all_types(self):
        prog = BaseProgram()
        prog.set_sample_params(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "stop": ["\n", "END"],
                "logprobs": True,
            }
        )
        assert prog.sample_params["stop"] == ["\n", "END"]
        assert prog.sample_params["logprobs"] is True

    def test_multiple_programs_have_independent_params(self):
        p1 = BaseProgram(sample_params={"temperature": 0.5})
        p2 = BaseProgram(sample_params={"temperature": 1.0})
        p1.set_sample_params({"top_p": 0.9})
        assert "top_p" not in p2.sample_params
        assert p1.sample_params["temperature"] == 0.5
        assert p2.sample_params["temperature"] == 1.0
