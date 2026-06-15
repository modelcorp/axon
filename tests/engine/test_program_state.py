import contextlib
import threading
import time

import pytest
import torch

from axon.engine.state.program_state import (
    ModelOutput,
    ModelStopReason,
    ProgramManager,
    ProgramState,
    Step,
    TerminationReason,
)


# ---------------------------------------------------------------------------
# ModelOutput dataclass
# ---------------------------------------------------------------------------
class TestModelOutput:
    def test_from_token_strs_builds_raw_response(self):
        output = ModelOutput.from_token_strs(
            token_ids=[10, 20],
            token_strs=["Answer", "<|im_end|>"],
            logprobs=[-0.1, -0.2],
            stop_reason=ModelStopReason.STOP,
            moe_routermap=[],
        )

        assert output.response == "Answer<|im_end|>"
        assert output.token_ids == [10, 20]
        assert output.token_strs == ["Answer", "<|im_end|>"]


# ---------------------------------------------------------------------------
# TerminationReason enum
# ---------------------------------------------------------------------------
class TestTerminationReason:
    """Verify TerminationReason used correctly in is_trainable strict mode."""

    def test_env_done_is_strict_allowed(self):
        """Only ENV_DONE should pass strict is_trainable check."""
        ps = ProgramState(uid="p1", session_id="s1", group_id="g1")
        ps.done = True
        ps.reward = 0.5
        s = Step(uid="s1", session_id="s1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]
        for reason in TerminationReason:
            ps.termination_reason = reason
            if reason == TerminationReason.ENV_DONE:
                assert ps.is_trainable(strict=True), "ENV_DONE should be strict-trainable"
            else:
                assert not ps.is_trainable(strict=True), f"{reason} should NOT be strict-trainable"


# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------
class TestStep:
    """Tests for the Step dataclass."""

    @pytest.fixture
    def step(self):
        return Step(uid="step-1", session_id="sess-1")

    def test_check_empty_partial_initially(self, step):
        assert step.check_empty_partial()

    # -- set_response --
    def test_set_response_basic(self, step):
        step.set_response(
            text="hello",
            tokens=[1, 2, 3],
            masks=[1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3],
            moe_routermap=[],
        )
        assert step.text == "hello"
        assert step.tokens == [1, 2, 3]
        assert step.token_len == 3
        assert step.masks == [1, 1, 1]
        assert step.logprobs == [-0.1, -0.2, -0.3]

    def test_set_response_accumulates(self, step):
        step.set_response(
            text="hel",
            tokens=[1, 2],
            masks=[1, 1],
            logprobs=[-0.1, -0.2],
            moe_routermap=[],
        )
        step.set_response(
            text="lo",
            tokens=[3],
            masks=[1],
            logprobs=[-0.3],
            moe_routermap=[],
        )
        assert step.text == "hello"
        assert step.tokens == [1, 2, 3]
        assert step.token_len == 3
        assert step.masks == [1, 1, 1]
        assert step.logprobs == [-0.1, -0.2, -0.3]

    def test_set_response_mismatched_lengths_raises(self, step):
        with pytest.raises(AssertionError):
            step.set_response(
                text="bad",
                tokens=[1, 2],
                masks=[1],
                logprobs=[-0.1, -0.2],
                moe_routermap=[],
            )

    def test_set_response_10_times_accumulation(self, step):
        """Call set_response 10 times, verify all tokens concatenated correctly."""
        all_text = ""
        all_tokens = []
        all_masks = []
        all_logprobs = []
        for i in range(10):
            text = f"w{i}"
            tokens = [100 + i]
            masks = [1]
            logprobs = [-0.1 * (i + 1)]
            step.set_response(text, tokens, masks, logprobs, [])
            all_text += text
            all_tokens.extend(tokens)
            all_masks.extend(masks)
            all_logprobs.extend(logprobs)

        assert step.text == all_text
        assert step.tokens == all_tokens
        assert step.token_len == 10
        assert step.masks == all_masks
        assert step.logprobs == all_logprobs

    def test_set_response_moe_routermap_full_length(self, step):
        """When moe_routermap has the same length as tokens, it replaces entirely."""
        routermap = torch.randn(3, 4, 8)  # [seq_len=3, num_layers=4, num_experts=8]
        step.set_response(
            text="abc",
            tokens=[1, 2, 3],
            masks=[1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3],
            moe_routermap=routermap,
        )
        assert isinstance(step.moe_routermap, torch.Tensor)
        assert step.moe_routermap.shape == (3, 4, 8)

    def test_set_response_moe_routermap_partial_with_padding(self, step):
        """Partial routermap: when old routermap is empty, should pad with -1."""
        # First set 2 tokens with no routermap
        step.set_response("ab", [1, 2], [1, 1], [-0.1, -0.2], [])
        # Now set 1 more token with routermap of length 1 (shorter than expected 3)
        partial_map = torch.randn(1, 4, 8)
        step.set_response("c", [3], [1], [-0.3], partial_map)
        # expected_len is now 3, routermap is 1, so cached_len = 2
        # old routermap is empty (list), so we go into the padding branch
        # Result should have length 3: 2 padded + 1 from partial
        if isinstance(step.moe_routermap, torch.Tensor):
            assert step.moe_routermap.shape[0] == 3
            # First 2 entries should be -1 padding
            assert torch.all(step.moe_routermap[0] == -1)
            assert torch.all(step.moe_routermap[1] == -1)

    def test_set_response_moe_routermap_truncation(self, step):
        """When moe_routermap is longer than expected, it should be truncated."""
        routermap = torch.randn(5, 4, 8)  # longer than tokens (3)
        step.set_response(
            text="abc",
            tokens=[1, 2, 3],
            masks=[1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3],
            moe_routermap=routermap,
        )
        assert isinstance(step.moe_routermap, torch.Tensor)
        assert step.moe_routermap.shape[0] == 3  # truncated to match tokens

    # -- mask_out_response --
    def test_mask_out_response(self, step):
        step.set_response(
            text="data",
            tokens=[10, 20, 30],
            masks=[1, 1, 1],
            logprobs=[-0.5, -0.6, -0.7],
            moe_routermap=[],
        )
        step.mask_out_response()
        assert step.masks == [0, 0, 0]
        assert step.logprobs == [0, 0, 0]
        # tokens and text are unchanged
        assert step.tokens == [10, 20, 30]
        assert step.text == "data"

    def test_mask_out_response_asserts_partial_empty(self, step):
        step.set_response(
            text="data",
            tokens=[10],
            masks=[1],
            logprobs=[-0.5],
            moe_routermap=[],
        )
        step.partial_tokens = [99]
        with pytest.raises(AssertionError):
            step.mask_out_response()

    # -- is_empty --
    def test_is_empty_all_zero_masks(self, step):
        step.set_response(
            text="x",
            tokens=[1, 2],
            masks=[0, 0],
            logprobs=[-0.1, -0.2],
            moe_routermap=[],
        )
        assert step.is_empty()

    def test_is_empty_some_masks_one(self, step):
        step.set_response(
            text="x",
            tokens=[1, 2],
            masks=[0, 1],
            logprobs=[-0.1, -0.2],
            moe_routermap=[],
        )
        assert not step.is_empty()

    # -- reset_partial_state --
    def test_reset_partial_state(self, step):
        step.partial_tokens = [1, 2]
        step.partial_text = "hi"
        step.partial_logprobs = [-0.1]
        step.partial_moe_routermap = [[1]]
        step.partial_token_strs = ["h", "i"]
        step.reset_partial_state()
        assert step.partial_tokens == []
        assert step.partial_text == ""
        assert step.partial_logprobs == []
        assert step.partial_moe_routermap == []
        assert step.partial_token_strs == []

    # -- reset --
    def test_reset(self, step):
        step.set_response(
            text="data",
            tokens=[10, 20],
            masks=[1, 1],
            logprobs=[-0.5, -0.6],
            moe_routermap=[],
        )
        step.partial_tokens = [99]
        step.partial_text = "partial"
        step.reset()
        assert step.text == ""
        assert step.tokens == []
        assert step.masks == []
        assert step.logprobs == []
        assert step.token_len == 0
        assert step.moe_routermap == []
        assert step.partial_tokens == []
        assert step.partial_text == ""
        assert step.partial_rollout_max_tokens == 0
        assert step.multi_modal_data is None

    # -- hardened edge cases --

    def test_set_response_empty_tokens(self, step):
        """set_response with no tokens should be a valid no-op."""
        step.set_response("", [], [], [], [])
        assert step.text == ""
        assert step.tokens == []
        assert step.token_len == 0

    def test_set_response_unicode_text_accumulates(self, step):
        """Unicode text should accumulate correctly across calls."""
        step.set_response("Hello \U0001f30d", [1, 2], [1, 1], [-0.1, -0.2], [])
        step.set_response(" \u4f60\u597d", [3], [1], [-0.3], [])
        assert step.text == "Hello \U0001f30d \u4f60\u597d"
        assert step.token_len == 3

    def test_is_empty_with_compensating_masks(self, step):
        """Masks [1, -1] sum to 0, but step has unmasked content at position 0."""
        step.set_response("ab", [1, 2], [1, -1], [-0.1, -0.2], [])
        # sum([1, -1]) = 0 → is_empty returns True, but mask=1 exists
        assert not step.is_empty(), (
            "Step with mask [1, -1] has unmasked content but is_empty() returns True "
            "because sum(masks) <= 0. is_empty should check any(m > 0 for m in masks)."
        )

    def test_is_empty_no_tokens(self, step):
        """Step with no tokens is definitely empty."""
        assert step.is_empty()


# ---------------------------------------------------------------------------
# ProgramState dataclass
# ---------------------------------------------------------------------------
class TestProgramState:
    """Tests for the ProgramState dataclass."""

    @pytest.fixture
    def ps(self):
        return ProgramState(uid="prog-1", session_id="sess-1", group_id="grp-1")

    # -- reset --
    def test_reset_resets_metrics(self, ps):
        ps.metrics.env_time = 99.0
        ps.reset()
        assert ps.metrics.env_time == 0.0

    def test_reset_preserves_identity(self, ps):
        ps.num_steps = 5
        ps.done = True
        ps.reward = 1.0
        ps.termination_reason = TerminationReason.ENV_DONE
        ps.steps = [Step(uid="s", session_id="s")]
        ps.step_rewards = {0: 1.0}
        ps.metadata = {"key": "val"}
        ps.reset()
        assert ps.num_steps == 0
        assert ps.done is False
        assert ps.reward == 0
        assert ps.termination_reason is None
        assert ps.steps == []
        assert ps.step_rewards == {}
        assert ps.metadata == {}
        # preserves identity fields
        assert ps.uid == "prog-1"
        assert ps.session_id == "sess-1"
        assert ps.group_id == "grp-1"

    # -- time bookkeeping --
    def test_record_env_time_increases(self, ps):
        ps.last_llm_call_time = time.monotonic() - 0.05
        ps.record_env_time()
        assert ps.metrics.env_time > 0
        assert ps.metrics.total_time > 0

    def test_record_llm_call_finish_time_updates(self, ps):
        before = time.monotonic()
        ps.record_llm_call_finish_time()
        after = time.monotonic()
        assert before <= ps.last_llm_call_time <= after

    # -- is_trainable --
    def test_is_trainable_not_done(self, ps):
        ps.done = False
        assert ps.is_trainable() is False

    def test_is_trainable_reward_inf(self, ps):
        ps.done = True
        ps.reward = float("inf")
        assert ps.is_trainable() is False

    def test_is_trainable_reward_neg_inf(self, ps):
        ps.done = True
        ps.reward = float("-inf")
        assert ps.is_trainable() is False

    def test_is_trainable_reward_nan(self, ps):
        ps.done = True
        ps.reward = float("nan")
        assert ps.is_trainable() is False

    def test_is_trainable_reward_large_positive(self, ps):
        ps.done = True
        ps.reward = 1e9
        assert ps.is_trainable() is False

    def test_is_trainable_reward_large_negative(self, ps):
        ps.done = True
        ps.reward = -1e9
        assert ps.is_trainable() is False

    def test_is_trainable_no_training_steps(self, ps):
        ps.done = True
        ps.reward = 0.5
        # steps list is empty, prefix tree is empty, training_steps is empty
        assert ps.is_trainable() is False

    def test_is_trainable_with_training_steps(self, ps):
        ps.done = True
        ps.reward = 0.5
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]
        assert ps.is_trainable() is True

    def test_is_trainable_strict_wrong_reason(self, ps):
        ps.done = True
        ps.reward = 0.5
        ps.termination_reason = TerminationReason.MAX_STEPS
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]
        assert ps.is_trainable(strict=False) is True
        assert ps.is_trainable(strict=True) is False

    def test_is_trainable_strict_env_done(self, ps):
        ps.done = True
        ps.reward = 0.5
        ps.termination_reason = TerminationReason.ENV_DONE
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]
        assert ps.is_trainable(strict=True) is True

    def test_is_trainable_boundary_reward_just_below_1e9(self, ps):
        """Test boundary: 1e9-1 should be trainable, 1e9 should not."""
        ps.done = True
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]

        ps.reward = 1e9 - 1
        assert ps.is_trainable() is True

        ps.reward = 1e9
        assert ps.is_trainable() is False

        ps.reward = 1e9 + 1
        assert ps.is_trainable() is False

    def test_is_trainable_boundary_reward_just_above_neg_1e9(self, ps):
        """Test boundary: -1e9+1 should be trainable, -1e9 should not."""
        ps.done = True
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]

        ps.reward = -1e9 + 1
        assert ps.is_trainable() is True

        ps.reward = -1e9
        assert ps.is_trainable() is False

        ps.reward = -1e9 - 1
        assert ps.is_trainable() is False

    # -- get_training_steps --
    def test_get_training_steps_returns_training_steps_if_set(self, ps):
        s = Step(uid="s1", session_id="sess-1")
        s.set_response("ok", [1], [1], [-0.1], [])
        ps.training_steps = [s]
        result = ps.get_training_steps()
        assert result is ps.training_steps

    def test_get_training_steps_from_step_rewards(self, ps):
        s1 = Step(uid="s1", session_id="sess-1")
        s1.set_response("a", [1], [1], [-0.1], [])  # non-empty
        s2 = Step(uid="s2", session_id="sess-1")
        s2.set_response("b", [2], [0], [-0.2], [])  # empty (mask sum = 0)
        ps.steps = [s1, s2]
        ps.step_rewards = {0: 1.0}
        result = ps.get_training_steps()
        assert len(result) == 1
        assert result[0].uid == "s1"

    def test_get_training_steps_empty_when_no_steps(self, ps):
        result = ps.get_training_steps()
        assert result == []

    # -- hardened edge cases --

    def test_get_training_steps_prefers_training_steps_field(self, ps):
        """training_steps field takes priority over prefix tree extraction."""
        s1 = Step(uid="s1", session_id="sess-1")
        s1.set_response("via field", [1], [1], [-0.1], [])
        ps.training_steps = [s1]
        ps.prefix_tree.insert([99], ["tree_token"], [1], [-0.5])
        result = ps.get_training_steps()
        assert result is ps.training_steps

    def test_get_steps_from_prefix_tree_with_branching(self, ps):
        """Branching in prefix tree should produce multiple steps."""
        ps.prefix_tree.insert([1, 2, 3], ["a", "b", "c"], [1, 1, 1], [-0.1, -0.2, -0.3])
        ps.prefix_tree.insert([1, 2, 4], ["a", "b", "d"], [1, 1, 1], [-0.1, -0.2, -0.4])
        steps = ps.get_steps_from_prefix_tree()
        assert len(steps) == 2, f"Expected 2 steps from branching tree, got {len(steps)}"
        token_seqs = {tuple(s.tokens) for s in steps}
        assert (1, 2, 3) in token_seqs
        assert (1, 2, 4) in token_seqs

    def test_get_steps_from_prefix_tree_counts_shared_prefix_once(self, ps):
        """Shared sampled prefix tokens should not receive duplicate branch credit."""
        ps.prefix_tree.insert([1, 2, 3], ["a", "b", "c"], [1, 1, 1], [-0.1, -0.2, -0.3])
        ps.prefix_tree.insert([1, 2, 4], ["a", "b", "d"], [1, 1, 1], [-0.1, -0.2, -0.4])

        steps = ps.get_steps_from_prefix_tree()
        by_tokens = {tuple(s.tokens): s.masks for s in steps}

        assert set(by_tokens) == {(1, 2, 3), (1, 2, 4)}
        assert sorted(by_tokens.values()) == [[0, 0, 1], [1, 1, 1]]
        assert sum(sum(masks) for masks in by_tokens.values()) == 4

    def test_get_steps_from_prefix_tree_single_path(self, ps):
        """Single path should produce exactly one step."""
        ps.prefix_tree.insert([10, 20, 30], ["x", "y", "z"], [1, 1, 1], [-0.1, -0.2, -0.3])
        steps = ps.get_steps_from_prefix_tree()
        assert len(steps) == 1
        assert steps[0].tokens == [10, 20, 30]
        assert steps[0].text == "xyz"

    def test_reset_clears_prefix_tree(self, ps):
        """Reset should clear the prefix tree."""
        ps.prefix_tree.insert([1, 2], ["a", "b"], [1, 1], [-0.1, -0.2])
        assert ps.prefix_tree.size() == 2
        ps.reset()
        assert ps.prefix_tree.size() == 0


# ---------------------------------------------------------------------------
# ProgramManager
# ---------------------------------------------------------------------------
class TestProgramManager:
    """Tests for the ProgramManager class."""

    def _make_program(self, uid, group_id, done=False):
        p = ProgramState(uid=uid, session_id=uid, group_id=group_id)
        p.done = done
        return p

    @pytest.fixture
    def manager(self):
        return ProgramManager()

    # -- add_programs & counts --
    def test_add_and_count(self, manager):
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("b1", "grpB"),
        ]
        manager.add_programs(programs)
        assert manager.get_num_programs() == 3
        assert manager.get_num_completed_programs() == 0
        assert manager.get_num_incomplete_programs() == 3

    def test_completed_count_after_marking_done(self, manager):
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("b1", "grpB"),
        ]
        manager.add_programs(programs)
        # Mark one as done
        manager.programs["grpA"]["a1"].done = True
        assert manager.get_num_completed_programs() == 1
        assert manager.get_num_incomplete_programs() == 2

    # -- pop_programs(completed=False) --
    def test_pop_incomplete(self, manager):
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("b1", "grpB"),
        ]
        manager.add_programs(programs)
        manager.programs["grpA"]["a1"].done = True
        incomplete = manager.pop_programs(completed=False)
        incomplete_uids = {p.uid for p in incomplete}
        assert incomplete_uids == {"a2", "b1"}
        # After popping incomplete, only the completed one remains
        assert manager.get_num_programs() == 1

    def test_pop_incomplete_empty_when_all_done(self, manager):
        programs = [self._make_program("a1", "grpA", done=True)]
        manager.add_programs(programs)
        incomplete = manager.pop_programs(completed=False)
        assert len(incomplete) == 0

    # -- pop_programs(completed=True) --
    def test_pop_completed_only_full_groups(self, manager):
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("b1", "grpB", done=True),
        ]
        manager.add_programs(programs)
        # grpA has incomplete members, grpB is fully done
        completed = manager.pop_programs(completed=True)
        completed_uids = {p.uid for p in completed}
        assert completed_uids == {"b1"}
        # grpA still present
        assert manager.get_num_programs() == 2

    def test_pop_completed_waits_for_all_in_group(self, manager):
        programs = [
            self._make_program("a1", "grpA", done=True),
            self._make_program("a2", "grpA"),
        ]
        manager.add_programs(programs)
        # grpA not fully done yet
        completed = manager.pop_programs(completed=True)
        assert len(completed) == 0
        assert manager.get_num_programs() == 2

    def test_pop_completed_all_done(self, manager):
        programs = [
            self._make_program("a1", "grpA", done=True),
            self._make_program("a2", "grpA", done=True),
        ]
        manager.add_programs(programs)
        completed = manager.pop_programs(completed=True)
        completed_uids = {p.uid for p in completed}
        assert completed_uids == {"a1", "a2"}
        assert manager.get_num_programs() == 0

    def test_pop_completed_removes_empty_groups(self, manager):
        programs = [self._make_program("b1", "grpB", done=True)]
        manager.add_programs(programs)
        manager.pop_programs(completed=True)
        assert "grpB" not in manager.programs

    # -- complex group semantics --
    def test_three_groups_complex_scenario(self, manager):
        """3 groups: grpA (2 members), grpB (1 member), grpC (3 members).
        Mark some done, pop completed, verify only fully-done groups pop."""
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("b1", "grpB"),
            self._make_program("c1", "grpC"),
            self._make_program("c2", "grpC"),
            self._make_program("c3", "grpC"),
        ]
        manager.add_programs(programs)

        # Mark grpB fully done and one member of grpC
        manager.programs["grpB"]["b1"].done = True
        manager.programs["grpC"]["c1"].done = True
        manager.programs["grpC"]["c2"].done = True

        # Pop completed: only grpB should come out (fully done)
        completed = manager.pop_programs(completed=True)
        assert {p.uid for p in completed} == {"b1"}
        assert manager.get_num_programs() == 5  # a1, a2, c1, c2, c3

        # Now finish grpC
        manager.programs["grpC"]["c3"].done = True
        completed = manager.pop_programs(completed=True)
        assert {p.uid for p in completed} == {"c1", "c2", "c3"}
        assert manager.get_num_programs() == 2  # a1, a2

    # -- concurrent access --
    def test_concurrent_add_pop(self, manager):
        """Add and pop from multiple threads simultaneously, verify no data corruption."""
        errors = []

        def add_programs(group_prefix, start, count):
            try:
                programs = [
                    self._make_program(f"{group_prefix}_{i}", group_prefix, done=(i % 2 == 0))
                    for i in range(start, start + count)
                ]
                manager.add_programs(programs)
            except Exception as e:
                errors.append(e)

        def pop_completed():
            try:
                for _ in range(10):
                    manager.pop_programs(completed=True)
            except Exception as e:
                errors.append(e)

        def pop_incomplete():
            try:
                for _ in range(10):
                    manager.pop_programs(completed=False)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=add_programs, args=(f"grp{i}", 0, 10)))
        threads.append(threading.Thread(target=pop_completed))
        threads.append(threading.Thread(target=pop_incomplete))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors in threads: {errors}"

    # -- hardened edge cases --

    def test_add_duplicate_uid_same_group_warns(self, manager):
        """Adding two programs with the same uid to the same group should log a warning."""
        import logging

        p1 = ProgramState(uid="dup", session_id="s1", group_id="g")
        p1.reward = 1.0
        p2 = ProgramState(uid="dup", session_id="s2", group_id="g")
        p2.reward = 2.0
        # The overwrite is logged; last writer wins
        with self._capture_log("axon.engine.state.program_state", logging.WARNING) as logs:
            manager.add_programs([p1, p2])
        assert any("Duplicate uid" in msg for msg in logs), "Expected a warning about duplicate uid"
        # Second program wins
        assert manager.get_num_programs() == 1
        assert manager.programs["g"]["dup"].reward == 2.0

    @staticmethod
    @contextlib.contextmanager
    def _capture_log(logger_name, level):
        """Capture log messages at *level* from *logger_name*."""
        import logging

        captured: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                captured.append(self.format(record))

        log = logging.getLogger(logger_name)
        handler = _Handler()
        handler.setLevel(level)
        log.addHandler(handler)
        old_level = log.level
        log.setLevel(level)
        try:
            yield captured
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)

    def test_pop_completed_mixed_groups_multiple_rounds(self, manager):
        """Multiple pop rounds should correctly drain groups as they complete."""
        programs = [
            self._make_program("a1", "grpA"),
            self._make_program("a2", "grpA"),
            self._make_program("a3", "grpA"),
            self._make_program("b1", "grpB"),
            self._make_program("b2", "grpB"),
        ]
        manager.add_programs(programs)
        manager.programs["grpB"]["b1"].done = True
        manager.programs["grpB"]["b2"].done = True
        completed = manager.pop_programs(completed=True)
        assert {p.uid for p in completed} == {"b1", "b2"}
        assert manager.get_num_programs() == 3

        manager.programs["grpA"]["a1"].done = True
        completed = manager.pop_programs(completed=True)
        assert len(completed) == 0  # grpA not fully done

        manager.programs["grpA"]["a2"].done = True
        manager.programs["grpA"]["a3"].done = True
        completed = manager.pop_programs(completed=True)
        assert {p.uid for p in completed} == {"a1", "a2", "a3"}
        assert manager.get_num_programs() == 0

    def test_get_completed_programs_statistics_empty(self, manager):
        """Statistics on empty manager should have all zero counts."""
        stats = manager.get_completed_programs_statistics()
        for key, val in stats.items():
            assert val == 0, f"Expected 0 for {key}, got {val}"

    def test_single_program_group_lifecycle(self, manager):
        """Single-program group should pop immediately when done."""
        p = self._make_program("solo", "grpSolo")
        manager.add_programs([p])
        assert manager.get_num_programs() == 1
        manager.programs["grpSolo"]["solo"].done = True
        completed = manager.pop_programs(completed=True)
        assert len(completed) == 1
        assert completed[0].uid == "solo"
        assert manager.get_num_programs() == 0
        assert "grpSolo" not in manager.programs
