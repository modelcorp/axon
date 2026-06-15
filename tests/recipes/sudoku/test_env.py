"""
Comprehensive tests for the Sudoku environment.

Covers:
  * Puzzle generation correctness (valid grid, uniqueness, clue mask).
  * Action parsing (canonical forms and variants).
  * Step semantics (placement, clear, replace, immutable clues, constraint
    violations, out-of-range, unparseable, no-op writes).
  * Reward bounds and termination.
  * Defense against known reward-hacking strategies.

Run:
    pytest tests/recipes/sudoku/test_env.py -v
"""

from __future__ import annotations

import random as _random
import sys
from pathlib import Path

import numpy as np
import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from sudoku.env import (  # noqa: E402
    SudokuEnv,
    _count_solutions,
    _generate_solved_grid,
    _is_full,
    _is_valid_complete,
    _make_puzzle,
    _parse_action,
    _violates_constraint,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_first_blank(env: SudokuEnv) -> tuple[int, int]:
    rs, cs = np.where(env.board == 0)
    return int(rs[0]), int(cs[0])


def solve_via_ground_truth(env: SudokuEnv) -> tuple[float, bool]:
    """Walk the solution and play it. Returns (final_reward, won)."""
    final_reward = 0.0
    for r in range(env.size):
        for c in range(env.size):
            if env.clue_mask[r, c]:
                continue
            v = int(env.solution[r, c])
            _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}={v}")
            final_reward = rew
            if done:
                return final_reward, env.success()
    return final_reward, env.success()


# --------------------------------------------------------------------------- #
# 1. Solution generator
# --------------------------------------------------------------------------- #
class TestSolutionGenerator:
    @pytest.mark.parametrize("size,seed", [(4, 0), (4, 7), (9, 0), (9, 13)])
    def test_shape(self, size, seed):
        rng = _random.Random(seed)
        sol = _generate_solved_grid(size, rng)
        assert sol.shape == (size, size)

    @pytest.mark.parametrize("size,seed", [(4, 0), (4, 7), (9, 0), (9, 13)])
    def test_valid_sudoku(self, size, seed):
        rng = _random.Random(seed)
        sol = _generate_solved_grid(size, rng)
        assert _is_valid_complete(sol)

    @pytest.mark.parametrize("size,seed", [(4, 0), (4, 7), (9, 0), (9, 13)])
    def test_unique_solution(self, size, seed):
        rng = _random.Random(seed)
        sol = _generate_solved_grid(size, rng)
        assert _count_solutions(sol.copy(), limit=2) == 1


# --------------------------------------------------------------------------- #
# 2. Puzzle uniqueness
# --------------------------------------------------------------------------- #
class TestPuzzleUniqueness:
    @pytest.mark.parametrize("size,num_clues", [(4, 6), (9, 36)])
    @pytest.mark.parametrize("seed", range(8))
    def test_unique_solution(self, size, num_clues, seed):
        rng = _random.Random(seed)
        sol = _generate_solved_grid(size, rng)
        puzzle, actual = _make_puzzle(sol, num_clues, rng, enforce_unique=True)
        assert _count_solutions(puzzle.copy(), limit=2) == 1

    @pytest.mark.parametrize("size,num_clues", [(4, 6), (9, 36)])
    @pytest.mark.parametrize("seed", range(8))
    def test_clues_match_solution(self, size, num_clues, seed):
        rng = _random.Random(seed)
        sol = _generate_solved_grid(size, rng)
        puzzle, _ = _make_puzzle(sol, num_clues, rng, enforce_unique=True)
        mask = puzzle != 0
        assert np.array_equal(puzzle[mask], sol[mask])

    def test_non_unique_mode_produces_multi_solution_puzzles(self):
        multi = 0
        for seed in range(6):
            rng = _random.Random(seed + 1000)
            sol = _generate_solved_grid(9, rng)
            puzzle, _ = _make_puzzle(sol, num_clues=20, rng=rng, enforce_unique=False)
            if _count_solutions(puzzle.copy(), limit=2) > 1:
                multi += 1
        assert multi >= 1


# --------------------------------------------------------------------------- #
# 3. Action parsing
# --------------------------------------------------------------------------- #
class TestActionParsing:
    @pytest.mark.parametrize(
        "action,expected",
        [
            ("R3C5=7", (3, 5, 7)),
            ("r3c5=7", (3, 5, 7)),
            ("R3C5=.", (3, 5, 0)),
            ("3,5,7", (3, 5, 7)),
            ("3 5 7", (3, 5, 7)),
            ("  R 3 C 5 = 7 ", (3, 5, 7)),
            ("R10C10=10", (10, 10, 10)),
            ("garbage", None),
            ("", None),
            (None, None),
            ("R3C5", None),
            ("=7", None),
        ],
    )
    def test_parse(self, action, expected):
        assert _parse_action(action) == expected


# --------------------------------------------------------------------------- #
# 4. Step semantics
# --------------------------------------------------------------------------- #
class TestStepSemantics:
    def test_valid_placement(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        obs, rew, done, info = env.step(f"R{r + 1}C{c + 1}={v}")
        assert rew == 0.0
        assert not done
        assert info["action_is_effective"]
        assert info["invalid_move"] == 0

    def test_noop_write_rejected(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        env.step(f"R{r + 1}C{c + 1}={v}")
        _, rew, _, info = env.step(f"R{r + 1}C{c + 1}={v}")
        assert rew == env.invalid_penalty
        assert info["invalid_move"] == 1
        assert info["reason"] == "noop_write"

    def test_clear_placed_cell(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        env.step(f"R{r + 1}C{c + 1}={v}")
        _, rew, _, info = env.step(f"R{r + 1}C{c + 1}=.")
        assert rew == 0.0
        assert info["reason"] == "clear"

    def test_clear_empty_cell(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        r, c = find_first_blank(env)
        _, rew, _, info = env.step(f"R{r + 1}C{c + 1}=.")
        assert rew == env.invalid_penalty
        assert info["reason"] == "clear_empty"

    def test_modify_clue(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        cr, cc = map(int, np.argwhere(env.clue_mask)[0])
        _, rew, _, info = env.step(f"R{cr + 1}C{cc + 1}=1")
        assert rew == env.invalid_penalty
        assert info["reason"] == "modify_clue"

    def test_out_of_range_cell(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        _, rew, _, info = env.step(f"R{env.size + 1}C1=1")
        assert rew == env.invalid_penalty
        assert info["reason"] == "out_of_range_cell"

    def test_out_of_range_value(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        r, c = find_first_blank(env)
        _, rew, _, info = env.step(f"R{r + 1}C{c + 1}={env.size + 1}")
        assert rew == env.invalid_penalty
        assert info["reason"] == "out_of_range_value"

    def test_unparseable(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        _, rew, _, info = env.step("not an action")
        assert rew == env.invalid_penalty
        assert info["reason"] == "unparseable"

    def test_nonstring_action(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        _, rew, _, info = env.step(None)
        assert rew == env.invalid_penalty
        assert info["reason"] == "unparseable"

    def test_constraint_violation(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=42)
        env.reset()
        rs, cs = np.where(env.board == 0)
        r0, c0 = int(rs[0]), int(cs[0])
        row_vals = [int(x) for x in env.board[r0] if x != 0]
        if row_vals:
            _, rew, _, info = env.step(f"R{r0 + 1}C{c0 + 1}={row_vals[0]}")
            assert rew == env.invalid_penalty
            assert info["reason"] == "constraint_violation"

    def test_immutable_clues_full_audit(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=80, seed=3)
        env.reset()
        initial_clues = env.puzzle.copy()
        for r in range(env.size):
            for c in range(env.size):
                if env.clue_mask[r, c]:
                    for v in range(0, env.size + 1):
                        env.step(f"R{r + 1}C{c + 1}={v}")
        assert np.array_equal(env.board * env.clue_mask, initial_clues)


# --------------------------------------------------------------------------- #
# 5. Termination & reward bounds
# --------------------------------------------------------------------------- #
class TestTermination:
    @pytest.mark.parametrize("size,num_clues", [(4, 8), (9, 36)])
    def test_solve_path(self, size, num_clues):
        env = SudokuEnv(size=size, num_clues=num_clues, max_turns=500, seed=1)
        env.reset()
        rew, won = solve_via_ground_truth(env)
        assert won
        assert rew == env.target_reward

    def test_after_done_is_zero(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=500, seed=1)
        env.reset()
        rew, _ = solve_via_ground_truth(env)
        assert rew == env.target_reward
        total = 0.0
        for _ in range(20):
            _, r, done, _ = env.step("R1C1=1")
            total += r
            assert done
        assert total == 0.0

    def test_max_turns_termination(self):
        env = SudokuEnv(size=9, num_clues=36, max_turns=10, seed=2)
        env.reset()
        for _ in range(10):
            _, r, done, _ = env.step("garbage")
        assert done
        assert env.turns == 10
        assert abs(r - env.invalid_penalty) < 1e-9

    def test_sparse_reward_at_timeout(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=20, seed=99)
        env.reset()
        blanks = [tuple(map(int, x)) for x in np.argwhere(~env.clue_mask)]
        for r, c in blanks[: len(blanks) // 2]:
            env.step(f"R{r + 1}C{c + 1}={int(env.solution[r, c])}")
        last_info = {}
        for _ in range(100):
            _, last, done, last_info = env.step("garbage")
            if done:
                break
        assert done
        assert abs(last - env.invalid_penalty) < 1e-9
        assert "progress" in last_info
        assert 0.0 < last_info["progress"] < 1.0

    def test_max_turns_1(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=1, seed=30)
        env.reset()
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        _, rew, done, info = env.step(f"R{r + 1}C{c + 1}={v}")
        assert done
        assert abs(rew - 0.0) < 1e-9
        assert "progress" in info


# --------------------------------------------------------------------------- #
# 6. Reward-hacking defense
# --------------------------------------------------------------------------- #
class TestRewardHacking:
    def test_invalid_spam_negative(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=50, seed=5)
        env.reset()
        total = sum_rewards_until_done(env, "nope")
        assert total <= 0.0

    def test_place_clear_cycle_no_positive(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=6)
        env.reset()
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        total = 0.0
        done = False
        while not done:
            _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}={v}")
            total += rew
            if done:
                break
            _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}=.")
            total += rew
        assert total <= 1.0

    def test_wrong_completion_blocked(self):
        bad = np.ones((4, 4), dtype=np.int64)
        assert not _is_valid_complete(bad)
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=7)
        assert _is_valid_complete(env.solution)

    def test_total_reward_bounded_by_target(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=500, seed=4)
        env.reset()
        total = 0.0
        done = False
        for r in range(env.size):
            for c in range(env.size):
                if env.clue_mask[r, c]:
                    continue
                _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}={int(env.solution[r, c])}")
                total += rew
                if done:
                    break
            if done:
                break
        assert abs(total - env.target_reward) < 1e-9

    def test_greedy_easy_cells_negative(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=40, seed=40)
        env.reset()
        blanks = [tuple(map(int, x)) for x in np.argwhere(~env.clue_mask)]
        total = 0.0
        done = False
        for r, c in blanks[: len(blanks) // 2]:
            _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}={int(env.solution[r, c])}")
            total += rew
            if done:
                break
        if not done:
            total += sum_rewards_until_done(env, "garbage")
        assert total < 0.0
        assert total < env.target_reward

    def test_grpo_distinguishability(self):
        seed, mt, size, nc = 50, 60, 4, 8

        # A: solve
        env_a = SudokuEnv(size=size, num_clues=nc, max_turns=mt, seed=seed)
        env_a.reset()
        total_a = 0.0
        done = False
        for r in range(size):
            for c in range(size):
                if not env_a.clue_mask[r, c]:
                    _, rew, done, _ = env_a.step(f"R{r + 1}C{c + 1}={int(env_a.solution[r, c])}")
                    total_a += rew
                    if done:
                        break
            if done:
                break

        # B: half fill + invalid spam
        env_b = SudokuEnv(size=size, num_clues=nc, max_turns=mt, seed=seed)
        env_b.reset()
        blanks = [tuple(map(int, x)) for x in np.argwhere(~env_b.clue_mask)]
        total_b = 0.0
        done = False
        for r, c in blanks[: len(blanks) // 2]:
            _, rew, done, _ = env_b.step(f"R{r + 1}C{c + 1}={int(env_b.solution[r, c])}")
            total_b += rew
            if done:
                break
        if not done:
            total_b += sum_rewards_until_done(env_b, "garbage")

        # C: all invalid
        env_c = SudokuEnv(size=size, num_clues=nc, max_turns=mt, seed=seed)
        env_c.reset()
        total_c = sum_rewards_until_done(env_c, "garbage")

        assert total_a > 0
        assert total_b < 0
        assert total_c < 0
        assert total_a > total_b > total_c

    def test_overwrite_attack_bounded(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=100, seed=21)
        env.reset()
        r, c = find_first_blank(env)
        vals = []
        for v in range(1, env.size + 1):
            env2 = SudokuEnv(size=4, num_clues=8, max_turns=100, seed=21)
            env2.reset()
            _, _, _, info = env2.step(f"R{r + 1}C{c + 1}={v}")
            if info["action_is_effective"]:
                vals.append(v)
        if len(vals) >= 2:
            env.step(f"R{r + 1}C{c + 1}={vals[0]}")
            total = 0.0
            for i in range(40):
                nv = vals[1] if i % 2 == 0 else vals[0]
                _, rew, done, _ = env.step(f"R{r + 1}C{c + 1}={nv}")
                total += rew
                if done:
                    break
            assert total <= 1.0

    def test_max_negative_reward(self):
        mt = 50
        env = SudokuEnv(size=4, num_clues=8, max_turns=mt, seed=20)
        env.reset()
        total = sum_rewards_until_done(env, "garbage")
        assert abs(total - mt * env.invalid_penalty) < 1e-9


# --------------------------------------------------------------------------- #
# 7. Input validation
# --------------------------------------------------------------------------- #
class TestInputValidation:
    @pytest.mark.parametrize(
        "cfg",
        [
            {"size": 3},
            {"size": 0},
            {"max_turns": 0},
            {"size": 4, "num_clues": -1},
            {"size": 4, "num_clues": 17},
        ],
    )
    def test_bad_config_raises(self, cfg):
        with pytest.raises(ValueError):
            SudokuEnv(**cfg)


# --------------------------------------------------------------------------- #
# 8. Reset / determinism / factory
# --------------------------------------------------------------------------- #
class TestResetAndFactory:
    def test_reset_idempotent(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=11)
        env.reset()
        board0 = env.board.copy()
        sol0 = env.solution.copy()
        mask0 = env.clue_mask.copy()
        r, c = find_first_blank(env)
        env.step(f"R{r + 1}C{c + 1}={int(env.solution[r, c])}")
        env.reset()
        assert np.array_equal(env.board, board0)
        assert np.array_equal(env.solution, sol0)
        assert np.array_equal(env.clue_mask, mask0)
        assert env.turns == 0
        assert not env._game_over
        assert not env._won

    def test_seed_determinism(self):
        env1 = SudokuEnv(size=9, num_clues=36, max_turns=200, seed=2025)
        env2 = SudokuEnv(size=9, num_clues=36, max_turns=200, seed=2025)
        assert np.array_equal(env1.solution, env2.solution)
        assert np.array_equal(env1.puzzle, env2.puzzle)

    def test_from_dict(self):
        env = SudokuEnv.from_dict({"size": 4, "num_clues": 8, "max_turns": 50, "seed": 17})
        assert env.size == 4 and env.max_turns == 50 and env.seed == 17
        obs, _ = env.reset()
        assert isinstance(obs, str) and len(obs) > 0


# --------------------------------------------------------------------------- #
# 9. Constraint checks
# --------------------------------------------------------------------------- #
class TestConstraintChecks:
    def test_empty_board_accepts(self):
        b = np.zeros((4, 4), dtype=np.int64)
        assert not _violates_constraint(b, 0, 0, 1)

    def test_row_conflict(self):
        b = np.zeros((4, 4), dtype=np.int64)
        b[0, 0] = 1
        assert _violates_constraint(b, 0, 1, 1)

    def test_column_conflict(self):
        b = np.zeros((4, 4), dtype=np.int64)
        b[0, 0] = 1
        assert _violates_constraint(b, 1, 0, 1)

    def test_box_conflict(self):
        b = np.zeros((4, 4), dtype=np.int64)
        b[0, 0] = 1
        assert _violates_constraint(b, 1, 1, 1)

    def test_no_conflict(self):
        b = np.zeros((4, 4), dtype=np.int64)
        b[0, 0] = 1
        assert not _violates_constraint(b, 2, 2, 1)

    def test_self_conflict_regression(self):
        b = np.zeros((4, 4), dtype=np.int64)
        b[0, 0] = 3
        assert _violates_constraint(b, 0, 0, 3)
        b[0, 0] = 0
        assert not _violates_constraint(b, 0, 0, 3)

    def test_every_placed_cell_valid(self):
        env = SudokuEnv(size=4, num_clues=6, max_turns=500, seed=70)
        env.reset()
        blanks = [tuple(map(int, x)) for x in np.argwhere(~env.clue_mask)]
        done = False
        for r, c in blanks:
            for v in range(1, env.size + 1):
                _, _, done, info = env.step(f"R{r + 1}C{c + 1}={v}")
                if info.get("action_is_effective") or done:
                    break
            if done:
                break
        board = env.board.copy()
        for r in range(env.size):
            for c in range(env.size):
                v = int(board[r, c])
                if v == 0:
                    continue
                board[r, c] = 0
                assert not _violates_constraint(board, r, c, v)
                board[r, c] = v


# --------------------------------------------------------------------------- #
# 10. Dead-end recovery, render, stress tests
# --------------------------------------------------------------------------- #
class TestMisc:
    def test_dead_end_recovery(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=500, seed=10)
        env.reset()
        r0, c0 = find_first_blank(env)
        correct_v = int(env.solution[r0, c0])
        wrong_v = None
        for v in range(1, env.size + 1):
            if v == correct_v:
                continue
            _, _, _, info = env.step(f"R{r0 + 1}C{c0 + 1}={v}")
            if info.get("action_is_effective"):
                wrong_v = v
                break
        if wrong_v is not None:
            assert int(env.board[r0, c0]) == wrong_v
            env.step(f"R{r0 + 1}C{c0 + 1}=.")
            env.step(f"R{r0 + 1}C{c0 + 1}={correct_v}")
            assert int(env.board[r0, c0]) == correct_v
            rew, won = solve_via_ground_truth(env)
            assert won and rew == env.target_reward

    @pytest.mark.parametrize("seed", range(5))
    def test_constraint_consistent_full_board(self, seed):
        env = SudokuEnv(size=4, num_clues=6, max_turns=500, seed=seed + 100)
        assert _is_valid_complete(env.solution)
        bad = env.solution.copy()
        bad[0, 0], bad[0, 1] = bad[0, 1], bad[0, 0]
        assert not _is_valid_complete(bad)

    @pytest.mark.parametrize("seed", range(20))
    def test_puzzle_not_presolved(self, seed):
        env = SudokuEnv(size=9, num_clues=36, max_turns=200, seed=seed)
        assert not _is_full(env.board)

    def test_render_consistency(self):
        env = SudokuEnv(size=4, num_clues=8, max_turns=200, seed=60)
        env.reset()
        obs_init = env.render()
        n_dots = obs_init.count(".")
        n_blanks = int((env.board == 0).sum())
        assert n_dots == n_blanks
        r, c = find_first_blank(env)
        v = int(env.solution[r, c])
        obs_after, _, _, _ = env.step(f"R{r + 1}C{c + 1}={v}")
        assert obs_after.count(".") == n_dots - 1
        assert str(v) in obs_after

    @pytest.mark.parametrize("seed", range(10))
    def test_valid_complete_equals_solution(self, seed):
        env = SudokuEnv(size=4, num_clues=6, max_turns=200, seed=seed + 200)
        board = env.puzzle.copy()
        for r in range(env.size):
            for c in range(env.size):
                if board[r, c] == 0:
                    board[r, c] = env.solution[r, c]
        assert _is_valid_complete(board)
        assert np.array_equal(board, env.solution)

    def test_hard_9x9(self):
        env = SudokuEnv(size=9, num_clues=28, max_turns=500, seed=300)
        env.reset()
        assert env.actual_num_clues >= 28
        rew, won = solve_via_ground_truth(env)
        assert won and rew == env.target_reward


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def sum_rewards_until_done(env: SudokuEnv, action: str) -> float:
    total = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(action)
        total += r
    return total
