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
"""
Sudoku environment for Axon.

A multi-turn Sudoku-solving env. Each step the agent emits a single placement
of the form `R<row>C<col>=<val>` (1-indexed). The env validates against
Sudoku rules (row/col/box uniqueness) and updates the board.

## Sizes
- size must be a perfect square: 4 (2x2 boxes), 9 (3x3 boxes — standard).
- For training, 9x9 with ~30-40 clues is a reasonable difficulty.

## Action format (string)
- "R<row>C<col>=<val>"   place a value (1..N)
- "R<row>C<col>=."       clear a previously placed (non-clue) cell
- Any unparseable / out-of-range string is treated as INVALID.

## Reward (sparse — no partial credit)
- Valid placement / clear:        +0.0
- Constraint violation:           -0.1, no state change (continue)
- Overwrite a clue or invalid:    -0.1, no state change (continue)
- Solve the puzzle (board full,
  matches the unique solution):   +2.0  (terminal)
- max_turns reached without
  solving:                        +0.0  (terminal, no partial credit)

Designed for GRPO with a difficulty curriculum (4x4 easy → 9x9 hard).
Sparse reward avoids local-optima hacks where the agent fills only easy
cells to farm partial credit.
"""

from __future__ import annotations

import math
import random
import re
import string
from typing import Any

import numpy as np

from axon.core import BaseEnv, register_env

DEFAULT_SIZE: int = 9
DEFAULT_NUM_CLUES: int = 36  # ~half of 81 cells
DEFAULT_MAX_TURNS: int = 200  # ~ enough turns to fill 81 cells with retries
DEFAULT_TARGET_REWARD: float = 1.0  # solve reward
DEFAULT_INVALID_PENALTY: float = -0.1
DEFAULT_ENFORCE_UNIQUE: bool = True  # ensures generated puzzles have one solution


# ----------------------------------------------------------------------------- #
# Sudoku puzzle generation
# ----------------------------------------------------------------------------- #
def _generate_solved_grid(size: int, rng: random.Random) -> np.ndarray:
    """Generate a random valid Sudoku solution using randomized backtracking."""
    box = int(math.isqrt(size))
    if box * box != size:
        raise ValueError(f"size must be a perfect square (4, 9, ...), got {size}")

    grid = np.zeros((size, size), dtype=np.int64)

    def is_safe(r: int, c: int, v: int) -> bool:
        if v in grid[r, :]:
            return False
        if v in grid[:, c]:
            return False
        br, bc = (r // box) * box, (c // box) * box
        if v in grid[br : br + box, bc : bc + box]:
            return False
        return True

    def fill(idx: int) -> bool:
        if idx == size * size:
            return True
        r, c = divmod(idx, size)
        values = list(range(1, size + 1))
        rng.shuffle(values)
        for v in values:
            if is_safe(r, c, v):
                grid[r, c] = v
                if fill(idx + 1):
                    return True
                grid[r, c] = 0
        return False

    if not fill(0):
        # Should be virtually impossible for valid sizes (4, 9).
        raise RuntimeError("Failed to generate a Sudoku solution")
    return grid


def _count_solutions(board: np.ndarray, limit: int = 2) -> int:
    """Count the number of Sudoku solutions of `board`, stopping early at `limit`.

    Uses backtracking with the "most constrained cell first" heuristic so that
    near-complete boards finish in microseconds and uniqueness checks during
    puzzle generation stay fast for 9x9. The board is mutated during the search
    and restored before return.
    """
    size = board.shape[0]
    found = [0]

    def find_best_empty() -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_n = size + 2
        for r in range(size):
            for c in range(size):
                if board[r, c] != 0:
                    continue
                # Count legal candidates; bail out as soon as we know this cell
                # is no better than the current best.
                n = 0
                for v in range(1, size + 1):
                    if not _violates_constraint(board, r, c, v):
                        n += 1
                        if n >= best_n:
                            break
                if n < best_n:
                    best_n = n
                    best = (r, c)
                    if n <= 1:
                        return best  # 0 or 1 candidates: can't do better
        return best

    def solve() -> None:
        if found[0] >= limit:
            return
        cell = find_best_empty()
        if cell is None:
            found[0] += 1
            return
        r, c = cell
        for v in range(1, size + 1):
            if not _violates_constraint(board, r, c, v):
                board[r, c] = v
                solve()
                board[r, c] = 0
                if found[0] >= limit:
                    return

    solve()
    return found[0]


def _make_puzzle(
    solution: np.ndarray,
    num_clues: int,
    rng: random.Random,
    enforce_unique: bool = True,
) -> tuple[np.ndarray, int]:
    """Build a puzzle from `solution` by removing cells.

    If ``enforce_unique`` (default), each removal is reverted when it would leave
    the puzzle with multiple completions. The function returns the puzzle along
    with the actual number of clues remaining (which may be > the requested
    `num_clues` when uniqueness blocks further removal).
    """
    size = solution.shape[0]
    n_total = size * size
    target_clues = max(0, min(n_total, int(num_clues)))
    n_remove_target = n_total - target_clues

    puzzle = solution.copy()
    flat_idx = list(range(n_total))
    rng.shuffle(flat_idx)

    n_removed = 0
    for k in flat_idx:
        if n_removed >= n_remove_target:
            break
        r, c = divmod(k, size)
        backup = int(puzzle[r, c])
        puzzle[r, c] = 0
        if enforce_unique and _count_solutions(puzzle, limit=2) != 1:
            puzzle[r, c] = backup  # restore — removing this cell broke uniqueness
        else:
            n_removed += 1

    actual_clues = n_total - n_removed
    return puzzle, actual_clues


# ----------------------------------------------------------------------------- #
# Action parsing
# ----------------------------------------------------------------------------- #
# Accepts:    R3C5=7    r3c5=7    R3,C5=7    3,5,7    3 5 7    3-5-7
# Clears:     R3C5=.    R3C5=0    3,5,0
_ACTION_RE = re.compile(
    r"""
    ^\s*
    (?:r\s*)?(\d+)              # row
    \s*[,\s\-x]*\s*
    (?:c\s*)?(\d+)              # col
    \s*[=,\s\-:]+\s*
    (\d+|\.)                    # value (digit or '.')
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_action(action: str | None) -> tuple[int, int, int] | None:
    """Parse an agent action string into (row_1idx, col_1idx, value).
    value 0 means "clear". Returns None if the string can't be parsed."""
    if action is None:
        return None
    s = str(action).strip()
    if not s:
        return None
    m = _ACTION_RE.match(s)
    if not m:
        return None
    r = int(m.group(1))
    c = int(m.group(2))
    v_str = m.group(3)
    v = 0 if v_str == "." else int(v_str)
    return r, c, v


# ----------------------------------------------------------------------------- #
# Render
# ----------------------------------------------------------------------------- #
def _render_board(board: np.ndarray) -> str:
    """Pretty-print the Sudoku board with box separators. Empty = '.'."""
    size = board.shape[0]
    box = int(math.isqrt(size))
    cell_w = len(str(size))
    lines: list[str] = []

    # column header (1-indexed columns)
    header = "    " + " ".join(str(c + 1).rjust(cell_w) for c in range(size))
    lines.append(header)

    # box separator line width matches the printed grid
    row_chars = (cell_w + 1) * size + (box - 1) * 2 - 1  # inner box dividers add 2 chars each ("| ")
    sep = "    " + "-" * row_chars

    for r in range(size):
        if r > 0 and r % box == 0:
            lines.append(sep)
        cells: list[str] = []
        for c in range(size):
            if c > 0 and c % box == 0:
                cells.append("|")
            v = int(board[r, c])
            cells.append(("." if v == 0 else str(v)).rjust(cell_w))
        lines.append(f"{str(r + 1).rjust(2)}: " + " ".join(cells))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- #
# Sudoku rules
# ----------------------------------------------------------------------------- #
def _violates_constraint(board: np.ndarray, r: int, c: int, v: int) -> bool:
    """Return True if placing v at (r, c) would conflict with another cell
    in its row, column, or box. (Treats the current contents of (r,c) as 0.)"""
    size = board.shape[0]
    box = int(math.isqrt(size))
    if v in board[r, :]:
        return True
    if v in board[:, c]:
        return True
    br, bc = (r // box) * box, (c // box) * box
    if v in board[br : br + box, bc : bc + box]:
        return True
    return False


def _is_full(board: np.ndarray) -> bool:
    return not bool(np.any(board == 0))


def _is_valid_complete(board: np.ndarray) -> bool:
    """True iff `board` is a full, internally-consistent Sudoku grid (every
    row, column and box contains 1..N exactly once). Used as a defense-in-depth
    win check independent of the stored solution."""
    size = board.shape[0]
    box = int(math.isqrt(size))
    if box * box != size:
        return False
    if not _is_full(board):
        return False
    expected = set(range(1, size + 1))
    for r in range(size):
        if set(int(v) for v in board[r, :]) != expected:
            return False
    for c in range(size):
        if set(int(v) for v in board[:, c]) != expected:
            return False
    for br in range(0, size, box):
        for bc in range(0, size, box):
            block = board[br : br + box, bc : bc + box].flatten()
            if set(int(v) for v in block) != expected:
                return False
    return True


# ----------------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------------- #
@register_env("sudoku")
class SudokuEnv(BaseEnv):
    """
    Sudoku-solving environment.

    ## Action
    A string action: ``"R<row>C<col>=<val>"`` with 1-indexed coords.
    Use ``=.`` or ``=0`` to clear a previously placed cell.

    ## Rewards (sparse)
    - Valid placement / clear:    0.0
    - Invalid action / overwrite
      a clue / constraint break:  ``invalid_penalty`` (default -0.1), no state change
    - Solve:                      +``target_reward`` (default 2.0), terminal
    - max_turns reached:          0.0 (no partial credit; progress is logged in ``info``)
    """

    INVALID_ACTION = "INVALID"

    def __init__(self, **kwargs: Any):
        self.size: int = int(kwargs.pop("size", DEFAULT_SIZE))
        self.num_clues: int = int(kwargs.pop("num_clues", DEFAULT_NUM_CLUES))
        self.max_turns: int = int(kwargs.pop("max_turns", DEFAULT_MAX_TURNS))
        self.seed: int = int(kwargs.pop("seed", 0))
        self.target_reward: float = float(kwargs.pop("target_reward", DEFAULT_TARGET_REWARD))
        self.invalid_penalty: float = float(kwargs.pop("invalid_penalty", DEFAULT_INVALID_PENALTY))
        self.enforce_unique: bool = bool(kwargs.pop("enforce_unique", DEFAULT_ENFORCE_UNIQUE))

        # Input validation. Catch bad configs early — silently producing a
        # 1x1 board or a 0-turn episode is a much worse failure mode at
        # training time.
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")
        box_size = int(math.isqrt(self.size))
        if box_size * box_size != self.size:
            raise ValueError(f"size must be a perfect square (4, 9, 16, ...), got {self.size}")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if not (0 <= self.num_clues <= self.size * self.size):
            raise ValueError(f"num_clues must be in [0, {self.size * self.size}], got {self.num_clues}")

        self._rng = random.Random(self.seed)
        self.id: str = "".join(self._rng.choices(string.ascii_letters + string.digits, k=6))

        self.solution: np.ndarray = _generate_solved_grid(self.size, self._rng)
        self.puzzle, self.actual_num_clues = _make_puzzle(
            self.solution, self.num_clues, self._rng, enforce_unique=self.enforce_unique
        )
        self.board: np.ndarray = self.puzzle.copy()
        # clue_mask[r,c] = True if the cell was a given clue (immutable).
        self.clue_mask: np.ndarray = self.puzzle != 0

        self.turns: int = 0
        self._game_over: bool = False
        self._won: bool = False

        self.env_kwargs = {
            "size": self.size,
            "num_clues": self.num_clues,
            "max_turns": self.max_turns,
            "seed": self.seed,
            "target_reward": self.target_reward,
            "invalid_penalty": self.invalid_penalty,
            "enforce_unique": self.enforce_unique,
        }

    # ------------------------------------------------------------------ #
    # Gym interface
    # ------------------------------------------------------------------ #
    def reset(self):
        self.__init__(**self.env_kwargs)
        return self.render(), {}

    def finished(self) -> bool:
        return self._game_over or self.turns >= self.max_turns

    def success(self) -> bool:
        return self._won

    def step(self, action: Any):
        if self.finished():
            return self.render(), 0.0, True, {"action_is_effective": False}

        info: dict[str, Any] = {"action_is_effective": False, "invalid_move": 0}
        self.turns += 1

        # Coerce action to string. Anything else (None, int, list, ...) is
        # treated as unparseable so we never misinterpret raw inputs.
        action_str = action if isinstance(action, str) else None
        parsed = _parse_action(action_str)
        if parsed is None:
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="unparseable")

        r1, c1, v = parsed
        # Convert 1-indexed -> 0-indexed.
        r, c = r1 - 1, c1 - 1

        # Bounds checks. Both must be in-range; values are 0..size (0 = clear).
        if not (0 <= r < self.size and 0 <= c < self.size):
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="out_of_range_cell")
        if not (0 <= v <= self.size):
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="out_of_range_value")

        # Clue cells are immutable — neither writable nor clearable.
        if self.clue_mask[r, c]:
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="modify_clue")

        # Clear a previously-placed cell.
        if v == 0:
            if self.board[r, c] == 0:
                # Clearing an already-empty non-clue cell is a no-op.
                info["invalid_move"] = 1
                return self._end_or_continue(self.invalid_penalty, info, reason="clear_empty")
            self.board[r, c] = 0
            info["action_is_effective"] = True
            # A clear cannot complete the puzzle (board now has an empty cell),
            # so no need to run a solve check here.
            return self._end_or_continue(0.0, info, reason="clear")

        # Overwriting / placing. Temporarily remove the old value so the
        # constraint check sees the cell as empty. Restore on rejection.
        prev = int(self.board[r, c])
        # Reject no-op writes (placing the same value already there). This
        # closes a loophole where the agent could repeatedly "place" the
        # same digit to feign progress without consuming useful reasoning.
        if prev == v:
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="noop_write")
        self.board[r, c] = 0
        if _violates_constraint(self.board, r, c, v):
            self.board[r, c] = prev  # restore
            info["invalid_move"] = 1
            return self._end_or_continue(self.invalid_penalty, info, reason="constraint_violation")

        self.board[r, c] = v
        info["action_is_effective"] = True
        info["overwrote_previous"] = prev != 0

        # Solve check. With unique-solution puzzles the two clauses are
        # equivalent, but we keep both as defense-in-depth: even if uniqueness
        # was disabled, we still award the win for any valid completed grid.
        if _is_full(self.board) and (bool(np.array_equal(self.board, self.solution)) or _is_valid_complete(self.board)):
            self._game_over = True
            self._won = True
            return self.render(), self.target_reward, True, info

        return self._end_or_continue(0.0, info, reason="placed")

    def _end_or_continue(self, reward: float, info: dict, reason: str):
        info["reason"] = reason
        if self.turns >= self.max_turns:
            self._game_over = True
            info["max_turns_reached"] = True
            # Log progress as a metric but do NOT add it to reward.
            # Sparse reward: only a full solve yields positive reward.
            info["progress"] = self._progress_fraction()
            return self.render(), reward, True, info
        return self.render(), reward, False, info

    # ------------------------------------------------------------------ #
    # Metrics (not used for reward — logged in info for diagnostics)
    # ------------------------------------------------------------------ #
    def _progress_fraction(self) -> float:
        """Fraction of blanks correctly filled, in [0, 1]. Exposed via
        ``info["progress"]`` for training dashboards / curriculum gating,
        but NOT included in the reward signal."""
        blanks = ~self.clue_mask
        n_blanks = int(blanks.sum())
        if n_blanks == 0:
            return 1.0
        correct = int(np.sum((self.board == self.solution) & blanks))
        return max(0.0, min(1.0, correct / n_blanks))

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def render(self, mode: str = "ansi") -> str:
        return _render_board(self.board)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_dict(env_info: dict) -> SudokuEnv:
        return SudokuEnv(
            size=env_info.get("size", DEFAULT_SIZE),
            num_clues=env_info.get("num_clues", DEFAULT_NUM_CLUES),
            max_turns=env_info.get("max_turns", DEFAULT_MAX_TURNS),
            seed=env_info.get("seed", 0),
            target_reward=env_info.get("target_reward", DEFAULT_TARGET_REWARD),
            invalid_penalty=env_info.get("invalid_penalty", DEFAULT_INVALID_PENALTY),
            enforce_unique=env_info.get("enforce_unique", DEFAULT_ENFORCE_UNIQUE),
        )
