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
#
# 2048 rules ported from OpenPipe ART (github.com/OpenPipe/ART), Apache-2.0.
"""
2048 environment for Axon — functionally identical to openpipe/art's 2048
implementation (examples/2048/utils.py + examples/2048/rollout.py).

Rules ported from ART exactly:
  - WINNING_VALUE defaults to 128 (configurable via target_value).
  - Each turn applies the move (slide + merge), then unconditionally spawns a
    new tile (2 w.p. 0.9, 4 w.p. 0.1) on a random empty cell — spawn happens
    even if the move didn't change the board (this matches ART, and differs
    from "real" 2048).
  - Game ends when max tile >= WINNING_VALUE, or the board is full (ART does
    not check whether merges are still possible when full).
  - An invalid action (integer not in {1,2,3,4}) terminates the episode with
    reward -1 — matches ART's `ValueError -> reward=-1, break`.
  - Terminal reward (ART formula from rollout.py):
        if agent_won (max == WINNING_VALUE):
            reward = 2.0
        else:
            max_value_reward   = (log2(max_tile) - 1) / (log2(WIN) - 1)
            board_value_reward = (log2(board_sum) - 1) / (log2(WIN * 16) - 1)
            reward = max_value_reward + 0.2 * board_value_reward
  - Board render matches ART's pipe-separated, dynamic-width, '_' empty format.

Note: `max_turns` is kept as an optional safety cap (default 10000 ~= no cap).
ART has no turn limit; games self-terminate when the board fills.
"""

import math
import random
import string
from typing import Any

import numpy as np

from axon.core import BaseEnv, register_env

DEFAULT_SIZE: int = 4
DEFAULT_TARGET: int = 128  # ART's WINNING_VALUE
DEFAULT_MAX_TURNS: int = 10_000  # effectively uncapped; ART has no limit


def _spawn_tile(board: np.ndarray, rng: random.Random) -> bool:
    """Port of ART's populate_random_cell: 90% 2, 10% 4, uniform over empties."""
    empties = list(zip(*np.where(board == 0), strict=False))
    if not empties:
        return False
    r, c = empties[rng.randrange(len(empties))]
    board[r, c] = 2 if rng.random() < 0.9 else 4
    return True


def _slide_row_left(row: np.ndarray) -> np.ndarray:
    """Port of ART's condense_sequence: remove gaps, merge greedy from front, pad."""
    nonzero = row[row != 0].tolist()
    merged: list[int] = []
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            merged.append(nonzero[i] * 2)
            i += 2
        else:
            merged.append(nonzero[i])
            i += 1
    merged.extend([0] * (len(row) - len(merged)))
    return np.array(merged, dtype=row.dtype)


def _apply_move(board: np.ndarray, direction: int) -> np.ndarray:
    """Apply a move. direction: 1=Left, 2=Down, 3=Right, 4=Up. Returns new board.

    Matches ART's condense_board exactly: no "did it change?" short-circuit.
    """
    b = board.copy()
    if direction == 1:  # Left
        return np.stack([_slide_row_left(r) for r in b], axis=0)
    if direction == 3:  # Right
        return np.stack([_slide_row_left(r[::-1])[::-1] for r in b], axis=0)
    if direction == 4:  # Up
        cols = [_slide_row_left(b[:, c]) for c in range(b.shape[1])]
        return np.stack(cols, axis=1)
    if direction == 2:  # Down
        cols = [_slide_row_left(b[:, c][::-1])[::-1] for c in range(b.shape[1])]
        return np.stack(cols, axis=1)
    return b


def _is_board_full(board: np.ndarray) -> bool:
    return not np.any(board == 0)


def _max_cell_value(board: np.ndarray) -> int:
    return int(board.max())


def _total_board_value(board: np.ndarray) -> int:
    return int(board.sum())


def _render_board_art(board: np.ndarray) -> str:
    """Render matching ART's render_board: pipe-separated, right-justified,
    '_' for empty, column width = width of the largest non-empty cell."""
    non_empty = [int(v) for v in board.flatten() if v != 0]
    max_cell_width = max((len(str(v)) for v in non_empty), default=1)
    lines = []
    for row in board:
        cells = [(str(int(v)) if v != 0 else "_").rjust(max_cell_width) for v in row]
        lines.append("|".join(cells))
    return "\n".join(lines) + "\n"


@register_env("2048")
class Game2048Env(BaseEnv):
    """
    2048 sliding puzzle — ART-compatible.

    ## Action Space
    Integer action:
    - 0 or any value outside {1,2,3,4}: invalid — episode ends with reward -1
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Rewards (ART formula)
    - Per-step: 0 for every valid move.
    - Invalid action: reward = -1, episode ends immediately.
    - Terminal (board full or win):
        win  -> +2.0
        else -> (log2(max)-1)/(log2(WIN)-1) + 0.2 * (log2(board_sum)-1)/(log2(WIN*16)-1)
    """

    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        3: "Right",
        4: "Up",
    }

    INVALID_ACTION = 0
    # Kept for backwards compatibility with the old env; ART doesn't use a
    # per-step penalty — an invalid action goes straight to reward=-1 and done.
    PENALTY_FOR_INVALID = -1.0

    def __init__(self, **kwargs):
        self.size: int = int(kwargs.pop("size", DEFAULT_SIZE))
        self.target_value: int = int(kwargs.pop("target_value", DEFAULT_TARGET))
        self.max_turns: int = int(kwargs.pop("max_turns", DEFAULT_MAX_TURNS))
        self.seed: int = int(kwargs.pop("seed", 0))

        self._rng = random.Random(self.seed)
        # ART also stores a random string id; harmless to mirror for debugging.
        self.id: str = "".join(self._rng.choices(string.ascii_letters + string.digits, k=6))
        self.board: np.ndarray = np.zeros((self.size, self.size), dtype=np.int64)
        self.turns: int = 0
        self._won: bool = False
        self._game_over: bool = False
        self._invalid_quit: bool = False

        # ART: two initial random cells.
        _spawn_tile(self.board, self._rng)
        _spawn_tile(self.board, self._rng)

        self.env_kwargs = {
            "size": self.size,
            "target_value": self.target_value,
            "max_turns": self.max_turns,
            "seed": self.seed,
        }

    def reset(self):
        self.__init__(**self.env_kwargs)
        return self.render(), {}

    # ART's check_game_finished: win (>=) OR board full.
    def _check_game_finished(self) -> bool:
        if _max_cell_value(self.board) >= self.target_value:
            return True
        return _is_board_full(self.board)

    def finished(self) -> bool:
        return self._game_over or self.turns >= self.max_turns

    def success(self) -> bool:
        # ART's agent_won uses ==, so we do too for identical semantics.
        return self._won and not self._invalid_quit

    # ART's terminal reward formula (rollout.py).
    def _terminal_reward(self) -> float:
        max_value = _max_cell_value(self.board)
        board_value = _total_board_value(self.board)
        agent_won = max_value == self.target_value
        if agent_won:
            return 2.0
        log_win = math.log(self.target_value, 2)
        max_value_reward = (math.log(max(max_value, 2), 2) - 1) / (log_win - 1)
        board_value_reward = (math.log(max(board_value, 2), 2) - 1) / (math.log(self.target_value * 16, 2) - 1)
        return max_value_reward + 0.2 * board_value_reward

    def step(self, action: int):
        if self.finished():
            return self.render(), 0.0, True, {"action_is_effective": False}

        # Normalize and validate action.
        try:
            action_int = int(action) if action is not None else self.INVALID_ACTION
        except (TypeError, ValueError):
            action_int = self.INVALID_ACTION

        info: dict[str, Any] = {"action_is_effective": False}

        if action_int not in (1, 2, 3, 4):
            self._game_over = True
            self._invalid_quit = True
            info["invalid_move"] = 1
            return self.render(), -1.0, True, info

        # Apply move then unconditionally spawn, matching ART's apply_agent_move.
        prev_board = self.board
        new_board = _apply_move(self.board, action_int)
        info["action_is_effective"] = not np.array_equal(prev_board, new_board)
        self.board = new_board
        _spawn_tile(self.board, self._rng)
        self.turns += 1

        # ART checks game_finished after apply_agent_move.
        if self._check_game_finished():
            self._game_over = True
            self._won = _max_cell_value(self.board) >= self.target_value
            info["invalid_move"] = 0
            return self.render(), self._terminal_reward(), True, info

        if self.turns >= self.max_turns:
            # Safety cap (not in ART); still emit ART-style terminal reward.
            self._game_over = True
            info["invalid_move"] = 0
            info["max_turns_reached"] = True
            return self.render(), self._terminal_reward(), True, info

        # Per-step reward in ART is 0 for all non-terminal steps.
        return self.render(), 0.0, False, info

    def render(self, mode: str = "tiny_rgb_array") -> str:
        return _render_board_art(self.board)

    @staticmethod
    def from_dict(env_info: dict) -> "Game2048Env":
        return Game2048Env(
            size=env_info.get("size", DEFAULT_SIZE),
            target_value=env_info.get("target_value", DEFAULT_TARGET),
            max_turns=env_info.get("max_turns", DEFAULT_MAX_TURNS),
            seed=env_info.get("seed", 0),
        )
