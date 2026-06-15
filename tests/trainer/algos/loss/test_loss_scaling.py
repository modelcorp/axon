"""Unit test for loss aggregation gradient correctness.

Verifies that agg_loss with different normalization schemes produces
the correct gradient when simulating D data-parallel workers.

Tests ALL combinations of:
- token_reduce: mean, sum, mean-norm, mean-program
- batch_reduce: token-mean, step-mean, program-mean
- DP sync: AVG (FSDP2) vs SUM (Megatron)
- Counts: global vs local
- Loss scaling: ×1, ×D, ×K, ×K×D

Usage:
    python tests/trainer/algos/loss/test_loss_scaling.py        # summary table
    pytest tests/trainer/algos/loss/test_loss_scaling.py -v     # unit tests
"""

import numpy as np
import torch
import torch.nn as nn

from axon.trainer.algos.loss.utils import agg_loss

# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------


def _make_batch(B=8, T=16, n_programs=4, seed=42):
    """Create a toy batch simulating multi-step programs.

    Returns loss_mat, mask, and metadata (num_program_steps, program_uids).
    Programs: 4 programs × 2 steps each = 8 rows.
    """
    g = torch.Generator().manual_seed(seed)
    loss_mat = torch.randn(B, T, generator=g)

    # Variable valid lengths per row
    valid_lens = [10, 8, 12, 6, 10, 8, 12, 6][:B]
    mask = torch.zeros(B, T)
    for i, vl in enumerate(valid_lens):
        mask[i, :vl] = 1.0

    # Program structure: n_programs programs, each with B/n_programs steps
    steps_per_prog = B // n_programs
    num_program_steps = torch.full((B,), steps_per_prog, dtype=torch.long)
    program_uids = np.array([f"prog_{i // steps_per_prog}" for i in range(B)], dtype=object)

    # Total tokens per program (for mean-program token_reduce)
    num_program_tokens = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        prog_id = i // steps_per_prog
        prog_start = prog_id * steps_per_prog
        prog_end = prog_start + steps_per_prog
        total_tokens = sum(valid_lens[j] for j in range(prog_start, prog_end))
        num_program_tokens[i] = total_tokens

    return loss_mat, mask, num_program_steps, num_program_tokens, program_uids


def _valid_counts(mask, program_uids=None):
    """Compute valid_batch_size, valid_token_count, valid_program_count."""
    n = mask.shape[0]
    valid_mask = mask.sum(dim=-1) > 0
    counts = {
        "valid_batch_size": torch.full((n,), int(valid_mask.sum().item()), dtype=torch.long),
        "valid_token_count": torch.full((n,), int(mask.sum().item()), dtype=torch.long),
    }
    if program_uids is not None:
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
        prog_count = len({program_uids[i] for i in valid_idx})
        counts["valid_program_count"] = torch.full((n,), prog_count, dtype=torch.long)
    return counts


# ---------------------------------------------------------------------------
# All reduce mode combinations
# ---------------------------------------------------------------------------

# All token_reduce × batch_reduce combinations
ALL_MODES = []
for tr in ["mean", "sum", "mean-norm", "mean-program"]:
    for br in ["token-mean", "step-mean", "program-mean"]:
        ALL_MODES.append({"token_reduce": tr, "batch_reduce": br})


def _mode_name(kw):
    tr = kw.get("token_reduce", "?")
    br = kw.get("batch_reduce", "")
    return f"{tr}" + (f"+{br}" if br else "")


# ---------------------------------------------------------------------------
# Gradient simulation
# ---------------------------------------------------------------------------


def _compute_grad(
    loss_mat,
    mask,
    D,
    sync_mode,
    dp_loss_scale,
    use_local_counts,
    num_program_steps,
    num_program_tokens,
    program_uids,
    **agg_kwargs,
):
    """Compute gradient simulating D workers with given sync mode.

    Args:
        D: number of DP workers (1 = single GPU reference)
        sync_mode: "avg" or "sum"
        dp_loss_scale: multiply loss by this before backward
        use_local_counts: recompute counts per shard if True
    """
    B = loss_mat.shape[0]
    assert B % D == 0
    shard = B // D

    global_counts = _valid_counts(mask, program_uids)

    total_grad = None
    for d in range(D):
        s = slice(d * shard, (d + 1) * shard)
        sl, sm = loss_mat[s], mask[s]
        s_nps = num_program_steps[s]
        s_npt = num_program_tokens[s]
        s_uids = program_uids[s] if program_uids is not None else None

        if use_local_counts:
            counts = _valid_counts(sm, s_uids)
        else:
            counts = {k: v[s] for k, v in global_counts.items()}

        # Build agg_loss kwargs
        kw = dict(agg_kwargs)
        kw.update(counts)

        # Add program metadata if needed
        tr = kw.get("token_reduce", "sum")
        br = kw.get("batch_reduce", "token-mean")
        if tr == "mean-program":
            kw["num_program_tokens"] = s_npt
        if br.startswith("program-"):
            kw["num_program_steps"] = s_nps

        # Compute grad through a scalar parameter
        w = nn.Linear(1, 1, bias=False)
        w.weight.data.fill_(1.0)
        w.zero_grad()
        loss = agg_loss(loss_mat=sl * w.weight, loss_mask=sm, **kw)
        loss = loss * dp_loss_scale
        loss.backward()

        if total_grad is None:
            total_grad = w.weight.grad.clone()
        else:
            total_grad += w.weight.grad

    if sync_mode == "avg":
        total_grad /= D
    return total_grad


# ---------------------------------------------------------------------------
# Main test: print comprehensive table
# ---------------------------------------------------------------------------


def test_all_combinations():
    """Test all (scaling × sync × counts × reduce_mode) and print summary."""
    loss_mat, mask, nps, npt, uids = _make_batch()
    D = 2
    K = 1  # no micro-batching

    configs = [
        # (name, sync_mode, dp_loss_scale, use_local_counts)
        ("ctrl+FSDP  (global,×D,avg)", "avg", D, False),
        ("ctrl+Mega  (global,×KD,sum)", "sum", K * D, False),
        ("ctrl+Mega  (global,×K,sum)", "sum", K, False),
        ("work+FSDP  (local,×D,avg)", "avg", D, True),
        ("work+FSDP  (local,×1,avg)", "avg", 1, True),
        ("work+Mega  (local,×KD,sum)", "sum", K * D, True),
        ("work+Mega  (local,×1,sum)", "sum", 1, True),
    ]

    print()
    header = f"{'Config':<38} {'Reduce Mode':<25} {'Ratio':>7} {'OK':>4}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for name, sync, scale, local in configs:
        for mode_kw in ALL_MODES:
            mn = _mode_name(mode_kw)
            try:
                ref = _compute_grad(loss_mat, mask, 1, "avg", 1, False, nps, npt, uids, **mode_kw)
                dist = _compute_grad(loss_mat, mask, D, sync, scale, local, nps, npt, uids, **mode_kw)

                if abs(ref.item()) < 1e-10:
                    ratio_str = "  n/a"
                    ok = "  -"
                else:
                    ratio = (dist / ref).item()
                    ratio_str = f"{ratio:7.2f}"
                    ok = " ✅" if abs(ratio - 1.0) < 0.01 else f" ×{ratio:.1f}"
            except Exception as e:
                ratio_str = "  err"
                ok = f" {str(e)[:20]}"

            print(f"  {name:<36} {mn:<25} {ratio_str} {ok}")
        print("-" * len(header))

    print()


# ---------------------------------------------------------------------------
# Pytest unit tests for the two correct configurations
# ---------------------------------------------------------------------------


class TestCorrectConfigs:
    """Verify the two known-correct configurations match single-GPU for all modes."""

    def _ref(self, **kw):
        lm, m, nps, npt, u = _make_batch()
        return _compute_grad(lm, m, 1, "avg", 1, False, nps, npt, u, **kw)

    def _check(self, D, sync, scale, local, **kw):
        lm, m, nps, npt, u = _make_batch()
        ref = _compute_grad(lm, m, 1, "avg", 1, False, nps, npt, u, **kw)
        dist = _compute_grad(lm, m, D, sync, scale, local, nps, npt, u, **kw)
        if abs(ref.item()) < 1e-10:
            return  # skip degenerate
        ratio = (dist / ref).item()
        assert abs(ratio - 1.0) < 0.01, f"ratio={ratio:.4f} for {_mode_name(kw)}"

    # Controller + FSDP (global counts, ×D, AVG) — current correct path
    def test_ctrl_fsdp_sum_token_mean(self):
        self._check(2, "avg", 2, False, token_reduce="sum", batch_reduce="token-mean")

    def test_ctrl_fsdp_mean_step_mean(self):
        self._check(2, "avg", 2, False, token_reduce="mean", batch_reduce="step-mean")

    def test_ctrl_fsdp_mean_program_mean(self):
        self._check(2, "avg", 2, False, token_reduce="mean", batch_reduce="program-mean")

    def test_ctrl_fsdp_sum_step_mean(self):
        self._check(2, "avg", 2, False, token_reduce="sum", batch_reduce="step-mean")

    # Controller + Megatron (global counts, ×K only, SUM)
    def test_ctrl_mega_sum_token_mean(self):
        self._check(2, "sum", 1, False, token_reduce="sum", batch_reduce="token-mean")

    def test_ctrl_mega_mean_step_mean(self):
        self._check(2, "sum", 1, False, token_reduce="mean", batch_reduce="step-mean")

    def test_ctrl_mega_mean_program_mean(self):
        self._check(2, "sum", 1, False, token_reduce="mean", batch_reduce="program-mean")

    # Worker + FSDP (local counts, ×1, AVG)
    def test_work_fsdp_sum_token_mean(self):
        self._check(2, "avg", 1, True, token_reduce="sum", batch_reduce="token-mean")

    def test_work_fsdp_mean_step_mean(self):
        self._check(2, "avg", 1, True, token_reduce="mean", batch_reduce="step-mean")

    def test_work_fsdp_mean_program_mean(self):
        self._check(2, "avg", 1, True, token_reduce="mean", batch_reduce="program-mean")


if __name__ == "__main__":
    test_all_combinations()
