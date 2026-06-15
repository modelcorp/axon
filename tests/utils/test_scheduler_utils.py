import pytest
import torch

from axon.utils.scheduler_utils import (
    build_lr_scheduler,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def _make_optimizer(lr=0.1):
    param = torch.nn.Parameter(torch.randn(2, 2))
    return torch.optim.SGD([param], lr=lr)


# ===================================================================
#  Cosine schedule with warmup
# ===================================================================


class TestCosineScheduleWithWarmup:
    def test_min_lr_ratio_floor(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=100, min_lr_ratio=0.1
        )

        # Advance to end of training
        for _ in range(100):
            scheduler.step()

        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(0.1, abs=1e-6)

    def test_min_lr_ratio_never_goes_below(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50, min_lr_ratio=0.2
        )

        for _ in range(100):
            scheduler.step()
            lr_ratio = scheduler.get_last_lr()[0] / 0.1
            assert lr_ratio >= 0.2 - 1e-6

    def test_init_lr_ratio_starting_point(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=100, init_lr_ratio=0.3
        )

        # At step 0, lr_lambda = init_lr_ratio + (1 - init_lr_ratio) * 0/10 = 0.3
        lr_ratio_0 = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio_0 == pytest.approx(0.3, abs=1e-6)

        # At step 5, lr_lambda = 0.3 + 0.7 * 5/10 = 0.65
        for _ in range(5):
            scheduler.step()
        lr_ratio_5 = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio_5 == pytest.approx(0.65, abs=1e-6)

    def test_full_schedule_values(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=100, min_lr_ratio=0.0
        )

        # At step 0 (no warmup), lr should be at peak
        lr_at_start = scheduler.get_last_lr()[0] / 0.1
        assert lr_at_start == pytest.approx(1.0, abs=1e-6)

        # Advance to end
        for _ in range(100):
            scheduler.step()
        lr_at_end = scheduler.get_last_lr()[0] / 0.1
        assert lr_at_end == pytest.approx(0.0, abs=1e-6)

    def test_warmup_equals_total_steps(self):
        """Edge case: warmup == total steps means entire schedule is warmup."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=100)
        for _ in range(100):
            scheduler.step()
        # At step 100, warmup complete, cosine has 0 steps of decay
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(1.0, abs=1e-6)

    def test_beyond_total_steps_lr_stays_at_min(self):
        """Steps beyond total_training_steps should floor at min_lr_ratio."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50, min_lr_ratio=0.1
        )
        for _ in range(200):
            scheduler.step()
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio >= 0.1 - 1e-6

    def test_num_cycles_two_produces_two_waves(self):
        """With num_cycles=2, lr oscillates twice over the training period."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=100, min_lr_ratio=0.0, num_cycles=2.0
        )
        # At 1/4 of training (25 steps), first wave should be at its trough (close to 0)
        for _ in range(25):
            scheduler.step()
        lr_at_quarter = scheduler.get_last_lr()[0] / 0.1
        assert lr_at_quarter < 0.1  # near trough of first wave
        # At 1/2 of training (50 steps), back at peak
        for _ in range(25):
            scheduler.step()
        lr_at_half = scheduler.get_last_lr()[0] / 0.1
        assert lr_at_half > 0.9  # near peak of second wave

    def test_init_lr_ratio_out_of_range_raises(self):
        optimizer = _make_optimizer(lr=0.1)
        with pytest.raises(AssertionError):
            get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100, init_lr_ratio=1.5)

    def test_min_lr_ratio_out_of_range_raises(self):
        optimizer = _make_optimizer(lr=0.1)
        with pytest.raises(AssertionError):
            get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=100, min_lr_ratio=-0.1)


# ===================================================================
#  Constant schedule with warmup
# ===================================================================


class TestConstantScheduleWithWarmup:
    def test_linear_warmup(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)

        # At step 0, ratio = 0/10 = 0.0
        lr_ratio_0 = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio_0 == pytest.approx(0.0, abs=1e-6)

        for step in range(1, 10):
            scheduler.step()
            lr_ratio = scheduler.get_last_lr()[0] / 0.1
            expected = float(step) / 10.0
            assert lr_ratio == pytest.approx(expected, abs=1e-6)

    def test_constant_after_warmup(self):
        optimizer = _make_optimizer(lr=0.1)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=5)

        # Advance past warmup
        for _ in range(5):
            scheduler.step()

        # Should be at full lr
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(1.0, abs=1e-6)

        # Stay constant for many more steps
        for _ in range(50):
            scheduler.step()
            lr_ratio = scheduler.get_last_lr()[0] / 0.1
            assert lr_ratio == pytest.approx(1.0, abs=1e-6)


# ===================================================================
#  build_lr_scheduler
# ===================================================================


class TestBuildLrScheduler:
    def test_cosine_type_with_warmup_ratio(self):
        optimizer = _make_optimizer(lr=0.1)
        lr_scheduler_args = {
            "total_training_steps": 100,
            "lr_warmup_steps_ratio": 0.1,
            "min_lr_ratio": 0.0,
        }
        scheduler = build_lr_scheduler(optimizer, "cosine", lr_scheduler_args, lr=0.1, rank=0)

        # Warmup should be 10 steps (0.1 * 100)
        for _ in range(10):
            scheduler.step()
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(1.0, abs=1e-6)

    def test_cosine_with_min_lr(self):
        optimizer = _make_optimizer(lr=0.1)
        lr_scheduler_args = {
            "total_training_steps": 100,
            "lr_warmup_steps": 0,
            "min_lr": 0.02,
        }
        scheduler = build_lr_scheduler(optimizer, "cosine", lr_scheduler_args, lr=0.1, rank=0)

        # min_lr_ratio = max(0.0, 0.02/0.1) = 0.2
        for _ in range(100):
            scheduler.step()
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(0.2, abs=1e-6)

    def test_warmup_ratio_and_steps_ratio_priority(self):
        """lr_warmup_steps_ratio is used when lr_warmup_steps is not provided."""
        optimizer = _make_optimizer(lr=0.1)
        lr_scheduler_args = {
            "total_training_steps": 200,
            "lr_warmup_steps_ratio": 0.05,  # = 10 steps
            "min_lr_ratio": 0.0,
        }
        scheduler = build_lr_scheduler(optimizer, "cosine", lr_scheduler_args, lr=0.1, rank=0)
        for _ in range(10):
            scheduler.step()
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        assert lr_ratio == pytest.approx(1.0, abs=1e-6)

    def test_min_lr_overrides_min_lr_ratio(self):
        """min_lr (absolute) takes precedence over min_lr_ratio when both could apply."""
        optimizer = _make_optimizer(lr=0.1)
        lr_scheduler_args = {
            "total_training_steps": 100,
            "lr_warmup_steps": 0,
            "min_lr_ratio": 0.5,  # would give floor of 0.05
            "min_lr": 0.01,  # gives floor of 0.1 (max(0.5, 0.01/0.1) = max(0.5, 0.1) = 0.5)
        }
        scheduler = build_lr_scheduler(optimizer, "cosine", lr_scheduler_args, lr=0.1, rank=0)
        for _ in range(100):
            scheduler.step()
        lr_ratio = scheduler.get_last_lr()[0] / 0.1
        # min_lr_ratio = max(0.5, 0.01/0.1) = max(0.5, 0.1) = 0.5
        assert lr_ratio == pytest.approx(0.5, abs=1e-6)

    def test_unsupported_type_raises(self):
        optimizer = _make_optimizer(lr=0.1)
        lr_scheduler_args = {"total_training_steps": 100, "lr_warmup_steps": 5}
        with pytest.raises(NotImplementedError, match="not supported"):
            build_lr_scheduler(optimizer, "polynomial", lr_scheduler_args, lr=0.1, rank=0)
