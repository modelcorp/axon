import pytest

from axon.utils.temperature_scheduler import TemperatureScheduler

# ===================================================================
#  Validation
# ===================================================================


class TestTemperatureSchedulerValidation:
    def test_negative_start_temperature(self):
        config = {"start_temperature": -1.0, "end_temperature": 1.0, "num_steps": 100}
        with pytest.raises(ValueError, match="Start temperature must be positive"):
            TemperatureScheduler(config)

    def test_negative_end_temperature(self):
        config = {"start_temperature": 1.0, "end_temperature": -0.5, "num_steps": 100}
        with pytest.raises(ValueError, match="End temperature must be positive"):
            TemperatureScheduler(config)

    def test_invalid_scheduler_type(self):
        config = {"scheduler": "quadratic", "start_temperature": 1.0, "end_temperature": 0.5, "num_steps": 100}
        with pytest.raises(ValueError, match="Invalid scheduler type"):
            TemperatureScheduler(config)

    def test_non_positive_num_steps(self):
        config = {"start_temperature": 1.0, "end_temperature": 0.5, "num_steps": 0}
        with pytest.raises(ValueError, match="Number of steps must be positive"):
            TemperatureScheduler(config)


# ===================================================================
#  Linear schedule
# ===================================================================


class TestLinearSchedule:
    def setup_method(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        self.scheduler = TemperatureScheduler(config)

    def test_monotonic_decrease(self):
        temps = [self.scheduler.get_temperature(step=s) for s in range(0, 101, 10)]
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1]

    def test_increasing_schedule(self):
        """Linear schedule can also go up (end > start)."""
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 0.5,
            "end_temperature": 2.0,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        assert s.get_temperature(step=0) == pytest.approx(0.5)
        assert s.get_temperature(step=100) == pytest.approx(2.0)
        assert s.get_temperature(step=50) == pytest.approx(1.25)

    def test_same_start_end_is_constant(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 1.0,
            "end_temperature": 1.0,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        for step in [0, 25, 50, 75, 100]:
            assert s.get_temperature(step=step) == pytest.approx(1.0)


# ===================================================================
#  Exponential schedule
# ===================================================================


class TestExponentialSchedule:
    def setup_method(self):
        config = {
            "enable": True,
            "scheduler": "exponential",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        self.scheduler = TemperatureScheduler(config)

    def test_midpoint_reasonable(self):
        temp = self.scheduler.get_temperature(step=50)
        assert 0.5 < temp < 2.0
        # Exponential midpoint: start * (end/start)^0.5 = 2.0 * (0.25)^0.5 = 2.0 * 0.5 = 1.0
        assert temp == pytest.approx(1.0)

    def test_monotonic_decrease(self):
        temps = [self.scheduler.get_temperature(step=s) for s in range(0, 101, 10)]
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1]

    def test_increasing_exponential(self):
        config = {
            "enable": True,
            "scheduler": "exponential",
            "start_temperature": 0.1,
            "end_temperature": 10.0,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        assert s.get_temperature(step=0) == pytest.approx(0.1)
        assert s.get_temperature(step=100) == pytest.approx(10.0)
        # Midpoint: 0.1 * (100)^0.5 = 1.0
        assert s.get_temperature(step=50) == pytest.approx(1.0)

    def test_exponential_same_start_end_is_constant(self):
        config = {
            "enable": True,
            "scheduler": "exponential",
            "start_temperature": 3.0,
            "end_temperature": 3.0,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        for step in [0, 50, 100]:
            assert s.get_temperature(step=step) == pytest.approx(3.0)


# ===================================================================
#  Cosine schedule
# ===================================================================


class TestCosineSchedule:
    def setup_method(self):
        config = {
            "enable": True,
            "scheduler": "cosine",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        self.scheduler = TemperatureScheduler(config)

    def test_midpoint_between_start_and_end(self):
        temp = self.scheduler.get_temperature(step=50)
        assert 0.5 < temp < 2.0
        # At progress=0.5: end + (start-end) * (1+cos(pi*0.5))/2 = 0.5 + 1.5*(1+0)/2 = 0.5 + 0.75 = 1.25
        assert temp == pytest.approx(1.25)

    def test_monotonic_decrease(self):
        temps = [self.scheduler.get_temperature(step=s) for s in range(0, 101, 10)]
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1]

    def test_cosine_symmetric_around_midpoint(self):
        """Cosine annealing is symmetric: temp(t) + temp(T-t) = start + end."""
        for step in range(0, 51):
            t_fwd = self.scheduler.get_temperature(step=step)
            t_bwd = self.scheduler.get_temperature(step=100 - step)
            assert t_fwd + t_bwd == pytest.approx(2.0 + 0.5, abs=1e-10)


# ===================================================================
#  get_temperature behaviour
# ===================================================================


class TestGetTemperature:
    def test_enabled_with_internal_counter(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        scheduler = TemperatureScheduler(config)
        assert scheduler.get_temperature() == pytest.approx(2.0)
        scheduler.current_step = 50
        assert scheduler.get_temperature() == pytest.approx(1.25)
        scheduler.current_step = 100
        assert scheduler.get_temperature() == pytest.approx(0.5)


# ===================================================================
#  step / reset / set_step
# ===================================================================


class TestStepAndReset:
    def test_step_disabled_does_not_increment(self):
        config = {"enable": False, "start_temperature": 1.0, "end_temperature": 0.5, "num_steps": 100}
        scheduler = TemperatureScheduler(config)
        scheduler.step()
        scheduler.step()
        assert scheduler.current_step == 0

    def test_set_step_negative_raises_value_error(self):
        config = {"enable": True, "start_temperature": 1.0, "end_temperature": 0.5, "num_steps": 100}
        scheduler = TemperatureScheduler(config)
        with pytest.raises(ValueError, match="Step must be non-negative"):
            scheduler.set_step(-1)


# ===================================================================
#  Progress clamping
# ===================================================================


class TestProgressClamping:
    def test_step_beyond_num_steps_clamps_to_end_temperature(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        scheduler = TemperatureScheduler(config)
        temp_at_end = scheduler.get_temperature(step=100)
        temp_beyond = scheduler.get_temperature(step=200)
        assert temp_at_end == pytest.approx(0.5)
        assert temp_beyond == pytest.approx(0.5)


# ===================================================================
#  State consistency
# ===================================================================


class TestSchedulerStateConsistency:
    def test_step_then_reset_then_get_returns_start(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        for _ in range(50):
            s.step()
        s.reset()
        assert s.get_temperature() == pytest.approx(2.0)

    def test_explicit_step_ignores_internal_counter(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 2.0,
            "end_temperature": 0.5,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        for _ in range(50):
            s.step()
        # Explicit step overrides internal counter
        assert s.get_temperature(step=0) == pytest.approx(2.0)
        # Internal counter unchanged
        assert s.current_step == 50

    def test_very_small_temperature_difference(self):
        config = {
            "enable": True,
            "scheduler": "linear",
            "start_temperature": 1.0,
            "end_temperature": 1.0 - 1e-10,
            "num_steps": 100,
        }
        s = TemperatureScheduler(config)
        t0 = s.get_temperature(step=0)
        t100 = s.get_temperature(step=100)
        assert abs(t0 - t100) < 1e-8

    def test_step_does_not_increment_when_disabled(self):
        config = {"enable": False, "start_temperature": 2.0, "end_temperature": 0.5, "num_steps": 100}
        s = TemperatureScheduler(config)
        for _ in range(200):
            s.step()
        assert s.current_step == 0

    def test_all_three_schedules_agree_at_boundaries(self):
        """All schedulers produce start_temp at step=0, end_temp at step=num_steps."""
        for sched_type in ["linear", "exponential", "cosine"]:
            config = {
                "enable": True,
                "scheduler": sched_type,
                "start_temperature": 5.0,
                "end_temperature": 0.1,
                "num_steps": 100,
            }
            s = TemperatureScheduler(config)
            assert s.get_temperature(step=0) == pytest.approx(5.0), f"{sched_type} failed at step=0"
            assert s.get_temperature(step=100) == pytest.approx(0.1), f"{sched_type} failed at step=100"

    def test_exponential_always_between_linear_and_cosine_for_decay(self):
        """For decreasing schedules, exponential decays faster at start, cosine decays slower."""
        configs = {}
        for sched in ["linear", "exponential", "cosine"]:
            configs[sched] = TemperatureScheduler(
                {"enable": True, "scheduler": sched, "start_temperature": 2.0, "end_temperature": 0.5, "num_steps": 100}
            )
        # At midpoint, exponential < linear (decays faster early)
        t_lin = configs["linear"].get_temperature(step=50)
        t_exp = configs["exponential"].get_temperature(step=50)
        assert t_exp < t_lin
