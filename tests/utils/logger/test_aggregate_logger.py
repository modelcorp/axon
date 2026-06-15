"""Tests for axon.utils.logger.aggregate_logger module."""

import logging
from unittest import mock

import pytest

from axon.utils.logger.aggregate_logger import (
    DecoratorLoggerBase,
    LocalLogger,
    concat_dict_to_str,
    log_with_rank,
    print_rank_0,
    print_with_rank_and_timer,
)


# ---------------------------------------------------------------------------
# concat_dict_to_str
# ---------------------------------------------------------------------------
class TestConcatDictToStr:
    def test_filters_non_numeric_values(self):
        result = concat_dict_to_str({"name": "test", "loss": 0.5, "data": [1, 2]}, step=1)
        assert "loss" in result
        assert "name" not in result
        assert "data" not in result

    def test_empty_dict_only_shows_step(self):
        assert concat_dict_to_str({}, step=42) == "step:42"

    def test_bool_included_as_numeric(self):
        """bool is a subclass of numbers.Number."""
        result = concat_dict_to_str({"flag": True}, step=0)
        assert "flag" in result

    def test_nan_and_inf_included(self):
        result = concat_dict_to_str({"nan": float("nan"), "inf": float("inf")}, step=0)
        assert "nan" in result
        assert "inf" in result

    def test_multiple_values_separated(self):
        result = concat_dict_to_str({"a": 1, "b": 2}, step=0)
        parts = result.split(" - ")
        assert len(parts) == 3  # step + 2 values


# ---------------------------------------------------------------------------
# LocalLogger
# ---------------------------------------------------------------------------
class TestLocalLogger:
    def test_silent_when_disabled(self, capsys):
        LocalLogger(print_to_console=False).log({"loss": 0.5}, step=1)
        assert capsys.readouterr().out == ""

    def test_log_with_title_delegates(self, capsys):
        """log_with_title should call through to print_utils.log_metrics."""
        logger = LocalLogger(print_to_console=True)
        # Should not raise even though it delegates
        logger.log_with_title({"loss": 0.5}, step=1, title="Test")
        out = capsys.readouterr().out
        assert "Test" in out or "loss" in out  # format depends on print_utils


# ---------------------------------------------------------------------------
# DecoratorLoggerBase
# ---------------------------------------------------------------------------
class TestDecoratorLoggerBase:
    def test_none_logger_dispatches_to_print(self, capsys):
        base = DecoratorLoggerBase(role="[X]", logger=None, rank=0)
        assert base.logging_function == base.log_by_print
        base.logging_function("hello")
        assert capsys.readouterr().out == "[X] hello\n"

    def test_provided_logger_dispatches_to_logging(self, caplog):
        py_logger = logging.getLogger("test_deco")
        base = DecoratorLoggerBase(role="[Y]", logger=py_logger, rank=0)
        assert base.logging_function == base.log_by_logging
        with caplog.at_level(logging.DEBUG, logger="test_deco"):
            base.logging_function("msg")
        assert "[Y] msg" in caplog.text

    def test_log_by_logging_raises_if_logger_none(self):
        base = DecoratorLoggerBase(role="[Z]", logger=None, rank=0)
        with pytest.raises(ValueError, match="Logger is not initialized"):
            base.log_by_logging("test")

    @pytest.mark.parametrize(
        "rank,log_only_rank_0,expect_output",
        [
            (0, True, True),
            (1, True, False),
            (5, False, True),
            (-1, True, False),
        ],
    )
    def test_rank_filtering(self, rank, log_only_rank_0, expect_output, capsys):
        base = DecoratorLoggerBase(role="[R]", rank=rank, log_only_rank_0=log_only_rank_0)
        base.logging_function("msg")
        out = capsys.readouterr().out
        assert ("msg" in out) == expect_output


# ---------------------------------------------------------------------------
# print_rank_0 – distributed awareness
# ---------------------------------------------------------------------------
class TestPrintRank0:
    def test_prints_when_not_distributed(self, capsys):
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            print_rank_0("hello")
        assert capsys.readouterr().out == "hello\n"

    def test_prints_on_rank_0(self, capsys):
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            print_rank_0("rank0")
        assert capsys.readouterr().out == "rank0\n"

    def test_suppressed_on_nonzero_rank(self, capsys):
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=3),
        ):
            print_rank_0("hidden")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# print_with_rank_and_timer
# ---------------------------------------------------------------------------
class TestPrintWithRankAndTimer:
    def test_output_format(self, capsys):
        print_with_rank_and_timer("timed", rank=7, log_only_rank_0=False)
        out = capsys.readouterr().out
        assert "[Rank 7]" in out
        assert "timed" in out
        import re

        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", out)

    def test_suppressed_on_nonzero_rank(self, capsys):
        print_with_rank_and_timer("secret", rank=2, log_only_rank_0=True)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# log_with_rank
# ---------------------------------------------------------------------------
class TestLogWithRank:
    def test_custom_level(self, caplog):
        py_logger = logging.getLogger("test_lwr")
        with caplog.at_level(logging.WARNING, logger="test_lwr"):
            log_with_rank("warn!", rank=5, logger=py_logger, level=logging.WARNING, log_only_rank_0=False)
        assert "[Rank 5] warn!" in caplog.text

    def test_rank_0_only_filtering(self, caplog):
        py_logger = logging.getLogger("test_lwr2")
        with caplog.at_level(logging.INFO, logger="test_lwr2"):
            log_with_rank("secret", rank=1, logger=py_logger, log_only_rank_0=True)
        assert "secret" not in caplog.text
