"""Tests for axon.utils.dataset_utils module."""

import json

import pytest

from axon.utils.dataset_utils import (
    TestDataset,
    TrainDataset,
    fetch_live_code_bench_system_prompt,
    load_dataset,
)


# ---------------------------------------------------------------------------
# load_dataset – directory construction and error handling
# ---------------------------------------------------------------------------
class TestLoadDataset:
    def test_train_math_resolves_to_train_dir(self, tmp_path, monkeypatch):
        """TrainDataset.Math enums should resolve to math/train/ subdirectory."""
        math_dir = tmp_path / "datasets" / "math" / "train"
        math_dir.mkdir(parents=True)
        data = [{"q": "1+1", "a": "2"}]
        (math_dir / "aime.json").write_text(json.dumps(data))

        import axon

        monkeypatch.setattr(axon, "__file__", str(tmp_path / "axon" / "__init__.py"))
        assert load_dataset(TrainDataset.Math.AIME) == data

    def test_test_code_resolves_to_test_dir(self, tmp_path, monkeypatch):
        """TestDataset.Code enums should resolve to code/test/ subdirectory."""
        code_dir = tmp_path / "datasets" / "code" / "test"
        code_dir.mkdir(parents=True)
        data = [{"task": "sort"}]
        (code_dir / "taco.json").write_text(json.dumps(data))

        import axon

        monkeypatch.setattr(axon, "__file__", str(tmp_path / "axon" / "__init__.py"))
        assert load_dataset(TestDataset.Code.TACO) == data

    def test_invalid_json_raises_value_error(self, tmp_path, monkeypatch):
        math_dir = tmp_path / "datasets" / "math" / "train"
        math_dir.mkdir(parents=True)
        (math_dir / "aime.json").write_text("{broken json")

        import axon

        monkeypatch.setattr(axon, "__file__", str(tmp_path / "axon" / "__init__.py"))
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_dataset(TrainDataset.Math.AIME)

    def test_category_is_derived_from_inner_class_name(self, tmp_path, monkeypatch):
        """Category dir should be 'math' (from Math class) not 'trainmath'."""
        # If this logic were wrong, it would look in wrong dir and try to download
        math_dir = tmp_path / "datasets" / "math" / "train"
        math_dir.mkdir(parents=True)
        (math_dir / "gsm8k.json").write_text("[]")

        import axon

        monkeypatch.setattr(axon, "__file__", str(tmp_path / "axon" / "__init__.py"))
        result = load_dataset(TrainDataset.Math.GSM8K)
        assert result == []

    def test_dataset_name_is_lowercased(self, tmp_path, monkeypatch):
        """Enum value 'AIME' should become filename 'aime.json'."""
        math_dir = tmp_path / "datasets" / "math" / "train"
        math_dir.mkdir(parents=True)
        (math_dir / "aime.json").write_text('[{"x": 1}]')

        import axon

        monkeypatch.setattr(axon, "__file__", str(tmp_path / "axon" / "__init__.py"))
        assert load_dataset(TrainDataset.Math.AIME) == [{"x": 1}]


# ---------------------------------------------------------------------------
# fetch_live_code_bench_system_prompt
# ---------------------------------------------------------------------------
class TestFetchLiveCodeBenchSystemPrompt:
    def test_without_starter_code_has_placeholder(self):
        result = fetch_live_code_bench_system_prompt("Solve this problem")
        assert "YOUR CODE HERE" in result
        assert "Solve this problem" in result
        assert "### Answer:" in result

    def test_with_starter_code_includes_it(self):
        result = fetch_live_code_bench_system_prompt("Complete", starter_code="def foo():\n    pass")
        assert "def foo():" in result
        assert "Complete" in result
        # Should NOT contain the placeholder when starter code is given
        assert "YOUR CODE HERE" not in result

    def test_empty_starter_code_treated_as_falsy(self):
        """Empty string is falsy, should behave like no starter code."""
        result = fetch_live_code_bench_system_prompt("task", starter_code="")
        assert "YOUR CODE HERE" in result

    def test_prompt_is_prepended_to_system_message(self):
        """The user prompt should come after the system message, not replace it."""
        from axon.utils.system_prompts import LCB_SYSTEM_MESSAGE_GENERIC

        result = fetch_live_code_bench_system_prompt("my prompt")
        idx_system = result.index(LCB_SYSTEM_MESSAGE_GENERIC)
        idx_prompt = result.index("my prompt")
        assert idx_system < idx_prompt
