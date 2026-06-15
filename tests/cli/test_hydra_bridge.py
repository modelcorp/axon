"""Tests for axon.cli._hydra_bridge module."""

import pytest

from axon.cli._hydra_bridge import (
    _flatten,
    _to_hydra_literal,
    build_hydra_overrides,
    flatten_yaml_to_overrides,
)

# ---------------------------------------------------------------------------
# _to_hydra_literal
# ---------------------------------------------------------------------------


class TestToHydraLiteral:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "null"),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (-7, "-7"),
            (0, "0"),
            (3.14, "3.14"),
            (0.0, "0.0"),
            ("hello", "hello"),
            ("", ""),
        ],
        ids=[
            "none",
            "true",
            "false",
            "int",
            "negative_int",
            "zero",
            "float",
            "float_zero",
            "str",
            "empty_str",
        ],
    )
    def test_all_types(self, value, expected):
        assert _to_hydra_literal(value) == expected

    def test_bool_int_ordering(self):
        """bool is a subclass of int; ensure 1 returns '1' not 'True'."""
        assert _to_hydra_literal(1) == "1"
        assert _to_hydra_literal(0) == "0"


# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flat_dict(self):
        result = _flatten("", {"lr": 0.001, "epochs": 10})
        assert "++lr=0.001" in result
        assert "++epochs=10" in result
        assert len(result) == 2

    def test_nested_dict(self):
        result = _flatten("", {"model": {"hidden_size": 256, "layers": 12}})
        assert "++model.hidden_size=256" in result
        assert "++model.layers=12" in result
        assert len(result) == 2

    def test_deeply_nested_dict(self):
        result = _flatten("", {"a": {"b": {"c": "deep"}}})
        assert result == ["++a.b.c=deep"]

    def test_list_values(self):
        result = _flatten("", {"gpus": [0, 1, 2]})
        assert result == ["++gpus=[0, 1, 2]"]

    def test_list_with_none(self):
        result = _flatten("", {"items": [None, True, 3]})
        assert result == ["++items=[null, true, 3]"]

    def test_scalar_at_top_level(self):
        """A non-dict, non-list value with a prefix produces a single override."""
        result = _flatten("learning_rate", 0.01)
        assert result == ["++learning_rate=0.01"]

    def test_empty_dict(self):
        result = _flatten("", {})
        assert result == []

    def test_mixed_nested_and_scalar(self):
        data = {
            "seed": 42,
            "optimizer": {"type": "adam", "lr": 0.001},
        }
        result = _flatten("", data)
        assert "++seed=42" in result
        assert "++optimizer.type=adam" in result
        assert "++optimizer.lr=0.001" in result
        assert len(result) == 3

    def test_prefix_propagation(self):
        """When called with a non-empty prefix, it prepends correctly."""
        result = _flatten("train", {"batch_size": 32})
        assert result == ["++train.batch_size=32"]

    # -- harder edge cases --

    def test_key_with_dot(self):
        """A dict key containing a dot becomes an ambiguous Hydra path."""
        result = _flatten("", {"a.b": 1})
        assert result == ["++a.b=1"]

    def test_integer_key(self):
        """Integer dict keys are stringified."""
        result = _flatten("", {0: "val"})
        assert result == ["++0=val"]

    def test_mixed_list_booleans_none_strings(self):
        """Lists with mixed types are rendered via _to_hydra_literal."""
        result = _flatten("", {"items": [True, None, "foo"]})
        assert result == ["++items=[true, null, foo]"]

    def test_list_with_empty_string(self):
        """An empty string in a list should still appear (as nothing between commas)."""
        result = _flatten("", {"tags": ["", "a"]})
        assert result == ["++tags=[, a]"]

    def test_list_of_dicts(self):
        """A list containing dicts: _to_hydra_literal calls str() on each dict."""
        result = _flatten("", {"items": [{"nested": True}]})
        assert result == ["++items=[{'nested': True}]"]

    def test_empty_list(self):
        """An empty list renders as '[]'."""
        result = _flatten("", {"tags": []})
        assert result == ["++tags=[]"]


# ---------------------------------------------------------------------------
# flatten_yaml_to_overrides
# ---------------------------------------------------------------------------


class TestFlattenYamlToOverrides:
    def test_invalid_yaml_not_a_mapping(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            flatten_yaml_to_overrides(yaml_file)

    def test_invalid_yaml_scalar(self, tmp_path):
        yaml_file = tmp_path / "scalar.yaml"
        yaml_file.write_text("42\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            flatten_yaml_to_overrides(yaml_file)

    def test_nested_yaml_with_list(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("data:\n  files: [a.txt, b.txt]\n  batch_size: 8\n")
        result = flatten_yaml_to_overrides(yaml_file)
        assert "++data.batch_size=8" in result
        assert "++data.files=[a.txt, b.txt]" in result
        assert len(result) == 2

    def test_complex_nested_yaml(self, tmp_path):
        content = "training:\n  optimizer:\n    type: adam\n    lr: 0.0001\n  scheduler:\n    warmup: 100\nseed: 42\n"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(content)
        result = flatten_yaml_to_overrides(yaml_file)
        assert "++training.optimizer.type=adam" in result
        assert "++training.optimizer.lr=0.0001" in result
        assert "++training.scheduler.warmup=100" in result
        assert "++seed=42" in result
        assert len(result) == 4

    # -- harder edge cases --

    def test_empty_yaml_file(self, tmp_path):
        """An empty YAML file yields None from safe_load, which must raise."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            flatten_yaml_to_overrides(yaml_file)

    def test_yaml_document_marker_only(self, tmp_path):
        """A YAML file containing only '---' yields None from safe_load."""
        yaml_file = tmp_path / "marker.yaml"
        yaml_file.write_text("---\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            flatten_yaml_to_overrides(yaml_file)

    def test_special_characters_in_values(self, tmp_path):
        """Values with spaces, colons, and other specials survive the round-trip."""
        yaml_file = tmp_path / "special.yaml"
        yaml_file.write_text('greeting: "hello: world"\npath: "/a b/c d"\n')
        result = flatten_yaml_to_overrides(yaml_file)
        assert "++greeting=hello: world" in result
        assert "++path=/a b/c d" in result
        assert len(result) == 2

    def test_missing_file(self):
        """A nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            flatten_yaml_to_overrides("/no/such/file.yaml")


# ---------------------------------------------------------------------------
# build_hydra_overrides
# ---------------------------------------------------------------------------


class TestBuildHydraOverrides:
    def test_no_kwargs(self):
        result = build_hydra_overrides()
        assert result == []

    def test_some_kwargs(self):
        result = build_hydra_overrides(model="gpt2", gpus=4)
        assert "model_path=gpt2" in result
        assert "num_gpus_per_node=4" in result
        assert len(result) == 2

    def test_all_kwargs(self):
        result = build_hydra_overrides(
            model="llama",
            train_data="train.jsonl",
            val_data="val.jsonl",
            gpus=8,
            nodes=2,
            experiment_name="exp1",
            output_dir="/out",
            resume="/ckpt",
        )
        assert "model_path=llama" in result
        assert "train_files=train.jsonl" in result
        assert "val_files=val.jsonl" in result
        assert "num_gpus_per_node=8" in result
        assert "num_nodes=2" in result
        assert "experiment_name=exp1" in result
        assert "output_dir=/out" in result
        assert "resume_from_checkpoint=/ckpt" in result
        assert len(result) == 8

    def test_unknown_kwargs_ignored(self):
        result = build_hydra_overrides(model="gpt2", unknown_flag="foo", another="bar")
        assert "model_path=gpt2" in result
        assert len(result) == 1

    def test_none_values_ignored(self):
        """Explicitly passing None for a known flag should not produce an override."""
        result = build_hydra_overrides(model=None, gpus=None)
        assert result == []

    # -- harder edge cases --

    def test_falsy_bool_produces_override(self):
        """model=False is falsy but not None; it must appear in the output."""
        result = build_hydra_overrides(model=False)
        assert result == ["model_path=False"]

    def test_zero_produces_override(self):
        """gpus=0 is falsy but not None; it must appear in the output."""
        result = build_hydra_overrides(gpus=0)
        assert result == ["num_gpus_per_node=0"]

    def test_output_order_matches_flag_map(self):
        """Overrides must appear in _FLAG_MAP iteration order, not insertion order."""
        result = build_hydra_overrides(
            resume="/ckpt",
            model="gpt2",
            gpus=4,
        )
        # _FLAG_MAP order: model, train_data, val_data, gpus, ..., resume
        assert result == [
            "model_path=gpt2",
            "num_gpus_per_node=4",
            "resume_from_checkpoint=/ckpt",
        ]

    def test_empty_string_value_produces_override(self):
        """model='' is not None, so it should produce an override with empty value."""
        result = build_hydra_overrides(model="")
        assert result == ["model_path="]
