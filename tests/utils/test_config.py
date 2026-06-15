"""Tests for axon.utils.config module."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from omegaconf import OmegaConf

from axon.utils.config import (
    _create_dataclass_from_dict,
    omega_conf_to_dataclass,
    update_dict_with_config,
)

# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SimpleConfig:
    name: str = "default"
    value: int = 0
    flag: bool = False


@dataclass
class InnerConfig:
    x: int = 1
    y: int = 2


@dataclass
class NestedConfig:
    inner: InnerConfig = field(default_factory=InnerConfig)
    label: str = "nested"


@dataclass
class RequiredFieldConfig:
    """Dataclass with required fields (no defaults)."""

    name: str
    count: int


@dataclass
class Level3:
    val: int = 0


@dataclass
class Level2:
    child: Level3 = field(default_factory=Level3)
    tag: str = "l2"


@dataclass
class Level1:
    child: Level2 = field(default_factory=Level2)
    tag: str = "l1"


@dataclass
class UnstructurableConfig:
    """A dataclass that OmegaConf.structured() cannot handle because of the
    Any-typed field with a mutable default_factory (a dict).  OmegaConf
    raises on this, which lets us exercise the except-fallback path in
    omega_conf_to_dataclass."""

    data: Any = field(default_factory=dict)
    name: str = "fallback"


class NotADataclass:
    pass


# ===========================================================================
# omega_conf_to_dataclass
# ===========================================================================


class TestOmegaConfToDataclass:
    """Tests for omega_conf_to_dataclass."""

    def test_simple_dataclass_from_dictconfig(self):
        """Pass a DictConfig and get back the right values."""
        cfg = OmegaConf.create({"name": "test", "value": 42, "flag": True})
        result = omega_conf_to_dataclass(cfg, dataclass_type=SimpleConfig)
        assert isinstance(result, SimpleConfig)
        assert result.name == "test"
        assert result.value == 42
        assert result.flag is True

    def test_non_dataclass_type_raises_value_error(self):
        """Passing a non-dataclass type should raise ValueError."""
        cfg = OmegaConf.create({"name": "test"})
        with pytest.raises(ValueError, match="must be a dataclass"):
            omega_conf_to_dataclass(cfg, dataclass_type=NotADataclass)

    def test_returns_raw_object_if_config_not_dict(self):
        """If config is not a dict/DictConfig (e.g., an int), return it as-is."""
        assert omega_conf_to_dataclass(42) == 42
        assert omega_conf_to_dataclass("hello") == "hello"
        assert omega_conf_to_dataclass(3.14) == 3.14

    def test_partial_config_uses_defaults(self):
        """Fields not in the config should use the dataclass defaults."""
        cfg = OmegaConf.create({"name": "partial"})
        result = omega_conf_to_dataclass(cfg, dataclass_type=SimpleConfig)
        assert isinstance(result, SimpleConfig)
        assert result.name == "partial"
        assert result.value == 0
        assert result.flag is False

    def test_target_key_is_stripped(self):
        """The _target_ key should be removed before creating the dataclass."""
        cfg = OmegaConf.create({"_target_": "some.module.SimpleConfig", "name": "stripped", "value": 5})
        result = omega_conf_to_dataclass(cfg, dataclass_type=SimpleConfig)
        assert isinstance(result, SimpleConfig)
        assert result.name == "stripped"
        assert result.value == 5

    # --- New edge-case tests ---

    def test_variable_interpolation_resolves(self):
        """OmegaConf ${...} interpolations should be resolved before conversion.

        The DictConfig is resolved via OmegaConf.to_container(resolve=True)
        inside omega_conf_to_dataclass, so interpolated values must appear
        correctly in the resulting dataclass.
        """
        # All interpolated keys reference fields that exist on SimpleConfig so
        # the merge step succeeds without hitting extra-key struct errors.
        cfg = OmegaConf.create({"name": "${value}_suffix", "value": 10, "flag": True})
        result = omega_conf_to_dataclass(cfg, dataclass_type=SimpleConfig)
        assert isinstance(result, SimpleConfig)
        assert result.name == "10_suffix"
        assert result.value == 10

    def test_extra_keys_raise_on_merge(self):
        """Extra keys in the config that are absent from the dataclass cause
        OmegaConf.merge to fail because the structured config from the dataclass
        has struct mode enabled, which rejects unknown keys."""
        from omegaconf.errors import ConfigKeyError

        cfg = OmegaConf.create({"name": "extra", "value": 1, "flag": False, "unknown_field": 999})
        with pytest.raises(ConfigKeyError):
            omega_conf_to_dataclass(cfg, dataclass_type=SimpleConfig)

    def test_target_with_no_dataclass_type_uses_hydra_instantiate(self):
        """When dataclass_type is None and _target_ is present, hydra.instantiate
        should be used to create the object."""
        cfg = OmegaConf.create(
            {
                "_target_": "tests.utils.test_config.SimpleConfig",
                "name": "hydra",
                "value": 77,
                "flag": True,
            }
        )
        result = omega_conf_to_dataclass(cfg, dataclass_type=None)
        assert isinstance(result, SimpleConfig)
        assert result.name == "hydra"
        assert result.value == 77
        assert result.flag is True

    def test_structured_config_fallback_path(self):
        """When OmegaConf.structured() raises (e.g. unsupported type annotation),
        omega_conf_to_dataclass should fall back to _create_dataclass_from_dict."""
        cfg = OmegaConf.create({"data": {"key": "value"}, "name": "fell_back"})
        result = omega_conf_to_dataclass(cfg, dataclass_type=UnstructurableConfig)
        assert isinstance(result, UnstructurableConfig)
        assert result.name == "fell_back"
        assert result.data == {"key": "value"}

    def test_none_config_returns_default_instance(self):
        """None is falsy, exercises the `if not config: return dataclass_type()` branch."""
        result = omega_conf_to_dataclass(None, dataclass_type=SimpleConfig)
        assert isinstance(result, SimpleConfig)
        assert result.name == "default"
        assert result.value == 0
        assert result.flag is False

    def test_listconfig_without_target_raises(self):
        """Passing a ListConfig without _target_ and no dataclass_type should raise."""
        cfg = OmegaConf.create(["a", "b"])
        with pytest.raises(AssertionError, match="_target_"):
            omega_conf_to_dataclass(cfg, dataclass_type=None)


# ===========================================================================
# _create_dataclass_from_dict
# ===========================================================================


class TestCreateDataclassFromDict:
    """Tests for _create_dataclass_from_dict."""

    def test_flat_dataclass(self):
        """Create a flat dataclass from a dict."""
        data = {"name": "flat", "value": 10, "flag": True}
        result = _create_dataclass_from_dict(SimpleConfig, data)
        assert isinstance(result, SimpleConfig)
        assert result.name == "flat"
        assert result.value == 10
        assert result.flag is True

    def test_nested_dataclass(self):
        """Nested dicts should be recursively converted to nested dataclasses."""
        data = {"inner": {"x": 10, "y": 20}, "label": "deep"}
        result = _create_dataclass_from_dict(NestedConfig, data)
        assert isinstance(result, NestedConfig)
        assert isinstance(result.inner, InnerConfig)
        assert result.inner.x == 10
        assert result.inner.y == 20
        assert result.label == "deep"

    def test_missing_fields_use_defaults(self):
        """Fields missing from the dict should fall back to dataclass defaults."""
        data = {"name": "only_name"}
        result = _create_dataclass_from_dict(SimpleConfig, data)
        assert isinstance(result, SimpleConfig)
        assert result.name == "only_name"
        assert result.value == 0
        assert result.flag is False

    def test_extra_keys_are_ignored(self):
        """Keys in the dict that are not fields on the dataclass are ignored."""
        data = {"name": "valid", "value": 1, "extra_key": "should_be_ignored"}
        result = _create_dataclass_from_dict(SimpleConfig, data)
        assert isinstance(result, SimpleConfig)
        assert result.name == "valid"
        assert result.value == 1
        assert not hasattr(result, "extra_key")

    # --- New edge-case tests ---

    def test_required_fields_missing_raises_type_error(self):
        """If the dataclass has required fields and data omits them, a TypeError
        should be raised (Python enforces positional args)."""
        with pytest.raises(TypeError):
            _create_dataclass_from_dict(RequiredFieldConfig, {"name": "only_name"})

    def test_deeply_nested_three_levels(self):
        """Three levels of nested dataclasses should all be recursively created."""
        data = {
            "child": {
                "child": {"val": 42},
                "tag": "inner_l2",
            },
            "tag": "outer_l1",
        }
        result = _create_dataclass_from_dict(Level1, data)
        assert isinstance(result, Level1)
        assert result.tag == "outer_l1"
        assert isinstance(result.child, Level2)
        assert result.child.tag == "inner_l2"
        assert isinstance(result.child.child, Level3)
        assert result.child.child.val == 42

    def test_dataclass_field_with_non_dict_value_passes_through(self):
        """When a field's type is a dataclass but the value is already an instance
        (not a dict), it should pass through unchanged."""
        inner_instance = InnerConfig(x=99, y=88)
        data = {"inner": inner_instance, "label": "prebuilt"}
        result = _create_dataclass_from_dict(NestedConfig, data)
        assert isinstance(result, NestedConfig)
        assert isinstance(result.inner, InnerConfig)
        assert result.inner.x == 99
        assert result.inner.y == 88
        assert result.label == "prebuilt"


# ===========================================================================
# update_dict_with_config
# ===========================================================================


class TestUpdateDictWithConfig:
    """Tests for update_dict_with_config."""

    def test_updates_matching_keys(self):
        """Keys present in both the dict and config are updated."""
        dictionary = {"name": "old", "value": 0}
        cfg = OmegaConf.create({"name": "new", "value": 99})
        update_dict_with_config(dictionary, cfg)
        assert dictionary["name"] == "new"
        assert dictionary["value"] == 99

    def test_leaves_non_matching_keys_alone(self):
        """Keys in the dict that are not in the config remain unchanged."""
        dictionary = {"name": "old", "extra": "untouched"}
        cfg = OmegaConf.create({"name": "new"})
        update_dict_with_config(dictionary, cfg)
        assert dictionary["name"] == "new"
        assert dictionary["extra"] == "untouched"

    def test_config_keys_not_in_dict_are_ignored(self):
        """Keys in the config that are not in the dict do not get added."""
        dictionary = {"name": "old"}
        cfg = OmegaConf.create({"name": "new", "other": "not_added"})
        update_dict_with_config(dictionary, cfg)
        assert dictionary == {"name": "new"}
        assert "other" not in dictionary

    # --- New edge-case test ---

    def test_nested_dictconfig_value_is_transferred(self):
        """When the config value is itself a nested DictConfig, the dict should
        receive the nested structure (as a DictConfig object via getattr)."""
        dictionary = {"settings": None}
        cfg = OmegaConf.create({"settings": {"lr": 0.01, "epochs": 5}})
        update_dict_with_config(dictionary, cfg)
        # The transferred value is a DictConfig (getattr returns the raw node)
        assert dictionary["settings"]["lr"] == 0.01
        assert dictionary["settings"]["epochs"] == 5
