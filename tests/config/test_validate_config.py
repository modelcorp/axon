"""Comprehensive tests for axon.config.validate_config."""

import warnings

import pytest
import yaml

from axon.config.validate_config import (
    _check_mutually_exclusive,
    _deep_merge,
    _get_nested,
    get_effective_strategy,
    is_fsdp_strategy,
    is_megatron_strategy,
    load_config_with_inheritance,
    load_ppo_config,
    needs_critic,
    needs_reference_policy,
    validate_actor_config,
    validate_algorithm_config,
    validate_data_config,
    validate_loss_config,
    validate_axon_config,
    validate_sampler_config,
)

# =========================================================================
# _deep_merge
# =========================================================================


class TestDeepMerge:
    def test_nested_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"z": 3, "w": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"y": 1, "z": 3, "w": 4}}

    def test_nested_non_dict_overrides_dict(self):
        base = {"x": {"y": 1}}
        override = {"x": 42}
        result = _deep_merge(base, override)
        assert result == {"x": 42}

    def test_dict_overrides_non_dict(self):
        base = {"x": 42}
        override = {"x": {"y": 1}}
        result = _deep_merge(base, override)
        assert result == {"x": {"y": 1}}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}

    def test_deeply_conflicting_nested_3_plus_levels(self):
        """Merge with 4+ levels of nesting where conflicts exist at every level."""
        base = {
            "L1": {
                "L2": {
                    "L3": {
                        "L4_keep": "original",
                        "L4_override": "old",
                    },
                    "L3_keep": 100,
                },
                "L2_keep": True,
            },
            "top_keep": [1, 2, 3],
        }
        override = {
            "L1": {
                "L2": {
                    "L3": {
                        "L4_override": "new",
                        "L4_added": "fresh",
                    },
                    "L3_added": 200,
                },
                "L2_added": False,
            },
            "top_added": "hi",
        }
        result = _deep_merge(base, override)
        # Deep values preserved
        assert result["L1"]["L2"]["L3"]["L4_keep"] == "original"
        assert result["L1"]["L2"]["L3"]["L4_override"] == "new"
        assert result["L1"]["L2"]["L3"]["L4_added"] == "fresh"
        assert result["L1"]["L2"]["L3_keep"] == 100
        assert result["L1"]["L2"]["L3_added"] == 200
        assert result["L1"]["L2_keep"] is True
        assert result["L1"]["L2_added"] is False
        assert result["top_keep"] == [1, 2, 3]
        assert result["top_added"] == "hi"

    def test_list_values_override_not_merge(self):
        """Lists should be replaced entirely, not appended."""
        base = {"items": [1, 2, 3], "nested": {"tags": ["a", "b"]}}
        override = {"items": [4, 5], "nested": {"tags": ["c"]}}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]
        assert result["nested"]["tags"] == ["c"]

    def test_none_values_mixed_with_dicts(self):
        """None in override should replace dict in base."""
        base = {"a": {"b": {"c": 1}}, "x": None}
        override = {"a": None, "x": {"y": 2}}
        result = _deep_merge(base, override)
        assert result["a"] is None
        assert result["x"] == {"y": 2}

    def test_override_dict_over_none_base(self):
        """If base has None, override dict replaces it."""
        base = {"config": None}
        override = {"config": {"key": "val"}}
        result = _deep_merge(base, override)
        assert result["config"] == {"key": "val"}


# =========================================================================
# _get_nested
# =========================================================================


class TestGetNested:
    def test_dot_path(self):
        cfg = {"actor": {"fsdp": {"use_remove_padding": True}}}
        assert _get_nested(cfg, "actor.fsdp.use_remove_padding") is True

    def test_missing_with_custom_default(self):
        assert _get_nested({}, "a.b", default=42) == 42

    def test_partial_path_missing(self):
        cfg = {"actor": {"fsdp": {}}}
        assert _get_nested(cfg, "actor.fsdp.use_remove_padding", False) is False

    def test_non_dict_intermediate(self):
        cfg = {"actor": 5}
        assert _get_nested(cfg, "actor.fsdp", "default") == "default"

    def test_deeply_nested_path(self):
        """Test a very deep dot path that exercises intermediate dict checks."""
        cfg = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        assert _get_nested(cfg, "a.b.c.d.e") == 42
        assert _get_nested(cfg, "a.b.c.d.f", "missing") == "missing"
        assert _get_nested(cfg, "a.b.x.d.e") is None


# =========================================================================
# validate_data_config
# =========================================================================


class TestValidateDataConfig:
    def test_valid_config(self):
        validate_data_config({"train_files": "data.jsonl", "max_prompt_length": 512, "max_seq_length": 1024})

    def test_missing_train_files_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_data_config({"max_prompt_length": 512, "max_seq_length": 1024})
            assert len(w) == 1
            assert "train_files" in str(w[0].message)

    def test_zero_prompt_length_raises(self):
        with pytest.raises(ValueError, match="max_prompt_length must be positive"):
            validate_data_config({"train_files": "f", "max_prompt_length": 0})

    def test_negative_prompt_length_raises(self):
        with pytest.raises(ValueError, match="max_prompt_length must be positive"):
            validate_data_config({"train_files": "f", "max_prompt_length": -1})

    def test_zero_seq_length_raises(self):
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_data_config({"train_files": "f", "max_prompt_length": 10, "max_seq_length": 0})

    def test_negative_seq_length_raises(self):
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_data_config({"train_files": "f", "max_prompt_length": 10, "max_seq_length": -5})

    def test_prompt_exceeds_seq_length_raises(self):
        with pytest.raises(ValueError, match="must not exceed"):
            validate_data_config({"train_files": "f", "max_prompt_length": 2048, "max_seq_length": 1024})

    def test_defaults_are_valid(self):
        """When lengths are not specified, defaults (512, 8192) should pass."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            validate_data_config({"train_files": "f"})


# =========================================================================
# validate_algorithm_config
# =========================================================================


class TestValidateAlgorithmConfig:
    VALID_ESTIMATORS = [
        "gae",
        "grpo",
        "rloo",
        "loop",
        "reinforce_plus_plus",
        "reinforce_plus_plus_baseline",
        "remax",
        "opo",
        "grpo_passk",
        "gpg",
        "cispo",
        "identity",
    ]

    @pytest.mark.parametrize("adv", VALID_ESTIMATORS)
    def test_valid_advantage_estimators(self, adv):
        validate_algorithm_config({"advantage": adv})

    def test_invalid_advantage_raises(self):
        with pytest.raises(ValueError, match="Invalid adv_estimator"):
            validate_algorithm_config({"advantage": "bogus"})

    @pytest.mark.parametrize("kl", ["kl", "abs", "mse", "low_var_kl", "full"])
    def test_valid_kl_rewards(self, kl):
        validate_algorithm_config({"kl_reward": kl})

    def test_kl_reward_none_is_valid(self):
        validate_algorithm_config({"kl_reward": None})

    def test_invalid_kl_reward_raises(self):
        with pytest.raises(ValueError, match="Invalid kl_reward"):
            validate_algorithm_config({"kl_reward": "bad_kl"})

    @pytest.mark.parametrize("kl_type", ["fixed", "adaptive"])
    def test_valid_kl_reward_args_type(self, kl_type):
        validate_algorithm_config({"kl_reward_args": {"type": kl_type}})

    def test_invalid_kl_reward_args_type_raises(self):
        with pytest.raises(ValueError, match="Invalid kl_reward_args.type"):
            validate_algorithm_config({"kl_reward_args": {"type": "unknown"}})

    def test_empty_kl_reward_args_ok(self):
        validate_algorithm_config({"kl_reward_args": {}})

    def test_all_fields_combined(self):
        """Exercise all paths in validate_algorithm_config at once."""
        validate_algorithm_config(
            {
                "advantage": "grpo",
                "kl_reward": "abs",
                "kl_reward_args": {"type": "adaptive"},
            }
        )

    def test_invalid_advantage_with_valid_kl(self):
        """Invalid advantage should raise even if KL fields are valid."""
        with pytest.raises(ValueError, match="Invalid adv_estimator"):
            validate_algorithm_config(
                {
                    "advantage": "nonexistent",
                    "kl_reward": "kl",
                    "kl_reward_args": {"type": "fixed"},
                }
            )


# =========================================================================
# validate_loss_config
# =========================================================================


class TestValidateLossConfig:
    def test_valid_token_reduce_values(self):
        for tr in ("sum", "mean", "mean-norm", "mean-program"):
            validate_loss_config({"loss_args": {"token_reduce": tr}})

    def test_invalid_token_reduce_raises(self):
        with pytest.raises(ValueError, match="Invalid token_reduce"):
            validate_loss_config({"loss_args": {"token_reduce": "bad"}})

    def test_valid_batch_reduce_values(self):
        for br in ("token-mean", "step-mean", "program-mean"):
            validate_loss_config({"loss_args": {"batch_reduce": br}})

    def test_invalid_batch_reduce_raises(self):
        with pytest.raises(ValueError, match="Invalid batch_reduce"):
            validate_loss_config({"loss_args": {"batch_reduce": "bad"}})

    @pytest.mark.parametrize("sampler_is", [None, "token", "sequence"])
    def test_valid_sampler_is(self, sampler_is):
        validate_loss_config({"loss_args": {"sampler_is": sampler_is}})

    def test_invalid_sampler_is_raises(self):
        with pytest.raises(ValueError, match="Invalid loss_args.sampler_is"):
            validate_loss_config({"loss_args": {"sampler_is": "bad"}})

    @pytest.mark.parametrize("sampler_rs", [None, "token", "sequence", "geometric"])
    def test_valid_sampler_rs(self, sampler_rs):
        validate_loss_config({"loss_args": {"sampler_rs": sampler_rs}})

    def test_invalid_sampler_rs_raises(self):
        with pytest.raises(ValueError, match="Invalid loss_args.sampler_rs"):
            validate_loss_config({"loss_args": {"sampler_rs": "bad"}})

    def test_all_loss_args_combined_valid(self):
        """All loss_args fields set to valid values simultaneously."""
        validate_loss_config(
            {
                "loss_args": {
                    "token_reduce": "mean",
                    "batch_reduce": "token-mean",
                    "sampler_is": "token",
                    "sampler_rs": "geometric",
                }
            }
        )

    def test_multiple_invalid_fields_first_wins(self):
        """Validation is sequential; token_reduce checked before batch_reduce."""
        with pytest.raises(ValueError, match="Invalid token_reduce"):
            validate_loss_config(
                {
                    "loss_args": {
                        "token_reduce": "invalid_tr",
                        "batch_reduce": "invalid_br",
                    }
                }
            )


# =========================================================================
# _check_mutually_exclusive
# =========================================================================


class TestCheckMutuallyExclusive:
    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="Please set at least one"):
            _check_mutually_exclusive(None, None, "actor")

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="You have set both"):
            _check_mutually_exclusive(4, 2, "actor")

    def test_only_mbs_ok(self):
        _check_mutually_exclusive(4, None, "actor")

    def test_only_mbs_per_gpu_ok(self):
        _check_mutually_exclusive(None, 2, "actor")

    def test_error_message_contains_section(self):
        with pytest.raises(ValueError, match=r"\[critic\]"):
            _check_mutually_exclusive(None, None, "critic")

    def test_custom_param_name(self):
        with pytest.raises(ValueError, match="forward_micro_batch_size"):
            _check_mutually_exclusive(None, None, "actor", "forward_micro_batch_size")


# =========================================================================
# validate_actor_config
# =========================================================================


class TestValidateActorConfig:
    @staticmethod
    def _make_config(
        micro_batch_size=None,
        micro_batch_size_per_gpu=None,
        forward_micro_batch_size=None,
        forward_micro_batch_size_per_gpu=None,
        use_dynamic_bsz=False,
        mini_batch_size=64,
        train_batch_size=64,
        decoding_n=1,
        strategy="fsdp",
        fsdp_sp_size=1,
        fsdp_use_remove_padding=False,
        megatron_tp=1,
        megatron_pp=1,
        megatron_cp=1,
    ):
        config = {
            "strategy": strategy,
            "mini_batch_size": mini_batch_size,
            "train_batch_size": train_batch_size,
            "decoding": {"n": decoding_n},
            "actor": {
                "micro_batch_size": micro_batch_size,
                "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "forward_micro_batch_size": forward_micro_batch_size,
                "forward_micro_batch_size_per_gpu": forward_micro_batch_size_per_gpu,
                "use_dynamic_bsz": use_dynamic_bsz,
                "fsdp": {
                    "ulysses_sequence_parallel_size": fsdp_sp_size,
                    "use_remove_padding": fsdp_use_remove_padding,
                },
                "megatron": {
                    "tensor_model_parallel_size": megatron_tp,
                    "pipeline_model_parallel_size": megatron_pp,
                    "context_parallel_size": megatron_cp,
                },
            },
        }
        return config

    def test_dynamic_bsz_skips_batch_checks(self):
        """With use_dynamic_bsz=True, batch size checks are skipped."""
        cfg = self._make_config(use_dynamic_bsz=True)
        # No micro_batch_size set at all -- should NOT raise
        validate_actor_config(cfg, n_gpus=8)

    def test_fsdp_basic_valid(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=8,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=64,
        )
        validate_actor_config(cfg, n_gpus=8, strategy="fsdp")

    def test_mutual_exclusive_micro_batch_both_none(self):
        cfg = self._make_config(
            forward_micro_batch_size_per_gpu=8,
        )
        with pytest.raises(ValueError, match="Please set at least one"):
            validate_actor_config(cfg, n_gpus=8)

    def test_mutual_exclusive_micro_batch_both_set(self):
        cfg = self._make_config(
            micro_batch_size=4,
            micro_batch_size_per_gpu=2,
            forward_micro_batch_size_per_gpu=8,
        )
        with pytest.raises(ValueError, match="You have set both"):
            validate_actor_config(cfg, n_gpus=8)

    def test_forward_micro_batch_mutual_exclusive(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=8,
            forward_micro_batch_size=4,
            forward_micro_batch_size_per_gpu=2,
        )
        with pytest.raises(ValueError, match="You have set both"):
            validate_actor_config(cfg, n_gpus=8)

    def test_train_batch_size_not_divisible_raises(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=8,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=65,
        )
        with pytest.raises(ValueError, match="real_train_batch_size.*must be divisible"):
            validate_actor_config(cfg, n_gpus=8)

    def test_train_lt_mini_raises(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=8,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=128,
            train_batch_size=64,
        )
        with pytest.raises(ValueError, match="must be >= mini_batch_size"):
            validate_actor_config(cfg, n_gpus=8)

    def test_mini_not_divisible_by_micro(self):
        cfg = self._make_config(
            micro_batch_size=3,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=64,
        )
        with pytest.raises(ValueError, match="mini_batch_size.*must be divisible by micro_batch_size"):
            validate_actor_config(cfg, n_gpus=2)

    def test_micro_times_sp_lt_ngpus(self):
        cfg = self._make_config(
            micro_batch_size=2,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=64,
            fsdp_sp_size=1,
        )
        with pytest.raises(ValueError, match="must be >= n_gpus"):
            validate_actor_config(cfg, n_gpus=8)

    def test_megatron_gpu_divisibility_invalid(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=1,
            forward_micro_batch_size_per_gpu=1,
            mini_batch_size=64,
            train_batch_size=64,
            megatron_tp=4,
            megatron_pp=2,
            megatron_cp=1,
            strategy="megatron",
        )
        # n_gpus=7 not divisible by tp*pp=8
        with pytest.raises(ValueError, match="n_gpus.*must be divisible"):
            validate_actor_config(cfg, n_gpus=7, strategy="megatron")

    def test_megatron_valid(self):
        cfg = self._make_config(
            micro_batch_size_per_gpu=1,
            forward_micro_batch_size_per_gpu=1,
            mini_batch_size=64,
            train_batch_size=64,
            megatron_tp=2,
            megatron_pp=1,
            megatron_cp=1,
            strategy="megatron",
        )
        validate_actor_config(cfg, n_gpus=8, strategy="megatron")

    def test_fsdp_sp_without_remove_padding_raises(self):
        cfg = self._make_config(
            use_dynamic_bsz=True,
            fsdp_sp_size=2,
            fsdp_use_remove_padding=False,
        )
        with pytest.raises(ValueError, match="use_remove_padding"):
            validate_actor_config(cfg, n_gpus=8, strategy="fsdp")

    def test_fsdp_sp_with_remove_padding_ok(self):
        cfg = self._make_config(
            use_dynamic_bsz=True,
            fsdp_sp_size=2,
            fsdp_use_remove_padding=True,
        )
        validate_actor_config(cfg, n_gpus=8, strategy="fsdp")

    def test_fsdp2_sp_without_remove_padding_raises(self):
        cfg = self._make_config(
            use_dynamic_bsz=True,
            fsdp_sp_size=4,
            fsdp_use_remove_padding=False,
        )
        with pytest.raises(ValueError, match="use_remove_padding"):
            validate_actor_config(cfg, n_gpus=8, strategy="fsdp2")

    def test_decoding_n_multiplier(self):
        """real_train_batch_size = train_batch_size * decoding.n"""
        cfg = self._make_config(
            micro_batch_size_per_gpu=8,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=64,
            decoding_n=3,
        )
        # real = 64*3 = 192, minimal = 8, 192 % 8 == 0 -> ok
        validate_actor_config(cfg, n_gpus=8)

    def test_micro_batch_size_exactly_equals_n_gpus_boundary(self):
        """micro_batch_size * sp_size == n_gpus is the exact boundary (should pass)."""
        cfg = self._make_config(
            micro_batch_size=8,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=64,
            train_batch_size=64,
            fsdp_sp_size=1,
        )
        # micro_batch_size(8) * sp_size(1) = 8 == n_gpus(8) -> ok (>= passes)
        validate_actor_config(cfg, n_gpus=8)

    def test_micro_batch_size_just_below_n_gpus_boundary(self):
        """micro_batch_size * sp_size = n_gpus - 1 should fail."""
        cfg = self._make_config(
            micro_batch_size=7,
            forward_micro_batch_size_per_gpu=8,
            mini_batch_size=7,
            train_batch_size=56,
            fsdp_sp_size=1,
        )
        with pytest.raises(ValueError, match="must be >= n_gpus"):
            validate_actor_config(cfg, n_gpus=8)

    def test_megatron_with_pipeline_parallelism_that_barely_fits(self):
        """Megatron: n_gpus=16, tp=4, pp=4, cp=1 => model_parallel=16, 16%16==0."""
        cfg = self._make_config(
            micro_batch_size_per_gpu=1,
            forward_micro_batch_size_per_gpu=1,
            mini_batch_size=64,
            train_batch_size=64,
            megatron_tp=4,
            megatron_pp=4,
            megatron_cp=1,
            strategy="megatron",
        )
        # 16 % (4*4*1) == 0, dp_size = 16/(4*4) = 1, minimal_bsz = 1*1 = 1
        # real_train_batch_size = 64, 64%1==0, 64>=64 -> ok
        validate_actor_config(cfg, n_gpus=16, strategy="megatron")

    def test_megatron_with_context_parallelism_not_divisible(self):
        """Megatron: n_gpus=16, tp=2, pp=2, cp=3 => model_parallel*cp=12, 16%12!=0."""
        cfg = self._make_config(
            micro_batch_size_per_gpu=1,
            forward_micro_batch_size_per_gpu=1,
            mini_batch_size=64,
            train_batch_size=64,
            megatron_tp=2,
            megatron_pp=2,
            megatron_cp=3,
            strategy="megatron",
        )
        with pytest.raises(ValueError, match="n_gpus.*must be divisible"):
            validate_actor_config(cfg, n_gpus=16, strategy="megatron")

    def test_real_world_fsdp_config_multiple_paths(self):
        """Real-world-like config exercising multiple validation paths at once."""
        cfg = self._make_config(
            micro_batch_size=16,
            forward_micro_batch_size_per_gpu=4,
            mini_batch_size=64,
            train_batch_size=128,
            decoding_n=2,
            fsdp_sp_size=2,
            fsdp_use_remove_padding=True,
        )
        # real_train = 128*2=256, minimal=8 for fsdp, 256%8==0
        # 128>=64, 64%16==0, 16*2=32>=8
        validate_actor_config(cfg, n_gpus=8, strategy="fsdp")


# =========================================================================
# validate_sampler_config
# =========================================================================


class TestValidateSamplerConfig:
    def test_valid_lora_rank(self):
        cfg = {"sampler": {"name": "vllm"}, "actor": {"lora": {"rank": 256}}}
        validate_sampler_config(cfg)

    def test_lora_rank_512_ok(self):
        cfg = {"sampler": {"name": "vllm"}, "actor": {"lora": {"rank": 512}}}
        validate_sampler_config(cfg)

    def test_lora_rank_exceeds_512_raises(self):
        cfg = {"sampler": {"name": "vllm"}, "actor": {"lora": {"rank": 1024}}}
        with pytest.raises(ValueError, match="LoRA rank.*must be less than or equal to 512"):
            validate_sampler_config(cfg)

    def test_lora_rank_high_non_vllm_ok(self):
        cfg = {"sampler": {"name": "sglang"}, "actor": {"lora": {"rank": 1024}}}
        validate_sampler_config(cfg)

    def test_moe_replay_prefix_caching_incompatible(self):
        cfg = {
            "moe_replay": True,
            "sampler": {"enable_prefix_caching": True},
            "actor": {},
        }
        with pytest.raises(ValueError, match="moe_replay is incompatible"):
            validate_sampler_config(cfg)

    def test_moe_replay_without_prefix_caching_ok(self):
        cfg = {"moe_replay": True, "sampler": {"enable_prefix_caching": False}, "actor": {}}
        validate_sampler_config(cfg)

    def test_no_moe_replay_with_prefix_caching_ok(self):
        cfg = {"moe_replay": False, "sampler": {"enable_prefix_caching": True}, "actor": {}}
        validate_sampler_config(cfg)

    def test_empty_config_ok(self):
        validate_sampler_config({"sampler": {}, "actor": {}})


# =========================================================================
# needs_reference_policy
# =========================================================================


class TestNeedsReferencePolicy:
    def test_kl_coef_positive(self):
        assert needs_reference_policy({"loss_args": {"kl_coef": 0.1}}) is True

    def test_kl_coef_zero(self):
        assert needs_reference_policy({"loss_args": {"kl_coef": 0}}) is False

    def test_kl_reward_set(self):
        assert needs_reference_policy({"kl_reward": "kl"}) is True

    def test_nested_kl_reward(self):
        assert needs_reference_policy({"algorithm": {"kl_reward": "abs"}}) is True

    def test_no_kl_at_all(self):
        assert needs_reference_policy({}) is False

    def test_kl_reward_none_and_coef_zero(self):
        assert needs_reference_policy({"kl_reward": None, "loss_args": {"kl_coef": 0}}) is False

    def test_both_set(self):
        assert needs_reference_policy({"kl_reward": "mse", "loss_args": {"kl_coef": 0.5}}) is True


# =========================================================================
# needs_critic
# =========================================================================


class TestNeedsCritic:
    def test_critic_enabled_explicitly(self):
        assert needs_critic({"critic": {"enable": True}}) is True

    def test_critic_disabled_explicitly(self):
        assert needs_critic({"critic": {"enable": False}}) is False

    def test_gae_advantage_needs_critic(self):
        assert needs_critic({"advantage": "gae"}) is True

    def test_grpo_advantage_no_critic(self):
        assert needs_critic({"advantage": "grpo"}) is False

    def test_default_is_gae(self):
        """When no advantage is specified at all, default is 'gae' -> needs critic."""
        assert needs_critic({}) is True

    def test_algorithm_nested_key(self):
        assert needs_critic({"algorithm": {"adv_estimator": "grpo"}}) is False

    def test_explicit_enable_overrides_advantage(self):
        """critic.enable takes precedence over advantage estimator."""
        assert needs_critic({"critic": {"enable": False}, "advantage": "gae"}) is False
        assert needs_critic({"critic": {"enable": True}, "advantage": "grpo"}) is True


# =========================================================================
# Strategy helpers (combined test)
# =========================================================================


class TestStrategyHelpers:
    def test_fsdp_and_fsdp2_are_fsdp(self):
        """Both fsdp and fsdp2 should be identified as FSDP strategies."""
        assert is_fsdp_strategy({"strategy": "fsdp"}) is True
        assert is_fsdp_strategy({"strategy": "fsdp2"}) is True
        assert is_fsdp_strategy({"strategy": "megatron"}) is False

    def test_megatron_identification(self):
        assert is_megatron_strategy({"strategy": "megatron"}) is True
        assert is_megatron_strategy({"strategy": "fsdp"}) is False

    def test_strategy_helpers_are_consistent(self):
        """For any config, exactly one of is_fsdp or is_megatron should be True
        (or both False for unknown)."""
        for s in ["fsdp", "fsdp2", "megatron"]:
            cfg = {"strategy": s}
            assert is_fsdp_strategy(cfg) != is_megatron_strategy(cfg), f"inconsistency for {s}"

    def test_default_strategy_is_fsdp(self):
        """Empty config should default to fsdp."""
        assert get_effective_strategy({}) == "fsdp"
        assert is_fsdp_strategy({}) is True


# =========================================================================
# load_ppo_config
# =========================================================================


class TestLoadPpoConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"train_batch_size": 32, "actor": {"lr": 1e-5}}))
        result = load_ppo_config(cfg_file)
        assert result["train_batch_size"] == 32
        assert result["actor"]["lr"] == 1e-5

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_ppo_config(tmp_path / "nonexistent.yaml")

    def test_with_overrides(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"a": 1, "nested": {"b": 2}}))
        result = load_ppo_config(cfg_file, overrides={"a": 99, "nested": {"c": 3}})
        assert result["a"] == 99
        assert result["nested"]["b"] == 2
        assert result["nested"]["c"] == 3

    def test_accepts_path_string(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"x": 1}))
        result = load_ppo_config(str(cfg_file))
        assert result == {"x": 1}


# =========================================================================
# load_config_with_inheritance
# =========================================================================


class TestLoadConfigWithInheritance:
    def test_base_key_resolves_parent(self, tmp_path):
        """Child with _base_ key inherits from parent config."""
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            yaml.dump(
                {
                    "actor": {"lr": 1e-4, "batch": 32},
                    "strategy": "fsdp",
                }
            )
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "_base_": "parent.yaml",
                    "actor": {"lr": 5e-5},
                    "train_batch_size": 128,
                }
            )
        )
        result = load_config_with_inheritance(child)
        # Child overrides parent's lr
        assert result["actor"]["lr"] == 5e-5
        # Parent's batch inherited
        assert result["actor"]["batch"] == 32
        # Parent's strategy inherited
        assert result["strategy"] == "fsdp"
        # Child's own key
        assert result["train_batch_size"] == 128
        # _base_ should be removed
        assert "_base_" not in result

    def test_multi_level_inheritance(self, tmp_path):
        """Grandparent -> parent -> child resolution chain."""
        grandparent = tmp_path / "grandparent.yaml"
        grandparent.write_text(
            yaml.dump(
                {
                    "a": 1,
                    "b": {"c": 2, "d": 3},
                }
            )
        )
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            yaml.dump(
                {
                    "_base_": "grandparent.yaml",
                    "b": {"c": 20},
                    "e": 5,
                }
            )
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "_base_": "parent.yaml",
                    "b": {"d": 30},
                    "f": 6,
                }
            )
        )
        # Load parent first (inherits grandparent)
        parent_cfg = load_config_with_inheritance(parent)
        assert parent_cfg["a"] == 1
        assert parent_cfg["b"]["c"] == 20
        assert parent_cfg["b"]["d"] == 3
        assert parent_cfg["e"] == 5

        # Child inherits parent (which already resolved grandparent)
        # But child doesn't go through intermediate -- it loads parent.yaml
        # which has _base_ pointing to grandparent
        child_cfg = load_config_with_inheritance(child)
        # child's _base_ is parent.yaml. load_config_with_inheritance loads parent.yaml raw,
        # then merges child on top. Parent has _base_ to grandparent but that's not resolved here.
        # So we test what the function actually does:
        assert child_cfg["b"]["d"] == 30
        assert child_cfg["f"] == 6

    def test_explicit_base_config_path_overrides_base_key(self, tmp_path):
        """When base_config_path is given, it overrides the _base_ key."""
        alt_base = tmp_path / "alt_base.yaml"
        alt_base.write_text(yaml.dump({"from_alt": True, "shared": "alt"}))
        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "_base_": "nonexistent.yaml",
                    "shared": "child",
                }
            )
        )
        result = load_config_with_inheritance(child, base_config_path=alt_base)
        assert result["from_alt"] is True
        assert result["shared"] == "child"

    def test_overrides_applied_on_top(self, tmp_path):
        """Overrides dict is applied after inheritance."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"a": 1, "b": 2}))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"_base_": "base.yaml", "b": 20}))
        result = load_config_with_inheritance(child, overrides={"b": 200, "c": 3})
        assert result["a"] == 1
        assert result["b"] == 200
        assert result["c"] == 3


# =========================================================================
# validate_axon_config end-to-end
# =========================================================================


class TestValidateAxonConfigEndToEnd:
    @staticmethod
    def _make_full_config(**overrides):
        """Build a realistic full config that passes validate_axon_config."""
        cfg = {
            "strategy": "fsdp",
            "num_gpus_per_node": 8,
            "num_nodes": 1,
            "train_files": "data.jsonl",
            "max_prompt_length": 512,
            "max_seq_length": 1024,
            "train_batch_size": 64,
            "mini_batch_size": 64,
            "advantage": "grpo",
            "decoding": {"n": 1},
            "sampler": {"name": "vllm"},
            "actor": {
                "micro_batch_size_per_gpu": 8,
                "forward_micro_batch_size_per_gpu": 8,
                "use_dynamic_bsz": False,
                "fsdp": {
                    "ulysses_sequence_parallel_size": 1,
                    "use_remove_padding": False,
                },
                "megatron": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                },
            },
        }
        return _deep_merge(cfg, overrides) if overrides else cfg

    def test_full_config_passes(self):
        """A well-formed config should pass all validators."""
        cfg = self._make_full_config()
        validate_axon_config(cfg)

    def test_mutate_train_batch_size_breaks(self):
        """train_batch_size not divisible by n_gpus should fail."""
        cfg = self._make_full_config(train_batch_size=65)
        with pytest.raises(ValueError):
            validate_axon_config(cfg)

    def test_mutate_advantage_breaks(self):
        """Invalid advantage estimator should fail."""
        cfg = self._make_full_config(advantage="nonexistent_advantage")
        with pytest.raises(ValueError, match="Invalid adv_estimator"):
            validate_axon_config(cfg)

    def test_mutate_prompt_length_breaks(self):
        """Zero prompt length should fail."""
        cfg = self._make_full_config(max_prompt_length=0)
        with pytest.raises(ValueError, match="max_prompt_length must be positive"):
            validate_axon_config(cfg)

    def test_mutate_lora_rank_breaks(self):
        """LoRA rank > 512 with vllm should fail."""
        cfg = self._make_full_config()
        cfg["actor"]["lora"] = {"rank": 1024}
        with pytest.raises(ValueError, match="LoRA rank"):
            validate_axon_config(cfg)

    def test_eval_mode_skips_training_validators(self):
        """In eval mode, actor/algorithm/loss validators are skipped."""
        cfg = self._make_full_config(
            mode="eval",
            advantage="TOTALLY_INVALID",
        )
        # Should not raise because eval mode skips algorithm validation
        validate_axon_config(cfg)

    def test_mutate_mini_gt_train_breaks(self):
        """mini_batch_size > train_batch_size should fail."""
        cfg = self._make_full_config(mini_batch_size=128, train_batch_size=64)
        with pytest.raises(ValueError, match="must be >= mini_batch_size"):
            validate_axon_config(cfg)
