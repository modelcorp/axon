"""Tests for axon.utils.state.utils checkpoint utility functions."""

import os

from axon.utils.state.utils import (
    delete_oldest_checkpoints,
    find_latest_ckpt_path,
    get_checkpoint_directories,
    is_valid_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fsdp_checkpoint(base, *, model_shards=True, hf_safetensors=False):
    """Create a valid FSDP-style checkpoint directory tree under *base*.

    Args:
        base: Root directory for the checkpoint.
        model_shards: If True, place model_world_size_*.pt files in actor/.
        hf_safetensors: If True, place .safetensors files in actor/huggingface/.
    """
    os.makedirs(os.path.join(base, "actor", "huggingface"), exist_ok=True)
    # data.pt at root
    open(os.path.join(base, "data.pt"), "w").close()
    # Required FSDP shard files
    open(os.path.join(base, "actor", "optim_world_size_2_rank_0.pt"), "w").close()
    open(os.path.join(base, "actor", "extra_state_world_size_2_rank_0.pt"), "w").close()
    if model_shards:
        open(os.path.join(base, "actor", "model_world_size_2_rank_0.pt"), "w").close()
    if hf_safetensors:
        open(os.path.join(base, "actor", "huggingface", "model-00001-of-00002.safetensors"), "w").close()


def _make_megatron_checkpoint(base, *, use_distcp=True, use_common=False):
    """Create a valid Megatron-style checkpoint directory tree under *base*."""
    os.makedirs(os.path.join(base, "actor", "huggingface"), exist_ok=True)
    open(os.path.join(base, "data.pt"), "w").close()
    if use_distcp:
        open(os.path.join(base, "actor", "shard_0.distcp"), "w").close()
    if use_common:
        open(os.path.join(base, "actor", "common.pt"), "w").close()


# ===========================================================================
# is_valid_checkpoint
# ===========================================================================


class TestIsValidCheckpoint:
    """Tests for is_valid_checkpoint."""

    def test_valid_fsdp_with_model_shards(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        _make_fsdp_checkpoint(str(ckpt), model_shards=True)
        assert is_valid_checkpoint(str(ckpt)) is True

    def test_valid_fsdp_with_hf_safetensors(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        _make_fsdp_checkpoint(str(ckpt), model_shards=False, hf_safetensors=True)
        assert is_valid_checkpoint(str(ckpt)) is True

    def test_valid_megatron_distcp(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        _make_megatron_checkpoint(str(ckpt), use_distcp=True, use_common=False)
        assert is_valid_checkpoint(str(ckpt)) is True

    def test_valid_megatron_common_pt(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        _make_megatron_checkpoint(str(ckpt), use_distcp=False, use_common=True)
        assert is_valid_checkpoint(str(ckpt)) is True

    def test_missing_fsdp_shard_files(self, tmp_path):
        """Has data.pt, actor/, huggingface/, and model shards but no optim/extra_state shards."""
        ckpt = tmp_path / "ckpt"
        os.makedirs(os.path.join(str(ckpt), "actor", "huggingface"))
        (ckpt / "data.pt").touch()
        # Only model shard, missing optim and extra_state shards
        open(os.path.join(str(ckpt), "actor", "model_world_size_2_rank_0.pt"), "w").close()
        assert is_valid_checkpoint(str(ckpt)) is False

    def test_missing_model_weights_no_shards_no_hf(self, tmp_path):
        """Has optim & extra_state shards but neither model shards nor HF weights."""
        ckpt = tmp_path / "ckpt"
        os.makedirs(os.path.join(str(ckpt), "actor", "huggingface"))
        (ckpt / "data.pt").touch()
        open(os.path.join(str(ckpt), "actor", "optim_world_size_2_rank_0.pt"), "w").close()
        open(os.path.join(str(ckpt), "actor", "extra_state_world_size_2_rank_0.pt"), "w").close()
        assert is_valid_checkpoint(str(ckpt)) is False

    def test_actor_dir_exists_but_is_file_not_directory(self, tmp_path):
        """actor exists but is a file, not a directory - should fail."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "data.pt").touch()
        (ckpt / "actor").touch()  # File, not dir!
        assert is_valid_checkpoint(str(ckpt)) is False

    def test_huggingface_dir_exists_but_empty_fsdp_should_fail(self, tmp_path):
        """huggingface dir exists but actor has no FSDP or megatron files."""
        ckpt = tmp_path / "ckpt"
        os.makedirs(os.path.join(str(ckpt), "actor", "huggingface"))
        (ckpt / "data.pt").touch()
        # actor/ and huggingface/ exist, but no model files at all
        assert is_valid_checkpoint(str(ckpt)) is False

    def test_both_megatron_and_fsdp_files_present(self, tmp_path):
        """Both megatron and FSDP files present - megatron detected first, should pass."""
        ckpt = tmp_path / "ckpt"
        os.makedirs(os.path.join(str(ckpt), "actor", "huggingface"))
        (ckpt / "data.pt").touch()
        # Megatron files
        open(os.path.join(str(ckpt), "actor", "shard_0.distcp"), "w").close()
        # FSDP files
        open(os.path.join(str(ckpt), "actor", "optim_world_size_2_rank_0.pt"), "w").close()
        open(os.path.join(str(ckpt), "actor", "extra_state_world_size_2_rank_0.pt"), "w").close()
        open(os.path.join(str(ckpt), "actor", "model_world_size_2_rank_0.pt"), "w").close()
        assert is_valid_checkpoint(str(ckpt)) is True


# ===========================================================================
# get_checkpoint_directories
# ===========================================================================


class TestGetCheckpointDirectories:
    """Tests for get_checkpoint_directories."""

    def test_finds_numeric_dirs(self, tmp_path):
        (tmp_path / "100").mkdir()
        (tmp_path / "200").mkdir()
        result = get_checkpoint_directories(str(tmp_path))
        assert result == [
            (100, str(tmp_path / "100")),
            (200, str(tmp_path / "200")),
        ]

    def test_finds_step_prefixed_dirs(self, tmp_path):
        (tmp_path / "step_10").mkdir()
        (tmp_path / "step_20").mkdir()
        result = get_checkpoint_directories(str(tmp_path))
        assert result == [
            (10, str(tmp_path / "step_10")),
            (20, str(tmp_path / "step_20")),
        ]

    def test_finds_global_step_prefixed_dirs(self, tmp_path):
        (tmp_path / "global_step_5").mkdir()
        (tmp_path / "global_step_15").mkdir()
        result = get_checkpoint_directories(str(tmp_path))
        assert result == [
            (5, str(tmp_path / "global_step_5")),
            (15, str(tmp_path / "global_step_15")),
        ]

    def test_mixed_formats_sorted(self, tmp_path):
        (tmp_path / "50").mkdir()
        (tmp_path / "step_10").mkdir()
        (tmp_path / "global_step_30").mkdir()
        result = get_checkpoint_directories(str(tmp_path))
        steps = [s for s, _ in result]
        assert steps == [10, 30, 50]

    def test_ignores_invalid_names(self, tmp_path):
        (tmp_path / "step_10").mkdir()
        (tmp_path / "not_a_checkpoint").mkdir()
        (tmp_path / "step_abc").mkdir()
        (tmp_path / "random_file.txt").touch()  # file, not dir
        result = get_checkpoint_directories(str(tmp_path))
        assert len(result) == 1
        assert result[0][0] == 10

    def test_adversarial_names(self, tmp_path):
        """Names that look numeric-ish but aren't valid checkpoint dirs."""
        (tmp_path / "step_").mkdir()  # empty step number
        (tmp_path / "123abc").mkdir()  # mixed chars
        (tmp_path / "step_-1").mkdir()  # negative
        (tmp_path / "step_10").mkdir()  # valid
        # A file with a matching name should not be picked up
        (tmp_path / "step_99").touch()  # file, not dir
        result = get_checkpoint_directories(str(tmp_path))
        assert len(result) == 1
        assert result[0] == (10, str(tmp_path / "step_10"))

    def test_very_large_step_number(self, tmp_path):
        """Very large step number should still be parseable."""
        (tmp_path / "step_999999999999").mkdir()
        result = get_checkpoint_directories(str(tmp_path))
        assert len(result) == 1
        assert result[0][0] == 999999999999


# ===========================================================================
# delete_oldest_checkpoints
# ===========================================================================


class TestDeleteOldestCheckpoints:
    """Tests for delete_oldest_checkpoints."""

    def test_deletes_oldest_keeps_max(self, tmp_path):
        """5 checkpoints, keep 3 -> delete oldest 2."""
        for i in range(1, 6):
            (tmp_path / f"step_{i}").mkdir()

        deleted = delete_oldest_checkpoints(str(tmp_path), max_to_keep=3)

        assert len(deleted) == 2
        assert str(tmp_path / "step_1") in deleted
        assert str(tmp_path / "step_2") in deleted

        # Oldest two should be gone, newest three should remain
        assert not (tmp_path / "step_1").exists()
        assert not (tmp_path / "step_2").exists()
        assert (tmp_path / "step_3").exists()
        assert (tmp_path / "step_4").exists()
        assert (tmp_path / "step_5").exists()

    def test_fewer_checkpoints_than_max(self, tmp_path):
        """2 checkpoints, keep 5 -> no deletion."""
        (tmp_path / "step_1").mkdir()
        (tmp_path / "step_2").mkdir()
        deleted = delete_oldest_checkpoints(str(tmp_path), max_to_keep=5)
        assert deleted == []
        assert (tmp_path / "step_1").exists()
        assert (tmp_path / "step_2").exists()

    def test_exactly_max_to_keep(self, tmp_path):
        """3 checkpoints, keep 3 -> no deletion."""
        for i in range(1, 4):
            (tmp_path / f"step_{i}").mkdir()
        deleted = delete_oldest_checkpoints(str(tmp_path), max_to_keep=3)
        assert deleted == []

    def test_deletion_failure_continues_others(self, tmp_path):
        """When deletion of one checkpoint fails, others should still be deleted."""
        for i in range(1, 6):
            d = tmp_path / f"step_{i}"
            d.mkdir()
            # Create a file inside so the dir is non-empty
            (d / "data.pt").touch()

        # Make step_1 unremovable by removing write permission on it
        bad_dir = tmp_path / "step_1"
        bad_file = bad_dir / "data.pt"
        bad_file.chmod(0o000)
        bad_dir.chmod(0o500)  # read+execute only, no write -> rmtree should fail

        try:
            deleted = delete_oldest_checkpoints(str(tmp_path), max_to_keep=3)
            # step_1 should have failed but step_2 should have been deleted
            assert str(tmp_path / "step_2") in deleted
            # step_1 should still exist (deletion failed)
            assert (tmp_path / "step_1").exists()
        finally:
            # Restore permissions for cleanup
            bad_dir.chmod(0o700)
            bad_file.chmod(0o644)

    def test_max_to_keep_none_and_nonpositive(self, tmp_path):
        """None, zero, and negative max_to_keep should all result in no deletion."""
        (tmp_path / "step_1").mkdir()
        for val in [None, 0, -1]:
            deleted = delete_oldest_checkpoints(str(tmp_path), max_to_keep=val)
            assert deleted == []
            assert (tmp_path / "step_1").exists()


# ===========================================================================
# find_latest_ckpt_path
# ===========================================================================


class TestFindLatestCkptPath:
    """Tests for find_latest_ckpt_path."""

    def test_returns_latest_valid(self, tmp_path):
        """Multiple checkpoints, only some valid -> returns latest valid."""
        # step_10: valid
        _make_fsdp_checkpoint(str(tmp_path / "step_10"))
        # step_20: valid
        _make_fsdp_checkpoint(str(tmp_path / "step_20"))
        # step_30: invalid (missing data.pt)
        (tmp_path / "step_30").mkdir()

        result = find_latest_ckpt_path(str(tmp_path))
        assert result == str(tmp_path / "step_20")

    def test_skips_invalid_returns_next(self, tmp_path):
        """Latest two are invalid, third from top is valid."""
        _make_fsdp_checkpoint(str(tmp_path / "step_5"))
        # step_10 and step_15: invalid (just empty dirs)
        (tmp_path / "step_10").mkdir()
        (tmp_path / "step_15").mkdir()

        result = find_latest_ckpt_path(str(tmp_path))
        assert result == str(tmp_path / "step_5")

    def test_no_valid_checkpoints(self, tmp_path):
        """All checkpoint dirs are malformed."""
        (tmp_path / "step_1").mkdir()
        (tmp_path / "step_2").mkdir()
        assert find_latest_ckpt_path(str(tmp_path)) is None

    def test_no_checkpoint_dirs_at_all(self, tmp_path):
        """Empty directory -> None."""
        assert find_latest_ckpt_path(str(tmp_path)) is None

    def test_nonexistent_path(self, tmp_path):
        assert find_latest_ckpt_path(str(tmp_path / "no_such_dir")) is None

    def test_returns_latest_among_all_valid(self, tmp_path):
        """All checkpoints valid -> returns the highest step."""
        _make_megatron_checkpoint(str(tmp_path / "global_step_100"))
        _make_megatron_checkpoint(str(tmp_path / "global_step_200"))
        _make_megatron_checkpoint(str(tmp_path / "global_step_300"))

        result = find_latest_ckpt_path(str(tmp_path))
        assert result == str(tmp_path / "global_step_300")

    def test_stress_many_checkpoints_only_middle_valid(self, tmp_path):
        """100 numbered checkpoints, only the 50th is valid."""
        for i in range(1, 101):
            d = tmp_path / f"step_{i}"
            if i == 50:
                _make_fsdp_checkpoint(str(d))
            else:
                d.mkdir()

        result = find_latest_ckpt_path(str(tmp_path))
        assert result == str(tmp_path / "step_50")

    def test_path_is_none(self):
        assert find_latest_ckpt_path(None) is None
