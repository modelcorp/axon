"""Tests for axon.utils.fs module."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from axon.utils.fs import local_mkdir_safe


class TestLocalMkdirSafe:
    """Tests for local_mkdir_safe."""

    def test_creates_new_directory(self, tmp_path):
        target = str(tmp_path / "new_dir")
        assert not os.path.exists(target)
        result = local_mkdir_safe(target)
        assert os.path.isdir(target)
        assert isinstance(result, str)

    def test_idempotent_existing_directory(self, tmp_path):
        target = str(tmp_path / "existing")
        os.makedirs(target)
        assert os.path.isdir(target)
        # Calling again on an existing directory should not raise.
        local_mkdir_safe(target)
        assert os.path.isdir(target)

    def test_creates_nested_directories(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        assert not os.path.exists(target)
        local_mkdir_safe(target)
        assert os.path.isdir(target)

    def test_relative_path_joined_with_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = local_mkdir_safe("relative_dir")
        expected = str(tmp_path / "relative_dir")
        assert result == expected
        assert os.path.isdir(expected)

    def test_concurrent_calls_do_not_fail(self, tmp_path):
        target = str(tmp_path / "concurrent_dir")
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(local_mkdir_safe, target) for _ in range(10)]
            results = [f.result() for f in futures]
        assert os.path.isdir(target)
        assert all(r == target for r in results)

    def test_unicode_path(self, tmp_path):
        """Paths with unicode characters should be created successfully."""
        target = str(tmp_path / "\u65e5\u672c\u8a9e" / "data")
        result = local_mkdir_safe(target)
        assert os.path.isdir(target)
        assert result == target

    def test_path_with_spaces(self, tmp_path):
        """Paths containing spaces should be created successfully."""
        target = str(tmp_path / "path with spaces" / "sub dir")
        result = local_mkdir_safe(target)
        assert os.path.isdir(target)
        assert result == target

    def test_file_at_intermediate_segment_raises(self, tmp_path):
        """Creating a path where an intermediate segment is a regular file should raise."""
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        target = str(blocker / "child" / "dir")
        with pytest.raises((OSError, NotADirectoryError)):
            local_mkdir_safe(target)

    def test_lock_file_created_in_temp_dir(self, tmp_path):
        """The locking mechanism should use a ckpt_*.lock file in the system temp dir."""
        import filelock

        target = str(tmp_path / "lock_test_dir")
        with patch.object(filelock, "FileLock", wraps=filelock.FileLock) as mock_lock:
            local_mkdir_safe(target)
            mock_lock.assert_called_once()
            lock_path = mock_lock.call_args[0][0]
            assert lock_path.startswith(tempfile.gettempdir())
            assert os.path.basename(lock_path).startswith("ckpt_")
            assert lock_path.endswith(".lock")
