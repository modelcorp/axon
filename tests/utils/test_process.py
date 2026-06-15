"""Tests for axon.utils.process module."""

import multiprocessing as mp
import signal
import time

from axon.utils.process import cleanup_subprocesses


def _sleeper():
    time.sleep(60)


def _quick_exit():
    pass


def _ignore_sigterm():
    """Process that traps SIGTERM and keeps running (needs SIGKILL)."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


class TestCleanupSubprocesses:
    def test_empty_list_is_noop(self):
        process_list = []
        cleanup_subprocesses(process_list)
        assert process_list == []

    def test_already_exited_process(self):
        p = mp.Process(target=_quick_exit)
        p.start()
        p.join(timeout=5)
        process_list = [p]
        cleanup_subprocesses(process_list)
        assert process_list == []

    def test_terminates_alive_process(self):
        p = mp.Process(target=_sleeper)
        p.start()
        assert p.is_alive()
        process_list = [p]
        cleanup_subprocesses(process_list)
        assert process_list == []
        assert not p.is_alive()

    def test_mixed_alive_and_dead(self):
        p_dead = mp.Process(target=_quick_exit)
        p_dead.start()
        p_dead.join(timeout=5)

        p_alive = mp.Process(target=_sleeper)
        p_alive.start()

        process_list = [p_dead, p_alive]
        cleanup_subprocesses(process_list)
        assert process_list == []
        assert not p_alive.is_alive()

    def test_sigterm_resistant_process_gets_sigkilled(self):
        """Process that ignores SIGTERM should still be cleaned up via SIGKILL."""
        p = mp.Process(target=_ignore_sigterm)
        p.start()
        assert p.is_alive()
        process_list = [p]
        cleanup_subprocesses(process_list)
        assert process_list == []
        assert not p.is_alive()

    def test_list_is_cleared_after_cleanup(self):
        """The input list should be empty after cleanup regardless of process state."""
        procs = [mp.Process(target=_sleeper) for _ in range(3)]
        for p in procs:
            p.start()
        process_list = list(procs)
        cleanup_subprocesses(process_list)
        assert process_list == []
        for p in procs:
            assert not p.is_alive()
