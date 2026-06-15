"""Tests for axon.utils.networking_utils module."""

import socket
from unittest import mock

import pytest

from axon.utils.networking_utils import ensure_port_available, get_free_port, is_valid_ipv6_address


# ---------------------------------------------------------------------------
# is_valid_ipv6_address
# ---------------------------------------------------------------------------
class TestIsValidIpv6Address:
    @pytest.mark.parametrize(
        "addr",
        [
            "::1",
            "::",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            "fe80::1",
            "::ffff:192.0.2.1",
        ],
    )
    def test_valid(self, addr):
        assert is_valid_ipv6_address(addr) is True

    @pytest.mark.parametrize(
        "addr",
        [
            "192.168.1.1",
            "not-an-address",
            "",
            "127.0.0.1",
            "::gggg",
            "2001:db8::1::1",
        ],
    )
    def test_invalid(self, addr):
        assert is_valid_ipv6_address(addr) is False


# ---------------------------------------------------------------------------
# get_free_port
# ---------------------------------------------------------------------------
class TestGetFreePort:
    def test_port_in_valid_range(self):
        port, sock = get_free_port("127.0.0.1")
        try:
            assert 1 <= port <= 65535
        finally:
            sock.close()

    def test_consecutive_calls_unique(self):
        socks = []
        ports = set()
        try:
            for _ in range(5):
                p, s = get_free_port("127.0.0.1")
                ports.add(p)
                socks.append(s)
            assert len(ports) == 5
        finally:
            for s in socks:
                s.close()

    def test_socket_bound_to_returned_port(self):
        port, sock = get_free_port("127.0.0.1")
        try:
            assert sock.getsockname()[1] == port
        finally:
            sock.close()

    def test_socket_has_reuse_options(self):
        _, sock = get_free_port("127.0.0.1")
        try:
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) != 0
        finally:
            sock.close()

    def test_ipv6_uses_af_inet6(self):
        try:
            port, sock = get_free_port("::1")
            try:
                assert sock.family == socket.AF_INET6
                assert 1 <= port <= 65535
            finally:
                sock.close()
        except OSError:
            pytest.skip("IPv6 not available")


# ---------------------------------------------------------------------------
# ensure_port_available
# ---------------------------------------------------------------------------
class TestEnsurePortAvailable:
    def test_free_port_is_noop(self):
        """A port with no listeners should return without error."""
        _, sock = get_free_port("127.0.0.1")
        port = sock.getsockname()[1]
        sock.close()
        # Port is now free
        ensure_port_available(port)  # should not raise

    def test_occupied_port_raises_without_force(self):
        """An occupied port should raise RuntimeError when force=False."""
        port, sock = get_free_port("127.0.0.1")
        try:
            # lsof may or may not find our socket (depends on listen state),
            # so we mock the subprocess to simulate an occupied port
            with mock.patch("axon.utils.networking_utils.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(stdout="12345\n")
                with pytest.raises(RuntimeError, match="already in use"):
                    ensure_port_available(port, force=False)
        finally:
            sock.close()

    def test_force_kills_process(self):
        """With force=True, should attempt os.kill on the PIDs."""
        with (
            mock.patch("axon.utils.networking_utils.subprocess.run") as mock_run,
            mock.patch("axon.utils.networking_utils.os.kill") as mock_kill,
            mock.patch("axon.utils.networking_utils.time.sleep"),
        ):
            mock_run.return_value = mock.Mock(stdout="111\n222\n")
            ensure_port_available(9999, force=True)
            assert mock_kill.call_count == 2
            mock_kill.assert_any_call(111, mock.ANY)
            mock_kill.assert_any_call(222, mock.ANY)

    def test_force_ignores_already_dead_process(self):
        """ProcessLookupError during kill should be silently ignored."""
        with (
            mock.patch("axon.utils.networking_utils.subprocess.run") as mock_run,
            mock.patch("axon.utils.networking_utils.os.kill", side_effect=ProcessLookupError),
            mock.patch("axon.utils.networking_utils.time.sleep"),
        ):
            mock_run.return_value = mock.Mock(stdout="999\n")
            ensure_port_available(9999, force=True)  # should not raise
