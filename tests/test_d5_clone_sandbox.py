"""Real sandbox controls; never substitute a mock for the egress proof."""

import sys

import pytest

from bench.instruments.d5_clone import sandbox
from bench.instruments.d5_clone.provider import Provider


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec required")
def test_positive_control():
    provider = Provider({}).start()
    try:
        assert sandbox.positive_control(provider.url + "/models") == {
            "external_blocked": True, "dns_blocked": True, "local_ok": True, "verdict": "PASS"}
    finally:
        provider.stop()


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec required")
def test_wrap():
    argv = ["python", "-c", "pass"]
    assert sandbox.wrap(argv) == ["/usr/bin/sandbox-exec", "-p", sandbox.PROFILE] + argv
    assert '(deny network-outbound)' in sandbox.PROFILE
    assert '(remote ip "localhost:*")' in sandbox.PROFILE
    assert '(remote unix-socket)' in sandbox.PROFILE
    resolver = '(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))'
    # The resolver deny must follow the unix-socket allow: the later rule wins.
    assert sandbox.PROFILE.index(resolver) > sandbox.PROFILE.index('(remote unix-socket))')


@pytest.mark.skipif(sys.platform == "darwin", reason="non-macOS contract")
def test_unavailable_off_macos():
    assert sandbox.available() is False
    with pytest.raises(RuntimeError, match="macOS sandbox-exec is unavailable"):
        sandbox.positive_control("http://127.0.0.1/")
