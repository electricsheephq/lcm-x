"""Real sandbox controls; never substitute a mock for the egress proof."""

from pathlib import Path
import socket
import subprocess
import sys

import pytest

from bench.instruments.d5_clone import sandbox
from bench.instruments.d5_clone.provider import Provider


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec required")
def test_positive_control():
    provider = Provider({}).start()
    try:
        assert sandbox.positive_control(provider.url + "/models") == {
            "external_blocked": True, "other_local_blocked": True, "dns_blocked": True, "local_ok": True,
            "verdict": "PASS"}
    finally:
        provider.stop()


def test_profile_allows_only_named_ports_and_socket_roots(tmp_path):
    text = sandbox.profile([8123], [tmp_path])
    assert "(deny network-outbound)" in text
    assert '(remote ip "localhost:8123")' in text
    assert f'(subpath "{tmp_path.resolve()}")' in text
    assert '"localhost:*"' not in text and "(remote unix-socket))" not in text
    argv = ["python", "-c", "pass"]
    assert sandbox.wrap(argv, ports=[8123]) == ["/usr/bin/sandbox-exec", "-p", sandbox.profile([8123])] + argv
    for bad in (0, 65536, "80", True):
        with pytest.raises(ValueError):
            sandbox.profile([bad])


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec required")
def test_unix_socket_outside_named_root_is_denied(tmp_path, monkeypatch):
    # AF_UNIX paths are limited to about 104 bytes, so bind by relative path from inside tmp_path.
    monkeypatch.chdir(tmp_path)
    servers = []
    for directory in ("run", "elsewhere"):
        (tmp_path / directory).mkdir()
        server = socket.socket(socket.AF_UNIX)
        server.bind(f"{directory}/s.sock")
        server.listen(1)
        servers.append(server)
    child = ("import errno, socket, sys\n"
             "def go(p):\n"
             "    s = socket.socket(socket.AF_UNIX)\n"
             "    try:\n"
             "        s.connect(p)\n"
             "        return 'ok'\n"
             "    except OSError as e:\n"
             "        return 'denied' if e.errno in (errno.EPERM, errno.EACCES) else 'error'\n"
             "print(go(sys.argv[1]), go(sys.argv[2]))\n")
    try:
        result = subprocess.run(sandbox.wrap([sys.executable, "-c", child, "run/s.sock", "elsewhere/s.sock"],
                                             ports=[], socket_roots=[tmp_path / "run"]),
                                capture_output=True, text=True, timeout=20, cwd=tmp_path)
    finally:
        for server in servers:
            server.close()
    assert result.stdout.split() == ["ok", "denied"]


@pytest.mark.skipif(sys.platform == "darwin", reason="non-macOS contract")
def test_unavailable_off_macos():
    assert sandbox.available() is False
    with pytest.raises(RuntimeError, match="macOS sandbox-exec is unavailable"):
        sandbox.positive_control("http://127.0.0.1:1/")


@pytest.mark.parametrize("root", ["/", "/private", "/private/var/run", "/private/tmp/some-gateway", "/var/run",
                                  "~", "~/.hermes", "~/.hermes/profiles", "/private/var/folders"])
def test_socket_roots_that_reach_system_sockets_are_refused(root):
    with pytest.raises(ValueError):
        sandbox.profile([], [Path(root).expanduser()])
