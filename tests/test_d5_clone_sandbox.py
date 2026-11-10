"""Real sandbox controls; never substitute a mock for the egress proof."""

import os
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


def test_profile_allows_only_named_ports_and_socket_roots():
    run_dir = Path.home() / "d5-clone-synthetic-run"  # Never created; outside every refused tree on macOS and Linux.
    text = sandbox.profile([8123], [run_dir])
    assert "(deny network-outbound)" in text
    assert '(remote ip "localhost:8123")' in text
    assert f'(subpath "{run_dir.resolve()}")' in text
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


@pytest.mark.parametrize("root", ["/", "/private", "/private/var/run", "/private/tmp/some-gateway", "/var/run", "/run",
                                  "/run/user", "/tmp/some-gateway", "/var/tmp",
                                  "~", "~/.hermes", "~/.hermes/profiles", "/private/var/folders"])
def test_socket_roots_that_reach_system_sockets_are_refused(root):
    with pytest.raises(ValueError):
        sandbox.profile([], [Path(root).expanduser()])


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS per-user temp dir")
def test_the_per_user_temp_dir_itself_is_refused_but_a_run_dir_inside_is_not():
    import subprocess as sp
    temp = Path(sp.run(["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True).stdout.strip())
    private = Path(os.path.realpath(temp))
    for root in (temp, temp.parent, private, Path("/System/Volumes/Data" + str(private)),
                 Path("/System/Volumes/Data/private/var/folders"), Path("/System/Volumes/Data" + str(Path.home()))):
        with pytest.raises(ValueError):
            sandbox.profile([], [root])
    assert "d5-run" in sandbox.profile([], [temp / "d5-run"])


@pytest.mark.parametrize("inside", ["/.hermes/profiles", "/.hermes"])
def test_a_firmlink_spelling_inside_a_refused_subtree_is_refused(inside):
    for root in ("/System/Volumes/Data" + str(Path.home().resolve()) + inside, "/System/Volumes/Data/private/var/run/x"):
        with pytest.raises(ValueError):
            sandbox.profile([], [root])


def test_a_denied_path_is_denied_in_every_spelling():
    text = sandbox.profile([], deny_paths=["/Users/someone/.hermes"])
    for spelling in ("/Users/someone/.hermes", "/System/Volumes/Data/Users/someone/.hermes"):
        assert f'(deny file-read* file-write* (subpath "{spelling}"))' in text
    with pytest.raises(ValueError):
        sandbox.profile([], deny_paths=['/tmp/a"b'])
    assert sandbox.hermes_home() == str(Path.home() / ".hermes")


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec required")
def test_a_child_cannot_read_a_denied_path_but_may_hash_its_name(tmp_path):
    """The replay edition only hashes --observed-state-db (a path under the live agent's home); under the deny,
    opening it fails, so a replay that opened it would error instead of reading it."""
    denied, allowed = tmp_path / "live", tmp_path / "other"
    for d in (denied, allowed):
        d.mkdir()
        (d / "state.db").write_bytes(b"synthetic")
    child = ("import hashlib, sys\n"
             "def go(p):\n"
             "    try:\n"
             "        open(p, 'rb').read()\n"
             "        return 'read'\n"
             "    except OSError:\n"
             "        return 'denied'\n"
             "print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:8], go(sys.argv[1]), go(sys.argv[2]))\n")
    result = subprocess.run(sandbox.wrap([sys.executable, "-c", child, str(denied / "state.db"), str(allowed / "state.db")],
                                         ports=[], deny_paths=[denied]),
                            capture_output=True, text=True, timeout=20, cwd=allowed)
    import hashlib
    assert result.stdout.split() == [hashlib.sha256(str(denied / "state.db").encode()).hexdigest()[:8], "denied", "read"]


def test_the_egress_control_child_never_sees_the_live_home(monkeypatch):
    """P5's child runs under the same profile as every host process, the live agent's home denied included."""
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"verdict": "PASS"}', stderr="")

    monkeypatch.setattr(sandbox, "available", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    assert sandbox.positive_control("http://127.0.0.1:9/v1/models") == {"verdict": "PASS"}
    (argv,) = seen
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert f'(deny file-read* file-write* (subpath "{os.path.abspath(sandbox.hermes_home())}"))' in argv[2]
