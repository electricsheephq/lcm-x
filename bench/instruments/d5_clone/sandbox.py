"""macOS egress restriction and an independently executed positive control.

The profile denies all outbound network except the named loopback ports and unix sockets under the named
directories. Other local services (other loopback ports, other unix sockets) are denied too, and so is the
system resolver's socket, so a host name lookup cannot leave the machine through the resolver.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import urlsplit


# Both spellings and their resolved forms, so the guard holds where /var/run or /tmp is a link (macOS, Linux).
_SYSTEM_SOCKETS = tuple(dict.fromkeys(d for raw in ("/private/var/run", "/private/tmp", "/private/var/tmp", "/var/run",
                                                   "/run", "/tmp", "/var/tmp")
                                      for d in (raw, os.path.realpath(raw), "/System/Volumes/Data" + raw)))


def _user_dirs():
    """macOS's per-user temp and cache dirs (the counterparts of /tmp, holding other apps' sockets), resolved."""
    if sys.platform != "darwin":
        return ()
    found = []
    for name in ("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR"):
        out = subprocess.run(["/usr/bin/getconf", name], capture_output=True, text=True).stdout.strip()
        if not out:
            raise ValueError("cannot resolve the per-user temp dirs")
        found.append(os.path.realpath(out))
    return tuple(found)


def profile(ports, socket_roots=()):
    lines = ["(version 1)", "(allow default)", "(deny network-outbound)"]
    for port in ports:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError("expected a TCP port number")
        lines.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
    for root in socket_roots:
        resolved = str(Path(root).resolve())
        if '"' in resolved or "\\" in resolved:
            raise ValueError("unsupported socket directory name")
        home = str(Path.home().resolve())
        subtrees = _SYSTEM_SOCKETS + (home + "/.hermes", os.path.realpath(home + "/.hermes"))  # Never inside these: other services' sockets live there.
        # Never these or above them; a dir made for the run INSIDE the per-user temp dir is fine.
        ancestors = subtrees + (home, "/private/var/folders") + _user_dirs()
        inside = any(resolved == d or resolved.startswith(d + "/") for d in subtrees)
        above = any(resolved == d or d.startswith(resolved.rstrip("/") + "/") for d in ancestors)
        if inside or above:
            raise ValueError("socket root would re-admit system sockets")
        lines.append(f'(allow network-outbound (remote unix-socket (subpath "{resolved}")))')
    return "\n".join(lines) + "\n"


def wrap(argv, *, ports, socket_roots=()):
    return ["/usr/bin/sandbox-exec", "-p", profile(ports, socket_roots)] + list(argv)


def available():
    return sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()


_CHILD = '''
import errno, json, socket, sys, urllib.request
def denied(address):
    try:
        with socket.create_connection(address, timeout=3):
            return False
    except OSError as exc:
        return exc.errno in (errno.EPERM, errno.EACCES)
blocked = denied(("1.1.1.1", 443))
other_blocked = denied(("127.0.0.1", int(sys.argv[2])))
dns_blocked = False
try:
    socket.getaddrinfo("example.com", 443)
except OSError:
    dns_blocked = True
local_ok = False
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(sys.argv[1], timeout=3) as response:
        local_ok = response.status == 200
except OSError:
    pass
ok = blocked and other_blocked and dns_blocked and local_ok
print(json.dumps({"external_blocked": blocked, "other_local_blocked": other_blocked, "dns_blocked": dns_blocked,
                  "local_ok": local_ok, "verdict": "PASS" if ok else "FAIL"}))
'''


def positive_control(local_url):
    """Under the profile that allows only *local_url*'s port: a public IP and a second local listener are
    denied, a public name does not resolve, and *local_url* answers."""
    if not available():
        raise RuntimeError("macOS sandbox-exec is unavailable")
    parts = urlsplit(local_url)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "::1") or not parts.port:
        raise ValueError("positive control requires a loopback URL with a port")
    other = socket.socket()
    try:
        other.bind(("127.0.0.1", 0))
        other.listen(1)
        argv = [sys.executable, "-c", _CHILD, local_url, str(other.getsockname()[1])]
        result = subprocess.run(wrap(argv, ports=[parts.port]), capture_output=True, text=True, timeout=20)
    finally:
        other.close()
    if result.returncode:
        raise RuntimeError("sandbox-exec child failed: " + result.stderr.strip())
    return json.loads(result.stdout)
