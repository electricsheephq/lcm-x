"""macOS egress restriction and an independently executed positive control.

Outbound traffic is allowed only to loopback and local unix sockets. The system resolver's socket is denied too,
so a host name lookup cannot leave the machine through the resolver; localhost still resolves locally.
"""

import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import ipaddress


PROFILE = '''(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))
'''


def wrap(argv):
    return ["/usr/bin/sandbox-exec", "-p", PROFILE] + list(argv)


def available():
    return sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()


def positive_control(local_url):
    if not available():
        raise RuntimeError("macOS sandbox-exec is unavailable")
    host = urlsplit(local_url).hostname
    if urlsplit(local_url).scheme != "http" or (
            host != "localhost" and not ipaddress.ip_address(host).is_loopback):
        raise ValueError("positive control requires a loopback URL")
    child = '''
import errno, json, socket, sys, urllib.request
blocked = False
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=3):
        pass
except OSError as exc:
    blocked = exc.errno in (errno.EPERM, errno.EACCES)
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
print(json.dumps({"external_blocked": blocked, "dns_blocked": dns_blocked, "local_ok": local_ok,
                  "verdict": "PASS" if blocked and dns_blocked and local_ok else "FAIL"}))
'''
    result = subprocess.run(wrap([sys.executable, "-c", child, local_url]),
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("sandbox-exec child failed: " + result.stderr.strip())
    return json.loads(result.stdout)
