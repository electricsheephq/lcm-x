"""Restore process-wide offline instrumentation after each isolated test."""
import socket
import subprocess
import shutil

import pytest

from benchmarking.standalone import ensure_agent_context_engine_importable

ensure_agent_context_engine_importable()


@pytest.fixture(autouse=True)
def fake_store_disk_headroom(monkeypatch):
    # Tiny synthetic stores do not require the production instrument's 150 GiB.
    usage = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(path)._replace(free=200*2**30))


@pytest.fixture(autouse=True)
def guard_state(monkeypatch):
    for owner, names in ((socket.socket, ("connect", "connect_ex", "sendto", "sendmsg")),
                         (socket, ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")),
                         (subprocess.Popen, ("__init__",))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, getattr(owner, name))
