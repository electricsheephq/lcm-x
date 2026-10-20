"""Restore process-wide offline instrumentation after each isolated test."""
import socket
import subprocess

import pytest


@pytest.fixture(autouse=True)
def guard_state(monkeypatch):
    for owner, names in ((socket.socket, ("connect", "connect_ex", "sendto", "sendmsg")),
                         (socket, ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")),
                         (subprocess.Popen, ("__init__",))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, getattr(owner, name))
