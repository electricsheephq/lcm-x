"""Declare the host capability used by frozen non-UID control fingerprints."""
import pytest


@pytest.fixture(autouse=True)
def non_uid_host(monkeypatch):
    from hermes_lcm import host_uid_emit

    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", False)
