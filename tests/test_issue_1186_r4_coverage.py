"""A preparing model must not hide a concurrent coverage-read failure."""
# ruff: noqa: F811 -- imported pytest fixture is injected by name
import sqlite3

from hermes_lcm import command, embedding_maintenance, tools
from hermes_lcm.diagnostics import doctor_guidance_for_check
from tests.test_issue_1186_r3 import default_engine  # noqa: F401


def test_preparing_with_coverage_error_keeps_coverage_remedy(default_engine, monkeypatch):
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: ("preparing", ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": "test", "available": True, "detail": "configured",
    })
    def unreadable(*args):
        raise sqlite3.OperationalError("synthetic coverage read failure")
    monkeypatch.setattr(command, "_embedding_read_connection", unreadable)
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "warn"
    assert "model: preparing; coverage: unavailable" in check["detail"]["readiness"]
    guidance = doctor_guidance_for_check(check)
    assert not any(marker in guidance["operator_action"] for marker in (
        "configured but not runnable", "hermes-lcm-x-local-embeddings",
    ))
    assert guidance["action"] == "inspect"
    assert "coverage" in guidance["operator_action"] and "SQLite" in guidance["operator_action"]
