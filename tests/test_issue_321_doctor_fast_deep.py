"""Fast doctor discloses skipped scans; deep preserves exhaustive diagnostics."""

import json

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.db_bootstrap import _record_integrity_failed
from hermes_lcm.engine import LCMEngine
from hermes_lcm.schemas import LCM_DOCTOR
from hermes_lcm.store import build_message_fts_spec


DEEP_CHECKS = [
    "database_integrity", "messages_fts_integrity", "nodes_fts_integrity",
    "payload_storage", "source_lineage_hygiene",
]


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    hermes_home = tmp_path / "hermes_home"
    e = LCMEngine(config=config, hermes_home=str(hermes_home))
    e._session_id = "test-session"
    e._session_platform = "telegram"
    e.update_model(
        "gpt-test",
        200000,
        provider="openai-codex",
        api_mode="responses",
    )
    return e


def _spy_scans(monkeypatch, engine):
    calls = []

    def wrap(owner, name):
        original = getattr(owner, name)

        def spy(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)

    for name in (
        "check_external_content_fts_integrity", "scan_sqlite_payload_risks",
        "scan_externalized_payload_integrity", "externalized_payload_stats",
    ):
        wrap(lcm_tools, name)
    wrap(engine._store, "get_source_stats")
    return calls


def _traced_doctor(engine, args):
    sql = []
    engine._store.connection.set_trace_callback(sql.append)
    try:
        return json.loads(lcm_tools.lcm_doctor(args, engine=engine)), sql
    finally:
        engine._store.connection.set_trace_callback(None)


def test_default_is_fast_and_skips_exhaustive_checks(engine, monkeypatch):
    calls = _spy_scans(monkeypatch, engine)
    result, sql = _traced_doctor(engine, {})
    assert calls == []
    assert any("pragma quick_check" in statement.lower() for statement in sql)
    assert not any("pragma integrity_check" in statement.lower() for statement in sql)
    assert result["mode"] == "fast"
    assert result["checks_not_run"] == DEEP_CHECKS


def test_fast_reports_not_run_and_never_claims_full_integrity(engine):
    result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    checks = {check["check"]: check for check in result["checks"]}
    for name in DEEP_CHECKS:
        assert checks[name]["status"] == "not_run"
    for name in DEEP_CHECKS[:3]:
        assert checks[name]["status"] != "pass"
    assert checks["sqlite_storage"]["status"] == "pass"
    assert checks["sqlite_storage"]["detail"]["quick_check"] == "ok"
    assert result["note"]
    assert not any(item["check"] in DEEP_CHECKS for item in result["guidance"])
    assert result["overall"] == "healthy"


def test_explicit_fast_equals_default(engine):
    results = [json.loads(lcm_tools.lcm_doctor(args, engine=engine)) for args in (
        {}, {"mode": "fast"}, {"mode": " FAST "},
    )]
    signatures = [[(check["check"], check["status"]) for check in result["checks"]] for result in results]
    assert signatures[0] == signatures[1] == signatures[2]
    assert [name for name, status in signatures[0] if status == "not_run"] == DEEP_CHECKS


def test_deep_runs_full_scan_in_base_order(engine, monkeypatch):
    fast = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    calls = _spy_scans(monkeypatch, engine)
    result, sql = _traced_doctor(engine, {"mode": "deep"})
    assert calls.count("check_external_content_fts_integrity") == 2
    assert set(calls) == {
        "check_external_content_fts_integrity", "scan_sqlite_payload_risks",
        "scan_externalized_payload_integrity", "externalized_payload_stats", "get_source_stats",
    }
    assert any("pragma integrity_check" in statement.lower() for statement in sql)
    assert [check["check"] for check in result["checks"]] == [check["check"] for check in fast["checks"]]
    assert not any(check["status"] == "not_run" for check in result["checks"])
    assert result["mode"] == "deep"
    assert result["checks_not_run"] == []
    assert "note" not in result


def test_fast_keeps_background_fts_flag(engine):
    """Regression guard: persisted corruption remains visible without a scan."""
    conn = engine._store.connection
    _record_integrity_failed(conn, build_message_fts_spec(), detail="x")
    conn.commit()
    result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    checks = {check["check"]: check for check in result["checks"]}
    assert checks["messages_fts_integrity_background_flag"]["status"] == "fail"
    assert result["overall"] == "unhealthy"


def test_unknown_mode_is_rejected(engine, monkeypatch):
    calls = _spy_scans(monkeypatch, engine)
    result, sql = _traced_doctor(engine, {"mode": "full"})
    assert "unknown lcm_doctor mode" in result["error"]
    assert calls == []
    assert sql == []


def test_action_branch_ignores_mode(engine):
    """Regression guard: repair preview ignores diagnostic depth."""
    result = json.loads(lcm_tools.lcm_doctor({"action": "repair_level3", "mode": "bogus"}, engine=engine))
    assert result["action"] == "repair_level3"
    assert "error" not in result


def test_schema_exposes_mode_enum():
    assert LCM_DOCTOR["parameters"]["properties"]["mode"]["enum"] == ["fast", "deep"]
    assert LCM_DOCTOR["parameters"]["required"] == []


def test_fast_sqlite_storage_failure_is_reported_not_swallowed(engine, monkeypatch):
    conn = engine._store.connection

    class ConnectionProxy:
        def execute(self, sql, *args, **kwargs):
            if "pragma quick_check" in sql.lower():
                raise RuntimeError("quick_check failed")
            return conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(conn, name)

    proxy = ConnectionProxy()
    monkeypatch.setattr(type(engine._store), "connection", property(lambda self: proxy))
    result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    checks = {check["check"]: check for check in result["checks"]}
    assert checks["sqlite_storage"]["status"] == "fail"
    assert checks["sqlite_storage"]["detail"] == "quick_check failed"
    assert checks["payload_storage"]["status"] == "not_run"
    assert result["overall"] == "unhealthy"
