"""D1/D1.1: doctor facts, background recovery and concise agent disclosure."""
# ruff: noqa: F811 -- imported fixture is injected by pytest
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance, tools
from hermes_lcm.command_doctor import collect_doctor, render_doctor
from hermes_lcm.config import LCMConfig, companion_enable_command
from hermes_lcm.diagnostics import doctor_guidance_for_check
from tests.test_issue_1186_r3 import default_engine  # noqa: F401

REAL_PREPARE = maintenance.start_local_model_prepare
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("empty", [("HERMES_HOME",), ("HERMES_PROFILE",), ("HERMES_HOME", "HERMES_PROFILE")])
def test_empty_target_matches_installer_printf(tmp_path, monkeypatch, empty):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes root"))
    monkeypatch.setenv("HERMES_PROFILE", "work")
    for name in empty:
        monkeypatch.setenv(name, "")
    # No installer dry mode: execute its target selection and actual printf,
    # stopping before any filesystem or host action.
    script = (ROOT / "scripts/install.sh").read_text()
    selection = script.split('HERMES_HOME_DIR=', 1)[1].split('if [[ "$INSTALL_PROFILE" != default ]]', 1)[0]
    result = subprocess.run(["bash", "-c", 'set -eu\nHERMES_HOME_DIR=' + selection +
        '\nprintf "%s plugins enable hermes-lcm-x-local-embeddings\\n" "$HERMES_COMMAND"'],
        env=os.environ.copy(), check=True, text=True, capture_output=True)
    check = {"check": "embedding_provider_health", "status": "warn", "detail": {"available": False}}
    guidance = doctor_guidance_for_check(check)["operator_action"]
    actual = companion_enable_command()
    assert shlex.split(actual) == shlex.split(result.stdout.strip())
    assert f"`{actual}`" in guidance


def probe(monkeypatch, state="present", available=True, reason="configured"):
    monkeypatch.setattr(tools, "effective_cpus", lambda: 8)
    monkeypatch.setattr(maintenance, "local_model_state", lambda config: (state, "offline" if state == "unavailable" else ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": "test", "available": available, "detail": reason})


def test_loaded_model_coverage_error_is_a_separate_issue(default_engine, monkeypatch):
    probe(monkeypatch)
    db = maintenance.database_identity(default_engine._store.db_path)
    maintenance._LAST_PASS[db] = (0, "complete")
    def unreadable(*args):
        raise sqlite3.OperationalError("synthetic coverage read failure")
    monkeypatch.setattr(command, "_embedding_read_connection", unreadable)
    check = tools._embedding_provider_health_check(default_engine)
    assert "model: present; coverage: unavailable (synthetic coverage read failure)" in check["detail"]["readiness"]
    assert not check["detail"].get("informational")
    text = render_doctor(collect_doctor(default_engine))
    assert "status: issues-found" in text
    assert "Model ready" in text and "synthetic coverage read failure" in text
    line = next(line for line in text.splitlines() if line.startswith("embedding_readiness:"))
    assert line.count("summaries") == 1 and "maintenance" not in line
    probe(monkeypatch, "preparing")
    text = render_doctor(collect_doctor(default_engine))
    assert "Local semantic recall preparing; full-text recall is active" in text
    assert "synthetic coverage read failure" in text
    probe(monkeypatch, "unavailable")
    text = render_doctor(collect_doctor(default_engine))
    assert "Local semantic recall unavailable (offline); full-text recall is active" in text
    assert "synthetic coverage read failure" in text
    probe(monkeypatch, "unavailable", available=False, reason="FastEmbed is not installed")
    text = render_doctor(collect_doctor(default_engine))
    assert "Local semantic recall unavailable (FastEmbed is not installed); full-text recall is active" in text
    assert "synthetic coverage read failure" in text and companion_enable_command() in text


def test_text_uses_three_spec_states(default_engine, monkeypatch):
    probe(monkeypatch, "preparing")
    assert "embedding_readiness: Local semantic recall preparing; full-text recall is active\n" in render_doctor(collect_doctor(default_engine))
    probe(monkeypatch, available=False, reason="FastEmbed is not installed")
    text = render_doctor(collect_doctor(default_engine))
    assert "Local semantic recall unavailable (FastEmbed is not installed); full-text recall is active" in text
    assert companion_enable_command() in text
    probe(monkeypatch)
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 1/1 summaries; chunks: caught up (coverage certified); maintenance: complete")
    text = render_doctor(collect_doctor(default_engine))
    line = next(line for line in text.splitlines() if line.startswith("embedding_readiness:"))
    assert line == "embedding_readiness: Model ready; summaries 1/1; chunks caught up (coverage certified)"


def test_catchup_failing_stalled_informational_and_complete(default_engine, monkeypatch):
    probe(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(tools.time, "time", lambda: clock[0])
    # Use the existing missing-profile receipt path plus real maintenance facts.
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 0/1 summaries; chunks: catching up")
    db = maintenance.database_identity(default_engine._store.db_path)
    maintenance._LAST_PASS[db] = (clock[0], "error")
    maintenance._FAILURES[db] = 1
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "warn" and not check["detail"].get("informational")
    assert "failing" in check["detail"]["reason"]
    assert "status: issues-found" in render_doctor(collect_doctor(default_engine))
    maintenance._FAILURES.pop(db)
    maintenance._LAST_PASS[db] = (clock[0], "summaries=complete:1")
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["informational"]
    assert "status: issues-found" not in render_doctor(collect_doctor(default_engine))
    clock[0] += command._embedding_backfill_lease_ttl_s() + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "warn" and "stalled" in check["detail"]["reason"]
    assert "status: issues-found" in render_doctor(collect_doctor(default_engine))
    maintenance._LAST_PASS.pop(db)
    current_profile = command._embedding_current_profile
    monkeypatch.setattr(command, "_embedding_current_profile", lambda conn: {
        "identity_hash": "fixture", "provider": "fastembed", "model_name": "test", "registered_at": 1000.0})
    command._record_embedding_catchup(default_engine._store.db_path, "summary", "fixture", 1)
    clock[0] += command._embedding_backfill_lease_ttl_s() + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert "stalled" in check["detail"]["reason"]  # persisted receipt after process restart
    # Registration is not a catch-up attempt, so its ISO timestamp cannot
    # date a new backlog before the first catch-up receipt.
    from hermes_lcm.vector_store import VectorStore
    vectors = VectorStore(default_engine._store.db_path, config=default_engine._config)
    try:
        vectors.register_profile("test", "fastembed", 2)
    finally:
        vectors.close()
    monkeypatch.setattr(command, "_embedding_current_profile", current_profile)
    conn = command._embedding_read_connection(default_engine._store.db_path)
    try:
        profile = current_profile(conn)
    finally:
        conn.close()
    assert isinstance(profile["registered_at"], str)
    clock[0] = tools.datetime.fromisoformat(profile["registered_at"]).timestamp()
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["informational"]  # registered, before the first receipt
    clock[0] += command._embedding_backfill_lease_ttl_s() + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["informational"]
    # A completed receipt describes the old backlog, not a new leaf queued
    # after an idle period. Only a pending receipt starts its stall window.
    command._record_embedding_catchup(default_engine._store.db_path, "summary", profile["identity_hash"], 0)
    clock[0] += command._embedding_backfill_lease_ttl_s() + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["informational"]
    assert "status: issues-found" not in render_doctor(collect_doctor(default_engine))
    command._record_embedding_catchup(default_engine._store.db_path, "summary", profile["identity_hash"], 1)
    clock[0] += command._embedding_backfill_lease_ttl_s() + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "warn" and "stalled" in check["detail"]["reason"]
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 1/1 summaries; chunks: caught up (coverage certified)")
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "pass"
    assert "status: issues-found" not in render_doctor(collect_doctor(default_engine))


def test_exhausted_prepare_backoff_background_only_and_success(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    config = LCMConfig()
    key = maintenance._local_model_key(config)
    attempts = []
    succeeds = [False]
    class Model:
        def warmup(self):
            attempts.append(clock[0])
            if not succeeds[0]:
                raise RuntimeError("offline")
    monkeypatch.setattr(command, "resolve_provider", lambda c: Model())
    db = tmp_path / "lcm.db"
    for i in range(6):
        assert REAL_PREPARE(db, config)
        maintenance._PREPARE_THREAD.join(2)
        if i < 5:
            clock[0] += 600
    callbacks = []
    # Existing scheduler continuation must stay armed at exhaustion, without
    # loading a model in this call. Its recorded delay is the prepare backoff.
    monkeypatch.setattr(maintenance, "_run_pass", lambda *a: ("model_prepare_exhausted", None))
    result = maintenance.run_incremental_embedding_pass(db, config, slice_state={})
    assert result["continue"] and result["delay"] == 300
    for delay in (300, 600, 1200, 1800, 1800):
        retry_at = maintenance._PREPARE_RETRY[key][1]
        assert retry_at == clock[0] + delay
        clock[0] = retry_at - 1
        assert not REAL_PREPARE(db, config, on_ready=lambda: callbacks.append("ready"), owner="owner")
        clock[0] += 1
        assert not REAL_PREPARE(db, config)  # foreground callers cannot restart exhausted preparation
        assert REAL_PREPARE(db, config, on_ready=lambda: callbacks.append("ready"), owner="owner")
        maintenance._PREPARE_THREAD.join(2)
        assert not REAL_PREPARE(db, config, on_ready=lambda: None, owner="owner")
    clock[0] = maintenance._PREPARE_RETRY[key][1]
    succeeds[0] = True
    assert REAL_PREPARE(db, config, on_ready=lambda: callbacks.append("ready"), owner="owner")
    maintenance._PREPARE_THREAD.join(2)
    assert len(attempts) == 12
    assert maintenance.local_model_state(config) == ("present", "")
    assert key not in maintenance._PREPARE_RETRY and key not in maintenance._PREPARE_READY
    assert callbacks == ["ready"]


def test_agent_reason_omits_operator_details_and_empty_suffix(default_engine, monkeypatch):
    probe(monkeypatch, "preparing")
    payload = json.loads(tools.lcm_recall({"query": "fixture"}, engine=default_engine))
    assert payload["degraded_reason"] == "local semantic recall preparing; using full-text recall"
    probe(monkeypatch, available=False, reason="missing companion at /synthetic/private; run hermes plugins enable companion")
    payload = json.loads(tools.lcm_recall({"query": "fixture"}, engine=default_engine))
    assert payload["degraded_reason"] == "local semantic recall unavailable; using full-text recall"
    with pytest.raises(RuntimeError, match="^local semantic recall preparing; using full-text recall$"):
        probe(monkeypatch, "preparing")
        tools._lcm_grep_resolve_provider(default_engine)


def test_live_catchup_lease_prevents_false_stall(default_engine, monkeypatch):
    probe(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(tools.time, "time", lambda: clock[0])
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 0/1 summaries; chunks: catching up")
    monkeypatch.setattr(command, "_embedding_current_profile", lambda conn: {
        "identity_hash": "fixture", "provider": "fastembed", "model_name": "test"})
    command._record_embedding_catchup(default_engine._store.db_path, "summary", "fixture", 1)
    ttl = command._embedding_backfill_lease_ttl_s()
    conn = default_engine._dag.connection
    lease = command._acquire_embedding_backfill_lease(conn, ttl_s=ttl, heartbeat_s=60)
    assert lease is not None
    clock[0] += ttl + 1
    assert lease.renew(now=clock[0])
    conn.commit()
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["informational"]
    assert "status: issues-found" not in render_doctor(collect_doctor(default_engine))
    clock[0] += ttl + 1
    check = tools._embedding_provider_health_check(default_engine)
    assert not check["detail"]["informational"] and "stalled" in check["detail"]["reason"]
    assert "status: issues-found" in render_doctor(collect_doctor(default_engine))
    maintenance._record(default_engine._store.db_path, "summaries=error", "synthetic catch-up disk error")
    assert lease.renew(now=clock[0])
    conn.commit()
    check = tools._embedding_provider_health_check(default_engine)
    assert not check["detail"]["informational"] and "failing" in check["detail"]["reason"]
    assert "synthetic catch-up disk error" in doctor_guidance_for_check(check)["operator_action"]


@pytest.mark.parametrize("cpus", [2, 8])
@pytest.mark.parametrize("reason", ["summary catch-up stalled (no recent catch-up progress)",
    "summary catch-up failing (last catch-up attempt recorded an error: synthetic disk error)"])
def test_stall_guidance_is_actionable_on_every_host(cpus, reason):
    check = {"check": "embedding_provider_health", "status": "warn", "detail": {
        "available": True, "cpu_limited": cpus <= 2, "reason": reason,
        "readiness": "model: present; coverage: 0/1 summaries; chunks: catching up"}}
    guidance = doctor_guidance_for_check(check)
    assert guidance["action"] == "inspect"
    assert reason in guidance["operator_action"]
    assert "`/lcm embed backfill --apply`" in guidance["operator_action"]
    assert "`/lcm doctor`" in guidance["operator_action"]
    assert "companion" not in guidance["operator_action"]
    assert "CPU-limited as expected" not in guidance["rationale"]


@pytest.mark.parametrize("raw", ["{", "[]", "null", '{"updated_at": []}',
    '{"updated_at": null}', '{"updated_at": "not-a-time"}'])
def test_malformed_catchup_receipt_falls_back_to_maintenance(default_engine, monkeypatch, raw):
    probe(monkeypatch)
    monkeypatch.setattr(tools.time, "time", lambda: 2000.0)
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 0/1 summaries; chunks: catching up")
    monkeypatch.setattr(command, "_embedding_current_profile", lambda conn: {
        "identity_hash": "fixture", "provider": "fastembed", "model_name": "test"})
    conn = default_engine._dag.connection
    conn.execute("INSERT INTO metadata(key,value) VALUES(?,?)", ("embedding_catchup:summary:fixture", raw))
    conn.commit()
    db = maintenance.database_identity(default_engine._store.db_path)
    maintenance._LAST_PASS[db] = (1000.0, "summaries=complete:1")
    check = tools._embedding_provider_health_check(default_engine)
    assert check["status"] == "warn" and "stalled" in check["detail"]["reason"]
    text = render_doctor(collect_doctor(default_engine))
    assert "status: issues-found" in text and "embedding_provider_health: fail" not in text
    maintenance._LAST_PASS[db] = (2000.0, "summaries=complete:1")
    assert tools._embedding_provider_health_check(default_engine)["detail"]["informational"]


@pytest.mark.parametrize("state,available", [("preparing", True), ("unavailable", True), ("preparing", False)])
def test_grep_and_recall_share_short_readiness_reason(default_engine, monkeypatch, state, available):
    probe(monkeypatch, state, available=available)
    recall = json.loads(tools.lcm_recall({"query": "fixture"}, engine=default_engine))
    grep = json.loads(tools.lcm_grep({"query": "fixture", "mode": "semantic"}, engine=default_engine))
    expected = f"local semantic recall {state if available else 'unavailable'}; using full-text recall"
    assert grep["degraded_reason"] == recall["degraded_reason"] == expected


def test_real_companion_reason_is_short_in_readiness(default_engine, monkeypatch):
    from hermes_lcm import embedding_provider
    monkeypatch.setattr(embedding_provider.importlib.util, "find_spec", lambda name: None)
    real_probe = embedding_provider.probe_provider_availability(default_engine._config)
    assert "install the dependency companion" in real_probe["detail"]
    probe(monkeypatch, available=False, reason=real_probe["detail"])
    text = render_doctor(collect_doctor(default_engine))
    line = next(line for line in text.splitlines() if line.startswith("embedding_readiness:"))
    assert line == "embedding_readiness: Local semantic recall unavailable (FastEmbed is not installed); full-text recall is active"
    assert text.splitlines().count(f"embedding_enable_command: {companion_enable_command()}") == 1
    assert companion_enable_command() not in line


def test_expired_prepare_backoff_blocked_by_other_key_repolls_at_sixty_seconds(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    config = LCMConfig()
    key = maintenance._local_model_key(config)
    maintenance._PREPARE[key] = ("unavailable", "offline")
    maintenance._PREPARE_RETRY[key] = (6, clock[0] - 1)
    class OtherPrepare:
        def is_alive(self):
            return True
    with monkeypatch.context() as patch:
        patch.setattr(maintenance, "_PREPARE_THREAD", OtherPrepare())
        assert not REAL_PREPARE(tmp_path / "lcm.db", config, on_ready=lambda: None, owner="fixture")
        patch.setattr(maintenance, "_run_pass", lambda *args: ("model_prepare_exhausted", None))
        result = maintenance.run_incremental_embedding_pass(tmp_path / "lcm.db", config, slice_state={})
        assert result["continue"] and result["delay"] >= 60
