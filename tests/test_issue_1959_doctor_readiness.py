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
    # Real registration timestamps are ISO strings, unlike receipt timestamps.
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
