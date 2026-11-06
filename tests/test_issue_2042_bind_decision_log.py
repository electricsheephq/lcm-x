# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""#2042 step 1: engine-direct decision/outcome diagnostics never carry content."""

from contextlib import contextmanager
import logging
import re
from types import SimpleNamespace

from hermes_lcm import aux_session as auxiliary_module
from hermes_lcm import engine as engine_module
import sqlite3

import pytest

from hermes_lcm.aux_session import record_subagent_start, record_subagent_stop
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_auxiliary_first_ingest import _FakeAgent
from tests.test_issue_1931_delegate_children import Agent, DELEGATE, host  # noqa: F401


SENTINEL = "PRIVATE_2042_CONTENT_7f36"
PATTERN_SENTINEL = "OPERATOR_2042_PATTERN_9ac2"
HISTORY = [{"role": "user", "content": f"question {SENTINEL}"},
           {"role": "assistant", "content": f"answer {SENTINEL}"}]


@pytest.fixture
def engines(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(auxiliary_module, "_DELEGATE_AUXILIARY_IDS", set())
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    opened = []

    def make(**kwargs):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"store-{len(opened)}.db"),
                                           embeddings_enabled=False, **kwargs),
                           hermes_home=str(tmp_path / "home"))
        opened.append(engine)
        return engine

    yield make
    for engine in reversed(opened):
        engine.shutdown()


def assert_safe_logs(caplog):
    records = [record for phase in ("setup", "call") for record in caplog.get_records(phase)]
    assert all(SENTINEL not in record.getMessage() for record in records)
    prefixes = ("LCM session decision:", "LCM session first store:", "LCM session skipped:")
    assert all(PATTERN_SENTINEL not in record.getMessage() for record in records
               if record.getMessage().startswith(prefixes))


def lines(caplog, prefix="LCM session decision:", session=None):
    return [record.getMessage() for record in caplog.records
            if record.getMessage().startswith(prefix)
            and (session is None or f"session={session} " in record.getMessage())]


def test_plain_bind_first_store_and_rebind_are_deduplicated(engines, caplog):
    engine = engines()
    engine.on_session_start("plain", platform="desktop")
    assert lines(caplog) == [
        "LCM session decision: session=plain platform=desktop parent=- decision=store reason=default "
        f"engine={engine._diagnostic_engine_token}"]
    engine.ingest(HISTORY)
    assert lines(caplog, "LCM session first store:") == ["LCM session first store: session=plain rows=2"]
    engine.ingest([*HISTORY, {"role": "user", "content": f"next {SENTINEL}"}])
    engine.on_session_start("plain", platform="desktop")
    assert len(lines(caplog)) == len(lines(caplog, "LCM session first store:")) == 1
    assert engine._store.get_session_count("plain") == 3
    assert_safe_logs(caplog)


@pytest.mark.parametrize("decision,reason,config_key", [
    ("ignored", "ignore_pattern", "ignore_session_patterns"),
    ("stateless", "stateless_pattern", "stateless_session_patterns"),
])
def test_filtered_bind_and_first_skipped_ingest(engines, caplog, decision, reason, config_key):
    engine = engines(**{config_key: ["filtered", f"unmatched-{PATTERN_SENTINEL}"]},
                     ignore_message_patterns=[PATTERN_SENTINEL])
    # Exercise the real matching wrapper even on minimal installs without regex.
    pattern = re.compile(PATTERN_SENTINEL)
    engine._compiled_ignore_message_patterns = [SimpleNamespace(
        pattern=PATTERN_SENTINEL, search=lambda text, timeout: pattern.search(text))]
    matched = {"role": "user", "content": f"{SENTINEL} {PATTERN_SENTINEL}"}
    assert engine._matches_ignore_message_patterns(matched)
    engine.on_session_start("filtered", platform="desktop")
    for _ in range(2):
        engine.ingest([*HISTORY, matched])
    engine.on_session_start("filtered", platform="desktop")
    assert len(lines(caplog)) == 1
    assert f"decision={decision} reason={reason}" in lines(caplog)[0]
    assert lines(caplog, "LCM session skipped:") == [f"LCM session skipped: session=filtered reason={reason}"]
    assert lines(caplog, "LCM session first store:") == []
    assert engine._store.get_session_count("filtered") == 0
    assert_safe_logs(caplog)


def test_first_skipped_and_first_store_are_separate_outcomes(engines, caplog):
    engine = engines(ignore_session_patterns=["cron:same", f"unmatched-{PATTERN_SENTINEL}"])
    engine.on_session_start("same", platform="cron")
    engine.ingest(HISTORY)
    engine.on_session_start("same", platform="desktop")
    engine.ingest(HISTORY)
    assert [line.split("decision=", 1)[1].split(" engine=")[0] for line in lines(caplog)] == [
        "ignored reason=ignore_pattern", "store reason=default"]
    assert len(lines(caplog, "LCM session skipped:")) == len(lines(caplog, "LCM session first store:")) == 1
    assert_safe_logs(caplog)


class Child(Agent):
    def ingest(self, engine):
        engine.ingest(HISTORY)

    def store(self, engine):
        return engine._store_delegate_child_turn(self.session_id, HISTORY, platform="subagent")


@pytest.mark.parametrize("markers,reason", [
    ({"enabled_toolsets": ["memory", "skills"]}, "toolsets_memory_only"),
    ({"_memory_write_origin": "background_review"}, "background_review"),
])
def test_auxiliary_bind_and_skipped_ingest(engines, caplog, markers, reason):
    engine = engines()
    engine.on_session_start("parent", platform="desktop")
    child = Child("aux", **markers)
    child.start(engine)
    child.ingest(engine)
    child.ingest(engine)
    child.start(engine)
    assert len(lines(caplog, session="aux")) == 1
    assert f"decision=auxiliary reason={reason}" in lines(caplog, session="aux")[0]
    assert "platform=subagent parent=parent" in lines(caplog, session="aux")[0]
    assert lines(caplog, "LCM session skipped:") == [f"LCM session skipped: session=aux reason={reason}"]
    assert engine._session_id == "parent"
    assert engine._store.get_session_count("aux") == 0
    assert_safe_logs(caplog)


def test_late_auxiliary_classification_logs_new_decision(engines, caplog):
    engine = engines()
    agent = _FakeAgent(engine, "late", "parent")
    agent.start_session()
    agent.ingest_history_after_background_marker(HISTORY)
    agent.ingest_history_after_background_marker(HISTORY)
    assert len(lines(caplog, session="late")) == 2
    assert "decision=store reason=default" in lines(caplog, session="late")[0]
    assert "decision=auxiliary reason=background_review" in lines(caplog, session="late")[1]
    assert lines(caplog, "LCM session skipped:") == ["LCM session skipped: session=late reason=background_review"]
    assert engine._store.get_session_count("late") == 0
    assert_safe_logs(caplog)


def test_delegated_child_bind_store_and_replay(engines, caplog):
    engine = engines()
    engine.on_session_start("parent", platform="desktop")
    child = Child("child", **DELEGATE)
    child.start(engine)
    child.ingest(engine)
    assert child.store(engine) == 2
    assert child.store(engine) == 0
    assert len(lines(caplog, session="child")) == 1
    assert "platform=subagent parent=parent decision=delegate_child reason=frame_marker" in lines(caplog, session="child")[0]
    assert lines(caplog, "LCM session first store:") == ["LCM session first store: session=child rows=2"]
    assert engine._session_id == "parent"
    assert engine._store.get_session_count("child") == 2
    assert_safe_logs(caplog)


@pytest.mark.parametrize("signal,reason", [("hook", "subagent_start"), ("state", "state_db_subagent")])
def test_delegate_store_without_bind_names_lineage_rule(engines, tmp_path, caplog, signal, reason):
    engine = engines()
    engine.on_session_start("parent", platform="desktop")
    if signal == "hook":
        record_subagent_start({"child_session_id": "unbound-child", "parent_session_id": "parent"})
    else:
        path = tmp_path / "home" / "state.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('unbound-child', 'subagent', 'parent')")
    try:
        assert engine._store_delegate_child_turn("unbound-child", HISTORY, platform="subagent") == 2
        assert engine._store_delegate_child_turn("unbound-child", HISTORY, platform="subagent") == 0
        assert len(lines(caplog, session="unbound-child")) == 1
        assert f"decision=delegate_child reason={reason}" in lines(caplog, session="unbound-child")[0]
        assert lines(caplog, "LCM session first store:") == ["LCM session first store: session=unbound-child rows=2"]
    finally:
        record_subagent_stop({"child_session_id": "unbound-child"})
    assert_safe_logs(caplog)


def test_off_current_s3_store_logs_its_session(engines, caplog):
    engine = engines()
    engine.on_session_start("ended", platform="desktop")
    engine.on_session_start("current", platform="desktop")
    engine.on_session_end("ended", HISTORY)
    assert len(lines(caplog, session="ended")) == 2
    assert "decision=store reason=off_current_s3" in lines(caplog, session="ended")[-1]
    assert lines(caplog, "LCM session first store:") == ["LCM session first store: session=ended rows=2"]
    assert engine._session_id == "current"
    assert_safe_logs(caplog)


def test_each_engine_logs_its_own_decision(engines, caplog):
    first, second = engines(), engines()
    for engine in (first, second):
        engine.on_session_start("same", platform="desktop")
        engine.ingest(HISTORY)
    assert len(lines(caplog)) == len(lines(caplog, "LCM session first store:")) == 2
    assert first._diagnostic_engine_token != second._diagnostic_engine_token
    assert_safe_logs(caplog)


def test_failed_ingest_does_not_claim_first_store(engines, caplog, monkeypatch):
    engine = engines()
    engine.on_session_start("failure", platform="desktop")

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic diagnostic test failure")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages_transaction", fail)
        engine.ingest(HISTORY)
    assert lines(caplog, "LCM session first store:") == []
    assert engine._store.get_session_count("failure") == 0
    engine.ingest(HISTORY)
    assert lines(caplog, "LCM session first store:") == ["LCM session first store: session=failure rows=2"]
    assert_safe_logs(caplog)


def test_resident_engine_conflict_skip_deduplicates_by_engine_and_session(host, caplog):
    caplog.set_level(logging.INFO)
    load, clone = host
    for _ in range(2):
        ctx, _ = load()
        parent = clone(ctx)
        parent.on_session_start("parent", platform="desktop")
        for session_id in ("unbound", "unbound", "other"):
            # No child markers: a resident parent prevents a cold bind and child storage.
            Agent(session_id).turn(ctx, HISTORY)
        assert ctx.offered._session_id == ""
        assert parent._session_id == "parent"
        assert parent._store.get_session_count("unbound") == 0
        assert parent._store.get_session_count("other") == 0
    assert lines(caplog, "LCM session skipped:") == [
        "LCM session skipped: session=unbound reason=resident_engine_conflict",
        "LCM session skipped: session=other reason=resident_engine_conflict",
    ] * 2
    assert sum("skipped ingest: stable engine use ended with resident_engine_conflict" in record.getMessage()
               for record in caplog.records) == 6
    assert_safe_logs(caplog)


class BadDiagnosticAgent:
    enabled_toolsets = ["memory", "skills"]

    @property
    def _parent_session_id(self):
        raise RuntimeError("diagnostic input failed")


def test_auxiliary_logging_input_failure_does_not_break_bind(engines, caplog, monkeypatch):
    engine = engines()
    engine.on_session_start("parent", platform="desktop")
    monkeypatch.setattr(engine, "_delegate_child_agent", lambda sid: BadDiagnosticAgent())
    child = Child("bad-bind", enabled_toolsets=["memory", "skills"])
    child.start(engine)
    child.ingest(engine)
    assert engine._session_id == "parent"
    assert engine._thread_context_has_auxiliary_session("bad-bind")
    assert engine._store.get_session_count("bad-bind") == 0
    assert_safe_logs(caplog)


def test_auxiliary_logging_input_failure_still_restores_foreground(engines, caplog, monkeypatch):
    engine = engines()
    foreground = _FakeAgent(engine, "parent")
    foreground.start_session()
    foreground.ingest_history_as_foreground(HISTORY)
    child = _FakeAgent(engine, "late", "parent")
    child.start_session()
    monkeypatch.setattr(engine, "_delegate_child_agent", lambda sid: BadDiagnosticAgent())
    child.ingest_history_after_background_marker(HISTORY)
    assert engine.current_session_id == "parent"
    assert engine._store.get_session_count("parent") == 2
    assert engine._store.get_session_count("late") == 0
    assert_safe_logs(caplog)


def test_bypassed_logging_input_failure_does_not_break_ingest(engines, caplog, monkeypatch):
    engine = engines(ignore_session_patterns=["ignored"])
    engine.on_session_start("ignored")

    original = engine._bypass_lcm_session_id
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        if calls == 1:  # Only the diagnostic lookup fails; prefix tracking stays real.
            raise RuntimeError("diagnostic session lookup failed")
        return original()

    monkeypatch.setattr(engine, "_bypass_lcm_session_id", fail)
    engine.ingest(HISTORY)
    assert engine._session_ignored
    assert engine._store.get_session_count("ignored") == 0
    assert_safe_logs(caplog)


def test_diagnostic_dedupe_is_bounded_and_evicts_oldest(engines, caplog):
    engine = engines()
    limit = engine_module._SESSION_DIAGNOSTIC_LOG_MAX
    for index in range(limit + 1):
        if index % 2:
            engine._log_session_ingest_outcome(f"s{index}", rows=1)
        else:
            engine._log_session_decision(f"s{index}", "store", "default")
    assert len(engine._session_diagnostic_logs) == limit
    assert all(key[1] != "s0" for key in engine._session_diagnostic_logs)
    assert next(iter(engine._session_diagnostic_logs))[1] == "s1"
    engine._log_session_ingest_outcome("s1", rows=1)
    assert len(lines(caplog, "LCM session first store:", "s1")) == 1
    engine._log_session_decision("s0", "store", "default")
    assert len(lines(caplog, session="s0")) == 2
    assert len(engine._session_diagnostic_logs) == limit
    assert all(key[1] != "s1" for key in engine._session_diagnostic_logs)
    assert_safe_logs(caplog)


@pytest.mark.parametrize("hook", [False, True])
def test_no_frame_live_auxiliary_bind_names_actual_decision(engines, caplog, hook):
    engine = engines()
    engine.on_session_start("parent", platform="desktop")
    engine._register_auxiliary_session("aux-parent")
    if hook:
        record_subagent_start({"child_session_id": "no-frame", "parent_session_id": "aux-parent"})
    try:
        engine.on_session_start("no-frame", platform="subagent", parent_session_id="aux-parent")
        assert f"decision=auxiliary reason={'subagent_start' if hook else 'live_aux_lineage'}" in lines(
            caplog, session="no-frame")[0]
        assert engine._session_id == "parent"
        assert engine._store.get_session_count("no-frame") == 0
    finally:
        record_subagent_stop({"child_session_id": "no-frame"})
    assert_safe_logs(caplog)


def test_no_frame_bypassed_ingest_names_inherited_rule(engines, caplog):
    engine = engines()
    engine.on_session_start("parent")
    engine._mark_thread_context_stateless("no-frame")
    engine.ingest(HISTORY)
    assert "decision=auxiliary reason=inherited_bypass" in lines(caplog, session="no-frame")[0]
    assert lines(caplog, "LCM session skipped:") == [
        "LCM session skipped: session=no-frame reason=inherited_bypass"]
    assert engine._session_id == "parent"
    assert engine._store.get_session_count("no-frame") == 0
    assert_safe_logs(caplog)


@pytest.mark.parametrize("late", [False, True])
def test_stale_auxiliary_compression_logs_new_session(engines, caplog, monkeypatch, late):
    engine = engines()
    engine.on_session_start("parent")
    calls = 0

    def generation(sid):
        nonlocal calls
        calls += 1
        return 7 if sid == "old-aux" and (not late or calls > 2) else 0

    engine._auxiliary_retired_session_generations["old-aux"] = {7}
    monkeypatch.setattr(engine, "_in_process_auxiliary_caller_generation", generation)
    engine.on_session_start("new-aux", platform="subagent", boundary_reason="compression", old_session_id="old-aux")
    assert "parent=old-aux decision=bypass reason=stale_auxiliary_compression" in lines(caplog, session="new-aux")[0]
    assert engine._session_id == "parent"
    assert engine._store.get_session_count("new-aux") == 0
    assert_safe_logs(caplog)


@pytest.mark.parametrize("shape,reason", [
    ("auxiliary", "auxiliary"), ("remembered", "auxiliary_lineage"),
    ("require_agent", "no_agent"), ("filtered", "bypass_pattern"),
    ("unproven", None), ("no_parent", "no_parent"),
    ("self_parent", "no_parent"), ("ignored_parent", "ignored_parent"),
    ("filtered_parent", "parent_bypass_pattern"),
])
def test_each_delegate_link_rejection_logs_skip(engines, caplog, monkeypatch, shape, reason):
    engine = engines(ignore_session_patterns=["filtered", "blocked-parent"])
    engine.on_session_start("parent")
    session = "filtered" if shape == "filtered" else "rejected-child"
    parent = "blocked-parent" if shape == "filtered_parent" else "" if shape == "no_parent" else (
        session if shape == "self_parent" else "parent")
    if shape == "auxiliary":
        monkeypatch.setattr(engine, "_delegate_child_agent", lambda sid: Child(sid, enabled_toolsets=["memory"]))
    elif shape == "remembered":
        key = engine._delegate_child_key(session)
        monkeypatch.setattr(auxiliary_module, "_DELEGATE_AUXILIARY_IDS", {key})
    if shape == "ignored_parent":
        engine._session_ignored = True
    hook = shape in {"no_parent", "self_parent", "ignored_parent", "filtered_parent"}
    if hook:
        record_subagent_start({"child_session_id": session, "parent_session_id": parent})
    try:
        for _ in range(2):
            assert engine._delegate_child_link(session, require_agent=shape == "require_agent") is None
        # An unproven child is not a skip: the caller still picks the session's route.
        assert lines(caplog, "LCM session skipped:", session) == (
            [] if reason is None else [f"LCM session skipped: session={session} reason={reason}"])
        assert engine._store.get_session_count(session) == 0
        assert engine._session_id == "parent"
    finally:
        record_subagent_stop({"child_session_id": session})
    assert_safe_logs(caplog)


def test_off_current_bypass_lineage_end_logs_skip(engines, caplog):
    engine = engines()
    engine.on_session_start("parent")
    engine._mark_lcm_bypass_lineage_session("bypassed", platform="subagent")
    engine.on_session_end("bypassed", HISTORY)
    assert lines(caplog, "LCM session skipped:", "bypassed") == [
        "LCM session skipped: session=bypassed reason=bypass_lineage_end"]
    assert engine._store.get_session_count("bypassed") == 0
    assert engine._session_id == "parent"
    assert_safe_logs(caplog)


def test_resident_conflict_delegate_store_does_not_log_conflict_skip(host, caplog):
    caplog.set_level(logging.INFO)
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent")
    Child("stored-child", **DELEGATE).turn(ctx, HISTORY)
    assert parent._store.get_session_count("stored-child") == 2
    assert lines(caplog, "LCM session skipped:", "stored-child") == []
    assert_safe_logs(caplog)


@pytest.mark.parametrize("early,fail_commit", [(False, False), (True, False), (False, True)])
def test_cold_tail_first_store_is_after_commit(engines, caplog, monkeypatch, early, fail_commit):
    engine = engines()
    engine.on_session_start("parent")
    store = engine._store
    atomic = store.ingest_atomic
    committed = False

    @contextmanager
    def transaction():
        nonlocal committed
        with atomic():
            yield
            assert lines(caplog, "LCM session first store:", "cold") == []
            if fail_commit:
                raise sqlite3.OperationalError("synthetic commit failure")
        committed = True

    monkeypatch.setattr(store, "ingest_atomic", transaction)
    monkeypatch.setattr(engine, "_cold_tail_stored", lambda *a, **kw: 0)
    monkeypatch.setattr(engine, "_cold_tail_one_conflict", lambda *a, **kw: HISTORY[0] if early else None)
    monkeypatch.setattr(engine, "_cold_tail_conflict_placed", lambda *a, **kw: True)
    tail = {"view": HISTORY, "session_id": "cold", "store": store, "source": "desktop", "conversation_id": "cold"}
    if fail_commit:
        with pytest.raises(sqlite3.OperationalError, match="synthetic commit failure"):
            engine._retry_cold_tail_s3(tail)
        assert lines(caplog, "LCM session first store:", "cold") == []
        assert store.get_session_count("cold") == 0
    else:
        assert engine._retry_cold_tail_s3(tail)
        assert committed
        assert lines(caplog, "LCM session first store:", "cold") == ["LCM session first store: session=cold rows=2"]
        assert store.get_session_count("cold") == 2
    assert_safe_logs(caplog)


def test_diagnostics_log_outside_auxiliary_lock(engines, caplog, monkeypatch):
    engine = engines()
    original = engine_module.logger.info

    def info(*args, **kwargs):
        assert not engine._auxiliary_session_lock._is_owned()
        original(*args, **kwargs)

    monkeypatch.setattr(engine_module.logger, "info", info)
    engine._log_session_decision("unlocked", "store", "default")
    engine._log_session_ingest_outcome("unlocked", rows=1)
    assert len(lines(caplog, session="unlocked")) == len(lines(caplog, "LCM session first store:")) == 1
    assert_safe_logs(caplog)
