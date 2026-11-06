# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""#2042 step 1: engine-direct decision/outcome diagnostics never carry content."""

import logging
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
def engines(tmp_path, caplog):
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
                     ignore_message_patterns=[f"unmatched-{PATTERN_SENTINEL}"])
    engine.on_session_start("filtered", platform="desktop")
    for _ in range(2):
        engine.ingest(HISTORY)
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
    caplog.clear()
    engine.on_session_end("ended", HISTORY)
    assert len(lines(caplog, session="ended")) == 1
    assert "decision=store reason=off_current_s3" in lines(caplog, session="ended")[0]
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
