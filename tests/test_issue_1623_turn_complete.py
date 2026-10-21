"""The host completion seam owns stopped turns without binding or replay state."""
from unittest.mock import Mock

import pytest
from agent.context_engine import ContextEngine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    # The unit host stub predates this optional callback; model the pinned hosts' no-op.
    monkeypatch.setattr(ContextEngine, "on_turn_complete", lambda *a, **k: None, raising=False)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("stopped-session", platform="acp", context_length=128000)
    yield engine
    engine.shutdown()


def stopped_rows():
    return [
        {"role": "user", "content": "read the synthetic file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-stop", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"fixture.txt"}'}}]},
        {"role": "tool", "content": "synthetic result", "tool_call_id": "call-stop"},
        {"role": "assistant", "content": "interrupted by user"},
    ]


def test_interrupted_callback_stores_rows_and_catchup_adds_no_duplicates(engine, monkeypatch):
    rows = stopped_rows()
    note = Mock(wraps=engine.note_turn_complete)
    monkeypatch.setattr(engine, "note_turn_complete", note)
    monkeypatch.setattr(engine, "_on_session_start_unlocked", Mock(side_effect=AssertionError("must not bind")))
    monkeypatch.setattr(engine, "_prepare_leaves_after_turn", Mock(side_effect=AssertionError("must not prepare")))
    engine.on_turn_complete(rows, interrupted=True, failed=False, session_id="wrong-meta-owner")
    assert engine._store.get_session_count("stopped-session") == len(rows)
    assert engine._session_id == "stopped-session"
    note.assert_called_once_with()
    engine.ingest(rows)
    assert engine._store.get_session_count("stopped-session") == len(rows)
    engine.ingest(rows + [{"role": "user", "content": "later turn"}, {"role": "assistant", "content": "done"}])
    stored = engine._store.get_session_messages("stopped-session")
    assert [r["role"] for r in stored] == [r["role"] for r in rows] + ["user", "assistant"]
    assert len(stored) == len(rows) + 2


@pytest.mark.parametrize("meta", [{}, {"interrupted": False}, {"failed": True, "interrupted": False}])
def test_completed_and_failed_only_callback_returns_before_engine_reads(engine, monkeypatch, meta):
    lease, ingest = Mock(), Mock()
    monkeypatch.setattr(engine, "_run_stably", lease)
    monkeypatch.setattr(engine, "ingest", ingest)
    engine.on_turn_complete(stopped_rows(), **meta)
    lease.assert_not_called()
    ingest.assert_not_called()


def test_optional_host_base_callback_is_preserved(engine, monkeypatch):
    base = Mock()
    monkeypatch.setattr(ContextEngine, "on_turn_complete", base)
    rows, usage = stopped_rows(), {"input_tokens": 12}
    engine.on_turn_complete(rows, usage=usage, interrupted=False)
    base.assert_called_once_with(rows, usage=usage, interrupted=False)


@pytest.mark.parametrize("bypass", ["unbound", "ignored", "stateless", "thread_stateless"])
def test_bypassed_callback_does_nothing(engine, monkeypatch, bypass):
    if bypass == "unbound":
        engine._session_id = ""
    elif bypass == "thread_stateless":
        monkeypatch.setattr(engine, "_thread_context_stateless", lambda: True)
    else:
        setattr(engine, "_session_" + bypass, True)
    ingest, note = Mock(), Mock()
    monkeypatch.setattr(engine, "ingest", ingest)
    monkeypatch.setattr(engine, "note_turn_complete", note)
    engine.on_turn_complete(stopped_rows(), interrupted=True)
    ingest.assert_not_called()
    note.assert_not_called()


@pytest.mark.parametrize("failure", ["ingest", "note_turn_complete"])
def test_callback_swallows_failure_and_always_notes_turn(engine, monkeypatch, caplog, failure):
    ingest, note = Mock(), Mock()
    {"ingest": ingest, "note_turn_complete": note}[failure].side_effect = RuntimeError("private detail sentinel")
    monkeypatch.setattr(engine, "ingest", ingest)
    monkeypatch.setattr(engine, "note_turn_complete", note)
    with caplog.at_level("DEBUG", logger="hermes_lcm.engine"):
        engine.on_turn_complete(stopped_rows(), interrupted=True)
    ingest.assert_called_once()
    note.assert_called_once_with()
    assert "private detail sentinel" not in caplog.text
    assert "RuntimeError" in caplog.text
