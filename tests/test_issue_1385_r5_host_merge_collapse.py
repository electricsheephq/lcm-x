"""I1/I2: fixed CI-shaped exchanges across Hermes role repair, without generation."""

from copy import deepcopy
import json
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import escalation, tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.conftest import _MAINTENANCE_SCHEDULERS, _wait_scheduler_idle


def _identity(row):
    content = row.get("content")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    calls = row.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    return row["role"], content, row.get("tool_call_id") or "", calls


@pytest.mark.parametrize("in_place", [True, False], ids=["same-session", "rotation"])
def test_fresh_structured_exchange_after_repeated_users_and_host_merge(tmp_path, monkeypatch, in_place):
    """Two equal list users never vouch for the distinct third exchange as replay."""
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k:
                        "Earlier turns retained.\nExpand for details about: turns")
    home = tmp_path / "home"
    home.mkdir()
    session = "identity-session"
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        conn.execute("INSERT INTO sessions VALUES (?, NULL, NULL)", (session,))
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
                                      fresh_tail_count=2, leaf_chunk_tokens=16), hermes_home=str(home))
    engine.on_session_start(session, platform="cli", conversation_id="identity-conversation", context_length=200_000)
    host, expected = [], []
    sequence = 0

    def stamp(row):
        nonlocal sequence
        sequence += 1
        return dict(row, timestamp=float(sequence), message_uid=f"occurrence-{sequence}")

    def assert_ledger():
        cursor = engine._store.connection.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages "
            "WHERE conversation_id = ? ORDER BY store_id", ("identity-conversation",))
        rows = [dict(zip((column[0] for column in cursor.description), row)) for row in cursor.fetchall()]
        assert list(map(_identity, rows)) == list(map(_identity, expected))

    def deliver(text, arguments):
        user = stamp({"role": "user", "content": [
            {"type": "text", "text": text}, {"type": "text", "text": "part two"}]})
        calls = [{"id": f"call-{sequence}-{i}", "type": "function",
                  "function": {"name": "inspect", "arguments": arg}} for i, arg in enumerate(arguments)]
        batch = [user, stamp({"role": "assistant", "content": "calling", "tool_calls": calls})]
        batch.extend(stamp({"role": "tool", "content": f"result {i}", "tool_call_id": call["id"]})
                     for i, call in enumerate(calls))
        batch.append(stamp({"role": "assistant", "content": f"answer {text} occurrence {sequence}"}))
        expected.extend(deepcopy(batch))
        host.extend(batch)
        engine.ingest(host)
        assert_ledger()

    try:
        deliver("alpha", ('{"query":1}', ' {"query": 0}'))
        deliver("alpha", (' {"query": 0}',))
        before = deepcopy(host)
        compressed = engine.compress(host, current_tokens=engine.threshold_tokens + 1, force=True)
        def without_persisted(rows):
            return [{key: value for key, value in row.items() if key != "_db_persisted"} for row in rows]

        assert without_persisted(compressed) != without_persisted(before), (
            "the scenario must adopt a changed compression output")
        engine.on_session_end(session, before)
        child = session if in_place else "identity-session-1"
        if not in_place:
            with sqlite3.connect(home / "state.db") as conn:
                conn.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = ?", (session,))
                conn.execute("INSERT INTO sessions VALUES (?, ?, NULL)", (child, session))
        engine.on_session_start(child, platform="cli", conversation_id="identity-conversation",
                                boundary_reason="compression", old_session_id=session)
        host = []
        # Match the state machine's host operation. List-content users are not
        # merged by Hermes; invoking this operation need not change the view.
        for message in compressed:
            previous = host[-1] if host else None
            if (previous is not None and previous["role"] == message["role"] == "user"
                    and isinstance(previous.get("content"), str) and isinstance(message.get("content"), str)):
                prefix, incoming = previous["content"], message["content"]
                previous["content"] = prefix + "\n\n" + incoming if prefix and incoming else prefix or incoming
                if prefix and incoming:
                    previous["_merged_turn_prefix"] = prefix + "\n\n"
                if previous["content"] != prefix:
                    previous.pop("_db_persisted", None)
            else:
                host.append(message)
        assert_ledger()
        deliver("βeta", ('{"query":1}',))
        engine.ingest(deepcopy(host))
        assert_ledger()
        engine.compress(host, current_tokens=engine.threshold_tokens + 1, force=True)
        assert_ledger()
    finally:
        engine.shutdown()
        for name in _MAINTENANCE_SCHEDULERS:
            scheduler = getattr(engine_module, name, None)
            if scheduler is not None:
                assert _wait_scheduler_idle(scheduler, 30.0), name
        tokens._count_tokens_cached.cache_clear()
