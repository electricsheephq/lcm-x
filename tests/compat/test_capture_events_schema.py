"""Claude Code capture ledger (``capture_events_v1``), per docs/process/schema-changes.md (I5).

Rollback: the previous GA opens a capture store and reads its rows; the ledger and the stamp
survive, and capture resumes without a duplicate. Forward: a store the previous GA wrote is
never adopted and never gets the ledger (capture stores are only created by the writer).
"""

import hashlib
import json

import pytest

from hermes_lcm import claude_code_store as cc
from tests.compat.helpers import PREVIOUS_GA, READERS, run_old

OLD_READ = r'''
store = store_mod.MessageStore(db, hermes_home=home)
try:
    rows = store.get_session_messages("s1")
    print(json.dumps({"rows": [[r["role"], r["content"]] for r in rows]}))
finally:
    store.close()
'''
OLD_WRITE = r'''
store = store_mod.MessageStore(db, hermes_home=home)
try:
    store.append_batch("S", [{"role": "user", "content": "Synthetic previous-release turn."}])
    print(json.dumps({"ok": True}))
finally:
    store.close()
'''


def _transcript(tmp_path, lines):
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for uid, role, text in lines:
            handle.write(json.dumps({"type": role, "uuid": uid, "sessionId": "s1",
                                     "message": {"role": role, "content": text}}) + "\n")
    return path


def _ledger(store):
    rows = store.connection.execute("SELECT * FROM capture_events ORDER BY event_id").fetchall()
    return len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()


def test_previous_ga_opens_a_capture_store_and_capture_resumes(tmp_path, old_tree, host_root):
    reference = tmp_path / "hermes-reference"
    path = _transcript(tmp_path, [("u1", "user", "Synthetic capture request."),
                                  ("a1", "assistant", "Synthetic capture reply.")])
    lineage = cc.Lineage("s1", None, path)
    store = cc.open_store(cc.admit(tmp_path, hermes_home=reference), tmp_path)
    try:
        cc.drain(store, lineage)
        ledger = _ledger(store)
        cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    report = run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_READ)
    assert report["rows"] == [["user", "Synthetic capture request."], ["assistant", "Synthetic capture reply."]]
    store = cc.open_store(cc.admit(tmp_path, hermes_home=reference), tmp_path)  # stamp intact after rollback
    try:
        assert _ledger(store) == ledger
        _transcript(tmp_path, [("u2", "user", "Synthetic turn after the rollback.")])
        cc.drain(store, lineage)
        assert [row["content"] for row in store.get_session_messages("s1")] == [
            "Synthetic capture request.", "Synthetic capture reply.", "Synthetic turn after the rollback."]
    finally:
        store.close()


def test_a_previous_ga_store_is_never_adopted(tmp_path, old_tree, host_root):
    assert run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_WRITE, mode="seed")["ok"]
    database = tmp_path / "lcm.db"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(cc.AdmissionError, match="store_unstamped"):
        cc.admit(tmp_path, hermes_home=tmp_path / "hermes-reference")
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert not (tmp_path / cc.OWNER_FILE).exists()
