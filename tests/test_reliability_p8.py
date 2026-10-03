"""P8 invariant and fail-open regressions using only sanitized fixtures."""
import hashlib
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import cells, probe
from bench.instruments.reliability.scorers import host_rewrite


COMMIT = dict(event="commit", session="S0", row_id=7, role="user", uid="hashed",
              active=1, target_role="user", target_uid="hashed", expected="digest", after="digest")
FLUSH = dict(event="flush_resolve", session="S0", row_id=7, target_id=7, role="user", uid="hashed",
             target_role="user", target_uid="hashed", active=1, path="row_id", active_count=1, action="MATCH", effect=True)


def scored(tmp_path, events=(), phases=None, cell=None):
    (tmp_path / "p8-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return host_rewrite.score(cell or {}, tmp_path, phases or [{"p8": {"supported": True}}])


@pytest.mark.parametrize("field,value", [("active", 0), ("active", None), ("target_role", "assistant"),
                                        ("target_uid", "wrong"), ("after", "stale"), ("expected", None)])
def test_i0_requires_address_role_uid_and_snapshot(tmp_path, field, value):
    out = scored(tmp_path, [dict(COMMIT, **{field: value})])
    assert out["failed_invariants"] == {"I0": {"count": 1, "row_ids": [7]}}


@pytest.mark.parametrize("action", ["REWRITE", "LEGACY", "ADOPT"])
def test_i1_archived_target(tmp_path, action):
    out = scored(tmp_path, [dict(FLUSH, active=0, action=action)])
    assert out["failed_invariants"]["I1"] == {"count": 1, "row_ids": [7, 7]}
    assert ("I3" in out["failed_invariants"]) == (action == "ADOPT")


@pytest.mark.parametrize("delta", [dict(target_role="assistant"), dict(target_uid="wrong"),
                                   dict(path="uid_snapshot", active_count=2),
                                   dict(path="uid_snapshot", active_count=0)])
def test_i2_live_identity_and_unique_resolution(tmp_path, delta):
    assert set(scored(tmp_path, [dict(FLUSH, **delta)])["failed_invariants"]) == {"I2"}


def test_i3_adopt_is_failure_even_for_host_made_dict(tmp_path):
    out = scored(tmp_path, [dict(FLUSH, action="ADOPT", lcm=False)])
    assert out["failed_invariants"] == {"I3": {"count": 1, "row_ids": [7, 7]}}


@pytest.mark.parametrize("where", ["transaction", "end"])
def test_i5_lcm_twins_fail_host_twins_are_reported(tmp_path, where):
    twins = dict(session="S0", role="user", uid="hashed", row_ids=[7, 9], lcm=True)
    for lcm in (True, False):
        item = dict(twins, lcm=lcm)
        events = [dict(event="sweep", duplicates=[item])] if where == "transaction" else []
        phases = [{"p8": dict(supported=True, duplicates=[item] if where == "end" else [])}]
        out = scored(tmp_path, events, phases)
        assert out["verdict"] == ("FAIL" if lcm else "PASS")
        assert bool(out["reported_duplicates"]) != lcm
        if lcm:
            assert out["failed_invariants"] == {"I5": {"count": 1, "row_ids": [7, 9]}}


def test_actions_and_counts_are_reported(tmp_path):
    events = [COMMIT] + [dict(FLUSH, action=a, target_id=None if a == "INSERT" else 7)
                        for a in ("INSERT", "REWRITE", "MATCH", "LEGACY")]
    out = scored(tmp_path, events)
    assert out["verdict"] == "PASS" and out["actions"] == dict.fromkeys(("INSERT", "REWRITE", "MATCH", "LEGACY"), 1)
    out = scored(tmp_path, [dict(COMMIT, active=0)] * 2)
    assert out["failed_invariants"]["I0"]["count"] == 2


def test_missing_disabled_process_or_partial_audit_is_unsupported(tmp_path):
    for phases in ([{}], [{"p8": {"supported": False}}], [{"p8": {"supported": True, "notes": ["OSError"]}}]):
        assert scored(tmp_path, [COMMIT], phases)["verdict"] == "UNSUPPORTED"
    for transport in ("acp-process", "gateway-process", "api-server"):
        assert scored(tmp_path, [COMMIT], cell={"transport": transport})["verdict"] == "UNSUPPORTED"
    (tmp_path / "p8-events.jsonl").unlink()
    assert host_rewrite.score({}, tmp_path, [{"p8": {"supported": True}}])["verdict"] == "UNSUPPORTED"


def fake_host(monkeypatch, tmp_path):
    import sys
    import types
    import agent
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,message_uid TEXT,active INT,content TEXT)")
    conn.execute("INSERT INTO messages VALUES(7,'S0','user','fixture-uid',1,'sanitized payload')")
    conn.commit()
    repair = types.ModuleType("agent.transcript_repair")
    repair.message_uid_or_none = lambda m: m.get("message_uid")
    repair.transcript_row_snapshot = lambda r: hashlib.sha256(r["content"].encode()).hexdigest()
    repair._active_message_row = lambda c, s, i, r: c.execute("SELECT * FROM messages WHERE session_id=? AND id=?", (s, i)).fetchone()
    repair._active_logical_message_row = lambda c, s, r, u: c.execute(
        "SELECT * FROM messages WHERE session_id=? AND role=? AND message_uid=? AND active=1 ORDER BY id DESC", (s, r, u)).fetchone()
    calls, marker = [], object()

    def resolve(c, sid, rows, *args, **kwargs):
        calls.append((c, sid, rows, args, kwargs))
        return marker

    repair.resolve_and_repair_transcript_batch = resolve
    persistence = types.ModuleType("agent.session_persistence")

    def write(ag, rows, live, messages):
        return repair.resolve_and_repair_transcript_batch(conn, ag.session_id, rows)

    persistence._db_flush_write = write
    compression = types.ModuleType("agent.conversation_compression")
    compression._commit_compaction = lambda ag, messages: SimpleNamespace(session_commit_succeeded=True, compressed=messages)
    for name, module in (("transcript_repair", repair), ("session_persistence", persistence), ("conversation_compression", compression)):
        monkeypatch.setitem(sys.modules, "agent." + name, module)
        monkeypatch.setattr(agent, name, module, raising=False)
    ag = SimpleNamespace(session_id="S0", _session_db=SimpleNamespace(_conn=conn, _lock=threading.Lock()))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_RELIABILITY_P8", raising=False)
    live = dict(role="user", message_uid="fixture-uid", _row_id=7,
                _db_row_snapshot=repair.transcript_row_snapshot(dict(conn.execute("SELECT * FROM messages").fetchone())))
    return repair, persistence, compression, ag, live, calls, marker


def test_wraps_call_through_log_only_hashes_and_pair_original_live_identity(monkeypatch, tmp_path):
    repair, persistence, compression, ag, live, calls, marker = fake_host(monkeypatch, tmp_path)
    state, pin, finish = probe.install_p8(tmp_path, "A", {}, set(), None, {})
    pin([live])
    assert persistence._db_flush_write(ag, [dict(live)], [live], [live]) is marker
    compression._commit_compaction(ag, [live])
    events = [json.loads(x) for x in (tmp_path / "p8-events.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["event"] == "flush_resolve")["lcm"] is True
    text = json.dumps(events)
    assert "sanitized payload" not in text and "fixture-uid" not in text
    assert finish()["notes"] == [] and len(calls) == 1
    row = dict(live, message_uid="other-uid")
    persistence._db_flush_write(ag, [row], [live], [live])
    assert state["notes"] == []


def test_observation_failure_never_changes_host_return_or_exception(monkeypatch, tmp_path):
    repair, persistence, _, ag, live, calls, marker = fake_host(monkeypatch, tmp_path)
    repair.transcript_row_snapshot = lambda r: (_ for _ in ()).throw(ValueError("private payload"))
    state, _, finish = probe.install_p8(tmp_path, "A", {}, set(), None, {})
    assert persistence._db_flush_write(ag, [live], [live], [live]) is marker
    assert state["notes"] == ["ValueError"] and len(calls) == 1
    assert "private payload" not in json.dumps(finish())
    # The original host exception propagates, and the host is never retried by the audit.
    def broken(*args, **kwargs):
        calls.append("broken")
        raise RuntimeError("host failure")
    repair.resolve_and_repair_transcript_batch = broken
    with pytest.raises(RuntimeError, match="host failure"):
        persistence._db_flush_write(ag, [live], [live], [live])
    assert calls.count("broken") == 1


def test_disabled_and_missing_seams_keep_original_callables(monkeypatch, tmp_path):
    repair, persistence, compression, _, _, _, _ = fake_host(monkeypatch, tmp_path)
    original = (repair.resolve_and_repair_transcript_batch, persistence._db_flush_write, compression._commit_compaction)
    monkeypatch.setenv("LCM_RELIABILITY_P8", "off")
    state, _, _ = probe.install_p8(tmp_path, "A", {}, set(), None, {})
    assert not state["supported"]
    assert original == (repair.resolve_and_repair_transcript_batch, persistence._db_flush_write, compression._commit_compaction)
    monkeypatch.delenv("LCM_RELIABILITY_P8")
    del repair._active_message_row
    assert not probe.install_p8(tmp_path, "A", {}, set(), None, {})[0]["supported"]
    assert original == (repair.resolve_and_repair_transcript_batch, persistence._db_flush_write, compression._commit_compaction)


def test_controls_are_exactly_registered():
    by = {c["id"]: c for c in cells.select("p8-control/*")}
    assert len(by) == 4 and by["p8-control/none"]["faults"] == []
    for variant in ("archived", "other-active", "random-snapshot"):
        assert by["p8-control/" + variant]["faults"] == [{"kind": "p8_inject", "variant": variant}]


def test_archived_legacy_read_without_write_or_adopt_is_only_reported(tmp_path):
    assert scored(tmp_path, [dict(FLUSH, active=0, action="LEGACY", effect=False)])["verdict"] == "PASS"


def test_control_flush_releases_non_reentrant_lock(monkeypatch, tmp_path):
    repair, persistence, compression, ag, live, calls, marker = fake_host(monkeypatch, tmp_path)
    persistence._db_flush_row = lambda ag, live, override: dict(live)
    state, pin, _ = probe.install_p8(tmp_path, "A", {"p8_inject": {"variant": "random-snapshot"}},
                                   set(), lambda *args, **kwargs: calls.append("fired"), {"turn": 9})
    pin([live])
    # Prove the host write is called only after the compaction lock is released.
    repair.resolve_and_repair_transcript_batch = lambda *args, **kwargs: assert_unlocked_host()
    def assert_unlocked_host():
        assert ag._session_db._lock.acquire(blocking=False)
        ag._session_db._lock.release()
        calls.append("unlocked")
        return marker
    compression._commit_compaction(ag, [live])
    compression._commit_compaction(ag, [live])
    assert "unlocked" in calls and "fired" in calls and state["notes"] == []
