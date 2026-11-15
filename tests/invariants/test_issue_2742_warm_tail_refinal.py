"""#2742 stays open: a busy session end's warm-tail retry stores the end's final reply a second time.

A session end on the current binding that hits SQLITE_BUSY freezes the rows it selected as new into a warm pending
tail, after the session's last stored row. When a later ordinary write stores the final reply (FINAL) first, and the
tail's first row is one that write treated differently (a first user row the host re-presents with a changed
timestamp, or a tool call whose arguments carry an inline encoded block after a profile rebind), the retry's
positional proof (`_match_pending_tail`) fails at row 0 and appends the whole tail: FINAL is stored twice, with no
relation linking the copies. The retry runs at a later session end, at shutdown plus a new engine on the same store,
or at a sibling engine's start. The inline-args shape is stored once since the receipt bridge (#2759).
The re-stamped final reply stays pinned until the follow-up.

Controls, which pass today: the same events without the re-stamp, with plain tool arguments, and the Hermes /retry
shape (#1179's host input), which this defect does not reach. Only real entry points are used (ingest,
on_session_end, on_session_start, shutdown), plus SQLITE_BUSY injected on the store's ingest admission during one
end. No private engine state is set.
"""

import base64
import copy
import json
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2742: a busy end's warm-tail retry fails its positional proof at row 0 and stores the final reply again",
)


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


def _open(tmp_path, monkeypatch, uid_mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
    for name in ("LCM_IDENTITY_ANCHOR", "LCM_INGEST_LEDGER", "LCM_HOST_MESSAGE_UID"):
        monkeypatch.delenv(name, raising=False)  # shipped defaults
    if uid_mode is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    # As the plugin's register() builds it: config from env, profile home from HERMES_HOME.
    engine = LCMEngine(config=LCMConfig.from_env(), hermes_home=str(tmp_path / "home"))
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **_kwargs: ("Earlier turns.", 1))
    _bind(engine)
    return engine


def _spy(monkeypatch):
    """Observation only: record each warm-tail proof (tail rows, after_store_id, result) without changing it."""
    seen, original = [], LCMEngine._match_pending_tail

    def wrapped(self, tail):
        result = original(self, tail)
        seen.append(([m.get("content") for m in tail["messages"]], tail["after_store_id"], result[0], result[3],
                     result[4]))
        return result

    monkeypatch.setattr(LCMEngine, "_match_pending_tail", wrapped)
    return seen


def _bind(engine, **kwargs):
    engine.on_session_start("parent", platform="cli", context_length=128000, conversation_id="conv", **kwargs)


def _row(text, stamp, role="user", uid=None):
    row = {"role": role, "content": text, "timestamp": stamp}
    if uid is not None:
        row["message_uid"] = uid
    return row


def _call(shape, cid, stamp):
    if shape == "data-uri-key":
        encoded = base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(1200))).decode("ascii")
        args = {"data:image\\/png;base64," + encoded: "v", "b": "x"}
    else:
        args = {"path": "a.txt", "content": "plain body"}
    return {"role": "assistant", "content": "", "timestamp": stamp, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "write_file", "arguments": json.dumps(args)}}]}


def _turn(n, tool=None, uids=True):
    stamp = float(n * 20 + 1)
    rows = [_row(f"question-{n}", stamp), _row(f"reply-{n}", stamp + 1, "assistant")]
    if tool is not None:
        rows += [_call(tool, f"call-{n}", stamp + 2),
                 {"role": "tool", "content": f"result-{n}", "tool_call_id": f"call-{n}", "timestamp": stamp + 3}]
    if uids:
        for i, row in enumerate(rows):
            row["message_uid"] = f"parent-{n}-{i}"
    return rows


def _busy_end(engine, monkeypatch, view):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "ingest_atomic", failed)
        engine.on_session_end("parent", copy.deepcopy(view))
    _require(calls, "the injected SQLITE_BUSY was reached")
    kinds = [t.get("kind") for t in engine._pending_session_tails]
    _require("warm" in kinds, f"the busy end queued a warm tail (tails={kinds})")


def _relations(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'").fetchone():
        return []
    return conn.execute("SELECT store_id, kind, related_store_id FROM message_relations ORDER BY rowid").fetchall()


def _count(engine, text):
    return sum(1 for r in engine._store.get_session_messages("parent") if r["content"] == text)


UID_MODES = [pytest.param(None, id="defaults"), pytest.param("off", id="uid-off")]


@pytest.mark.parametrize("uid_mode", UID_MODES)
@pytest.mark.parametrize("uids", [True, False], ids=["uids", "no-uids"])
@pytest.mark.parametrize("noop", [True, False], ids=["noop", "no-noop"])
@pytest.mark.parametrize("drain", ["end", "shutdown"])
@pytest.mark.parametrize("restamp", [pytest.param(True, id="restamp", marks=_PIN),
                                     pytest.param(False, id="control")])
def test_stamp_only_final_stored_once(tmp_path, monkeypatch, uid_mode, uids, noop, drain, restamp):
    """Ingest; the host re-stamps the first user row; a busy warm end adds FINAL; next turn; drain the tail.
    Correct engine: FINAL is stored once. The control drops the re-stamp."""
    engine = _open(tmp_path, monkeypatch, uid_mode)
    proofs = _spy(monkeypatch)
    try:
        live = _turn(0, uids=uids)
        engine.ingest(copy.deepcopy(live))
        if noop:
            engine.ingest(copy.deepcopy(live))  # the host's no-op re-presentation
        if restamp:
            live[0]["timestamp"] += 1000
        live.append(_row("FINAL", 1003.0, "assistant"))
        _busy_end(engine, monkeypatch, live)
        live += _turn(1, uids=uids)
        engine.ingest(copy.deepcopy(live))
        if drain == "end":
            engine.on_session_end("parent", copy.deepcopy(live))
        else:
            engine.shutdown()
            engine = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
            _bind(engine)
        _require(proofs, "the warm-tail retry ran its proof (_match_pending_tail)")
        rows = [(r["role"], r["content"], r.get("observed_at")) for r in engine._store.get_session_messages("parent")]
        _require(_count(engine, "FINAL") >= 1, f"FINAL is stored at least once: {rows}")
        assert _count(engine, "FINAL") == 1, (rows, proofs, _relations(engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("uid_mode", UID_MODES)
@pytest.mark.parametrize("tool", [pytest.param("data-uri-key", id="inline"),
                                  pytest.param("plain", id="control")])
def test_inline_args_final_stored_once(tmp_path, monkeypatch, uid_mode, tool):
    """Ingest with a tool call; profile rebind and back; a busy warm end adds FINAL; next turn; drain through a
    sibling engine's start on the same store and a later end. Correct engine: FINAL is stored once. The control
    uses plain tool arguments."""
    engine = _open(tmp_path, monkeypatch, uid_mode)
    proofs = _spy(monkeypatch)
    sibling = None
    try:
        live = _turn(0, tool)
        engine.ingest(copy.deepcopy(live))
        alternate = tmp_path / "profile-1"
        alternate.mkdir()
        home = engine._hermes_home
        _bind(engine, hermes_home=str(alternate))
        _bind(engine, hermes_home=home)
        live.append(_row("FINAL", 1002.0, "assistant"))
        _busy_end(engine, monkeypatch, live)
        live += _turn(1)
        engine.ingest(copy.deepcopy(live))
        sibling = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
        _bind(sibling)
        engine.on_session_end("parent", copy.deepcopy(live))
        _require(proofs, "the warm-tail retry ran its proof (_match_pending_tail)")
        rows = [(r["role"], r["content"], r.get("observed_at")) for r in engine._store.get_session_messages("parent")]
        _require(_count(engine, "FINAL") >= 1, f"FINAL is stored at least once: {rows}")
        assert _count(engine, "FINAL") == 1, (rows, proofs)
    finally:
        if sibling is not None:
            sibling.shutdown()
        engine.shutdown()


@pytest.mark.parametrize("uid_mode", UID_MODES)
@pytest.mark.parametrize("retry", [True, False], ids=["retry", "control"])
def test_control_retry_shaped_final_stored_once(tmp_path, monkeypatch, uid_mode, retry):
    """Control, both arms: the Hermes /retry shape (#1179's host input). The last exchange is archived and its user
    text re-sent with a new stamp and uid; the busy end carries the new reply; the next turn is ingested; the session
    ends normally. FINAL is stored once today, and must stay so."""
    engine = _open(tmp_path, monkeypatch, uid_mode)
    proofs = _spy(monkeypatch)
    try:
        live = _turn(0) + _turn(1)
        engine.ingest(copy.deepcopy(live))
        if retry:
            live = live[:2] + [_row("question-1", 500.0, uid="parent-retry-1")]
        live.append(_row("FINAL", 501.0, "assistant", uid="parent-final"))
        _busy_end(engine, monkeypatch, live)
        live += _turn(2)
        engine.ingest(copy.deepcopy(live))
        engine.on_session_end("parent", copy.deepcopy(live))
        rows = [(r["role"], r["content"], r.get("observed_at")) for r in engine._store.get_session_messages("parent")]
        assert _count(engine, "FINAL") == 1, (rows, proofs)
    finally:
        engine.shutdown()
