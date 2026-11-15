"""#2550 regressions on two paths: a turn-0 tool call with inline-argument content, and its result, are stored again.

Both paths need an inline-argument shape in turn 0's tool call (line-wrapped base64 in a file body, or a JSON-escaped
`data:` URI key), and both store the unchanged call and its unchanged result a second time.

Path 1, a busy session end. A session end hits SQLITE_BUSY on the store's ingest admission (and, for the
before-append shape, on the session-count read), so the end runs its frozen read-only re-run. That re-run has no
ingest-ledger attempt, so the ledger's cold cursor is skipped; reconcile's stored-tail compare fails on the
inline-argument call, the cursor falls to 0, and the call and result are stored again. It is reached after a restart
with no stamp, or with the host re-presenting the first user row with a changed timestamp or message_uid.

Path 2, no busy end: the host re-presents the first user row with a changed timestamp or message_uid, then the
binding is cut (another session's start and back, a profile rebind, or a restart), then the same view is presented
again. The stamp breaks the ledger's contiguous proof at row 0, reconcile falls to cursor 0, and the whole-segment
rule (#2191) drops the recognised result because the call is unrecognised, so both are stored again. Later
presentations do not remove the second copy.

These cases pin both duplicates (strict xfails before the repair). The controls, which pass today, run the same events with
plain tool arguments, without the stamp, without the busy fault, or without the binding cut. The re-stamped user row
may be stored as a new occurrence by design (#1179, #2720), so it is never asserted. Only real entry points are used
(ingest, on_session_end, on_session_start, shutdown and the exit flush hook), plus storage calls monkeypatched to raise
sqlite3.OperationalError('database is locked'). No private engine state is set.
"""

import base64
import copy
import json
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import maintenance_scheduler
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.harness.engine import make_engine


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


def _encoded(size):
    return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(size))).decode("ascii")


def _args(shape):
    if shape == "wrapped-base64":
        body = _encoded(3420)
        return {"path": "a.b64",
                "content": "hdr\n" + "\n".join(body[i:i + 76] for i in range(0, len(body), 76)) + "\nend"}
    if shape == "data-uri-key":
        return {"data:image\\/png;base64," + _encoded(1200): "v", "b": "x"}
    return {"path": "a.txt", "content": "plain body"}


# --- Path 1: a busy session end -------------------------------------------------------------------------------------

ENV_DEFAULTS = {}  # shipped defaults: anchor on, ledger shadow, host uid shadow
ENV_GENERATOR = {"LCM_IDENTITY_ANCHOR": "1", "LCM_INGEST_LEDGER": "shadow", "LCM_HOST_MESSAGE_UID": "off"}


def _urow(role, text, ts, uid):
    return {"role": role, "content": text, "timestamp": ts, "message_uid": uid}


def _turn(n, shape):
    stamp = float(n * 20 + 1)
    rows = [_urow("user", f"question-{n}", stamp, f"u-{n}-0"),
            _urow("assistant", f"reply-{n}", stamp + 1, f"u-{n}-1")]
    if shape != "none":
        cid = f"call-{n}"
        rows.append({"role": "assistant", "content": "", "timestamp": stamp + 2, "message_uid": f"u-{n}-2",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": "write_file", "arguments": json.dumps(_args(shape))}}]})
        rows.append({"role": "tool", "content": f"result-{n}", "tool_call_id": cid, "timestamp": stamp + 3,
                     "message_uid": f"u-{n}-3"})
    return rows


class _Probe:
    """Observation only: counts injected busy errors and frozen read-only re-runs without changing either."""

    def __init__(self, monkeypatch):
        self.busy = 0
        self.frozen_reruns = 0
        original = LCMEngine._ingest_messages_transaction

        def wrapped(engine, *args, **kwargs):
            if getattr(engine, "_ingest_read_only", None) is not None:
                self.frozen_reruns += 1
            return original(engine, *args, **kwargs)

        monkeypatch.setattr(LCMEngine, "_ingest_messages_transaction", wrapped)

    def locked(self, *_args, **_kwargs):
        self.busy += 1
        raise sqlite3.OperationalError("database is locked")

    def busy_end(self, patch, engine, kind, view):
        busy, frozen = self.busy, self.frozen_reruns
        patch.setattr(engine._store, "ingest_atomic", self.locked)
        if kind == "before-append":
            patch.setattr(engine._store, "get_session_count", self.locked)
        engine.on_session_end("parent", copy.deepcopy(view))
        _require(self.busy > busy, "the injected SQLITE_BUSY was reached")
        _require(self.frozen_reruns > frozen, "the busy end ran its frozen read-only re-run")


def _stamp(live, field):
    row = next(m for m in live if m["role"] == "user")
    if field == "timestamp":
        row["timestamp"] = row["timestamp"] + 1000
    else:
        row["message_uid"] = "changed-uid"


def _counts(engine):
    rows = engine._store.get_session_messages("parent")
    results = sum(1 for r in rows if r["role"] == "tool" and r["content"] == "result-0")
    calls = sum(1 for r in rows if r["role"] == "assistant" and "call-0" in json.dumps(r.get("tool_calls") or ""))
    summary = " | ".join(f'{r["role"][0]}:{(r["content"] or "")[:12]}@{r.get("observed_at")}' for r in rows)
    return results, calls, summary


@pytest.fixture
def env(monkeypatch, request):
    for key in ("LCM_IDENTITY_ANCHOR", "LCM_INGEST_LEDGER", "LCM_HOST_MESSAGE_UID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in request.param.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *args, **kwargs: None)
    return request.param


ENVS = [pytest.param(ENV_DEFAULTS, id="defaults"), pytest.param(ENV_GENERATOR, id="generator-env")]


def _warm_ts(engine, monkeypatch, probe, shape, stamp):
    """Warm busy end with a final reply, plain ingest of the next turn, optional stamp, then an end."""
    live = _turn(0, shape)
    engine.ingest(copy.deepcopy(live))
    live.append(_urow("assistant", "final-1", 1001.0, "f-1"))
    with monkeypatch.context() as patch:
        probe.busy_end(patch, engine, "warm", live)
    live += _turn(1, "plain")
    engine.ingest(copy.deepcopy(live))
    if stamp:
        _stamp(live, stamp)
    engine.on_session_end("parent", copy.deepcopy(live))
    maintenance_scheduler._flush_pending_tails_at_exit()


def _before_append(engine, monkeypatch, probe, shape, stamp):
    """Optional stamp, a busy end that also fails the session-count read, then an end."""
    live = _turn(0, shape)
    engine.ingest(copy.deepcopy(live))
    if stamp:
        _stamp(live, stamp)
    with monkeypatch.context() as patch:
        probe.busy_end(patch, engine, "before-append", live)
    engine.on_session_end("parent", copy.deepcopy(live))
    maintenance_scheduler._flush_pending_tails_at_exit()


def _restart_warm(engine, monkeypatch, probe, shape, stamp, fault="warm"):
    """Ingest; a new engine on the same store binds the session; a busy end with a final reply; then a sibling bind
    and the exit flush retry the tail."""
    live = _turn(0, shape)
    engine.ingest(copy.deepcopy(live))
    engine.shutdown()
    second = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
    second.on_session_start("parent", platform="cli", context_length=128000, conversation_id="conv")
    if stamp:
        _stamp(live, stamp)
    live.append(_urow("assistant", "final-2", 1002.0, "f-2"))
    with monkeypatch.context() as patch:
        if fault != "none":
            probe.busy_end(patch, second, fault, live)
        else:
            second.on_session_end("parent", copy.deepcopy(live))
    sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
    sibling.on_session_start("parent", platform="cli", context_length=128000, conversation_id="conv")
    maintenance_scheduler._flush_pending_tails_at_exit()
    return [second, sibling]


def _restart_end(engine, monkeypatch, probe, shape, stamp):
    """Control for restart-warm: the same restart and end with no busy fault."""
    return _restart_warm(engine, monkeypatch, probe, shape, stamp, fault="none")


SCENARIOS = {"warm-ts": _warm_ts, "before-append": _before_append, "restart-warm": _restart_warm,
             "restart-end": _restart_end}


def _run_busy(tmp_path, monkeypatch, scenario, shape, stamp):
    engine = make_engine(tmp_path, window=128000, session_id="parent", conversation_id="conv")
    probe = _Probe(monkeypatch)
    extra = []
    try:
        extra = SCENARIOS[scenario](engine, monkeypatch, probe, shape, stamp) or []
        reader = extra[-1] if extra else engine
        return _counts(reader)
    finally:
        for instance in [*reversed(extra), engine]:
            instance.shutdown()


BUSY_CASES = [
    ("warm-ts", "wrapped-base64", "timestamp"),
    ("warm-ts", "data-uri-key", "timestamp"),
    ("warm-ts", "wrapped-base64", "message_uid"),
    ("before-append", "data-uri-key", "message_uid"),
    ("before-append", "wrapped-base64", "timestamp"),
    ("before-append", "data-uri-key", "timestamp"),
    ("restart-warm", "wrapped-base64", None),
    ("restart-warm", "data-uri-key", None),
    ("restart-warm", "wrapped-base64", "timestamp"),
]
BUSY_CONTROLS = [
    # same events, plain arguments
    ("warm-ts", "plain", "timestamp"),
    ("before-append", "plain", "message_uid"),
    ("before-append", "plain", "timestamp"),
    ("restart-warm", "plain", "timestamp"),
    ("restart-warm", "plain", None),
    # same restart and end, no busy fault
    ("restart-end", "wrapped-base64", None),
    ("restart-end", "data-uri-key", None),
    # same events, no stamp
    ("warm-ts", "wrapped-base64", None),
    ("before-append", "data-uri-key", None),
    ("before-append", "wrapped-base64", None),
]


@pytest.mark.parametrize("env", ENVS, indirect=True)
@pytest.mark.parametrize("scenario,shape,stamp", [pytest.param(*case) for case in BUSY_CASES])
def test_inline_call_stored_once_after_busy_end(env, tmp_path, monkeypatch, scenario, shape, stamp):
    """Correct engine: the turn-0 tool call and its result are each stored once."""
    results, calls, summary = _run_busy(tmp_path, monkeypatch, scenario, shape, stamp)
    # A missing row must fail loudly, never be absorbed by a strict xfail that expects the duplicate.
    _require(results >= 1 and calls >= 1, f"turn 0's call and result are each stored at least once: {summary}")
    assert (results, calls) == (1, 1), summary


@pytest.mark.parametrize("env", ENVS, indirect=True)
@pytest.mark.parametrize("scenario,shape,stamp", BUSY_CONTROLS)
def test_control_stored_once_after_busy_end(env, tmp_path, monkeypatch, scenario, shape, stamp):
    results, calls, summary = _run_busy(tmp_path, monkeypatch, scenario, shape, stamp)
    assert (results, calls) == (1, 1), summary


# --- Path 2: a re-stamped first user row plus a binding cut ---------------------------------------------------------


def _view(shape):
    rows = [
        {"role": "user", "content": "parent-question-0", "timestamp": 1.0},
        {"role": "assistant", "content": "parent-reply-0", "timestamp": 2.0},
        {"role": "assistant", "content": "", "timestamp": 3.0, "tool_calls": [{
            "id": "call-0", "type": "function",
            "function": {"name": "write_file", "arguments": json.dumps(_args(shape))}}]},
        {"role": "tool", "content": "parent-result-0", "tool_call_id": "call-0", "timestamp": 4.0},
    ]
    for i, row in enumerate(rows):
        row["message_uid"] = f"parent-0-{i}"
    return rows


def _new_engine(home):
    return LCMEngine(config=LCMConfig.from_env(), hermes_home=str(home))


def _bind(engine, sid, **kwargs):
    engine.on_session_start(sid, platform="cli", context_length=128000,
                            conversation_id="conv" if sid == "parent" else "other-conv", **kwargs)


def _restamp_env(tmp_path, monkeypatch, uid_mode=None):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
    if uid_mode is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_kw: ("Earlier turns.", 1))
    return home


def _run_restamp(tmp_path, monkeypatch, *, shape, stamp, cut, present, uid_mode):
    home = _restamp_env(tmp_path, monkeypatch, uid_mode)
    made = []
    try:
        engine = _new_engine(home)
        made.append(engine)
        _bind(engine, "parent")
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        _require(len(engine._store.get_session_messages("parent")) == 4, "the first ingest stored turn 0's 4 rows")
        if stamp == "timestamp":
            view[0]["timestamp"] = view[0]["timestamp"] + 1000
        elif stamp == "message_uid":
            view[0]["message_uid"] = "changed-3"
        if cut == "start-other":
            _bind(engine, "other")
            _bind(engine, "parent")
        elif cut == "rebind":
            alternate = tmp_path / "profile-alt"
            alternate.mkdir()
            _bind(engine, "parent", hermes_home=str(alternate))
            _bind(engine, "parent", hermes_home=str(home))
        elif cut == "restart":
            engine.shutdown()
            engine = _new_engine(home)
            made.append(engine)
            _bind(engine, "parent")
        if present == "end":
            engine.on_session_end("parent", copy.deepcopy(view))
        else:
            engine.ingest(copy.deepcopy(view))
        rows = engine._store.get_session_messages("parent")
        recon = dict(getattr(engine, "_last_ingest_reconciliation", {}) or {})
        return rows, recon
    finally:
        for instance in reversed(made):
            instance.shutdown()


def _assert_once(rows, recon):
    results = [r for r in rows if r["role"] == "tool" and r["content"] == "parent-result-0"]
    calls = [r for r in rows if r["role"] == "assistant" and r.get("tool_calls")]
    replies = [r for r in rows if r["role"] == "assistant" and r["content"] == "parent-reply-0"]
    detail = " | ".join(f"{r['store_id']}:{r['role']}:{(r['content'] or '')[:18]}@{r.get('observed_at')}"
                        f"{':TC' if r.get('tool_calls') else ''}" for r in rows) + " || " + repr(
        {k: recon.get(k) for k in ("action", "reason", "cursor", "stored_after_first_cut") if k in recon})
    _require(replies and results and calls, f"turn 0's reply, call and result are each stored at least once: {detail}")
    assert len(replies) == 1, detail
    assert len(results) == 1, detail
    assert len(calls) == 1, detail


UID_MODES = [pytest.param(None, id="uid-default"), pytest.param("off", id="uid-off")]

RESTAMP_CASES = [
    # (shape, stamp, cut, present)
    ("data-uri-key", "timestamp", "start-other", "end"),
    ("wrapped-base64", "timestamp", "restart", "ingest"),
    ("data-uri-key", "timestamp", "rebind", "ingest"),
    ("wrapped-base64", "timestamp", "start-other", "end"),
    ("data-uri-key", "message_uid", "start-other", "end"),
]
RESTAMP_CONTROLS = [
    ("plain", "timestamp", "start-other", "end"),      # no inline transform
    ("plain", "timestamp", "restart", "ingest"),
    ("plain", "timestamp", "rebind", "ingest"),
    ("data-uri-key", None, "start-other", "end"),       # no stamp
    ("wrapped-base64", None, "restart", "ingest"),
    ("data-uri-key", None, "rebind", "ingest"),
    ("data-uri-key", "timestamp", None, "end"),         # no binding cut
    ("wrapped-base64", "timestamp", None, "ingest"),
]


@pytest.mark.parametrize("uid_mode", UID_MODES)
@pytest.mark.parametrize("shape,stamp,cut,present",
                         [pytest.param(*c, id="-".join(map(str, c))) for c in RESTAMP_CASES])
def test_restamp_cut_call_and_result_stored_once(tmp_path, monkeypatch, shape, stamp, cut, present, uid_mode):
    """Correct engine: the reply, the unchanged call and the unchanged result are each stored once."""
    rows, recon = _run_restamp(tmp_path, monkeypatch, shape=shape, stamp=stamp, cut=cut, present=present,
                               uid_mode=uid_mode)
    _assert_once(rows, recon)


@pytest.mark.parametrize("uid_mode", UID_MODES)
@pytest.mark.parametrize("shape,stamp,cut,present", RESTAMP_CONTROLS,
                         ids=["-".join(map(str, c)) for c in RESTAMP_CONTROLS])
def test_control_restamp_cut_call_and_result_stored_once(tmp_path, monkeypatch, shape, stamp, cut, present,
                                                         uid_mode):
    rows, recon = _run_restamp(tmp_path, monkeypatch, shape=shape, stamp=stamp, cut=cut, present=present,
                               uid_mode=uid_mode)
    _assert_once(rows, recon)


@pytest.mark.parametrize("repeat_stamp", [False, True], ids=["plain-followups", "second-stamp-and-cut"])
def test_restamp_cut_later_presentations_store_call_and_result_once(tmp_path, monkeypatch, repeat_stamp):
    """After the forcing sequence, later real presentations (ingest, an optional second stamp and cut, a restart and
    an end). Correct engine: the call and result are each stored once at the end; today nothing later removes the
    second copy, and a second stamp plus cut adds another."""
    home = _restamp_env(tmp_path, monkeypatch)
    made = []
    try:
        engine = _new_engine(home)
        made.append(engine)
        _bind(engine, "parent")
        view = _view("data-uri-key")
        engine.ingest(copy.deepcopy(view))
        _require(len(engine._store.get_session_messages("parent")) == 4, "the first ingest stored turn 0's 4 rows")
        view[0]["timestamp"] += 1000
        _bind(engine, "other")
        _bind(engine, "parent")
        engine.ingest(copy.deepcopy(view))
        first = len(engine._store.get_session_messages("parent"))
        # 7 today (re-stamped user row plus a second call and result); a fix stores fewer. Reported, not required.
        _require(first >= 4, f"the re-presentation kept turn 0's rows (stored={first})")
        view.append({"role": "assistant", "content": "parent-final", "timestamp": 5.0, "message_uid": "parent-0-4"})
        engine.ingest(copy.deepcopy(view))
        if repeat_stamp:
            view[0]["timestamp"] += 1000
            _bind(engine, "other")
            _bind(engine, "parent")
            engine.ingest(copy.deepcopy(view))
        engine.shutdown()
        engine = _new_engine(home)
        made.append(engine)
        _bind(engine, "parent")
        engine.on_session_end("parent", copy.deepcopy(view))
        rows = engine._store.get_session_messages("parent")
        recon = dict(getattr(engine, "_last_ingest_reconciliation", {}) or {})
        recon["stored_after_first_cut"] = first
        _require(any(r["role"] == "assistant" and r["content"] == "parent-final" for r in rows),
                 "the later final reply is stored")
        _assert_once(rows, recon)
    finally:
        for instance in reversed(made):
            instance.shutdown()
