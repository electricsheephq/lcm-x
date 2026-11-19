"""#2856: live replay identities of rows that read no state are remembered, and equal the unmemoised ones.

A1 oracle (memo == uncached under interleaved state changes), A2 reads-no-state enforcement, A3 real artifacts the
gate must reject, A5 memo bound. Synthetic data only.
"""
import base64
import builtins
import copy
import functools
import gc
import json
import os
import pathlib
import random
import sys
import threading
import tracemalloc

import pytest

import hermes_lcm.externalize as externalize
import hermes_lcm.ingest_integrity as ingest_integrity
import hermes_lcm.ingest_persisted as ingest_persisted
import hermes_lcm.ingest_protection as ingest_protection
import hermes_lcm.reconcile as reconcile
from hermes_lcm.dag import SummaryNode
from hermes_lcm.reconcile import (
    ReconcileMixin, _HERMES_PRUNED_TOOL_PLACEHOLDER, _freeze_replay_identity_value, _replay_identity_memo_gate,
)
from tests.harness import engine_context
import tests.test_compression_boundary as compression_boundary
from tests.test_issue_10_persisted_output_spillover import _write_output
from tests.test_reconcile_oob_message_block import OOB_BLOCK

FLAGS = [dict(strip_carrier=s, allow_pruned_tool_alias=a) for s in (True, False) for a in (True, False)]
WORD = " lorem ipsum dolor"
PROJECTION = ("[LCM survival fit: this user message (900 tokens) is stored verbatim as store id {sid}; "
              "lcm_expand / lcm_grep reach the full text. (projection 1200/600)]")
NEAR_MISSES = ["[LCM survival fit: that", "[LCM survival fit:this ", "lcm_survival_fi", "[Old tool output cleared",
               "[OUT-OF-BAND USER MESSAG", "<persisted-output", "[Externalized", "[GC'd externalize",
               "[Externalized LCM ingest payload", "[Expand for details", "Expand for details about: x"]
TODO = ("[Your active task list was preserved across context compression]\n- [>] 1. ship it (in_progress)\n\n"
        "and then the next request")
SWITCH = "[Note: model was just switched from a to b.]\n\nhello after the switch"
# No embedding or rollup work: their background jobs read and write on their own threads and schedule, whatever the
# environment's TMPDIR, LANG or embedding default.
QUIET = dict(embeddings_enabled=False, temporal_rollups_enabled=False)
OTHER_THREAD_READS = []  # reads outside the identity body while A2's patches were in place (evidence only)


def _engine(tmp_path):
    return engine_context(tmp_path, window=200_000, conversation_id="conv",
                          large_output_externalization_enabled=True,
                          large_output_externalization_threshold_chars=200,
                          large_output_externalization_path=str(tmp_path / "ext"), **QUIET)


def _memo(engine):
    memo = engine.__dict__.get("_replay_identity_memo")
    return {} if memo is None else memo.entries


def _admitted(msg):
    """The wrapper's own fill condition, restated: a plain dict whose key builds and whose gate is empty."""
    if type(msg) is not dict:
        return False
    try:
        frozen = _freeze_replay_identity_value(msg.get("tool_calls"))
    except (TypeError, RecursionError):
        return False
    content = reconcile.normalize_content_value(msg.get("content")) or ""
    return not _replay_identity_memo_gate(str(msg.get("role") or "unknown"), content, frozen)


def _freezes(row):
    try:
        _freeze_replay_identity_value(row.get("tool_calls"))
        return True
    except (TypeError, RecursionError):
        return False


class _Counter:
    def __init__(self, monkeypatch):
        self.calls, original = 0, ReconcileMixin._message_replay_identity_uncached

        def counted(engine, msg, **kwargs):
            self.calls += not kwargs.get("stored_row")
            return original(engine, msg, **kwargs)
        monkeypatch.setattr(ReconcileMixin, "_message_replay_identity_uncached", counted)
        self.original = original


def _impure(rng, kind, sid):
    """Each impure kind built the way A3's real artifacts carry it (the constants and host shapes)."""
    text = f"row {rng.randrange(50)}" + WORD * rng.randrange(1, 4)
    if kind == "projection":
        return {"role": "user", "content": PROJECTION.format(sid=sid)}
    if kind == "projection_call":
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "pc", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"lcm_survival_fit": PROJECTION.format(sid=sid)})}}]}
    if kind == "pruned":
        return {"role": "tool", "tool_call_id": f"t{rng.randrange(4)}", "name": "read_file",
                "content": _HERMES_PRUNED_TOOL_PLACEHOLDER}
    if kind == "oob":
        return {"role": "tool", "tool_call_id": f"t{rng.randrange(4)}", "content": text + OOB_BLOCK}
    if kind == "persisted":
        return {"role": "tool", "tool_call_id": "t1", "content": "<persisted-output>\nFull output saved to: /x\n"
                "</persisted-output>"}
    if kind == "externalized":
        return {"role": "tool", "tool_call_id": "t2", "content": "[Externalized tool output: tool=read_file; "
                "tool_call_id=t2; chars=900; ref=lcm_000000000000_y.json]"}
    if kind == "gc":
        return {"role": "tool", "tool_call_id": "t2", "content": "[GC'd externalized tool output: tool_call_id=t2; "
                "chars=9; ref=lcm_000000000000_y.json]"}
    if kind == "ingest_placeholder":
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "ip", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"content": "[Externalized LCM ingest payload: 9000 chars;"
                                                                    " ref=lcm_000000000000_y.json]"})}}]}
    return {"role": "user", "content": f"[Recent Summary (d0, node {rng.randrange(1, 6)})]\nEarlier turns.\n"
            "[Expand for details: turns]"
            "\n\n" + text}


def _pure(rng):
    text = f"row {rng.randrange(60)}" + WORD * rng.randrange(0, 5)
    shape = rng.randrange(12)
    if shape == 0:
        return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": "x"}]}
    if shape == 1:
        return {"role": "user", "content": text + "\ud800 lone"}
    if shape == 2:
        args = rng.choice([True, 1, 1.0, None, 0, -0.0, float("nan"), "x"])
        nested = json.dumps({"a": [1, {"b": args}], "z": "y"})
        calls = [{"id": f"c{rng.randrange(3)}", "type": "function",
                  "function": {"name": "f", "arguments": rng.choice([nested, {"k": args}, "  [1, 2]", "{bad"])}}]
        return {"role": "assistant", "content": text, "tool_calls": rng.choice([calls, tuple(calls)])}
    if shape == 3:
        return {"role": "assistant", "content": text, "tool_calls": [rng.choice([{1: "a"}, {"1": "a"}, {True: "a"}])]}
    if shape == 4:
        return {"role": "tool", "tool_call_id": f"t{rng.randrange(4)}", "name": rng.choice(["read_file", "grep"]),
                "content": text}
    if shape == 5:
        return {"role": "user", "content": rng.choice([TODO, SWITCH])}
    if shape == 6:
        return {"role": "user", "content": rng.choice(NEAR_MISSES) + " " + text}
    if shape == 7:
        return {"role": "assistant", "content": text, "tool_calls": [{"f": {1, 2}}]}  # key does not build
    if shape == 8:
        return {"role": rng.choice([None, "", "system"]), "content": rng.choice([None, "", 5, text])}
    if shape == 9:
        return {"role": "tool", "tool_call_id": "t1", "tool_name": "read_file", "content": text}
    return {"role": rng.choice(["user", "assistant"]), "content": text}


def _variants(rng, row):
    """Rows differing only in keys the identity does not read on an admitted row."""
    out = []
    for key, values in (("timestamp", [1.0, 2.0, None]), ("message_uid", ["u1", "u2"]), ("session_id", ["S", "X"]),
                        ("store_id", [3, 4]), ("name", ["alice", "bob"])):
        if rng.random() < 0.3 and not (key == "name" and row.get("role") == "tool"):
            out.append({**copy.deepcopy(row), key: rng.choice(values)})
    return out


def _case(rng, sid):
    rows = []
    for _ in range(rng.randrange(2, 6)):
        if rng.random() < 0.3:
            rows.append(_impure(rng, rng.choice(["projection", "projection_call", "pruned", "oob", "persisted",
                                                 "externalized", "gc", "ingest_placeholder", "summary"]), sid))
        else:
            rows.append(_pure(rng))
        rows += _variants(rng, rows[-1])
        if rng.random() < 0.2:
            rows.append(copy.deepcopy(rows[-1]))  # duplicate row
    return rows


def _mutate(rng, engine, tmp_path, step, rows):
    """Store writes, DAG inserts, payload files, OOB-durable rows, a config swap, a rebind, a session switch.
    The writes are aimed at the case's rows: a durable copy of an OOB row, a source for a pruned row, the DAG
    node a summary row names, the payload an externalized row names."""
    choice = rng.randrange(9)
    oob = [row for row in rows if OOB_BLOCK in str(row.get("content"))]
    pruned = [row for row in rows if row.get("content") == _HERMES_PRUNED_TOOL_PLACEHOLDER]
    if choice == 0:
        row = rng.choice(oob) if oob else {"role": "tool", "tool_call_id": "t0", "content": "x" + OOB_BLOCK}
        engine._store.append(engine._session_id, {"role": "tool", "tool_call_id": row["tool_call_id"],
                                                  "content": row["content"]}, conversation_id="conv")
    elif choice == 7:
        cid = rng.choice(pruned)["tool_call_id"] if pruned else "t1"
        engine._store.append(engine._session_id, {"role": "tool", "tool_call_id": cid, "tool_name": "read_file",
                                                  "content": f"source {step}" + WORD * 30}, conversation_id="conv")
    elif choice in (1, 8):
        sid = engine._store.append(engine._session_id, {"role": "user", "content": "src" + WORD * 400},
                                   conversation_id="conv")
        engine._dag.add_node(SummaryNode(session_id=engine._session_id, depth=0, summary="Earlier turns.",
                                         source_ids=[sid], source_type="messages", expand_hint="turns",
                                         created_at=1.0))
    elif choice == 2:
        directory = tmp_path / "ext"
        directory.mkdir(exist_ok=True)
        (directory / f"lcm_00000000000{step % 10}_y.json").write_text(json.dumps(
            {"kind": "tool_result", "content": "payload " + str(step)}), encoding="utf-8")
    elif choice == 3:
        engine.__dict__.setdefault("_out_of_band_durable_rows", set()).add(str(step))
    elif choice == 4:
        engine._config = copy.copy(engine._config)
    elif choice == 5:
        engine.on_session_start(engine._session_id, platform="cli", context_length=200_000, conversation_id="conv")
    else:
        other = "S2" if engine._session_id == "S" else "S"
        engine.on_session_start(other, platform="cli", context_length=200_000, conversation_id="conv")


def test_a1_memoised_identity_equals_the_uncached_one(tmp_path, monkeypatch):
    counter = _Counter(monkeypatch)
    rng = random.Random(246)
    with _engine(tmp_path) as engine:
        sid = engine._store.append("S", {"role": "user", "content": "projected source" + WORD * 600},
                                   conversation_id="conv")
        configs, cases, rows_checked, hits, admitted_seen = [engine._config], 2000, 0, 0, 0
        for step in range(cases):
            rows = _case(rng, sid)
            for row in rows:
                for flags in FLAGS:
                    expected = counter.original(engine, copy.deepcopy(row), **flags)
                    first = engine._message_replay_identity(row, **flags)
                    assert first == expected, (step, row, flags)
                    if rng.random() < 0.05:
                        _mutate(rng, engine, tmp_path, step, rows)
                    configs.append(engine._config)  # keep every config alive: id() is never recycled here
                    expected = counter.original(engine, copy.deepcopy(row), **flags)
                    before = counter.calls
                    second = engine._message_replay_identity(row, **flags)
                    assert second == expected, (step, row, flags)
                    if _admitted(row) and counter.calls == before:
                        hits += 1
                    if _admitted(row):
                        admitted_seen += 1
                    elif type(row) is dict and _freezes(row):
                        found = (str(row.get("role") or "unknown"),
                                 reconcile.normalize_content_value(row.get("content")) or "",
                                 _freeze_replay_identity_value(row.get("tool_calls")))
                        assert all((key[3], key[4], key[6]) != found for key in _memo(engine)), (step, row)
                    rows_checked += 1
            if rng.random() < 0.5:
                _mutate(rng, engine, tmp_path, step, rows)
                configs.append(engine._config)
    assert rows_checked >= 2000 * 4
    # Every admitted second call with no epoch change between it and its first call is a hit.
    assert hits >= admitted_seen * 0.8, (hits, admitted_seen)
    _evidence("a1_oracle.json", dict(cases=cases, row_flag_checks=rows_checked, admitted=admitted_seen, hits=hits))


class _Touched(BaseException):
    """Raised on any state read; a BaseException so the body's ``except Exception`` cannot swallow it."""


def _caller():
    frame = sys._getframe(2)
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}:{frame.f_code.co_name}"


def _in_identity(owner):
    """True on the identity's own thread while the identity body is on the stack: a read anywhere else (a
    background job, a finalizer, pytest formatting a traceback through linecache) is not the identity's."""
    if threading.get_ident() != owner:
        return False
    body, frame = ReconcileMixin._message_replay_identity_uncached.__code__, sys._getframe(2)
    while frame is not None:
        if frame.f_code is body:
            return True
        frame = frame.f_back
    return False


class _Raiser:
    """Inside the identity body, answers only an identity test for None (no attribute access at all) and raises
    on everything else, recording the caller's frame. Anything else (another thread, or this thread outside the
    body) is handed the real object."""

    def __init__(self, name, log, real, owner):
        for key, value in (("_name", name), ("_log", log), ("_real", real), ("_owner", owner)):
            object.__setattr__(self, key, value)

    def __getattribute__(self, attr):
        if not _in_identity(object.__getattribute__(self, "_owner")):
            return getattr(object.__getattribute__(self, "_real"), attr)
        object.__getattribute__(self, "_log").append((object.__getattribute__(self, "_name"), attr, _caller()))
        raise _Touched(attr)

    def _fail(self, *args, **kwargs):
        object.__getattribute__(self, "_log").append((object.__getattribute__(self, "_name"), "special", _caller()))
        raise _Touched("special")

    __bool__ = __len__ = __iter__ = __contains__ = __getitem__ = __eq__ = __hash__ = __call__ = _fail


def _raising_io(monkeypatch, log, owner):
    """Patch the reads to raise inside the identity body only (recording the caller's frame); any other caller
    (another thread, a finalizer, a traceback formatter's linecache) gets the real function."""
    def make(name, real):
        def fail(*args, **kwargs):
            if not _in_identity(owner):
                OTHER_THREAD_READS.append((name, threading.current_thread().name, _caller()))
                return real(*args, **kwargs)
            log.append(("io", name, _caller()))
            raise _Touched(name)
        return fail
    for attr in ("stat", "open", "read_text", "read_bytes", "exists", "glob", "is_file", "is_dir", "resolve"):
        monkeypatch.setattr(pathlib.Path, attr, make(f"Path.{attr}", getattr(pathlib.Path, attr)))
    monkeypatch.setattr(os, "stat", make("os.stat", os.stat))
    monkeypatch.setattr(builtins, "open", make("open", builtins.open))
    for module in (externalize, ingest_persisted, ingest_integrity, ingest_protection, reconcile):
        for name in dir(module):
            value = getattr(module, name)
            if callable(value) and not isinstance(value, type) and any(
                    part in name for part in ("load", "find_externalized", "recover", "_safe_", "read_")):
                monkeypatch.setattr(module, name, make(f"{module.__name__}.{name}", value))


def test_a2_admitted_rows_read_no_state(tmp_path, monkeypatch):
    rng = random.Random(246)
    with _engine(tmp_path) as engine:
        sid = engine._store.append("S", {"role": "user", "content": "projected source" + WORD * 600},
                                   conversation_id="conv")
        admitted = [row for _ in range(2000) for row in _case(rng, sid) if _admitted(row)]
        normal = [[ReconcileMixin._message_replay_identity_uncached(engine, row, **flags) for flags in FLAGS]
                  for row in admitted]
        store, dag, config, log = engine._store, engine._dag, engine._config, []
        with monkeypatch.context() as patch:
            owner = threading.get_ident()
            engine._store, engine._dag, engine._config = (
                _Raiser(n, log, real, owner) for n, real in (("store", store), ("dag", dag), ("config", config)))
            _raising_io(patch, log, owner)
            gc.disable()  # no finalizer runs inside the window
            try:
                isolated = [[ReconcileMixin._message_replay_identity_uncached(engine, row, **flags) for flags in FLAGS]
                            for row in admitted]
            finally:
                gc.enable()
                engine._store, engine._dag, engine._config = store, dag, config
        assert log == []
        assert isolated == normal
    assert len(admitted) >= 2000
    _evidence("a2_reads_no_state.json", dict(admitted_rows=len(admitted), state_reads=len(log),
                                             outside_identity_reads=sorted(set(OTHER_THREAD_READS))))


def real_artifacts(tmp_path):
    """Each impure kind as a real engine (or the host shape LCM's own tests pin) produces it: (kind, row)."""
    out = []
    with pytest.MonkeyPatch.context() as patch:  # the shared fixture's config, with QUIET added
        patch.setattr(compression_boundary, "LCMConfig", functools.partial(compression_boundary.LCMConfig, **QUIET))
        engine, _compacted, tail, _ids = compression_boundary._summary_carrier_fixture(tmp_path / "carrier")
    try:
        carrier = engine._assemble_context(None, tail)[0]
        assert engine._generated_context_carrier_remainder(carrier) == tail[0]["content"]
        out.append(("summary", dict(carrier)))
    finally:
        engine.shutdown()
    with _engine(tmp_path / "real") as engine:
        source = {"role": "user", "content": "x" * 6000}
        store_id = engine._store.append("S", source, conversation_id="conv")
        projected = engine._survival_projection([source], {id(source): store_id}, 0)[0]
        assert "[LCM survival fit: this " in projected["content"]
        out.append(("projection", projected))
        long = base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(3420))).decode("ascii")
        args = json.dumps({"path": "a.b64", "content": "hdr\n" + "\n".join(
            long[i:i + 76] for i in range(0, len(long), 76)) + "\nend"})
        engine.ingest([{"role": "user", "content": "go"},
                       {"role": "assistant", "content": "", "tool_calls": [
                           {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": args}}]},
                       {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "out " * 2000},
                       {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": "full " * 100},
                       {"role": "assistant", "content": "done"}])
        stored = engine._store.get_session_messages("S")
        payload_row = next(r for r in stored if r.get("tool_calls"))
        out.append(("ingest_placeholder", {"role": "assistant", "content": payload_row["content"],
                                           "tool_calls": json.loads(payload_row["tool_calls"])
                                           if isinstance(payload_row["tool_calls"], str) else payload_row["tool_calls"]}))
        tool_row = next(r for r in stored if r["role"] == "tool" and r["tool_call_id"] == "c1")
        assert tool_row["content"].startswith("[Externalized tool output:")
        out.append(("externalized", {"role": "tool", "tool_call_id": "c1", "name": "write_file",
                                     "content": tool_row["content"]}))
    out.append(("pruned", {"role": "tool", "tool_call_id": "c2", "name": "read_file",
                           "content": _HERMES_PRUNED_TOOL_PLACEHOLDER}))
    out.append(("oob", {"role": "tool", "tool_call_id": "c2", "content": "full output" + OOB_BLOCK}))
    marker = _write_output(tmp_path / "real" / "home" / "cache" / "spillover" / "call_spill.txt")
    out.append(("persisted", {"role": "tool", "tool_call_id": "call_spill", "content": marker}))
    return out


def test_a3_real_artifacts_are_rejected_and_never_remembered(tmp_path):
    artifacts = real_artifacts(tmp_path)
    assert {kind for kind, _ in artifacts} == {"summary", "projection", "ingest_placeholder", "externalized",
                                               "pruned", "oob", "persisted"}
    verdicts = {}
    with _engine(tmp_path / "check") as engine:
        engine._store.append("S", {"role": "tool", "tool_call_id": "c2", "tool_name": "read_file",
                                   "content": "full " * 100}, conversation_id="conv")
        for kind, row in artifacts:
            frozen = _freeze_replay_identity_value(row.get("tool_calls"))
            kinds = _replay_identity_memo_gate(row["role"], reconcile.normalize_content_value(row["content"]) or "",
                                               frozen)
            assert kind in kinds, (kind, kinds)
            verdicts[kind] = sorted(kinds)
            for flags in FLAGS:
                assert engine._message_replay_identity(row, **flags) == \
                    ReconcileMixin._message_replay_identity_uncached(engine, row, **flags)
            assert not any(key[6] == frozen and key[4] == (reconcile.normalize_content_value(row["content"]) or "")
                           for key in _memo(engine)), kind
    _evidence("a3_real_artifacts.json", verdicts)


@pytest.mark.parametrize("row_chars", [500, 16_000])
def test_a5_memo_stays_inside_its_bound(tmp_path, row_chars):
    with _engine(tmp_path) as engine:
        engine._message_replay_identity({"role": "user", "content": "warm"})  # opens storage, makes the epoch
        memo_of = lambda: engine.__dict__["_replay_identity_memo"]  # noqa: E731
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            total, peak_size, peak_entries, peak_held, i = 0, 0, 0, 0, 0
            while total < 100 * 1024 * 1024:
                text = f"{i:012d}" + "y" * (row_chars - 12)
                engine._message_replay_identity({"role": "assistant", "content": text})
                total, i = total + len(text), i + 1
                memo = memo_of()
                peak_size, peak_entries = max(peak_size, memo.size), max(peak_entries, len(memo.entries))
                if i % 512 == 0:
                    peak_held = max(peak_held, tracemalloc.get_traced_memory()[0] - base)
            peak_held = max(peak_held, tracemalloc.get_traced_memory()[0] - base)
        finally:
            tracemalloc.stop()
        assert peak_size <= 32 * 1024 * 1024
        assert peak_entries <= 65_536
        assert peak_held <= 40 * 1024 * 1024, peak_held
        big = {"role": "user", "content": "z" * (1024 * 1024 + 1)}
        engine._message_replay_identity(big)
        assert not any(len(key[4]) > 1024 * 1024 for key in memo_of().entries)
    _evidence(f"a5_bound_{row_chars}.json", dict(inserted_chars=total, rows=i, peak_accounted=peak_size,
                                                 peak_entries=peak_entries, peak_traced_growth=peak_held))


def _evidence(name, payload):
    directory = os.environ.get("LCMX_FIXEDWORK_EVIDENCE")
    if directory:
        pathlib.Path(directory, name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
