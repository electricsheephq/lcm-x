"""I1/I2: the #1511 full-literal cold replay partition and unchanged witnesses.

The 53 synthetic probe cases exercise real ingest. Control fingerprints cover
all ordered message columns, observed stamps, relations and host-UID bindings;
only the home-derived lineage tag is normalized. No model calls are made.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time

import pytest

from tests.harness import engine_context, pin_char_counter
from tests.messages_snapshot import messages_snapshot_columns
from tests.invariants import test_issue_1519_alias_uid_conflict as uid_fixture


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def message(text, stamp, role="user", **keys):
    return dict(role=role, content=text, timestamp=stamp, **keys)


def lineage(root):
    (root / "home").mkdir()
    with sqlite3.connect(root / "home/state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                       [("parent", None, "compression"), ("child", "parent", None), ("other", None, None),
                        ("stub-first-proof", None, "compression"),
                        ("stub-first-proof-child", "stub-first-proof", None)])


def snapshot(engine):
    db = engine._store._conn
    def query(sql):
        cursor = db.execute(sql)
        return [dict(zip([d[0] for d in cursor.description], row)) for row in cursor]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    bindings = query("SELECT * FROM host_uid_bindings ORDER BY rowid") if "host_uid_bindings" in tables else []
    for row in bindings:
        key = row["lineage_key"]
        if ":" in key and len(key.split(":", 1)[0]) == 16:
            row["lineage_key"] = "HOME:" + key.split(":", 1)[1]
    rows = query(f"SELECT {messages_snapshot_columns(db)} FROM messages ORDER BY store_id")
    return dict(rows=rows, stamps=[r.get("observed_at") for r in rows], bindings=bindings,
                relations=query("SELECT * FROM message_relations ORDER BY rowid") if "message_relations" in tables else [])


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def clear(self):
        self.records.clear()

    @property
    def text(self):
        return "\n".join(self.records)


COLD = tuple(f"target-cold-s{s}-a{a}-r{r}" for s in (0, 1) for a in (0, 1) for r in (0, 1))
NARROW_CONTROLS = (
    "users-restamp-0.5", "users-restamp-5", "users-restamp-3600", "all-restamp",
    "tools-restamp", "per-row-offsets", "iso-submicro-drift", "earlier-and-later",
    "null-steer-suffix", "null-tool-suffix", "unknown-incoming-suffix",
    "proofed-later-binding", "proofed-later-alt_stamp", "proofed-later-carried_stamp",
)
CASES = (*COLD,
         *(name.replace("cold", "warm") for name in COLD),
         *("plain-" + kind for kind in ("unchanged", "earlier", "different", "assistant")),
         *("uid-" + kind for kind in uid_fixture.CASES),
         "proof-alt-stamp-False", "proof-alt-stamp-True",
         "proof-pure-restamp-False", "proof-pure-restamp-True",
         "proof-composite-False", "proof-composite-True", "proof-remainder-False", "proof-remainder-True",
         "proof-whitespace", "proof-carried-stamp", "proof-live-recovery", "proof-current-turn-alias",
         "proof-null-canonical-binding", "proof-emitted-carrier",
         "retry-r3-000", "retry-r3-100", "retry-r3-110", "retry-r3-101", "proof-stub-first-adopted-child",
         "relation-undo-resend", "relation-alt-stamp", *NARROW_CONTROLS)
REFERENCE = Path(__file__).with_name("issue_1511_control_hashes.json")


@pytest.fixture(autouse=True)
def deterministic_host(monkeypatch):
    import hermes_lcm.engine as engine_module
    from hermes_lcm import escalation
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "1")
    # The stub helper changes this level; preserve it across same-process tests.
    logger = logging.getLogger("hermes_lcm")
    monkeypatch.setattr(logger, "level", logger.level)
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(time, "time", lambda: 1800000000.)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: "Earlier turns retained.\nExpand for details about: turns")
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **k: ("Summary: " + "lorem ipsum " * 200 + "\nExpand for details about: turns", 1))


def run_case(root, selected):
    root.mkdir(exist_ok=True)
    lineage(root)
    events = []
    def context(root, sid="parent", **kw):
        return engine_context(root, window=kw.pop("window", 128000), session_id=sid, conversation_id="conv", **kw)
    def snap(engine, phase):
        events.append(dict(phase=phase, **snapshot(engine)))
    def case(name, body, category, proof):
        if name == selected:
            body(root, snap)
    def target(root, snap, structured, shared, reply, cold, control=None):
        a = message([{"type":"text", "text":"Synthetic A"}] if structured else "Synthetic A", 20. if shared else 15.)
        b = message("Synthetic B", 20., "assistant" if control == "assistant" else "user")
        initial = [a, *([message("Synthetic reply", 16., "assistant")] if reply else []), b]
        replay = deepcopy(initial)
        replay[-1]["timestamp"] = 10. if control == "earlier" else 20. if control == "unchanged" else 30.
        if control == "different":
            replay[-1]["content"] = "Synthetic different B"
        with context(root) as engine:
            engine.ingest(deepcopy(initial))
            snap(engine, "initial")
            if not cold:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
                engine.ingest(deepcopy(replay))
                snap(engine, "repeat")
        if cold:
            with context(root) as engine:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
                engine.ingest(deepcopy(replay))
                snap(engine, "repeat")
        return dict(target_stamp=30., target_text="Synthetic B")

    for cold in (True, False):
        for structured in (False, True):
            for shared in (False, True):
                for reply in (False, True):
                    case(f"target-{'cold' if cold else 'warm'}-s{int(structured)}-a{int(shared)}-r{int(reply)}",
                         lambda r,s,st=structured,sh=shared,rp=reply,cl=cold: target(r,s,st,sh,rp,cl),
                         "cold_target" if cold else "warm_target", "no occurrence proof")
    for control in ("unchanged", "earlier", "different", "assistant"):
        case("plain-" + control, lambda r,s,c=control: target(r,s,False,False,False,True,c), "plain", control)

    def uid_case(root, snap, kind):
        def prepare(engine):
            import hermes_lcm.host_uid_emit as emit
            with pytest.MonkeyPatch.context() as local:
                if kind == "engine_incoming":
                    local.setattr(emit, "engine_uid", lambda *a: "synthetic-engine-uid")
                return uid_fixture.prepare(engine, kind)
        with context(root) as engine:
            replay = prepare(engine)
            snap(engine, "prepared")
        with context(root) as engine:
            engine.ingest(deepcopy(replay))
            snap(engine, "replay")
            engine.ingest(deepcopy(replay))
            snap(engine, "repeat")
        diagnostic = root / "diagnostic"
        diagnostic.mkdir()
        lineage(diagnostic)
        with context(diagnostic) as engine:
            view = prepare(engine)
        with context(diagnostic) as engine:
            before = snapshot(engine)
            plan = engine._identity_anchor_prematch(view, view, 0)
            edge = {str(i): [r["store_id"] for r in rows] for i, rows in plan.get("matched", {}).items()}
            assert snapshot(engine) == before
        return dict(prematch_edges=edge, case_1519=kind)

    for kind in uid_fixture.CASES:
        case("uid-" + kind, lambda r,s,k=kind: uid_case(r,s,k), "uid", "canonical/alias " + kind)

    def alias(root, snap, cold):
        with context(root) as engine:
            replay = uid_fixture.prepare(engine, "old_view_absent")
            snap(engine, "prepared")
            if not cold:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
        if cold:
            with context(root) as engine:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
    for cold in (False, True):
        case("proof-alt-stamp-" + str(cold), lambda r,s,c=cold: alias(r,s,c), "proof", "R1 recorded alt_stamp")

    def pure_restamp(root, snap, cold):
        with context(root) as engine:
            replay = uid_fixture.prepare(engine, "old_view_absent")[:1]
            before = snapshot(engine)
            snap(engine, "prepared")
            if not cold:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
                assert snapshot(engine)["rows"] == before["rows"]
        if cold:
            with context(root) as engine:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
                assert snapshot(engine)["rows"] == before["rows"]
        return dict(restamped_user_canonical_stamp=20.,incoming_stamp=30.,row_delta=0,
                    occurrence_count=1,proof="durable alt_stamp@30")
    for cold in (False, True):
        case("proof-pure-restamp-" + str(cold), lambda r,s,c=cold: pure_restamp(r,s,c), "proof", "later stamp, durable alt_stamp, no new rows")

    def composite(root, snap, kind, cold):
        head, tail = message("Synthetic held head", 20.), message("Synthetic remainder", 30.)
        with context(root) as engine:
            engine.ingest([head] if kind == "remainder" else [head, tail])
            merged = message(head["content"] + "\n\n" + tail["content"], 20.)
            engine.ingest([merged])
            snap(engine, "witnessed")
            replay = [head, tail] if kind == "remainder" else [merged, message(tail["content"], 20.)]
            if not cold:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
        if cold:
            with context(root) as engine:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
    for kind in ("composite", "remainder"):
        for cold in (False, True):
            case(f"proof-{kind}-{cold}", lambda r,s,k=kind,c=cold: composite(r,s,k,c), "proof", "R2 witnessed composite" if kind == "composite" else "R3 constituent/R5 NULL backfill")

    def whitespace(root, snap):
        with context(root) as engine:
            engine.ingest([message("Synthetic whitespace \n", 20.)])
            snap(engine, "initial")
        with context(root, "child") as engine:
            engine.ingest([message("Synthetic whitespace", 20.)])
            snap(engine, "replay")
    case("proof-whitespace", whitespace, "proof", "R1-ws same-stamp whitespace")

    def carry(root, snap):
        previous, current = message("Synthetic previous", 10.), message("Synthetic current", 20.)
        before, reply = message("Synthetic prior reply", 9., "assistant"), message("Synthetic reply", 21., "assistant")
        with context(root) as engine:
            engine.ingest([before, previous])
            engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv", boundary_reason="compression")
            engine.ingest([before, {**current, "timestamp":10.}, reply])
            snap(engine, "prepared")
        with context(root, "child") as engine:
            engine.ingest([previous, current, reply])
            snap(engine, "replay")
            engine.ingest([previous, current, reply])
            snap(engine, "repeat")
    case("proof-carried-stamp", carry, "uid", "durable carried_stamp + verified ancestry")

    def live_recovery(root, snap):
        from tests.invariants.test_issue_1505_recovery_stamp import recover
        with context(root) as engine:
            view, original = recover(engine, text="Synthetic recovery request " * 500)
            before = snapshot(engine)
            snap(engine, "prepared")
            view[-1]["timestamp"] = 30.
            engine.ingest(deepcopy(view))
            view[-1]["content"] = original["content"]
            engine.ingest(view)
            snap(engine, "replay")
            assert snapshot(engine)["rows"] == before["rows"]
        return dict(restamped_user_canonical_stamp=20.,incoming_stamp=30.,row_delta=0,
                    occurrence_count=1,proof="same live recovery object")
    case("proof-live-recovery", live_recovery, "proof", "R5 live recovery-object receipt")

    def recent(root, snap):
        head = message("Synthetic current head", 20.)
        whole = message("Synthetic current head\n\nSynthetic new remainder", 30.)
        with context(root) as engine:
            engine.ingest([head])
            engine.ingest([head, whole])
            snap(engine, "prepared")
            engine.ingest([{**whole, "timestamp":20.}])
            snap(engine, "replay")
    case("proof-current-turn-alias", recent, "proof", "R5 recent current-turn stamp witness")

    def null_uid(root, snap):
        row = message("Synthetic NULL binding", None, message_uid="host-null")
        with context(root) as engine:
            engine.ingest([message("Synthetic old prefix", 10.), row])
            snap(engine, "prepared")
            engine.ingest([message("Synthetic changed prefix", 11.), {**row, "timestamp":30.}])
            snap(engine, "replay")
            assert sum(r["content"] == row["content"] for r in engine._store.get_session_messages("parent")) == 1
    case("proof-null-canonical-binding", null_uid, "uid", "R5 NULL stamp + canonical UID")

    def emitted_carrier(root, snap):
        import tests.test_issue_650_survival_fit_keeps_summary as f
        with ExitStack() as stack, pytest.MonkeyPatch.context() as local:
            host = f.host.__wrapped__(local)
            def bind(path, window=24000):
                return stack.enter_context(engine_context(path, window=window, session_id="S", platform="telegram",
                    conversation_id="conv", fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=.75,
                    threshold_full_sweep_enabled=True))
            local.setattr(f, "_engine", bind)
            engine, fitted, rest = f._held_fit_after_inplace_boundary(root, host, local)
            snap(engine, "prepared")
            fitted[0]["timestamp"] = 40000.
            engine.ingest([*fitted, message("Synthetic final reply", 30001., "assistant")])
            snap(engine, "replay")
            assert sum(r["content"] == rest for r in engine._store.get_session_messages("S")) == 1
    case("proof-emitted-carrier", emitted_carrier, "proof", "R8 emitted carrier descriptor/retained source")

    def retry(root, snap, compacted, cold, remapped):
        from tests.test_compression_boundary import _host_merge_consecutive_users
        with context(root, fresh_tail_count=2, leaf_chunk_tokens=16) as engine:
            view = [message("Synthetic alpha", 1., message_uid="host-1"), message("Synthetic reply1", 2., "assistant", message_uid="host-2"),
                    message("Synthetic same text", 3., message_uid="host-3"), message("Synthetic reply2", 4., "assistant", message_uid="host-4")]
            engine.ingest(deepcopy(view))
            if compacted:
                view = engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
                assert engine._compress_commit_proof
            if remapped:
                view = _host_merge_consecutive_users(view)
            snap(engine, "prepared")
            replay = deepcopy(view)
            idx = len(replay)-1 if compacted else len(replay)-2
            replay[idx].update(timestamp=50., message_uid="host-replacement")
            if not compacted:
                replay.append(message("Synthetic next", 51., message_uid="host-next"))
            if not cold:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
        if cold:
            with context(root, fresh_tail_count=2, leaf_chunk_tokens=16) as engine:
                engine.ingest(deepcopy(replay))
                snap(engine, "replay")
    for compacted,cold,remapped in ((False,False,False),(True,False,False),(True,True,False),(True,False,True)):
        case(f"retry-r3-{int(compacted)}{int(cold)}{int(remapped)}", lambda r,s,c=compacted,cl=cold,m=remapped: retry(r,s,c,cl,m), "proof", "R3 retained/durable/remapped metadata")

    def stub(root, snap):
        from tests.test_issue_1116_stub_first_exit_commit_proof import Host, CONTEXT, committed_session, grow_over_threshold, stub_first_result
        capture = Capture()
        logger = logging.getLogger("hermes_lcm")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        with ExitStack() as stack, pytest.MonkeyPatch.context() as local:
            def bind(host, **start):
                engine = stack.enter_context(engine_context(root, window=CONTEXT, session_id=host.sid,
                    conversation_id="stub-first-proof-conv", fresh_tail_count=host.tail_count,
                    fresh_tail_max_tokens=0, leaf_chunk_tokens=8000, context_threshold=.75,
                    large_output_externalization_path=str(root/"externalized")))
                return engine
            local.setattr(Host, "_bind", bind)
            host = Host(root, 7)
            committed_session(host, "summary")
            grow_over_threshold(host)
            try:
                result, before = stub_first_result(host, capture)
            except AssertionError as exc:
                snap(host.engine, "stub-construction-failure")
                raise AssertionError(str(exc) + "; status=" + str(host.engine._last_compression_status)
                                     + "; logs=" + repr(capture.records[-8:])) from exc
            host.rotate(result, before, "stub-first-proof-child")
            snap(host.engine, "adopted")
            assert host.engine._durable_commit_proof_payload()["session_id"] == host.sid
            host.cold_rebind()
            snap(host.engine, "replay")
            assert host.stored_rows() == 0
        logger.removeHandler(capture)
    case("proof-stub-first-adopted-child", stub, "adopted", "R8 adopted stub-first durable proof")

    def relation_control(root, snap, kind):
        q = message("Synthetic earlier reply", 940., "assistant")
        p = message("Synthetic original request", 1000.)
        r = message("Synthetic revised request", 1000.)
        with context(root) as engine:
            engine.ingest(deepcopy([q, p]))
            if kind == "undo-resend":
                engine.ingest(deepcopy([q, r]))
                replay = [q, p, {**r, "timestamp": 1010.}]
                row = engine._store.get_session_messages("parent")[-1]
                relation_kind = "carried_stamp"
            else:
                row = engine._store.get_session_messages("parent")[-1]
                # A prior witnessed alternate stamp explains this row, even when
                # the incoming stamp is later than that witness.
                engine._store.add_message_relations([
                    [(row["store_id"], "alt_stamp", None, 0, 1005.)]])
                replay = [q, {**p, "timestamp": 1010.}]
                relation_kind = "alt_stamp"
            assert engine._store.get_message_relations([row["store_id"]], relation_kind)
            snap(engine, "prepared")
        with context(root) as engine:
            before = snapshot(engine)
            engine.ingest(deepcopy(replay))
            snap(engine, "replay")
            assert snapshot(engine) == before
            engine.ingest(deepcopy(replay))
            snap(engine, "repeat")
            assert snapshot(engine) == before

    for kind in ("undo-resend", "alt-stamp"):
        case("relation-" + kind, lambda r,s,k=kind: relation_control(r,s,k),
             "proof", "durable stamp-explaining relation")

    def narrow_control(root, snap, kind):
        initial = [message("Synthetic A", 995.), message("Synthetic reply A", 996., "assistant"),
                   message("Synthetic B", 1000.), message("Synthetic reply B", 1001., "assistant")]
        if kind in ("tools-restamp", "null-tool-suffix"):
            initial = [message("Synthetic A", 995.),
                       message("", 996., "assistant", tool_calls=[dict(id="synthetic-call", type="function",
                               function=dict(name="read_file", arguments="{}"))]),
                       message("Synthetic tool result", 997., "tool", tool_call_id="synthetic-call", tool_name="read_file"),
                       message("Synthetic done", 998., "assistant")]
            if kind == "null-tool-suffix":
                for row in initial[1:]:
                    row["timestamp"] = None
        if kind == "null-steer-suffix":
            initial = [message("Synthetic A", 995.), message("Synthetic B", 1000.),
                       message("Synthetic /steer", None), message("Synthetic steer reply", None, "assistant")]
        if kind.startswith("proofed-later-"):
            initial = [message("Synthetic A", 995.), message("Synthetic B", 1000.), message("Synthetic C", 1001.)]
            if kind.endswith("binding"):
                initial[-1]["message_uid"] = "synthetic-canonical-c"
        if kind == "iso-submicro-drift":
            for i, row in enumerate(initial):
                row["timestamp"] = 1730000000.1234567 + i
        replay = deepcopy(initial)
        if kind.startswith("users-restamp-"):
            for row in replay:
                if row["role"] == "user":
                    row["timestamp"] += float(kind.rsplit("-", 1)[1])
        elif kind in ("all-restamp", "tools-restamp", "per-row-offsets"):
            for i, row in enumerate(replay):
                row["timestamp"] += 5. if kind != "per-row-offsets" else (i + 1) * .25
        elif kind == "iso-submicro-drift":
            for row in replay:
                row["timestamp"] = datetime.fromtimestamp(row["timestamp"], timezone.utc).isoformat(timespec="microseconds")
        elif kind == "earlier-and-later":
            replay[0]["timestamp"] -= 1.
            replay[2]["timestamp"] += 10.
        elif kind == "null-steer-suffix":
            replay[1]["timestamp"] += 10.
            replay[2]["timestamp"], replay[3]["timestamp"] = 1002., 1003.
        elif kind == "null-tool-suffix":
            for i, row in enumerate(replay):
                row["timestamp"] = 1010. + i
        elif kind == "unknown-incoming-suffix":
            replay[2]["timestamp"] += 10.
            replay[3]["timestamp"] = None
        else:
            replay[1]["timestamp"] += 10.
            replay[2]["timestamp"] += 10.
            replay[2].pop("message_uid", None)
        with context(root) as engine:
            engine.ingest(deepcopy(initial))
            if kind in ("proofed-later-alt_stamp", "proofed-later-carried_stamp"):
                sid = engine._store.get_session_messages("parent")[-1]["store_id"]
                engine._store.add_message_relations([[(sid, kind.removeprefix("proofed-later-"), None, 0, 1005.)]])
            snap(engine, "prepared")
        with context(root) as engine:
            before = snapshot(engine)
            engine.ingest(deepcopy(replay))
            snap(engine, "replay")
            after = snapshot(engine)
            assert len(after["rows"]) == len(before["rows"])
            if kind not in ("null-steer-suffix", "null-tool-suffix"):
                assert after == before
            engine.ingest(deepcopy(replay))
            snap(engine, "repeat")
            assert snapshot(engine) == after
            continued = [*replay, message("Synthetic follow-up", 1100.), message("Synthetic follow-up reply", 1101., "assistant")]
            engine.ingest(deepcopy(continued))
            snap(engine, "warm")
            assert snapshot(engine)["rows"][:len(after["rows"])] == after["rows"]
            assert len(snapshot(engine)["rows"]) == len(after["rows"]) + 2
            engine.ingest(deepcopy(continued))
            snap(engine, "warm-repeat")
            assert snapshot(engine) == {k: v for k, v in events[-2].items() if k != "phase"}
    for kind in NARROW_CONTROLS:
        case(kind, lambda r,s,k=kind: narrow_control(r,s,k), "control", "exactly one difference and clean suffix")

    assert events, selected
    return events


@pytest.mark.parametrize("case", CASES)
def test_cold_replay_partition(tmp_path, case, record_property):
    events = run_case(tmp_path, case)
    fingerprints = {event["phase"]: digest(event) for event in events}
    record_property("snapshots", json.dumps(events, sort_keys=True))
    if case in COLD:
        initial, replay, repeat = events
        added = replay["rows"][len(initial["rows"]):]
        assert len(added) == 1 and added[0]["observed_at"] == 30.
        assert added[0]["role"] == "user" and added[0]["content"] == "Synthetic B"
        assert replay["rows"][:len(initial["rows"])] == initial["rows"]
        assert {k: v for k, v in replay.items() if k != "phase"} == {
            k: v for k, v in repeat.items() if k != "phase"}
    elif case.startswith("uid-") and case[4:] in uid_fixture.V2_UID_CONFLICTS:
        # PR-B's positive v2 row proof supersedes the old new-writer UID
        # observation expectation; the actual BASE-writer controls live in #1519.
        prepared, replay, repeat = events
        assert fingerprints["prepared"] == json.loads(REFERENCE.read_text())[case]["prepared"]
        before = prepared["rows"]
        assert replay["rows"][:len(before)] == before
        added = replay["rows"][len(before):]
        assert [(r["role"], r["observed_at"]) for r in added] == [
            ("user", 20. if case == "uid-non_alias" else 30.), ("assistant", 51.)]
        assert {k: v for k, v in replay.items() if k != "phase"} == {
            k: v for k, v in repeat.items() if k != "phase"}
    else:
        assert fingerprints == json.loads(REFERENCE.read_text())[case]


@pytest.mark.parametrize("control", ["unstamped-tail", "canonical-binding", "partial-tail"])
def test_narrowing_controls(tmp_path, control):
    lineage(tmp_path)
    initial = [message("Synthetic A", 995.), message("Synthetic B", 1000.)]
    if control == "unstamped-tail":
        initial.append(message("Synthetic reply2", None, "assistant"))
    if control == "canonical-binding":
        initial[1]["message_uid"] = "host-canonical-b"
    replay = deepcopy(initial)
    replay[-2 if control == "unstamped-tail" else -1]["timestamp"] = 1010.
    for row in replay:
        row.pop("message_uid", None)
    if control == "partial-tail":
        replay.append(message("Synthetic new reply", 1011., "assistant"))
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(initial))
        before = snapshot(engine)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(replay))
        after = snapshot(engine)
        if control == "partial-tail":
            assert after["rows"][:2] == before["rows"]
            assert len(after["rows"]) == 3
            assert after["rows"][-1]["content"] == "Synthetic new reply"
            assert after["relations"] == before["relations"] and after["bindings"] == before["bindings"]
        else:
            assert after == before
        engine.ingest(deepcopy(replay))
        assert snapshot(engine) == after


def test_retired_proof_recursion_keeps_replayed_prefix(tmp_path, monkeypatch):
    """Exercise #671's recursive reconcile and its reason-prefix consumer.

    Force only the first match miss and the retirement decision; the recursive
    matcher, #1511 rewind, prefix check, occurrence audit and append are real.
    Existing retirement-shape tests cover when retirement is allowed.
    """
    lineage(tmp_path)
    initial = [message("Synthetic A", 995.), message("Synthetic B", 1000.)]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(initial))
        before = snapshot(engine)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        original = engine._find_reconciled_cursor_for_store_tail
        matches = []
        def first_miss(*args, **kwargs):
            matches.append(True)
            return None if len(matches) == 1 else original(*args, **kwargs)
        monkeypatch.setattr(engine, "_find_reconciled_cursor_for_store_tail", first_miss)
        monkeypatch.setattr(engine, "_retire_stale_durable_proof", lambda messages: True)
        engine.ingest([initial[0], {**initial[1], "timestamp": 1010.}])
        after = snapshot(engine)
        assert len(matches) == 2
        assert len(after["rows"]) == 3
        assert after["rows"][:2] == before["rows"]
        assert after["rows"][2]["content"] == "Synthetic B"
        assert after["rows"][2]["observed_at"] == 1010.
        reason = engine._last_ingest_reconciliation["reason"]
        assert reason.startswith("replayed ") and reason.endswith("past a stale commit proof")


def test_same_stamp_full_replay_keeps_cursor(tmp_path):
    """The strict-later guard must avoid even a needless same-stamp rewind."""
    lineage(tmp_path)
    view = [message("Synthetic A", 995.), message("Synthetic B", 1000.)]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(view))
        before = snapshot(engine)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        assert engine._reconcile_ingest_cursor_from_store(deepcopy(view)) == len(view)
        assert snapshot(engine) == before


def test_uid_off_ignores_existing_binding(tmp_path, monkeypatch, record_property):
    """Bindings captured earlier cannot suppress a UID-less resend in off mode."""
    lineage(tmp_path)
    initial = [message("Synthetic A", 995.), message("Synthetic B", 1000., message_uid="synthetic-bound-b")]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(initial))
        before = snapshot(engine)
        assert before["bindings"]
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    replay = [initial[0], message("Synthetic B", 1010.)]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        reads = []
        engine._store._conn.set_trace_callback(lambda sql: reads.append(sql) if "host_uid_bindings" in sql.lower() else None)
        cursor = engine._reconcile_ingest_cursor_from_store(deepcopy(replay))
        record_property("off_binding_reads", len(reads))
        assert not reads
        assert cursor == 1
        engine.ingest(deepcopy(replay))
        assert not reads
        engine._store._conn.set_trace_callback(None)
        after = snapshot(engine)
        assert len(after["rows"]) == 3
        assert after["rows"][:2] == before["rows"]
        assert after["rows"][-1]["observed_at"] == 1010.
        assert after["bindings"] == before["bindings"]
        engine.ingest(deepcopy(replay))
        assert snapshot(engine) == after


@pytest.mark.parametrize("prefix", ["none", "clean", "unknown-stored", "unknown-incoming"])
def test_later_occurrence_reason_and_clean_suffix(tmp_path, prefix):
    lineage(tmp_path)
    initial = ([] if prefix == "none" else [message("Synthetic prefix", None if prefix == "unknown-stored" else 995.)])
    index = len(initial)
    initial += [message("Synthetic resend", 1000.), message("Synthetic reply", 1001., "assistant")]
    replay = deepcopy(initial)
    replay[index]["timestamp"] = 1010.
    if prefix == "unknown-stored":
        replay[0]["timestamp"] = 995.
    if prefix == "unknown-incoming":
        replay[0]["timestamp"] = None
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(initial))
        before = snapshot(engine)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        assert engine._reconcile_ingest_cursor_from_store(deepcopy(replay)) == index
        assert engine._last_ingest_reconciliation["reason"] == "replayed durable tail before a later user occurrence"
        engine.ingest(deepcopy(replay))
        after = snapshot(engine)
        expected = deepcopy(before["rows"])
        if prefix == "unknown-stored":
            expected[0].update(observed_at=995., observed_at_source="host_message_timestamp")
        assert after["rows"][:len(expected)] == expected
        assert len(after["rows"]) == len(before["rows"]) + 1
        assert after["rows"][-1]["observed_at"] == 1010.
        engine.ingest(deepcopy(replay))
        assert snapshot(engine) == after
