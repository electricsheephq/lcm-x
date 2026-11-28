"""Natural re-ownership and mixed-window replay (#1250, I1/I2/I5/I14)."""

import json
import sqlite3

import pytest

from hermes_lcm import engine as engine_module, message_patterns
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from tests.engine_helpers import _FakeTimeoutRegexEngine
from tests.invariants.test_issue_1250_publication_reown import (
    EARLY, GATEWAY, LATER, SESSION, _bind_gateway, _digests, _hook_for, _rows,
    open_engine as _open_engine,
)

open_engine = _open_engine
pytestmark = pytest.mark.issue(1250)
REOWN_KEY = f"reowned_session_rows:{SESSION}"


@pytest.fixture
def uid_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES (?, NULL, NULL)", (SESSION,))


def _reown_then_append(open_engine, early):
    writer = open_engine(fresh_tail_count=0)
    _start(writer)
    writer.ingest(early)
    writer.shutdown()
    gateway = open_engine(fresh_tail_count=0)
    _bind_gateway(gateway)
    gateway.compress(early, force=True)
    assert gateway.last_compression_status == "compacted"
    assert gateway._store.read_metadata_json(REOWN_KEY) == [row["store_id"] for row in _rows(gateway)]
    gateway.ingest(early + LATER[:2])
    assert len(_rows(gateway)) == len(early) + 2
    gateway.shutdown()


def _cold_replay(open_engine, raw, passes, *, fresh=0):
    for _ in range(passes):
        engine = open_engine(fresh_tail_count=fresh)
        _start(engine)
        engine.ingest(raw)
        yield _rows(engine)
        engine.shutdown()


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("mode", ["off", "shadow", "on", "on-ignore"])
@pytest.mark.parametrize("shape", ["unstamped", "new-uids", "same-uids", "stamps"])
def test_provisional_rebind_suffix_requires_occurrence_evidence(open_engine, monkeypatch, mode, shape):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on" if mode == "on-ignore" else mode)
    if mode == "on-ignore" and message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    settings = dict(fresh_tail_count=0, ignore_message_patterns=["^HEARTBEAT"] if mode == "on-ignore" else [])
    writer = open_engine(**settings)
    _start(writer)
    writer.ingest(EARLY)
    emitted = writer.compress(EARLY, force=True)
    late = [dict(message) for message in LATER[:2]]
    if shape == "stamps":
        late = [dict(row, timestamp=200 + index) for index, row in enumerate(late)]
    if shape in {"new-uids", "same-uids"}:
        late = [dict(row, message_uid=f"late-{index}") for index, row in enumerate(late)]
    writer.ingest(emitted + late)
    writer.shutdown()
    if shape == "new-uids":
        late = [dict(row, message_uid=f"new-{index}") for index, row in enumerate(late)]
    copies = shape != "stamps" and (shape != "same-uids" or mode != "on")
    for pass_number in range(1, 7):
        gateway = open_engine(**settings)
        _bind_gateway(gateway)
        before = _rows(gateway)
        gateway.ingest(emitted + late)
        stored = _rows(gateway)
        assert stored[:len(before)] == before
        repeats = pass_number if copies else 0
        if shape == "new-uids" and mode == "on":
            repeats = 1
        assert _digests(stored) == _digests(EARLY + late + late * repeats)
        gateway.shutdown()


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("other_binding", [False, True], ids=["different-uid", "canonical-uid-on-other-row"])
def test_provisional_rebind_suffix_stops_at_uid_conflict_even_with_equal_stamp(
    open_engine, monkeypatch, mode, other_binding,
):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    early = [dict(EARLY[0], message_uid="early-0"), *EARLY[1:]]
    late = [dict(row, timestamp=200 + index) for index, row in enumerate(LATER[:2])]
    if not other_binding:
        late[1]["message_uid"] = "original"
    writer = open_engine(fresh_tail_count=0)
    _start(writer)
    writer.ingest(early)
    emitted = writer.compress(early, force=True)
    writer.ingest(emitted + late)
    writer.shutdown()

    gateway = open_engine(fresh_tail_count=0)
    _bind_gateway(gateway)
    repeated = dict(late[1], message_uid="early-0" if other_binding else "new")
    gateway.ingest(emitted + [late[0], repeated])
    assert gateway._last_ingest_reconciliation["cursor"] == len(emitted) + 1
    assert _digests(_rows(gateway)) == _digests(early + late + [repeated])


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("stamp_first", [True, False], ids=["stamp-first", "uid-first"])
def test_receipt_row_cannot_prove_two_occurrences(open_engine, stamp_first):
    original = dict(EARLY[0], timestamp=100, message_uid="early-0")
    _reown_then_append(open_engine, [original])
    with_stamp = dict(EARLY[0], timestamp=100)
    with_uid = dict(EARLY[0], message_uid="early-0")
    raw = [with_stamp, with_uid] if stamp_first else [with_uid, with_stamp]
    assert [sum(row["content"] == original["content"] for row in stored)
            for stored in _cold_replay(open_engine, raw, 5)] == [2] * 5


@pytest.mark.usefixtures("uid_lineage")
def test_receipt_claim_does_not_hide_pruned_tool_source(open_engine):
    call_id = "r12-call"
    early = [dict(EARLY[0], timestamp=100), {
        "role": "assistant", "content": "", "timestamp": 101, "message_uid": "early-1",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}],
    }, {"role": "tool", "content": "An original result. " * 20, "tool_call_id": call_id, "message_uid": "early-2"}]
    _reown_then_append(open_engine, early)
    raw = early[:2] + [dict(early[2], content=_HERMES_PRUNED_TOOL_PLACEHOLDER)]
    # Keeping the claimed source visible lets the resolver recognize the tool
    # group, while the post-filter still prevents spending the assistant twice.
    assert [len(stored) for stored in _cold_replay(open_engine, raw, 6)] == [5] * 6


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("proof", ["partial-stamp", "uid-call", "uid-result", "both-uid", "stamps"])
def test_receipt_checks_tool_segment_after_combining_proofs(open_engine, proof):
    call_id = "receipt-tool-call"
    original = [dict(EARLY[0], timestamp=100), {
        "role": "assistant", "content": "", "timestamp": 101,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}],
    }, {
        "role": "tool", "content": "Original tool result.", "tool_call_id": call_id, "timestamp": 102,
    }]
    uid_indexes = {"uid-call": [1], "uid-result": [2], "both-uid": [1, 2]}.get(proof, [])
    for index in uid_indexes:
        original[index]["message_uid"] = f"receipt-tool-{index}"
    _reown_then_append(open_engine, original)

    raw = [dict(row) for row in original]
    for index in uid_indexes:
        raw[index].pop("timestamp")
    if proof == "partial-stamp":
        raw[2].pop("timestamp")
        raw[2]["content"] = "New tool result."

    resumed = open_engine(fresh_tail_count=0)
    _start(resumed)
    before = _rows(resumed)
    if proof != "partial-stamp":
        # A separate durable-tail matcher can also replay these bytes. Verify
        # that the receipt itself can prove the group with both kinds of evidence.
        assert resumed._cursor_from_reowned_session_rows(raw, before, len(before)) == (
            len(raw), {index: row["store_id"] for index, row in enumerate(before[:len(raw)])},
        )
    resumed.ingest(raw)
    after = _rows(resumed)
    assert after[:len(before)] == before
    added = after[len(before):]
    if proof == "partial-stamp":
        assert [(row["conversation_id"], row["role"]) for row in added] == [
            (SESSION, "assistant"), (SESSION, "tool"),
        ]
        assert added[0]["tool_calls"][0]["id"] == added[1]["tool_call_id"] == call_id
        assert added[1]["content"] == raw[2]["content"]
    else:
        assert added == []


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("mixed, expected", [(True, [8, 9, 10, 11, 12]), (False, [6, 8, 10, 12, 14])],
                         ids=["mixed-replay", "same-uids-replay-plus-pair"])
def test_receipt_uid_replay_matches_ga_sequence(open_engine, mixed, expected):
    early = [dict(row, message_uid=f"early-{i}") for i, row in enumerate(EARLY if mixed else EARLY[:2])]
    _reown_then_append(open_engine, early)
    if mixed:
        raw = early[:2] + [dict(EARLY[2], message_uid="changed"), EARLY[3]]
    else:
        raw = early + early
    assert [len(stored) for stored in _cold_replay(open_engine, raw, 5)] == expected


@pytest.mark.usefixtures("uid_lineage")
def test_receipt_uid_replay_with_stall_rows_matches_ga_sequence(open_engine):
    early = [dict(row, message_uid=f"early-{i}") for i, row in enumerate(EARLY)]
    writer, _, _ = _mixed(open_engine, early=early)
    held = [dict(role="assistant", content="Reply stored during the stall.", message_uid="held-0"),
            dict(role="user", content="Request stored during the stall.", message_uid="held-1")]
    writer.ingest(writer.compress(early) + held)
    writer.shutdown()
    raw = early + held + [dict(role="assistant", content="New reply after restart."),
                          dict(role="user", content="New request after restart.")]
    counts = []
    for _ in range(5):
        engine = open_engine()
        _start(engine)
        engine.compress(raw, force=True)
        counts.append(len(_rows(engine)))
        engine.shutdown()
    assert counts == [13, 15, 17, 19, 21]


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("receipt", [None, "missing", "foreign", "rotation", "duplicate"])
def test_unusable_receipt_does_not_widen_uid_resolution(open_engine, monkeypatch, receipt):
    early = [dict(EARLY[0], timestamp=100, message_uid="early-0")]
    if receipt is not None:
        _reown_then_append(open_engine, early)
    cold = open_engine(fresh_tail_count=0)
    session = "S2-1250" if receipt == "rotation" else SESSION
    cold.on_session_start(session, platform="gui", conversation_id=SESSION, context_length=200_000)
    if receipt == "rotation":
        rotated = cold._store.append(session, early[0], conversation_id=GATEWAY)
        cold._store.write_metadata_json([f"reowned_session_rows:{session}"], json.dumps([rotated]))
    if receipt in {"missing", "foreign", "duplicate"}:
        first_id = _rows(cold)[0]["store_id"]
        invalid_id = max(row["store_id"] for row in _rows(cold)) + 1
        if receipt == "foreign":
            invalid_id = cold._store.append("foreign-session", early[0], conversation_id=SESSION)
        elif receipt == "duplicate":
            invalid_id = first_id
        cold._store.write_metadata_json([REOWN_KEY], json.dumps([first_id, invalid_id]))
    seen = []
    resolve = cold._host_uid_resolve

    def spy(*args, **kwargs):
        seen.append(kwargs.get("reowned", ()))
        return resolve(*args, **kwargs)

    monkeypatch.setattr(cold, "_host_uid_resolve", spy)
    cold.ingest(early)
    assert seen and not any(seen)
    assert cold._reowned_receipt_store_ids() == frozenset()
    if receipt is None:
        assert cold._last_ingest_reconciliation == {"action": "none", "reason": "not run"}
    elif receipt != "rotation":
        # The ordinary ledger can still prove this stamped row in round 11.
        assert cold._last_ingest_reconciliation["reason"] == "ledger durable cursor (ingest vector)"
        assert cold._last_ingest_reconciliation["cursor"] == 1


@pytest.mark.usefixtures("uid_lineage")
@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
def test_receipt_gate_uses_two_sql_variables_for_1200_rows(open_engine, monkeypatch, mode):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    engine = open_engine(fresh_tail_count=0)
    _start(engine)
    early = [dict(role="user", content=f"Synthetic row {i}", timestamp=100 + i) for i in range(1200)]
    with engine._store.atomic(immediate=True):
        ids = [engine._store.append(SESSION, message, conversation_id=GATEWAY) for message in early]
        engine._store.write_metadata_json([REOWN_KEY], json.dumps(ids))
    engine.shutdown()
    cold = open_engine(fresh_tail_count=0)
    _start(cold)
    conn = cold._store.connection
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    statements = []
    conn.set_trace_callback(statements.append)
    cold.ingest(early)
    assert cold._reowned_receipt_store_ids() == frozenset(ids)
    assert [row["store_id"] for row in _rows(cold)] == ids
    assert sum("FROM json_each(" in sql and "JOIN messages m ON m.store_id" in sql for sql in statements) == 1


@pytest.fixture
def differential_base(request):
    return request.path.name.startswith("test_zz_1250_r7_")


def _start(writer):
    writer.on_session_start(SESSION, platform="gui", conversation_id=SESSION, context_length=200_000)


def _step(engine, name, view, action, new=(), *, copies=(), may_reown=False):
    before_rows = _rows(engine)
    before = _digests(before_rows)
    expected = before + _digests(new)
    result = action()
    after_rows = _rows(engine)
    after = _digests(after_rows)
    delta = after - expected
    print(json.dumps({
        "step": name, "binding": engine._conversation_id, "rows": [before.total(), after.total()],
        "view": [[*key, count] for key, count in sorted(_digests(view).items())],
        "stored": [[*key, count] for key, count in sorted(after.items())],
        "extra": [[*key, count] for key, count in sorted(delta.items())],
    }))
    assert after == expected + _digests(copies)
    after_by_id = {row["store_id"]: row for row in after_rows}
    for row in before_rows:
        current = after_by_id[row["store_id"]]
        assert row == (dict(current, conversation_id=row["conversation_id"]) if may_reown else current)
    return result


def _mixed(open_engine, *, sequential=False, early=EARLY):
    writer, gateway = open_engine(), open_engine()
    _start(writer)
    _step(writer, "early ingest", early, lambda: writer.ingest(list(early)), early)
    if sequential:
        writer.shutdown()
    _bind_gateway(gateway)
    full = early + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    before = _rows(gateway)
    gateway_view = _step(gateway, "gateway compress", full, lambda: gateway.compress(full), may_reown=True)
    expected_ids = ([row["store_id"] for row in before if row["conversation_id"] == SESSION]
                    if gateway.last_compression_status == "compacted" else None)
    receipt = gateway._store.read_metadata_json(REOWN_KEY)
    assert (sorted(receipt) if receipt is not None else None) == expected_ids
    owners = [GATEWAY] * len(full) if expected_ids else [SESSION] * len(early) + [GATEWAY] * len(LATER)
    assert [row["conversation_id"] for row in _rows(gateway)] == owners
    return writer, gateway, gateway_view


@pytest.mark.parametrize("mode", ["off", "on"])
def test_new_unstamped_pair_after_gateway_turns_is_stored(open_engine, monkeypatch, mode):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    writer, gateway = open_engine(fresh_tail_count=0), open_engine(fresh_tail_count=0)
    original = EARLY[:2]
    _start(writer)
    writer.ingest(list(original))
    writer.shutdown()
    _bind_gateway(gateway)
    gateway.compress(list(original), force=True)
    assert gateway.last_compression_status == "compacted"
    gateway.ingest(original + LATER[:2])
    assert _digests(_rows(gateway)) == _digests(original + LATER[:2])
    gateway.shutdown()

    resumed = open_engine(fresh_tail_count=0)
    _start(resumed)
    new_pair = [dict(message) for message in original]
    # v0.30.4 stores both new occurrences (4 -> 6) in both modes.
    _step(resumed, "new unstamped pair", new_pair, lambda: resumed.ingest(new_pair), new_pair)


@pytest.mark.parametrize("shape", [
    "same-uids", "all-new-uids", "middle-new-uid", "uid-bound-to-other-row", "version-on-other-row",
])
def test_reowned_replay_checks_host_uid_bindings(open_engine, tmp_path, monkeypatch, shape):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES (?, NULL, NULL)", (SESSION,))

    writer, gateway = open_engine(), open_engine()
    original = [dict(row, message_uid=f"early-{i}") for i, row in enumerate(EARLY)]
    if shape == "uid-bound-to-other-row":
        original[2].pop("message_uid")
    _start(writer)
    writer.ingest(original)
    lineage, _ = writer._host_uid_lineage_key()
    assert lineage is not None
    stored = _rows(writer)
    bound = writer._store.host_uid_uids_of_stores(lineage, [row["store_id"] for row in stored])
    assert bound == {row["store_id"]: {host["message_uid"]} for row, host in zip(stored, original)
                     if "message_uid" in host}
    _bind_gateway(gateway)
    gateway.ingest(original + LATER)
    gateway.compress(original + LATER)
    assert gateway.last_compression_status == "compacted"
    before = _rows(gateway)
    assert len(before) == 9 and all(row["conversation_id"] == GATEWAY for row in before)
    assert set(gateway._store.read_metadata_json(REOWN_KEY)) == {row["store_id"] for row in before[:4]}
    if shape == "version-on-other-row":
        gateway._store.add_host_uid_bindings(lineage, [
            (before[1]["store_id"], original[0]["message_uid"], "version", "test"),
        ])

    resumed = open_engine()
    _start(resumed)
    incoming = [dict(row) for row in original]
    new_uids = {}
    if shape == "all-new-uids":
        new_uids = {i: f"new-{i}" for i in range(len(incoming))}
    elif shape == "middle-new-uid":
        new_uids = {2: "new-2"}
        # The receipt's content view need not carry the original UID.
        cold_reader = resumed._cold_reader_replay_messages
        monkeypatch.setattr(resumed, "_cold_reader_replay_messages", lambda *args: [
            {key: value for key, value in row.items() if key != "message_uid"} for row in cold_reader(*args)
        ])
    elif shape == "uid-bound-to-other-row":
        incoming[2]["message_uid"] = incoming[0]["message_uid"]
    for index, uid in new_uids.items():
        incoming[index]["message_uid"] = uid

    resumed.ingest(incoming)
    after = _rows(resumed)
    added = after[len(before):]
    print(f"rows={len(before)}->{len(after)} added={[row['content'] for row in added]}")
    assert after[:len(before)] == before
    if shape in {"same-uids", "version-on-other-row"}:
        assert not added
        assert resumed._last_ingest_reconciliation["reason"] == "replayed UID-bound prefix"
    else:
        # v0.30.4 adds all four all-new rows, or one contradicted row in each
        # partial case. The receipt here only proves the prefix before the veto.
        cursor = 0 if shape == "all-new-uids" else 2
        assert _digests(added) == _digests(incoming if cursor == 0 else incoming[2:3])
        new_bindings = resumed._store.host_uid_bindings_for(lineage, new_uids.values())
        assert new_bindings == {uid: [(added[i - cursor]["store_id"], "canonical")] for i, uid in new_uids.items()}
        if cursor:
            resumed.shutdown()
            for _ in range(4):
                cold = open_engine()
                _start(cold)
                if shape == "middle-new-uid":
                    reader = cold._cold_reader_replay_messages
                    monkeypatch.setattr(cold, "_cold_reader_replay_messages", lambda *args: [
                        {key: value for key, value in row.items() if key != "message_uid"} for row in reader(*args)
                    ])
                cold.ingest(incoming)
                assert _rows(cold) == after
                cold.shutdown()


@pytest.mark.parametrize("restarts,plain", [(1, False), (2, False), (1, True)],
                         ids=["a", "b", "plain-ingest"])
def test_raw_replay_after_natural_stall(open_engine, differential_base, restarts, plain):
    writer, gateway, gateway_view = _mixed(open_engine)
    writer_view = list(EARLY)
    for i in range(3):
        writer_view = _step(writer, f"writer compress {i + 1}", writer_view,
                            lambda: writer.compress(writer_view))
    assert (gateway.last_compression_status == "error") is differential_base
    stalled, raw = (gateway, gateway_view) if differential_base else (writer, writer_view)
    binding = stalled._conversation_id
    assert raw == (EARLY + LATER if differential_base else EARLY)
    for i in range(restarts):
        _step(stalled, f"start {i + 1}", [], lambda: stalled.on_session_start(
            SESSION, conversation_id=binding, context_length=200_000))
        assert stalled._conversation_id == binding
        # The original GA control replays in G without copies (9 -> 9).
        # The returning S window has no stamps or UIDs for the moved rows.
        copies = [] if differential_base else EARLY
        if plain:
            _step(stalled, "plain ingest", raw, lambda: stalled.ingest(raw), copies=copies)
        else:
            _step(stalled, f"raw compress {i + 1}", raw, lambda: stalled.compress(raw, force=True), copies=copies)


def test_restarted_window_with_self_stored_and_new_rows(open_engine, differential_base):
    writer, gateway, _ = _mixed(open_engine)
    assert (gateway.last_compression_status == "error") is differential_base
    held = [dict(role="assistant", content="Reply stored by the session window during the stall."),
            dict(role="user", content="Request stored by the session window during the stall.")]
    new = [dict(role="assistant", content="New reply after restart."),
           dict(role="user", content="New request after restart.")]
    # Keep the raw snapshot to model the restarted host's transcript. The live
    # engine adopts the compression result when it adds the turns during the stall.
    view = _step(writer, "writer compress", EARLY, lambda: writer.compress(list(EARLY)))
    view += held
    _step(writer, "self-stored turns", view, lambda: writer.ingest(view), held)
    _step(writer, "same-binding start", [], lambda: _start(writer))
    raw = EARLY + held + new
    # On GA the G publication refuses and S adopts a compacted view. Resuming
    # S with its raw transcript copies the six previously stored rows on GA
    # (11 -> 19) and on the head because there is no occurrence evidence.
    view = _step(writer, "mixed raw compress", raw, lambda: writer.compress(raw, force=True), new,
                 copies=EARLY + held)
    _step(writer, "adopted view", view, lambda: writer.ingest(view))


def test_cold_post_turn_hook_after_gateway_publication(open_engine, differential_base):
    _, gateway, _ = _mixed(open_engine, sequential=True)
    assert (gateway.last_compression_status == "error") is differential_base
    gateway.shutdown()
    full = EARLY + LATER
    for i in range(3):
        cold = open_engine()
        assert not cold._session_id
        hook = _hook_for(cold)
        turn = [dict(role="assistant", content=f"Cold reply {i + 1}."),
                dict(role="user", content=f"Cold follow-up {i + 1}.")]
        full += turn
        _step(cold, f"cold hook {i + 1}", full,
              lambda: hook(session_id=SESSION, platform="gui", conversation_history=full), turn)
        assert cold._session_id == SESSION
        cold.shutdown()


def test_sequential_explicit_session_id_resume(open_engine, differential_base):
    _, gateway, _ = _mixed(open_engine, sequential=True)
    assert (gateway.last_compression_status == "error") is differential_base
    resumed = open_engine()
    _step(resumed, "explicit S start", [], lambda: _start(resumed))
    assert resumed._conversation_id == SESSION
    full = EARLY + LATER
    _step(resumed, "explicit S full ingest", full, lambda: resumed.ingest(full))
    _step(resumed, "explicit S full compress", full, lambda: resumed.compress(full, force=True))


def test_single_window_same_binding_restart(open_engine):
    writer = open_engine()
    _start(writer)
    view = EARLY + LATER
    _step(writer, "single-window ingest", view, lambda: writer.ingest(view), view)
    for i in range(3):
        view = _step(writer, f"single-window compress {i + 1}", view, lambda: writer.compress(view))
    _step(writer, "single-window start", [], lambda: _start(writer))
    _step(writer, "single-window adopted replay", view, lambda: writer.compress(view, force=True))


@pytest.mark.parametrize("rotate_before_conflict", [False, True])
@pytest.mark.parametrize("finalize_next", [False, True], ids=["one-transition", "two-transitions"])
def test_competing_gateway_rotates_after_reownership(open_engine, monkeypatch, rotate_before_conflict, finalize_next):
    writer, gateway, gateway_view = _mixed(open_engine)
    calls = []

    def summarize(**_):
        calls.append(None)
        return "Cobalt summary.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)

    def rotate():
        gateway.on_session_end(SESSION, gateway_view)
        gateway.on_session_start("S2-1250", platform="desktop", conversation_id=GATEWAY)
        binding = gateway._lifecycle.get_by_conversation(GATEWAY)
        assert (binding.current_session_id, binding.last_finalized_session_id) == ("S2-1250", SESSION)
        if finalize_next:
            gateway.on_session_end("S2-1250", [])
            binding = gateway._lifecycle.get_by_conversation(GATEWAY)
            assert (binding.current_session_id, binding.last_finalized_session_id) == (None, "S2-1250")

    if rotate_before_conflict:
        rotate()
    view = _step(writer, "first stalled compress", EARLY, lambda: writer.compress(list(EARLY)))
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert len(calls) == 1
    if not rotate_before_conflict:
        rotate()
    assert _step(writer, "rotated gateway back-off", view, lambda: writer.compress(view, force=True)) == view
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert len(calls) == 1


def test_replay_receipt_skips_ignored_host_message(open_engine, monkeypatch):
    if message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    config = {"ignore_message_patterns": ["^HEARTBEAT"]}
    writer, gateway = open_engine(**config), open_engine(**config)
    heartbeat = {"role": "user", "content": "HEARTBEAT ok"}
    early = [dict(row, timestamp=100 + i) for i, row in enumerate(EARLY)]
    raw = early[:2] + [heartbeat] + early[2:]
    _start(writer)
    _step(writer, "initial ingest with heartbeat", raw, lambda: writer.ingest(raw), EARLY)
    _bind_gateway(gateway)
    full = raw + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    _step(gateway, "gateway re-owns the rows", full, lambda: gateway.compress(full), may_reown=True)
    assert gateway.last_compression_status == "compacted"
    assert set(gateway._store.read_metadata_json(REOWN_KEY)) == {
        row["store_id"] for row in _rows(writer)[:len(EARLY)]
    }
    writer.shutdown()

    resumed = open_engine(**config)
    _start(resumed)
    new = [
        {"role": "user", "content": "A new request after the heartbeat."},
        {"role": "assistant", "content": "A new reply after the heartbeat."},
    ]
    _step(resumed, "raw replay with heartbeat", raw + new, lambda: resumed.ingest(raw + new), new)
    assert resumed._last_ingest_reconciliation["reason"] == "replayed re-owned session rows"
    assert resumed._last_ingest_reconciliation["cursor"] == len(raw)
    assert resumed._last_ingest_reconciliation["effective_incoming"] == len(EARLY) + len(new)


@pytest.mark.parametrize("stored_stamp, incoming_stamp", [(None, None), (100, None), (None, 100), (100, 100)])
def test_singleton_replay_receipt_needs_occurrence_proof(open_engine, stored_stamp, incoming_stamp):
    writer, gateway = open_engine(), open_engine()
    _start(writer)
    original = dict(EARLY[0], timestamp=stored_stamp)
    writer.ingest([original])
    _bind_gateway(gateway)
    full = [original, *LATER]
    gateway.ingest(full)
    gateway.compress(full)
    assert gateway.last_compression_status == "compacted"
    assert gateway._store.read_metadata_json(REOWN_KEY) == [_rows(writer)[0]["store_id"]]

    resumed = open_engine()
    _start(resumed)
    repeat = dict(original, timestamp=incoming_stamp)
    is_replay = stored_stamp is not None and stored_stamp == incoming_stamp
    _step(resumed, "singleton replay", [repeat], lambda: resumed.ingest([repeat]), [] if is_replay else [repeat])
    if is_replay:
        assert resumed._last_ingest_reconciliation["reason"] == "replayed re-owned session rows"


def test_receipt_replay_keeps_new_message_matching_only_after_redaction(open_engine, monkeypatch):
    if message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    _, gateway, _ = _mixed(open_engine, sequential=True)
    new = {"role": "user", "content": "api_key=sk-synthetic...test needs redaction"}
    resumed = open_engine(
        ignore_message_patterns=[r"\[LCM sensitive redaction:"],
        sensitive_patterns_enabled=True, sensitive_patterns=["api_key"],
    )
    _start(resumed)
    redacted = resumed._redact_active_replay_messages([new])[0]
    assert redacted != new
    assert not resumed._matches_ignore_message_patterns(new)
    assert resumed._matches_ignore_message_patterns(redacted)
    raw = EARLY[:2] + [new] + EARLY[2:]
    before = _rows(gateway)
    resumed.ingest(raw)
    after = _rows(resumed)
    assert after[:len(before)] == before
    assert [row["content"] for row in after[len(before):]].count(redacted["content"]) == 1


def test_receipt_replay_skips_cached_active_placeholder(open_engine, monkeypatch):
    if message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    config = {"ignore_message_patterns": ["^HEARTBEAT"]}
    writer, gateway = open_engine(**config), open_engine(**config)
    _start(writer)
    ignored = {"role": "user", "content": "HEARTBEAT ignored in active replay"}
    early = [dict(row, timestamp=100 + i) for i, row in enumerate(EARLY)]
    raw = early[:2] + [ignored] + early[2:]
    writer.ingest(raw)
    replay = list(writer._last_active_replay_messages)
    assert replay[2]["content"].startswith("[LCM active replay placeholder: message ignored;")
    _bind_gateway(gateway)
    gateway.ingest(raw + LATER)
    gateway.compress(raw + LATER)
    assert gateway.last_compression_status == "compacted"

    writer._schedule_ingest_cursor_reconciliation()
    new = {"role": "user", "content": "New message following the active replay."}
    _step(writer, "active placeholder replay", replay + [new], lambda: writer.ingest(replay + [new]), [new])
    assert writer._last_ingest_reconciliation["reason"] == "replayed re-owned session rows"
    assert writer._last_ingest_reconciliation["cursor"] == len(replay)
    assert writer._last_ingest_reconciliation["effective_incoming"] == len(EARLY) + 1


@pytest.mark.parametrize("has_stamps", [False, True])
def test_replay_receipt_stops_at_unmatched_host_message(open_engine, has_stamps):
    early = [dict(row, timestamp=100 + i) for i, row in enumerate(EARLY)] if has_stamps else EARLY
    writer, gateway, _ = _mixed(open_engine, early=early)
    raw = early[:2] + [{"role": "user", "content": "An unmatched request."}] + early[2:]
    stored = _rows(gateway)
    # v0.30.4 has no receipt matcher.
    cursor, pairs = writer._cursor_from_reowned_session_rows(raw, stored, len(stored))
    assert cursor == (2 if has_stamps else 0)
    assert pairs == {i: row["store_id"] for i, row in enumerate(stored[:cursor])}


def test_short_view_skips_full_receipt_scan(open_engine, monkeypatch):
    writer, gateway, _ = _mixed(open_engine)
    stored = _rows(gateway)

    def unexpected_read(*args, **kwargs):
        pytest.fail("a view shorter than the receipt should not load the session")

    monkeypatch.setattr(writer._store, "get_session_messages", unexpected_read)
    assert writer._cursor_from_reowned_session_rows(EARLY[:2], stored[-1:], len(stored)) == (0, {})


@pytest.mark.parametrize("prior, retained", [
    ('malformed', []), ('"scalar"', []), ('{"foreign": 999}', []), ('999', []),
    ('[999, "unusable", 1.5, true, null, {}]', [999]),
])
def test_malformed_replay_receipt_does_not_prevent_reownership(open_engine, prior, retained):
    writer, gateway = open_engine(), open_engine()
    _start(writer)
    writer.ingest(list(EARLY))
    before = _rows(writer)
    writer._store.write_metadata_json([REOWN_KEY], prior)
    _bind_gateway(gateway)
    full = EARLY + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    _step(gateway, "gateway publishes with malformed receipt", full, lambda: gateway.compress(full), may_reown=True)
    assert gateway.last_compression_status == "compacted"
    assert sorted(gateway._store.read_metadata_json(REOWN_KEY)) == [row["store_id"] for row in before] + retained
    assert all(row["conversation_id"] == GATEWAY for row in _rows(gateway))
