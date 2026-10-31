"""#1124: copies need positive evidence and actual summary coverage, never a frontier guess."""

import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.store_complete as lcm_store_complete
from hermes_lcm.dag import SummaryNode
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens

from tests.test_host_uid_shadow import (
    PAD,
    ROLLBACK_READERS,
    _bindings,
    _engine as _uid_engine,
    _m,
    _skip_or_fail_in_ci,
    _state_db,
)
from tests.test_issue_581_582_store_complete_survival import (
    _assert_contiguous,
    _engine,
    _frontier,
    _rows,
    _turn,
)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    instance = _engine(tmp_path, temporal_rollups_enabled=False)
    monkeypatch.setattr(
        instance, "_host_uid_lineage_key", lambda: ("test:lineage", None)
    )
    yield instance
    instance.shutdown()


def _text(content="same text", role="assistant"):
    return {"role": role, "content": content}


def _segment():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_copy",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_copy",
            "content": "result",
            "name": "read",
        },
    ]


def _append(engine, messages, session="S", conversation="conv"):
    return [
        engine._store.append(session, message, conversation_id=conversation)
        for message in messages
    ]


def _bind(engine, store_id, uid="original", kind="canonical", lineage="test:lineage"):
    engine._store.add_host_uid_bindings(lineage, [(store_id, uid, kind, "agree_new")])
    engine._store.record_host_uid_checks(lineage, [(uid, store_id, False)])


def _node(engine, ids, *, before_commit=None):
    return engine._dag.add_node(
        SummaryNode(
            session_id="S",
            summary="Earlier turns",
            token_count=2,
            source_token_count=20,
            source_ids=ids,
            source_type="messages",
        ),
        before_commit=before_commit,
    )


def _walk(
    engine, *, chunk=(), claims=None, mapping=None, view=(), carry=(), budget=10_000
):
    return engine._store_complete_input(
        list(chunk),
        claims or {},
        mapping or {},
        list(view),
        list(chunk),
        _frontier(engine),
        list(carry),
        budget,
    )


def _ids(out, mapping=None):
    mapping = mapping or {}
    return [
        sid
        for message, claims in out
        for sid in [
            *claims,
            *([mapping[id(message)]] if id(message) in mapping else []),
        ]
    ]


def _publish(engine, out):
    ids = _ids(out)
    frontier = _frontier(engine)
    proofs = getattr(engine, "_store_complete_copy_exclusion_proofs", {})
    _node(
        engine,
        ids,
        before_commit=lambda conn, node_id: (
            engine._lifecycle.stage_compaction_publication(
                conn,
                "conv",
                "S",
                node_id,
                frontier,
                ids,
                excluded_store_ids=getattr(engine, "_store_complete_excluded", []),
                copy_exclusion_proofs=proofs,
            )
        ),
    )


def _control(engine):
    """A proven copy alongside each negative control makes classifier precision red on the base."""
    twin, copy = _append(engine, [_text("control twin"), _text("control twin")])
    _bind(engine, twin, "control")
    return twin, copy


def _assert_control(engine, out, control):
    twin, copy = control
    assert twin in _ids(out) and copy not in _ids(out)
    assert engine._store_complete_copy_exclusion_proofs[copy] == twin


@pytest.mark.parametrize("evidence", ["E1", "E2"])
def test_01_same_leaf_copy_is_excluded(engine, evidence):
    messages = _segment() if evidence == "E1" else [_text()]
    originals = _append(engine, messages)
    copies = _append(engine, [{**message, "timestamp": 500.0} for message in messages])
    if evidence == "E2":
        _bind(engine, originals[0])
    end = _append(engine, [_text("next turn", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [*originals, end]
    assert engine._store_complete_copy_exclusion_proofs == dict(zip(copies, originals))
    assert set(copies) <= set(engine._store_complete_excluded)


def test_01_e2_is_disabled_after_host_uid_mode_is_off(engine, monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    twin, repeat = _append(engine, [_text(), _text()])
    _bind(engine, twin)
    end = _append(engine, [_text("next turn", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [twin, end]
    assert engine._store_complete_copy_exclusion_proofs == {repeat: twin}

    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    out = _walk(engine)
    assert _ids(out) == [twin, repeat, end]
    assert repeat not in engine._store_complete_excluded
    assert engine._store_complete_copy_exclusion_proofs == {}


def test_02_node_covered_twin_is_excluded_next_leaf(engine):
    originals = _append(engine, _segment())
    twin = originals[-1]
    _publish(engine, [(message, [sid]) for message, sid in zip(_segment(), originals)])
    assert _frontier(engine) == twin
    copy = _append(engine, [_segment()[1]])[0]
    end = _append(engine, [_text("next", "user")])[0]
    out = _walk(engine)
    assert copy not in _ids(out) and end in _ids(out)
    assert engine._store_complete_copy_exclusion_proofs == {copy: twin}
    _publish(engine, out)
    assert _frontier(engine) == end


def test_partial_tool_group_repeat_keeps_the_caller_with_its_new_result(engine):
    originals = _append(engine, _segment())
    _publish(engine, list(zip(_segment(), ([sid] for sid in originals))))
    call, result = _segment()
    later = _append(engine, [call, {**result, "content": "different result"}])
    end = _append(engine, [_text("end", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [*later, end]
    assert not engine._store_complete_copy_exclusion_proofs
    _publish(engine, out)
    assert _frontier(engine) == end


def test_partial_tool_group_repeat_inside_one_leaf_is_not_attached_to_the_twin(engine):
    originals = _append(engine, _segment())
    call, result = _segment()
    later = _append(engine, [call, {**result, "content": "different result"}])
    end = _append(engine, [_text("end", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [*originals, *later, end]
    assert not engine._store_complete_copy_exclusion_proofs
    _publish(engine, out)
    assert _frontier(engine) == end


def test_full_tool_group_copy_is_still_excluded_whole(engine):
    originals = _append(engine, _segment())
    _publish(engine, list(zip(_segment(), ([sid] for sid in originals))))
    copies = _append(engine, _segment())
    end = _append(engine, [_text("end", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [end]
    assert engine._store_complete_copy_exclusion_proofs == dict(zip(copies, originals))
    assert set(copies) <= set(engine._store_complete_excluded)
    _publish(engine, out)
    assert _frontier(engine) == end


@pytest.mark.parametrize("own_binding", [True, False], ids=["bound", "unbound"])
def test_e1_spares_a_tool_exchange_with_its_own_uid_binding(engine, own_binding):
    originals = _append(engine, _segment())
    for sid in originals:
        _bind(engine, sid, f"original-{sid}")
    _publish(engine, list(zip(_segment(), ([sid] for sid in originals))))
    later = _append(engine, _segment())
    if own_binding:
        for sid in later:
            _bind(engine, sid, f"occurrence-{sid}")
    end = _append(engine, [_text("end", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == ([*later, end] if own_binding else [end])
    assert engine._store_complete_copy_exclusion_proofs == (
        {} if own_binding else dict(zip(later, originals))
    )
    _publish(engine, out)
    assert _frontier(engine) == end


@pytest.mark.parametrize(
    "text", ["continue", "deliberate long repeated instruction " * 400]
)
def test_03_genuine_user_repeat_with_own_uid_stays(engine, text):
    originals = _append(engine, [_text(text, "user"), _text(text, "user")])
    for sid in originals:
        _bind(engine, sid, f"occurrence-{sid}")
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert set(originals) <= set(_ids(out))
    _assert_control(engine, out, control)


@pytest.mark.parametrize(
    "binding", ["absent", "version", "other-lineage", "unresolved"]
)
def test_04_text_without_canonical_lineage_evidence_stays(engine, monkeypatch, binding):
    originals = _append(engine, [_text(), _text()])
    if binding == "version":
        _bind(engine, originals[0], kind="version")
    elif binding == "other-lineage":
        _bind(engine, originals[0], lineage="other:lineage")
    elif binding == "unresolved":
        _bind(engine, originals[0])
        monkeypatch.setattr(
            engine, "_host_uid_lineage_key", lambda: (None, "unresolved")
        )
    # E1 still works without any uid, including an unresolved lineage.
    twin, copy = _append(engine, [_segment()[1], _segment()[1]])
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert set(originals) <= set(_ids(out))
    assert twin in _ids(out) and copy not in _ids(out)


def test_e2_never_excludes_an_unbound_user_repeat(engine):
    twin, repeat = _append(engine, [_text("continue", "user")] * 2)
    _bind(engine, twin)
    _node(engine, [twin])
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert repeat not in engine._store_complete_copy_exclusion_proofs
    assert repeat in _ids(out)


def test_05_retained_unmapped_copy_stays_blocked(engine):
    control = _control(engine)
    _append(engine, [_text("before blocked", "user")])
    twin, copy = _append(engine, [_text(), _text()])
    _bind(engine, twin)
    _node(engine, [twin])
    _append(engine, [_text("later", "user")])
    out = _walk(engine, view=[_text()])
    _assert_control(engine, out, control)
    assert copy not in engine._store_complete_excluded and copy not in _ids(out)
    assert max(_ids(out)) < copy


@pytest.mark.parametrize("claimed", [False, True], ids=["mapped", "claimed"])
def test_06_inverted_mapping_and_claimed_rows_are_never_copies(engine, claimed):
    control = _control(engine)
    twin, copy = _append(engine, [_segment()[1], _segment()[1]])
    message = _segment()[1]
    mapping = {} if claimed else {id(message): copy}
    claims = {id(message): [copy]} if claimed else {}
    _append(engine, [_text("end", "user")])
    out = _walk(engine, chunk=[message], mapping=mapping, claims=claims, view=[message])
    assert {twin, copy} <= set(_ids(out, mapping))
    _assert_control(engine, out, control)
    assert copy not in engine._store_complete_excluded


def test_07_frontier_is_not_summary_coverage(engine):
    twin = _append(engine, [_segment()[1]])[0]
    engine._lifecycle.stage_frontier_advance(
        engine._store.connection, "conv", "S", twin
    )
    copy = _append(engine, [_segment()[1]])[0]
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert copy in _ids(out) and copy not in engine._store_complete_excluded
    _assert_control(engine, out, control)


def test_08_publication_advances_contiguously_over_non_copies(engine, monkeypatch):
    control = _control(engine)
    end = _append(engine, [_text("end", "user")])[0]
    tail = [_text("fresh", "user"), _text("fresh reply")]
    tail_ids = _append(engine, tail)
    mapping = dict(zip(map(id, tail), tail_ids))
    before = _rows(engine)
    out = _walk(engine, view=tail, mapping=mapping)
    _assert_control(engine, out, control)
    monkeypatch.setattr(
        lcm_engine, "summarize_with_escalation", lambda **kwargs: ("Earlier turns", 1)
    )
    engine._ingest_cursor = len(tail)
    engine.compress(tail)
    assert engine._last_compression_status == "compacted", (
        engine._last_compression_noop_reason
    )
    assert _frontier(engine) == end and _rows(engine) == before
    coverage = Counter(
        sid for node in engine._dag.get_session_nodes("S") for sid in node.source_ids
    )
    assert all(coverage[sid] == 1 for sid in (control[0], end))
    assert coverage[control[1]] == 0 and coverage[control[0]] == 1
    _assert_contiguous(engine, {control[1]: control[0]})


def test_09_copy_only_backlog_is_drained_without_hidden_only_hold(engine, monkeypatch):
    twin = _append(engine, [_text()])[0]
    _bind(engine, twin)
    _node(engine, [twin])
    _append(engine, [_text(), _text()])
    # Fresh tail is mapped and retained. No summary call or hidden-only hold is warranted.
    tail = [_text("fresh", "user"), _text("fresh reply")]
    tail_ids = _append(engine, tail)
    mapping = dict(zip(map(id, tail), tail_ids))
    monkeypatch.setattr(
        engine, "_get_store_id_map_for_messages", lambda _messages: mapping
    )
    backlog = engine._store_complete_backlog(tail, 0)
    assert not backlog and backlog.rows == 0 and not backlog.truncated
    assert engine._hidden_backlog_status() == {"hidden_rows": 0}
    engine._ingest_cursor = len(tail)
    result = engine.compress(tail)
    assert result == tail and engine._last_compression_status != "compacted"
    assert engine._no_progress_hold is None


def test_10_exit_fit_and_uncovered_counter_use_covered_view_rows(engine):
    control = _control(engine)
    turns = [
        _text(f"turn {i} " * 30, role)
        for i in range(3)
        for role in ("user", "assistant")
    ]
    ids = _append(engine, turns)
    out = _walk(engine)
    _assert_control(engine, out, control)
    _publish(engine, out)
    mapping = dict(zip(map(id, turns), ids))
    budget = engine._survival_measure(turns[4:]) + 10
    cut = engine._survival_cut(
        turns, 0, budget, True, "exit_fit:compacted", mapping, True, keep_from=4
    )
    assert cut is not None
    fitted, count, dropped, projected, notice = cut
    assert dropped and control[1] not in dropped
    assert set(dropped) <= set(ids)
    engine._survival_record(
        "exit_fit:compacted",
        count,
        dropped,
        1000,
        engine._survival_measure(fitted),
        budget,
        projected,
        notice,
        warn_user=False,
    )
    assert engine._last_survival_fit["uncovered_rows"] == 0
    assert (
        engine._store.read_metadata_json("survival_fit:counter")["last_uncovered_rows"]
        == 0
    )
    # A view that explicitly maps to the excluded copy must still fail the #738 gate.
    assert (
        engine._survival_cut(
            turns,
            0,
            budget,
            True,
            "exit_fit:compacted",
            {**mapping, id(turns[0]): control[1]},
            True,
            keep_from=4,
        )
        is None
    )


def test_11_copy_injection_preserves_freed_per_leaf(tmp_path, monkeypatch):
    outcomes = []
    for copies in (False, True):
        instance = _engine(tmp_path / str(copies), temporal_rollups_enabled=False)
        monkeypatch.setattr(
            instance, "_host_uid_lineage_key", lambda: ("test:lineage", None)
        )
        try:
            first = _text("leaf text " * 35)
            twin = _append(instance, [first])[0]
            _bind(instance, twin)
            if copies:
                _append(instance, [first] * 4)
            _append(instance, [_text("next turn " * 35, "user"), _text("later " * 35)])
            out = _walk(instance, budget=260)
            messages = [m for m, _claims in out]
            outcomes.append((messages, count_messages_tokens(messages) - 2))
        finally:
            instance.shutdown()
    assert outcomes[0] == outcomes[1]


def test_12_v0262_reader_accepts_store_after_copy_publication(engine):
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    _assert_control(engine, out, control)
    _publish(engine, out)
    frontier = _frontier(engine)
    root = Path(__file__).resolve().parents[1]
    commit = ROLLBACK_READERS["v0.26.2"]
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        capture_output=True,
    ).returncode:
        _skip_or_fail_in_ci(f"the v0.26.2 commit {commit[:8]} is not in this checkout")
    code = subprocess.check_output(
        ["git", "show", f"{commit}:lifecycle_state.py"], cwd=root, text=True
    )
    module = types.ModuleType("hermes_lcm._rollback_1124")
    module.__package__ = "hermes_lcm"
    # Dataclasses require the module registered while loading the historical reader.
    import sys

    sys.modules[module.__name__] = module
    try:
        exec(compile(code, "v0.26.2:lifecycle_state.py", "exec"), module.__dict__)
        reader = module.LifecycleStateStore(engine._config.database_path)
        try:
            assert (
                reader.get_by_conversation("conv").current_frontier_store_id == frontier
            )
            later = _append(engine, [_text("after rollback", "user")])[0]
            _node(
                engine,
                [later],
                before_commit=lambda conn, node_id: reader.stage_compaction_publication(
                    conn,
                    "conv",
                    "S",
                    node_id,
                    frontier,
                    [later],
                    excluded_store_ids=[],
                ),
            )
            assert reader.get_by_conversation("conv").current_frontier_store_id == later
        finally:
            reader.close()
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize(
    "forgery",
    [
        "bytes",
        "uncovered",
        "consumed-not-summarized",
        "foreign-session",
        "foreign-conversation",
        "later",
        "not-excluded",
        "changed-before-commit",
    ],
)
def test_13_fence_rejects_forged_copy_proof(engine, forgery):
    original = _segment()[1]
    twin = _append(
        engine,
        [original],
        session="OTHER" if forgery == "foreign-session" else "S",
        conversation="other" if forgery == "foreign-conversation" else "conv",
    )[0]
    if forgery not in ("uncovered", "consumed-not-summarized"):
        _node(engine, [twin])
    copy = _append(engine, [_text("wrong bytes") if forgery == "bytes" else original])[
        0
    ]
    end = _append(engine, [original if forgery == "later" else _text("end", "user")])[0]
    if forgery in ("foreign-session", "foreign-conversation"):
        engine._lifecycle.stage_frontier_advance(
            engine._store.connection, "conv", "S", twin
        )
        engine._store.connection.commit()
    proof = {copy: end if forgery == "later" else twin}
    before = _frontier(engine)

    def stage(conn, node_id):
        if forgery == "changed-before-commit":
            conn.execute(
                "UPDATE messages SET tool_name='changed' WHERE store_id=?", (twin,)
            )
        engine._lifecycle.stage_compaction_publication(
            conn,
            "conv",
            "S",
            node_id,
            before,
            [twin, end] if forgery == "consumed-not-summarized" else [end],
            excluded_store_ids=(
                [copy]
                if forgery
                in (
                    "consumed-not-summarized",
                    "foreign-session",
                    "foreign-conversation",
                )
                else [twin, copy]
                if forgery != "not-excluded"
                else [twin]
            ),
            copy_exclusion_proofs=proof,
        )

    with pytest.raises(LifecyclePublicationConflictError):
        _node(engine, [end], before_commit=stage)
    assert _frontier(engine) == before
    assert not any(
        end in node.source_ids for node in engine._dag.get_session_nodes("S")
    )
    assert engine._store.get(twin).get("tool_name") != "changed"


@pytest.mark.parametrize(
    "risk", ["lossy", "bound-copy", "raw-json-bytes", "foreign-twin"]
)
def test_copy_evidence_is_fail_safe(engine, risk):
    original = _segment()[1]
    if risk == "lossy":
        original = {
            **original,
            "content": "[LCM sensitive redaction: name=password_assignment; chars=12]",
        }
    if risk == "raw-json-bytes":
        original = _segment()[0]
    twin = _append(
        engine, [original], session="OTHER" if risk == "foreign-twin" else "S"
    )[0]
    copy = _append(engine, [original])[0]
    if risk == "bound-copy":
        original = _text()
        engine._store.connection.execute(
            "UPDATE messages SET role='assistant', content=?, tool_call_id=NULL, "
            "tool_name=NULL WHERE store_id IN (?, ?)",
            ("text", twin, copy),
        )
        engine._store.connection.commit()
        _bind(engine, twin)
        _bind(engine, copy, "own-copy", kind="version")
    elif risk == "raw-json-bytes":
        engine._store.connection.execute(
            "UPDATE messages SET tool_calls=replace(tool_calls, ': ', ':') "
            "WHERE store_id=?",
            (copy,),
        )
        engine._store.connection.commit()
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert copy in _ids(out) and copy not in engine._store_complete_excluded
    _assert_control(engine, out, control)


@pytest.mark.parametrize("error", ["timeout", "context", "size-context"])
def test_r1_rescue_prefix_before_copy_publishes(engine, monkeypatch, error):
    calls = []

    def summarize(**kwargs):
        text = kwargs["text"]
        calls.append(text)
        if error == "size-context":
            refused = len(text) > 450
        else:
            refused = sum(line.startswith("[") for line in text.splitlines()) >= 4
        if refused:
            if error == "timeout":
                raise TimeoutError("summary timed out")
            raise RuntimeError("prompt is too long: maximum context length exceeded")
        return "Earlier turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    fill = [
        _text(f"filler {i} " + "x" * 60, "user" if i % 2 == 0 else "assistant")
        for i in range(4 if error == "size-context" else 2)
    ]
    ids = _append(
        engine,
        [_text("control twin")]
        + fill
        + [_text("control twin"), _text("later " + "y" * 200, "user")],
    )
    twin, copy = ids[0], ids[-2]
    _bind(engine, twin, "control")
    tail = [_text("fresh", "user"), _text("fresh reply")]
    _append(engine, tail)
    engine._ingest_cursor = len(tail)
    engine.compress(list(tail))
    assert len(calls) >= 2
    assert engine._last_compression_status == "compacted", (
        engine._last_compression_noop_reason
    )
    assert twin <= _frontier(engine) < copy
    for _ in range(2):
        engine.note_turn_complete()
        engine.compress(list(tail))
    assert _frontier(engine) >= twin


@pytest.mark.parametrize("path", ["r3-remainder", "mode-off"])
def test_r2_real_ingest_genuine_unbound_repeat_stays(tmp_path, monkeypatch, path):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    _state_db(tmp_path, [("S", None, None)])
    instance = _uid_engine(tmp_path)
    try:
        system = _text("stable system prompt", "system")
        first = _m("user", "continue", 10.0, "u-c1")
        reply = _m("assistant", "ok, continuing" + PAD, 11.0, "a-1")
        head = _m("user", "head R" + PAD, 500.0, "u-r")
        view = [system, first, reply] + ([head] if path == "r3-remainder" else [])
        instance.ingest([dict(m) for m in view])
        if path == "r3-remainder":
            merged = _m("user", head["content"] + "\n\ncontinue", 500.0, "u-r")
            view = [
                system,
                first,
                reply,
                merged,
                _m("assistant", "reply 2", 501.0, "a-2"),
            ]
        else:
            monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
            view += [
                _m("user", "continue", 20.0, "u-c2"),
                _m("assistant", "again" + PAD, 21.0, "a-2"),
            ]
        instance.ingest([dict(m) for m in view])
        twin, repeat = [
            int(r["store_id"])
            for r in _rows(instance)
            if r["role"] == "user" and r["content"] == "continue"
        ]
        bound = {b[0]: b[3] for b in _bindings(instance)}
        assert bound.get(twin) == "canonical" and repeat not in bound
        _node(instance, [twin])
        out = _walk(instance)
        assert repeat in _ids(out)
        assert repeat not in instance._store_complete_copy_exclusion_proofs
        _publish(instance, out)
        assert _frontier(instance) >= repeat
        assert any(repeat in n.source_ids for n in instance._dag.get_session_nodes("S"))
    finally:
        instance.shutdown()


def test_r2_recorded_composite_member_stays_after_twin_reingest(engine):
    twin, repeat = _append(engine, [_text("continue", "user")] * 2)
    _bind(engine, twin)
    _node(engine, [twin])
    engine._store.add_message_relations(
        [[(twin, "composite", twin, 0, 10.0), (twin, "composite", repeat, 1, 10.0)]]
    )
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert (
        repeat in _ids(out)
        and repeat not in engine._store_complete_copy_exclusion_proofs
    )


def test_r2_canonical_without_disagreement_is_not_evidence(engine):
    twin, repeat = _append(engine, [_text()] * 2)
    engine._store.add_host_uid_bindings(
        "test:lineage", [(twin, "original", "canonical", "agree_new")]
    )
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert repeat in _ids(out) and not engine._store_complete_copy_exclusion_proofs


def _tagged_segment(tag):
    call, result = _segment()
    call["tool_calls"][0]["id"] = f"call_{tag}"
    return [call, {**result, "tool_call_id": f"call_{tag}", "content": f"result {tag}"}]


@pytest.mark.parametrize("page", [4, 2000])
def test_r3_copy_page_schedules_and_advances(engine, monkeypatch, page):
    monkeypatch.setattr(lcm_store_complete, "_SCAN_LIMIT", page)
    originals = [m for i in range(page // 2 + 1) for m in _tagged_segment(i)]
    ids = _append(engine, originals)
    _publish(engine, [(m, [sid]) for m, sid in zip(originals, ids)])
    copies = _append(engine, originals)
    _append(
        engine, [_text("real new turn " * 20, "user"), _text("real new reply " * 20)]
    )
    host = [_text(f"host turn {i} " * 20, "user") for i in range(4)]
    tail = [_text("fresh", "user"), _text("fresh reply")]
    view = host + tail
    view_ids = _append(engine, view)
    mapping = dict(zip(map(id, view), view_ids))
    monkeypatch.setattr(
        engine, "_get_store_id_map_for_messages", lambda _messages: mapping
    )
    backlog = engine._store_complete_backlog(view, 0)
    assert backlog.truncated and backlog.rows >= page
    out = _walk(engine, budget=400)
    assert out and copies[0] in _ids(out)
    _publish(engine, out)
    assert _frontier(engine) > ids[-1]


@pytest.mark.parametrize("calls", [1, 3], ids=["short-control", "full-page"])
def test_r3_forced_copy_page_drains_real_rows(engine, monkeypatch, calls):
    monkeypatch.setattr(lcm_store_complete, "_SCAN_LIMIT", 4)
    originals = [m for i in range(calls) for m in _tagged_segment(i)]
    ids = _append(engine, originals)
    _publish(engine, [(m, [sid]) for m, sid in zip(originals, ids)])
    _append(engine, originals)
    real = _append(
        engine, [_text("real new turn " * 20, "user"), _text("real new reply " * 20)]
    )
    monkeypatch.setattr(
        lcm_engine, "summarize_with_escalation", lambda **kwargs: ("Earlier turns", 1)
    )
    view = []
    frontiers = []
    for turn in range(6):
        host = [_text(f"host turn {turn}-{i} " * 20, "user") for i in range(4)]
        tail = [_text("fresh", "user"), _text("fresh reply")]
        _append(engine, host + tail)
        view += host + tail
        engine._ingest_cursor = len(view)
        view = engine.compress(view, force=True, bypass_cooldown=True)
        frontiers.append(_frontier(engine))
        engine.note_turn_complete()
    assert frontiers[-1] >= real[-1], frontiers


def test_r5_ordinary_leaf_skips_unusable_lookup(tmp_path, monkeypatch):
    instance = _engine(
        tmp_path, dynamic_leaf_chunk_enabled=True, dynamic_leaf_chunk_max=400
    )
    lookups, hidden = [], []
    cls = type(instance)
    real_twins, real_messages = (
        cls._store_complete_copy_twins,
        cls._store_complete_messages,
    )
    monkeypatch.setattr(
        cls,
        "_store_complete_copy_twins",
        lambda self, rows, carry: (
            lookups.append(len(rows)) or real_twins(self, rows, carry)
        ),
    )
    monkeypatch.setattr(
        cls,
        "_store_complete_messages",
        lambda self, rows: hidden.append(len(rows)) or real_messages(self, rows),
    )
    monkeypatch.setattr(
        lcm_engine, "summarize_with_escalation", lambda **kwargs: ("Earlier turns", 1)
    )
    try:
        view = [
            row for i in range(12) for row in _turn(f"T{i}", 100.0 * i, tool=i % 3 == 0)
        ]
        instance.ingest(view)
        instance.compress(view)
        assert instance._last_compression_status == "compacted"
        assert hidden and not any(hidden)
        assert not lookups
    finally:
        instance.shutdown()


def test_r5_empty_backlog_skips_lookup(engine, monkeypatch):
    _append(engine, [_text("system", "system")])
    monkeypatch.setattr(
        engine,
        "_store_complete_copy_twins",
        lambda *_: pytest.fail("lookup cannot be used"),
    )
    assert not engine._store_complete_backlog([], 0)


def test_r5_extended_hidden_scan_computes_skipped_lookup(engine, monkeypatch):
    monkeypatch.setattr(lcm_store_complete, "_SCAN_LIMIT", 2)
    call, result = _segment()
    ids = _append(engine, [call, result, result, _text("end", "user")])
    end = _text("end", "user")
    mapping = {id(call): ids[0], id(result): ids[1], id(end): ids[3]}
    out = _walk(
        engine, chunk=[call, result, end], mapping=mapping, view=[call, result, end]
    )
    assert out is not None and _ids(out, mapping) == [ids[0], ids[1], ids[3]]
    assert engine._store_complete_copy_exclusion_proofs == {ids[2]: ids[1]}


def test_r6_messages_lookup_failure_keeps_copies_summarized(
    engine, monkeypatch, caplog
):
    twin, copy = _append(engine, [_segment()[1]] * 2)
    end = _append(engine, [_text("end", "user")])[0]
    connection = engine._store.connection

    def deny_lookup(action, table, column, database, source):
        import sqlite3

        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ
            and table == "messages"
            and column == "tool_calls"
            else sqlite3.SQLITE_OK
        )

    # Deny the SELECT's raw tool_calls column only after the ordinary owned-row read.
    real_owned = engine._store_complete_owned_rows

    def owned(*args, **kwargs):
        rows = real_owned(*args, **kwargs)
        connection.set_authorizer(deny_lookup)
        return rows

    monkeypatch.setattr(engine, "_store_complete_owned_rows", owned)
    try:
        with caplog.at_level("DEBUG"):
            out = _walk(engine)
        assert _ids(out) == [twin, copy, end]
        assert not engine._store_complete_copy_exclusion_proofs
        assert "copy lookup unavailable" in caplog.text
    finally:
        connection.set_authorizer(None)


def test_r7_copy_past_last_covered_row_has_no_proof(engine):
    twin, copy = _append(engine, [_segment()[1]] * 2)
    out = _walk(engine)
    assert _ids(out) == [twin]
    assert not engine._store_complete_copy_exclusion_proofs
    assert copy not in engine._store_complete_excluded
    _publish(engine, out)
    assert _frontier(engine) == twin


def test_r7_tool_group_rollback_removes_admitted_twin(engine):
    prefix = _text("prefix", "user")
    call, result = _segment()
    result = {**result, "content": "large result " * 100}
    ids = _append(engine, [prefix, call, result, call, _text("end", "user")])
    observed = []
    original_trace = sys.gettrace()

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is engine._store_complete_input.__func__.__code__
        ):
            observed.append(set(frame.f_locals["admitted"]))
        return trace

    # The rollback ends this walk immediately, so pin its local coverage invariant at return.
    try:
        sys.settrace(trace)
        out = _walk(
            engine, budget=count_message_tokens(prefix) + count_message_tokens(call) + 1
        )
    finally:
        sys.settrace(original_trace)
    assert _ids(out) == [ids[0]]
    assert observed == [{ids[0]}]
    assert not engine._store_complete_copy_exclusion_proofs


def test_r7_retained_unmapped_copy_stays_in_backlog_label(engine, monkeypatch):
    twin = _append(engine, [_text()])[0]
    _bind(engine, twin)
    _publish(engine, [(_text(), [twin])])
    _append(engine, [_text()])
    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda _messages: {})
    backlog = engine._store_complete_backlog([_text()], 0)
    assert backlog.rows == 1 and not backlog.truncated
    assert engine._hidden_backlog_status() == {"hidden_rows": 1}
