"""A post-compress prompt inherits the prior turn's stamp, then returns at its own."""

import base64
from copy import deepcopy

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX
import tests.invariants.test_issue_2378_restamp_link as restamp_link
from tests.invariants.test_issue_2378_restamp_link import (
    T, T_COMMIT, T_REQ, all_rows, build as build, history, replay_links, run,
)

pytestmark = pytest.mark.issue(2378)
P = "please check the next deployment step now"
NEW = "a distinct new turn after the rebind"


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("restamp_t", [False, True], ids=["kept-T", "restamped-T"])
@pytest.mark.parametrize("retained_p", [False, True], ids=["transferred-P", "new-repeat-P"])
def test_next_prompt_stamp_transfer_is_not_silent(build, cold, restamp_t, retained_p):
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    assert sum(row["content"] == P for row in all_rows(engine)) == 1
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    if retained_p:
        view.append({"role": "user", "content": P, "timestamp": T_COMMIT})
    else:
        next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    if restamp_t:
        next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    copies = [row["store_id"] for row in rows if row["content"] == P]
    # Content-free failure evidence: ids and relation kinds only.
    evidence = {"rows": copies,
                "relations": [(a, "possible_replay_of", b) for a, b in sorted(links)]}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(copies) == 2 if retained_p else len(copies) in (1, 2), evidence  # a genuine repeat is stored
    expected = set() if retained_p or len(copies) == 1 else {(copies[1], copies[0])}
    assert {(a, b) for a, b in links if a in copies} == expected, evidence
    by_id = {row["store_id"]: row for row in rows}
    assert all((by_id[a]["role"], by_id[a]["content"].strip()) ==
               (by_id[b]["role"], by_id[b]["content"].strip()) for a, b in links), evidence
    assert len({a for a, _b in links}) == len(links), evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_tool_linked_carrier_still_starts_the_transfer(build, cold):
    """#2377 v7 merge (Astra): a re-stamped tool carrier the tool writer linked still carries its stamp."""
    call_row = {"role": "assistant", "content": "", "timestamp": T_REQ, "tool_calls": [
        {"id": "call-carrier", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "call-carrier", "content": "carrier result", "timestamp": T_REQ}
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ - 1}, call_row, result]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    for row in view:
        if row.get("tool_calls") or row.get("role") == "tool":
            row["timestamp"] = T_COMMIT - 1
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    copies = [row["store_id"] for row in rows if row["content"] == P]
    evidence = {"rows": copies, "relations": sorted(links)}
    assert len(copies) in (1, 2), evidence
    if len(copies) == 2:
        assert (copies[1], copies[0]) in links, evidence


@pytest.mark.parametrize("fault", [False, True], ids=["control", "fault"])
def test_link_pending_source_is_not_also_linked_by_the_transfer(build, monkeypatch, fault):
    """#2377 v7: a failed link computation marks the batch link_pending; the transfer never links one too.

    The re-stamped row is link_pending as well, so no per-row link records a carried stamp and the
    transfer has no target. The control shows the same shape links without the fault.
    """
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    engine.on_session_end("S", deepcopy(view))
    engine.shutdown()
    engine = build()
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1  # T re-stamped and linked
    call = {"id": "call-2377", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    view += [{"role": "assistant", "content": "", "tool_calls": [call], "timestamp": T_COMMIT + 5},
             {"role": "tool", "tool_call_id": "call-2377", "content": "lookup done", "timestamp": T_COMMIT + 6},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]

    def fail(*_args, **_kwargs):
        raise ValueError("planned computation fault")

    if fault:
        monkeypatch.setattr(engine, "_plan_tool_replay_links", fail)
    engine.ingest(deepcopy(view))
    copies = [row["store_id"] for row in all_rows(engine) if row["content"] == P]
    pending = {row["store_id"] for row in engine._store.get_message_relations(copies, "link_pending")}
    linked = {a for a, _b in replay_links(engine)}
    evidence = {"rows": copies, "pending": sorted(pending), "linked": sorted(linked & set(copies))}
    assert len(copies) == 2, evidence
    if not fault:  # the control: this shape takes the transfer link
        assert copies[1] in linked and not pending, evidence
        return
    assert copies[1] in pending, evidence  # the fault path marked the new copy
    assert not pending & linked, evidence


def test_unreadable_placeholder_never_links_and_keeps_primary_links(build, monkeypatch):
    """Restoring a target's stored form can fail (an unreadable payload): the target is treated as retained, so the
    transferred prompt stays unlinked, and the re-stamped T link written before it survives."""
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    engine.on_session_end("S", deepcopy(view))
    engine.on_session_start("S", conversation_id="conversation", platform="cli")

    restore, link, linking = (engine._stored_row_externalized_text_parts_for_pattern_matching,
                              engine._link_restamped_occurrences, [])

    def unreadable(*args, **kwargs):  # only the linker's read fails; other readers of the helper are unchanged
        if linking:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return restore(*args, **kwargs)

    def linker(*args):
        linking.append(True)
        try:
            return link(*args)
        finally:
            linking.pop()

    monkeypatch.setattr(engine, "_stored_row_externalized_text_parts_for_pattern_matching", unreadable)
    monkeypatch.setattr(engine, "_link_restamped_occurrences", linker)
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    t_copies = [row["store_id"] for row in rows if row["content"] == T]
    p_copies = [row["store_id"] for row in rows if row["content"] == P]
    evidence = {"t": t_copies, "p": p_copies, "relations": sorted(links)}
    assert len(t_copies) == 2 and (t_copies[1], t_copies[0]) in links, evidence
    # The transfer fails closed: no consuming link from a P copy. The borrowed-stamp linker (#2421) reads no
    # stored form, and may attribute the moved P with its non-consuming link (its borrowed stamp recorded).
    from_p = {(source, target, stamp) for source, target, stamp in engine._store._conn.execute(
        "SELECT store_id, related_store_id, observed_at FROM message_relations WHERE kind='possible_replay_of'")
        if source in p_copies}
    assert len(p_copies) == 2 and from_p == {(p_copies[1], p_copies[0], T_REQ)}, evidence


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="625 (#2488, named surplus limit, v0.30.0): the "
                   "todo composite is treated as scaffold before R2, so no composite witness is recorded")
def test_625_todo_prompt_composite_with_stamp_transfer_copy_is_replay(build):
    engine = build()
    todo = _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [>] 1. check deployment (in_progress)"
    head = history(False)
    parts = [{"role": "user", "content": todo, "timestamp": T_REQ},
             {"role": "user", "content": P, "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ + 2}]
    engine.ingest(deepcopy(head + parts))
    before = all_rows(engine)
    if sum(row["content"] == P for row in before) != 2 or sum(row["content"] == todo for row in before) != 1:
        pytest.fail("setup: the todo row and both P rows must be stored once each")
    engine.on_session_end("S", deepcopy(head + parts))
    engine.on_session_start("S", conversation_id="conversation", platform="cli")
    composite = {"role": "user", "content": todo + "\n\n" + P, "timestamp": T_REQ}
    engine.ingest(deepcopy(head + [composite,
                                  {"role": "user", "content": NEW, "timestamp": T_COMMIT}]))
    rows = all_rows(engine)
    if sum(row["content"] == NEW for row in rows) != 1:
        pytest.fail("not the 625 shape: the new turn must be stored once")
    # Only the composite-surplus assertions below may raise the expected AssertionError.
    assert not any(row["content"] == composite["content"] for row in rows)
    relations = list(engine._store._conn.execute(
        "SELECT store_id,kind,related_store_id FROM message_relations WHERE kind='composite'"))
    ids = {row["store_id"] for row in before if row["content"] in (todo, P)}
    assert len(relations) == 2
    assert all(a in ids and b in ids for a, _kind, b in relations)


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("host_form", ["trimmed", "raw", "structured"])
def test_retained_prompt_vetoes_a_genuine_repeat(build, cold, host_form):
    """The host still shows the transferred prompt at its old stamp (trimmed, byte-equal as the control, or as
    structured content parts), and the user then really repeats it at a new stamp: that repeat is a new occurrence,
    never linked to the old copy."""
    first = [{"type": "text", "text": P}] if host_form == "structured" else P + "\n"
    again = deepcopy(first) if host_form == "structured" else P
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": deepcopy(first), "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    old = [row["store_id"] for row in all_rows(engine)
           if row["content"].strip() == normalize_content_value(first).strip()]
    assert len(old) == 1
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    if host_form == "trimmed":
        next(row for row in view if row.get("content") == P + "\n")["content"] = P
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view += [{"role": "user", "content": again, "timestamp": T_COMMIT},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    repeat = [row["store_id"] for row in rows
              if row["content"] == normalize_content_value(again) and row["observed_at"] == T_COMMIT]
    evidence = {"old": old, "repeat": repeat, "relations": sorted(links)}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(repeat) == 1, evidence
    assert not any(a == repeat[0] and b == old[0] for a, b in links), evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_retained_redacted_prompt_vetoes_a_genuine_repeat(build, cold):
    """With durable redaction on, the stored copy of a prompt holding a credential is redacted while the host still
    shows it raw. The retained original must still veto, so a genuine repeat at a new stamp stays unlinked."""
    secret_p = P + " with api_key=sk-test-0123456789abcdefABCDEF0123"
    engine = build(sensitive_patterns_enabled=True)
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": secret_p, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    old = [row["store_id"] for row in all_rows(engine) if row["content"].startswith(P)]
    assert len(old) == 1
    assert not any("sk-test-" in row["content"] for row in all_rows(engine))  # stored redacted
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build(sensitive_patterns_enabled=True)
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view += [{"role": "user", "content": secret_p, "timestamp": T_COMMIT},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    repeat = [row["store_id"] for row in rows if row["content"].startswith(P) and row["observed_at"] == T_COMMIT]
    evidence = {"old": old, "repeat": repeat, "relations": sorted(links)}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(repeat) == 1, evidence
    assert not any(a == repeat[0] for a, _b in links), evidence


def _rotate_then_store_p_at_carried_stamp(build):
    """The compaction commits by rotating S into the child ``rotated``; the next prompt P then inherits T's
    carried stamp and is stored first in the child."""
    engine = build(rotating=True)
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    engine.on_session_end("S", deepcopy(before))
    engine.on_session_start("rotated", boundary_reason="compression", old_session_id="S",
                            conversation_id="conversation", platform="cli")
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    assert [row["session_id"] for row in all_rows(engine) if row["content"] == P] == ["rotated"]
    return engine, view


def _rebind_child(build, engine, view, cold):
    """As the in-place cells: the child session ends, then binds again (warm) or after a restart (cold)."""
    engine.on_session_end("rotated", deepcopy(view))
    if cold:
        engine.shutdown()
        return build(rotating=True, session_id="rotated")
    engine.on_session_start("rotated", conversation_id="conversation", platform="cli")
    return engine


def _assert_no_silent_copy(engine, *, linked):
    rows, links = all_rows(engine), replay_links(engine)
    copies = [row["store_id"] for row in rows if row["content"] == P]
    # Content-free failure evidence: ids, sessions and relation kinds only.
    evidence = {"rows": [(row["store_id"], row["session_id"]) for row in rows if row["content"] in (P, T)],
                "relations": [(a, "possible_replay_of", b) for a, b in sorted(links)]}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    t_copies = [row["store_id"] for row in rows if row["content"] == T]
    assert len(t_copies) == 2 and (t_copies[1], t_copies[0]) in links, evidence
    assert len(copies) == 2, evidence
    expected = {(copies[1], copies[0])} if linked else set()
    assert {(a, b) for a, b in links if a in copies} == expected, evidence
    by_id = {row["store_id"]: row for row in rows}
    assert all((by_id[a]["role"], by_id[a]["content"].strip()) ==
               (by_id[b]["role"], by_id[b]["content"].strip()) for a, b in links), evidence
    assert len({a for a, _b in links}) == len(links), evidence
    assert len({b for _a, b in links}) == len(links), evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_rotation_child_stamp_transfer_is_not_silent(build, cold):
    """The child holds P at T's carried stamp; the host then shows T re-stamped and P at its own stamp.
    The child's own-stamp copy of P is the same occurrence, so it is linked to the child's first copy."""
    engine, view = _rotate_then_store_p_at_carried_stamp(build)
    engine = _rebind_child(build, engine, view, cold)
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    _assert_no_silent_copy(engine, linked=True)


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_rotation_child_genuine_repeat_is_not_linked(build, cold):
    """Control: the host still shows the child's P at the carried stamp and the user really repeats P at a new
    stamp. That repeat is a new occurrence, never linked to the old copy."""
    engine, view = _rotate_then_store_p_at_carried_stamp(build)
    engine = _rebind_child(build, engine, view, cold)
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view += [{"role": "user", "content": P, "timestamp": T_COMMIT},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]
    engine.ingest(deepcopy(view))
    _assert_no_silent_copy(engine, linked=False)


# The stored form of a long prompt is a whole-message placeholder; of inline media, a substring placeholder.
STORED_FORMS = {
    "long": (P + " " + "word " * 25_000, {"large_output_externalization_enabled": True}),
    "inline-media": (P + " data:image/png;base64," + base64.b64encode(bytes(range(256)) * 16).decode(), {}),
}


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("form", sorted(STORED_FORMS))
def test_retained_externalized_prompt_vetoes_a_genuine_repeat(build, cold, form):
    """The retained original is stored as a placeholder; the host still shows it in full. It must still veto, so a
    genuine repeat at a new stamp (stored as the same placeholder) stays unlinked."""
    prompt, overrides = STORED_FORMS[form]
    engine = build(**overrides)
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": prompt, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    old = [row for row in all_rows(engine) if row["observed_at"] == T_REQ and row["role"] == "user"
           and row["content"] != T]
    assert len(old) == 1 and old[0]["content"] != prompt  # stored as a placeholder
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build(**overrides)
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view += [{"role": "user", "content": prompt, "timestamp": T_COMMIT},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    repeat = [row["store_id"] for row in rows if row["observed_at"] == T_COMMIT and row["role"] == "user"]
    evidence = {"old": old[0]["store_id"], "repeat": repeat, "relations": sorted(links)}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(repeat) == 1, evidence
    assert not any(a == repeat[0] for a, _b in links), evidence


NOTE = "[Note: model was just switched from X to Y]\n\n"


def test_rotation_widening_never_redirects_a_proof_session_link(build, caplog, monkeypatch):
    """The predecessor S holds a model-switch note + T at the carried stamp; the child holds a genuine repeat of T that
    inherited that stamp. The host strips the note from the re-stamped carried row: it links to S's original, never
    to the child's repeat."""
    monkeypatch.setattr(restamp_link, "T", NOTE + T)
    repeat = {}

    def strip_and_repeat(engine, view):
        next(row for row in view if row.get("content") == NOTE + T)["content"] = T
        repeat["sid"] = engine._store.append("rotated", {"role": "user", "content": T, "timestamp": T_REQ},
                                             conversation_id=engine._conversation_id)
        view.append({"role": "user", "content": T, "timestamp": T_COMMIT + 5})

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=strip_and_repeat)
    links = replay_links(engine)
    original = next(row["store_id"] for row in rows if row["session_id"] == "S" and row["content"] == NOTE + T)
    copy = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == T
                and row["observed_at"] == T_COMMIT)
    evidence = {"original": original, "repeat": repeat["sid"], "copy": copy, "relations": sorted(links)}
    assert (copy, repeat["sid"]) not in links, evidence
    assert {b for a, b in links if a == copy} == {original}, evidence


def _child_bound(build):
    engine = build(rotating=True)
    engine.on_session_end("S", [])
    engine.on_session_start("rotated", boundary_reason="compression", old_session_id="S",
                            conversation_id="conversation", platform="cli")

    def store(session, content, stamp):
        return engine._store.append(session, {"role": "user", "content": content, "timestamp": stamp},
                                    conversation_id="conversation")

    return engine, store


def _drive_linker(engine, rows):
    """Call the linker as ingest does: each row is (stored copy, proof session, proof stamp, host stamp)."""
    stored = engine._store.get_batch([sid for sid, *_rest in rows])
    messages, stored_at, previous = [], {}, {}
    for idx, (sid, session, old_stamp, new_stamp) in enumerate(rows):
        identity = engine._message_replay_identity(stored[sid], stored_row=True)
        digest = engine_module._commit_proof_identity_digest(engine_module._proof_user_identity(identity))
        messages.append({"role": "user", "content": stored[sid]["content"], "timestamp": new_stamp})
        stored_at[idx] = sid
        previous[idx] = (session, {"timestamp": old_stamp, "identity_sha256": digest})
    engine._link_restamped_occurrences(messages, stored_at, previous, False)
    return replay_links(engine)


@pytest.mark.parametrize("a_first", [True, False], ids=["A-then-B", "B-then-A"])
def test_rotation_tiers_are_per_row_not_order_dependent(build, a_first):
    """Two re-stamped rows recorded by the child's proof: A's original lives only in the predecessor; B's lives in
    the child, and an equal B sits in the predecessor at the same stamp. Each links to its own original whatever the
    row order. The ingest path admits one such row per bind today, so the linker is driven directly."""
    engine, store = _child_bound(build)
    a_orig, b_pred, b_child = store("S", "alpha prompt", 100.0), store("S", "beta prompt", 200.0), \
        store("rotated", "beta prompt", 200.0)
    a_new, b_new = store("rotated", "alpha prompt", 300.0), store("rotated", "beta prompt", 301.0)
    rows = [(a_new, "rotated", 100.0, 300.0), (b_new, "rotated", 200.0, 301.0)]
    links = _drive_linker(engine, rows if a_first else rows[::-1])
    assert {(a_new, a_orig), (b_new, b_child)} <= links, {"b_pred": b_pred, "relations": sorted(links)}


@pytest.mark.parametrize("x_first", [True, False], ids=["X-then-Y", "Y-then-X"])
def test_rotation_mixed_proof_sessions_keep_the_predecessor_tier(build, x_first):
    """One batch mixes proofs: X's names the child, Y's names the predecessor S, and both originals live in S. Y's
    proof session must not empty X's predecessor tier. Driven directly (a mixed batch needs both proof sources)."""
    engine, store = _child_bound(build)
    x_orig, y_orig = store("S", "x prompt", 100.0), store("S", "y prompt", 200.0)
    x_new, y_new = store("rotated", "x prompt", 300.0), store("rotated", "y prompt", 301.0)
    rows = [(x_new, "rotated", 100.0, 300.0), (y_new, "S", 200.0, 301.0)]
    links = _drive_linker(engine, rows if x_first else rows[::-1])
    assert links == {(x_new, x_orig), (y_new, y_orig)}, sorted(links)


def test_one_unreadable_equal_target_fails_the_whole_source(build, monkeypatch):
    """P's carried-stamp copy sits in the child and an equal P sits in the predecessor at the same stamp: two
    candidates, so nothing links. When the child copy's stored form cannot be read, the predecessor copy must not
    become the unique candidate: the source links nothing, and the re-stamped T link survives."""
    engine, view = _rotate_then_store_p_at_carried_stamp(build)
    engine = _rebind_child(build, engine, view, False)
    unreadable_id = next(row["store_id"] for row in all_rows(engine) if row["content"] == P)
    engine._store.append("S", {"role": "user", "content": P, "timestamp": T_REQ}, conversation_id="conversation")
    parts, link, linking = (engine._stored_row_externalized_text_parts_for_pattern_matching,
                            engine._link_restamped_occurrences, [])

    def unreadable(target):  # only the linker's read of the child copy fails
        if linking and target["store_id"] == unreadable_id:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return parts(target)

    def linker(*args):
        linking.append(True)
        try:
            return link(*args)
        finally:
            linking.pop()

    monkeypatch.setattr(engine, "_stored_row_externalized_text_parts_for_pattern_matching", unreadable)
    monkeypatch.setattr(engine, "_link_restamped_occurrences", linker)
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    t_copies = [row["store_id"] for row in rows if row["content"] == T]
    p_copies = [row["store_id"] for row in rows if row["content"] == P]
    evidence = {"t": t_copies, "p": p_copies, "unreadable": unreadable_id, "relations": sorted(links)}
    assert len(t_copies) == 2 and (t_copies[1], t_copies[0]) in links, evidence
    assert len(p_copies) == 3 and not any(a in p_copies for a, _b in links), evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_rotation_fast_path_restamp_links_to_the_parent_original(build, cold):
    """End to end: the host compresses with no ingest after it, ends S with exactly the compress input (the fast
    end path marks the proof consumed), starts the child with a compression boundary, and the child's FIRST ingest
    carries T re-stamped. That copy links once, to S's original, and no row is lost."""
    engine = build(rotating=True)
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    output = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    stored_before = all_rows(engine)
    engine.on_session_end("S", deepcopy(before))
    assert engine._compress_commit_proof["end_consumed"]  # the fast end path took it
    engine.on_session_start("rotated", boundary_reason="compression", old_session_id="S",
                            conversation_id="conversation", platform="cli")
    if cold:
        engine.shutdown()
        engine = build(rotating=True, session_id="rotated")
    view = deepcopy(output)
    carried = [row for row in view if row.get("role") == "user" and row.get("content") == T]
    assert len(carried) == 1
    carried[0]["timestamp"] = T_COMMIT
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    copies = [(row["store_id"], row["session_id"], row["observed_at"]) for row in rows if row["content"] == T]
    evidence = {"copies": copies, "relations": sorted(links)}
    assert {row["store_id"] for row in stored_before} <= {row["store_id"] for row in rows}, evidence
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert [(session, stamp) for _sid, session, stamp in copies] == [("S", T_REQ), ("rotated", T_COMMIT)], evidence
    assert {(a, b) for a, b in links if a == copies[1][0]} == {(copies[1][0], copies[0][0])}, evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_retained_prompt_with_a_model_switch_note_still_vetoes(build, cold):
    """The host prefixes a model-switch note to the retained old-stamp prompt; it is still that prompt, so a genuine
    repeat at a new stamp stays unlinked."""
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    old = [row["store_id"] for row in all_rows(engine) if row["content"] == P]
    assert len(old) == 1
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    next(row for row in view if row.get("content") == P)["content"] = NOTE + P
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view += [{"role": "user", "content": P, "timestamp": T_COMMIT},
             {"role": "user", "content": NEW, "timestamp": T_COMMIT + 10}]
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    repeat = [row["store_id"] for row in rows if row["content"] == P and row["observed_at"] == T_COMMIT]
    evidence = {"old": old, "repeat": repeat, "relations": sorted(links)}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(repeat) == 1, evidence
    assert not any(a == repeat[0] for a, _b in links), evidence


def test_transfer_target_found_at_its_alt_stamp_too_links_once(build):
    """Two carried turns are re-stamped and linked; the transferred prompt's original carries an alt_stamp at the
    second turn's stamp, so the stamp lookup returns it once per stamp. It is one occurrence: link exactly once."""
    t2, t2_req = "and then also check the staging logs please", T_REQ + 0.5
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ},
                               {"role": "user", "content": t2, "timestamp": t2_req}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    original = next(row["store_id"] for row in all_rows(engine) if row["content"] == P)
    engine._store.add_message_relations([[(original, "alt_stamp", None, None, t2_req)]])
    engine.on_session_end("S", deepcopy(view))
    engine.on_session_start("S", conversation_id="conversation", platform="cli")
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 2
    next(row for row in view if row.get("content") == t2)["timestamp"] = T_COMMIT - 1
    next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    copies = {content: [row["store_id"] for row in rows if row["content"] == content] for content in (T, t2, P)}
    evidence = {"copies": copies, "original": original, "relations": sorted(links)}
    assert all(len(ids) == 2 and (ids[1], ids[0]) in links for ids in copies.values()), evidence
    assert [(a, b) for a, b in sorted(links) if a == copies[P][1]] == [(copies[P][1], original)], evidence


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_scaffold_stripping_never_removes_a_raw_veto(build, cold):
    """The prompt itself is NOTE + P (stored and repeated as such); the host shows the retained old-stamp copy with a
    leading space. The raw stripped forms match, so the veto holds: the genuine repeat stays unlinked. Through ingest
    the leading-space row is re-stored and T's link is not formed, so the linker is driven directly."""
    engine = build()

    def store(content, stamp):
        return engine._store.append("S", {"role": "user", "content": content, "timestamp": stamp},
                                    conversation_id="conversation")

    t_orig, p_orig = store(T, T_REQ), store(NOTE + P, T_REQ)
    t_new, repeat = store(T, T_COMMIT - 1), store(NOTE + P, T_COMMIT)
    if cold:
        engine.shutdown()
        engine = build()
    identity = engine._message_replay_identity(engine._store.get_batch([t_new])[t_new], stored_row=True)
    digest = engine_module._commit_proof_identity_digest(engine_module._proof_user_identity(identity))
    messages = [{"role": "user", "content": T, "timestamp": T_COMMIT - 1},
                {"role": "user", "content": " " + NOTE + P, "timestamp": T_REQ},
                {"role": "user", "content": NOTE + P, "timestamp": T_COMMIT}]
    engine._link_restamped_occurrences(messages, {0: t_new, 2: repeat},
                                       {0: ("S", {"timestamp": T_REQ, "identity_sha256": digest})}, False)
    links = replay_links(engine)
    evidence = {"p_orig": p_orig, "repeat": repeat, "relations": sorted(links)}
    assert (t_new, t_orig) in links, evidence
    assert not any(a == repeat for a, _b in links), evidence
