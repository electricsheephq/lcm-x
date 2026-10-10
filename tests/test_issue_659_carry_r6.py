"""#659 round 6: the carry is never normalised in storage, and never nests.

Every user row is stored exactly as submitted unless an unchanged base mechanism consumes it (the compress commit
proof, the cursor, the pure-summary-scaffold rule). On the commit-proof path (compress, session end, rotation,
then the emitted window) storage equals base. A carry-bearing head stored whole on a no-proof path is never
carried again: carry eligibility reads stored rows through a read-only view, storage keeps the bytes.
"""

import json
import re

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tools as lcm_tools
import tests.test_issue_659_user_carry_packet as carry_packet
from tests.test_issue_659_carry_r4 import _generated
from tests.test_issue_659_user_carry_packet import CARRY, SEP, _compact, _history, _host_merge, _tools

make = carry_packet.make  # the shared fixture

PRE = [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}]
# Hermes user-role continuations that are not human turns (hermes-agent f925b017, agent/conversation_loop.py;
# classified synthetic by agent/conversation_compression.py _is_real_user_message). REVIEW5 R5-N1.
NUDGES = {
    "codex-ack": {"role": "user", "content": "[System: Continue now. Execute the required tool calls and only send "
                                             "your final answer after completing the task.]"},
    "degenerate-final": {"role": "user", "content": (
        "[System: Your previous message ended the turn with a fragment that is not a usable answer. If the task is "
        "unfinished, continue it and then give the complete answer. If that fragment WAS your complete answer, "
        "send it again exactly as before.]")},
    "dropped-toolcall": {"role": "user", "_dropped_toolcall_nudge": True, "content": (
        "Your previous turn indicated a tool call but none was included. Do not narrate a plan or restate intent "
        "— issue the actual tool call now to continue the task.")},
}
TODO = "[Your active task list was preserved across context compression]\n- [ ] 1. finish the task (pending)"


def _rotate(engine, hops=1, start=1):
    for i in range(start, start + hops):
        engine.on_session_start(f"P{i}", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression", old_session_id=f"P{i - 1}")
    assert engine._compression_boundary_ingest_pending


def _rows(engine):
    return [(r["role"], r["content"]) for r in engine._store.get_session_messages(engine._session_id)]


@pytest.mark.parametrize("nudge", sorted(NUDGES))
def test_one_human_quote_with_a_host_continuation_is_stored_whole(make, nudge):
    """R5-N1: one human turn quoting the emitted head, then a synthetic Hermes continuation: 871 bytes stay 871."""
    engine = make(session="P0")
    _out, pure = _generated(engine)
    engine.on_session_end("P0", PRE)
    _rotate(engine)
    assert len(pure.encode("utf-8")) == 871
    engine.ingest([{"role": "user", "content": pure}, {"role": "assistant", "content": "interim response"},
                   dict(NUDGES[nudge]), {"role": "assistant", "content": "final response"}])
    assert _rows(engine)[0] == ("user", pure)


@pytest.mark.parametrize("between", [False, True])
def test_two_queued_humans_after_a_quote_store_the_quote_whole(make, between):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    engine.on_session_end("P0", PRE)
    _rotate(engine)
    engine.ingest([{"role": "user", "content": pure}, *([{"role": "assistant", "content": "own answer"}] * between),
                   {"role": "user", "content": "NEXT QUEUED USER TURN"}])
    assert _rows(engine)[0] == ("user", pure) and _rows(engine)[-1] == ("user", "NEXT QUEUED USER TURN")


def _shape(shape):
    return [*_history(8), *{
        "summary": [{"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}],
        "objective": [{"role": "user", "content": "ACTIVE request"}, *_tools(70), *_tools(71)],
        "fresh-user-merge": [{"role": "user", "content": "TAIL-USER: one more thing."}, *_tools(90, 50),
                             {"role": "assistant", "content": "tail reply"}]}[shape]]


@pytest.mark.parametrize("todo", [False, True])
@pytest.mark.parametrize("shape", ["summary", "objective", "fresh-user-merge"])
def test_normal_path_host_merge_of_real_compress_output_stores_what_base_stores(make, shape, todo):
    """REVIEW5 P-NORMAL-PATH, non-empty fresh tail: the commit proof consumes the emitted head; rows and cursor are
    base's (ce186095, recorded in r6-probes/r6-ce186095-normal-*.json)."""
    engine = make(session="P0")
    history = _shape(shape)
    out = _compact(engine, history)
    assert CARRY in out[0]["content"]
    engine.on_session_end("P0", history)
    _rotate(engine)
    request = TODO + "\n\nNEW REAL REQUEST" if todo else "NEW REAL REQUEST"
    host = _host_merge([*out, *([{"role": "user", "content": TODO}] if todo else []),
                        {"role": "user", "content": "NEW REAL REQUEST"}, {"role": "assistant", "content": "new reply"}])
    engine.ingest(host)
    assert _rows(engine) == [("user", request), ("assistant", "new reply")]
    assert engine._ingest_cursor == len(host) and not engine._compression_boundary_ingest_pending


@pytest.mark.parametrize("hops", [1, 2, 33])
@pytest.mark.parametrize("kind", ["summary", "objective"])
def test_rotated_identity_on_the_commit_proof_path_equals_base(make, kind, hops, tmp_path):
    """P-ROTATED-IDENTITY where the commit proof applies: child rows and cursor, then a cold bind, equal base's."""
    engine = make(session="P0", home=tmp_path)
    history = _shape(kind)
    out = _compact(engine, history)
    engine.on_session_end("P0", history)
    _rotate(engine, hops)
    tail = [{"role": "user", "content": "new request"}, {"role": "assistant", "content": "ok"}]
    engine.ingest([*out, *tail])
    assert _rows(engine) == [("user", "new request"), ("assistant", "ok")] and engine._ingest_cursor == len(out) + 2
    engine.shutdown()
    resumed = make(session=f"P{hops}", home=tmp_path)
    resumed.ingest([*out, *tail, {"role": "user", "content": "next"}, {"role": "assistant", "content": "next answer"}])
    assert _rows(resumed) == [("user", "new request"), ("assistant", "ok"), ("user", "next"),
                              ("assistant", "next answer")]
    assert resumed._last_ingest_reconciliation["cursor"] == len(out) + 2


def _packet(engine, out):
    """(carry part, the store ids it carries)."""
    carry = carry_packet._generated(engine, out)[1]
    header = engine_module._USER_CARRY_HEADER_RE.match(carry)
    return carry, {int(i) for i in header.group(1).split(", ")} if header else set()


def _stored_ids(engine, text):
    return {r["store_id"] for r in engine._store.get_session_messages(engine._session_id) if r["content"] == text}


def _new_turns(cycle):
    return [row for i in range(4) for row in carry_packet._turn(100 * cycle + i)]


@pytest.mark.parametrize("path", ["lone-quote", "prefixed-quote", "window-without-session-end"])
def test_a_carry_head_stored_whole_is_never_carried_again(make, path):
    """No nesting: a no-proof rotation stores the carry-bearing head whole; the next packet holds none of it. A
    user's own paste of a verified carry block keeps its bytes in the store; only its re-carry is skipped."""
    engine = make(session="P0")
    out, pure = _generated(engine)
    _rotate(engine)
    text = "Please inspect this copied context:" + SEP + pure if path == "prefixed-quote" else pure
    window = [*out, {"role": "user", "content": "new request"}] if path == "window-without-session-end" else [
        {"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]
    engine.ingest(window)
    head_ids = _stored_ids(engine, out[0]["content"] if path == "window-without-session-end" else text)
    assert head_ids  # stored whole
    active = window if path == "window-without-session-end" else [*out, *window]  # what the host compresses
    out2 = _compact(engine, [*active, *_new_turns(1), {"role": "user", "content": "latest 1"},
                             {"role": "assistant", "content": "ok"}])
    assert engine._last_compression_status == "compacted" and engine._last_compacted_store_id > max(head_ids)
    stored = _stored_ids(engine, out[0]["content"] if path == "window-without-session-end" else text)
    packet, carried = _packet(engine, out2)
    assert head_ids <= stored and packet.count(CARRY) == 1 and carried and not carried & stored
    status = engine._last_user_carry
    assert status["entries"] > 0 and status["delivered_tokens"] <= status["budget_tokens"]


@pytest.mark.parametrize("shape", ["window", "empty-tail-composite"])
def test_three_no_proof_rotations_do_not_grow_the_packet(make, monkeypatch, shape):
    """Each cycle stores the emitted head whole (no session end, so no commit proof; or the host glued the next
    request onto a lone head); the packet carries real rows only, so its size stays flat over the rotations."""
    monkeypatch.setattr(engine_module, "_USER_CARRY_MAX_TOKENS", 400, raising=False)
    engine = make(session="P0", fresh_tail_count=0 if shape == "empty-tail-composite" else 4)
    out = _compact(engine, [*_history(8), {"role": "user", "content": "latest 0"},
                            {"role": "assistant", "content": "ok"}])
    sizes, heads = [], set()
    for cycle in range(1, 4):
        _rotate(engine, start=cycle)
        window = _host_merge([*out, *_new_turns(cycle), {"role": "user", "content": f"latest {cycle}"},
                              {"role": "assistant", "content": "ok"}])
        engine.ingest(window)
        heads |= _stored_ids(engine, window[0]["content"])
        assert len(heads) == cycle and CARRY in window[0]["content"]  # stored whole: no proof, no normalisation
        out = _compact(engine, window)
        assert engine._last_compression_status == "compacted"
        packet, carried = _packet(engine, out)
        assert packet.count(CARRY) == 1 and carried and not carried & heads
        assert engine._last_user_carry["delivered_tokens"] <= engine._last_user_carry["budget_tokens"]
        sizes.append(len(packet))
    assert max(sizes) <= 1.25 * min(sizes), sizes  # budget granularity (whole rows, one excerpt), not growth


@pytest.mark.parametrize("lead", ["tool", "assistant", "two-rotations"])
def test_a_first_new_only_quote_is_stored_whole_after_any_lead(make, lead):
    """REVIEW4/5 pending cases: a quote behind a leading non-user row, or after two rotations, keeps its bytes."""
    engine = make(session="P0")
    _out, pure = _generated(engine)
    engine.on_session_end("P0", PRE)
    _rotate(engine, 2 if lead == "two-rotations" else 1)
    first = [] if lead == "two-rotations" else [{"role": lead, "content": "FIRST ROW"}]
    engine.ingest([*first, {"role": "user", "content": pure}, {"role": "assistant", "content": "ok"}])
    assert ("user", pure) in _rows(engine)


def _carried(out):
    """(rendered text of the output, every store id named by a carry header in it)."""
    text = "\n".join(str(m.get("content") or "") for m in out)
    ids = set()
    for header in engine_module._USER_CARRY_HEADER_RE.finditer(text):
        ids.update(int(i) for i in header.group(1).split(", "))
    return text, ids


@pytest.mark.parametrize("defaults", [False, True])
@pytest.mark.parametrize("shape", ["glued", "merged-user", "middle", "prefixed", "externalized",
                                   "large-emitted-head", "block-at-start"])
def test_stored_head_is_not_recarried(make, monkeypatch, defaults, shape):
    """REVIEW6 nesting probes: a verified carry stored whole (at row start, glued, quoted, externalized, or as a
    large emitted head) stays recoverable through lcm_expand and is never carried again over three no-proof
    rotations; the packet never nests and stays within budget."""
    engine = make(session="P0", temporal_rollups_enabled=False, large_output_externalization_enabled=defaults,
                  large_output_active_replay_stubbing_enabled=defaults)
    if shape == "large-emitted-head":
        large_summary = "Ordinary summarized history. " * 3800
        monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: (large_summary, 1))
    _out, pure = _generated(engine)
    if shape == "large-emitted-head":
        assert len(pure) > 100000 and CARRY in pure
    block = pure[pure.index(SEP + CARRY) + len(SEP):]
    text = {
        "glued": pure + "\n\nNEW REAL REQUEST",
        "merged-user": pure + "\n\nFIRST USER TURN\n\nSECOND USER TURN",
        "middle": "MY QUOTATION" + SEP + pure + "\n\nMY SUFFIX TEXT",
        "prefixed": "MY QUOTATION" + SEP + pure,
        "externalized": ("Ordinary human document paragraph. " * 3500) + SEP + pure + "\n\nMY SUFFIX TEXT",
        "large-emitted-head": pure,
        "block-at-start": block + "\n\nMY SUFFIX TEXT",
    }[shape]
    _rotate(engine)
    window = [{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]
    active = engine._ingest_messages(window)
    head = engine._store.get_session_messages("P1")[0]
    sid = head["store_id"]
    ext = head["content"].startswith("[Externalized payload:")
    recovered = json.loads(lcm_tools.lcm_expand({"store_id": sid, "max_tokens": 1000000}, engine=engine)).get("content")
    if ext:
        refs = re.findall(r";\s*ref=([^;\]\s]+)", head["content"])
        assert refs
        recovered = json.loads(lcm_tools.lcm_expand({"externalized_ref": refs[0], "max_tokens": 1000000},
                                                    engine=engine)).get("content")
    assert recovered == text, "every submitted byte is recoverable through lcm_expand"
    cycles = []  # three no-proof rotations with real compress output; each new head retained whole
    for cycle in range(1, 4):
        history = [*active, *_new_turns(cycle), {"role": "user", "content": f"latest {cycle}"},
                   {"role": "assistant", "content": "ok"}]
        out2 = _compact(engine, history)
        rendered, ids = _carried(out2)
        cycles.append({"head_id_carried": sid in ids, "carry_blocks": rendered.count(CARRY),
                       "delivered": engine._last_user_carry})
        if cycle < 3:
            engine.on_session_start(f"P{cycle + 1}", platform="cli", context_length=128000, conversation_id="conv",
                                    boundary_reason="compression", old_session_id=f"P{cycle}")
            active = engine._ingest_messages(out2)
    assert not any(c["head_id_carried"] for c in cycles), "whole-stored verified carry head must never be selected"
    assert all(c["carry_blocks"] <= 1 for c in cycles), "the carry packet must not nest"
    assert all(c["delivered"]["delivered_tokens"] <= c["delivered"]["budget_tokens"] for c in cycles)


@pytest.mark.parametrize("shape", ["plain", "malformed-header", "changed-id", "edited-body", "separator-only",
                                   "manifest-only"])
def test_unverified_genuine_row_is_carried(make, shape):
    """REVIEW6 false positives: a genuine user row that is not a verified carry block is still carried."""
    engine = make(session="P0", temporal_rollups_enabled=False)
    _out, pure = _generated(engine)
    text = {
        "plain": "REAL USER: instructions after the compaction",
        "malformed-header": "REAL USER" + SEP + CARRY + ": mine]",
        "changed-id": "REAL USER" + SEP + pure.replace("store ids 1,", "store ids 999999,"),
        "edited-body": "REAL USER" + SEP + pure.replace("REQ-0:", "EDITED-0:"),
        "separator-only": "REAL USER" + SEP + "unrelated text",
        "manifest-only": "REAL USER" + SEP + "[Summary parts omitted for space: MY OWN TEXT]",
    }[shape]
    _rotate(engine)
    engine._ingest_messages([{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}])
    head = engine._store.get_session_messages("P1")[0]
    verified = engine._holds_verified_user_carry(head)
    engine._last_compacted_store_id = head["store_id"]
    engine._user_carry_cache = None
    parts = engine._user_carry_parts([], "user", [], [], [], [], [], None, None)
    _rendered, ids = _carried([{"role": "user", "content": SEP.join(parts)}])
    assert not verified and head["store_id"] in ids
