"""#1162 fix E: a capped assembly keeps the newest visible reply when it would drop it (LCM-X v0.27.1).

Engine-level on scratch stores with stubbed summaries. ``_host_keep_last_reply`` is a faithful copy of the Hermes
rule that re-adds a reply the engine dropped; the flow tests commit its result as the next host view."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens

PAD = " alpha beta gamma delta" * 30
RESIDUAL = "reply_keep=residual"


@pytest.fixture(autouse=True)
def _summaries(monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                        lambda **_kw: ("Earlier turns.\nExpand for details about: turns", 1))


# ---- Faithful copy of the Hermes placement rule: v0.21.7 agent/conversation_compression_reply_anchor.py
# _ensure_compressed_keeps_last_assistant_reply / _reply_insertion_index / _same_visible_content /
# _reused_tool_call_ids (module byte-identical on r34.x and upstream main). Test data carries no host summary
# rows, so is_compaction_summary_message is False; _is_real_user_message is "a user row with text" (the host
# classes LCM's summaries as real users); estimate_messages_tokens_rough is len(json) / 4 per row.
def _host_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or p.get("content") or "") for p in content if isinstance(p, dict))
    return ""


def _call_ids(message) -> frozenset:
    return frozenset(filter(None, (str(tc.get("call_id") or tc.get("id") or "").split("|", 1)[0].strip()
                                   for tc in message.get("tool_calls") or ())))


def _result_id(message) -> str:
    return str(message.get("tool_call_id") or "").split("|", 1)[0].strip()


def _same_visible(left, right, right_text) -> bool:
    left_text = _host_text(left).strip()
    return left_text == right_text if (left_text or right_text) else left.get("content") == right.get("content")


def _rough(messages) -> int:
    return sum(len(json.dumps(m, default=str)) // 4 for m in messages)


def _real_user(message) -> bool:
    return isinstance(message, dict) and message.get("role") == "user" and bool(_host_text(message).strip())


def _insertion_index(followers, compressed, reused_ids):
    def row_ids(message):
        return _call_ids(message) | frozenset(filter(None, (_result_id(message),)))

    ambiguous = False
    for follower in followers:
        role, text, follower_calls = follower.get("role"), _host_text(follower).strip(), _call_ids(follower)
        if row_ids(follower) & reused_ids:
            ambiguous = ambiguous or any(row_ids(m) & reused_ids for m in compressed)
            continue
        for pos in range(len(compressed) - 1, -1, -1):
            message = compressed[pos]
            if message.get("role") != role:
                continue
            message_calls = _call_ids(message)
            if follower_calls or message_calls:
                if message_calls == follower_calls:
                    return pos
                continue
            if role == "tool" and (_result_id(follower) or _result_id(message)):
                if _result_id(message) == _result_id(follower):
                    return pos
                continue
            if _same_visible(message, follower, text):
                return pos
    if ambiguous:
        return None
    if followers and any(_real_user(f) for f in followers) and compressed and _real_user(compressed[-1]):
        return len(compressed) - 1
    return len(compressed)


def _host_keep_last_reply(original, compressed):
    """Hermes' post-compress reply anchor on ``compressed`` (mutated); returns the inserted row or None."""
    reply, reply_text = None, ""
    for message in reversed(original):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        reply_text = _host_text(message).strip()
        if reply_text:
            reply = message
            break
    if reply is None:
        return None

    def twin(message):
        return (message.get("role") == "assistant" and not message.get("tool_calls")
                and _same_visible(message, reply, reply_text))

    if any(twin(m) for m in compressed):
        return None
    seen = Counter(c for m in original if m.get("role") == "assistant" for c in _call_ids(m))
    reply_pos = next(i for i, m in enumerate(original) if m is reply)
    index = _insertion_index(original[reply_pos + 1:], compressed, frozenset(c for c, n in seen.items() if n > 1))
    if index is None:
        return None
    neighbours = [compressed[i] for i in (index - 1, index) if 0 <= i < len(compressed)]
    if any(twin(m) for m in neighbours) or any(m.get("role") == "assistant" for m in neighbours):
        return None
    anchor = dict(reply)
    if _rough(compressed) < _rough(original) < _rough(compressed) + _rough([anchor]):
        return None
    compressed.insert(index, anchor)
    return anchor
# ---- end of the host copy


class _Host:
    """Rows as Hermes builds them: a timestamp each, and a message_uid when ``uid``."""

    def __init__(self, uid: bool = True):
        self.uid, self.clock = uid, 1_700_000_000.0

    def _stamp(self, row: dict) -> dict:
        self.clock += 1.0
        row["timestamp"] = self.clock
        if self.uid:
            row["message_uid"] = f"host-{int(self.clock)}"
        return row

    def user(self, text: str, pad: str = PAD) -> dict:
        return self._stamp({"role": "user", "content": f"{text}{pad}"})

    def reply(self, text: str, pad: str = PAD) -> dict:
        return self._stamp({"role": "assistant", "content": f"{text}{pad}"})

    def call(self, i: int) -> dict:
        return self._stamp({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": f'{{"n": {i}}}'}}]})

    def result(self, i: int, pad: str = PAD) -> dict:
        return self._stamp({"role": "tool", "tool_call_id": f"call_{i}", "content": f"file {i}{pad}"})


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "dynamic_leaf_chunk_max": 400, "temporal_rollups_enabled": False,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id or 0)


def _stops(engine) -> list:
    """Records each pass whose leaf selection ended before an unresolved retained occurrence (the #1162 stop)."""
    seen, original = [False], engine._identity_anchor_summary_input

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen[-1] = seen[-1] or out == []
        return out

    engine._identity_anchor_summary_input = spy
    return seen


def _assert_lossless(engine) -> None:
    """Every row at or below the frontier is covered by exactly one message leaf, leaves in store order."""
    leaves = [n for n in engine._dag.get_session_nodes("S") if n.source_type == "messages"]
    covered = Counter(int(s) for n in leaves for s in n.source_ids)
    frontier = _frontier(engine)
    below = [int(r["store_id"]) for r in engine._store.get_session_messages("S", limit=100_000)
             if int(r["store_id"]) <= frontier]
    assert below and all(covered[s] == 1 for s in below), (frontier, below, covered)
    spans = [sorted(int(s) for s in n.source_ids) for n in sorted(leaves, key=lambda n: n.node_id)]
    assert all(a[-1] < b[0] for a, b in zip(spans, spans[1:])), spans


def _e2(tmp_path, *, sweep=True, uid=True, tail=7, cap=1300, passes=6):
    """The #1162 `e2` wedge: a capped first pass drops the uncovered reply A whose follower "continue" has a later
    twin after a tool round; the host rule commits each result as the next view."""
    engine = _engine(tmp_path, fresh_tail_count=tail, max_assembly_tokens=cap, threshold_full_sweep_enabled=sweep)
    stops, host = _stops(engine), _Host(uid)
    big = PAD * 3
    view = [m for i in range(1, 5) for m in (host.user(f"[T{i}] ask"), host.reply(f"reply to T{i}"))]
    view += [host.user("continue", pad=big), host.call(1), host.result(1), host.user("continue", pad=big),
             host.call(2), host.result(2)]
    trace = []
    try:
        for k in range(passes + 1):
            if k:
                view = view + [host.call(k + 2), host.result(k + 2)]
                engine.ingest(view)
                engine.note_turn_complete()
                view = view + [host.reply(f"reply {k + 2}"), host.user(f"[N{k + 2}] ask")]
            stops.append(False)
            out = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
            engine._no_progress_hold = None
            first = list(out)
            inserted = _host_keep_last_reply(view, out)
            trace.append({"frontier": _frontier(engine), "stop": stops[-1], "inserted": inserted is not None,
                          "status": engine._last_compression_status, "roles": "".join(m["role"][0] for m in out),
                          "out": first})
            engine._config.max_assembly_tokens = 0  # the cap was an overflow-like event on the first pass only
            view = out
    except BaseException:
        engine.shutdown()
        raise
    return engine, trace


def _brief(trace) -> list:
    return [{k: v for k, v in t.items() if k != "out"} for t in trace]


@pytest.mark.parametrize("sweep", [True, False])
@pytest.mark.parametrize("uid", [True, False])
def test_p1_capped_pass_keeps_uncovered_reply_and_the_wedge_never_forms(tmp_path, sweep, uid):
    """P1: on the base the host re-adds A behind F2's tool round and every later pass stops (`[]`); with E the
    capped output already shows A at its store-order slot, the host inserts nothing and the frontier rises."""
    engine, trace = _e2(tmp_path, sweep=sweep, uid=uid)
    engine.shutdown()
    frontiers = [t["frontier"] for t in trace]
    run = max(map(len, "".join("x" if t["stop"] else " " for t in trace).split()), default=0)
    inserts = [i for i, t in enumerate(trace) if t["inserted"]]
    assert run == 0 and not inserts, (
        f"{run} consecutive retained-occurrence stops; host inserts at passes {inserts}; frontiers {frontiers}")
    assert all(b > a for a, b in zip(frontiers, frontiers[1:])), _brief(trace)
    first = trace[0]["out"]
    texts = [_host_text(m) for m in first]
    reply = next(i for i, t in enumerate(texts) if t.startswith("reply to T4"))
    assert first[0]["role"] == "user" and reply == 1, _brief(trace)  # [summary, A, F2, ...]: step 2
    assert texts[reply + 1].startswith("continue")


def test_p4_interior_gap_rows_are_summarised_in_store_order(tmp_path, caplog):
    """P4: the step-2 view hides the uncovered rows between A and F2; they reach a leaf in store order, with no
    retained-occurrence stop and no publication error, and the lossless audit is clean."""
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    engine, trace = _e2(tmp_path, passes=8)
    try:
        rows = engine._store.get_session_messages("S", limit=100_000)
        _assert_lossless(engine)
        frontier = _frontier(engine)
    finally:
        engine.shutdown()
    assert not any(t["stop"] or t["inserted"] or t["status"] == "error" for t in trace), _brief(trace)
    assert not [r for r in caplog.records if "publication" in r.getMessage()]
    reply_id = next(int(r["store_id"]) for r in rows if str(r["content"]).startswith("reply to T4"))
    twin_id = [int(r["store_id"]) for r in rows if str(r["content"]).startswith("continue")][1]
    assert frontier > twin_id > reply_id + 1  # the hidden gap (F, tc1, tr1) sits under the frontier


def _steps(monkeypatch) -> list:
    """The E step each assembly applied (1, 2) or None, in call order (re-entries included); empty on the base."""
    applied, original = [], getattr(LCMEngine, "_keep_newest_reply", None)
    if original is None:
        return applied

    def spy(self, *args):
        out = original(self, *args)
        applied.append(out[2]["step"] if out[2] else None)
        return out

    monkeypatch.setattr(LCMEngine, "_keep_newest_reply", spy)
    return applied


def _disable_e(monkeypatch) -> None:
    """The base selection (on 76769913 the attribute does not exist and this is a no-op)."""
    monkeypatch.setattr(LCMEngine, "_keep_newest_reply", lambda self, *args: (args[-3], args[-2], None),
                        raising=False)


def _overflow_view(engine, host, fpad=PAD * 3, *, objective=False):
    """A summary over T1-T3, then [U4, A4, F, tc1, tr1, F2, tc2, tr2] (or, with ``objective``, a tool loop behind
    a commentary reply under the newest user)."""
    view = [m for i in range(1, 4) for m in (host.user(f"[T{i}] ask"), host.reply(f"reply to T{i}"))]
    view = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
    engine._no_progress_hold = None
    if objective:
        todo = host.user("[Your active task list was preserved across context compression]\n- [ ] ship it", pad="")
        return view + [host.user("[OBJ] run the release workflow", pad=""), host.reply("reply to OBJ", pad=PAD * 2), todo,
                       host.call(1), host.result(1), host.call(2), host.result(2)]
    return view + [host.user("[T4] ask"), host.reply("reply to T4"), host.user("continue", pad=fpad), host.call(1),
                   host.result(1), host.user("continue", pad=fpad), host.call(2), host.result(2)]


def _overflow_pass(engine, view, cap):
    """A forced overflow recovery with no leaf (the fresh tail holds every row): overflow assembly under ``cap``."""
    engine._config.fresh_tail_count, engine._config.max_assembly_tokens = 64, cap
    out = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
    engine._no_progress_hold = None
    engine._config.fresh_tail_count, engine._config.max_assembly_tokens = 2, 0
    return out


@pytest.mark.parametrize(("fpad", "cap", "step"), [(" small", 560, 1), (PAD * 3, 900, 2)], ids=["step1", "step2"])
def test_p2_overflow_recovery_keeps_reply_status_and_next_pass(tmp_path, monkeypatch, fpad, cap, step):
    """P2: overflow recovery (stub_over_cap_tool_results) with a small follower (step 1: the gap fits) and a big
    one (step 2). The finalizer reports overflow_recovery and not failed (cap + tokens(A)), the host inserts
    nothing, and the next pass compacts with no retained-occurrence stop."""
    applied = _steps(monkeypatch)
    engine, host = _engine(tmp_path), _Host()
    stops = _stops(engine)
    try:
        view = _overflow_view(engine, host, fpad)
        reply = next(m for m in view if _host_text(m).startswith("reply to T4"))
        out = _overflow_pass(engine, view, cap)
        texts = [_host_text(m) for m in out]
        assert out[0]["role"] == "user" and texts[1] == reply["content"] and texts[2].startswith("continue"), texts
        assert engine._last_compression_status == "overflow_recovery"
        assert cap < count_messages_tokens(out) <= cap + count_message_tokens(reply)
        assert engine._last_overflow_recovery_failed is False
        assert applied[-1] == step, applied
        assert _host_keep_last_reply(view, out) is None
        before = _frontier(engine)
        view = out + [host.call(3), host.result(3), host.reply("reply 3"), host.user("[N3] ask")]
        stops.append(False)
        engine.compress(view, current_tokens=count_messages_tokens(view))
        assert stops[-1] is False and _frontier(engine) > before
    finally:
        engine.shutdown()


def _shape_output(tmp_path, shape):
    """P3 shapes through the overflow assembly; returns the compressed list."""
    engine, host = _engine(tmp_path), _Host()
    try:
        if shape == "no_reply":
            view = [m for i in range(1, 7) for m in (host.user(f"[U{i}] ask"), host.call(i), host.result(i))]
            view = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
            return _overflow_pass(engine, view + [host.call(7), host.result(7)], 700)
        view = _overflow_view(engine, host)
        if shape == "uncapped":
            engine._config.fresh_tail_count = 64
            return list(engine.compress(view, current_tokens=count_messages_tokens(view)))
        if shape == "covered":
            # Hermes re-added A after it was summarised (covered): the copy sits right after the summary.
            engine._config.fresh_tail_count = 6
            out = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
            reply = next(m for m in view if _host_text(m).startswith("reply to T4"))
            rows = engine._store.get_session_messages("S", limit=100_000)
            assert next(int(r["store_id"]) for r in rows if r["content"] == reply["content"]) <= _frontier(engine)
            view = out[:1] + [dict(reply)] + out[1:]
            return _overflow_pass(engine, view, 900)
        return _overflow_pass(engine, view, {"kept": 2400, "empty": 1}[shape])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["covered", "uncapped", "kept", "empty", "no_reply"])
def test_p3_other_paths_are_byte_identical(tmp_path, monkeypatch, caplog, shape):
    """P3: covered A, uncapped, A already kept, empty suffix, no visible reply: the output equals the base
    selection's byte for byte, and E logs nothing."""
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    with_e = _shape_output(tmp_path / "e", shape)
    assert not [r for r in caplog.records if "reply_keep" in r.getMessage()]
    shown = any(_host_text(m).startswith("reply to T4") for m in with_e)
    assert shown is (shape in ("uncapped", "kept")), shape
    _disable_e(monkeypatch)
    base = _shape_output(tmp_path / "base", shape)
    assert json.dumps(with_e, sort_keys=True) == json.dumps(base, sort_keys=True)


def test_p5_preserved_objective_stays_when_e_fires(tmp_path, monkeypatch):
    """P5: the newest real user row sits before A (a todo snapshot follows A), so its preserved objective heads the
    summary. E keeps A over the cap, and the over-cap pass, measured against cap + tokens(A), keeps the objective."""
    applied = _steps(monkeypatch)
    engine, host = _engine(tmp_path), _Host()
    try:
        view = _overflow_view(engine, host, objective=True)
        reply = next(m for m in view if _host_text(m).startswith("reply to OBJ"))
        out = _overflow_pass(engine, view, 440)
        failed = engine._last_overflow_recovery_failed
    finally:
        engine.shutdown()
    assert out[1]["content"] == reply["content"] and out[2]["content"].startswith("[Your active task list")
    assert out[0]["content"].startswith("[Current user objective preserved from compacted history]")
    assert 440 < count_messages_tokens(out) <= 440 + count_message_tokens(reply) and failed is False
    assert applied[-1] == 1, applied


def _zero_room_pass(tmp_path):
    engine, host = _engine(tmp_path), _Host()
    try:
        view = _overflow_view(engine, host)
        cap = sum(count_message_tokens(m) for m in view[-5:])  # tc1, tr1, F2, tc2, tr2 fill the cap exactly
        return view, _overflow_pass(engine, view, cap)
    finally:
        engine.shutdown()


def test_p8_zero_summary_room_rejects_step_1_and_takes_step_2(tmp_path, monkeypatch):
    """P8: the kept rounds fill the cap exactly, so the base view has no summary room (it starts at F2). Step 1 is
    rejected for room; step 2 frees tc1/tr1 for the summary, which keeps the leading user row."""
    applied = _steps(monkeypatch)
    view, out = _zero_room_pass(tmp_path / "e")
    assert [_host_text(m)[:11] for m in out[:3]] == ["[Recent Sum", "reply to T4", "continue al"]
    assert applied == [2], applied
    assert _host_keep_last_reply(view, list(out)) is None
    _disable_e(monkeypatch)
    _view, base = _zero_room_pass(tmp_path / "base")
    assert [_host_text(m)[:11] for m in base[:1]] == ["continue al"]


def _retained_user_assembly(tmp_path, carrier: bool, cap: int):
    """The retained sole-user path (summary role assistant): A, a todo snapshot, then tool rounds, assembled under
    ``cap`` without writes. ``carrier``: A arrives already folded with the summary by a previous assembly."""
    todo = "[Your active task list was preserved across context compression]\n- [ ] ship it"
    engine, host = _engine(tmp_path), _Host()
    system, prompt = {"role": "system", "content": "stable system prompt"}, host.user("the only real user prompt")
    try:
        view = [system, prompt] + [m for i in (1, 2, 3) for m in (host.call(i), host.result(i))]
        out0 = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
        tail = [host.reply("reply to the prompt"), host.user(todo, pad="")]
        tail += [m for i in (4, 5) for m in (host.call(i), host.result(i))]
        engine.ingest(out0 + tail)
        engine._pending_context_anchor_messages = view[2:] + tail  # the host input, as the compaction path sets it
        if carrier:
            folded = engine._assemble_context(system, tail, assembly_cap_override=820, include_lcm_note=False,
                                              retained_user_message=prompt)
            assert folded[2]["content"].startswith("[Recent Summary") and "reply to the prompt" in folded[2]["content"]
            # the host commits the assembly; compress() records it the same way
            engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(folded), False
            tail = folded[2:] + [host.call(6), host.result(6)]
            engine.ingest(folded + tail[-2:])  # the next turn reaches LCM through the host's per-turn ingest
            engine._pending_context_anchor_messages = view[2:] + tail
        out = engine._assemble_context(system, tail, assembly_cap_override=cap, include_lcm_note=False,
                                       retained_user_message=prompt, persist=False)
        return tail, out
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("carrier", "cap", "kept"), [(False, 700, False), (True, 880, False), (True, 800, True)],
                         ids=["newly_folded", "carrier_residual", "carrier_kept"])
def test_p6_retained_user_and_carrier_shapes(tmp_path, monkeypatch, caplog, carrier, cap, kept):
    """P6: with a retained user the kept A heads the tail, so the summary folds into it (a newly folded A, or an A
    that already arrived folded): the twin is gone, both steps fail the postcondition, and the base output comes
    back with one residual line. Where no summary fits, A stays unfolded and the postcondition holds. Neither
    branch adds assistant;assistant adjacency."""
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    tail, out = _retained_user_assembly(tmp_path / "e", carrier, cap)
    residual = [r.getMessage() for r in caplog.records if RESIDUAL in r.getMessage()]
    roles = [m["role"] for m in out]
    assert not any(a == b == "assistant" for a, b in zip(roles, roles[1:])), roles
    if kept:
        assert not residual and _host_keep_last_reply(tail, list(out)) is None
        assert out[2]["content"] == tail[0]["content"]
        return
    assert len(residual) == 1 and "step1=postcondition, step2=postcondition" in residual[0], residual
    _disable_e(monkeypatch)
    _base_tail, base = _retained_user_assembly(tmp_path / "base", carrier, cap)
    assert json.dumps(out, sort_keys=True) == json.dumps(base, sort_keys=True)


def _carrier_assembly(tmp_path):
    """Review construction (#1162 E r2): e2 tail [A, huge F, carrier(tc1), tr1, F2, tc2, tr2] under cap 1300, where
    carrier(tc1) is a generated-context fold of tc1 registered through the folded-tail lineage, so the store-id map
    resolves it to tc1's raw row. Step 1 has no room; step 2 would hide the carrier."""
    engine, host = _engine(tmp_path), _Host()
    try:
        view = [m for i in range(1, 4) for m in (host.user(f"[T{i}] ask"), host.reply(f"reply to T{i}"))]
        out0 = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
        big = PAD * 3
        rows = [host.user("[T4] ask"), host.reply("reply to T4"), host.user("continue", pad=big), host.call(1),
                host.result(1), host.user("continue", pad=big), host.call(2), host.result(2)]
        engine.ingest(out0 + rows)
        call_id = next(int(r["store_id"]) for r in engine._store.get_session_messages("S", limit=100_000)
                       if r["role"] == "assistant" and "call_1" in json.dumps(r.get("tool_calls")))
        carrier = engine._prepend_generated_context_to_message(rows[3], out0[0]["content"])
        assert engine._write_folded_tail_lineage(carrier, call_id)
        tail = rows[1:3] + [carrier] + rows[4:]
        assert engine._get_store_id_map_for_messages(tail)[id(carrier)] == call_id
        return engine._assemble_context(None, tail, assembly_cap_override=1300, persist=False)
    finally:
        engine.shutdown()


def test_step2_never_hides_a_generated_carrier(tmp_path, monkeypatch, caplog):
    """Step 2 hides only raw stored occurrences: a registered fold (generated context + tc1) maps to tc1's store id,
    yet it is generated, so step 2 refuses it and the shape yields the base output with one residual line."""
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    out = _carrier_assembly(tmp_path / "e")
    residual = [r.getMessage() for r in caplog.records if RESIDUAL in r.getMessage()]
    assert len(residual) == 1 and "step1=room, step2=not_raw_row" in residual[0], [_host_text(m)[:20] for m in out]
    _disable_e(monkeypatch)
    base = _carrier_assembly(tmp_path / "base")
    assert json.dumps(out, sort_keys=True) == json.dumps(base, sort_keys=True)


def test_hidden_survival_projection_raw_row_is_covered_in_store_order(tmp_path, monkeypatch, caplog):
    """Re-check r2 construction: tr1 arrives as its survival-fit projection (a bounded copy whose only source is
    raw row tr1), which the store-id map resolves to tr1's id, so step 2 hides it. That is lossless: once the host
    commits the step-2 view, later passes cover tr1's raw row in exactly one leaf, in store order."""
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    applied = _steps(monkeypatch)
    engine, host = _engine(tmp_path), _Host()
    stops = _stops(engine)
    try:
        view = [m for i in range(1, 4) for m in (host.user(f"[T{i}] ask"), host.reply(f"reply to T{i}"))]
        out0 = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
        rows = [host.user("[T4] ask"), host.reply("reply to T4"), host.user("continue", pad=PAD * 3), host.call(1),
                host.result(1, pad=PAD * 10), host.user("continue", pad=PAD * 3), host.call(2), host.result(2)]
        engine.ingest(out0 + rows)
        source = next(int(r["store_id"]) for r in engine._store.get_session_messages("S", limit=100_000)
                      if r.get("tool_call_id") == "call_1")
        projected = engine._survival_projection([rows[4]], {id(rows[4]): source}, 0)[0]
        assert projected != rows[4] and len(_host_text(projected)) < len(_host_text(rows[4]))
        tail = rows[1:4] + [projected] + rows[5:]
        assert engine._get_store_id_map_for_messages(tail)[id(projected)] == source
        out = engine._assemble_context(None, tail, assembly_cap_override=1300)
        assert applied == [2] and projected not in out and rows[3] not in out, applied  # step 2 hid tc1 + projection
        assert [_host_text(m)[:11] for m in out[1:3]] == ["reply to T4", "continue al"]
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(out), False  # the host commits it
        view, frontiers = out, [_frontier(engine)]
        for k in range(3, 9):
            view = view + [host.call(k), host.result(k)]
            engine.ingest(view)
            engine.note_turn_complete()
            view = view + [host.reply(f"reply {k}"), host.user(f"[N{k}] ask")]
            stops.append(False)
            out = list(engine.compress(view, current_tokens=count_messages_tokens(view)))
            engine._no_progress_hold = None
            assert _host_keep_last_reply(view, out) is None and engine._last_compression_status != "error"
            frontiers.append(_frontier(engine))
            view = out
        leaves = [n for n in engine._dag.get_session_nodes("S") if n.source_type == "messages"]
        holding = [sorted(int(s) for s in n.source_ids) for n in leaves if source in {int(s) for s in n.source_ids}]
        _assert_lossless(engine)
    finally:
        engine.shutdown()
    assert all(b > a for a, b in zip(frontiers, frontiers[1:])), frontiers
    assert not any(stops) and not [r for r in caplog.records if "publication" in r.getMessage()]
    assert frontiers[-1] > source and len(holding) == 1, holding
    assert holding[0] == list(range(holding[0][0], holding[0][-1] + 1)), holding  # contiguous store-order span


def test_host_growth_rejects_kept_reply_and_returns_residual(tmp_path, monkeypatch, caplog):
    """A host-only growth rejection tries both E steps, then returns the base view once."""
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    applied = _steps(monkeypatch)
    checked, original = [], LCMEngine._kept_reply_rejection

    def estimate(messages):
        texts = [_host_text(m) for m in messages]
        return 101 if any(t.startswith("reply to T4") for t in texts) and not any(
            t.startswith("[T4] ask") for t in texts) else 100

    def rejection(self, result, reply, objective_due, host_input):
        checked.append((count_messages_tokens(result), count_messages_tokens(host_input),
                        estimate(result), estimate(host_input)))
        return original(self, result, reply, objective_due, host_input)

    monkeypatch.setattr(lcm_engine, "_host_estimate", estimate, raising=False)  # semantic RED on be9b4b0d
    monkeypatch.setattr(LCMEngine, "_kept_reply_rejection", rejection)
    engine, host = _engine(tmp_path / "e"), _Host()
    try:
        view = _overflow_view(engine, host, fpad=" small")
        out = _overflow_pass(engine, view, 560)
    finally:
        engine.shutdown()
    assert checked and all(tokens <= input_tokens and rough > input_rough
                           for tokens, input_tokens, rough, input_rough in checked), checked
    assert estimate(out) <= estimate(view)
    assert applied == [1, 2, None], applied
    residual = [r.getMessage() for r in caplog.records if RESIDUAL in r.getMessage()]
    assert len(residual) == 1 and "step1=growth, step2=growth" in residual[0], residual
    _disable_e(monkeypatch)
    engine, host = _engine(tmp_path / "base"), _Host()
    try:
        base = _overflow_pass(engine, _overflow_view(engine, host, fpad=" small"), 560)
    finally:
        engine.shutdown()
    assert json.dumps(out, sort_keys=True) == json.dumps(base, sort_keys=True)


@pytest.mark.parametrize("estimate", [100, None], ids=["host_equal", "host_none"])
def test_host_equal_or_none_preserves_p1_e2_base_bytes(tmp_path, monkeypatch, estimate):
    """Break-even and unavailable host estimates retain the exact be9b4b0d P1 e2 output."""
    monkeypatch.setattr(lcm_engine, "_host_estimate", lambda _messages: estimate, raising=False)
    engine, trace = _e2(tmp_path, passes=0)
    engine.shutdown()
    out = trace[0]["out"]
    assert out[1]["content"].startswith("reply to T4") and not trace[0]["inserted"]
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    assert digest == "753a59bec6fddd1e0c9ec4777d5e3b0b808d69edefaee038d38b65639f1acbad", digest
