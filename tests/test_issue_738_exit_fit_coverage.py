"""#738: an automatic exit fit (#668) drops only whole turns a summary node covers, and with the full sweep off an
automatic threshold pass forms a leaf even when the backlog outside the fresh tail is under the leaf chunk.

The H1 fixture shape (64k window, threshold 0.5, a 32-message fresh tail, a 20k leaf chunk, the full sweep off)
driven through the host's preflight order: ``should_compress(observed)`` -> ``compress``, else below the threshold
``should_compress_preflight`` -> ``compress``. In v0.24.9-rc1 the exit fit after a pass that stored no leaf, or a
leaf that covered only part of the backlog, cut the oldest whole turns no summary covers. Summaries are stubbed;
the sweep-on and the non-exit cells are pinned to their v0.24.9-rc1 bytes."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SUMMARY = "Earlier turns.\nExpand for details about: turns"
PAD = "alpha beta gamma delta "
TAG = re.compile(r"\[T(\d+)\]")


@pytest.fixture(autouse=True)
def estimators(monkeypatch):
    """As tests/test_issue_668_exit_fit.py: LCM's character estimate is the host's estimate too."""
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", tokens.count_messages_tokens)


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return SUMMARY, 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, monkeypatch, *, sweep: bool = False, **config) -> LCMEngine:
    monkeypatch.setenv("LCM_THRESHOLD_FULL_SWEEP_ENABLED", "true" if sweep else "false")
    settings = {"fresh_tail_count": 32, "leaf_chunk_tokens": 20_000, "context_threshold": 0.5,
                "database_path": str(tmp_path / "lcm.db"), **config}
    cfg = LCMConfig.from_env()
    for key, value in settings.items():
        setattr(cfg, key, value)
    assert cfg.threshold_full_sweep_enabled is sweep
    engine = LCMEngine(config=cfg)
    engine.on_session_start("S", platform="acp", conversation_id="conv", context_length=64_000)
    return engine


def _user(turn: int, repeat: int = 250) -> dict:
    return {"role": "user", "content": f"[T{turn:02d}] user turn {turn}: {PAD * repeat}end."}


def _reply(turn: int) -> dict:
    return {"role": "assistant", "content": f"reply to T{turn:02d}: noted item {turn}."}


def _turns(count: int) -> list[dict]:
    return [row for turn in range(1, count + 1) for row in (_user(turn), _reply(turn))]


def _drive(engine, turns: int = 60, *, size=lambda turn: 250, overhead: int = 800, outputs: list | None = None,
           fits: list | None = None, gate: bool = False) -> tuple[list[dict], int]:
    """The host loop of the gauntlet's red test; returns the live list and the turn of the last threshold pass.
    ``gate``: the host reads ``_automatic_compression_blocked`` on the type before an automatic compress."""
    messages, last_compress_turn = [], 0
    for turn in range(1, turns + 1):
        messages.append(_user(turn, size(turn)))
        observed = tokens.count_messages_tokens(messages) + overhead
        engine.last_prompt_tokens = observed
        engine._last_survival_fit = None
        if gate and type(engine)._automatic_compression_blocked(engine):
            pass
        elif engine.should_compress(observed):
            messages = engine.compress(messages, current_tokens=observed)
            last_compress_turn = turn
        elif observed < engine.threshold_tokens and engine.should_compress_preflight(messages):
            messages = engine.compress(messages)
        else:
            messages.append(_reply(turn))
            continue
        if outputs is not None:
            outputs.append(messages)
        if fits is not None and engine._last_survival_fit:
            fits.append(engine._last_survival_fit)
        messages.append(_reply(turn))
    return messages, last_compress_turn


def _stranded(engine, live_messages: list[dict], before_turn: int) -> list[int]:
    """Turns before ``before_turn`` neither in the live list nor covered by a summary node's sources."""
    rows = [row for row in engine._store.get_session_messages("S", limit=100_000) if row.get("role") == "user"]
    turn_of = {int(row["store_id"]): int(match.group(1)) for row in rows
               for match in [TAG.match(str(row.get("content") or ""))] if match}
    covered = {turn_of[i] for i in engine._store_complete_node_covered(list(turn_of)) if i in turn_of}
    live = {int(match.group(1)) for message in live_messages if message.get("role") == "user"  # carriers too
            for match in [TAG.search(str(message.get("content") or ""))] if match}
    return sorted(set(range(1, before_turn)) - live - covered)


def _exit_fit_dropped_uncovered(fit) -> bool:
    return bool(fit) and str(fit.get("reason", "")).startswith("exit_fit:") and fit.get("dropped_rows") \
        and fit.get("uncovered_rows") != 0


def _digest(engine, outputs: list) -> str:
    """The returned lists and the stored nodes (depth, sources, summary): the bytes a pin compares."""
    nodes = [(node.depth, list(node.source_ids), node.summary)
             for node in engine._dag.get_session_nodes("S", limit=100_000)]
    payload = {"outputs": outputs, "nodes": nodes, "status": [engine._last_compression_status,
                                                              engine._last_compression_noop_reason]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _nodes(engine) -> list:
    return engine._dag.get_session_nodes("S", limit=100_000)


# -- T1: the red test of the gauntlet (rc1: 0 nodes, 40 turns stranded by 20 exit fits) ---------------------------

@pytest.mark.parametrize("dynamic", [False, True], ids=["static_minimum", "dynamic_minimum"])
def test_t1_sweep_off_threshold_passes_strand_no_turn(tmp_path, monkeypatch, summaries, dynamic):
    engine = _engine(tmp_path, monkeypatch, dynamic_leaf_chunk_enabled=dynamic)
    try:
        fits: list = []
        live, last_compress_turn = _drive(engine, fits=fits)
        assert last_compress_turn > 0
        assert not [fit for fit in fits if _exit_fit_dropped_uncovered(fit)], fits
        stranded = _stranded(engine, live, last_compress_turn)
        assert stranded == [] and _nodes(engine) and summaries, (len(stranded), stranded[:5])
    finally:
        engine.shutdown()


# -- T2: the hold is armed and the held fit-only return happens at the survival ceiling -----------------------------

def test_t2_a_held_pass_makes_no_uncovered_exit_cut(tmp_path, monkeypatch, summaries):
    """The host rejects the first threshold compaction: the #651 hold blocks the gate until the context reaches
    the survival ceiling, where the held fit-only return runs (#618 item 3). Its exit fit drops no uncovered row;
    over the window budget the plain fit (v0.24.8's protection) still applies."""
    engine = _engine(tmp_path, monkeypatch)
    try:
        rejected = []
        original = engine.compress

        def first_rejected(messages, *args, **kwargs):
            if not rejected:
                rejected.append(True)
                engine.record_rejected_compaction()
                return messages
            return original(messages, *args, **kwargs)

        monkeypatch.setattr(engine, "compress", first_rejected)
        fits: list = []
        _drive(engine, gate=True, fits=fits)
        assert rejected and not [fit for fit in fits if _exit_fit_dropped_uncovered(fit)], fits
    finally:
        engine.shutdown()


def test_t2_the_sweep_on_held_pass_at_the_ceiling_is_still_fit_only(tmp_path, monkeypatch, summaries):
    """#618 item 3 with the sweep on (the fleet path) is unchanged."""
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        view = _turns(40)
        engine.ingest(view)
        request = engine._survival_measure(view) + 800
        engine.record_rejected_compaction()
        engine.compress(view, current_tokens=request)
        assert summaries == [] and (engine._last_compression_status, engine._last_compression_noop_reason) == (
            "noop", "held")
    finally:
        engine.shutdown()


# -- T3: the sweep on is byte-identical to v0.24.9-rc1 ---------------------------------------------------------------

_SWEEP_ON_RC1 = {
    "loop": "092beb472ca7f5c13a48f6e5c1b874c458cb44ad2a3f859b746610cd946c13d9",
    "held_ceiling": "f8c376d95c36a2bd84394490fb628e3d740a663c625b8d61244178e6656c44fd",
}


@pytest.mark.parametrize("cell", sorted(_SWEEP_ON_RC1))
def test_t3_sweep_on_output_is_byte_identical_to_rc1(tmp_path, monkeypatch, summaries, cell):
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        outputs: list = []
        if cell == "loop":
            _drive(engine, outputs=outputs)
        else:
            view = _turns(40)
            engine.ingest(view)
            engine.record_rejected_compaction()
            outputs.append(engine.compress(view, current_tokens=engine._survival_measure(view) + 800))
        assert _digest(engine, outputs) == _SWEEP_ON_RC1[cell]
    finally:
        engine.shutdown()


# -- T4: passes that end in no exit fit are unchanged ----------------------------------------------------------------

_UNCHANGED_RC1 = {
    "forced_overflow": "6301e5d4ffdb1ad3e0963a11ed8f90492357bebe9d83523c76cc01f2e2196ad3",
    "manual": "a15e17a6d9d528acafca68e160a804dcf2a2fac340572f22074fb2d2d0345cc9",
    "recovery": "aeb06806911cc146266835229d4b870b0b89957a4c92790590673be88e4a02d3",
    "below_threshold": "a15e17a6d9d528acafca68e160a804dcf2a2fac340572f22074fb2d2d0345cc9",
    "native": "dbd68c16d1a89a454aa7d526ad3c9cdd95c8e5b291d369be8d7206424cccae20",
}


def _small_backlog_view(engine, turns: int = 24) -> tuple[list[dict], int]:
    """Over the 32k threshold, with a raw backlog outside the 32-message tail under the 20k leaf chunk (8 turns,
    about 11.6k tokens)."""
    view = _turns(turns)
    engine.ingest(view)
    return view, engine._survival_measure(view) + 800


@pytest.mark.parametrize("cell", sorted(_UNCHANGED_RC1))
def test_t4_passes_that_are_not_automatic_threshold_passes_are_unchanged(tmp_path, monkeypatch, summaries, cell):
    config = {"max_assembly_tokens": 20_000} if cell == "forced_overflow" else {}
    if cell == "native":
        config["native_recovery"] = True
    engine = _engine(tmp_path, monkeypatch, **config)
    try:
        view, observed = _small_backlog_view(engine)
        if cell == "forced_overflow":
            assert engine._should_force_overflow_recovery(observed_tokens=observed, messages=view)
            result = engine.compress(view, current_tokens=observed)
        elif cell == "manual":
            result = engine.compress(view, current_tokens=observed, force=True)
        elif cell == "recovery":
            result = engine.compress(view, current_tokens=observed, bypass_cooldown=True)
        elif cell == "below_threshold":
            result = engine.compress(view, current_tokens=engine.threshold_tokens - 1)
        else:
            result = engine.compress(view, current_tokens=observed)
        assert _digest(engine, [result]) == _UNCHANGED_RC1[cell]
    finally:
        engine.shutdown()


# -- D1: a capped leaf in the ordinary loop (review of e965c62e, finding 1) ------------------------------------------

def test_d1_a_capped_leaf_then_the_exit_fit_drops_no_uncovered_turn(tmp_path, monkeypatch, summaries):
    """Turns 1-20 large, 21-60 small, turn 61 huge: the first threshold pass (turn 61, ~56.8k observed) stores one
    leaf capped at 40% of the window; rc1 and e965c62e then cut turns 18-19, which no summary covers."""
    engine = _engine(tmp_path, monkeypatch)
    try:
        fits: list = []
        live, last_compress_turn = _drive(
            engine, 61, size=lambda turn: 250 if turn <= 20 else (2 if turn <= 60 else 4500), fits=fits)
        assert last_compress_turn == 61 and _nodes(engine) and summaries
        assert not [fit for fit in fits if _exit_fit_dropped_uncovered(fit)], fits
        assert _stranded(engine, live, 62) == []
    finally:
        engine.shutdown()


# -- D2: every summary route refused ---------------------------------------------------------------------------------

def test_d2_with_every_route_refused_the_exit_fit_drops_no_uncovered_turn(tmp_path, monkeypatch, summaries):
    engine = _engine(tmp_path, monkeypatch)
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(engine._config.summary_model)
    assert not breaker.allows(engine._config.summary_model)
    try:
        view, observed = _small_backlog_view(engine)
        assert 35_000 < observed < engine._survival_ceiling()
        result = engine.compress(view, current_tokens=observed)
        assert summaries == [] and _nodes(engine) == []
        assert not _exit_fit_dropped_uncovered(engine._last_survival_fit), engine._last_survival_fit
        assert _stranded(engine, result, 25) == []
    finally:
        engine.shutdown()


# -- D3: #677 held maintenance when a caller skips the host gate -----------------------------------------------------

def test_d3_held_maintenance_from_a_caller_that_skips_the_gate_drops_no_uncovered_turn(
        tmp_path, monkeypatch, summaries):
    engine = _engine(tmp_path, monkeypatch)
    try:
        view, observed = _small_backlog_view(engine)
        engine.record_rejected_compaction()
        assert engine.threshold_tokens <= observed < engine._survival_ceiling()
        assert engine._no_progress_hold_blocks(observed)
        result = engine.compress(view, current_tokens=observed)
        assert summaries == [] and engine._last_compression_status == "sanitized"
        assert not _exit_fit_dropped_uncovered(engine._last_survival_fit), engine._last_survival_fit
        assert _stranded(engine, result, 25) == []
    finally:
        engine.shutdown()


# -- D4: the native held path is unchanged (review of e965c62e, finding 3) ------------------------------------------

def test_d4_a_held_native_recovery_pass_at_the_ceiling_never_reaches_the_native_compressor(
        tmp_path, monkeypatch, summaries):
    """rc1: native.compress calls=0, status noop, reason held (e965c62e: 1 call, error, no_progress)."""
    calls: list = []

    class ContextCompressor:
        def __init__(self, **kwargs):
            calls.append("init")

        def compress(self, *args, **kwargs):
            calls.append("compress")
            return args[0] if args else []

    module = types.ModuleType("agent.context_compressor")
    module.ContextCompressor = ContextCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", module)
    engine = _engine(tmp_path, monkeypatch, native_recovery=True)
    engine._compression_cancelled_check = lambda: False
    try:
        view = _turns(40)
        engine.ingest(view)
        request = engine._survival_measure(view) + 800
        engine.record_rejected_compaction()
        assert engine._hold_fit_only_applies(request)
        engine.compress(view, current_tokens=request)
        assert calls == []
        assert (engine._last_compression_status, engine._last_compression_noop_reason) == ("noop", "held")
    finally:
        engine.shutdown()


# -- D5: rejected passes at the ceiling make no more summary calls than rc1 (review of e965c62e, finding 2) ----------

_REJECTED_CEILING_CALLS_RC1 = "[0, 0, 0, 0, 0, 0]"  # rc1: #618 item 3, the held pass at the ceiling only fits


def test_d5_rejected_level_3_passes_at_the_ceiling_call_no_more_than_rc1(tmp_path, monkeypatch):
    """24 turns, a fixed host overhead of 19,661 tokens, the hold armed, every summary a non-verbatim level 3
    result (#652 rejects it): six more turns through the gated host loop, summariser invocations per turn."""
    per_turn: list[int] = []
    calls: list = []
    held = 0

    def rejected(**kwargs):
        calls.append(1)
        return "x " * 50, 3

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", rejected)
    engine = _engine(tmp_path, monkeypatch)
    try:
        messages = _turns(24)
        engine.ingest(messages)
        engine.record_rejected_compaction()
        for turn in range(25, 31):
            messages.append(_user(turn))
            observed = tokens.count_messages_tokens(messages) + 19_661
            engine.last_prompt_tokens = observed
            before = len(calls)
            held += engine._hold_fit_only_applies(observed)
            if not type(engine)._automatic_compression_blocked(engine) and engine.should_compress(observed):
                messages = engine.compress(messages, current_tokens=observed)
            per_turn.append(len(calls) - before)
            messages.append(_reply(turn))
        rc1 = json.loads(_REJECTED_CEILING_CALLS_RC1)
        assert held and len(per_turn) == len(rc1) == 6
        assert all(now <= then for now, then in zip(per_turn, rc1)), (per_turn, rc1)
    finally:
        engine.shutdown()


# -- D6: with the sweep on the exit fit still cuts fully covered turns (#668 kept) ----------------------------------

@pytest.mark.parametrize("unpinned", [False, True], ids=["mapped_rows", "an_unpinned_row"])
def test_d6_an_exit_fit_still_cuts_turns_a_summary_covers(tmp_path, monkeypatch, summaries, unpinned):
    """The sweep on, 24 stored turns, and a message-sourced leaf covering the oldest 8 that the host list still
    shows: the automatic exit fit cuts those turns (#668 headroom kept). A row the store-id map cannot pin (a stub)
    is no coverage proof: the cut never passes it."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        view, observed = _small_backlog_view(engine)
        mapping = engine._get_store_id_map_for_messages(view)
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=SUMMARY, token_count=20,
                                         source_token_count=11_000, source_type="messages", created_at=1.0,
                                         source_ids=[mapping[id(m)] for m in view[:16]]))
        if unpinned:  # a stubbed copy of a covered stored reply: durable (the cursor covers it), never pinned
            view[3] = {**view[3], "content": "[stubbed: reply stored verbatim]"}
            assert id(view[3]) not in engine._get_store_id_map_for_messages(view)
        assert engine._ingest_cursor == len(view)
        args = engine._survival_fit_args(view, observed, "compacted", False, automatic=True)
        result = engine._survival_fit(view, view, observed, **args)
        fit = engine._last_survival_fit
        if unpinned:  # only turn 1 could leave before the stub, and that cut does not reach the cap: no cut
            assert result is view and fit is None
        else:
            assert fit and fit["reason"] == "exit_fit:compacted" and fit["dropped_rows"] > 0, fit
            assert fit["uncovered_rows"] == 0 and _stranded(engine, result, 25) == []
            assert engine._survival_measure(result) + 800 <= int(engine.threshold_tokens * 0.95)
    finally:
        engine.shutdown()


# -- D7: a merged live row stands for an uncovered stored row (review of 6e1739b3) -----------------------------------

@pytest.mark.parametrize("sweep", [False, True], ids=["sweep_off", "sweep_on"])
def test_d7_a_merged_row_hiding_an_uncovered_stored_row_is_not_cut(tmp_path, monkeypatch, summaries, sweep):
    """Stored: user1, assistant1, assistant2 ("UNCOVERED reply"), user2, assistant3; a message leaf over rows 1-2.
    Held maintenance sanitizes the list, merging assistant1 + assistant2 into one row the map cannot pin; the
    exit fit (threshold 600, cap 570) must not drop it, since it carries row 3, which no summary covers."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, monkeypatch, sweep=sweep, fresh_tail_count=2)
    try:
        body = " alpha beta gamma delta" * 30
        view = [{"role": "user", "content": "[T01] user one" + body},
                {"role": "assistant", "content": "assistant one" + body},
                {"role": "assistant", "content": "UNCOVERED reply" + body},
                {"role": "user", "content": "[T02] user two" + body},
                {"role": "assistant", "content": "assistant three" + body}]
        engine.ingest(view)
        ids = [int(r["store_id"]) for r in engine._store.get_session_messages("S", limit=100)]
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=SUMMARY, token_count=20,
                                         source_token_count=400, source_type="messages", created_at=1.0,
                                         source_ids=ids[:2]))
        engine.threshold_tokens = 600
        observed = engine._survival_measure(view) + 20
        engine.record_rejected_compaction()
        assert engine._no_progress_hold_blocks(observed)
        engine._last_survival_fit = None
        result = engine.compress(view, current_tokens=observed)
        assert engine._last_compression_status == "sanitized"
        assert any("UNCOVERED reply" in str(m.get("content")) for m in result)
        fit = engine._last_survival_fit
        assert not (fit and fit["reason"].startswith("exit_fit:") and fit["dropped_rows"]), fit
    finally:
        engine.shutdown()


# -- D8: an uncovered stored tool reply shown as a stub -------------------------------------------------------------

def test_d8_a_stubbed_uncovered_tool_reply_is_not_cut(tmp_path, monkeypatch, summaries):
    """Turn 1 (user, tool call, tool reply, assistant) is covered except its tool reply, which the live list shows
    as a stub the map cannot pin: the exit fit keeps turn 1."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, monkeypatch, fresh_tail_count=2)
    try:
        body = " alpha beta gamma delta" * 30
        call = {"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        view = [{"role": "user", "content": "[T01] user one" + body},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call-1", "content": "UNCOVERED tool output" + body * 3},
                {"role": "assistant", "content": "assistant one" + body},
                {"role": "user", "content": "[T02] user two" + body},
                {"role": "assistant", "content": "assistant two" + body}]
        engine.ingest(view)
        mapping = engine._get_store_id_map_for_messages(view)
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=SUMMARY, token_count=20,
                                         source_token_count=400, source_type="messages", created_at=1.0,
                                         source_ids=[mapping[id(view[i])] for i in (0, 1, 3)]))
        view[2] = {**view[2], "content": "[tool output stubbed: stored verbatim]"}
        assert id(view[2]) not in engine._get_store_id_map_for_messages(view) and engine._ingest_cursor == len(view)
        engine.threshold_tokens = 600
        observed = engine._survival_measure(view) + 20
        args = engine._survival_fit_args(view, observed, "compacted", False, automatic=True)
        assert args["reason"] == "exit_fit:compacted"
        result = engine._survival_fit(view, view, observed, **args)
        assert result is view and engine._last_survival_fit is None
    finally:
        engine.shutdown()


# -- D9: verified scaffold first, then covered stored turns: the cut passes the scaffold row ------------------------

def test_d9_a_region_of_scaffold_alone_is_skipped_not_final(tmp_path, monkeypatch, summaries):
    """A body that starts with a verified scaffold row (a preserved objective): the scaffold-only region is not cut
    (no stored row), and the next region, which adds covered stored turns, is (``continue``, not ``break``)."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, monkeypatch, sweep=True, fresh_tail_count=2)
    try:
        engine.threshold_tokens = 600
        view = [row for turn in range(1, 5) for row in (_user(turn, 40), _reply(turn))]
        engine.ingest(view)
        mapping = engine._get_store_id_map_for_messages(view)
        node = SummaryNode(session_id="S", depth=0, summary=SUMMARY, token_count=20, source_token_count=900,
                           source_type="messages", created_at=1.0, source_ids=[mapping[id(m)] for m in view[:4]])
        engine._dag.add_node(node)
        scaffold = {"role": "user", "content": "[Current user objective preserved from compacted history]\nship it"}
        assert engine._is_verified_replay_scaffold_message(scaffold)
        listed = [scaffold, *view]
        store_ids = engine._get_store_id_map_for_messages(listed)
        budget = engine._survival_measure(listed[5:]) + 10  # reachable once turns 1-2 leave
        cut = engine._survival_cut(listed, 0, budget, True, "exit_fit:compacted", store_ids, True,
                                   keep_from=len(listed) - 2)
        assert cut is not None
        fitted = cut[0]
        assert all(m is not listed[1] for m in fitted) and any(m is listed[5] for m in fitted)
    finally:
        engine.shutdown()
