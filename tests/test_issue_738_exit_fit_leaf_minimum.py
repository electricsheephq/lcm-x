"""#738: with the full sweep off, no automatic threshold fit strands turns while the leaf-chunk minimum blocks a leaf.

The H1 fixture shape (64k window, threshold 0.5, a 32-message fresh tail, a 20k leaf chunk) driven through the
host's preflight order: ``should_compress(observed)`` -> ``compress``, else below the threshold
``should_compress_preflight`` -> ``compress``. Invariant: every earlier turn is in the live list or covered by a
summary node. Summaries are stubbed; the sweep-on and below-threshold cells are pinned to their v0.24.9-rc1 bytes."""

from __future__ import annotations

import hashlib
import json
import re
import time

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SUMMARY = "Earlier turns.\nExpand for details about: turns"
TURN_PAD = "alpha beta gamma delta " * 250
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


def _user(turn: int) -> dict:
    return {"role": "user", "content": f"[T{turn:02d}] user turn {turn}: {TURN_PAD}end."}


def _reply(turn: int) -> dict:
    return {"role": "assistant", "content": f"reply to T{turn:02d}: noted item {turn}."}


def _drive(engine, turns: int = 60, outputs: list | None = None, *, reject_first: bool = False,
           held: list | None = None) -> tuple[list[dict], int]:
    """The host loop of the gauntlet's red test; returns the live list and the turn of the last threshold pass.
    ``reject_first``: the host rejects the first threshold compaction (the #651 hold starts) and reads its gate
    (``_automatic_compression_blocked`` on the type) before every automatic compress, as Hermes does."""
    messages, last_compress_turn = [], 0
    for turn in range(1, turns + 1):
        messages.append(_user(turn))
        observed = tokens.count_messages_tokens(messages) + 800
        engine.last_prompt_tokens = observed
        if reject_first and engine.should_compress(observed) and last_compress_turn == 0:
            engine.record_rejected_compaction()
            last_compress_turn = turn
        elif reject_first and type(engine)._automatic_compression_blocked(engine):
            pass
        elif engine.should_compress(observed):
            if held is not None and engine._hold_fit_only_applies(observed):
                held.append(turn)
            messages = engine.compress(messages, current_tokens=observed)
            last_compress_turn = turn
            if outputs is not None:
                outputs.append(messages)
        elif observed < engine.threshold_tokens and engine.should_compress_preflight(messages):
            messages = engine.compress(messages)
            if outputs is not None:
                outputs.append(messages)
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


def _digest(engine, outputs: list) -> str:
    """The returned lists and the stored nodes (depth, sources, summary): the bytes a pin compares."""
    nodes = [(node.depth, list(node.source_ids), node.summary)
             for node in engine._dag.get_session_nodes("S", limit=100_000)]
    payload = {"outputs": outputs, "nodes": nodes, "status": [engine._last_compression_status,
                                                              engine._last_compression_noop_reason]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# -- T1: the red test of the gauntlet (rc1: 0 nodes, 40 turns stranded) ----------------------------------------

@pytest.mark.parametrize("dynamic", [False, True], ids=["static_minimum", "dynamic_minimum"])
def test_t1_sweep_off_threshold_passes_strand_no_turn(tmp_path, monkeypatch, summaries, dynamic):
    engine = _engine(tmp_path, monkeypatch, dynamic_leaf_chunk_enabled=dynamic)
    try:
        live, last_compress_turn = _drive(engine)
        stranded = _stranded(engine, live, last_compress_turn)
        nodes = engine._dag.get_session_nodes("S", limit=100_000)
        assert last_compress_turn > 0 and stranded == [], (len(nodes), len(stranded), stranded[:5])
        assert nodes and summaries
    finally:
        engine.shutdown()


# -- T2: the fixA shape: the hold is armed and the held fit-only return happens at the survival ceiling ---------

def _ceiling_view(engine, turns: int = 40) -> tuple[list[dict], int]:
    view = [row for turn in range(1, turns + 1) for row in (_user(turn), _reply(turn))]
    engine.ingest(view)
    request = engine._survival_measure(view) + 800
    assert request >= engine._survival_ceiling()
    return view, request


def test_t2_a_held_pass_at_the_ceiling_with_the_sweep_off_strands_no_turn(tmp_path, monkeypatch, summaries):
    """The host rejects the first threshold compaction: the #651 hold blocks the gate until the context reaches
    the survival ceiling, where rc1 took the held fit-only return and dropped uncovered turns."""
    engine = _engine(tmp_path, monkeypatch)
    try:
        held: list[int] = []
        live, _rejected = _drive(engine, reject_first=True, held=held)
        assert held  # at least one automatic pass met the hold at the survival ceiling
        stranded = _stranded(engine, live, 61)
        assert stranded == [], (len(stranded), stranded[:5])
        assert engine._dag.get_session_nodes("S", limit=100_000) and summaries
    finally:
        engine.shutdown()


def test_t2_the_sweep_on_held_pass_at_the_ceiling_is_still_fit_only(tmp_path, monkeypatch, summaries):
    """#618 item 3 with the sweep on (the fleet path) is unchanged."""
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        view, request = _ceiling_view(engine)
        engine.record_rejected_compaction()
        engine.compress(view, current_tokens=request)
        assert summaries == [] and (engine._last_compression_status, engine._last_compression_noop_reason) == (
            "noop", "held")
    finally:
        engine.shutdown()


# -- T3: invariant 1, the sweep on is byte-identical to v0.24.9-rc1 ----------------------------------------------

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
            view, request = _ceiling_view(engine)
            engine.record_rejected_compaction()
            outputs.append(engine.compress(view, current_tokens=request))
        assert _digest(engine, outputs) == _SWEEP_ON_RC1[cell]
    finally:
        engine.shutdown()


# -- T4: invariant 2, forced overflow, manual, recovery, below-threshold and native passes are unchanged ---------

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
    view = [row for turn in range(1, turns + 1) for row in (_user(turn), _reply(turn))]
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


# -- invariant 3: the #605 foreground budget governs the waived leaf ---------------------------------------------

class _Clock:
    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


class _Provider:
    """``seconds`` per call on the clock, then a summary; ``None`` hangs to its timeout."""

    def __init__(self, clock: _Clock, seconds: float | None):
        self.clock, self.seconds, self.calls = clock, seconds, []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append({"timeout": timeout, "started": self.clock.offset})
        if self.seconds is None or (timeout is not None and self.seconds > timeout):
            self.clock.offset += timeout
            raise TimeoutError("provider timed out")
        self.clock.offset += self.seconds
        return SUMMARY


def _budget_engine(tmp_path, monkeypatch) -> tuple[LCMEngine, list[dict], int]:
    engine = _engine(tmp_path, monkeypatch, l3_truncate_tokens=2)  # every source needs a call (no F2 verbatim)
    view, observed = _small_backlog_view(engine)
    return engine, view, observed


def test_invariant3_a_hanging_route_bounds_the_waived_leaf_and_never_yields_level_3(tmp_path, monkeypatch, clock):
    provider = _Provider(clock, None)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    engine, view, observed = _budget_engine(tmp_path, monkeypatch)
    try:
        started = time.monotonic()
        engine.compress(view, current_tokens=observed)
        hard = engine._foreground_budget_seconds()[1]
        assert provider.calls  # the minimum no longer blocks the leaf: its call is admitted
        assert all(call["started"] + call["timeout"] <= hard + 0.5 for call in provider.calls)
        assert time.monotonic() - started <= hard + 1.0
        assert not any(escalation._L3_TRUNCATION_MARKER in node.summary
                       for node in engine._dag.get_session_nodes("S", limit=100_000))
    finally:
        engine.shutdown()


def test_invariant3_a_time_stop_before_the_waived_leaf_makes_no_call(tmp_path, monkeypatch, clock):
    provider = _Provider(clock, 30.0)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    engine, view, observed = _budget_engine(tmp_path, monkeypatch)
    original = engine._prepare_retained_user_anchor

    def slow(*args, **kwargs):  # 110 s of pre-work: 5 s of usable time left, under the leaf's estimate
        result = original(*args, **kwargs)
        clock.offset += 110.0
        return result

    monkeypatch.setattr(engine, "_prepare_retained_user_anchor", slow)
    try:
        engine.compress(view, current_tokens=observed)
        assert provider.calls == [] and engine._dag.get_session_nodes("S", limit=100_000) == []
        assert engine._last_compression_status != "error"
        assert engine._last_compression_noop_reason.endswith("time budget spent before the first leaf")
    finally:
        engine.shutdown()


# -- invariant 4: a pass that stores the waived leaf is progress; the #651 hold is not armed ----------------------

def test_invariant4_the_waived_leaf_is_progress_and_arms_no_hold(tmp_path, monkeypatch, summaries):
    engine = _engine(tmp_path, monkeypatch)
    try:
        view, observed = _small_backlog_view(engine)
        result = engine.compress(view, current_tokens=observed)
        assert len(engine._dag.get_session_nodes("S", limit=100_000)) == 1 and summaries
        assert engine._last_compression_status == "compacted"
        assert not engine._no_progress_hold_active() and engine._compression_block_reason() is None
        assert _stranded(engine, result, 25) == []
    finally:
        engine.shutdown()
