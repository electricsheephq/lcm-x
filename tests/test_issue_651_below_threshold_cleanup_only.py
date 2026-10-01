"""#651: below the host threshold an automatic compress() is cleanup-only; a finite no-progress hold; the host
breaker hooks.

Engine-level: a host list in; the returned list, stored rows, DAG leaves, the public status and the host-shaped
breaker calls out. Summaries are stubbed and counted. The last cell runs the host's own refusal site and breaker
gate from a reliability host tree when one is configured (skipped otherwise)."""

from __future__ import annotations

import inspect
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

PAD = " alpha beta gamma delta" * 30
HOLD = lcm_engine._SWEEP_BUDGET_HOLD_SECONDS


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """LCM's character estimate and no host estimator, as in the #608 tests (#615)."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 100, "fresh_tail_pressure_yield_enabled": False,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _leaves(engine) -> int:
    return len(engine._dag.get_session_nodes(engine._session_id))


def _stored(engine) -> int:
    return engine._store.get_session_count(engine._session_id)


def _maintenance_engine(tmp_path, threshold: int | None = None, **config):
    """Deferred maintenance under critical budget pressure, below the host threshold: the preflight asks for
    the engine-driven sub-threshold compress() the host logs as 'below N threshold'."""
    engine = _engine(tmp_path, deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5, **config)
    view = _view()
    rough = count_messages_tokens(view)
    engine.threshold_tokens = 100_000 if threshold is None else threshold
    engine.context_length = int(rough * 1.6)  # pressure ~0.62 >= 0.5; the survival ceiling stays above rough
    assert rough < engine.threshold_tokens
    assert engine.should_compress_preflight(view) is True
    return engine, view, rough


def _host_gate(compressor, *, bypass_cooldown: bool = False, force: bool = False) -> bool:
    """The r34.3 / 0.21.2 ``not force and _automatic_compression_gate_blocks`` read: the gate on the TYPE,
    ``ignore_cooldown`` only when the signature accepts it and the attempt bypasses the cooldown; a missing gate
    never blocks, and manual /compress (``force``) never consults it."""
    if force:
        return False
    blocked = getattr(type(compressor), "_automatic_compression_blocked", None)
    if not callable(blocked):
        return False
    accepts = bypass_cooldown and "ignore_cooldown" in inspect.signature(blocked).parameters
    return bool(blocked(compressor, ignore_cooldown=True) if accepts else blocked(compressor))


# -- T1. below the threshold an automatic compress() is cleanup-only ---------------------------------------

def test_t1_below_threshold_automatic_compress_writes_no_leaf_and_still_ingests(tmp_path, summaries):
    engine, view, rough = _maintenance_engine(tmp_path)
    try:
        result = engine.compress(view, current_tokens=rough)
        assert summaries == [] and _leaves(engine) == 0
        assert engine._last_compression_status == "sanitized"
        assert _stored(engine) == len(view)  # ingest ran: every host row is stored
        assert engine._ingest_cursor == len(result) == len(view)  # the cursor indexes the returned list
        assert engine._no_progress_hold is None  # cleanup-only is not a threshold pass
        assert engine._preflight_below_threshold_cleanup_only is False  # the handoff is one-shot
    finally:
        engine.shutdown()


def test_t1_the_handoff_needs_the_preflight_request_and_its_own_tokens_below_threshold(tmp_path, summaries):
    engine, view, rough = _maintenance_engine(tmp_path)
    try:  # the host observed the threshold after all: a normal pass, not cleanup-only
        engine.compress(view, current_tokens=engine.threshold_tokens)
        assert summaries and _leaves(engine) >= 1
    finally:
        engine.shutdown()


# -- T2. /compress, forced overflow and a host recovery attempt still run leaves -----------------------------

@pytest.mark.parametrize("call", ["manual", "forced_overflow", "recovery"])
def test_t2_manual_forced_and_recovery_passes_below_threshold_run_leaves(tmp_path, summaries, call):
    config = {"max_assembly_tokens": 0}
    if call == "forced_overflow":
        config["max_assembly_tokens"] = 3_000  # above the view: preflight is not forced, compress() is
    engine, view, rough = _maintenance_engine(tmp_path, **config)
    try:
        assert engine._preflight_below_threshold_cleanup_only is True
        if call == "manual":
            engine.compress(view, current_tokens=rough, force=True)
        elif call == "forced_overflow":
            assert rough < 3_000 < engine.threshold_tokens
            engine.compress(view, current_tokens=3_001)
        else:
            engine.compress(view, current_tokens=rough, bypass_cooldown=True)
        assert summaries and _leaves(engine) >= 1, engine._last_compression_noop_reason
    finally:
        engine.shutdown()


# -- T3. the no-progress hold -------------------------------------------------------------------------------

def _stuck_engine(tmp_path):
    """One unshortenable newest turn: neither a leaf nor the #668 exit fit can make progress."""
    engine = _engine(tmp_path, leaf_chunk_tokens=100_000)
    view = _view(1)
    rough = count_messages_tokens(view)
    engine.threshold_tokens = 100
    assert engine.should_compress(rough) is True
    return engine, view, rough


def test_t3_a_no_progress_threshold_pass_arms_the_hold(tmp_path, summaries):
    engine, view, rough = _stuck_engine(tmp_path)
    try:
        result = engine.compress(view, current_tokens=rough)
        assert summaries == [] and _leaves(engine) == 0 and len(result) == len(view)
        hold = engine.get_status()["no_progress_hold"]
        assert hold["reason"] == "no_progress" and hold["until"] > 0
        # the next automatic pass is skipped at every automatic decision point
        assert engine.should_compress(rough) is False
        assert engine.should_compress_preflight(view) is False
        assert _host_gate(engine) is True
        assert engine._compression_block_reason() == "cooldown:lcm_no_progress"  # host: transient, never exhausts
        # a host recovery attempt and manual /compress are never blocked
        assert _host_gate(engine, bypass_cooldown=True) is False
        engine.compress(view, current_tokens=rough, force=True)
    finally:
        engine.shutdown()


def test_t3_the_hold_expires_by_time(tmp_path, summaries, monkeypatch):
    engine, view, rough = _stuck_engine(tmp_path)
    try:
        engine.compress(view, current_tokens=rough)
        assert engine.should_compress(rough) is False
        now = lcm_engine.time.monotonic()  # #618: the hold reads the monotonic clock
        monkeypatch.setattr(lcm_engine.time, "monotonic", lambda: now + HOLD + 1)
        assert engine.should_compress(rough) is True
        assert engine.get_status()["no_progress_hold"] is None and engine._no_progress_hold is None
    finally:
        engine.shutdown()


def test_t3_the_hold_survives_a_following_turn_that_stores_new_rows(tmp_path, summaries):
    engine, view, rough = _stuck_engine(tmp_path)
    try:
        engine.compress(view, current_tokens=rough)
        rows = _stored(engine)
        next_turn = [*view, *_turn("T9", 900.0)]
        next_rough = count_messages_tokens(next_turn)
        assert engine.should_compress_preflight(next_turn) is False  # preflight ingests the new turn
        assert _stored(engine) > rows
        assert engine.should_compress(next_rough) is False
        assert _host_gate(engine) is True
        assert engine.get_status()["no_progress_hold"]["reason"] == "no_progress"
    finally:
        engine.shutdown()


def test_t3_a_hidden_only_leaf_is_progress_and_does_not_arm_the_hold(tmp_path, summaries):
    """#581 (b): the view is all fresh tail; the leaf covers owned rows the view does not show. It consumes no
    host row and the returned list grows, yet coverage advanced: no hold."""
    engine = _engine(tmp_path, leaf_chunk_tokens=400, context_threshold=0.001)
    old = [*_turn("H1", 100.0), *_turn("H2", 110.0)]
    tail = _turn("T9", 900.0)
    try:
        engine.ingest([*old, *tail])
        result = engine.compress(list(tail))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert _leaves(engine) == 1 and len(result) >= len(tail)
        assert count_messages_tokens(result) >= count_messages_tokens(tail)
        assert engine._no_progress_hold is None and engine.get_status()["no_progress_hold"] is None
    finally:
        engine.shutdown()


def test_t3_a_stored_leaf_ends_the_hold(tmp_path, summaries):
    engine = _engine(tmp_path)
    view = _view()
    try:
        engine.threshold_tokens = 100
        engine.record_rejected_compaction()
        assert _host_gate(engine) is True
        engine.compress(view, current_tokens=count_messages_tokens(view), force=True)  # manual /compress
        assert _leaves(engine) >= 1 and engine._no_progress_hold is None and _host_gate(engine) is False
    finally:
        engine.shutdown()


@pytest.mark.parametrize("call", ["manual", "forced_overflow", "recovery"])
def test_t3_manual_forced_overflow_and_recovery_are_never_blocked_while_it_holds(tmp_path, summaries, call):
    engine = _engine(tmp_path, max_assembly_tokens=3_000)
    view = _view()
    rough = count_messages_tokens(view)
    try:
        engine.threshold_tokens = 100
        assert rough < 3_000
        engine.record_rejected_compaction()
        assert engine.should_compress(rough) is False and _host_gate(engine) is True  # it holds
        if call == "manual":
            assert _host_gate(engine, force=True) is False
            engine.compress(view, current_tokens=rough, force=True)
        elif call == "forced_overflow":
            assert engine.should_compress(3_001) is True  # the LCM answer
            assert _host_gate(engine) is False  # the gate saw the over-cap observation
            engine.compress(view, current_tokens=3_001)
        else:
            assert _host_gate(engine, bypass_cooldown=True) is False
            engine.compress(view, current_tokens=rough, bypass_cooldown=True)
        assert summaries and _leaves(engine) >= 1, engine._last_compression_noop_reason
    finally:
        engine.shutdown()


# -- T4. the survival ceiling --------------------------------------------------------------------------------

def test_t4_the_hold_does_not_apply_at_the_survival_ceiling(tmp_path, summaries):
    engine, view, rough = _stuck_engine(tmp_path)
    try:
        engine.compress(view, current_tokens=rough)
        assert engine.should_compress(rough) is False
        ceiling = int(engine.context_length * (1 - engine._config.survival_reserve))
        assert engine.should_compress(ceiling) is True  # the host calls compress(), whose survival fit runs
        assert _host_gate(engine) is False  # the gate saw the ceiling observation
        engine.last_prompt_tokens = ceiling - 1
        assert engine.should_compress(ceiling - 1) is False and _host_gate(engine) is True
    finally:
        engine.shutdown()


# -- F1 (PR #655 review). a bypassed (auxiliary) session is never governed by the foreground hold ----------

def _review_trace_engine(tmp_path):
    """The review's trace: window 100k, threshold 35k, reserve 0.15, assembly cap 90k; a foreground no-progress
    pass at 40k arms the hold; an in-process auxiliary (LCM-bypassed) call then runs on this thread."""
    engine = _engine(tmp_path, context_length=100_000, leaf_chunk_tokens=100_000, max_assembly_tokens=90_000)
    view = _view(1)  # #668: an unshortenable newest turn still arms the no-progress hold
    assert engine.threshold_tokens == 35_000
    engine.last_prompt_tokens = 40_000
    assert engine.should_compress(40_000) is True
    engine.compress(view, current_tokens=40_000)
    assert engine.get_status()["no_progress_hold"]["reason"] == "no_progress" and _host_gate(engine) is True
    engine._mark_thread_context_stateless("aux-1")
    engine.update_from_response({"prompt_tokens": 95_000, "completion_tokens": 1, "total_tokens": 95_001})
    assert engine.last_prompt_tokens == 40_000  # auxiliary usage keeps the foreground observation
    assert engine._bypasses_lcm_context_management() and engine.threshold_tokens == 35_000
    return engine, view


def test_f1_auxiliary_forced_overflow_is_not_blocked_by_the_foreground_hold(tmp_path, summaries):
    engine, view = _review_trace_engine(tmp_path)
    try:
        assert engine.should_compress(95_000) is True  # over the 90k cap and the 85k survival ceiling
        assert _host_gate(engine) is False  # the host proceeds to compress()
        big = [{"role": "system", "content": "system prompt"},
               *[row for i in range(40) for row in _turn(f"A{i}", 10.0 * (i + 1))]]
        result = engine.compress(big, current_tokens=95_000)
        assert count_messages_tokens(result) < count_messages_tokens(big)  # the bypass path bounded it
        assert engine.get_status()["no_progress_hold"] is not None  # the foreground hold itself is untouched
    finally:
        engine.shutdown()


def test_f1_a_foreground_call_after_an_auxiliary_call_is_judged_on_its_own_tokens(tmp_path, summaries):
    engine, _view_ = _review_trace_engine(tmp_path)
    try:
        assert engine.should_compress(95_000) is True and _host_gate(engine) is False  # auxiliary
        engine._clear_thread_context_stateless("aux-1")
        assert not engine._bypasses_lcm_context_management()
        assert engine.should_compress(85_000) is True and _host_gate(engine) is False  # at the ceiling
        assert engine.should_compress(40_000) is False and _host_gate(engine) is True  # not the auxiliary 95k
    finally:
        engine.shutdown()


# -- G1 (PR #655). a held pass stays cleanup-only when the host's tokens reach the threshold ------------------

def _held_cleanup_request(tmp_path):
    """The preflight's message estimate is below the threshold; the host's own count (system prompt, tool
    schemas) is above it and below the survival ceiling. The no-progress hold is armed."""
    rough = count_messages_tokens(_view())
    engine, view, rough = _maintenance_engine(tmp_path, threshold=rough + 100)
    ceiling = int(engine.context_length * (1 - engine._config.survival_reserve))
    assert rough < engine.threshold_tokens < rough + 200 < ceiling
    engine.record_rejected_compaction()
    assert engine._preflight_below_threshold_cleanup_only is True
    return engine, view, rough, ceiling


def test_g1_a_held_pass_above_the_threshold_stays_cleanup_only(tmp_path, summaries):
    engine, view, rough, ceiling = _held_cleanup_request(tmp_path)
    try:
        assert engine._no_progress_hold_blocks(rough + 200) is True  # the one predicate the gate also uses
        engine.compress(view, current_tokens=rough + 200)
        assert summaries == [] and _leaves(engine) == 0
        assert engine._last_compression_status == "sanitized"
        assert engine.get_status()["no_progress_hold"]["reason"] == "host_rejected"
    finally:
        engine.shutdown()


def test_g1_at_the_survival_ceiling_a_held_pass_is_not_forced_cleanup_only(tmp_path, summaries):
    """#738: with the sweep off the held pass at the ceiling summarises the raw backlog outside the fresh tail
    instead of returning the fit."""
    engine, view, rough, ceiling = _held_cleanup_request(tmp_path)
    try:
        assert engine._no_progress_hold_blocks(ceiling) is False
        engine.compress(view, current_tokens=ceiling)
        # #618 item 3: not cleanup-only; #738: and with the sweep off not fit-only, so no exit fit strands a turn
        assert summaries and _leaves(engine) >= 1
        assert engine._last_compression_noop_reason != "held"
        assert (engine._last_survival_fit or {}).get("uncovered_rows", 0) == 0
    finally:
        engine.shutdown()


# -- G2 (PR #655). a hold never leaks across a session rebind ------------------------------------------------

def test_g2_a_rebind_to_another_session_is_not_held(tmp_path, summaries):
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = 100
        engine.record_rejected_compaction()
        assert engine.should_compress(1_000) is False and _host_gate(engine) is True
        engine.on_session_start("S2", platform="telegram", context_length=200_000, conversation_id="conv2")
        engine.threshold_tokens = 100
        assert engine.get_status()["no_progress_hold"] is None
        assert engine.should_compress(1_000) is True and _host_gate(engine) is False
    finally:
        engine.shutdown()


# -- T5. the host breaker hooks ------------------------------------------------------------------------------

def test_t5_record_rejected_compaction_arms_the_hold_per_the_host_contract(tmp_path, summaries):
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = 100
        assert "ignore_cooldown" in inspect.signature(LCMEngine._automatic_compression_blocked).parameters
        assert _host_gate(engine) is False and engine._compression_block_reason() is None
        assert engine.record_rejected_compaction() is None  # called without arguments
        assert _host_gate(engine) is True
        assert _host_gate(engine, bypass_cooldown=True) is False
        assert engine._compression_block_reason() == "cooldown:lcm_host_rejected"
        assert engine.get_status()["no_progress_hold"]["reason"] == "host_rejected"
        assert engine.should_compress(1_000) is False
    finally:
        engine.shutdown()


def test_t5_a_pending_below_threshold_cleanup_pass_is_not_blocked(tmp_path, summaries):
    engine, view, rough = _maintenance_engine(tmp_path)
    try:
        engine.record_rejected_compaction()
        assert _host_gate(engine) is False  # the cleanup-only pass the preflight asked for goes through
        engine.compress(view, current_tokens=rough)
        assert summaries == [] and _host_gate(engine) is True
    finally:
        engine.shutdown()


_HOST_CELL = textwrap.dedent("""
    import importlib.util, json, sys, time, types
    plugin_dir, db = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", plugin_dir + "/__init__.py", submodule_search_locations=[plugin_dir])
    sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from agent import conversation_compression as cc

    engine = LCMEngine(config=LCMConfig(database_path=db, fresh_tail_count=2))
    engine.on_session_start("S", platform="cli", context_length=200000, conversation_id="conv")
    engine.threshold_tokens = 100
    agent = types.SimpleNamespace(context_compressor=engine, session_id="S", _cached_system_prompt="sys",
                                  _emit_warning=lambda *a, **k: None)
    out = {"before": cc._automatic_compression_gate_blocks(agent, False)}
    small = [{"role": "user", "content": "hi"}]
    grown = [{"role": "user", "content": "hi " * 400}]
    out["refused"] = cc._salvage_or_refuse_grown_transcript(
        agent, small, grown, system_message="sys", attempt_started_at=time.monotonic(), attempt_snapshot={})[0] is None
    out["hold"] = (engine.get_status()["no_progress_hold"] or {}).get("reason")
    out["blocked"] = cc._automatic_compression_gate_blocks(agent, False)
    out["transient"] = cc.compression_blocked_transiently(agent)
    out["recovery"] = cc._automatic_compression_gate_blocks(agent, True)
    engine.shutdown()
    print(json.dumps(out))
""")


def _reliability_host(name: str):
    try:
        from bench.instruments.reliability import hosts
    except Exception:
        return None
    try:
        return hosts.load(hosts.hosts_file(None), [name])[name]
    except Exception:
        return None


@pytest.mark.parametrize("host_name", ["r34.3-0.21.5", "customer-0.21.2"])
def test_t5_host_tree_cell_refusal_records_a_strike_and_the_gate_blocks(tmp_path, host_name):
    """The host's own refusal site ('would be larger') and breaker gate, run under the host's python."""
    host = _reliability_host(host_name)
    if host is None or not Path(host["python"]).exists():
        pytest.skip(f"reliability host {host_name} not configured")
    plugin_dir = Path(__file__).resolve().parent.parent
    proc = subprocess.run([host["python"], "-c", _HOST_CELL, str(plugin_dir), str(tmp_path / "lcm.db")],
                          cwd=host["src"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"before": False, "refused": True, "hold": "host_rejected", "blocked": True,
                   "transient": True, "recovery": False}, out
