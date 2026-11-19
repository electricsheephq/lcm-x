"""#2982 return guard: an automatic pass never returns a list larger than its input by the host's estimator
(agent.model_metadata.estimate_messages_tokens_rough), after the host's post-steps. On a hit the host gets its own
list object back (no list progress), the published leaves stay stored, and one turn-end hold (``hidden_only``) is armed.
Forced and recovery passes are unchanged. The walks print per-pass sizes by the host's estimator and N, the passes
until the first smaller return.

Shapes come from tests/test_issue_2982_pass_never_grows.py (placeholder-heavy, hidden-heavy) and are synthetic.
"""

from __future__ import annotations

import copy

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.compaction import CompactionMixin
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

from tests.test_issue_2982_guarded_walk import (
    COMPRESSED_SUMMARY_METADATA_KEY, _SUMMARY_END_MARKER, _looks_like_compaction_summary, _salvage,
)
from tests.test_issue_2982_pass_never_grows import CLEARED, QUIET, _summary, _turn, _words, host_estimate

pytestmark = pytest.mark.issue(2982)

CALLS: list[int] = []


@pytest.fixture
def make(tmp_path, monkeypatch):
    def summarise(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        CALLS.append(max_tokens)
        return _summary(max_tokens)

    CALLS.clear()
    monkeypatch.setattr(escalation, "_invoke_summary_llm", summarise)
    made = []

    def build(**overrides):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"lcm{len(made)}.db"), fresh_tail_count=4,
                                            **QUIET, **overrides), hermes_home=str(tmp_path))
        engine.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
        made.append(engine)
        return engine

    yield build
    for engine in made:
        engine.shutdown()


def _shape(*, turns=40, old=32, hidden=2, output_tokens=4_000, thinking=0):
    """Two in three of the oldest ``old`` turns hidden; every tool output but the newest two a placeholder in the list.
    ``thinking``: stored thinking (``reasoning_content``) of that many tokens on every assistant row."""
    def turn(i):
        rows = _turn(i, 15, output_tokens)
        if thinking:
            rows[1] = {**rows[1], "reasoning_content": _words(thinking, f"think{i}z")}
        return rows

    def is_hidden(i):
        return hidden and i < old and i % (hidden + 1) != hidden

    stored = [{"role": "system", "content": "system prompt"}] + [r for i in range(turns) for r in turn(i)]
    shown = [r for i in range(turns) if not is_hidden(i) for r in turn(i)]
    tools = [k for k, r in enumerate(shown) if r["role"] == "tool"]
    for k in tools[:-2]:
        shown[k] = {**shown[k], "content": CLEARED}
    return stored, [stored[0]] + shown + [{"role": "user", "content": "latest ask"}]


def _nodes(engine):
    return engine._dag.get_session_nodes("S")


def _twice(engine):
    seen, twice = set(), set()
    for node in _nodes(engine):
        if node.depth == 0:
            for sid in node.source_ids:
                (twice if sid in seen else seen).add(sid)
    return sorted(twice)


def _stored(engine):
    return [(r["role"], r.get("content")) for r in engine._store.get_session_messages("S", limit=100_000)]


def _pressure(engine):
    return engine.threshold_tokens + 20_000


def _summary_rows_safe(rows):
    return not any(isinstance(m.get("content"), str) and _looks_like_compaction_summary(m, m["content"]) for m in rows)


@pytest.mark.parametrize("thinking", [0, 1_500], ids=["no-thinking", "stored-thinking"])
def test_r0_and_k_grown_pass_is_held_back(make, thinking):
    """R0: the pass that grew the list now returns the host's own list; K: the host's no-progress test (equality with
    its snapshot, before every post-step) rejects it, so no salvage and no strike follow. Stored thinking is charged by
    the host's estimator only; the guard decides by the host's estimator."""
    engine = make()
    stored, view = _shape(thinking=thinking)
    engine.ingest(stored)
    snapshot = copy.deepcopy(view)
    before_rows = len(_stored(engine))
    result = engine.compress(view, current_tokens=_pressure(engine))
    status = engine.get_status()
    report = {"in_host": host_estimate(view), "out_host": host_estimate(result), "in_plugin": count_messages_tokens(view),
              "nodes": len(_nodes(engine)), "calls": len(CALLS), "held_back": status["held_back_returns"],
              "hold": engine._compression_block_reason(), "stored_delta": len(_stored(engine)) - before_rows}
    print("REPORT R0/K", thinking, report)
    assert host_estimate(result) <= host_estimate(view), report
    assert _summary_rows_safe(result), report
    if status["held_back_returns"]:  # the grown pass: K
        assert result is view and result == snapshot, report  # _candidate_rejected: no progress, nothing else runs
        assert getattr(engine, "_last_compression_made_progress", False) is False, report  # the line has no flag
        assert report["hold"] == "cooldown:lcm_hidden_only", report  # one turn-end reason, never no_progress
        assert status["held_back_returns"]["passes"] == 1 and status["held_back_returns"]["deficit_tokens"] > 0
        assert report["nodes"] >= 1 and report["calls"] == report["nodes"], report  # leaves kept; one call per leaf
    assert report["stored_delta"] == 1, report  # only the new user row
    assert not _twice(engine), report


def test_unguarded_grown_return_is_refused_by_the_host_and_never_elides_our_summary(make):
    """Forced passes keep today's path: the grown list goes to the host, whose salvage cannot shrink it (refused) and
    never elides an LCM summary row (no end marker, no _compressed_summary key)."""
    engine = make()
    stored, view = _shape()
    engine.ingest(stored)
    result = engine.compress(view, current_tokens=_pressure(engine), force=True)
    salvaged, elided = _salvage(view, result)
    print("REPORT forced", host_estimate(view), host_estimate(result), salvaged is None, elided)
    assert host_estimate(result) > host_estimate(view)  # forced: unchanged, the guard is silent
    assert engine.get_status()["held_back_returns"] is None
    assert elided == 0 and _summary_rows_safe(result)
    assert not any(str(m.get("content") or "").rstrip().endswith(_SUMMARY_END_MARKER)
                   or m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in result)


def test_recovery_pass_is_unchanged(make):
    engine = make()
    stored, view = _shape()
    engine.ingest(stored)
    result = engine.compress(view, current_tokens=_pressure(engine), bypass_cooldown=True)
    assert engine.get_status()["held_back_returns"] is None
    assert result is not view


def test_shrinking_pass_is_not_held_back(make):
    engine = make()
    stored, view = _shape(hidden=0)
    engine.ingest(stored)
    result = engine.compress(view, current_tokens=_pressure(engine))
    assert host_estimate(result) < host_estimate(view)
    assert engine.get_status()["held_back_returns"] is None
    assert engine._compression_block_reason() is None


def test_exception_path_never_grows(make, monkeypatch):
    engine = make()
    stored, view = _shape()
    engine.ingest(stored)

    def boom(self, *args, **kwargs):
        raise RuntimeError("synthetic failure after the leaves")

    monkeypatch.setattr(CompactionMixin, "_finish_with_leaf", boom)
    try:
        result = engine.compress(view, current_tokens=_pressure(engine))
    except RuntimeError:
        result = None  # re-raised to the host: no list returned
    print("REPORT exception-path", None if result is None else (host_estimate(view), host_estimate(result)))
    assert result is None or host_estimate(result) <= host_estimate(view)


def test_next_turn_pass_runs_and_adopts_without_summarising_twice(make):
    """The armed reason is hidden_only, released at turn end; the next turn's pass runs, publishes only the new rows'
    leaf (one summariser call per new leaf) and covers no store id twice."""
    engine = make()
    stored, view = _shape()
    engine.ingest(stored)
    first = engine.compress(view, current_tokens=_pressure(engine))
    nodes1, calls1 = len(_nodes(engine)), len(CALLS)
    assert first is view and engine._compression_block_reason() == "cooldown:lcm_hidden_only"
    assert not engine.should_compress(_pressure(engine))
    engine.note_turn_complete()
    assert engine._compression_block_reason() is None and engine.should_compress(_pressure(engine))
    nxt = first + _turn(100, 15, 4_000) + [{"role": "assistant", "content": "done"}]
    engine.ingest(nxt)
    second = engine.compress(nxt, current_tokens=_pressure(engine))
    report = {"pass2": [host_estimate(nxt), host_estimate(second)], "new_nodes": len(_nodes(engine)) - nodes1,
              "new_calls": len(CALLS) - calls1, "status": engine._last_compression_status,
              "held_back": engine.get_status()["held_back_returns"], "twice": _twice(engine)}
    print("REPORT next-turn", report)
    assert report["status"] == "compacted", report
    assert report["new_calls"] == report["new_nodes"], report
    assert not report["twice"], report
    assert host_estimate(second) <= host_estimate(nxt), report


def test_stall_fully_covered_chunk_same_list_again(make):
    """A second pass on the SAME list after a held-back pass (every old row already covered). Reports what it arms;
    it must never grow the list, never summarise a covered row again, and never arm no_progress for the guard."""
    engine = make()
    stored, view = _shape()
    engine.ingest(stored)
    engine.compress(view, current_tokens=_pressure(engine))
    engine.note_turn_complete()
    nodes1, calls1 = len(_nodes(engine)), len(CALLS)
    again = engine.compress(view, current_tokens=_pressure(engine))
    report = {"out_le_in": host_estimate(again) <= host_estimate(view), "new_nodes": len(_nodes(engine)) - nodes1,
              "new_calls": len(CALLS) - calls1, "status": engine._last_compression_status,
              "noop": engine._last_compression_noop_reason, "hold": engine._compression_block_reason(),
              "held_back": engine.get_status()["held_back_returns"], "adoption": engine._compress_adoption}
    print("REPORT stall-covered", report)
    assert report["out_le_in"] and not _twice(engine) and report["new_calls"] == report["new_nodes"], report


def _walk(engine, view, new_turn, passes):
    host, steps, first_smaller = view, [], None
    for step in range(passes):
        before_hb = (engine.get_status()["held_back_returns"] or {}).get("passes", 0)
        calls = len(CALLS)
        result = engine.compress(host, current_tokens=_pressure(engine))
        held = result is host and (engine.get_status()["held_back_returns"] or {}).get("passes", 0) > before_hb
        steps.append([step + 1, host_estimate(host), host_estimate(result), "held" if held else "",
                      len(CALLS) - calls, len(_twice(engine))])
        if first_smaller is None and host_estimate(result) < host_estimate(host):
            first_smaller = step + 1
        host = result
        engine.note_turn_complete()
        host = host + new_turn(step)
        engine.ingest(host)
    return first_smaller, steps


@pytest.mark.parametrize(("label", "condense", "output_tokens"), [
    ("full-new-turns", False, 4_000), ("full-new-turns-condensing", True, 4_000), ("small-new-turns", False, 150),
    ("small-new-turns-condensing", True, 150)])
def test_walk(make, label, condense, output_tokens):
    """N = the passes until the first smaller return. small-new-turns is the deciding walk: new turns of a short ask,
    a tool call, a 150-token output and a short reply. condensing: a 3,000-token summary prefix target."""
    engine = make(**({"summary_prefix_target_tokens": 3_000} if condense else {}))
    stored, view = _shape()
    engine.ingest(stored)
    first_smaller, steps = _walk(
        engine, view, lambda step: _turn(200 + step, 15, output_tokens) + [{"role": "assistant", "content": f"done {step}"}],
        30 if output_tokens < 1_000 else 10)
    held = sum(1 for s in steps if s[3])
    print("REPORT walk", label, {"N": first_smaller, "held_back_passes": held,
                                 "status": engine.get_status()["held_back_returns"], "steps": steps})
    assert not any(s[5] for s in steps), steps  # no store id covered twice
    assert all(s[2] <= s[1] for s in steps), steps  # never larger by the host's estimator


def test_stored_condensation_that_frees_nothing_keeps_the_hold_and_the_streak(make, monkeypatch):
    """Leaf summaries of about 120 tokens and a 200-token prefix target: after the leaves the sweep condenses sources
    under the level 3 bound, which are stored whole (no call) and free nothing ('condensation_no_progress'). The sweep
    clears only a no_progress hold, and the guard arms its turn-end hold after the sweep, so the armed reason stays; a
    return that frees nothing never ends the held-back streak (strictly smaller only)."""
    monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda prompt, max_tokens, model="", timeout=None,
                        reasoning_effort="": (CALLS.append(max_tokens), "Earlier work.\n" + _words(120, "fact"))[1])
    engine = make(summary_prefix_target_tokens=200)
    stored, view = _shape()
    engine.ingest(stored)
    result = engine.compress(view, current_tokens=_pressure(engine))
    sweep = engine.get_status()["threshold_full_sweep"]
    report = {"in": host_estimate(view), "out": host_estimate(result), "stop": sweep.get("stop_reason"),
              "condensations": sweep.get("condensation_passes"), "hold": engine._compression_block_reason(),
              "held_back": engine.get_status()["held_back_returns"]}
    print("REPORT no-shrink-condensation", report)
    assert host_estimate(result) <= host_estimate(view), report
    if report["held_back"]:
        assert report["hold"] == "cooldown:lcm_hidden_only" and report["held_back"]["open"], report
