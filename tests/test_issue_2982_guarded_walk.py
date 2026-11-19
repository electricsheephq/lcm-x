"""#2982 read-only probes for the return-guard plan (T1-372); no product code. The guard is EMULATED at the test level:
when a pass returns a list larger than its input by the host's estimator, the "host" keeps its input list (the object
the guarded engine would hand back), and the published leaves stay stored.

- R0x: on the grown return of the reproduction shape, can the host's salvage elide one of OUR summary rows? (vendored
  copies of the host's ``_looks_like_compaction_summary`` and ``salvage_grown_transcript`` tool/summary steps,
  hermes-agent 71d34bad10 agent/context_compressor.py:526-591), and is the salvage enough (else the host refuses)?
- K: the guard's return (the input object) is equal to the host's pre-compression snapshot, so the host's
  ``_candidate_rejected`` (conversation_compression.py:3526, equality at :3557) rejects it before every post-step.
- C: the pass AFTER a guarded return, on the next turn: does it adopt the covered rows or summarise them a second time,
  and does the turn-paced hold (#904's ``hidden_only``) block it?
- D: a walk (placeholder rows first, then one full-size turn a turn) under the emulated guard: per-pass sizes by the
  host's estimator, and the first pass that returns a smaller list.
Reports print as REPORT lines; the asserts only check premises (these are probes, not bars).
"""

from __future__ import annotations

import copy

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

from tests.test_issue_2982_pass_never_grows import CLEARED, QUIET, _summary, _turn, host_estimate

pytestmark = pytest.mark.issue(2982)

# Vendored from hermes-agent 71d34bad10 agent/context_compressor.py (read-only reference).
_SUMMARY_END_MARKER = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
_MERGED_PRIOR_CONTEXT_HEADER = "[PRIOR CONTEXT — for reference only; not a new message]"
_SALVAGE_SUMMARY_MAX_CHARS, _SALVAGE_KEEP_RECENT_TOOLS, _PRUNE_MIN_CHARS = 8_000, 2, 200
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"
COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"


def _looks_like_compaction_summary(msg, content) -> bool:  # host :526-539
    role = msg.get("role")
    if (not content.rstrip().endswith(_SUMMARY_END_MARKER) or content.startswith(_MERGED_PRIOR_CONTEXT_HEADER)
            or role == "tool" or (role in ("user", "assistant") and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY))):
        return False
    head = content[:280]
    return bool(msg.get(COMPRESSED_SUMMARY_METADATA_KEY)) or "CONTEXT COMPACTION" in head or "Conversation Summary" in head


def _salvage(original, candidate):
    """The tool-placeholder and summary-elision steps of salvage_grown_transcript (no reasoning, no todo snapshot in
    these lists): the salvaged list if strictly smaller than the original, else None (the host refuses)."""
    budget = host_estimate(original)
    out = [dict(m) for m in candidate]
    tools = [i for i, m in enumerate(out) if m.get("role") == "tool"]
    keep = set(tools[-_SALVAGE_KEEP_RECENT_TOOLS:])
    elided = 0
    for index, msg in enumerate(out):
        if msg.get("role") == "tool" and index not in keep and isinstance(msg.get("content"), str) \
                and len(msg["content"]) > _PRUNE_MIN_CHARS:
            msg["content"] = _PRUNED_TOOL_PLACEHOLDER
        content = msg.get("content")
        if isinstance(content, str) and len(content) > _SALVAGE_SUMMARY_MAX_CHARS and _looks_like_compaction_summary(msg, content):
            elided += 1
    return (out if host_estimate(out) < budget else None), elided


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(escalation, "_invoke_summary_llm",
                        lambda prompt, max_tokens, model="", timeout=None, reasoning_effort="": _summary(max_tokens))
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=4, **QUIET),
                         hermes_home=str(tmp_path))
    instance.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
    yield instance
    instance.shutdown()


def _shape(turns=40, old=32, hidden=2, output_tokens=4_000):
    """The reproduction's two-in-three-hidden shape: (stored rows, host view)."""
    def is_hidden(i):
        return i < old and i % (hidden + 1) != hidden

    stored = [{"role": "system", "content": "system prompt"}] + [r for i in range(turns) for r in _turn(i, 15, output_tokens)]
    shown = [r for i in range(turns) if not is_hidden(i) for r in _turn(i, 15, output_tokens)]
    tools = [k for k, r in enumerate(shown) if r["role"] == "tool"]
    for k in tools[:-2]:
        shown[k] = {**shown[k], "content": CLEARED}
    return stored, [stored[0]] + shown + [{"role": "user", "content": "latest ask"}]


def _nodes(engine):
    return engine._dag.get_session_nodes("S")


def _twice(engine):
    """Store ids covered by more than one depth-0 node (summarised twice)."""
    seen, twice = set(), set()
    for node in _nodes(engine):
        if node.depth == 0:
            for sid in node.source_ids:
                (twice if sid in seen else seen).add(sid)
    return sorted(twice)


def _stored(engine):
    return len(engine._store.get_session_messages("S", limit=100_000))


def test_r0x_and_k_grown_return_salvage_and_equality(engine):
    stored, view = _shape()
    engine.ingest(stored)
    snapshot = copy.deepcopy(view)  # the host's messages_before_compression
    result = engine.compress(view, current_tokens=engine.threshold_tokens + 20_000)
    salvaged, elided = _salvage(view, result)
    summary_rows = [m for m in result if m.get("role") != "tool" and isinstance(m.get("content"), str)
                    and len(m["content"]) > _SALVAGE_SUMMARY_MAX_CHARS]
    report = {"in_host": host_estimate(view), "out_host": host_estimate(result),
              "rows_out": [(m.get("role"), len(str(m.get("content") or ""))) for m in result],
              "summary_rows_over_8000_chars": len(summary_rows),
              "summary_rows_matching_host_summary_test": sum(_looks_like_compaction_summary(m, m["content"]) for m in summary_rows),
              "rows_with_end_marker": sum(str(m.get("content") or "").rstrip().endswith(_SUMMARY_END_MARKER) for m in result),
              "rows_with_compressed_summary_key": sum(bool(m.get(COMPRESSED_SUMMARY_METADATA_KEY)) for m in result),
              "salvage": None if salvaged is None else host_estimate(salvaged), "salvage_elided_summaries": elided,
              "host_outcome": "refused" if salvaged is None else "salvaged",
              "k_input_object_equals_snapshot": view == snapshot, "k_grown_result_equals_snapshot": result == snapshot}
    print("REPORT R0x+K", report)
    assert report["out_host"] > report["in_host"], report  # premise: this is the growing pass
    assert report["k_input_object_equals_snapshot"], report


def test_c_pass_after_a_guarded_return(engine):
    stored, view = _shape()
    engine.ingest(stored)
    first = engine.compress(view, current_tokens=engine.threshold_tokens + 20_000)
    nodes1 = len(_nodes(engine))
    grew = host_estimate(first) > host_estimate(view)
    # Emulated guard: the host keeps `view`; the guarded engine arms #904's turn-paced hold.
    engine._start_no_progress_hold("hidden_only")
    pressure = engine.threshold_tokens + 20_000
    held = {"block_reason": engine._compression_block_reason(), "should_compress": engine.should_compress(pressure)}
    engine.note_turn_complete()  # the host's post_llm_call at the end of the turn
    released = {"block_reason": engine._compression_block_reason(), "should_compress": engine.should_compress(pressure)}
    nxt = view + [{"role": "assistant", "content": "done with that"}, {"role": "user", "content": "next ask"}]
    engine.ingest(nxt)
    stored_before2 = _stored(engine)
    second = engine.compress(nxt, current_tokens=pressure)
    nodes2 = _nodes(engine)
    report = {"pass1": {"in_host": host_estimate(view), "out_host": host_estimate(first), "grew": grew, "nodes": nodes1},
              "hold_armed": held, "after_turn_end": released,
              "pass2": {"in_host": host_estimate(nxt), "out_host": host_estimate(second), "rows_in": len(nxt),
                        "rows_out": len(second), "nodes_after": len(nodes2), "new_nodes": len(nodes2) - nodes1,
                        "store_ids_summarised_twice": _twice(engine), "stored_before": stored_before2,
                        "stored_after": _stored(engine), "status": engine._last_compression_status,
                        "noop_reason": engine._last_compression_noop_reason,
                        "progress_reported": getattr(engine, "_last_compression_made_progress", "n/a")}}
    print("REPORT C", report)
    assert grew, report


@pytest.mark.parametrize("arm_hold", [True, False], ids=["hold-armed", "no-hold"])
def test_d_walk_under_an_emulated_guard(engine, arm_hold):
    stored, view = _shape()
    engine.ingest(stored)
    host, steps, first_smaller = view, [], None
    for step in range(10):
        before = host
        result = engine.compress(before, current_tokens=engine.threshold_tokens + 20_000)
        grew = host_estimate(result) > host_estimate(before)
        steps.append({"pass": step + 1, "in_host": host_estimate(before), "out_host": host_estimate(result),
                      "rows_in": len(before), "rows_out": len(result), "guarded": grew,
                      "nodes": len(_nodes(engine)), "twice": len(_twice(engine)),
                      "status": engine._last_compression_status, "noop": engine._last_compression_noop_reason})
        if grew:
            if arm_hold:
                engine._start_no_progress_hold("hidden_only")
        else:
            host = result
            if first_smaller is None and host_estimate(result) < host_estimate(before):
                first_smaller = step + 1
        engine.note_turn_complete()
        new = _turn(100 + step, 15, 4_000) + [{"role": "assistant", "content": f"done {step}"}]
        host = host + new
        engine.ingest(host)
    report = {"arm_hold": arm_hold, "first_pass_smaller": first_smaller, "steps": steps}
    print("REPORT D", report)
    assert steps[0]["guarded"], report
