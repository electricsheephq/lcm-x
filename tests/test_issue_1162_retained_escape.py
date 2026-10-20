"""#1162: a retained occurrence that never resolves stopped every pass with "leaf would end before an unresolved
retained occurrence" (leaves=0), and nothing named the row or warned. The stop now names the blocking row (store_id,
role, kind) on the stop line, the no-op line, the sweep record and the status; a sanitizing pass keeps that reason;
doctor warns after consecutive over-threshold passes without a leaf.

Engine-level, on a scratch store with a stub summariser (the repro of the #1162 root-cause comment)."""

from __future__ import annotations

import json
import re
import logging

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools
from hermes_lcm.engine import LCMEngine
from tests.harness import make_engine, pin_char_counter

PAD = " alpha beta gamma delta" * 30
RETAINED = "leaf would end before an unresolved retained occurrence"


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[tuple[int, str]] = []

    def emit(self, record):
        self.lines.append((record.levelno, record.getMessage()))


@pytest.fixture
def lines():
    handler, logger = _Lines(), logging.getLogger("hermes_lcm")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(level)


def _engine(tmp_path, sweep: bool, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "dynamic_leaf_chunk_max": 400, "threshold_full_sweep_enabled": sweep,
                "temporal_rollups_enabled": False, **config}
    engine = make_engine(tmp_path, **settings)
    # The host-identity fixture keeps state.db beside lcm.db.
    engine._hermes_home = str(tmp_path)
    engine._store._hermes_home = str(tmp_path)
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, tool: bool = False) -> list[dict]:
    rows = [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts}]
    if tool:
        call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        rows += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag}{PAD}"}]
    return rows + [{"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages("S", limit=100_000)


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id or 0)


def _passes(engine, view, n, lines, label="") -> tuple[list, list[dict]]:
    """``n`` threshold passes, each a foreground turn; the 600 s no-progress hold only paces passes (a real host
    retries after it), so it is cleared between them."""
    trace = []
    for _ in range(n):
        start = len(lines)
        view = engine.compress(view, current_tokens=10_000)
        engine.note_turn_complete()
        engine._no_progress_hold = None
        new = lines[start:]
        stop = next((msg for _lvl, msg in new if msg.startswith("LCM compaction stop:")), "")
        trace.append({
            "label": label, "status": engine._last_compression_status,
            "reason": engine._last_compression_noop_reason, "frontier": _frontier(engine),
            "leaves": int(stop.split("leaves=", 1)[1].split(" ", 1)[0]) if "leaves=" in stop else None,
            "stop": stop.split(" elapsed=", 1)[0], "line": stop, "rows": len(view),
            "warnings": [msg for lvl, msg in new if lvl >= logging.WARNING],
        })
    return view, trace


def _show(trace) -> str:
    return "\n".join(f"{t['label']} status={t['status']} frontier={t['frontier']} rows={t['rows']} | {t['stop']} "
                     f"| noop={t['reason']!r}" for t in trace)


def _reorder(tmp_path, sweep, *, bound=True):
    """#581 test (c): R moves to the END; #1403 requires proof that it is the same occurrence."""
    if bound:
        from tests.test_host_uid_shadow import _state_db

        _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, sweep, fresh_tail_count=1, leaf_chunk_tokens=1)
    r = {"role": "user", "content": "[R] retained occurrence" + PAD, "message_uid": "retained-r"}
    stored = [*_turn("T1", 100.0), r, *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    engine.ingest(stored)
    view = [m for m in stored if m is not r] + [dict(r)]
    engine._ingest_cursor = len(view)
    return engine, view


def _restore(tmp_path, sweep):
    """The host replaces an older tool output with new, non-salvage content: the ingest stores it and the
    unstamped rows after it again at the END, and the oldest host row binds to a newer store id."""
    engine = _engine(tmp_path, sweep, fresh_tail_count=4, leaf_chunk_tokens=400)
    view = [row for i in range(1, 13) for row in _turn(f"T{i}", 100.0 * i, tool=True)]
    view = engine.compress(view, current_tokens=10_000)
    engine.note_turn_complete()
    engine._no_progress_hold = None
    tools = [m for m in view if m.get("role") == "tool"]
    tools[0]["content"] = "[Host replaced this tool output with genuinely new content]"
    return engine, view + _turn("T13", 1300.0, tool=True)


def _scenario(name, tmp_path, sweep):
    return (_reorder if name == "reorder" else _restore)(tmp_path, sweep)


def _r_id(engine) -> int:
    return next(int(x["store_id"]) for x in _rows(engine) if x["content"].startswith("[R] retained"))


SCENARIOS = pytest.mark.parametrize("name", ["reorder", "restore"])
SWEEP = pytest.mark.parametrize("sweep", [True, False], ids=["sweep_on", "sweep_off"])


def test_unbound_reorder_keeps_one_copy_across_compactions(tmp_path, summaries, lines):
    """#1403: without occurrence proof, keep a copy once, not once per later compaction/audit."""
    engine, view = _reorder(tmp_path, False, bound=False)
    try:
        before = len(_rows(engine))
        engine.ingest(view)
        assert len(_rows(engine)) == before + 1
        _passes(engine, view, 3, lines)
        assert len(_rows(engine)) == before + 1
        assert sum(row["content"].startswith("[R] retained") for row in _rows(engine)) == 2
    finally:
        engine.shutdown()


# -- observability ---------------------------------------------------------------------------------------------

@SCENARIOS
@SWEEP
def test_the_first_retained_stop_names_the_blocking_row(tmp_path, summaries, lines, name, sweep):
    """The stop line, the no-op line, the sweep record and the status all carry the blocking row (store_id, role,
    kind); with the sweep off a sanitizing pass no longer erases the reason."""
    engine, view = _scenario(name, tmp_path, sweep)
    try:
        trace, start = [], 0
        while len(trace) < 3 and not (trace and trace[-1]["reason"].startswith(RETAINED)):
            start = len(lines)
            view, step = _passes(engine, view, 1, lines, name)  # leaves may still form before the stop
            trace += step
        named = trace[-1]["reason"]
        assert named.startswith(RETAINED + " (store_id=") and named.endswith(" kind=shown)"), _show(trace)
        if name == "reorder":
            assert named == f"{RETAINED} (store_id={_r_id(engine)} role=user kind=shown)"
        store_id, role = named.split("(store_id=", 1)[1].split(" kind=", 1)[0].split(" role=")
        line = trace[-1]["line"]  # reason= exactly as on base (sweep on: the #581 text); the row is its own field
        assert trace[-1]["leaves"] == 0 and line.endswith(f" blocked=store_id:{store_id},role:{role},kind:shown")
        assert re.search(f"reason={RETAINED} leaves=0 progress=" if sweep else "reason=(noop|sanitized) leaves=0 "
                         "progress=", line) and "(store_id=" not in line, line
        assert f"LCM compression no-op: {named}" in [msg for _lvl, msg in lines[start:]]
        status = engine.get_status()
        assert status["last_compression_noop_reason"] == named
        assert status["zero_leaf_streak"]["last_reason"] == named
        if sweep:
            assert status["threshold_full_sweep"]["stop_reason"] == named
    finally:
        engine.shutdown()


def test_a_sanitizing_retained_stop_is_not_pressure_relief(tmp_path, summaries, lines, monkeypatch):
    """With the sweep off the restore stop returns a sanitized list: no leaf was written, so the fresh-tail
    pressure evidence is not reset and the reason is kept."""
    engine, view = _restore(tmp_path, False)
    relieved = []
    monkeypatch.setattr(engine, "_note_fresh_tail_pressure_relieved", lambda: relieved.append(1))
    try:
        trace = []
        while not (trace and trace[-1]["reason"].startswith(RETAINED)) and len(trace) < 4:
            relieved.clear()
            view, step = _passes(engine, view, 1, lines, "restore")  # leaves still form before the stop
            trace += step
        assert trace[-1]["status"] == "sanitized" and trace[-1]["reason"].startswith(RETAINED), _show(trace)
        assert relieved == []
    finally:
        engine.shutdown()


def _streak(engine):
    return (engine.get_status()["zero_leaf_streak"] or {}).get("passes")


def _drop_r(view) -> list:
    return [m for m in view if not str(m.get("content", "")).startswith("[R] retained")]


def test_doctor_warns_on_a_zero_leaf_streak_and_clears_after_a_leaf(tmp_path, summaries, lines):
    """Three consecutive over-threshold passes that select a chunk and store no leaf warn with the last reason (in
    lcm_status too); a stored leaf clears it."""
    engine, view = _reorder(tmp_path, False)

    def doctor():
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        return next(check for check in report["checks"] if check["check"] == "compaction_progress")

    try:
        assert doctor() == {"check": "compaction_progress", "status": "pass", "detail": {"passes": 0}}
        streaks = []
        for _ in range(4):  # the leaf before R, then R stops every pass
            view, _trace = _passes(engine, view, 1, lines)
            streaks.append(_streak(engine))
            if len(streaks) < 4:
                assert doctor()["status"] == "pass"
        assert streaks == [None, 1, 2, 3]
        check, named = doctor(), f"{RETAINED} (store_id={_r_id(engine)} role=user kind=shown)"
        assert check["status"] == "warn" and check["detail"] == {"passes": 3, "last_reason": named}
        assert json.loads(lcm_tools.lcm_status({}, engine=engine))["zero_leaf_streak"] == check["detail"]
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        guidance = next(item for item in report["guidance"] if item["check"] == "compaction_progress")
        assert guidance["warning_only"] and "blocked=" in guidance["operator_action"]
        view, trace = _passes(engine, _drop_r(view), 1, lines)  # the host stops showing R: a leaf
        assert trace[0]["leaves"] >= 1, _show(trace)
        assert _streak(engine) is None and doctor()["status"] == "pass"
    finally:
        engine.shutdown()


# -- review of 559ce54f: the streak counts only real over-threshold passes; the stop line keeps base's reason ---

def _streak_of_three(tmp_path, lines):
    engine, view = _reorder(tmp_path, False)
    view, _trace = _passes(engine, view, 4, lines)
    assert _streak(engine) == 3
    return engine, view


def _one_pass(engine, view, **kwargs):
    view = engine.compress(view, **kwargs)
    engine.note_turn_complete()
    engine._no_progress_hold = None
    return view


def test_a_pass_at_or_under_the_threshold_clears_the_streak(tmp_path, summaries, lines):
    """Review 2: the streak is consecutive OVER-threshold passes: a pass the host observes at or under it starts it
    over, and so does the pressure-yield entry point that sees the session under the threshold."""
    engine, view = _streak_of_three(tmp_path, lines)
    try:
        view = _one_pass(engine, view, current_tokens=1)
        assert _streak(engine) is None
        view, _trace = _passes(engine, view, 1, lines)
        view = _one_pass(engine, view, current_tokens=engine.threshold_tokens)  # exactly at the threshold
        assert _streak(engine) is None
        view, _trace = _passes(engine, view, 1, lines)
        assert _streak(engine) == 1
        engine._config.fresh_tail_pressure_yield_enabled = True
        engine._maybe_engage_fresh_tail_pressure_yield(view, 1, eligible_tokens=0)
        assert _streak(engine) is None
    finally:
        engine.shutdown()


def test_an_unknown_host_count_counts_the_view(tmp_path, summaries, lines):
    """PR round 1: ``current_tokens=0`` is an unknown host count (as in compress()): the view decides, so an
    over-threshold pass that stores no leaf still counts."""
    engine, view = _streak_of_three(tmp_path, lines)
    try:
        _one_pass(engine, view, current_tokens=0)
        assert _streak(engine) == 4
    finally:
        engine.shutdown()


def test_only_passes_that_select_a_leaf_count(tmp_path, summaries, lines, monkeypatch):
    """Review 3: calls the #651 hold turns into cleanup-only maintenance, and passes that select no chunk, never
    count; a new session that keeps the conversation id starts without the old streak."""
    engine, view = _reorder(tmp_path, False)
    try:
        view, _trace = _passes(engine, view, 1, lines)  # the leaf before R
        view = engine.compress(view, current_tokens=10_000)  # R stops it; the #651 hold arms
        assert _streak(engine) == 1 and engine._no_progress_hold is not None
        for _ in range(2):  # held: the hold is not cleared between these calls
            view = engine.compress(view, current_tokens=10_000)
            assert engine._last_compression_status == "sanitized"
        assert _streak(engine) == 1
        engine._no_progress_hold = None
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_select_oldest_leaf_chunk", lambda *_args, **_kwargs: [])
            patch.setattr(engine, "_store_complete_backlog", lambda *_args, **_kwargs: False)
            view = _one_pass(engine, view, current_tokens=10_000)  # no chunk selected
            assert engine._last_compression_noop_reason == "no eligible leaf chunk selected"
        assert _streak(engine) == 1
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv")
        assert _streak(engine) is None
    finally:
        engine.shutdown()


def test_an_overlapping_compaction_keeps_this_pass_counted(tmp_path, summaries, lines, monkeypatch):
    """PR round 1: the selection mark belongs to its compress() call (its budget), so a compaction that overlaps it on
    the shared engine (a child's call, here an empty one inside this pass) does not unmark this pass."""
    engine, view = _streak_of_three(tmp_path, lines)
    extend, overlapped = engine._identity_anchor_extend_chunk, []

    def overlapping(chunk, candidates):
        if not overlapped:
            overlapped.append(engine.compress([], current_tokens=0))
        return extend(chunk, candidates)

    monkeypatch.setattr(engine, "_identity_anchor_extend_chunk", overlapping)
    try:
        _one_pass(engine, view, current_tokens=10_000)
        assert overlapped and engine._last_compression_noop_reason.startswith(RETAINED)
        assert _streak(engine) == 1  # the overlapping empty call cleared the streak; this pass still counted
    finally:
        engine.shutdown()


def test_a_sweep_that_stores_leaves_then_stops_names_the_row(tmp_path, summaries, lines):
    """PR round 1: a threshold sweep that stores leaves and then stops at a retained occurrence (its no-op reason is
    empty after the leaves; the sweep record keeps the reason) keeps blocked= on its stop line, reason= as on base."""
    from hermes_lcm.escalation import ForegroundBudget, ForegroundEstimates

    engine, _view = _reorder(tmp_path, True)
    try:
        budget = ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0, estimates=ForegroundEstimates())
        budget.sweep_active, budget.leaves, budget.selected = True, 2, True
        engine._last_compression_status, engine._last_compression_noop_reason = "compacted", ""
        engine._last_threshold_full_sweep = {"stop_reason": f"{RETAINED} (store_id=3 role=user kind=shown)"}
        engine._store_complete_blocked = (3, "user", "shown", 0, frozenset({3}))
        engine._leaf_selection_ran = True  # (the per-engine mark of the previous revision, for its red receipt)
        start = len(lines)
        engine._finish_foreground_budget(budget, [], 10_000)
        line = next(msg for _lvl, msg in lines[start:] if msg.startswith("LCM compaction stop:"))
        assert f"reason={RETAINED} leaves=2 " in line and line.endswith(" blocked=store_id:3,role:user,kind:shown")
    finally:
        engine.shutdown()


def test_the_stop_line_parsers_read_the_blocked_line(tmp_path, summaries, lines):
    """Review 4: the in-repo stop-line parsers (#605, the host fallback, #671) read a stop line with ``blocked=``."""
    from tests.test_host_fallback_stop_line import STOP as host_fallback
    from tests.test_issue_605_foreground_budget import STOP_LINE as foreground_budget
    from tests.test_issue_671_stub_first_exit import STOP as stub_first_exit

    engine, view = _reorder(tmp_path, False)
    try:
        view, trace = _passes(engine, view, 3, lines)
        line = next(t["line"] for t in trace if " blocked=" in t["line"])
        for parser in (foreground_budget, host_fallback, stub_first_exit):
            found = parser.search(line)
            assert found and found.group(1) == "noop" and found.group(2) == "0", (parser.pattern, line)
    finally:
        engine.shutdown()


def _no_source(tmp_path, monkeypatch, sweep, empty):
    """Inject the selected no-source boundary; attribution and all stop surfaces stay real."""
    pin_char_counter(monkeypatch)
    engine = _engine(tmp_path, sweep)
    view = [row for i in range(1, 5) for row in _turn(f"T{i}", 100.0 * i)]
    summary = {"role": "user", "content": "[CONTEXT COMPACTION] Earlier turns were compacted."}
    select = engine._select_leaf_candidate

    def select_without_source(state, leaf, *args):
        action = select(state, leaf, *args)
        assert action == ("next", None)
        # A native summary has no raw source; the real mapper bounds the scan below
        # the first retained mapped row. An empty hidden scan has the same outcome.
        leaf.to_compact = [] if empty else [summary]
        return action

    monkeypatch.setattr(engine, "_select_leaf_candidate", select_without_source)
    return engine, view


@SWEEP
@pytest.mark.parametrize("empty", [False, True], ids=["host_summary", "empty_scan"])
def test_no_source_stop_reaches_log_status_and_doctor(tmp_path, summaries, lines, monkeypatch, sweep, empty):
    engine, view = _no_source(tmp_path, monkeypatch, sweep, empty)
    try:
        engine.ingest(view)
        before = _rows(engine)
        for _ in range(3):
            start = len(lines)
            view, _ = _passes(engine, view, 1, lines)
            reason = "leaf input has no stored source rows"
            line = next(msg for _, msg in lines[start:] if msg.startswith("LCM compaction stop:"))
            assert re.search(f"reason={reason} leaves=0 " if sweep else "reason=(noop|sanitized) leaves=0 ", line), line
            assert f" empty_input=no_source,rows:{0 if empty else 1}" in line, line
            assert "(input_rows_without_source=" not in line and "blocked=" not in line and RETAINED not in line, line
            full_reason = f"{reason} (input_rows_without_source={0 if empty else 1})"
            assert f"LCM compression no-op: {full_reason}" in [msg for _, msg in lines[start:]]
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert status["last_compression_noop_reason"] == full_reason
        assert status["zero_leaf_streak"]["last_reason"] == full_reason
        progress = next(check for check in json.loads(lcm_tools.lcm_doctor({}, engine=engine))["checks"]
                        if check["check"] == "compaction_progress")
        assert progress["status"] == "warn"
        assert progress["detail"] == status["zero_leaf_streak"]
        assert progress["detail"]["passes"] == 3
        assert _rows(engine) == before and not summaries
    finally:
        engine.shutdown()


@pytest.mark.parametrize("empty", [False, True], ids=["host_summary", "empty_scan"])
def test_the_stop_line_parsers_read_the_no_source_line(tmp_path, summaries, lines, monkeypatch, empty):
    from tests.test_host_fallback_stop_line import STOP as host_fallback
    from tests.test_issue_605_foreground_budget import STOP_LINE as foreground_budget
    from tests.test_issue_671_stub_first_exit import STOP as stub_first_exit

    engine, view = _no_source(tmp_path, monkeypatch, False, empty)
    try:
        _view, trace = _passes(engine, view, 1, lines)
        for parser in (foreground_budget, host_fallback, stub_first_exit):
            found = parser.search(trace[0]["line"])
            assert found and found.group(1) in {"noop", "sanitized"} and found.group(2) == "0", trace[0]["line"]
    finally:
        engine.shutdown()


@SWEEP
def test_contiguity_cut_does_not_claim_a_retained_blocker(tmp_path, monkeypatch, lines, sweep):
    """A real contiguity cut supplies the reason; exercise its final logger at unit level."""
    from hermes_lcm.escalation import ForegroundBudget, ForegroundEstimates

    pin_char_counter(monkeypatch)
    engine = _engine(tmp_path, sweep)
    rows = [{"role": "user", "content": f"row {i}" + PAD} for i in range(3)]
    try:
        engine.ingest(rows)
        stored = _rows(engine)
        mapping = {id(row): saved["store_id"] for row, saved in zip(rows, stored)}
        chunk = [rows[2], rows[0]]
        result = engine._store_complete_input(chunk, {}, {id(row): mapping[id(row)] for row in chunk},
                                              chunk, chunk, 0, {}, 1, scanned=(stored, False))
        assert result == [] and engine._store_complete_blocked is None
        reason = "leaf input cannot form a contiguous stored prefix"
        assert engine._store_complete_empty_reason == reason
        assert _rows(engine) == stored
        budget = ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0, estimates=ForegroundEstimates())
        budget.sweep_active, budget.selected = sweep, True
        engine._last_compression_status, engine._last_compression_noop_reason = "noop", reason
        engine._last_threshold_full_sweep = {"stop_reason": reason} if sweep else {}
        engine._finish_foreground_budget(budget, [], 10_000)
        line = next(msg for _, msg in lines if msg.startswith("LCM compaction stop:"))
        assert f"reason={reason if sweep else 'noop'} leaves=0 " in line, line
        assert line.endswith(" empty_input=contiguity") and "blocked=" not in line, line
    finally:
        engine.shutdown()


def test_prepared_input_probe_preserves_the_active_stop_diagnostics(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    engine = _engine(tmp_path, False)
    summary = {"role": "user", "content": "[CONTEXT COMPACTION] Earlier turns were compacted."}
    try:
        engine._store_complete_blocked = (7, "user", "shown", 0)
        engine._store_complete_empty_reason = "previous diagnostic"
        assert engine._prepared_chunk_consumption([summary], [summary], [], set()) is None
        assert engine._store_complete_blocked == (7, "user", "shown", 0)
        assert engine._store_complete_empty_reason == "previous diagnostic"
    finally:
        engine.shutdown()
