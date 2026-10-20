"""#1162: distinguish no-call survival fitting from a recorded summary failure."""
import json

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.dag import SummaryNode
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tools import lcm_doctor, lcm_grep, lcm_load_session, lcm_status
from tests import test_issue_581_582_store_complete_survival as fit
from tests import test_issue_1162_retained_escape as retained
from tests.harness import make_engine

FITTED = ("To keep this conversation within the model's context window, {n} older messages left live context "
          "during LCM context fitting; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
          "/lcm doctor reports why.")
FAILED = ("LCM could not summarise part of this conversation in time. To keep the session alive, {n} older "
          "messages left live context; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
          "/lcm doctor reports it.")
summaries = retained.summaries
host_estimator = fit.host_estimator


@pytest.fixture
def no_call(tmp_path, monkeypatch, summaries):
    engine, view = retained._reorder(tmp_path, False)
    # Reach the retained stop with the existing repro, before applying a fit budget.
    for _ in range(3):
        view = engine.compress(view, current_tokens=10_000)
        engine.note_turn_complete()
        engine._no_progress_hold = None
        if engine._last_compression_noop_reason.startswith(retained.RETAINED):
            break
    assert engine._last_compression_noop_reason.startswith(retained.RETAINED)
    summaries.clear()
    monkeypatch.setattr(engine, "_survival_fit_budget", lambda *args, **kwargs: 850)
    anchored = []
    original = engine._identity_anchor_summary_input

    def anchor(*args, **kwargs):
        result = original(*args, **kwargs)
        anchored.append(result)
        return result

    monkeypatch.setattr(engine, "_identity_anchor_summary_input", anchor)
    records = []
    record = engine._survival_record

    def capture(reason, count, ids, *args, **kwargs):
        records.append(list(ids))
        return record(reason, count, ids, *args, **kwargs)

    monkeypatch.setattr(engine, "_survival_record", capture)
    try:
        stored = retained._rows(engine)
        result = engine.compress(view, current_tokens=engine._survival_measure(view))
        assert anchored == [[]] and summaries == []
        assert engine._last_compression_noop_reason.startswith(retained.RETAINED)
        assert len(result) < len(view) and len(records) == 1 and records[0]
        assert engine._last_survival_fit["uncovered_rows"] == len(records[0])
        yield engine, records[0], stored
    finally:
        engine.shutdown()


def test_regression_no_call_warning(no_call):
    engine, ids, _ = no_call
    warning = engine._survival_fit_pending_warning[1]
    assert warning == FITTED.format(n=len(ids)) and "in time" not in warning


def test_regression_no_call_cause(no_call):
    engine, _, _ = no_call
    cause = engine._last_compression_noop_reason
    assert engine._last_survival_fit["cause"] == cause
    assert engine._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)["last_cause"] == cause


def test_regression_status_cause(no_call):
    engine, _, _ = no_call
    assert json.loads(lcm_status({}, engine=engine))["last_survival_fit"]["cause"] == (
        engine._last_compression_noop_reason)


def test_regression_command_doctor_cause(no_call):
    engine, _, _ = no_call
    observation = next(line for line in handle_lcm_command("doctor", engine).splitlines()
                       if "survival_fit: " in line)
    assert f"last cause {engine._last_compression_noop_reason}" in observation


def test_regression_tool_doctor_cause(no_call):
    engine, _, _ = no_call
    report = json.loads(lcm_doctor({}, engine=engine))
    record = engine._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)
    assert report["survival_fit"] == record
    assert report["survival_fit"]["last_cause"] == engine._last_compression_noop_reason
    assert not any(item["check"] == "survival_fit" for item in report["checks"])
    command = handle_lcm_command("doctor", engine)
    assert f"last cause {engine._last_compression_noop_reason}" in command


def test_control_fitted_rows_remain_retrievable(no_call):
    engine, ids, stored = no_call
    loaded = json.loads(lcm_load_session({"session_id": "S", "limit": 100}, engine=engine))
    loaded_by_id = {row["store_id"]: row for row in loaded["messages"]}
    for row in stored:
        if row["store_id"] not in ids:
            continue
        assert loaded_by_id[row["store_id"]]["content"] == row["content"]
        hits = json.loads(lcm_grep({"query": row["content"], "content_scope": "history",
                                   "limit": 100}, engine=engine))
        assert row["store_id"] in {hit["store_id"] for hit in hits["results"]}
    assert retained._rows(engine) == stored


def test_control_summary_timeout_keeps_warning(tmp_path, monkeypatch, host_estimator):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("synthetic summary timeout")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", timeout)
    try:
        view = fit._long_view()
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        assert calls
        assert engine._survival_fit_pending_warning[1] == FAILED.format(
            n=engine._last_survival_fit["dropped_rows"])
    finally:
        engine.shutdown()


@pytest.fixture
def publication(tmp_path, monkeypatch, summaries, host_estimator):
    engine = make_engine(tmp_path, window=fit.WINDOW, conversation_id="conv", platform="telegram",
                         fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
                         threshold_full_sweep_enabled=False)

    def conflict(*args, **kwargs):
        raise LifecyclePublicationConflictError("synthetic publication conflict")

    monkeypatch.setattr(engine._lifecycle, "stage_compaction_publication", conflict)
    try:
        view = fit._long_view()
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        assert summaries and engine._last_survival_fit["reason"] == "publication_invariant_conflict"
        yield engine
    finally:
        engine.shutdown()


def test_regression_publication_warning(publication):
    assert publication._survival_fit_pending_warning[1] == FITTED.format(
        n=publication._last_survival_fit["dropped_rows"])


def test_control_publication_reason_is_unchanged(publication):
    counter = publication._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)
    assert publication._last_survival_fit["reason"] == counter["last_reason"] == "publication_invariant_conflict"


@pytest.mark.parametrize("reason", ["sqlite_publication_locked", "lifecycle_binding_changed",
                                  "publication_invariant_conflict"])
def test_regression_unproven_reason_is_neutral(tmp_path, monkeypatch, host_estimator, reason):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda *a, **k: "")

    def publication_stop(messages, **kwargs):
        # The pass attempted a call that failed, then hit a publication stop.
        budget = engine._foreground_call_budget()
        escalation._invoke_summary_llm_chain("source", 10, budget=budget)
        monkeypatch.setattr(escalation, "_invoke_summary_llm", _timeout)
        escalation._invoke_summary_llm_chain("source", 10, budget=budget)
        engine._last_compression_status = "noop"
        engine._survival_fit_reason = reason
        return messages

    monkeypatch.setattr(engine, "_compress_impl", publication_stop)
    try:
        view = fit._long_view()
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        record = engine._last_survival_fit
        assert record["reason"] == reason
        assert "cause" not in record
        assert engine._survival_fit_pending_warning[1] == FITTED.format(n=record["dropped_rows"])
    finally:
        engine.shutdown()


def _timeout(*args, **kwargs):
    raise TimeoutError("synthetic summary timeout")


@pytest.mark.parametrize("sweep", [False, True])
def test_second_compaction_timeout(tmp_path, monkeypatch, host_estimator, sweep):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW,
                         threshold_full_sweep_enabled=sweep, l3_truncate_tokens=1)
    failing = False
    calls = []

    def summarize(*args, **kwargs):
        calls.append(failing)
        if failing:
            return _timeout()
        return "Earlier turns.\nExpand for details about: turns"

    monkeypatch.setattr(escalation, "_invoke_summary_llm", summarize)
    try:
        view = fit._long_view(turns=6)
        engine.ingest(view)
        first = engine.compress(view, current_tokens=host_estimator(view))
        engine._survival_fit_warned.clear()
        engine._survival_fit_pending_warning = engine._last_survival_fit = None
        engine._no_progress_hold = None
        engine._sweep_budget_hold_until = 0
        engine.note_turn_complete()
        extra = [row for i in range(6, 30) for row in fit._turn(f"L{i}", 10.0 * i, tool=i % 3 == 0)]
        failing = True
        view = [*first, *extra]
        engine.compress(view, current_tokens=host_estimator(view))
        assert False in calls and True in calls
        assert engine._last_compression_status in {"sanitized", "compacted"}
        assert engine._survival_fit_pending_warning[1] == FAILED.format(
            n=engine._last_survival_fit["dropped_rows"])
        assert engine._last_survival_fit["cause"] == "summary call failed: TimeoutError"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("outcome", ["rejection", "success", "all_rejected"])
def test_primary_timeout_then_fallback_rejection(tmp_path, monkeypatch, host_estimator, outcome):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW, summary_fallback_models=["fb"])
    calls = []

    def summarize(*args, model="", **kwargs):
        calls.append(model)
        if outcome == "all_rejected":
            return ""
        if model == "fb" and outcome == "success":
            return "Earlier turns.\nExpand for details about: turns"
        return "" if model == "fb" else _timeout()

    monkeypatch.setattr(escalation, "_invoke_summary_llm", summarize)
    try:
        view = fit._long_view(turns=200 if outcome == "success" else 80)
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        assert "" in calls and "fb" in calls
        assert escalation._summary_call.error is None
        failed = outcome == "rejection"
        assert engine._survival_fit_pending_warning[1] == (FAILED if failed else FITTED).format(
            n=engine._last_survival_fit["dropped_rows"])
        if failed:
            assert engine._last_survival_fit["cause"] == "summary call failed: TimeoutError"
        else:
            assert not engine._last_survival_fit.get("cause", "").startswith("summary call failed:")
        if outcome == "success":
            assert engine._last_compression_status == "compacted"
            assert engine._dag.get_session_nodes("S")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("status,mode", [("sanitized", "normal"), ("compacted", "normal"),
                         ("overflow_recovery", "recovery"), ("error", "normal"),
                         ("exception:RuntimeError", "normal"), ("noop", "exit")])
def test_pass_failure_inheritance(tmp_path, monkeypatch, host_estimator, status, mode):
    engine = fit._engine(tmp_path, context_length=100_000 if mode == "exit" else fit.WINDOW)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", _timeout)

    def stopped(messages, **kwargs):
        escalation._invoke_summary_llm_chain("source", 10, budget=engine._foreground_call_budget())
        # A later successful attempt must not erase the pass failure either.
        monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda *a, **k: "usable summary")
        escalation._invoke_summary_llm_chain("source", 10, budget=engine._foreground_call_budget())
        engine._foreground_budget = None  # another thread clearing the engine-wide budget is irrelevant
        if status.startswith("exception:"):
            raise RuntimeError("synthetic compaction exception")
        engine._last_compression_status = status
        return messages

    monkeypatch.setattr(engine, "_compress_impl", stopped)
    try:
        view = fit._long_view()
        engine.ingest(view)
        if mode == "exit":
            ids = [r["store_id"] for r in fit._rows(engine)]
            engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier turns", source_ids=ids))
            engine.threshold_tokens = host_estimator(view) // 2
        engine.compress(view, current_tokens=host_estimator(view), bypass_cooldown=mode == "recovery")
        record = engine._last_survival_fit
        prefix = {"normal": "", "recovery": "recovery_attempt:", "exit": "exit_fit:"}[mode]
        assert record["reason"] == prefix + status
        assert record["cause"] == "summary call failed: TimeoutError"
        assert engine._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)["last_cause"] == record["cause"]
        if mode == "exit":
            assert engine._survival_fit_pending_warning is None
        else:
            assert engine._survival_fit_pending_warning[1] == FAILED.format(n=record["dropped_rows"])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("boundary", ["reset", "start", "rollover"])
def test_status_drops_previous_session_fit(no_call, boundary):
    engine, _, _ = no_call
    assert json.loads(lcm_status({}, engine=engine))["last_survival_fit"] is not None
    if boundary == "reset":
        engine.on_session_reset()
    elif boundary == "start":
        engine.on_session_start("S2", conversation_id="conv2")
    else:
        engine.rollover_session("S", "S2", previous_messages=[], context_length=20_000)
    assert json.loads(lcm_status({}, engine=engine))["last_survival_fit"] is None


@pytest.mark.parametrize("record,expected", [({"count": 0, "projected_count": 0}, False),
                         ({"count": 0}, False), ({"count": 1, "projected_count": 0}, True),
                         ({"count": 0, "count_lost": True}, True),
                         ({"count": 0, "projected_count": 1}, True), ({"count": "damaged"}, True)])
def test_doctor_surfaces_agree(tmp_path, record, expected):
    engine = retained._engine(tmp_path, False, context_threshold=0.75)
    try:
        engine.ingest([{"role": "user", "content": "ordinary stored turn"}])
        baseline_command_status = handle_lcm_command("doctor", engine).splitlines()[1]
        baseline_tool_status = json.loads(lcm_doctor({}, engine=engine))["overall"]
        assert baseline_command_status == "status: ok"
        assert baseline_tool_status == "healthy"
        record = {**record, "last_cause": "synthetic fit cause"}
        engine._store.write_metadata_json([survival.SURVIVAL_FIT_COUNTER_KEY], json.dumps(record))
        command = handle_lcm_command("doctor", engine)
        tool = json.loads(lcm_doctor({}, engine=engine))
        assert tool["survival_fit"] == (record if expected else None)
        assert not any(c["check"] == "survival_fit" for c in tool["checks"])
        assert ("survival_fit: applied" in command) == expected
        assert ("last cause synthetic fit cause" in command) == expected
        assert tool["overall"] == baseline_tool_status
        assert command.splitlines()[1] == baseline_command_status
    finally:
        engine.shutdown()
