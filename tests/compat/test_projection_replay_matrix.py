"""Replay protocol floors, including the intentional #919 old-reader boundary."""

import json

import pytest

from tests.compat import codec_adapter
from tests.compat.helpers import CASES, GOLDEN, READERS, emit_case, open_engine, run_old

# diagnostics.survival_fit_projection_floor_applies and test_issue_919 define
# the conservative v0.27.0 floor. Legacy and long forms predate every pinned
# reader; test_issue_916 and test_issue_959 define the changed short forms.
# The pinned readers start at v0.25.0-rc1. The operator guide's v0.24.5
# plugin-only rollback floor is not exercised here (follow-up on #1271).
SHORT_READER_FLOOR = "v0.27.0"
READER_ORDER = ("v0.25.0-rc1", "v0.25.1", "v0.26.2", "v0.27.0", "v0.28.0")
EXPECTED = {
    "mark_only": "supported",
    "head_mark": "documented_short_boundary",
    "head_mark_tail": "supported",
    "bounded_calls": "supported",
    "stored_calls": "documented_short_boundary",
    "followers": "supported",
    "keep_reply": "supported",
}
# Readers below SHORT_READER_FLOOR use EXPECTED; newer readers support all forms.
# S6-F1 remains a strict failure on every reader's supported name-only tool-result cells
# S6-F2 applies only to non-empty bounded calls below the floor.


def below_short_floor(reader):
    return READER_ORDER.index(reader) < READER_ORDER.index(SHORT_READER_FLOOR)


class ReplayMismatch(AssertionError):
    """Only the proven duplicate-row failure may satisfy a strict xfail."""


def replay_cases(old_reader=False):
    cases = []
    for case in CASES:
        reason = None
        if case.endswith("-named_tool_result") and (not old_reader or not case.startswith("head_mark-")):
            reason = "S6-F1: name-only tool result loses tool_name; cold replay stores the tool pair again"
        elif old_reader and case in ("bounded_calls-text", "bounded_calls-structured"):
            reason = "S6-F2: old codec expects mark-only content and re-stores preserved non-empty bounded-call rows"
        marks = pytest.mark.xfail(strict=True, raises=ReplayMismatch, reason=reason) if reason else ()
        cases.append(pytest.param(case, marks=marks, id=case))
    return cases

REPLAY = r'''
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    before = engine._store.get_session_messages("S")
    view = json.loads(Path(data).read_text())
    engine.ingest(view)
    first = engine._store.get_session_messages("S")
    engine.ingest(view)
    second = engine._store.get_session_messages("S")
    assert first[:len(before)] == second[:len(before)] == before
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"first_added": len(first) - len(before), "replay_added": len(second) - len(first),
                      "added_rows": first[len(before):]}))
finally:
    engine.shutdown()
'''


def assert_added_rows(rows, messages):
    from hermes_lcm.message_content import normalize_content_value

    keys = ("role", "content", "tool_calls", "tool_call_id", "tool_name", "observed_at")
    expected = [{"role": message["role"], "content": normalize_content_value(message.get("content")),
                 "tool_calls": message.get("tool_calls"), "tool_call_id": message.get("tool_call_id"),
                 "tool_name": message.get("tool_name"), "observed_at": message.get("timestamp")}
                for message in messages]
    assert [{key: row.get(key) for key in keys} for row in rows] == expected


@pytest.mark.parametrize("frozen", [False, True], ids=["emitted", "frozen"])
@pytest.mark.parametrize("case", replay_cases())
def test_candidate_warm_and_cold_replay(tmp_path, case, frozen):
    engine = open_engine(tmp_path)
    try:
        emitted, before = emit_case(engine, case)
        if frozen:
            emitted = json.loads(json.loads(GOLDEN.read_text())[case])
        engine.ingest(emitted)
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()
    engine = open_engine(tmp_path)
    try:
        engine.ingest(emitted)
        engine.ingest(emitted)
        after = engine._store.get_session_messages("S")
        assert after[:len(before)] == before
        if len(after) != len(before):
            assert_added_rows(after[len(before):], emitted[-2:])
            raise ReplayMismatch(f"cold replay added {len(after) - len(before)} rows")
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("reader", "case"), [
    pytest.param(reader, case.values[0], marks=case.marks, id=f"{reader}-{case.id}")
    for reader in READERS for case in replay_cases(old_reader=below_short_floor(reader))
])
def test_old_reader_projection_replay(tmp_path, old_tree, host_root, reader, case):
    from hermes_lcm.diagnostics import doctor_guidance_for_check, survival_fit_projection_floor_applies
    from tests.test_host_uid_shadow import ROLLBACK_READERS

    assert READERS == ROLLBACK_READERS
    old = old_tree(reader)
    engine = open_engine(tmp_path)
    try:
        emitted, _ = emit_case(engine, case)
    finally:
        engine.shutdown()
    host = tmp_path / "view.json"
    host.write_text(json.dumps(emitted))
    result = run_old(old, tmp_path, host_root, REPLAY, data=host)
    assert result["replay_added"] == 0, result
    expectation = EXPECTED[case.split("-")[0]] if below_short_floor(reader) else "supported"
    if expectation == "documented_short_boundary":
        # Short text/structured rows duplicate once; tool forms duplicate the pair.
        expected_rows = emitted[-2:] if ("tool" in case or case.startswith("stored_calls-")) else emitted[-1:]
        assert result["first_added"] == len(expected_rows), result
        assert_added_rows(result["added_rows"], expected_rows)
        diagnostic_dir = tmp_path / "diagnostic"
        diagnostic_dir.mkdir()
        diagnostic = open_engine(diagnostic_dir)
        try:
            # Generate the doctor record through the real fit, never a literal counter.
            view = [{"role": "user", "content": "diagnostic " * 1200, "timestamp": 10.0}]
            diagnostic.ingest(view)
            assert codec_adapter.fit(diagnostic, view, 650) != view
            detail = diagnostic._store.read_metadata_json("survival_fit:counter")
            assert detail["count"] == detail["projected_count"] == 1
        finally:
            diagnostic.shutdown()
        assert survival_fit_projection_floor_applies(detail)
        guidance = doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": detail})
        assert SHORT_READER_FLOOR in guidance["operator_action"]
        assert "v0.26.x or earlier cannot recognise" in guidance["operator_action"]
    else:
        if result["first_added"] != 0:
            assert result["first_added"] == 2, "different failure from S6-F1/S6-F2"
            assert_added_rows(result["added_rows"], emitted[-2:])
            raise ReplayMismatch(result)


def test_unprojected_fit_has_no_reader_floor(tmp_path):
    from hermes_lcm.diagnostics import survival_fit_projection_floor_applies

    engine = open_engine(tmp_path)
    try:
        view = [{"role": "user", "content": "earlier " * 1200, "timestamp": 1.0},
                {"role": "assistant", "content": "", "timestamp": 2.0},
                {"role": "user", "content": "Next request.", "timestamp": 3.0}]
        engine.ingest(view)
        assert codec_adapter.fit(engine, view, 650) == view[-1:]
        detail = engine._store.read_metadata_json("survival_fit:counter")
        assert detail["count"] == 1 and detail["projected_count"] == 0
        assert not survival_fit_projection_floor_applies(detail)
    finally:
        engine.shutdown()
