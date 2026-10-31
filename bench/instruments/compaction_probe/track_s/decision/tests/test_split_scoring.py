"""Deterministic split verdicts: offline fixtures, no reader or provider calls."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_scoring import fact_verdicts, request_ground_truth, score_event  # noqa: E402


def fact(kind="tool_number", value="rows_verified=12345", fixture="alpha"):
    return dict(id="F1", fixture=fixture, value=value, answer=value, **{"class": kind})


@pytest.mark.parametrize("context,expected", [("rows_verified=12345", True), ("12345", False)])
def test_verbatim_context_is_independent_of_answer(context, expected):
    assert fact_verdicts(fact(), "12345", context, [fact()])["verbatim"] is expected


@pytest.mark.parametrize("answer,expected", [("12345", True), ("123456", False), ("912345", False), ("unknown", False)])
def test_scalar_exact_digits_without_prefix(answer, expected):
    v = fact_verdicts(fact(), answer, "rows_verified=12345", [fact()])
    assert v["scalar"] is expected
    if answer == "12345":
        assert not v["strict (historical)"]
        assert v["association"]


def test_non_numeric_identifiers_have_no_scalar_verdict():
    f = fact("build_id", "abc12345def")
    assert fact_verdicts(f, f["value"], f["value"], [f])["scalar"] is None


@pytest.mark.parametrize("kind,value,positive,negative", [
    ("name", "workspace-abcdef", "workspace-abcdef", "workspace-bbbbbb"),
    ("path", "src/abcdef/settings.toml", "src/abcdef/settings.toml", "src/bbbbbb/settings.toml"),
    ("limit", "12345 MiB", "12345 MiB", "12345 bytes"),
    ("build_id", "abc123def456", "abc123def456", "bbb123def456"),
    ("prefix", "abc123def456-", "abc123def456-", "bbb123def456-"),
    ("decision", "snapshot-abc123def456 because read-only replay preserves ownership",
     "snapshot-abc123def456; read only replay preserves ownership", "snapshot-abc123def456"),
    ("superseded_value", "abc123def456", "abc123def456", "bbb123def456"),
    ("error_with_fix", "E_abc123def456: cursor missing; fixed by rebuilding the fixture index",
     "E_abc123def456: missing cursor, fixed via rebuild of fixture indexes", "E_abc123def456: rebuilding index"),
    ("pending_task", "verify abc123def456 rollback before release",
     "abc123def456 rollback verification before release", "abc123def456 rollback remains pending"),
    ("early_user_constraint", "Never modify fixtures/abc123def456/source.json",
     "Preserve fixtures/abc123def456/source.json unchanged", "Modify fixtures/abc123def456/source.json"),
    ("tool_number", "rows_verified=12345", "verified rows: 12345", "12345"),
    ("file_change", "src/abc123def456/loader.py: preserve tool_call_id in replay",
     "src/abc123def456/loader.py preserves replay tool_call_id", "src/abc123def456/loader.py"),
])
def test_semantic_requires_identifier_and_qualifiers(kind, value, positive, negative):
    f = fact(kind, value)
    assert fact_verdicts(f, positive, value, [f])["semantic"]
    assert not fact_verdicts(f, negative, value, [f])["semantic"]


@pytest.mark.parametrize("answer,expected", [("12345", True), ("67890", False),
    ("12345 or 67890", False), ("beta: 12345", False)])
def test_association_rejects_sibling_values_and_other_fixture(answer, expected):
    a, b = fact(), dict(fact(value="rows_verified=67890", fixture="beta"), id="F2")
    assert fact_verdicts(a, answer, a["value"], [a, b])["association"] is expected


def test_stale_value_and_missing_answer_do_not_pass():
    f = dict(fact("name", "new-workspace"), stale="old-workspace")
    v = fact_verdicts(f, "new-workspace or old-workspace", f["value"], [f])
    assert not v["association"] and not v["semantic"]
    # Historical classifier is deliberately unchanged, including this old weakness.
    assert v["strict (historical)"]
    assert fact_verdicts(f, None, None, [f])["semantic"] is None
    assert fact_verdicts(f, None, None, [f])["verbatim"] is None


def material():
    rows = [dict(id="r0", role="user", content="Audit fixture replay seed 1 and prepare the read-only rollback report."),
            dict(id="r1", role="user", content="Continue the read-only fixture audit; leave the rollback task pending."),
            dict(id="r2", role="user", content="Correction: use abc123; prior value is obsolete.")]
    specs = [dict(id="R", kind="current_request", answer=rows[0]["content"], row_index=0, row_id="r0"),
             dict(id="Q", kind="constraint_quote", answer="Keep fixtures read-only."),
             dict(id="O", kind="constraint_obey", answer="READ_ONLY_AUDIT", verbatim="Keep fixtures read-only.")]
    return rows, specs


def test_request_split_latest_correction_and_explicit_supersession():
    rows, specs = material()
    truth = request_ground_truth(rows, specs)
    assert truth["initial"]["superseded_by"] is None
    assert truth["latest_rule"] == "latest user row, including fact corrections"
    receipt = dict(answers=dict(R=rows[1]["content"], Q="Keep fixtures read-only.", O="READ_ONLY_AUDIT"),
                   checks=[dict(kind="constraint_quote", present_verbatim=False, reader_pass=True),
                           dict(kind="constraint_obey", present_verbatim=False, reader_pass=True)])
    v = score_event(receipt, rows[0]["content"], 1, truth)
    assert v["R-initial"]["retained"] and v["R-latest"]["reader_compliance"]
    assert not v["Q"]["verbatim"] and v["Q"]["reader_compliance"]
    assert not v["O"]["verbatim"] and v["O"]["reader_compliance"]
    assert not score_event(receipt, "", 2, truth)["R-latest"]["reader_compliance"]
    truth["initial"]["superseded_by"] = dict(row_index=2, row_id="r2")
    assert not score_event(receipt, rows[0]["content"], 2, truth)["R-initial"]["operative"]
    assert score_event(receipt, rows[0]["content"], 2, truth)["R-initial"]["retained"] is None


def test_initial_semantic_and_missing_event_receipts():
    rows, specs = material()
    truth = request_ground_truth(rows, specs)
    v = score_event(dict(answers=dict(R="Read only audit of seed 1 fixture replay; prepare rollback report")), "", 1, truth)
    assert v["R-initial"]["semantic"] and v["R-initial"]["retained"]
    assert not v["R-latest"]["reader_compliance"]
    v = score_event(dict(answers=dict(R="Audit seed 2")), "", 1, truth)
    assert not v["R-initial"]["semantic"] and not v["R-initial"]["retained"]
    v = score_event({}, None, 1, truth)
    assert v["R-initial"]["retained"] is None
    assert v["O"]["reader_compliance"] is None
