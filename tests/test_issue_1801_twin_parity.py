"""Carried date spelling parity and the approved mixed-date exception."""
from datetime import date
import json
import re

import pytest

from hermes_lcm import tools
from hermes_lcm.evidence_compiler import compile_preanswer_evidence
from hermes_lcm.reasoning import _quote_supports_day
from hermes_lcm.requirements_compiler import _carried_absolute_day
from tests.harness import engine_context
from tests.test_issue_1801_carried_followups import REASON, args, plant

FORMS = ["2023-02-10", "February 10, 2023", "Feb 10, 2023",
         "February 10 2023", "Feb 10th, 2023"]
QUESTIONS = {"date_filter": "Which vacation did I take in {month}?",
             "order": "What is the order of events in {month}?",
             "finite": "How many vacations did I take in {month}?"}


def normalize(value, spelling, iso=FORMS[0]):
    """Compare answer fields; ignore spelling, spans, digests and diagnostic metrics."""
    if isinstance(value, dict):
        return {k: normalize(v, spelling, iso) for k, v in value.items()
                if not k.endswith("sha256") and k not in {
                    "metrics", "latency_ms", "span_start", "span_end", "char_start", "char_end"}}
    if isinstance(value, list):
        return [normalize(v, spelling, iso) for v in value]
    if isinstance(value, str):
        return re.sub(r"lcm:(\d+):\d+-\d+", r"lcm:\1:SPAN",
                      value.replace(spelling, iso) if spelling else value)
    return value


def run(engine, refs, operation, window):
    month = "February" if window == "inside" else "March"
    return json.loads(tools.lcm_compile_evidence(args(refs,
        QUESTIONS[operation].format(month=month), "2023-03-22"), engine=engine))


def quote(shape, spelling):
    if shape == "mixed":
        return f"I reviewed the plan on {spelling}. I took a vacation to Bali yesterday."
    if shape == "two-date":
        return f"I took a vacation to Bali on {spelling} (booked February 1, 2023)."
    if shape == "time-suffix":
        return f"I took a vacation to Bali on {spelling}T09:00."
    if shape == "yearless":
        return "I took a vacation to Bali on " + re.sub(r"^2023-|,? 2023$", "", spelling) + "."
    return f"I took a vacation to Bali on {spelling}."


@pytest.mark.parametrize("shape", ["single", "mixed", "time-suffix", "two-date", "yearless"])
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("operation", QUESTIONS)
@pytest.mark.parametrize("window", ["inside", "outside"])
def test_carried_twin_parity(tmp_path, shape, form, operation, window):
    # A yearless ISO twin is also yearless; it must not acquire a year.
    iso_form = "02-10" if shape == "yearless" else FORMS[0]
    results = []
    for arm, spelling in enumerate([iso_form, form]):
        with engine_context(tmp_path / str(arm), embeddings_enabled=False) as engine:
            text = quote(shape, spelling)
            if shape == "two-date" and arm == 0 and form != FORMS[0]:
                # E7's twin respells EVERY named date, including parentheticals.
                text = text.replace("February 1, 2023", "2023-02-01")
            refs = plant(engine, [text,
                "I took a vacation to Kyoto on 2023-03-05."])
            before = engine._store.connection.total_changes
            result = run(engine, refs, operation, window)
            assert engine._store.connection.total_changes == before
            normalized = normalize(result, spelling)
            if shape == "two-date" and form != FORMS[0]:
                normalized = normalize(normalized, "February 1, 2023", "2023-02-01")
            results.append(normalized)
    iso, named = results
    if shape == "time-suffix" and form != FORMS[0] and named != iso:
        # Named-month + ISO T-time hybrids are unsupported (e4). Each differing
        # cell must be fail-closed, never a new answer or a certain carried time.
        assert named["state"] == "unknown"
        assert named["direct_fact"] is None and named["computation"] is None
        assert named["reason_code"] == REASON or named["reason_code"].startswith("finite_grounding_failed:")
        assert iso["state"] == "answer_sufficient" or iso["reason_code"] == REASON
    else:
        assert named == iso


@pytest.mark.parametrize("form", ["2023-01-10", "January 10, 2023"])
@pytest.mark.parametrize("operation", QUESTIONS)
@pytest.mark.parametrize("window", ["inside", "outside"])
def test_mixed_carried_withheld_and_recorded(tmp_path, form, operation, window):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [f"I reviewed the plan on {form}. I took a vacation to Bali yesterday.",
            "I took a vacation to Kyoto on 2023-01-12."])
        request = args(refs, QUESTIONS[operation].format(
            month="January" if window == "inside" else "March"), "2023-03-22")
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compile_evidence(request, engine=engine))
        assert result["state"] == "unknown"
        assert result["reason_code"] == REASON
        assert result["direct_fact"] is None and result["computation"] is None
        assert any(x["time_uncertain"] == "carried" and x["reason_code"] == REASON
                   for x in result.get("exclusions", []))
        hook = compile_preanswer_evidence(request["question"], engine=engine,
            question_date=request["question_date"], baseline_refs=refs, enabled=True,
            render_baseline_context=True, budgets={"max_retrieval_calls": 0})
        assert hook["context"] is None
        assert hook["direct_fact"] is None and hook["computation"] is None
        assert any(x["reason_code"] == REASON for x in hook.get("exclusions", []))
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("text", ["I took a vacation to Bali on 2023-02-10T09:00.",
    "I took a vacation to Bali on 2023-02-10 (booked February 1, 2023)."])
@pytest.mark.parametrize("operation", QUESTIONS)
@pytest.mark.parametrize("window", ["inside", "outside"])
def test_explicit_carried_window_keeps_base(tmp_path, text, operation, window):
    # The ordinary twin is a control for base's explicit-event window filtering.
    outputs = []
    for arm, carried_index in enumerate([0, 2]):
        with engine_context(tmp_path / str(arm), embeddings_enabled=False) as engine:
            refs = plant(engine, [text, "I took a vacation to Kyoto on 2023-03-05.",
                "Unrelated synthetic row."], carried_index=carried_index)[:2]
            before = engine._store.connection.total_changes
            outputs.append(run(engine, refs, operation, window))
            assert engine._store.connection.total_changes == before
    carried, ordinary = outputs
    if "booked" in text:
        # #1814 supersedes the former ISO-first exception for distinct days.
        for output in outputs:
            assert output["state"] == "unknown"
            assert output["direct_fact"] is None and output["computation"] is None
            assert output["exclusions"]
    elif window == "outside":
        assert normalize(carried, "") == normalize(ordinary, "")
        if operation == "date_filter":
            assert "Kyoto" in carried["direct_fact"]["value"]
    elif operation == "date_filter":
        assert carried["state"] == ordinary["state"] == "answer_sufficient"
        assert carried["direct_fact"]["value"] == ordinary["direct_fact"]["value"]
        assert carried["direct_fact"]["time_uncertain"] == "carried"
    else:
        # Base deliberately withholds carried computation with an as-of cutoff.
        assert carried["state"] == "unknown"
        assert carried["direct_fact"] is None and carried["computation"] is None


@pytest.mark.parametrize("form", FORMS)
def test_single_date_parser_parity(form):
    text = quote("single", form)
    expected = date(2023, 2, 10)
    assert _carried_absolute_day(dict(time_uncertain="carried", quote=text)) == expected
    assert _quote_supports_day(text, expected, "")
    assert not _quote_supports_day(text, date(2023, 2, 11), "")


def test_two_date_parser_divergence_is_deliberate():
    text = quote("two-date", FORMS[0])
    assert _carried_absolute_day(dict(time_uncertain="carried", quote=text)) is None
    assert _quote_supports_day(text, date(2023, 2, 10), "")
    assert _quote_supports_day(text, date(2023, 2, 1), "")


def test_ordinary_mixed_date_binding_characterization(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, ["I reviewed the plan on 2023-01-10. I took a vacation to Bali yesterday.",
            "Unrelated synthetic row."], carried_index=1)[:1]
        result = json.loads(tools.lcm_compile_evidence(args(refs,
            "Which vacation did I take in January?", "2023-03-22"), engine=engine))
        assert result["direct_fact"] is None
