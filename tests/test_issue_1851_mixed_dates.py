"""Synthetic date pairs from #1851: ordinary/carried agreement at public tools."""
import json
from datetime import date

import pytest

from hermes_lcm import occurrence_time, tools
from hermes_lcm.requirements_compiler import _carried_absolute_day
from tests.harness import engine_context
from tests.test_issue_1801_carried_followups import args, plant

REASON = "ambiguous_multiple_explicit_dates"
QUESTIONS = [
    ("order", "What is the order of events?"),
    ("date_filter", "Which vacation did I take in January?"),
    ("date_interval", "How many days between the PlankChallenge and launch?"),
    ("scalar", "How many vacations did I take in January?"),
]
PAIRS = [
    ("2023-01/10", "January 12, 2023"),
    ("2023/01-10", "January 12, 2023"),
    ("2023-01/31", "February 2, 2023"),
    ("2023/01-31", "February 2, 2023"),
]


def row(pair):
    return f"I took a vacation to Bali on {pair[0]} and {pair[1]} for the PlankChallenge."


def assert_withheld(engine, refs, operation, question):
    result = json.loads(tools.lcm_compile_evidence(args(refs, question, "2023-03-22"), engine=engine))
    assert result["contract"]["operation"] == operation
    assert result["state"] == "unknown"
    assert result["direct_fact"] is None and result["computation"] is None


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("operation,question", QUESTIONS)
def test_conflicting_days_withhold_both_paths(tmp_path, pair, carried, operation, question):
    text = row(pair)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text, "The launch happened on 2023-01-20.", "Unrelated synthetic row."],
                     carried_index=0 if carried else 2)[:2]
        before = engine._store.connection.total_changes
        assert_withheld(engine, refs, operation, question)
        assert occurrence_time._quote_date_info(text)[1] == REASON
        assert occurrence_time.resolve_occurrence_time(text, observed_at=0)["reason"] == REASON
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("carried", [False, True])
def test_conflicting_days_withhold_compute(tmp_path, pair, carried):
    texts = [row(pair), "The launch happened on 2023-01-20.", "Unrelated synthetic row."]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, texts, carried_index=0 if carried else 2)
        operands = [dict(store_id=int(ref.split(":")[1]), span_start=0, span_end=len(text),
                         quote=text, label=label,
                         occurrence_time=dict(event_time_source="explicit", event_date=day))
                    for ref, text, label, day in zip(refs, texts, ["PlankChallenge", "launch"],
                                                     [pair[0].replace("/", "-"), "2023-01-20"])]
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compute(dict(question=QUESTIONS[2][1], evidence_complete=True,
                                                  operands=operands), engine=engine))
        assert result["status"] != "computed"
        assert occurrence_time._quote_date_info(texts[0])[1] == REASON
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("month", ["January", "February"])
def test_month_boundary_filter_withheld(tmp_path, carried, month):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        text = row(PAIRS[2])
        refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        assert_withheld(engine, refs, "date_filter", f"Which vacation did I take in {month}?")
        assert occurrence_time._quote_date_info(text)[1] == REASON


@pytest.mark.parametrize("spelling", ["2023-01/10", "2023/01-10"])
def test_mixed_only_matches_uniform_twin(tmp_path, spelling):
    expected = date(2023, 1, 10)
    assert occurrence_time._date_value(spelling) == expected
    for i, form in enumerate([spelling, "2023-01-10", "2023/01/10"]):
        text = f"I took a vacation to Bali on {form}."
        assert occurrence_time.resolve_occurrence_time(text, observed_at=0)["event_date"] == expected.isoformat()
        assert _carried_absolute_day(dict(time_uncertain="carried", quote=text)) == expected
        for carried in [False, True]:
            with engine_context(tmp_path / f"{i}-{carried}", embeddings_enabled=False) as engine:
                refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
                result = json.loads(tools.lcm_compile_evidence(args(refs, QUESTIONS[1][1], "2023-03-22"), engine=engine))
                assert result["state"] == "answer_sufficient"
                assert result["direct_fact"]["value"] == text


def test_ordinal_is_already_normalized():
    assert occurrence_time._quote_date_info("Jan 17th, 2023")[0] == date(2023, 1, 17)
    assert _carried_absolute_day(dict(time_uncertain="carried", quote="Jan 17th, 2023")) == date(2023, 1, 17)


@pytest.mark.parametrize("pair", [
    pytest.param(("2023-02/28", "Jan 17th, 2023"), id="idd00128"),
    pytest.param(("March 12, 2023", "2023-01/30"), id="idd03153"),
    pytest.param(("2023-01/14", "January 16 2023"), id="idd03159"),
])
@pytest.mark.parametrize("carried", [False, True])
def test_issue_cells_fixture_only(tmp_path, pair, carried):
    text = row(pair)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        assert_withheld(engine, refs, "order", QUESTIONS[0][1])
        assert occurrence_time._quote_date_info(text)[1] == REASON
