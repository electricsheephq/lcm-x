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
    assert result["reason_code"] == "uncertain_source_observation_time"
    assert result["direct_fact"] is None and result["computation"] is None


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("operation,question", QUESTIONS)
# Regression: mixed-separator date assertions fail at the original base.
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
# Regression: mixed-separator date assertions fail at the original base.
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
# Regression: mixed-separator date assertions fail at the original base.
def test_month_boundary_filter_withheld(tmp_path, carried, month):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        text = row(PAIRS[2])
        refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        assert_withheld(engine, refs, "date_filter", f"Which vacation did I take in {month}?")
        assert occurrence_time._quote_date_info(text)[1] == REASON


@pytest.mark.parametrize("spelling", ["2023-01/10", "2023/01-10"])
# Regression: mixed-separator date assertions fail at the original base.
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
                answered = json.loads(tools.lcm_compile_evidence(args(refs,
                    "When did I take a vacation to Bali?", "2023-03-22"), engine=engine))
                assert answered["state"] == "answer_sufficient"
                assert date.fromisoformat(answered["direct_fact"]["value"].replace("/", "-")) == expected


# Compatibility guard: ordinal parsing already passes at the original base.
def test_ordinal_is_already_normalized():
    assert occurrence_time._quote_date_info("Jan 17th, 2023")[0] == date(2023, 1, 17)
    assert _carried_absolute_day(dict(time_uncertain="carried", quote="Jan 17th, 2023")) == date(2023, 1, 17)


@pytest.mark.parametrize("pair", [
    pytest.param(("2023-02/28", "Jan 17th, 2023"), id="idd00128"),
    pytest.param(("2023/02-28", "Jan 17th, 2023"), id="idd00128-reverse"),
    pytest.param(("March 12, 2023", "2023-01/30"), id="idd03153"),
    pytest.param(("March 12, 2023", "2023/01-30"), id="idd03153-reverse"),
    pytest.param(("2023-01/14", "January 16 2023"), id="idd03159"),
    pytest.param(("2023/01-14", "January 16 2023"), id="idd03159-reverse"),
])
@pytest.mark.parametrize("carried", [False, True])
# Regression: mixed-separator date assertions fail at the original base.
def test_issue_cells_fixture_only(tmp_path, pair, carried):
    text = row(pair)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        assert_withheld(engine, refs, "order", QUESTIONS[0][1])
        assert occurrence_time._quote_date_info(text)[1] == REASON


@pytest.mark.parametrize("mixed", ["2023-01/10", "2023/01-10"])
@pytest.mark.parametrize("other", ["2023-01-12", "January 12, 2023"])
# Compatibility guard: preserve the established non-date/relative/operand behavior.
def test_non_date_carried_fact_keeps_base_parse_mode(tmp_path, mixed, other):
    text = f"Batch dates: {mixed} and {other}. My project status is green."
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text])
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compile_evidence(args(refs), engine=engine))
        assert result["state"] == "answer_sufficient"
        assert result["direct_fact"]["value"] == text
        assert _carried_absolute_day(dict(time_uncertain="carried", quote=text),
                                     date_dependent=False) == date(2023, 1, 12)
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("mixed", ["2023-01/10", "2023/01-10"])
@pytest.mark.parametrize("carried", [False, True])
# Compatibility guard: preserve the established non-date/relative/operand behavior.
def test_single_mixed_day_keeps_relative_verdict(tmp_path, mixed, carried):
    text = f"I took a vacation to Bali on {mixed}; I jogged yesterday and today."
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        result = json.loads(tools.lcm_compile_evidence(args(refs, QUESTIONS[1][1], "2023-03-22"), engine=engine))
        assert result["state"] == "answer_sufficient"
        assert occurrence_time._quote_date_info(text)[1] is None
        assert occurrence_time.resolve_occurrence_time(text, observed_at=0)["event_date"] == "2023-01-10"


@pytest.mark.parametrize("mixed", ["2023-01/10", "2023/01-10"])
# Compatibility guard: preserve the established non-date/relative/operand behavior.
def test_mixed_day_does_not_add_unanchored_interval_operand(tmp_path, mixed):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, ["I visited the Louvre on 2023-01-12.",
                             "I visited the Prado on 2023-01-20.",
                             f"I visited the Uffizi on {mixed}.", "Unrelated synthetic row."],
                     carried_index=3)[:3]
        result = json.loads(tools.lcm_compile_evidence(args(refs,
            "How many days passed between my visit to the Louvre and my visit to the Prado?"), engine=engine))
        assert result["state"] == "computation_sufficient"


@pytest.mark.parametrize("quote,expected", [
    ("Batch date: 2023-01-10.", frozenset({date(2023, 1, 10)})),
    ("Batch date: Jan 10.", frozenset()),
])
# Regression: the pre-fix head returns mutable, uncached legacy days.
def test_carried_legacy_days_uses_bounded_immutable_cache(quote, expected):
    from hermes_lcm.requirements_compiler import _carried_legacy_days, _small_carried_legacy_days

    days = _carried_legacy_days(quote)
    assert isinstance(days, frozenset)
    assert days == expected
    assert _small_carried_legacy_days.cache_info().maxsize == 4096
    hits = _small_carried_legacy_days.cache_info().hits
    assert _carried_legacy_days(quote) is days
    assert _small_carried_legacy_days.cache_info().hits == hits + 1


# Regression: the non-date carried cache must not retain large rows by value.
def test_carried_legacy_days_retains_results_only_for_large_rows():
    import gc
    import tracemalloc
    from hermes_lcm.requirements_compiler import _carried_mixed_dates

    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for i in range(64):
        quote = f"Synthetic carried {i}: 2023-01-10 and yesterday. " + "x" * (256 * 1024)
        assert _carried_mixed_dates({"quote": quote, "content": quote, "time_uncertain": "carried"},
                                    date_dependent=False)
    del quote
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert retained < 2 * 1024 * 1024, retained


@pytest.mark.parametrize("first", ["2023-01/10", "2023/01-10", "2023-01-10"])
@pytest.mark.parametrize("glued", ["X2023-01-12", "build_2023-01-12", "v2023-02-03"])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("operation,question", [
    ("order", "What is the order of events?"),
    ("date_filter", "Which challenge did I finish in January?"),
])
# Compatibility guard: fallback stays unknown at base, pre-fix head and fixed head.
def test_order_event_fallback_stays_unknown(tmp_path, first, glued, carried, operation, question):
    text = f"I finished the PlankChallenge on {first} per ticket {glued}."
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [text, "The launch happened on 2023-01-20.", "Unrelated synthetic row."],
                     carried_index=0 if carried else 2)[:2]
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compile_evidence(args(refs, question, "2023-03-22"), engine=engine))
        assert result["contract"]["operation"] == operation
        assert result["state"] == "unknown"
        assert result["direct_fact"] is None and result["computation"] is None
        assert engine._store.connection.total_changes == before
