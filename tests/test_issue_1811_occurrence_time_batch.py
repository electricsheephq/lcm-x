"""One grammar and conservative occurrence metadata across read paths."""
import json
from pathlib import Path

import pytest

from hermes_lcm import tools
from hermes_lcm.occurrence_time import resolve_occurrence_time
from tests.harness import engine_context
from tests.test_issue_1801_carried_followups import args, plant

FORMS = ["2023-01-10", "January 10, 2023", "Jan 10, 2023",
         "January 10 2023", "Jan 10th, 2023"]
PHRASES = ["yesterday", "today", "last monday", "3 days ago", "2 weeks ago", "1 month ago",
           "last week", "last month", "last year", "last night", "last weekend",
           "2 years ago", "a day ago", "an year ago", "one week ago", "two days ago",
           "three weeks ago", "four months ago", "five years ago", "six days ago",
           "seven days ago", "eight days ago", "nine days ago", "ten days ago",
           "eleven days ago", "twelve days ago", "a few days ago", "a couple of weeks ago",
           "the day before", "the day before yesterday", "the other day",
           "earlier this week", "earlier this month", "earlier this year"]
QUESTIONS = ["Which vacation did I take in January?", "How many vacations did I take in January?",
             "What is the order of events in January?", "What is the order of events?"]


def mixed(form, phrase):
    return f"I reviewed the plan on {form}. I took a vacation to Bali {phrase}."


@pytest.mark.parametrize("phrase", PHRASES)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("carried", [False, True])
def test_shared_grammar_mixed_withheld_and_recorded(tmp_path, phrase, form, carried):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [mixed(form, phrase), "I took a vacation to Kyoto on 2023-01-12.",
                             "Unrelated synthetic row."], carried_index=0 if carried else 2)[:2]
        before = engine._store.connection.total_changes
        for question in QUESTIONS:
            for question_date in ([None, "2023-03-22"] if question == QUESTIONS[-1] else ["2023-03-22"]):
                result = json.loads(tools.lcm_compile_evidence(args(refs, question, question_date), engine=engine))
                assert result["direct_fact"] is None and result["computation"] is None
                assert result["state"] == "unknown"
                assert any(x["exact_ref"].startswith(refs[0].split(":0-")[0] + ":")
                           for x in result.get("exclusions", []))
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("form", ["February 10, 2023", "2023-02-10"])
def test_iso_and_named_conflicting_dates_withheld(tmp_path, carried, form):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [f"I booked the trip on 2023-01-01. The vacation to Bali was on {form}.",
                             "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
        result = json.loads(tools.lcm_compile_evidence(args(refs, QUESTIONS[0], "2023-03-22"), engine=engine))
        assert result["direct_fact"] is None
        assert result.get("exclusions")


@pytest.mark.parametrize("form", ["2023-02-10", "Feb 10th, 2023", "February 10th 2023",
                                 "Friday, February 10, 2023", "Friday Feb 10th, 2023"])
@pytest.mark.parametrize("question_date", [None, "2023-03-22"])
@pytest.mark.parametrize("carried", [False, True])
def test_order_label_shared_grammar_parity(tmp_path, form, question_date, carried):
    from hermes_lcm.answer_contract import compile_answer_contract
    from hermes_lcm.requirements_compiler import _normalize_ref, _ordered_event_candidate

    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [f"I took a vacation to Bali on {form}.", "Unrelated synthetic row."],
                     carried_index=0 if carried else 1)
        source = _normalize_ref(refs[0], engine=engine, origin="baseline")
        if carried:
            source["time_uncertain"] = "carried"
        contract = compile_answer_contract("What is the order of events?", question_as_of=question_date).contract
        candidate = _ordered_event_candidate(source, contract)
        if not carried and "th" in form:
            # Ordinary ordinal-only dates had no day at base; never newly answer.
            assert candidate is None
        else:
            assert candidate["label"] == "Bali"
        result = json.loads(tools.lcm_compile_evidence(args(refs, "What is the order of events?", question_date), engine=engine))
        assert "Feb" not in str(result.get("computation"))


def test_base_unknown_identity_and_every_changed_object_strictly_shrinks():
    # Recorded on the pinned base before implementation; includes every table quote.
    cases = json.loads(Path(__file__).with_name("issue_1811_base_occurrences.json").read_text())
    changed = 0
    for case in cases:
        base = case["base"]
        head = resolve_occurrence_time(case["quote"], observed_at=123, session_date=case["session_date"])
        if base["event_time_source"] == "unknown":
            assert head == base, case["quote"]
        if head != base:
            changed += 1
            assert head["event_time_source"] == "unknown", case["quote"]
            assert len(json.dumps(head)) < len(json.dumps(base)), case["quote"]
    assert changed


@pytest.mark.parametrize("phrase", ["yesterday", "two days ago", "last week", "the day before yesterday"])
def test_resolver_mixed_precedence_and_relative_ambiguity(phrase):
    result = resolve_occurrence_time(mixed(FORMS[0], phrase), observed_at=123, session_date="2023-03-22")
    assert result["reason"] == "mixed_absolute_and_relative_dates"
    result = resolve_occurrence_time(f"I left yesterday and returned {phrase}.", observed_at=123, session_date="2023-03-22")
    assert result["reason"] == "ambiguous_relative_expression"


def test_resolver_month_name_conflict_and_support():
    result = resolve_occurrence_time("2023-01-01 and February 10, 2023", observed_at=123)
    assert result["reason"] == "ambiguous_multiple_explicit_dates"
    result = resolve_occurrence_time("2023-01-01 and January 1, 2023", observed_at=123)
    assert result["event_date"] == "2023-01-01"
    assert result["support"] == dict(quote="2023-01-01", char_start=0, char_end=10)


NOISE_ROWS = [
    "I reviewed the plan on 2023-01-10 and took a vacation yesterday.",
    "I reviewed the plan on 2023-01-10 and again on 2023-01-12.",
    "We met on January 3, 2023 and again last week.",
]
NON_DATE_CASES = [
    ("What is the project status?", "My project status is green."),
    ("What is my current project status?", "My project status is green."),
    ("What is my favorite color?", "My favorite color is blue."),
    ("How many engineers do I directly lead on the platform team?", "I directly lead 7 engineers on the platform team."),
    ("How much did I spend in total on lunch and dinner?", "I spent $12 on lunch and $18 on dinner."),
]


def non_date_matrix_result(tmp_path, monkeypatch, carried, noise_index, case_index, partial):
    from hermes_lcm import requirements_compiler

    monkeypatch.setattr(requirements_compiler.time, "perf_counter", lambda: 1.0)
    question, fact = NON_DATE_CASES[case_index]
    noise = NOISE_ROWS[noise_index]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        if partial:
            row = noise + " " + fact
            refs = plant(engine, [row, "Unrelated synthetic row."], carried_index=0 if carried else 1)[:1]
            start = len(noise) + 1
            refs = [f"lcm:1:{start}-{len(row)}"]
        else:
            refs = plant(engine, [fact, noise, "Unrelated synthetic row."], carried_index=1 if carried else 2)[:2]
        before = engine._store.connection.total_changes
        raw = tools.lcm_compile_evidence(args(refs, question), engine=engine)
        assert engine._store.connection.total_changes == before
        result = json.loads(raw)
        assert result["contract"]["temporal_window"] is None
        assert result["contract"]["operation"] not in {"date_filter", "order", "date_interval"}
        return raw


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("case_index", range(5))
@pytest.mark.parametrize("noise_index", range(3))
@pytest.mark.parametrize("carried", [False, True])
def test_non_date_contract_base_identity(tmp_path, monkeypatch, carried, noise_index, case_index, partial):
    import hashlib

    expected = json.loads(Path(__file__).with_name("issue_1811_non_date_base_hashes.json").read_text())
    key = f"{int(carried)}-{noise_index}-{case_index}-{int(partial)}"
    raw = non_date_matrix_result(tmp_path, monkeypatch, carried, noise_index, case_index, partial)
    assert hashlib.sha256(raw.encode()).hexdigest() == expected[key]


@pytest.mark.parametrize("carried", [False, True])
def test_ordinary_partial_date_filter_matches_own_row(tmp_path, monkeypatch, carried):
    from hermes_lcm import requirements_compiler

    monkeypatch.setattr(requirements_compiler.time, "perf_counter", lambda: 1.0)
    results = []
    fact = "I took a vacation to Bali yesterday."
    for partial in [False, True]:
        noise = "I reviewed the plan on 2023-01-10. " if partial else ""
        row = noise + fact
        with engine_context(tmp_path / str(partial), embeddings_enabled=False) as engine:
            plant(engine, [row, "Unrelated synthetic row."], carried_index=0 if carried else 1)
            engine._session_occurrence_dates = {"0": "2023-01-12"}
            ref = f"lcm:1:{len(noise)}-{len(row)}"
            result = json.loads(tools.lcm_compile_evidence(args([ref], QUESTIONS[0], "2023-03-22"), engine=engine))
            results.append(result)
    own, partial = results
    if carried:
        assert partial["state"] == "unknown"
        assert partial["reason_code"] == "uncertain_source_observation_time"
    else:
        assert partial["state"] == own["state"] == "answer_sufficient"
        assert partial["reason_code"] == own["reason_code"]
        assert partial["direct_fact"]["value"] == own["direct_fact"]["value"]
        assert partial.get("exclusions", []) == own.get("exclusions", [])


@pytest.mark.parametrize("form", [f"{day}{dot}, Feb 10, 2023" for day in
    ["Mon", "Tue", "Tues", "Wed", "Thu", "Thur", "Thurs", "Fri", "Sat", "Sun"] for dot in ["", "."]])
@pytest.mark.parametrize("question_date", [None, "2023-03-22"])
@pytest.mark.parametrize("carried", [False, True])
def test_abbreviated_weekday_order_label(tmp_path, form, question_date, carried):
    from hermes_lcm.answer_contract import compile_answer_contract
    from hermes_lcm.requirements_compiler import _normalize_ref, _ordered_event_candidate

    candidates = []
    for i, spelling in enumerate(["2023-02-10", form]):
        with engine_context(tmp_path / str(i), embeddings_enabled=False) as engine:
            refs = plant(engine, [f"The Kyoto Festival happened on {spelling}.", "Unrelated synthetic row."],
                         carried_index=0 if carried else 1)
            source = _normalize_ref(refs[0], engine=engine, origin="baseline")
            if carried:
                source["time_uncertain"] = "carried"
            contract = compile_answer_contract("What is the order of events?", question_as_of=question_date).contract
            candidates.append(_ordered_event_candidate(source, contract))
    assert candidates[0]["label"] == candidates[1]["label"]
    assert candidates[0]["date"] == candidates[1]["date"] == "2023-02-10"


@pytest.mark.parametrize("carried", [False, True])
def test_large_quote_cache_retains_results_only(carried):
    import gc
    import tracemalloc
    from hermes_lcm.requirements_compiler import _carried_mixed_dates

    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for i in range(64):
        quote = f"Synthetic carried={carried} {i}: 2023-01-10 and yesterday. " + "x" * (256 * 1024)
        source = {"quote": quote, "content": quote}
        if carried:
            source["time_uncertain"] = "carried"
        assert _carried_mixed_dates(source)
    del quote, source
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert retained < 2 * 1024 * 1024, retained


def test_cached_date_info_equals_uncached():
    from hermes_lcm import occurrence_time

    for i in range(3000):
        quote = f"Synthetic {i}: " + ["2023-01-10", "January 3, 2023 and last week",
            "2023-01-10 and 2023-01-12", "yesterday", "no date"][i % 5]
        if i % 7 == 0:
            quote += "x" * 1300
        assert occurrence_time._quote_date_info(quote) == occurrence_time._parse_quote_date_info(quote)
