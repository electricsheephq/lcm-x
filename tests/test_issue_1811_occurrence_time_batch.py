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
