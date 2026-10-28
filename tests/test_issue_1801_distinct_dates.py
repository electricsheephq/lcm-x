"""E7: every named date participates in a carried quote's ISO twin."""
import json

import pytest

from hermes_lcm import tools
from hermes_lcm.evidence_compiler import compile_preanswer_evidence
from tests.harness import engine_context
from tests.test_issue_1801_carried_followups import REASON, args, plant
from tests.test_issue_1801_twin_parity import QUESTIONS, normalize

PAIRS = [
    ("January 1, 2023", "February 10, 2023"),
    ("Jan 1, 2023", "Feb 10, 2023"),
    ("January 1 2023", "February 10 2023"),
    ("Jan 1st, 2023", "Feb 10th, 2023"),
]


def distinct_quote(first, second):
    return f"I booked the vacation on {first}. I took a vacation to Bali on {second}."


def exercise(tmp_path, first, second, operation, month):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, [distinct_quote(first, second),
            "I took a vacation to Kyoto on 2023-03-05."])
        question = QUESTIONS[operation].format(month=month)
        request = args(refs, question, "2023-03-22")
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compile_evidence(request, engine=engine))
        hook = compile_preanswer_evidence(question, engine=engine,
            question_date=request["question_date"], baseline_refs=refs, enabled=True,
            render_baseline_context=True, budgets={"max_retrieval_calls": 0})
        assert engine._store.connection.total_changes == before
        return result, hook


def assert_withheld(result):
    assert result["state"] == "unknown"
    assert result["reason_code"] == REASON
    assert result["direct_fact"] is None and result["computation"] is None
    assert any(x["reason_code"] == REASON and x["time_uncertain"] == "carried"
               for x in result.get("exclusions", []))


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("operation", QUESTIONS)
@pytest.mark.parametrize("month", ["February", "March"])
def test_distinct_named_dates_match_full_iso_twin(tmp_path, pair, operation, month):
    iso = exercise(tmp_path / "iso", "2023-01-01", "2023-02-10", operation, month)
    named = exercise(tmp_path / "named", *pair, operation, month)
    for iso_result, named_result in zip(iso, named):
        assert_withheld(iso_result)
        assert_withheld(named_result)
        normalized = normalize(normalize(named_result, pair[0], "2023-01-01"), pair[1])
        expected = normalize(iso_result, "")
        assert normalized == expected
    assert named[1]["context"] is None


def test_exact_thread_4343710383_january_vacation(tmp_path):
    result, hook = exercise(tmp_path, *PAIRS[0], "date_filter", "January")
    assert_withheld(result)
    assert_withheld(hook)
    assert hook["context"] is None
