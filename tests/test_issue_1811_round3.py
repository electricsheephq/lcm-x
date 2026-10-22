"""Final-round multi-relative exclusion, bounded order labels and compute parity."""

import json

import pytest

from hermes_lcm import evidence_tools, occurrence_time, requirements_compiler, tools
from tests.harness import engine_context

STAMP = 1676462400.0


@pytest.mark.parametrize("operation,question", [
    ("order", "What is the order of the museums I visited, earliest to latest?"),
    ("date_filter", "Which museum did I visit in January?"),
])
@pytest.mark.parametrize("qd", [None, "2023-02-20"])
def test_louvre_prado_uffizi_multi_relative_refusal(tmp_path, monkeypatch, operation, question, qd):
    monkeypatch.setattr(requirements_compiler.time, "perf_counter", lambda: 1.0)
    rows = ["I visited the Louvre museum last week and again yesterday.",
            "I visited the Prado museum on 2023-01-15.",
            "I visited the Uffizi museum on 2023-02-08."]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = [engine._store.append(str(i), dict(role="user", content=t, timestamp=STAMP, source="cli"))
               for i, t in enumerate(rows)]
        refs = [f"lcm:{sid}:0-{len(t)}" for sid, t in zip(ids, rows)]
        result = json.loads(tools.lcm_compile_evidence(dict(mode="auto", question=question,
                            question_date=qd, baseline_refs=refs, budgets={"max_retrieval_calls": 0}), engine=engine))
    assert result["contract"]["operation"] == operation
    assert result["state"] == "unknown"
    assert result["reason_code"] == "uncertain_source_observation_time"
    assert any(e["exact_ref"] == refs[0] and e["reason_code"] == result["reason_code"]
               for e in result["exclusions"])


@pytest.mark.parametrize("qd", [None, "2023-03-22"])
@pytest.mark.parametrize("case", range(3))
def test_lcm_compute_non_date_sum_base_parity(tmp_path, monkeypatch, qd, case):
    monkeypatch.setattr(evidence_tools.time, "perf_counter", lambda: 1.0)
    quotes = ["On 2023-01-10 I paid $140 for the tent and yesterday $35 for the stove.",
              "I paid $140 for the tent on 2023-01-10.",
              "I paid $35 for the stove on 2023-01-12 and again last week."]
    selected = [0 if case == 0 else 1, 2]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = [engine._store.append(str(i), dict(role="user", content=t, timestamp=STAMP, source="cli"))
               for i, t in enumerate(quotes)]
        operands = []
        for r, label, value, day in zip(selected, ["tent", "stove"], [140, 35], ["2023-01-10", "2023-01-12"]):
            operand = dict(store_id=ids[r], span_start=0, span_end=len(quotes[r]), quote=quotes[r],
                           label=label, value=value, unit="usd")
            if case != 2:
                operand["occurrence_time"] = dict(event_time_source="explicit", event_date=day)
            operands.append(operand)
        result = json.loads(tools.lcm_compute(dict(question="How much did I pay in total for the tent and the stove?",
                            question_date=qd, evidence_complete=True, operands=operands), engine=engine))
    assert result["status"] == "computed"
    assert result["trace"]["result"] == 175


def test_long_order_label_name_is_bounded_and_reused(monkeypatch):
    quote = "I visited the Louvre on Fri, January 10th 2023." + " We packed bags." * 600
    name = occurrence_time._order_label_name(quote)
    assert name == "We"
    assert max(len(k) for k in occurrence_time._large_order_label_names) == 32
    assert all(v is None or len(v) <= 256 for v in occurrence_time._large_order_label_names.values())
    def forbidden(_):
        raise AssertionError("warm large label repeated the strip")
    monkeypatch.setattr(occurrence_time, "_order_label_text", forbidden)
    assert occurrence_time._order_label_name(quote) == name
