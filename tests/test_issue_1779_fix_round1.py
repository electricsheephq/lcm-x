"""Finite window disclosure, proposal propagation and bounded provenance reads."""
import json
import sqlite3

import pytest

from hermes_lcm import tools
from hermes_lcm.evidence_compiler import SELECTOR_SCHEMA_VERSION
from hermes_lcm.store import MessageStore
from tests.harness import engine_context
from tests.test_issue_1507_carried_time import source  # noqa: F401
from tests.test_issue_1779_carried_time import STAMP, compile_args, plant


def proposal_args(ids, texts, question_date=None):
    args = compile_args(ids, texts, question="What is the project status?", question_date=question_date)
    args.update(mode="proposal", proposal=dict(version=SELECTOR_SCHEMA_VERSION,
        selections=[dict(claim_id=f"claim-{i}", facet="answer", exact_ref=ref, quote=text)
            for i, (ref, text) in enumerate(zip(args["baseline_refs"], texts))], missing_facets=[]))
    args["budgets"].update(max_quote_chars=3000, max_output_tokens=10000)
    return args


@pytest.mark.parametrize("absolute", [False, True])
def test_finite_window_discloses_uncertain_exclusion(tmp_path, absolute):
    texts = ["I took a vacation to Kyoto " + ("on 2023-12-20." if absolute else "3 months ago."),
        "I took a vacation to Oslo on 2024-03-10."]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, texts, True)
        before = engine._store.connection.total_changes
        obj = json.loads(tools.lcm_compile_evidence(compile_args(ids, texts,
            question="How many vacations did I take this year?", question_date="2024-03-22"), engine=engine))
        if absolute:
            assert obj["state"] == "computation_sufficient"
            assert obj["computation"]["result_value"] == 1
        else:
            assert obj["computation"] is None
            assert obj["reason_code"] == "uncertain_source_observation_time"
            assert any(x.get("time_uncertain") == "carried" and x.get("reason_code")
                == "event_outside_temporal_window" for x in obj.get("exclusions", []))
        assert engine._store.connection.total_changes == before


def test_proposal_preserves_cutoff_exclusion(source):  # noqa: F811
    engine, row, uncertain = source
    obj = json.loads(tools.lcm_compile_evidence(proposal_args([row["store_id"]],
        [row["content"]], "2024-03-19"), engine=engine))
    if not uncertain:
        assert "exclusions" not in obj
        assert all(x["reason_code"] == "claim_grounding_failed" for x in obj["rejections"])
        return
    assert any(x.get("reason_code") == "source_observed_after_question_as_of"
        and x.get("time_uncertain") == ("carried" if uncertain else None)
        for x in obj.get("exclusions", []))
    assert any(x["reason_code"] == "source_observed_after_question_as_of" for x in obj["rejections"])


@pytest.mark.parametrize("count", [1, 10, 40])
def test_proposal_uses_one_provenance_batch(tmp_path, count):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = [f"My project status is amber{i}." for i in range(count)]
        ids = plant(engine, texts, True)
        sql = []
        engine._store.connection.set_trace_callback(lambda s: sql.append(s) if "message_relations" in s else None)
        try:
            obj = json.loads(tools.lcm_compile_evidence(proposal_args(ids, texts), engine=engine))
        finally:
            engine._store.connection.set_trace_callback(None)
        assert obj["metrics"]["selected_claims"] == count
        assert 1 <= len(sql) <= 2


def test_relation_existence_cache_is_positive_and_per_connection(tmp_path):
    store = MessageStore(str(tmp_path / "cache.db"))
    conn = sqlite3.connect(tmp_path / "cache.db")
    sql = []
    conn.set_trace_callback(lambda s: sql.append(s) if "sqlite_master" in s else None)
    try:
        assert store.get_carried_stamp_ids([1], connection=conn) == set()
        assert len(sql) == 1
        store.add_message_relations([[(1, "carried_stamp", None, 0, STAMP)]])
        assert store.get_carried_stamp_ids([1], connection=conn) == {1}
        assert len(sql) == 2
        assert store.get_carried_stamp_ids([1], connection=conn) == {1}
        assert len(sql) == 3  # private caller connections are intentionally not retained
        own_sql = []
        store.connection.set_trace_callback(lambda s: own_sql.append(s) if "sqlite_master" in s else None)
        assert store.get_carried_stamp_ids([1]) == {1}
        assert store.get_carried_stamp_ids([1]) == {1}
        assert own_sql == []  # writer already cached its owned connection
    finally:
        conn.close()
        store.close()
