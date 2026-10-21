"""Carried stamps stay lower bounds on compiler and computation surfaces."""
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid
from unittest.mock import patch

import pytest

from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.evidence_compiler import compile_preanswer_evidence
from tests.harness import engine_context
from tests.test_issue_1507_carried_time import source  # noqa: F401

STAMP = 1710936000.0
REASON = "uncertain_source_observation_time"
REL = ["The PlankChallenge happened yesterday.", "The Launch happened 3 days ago."]
ABS = ["The PlankChallenge happened on 2024-03-19.", "The Launch happened on 2024-03-17."]
QUESTION = "How many days between the PlankChallenge and launch?"
ORDER = "What is the order of events?"


def plant(engine, texts, carried=False, session=None):
    ids = [engine._store.append(session or str(i), dict(role="user", content=t,
        timestamp=STAMP, source="cli", conversation_id="conversation-a"))
        for i, t in enumerate(texts)]
    if carried:
        engine._store.add_message_relations([[(ids[0], "carried_stamp", None, 0, STAMP)]])
    return ids


def compute_args(ids, texts, question=QUESTION, question_date=None):
    return dict(question=question, question_date=question_date, evidence_complete=True,
        operands=[dict(store_id=sid, span_start=0, span_end=len(text), quote=text,
            label=label, occurrence_time=dict(event_time_source=("explicit" if "2024" in text
                else "relative_to_session"), event_date=day))
            for sid, text, label, day in zip(ids, texts, ["PlankChallenge", "launch"],
                ["2024-03-19", "2024-03-17"])])


def compile_args(ids, texts, question=ORDER, question_date=None):
    return dict(mode="auto", question=question, question_date=question_date,
        baseline_refs=[f"lcm:{sid}:0-{len(text)}" for sid, text in zip(ids, texts)],
        budgets={"max_retrieval_calls": 0})


@pytest.mark.parametrize("question_date", [None, "2024-03-19", "2024-03-22"])
def test_compile_disclosure_for_all_marker_causes(source, question_date):  # noqa: F811
    engine, row, uncertain = source
    ref = f"lcm:{row['store_id']}:0-{len(row['content'])}"
    obj = json.loads(tools.lcm_compile_evidence(dict(mode="auto", question="What is the project status?",
        baseline_refs=[ref], question_date=question_date, budgets={"max_retrieval_calls": 0}), engine=engine))
    if question_date == "2024-03-19" and uncertain:
        assert any(x.get("exact_ref") == ref and x.get("time_uncertain") == "carried"
            and x.get("reason_code") == "source_observed_after_question_as_of"
            for x in obj.get("exclusions", []))
    elif question_date != "2024-03-19":
        marker = obj.get("direct_fact", {}).get("time_uncertain")
        assert marker == ("carried" if uncertain else None)
        hook = compile_preanswer_evidence("What is the project status?", engine=engine,
            baseline_refs=[ref], question_date=question_date, enabled=True, render_baseline_context=True,
            budgets={"max_retrieval_calls": 0})
        assert hook["evidence"][0].get("time_uncertain") == ("carried" if uncertain else None)
        assert ("time_uncertain: carried" in hook["context"]) is uncertain


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("question_date", [None, "2024-03-22"])
def test_compile_order_and_compute(tmp_path, monkeypatch, carried, absolute, question_date):
    monkeypatch.setattr("time.time", lambda: 1800000000.0)
    texts = ABS if absolute else REL
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, texts, carried)
        before = engine._store.connection.total_changes
        computed = json.loads(tools.lcm_compute(compute_args(ids, texts, question_date=question_date), engine=engine))
        compiled = json.loads(tools.lcm_compile_evidence(compile_args(ids, texts, question_date=question_date), engine=engine))
        blocked = carried and not absolute
        assert (computed["status"] == "computed") is not blocked
        assert (compiled["state"] == "computation_sufficient") is not blocked
        if blocked:
            assert computed["reason"] == REASON
            assert compiled["reason_code"] == REASON
            assert compiled["computation"] is None
        assert engine._store.connection.total_changes == before


def test_latest_observation_fallback_is_withheld(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = ["My project status is green."]
        ids = plant(engine, texts, True)
        obj = json.loads(tools.lcm_compile_evidence(compile_args(ids, texts,
            question="What is my latest project status?"), engine=engine))
        assert obj["reason_code"] == REASON
        assert obj.get("direct_fact") is None


def retrieve_finish(engine, ids, texts, question=QUESTION, question_date=None):
    def call(**args):
        return json.loads(engine.handle_tool_call("lcm_retrieve", args))
    identity = dict(intent_type="event memory", operation="date_interval", subject_key="user:self",
        predicate_key="event.happened", scope_key="personal", conversation_id="conversation-a",
        time_mode="none", policy_version="adaptive-v1")
    start = call(action="start", question=question, question_date=question_date, identity=identity,
        requirements=[dict(slot_id="events", description="events", minimum_refs=len(ids))])
    rid = start["retrieval_id"]
    refs = []
    for sid in ids:
        search = call(action="search", retrieval_id=rid, missing_slot="events", tool="lcm_expand",
            tool_args={"store_id": sid}, **({"resolved_slots": [dict(slot_id="events", evidence_refs=refs)]} if refs else {}))
        refs = [search["evidence"][0]["citation"]]
    return call(action="finish", retrieval_id=rid,
        resolved_slots=[dict(slot_id="events", evidence_refs=refs)],
        selected_refs=[f"lcm:{sid}:0-{len(text)}" for sid, text in zip(ids, texts)],
        computation=dict(operands=compute_args(ids, texts, question, question_date)["operands"]))


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_retrieve_cannot_publish_uncertain_computation(tmp_path, carried, absolute):
    texts = ABS if absolute else REL
    with patch("time.time", return_value=1800000000.0):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), adaptive_retrieval_enabled=True))
        engine.on_session_start("session-a", conversation_id="conversation-a")
        try:
            ids = plant(engine, texts, carried, "session-a")
            obj = retrieve_finish(engine, ids, texts)
            blocked = carried and not absolute
            assert (obj["computation"]["status"] == "computed") is not blocked
            persistence = obj["query_view"]["persistence"]["status"]
            assert persistence == ("skipped" if blocked else "published")
            if blocked:
                assert obj["computation"]["reason"] == REASON
                assert obj["status"] == "fallback"
        finally:
            engine.shutdown()


# Full serialized output at ba88a3cd2d36ab266fb159742a6b9834151e3f43; clocks/UUID fixed.
BASELINE_HASHES = {'compile-relative-undated': '6c7536e3ca28a29fafc998e2ecb6f81fe81187e7eb574f207c3dbf3710938a72', 'compile-relative-dated': '4724d0ab7d5e18fd4494e42b4e520179f24a69458bd8d869bb79dda678628660', 'compile-absolute-undated': 'c00ce98742112872e376c552e4184f97a55ce9ba1ff5cb9bc70b923e446efdcd', 'compile-absolute-dated': '17b416a90fea71093826654832a11fcaec5e7d91fbf7f5d7b2b3cd6fb539e15a', 'compute-relative-undated': '17f1b64123ee9e4df01f6ad9ac7e461d85120b9fc33588b4553ba6b69012903f', 'compute-relative-dated': '6907e06a5ec09e51f0f6cad9e5476c142e0cb07bf651baaa4dd0822e13286484', 'compute-absolute-undated': 'c014a8969cf463016fe30bcdf89a68ac27e9526bd725d0276d8d191d4b922a73', 'compute-absolute-dated': '68f1af425faa382a7ae3bdb4a233992be012f0066bec99f710f664b484939a43', 'retrieve-relative-undated': 'bad8cf9da7ae0c0c095d81c675978daaf2bd33cba6a4c31fbf2e3b003d70f685', 'retrieve-relative-dated': '4504ffcbd542ae3f83ed3fc584df3657b4a1bc1c99fec56840999aa656ceec4e', 'retrieve-absolute-undated': '1900741cf8c6c5559dabca2211ac34266b2ca814961605996d4b264cfe4527bc', 'retrieve-absolute-dated': 'cb2d926676c8e67b5d726d27deac146dd24e2f93bfdaf047059f703a03a13121'}


@pytest.mark.parametrize("shape", list(BASELINE_HASHES))
def test_ordinary_serialization_matches_pinned_base(tmp_path, shape):
    surface, kind, dated = shape.split("-")
    texts = ABS if kind == "absolute" else REL
    question_date = "2024-03-22" if dated == "dated" else None
    with patch("time.time", return_value=1800000000.0), patch("time.perf_counter", return_value=100.0), patch("uuid.uuid4", return_value=uuid.UUID(int=1)):
        if surface == "retrieve":
            engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), adaptive_retrieval_enabled=True))
            engine.on_session_start("session-a", conversation_id="conversation-a")
            try:
                ids = plant(engine, texts, session="session-a")
                raw = json.dumps(retrieve_finish(engine, ids, texts, question_date=question_date))
            finally:
                engine.shutdown()
        else:
            with engine_context(tmp_path, embeddings_enabled=False) as engine:
                ids = plant(engine, texts)
                raw = (tools.lcm_compute(compute_args(ids, texts, question_date=question_date), engine=engine)
                    if surface == "compute" else tools.lcm_compile_evidence(compile_args(ids, texts, question_date=question_date), engine=engine))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    assert digest == BASELINE_HASHES[shape]


@pytest.mark.parametrize("absolute", [False, True])
def test_finite_enumeration_respects_carried_dates(tmp_path, absolute):
    texts = ["I took a vacation to Bali " + ("on 2024-03-19." if absolute else "yesterday.")]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, texts, True)
        obj = json.loads(tools.lcm_compile_evidence(compile_args(ids, texts,
            question="How many vacations did I take this year?", question_date="2024-03-22"), engine=engine))
        if absolute:
            assert obj["state"] == "computation_sufficient"
            assert obj["evidence"][0]["time_uncertain"] == "carried"
        else:
            assert obj["reason_code"] == REASON
            assert obj["computation"] is None


def test_real_preanswer_hook_discloses_time(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hermes_lcm.issue1779_plugin", Path(__file__).parents[1] / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    plugin.__package__ = "hermes_lcm"
    spec.loader.exec_module(plugin)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = ["My project status is green."]
        ids = plant(engine, texts, True)
        engine._config.preanswer_evidence_enabled = True
        engine._config.preanswer_evidence_mode = "requirements_v1"
        refs = compile_args(ids, texts)["baseline_refs"]
        monkeypatch.setattr(plugin, "_answer_ready_baseline", lambda *_: refs)
        obj = plugin._pre_llm_context(engine, {"user_message": "What is the project status?"})
        assert "time_uncertain: carried" in (obj or {}).get("context", "")


@pytest.mark.parametrize("count", [1, 10, 40])
def test_relation_reads_are_batched(tmp_path, count):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = [f"My project status is green {i}." for i in range(count)]
        ids = plant(engine, texts, True)
        statements = []
        engine._store.connection.set_trace_callback(lambda sql: statements.append(1) if "message_relations" in sql else None)
        try:
            tools.lcm_compile_evidence(compile_args(ids, texts, question="What is the project status?"), engine=engine)
        finally:
            engine._store.connection.set_trace_callback(None)
        assert 1 <= len(statements) <= 6
