"""Round-one read-path regressions: batching and lower-bound time disclosure."""
import json
import sqlite3

import pytest

from hermes_lcm import evidence_pack, tools
from tests.harness import engine_context
from tests.test_issue_1507_carried_time import source  # noqa: F401


def ref(engine, sid):
    return {"exact_ref": f"lcm:{sid}:0-{len(engine._store.get(sid)['content'])}"}


@pytest.mark.parametrize("surface", ["snippets", "answer_ready", "occurrence", "session", "pack"])
def test_one_relation_batch(tmp_path, monkeypatch, surface):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = []
        for i in range(25):
            ids.append(engine._store.append(f"session-{i}", dict(
                role="user", content=f"batchprobe synthetic record {i}.", timestamp=1710936000 + i)))
        engine._store.add_message_relations([
            [(sid, "carried_stamp", None, 0, 1710936000)] for sid in ids])
        statements = []
        connect = sqlite3.connect

        def trace(sql):
            if "message_relations" in sql and sql.lstrip().upper().startswith("SELECT"):
                statements.append(sql)

        def traced_connect(*args, **kwargs):
            conn = connect(*args, **kwargs)
            conn.set_trace_callback(trace)
            return conn

        engine._store.connection.set_trace_callback(trace)
        monkeypatch.setattr(sqlite3, "connect", traced_connect)
        if surface == "pack":
            payload = json.loads(evidence_pack.build_evidence_pack(dict(
                question="What is the latest deadline?", baseline_refs=[ref(engine, sid) for sid in ids]),
                engine=engine))
            assert payload["selection"]["reason_code"] == "uncertain_source_observation_time"
        else:
            payload = json.loads(tools.lcm_recall(dict(
                query="batchprobe", include="verbatim", limit=25,
                detail="snippets" if surface == "snippets" else "answer_ready",
                include_occurrence_time=surface in {"occurrence", "session"}),
                engine=engine, delivery_unit="session" if surface == "session" else "hits"))
            assert len(payload["hits"]) == 25
            assert all(h.get("time_uncertain") == "carried" for h in payload["hits"])
        assert len(statements) <= 2


def test_session_nested_all_causes(source):  # noqa: F811
    engine, row, uncertain = source
    payload = json.loads(tools.lcm_recall(dict(
        query="uncertaintyprobe", include="verbatim", detail="answer_ready",
        include_occurrence_time=True), engine=engine, delivery_unit="session"))
    hit = next(h for h in payload["hits"] if h["store_id"] == row["store_id"])
    for field in ("observation_time", "occurrence_time"):
        assert hit[field].get("time_uncertain") == ("carried" if uncertain else None)


@pytest.mark.parametrize("observed,question_date,included", [
    (1708344000, "2024-03-01", True),
    (1710936000, "2024-03-01", False),
])
def test_as_of_lower_bound(tmp_path, observed, question_date, included):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        sid = engine._store.append("s", dict(role="user", content="Project status is green.", timestamp=observed))
        engine._store.add_message_relations([[(sid, "carried_stamp", None, 0, observed)]])
        payload = json.loads(evidence_pack.build_evidence_pack(dict(
            question="What is the latest project status?", question_date=question_date,
            baseline_refs=[ref(engine, sid)]), engine=engine))
        assert bool(payload["evidence"]) is included
        if included:
            assert payload["evidence"][0]["observation_time"]["time_uncertain"] == "carried"
            assert payload["selection"]["reason_code"] == "uncertain_source_observation_time"
        else:
            assert payload["exclusions"] == [dict(exact_ref=ref(engine, sid)["exact_ref"],
                reason_code="source_observed_after_question_as_of", time_uncertain="carried")]


@pytest.mark.parametrize("quote_date,boundary", [
    ("2024-03-12", "2024-03-01"), ("2024-03-25", "2024-03-19")])
def test_quote_occurrence_cutoff_and_true_index(tmp_path, quote_date, boundary):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = [engine._store.append("s", dict(role="user", timestamp=1708344000,
            content="Project status is green.")) for _ in range(13)]
        sid = engine._store.append("s", dict(role="user", timestamp=1708344000,
            content=f"The deadline is {quote_date}."))
        engine._store.add_message_relations([[(sid, "carried_stamp", None, 0, 1708344000)]])
        payload = json.loads(evidence_pack.build_evidence_pack(dict(
            question="What is the project status?", question_date=boundary,
            baseline_refs=[ref(engine, i) for i in [*ids, sid]]), engine=engine))
        assert payload["status"] == "fallback"
        assert "operands[13]" in payload["rejections"][-1]["reason"]
        assert any(x["exact_ref"] == ref(engine, sid)["exact_ref"]
                   and x.get("time_uncertain") == "carried" for x in payload["exclusions"])


def test_marker_optional_with_time():
    from hermes_lcm.recall_delivery import _without_event_time

    hit = dict(store_id=1, event_time="2024-03-20T12:00:00Z",
               event_time_source="host", time_uncertain="carried")
    assert _without_event_time(hit) == {"store_id": 1}


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("boundary", [None, "2024-02-01", "2024-03-22"])
def test_selection_as_of_review_scenarios(tmp_path, mixed, boundary):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = [engine._store.append(f"s-{i}", dict(role="user", timestamp=1710936000 + i,
            content=f"The project status is green, synthetic record {i}.")) for i in range(12)]
        selected = ids[6:12] if mixed else [ids[2]]
        carried = ids[8] if mixed else ids[2]
        engine._store.add_message_relations([[(carried, "carried_stamp", None, 0, 1710936000)]])
        payload = json.loads(evidence_pack.build_evidence_pack(dict(
            question="What is the latest deadline?", question_date=boundary,
            baseline_refs=[ref(engine, sid) for sid in selected]), engine=engine))
        if boundary == "2024-02-01":
            assert payload["status"] == "fallback"
            assert payload["selection"] is None
            assert len(payload["exclusions"]) == len(selected)
            assert any(x["exact_ref"] == ref(engine, carried)["exact_ref"]
                       and x.get("time_uncertain") == "carried" for x in payload["exclusions"])
        else:
            assert payload["selection"]["status"] == "fallback"
            assert payload["selection"]["reason_code"] == "uncertain_source_observation_time"
            assert payload["selection"]["uncertain_refs"] == [ref(engine, carried)["exact_ref"]]


def test_failed_batch_never_shows_carried_time_as_certain(tmp_path, monkeypatch):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        sid = engine._store.append("s", dict(role="user", content="deadlineprobe status.", timestamp=1710936000))
        engine._store.add_message_relations([[(sid, "carried_stamp", None, 0, 1710936000)]])
        monkeypatch.setattr(tools, "_lcm_recall_event_time_rows", lambda *a, **kw: ({}, "deadline"))
        payload = json.loads(tools.lcm_recall(dict(query="deadlineprobe", include="verbatim",
            detail="answer_ready", include_occurrence_time=True), engine=engine))
        assert payload["hits"]
        assert payload["event_time_unavailable_reason"] == "deadline"
        assert all(not any(k in h for k in ("event_time", "observation_time", "occurrence_time"))
                   for h in payload["hits"])


@pytest.mark.parametrize("unit", ["hits", "session"])
def test_marker_does_not_displace_ordinary_coverage(tmp_path, monkeypatch, unit):
    from hermes_lcm import recall_session

    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = [engine._store.append(f"session-{i}", dict(role="user",
            content=f"capprobe synthetic record {i}. " * 5, timestamp=1710936000 + i))
            for i in range(8)]
        args = dict(query="capprobe", include="verbatim", detail="answer_ready", limit=8)
        caps = (3000, 4000, 5000, 6000, 7000, 8000)

        def read(cap):
            monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
            monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
            return json.loads(tools.lcm_recall(args, engine=engine, delivery_unit=unit))

        ordinary = [read(cap) for cap in caps]
        engine._store.add_message_relations([[(ids[0], "carried_stamp", None, 0, 1710936000)]])
        for cap, before in zip(caps, ordinary):
            after = read(cap)
            old_ids = {h["store_id"] for h in before["hits"]} - {ids[0]}
            assert old_ids <= {h["store_id"] for h in after["hits"]}
            for hit in after["hits"]:
                if hit["store_id"] == ids[0] and hit.get("event_time"):
                    assert hit.get("time_uncertain") == "carried"
            if unit == "session":
                old_hints = {s["session_id"] for s in before["provenance"]["delivery"]["sessions"]
                             if s.get("expand_hint") and s["session_id"] != "session-0"}
                new_hints = {s["session_id"] for s in after["provenance"]["delivery"]["sessions"]
                             if s.get("expand_hint")}
                assert old_hints <= new_hints
