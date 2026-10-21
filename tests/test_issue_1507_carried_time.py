"""Read-only carried-time disclosure, including the marker's ambiguous causes."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import sqlite3
import time

import pytest

from hermes_lcm import evidence_pack, tools
from tests.harness import engine_context

STAMP = 1_710_936_000.0  # 2024-03-20T12:00:00Z, after the historical cutoff.
TEXT = "uncertaintyprobe project status is green."


def _hashes(store):
    return tuple(hashlib.sha256(json.dumps(store.connection.execute(
        f"SELECT * FROM {table} ORDER BY 1"
    ).fetchall()).encode()).hexdigest() for table in ("messages", "message_relations"))


@pytest.fixture(params=["carried", "in-place-edit", "undo-resend", "ordinary"])
def source(request, tmp_path):
    shape = request.param
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                       [("parent", None, "compression"), ("child", "parent", None)])
    q = dict(role="assistant", content="earlier reply", timestamp=STAMP - 60)
    p = dict(role="user", content="original request", timestamp=STAMP)
    r = dict(role="user", content=TEXT, timestamp=STAMP)
    with ExitStack() as stack:
        engine = stack.enter_context(engine_context(
            tmp_path, window=128000, session_id="parent", conversation_id="conv", embeddings_enabled=False))
        engine.ingest(deepcopy([q, p]))
        if shape == "carried":
            engine.on_session_start("child", platform="cli", context_length=128000,
                                    conversation_id="conv", boundary_reason="compression")
        if shape == "ordinary":
            r["timestamp"] += 86400
            engine.ingest(deepcopy([q, p, r]))
        else:
            engine.ingest(deepcopy([q, r]))
            if shape == "undo-resend":
                stack.close()
                engine = stack.enter_context(engine_context(
                    tmp_path, window=128000, session_id="parent", conversation_id="conv", embeddings_enabled=False))
                engine.ingest(deepcopy([q, p, {**r, "timestamp": STAMP + 86400}]))
        rows = engine._store.get_session_messages(engine.current_session_id)
        (row,) = [row for row in rows if row["content"] == TEXT]
        relations = engine._store.get_message_relations([row["store_id"]], "carried_stamp")
        assert bool(relations) is (shape != "ordinary")
        assert row["observed_at"] == STAMP + (86400 if shape == "ordinary" else 0)
        before = _hashes(engine._store)
        yield engine, row, shape != "ordinary"
        assert _hashes(engine._store) == before


@pytest.mark.parametrize("mode", ["snippets", "answer_ready", "non-strict", "unhydrated", "session"])
def test_recall_discloses_without_rewriting(source, monkeypatch, mode):
    engine, row, uncertain = source
    if mode == "non-strict":
        engine._config.recall_reference_strict = False
    if mode == "unhydrated":
        monkeypatch.setattr(tools, "_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT", 0)
        engine._config.recall_reference_strict = False
    args = dict(query="uncertaintyprobe", include="verbatim", limit=1,
                detail="snippets" if mode == "snippets" else "answer_ready",
                include_occurrence_time=mode != "snippets")
    payload = json.loads(tools.lcm_recall(args, engine=engine,
                         delivery_unit="session" if mode == "session" else "hits"))
    (hit,) = [hit for hit in payload["hits"] if hit["store_id"] == row["store_id"]]
    assert hit.get("time_uncertain") == ("carried" if uncertain else None)
    assert hit["event_time"] == ("2024-03-21T12:00:00Z" if not uncertain else "2024-03-20T12:00:00Z")
    if mode not in {"snippets", "session"}:
        for field in ("observation_time", "occurrence_time"):
            assert hit[field].get("time_uncertain") == ("carried" if uncertain else None)


def test_event_time_helper_discloses_relation(source):
    engine, row, uncertain = source
    rows, reason = tools._lcm_recall_event_time_rows(
        engine, [row["store_id"]], deadline=time.monotonic() + 5)
    assert reason is None
    assert tools._lcm_recall_event_time(rows[row["store_id"]]).get("time_uncertain") == (
        "carried" if uncertain else None)


@pytest.mark.parametrize("question_date", [None, "2024-03-19", "2024-03-22"])
def test_evidence_as_of_retains_uncertain_rows(source, question_date):
    engine, row, uncertain = source
    payload = json.loads(evidence_pack.build_evidence_pack(dict(
        question="What is the project status?", question_date=question_date,
        baseline_refs=[dict(exact_ref=f"lcm:{row['store_id']}:0-{len(TEXT)}")],
    ), engine=engine))
    if question_date == "2024-03-19" and not uncertain:
        assert payload["evidence"] == []
        assert payload["exclusions"][0]["reason_code"] == "source_observed_after_question_as_of"
        return
    (item,) = payload["evidence"]
    for field in ("observation_time", "occurrence_time"):
        assert item[field].get("time_uncertain") == ("carried" if uncertain else None)
    assert item["observation_time"]["observed_at"] == row["observed_at"]
    assert payload["exclusions"] == []
    if uncertain and question_date:
        assert payload["computation"] is None
        assert payload["computation_fallback_reason"] == "uncertain_source_observation_time"


def test_observation_time_helper_discloses(source):
    engine, row, uncertain = source
    # Resolve through the authoritative relation read used by evidence packs.
    candidate, error = evidence_pack._resolve_candidate(
        dict(exact_ref=f"lcm:{row['store_id']}:0-{len(TEXT)}"), engine=engine,
        budgets=evidence_pack._parse_budgets(None), plan=None)
    assert error is None
    assert candidate.public["observation_time"].get("time_uncertain") == (
        "carried" if uncertain else None)


def test_as_of_mixed_sources_keep_ordinary_filter(source):
    engine, row, uncertain = source
    prior = engine._store.get(2)
    payload = json.loads(evidence_pack.build_evidence_pack(dict(
        question="What is the project status?", question_date="2024-03-19",
        baseline_refs=[dict(exact_ref=f"lcm:{item['store_id']}:0-{len(item['content'])}")
                       for item in (prior, row)],
    ), engine=engine))
    assert [item["store_id"] for item in payload["evidence"]] == ([row["store_id"]] if uncertain else [])
    assert len(payload["exclusions"]) == (1 if uncertain else 2)


def test_relation_read_does_not_create_schema(tmp_path):
    from hermes_lcm.store import MessageStore

    store = MessageStore(str(tmp_path / "read.db"))
    try:
        store_id = store.append("ordinary", dict(role="user", content=TEXT, timestamp=STAMP))
        before = store.connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
        store.connection.execute("PRAGMA query_only=ON")
        assert store.get_carried_stamp_ids([store_id]) == set()
        assert store.connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == before
    finally:
        store.close()
