"""Issue #265: recover capped FTS when semantic arms produce no evidence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
import hermes_lcm.vector_store as vector_store_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.ingest_protection import (
    embedding_privacy_revision,
    embedding_provider_requires_privacy,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import VectorStore

CURRENT = "session-cur"
CAPPED_REASON = "full-text arm capped to preserve semantic recall budget"
QUERY = "kanban migration"


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.queries: list[str] = []
        self.last_usage_tokens = 7

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)


@pytest.fixture
def recall_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "recall.db"),
        embeddings_enabled=True,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
        sensitive_patterns_enabled=True,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_path),
        current_session_id=CURRENT,
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _add_summary(
    engine,
    summary,
    *,
    session_id,
    created_at,
    latest_at=None,
    source_ids=None,
):
    return engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=20,
            source_token_count=40,
            source_ids=list(source_ids or []),
            source_type="messages",
            created_at=created_at,
            earliest_at=created_at,
            latest_at=latest_at if latest_at is not None else created_at,
            expand_hint=f"Expand {summary[:20]}",
        )
    )


def _seed_summary_vectors(engine, rows, *, provider="mock", model="mock-model"):
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        revision = (
            embedding_privacy_revision(engine._config)
            if embedding_provider_requires_privacy(provider)
            else ""
        )
        store.register_profile(model, provider, 2, revision=revision)
        identity = store.capture_identity(model, provider=provider)
        for node_id, vector in rows:
            store.record_embedding(str(node_id), "summary", model, vector, identity=identity)
    finally:
        store.close()


def _freeze_vector_store_clock(monkeypatch, clock):
    """Put the vector store on the same synthetic clock as the tool.

    The recall budget is one absolute deadline read by both ``tools`` and the
    store's full-corpus scan. Freezing only the tool's clock leaves the store on
    real time, so it sees the synthetic deadline as already expired and reports
    ``coverage='bounded'`` before scoring a single batch.
    """
    monkeypatch.setattr(vector_store_module, "_monotonic", lambda: clock[0])


@pytest.fixture
def recall_case(recall_engine, monkeypatch):
    recall_engine._config.recall_query_timeout_s = 8.0
    clock = [100.0]
    monkeypatch.setattr(lcm_tools.time, "monotonic", lambda: clock[0])
    _freeze_vector_store_clock(monkeypatch, clock)
    node = _add_summary(
        recall_engine, "quarterly kanban migration summary",
        session_id="session-s", created_at=1.0,
    )
    sid = recall_engine._store.append(
        "session-b",
        {"role": "user", "content": "the quarterly kanban migration plan for the platform team"},
        source="chat",
    )
    real_fts = lcm_tools._lcm_recall_fts_arm
    calls = []

    def make_fts(second="real", *, capped=True, advance=True):
        if capped:
            _seed_summary_vectors(recall_engine, [(node, (1.0, 0.0))])

        def fts(engine, query, *, candidate_limit, deadline, excluded_session_ids):
            calls.append((deadline, candidate_limit, frozenset(excluded_session_ids)))
            if len(calls) == 1 or second == "expire":
                if advance:
                    clock[0] = deadline
                return [], {
                    "error": "lcm_grep request deadline exceeded",
                    "mode": "recall",
                    "timeout": True,
                    "timeout_stage": "full_text",
                }
            assert len(calls) == 2
            return real_fts(
                engine, query, candidate_limit=candidate_limit, deadline=deadline,
                excluded_session_ids=excluded_session_ids,
            )

        monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", fts)

    return SimpleNamespace(
        engine=recall_engine, clock=clock, node=node, sid=sid,
        calls=calls, make_fts=make_fts,
    )


def _recall(case, **args):
    return json.loads(lcm_tools.lcm_recall(
        {"query": QUERY, "include": "all", "limit": 5, **args}, engine=case.engine,
    ))


def _reasons(payload):
    return payload.get("degraded_reason", "").split("; ")


@pytest.mark.parametrize("outcome", ("raises", "none"))
def test_capped_fts_reruns_when_provider_resolution_fails(recall_case, monkeypatch, outcome):
    case = recall_case
    case.make_fts()

    def resolve(*_args, **_kwargs):
        if outcome == "raises":
            raise RuntimeError("provider down")
        return None

    monkeypatch.setattr(lcm_tools, "_resolve_recall_provider", resolve)
    payload = _recall(case)
    assert len(case.calls) == 2
    assert case.calls[0][0] == 102.0
    assert case.calls[1][0] == 108.0
    assert case.calls[0][1:] == case.calls[1][1:]
    provenance = payload["provenance"]
    assert provenance["coverage"]["fts"] == "ok"
    assert provenance["arms_run"] == ["fts"]
    assert case.sid in {hit["store_id"] for hit in payload["hits"]}
    assert payload.get("timeout") is not True
    assert CAPPED_REASON not in _reasons(payload)
    reason = ("embedding provider unavailable: provider down" if outcome == "raises"
              else "embedding provider is not configured")
    assert reason in _reasons(payload)
    assert provenance["fts_rerun"] is True


def test_capped_fts_reruns_when_strict_summaries_yield_only_leads(recall_case, monkeypatch):
    case = recall_case
    case.make_fts()
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    assert case.engine._config.recall_reference_strict is True
    payload = _recall(case, detail="answer_ready")
    assert len(case.calls) == 2
    assert case.calls[0][0] == 102.0
    assert case.calls[1][0] == 108.0
    assert payload["hits"]
    assert all(hit["kind"] == "message_excerpt" for hit in payload["hits"])
    assert case.sid in {hit["store_id"] for hit in payload["hits"]}
    provenance = payload["provenance"]
    assert case.node in {lead["node_id"] for lead in provenance["answer_ready"]["summary_leads"]}
    assert provenance["coverage"]["fts"] == "ok"
    assert provenance["coverage"]["summary"] == "full"
    assert CAPPED_REASON not in _reasons(payload)
    assert payload.get("timeout") is not True
    assert provenance["fts_rerun"] is True


def test_rerun_that_expires_reports_request_timeout(recall_case, monkeypatch):
    case = recall_case
    case.make_fts("expire")
    monkeypatch.setattr(lcm_tools, "_resolve_recall_provider", lambda *_a, **_k: None)
    payload = _recall(case)
    assert len(case.calls) == 2
    assert [call[0] for call in case.calls] == [102.0, 108.0]
    assert payload["provenance"]["coverage"]["fts"] == "none"
    assert "full-text arm unavailable" in _reasons(payload)
    assert CAPPED_REASON not in _reasons(payload)
    assert payload["timeout"] is True
    assert payload["provenance"]["fts_rerun"] is True
    assert payload["hits"] == []


def test_no_rerun_when_semantic_arm_returns_hits(recall_case, monkeypatch):
    case = recall_case
    case.make_fts()
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools, "_lcm_recall_summary_arm", lambda *_a, **_k: (
        [{"kind": "summary", "node_id": case.node, "session_id": "session-s",
          "timestamp": 1.0, "snippet": "semantic hit lands while full-text is fenced",
          "from_current_session": False, "expand_hint": f"lcm_expand(node_id={case.node})"}],
        "full", 1, 1, [],
    ))
    monkeypatch.setattr(lcm_tools, "_lcm_recall_chunk_arm", lambda *_a, **_k: ([], "none", 0, 0))
    payload = _recall(case)
    assert len(case.calls) == 1
    assert case.calls[0][0] == 102.0
    assert payload["provenance"]["arms_run"] == ["summary"]
    assert CAPPED_REASON in _reasons(payload)
    assert payload.get("timeout") is not True
    assert "fts_rerun" not in payload["provenance"]


def test_no_rerun_when_request_deadline_already_spent(recall_case, monkeypatch):
    case = recall_case
    case.make_fts()

    def resolve(*_args, **_kwargs):
        case.clock[0] = 108.0
        return None

    monkeypatch.setattr(lcm_tools, "_resolve_recall_provider", resolve)
    payload = _recall(case)
    assert len(case.calls) == 1
    assert case.calls[0][0] == 102.0
    assert CAPPED_REASON in _reasons(payload)
    assert "fts_rerun" not in payload["provenance"]


@pytest.mark.parametrize("mode", ("saturation", "uncapped"))
def test_no_rerun_without_capped_expiry(recall_case, monkeypatch, mode):
    case = recall_case
    case.make_fts(capped=mode == "saturation", advance=mode != "saturation")
    monkeypatch.setattr(lcm_tools, "_resolve_recall_provider", lambda *_a, **_k: None)
    payload = _recall(case)
    assert len(case.calls) == 1
    assert case.calls[0][0] == (102.0 if mode == "saturation" else 108.0)
    assert "full-text arm unavailable" in _reasons(payload)
    assert CAPPED_REASON not in _reasons(payload)
    assert "fts_rerun" not in payload["provenance"]
    assert payload["timeout"] is True
