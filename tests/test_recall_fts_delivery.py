"""Exact FTS delivery beyond hydration and match-centered source windows."""
import hashlib
import json
import random

import pytest

from hermes_lcm import tools
from tests.test_lcm_recall import recall_engine as recall_engine


def seed(engine, texts):
    engine._config.embeddings_enabled = False
    for i, text in enumerate(texts):
        store_id = engine._store.append(f"archive-{i // 5}", {
            "role": "user", "content": text, "timestamp": 1700000000 + i,
        })
        engine._store.connection.execute(
            "UPDATE messages SET timestamp=? WHERE store_id=?", (1700000000 + i, store_id),
        )
    engine._store.connection.commit()


def recall(engine, query="evidence", **kwargs):
    return tools.lcm_recall({"query": query, "limit": 25,
                             "detail": "answer_ready", **kwargs}, engine=engine)


def check_spans(engine, raw):
    payload = json.loads(raw)
    for hit in payload["hits"]:
        row = engine._store.get(hit["store_id"])
        text = hit.get("content", hit["snippet"])
        offset = hit["content_offset"]
        assert row["content"][offset:offset + len(text)] == text
        assert hit["content_returned_chars"] == len(text)
    assert len(raw) <= tools._LCM_RECALL_RESPONSE_CHAR_CAP
    assert len(raw.encode()) <= tools._LCM_RECALL_RESPONSE_CHAR_CAP
    return payload


def test_25_fts_rows_survive_eight_hydrations(recall_engine):
    seed(recall_engine, [f"preface {i} evidence end" for i in range(25)])
    payload = check_spans(recall_engine, recall(recall_engine))
    assert len(payload["hits"]) == 25
    assert sum("content" in hit for hit in payload["hits"]) == 8
    assert max(sum(h["session_id"] == hit["session_id"] for h in payload["hits"])
               for hit in payload["hits"]) <= 5


@pytest.mark.parametrize(("query", "match"), [
    ("runs", "running"), ("cafe", "café"), ('"running quickly"', "running quickly"),
])
def test_late_fts_match_survives_hydration(recall_engine, query, match):
    seed(recall_engine, ["padding " * 400 + f"The evidence concerns {match}."])
    payload = check_spans(recall_engine, recall(recall_engine, query))
    assert len(payload["hits"]) == 1
    hit = payload["hits"][0]
    assert match in hit["content"]
    assert hit["evidence_span"]["char_start"] > 2400


@pytest.mark.parametrize("random_seed", [0, 1, 2])
def test_answer_ready_excerpt_property(recall_engine, random_seed):
    rng = random.Random(random_seed)
    texts = ["prefix " * rng.randrange(0, 600) + f"evidence row {i} café "
             + "suffix " * rng.randrange(0, 80) for i in range(35)]
    seed(recall_engine, texts)
    payload = check_spans(recall_engine, recall(recall_engine))
    assert len(payload["hits"]) > 8


def test_snippets_fixture_unchanged(recall_engine, monkeypatch):
    seed(recall_engine, [f"preface {i} evidence end" for i in range(25)])
    monkeypatch.setattr(tools.time, "time", lambda: 1700000100)
    raw = recall(recall_engine, detail="snippets")
    assert hashlib.sha256(raw.encode()).hexdigest() == "b0bb091327843b3eee9e959f1aada0bbfc34ad7b9eb751b8c20bfca619c810ec"


def test_answer_ready_byte_budget_unchanged(recall_engine, monkeypatch):
    seed(recall_engine, ["prefix " * 400 + f"evidence row {i}" for i in range(25)])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 12000)
    payload = check_spans(recall_engine, recall(recall_engine))
    assert payload["provenance"]["answer_ready"]["response_truncated"]
