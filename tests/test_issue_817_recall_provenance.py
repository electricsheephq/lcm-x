"""Unit-level recall health receipts; no dataset or provider benchmark runs."""

import json
from types import SimpleNamespace

import pytest

import benchmarking.longmemeval as lme


def _question(question_id="q-health"):
    # One evidence session, following the offline harness's synthetic fixture.
    return lme.parse_question({
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "what is my locker passcode ZEBRA",
        "answer": "ZEBRA",
        "question_date": "2023-01-01",
        "haystack_session_ids": ["s-evidence"],
        "haystack_dates": ["2023-01-01"],
        "haystack_sessions": [[{
            "role": "user", "content": "locker passcode ZEBRA", "has_answer": True,
        }]],
        "answer_session_ids": ["s-evidence"],
    })


_HITS = [{"kind": "message_excerpt", "session_id": "s-evidence", "store_id": 1}]
_CASES = [
    pytest.param(_HITS, "ok", False, id="hits-fts-ok"),
    pytest.param([], "ok", False, id="empty-fts-ok"),
    pytest.param([], "none", True, id="empty-fts-none"),
]


def _payload(hits, fts, degraded):
    return {
        "hits": hits,
        "degraded": degraded,
        "degraded_reason": "full-text arm unavailable" if degraded else "",
        "provenance": {
            "coverage": {"fts": fts}, "rerank": "disabled", "rerank_scores": [],
        },
    }


def _install_recall(monkeypatch, payload):
    lme._ensure_hermes_lcm_package()
    import hermes_lcm.tools as lcm_tools

    monkeypatch.setattr(lcm_tools, "lcm_recall", lambda *_a, **_k: json.dumps(payload))


@pytest.mark.parametrize("hits,fts,degraded", _CASES)
@pytest.mark.parametrize("return_status", [False, True], ids=["list", "tuple"])
def test_production_recall_records_health_without_changing_return(
    tmp_path, monkeypatch, hits, fts, degraded, return_status,
):
    _install_recall(monkeypatch, _payload(hits, fts, degraded))
    health = {}
    accounting = lme.ProviderAccounting()
    result = lme.production_recall_hits(
        _question(), SimpleNamespace(), None, None, None,
        provider_name="none", tmp_dir=tmp_path, embeddings_enabled=False,
        limit=10, return_status=return_status, recall_health=health,
        accounting=accounting,
    )
    assert result == ((hits, "disabled", []) if return_status else hits)
    assert health == {"degraded": degraded, "coverage": {"fts": fts}}
    assert accounting.degraded_outcomes == []


@pytest.mark.parametrize("provenance,coverage", [
    pytest.param({}, {}, id="absent"),
    pytest.param({"coverage": {"fts": "ok", "summary": "full", "chunk": 7}},
                 {"fts": "ok", "summary": "full"}, id="strings-only"),
])
def test_production_recall_copies_only_string_coverage(
    tmp_path, monkeypatch, provenance, coverage,
):
    _install_recall(monkeypatch, {"hits": [], "provenance": provenance})
    health = {}
    assert lme.production_recall_hits(
        _question(), SimpleNamespace(), None, None, None,
        provider_name="none", tmp_dir=tmp_path, embeddings_enabled=False,
        limit=10, recall_health=health,
    ) == []
    assert health == {"degraded": False, "coverage": coverage}
    assert health["coverage"] is not provenance.get("coverage")


def _record(question_id="q-health"):
    metrics = {"ingest_ms": 1.0}
    for arm in lme.ARMS:
        metrics[arm] = {
            "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0,
            "ndcg@10": 0.0, "latency_ms": 1.0,
            "turn": {
                "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0,
                "ndcg@10": 0.0, "session_granularity": False,
            },
        }
    return lme._question_checkpoint_record(_question(question_id), metrics)


def _counts(**overrides):
    counts = dict(fts_ok=0, fts_none=0, fts_absent=0, degraded=0, unrecorded=0)
    counts.update(overrides)
    return counts


def _accumulate(records):
    counts = _counts()
    state = dict(by_category={}, overall=lme._new_arm_samples(), ingest_samples=[],
                 rerank_mode_counts={}, recall_health_counts=counts)
    for record in records:
        lme._accumulate_question_checkpoint(record, **state)
    return counts


def test_live_and_legacy_rows_accumulate_recall_health():
    records = []
    for index, (fts, degraded) in enumerate((("ok", False), ("ok", False), ("none", True))):
        record = _record(f"q{index}")
        record["arms"]["lcm_recall"]["recall_health"] = {
            "degraded": degraded, "coverage": {"fts": fts},
        }
        records.append(record)
    records.append(_record("q-legacy"))
    assert _accumulate(records) == _counts(fts_ok=2, fts_none=1, degraded=1, unrecorded=1)


def test_absent_fts_and_stub_health_exclude_abstentions():
    records = []
    for degraded in (False, None):
        record = _record()
        record["arms"]["lcm_recall"]["recall_health"] = {
            "degraded": degraded, "coverage": {},
        }
        records.append(record)
    records.append({"abstention": True})
    assert _accumulate(records) == _counts(fts_absent=2)


def _run(tmp_path, questions, checkpoint=None, **kwargs):
    return lme.run_harness(
        questions, provider_name="stub", model="", tmp_dir=tmp_path,
        embeddings_enabled=False, reuse_db_template=False,
        checkpoint_path=checkpoint, **kwargs,
    )


def test_report_records_failed_search_and_checkpoint_health(tmp_path, monkeypatch):
    _install_recall(monkeypatch, _payload([], "none", True))
    checkpoint = tmp_path / "checkpoint.jsonl"
    report = _run(tmp_path, [_question()], checkpoint)
    assert report["lcm_recall_health"] == {**_counts(fts_none=1, degraded=1), "failed_searches": 1}
    row = json.loads(checkpoint.read_text().splitlines()[1])
    assert row["arms"]["lcm_recall"]["recall_health"] == {
        "degraded": True, "coverage": {"fts": "none"},
    }
    assert row["arms"]["lcm_recall"]["recall@10"] == 0.0


def test_old_checkpoint_resumes_and_counts_unrecorded(tmp_path, monkeypatch):
    _install_recall(monkeypatch, _payload([], "none", True))
    questions = [_question("q-old"), _question("q-live")]
    checkpoint = tmp_path / "checkpoint.jsonl"
    _run(tmp_path, questions[:1], checkpoint)
    lines = checkpoint.read_text().splitlines()
    row = json.loads(lines[1])
    row["arms"]["lcm_recall"].pop("recall_health", None)
    checkpoint.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    report = _run(tmp_path, questions, checkpoint, resume=True)
    assert report["scored_count"] == 2
    assert report["lcm_recall_health"] == {
        **_counts(fts_none=1, degraded=1, unrecorded=1), "failed_searches": 1,
    }


def test_empty_stub_dict_records_unknown_health(tmp_path, monkeypatch):
    monkeypatch.setattr(lme, "production_recall_hits", lambda *_a, **_k: [])
    checkpoint = tmp_path / "checkpoint.jsonl"
    report = _run(tmp_path, [_question()], checkpoint)
    row = json.loads(checkpoint.read_text().splitlines()[1])
    assert row["arms"]["lcm_recall"]["recall_health"] == {"degraded": None, "coverage": {}}
    assert report["lcm_recall_health"] == {**_counts(fts_absent=1), "failed_searches": 0}


def test_no_scored_questions_report_zero_health_counts(tmp_path):
    question = _question("q-health_abs")
    report = _run(tmp_path, [question])
    assert report["scored_count"] == 0
    assert report["lcm_recall_health"] == {**_counts(), "failed_searches": 0}


@pytest.mark.parametrize("health", [
    pytest.param([], id="not-object"),
    pytest.param({"degraded": "false", "coverage": {}}, id="bad-degraded"),
    pytest.param({"degraded": False, "coverage": []}, id="bad-coverage"),
    pytest.param({"degraded": False, "coverage": {"fts": False}}, id="non-string-fts"),
    pytest.param({"degraded": False, "coverage": {"fts": "partial"}}, id="unknown-fts"),
])
def test_restored_malformed_recall_health_raises(tmp_path, health):
    record = _record()
    record["arms"]["lcm_recall"]["recall_health"] = health
    with pytest.raises(ValueError, match=r"checkpoint line 2 field arms.lcm_recall.recall_health"):
        lme._validate_restored_checkpoint_metrics(
            record, line_number=2, path=tmp_path / "checkpoint.jsonl",
        )


@pytest.mark.parametrize("fts", [False, "partial", None])
def test_unknown_fts_status_fails_aggregation(fts):
    record = _record("q-unknown")
    record["arms"]["lcm_recall"]["recall_health"] = {
        "degraded": False, "coverage": {"fts": fts},
    }
    with pytest.raises(ValueError, match=r"unknown recall_health coverage.fts"):
        _accumulate([record])
