"""Pinned500 topology/provenance checks with synthetic rows and mocked capture."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarking import longmemeval as instrument
from benchmarking import portable_recall_eval as evaluation
from scripts import eval_portable_recall as cli


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    dataset = tmp_path / "longmemeval_s"
    raw = [{"question_id": f"public-{i:03}", "question_type": "single-session-user",
            "question": f"synthetic question {i}", "haystack_sessions": []} for i in range(500)]
    dataset.write_text(json.dumps(raw))
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    calls = []
    def evaluate(question, provider, *, candidate_sink, **config):
        assert config == {"chunk_provider": "chunk", "provider_name": "fastembed",
                          "tmp_dir": tmp_path, "embeddings_enabled": True, "top_k": 10}
        calls.append(question.question_id)
        candidate_sink(question=question, store_id_to_turn={}, capture_ms=1)
        return {"synthetic_metric": question.question_id, "nested": {"latency_ms": 2}}
    def freeze(*, question, scope_id, split, span_annotations):
        assert scope_id == question.question_id and span_annotations is None
        if question.question_id == "public-001":
            raise evaluation.SpanCaptureUnavailable("synthetic projection failure")
        return evaluation.FrozenCase(question.question_id, scope_id, split, question.category,
                                     question.question, (), (), None)
    monkeypatch.setattr(instrument, "evaluate_question", evaluate)
    monkeypatch.setattr(instrument, "resolve_harness_providers", lambda *args: SimpleNamespace(
        summary="summary", chunk="chunk", summary_binding=("fastembed", "pinned-summary"),
        chunk_binding=("fastembed", "pinned-chunk")))
    monkeypatch.setattr(evaluation, "freeze_production_pool", freeze)
    def capture(index=0, count=1):
        return evaluation.capture_public(dataset, expected_sha256=checksum, tmp_dir=tmp_path,
            posture="LOCAL", local_model="pinned-summary", shard_index=index, shard_count=count,
            dev_header={"corpus_sha256": "d" * 64, "source_identity": "registered-source:LOCAL"})
    return dataset, checksum, calls, capture


@pytest.fixture
def shards(cohort, tmp_path):
    dataset, checksum, calls, capture = cohort
    baselines, pools = [], []
    for index in range(10):
        cases, baseline = capture(index, 10)
        baseline["source_identity"] = "candidate:LOCAL"
        pool, report = tmp_path / f"pool-{index}.jsonl", tmp_path / f"baseline-{index}.json"
        header = evaluation.write_frozen(pool, cases, input_kind="public", posture="LOCAL",
            source_identity=baseline["source_identity"], provenance=baseline["provenance"])
        baseline["frozen_corpus_sha256"] = header["corpus_sha256"]
        report.write_text(json.dumps(baseline))
        baselines.append(report)
        pools.append(pool)
    return dataset, checksum, baselines, pools


def merge(shards):
    dataset, checksum, baselines, pools = shards
    return evaluation.merge_public(dataset, expected_sha256=checksum,
                                   baseline_inputs=baselines, frozen_inputs=pools)


def test_partition_and_complete_merge(cohort, shards):
    dataset, checksum, calls, capture = cohort
    assert len(calls) == len(set(calls)) == 500
    assert calls[:50] == [f"public-{i:03}" for i in range(0, 500, 10)]
    cases, baseline = merge(shards)
    assert [r["question_id"] for r in baseline["per_question"]] == [f"public-{i:03}" for i in range(500)]
    assert all(r["current_instrument_metrics"]["synthetic_metric"] == r["question_id"]
               for r in baseline["per_question"])
    assert len(cases) == 499 and all(c.gold is None for c in cases)
    assert baseline["provenance"]["cohort_complete"] is True
    assert baseline["provenance"]["unmeasured_capture_questions"] == 1
    assert baseline["provenance"]["aggregation"]["shard_count"] == 10
    assert merge((dataset, checksum, list(reversed(shards[2])), list(reversed(shards[3])))) == (cases, baseline)


def test_default_unsharded(cohort):
    _, _, calls, capture = cohort
    cases, baseline = capture()
    assert len(calls) == 500 and len(cases) == 499
    assert baseline["provenance"]["shard"]["count"] == 1
    assert baseline["provenance"]["cohort_complete"] is True


@pytest.mark.parametrize("fault", ["duplicate", "missing", "foreign", "source", "model", "digest",
                                  "count", "capture", "partial", "shard", "metric", "pool", "dev"])
def test_reject_incomplete_or_mixed(shards, fault):
    dataset, checksum, baselines, pools = shards
    report = json.loads(baselines[0].read_text())
    if fault == "duplicate": report["per_question"][1] = report["per_question"][0]
    if fault == "missing": report["per_question"].pop()
    if fault == "foreign": report["per_question"][0]["question_id"] = "foreign"
    if fault == "source": report["source_identity"] = "other-source"
    if fault == "model": report["provenance"]["local_model"] = "other-model"
    if fault == "digest": report["provenance"]["shard"]["full_question_ids_sha256"] = "f" * 64
    if fault == "count": report["provenance"]["evaluated_questions"] = 500
    if fault == "capture": report["per_question"][0]["capture"]["status"] = "MISSING_CAPTURE"
    if fault == "partial": baselines.pop(); pools.pop()
    if fault == "shard": baselines[1] = baselines[0]; pools[1] = pools[0]
    if fault == "metric": report["per_question"][0]["current_instrument_metrics"] = None
    if fault == "pool": report["frozen_corpus_sha256"] = "f" * 64
    if fault == "dev": report["provenance"]["dev_registration"]["corpus_sha256"] = ""
    baselines[0].write_text(json.dumps(report))
    with pytest.raises(ValueError): merge((dataset, checksum, baselines, pools))


def test_full_dataset_validated_before_partition(cohort):
    dataset, _, calls, capture = cohort
    raw = json.loads(dataset.read_text())
    raw[499]["question_id"] = raw[0]["question_id"]
    dataset.write_text(json.dumps(raw))
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.public_questions(dataset, checksum)
    with pytest.raises(ValueError, match="checksum"):
        capture(0, 10)
    assert not calls


def test_merge_cli_create_only_and_validation_before_write(shards, tmp_path, monkeypatch):
    dataset, checksum, baselines, pools = shards
    output, frozen = tmp_path / "complete.json", tmp_path / "complete.jsonl"
    args = ["eval", "merge-public", "--dataset", str(dataset), "--dataset-sha256", checksum,
            "--baseline-output", str(output), "--frozen-output", str(frozen)]
    for baseline, pool in zip(baselines, pools):
        args += ["--baseline-input", str(baseline), "--frozen-input", str(pool)]
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    before = output.read_bytes(), frozen.read_bytes()
    with pytest.raises(SystemExit): cli.main()
    assert before == (output.read_bytes(), frozen.read_bytes())
    monkeypatch.setattr("sys.argv", [a.replace("complete.", "partial.") for a in args[:-4]])
    with pytest.raises(SystemExit): cli.main()
    assert not (tmp_path / "partial.json").exists() and not (tmp_path / "partial.jsonl").exists()
