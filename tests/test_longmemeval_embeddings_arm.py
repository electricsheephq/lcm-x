"""Offline coverage for the explicit embedding arm and unrun-arm reporting."""

import importlib.util
import json
from pathlib import Path

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli
from tests.test_longmemeval_harness import _synthetic_dataset
from tests.test_longmemeval_medium import _zero_timing


class _HarnessReached(Exception):
    pass


@pytest.mark.parametrize("flag, expected", [("off", False), (None, None), ("on", None)])
def test_cli_threads_embedding_mode(tmp_path, monkeypatch, flag, expected):
    cli = load_cli()
    source = tmp_path / "longmemeval_s"
    source.touch()
    monkeypatch.setattr(
        cli, "load_questions_with_sha256",
        lambda *_args, **_kwargs: (_synthetic_dataset(), "a" * 64),
    )
    captured = {}

    def capture(_questions, **kwargs):
        captured.update(kwargs)
        raise _HarnessReached

    monkeypatch.setattr(cli, "run_harness", capture)
    argv = [
        "run", "--dataset", str(source), "--output", str(tmp_path / "out"),
        "--allow-external-output", "--provider", "stub",
    ]
    if flag is not None:
        argv += ["--embeddings", flag]
    with pytest.raises(_HarnessReached):
        cli.main(argv)
    assert captured.get("embeddings_enabled") is expected


def test_cli_off_refuses_metered_provider_before_resolution(tmp_path, monkeypatch, capsys):
    cli = load_cli()

    def forbidden(*_args, **_kwargs):
        pytest.fail("off mode must refuse before resolving any provider")

    monkeypatch.setattr(cli, "resolve_harness_provider", forbidden)
    monkeypatch.setattr(lme, "resolve_harness_provider", forbidden)
    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "run", "--dataset", str(tmp_path / "longmemeval_s"),
            "--output", str(tmp_path / "out"), "--allow-external-output",
            "--embeddings", "off", "--provider", "voyage", "--model", "unused",
        ])
    assert exc.value.code == 2
    assert "--embeddings off requires --provider stub" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _run(tmp_path, enabled):
    return lme.run_harness(
        _synthetic_dataset(), provider_name="stub", model="",
        tmp_dir=tmp_path, embeddings_enabled=enabled,
    )


def test_off_reports_unrun_vector_arms_and_numeric_lexical_arms(tmp_path):
    report = _run(tmp_path, False)
    for arms in [report["arms"], *report["per_category"].values()]:
        for arm, row in arms.items():
            ran = arm in {"fts", "lcm_recall"}
            assert row["run"] is ran
            assert row["n"] == (3 if ran else 0)
            for metrics in (row, row["turn"]):
                for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10"):
                    if ran:
                        assert isinstance(metrics[metric], (int, float))
                    else:
                        assert metrics[metric] is None
            for latency in row["latency_ms"].values():
                assert isinstance(latency, (int, float)) if ran else latency is None
    markdown = lme.render_markdown(report)
    for arm in set(lme.ARMS) - {"fts", "lcm_recall"}:
        assert f"| {arm} | not run |" in markdown


@pytest.mark.parametrize("enabled", [False, True])
def test_embedding_calls_have_off_on_positive_control(tmp_path, monkeypatch, enabled):
    calls = {"documents": 0, "query": 0}
    documents = lme.StubEmbedder.embed_documents
    query = lme.StubEmbedder.embed_query

    def count_documents(self, texts):
        calls["documents"] += 1
        return documents(self, texts)

    def count_query(self, text):
        calls["query"] += 1
        return query(self, text)

    monkeypatch.setattr(lme.StubEmbedder, "embed_documents", count_documents)
    monkeypatch.setattr(lme.StubEmbedder, "embed_query", count_query)
    _run(tmp_path, enabled)
    if enabled:
        assert calls["documents"] > 0
        assert calls["query"] > 0
    else:
        assert calls == {"documents": 0, "query": 0}


@pytest.mark.parametrize("enabled", [False, True])
def test_retrieval_config_matches_resolved_mode(tmp_path, enabled):
    report = _run(tmp_path, enabled)
    assert report["retrieval_config"] == {
        "embeddings_enabled": enabled,
        "provider": report["provider"],
        "lcm_recall_mode": "semantic_or_hybrid" if enabled else "full_text",
    }
    assert report["embeddings_enabled"] is enabled
    assert all(
        row["run"] is (enabled or arm in {"fts", "lcm_recall"})
        for arm, row in report["arms"].items()
    )


def test_no_samples_reports_null_metrics_even_when_embeddings_on(tmp_path):
    report = lme.run_harness(
        [], provider_name="stub", model="", tmp_dir=tmp_path,
    )
    assert all(row["run"] is False for row in report["arms"].values())
    assert all(row["recall@1"] is None for row in report["arms"].values())
    assert "not run" in lme.render_markdown(report)


def test_off_resume_keeps_vector_arms_unrun(tmp_path):
    questions = _synthetic_dataset()
    checkpoint = tmp_path / "checkpoint.jsonl"
    kwargs = {
        "provider_name": "stub", "model": "", "embeddings_enabled": False,
        "checkpoint_path": checkpoint,
    }
    original = lme.run_harness(questions, tmp_dir=tmp_path / "first", **kwargs)
    resumed = lme.run_harness(
        questions, tmp_dir=tmp_path / "resumed", resume=True, **kwargs,
    )
    assert resumed["arms"] == original["arms"]
    assert resumed["per_category"] == original["per_category"]
    assert resumed["retrieval_config"] == original["retrieval_config"]


def test_off_rerank_aggregate_is_disabled_live_and_resumed(tmp_path):
    questions = _synthetic_dataset()
    kwargs = {
        "provider_name": "stub", "model": "", "embeddings_enabled": False,
        "checkpoint_path": tmp_path / "checkpoint.jsonl",
    }
    live = lme.run_harness(questions, tmp_dir=tmp_path / "live", **kwargs)
    resumed = lme.run_harness(
        questions, tmp_dir=tmp_path / "resumed", resume=True, **kwargs,
    )
    for report in (live, resumed):
        assert report["scored_count"] == len(questions)
        assert report["rerank"]["mode"] == "disabled"
        assert report["rerank"]["real_count"] == 0
        assert report["rerank"]["placeholder_count"] == 0
        assert report["rerank"]["counts"] == {}


def test_off_checkpoint_and_candidates_contain_only_measured_arms(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    dump = tmp_path / "candidates.jsonl"
    lme.run_harness(
        _synthetic_dataset(), provider_name="stub", model="",
        tmp_dir=tmp_path / "dbs", embeddings_enabled=False,
        checkpoint_path=checkpoint, dump_candidates_path=dump,
    )
    # Exercise the rebank consumer unchanged: absent arms project to no measurement.
    spec = importlib.util.spec_from_file_location(
        "result_identity",
        Path(__file__).resolve().parents[1] / "bench/tools/v1m-rebank/result_identity.py",
    )
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    checkpoint_rows = list(map(json.loads, checkpoint.read_text().splitlines()[1:]))
    dump_rows = list(map(json.loads, dump.read_text().splitlines()[1:]))
    for record, candidates in zip(checkpoint_rows, dump_rows, strict=True):
        assert set(record["arms"]) == {"fts", "lcm_recall"}
        assert set(candidates["arms"]) == {"fts", "lcm_recall"}
        projection = identity._project(record)
        for arm in lme.ARMS:
            assert (projection["arms"][arm] is not None) is (arm in record["arms"])
        for arm, ranking in candidates["arms"].items():
            assert lme.recall_at_k(
                ranking["sessions_top10"], set(candidates["gold_sessions"]), 10,
            ) == record["arms"][arm]["recall@10"]
            assert lme.recall_at_k(
                [tuple(turn) for turn in ranking["turns_top10"]],
                {tuple(turn) for turn in candidates["gold_turns"]}, 10,
            ) == record["arms"][arm]["turn"]["recall@10"]
    assert identity.compare(checkpoint, checkpoint) == (len(checkpoint_rows), [], [], [])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_off_resume_accepts_old_and_new_checkpoint_arms(tmp_path, monkeypatch, legacy, partial):
    _zero_timing(monkeypatch)
    questions = _synthetic_dataset()
    checkpoint = tmp_path / "checkpoint.jsonl"
    kwargs = {
        "provider_name": "stub", "model": "", "embeddings_enabled": False,
        "checkpoint_path": checkpoint,
    }
    writer = lme._question_checkpoint_record

    def legacy_writer(*args, **kwargs):
        # The old off writer serialized all seven arms from the scored object.
        kwargs["embeddings_enabled"] = True
        return writer(*args, **kwargs)

    with monkeypatch.context() as initial:
        if legacy:
            initial.setattr(lme, "_question_checkpoint_record", legacy_writer)
        original = lme.run_harness(questions, tmp_dir=tmp_path / "live", **kwargs)
    lines = checkpoint.read_bytes().splitlines(keepends=True)
    assert set(json.loads(lines[1])["arms"]) == (
        set(lme.ARMS) if legacy else {"fts", "lcm_recall"}
    )
    if partial:
        checkpoint.write_bytes(b"".join(lines[:2]))
    evaluated = []
    evaluate = lme.evaluate_question

    def track(question, *args, **kwargs):
        evaluated.append(question.question_id)
        return evaluate(question, *args, **kwargs)

    monkeypatch.setattr(lme, "evaluate_question", track)
    resumed = lme.run_harness(
        questions, tmp_dir=tmp_path / "resumed", resume=True, **kwargs,
    )
    assert resumed == original
    assert evaluated == ([q.question_id for q in questions[1:]] if partial else [])
    if partial:
        new_rows = list(map(json.loads, checkpoint.read_text().splitlines()[2:]))
        assert all(set(row["arms"]) == {"fts", "lcm_recall"} for row in new_rows)
