"""Offline coverage for the explicit embedding arm and unrun-arm reporting."""

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli
from tests.test_longmemeval_harness import _synthetic_dataset


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
