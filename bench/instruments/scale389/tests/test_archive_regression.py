from __future__ import annotations

import os

import pytest

from bench.instruments.scale389.archive_regression import (
    DEFAULT_DATASET,
    DEFAULT_SIDECAR,
    run_regression,
)
from bench.instruments.scale389 import archive_regression, build_corpus, phase1a, run_scale
from bench.tools.benchpaths import bench_path


@pytest.mark.skipif(
    not all(os.environ.get(name) for name in (
        "LCMX_BENCH_ROOT", "SCALE389_ARCHIVE_RESULTS", "SCALE389_ARCHIVE_CORPUS",
    )),
    reason="the machine-local read-only F34 archive is unavailable",
)
def test_archived_f34_outputs_reproduce_and_turn_join_is_non_degenerate():
    result = run_regression(
        archive_regression.required_path("SCALE389_ARCHIVE_RESULTS"),
        archive_regression.required_path("SCALE389_ARCHIVE_CORPUS") / "qeval.json",
        bench_path(DEFAULT_DATASET),
        bench_path(DEFAULT_SIDECAR),
    )

    assert result["archive_read_only"] is True
    assert result["session_regression"]["status"] == "exact"
    assert result["session_regression"]["published_table"]["A3"] == {
        "500": 0.82,
        "2000": 0.60,
        "8000": 0.34,
        "19829": 0.233,
    }
    assert sorted(result["turn_join_check"]["values"]) == [0, 1]


@pytest.mark.parametrize("name, getter", [
    ("SCALE389_CORPUS", phase1a._corpus),
    ("SCALE389_RESULTS", phase1a._results),
])
def test_run_locations_require_explicit_environment(monkeypatch, tmp_path, name, getter):
    monkeypatch.setenv("LCMX_BENCH_ROOT", str(tmp_path))
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match=name):
        getter()
    monkeypatch.setenv(name, "")
    with pytest.raises(SystemExit, match=name):
        getter()
    monkeypatch.setenv(name, str(tmp_path / "chosen"))
    assert getter() == tmp_path / "chosen"


@pytest.mark.parametrize("module, argv, environment, argument", [
    (build_corpus, ["--dataset", "fixture.json"], "SCALE389_CORPUS", "--output"),
    (run_scale, ["ingest"], "SCALE389_RESULTS", "--results"),
    (archive_regression, [], "SCALE389_ARCHIVE_RESULTS", "--results"),
    (archive_regression, ["--results", "fixture"], "SCALE389_ARCHIVE_CORPUS", "--qeval"),
])
def test_cli_names_missing_run_location(monkeypatch, tmp_path, module, argv, environment, argument):
    monkeypatch.setenv("LCMX_BENCH_ROOT", str(tmp_path))
    monkeypatch.delenv(environment, raising=False)
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert environment in str(exc.value)
    assert argument in str(exc.value)


@pytest.mark.parametrize("explicit", [False, True])
def test_build_cli_uses_selected_output(monkeypatch, tmp_path, explicit):
    selected = tmp_path / "selected"
    monkeypatch.setenv("SCALE389_CORPUS", str(tmp_path / "environment" if explicit else selected))
    seen = []
    monkeypatch.setattr(build_corpus, "build_corpus", lambda dataset, output, **kw: seen.append(output))
    argv = ["--dataset", "fixture.json"]
    if explicit:
        argv += ["--output", str(selected)]
    assert build_corpus.main(argv) == 0
    assert seen == [selected]


@pytest.mark.parametrize("explicit", [False, True])
def test_scale_cli_uses_selected_results(monkeypatch, tmp_path, explicit):
    selected = tmp_path / "selected"
    monkeypatch.setenv("SCALE389_RESULTS", str(tmp_path / "environment" if explicit else selected))
    seen = []
    monkeypatch.setattr(run_scale, "run_ingest", lambda args: seen.append(args.results))
    argv = ["ingest"]
    if explicit:
        argv += ["--results", str(selected)]
    assert run_scale.main(argv) == 0
    assert seen == [selected]


@pytest.mark.parametrize("explicit", [False, True])
def test_archive_cli_uses_selected_inputs(monkeypatch, tmp_path, explicit):
    results, corpus = tmp_path / "results", tmp_path / "corpus"
    monkeypatch.setenv("LCMX_BENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("SCALE389_ARCHIVE_RESULTS", str(tmp_path / "unused" if explicit else results))
    monkeypatch.setenv("SCALE389_ARCHIVE_CORPUS", str(tmp_path / "unused" if explicit else corpus))
    seen = []

    def fake_regression(*paths):
        seen.append(paths)
        return {}

    monkeypatch.setattr(archive_regression, "run_regression", fake_regression)
    argv = []
    if explicit:
        argv = ["--results", str(results), "--qeval", str(corpus / "qeval.json")]
    assert archive_regression.main(argv) == 0
    assert seen == [(results, corpus / "qeval.json", bench_path(DEFAULT_DATASET), bench_path(DEFAULT_SIDECAR))]
