"""Offline recall lever arms, with real store opens and synthetic model calls."""
import importlib.util
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import common, delivery, mcp_reader, run, score, stores
from bench.fmp.tests.test_score_commits import COMMITS, arm
from bench.fmp.tests.test_switches import conversations

REPO = Path(__file__).resolve().parents[3]
BASE = "b1768887f9e4e717593a92ae6935b53efd92fb96"
OPTION = ("--recall-option", "recall_later_statements=2")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    lme = common.load_lcm(REPO)
    from hermes_lcm import tokens, tools
    tokens._encoder_ready, tokens._encoder = True, None
    question = conversations()[0]
    monkeypatch.setattr(run, "commit", lambda _: BASE)
    monkeypatch.setattr(stores, "commit", lambda _: BASE)
    monkeypatch.setattr(run.datasets, "load", lambda *a: ([question], "dataset", "ids", {}))
    monkeypatch.setattr(run, "versions", lambda: {})
    built = Mock(wraps=stores.Stores)
    monkeypatch.setattr(run, "Stores", built)
    backends = Mock(return_value=SimpleNamespace(close=Mock()))
    monkeypatch.setattr(run, "Backends", backends)
    seen, transcripts = [], []

    def recall(arguments, engine):
        return json.dumps({"later": engine._config.recall_later_statements,
                           "provenance": {"delivery": {"unit": engine._config.recall_delivery_unit}}})

    monkeypatch.setattr(tools, "lcm_recall", recall)

    def answer(q, mode, client, reader, engine=None, speakers=None, **kwargs):
        with ExitStack() as stack:
            if mode == "agentic":
                spec = {**kwargs["server"], "log": str(tmp_path / "mcp.jsonl")}
                reader = mcp_reader.prepare(spec, stack)
                engine = reader.engine
                offered = reader.offered
                raw, error = reader.call("lcm_recall", {})
                assert not error
            else:
                offered = delivery.schemas(engine)
                raw = delivery.dispatch("lcm_recall", {}, engine, offered)
            seen.append((engine._config, offered, raw))
            transcripts.append(json.dumps({"schemas": offered, "result": raw}).encode())
        return {"status": "ok", "hypothesis": "yes", "calls": []}

    monkeypatch.setattr(delivery, "answer", answer)
    monkeypatch.setattr(run, "canary", lambda args, *a: seen.append(args.recall_options))
    monkeypatch.setattr(run.judges, "judge_row", lambda row, *a, **k: {
        **row, "judges": {run.CODEX_DEFAULT_JUDGE: {"status": "ok", "correct": True}}})

    def args(command="answer", *extra):
        return run.parser().parse_args([
            command, "--split-label", "synthetic", "--data", str(tmp_path / "data.json"),
            "--store-root", str(tmp_path / "stores"), "--output", str(tmp_path / "output"),
            "--agreement-judge", "none", "--workers", "1", *extra])

    return SimpleNamespace(args=args, seen=seen, transcripts=transcripts, built=built,
                           backends=backends, lme=lme, question=question)


@pytest.mark.parametrize("option,expected", [
    ("recall_later_statements=2", 2), ("recall_later_statements=999", 5),
    ("recall_adjacent_reply=YES", True), ("recall_fts_anchor=off", False),
    ("recall_delivery_order= TIME ", "time"), ("recall_scan_budget_s=0.25", 0.25),
    ("recall_arm_weights=fts=0.25,chunk=0", {"fts": 0.25, "summary": 1.0, "chunk": 0.0}),
])
def test_recall_option_parsing(cli, option, expected):
    args = cli.args("canary", "--recall-option", option)
    run.execute(args)
    key = option.partition("=")[0]
    assert cli.seen == [{key: expected}]
    assert type(cli.seen[0][key]) is type(expected)


@pytest.mark.parametrize("option", [
    "recall_missing=1", "fresh_tail_count=1", "recall_later_statements=bad",
    "recall_later_statements=-1", "recall_adjacent_reply=maybe", "recall_scan_rows=bad",
    "recall_delivery_order=invalid", "recall_arm_weights=fts=bad", "recall_scan_rows=",
    "recall_later_statements", "recall_query_timeout_s=bad",
])
def test_recall_option_refused_before_stores_or_backends(cli, option):
    args = cli.args("answer", "--recall-option", option)
    with pytest.raises(ValueError, match="recall-option"):
        run.execute(args)
    cli.built.assert_not_called()
    cli.backends.assert_not_called()
    assert not args.output.exists()


@pytest.mark.parametrize("mode", ["single", "agentic"])
def test_recall_option_answer_and_mcp(cli, tmp_path, mode):
    args = cli.args("answer", "--delivery", mode, *OPTION,
                    "--recall-option", "recall_delivery_unit=session")
    run.execute(args)
    config, offered, raw = cli.seen[0]
    assert config.recall_later_statements == 2
    assert config.recall_delivery_unit == "session"
    assert "Session delivery" in next(s["function"]["description"] for s in offered
                                      if s["function"]["name"] == "lcm_recall")
    assert json.loads(raw)["later"] == 2
    spec = run.server_spec(args, "dataset", cli.question)
    expected = {"recall_delivery_unit": "session", "recall_later_statements": 2}
    assert spec["recall_options"] == expected
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert manifest["recall_options"] == expected
    assert list(manifest["recall_options"]) == sorted(expected)


def test_recall_option_build_ignores_options(cli, tmp_path, monkeypatch):
    from hermes_lcm.store import MessageStore
    observed = []
    original = MessageStore.__init__

    def opened(self, *a, **kwargs):
        observed.append(kwargs["ingest_protection_config"].recall_later_statements)
        original(self, *a, **kwargs)

    monkeypatch.setattr(MessageStore, "__init__", opened)
    args = cli.args("build-stores", *OPTION)
    run.execute(args)
    assert observed == [0]
    assert "recall_options" not in (args.output / "build-receipt.json").read_text()
    control = stores.Stores(args.store_root, "off", REPO, "dataset", cli.lme)
    assert control.build(cli.question) == "reused"
    # The answer store also builds with defaults, then opens frozen with options.
    args = cli.args("answer", *OPTION)
    run.execute(args)
    assert observed == [0, 2, 2]


@pytest.mark.parametrize("answer_options,judge_options", [
    ((), OPTION), (OPTION, ()), (OPTION, ("--recall-option", "recall_later_statements=1")),
])
def test_recall_option_judge_mismatch(cli, answer_options, judge_options):
    run.execute(cli.args("answer", *answer_options))
    manifest = cli.args().output / "run-manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="judging run identity mismatch: recall_options"):
        run.execute(cli.args("judge", *judge_options))
    assert manifest.read_bytes() == before
    assert not (manifest.parent / "judged.jsonl").exists()


def test_recall_option_judge_matches_and_cli_error(cli, monkeypatch, capsys):
    run.execute(cli.args("answer", *OPTION))
    argv = ["judge", "--split-label", "synthetic", "--data", str(cli.args().data),
            "--store-root", str(cli.args().store_root), "--output", str(cli.args().output),
            "--agreement-judge", "none", "--workers", "1"]
    monkeypatch.setattr("sys.argv", ["run.py", *argv])
    assert run.main() == 1
    assert "judging run identity mismatch: recall_options" in capsys.readouterr().err
    run.execute(cli.args("judge", *OPTION))
    assert json.loads((cli.args().output / "run-manifest.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("cross_commit", [False, True])
def test_recall_option_score_axis(tmp_path, cross_commit):
    paths = [arm(tmp_path, f"{i}-{j}", commit, [bool(j)], harness_tree_sha256="same",
                 **({"recall_options": {"recall_later_statements": 2, "recall_adjacent_reply": True}} if j else {}))
             for i, commit in enumerate(COMMITS if cross_commit else COMMITS[:1]) for j in range(2)]
    report = score.score(paths, tmp_path / "score", pair_across_commits=cross_commit)
    assert len(report["arms"]) == len(paths)
    assert len(report["paired"]) == (2 if cross_commit else 1)
    assert sum("/opt:recall_adjacent_reply=true,recall_later_statements=2" in name
               for name in report["arms"]) == len(paths) // 2
    assert "recall_options" in score.IDENTITY
    manifest = paths[1] / "run-manifest.json"
    common.write_json(manifest, {**json.loads(manifest.read_text()), "harness_tree_sha256": "other"})
    with pytest.raises(ValueError, match="identical harness_tree_sha256"):
        score.score(paths, tmp_path / "mismatch", pair_across_commits=cross_commit)


def base_module(name):
    """Load pinned main's module with current imports, without a second checkout."""
    path = Path(run.__file__).with_name(name + ".py")
    source = subprocess.check_output(["git", "show", f"{BASE}:bench/fmp/{name}.py"], cwd=REPO)
    spec = importlib.util.spec_from_loader("bench.fmp._base_" + name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def test_recall_option_default_bytes_match_main(cli, tmp_path, monkeypatch):
    baseline_run, baseline_score = base_module("run"), base_module("score")
    for name in ("commit", "versions", "Backends", "canary"):
        monkeypatch.setattr(baseline_run, name, getattr(run, name))
    # Harness provenance changes with source edits; pin it and all clocks equally.
    for module in (baseline_run, run):
        monkeypatch.setattr(module, "sha", lambda _: "pinned-provenance")
        monkeypatch.setattr(module, "utc", lambda: "2025-01-01T00:00:00+00:00")
    monkeypatch.setattr(run.time, "monotonic", lambda: 100.0)
    snapshots = []
    for module, scorer in ((baseline_run, baseline_score), (run, score)):
        args = cli.args()
        args.output = tmp_path / "comparison"
        module.execute(args)
        assert "recall_options" not in json.loads((args.output / "run-manifest.json").read_text())
        spec = module.server_spec(args, "dataset", cli.question)
        assert "recall_options" not in spec
        args.command = "judge"
        module.execute(args)
        report = tmp_path / "report"
        scorer.score([args.output], report)
        files = [args.output / n for n in ("run-manifest.json", "answers.jsonl", "judged.jsonl")]
        files += [report / n for n in ("report.json", "report.md", "adjudicate.jsonl")]
        snapshots.append(([p.read_bytes() for p in files], cli.transcripts[-1], spec))
        import shutil
        shutil.rmtree(args.output)
        shutil.rmtree(report)
    assert snapshots[0] == snapshots[1]
