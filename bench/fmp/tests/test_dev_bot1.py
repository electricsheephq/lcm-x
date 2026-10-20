"""Offline regressions for the four PR 1682 round-one review threads."""
import ast
import json
import math
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import dev, dev_cache
from bench.fmp.common import sha
from bench.fmp.dev_panel import encoded


@pytest.mark.parametrize("depth", [1, 12, 13, 25, 50, 100])
def test_caps_match_product_formulas(depth):
    tree = ast.parse((Path(__file__).resolve().parents[3] / "tools.py").read_text())
    recall = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lcm_recall")
    names = {"candidate_limit", "rerank_window"}
    formulas = {n.targets[0].id: n.value for n in recall.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}
    cap = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_LCM_GREP_HYBRID_CANDIDATE_CAP" for t in n.targets))
    values = {name: eval(compile(ast.Expression(expr), "tools.py", "eval"),
              {"min": min, "max": max, "limit": depth, "_LCM_GREP_HYBRID_CANDIDATE_CAP": cap})
              for name, expr in formulas.items()}
    assert dev.caps(depth) == dict(limit=depth, candidate_pool=values["candidate_limit"],
        rerank_window=values["rerank_window"], per_session=math.ceil(5 * depth / 25),
        expanded=math.ceil(8 * depth / 25))


def test_source_identity_hashes_tracked_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_cache.subprocess, "check_output", lambda *a, **kw: b"product.py\0")
    assert dev_cache.source_identity(tmp_path) == sha(encoded({"product.py": sha(b"deleted")}))
    (tmp_path / "product.py").write_text("synthetic product")
    assert dev_cache.source_identity(tmp_path) != sha(encoded({"product.py": sha(b"deleted")}))


@pytest.fixture
def screen(tmp_path, monkeypatch):
    questions = [SimpleNamespace(qid=f"q{i}", question_date="", question=f"question {i}") for i in range(4)]
    monkeypatch.setattr(dev, "load_lcm", lambda repo: None)
    monkeypatch.setattr(dev, "load", lambda *a: (questions, {"items": []}, {"items": []}))
    monkeypatch.setattr(dev.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 1024 ** 3))
    panel = tmp_path / "panel.json"
    panel.write_text("{}")
    retrieval = tmp_path / "cache" / "retrieval"
    retrieval.mkdir(parents=True)
    calls, commands = [], []

    def worker_run(cmd, **kw):
        spec = json.loads(Path(cmd[-1]).read_bytes())
        arm = Path(spec["receipt"]).name.split("-")[0]
        rows = []
        for q in questions:
            key = f"{arm}-{q.qid}"
            (retrieval / (key + ".json")).write_text(json.dumps({"value": {"evidence": key}}))
            rows.append(dict(id=q.qid, cache_key=key, sources={}, retrieval_sha256=key))
        Path(spec["receipt"]).write_text(json.dumps(dict(product="fixture", rows=rows)))
        return SimpleNamespace(returncode=0)

    class Cache:
        counts = {}

        def call(self, spec, messages, records, *, role, **kw):
            if role == "reader":
                calls.append(messages[1]["content"].splitlines()[1])
            return {"content": "yes"}

    class Card:
        def parser(self):
            return SimpleNamespace(parse_args=lambda args: args)

        def execute(self, args):
            if args[0] == "init":
                commands.append(args[args.index("--command") + 1])
            return 0

    monkeypatch.setattr(dev.subprocess, "run", worker_run)
    monkeypatch.setattr(dev, "MemoryCodex", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(dev, "Responses", lambda *a: Cache())
    monkeypatch.setattr(dev, "judge_messages", lambda *a: ([], "lme-s"))
    monkeypatch.setattr(dev, "evaluate", lambda *a, **kw: {"verdict": "NO-SIGNAL"})
    monkeypatch.setattr(dev, "card_tool", lambda: Card())

    def run(*extra):
        args = dev.parser().parse_args(["--baseline", str(tmp_path / "baseline"),
            "--cache-root", str(retrieval.parent), "--output", str(tmp_path / f"out{len(commands)}"),
            "--panel", str(panel), "--guard", str(panel), "--lme-data", str(panel),
            "--locomo-data", str(panel), "--workers", "1", *extra])
        assert dev.execute(args) == 0
        return args

    return run, calls, commands


def test_recorded_command_tracks_parsed_invocation(screen):
    run, _, commands = screen
    first = run("--candidate-config", '{"recall_depth":12}')
    run("--reader", "codex:fixture@low", "--candidate-config", '{"recall_depth":12}')
    run("--workers", "2", "--candidate-config", '{"recall_depth":12}')
    assert len(set(commands)) == 3
    tokens = shlex.split(commands[0])
    for name in ("reader", "judge", "workers", "deadline_min", "min_gain", "guard_margin",
                 "config", "candidate_config"):
        assert tokens[tokens.index("--" + name.replace("_", "-")) + 1] == str(getattr(first, name))
    for name in ("panel", "guard", "cache_root", "baseline", "output", "lme_data", "locomo_data"):
        assert tokens[tokens.index("--" + name.replace("_", "-")) + 1] == getattr(first, name).name
    assert str(first.output.parent) not in commands[0]


def test_cold_screen_alternates_arm_order_by_question(screen):
    run, calls, _ = screen
    run()
    assert calls == ["baseline-q0", "candidate-q0", "candidate-q1", "baseline-q1",
                     "baseline-q2", "candidate-q2", "candidate-q3", "baseline-q3"]
