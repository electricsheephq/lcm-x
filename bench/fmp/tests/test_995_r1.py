"""Fix-round regressions through production writers; synthetic inputs, no models."""
import dataclasses
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bench.fmp import closeout, common, datasets, delivery, dev_panel, judges, run, score
from bench.fmp.tests.test_995 import locomo, make_answers
from bench.fmp.tests.test_fmp import codex_backend
from bench.fmp.tests.test_r3 import MODEL, arm, row
from bench.fmp.tests.test_switches import cli

BASE = "20891a980cae075cad177da7c19c549bac05e63c"
REPO = Path(run.__file__).resolve().parents[2]


def base_module(name):
    module = ModuleType("bench.fmp.baseline_" + name)
    module.__package__ = "bench.fmp"
    module.__file__ = str(REPO / "bench/fmp" / (name + ".py"))
    source = subprocess.check_output(["git", "show", f"{BASE}:bench/fmp/{name}.py"], cwd=REPO)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def test_frozen_fingerprint_and_judge_boundary(tmp_path):
    raw = json.loads(locomo(tmp_path).read_text())[0]
    for q, qa in zip(datasets.normalize_locomo(raw, adversarial=True), raw["qa"], strict=True):
        old = dataclasses.replace(q, gold=str(qa.get("answer", qa.get("adversarial_answer", ""))))
        assert dev_panel.fingerprint(q) == dev_panel.fingerprint(old)
        if q.qtype == "5":
            sent = []
            client = SimpleNamespace(call=lambda model, messages, records, **kw: sent.append(messages) or {"content": "yes"})
            judges.judge(q, "unknown", client, models=[MODEL])
            assert "DECOY" not in json.dumps(sent)


def written_answer(tmp_path, monkeypatch, label, fallback=False, implementation=run):
    common.load_lcm(REPO)
    monkeypatch.setattr(implementation.delivery, "schemas", lambda _: [])
    owner = tmp_path / "owner/.codex"
    owner.mkdir(parents=True)
    (owner / "auth.json").write_text("{}")
    monkeypatch.setenv("HOME", str(owner.parent))
    backend, _ = codex_backend(tmp_path)
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=lambda ids: {1: {
        "store_id": 1, "role": "user", "observed_at": None, "content": "Synthetic teal key"}}))
    monkeypatch.setattr(delivery, "dispatch", lambda *a, **k: json.dumps({"hits": [{"store_id": 1}]}))
    monkeypatch.setattr(delivery, "schemas", lambda _: [])
    monkeypatch.setattr(implementation, "versions", lambda: {})
    monkeypatch.setattr(implementation, "commit", lambda _: "fixture-commit")
    monkeypatch.setattr(implementation, "utc", lambda: "fixture-time")
    args = cli(tmp_path, "answer", "--delivery", "single", "--store-text", "plain", "--hit-label", label)
    args.output = backend.root.parent
    store_path = tmp_path / "store"
    store_path.mkdir()
    (store_path / "speakers.json").write_text(json.dumps({} if fallback else {"1": "Ada"}))
    store = SimpleNamespace(generation="fixture", engine=lambda _: nullcontext((engine, None)), directory=lambda _: store_path)
    try:
        job = implementation.run(args, backend, [q], "dataset", "ids", {}, store)
        work, source, _ = next(job)
        answer = work(source[0])
        job.send(answer)
        with pytest.raises(StopIteration):
            job.send(None)
        assert answer["status"] == "ok"
        assert backend.runner.stdins and not list(args.output.rglob("prompt.txt"))
        assert json.loads((args.output / "answers.jsonl").read_text()) == answer
        return args.output, answer, backend.runner.stdins[0]
    finally:
        backend.close()


@pytest.mark.parametrize("fallback", [False, True])
def test_real_harness_write_path(tmp_path, monkeypatch, fallback):
    a0, left, prompt = written_answer(tmp_path / "a0", monkeypatch, "role")
    a1, right, _ = written_answer(tmp_path / "a1", monkeypatch, "speaker", fallback)
    evidence = prompt.split("History evidence:\n")[1].rsplit("\nQuestion: ", 1)[0]
    assert left["evidence_labels"] == {"role": 1, "speaker": 0, "summary": 0}
    assert left["evidence_sha256"] == common.sha(evidence.encode())
    report = run.check_arms([a0, a1], tmp_path / "check.json")
    assert report["retrieval_differing_qids"] == 0
    assert report["passed"] is (not fallback)
    assert [a["arm"] for a in report["arms"]] == ["A0", "A1"]


@pytest.mark.parametrize("failure", ["two_role", "two_speaker", "store", "dataset", "empty_role", "empty_speaker", "legacy"])
def test_declared_modes_and_dataset_bytes(tmp_path, failure):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    manifest_path = paths[1] / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "two_role":
        manifest["hit_label"] = "role"
    if failure == "two_speaker":
        p = paths[0] / "run-manifest.json"
        m = json.loads(p.read_text())
        p.write_text(json.dumps({**m, "hit_label": "speaker"}))
    if failure == "store":
        manifest["store_text"] = "speaker-prefix"
    if failure == "dataset":
        manifest["dataset_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    for path in paths:
        p = path / "answers.jsonl"
        payload = json.loads(p.read_text())
        if failure == "legacy":
            payload.pop("evidence_labels")
            payload.pop("evidence_sha256")
        elif failure == "empty_role" and path == paths[0] or failure == "empty_speaker" and path == paths[1]:
            payload["evidence_labels"] = {"role": 0, "speaker": 0, "summary": 1}
        p.write_text(json.dumps(payload))
    report = run.check_arms(paths, tmp_path / "check.json")
    assert report["passed"] is False
    assert report["declared_modes_valid"] is (failure not in ("two_role", "two_speaker", "store"))
    assert report["dataset_hash_equal"] is (failure != "dataset")
    if failure == "legacy":
        assert all(a["label_check"] == "unavailable" and "label check unavailable" in a["reason"] for a in report["arms"])


def test_per_call_retrieval_identity(tmp_path):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    p = paths[1] / "answers.jsonl"
    payload = json.loads(p.read_text())
    payload["trace"] = [{"tool": "lcm_recall", "result": {"hits": [{"store_id": i}]}} for i in (1, 2)]
    p.write_text(json.dumps(payload))
    report = run.check_arms(paths, tmp_path / "check.json")
    assert report["retrieval_differing_qids"] == 1 and not report["passed"]


def test_output_path_and_write_error(tmp_path, monkeypatch, capsys):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    output = tmp_path / "custom.json"
    monkeypatch.setattr(run.sys, "argv", ["fmp", "check-arms", "--runs", *map(str, paths), "--output", str(output)])
    assert run.main() == 0 and output.exists()
    monkeypatch.setattr(run.sys, "argv", ["fmp", "check-arms", "--runs", *map(str, paths), "--output", str(output)])
    assert run.main() == 1
    assert str(output) in capsys.readouterr().err


def test_population_pairs_and_card_headline(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(closeout, "card_tool", lambda: recording_card(commands))
    paths = [arm(tmp_path, "off", [row(0, False, qtype="1"), row(1, True, qtype="5")]),
             arm(tmp_path, "local", [row(0, True, qtype="1"), row(1, False, qtype="5")])]
    for path in paths:
        p = path / "run-manifest.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), "include_adversarial": True}))
    report = score.score(paths, tmp_path / "report")
    pair = next(iter(report["paired"].values()))
    for population, expected in (("all categories", (1, 1, 1, 0)), ("cats 1-4", (1, 0, 1, -1)), ("category 5", (0, 1, 1, 1))):
        stat = pair["populations"][population]["per_judge"][MODEL]
        assert (stat["b"], stat["c"], stat["p"], stat["accuracy_delta"]) == expected
    text = (tmp_path / "report/RESULT.md").read_text()
    assert "cats 1-4" in text and "all categories" in text
    headlines = [c for c in commands if c[0] == "headline"]
    assert [(c[c.index("--value") + 1], c[c.index("--n") + 1]) for c in headlines] == [("0.0", "1"), ("1.0", "1")]
    assert all("cats 1-4" in c[c.index("--metric") + 1] for c in headlines)


def recording_card(commands):
    return SimpleNamespace(parser=lambda: SimpleNamespace(parse_args=lambda args: args),
                           execute=lambda args: commands.append(args) and 0)


def test_flag_off_score_and_card_bytes(tmp_path, monkeypatch):
    paths = [arm(tmp_path, "off", [row(0, qtype="1"), row(1, False, qtype="2")])]
    baseline_score, baseline_closeout = base_module("score"), base_module("closeout")
    monkeypatch.setattr(baseline_score, "result_card", baseline_closeout.result_card)
    base_commands, head_commands = [], []
    monkeypatch.setattr(baseline_closeout, "card_tool", lambda: recording_card(base_commands))
    monkeypatch.setattr(closeout, "card_tool", lambda: recording_card(head_commands))
    monkeypatch.setattr(closeout.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed-snapshot"))
    baseline_score.score(paths, tmp_path / "base")
    score.score(paths, tmp_path / "head")
    for name in ("report.json", "report.md", "adjudicate.jsonl", "RESULT.md"):
        assert (tmp_path / "base" / name).read_bytes() == (tmp_path / "head" / name).read_bytes(), name
    assert [c[:1] + c[2:] for c in base_commands] == [c[:1] + c[2:] for c in head_commands]


def test_flag_off_answer_and_manifest_identity(tmp_path, monkeypatch):
    baseline_run, baseline_delivery = base_module("run"), base_module("delivery")
    with monkeypatch.context() as old:
        old.setattr(baseline_delivery, "dispatch", lambda *a, **k: json.dumps({"hits": [{"store_id": 1}]}))
        old.setattr(baseline_run, "delivery", baseline_delivery)
        base_path, before, base_prompt = written_answer(tmp_path / "base", old, "role", implementation=baseline_run)
    path, after, prompt = written_answer(tmp_path / "head", monkeypatch, "role")
    assert prompt == base_prompt
    assert set(after) - set(before) == {"evidence_labels", "evidence_sha256"}
    for value in (before, after):
        value.pop("latency_s")
        for call in value["calls"]:
            for key in ("call_dir", "latency_s", "at"):
                call.pop(key, None)
    assert {k: v for k, v in after.items() if k not in ("evidence_labels", "evidence_sha256")} == before
    assert json.loads((base_path / "run-manifest.json").read_text()) == json.loads((path / "run-manifest.json").read_text())
