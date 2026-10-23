"""#995 harness fixtures: category 5 and pre-judge arm integrity, no live calls."""
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import common, datasets, dev, judges, run, score
from bench.fmp.tests.test_r3 import MODEL, arm, row
from bench.fmp.tests.test_switches import cli


def locomo(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps([{"sample_id": "fixture", "conversation": {
        "speaker_a": "Ada", "speaker_b": "Bo", "session_1_date_time": "2025-01-01",
        "session_1": [{"speaker": "Ada", "text": "Synthetic teal key"}]}, "qa": [
        {"category": 1, "question": "Synthetic key?", "answer": "teal"},
        {"category": 5, "question": "Synthetic unknown?", "adversarial_answer": "DECOY"}]}]))
    return path


def test_default_loader_identity(tmp_path):
    path = locomo(tmp_path)
    questions, data_hash, ids_hash, excluded = datasets.load(path, "locomo", None)
    assert [q.qid for q in questions] == ["fixture:0"]
    assert questions[0].gold == "teal"
    assert data_hash == common.sha(path.read_bytes()) and ids_hash is None
    assert excluded == {"locomo_category_5_adversarial": 1}
    assert "include_adversarial" not in run.switches(cli(tmp_path))


@pytest.mark.parametrize("flag", [False, True])
def test_flag_load_and_manifest(tmp_path, monkeypatch, flag):
    questions, _, _, excluded = datasets.load(locomo(tmp_path), "locomo", None, include_adversarial=flag)
    assert [q.qid for q in questions] == (["fixture:0", "fixture:1"] if flag else ["fixture:0"])
    if flag:
        assert questions[1].gold == "DECOY" and questions[1].is_abstention and excluded == {}
    args = cli(tmp_path, "answer", "--delivery", "single", *(["--include-adversarial"] if flag else []))
    args.output.mkdir()
    store = SimpleNamespace(generation="synthetic", engine=lambda q: nullcontext((None, None)))
    monkeypatch.setattr(run.delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run, "versions", lambda: {})
    job = run.run(args, None, questions, "dataset-hash", "ids-hash", excluded, store)
    next(job)
    job.close()
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert (manifest.get("include_adversarial") is True) == flag
    assert flag or "include_adversarial" not in manifest
    assert manifest["excluded"] == ({} if flag else {"locomo_category_5_adversarial": 1})
    args.command, args.include_adversarial = "judge", not flag
    with pytest.raises(ValueError, match="include_adversarial"):
        next(run.run(args, None, questions, "dataset-hash", "ids-hash", excluded, None))


@pytest.mark.parametrize("flag", [False, True])
def test_execute_passes_loader_flag(tmp_path, monkeypatch, flag):
    args = cli(tmp_path, "answer", "--data", str(locomo(tmp_path)), *(["--include-adversarial"] if flag else []))
    monkeypatch.setattr(run, "commit", lambda _: "synthetic")
    monkeypatch.setattr(run, "load_lcm", lambda _: None)
    def load(*args, **kwargs):
        assert args[-1] is flag
        raise RuntimeError("fixture loader reached before any client")
    monkeypatch.setattr(datasets, "load", load)
    with pytest.raises(RuntimeError, match="fixture loader reached"):
        run.execute(args)


def test_dev_judge_bytes_identity():
    q = SimpleNamespace(dataset="locomo", qtype="5", question="Synthetic unknown?", gold="DECOY", is_abstention=True)
    encoded = json.dumps(dev.judge_messages(q, "I do not know"), ensure_ascii=False).encode()
    assert common.sha(encoded) == "9a23dd012449f60f858cd14646f09b9debda46c2de0395b2bda5efc7bbfacf72"


def test_category5_run_judging_and_score_reparse(tmp_path):
    q = SimpleNamespace(dataset="locomo", qtype="5", question="Synthetic unknown?", gold="DECOY", is_abstention=True)
    sent = []
    def call(model, messages, records, **kwargs):
        sent.append((messages, kwargs))
        return {"content": "yes"}
    result, _ = judges.judge(q, "I do not know", SimpleNamespace(call=call), models=[MODEL])
    assert result[MODEL]["correct"] is True
    assert sent[0][0] == dev.judge_messages(q, "I do not know")[0]
    assert "DECOY" not in json.dumps(sent)
    path = arm(tmp_path, "off", [row(0, qtype="5", judges=result), row(1, False, qtype="1")])
    manifest = path / "run-manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "include_adversarial": True}))
    report = score.score([path], tmp_path / "score")
    summary = next(iter(report["arms"].values()))
    assert summary["by_type"]["5"][MODEL] == 1
    assert summary["category_totals"]["all categories"]["scored_n"] == 2
    assert summary["category_totals"]["all categories"]["accuracy"][MODEL] == .5
    assert summary["category_totals"]["cats 1-4"]["scored_n"] == 1
    assert summary["category_totals"]["cats 1-4"]["accuracy"][MODEL] == 0
    assert "| off/agentic/" in (tmp_path / "score/report.md").read_text()
    assert "all categories" in (tmp_path / "score/report.md").read_text()


def test_score_refuses_mixed_adversarial_flag(tmp_path):
    paths = [arm(tmp_path, c, [row(0)]) for c in ("off", "local")]
    manifest = paths[1] / "run-manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "include_adversarial": True}))
    with pytest.raises(ValueError, match="include_adversarial"):
        score.score(paths, tmp_path / "score")


def make_answers(root, name, label="speaker", hit=2, qid="synthetic-q", ids_hash="same-hash", evidence=None):
    path = root / name
    path.mkdir()
    evidence = evidence or f"[2025-01-01] {'Ada' if label == 'speaker' else 'user'}: PRIVATE_EVIDENCE"
    role = int("assistant:" in evidence or "user:" in evidence)
    (path / "run-manifest.json").write_text(json.dumps({"delivery": "single", "hit_label": label, "store_text": "plain", "dataset_sha256": "dataset",
        "id_list_sha256": ids_hash, "question_ids": [qid]}))
    (path / "answers.jsonl").write_text(json.dumps({"qid": qid, "question": "PRIVATE_QUESTION",
        "hypothesis": "PRIVATE_ANSWER", "evidence_labels": {"role": role, "speaker": 1-role, "summary": 0},
        "evidence_sha256": common.sha(evidence.encode()), "trace": [{"tool": "lcm_recall", "result": json.dumps({
            "hits": [{"store_id": 1}, {"store_id": hit}], "snippet": "PRIVATE_EVIDENCE"})}],
        "calls": [{"call_dir": "call", "prompt_tokens_estimate": 100}]}))
    return path


@pytest.mark.parametrize("failure", [None, "retrieval", "order", "labels", "qids", "ids_hash", "missing_receipt"])
def test_check_arms(tmp_path, failure):
    a0 = make_answers(tmp_path, "A0", "role")
    a1 = make_answers(tmp_path, "A1", hit=3 if failure == "retrieval" else 2,
                      qid="other-q" if failure == "qids" else "synthetic-q",
                      ids_hash="different" if failure == "ids_hash" else "same-hash",
                      evidence="[2025-01-01] assistant: PRIVATE_EVIDENCE" if failure == "labels" else None)
    if failure == "missing_receipt":
        source = json.loads((a1 / "answers.jsonl").read_text())
        source.pop("evidence_labels")
        (a1 / "answers.jsonl").write_text(json.dumps(source))
    if failure == "order":
        source = json.loads((a1 / "answers.jsonl").read_text())
        source["trace"][0]["result"] = {"hits": [{"store_id": 2}, {"store_id": 1}]}
        (a1 / "answers.jsonl").write_text(json.dumps(source))
    before = {p: p.read_bytes() for directory in (a0, a1) for p in directory.rglob("*") if p.is_file()}
    report = run.check_arms([a0, a1], tmp_path / "check.json")
    assert report["passed"] is (failure is None)
    assert report["retrieval_differing_qids"] == int(failure in ("retrieval", "order"))
    assert report["arms"][0]["role_lines"] == 1
    assert report["arms"][1]["role_lines"] == int(failure == "labels")
    assert report["arms"][0]["prompt_tokens_estimate"] == {"median": 100, "p90": 100}
    assert all(p.read_bytes() == data for p, data in before.items())
    output = json.loads((tmp_path / "check.json").read_text())
    assert output == report
    assert "PRIVATE_" not in json.dumps(output) and "synthetic-q" not in json.dumps(output)
    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)
    assert not {"question", "answer", "hypothesis", "evidence", "gold", "snippet", "text"} & set(keys(output))


@pytest.mark.parametrize("bad", [False, True])
def test_check_arms_cli(tmp_path, bad):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1", hit=3 if bad else 2)]
    repo = Path(run.__file__).resolve().parents[2]
    env = {**os.environ, "FMP_PYTHON": os.sys.executable}
    result = subprocess.run(["bash", "bench/fmp/run.sh", "check-arms", "--runs", *map(str, paths)],
                            cwd=repo, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == int(bad)
    assert "PRIVATE_" not in result.stdout + result.stderr
    reports = list(tmp_path.glob("check-arms-*.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["passed"] is (not bad)


def test_check_arms_even_median(tmp_path):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    for path in paths:
        payload = json.loads((path / "answers.jsonl").read_text())
        payload["calls"].append({"prompt_tokens_estimate": 300})
        (path / "answers.jsonl").write_text(json.dumps(payload))
    report = run.check_arms(paths, tmp_path / "check.json")
    assert report["arms"][0]["prompt_tokens_estimate"] == {"median": 200, "p90": 300}


def test_fix1_check_arms_storeless_hit(tmp_path):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    for path in paths:
        row = json.loads((path / "answers.jsonl").read_text())
        row["trace"][0]["result"] = {"hits": [
            dict(kind="retraction", snippet="[Withdrawn]", withdrawn_store_id=1, cites=[2]),
        ]}
        (path / "answers.jsonl").write_text(json.dumps(row))
    report = run.check_arms(paths, tmp_path / "check.json")
    assert report["passed"]
    assert report["retrieval_differing_qids"] == 0
    row["trace"][0]["result"]["hits"][0]["store_id"] = 3
    (paths[1] / "answers.jsonl").write_text(json.dumps(row))
    assert run.check_arms(paths, tmp_path / "different.json")["retrieval_differing_qids"] == 1
