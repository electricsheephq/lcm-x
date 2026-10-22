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


def test_flag_load_and_manifest(tmp_path, monkeypatch):
    questions, _, _, excluded = datasets.load(locomo(tmp_path), "locomo", None, include_adversarial=True)
    assert [q.qid for q in questions] == ["fixture:0", "fixture:1"]
    assert questions[1].gold == "" and questions[1].is_abstention and excluded == {}
    args = cli(tmp_path, "answer", "--include-adversarial", "--delivery", "single")
    args.output.mkdir()
    store = SimpleNamespace(generation="synthetic", engine=lambda q: nullcontext((None, None)))
    monkeypatch.setattr(run.delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run, "versions", lambda: {})
    job = run.run(args, None, questions, "dataset-hash", "ids-hash", excluded, store)
    next(job)
    job.close()
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert manifest["include_adversarial"] is True and manifest["excluded"] == {}


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
    prompt = path / "harness/call/prompt.txt"
    prompt.parent.mkdir(parents=True)
    evidence = evidence or f"[2025-01-01] {'Ada' if label == 'speaker' else 'user'}: PRIVATE_EVIDENCE"
    prompt.write_text("Instructions\n\nHistory evidence:\n" + evidence + "\nQuestion: PRIVATE_QUESTION")
    (path / "run-manifest.json").write_text(json.dumps({"delivery": "single", "hit_label": label,
        "id_list_sha256": ids_hash, "question_ids": [qid]}))
    (path / "answers.jsonl").write_text(json.dumps({"qid": qid, "question": "PRIVATE_QUESTION",
        "hypothesis": "PRIVATE_ANSWER", "trace": [{"tool": "lcm_recall", "result": json.dumps({
            "hits": [{"store_id": 1}, {"store_id": hit}], "snippet": "PRIVATE_EVIDENCE"})}],
        "calls": [{"call_dir": "call", "prompt_tokens_estimate": 100}]}))
    return path


@pytest.mark.parametrize("failure", [None, "retrieval", "order", "labels", "qids", "ids_hash", "missing_prompt"])
def test_check_arms(tmp_path, failure):
    a0 = make_answers(tmp_path, "A0", "role")
    a1 = make_answers(tmp_path, "A1", hit=3 if failure == "retrieval" else 2,
                      qid="other-q" if failure == "qids" else "synthetic-q",
                      ids_hash="different" if failure == "ids_hash" else "same-hash",
                      evidence="[2025-01-01] assistant: PRIVATE_EVIDENCE" if failure == "labels" else None)
    if failure == "missing_prompt":
        source = json.loads((a1 / "answers.jsonl").read_text())
        source["calls"][0]["call_dir"] = "absent"
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
