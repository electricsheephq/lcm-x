"""Transient-call regressions through the answer writer and real event parser; no providers."""
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import common, datasets, delivery, dev_panel, judges, models, run
from bench.fmp.tests.test_995 import locomo, make_answers
from bench.fmp.tests.test_995_r1 import base_module
from bench.fmp.tests.test_fmp import Scripted, codex_backend
from bench.fmp.tests.test_switches import cli

REPO = Path(run.__file__).resolve().parents[2]
BASE = "6926223831ebd995eca53dadc70a32166c066068"


def written(tmp_path, monkeypatch, failures=1, label="role", marker="transient transport", event="turn.failed", exit_code=1):
    common.load_lcm(REPO)
    owner = tmp_path / "owner/.codex"
    owner.mkdir(parents=True)
    (owner / "auth.json").write_text("{}")
    monkeypatch.setenv("HOME", str(owner.parent))
    backend, _ = codex_backend(tmp_path)
    fake, sent, sleeps, recalls = backend.runner, [], [], []

    def transport(*args, **kwargs):
        proc = fake(*args, **kwargs)
        sent.append(fake.stdins[-1])
        if len(sent) <= failures:
            events = [json.loads(line) for line in proc.stdout.splitlines()]
            events.append({"type": event, "error": {"message": marker}})
            return subprocess.CompletedProcess(proc.args, exit_code, "\n".join(map(json.dumps, events)), "")
        return proc

    backend.runner = transport
    monkeypatch.setattr(models.time, "sleep", sleeps.append)
    monkeypatch.setattr(delivery, "dispatch", lambda *a, **kw: recalls.append(1) or '{"hits":[{"store_id":1}]}')
    monkeypatch.setattr(delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run, "versions", lambda: {})
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=lambda ids: {1: {
        "store_id": 1, "role": "user", "observed_at": None, "content": "Synthetic teal key"}}))
    store_path = tmp_path / "store"
    store_path.mkdir()
    (store_path / "speakers.json").write_text('{"1":"Ada"}')
    store = SimpleNamespace(generation="fixture", engine=lambda _: nullcontext((engine, None)), directory=lambda _: store_path)
    args = cli(tmp_path, "answer", "--delivery", "single", "--store-text", "plain", "--hit-label", label)
    args.output = backend.root.parent
    try:
        job = run.run(args, backend, [q], "dataset", "ids", {}, store)
        work, source, _ = next(job)
        row = work(source[0])
        job.send(row)
        with pytest.raises(StopIteration):
            job.send(None)
        assert json.loads((args.output / "answers.jsonl").read_text()) == row
        return args.output, row, sent, sleeps, recalls
    finally:
        backend.close()


@pytest.mark.parametrize("event,exit_code", [("turn.failed", 1), ("turn.failed", 0), ("error", 0), ("other", 1)])
def test_reader_recovers_same_receipts_and_pair_passes(tmp_path, monkeypatch, event, exit_code):
    paths = []
    for label in ("role", "speaker"):
        path, row, sent, sleeps, recalls = written(tmp_path / label, monkeypatch, label=label, event=event, exit_code=exit_code)
        paths.append(path)
        assert row["status"] == "ok" and row["attempts"] == 2
        assert row["attempt_errors"] == ["codex_failed"]
        assert sleeps == [5] and recalls == [1] and sent[0] == sent[1]
        evidence = sent[0].split("History evidence:\n")[1].rsplit("\nQuestion: ", 1)[0]
        assert row["evidence_sha256"] == common.sha(evidence.encode())
    assert run.check_arms(paths, tmp_path / "check.json")["passed"]


def test_exhausted_reader_preserves_receipts_and_fails_closed(tmp_path, monkeypatch):
    path, row, sent, sleeps, recalls = written(tmp_path / "failed", monkeypatch, failures=3)
    other = make_answers(tmp_path, "A1", qid=row["qid"], hit=1, ids_hash="ids")
    assert row["status"] == "error" and row["attempts"] == 3
    assert row["attempt_errors"] == ["codex_failed"] * 3
    assert len(sent) == 3 and len(set(sent)) == 1 and sleeps == [5, 20] and recalls == [1]
    assert row["trace"] and row["evidence_labels"]["role"] == 1
    report = run.check_arms([path, other], tmp_path / "check.json")
    assert not report["passed"] and report["arms"][0]["missing_labels"] == 0
    assert report["arms"][0]["reason"] == f"1 answer rows errored: {row['qid']}"


@pytest.mark.parametrize("marker", ["quota exceeded", "usage limit reached", "out of credits"])
def test_reader_limit_markers_never_retry(tmp_path, monkeypatch, marker):
    _, row, sent, sleeps, _ = written(tmp_path, monkeypatch, failures=3, marker=marker)
    assert row["status"] == "not_run" and row["reason"] == "budget_stopped"
    assert len(sent) == 1 and not sleeps
    assert row["trace"] and row["evidence_sha256"]


@pytest.mark.parametrize("status,reason", [("error", "errored"), ("not_run", "not run")])
@pytest.mark.parametrize("receipts", [True, False])
def test_check_arms_error_reason_precedes_legacy_reason(tmp_path, status, reason, receipts):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    for path in paths:
        file = path / "answers.jsonl"
        row = json.loads(file.read_text())
        row["status"] = status
        if not receipts:
            for key in ("trace", "evidence_labels", "evidence_sha256"):
                row.pop(key)
        file.write_text(json.dumps(row))
    report = run.check_arms(paths, tmp_path / "check.json")
    assert not report["passed"]
    assert all(a["reason"] == f"1 answer rows {reason}: synthetic-q" for a in report["arms"])


class FailingJudge(Scripted):
    def __init__(self, failures, marker=None):
        super().__init__([{"content": "CORRECT"}])
        self.failures, self.marker, self.sent = failures, marker, []

    def call(self, model, messages, records, **kwargs):
        self.sent.append(json.dumps(messages))
        if len(self.sent) <= self.failures:
            records.append({"status": "error", "error": "usage_limit" if self.marker else "codex_failed"})
            if self.marker:
                raise models.BudgetStopped()
            raise RuntimeError("codex_call_failed")
        return super().call(model, messages, records, **kwargs)


@pytest.mark.parametrize("failures", [1, 3])
def test_judge_bounded_retry(tmp_path, monkeypatch, failures):
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    client, sleeps = FailingJudge(failures), []
    monkeypatch.setattr(models.time, "sleep", sleeps.append)
    result, _ = judges.judge(q, "teal", client, models=[models.CODEX_DEFAULT_JUDGE])
    result = result[models.CODEX_DEFAULT_JUDGE]
    assert result["status"] == ("ok" if failures == 1 else "judge_error")
    assert result["attempts"] == min(failures + 1, 3)
    assert result["attempt_errors"] == ["codex_failed"] * failures
    assert sleeps == ([5] if failures == 1 else [5, 20]) and len(set(client.sent)) == 1


def test_judge_quota_stop_not_retried(tmp_path, monkeypatch):
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    client, sleeps = FailingJudge(3, marker=True), []
    monkeypatch.setattr(models.time, "sleep", sleeps.append)
    with pytest.raises(models.BudgetStopped):
        judges.judge(q, "teal", client, models=[models.CODEX_DEFAULT_JUDGE])
    assert len(client.sent) == 1 and sleeps == []


def test_no_failure_flag_off_output_bytes_match_main(tmp_path, monkeypatch):
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    baseline_delivery, baseline_judges = base_module("delivery"), base_module("judges")
    for module in (baseline_delivery, baseline_judges):
        name = "delivery" if module is baseline_delivery else "judges"
        source = subprocess.check_output(["git", "show", f"{BASE}:bench/fmp/{name}.py"], cwd=REPO)
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=lambda ids: {1: {
        "role": "user", "observed_at": None, "content": "Synthetic teal key"}}))
    rows = []
    for module, judge_module in ((baseline_delivery, baseline_judges), (delivery, judges)):
        monkeypatch.setattr(module, "single", lambda *a, **kw: ("[unknown] user: Synthetic teal key", '{"hits":[{"store_id":1}]}'))
        row = module.answer(q, "single", Scripted([{"content": "teal"}]), models.CODEX_DEFAULT_READER, engine,
                            hit_label="role", store_text="plain")
        rows.append(judge_module.judge_row(row, q, Scripted([{"content": "CORRECT"}]), models=[models.CODEX_DEFAULT_JUDGE]))
    assert json.dumps(rows[0]).encode() == json.dumps(rows[1]).encode()


def test_dev_panel_keeps_same_151_ids():
    panels = [json.loads((dev_panel.PANELS / name).read_bytes()) for name in ("dev100.json", "guard.json")]
    assert len({item["id"] for panel in panels for item in panel["items"]}) == 151
    for name in ("dev100.json", "guard.json"):
        assert (dev_panel.PANELS / name).read_bytes() == subprocess.check_output(
            ["git", "show", f"{BASE}:bench/fmp/panels/{name}"], cwd=REPO)


def test_successful_check_arms_bytes_match_main(tmp_path):
    baseline = base_module("run")
    source = subprocess.check_output(["git", "show", f"{BASE}:bench/fmp/run.py"], cwd=REPO)
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    baseline.check_arms(paths, tmp_path / "before.json")
    run.check_arms(paths, tmp_path / "after.json")
    assert (tmp_path / "before.json").read_bytes() == (tmp_path / "after.json").read_bytes()
