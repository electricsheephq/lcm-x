"""Synthetic D4 replacement runs; no bridge or provider is executed."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.tools import d4_health_rerun as d
from bench.tools import d8_guard as g


FIXTURES = Path(__file__).parent / "fixtures" / "d4_health_rerun"
HEALTH = json.loads((FIXTURES / "report.json").read_text())["evaluations"][0]["recallHealth"]
BAD = {**HEALTH, "timeout": True, "degradedReason": "timeout"}
LOCAL = {**HEALTH, "coverage": {"fts": "ok", "summary": "full_approx", "chunk": "full"},
         "degraded": True, "degradedReason": "summary arm coverage full_approx: fixture"}


def fixture(n=1, config="OFF"):
    report = json.loads((FIXTURES / "report.json").read_text())
    checkpoint = json.loads((FIXTURES / "checkpoint.json").read_text())
    row, question = report["evaluations"][0], checkpoint["questions"]["q000"]
    report["recallConfig"] = checkpoint["recallConfig"] = config
    report["evaluations"], checkpoint["questions"] = [], {}
    for i in range(n):
        qid = f"q{i:03d}"
        report["evaluations"].append({**copy.deepcopy(row), "questionId": qid})
        checkpoint["questions"][qid] = {**copy.deepcopy(question), "questionId": qid,
                                        "containerTag": qid + "-X"}
    return report, checkpoint


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "data/runs/X"
    run_dir.mkdir(parents=True)
    state = SimpleNamespace(run=run_dir, out=tmp_path / "health-merged/report.json",
                            receipt=tmp_path / "health-receipt.json", calls=[])

    def apply(report, checkpoint, *, replacement_health=HEALTH, mutate=None, score=1):
        rp, ck = run_dir / "report.json", run_dir / "checkpoint.json"
        rp.write_text(json.dumps(report))
        ck.write_text(json.dumps(checkpoint))
        state.before = (rp.read_bytes(), ck.read_bytes())

        def execute(command, cwd, replacement_dir):
            state.calls.append(command)
            ids = command[command.index("--questions") + 1].split(",")
            rr, cc = copy.deepcopy(report), copy.deepcopy(checkpoint)
            rr["runId"] = cc["runId"] = rr["dataSourceRunId"] = cc["dataSourceRunId"] = "X-health-rerun1"
            rr["evaluations"] = [r for r in rr["evaluations"] if r["questionId"] in ids]
            cc["questions"] = {q: cc["questions"][q] for q in ids}
            for row in rr["evaluations"]:
                row.update(recallHealth=copy.deepcopy(replacement_health), score=score,
                           label="correct" if score else "incorrect", hypothesis="replacement")
            for question in cc["questions"].values():
                question["phases"]["evaluate"].update(score=score, label="correct" if score else "incorrect")
            if mutate:
                mutate(rr, cc)
            replacement_dir.mkdir()
            (replacement_dir / "report.json").write_text(json.dumps(rr))
            (replacement_dir / "checkpoint.json").write_text(json.dumps(cc))

        result = d.process(rp, ck, state.out, state.receipt, report["recallConfig"], g,
                           ["bun", "run", "/bridge/src/index.ts"], execute=execute)
        merged = json.loads(state.out.read_text()) if state.out.exists() else None
        return result, merged

    state.apply = apply
    return state


def test_healthy_pass_through(run):
    receipt, merged = run.apply(*fixture())
    assert receipt["status"] == "COMPLETE"
    assert run.calls == []
    assert merged["evaluations"][0]["healthAttempt"] == 1


def test_replacement_score_and_health(run):
    report, cp = fixture()
    report["evaluations"][0].update(recallHealth=BAD, score=0, label="incorrect")
    original = copy.deepcopy(report["evaluations"][0])
    receipt, merged = run.apply(report, cp)
    row = merged["evaluations"][0]
    assert row["score"] == 1
    assert row["recallHealth"] == HEALTH
    assert row["hypothesis"] == "replacement"
    assert row["healthAttempt"] == 2
    assert row["replacedFrom"] == hashlib.sha256(d.canonical(original)).hexdigest()
    assert receipt["still_unhealthy"] == 0
    assert merged["summary"] == {"totalQuestions": 1, "correctCount": 1, "accuracy": 1.0}
    assert len(run.calls) == 1


def test_persistent_unhealthy_counts_against(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    receipt, merged = run.apply(report, cp, replacement_health=BAD)
    row = merged["evaluations"][0]
    assert (row["score"], row["label"], row["healthUnhealthy"]) == (0, "incorrect", True)
    assert receipt["degradedReasons"] == {"timeout": 1}


def test_abstention_exempt(run):
    report, cp = fixture()
    report["evaluations"][0].update(questionId="q_abs", recallHealth=None)
    cp["questions"] = {"q_abs": cp["questions"].pop("q000")}
    receipt, merged = run.apply(report, cp)
    assert receipt["status"] == "COMPLETE"
    assert not merged["evaluations"][0]["healthUnhealthy"]
    assert run.calls == []


def test_off_expected_degraded_healthy(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = {
        **HEALTH, "degraded": True, "degradedReason": "semantic retrieval is disabled"}
    receipt, _ = run.apply(report, cp)
    assert receipt["status"] == "COMPLETE"
    assert run.calls == []


@pytest.mark.parametrize("n,status", [(100, "COMPLETE"), (99, "INCONCLUSIVE")])
def test_one_percent_boundary(run, n, status):
    report, cp = fixture(n)
    report["evaluations"][0]["recallHealth"] = BAD
    receipt, _ = run.apply(report, cp, replacement_health=BAD)
    assert receipt["still_unhealthy"] == 1
    assert receipt["status"] == status


def test_config_mismatch(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    with pytest.raises(ValueError, match="STOP: replacement identity mismatch: recallConfig"):
        run.apply(report, cp, mutate=lambda rr, cc: cc.update(recallConfig="LOCAL"))


def test_reused_datasource(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    with pytest.raises(ValueError, match="STOP: replacement dataSourceRunId reuse/mismatch"):
        run.apply(report, cp, mutate=lambda rr, cc: cc.update(dataSourceRunId="X"))


def test_missing_completed_health(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = None
    receipt, merged = run.apply(report, cp)
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["reason"] == "completed_question_missing_health"
    assert merged is None
    assert run.calls == []


def test_original_bytes_unchanged(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    receipt, _ = run.apply(report, cp)
    assert run.before == ((run.run / "report.json").read_bytes(),
                          (run.run / "checkpoint.json").read_bytes())
    for path, sha in receipt["inputs"].items():
        assert d.digest(path) == sha


def test_exact_command_builder():
    report, _ = fixture()
    assert d.replacement_command(report, ["z", "a"], ["bun", "run", "bridge.ts"], 2) == [
        "bun", "run", "bridge.ts", "run", "-p", "hermes-lcm", "-b", "longmemeval",
        "-r", "X-health-rerun1", "--questions", "a,z", "-j", "judge", "-m", "reader",
        "--concurrency-answer", "2", "--concurrency-evaluate", "2"]


@pytest.mark.parametrize("health,healthy", [
    (LOCAL, True),
    ({**LOCAL, "coverage": {"fts": "ok", "summary": "full_approx"}}, False),
], ids=["disclosed-full-approx", "missing-chunk"])
def test_local_health(run, health, healthy):
    report, cp = fixture(config="LOCAL")
    report["evaluations"][0]["recallHealth"] = health
    receipt, merged = run.apply(report, cp, replacement_health=health)
    assert receipt["status"] == ("COMPLETE" if healthy else "INCONCLUSIVE")
    assert receipt["still_unhealthy"] == (0 if healthy else 1)
    assert len(run.calls) == (0 if healthy else 1)
    assert merged["evaluations"][0]["healthUnhealthy"] is (not healthy)


def test_failed_answer_not_retried(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    cp["questions"]["q000"]["phases"]["answer"]["status"] = "failed"
    _, merged = run.apply(report, cp)
    assert run.calls == []
    assert merged["evaluations"][0]["score"] == 0


def test_aborted_no_replacement(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    cp["status"] = "failed"
    receipt, merged = run.apply(report, cp)
    assert receipt["status"] == "INCOMPLETE"
    assert run.calls == []
    assert merged is None


def test_no_second_attempt(run):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    run.apply(report, cp)
    # A new destination does not authorize a second replacement run.
    run.out = run.out.with_name("another-report.json")
    run.receipt = run.receipt.with_name("another-receipt.json")
    with pytest.raises(ValueError, match="STOP: replacement run already exists; no second attempt"):
        run.apply(report, cp)


@pytest.mark.parametrize("health", [BAD, HEALTH], ids=["unhealthy", "healthy"])
def test_judge_fallback_counts_against(run, health):
    report, cp = fixture()
    report["evaluations"][0].update(recallHealth=health, explanation="Failed to parse judge response")
    _, merged = run.apply(report, cp)
    assert run.calls == []
    assert merged["evaluations"][0]["score"] == 0
    assert merged["evaluations"][0]["_failed"]


def test_guard_from_same_tree(tmp_path):
    root = Path(d.__file__).resolve().parents[2]
    assert d.guard_from(root) is g
    with pytest.raises(ValueError, match="STOP: instruments import outside bound tree"):
        d.guard_from(tmp_path)


@pytest.mark.parametrize("role", ["answererProvenance", "judgeProvenance"])
@pytest.mark.parametrize("field,value", [
    ("model", "executing-model"), ("configuredModel", "configured-model"),
    ("provider", "openai"), ("reasoningEffort", "high"), ("transport", "codex-cli"),
    ("transportVersion", "1.0"), ("modelExplicit", True), ("serviceTier", "priority"),
    ("isolated", True), ("modelPinSource", "explicit-cli-argv"),
    ("eventUsageCapture", "codex-jsonl"), ("eventModelField", "not-emitted-by-codex-jsonl"),
    ("tokenizerModel", "tokenizer"), ("executionIdentityCount", 1),
    ("mixedExecutionIdentity", False), ("callLedgerComplete", True),
])
@pytest.mark.parametrize("source", ["report", "checkpoint"])
def test_execution_provenance_mismatch(run, role, field, value, source):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    report[role] = {field: value}
    cp[role] = {field: value}

    def change(rr, cc):
        (rr if source == "report" else cc)[role][field] = "changed"

    with pytest.raises(ValueError, match=r"STOP: .*" + role + r"\." + field):
        run.apply(report, cp, mutate=change)
    assert not run.out.exists()


@pytest.mark.parametrize("field", [
    "datasetPath", "datasetSha256", "promptVersion", "readerPromptVersion",
    "scoringPath", "answerPresentationMode",
])
def test_dataset_scorer_pin_mismatch(run, field):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    report[field] = cp[field] = "original"
    with pytest.raises(ValueError, match="STOP: .*" + field):
        run.apply(report, cp, mutate=lambda rr, cc: cc.update({field: "changed"}))


@pytest.mark.parametrize("recorded_attempt", ["first", "replacement"])
def test_provenance_present_on_only_one_attempt(run, recorded_attempt):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    if recorded_attempt == "first":
        report["answererProvenance"] = {"model": "reader"}

    def change(rr, cc):
        rr.pop("answererProvenance", None)
        if recorded_attempt == "replacement":
            cc["answererProvenance"] = {"model": "reader"}

    with pytest.raises(ValueError, match=r"STOP: .*answererProvenance\.model"):
        run.apply(report, cp, mutate=change)


def test_matching_provenance_with_subset_counters(run):
    report, cp = fixture(2)
    report["evaluations"][0]["recallHealth"] = BAD
    report["answererProvenance"] = cp["answererProvenance"] = {
        "model": "reader", "reasoningEffort": "high", "callCount": 2, "retryCount": 1}
    for field in ("questionSetSha256", "normalizedInputsSha256", "loaderOrderSha256"):
        report[field] = cp[field] = "full-population"

    def subset(rr, cc):
        for doc in (rr, cc):
            doc["answererProvenance"].update(callCount=1, retryCount=0)
            for field in ("questionSetSha256", "normalizedInputsSha256", "loaderOrderSha256"):
                doc[field] = "selected-population"

    receipt, merged = run.apply(report, cp, mutate=subset)
    assert receipt["status"] == "COMPLETE"
    assert merged["answererProvenance"]["model"] == "reader"


@pytest.mark.parametrize("marker", ["report-id", "checkpoint-id", "checkpoint-marker", "merged-report"])
def test_replacement_input_stops_before_command(run, monkeypatch, marker):
    report, cp = fixture()
    report["evaluations"][0]["recallHealth"] = BAD
    if marker.endswith("-id"):
        (report if marker == "report-id" else cp)["runId"] += "-health-rerun1"
    else:
        (cp if marker == "checkpoint-marker" else report)["healthReplacement"] = {}

    def no_command(*args, **kwargs):
        pytest.fail("replacement command built for replacement input")

    monkeypatch.setattr(d, "replacement_command", no_command)
    with pytest.raises(ValueError, match="STOP: input is already a health replacement"):
        run.apply(report, cp)
    assert run.calls == []
    assert not run.out.exists()


@pytest.mark.parametrize("healthy", [True, False])
def test_writer_category_aggregates_follow_merged_rows(run, healthy):
    pair = FIXTURES.parent / "d8_writer_pair/a"
    report = json.loads((pair / "report.json").read_text())
    cp = json.loads((pair / "checkpoint.json").read_text())
    report["runId"] = cp["runId"] = "X"
    row = report["evaluations"][1]
    row.update(score=int(not healthy), label="incorrect" if healthy else "correct", recallHealth=BAD)
    report["byQuestionType"]["knowledge-update"].update(correct=int(not healthy), accuracy=int(not healthy))
    def timings(rr, cc):
        rr["evaluations"][0].update(searchDurationMs=25, answerDurationMs=50, totalDurationMs=100,
                                    retrievalMetrics={key: 1 for key in
                                        ("hitAtK", "precisionAtK", "recallAtK", "f1AtK", "mrr", "ndcg", "k")})

    _, merged = run.apply(report, cp, replacement_health=HEALTH if healthy else BAD, mutate=timings)
    category = merged["byQuestionType"]["knowledge-update"]
    assert category["correct"] == int(healthy)
    assert category["accuracy"] == float(healthy)
    assert category["total"] == 1
    assert category["latency"]["search"] == {
        "min": 25, "max": 25, "mean": 25, "median": 25, "p95": 25, "p99": 25, "stdDev": 0, "count": 1}
    assert category["retrieval"]["hitAtK"] == 1
    assert "latency" not in merged
    assert merged["aggregatesRemoved"] == ["latency"]
