#!/usr/bin/env python3
"""A3: one fresh unhealthy-question replacement, immutable inputs, merged-only output."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_from(tree):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    module = importlib.import_module("bench.tools.d8_guard")
    if Path(tree).resolve() != root or not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("STOP: instruments import outside bound tree")
    return module


def replacement_command(report, ids, bridge, workers=4):
    if not ids or not 1 <= workers <= 4:
        raise ValueError("STOP: invalid replacement selection/workers")
    return [*bridge, "run", "-p", report["provider"], "-b", report["benchmark"],
            "-r", report["runId"] + "-health-rerun1", "--questions", ",".join(sorted(ids)),
            "-j", report["judge"], "-m", report["answeringModel"],
            "--concurrency-answer", str(workers), "--concurrency-evaluate", str(workers)]


def rows_by_id(report):
    rows = report["evaluations"]
    out = {r["questionId"]: r for r in rows}
    if len(rows) != len(out):
        raise ValueError("STOP: duplicate report question ids")
    return out


def phase_failed(question):
    return any(question["phases"].get(p, {}).get("status") != "completed"
               for p in ("ingest", "indexing", "search", "answer", "evaluate"))


def outside_recall_error(question, row, guard):
    return (phase_failed(question) or guard._failed(row) or
            row.get("explanation") == "Failed to parse judge response")


def check_provenance(first, second, guard):
    # These hashes describe the selected subset; counters describe its call volume.
    subset = {"questionSetSha256", "normalizedInputsSha256", "loaderOrderSha256"}
    fields = set(guard.IDENTITY_FIELDS) - subset
    for role in ("answererProvenance", "judgeProvenance"):
        fields.update(role + "." + key for doc in (*first, *second)
                      for key in (doc.get(role) or {}) if key not in {"callCount", "retryCount"})
    def values(docs, field):
        found = []
        for doc in docs:
            value = doc
            for key in field.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                found.append(canonical(value))
        return found
    for field in sorted(fields | subset):
        a, b = values(first, field), values(second, field)
        if (field not in subset and bool(a) != bool(b)) or len(set(a + b if field not in subset else a)) > 1 or len(set(b)) > 1:
            raise ValueError("STOP: replacement provenance mismatch: " + field)
        if a:
            parent, _, key = field.rpartition(".")
            (first[0].setdefault(parent, {}) if parent else first[0]).setdefault(key, json.loads(a[0]))


def category_stats(rows):
    def latency(values):
        v = sorted(x for x in values if x)
        n, mean = len(v), sum(v) / len(v) if v else 0
        return dict(min=v[0] if v else 0, max=v[-1] if v else 0, mean=math.floor(mean + .5),
                    median=v[len(v)//2] if v else 0, p95=v[min(math.floor(n*.95), n-1)] if v else 0,
                    p99=v[min(math.floor(n*.99), n-1)] if v else 0,
                    stdDev=math.floor(math.sqrt(sum((x-mean)**2 for x in v)/n) + .5) if v else 0, count=n)
    result = {}
    for kind in dict.fromkeys(r.get("questionType") for r in rows):
        group = [r for r in rows if r.get("questionType") == kind]
        correct = sum(r.get("score") == 1 for r in group)
        result[kind] = dict(total=len(group), correct=correct, accuracy=correct/len(group), latency={
            phase: latency(r.get(phase + "DurationMs", 0) for r in group) for phase in ("search", "answer", "total")})
        metrics = [r["retrievalMetrics"] for r in group if r.get("retrievalMetrics")]
        if metrics:
            result[kind]["retrieval"] = {key: sum(m[key] for m in metrics)/len(metrics)
                                        for key in ("hitAtK", "precisionAtK", "recallAtK", "f1AtK", "mrr", "ndcg")}
            result[kind]["retrieval"]["k"] = metrics[-1]["k"]
    return result


def process(report_path, checkpoint_path, output, receipt_path, config, guard,
            bridge, *, workers=4, execute=None):
    report_path, checkpoint_path, output, receipt_path = map(
        Path, (report_path, checkpoint_path, output, receipt_path))
    if output.exists() or receipt_path.exists():
        raise ValueError("STOP: output/receipt already exists")
    original = {str(p): digest(p) for p in (report_path, checkpoint_path)}
    r1, cp = (json.loads(p.read_text()) for p in (report_path, checkpoint_path))
    if any(str(doc.get("runId", "")).endswith("-health-rerun1") or "healthReplacement" in doc for doc in (r1, cp)):
        raise ValueError("STOP: input is already a health replacement")
    receipt = {"status": "INCOMPLETE", "inputs": original, "replacement_command": None}
    def finish(status, merged=None):
        receipt["status"] = status
        if any(digest(p) != h for p, h in receipt["inputs"].items()):
            raise ValueError("STOP: input bytes changed")
        if merged is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            receipt["merged_sha256"] = digest(output)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open("x") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
        return receipt
    if cp.get("status") != "completed":
        receipt["reason"] = "aborted_or_partial_run_no_replacement"
        return finish("INCOMPLETE")
    for key in ("runId", "provider", "benchmark", "judge", "answeringModel", "recallConfig"):
        if not r1.get(key) or r1[key] != cp.get(key):
            raise ValueError("STOP: first report/checkpoint identity mismatch: " + key)
    if r1["recallConfig"] != config:
        raise ValueError("STOP: first recallConfig mismatch")
    check_provenance((r1, cp), (r1, cp), guard)
    questions, rows = cp["questions"], rows_by_id(r1)
    ids = cp.get("targetQuestionIds") or list(questions)
    if len(ids) != len(set(ids)) or set(ids) != set(questions) or set(rows) - set(ids):
        raise ValueError("STOP: first question set mismatch")
    states, unhealthy = {}, []
    for qid in ids:
        q = questions[qid]
        if qid not in rows:
            if not phase_failed(q):
                receipt["reason"] = "completed_question_missing_report"
                return finish("INCOMPLETE")
            rows[qid] = {"questionId": qid, "questionType": q.get("questionType"),
                         "score": 0, "label": "incorrect", "_failed": True,
                         "recallHealth": q["phases"].get("search", {}).get("recallHealth")}
        state = "healthy" if qid.endswith("_abs") else guard.classify_recall_health(rows[qid].get("recallHealth"), config)
        if state == "missing" and not phase_failed(q):
            receipt["reason"] = "completed_question_missing_health"
            return finish("INCOMPLETE")
        states[qid] = state
        if state == "unhealthy" and not outside_recall_error(q, rows[qid], guard):
            unhealthy.append(qid)
    if not ids:
        raise ValueError("STOP: empty question set")
    unhealthy.sort()
    receipt.update(total=len(ids), selected_ids=unhealthy)
    merged = copy.deepcopy(r1)
    merged_rows = {qid: copy.deepcopy(rows[qid]) for qid in ids}
    for row in merged_rows.values():
        row["healthAttempt"] = 1
    replacement = None
    if unhealthy:
        rid = r1["runId"] + "-health-rerun1"
        run_dir = report_path.parent.parent / rid
        if run_dir.exists():
            raise ValueError("STOP: replacement run already exists; no second attempt")
        command = replacement_command(r1, unhealthy, bridge, workers)
        receipt["replacement_command"] = command
        cwd = report_path.parent.parents[2]
        if execute is None:
            def execute(command, cwd, run_dir):
                with (report_path.parent / "health-rerun-launch.log").open("x") as log:
                    subprocess.run(command, cwd=cwd, env=os.environ.copy(), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
        execute(command, cwd, run_dir)
        rp, ckp = run_dir / "report.json", run_dir / "checkpoint.json"
        receipt["inputs"].update({str(p): digest(p) for p in (rp, ckp)})
        replacement, cp2 = (json.loads(p.read_text()) for p in (rp, ckp))
        if (cp2.get("dataSourceRunId") != rid or cp2.get("runId") != rid or
                replacement.get("dataSourceRunId") != rid or replacement.get("runId") != rid):
            raise ValueError("STOP: replacement dataSourceRunId reuse/mismatch")
        for key in ("provider", "benchmark", "judge", "answeringModel", "recallConfig"):
            if cp2.get(key) != r1[key] or replacement.get(key) != r1[key]:
                raise ValueError("STOP: replacement identity mismatch: " + key)
        check_provenance((r1, cp), (replacement, cp2), guard)
        if cp2.get("status") != "completed":
            receipt["reason"] = "partial_replacement_no_second_attempt"
            return finish("INCOMPLETE")
        rr = rows_by_id(replacement)
        if set(rr) != set(unhealthy) or set(cp2["questions"]) != set(unhealthy):
            raise ValueError("STOP: replacement question set mismatch")
        receipt["replacement_checkpoint_sha256"] = digest(ckp)
        receipt["replacement_checkpoint"] = str(ckp.relative_to(output.parent)) if ckp.is_relative_to(output.parent) else os.path.relpath(ckp, output.parent)
        for qid in unhealthy:
            row = copy.deepcopy(rr[qid])
            state = guard.classify_recall_health(row.get("recallHealth"), config)
            if state == "missing" and not phase_failed(cp2["questions"][qid]):
                receipt["reason"] = "completed_replacement_missing_health"
                return finish("INCOMPLETE")
            row.update(healthAttempt=2, replacedFrom=hashlib.sha256(canonical(rows[qid])).hexdigest())
            merged_rows[qid], states[qid] = row, state
    remaining = [qid for qid in ids if states[qid] != "healthy"]
    for qid, row in merged_rows.items():
        if qid in remaining or outside_recall_error((cp2 if qid in unhealthy else cp)["questions"][qid], row, guard):
            row.update(score=0, label="incorrect", _failed=True)
        row["healthUnhealthy"] = qid in remaining
    receipt.update(still_unhealthy=len(remaining), degradedReasons=dict(Counter(
        str(merged_rows[q].get("recallHealth", {}).get("degradedReason") or
            ("timeout" if (merged_rows[q].get("recallHealth") or {}).get("timeout") else "missing_or_unhealthy"))
        if isinstance(merged_rows[q].get("recallHealth"), dict) else "missing" for q in remaining)))
    merged["evaluations"] = [merged_rows[q] for q in ids]
    merged["healthReplacement"] = {k: receipt[k] for k in
        ("inputs", "selected_ids", "still_unhealthy", "degradedReasons", "replacement_checkpoint", "replacement_checkpoint_sha256") if k in receipt}
    merged["summary"] = {"totalQuestions": len(ids), "correctCount": sum(r.get("score") == 1 for r in merged_rows.values())}
    merged["summary"]["accuracy"] = merged["summary"]["correctCount"] / len(ids)
    if "byQuestionType" in merged:
        merged["byQuestionType"] = category_stats(merged["evaluations"])
    if "retrieval" in merged:
        merged["retrieval"] = category_stats([dict(r, questionType="all") for r in merged["evaluations"]])["all"].get("retrieval")
    removed = [key for key in ("latency", "tokens", "memscore", "memscoreComponents", "cliLedger") if key in merged]
    for key in removed:
        del merged[key]
    if removed:
        merged["aggregatesRemoved"] = removed
    return finish("INCONCLUSIVE" if len(remaining) * 100 > len(ids) else "COMPLETE", merged)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "checkpoint", "output", "receipt", "instruments"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--config", required=True, choices=("OFF", "LOCAL"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("bridge", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    try:
        bridge = a.bridge[1:] if a.bridge[:1] == ["--"] else a.bridge
        if not bridge:
            raise ValueError("STOP: bridge command required")
        result = process(a.report, a.checkpoint, a.output, a.receipt, a.config,
                         guard_from(a.instruments), bridge, workers=a.workers)
        print(json.dumps({k: result[k] for k in ("status", "total", "still_unhealthy") if k in result}))
        return 0 if result["status"] == "COMPLETE" else 3
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        # Never emit arbitrary report/exception content.
        print(str(error) if isinstance(error, ValueError) and str(error).startswith("STOP:") else
              "STOP: health replacement " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
