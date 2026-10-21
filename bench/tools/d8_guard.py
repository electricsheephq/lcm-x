"""D8 single-shot QA guard; recording capability, not a release-row certificate."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

from bench.tools.failclose import _correct, _failed, _qid, load_rows


# Memorybench report/checkpoint identities, plus optional dataset/scorer pins.
# Missing pins on both sides are disclosed, never invented from local files.
IDENTITY_FIELDS = (
    "datasetPath", "datasetSha256", "datasetSha", "dataset.path", "dataset.sha256",
    "answeringModel", "answererProvenance.model", "answererProvenance.configuredModel",
    "answererProvenance.reasoningEffort", "judge", "judgeProvenance.model",
    "judgeProvenance.configuredModel", "judgeProvenance.reasoningEffort",
    "scoringPath", "promptVersion", "answerPresentationMode", "recallConfig",
)


def classify_recall_health(health, config):
    """D1 OFF and #1723 LOCAL rules. Missing coverage (quota path) is missing."""
    if config not in {"OFF", "LOCAL"} or not isinstance(health, dict):
        return "missing"
    if health.get("timeout") is True:
        return "unhealthy"
    coverage = health.get("coverage")
    if not isinstance(coverage, dict):
        return "missing"
    if not isinstance(health.get("degraded"), bool) or health.get("timeout") is not False:
        return "missing"
    reason = health.get("degradedReason")
    if reason is not None and not isinstance(reason, str):
        return "missing"
    if config == "OFF":
        valid_reason = reason == "semantic retrieval is disabled" if health["degraded"] else not reason
        return "healthy" if coverage.get("fts") == "ok" and valid_reason else "unhealthy"
    arms = ("summary", "chunk")
    reasons = reason.split("; ") if reason else []
    if any(coverage.get(arm) not in {"full", "full_approx"} for arm in arms):
        return "unhealthy"
    if coverage.get("fts") != "ok":
        return "unhealthy"
    if health["degraded"] != bool(reasons):
        return "unhealthy"
    if any(not any(r.startswith(f"{arm} arm coverage full_approx:")
                       and coverage[arm] == "full_approx" for arm in arms) for r in reasons):
        return "unhealthy"
    if any(coverage[arm] == "full_approx" and not any(
        r.startswith(f"{arm} arm coverage full_approx:") for r in reasons
    ) for arm in arms):
        return "unhealthy"
    return "healthy"


def _is_correct(row):
    if _failed(row):
        return False
    try:
        return _correct(row)
    except ValueError:  # absent evaluation is incorrect, never a dropped pair
        return False


def _unscorable(row):
    if _failed(row):
        return False
    try:
        _correct(row)
    except ValueError:
        return True
    return False


def degraded_rate(rows, config):
    states = [classify_recall_health(row.get("recallHealth"), config) for row in rows]
    missing = states.count("missing")
    unscorable = sum(_unscorable(row) for row in rows)
    degraded = sum(state != "healthy" or _failed(row) or _unscorable(row) for row, state in zip(rows, states))
    result = {"total": len(rows), "degraded": degraded,
            "rate_points": 100 * degraded / len(rows) if rows else None,
            "missing": missing, "INCOMPLETE": bool(missing or unscorable) or not rows}
    if unscorable:
        result["unscorable"] = unscorable
    return result


def bootstrap_bounds(clusters):
    """10,000 seed-1607 resamples; nearest-rank p2.5/p97.5 (250th/9,750th).

    Each cluster contains paired 0/1 correctness differences. Pool sampled
    questions (ratio of sums), retaining the original per-question weighting.
    """
    if not clusters:
        return None, None
    rng = random.Random(1607)
    totals = [(sum(values), len(values)) for values in clusters]
    samples = []
    for _ in range(10_000):
        picked = rng.choices(totals, k=len(totals))
        samples.append(100 * sum(s for s, _ in picked) / sum(n for _, n in picked))
    samples.sort()
    return (samples[math.ceil(0.025 * len(samples)) - 1],
            samples[math.ceil(0.975 * len(samples)) - 1])


def _record_identities(document, identities, issues):
    for role in ("answererProvenance", "judgeProvenance"):
        provenance = document.get(role)
        if not isinstance(provenance, dict):
            continue
        for key, value in provenance.items():
            mixed = (value in ("mixed", "mixed stored CLI call identities")
                     or (key == "mixedExecutionIdentity" and value is True)
                     or (key == "executionIdentityCount" and isinstance(value, (int, float))
                         and value > 1))
            issue = f"identity mixed within run: {role}.{key}"
            if mixed and issue not in issues:
                issues.append(issue)
    for field in IDENTITY_FIELDS:
        value = document
        for key in field.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and not value.strip():
            value = None
        if value is not None and value not in identities[field]:
            identities[field].append(value)


def _check_rows(values, source, identities, issues):
    if not isinstance(values, list):
        raise ValueError(f"{source} evaluations must be a list")
    seen, duplicates = set(), []
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"{source} row is not a JSON object")
        try:
            qid = _qid(row)
        except ValueError as error:
            raise ValueError(f"{source}: {error}") from error
        if qid in seen:
            duplicates.append(qid)
        seen.add(qid)
        _record_identities(row, identities, issues)
    if duplicates:
        issues.append(f"duplicate question ids: {len(duplicates)}; first id: {duplicates[0]} ({source})")


def _load_run(root, *, baseline=False):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    identities = {field: [] for field in IDENTITY_FIELDS}
    issues = []
    _record_identities(report, identities, issues)
    _check_rows(report.get("evaluations", []), "report.json", identities, issues)
    # Validate each input before load_rows collapses IDs. A single detail row
    # overlaying a report row is legitimate; repeated IDs within a file are not.
    detail = next((path for path in (root / "per_question.jsonl", root / "results/per_question.jsonl")
                   if path.is_file()), None)
    detail_rows = []
    if detail:
        detail_rows = [json.loads(line) for line in detail.read_text().splitlines() if line.strip()]
        _check_rows(detail_rows, detail.name, identities, issues)
    has_rows = (report.get("evaluations") or (root / "per_question.jsonl").exists()
                or (root / "results/per_question.jsonl").exists())
    rows = {_qid(row): row for row in load_rows(root)} if has_rows else {}
    if baseline:
        report_rows = {_qid(row): row for row in report.get("evaluations", [])}
        contradictions = []
        for qid, detail_row in {_qid(row): row for row in detail_rows}.items():
            if _failed(report_rows.get(qid, {})) and not _failed(detail_row):
                try:
                    _correct(rows[qid])  # An annotation may retain the report verdict.
                except ValueError:
                    continue
                contradictions.append(qid)
        if contradictions:
            issues.append("failure contradicted by detail: " + ", ".join(sorted(contradictions)))
    checkpoint_path = root / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        _record_identities(checkpoint, identities, issues)
        questions = checkpoint.get("questions", {})
        targets = [{"questionId": qid} for qid in checkpoint.get("targetQuestionIds") or questions]
        _check_rows(targets, "checkpoint.json targetQuestionIds", identities, issues)
        for target in targets:
            qid = _qid(target)
            question = questions.get(qid, {})
            phases = question.get("phases", {})
            row = rows.setdefault(qid, {"questionId": qid,
                                       "questionType": question.get("questionType")})
            if "recallHealth" not in row:
                row["recallHealth"] = phases.get("search", {}).get("recallHealth")
            row["_failed"] = row.get("_failed", False) or any(
                phases.get(p, {}).get("status") != "completed"
                for p in ("search", "answer", "evaluate"))
    issues += [f"identity mixed within run: {field}" for field, values in identities.items()
               if len(values) > 1]
    return report, rows, identities, issues


def _compare_identities(candidate, baseline, *, placebo=False):
    issues, missing_both = [], []
    for field in IDENTITY_FIELDS:
        if field in {"recallConfig", "answerPresentationMode"} and not placebo:
            continue
        left, right = candidate[field], baseline[field]
        if not left and not right:
            missing_both.append(field)
        elif not left or not right:
            issues.append(f"identity missing: {field}")
        elif len(left) != 1 or len(right) != 1 or left != right:
            issues.append(f"identity mismatch: {field}")
    return issues, missing_both


def _category(row, name):
    if name == "abstention":
        return _qid(row).endswith("_abs")
    kind = row.get("questionType", row.get("category"))
    return kind == "knowledge-update" if name == "knowledge-update" else str(kind) in {"5", "adversarial"}


def compare_runs(candidate_run, baseline_run, *, subset=False, placebo=False):
    candidate_report, candidate, candidate_ids, candidate_issues = _load_run(candidate_run)
    baseline_report, baseline, baseline_ids, baseline_issues = _load_run(baseline_run, baseline=True)
    issues, missing_both = _compare_identities(candidate_ids, baseline_ids, placebo=placebo)
    issues += [f"{side} {issue}" if issue.startswith("failure contradicted by detail:")
               else f"{side}: {issue}" for side, values in
               (("candidate", candidate_issues), ("baseline", baseline_issues)) for issue in values]
    benchmark = baseline_report.get("benchmark", "")
    if benchmark != candidate_report.get("benchmark") or benchmark not in {"locomo", "longmemeval"}:
        raise ValueError("matching locomo or longmemeval benchmark required")
    names = ["adversarial"] if benchmark == "locomo" else ["abstention", "knowledge-update"]
    # Baseline questions absent from the candidate remain failed/missing, never dropped.
    extra = len(candidate.keys() - baseline.keys())
    for qid, row in baseline.items():
        candidate.setdefault(qid, {"questionId": qid, "questionType": row.get("questionType", row.get("category")),
                                   "_failed": True})
    cc, bc = candidate_report.get("recallConfig"), baseline_report.get("recallConfig")
    overall = {"candidate": degraded_rate(list(candidate.values()), cc),
               "baseline": degraded_rate(list(baseline.values()), bc)}
    issues += [f"{side}: unscorable rows: {rate['unscorable']}"
               for side, rate in overall.items() if rate.get("unscorable")]
    incomplete = (bool(issues) or bool(extra) or any(row["INCOMPLETE"] for row in overall.values())
                  or (not subset and len(baseline) != {"longmemeval": 500, "locomo": 1986}[benchmark]))
    cr, br = overall["candidate"], overall["baseline"]
    rate_pass = None if placebo else (not cr["INCOMPLETE"] and not br["INCOMPLETE"]
                                     and cr["rate_points"] <= br["rate_points"] + 1)
    result = {"categories": {}, "runs": overall, "excluded_absent_from_baseline": extra,
              "recall_configs": {"candidate": cc, "baseline": bc},
              "answer_presentation_modes": {
                  "candidate": candidate_report.get("answerPresentationMode"),
                  "baseline": baseline_report.get("answerPresentationMode")},
              "INCOMPLETE": incomplete, "d8_row": not subset,
              "mode": "placebo" if placebo else "non-inferiority", "rate_pass": rate_pass}
    result["identities_missing_both"] = missing_both
    if issues:
        result["reason"] = "; ".join(issues)
    for name in names:
        base = {qid: row for qid, row in baseline.items() if _category(row, name)}
        clusters = {}
        for qid, row in sorted(base.items()):
            # LoCoMo module encodes `${sample_id}-q${index}`; sample_id is conversation.
            match = re.fullmatch(r"(.+)-q\d+", qid) if benchmark == "locomo" else None
            if benchmark == "locomo" and not match:
                raise ValueError(f"invalid LoCoMo question id: {qid}")
            cluster = match[1] if match else qid
            clusters.setdefault(cluster, []).append(int(_is_correct(candidate[qid])) - int(_is_correct(row)))
        lower, upper = bootstrap_bounds(list(clusters.values()))
        cr = degraded_rate([row for row in candidate.values() if _category(row, name)], cc)
        br = degraded_rate(list(base.values()), bc)
        interval_pass = lower is not None and lower > -2.0 and (not placebo or upper < 2.0)
        result["categories"][name] = {"paired_n": len(base), "clusters": len(clusters),
            "difference_points": 100 * sum(sum(v) for v in clusters.values()) / len(base) if base else None,
            "lower_bound_points": lower, "upper_bound_points": upper, "interval_pass": interval_pass,
            "candidate": cr, "baseline": br, "pass": interval_pass}
    result["pass"] = (all(row["pass"] for row in result["categories"].values())
                      and (placebo or rate_pass) and not incomplete)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("--subset", action="store_true",
                        help="allow a held-out split; output is not a full D8 row")
    parser.add_argument("--placebo", action="store_true",
                        help="compare A-prime (candidate) with A (baseline); require intervals strictly inside +/-2 points")
    args = parser.parse_args(argv)
    try:
        result = compare_runs(args.candidate_run, args.baseline_run, subset=args.subset, placebo=args.placebo)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"pass": False, "INCOMPLETE": True, "error": str(error),
                          "d8_row": not args.subset, "mode": "placebo" if args.placebo else "non-inferiority"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
