#!/usr/bin/env python3
"""Capture delivered-hit profiles and catch silent retrieval-arm swaps.

Usage::

    python bench/tools/deliveryprofile.py capture RESULTS_DIR -o profile.json
    python bench/tools/deliveryprofile.py compare baseline.json candidate.json

The reader accepts the scale389 ``query-*.jsonl`` shape as well as one-hit-per-
line JSONL.  Missing attribution, character, or token fields are recorded as
unattributed/missing rather than turning a diagnostic profile into a crash.
All commands accept ``--json`` and write a short human summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ATTRIBUTION_KEYS = (
    "arm",
    "source",
    "retrieval_arm",
    "retrievalArm",
    "contributor",
    "mechanism",
)
CHAR_KEYS = (
    "delivered_chars",
    "content_chars",
    "char_count",
    "chars",
    "payload_chars",
)
TOKEN_KEYS = (
    "delivered_tokens",
    "content_tokens",
    "token_count",
    "tokens",
    "payload_tokens",
    "n_tokens",
)
CONTENT_KEYS = ("content", "snippet", "text", "payload", "body")


def _read_json(path: str | Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _input_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise NotADirectoryError(path)
    query_files = sorted(path.glob("query-*.jsonl"))
    return query_files or sorted(path.glob("*.jsonl"))


def _first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    lowered = {str(key).lower().replace("-", "_"): value for key, value in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower().replace("-", "_"))
        if value not in (None, ""):
            return value
    return None


def _containers(hit: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    values = [hit]
    for key in ("metadata", "provenance", "source_metadata"):
        nested = hit.get(key)
        if isinstance(nested, dict):
            values.append(nested)
    if row is not hit:
        values.append(row)
    return values


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _attribution(hit: dict[str, Any], row: dict[str, Any]) -> str:
    for container in _containers(hit, row):
        value = _first(container, ATTRIBUTION_KEYS)
        if isinstance(value, dict):
            value = _first(value, ATTRIBUTION_KEYS)
        if value not in (None, ""):
            return str(value)
    return "unattributed"


def _payload_metric(hit: dict[str, Any], row: dict[str, Any], keys: Iterable[str], content: Any) -> float | None:
    for container in _containers(hit, row):
        value = _first(container, keys)
        number = _as_number(value)
        if number is not None:
            return number
    if content is not None:
        return float(len(str(content))) if tuple(keys) == CHAR_KEYS else None
    return None


def _content(hit: dict[str, Any]) -> Any:
    return _first(hit, CONTENT_KEYS)


def _hits_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("delivered_hits", "deliveredHits", "hits", "delivered", "results"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]
    if _first(row, CONTENT_KEYS) is not None:
        return [row]
    return []


def _question_key(row: dict[str, Any], file_path: Path, line_number: int) -> str:
    qid = _first(row, ("question_id", "questionId", "qid", "id"))
    if qid in (None, ""):
        qid = f"{file_path.name}:{line_number}"
    rep = _first(row, ("rep", "repeat", "trial"))
    return f"{qid}#rep={rep}" if rep not in (None, "") else str(qid)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "median": statistics.median(numbers),
        "p90": _percentile(numbers, 0.90),
        "max": max(numbers),
    }


def capture_profile(run_results_dir: str | Path) -> dict[str, Any]:
    """Build a bounded profile from query JSONL files without requiring all fields."""
    files = _input_files(run_results_dir)
    if not files:
        raise ValueError(f"no JSONL result files found in {run_results_dir}")

    arm_counts: dict[str, int] = defaultdict(int)
    chars: list[float] = []
    tokens: list[float] = []
    question_hits: dict[str, int] = defaultdict(int)
    total_hits = 0
    missing_chars = 0
    missing_tokens = 0
    rows_seen = 0

    for file_path in files:
        for line_number, row in enumerate(_read_jsonl(file_path), 1):
            rows_seen += 1
            question = _question_key(row, file_path, line_number)
            hits = _hits_for_row(row)
            question_hits[question] += len(hits)
            for hit in hits:
                total_hits += 1
                arm_counts[_attribution(hit, row)] += 1
                content = _content(hit)
                char_count = _payload_metric(hit, row, CHAR_KEYS, content)
                token_count = _payload_metric(hit, row, TOKEN_KEYS, content)
                if char_count is None:
                    missing_chars += 1
                else:
                    chars.append(char_count)
                if token_count is None:
                    missing_tokens += 1
                else:
                    tokens.append(token_count)

    arm_counts = dict(sorted(arm_counts.items()))
    denominator = total_hits
    shares = {
        arm: (count / denominator if denominator else None)
        for arm, count in arm_counts.items()
    }
    profile = {
        "version": 1,
        "run_results_dir": str(Path(run_results_dir)),
        "input_files": [str(path) for path in files],
        "rows": rows_seen,
        "questions": len(question_hits),
        "total_hits": total_hits,
        "attributed_hits": total_hits - arm_counts.get("unattributed", 0),
        "unattributed_hits": arm_counts.get("unattributed", 0),
        "arm_hit_counts": arm_counts,
        "arm_contribution_shares": shares,
        "delivered_chars": {
            **_distribution(chars),
            "missing": missing_chars,
        },
        "delivered_tokens": {
            **_distribution(tokens),
            "missing": missing_tokens,
        },
        "hits_per_question": _distribution(question_hits.values()),
    }
    return profile


def write_profile(run_results_dir: str | Path, output: str | Path) -> dict[str, Any]:
    profile = capture_profile(run_results_dir)
    Path(output).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def _profile_counts(profile: dict[str, Any]) -> dict[str, float]:
    value = profile.get("arm_hit_counts", profile.get("per_arm_hit_counts", {}))
    return {
        str(arm): float(count)
        for arm, count in value.items()
        if _as_number(count) is not None
    }


def _profile_shares(profile: dict[str, Any], counts: dict[str, float]) -> dict[str, float]:
    value = profile.get("arm_contribution_shares", profile.get("arm_shares"))
    if isinstance(value, dict):
        return {
            str(arm): float(share)
            for arm, share in value.items()
            if _as_number(share) is not None
        }
    total = sum(counts.values())
    return {arm: count / total for arm, count in counts.items()} if total else {}


def _median(profile: dict[str, Any], metric: str) -> float | None:
    value = profile.get(f"delivered_{metric}")
    if isinstance(value, dict):
        return _as_number(value.get("median"))
    for key in (f"median_delivered_{metric}", f"delivered_{metric}_median"):
        number = _as_number(profile.get(key))
        if number is not None:
            return number
    return None


def compare_profiles(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    max_share_drift: float = 0.15,
    max_median_drift: float = 0.25,
) -> dict[str, Any]:
    """Compare arm mix and payload medians, naming every violated clause."""
    if max_share_drift < 0 or max_median_drift < 0:
        raise ValueError("drift bounds must be non-negative")
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("profiles must be JSON objects")
    baseline_counts = _profile_counts(baseline)
    candidate_counts = _profile_counts(candidate)
    baseline_shares = _profile_shares(baseline, baseline_counts)
    candidate_shares = _profile_shares(candidate, candidate_counts)
    clauses: list[dict[str, Any]] = []
    failures: list[str] = []

    for arm in sorted(set(baseline_shares) | set(candidate_shares)):
        before = baseline_shares.get(arm, 0.0)
        after = candidate_shares.get(arm, 0.0)
        drift = abs(after - before)
        ok = drift <= max_share_drift
        clause = {
            "name": f"arm-share:{arm}",
            "baseline": before,
            "candidate": after,
            "drift": drift,
            "bound": max_share_drift,
            "ok": ok,
        }
        clauses.append(clause)
        if not ok:
            failures.append(clause["name"])
        if before > 0.01 and after <= 0.01:
            death = {
                "name": f"arm-death:{arm}",
                "baseline": before,
                "candidate": after,
                "threshold": 0.01,
                "ok": False,
            }
            clauses.append(death)
            failures.append(death["name"])

    for metric in ("chars", "tokens"):
        before = _median(baseline, metric)
        after = _median(candidate, metric)
        if before is None or after is None:
            clauses.append({
                "name": f"median-delivered-{metric}",
                "baseline": before,
                "candidate": after,
                "bound": max_median_drift,
                "ok": True,
                "skipped": "missing median",
            })
            continue
        drift = abs(after - before) / abs(before) if before else (0.0 if after == 0 else math.inf)
        ok = drift <= max_median_drift
        clause = {
            "name": f"median-delivered-{metric}",
            "baseline": before,
            "candidate": after,
            "relative_drift": drift,
            "bound": max_median_drift,
            "ok": ok,
        }
        clauses.append(clause)
        if not ok:
            failures.append(clause["name"])

    return {
        "version": 1,
        "baseline": str(Path(baseline_path)),
        "candidate": str(Path(candidate_path)),
        "bounds": {
            "max_share_drift": max_share_drift,
            "max_median_drift": max_median_drift,
        },
        "clauses": clauses,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "ok": not failures,
    }


# Small aliases make the capture/compare functions convenient for bench code.
capture = capture_profile
compare = compare_profiles


def _summary(result: dict[str, Any]) -> str:
    verdict = result.get("verdict", "UNKNOWN")
    failures = result.get("failures", [])
    return f"{verdict}" + (f" failures={','.join(failures)}" if failures else "")


def _emit(result: dict[str, Any], json_output: bool = True) -> None:
    print(_summary(result), file=sys.stderr)
    if json_output:
        print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and compare delivered retrieval profiles.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="emit the bounded JSON report")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture a delivery profile")
    capture.add_argument("run_results_dir", type=Path)
    capture.add_argument("-o", "--output", type=Path, required=True)
    capture.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)

    compare = subparsers.add_parser("compare", help="compare two delivery profiles")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--max-share-drift", type=float, default=0.15)
    compare.add_argument("--max-median-drift", type=float, default=0.25)
    compare.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_output = True
    try:
        if args.command == "capture":
            result = write_profile(args.run_results_dir, args.output)
            result = {**result, "output": str(args.output)}
            _emit(result, json_output)
            return 0
        result = compare_profiles(
            args.baseline,
            args.candidate,
            max_share_drift=args.max_share_drift,
            max_median_drift=args.max_median_drift,
        )
        _emit(result, json_output)
        return 0 if result["ok"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
