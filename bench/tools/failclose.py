#!/usr/bin/env python3
"""Account for harness fail-closes without hiding them in the score."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


FAIL_CLOSE_MARKERS = (
    "fail-close",
    "fail close",
    "failed closed",
    "no validated exact source reference",
)


def _qid(row: dict[str, Any]) -> str:
    for key in ("questionId", "question_id", "qid", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("per-question row has no question id")


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _failure_texts(row: dict[str, Any]) -> Iterable[str]:
    for key in (
        "fail_close_signature",
        "failClosedReason",
        "fail_closed_reason",
        "error",
        "exception",
        "message",
        "explanation",
        "hypothesis",
        "answer",
        "response",
        "status",
        "label",
    ):
        value = row.get(key)
        if value not in (None, ""):
            yield _normalise(value)


def fail_close_signature(row: dict[str, Any]) -> str | None:
    """Return the row's fail-close signature, or None for an ordinary row."""
    explicit = any(
        row.get(key) is True
        for key in ("fail_closed", "failClosed", "failed_closed")
    )
    texts = list(_failure_texts(row))
    for text in texts:
        lowered = text.lower()
        if any(marker in lowered for marker in FAIL_CLOSE_MARKERS):
            return text
    if explicit:
        return texts[0] if texts else "fail_closed"
    return None


def _correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    if "score" in row:
        return row["score"] == 1
    label = str(row.get("label", "")).strip().lower()
    if label:
        return label in {"correct", "pass", "passed"}
    raise ValueError(f"question {_qid(row)!r} has no score")


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


def load_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load report evaluations, overlaid by per_question.jsonl when present."""
    root = Path(run_dir)
    report_path = root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing harness report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluations = report.get("evaluations", [])
    if not isinstance(evaluations, list):
        raise ValueError("report.json evaluations must be a list")

    by_qid: dict[str, dict[str, Any]] = {}
    for value in evaluations:
        if not isinstance(value, dict):
            raise ValueError("report.json evaluation is not an object")
        by_qid[_qid(value)] = dict(value)

    detail_path = next(
        (
            candidate
            for candidate in (
                root / "per_question.jsonl",
                root / "results" / "per_question.jsonl",
            )
            if candidate.is_file()
        ),
        None,
    )
    if detail_path:
        for detail in _read_jsonl(detail_path):
            qid = _qid(detail)
            by_qid[qid] = {**by_qid.get(qid, {}), **detail}

    if not by_qid:
        raise ValueError("run has no per-question results")
    return [by_qid[qid] for qid in sorted(by_qid)]


def _score(rows: Iterable[dict[str, Any]]) -> dict[str, int | float | None]:
    values = list(rows)
    correct = sum(_correct(row) for row in values)
    total = len(values)
    return {
        "correct": correct,
        "total": total,
        "rate": correct / total if total else None,
    }


def account_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return the first-class raw and fail-close-adjusted accounting object."""
    values = list(rows)
    signatures = {_qid(row): fail_close_signature(row) for row in values}
    failed = sorted(qid for qid, signature in signatures.items() if signature)
    adjusted = [row for row in values if _qid(row) not in set(failed)]
    return {
        "score_raw": _score(values),
        "score_adjusted": _score(adjusted),
        "fail_closed_n": len(failed),
        "fail_closed_qids": failed,
        "per_row_signature": dict(sorted(signatures.items())),
    }


def account_run(run_dir: str | Path) -> dict[str, Any]:
    return account_rows(load_rows(run_dir))


def compare_rows(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare paired rows after dropping either arm's fail-closes from both."""
    left = {_qid(row): row for row in left_rows}
    right = {_qid(row): row for row in right_rows}
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise ValueError(
            "paired qids differ: "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )

    dropped = sorted(
        qid
        for qid in left
        if fail_close_signature(left[qid]) or fail_close_signature(right[qid])
    )
    kept = sorted(left.keys() - set(dropped))
    left_only = sum(_correct(left[qid]) and not _correct(right[qid]) for qid in kept)
    right_only = sum(_correct(right[qid]) and not _correct(left[qid]) for qid in kept)
    return {
        "drop_convention": "union-drop",
        "dropped_qids": dropped,
        "paired_n": len(kept),
        "left_score": _score(left[qid] for qid in kept),
        "right_score": _score(right[qid] for qid in kept),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report raw and fail-close-adjusted harness scores."
    )
    parser.add_argument("run_dir", type=Path, help="run directory containing report.json")
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="RUN_DIR",
        help="add a paired comparison using the named union-drop convention",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.run_dir)
    result = account_rows(rows)
    if args.compare:
        result["paired_comparison"] = compare_rows(rows, load_rows(args.compare))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
