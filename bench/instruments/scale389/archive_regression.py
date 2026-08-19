#!/usr/bin/env python3
"""Reproduce F34 session metrics and exercise the new answer-turn join offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.instruments.scale389.metrics import (
    answer_turn_delivery_metrics,
    answer_turns_from_question,
    select_questions,
    summarise_rows,
)
from bench.tools.pinverify import sha256_file
from bench.tools.storefreeze import snapshot


DEFAULT_RESULTS = Path(
    "/Volumes/LEXAR/Codex/session-notes/2026-07-29/"
    "hermes-phase1b-rerun/artifacts/results"
)
DEFAULT_QEVAL = Path(
    "/Volumes/LEXAR/Codex/session-notes/2026-07-25/"
    "hermes-v1-scale/artifacts/corpus/qeval.json"
)
DEFAULT_DATASET = Path("/Volumes/LEXAR/hermes-work/longmemeval-data/longmemeval_s")
DEFAULT_SIDECAR = Path(
    "/Volumes/LEXAR/hermes-work/mb-workdir-phase1a/"
    "phase1a-scale-500.dates.json"
)
EXPECTED_PUBLISHED = {
    "A3": {"500": 0.82, "2000": 0.60, "8000": 0.34, "19829": 0.233},
    "B": {"500": 0.60, "2000": 0.36, "8000": 0.26, "19829": 0.20},
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def reproduce_session_table(results_dir: Path) -> dict[str, Any]:
    table: dict[str, dict[str, float]] = {}
    checked_summaries = 0
    for arm, scales in EXPECTED_PUBLISHED.items():
        for scale, expected in scales.items():
            query_path = results_dir / f"query-{arm}-{scale}.jsonl"
            summary_path = results_dir / f"summary-{arm}-{scale}.json"
            computed = summarise_rows(_load_jsonl(query_path))
            archived = json.loads(summary_path.read_text(encoding="utf-8"))
            for key, value in archived.items():
                if computed.get(key) != value:
                    raise AssertionError(
                        f"{summary_path.name} mismatch for {key}: "
                        f"computed={computed.get(key)!r} archived={value!r}"
                    )
            published = round(float(computed["session_gold_all"]), 3)
            if published != expected:
                raise AssertionError(
                    f"published F34 {arm}/{scale}: got {published}, want {expected}"
                )
            table.setdefault(arm, {})[scale] = published
            checked_summaries += 1
    return {
        "checked_summaries": checked_summaries,
        "metric": "session_gold_all",
        "published_table": table,
        "status": "exact",
    }


def exercise_turn_join(
    qeval_path: Path, dataset_path: Path, sidecar_path: Path
) -> dict[str, Any]:
    qeval = json.loads(qeval_path.read_text(encoding="utf-8"))
    primary_ids = {
        row["question_id"] for row in select_questions(qeval, "A3")
    }
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    candidates = []
    for question in dataset:
        if question["question_id"] not in primary_ids:
            continue
        corpus_question = {
            "gold": list(question["answer_session_ids"]),
            "question_id": question["question_id"],
        }
        turns = answer_turns_from_question(
            corpus_question, qeval_path.parent / "union.jsonl"
        )
        if turns and all(len(turn["content"]) >= 25 for turn in turns):
            candidates.append((question, turns, corpus_question))
    if not candidates:
        raise AssertionError("F34 primary set has no usable labeled answer-turn fixture")

    question, turns, corpus_question = candidates[0]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    complete_hits = [
        {"content": turn["content"], "session_id": turn["session_id"]}
        for turn in turns
    ]
    complete = answer_turn_delivery_metrics(
        corpus_question,
        complete_hits,
        sidecar,
        qeval_path.parent / "union.jsonl",
    )
    incomplete = answer_turn_delivery_metrics(
        corpus_question,
        complete_hits[:-1],
        sidecar,
        qeval_path.parent / "union.jsonl",
    )
    values = [
        incomplete["answer_turn_delivered_complete"],
        complete["answer_turn_delivered_complete"],
    ]
    if sorted(values) != [0, 1]:
        raise AssertionError(f"turn metric is degenerate: {values}")
    return {
        "fixture": "F34 primary question + archived .dates.json + controlled payload",
        "metric": "answer_turn_delivered_complete",
        "question_id": question["question_id"],
        "values": values,
    }


def run_regression(
    results_dir: Path,
    qeval_path: Path,
    dataset_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    before = snapshot(results_dir)
    input_shas = {
        "dataset": sha256_file(dataset_path),
        "qeval": sha256_file(qeval_path),
        "sidecar": sha256_file(sidecar_path),
    }
    session = reproduce_session_table(results_dir)
    turn = exercise_turn_join(qeval_path, dataset_path, sidecar_path)
    after = snapshot(results_dir)
    if before["self_sha256"] != after["self_sha256"]:
        raise AssertionError("archived F34 results changed during reproduction")
    return {
        "archive_read_only": True,
        "archive_snapshot_sha256": before["self_sha256"],
        "input_sha256": input_shas,
        "session_regression": session,
        "turn_join_check": turn,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce F34's published session table and self-test turn joins."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--qeval", type=Path, default=DEFAULT_QEVAL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and args.output.resolve().is_relative_to(args.results.resolve()):
        raise SystemExit("refusing to write regression output inside the F34 archive")
    result = run_regression(
        args.results, args.qeval, args.dataset, args.sidecar
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
