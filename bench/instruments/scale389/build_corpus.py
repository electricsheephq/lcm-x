#!/usr/bin/env python3
"""Build the deduplicated scale ladder and persist answer-turn labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.tools.pinverify import sha256_file


DEFAULT_DATASET = Path("/Volumes/LEXAR/hermes-work/longmemeval-data/longmemeval_s")
DEFAULT_OUTPUT = Path(
    "/Volumes/LEXAR/Codex/session-notes/2026-07-29/"
    "hermes-r3-1/artifacts/laneINSTPREP-logs/scale389-corpus"
)
SEED = 20260725
DEFAULT_SCALES = (500, 2000, 8000, 19829)


def _raw_answer_turns(question: dict[str, Any]) -> list[dict[str, str]]:
    """Persist answer labels while building the corpus from raw input."""
    ids = question["haystack_session_ids"]
    dates = question["haystack_dates"]
    sessions = question["haystack_sessions"]
    turns: list[dict[str, str]] = []
    for session_id, date, messages in zip(ids, dates, sessions):
        for message in messages:
            if message.get("has_answer") is True:
                turns.append(
                    {
                        "session_id": str(session_id),
                        "date": str(date),
                        "content": str(message.get("content", "")),
                    }
                )
    return turns


def build_corpus(
    dataset_path: Path,
    output_dir: Path,
    *,
    n_eval_questions: int = 100,
    scales: tuple[int, ...] = DEFAULT_SCALES,
) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    union: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    for raw in data:
        ids = raw["haystack_session_ids"]
        dates = raw["haystack_dates"]
        sessions = raw["haystack_sessions"]
        for session_id, session, date in zip(ids, sessions, dates):
            value = {"sid": session_id, "date": date, "messages": session}
            previous = union.setdefault(session_id, value)
            if previous != value:
                raise ValueError(f"conflicting content for session {session_id!r}")
        questions.append(
            {
                "answer": raw["answer"],
                "answer_turns": _raw_answer_turns(raw),
                "gold": sorted(set(raw["answer_session_ids"])),
                "haystack": list(ids),
                "question": raw["question"],
                "question_date": raw["question_date"],
                "question_id": raw["question_id"],
                "question_type": raw["question_type"],
            }
        )

    if n_eval_questions > len(questions):
        raise ValueError("requested more eval questions than the dataset contains")
    all_sids = sorted(union)
    rng = random.Random(SEED)
    qeval = sorted(
        rng.sample(questions, n_eval_questions), key=lambda row: row["question_id"]
    )
    gold = sorted({sid for question in qeval for sid in question["gold"]})
    if not set(gold) <= set(union):
        raise ValueError("an eval gold session is absent from the union")

    rest = [sid for sid in all_sids if sid not in set(gold)]
    random.Random(SEED + 1).shuffle(rest)
    pool = gold + rest
    ladder: dict[str, Any] = {"scales": {}, "seed": SEED}
    for n in scales:
        if not len(gold) <= n <= len(pool):
            raise ValueError(f"scale {n} cannot contain the {len(gold)} gold sessions")
        members = pool[:n]
        order = list(members)
        random.Random(SEED + 1000 + n).shuffle(order)
        ladder["scales"][str(n)] = {
            "ingest_order": order,
            "n_gold_in_scope": len(set(gold) & set(members)),
            "n_messages": sum(len(union[sid]["messages"]) for sid in members),
            "n_sessions": len(members),
        }

    union_path = output_dir / "union.jsonl"
    with union_path.open("w", encoding="utf-8") as handle:
        for session_id in all_sids:
            handle.write(json.dumps(union[session_id], sort_keys=True) + "\n")
    qeval_path = output_dir / "qeval.json"
    qeval_path.write_text(
        json.dumps(
            {
                "gold_union": gold,
                "n_questions": len(qeval),
                "questions": qeval,
                "seed": SEED,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ladder_path = output_dir / "ladder.json"
    ladder_path.write_text(
        json.dumps(ladder, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset_sha256": sha256_file(dataset_path),
        "ladder": str(ladder_path),
        "qeval": str(qeval_path),
        "qeval_sha256": sha256_file(qeval_path),
        "union": str(union_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build scale389 corpus files with persisted has_answer turns."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_corpus(
        args.dataset,
        args.output,
        n_eval_questions=args.questions,
        scales=tuple(args.scales),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
