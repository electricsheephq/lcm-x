#!/usr/bin/env python3
"""RUN-SHEET-V1M-REBANK §4.3 bar 3 — the executable result-identity projection.

Usage: result_identity.py <f53_per_question_checkpoint.jsonl> <rebank_per_question_checkpoint.jsonl> [more pairs...]
       (pass shard files pairwise: f53-shard-0 rebank-shard-0 f53-shard-1 rebank-shard-1 ...)

Projection per question_id: (category, abstention, {arm: (recall@1, recall@5, recall@10, ndcg@10,
turn.recall@1, turn.recall@5, turn.recall@10, turn.ndcg@10, turn.session_granularity)}) for the
seven arms, compared by exact equality after a JSON round-trip. Excluded on purpose: ingest_ms, every
latency_ms, rerank_mode / recall_rerank_status (configuration echoes, pinned once as a sanity check),
and the fields F53 never wrote (privacy, corpus_counts, embed_cache, chunk_embedding_mode).
A question_id present in only one side is a delta. Exit 0 = identical; exit 1 = deltas; exit 2 = usage/parse error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("fts", "summary_vectors", "hybrid_rrf", "hybrid_rerank", "chunk_vectors", "hybrid_rrf3", "lcm_recall")
SESSION_METRICS = ("recall@1", "recall@5", "recall@10", "ndcg@10")
TURN_METRICS = ("recall@1", "recall@5", "recall@10", "ndcg@10", "session_granularity")
# Line 1 of a checkpoint is the {"__checkpoint_header__": ...} object; any object without a string
# question_id is skipped as a header.


class InputError(Exception):
    """A malformed or unreadable checkpoint file (exit 2, never a verdict)."""


def _rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError as exc:
            raise InputError(f"malformed JSON in {path} at line {line_no}: {exc}") from exc
        qid = record.get("question_id") if isinstance(record, dict) else None
        if not isinstance(qid, str):
            continue  # header line
        if qid in rows:
            raise InputError(f"duplicate question_id {qid!r} in {path} at line {line_no}")
        rows[qid] = record
    return rows


def _project(record: dict) -> dict:
    arms = record.get("arms") or {}
    projected = {}
    for arm in ARMS:
        metrics = arms.get(arm)
        if not isinstance(metrics, dict):
            projected[arm] = None  # an absent arm is part of the identity (abstention rows have arms == {})
            continue
        turn = metrics.get("turn") or {}
        projected[arm] = (
            tuple(metrics.get(m) for m in SESSION_METRICS)
            + tuple(turn.get(m) for m in TURN_METRICS)
        )
    return {
        "category": record.get("category"),
        "abstention": record.get("abstention"),
        "arms": projected,
    }


def _canon(value):
    return json.loads(json.dumps(value, sort_keys=True))


def compare(f53: Path, rebank: Path) -> tuple[int, list[str], list[str], list[str]]:
    a, b = _rows(f53), _rows(rebank)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    identical = 0
    deltas: list[str] = []
    for qid in sorted(set(a) & set(b)):
        pa, pb = _canon(_project(a[qid])), _canon(_project(b[qid]))
        if pa == pb:
            identical += 1
            continue
        moved_arms = [arm for arm in ARMS if pa["arms"][arm] != pb["arms"][arm]]
        head = []
        if pa["category"] != pb["category"]:
            head.append("category")
        if pa["abstention"] != pb["abstention"]:
            head.append("abstention")
        deltas.append(f"{qid}: moved arms={moved_arms or head}")
    return identical, deltas, only_a, only_b


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__, file=sys.stderr)
        return 2
    total_identical = 0
    total_deltas: list[str] = []
    for f53, rebank in zip(argv[0::2], argv[1::2]):
        try:
            identical, deltas, only_a, only_b = compare(Path(f53), Path(rebank))
        except InputError as exc:
            print(f"INPUT ERROR: {exc}", file=sys.stderr)
            return 2
        total_identical += identical
        total_deltas.extend(deltas)
        total_deltas.extend(f"{qid}: present only in F53 ({f53})" for qid in only_a)
        total_deltas.extend(f"{qid}: present only in re-bank ({rebank})" for qid in only_b)
        print(f"{Path(rebank).parent.name}: identical={identical} deltas={len(deltas)} only_f53={len(only_a)} only_rebank={len(only_b)}")
    print(f"TOTAL identical={total_identical} deltas={len(total_deltas)}")
    for line in total_deltas:
        print("  DELTA " + line)
    return 0 if not total_deltas else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
