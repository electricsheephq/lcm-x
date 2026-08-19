"""Shared metrics and probe selection for the 389x scale instrument."""

from __future__ import annotations

import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from bench.tools.failclose import fail_close_signature
from bench.tools.pinverify import sha256_file, verify


PRIMARY_N = 50
DEFAULT_UNCENSORED_N = 50
UNCENSORED_ARMS = {"A2u", "A3u"}
PRIMARY_SEED = 20260725 + 7
UNCENSORED_SEED = 20260725 + 8


def normalise_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def content_matches(delivered: str, answer_turn: str) -> bool:
    """Return the F29/F37 content join, never a positional match."""
    content = normalise_content(delivered)
    turn = normalise_content(answer_turn)
    return bool(content) and len(content) >= 25 and (
        content in turn or turn[:120] in content or content[:120] in turn
    )


def answer_turns_from_question(
    question: dict[str, Any],
    union_path: str | Path | None = None,
    union_records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Read persisted turns or derive them from the corpus union.

    The persisted-turn path is used for corpus-built qeval rows. A row without
    persisted turns must provide a corpus union path; raw dataset dates are not
    an identity source for scoring.
    """
    persisted = question.get("answer_turns")
    if persisted is not None:
        if not isinstance(persisted, list):
            raise ValueError("answer_turns must be a list")
        return [
            {
                "session_id": str(turn["session_id"]),
                "date": str(turn["date"]),
                "content": str(turn["content"]),
            }
            for turn in persisted
        ]

    question_id = str(question.get("question_id", "<unknown>"))
    if union_path is None:
        raise ValueError(
            f"answer_turns missing for question {question_id!r}; "
            "corpus union.jsonl path is required"
        )
    path = Path(union_path)
    if path.is_dir():
        path /= "union.jsonl"

    if union_records is None:
        try:
            union_records = {}
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(
                        row.get("sid"), str
                    ):
                        raise ValueError(
                            f"line {line_number} is not a valid union record"
                        )
                    union_records[row["sid"]] = row
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot read corpus union {path} for question {question_id!r}: {exc}"
            ) from exc

    gold = question.get("gold")
    if not isinstance(gold, list):
        raise ValueError(
            f"question {question_id!r} has no gold session ids for corpus union {path}"
        )
    turns: list[dict[str, str]] = []
    for value in gold:
        session_id = str(value)
        record = union_records.get(session_id)
        if record is None:
            raise ValueError(
                f"corpus union {path} is missing gold session {session_id!r} "
                f"for question {question_id!r}"
            )
        try:
            corpus_session_id = str(record["sid"])
            date = str(record["date"])
            messages = record["messages"]
            if not isinstance(messages, list):
                raise TypeError("messages is not a list")
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"cannot read corpus union {path} for question {question_id!r}: "
                f"invalid record for session {session_id!r}: {exc}"
            ) from exc
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    f"cannot read corpus union {path} for question {question_id!r}: "
                    f"invalid message for session {session_id!r}"
                )
            if message.get("has_answer") is True:
                turns.append(
                    {
                        "session_id": corpus_session_id,
                        "date": str(date),
                        "content": str(message.get("content", "")),
                    }
                )
    return turns


def _hit_content(hit: dict[str, Any]) -> str:
    return str(hit.get("snippet") or hit.get("content") or "")


def _hit_session_id(hit: dict[str, Any]) -> str:
    value = hit.get("session_id")
    if value is None:
        metadata = hit.get("metadata") or {}
        value = metadata.get("session_id")
    if value is None:
        raise ValueError("delivered hit has no session_id")
    return str(value)


def answer_turn_delivery_metrics(
    question: dict[str, Any],
    delivered_hits: Iterable[dict[str, Any]],
    sidecar_dates: dict[str, Any],
    union_path: str | Path | None = None,
) -> dict[str, int | None]:
    """Score labeled answer turns against the exact delivered hit payload.

    The sidecar is the stable identity boundary. Every delivered session must
    occur in it; no hit or answer turn is mapped by ingest position.
    """
    turns = answer_turns_from_question(question, union_path)
    if not turns:
        return {
            "answer_turn_delivered_complete": None,
            "answer_turn_delivered_found": 0,
            "answer_turn_delivered_total": 0,
        }

    for turn in turns:
        session_id = turn["session_id"]
        if session_id not in sidecar_dates:
            raise ValueError(
                f"answer session {session_id!r} is absent from the .dates.json sidecar"
            )
        if str(sidecar_dates[session_id]) != turn["date"]:
            raise ValueError(
                f"answer session {session_id!r} date disagrees with the .dates.json sidecar"
            )

    payloads: list[str] = []
    for hit in delivered_hits:
        session_id = _hit_session_id(hit)
        if session_id not in sidecar_dates:
            raise ValueError(
                f"delivered session {session_id!r} is absent from the .dates.json sidecar"
            )
        payloads.append(_hit_content(hit))

    covered = [
        any(content_matches(payload, turn["content"]) for payload in payloads)
        for turn in turns
    ]
    found = sum(covered)
    return {
        "answer_turn_delivered_complete": int(found == len(turns)),
        "answer_turn_delivered_found": found,
        "answer_turn_delivered_total": len(turns),
    }


def session_gold_metrics(
    question: dict[str, Any], delivered_hits: Iterable[dict[str, Any]]
) -> dict[str, int | float | None]:
    returned = {_hit_session_id(hit) for hit in delivered_hits}
    gold = {str(value) for value in question.get("gold", [])}
    found = len(returned & gold)
    complete = int(found == len(gold))
    return {
        "gold_total": len(gold),
        "gold_found": found,
        "recall": found / len(gold) if gold else None,
        "hit_at_k": int(found > 0),
        "session_gold_all": complete,
        # Compatibility alias for the F34 files. New claims use the explicit key.
        "all_gold": complete,
    }


def select_questions(
    qeval: dict[str, Any],
    arm: str,
    *,
    primary_n: int = PRIMARY_N,
    uncensored_n: int = DEFAULT_UNCENSORED_N,
) -> list[dict[str, Any]]:
    questions = qeval["questions"]
    if not 1 <= primary_n <= len(questions):
        raise ValueError(f"primary_n must be in [1, {len(questions)}]")
    primary = sorted(
        random.Random(PRIMARY_SEED).sample(questions, primary_n),
        key=lambda row: row["question_id"],
    )
    if arm not in UNCENSORED_ARMS:
        return primary
    if not 1 <= uncensored_n <= len(primary):
        raise ValueError(f"uncensored_n must be in [1, {len(primary)}]")
    return sorted(
        random.Random(UNCENSORED_SEED).sample(primary, uncensored_n),
        key=lambda row: row["question_id"],
    )


def emit_probe_question_pin(
    arm: str, questions: Iterable[dict[str, Any]], output_dir: str | Path
) -> dict[str, str]:
    """Freeze the exact probe qids and verify them through bench.tools.pinverify."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    probe_path = root / f"probe-questions-{arm}.txt"
    content = "".join(f"{row['question_id']}\n" for row in questions)
    if probe_path.exists() and probe_path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to replace drifted probe list: {probe_path}")
    if not probe_path.exists():
        probe_path.write_text(content, encoding="utf-8")

    digest = sha256_file(probe_path)
    pins_path = root / f"probe-questions-{arm}.pins.yaml"
    pins = {
        "files": {
            "probe_questions": {"path": str(probe_path), "sha256": digest}
        },
        "version": 1,
    }
    encoded = json.dumps(pins, sort_keys=True, separators=(",", ":")) + "\n"
    if pins_path.exists() and pins_path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"refusing to replace drifted probe pin: {pins_path}")
    if not pins_path.exists():
        pins_path.write_text(encoded, encoding="utf-8")

    report_path = root / f"PINS-PROBE-{arm}-PRERUN.txt"
    passed, _, _ = verify(pins_path, "pre-run", report_path)
    if not passed:
        raise ValueError(f"probe question pin verification failed: {report_path}")
    return {
        "path": str(probe_path),
        "sha256": digest,
        "pins": str(pins_path),
        "report": str(report_path),
    }


def _hist(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key))
        counts[label] = counts.get(label, 0) + 1
    return counts


def summarise_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the F34 summary while adding explicit metric granularity."""
    values = list(rows)
    if not values:
        raise ValueError("cannot summarise zero rows")
    signatures = [fail_close_signature(row) for row in values]
    if any(signatures):
        raise ValueError("scale metric rows contain fail-closed results")

    latencies = [float(row["latency_ms"]) for row in values]
    latencies_sorted = sorted(latencies)
    reps = sorted({int(row["rep"]) for row in values})
    per_rep = {
        rep: round(
            statistics.mean(
                float(row["latency_ms"]) for row in values if int(row["rep"]) == rep
            ),
            2,
        )
        for rep in reps
    }
    turn_values = [
        int(row["answer_turn_delivered_complete"])
        for row in values
        if row.get("answer_turn_delivered_complete") is not None
    ]
    session_values = [
        int(row.get("session_gold_all", row["all_gold"])) for row in values
    ]
    latency_mean = statistics.mean(latencies)
    session_mean = round(statistics.mean(session_values), 4)
    summary = {
        "arm": values[0]["arm"],
        "scale": values[0]["scale"],
        "n_questions": len({row["question_id"] for row in values}),
        "reps": len(reps),
        "latency_mean_ms": round(latency_mean, 2),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(
            latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))], 2
        ),
        "latency_max_ms": round(max(latencies), 2),
        "rep_means_ms": [per_rep[rep] for rep in reps],
        "rep_mean_spread_pct": round(
            100 * (max(per_rep.values()) - min(per_rep.values())) / latency_mean,
            2,
        ),
        "recall_macro": round(
            statistics.mean(float(row["recall"]) for row in values), 4
        ),
        "hit_at_k": round(
            statistics.mean(float(row["hit_at_k"]) for row in values), 4
        ),
        "session_gold_all": session_mean,
        "all_gold": session_mean,
        "answer_turn_delivered_complete": (
            round(statistics.mean(turn_values), 4) if turn_values else None
        ),
        "answer_turn_delivered_complete_n": len(turn_values),
        "rows_returned_mean": round(
            statistics.mean(float(row["rows_returned"]) for row in values), 2
        ),
        "degraded_frac": round(
            statistics.mean(1 if row.get("degraded") else 0 for row in values), 4
        ),
        "empty_frac": round(
            statistics.mean(float(row.get("empty", 0)) for row in values), 4
        ),
        "cov_fts_hist": _hist(values, "cov_fts"),
        "cov_summary_hist": _hist(values, "cov_summary"),
        "cov_chunk_hist": _hist(values, "cov_chunk"),
        "degraded_reason_hist": _hist(values, "degraded_reason"),
        "fail_closed_n": 0,
    }
    return summary
