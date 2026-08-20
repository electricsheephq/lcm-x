#!/usr/bin/env python3
"""Mechanically score compaction-probe answers."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ABSTAIN_PATTERNS = [
    r"(?i)\b(don'?t know|do not know|no (record|memory|information)|not (sure|mentioned|specified)|cannot (recall|find)|unsure)\b",
]
ABSTAIN_RE = tuple(re.compile(pattern) for pattern in ABSTAIN_PATTERNS)


def normalize(value: Any) -> str:
    """Casefold and remove punctuation/whitespace for substring matching."""
    text = str(value or "").casefold()
    return "".join(
        char
        for char in text
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> tuple[list[Any], int]:
    rows: list[Any] = []
    invalid = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid += 1
    return rows, invalid


def _probe_id(row: Any, index: int) -> str:
    if isinstance(row, dict):
        for key in ("probe_id", "id", "name"):
            if row.get(key) is not None:
                return str(row[key])
    return f"probe-{index}"


def _metadata(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _field(row: Any, *names: str, default: Any = None) -> Any:
    if not isinstance(row, dict):
        return default
    metadata = _metadata(row)
    for name in names:
        if row.get(name) is not None:
            return row[name]
        if metadata.get(name) is not None:
            return metadata[name]
    return default


def _canary_map(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "canaries" in payload:
        payload = payload["canaries"]
    if isinstance(payload, list):
        result = {}
        for index, row in enumerate(payload):
            if isinstance(row, dict):
                key = row.get("canary_id", row.get("id", row.get("name", index)))
                value = row.get(
                    "value",
                    row.get(
                        "answer",
                        row.get("expected", row.get("expected_value", row.get("canary"))),
                    ),
                )
                result[str(key)] = value
            else:
                result[str(index)] = row
        return result
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                result[str(key)] = value.get(
                    "value",
                    value.get(
                        "answer",
                        value.get("expected", value.get("expected_value", value.get("canary"))),
                    ),
                )
            else:
                result[str(key)] = value
        return result
    raise ValueError("canaries JSON must be a list or object")


def _canary_value(probe: Any, canaries: dict[str, Any]) -> Any:
    direct = _field(probe, "canary_value", "expected_value", "expected", "value")
    if direct is not None:
        return direct
    canary_id = _field(probe, "canary_id", "canary")
    if isinstance(canary_id, dict):
        return canary_id.get("value", canary_id.get("answer"))
    if canary_id is not None:
        return canaries.get(str(canary_id))
    return None


def _is_trap(probe: Any, canary_value: Any) -> bool:
    value = _field(probe, "trap", "is_trap", "negative", default=False)
    if isinstance(value, str):
        value = value.casefold() in {"1", "true", "yes", "trap", "negative"}
    return bool(value) or canary_value is None


def _raw_answer(result: Any) -> tuple[str, bool]:
    if not isinstance(result, dict):
        return "", True
    value = result.get("raw_answer", result.get("answer", result.get("output")))
    if not isinstance(value, str) or not value.strip():
        return "", True
    return value, False


def classify(raw_answer: Any, canary_value: Any, trap: bool) -> tuple[str, bool]:
    answer, unparseable = _raw_answer(raw_answer if isinstance(raw_answer, dict) else {"raw_answer": raw_answer})
    expected = normalize(canary_value)
    if not unparseable and expected and expected in normalize(answer):
        return "CORRECT", False
    if not unparseable and any(pattern.search(answer) for pattern in ABSTAIN_RE):
        return "ABSTAIN", False
    return "HALLUCINATE", unparseable


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    traps = sum(bool(row["trap"]) for row in rows)
    canaries = len(rows) - traps
    correct_negative = sum(row["classification"] == "ABSTAIN" and row["trap"] for row in rows)
    correct = counts["CORRECT"]
    total = len(rows)
    result: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "abstain": counts["ABSTAIN"],
        "hallucinate": counts["HALLUCINATE"],
        "correct_negative": correct_negative,
        "canary_total": canaries,
        "trap_total": traps,
        "unparseable": sum(bool(row["unparseable"]) for row in rows),
        "retention": (correct / canaries) if canaries else None,
        "accuracy": ((correct + correct_negative) / total) if total else None,
    }
    return result


def score(results_path: Path, canaries_path: Path, probes_path: Path) -> dict[str, Any]:
    canaries = _canary_map(_load_json(canaries_path))
    probe_rows, invalid_probe_lines = _load_jsonl(probes_path)
    result_rows, invalid_result_lines = _load_jsonl(results_path)
    if invalid_probe_lines:
        raise ValueError(f"probes JSONL contains {invalid_probe_lines} invalid line(s)")

    probes: list[dict[str, Any]] = []
    probe_by_id: dict[str, dict[str, Any]] = {}
    for index, probe in enumerate(probe_rows, 1):
        pid = _probe_id(probe, index)
        if pid in probe_by_id:
            raise ValueError(f"duplicate probe id in probes JSONL: {pid}")
        value = _canary_value(probe, canaries)
        scored = {
            "probe_id": pid,
            "epoch": str(_field(probe, "epoch", "epoch_id", default="unknown")),
            "class": str(_field(probe, "class", "info_class", "category", default="unknown")),
            "trap": _is_trap(probe, value),
            "canary_id": _field(probe, "canary_id", "canary"),
            "canary_value": value,
        }
        probe_by_id[pid] = scored
        probes.append(scored)

    result_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in result_rows:
        if not isinstance(row, dict) or row.get("kind") != "probe":
            continue
        pid = row.get("probe_id")
        if pid is None:
            continue
        pid = str(pid)
        if pid in result_by_id:
            duplicate_ids.add(pid)
        else:
            result_by_id[pid] = row

    scored_rows: list[dict[str, Any]] = []
    for probe in probes:
        pid = probe["probe_id"]
        result = result_by_id.get(pid)
        duplicate = pid in duplicate_ids
        if duplicate:
            classification, unparseable = "HALLUCINATE", True
            raw_answer = ""
        else:
            raw_answer, unparseable = _raw_answer(result)
            classification, classified_unparseable = classify(
                result or {}, probe["canary_value"], probe["trap"]
            )
            unparseable = unparseable or classified_unparseable or result is None
        scored_rows.append(
            {
                **probe,
                "raw_answer": raw_answer,
                "classification": classification,
                "score": classification,
                "unparseable": bool(unparseable),
                "correct_negative": bool(probe["trap"] and classification == "ABSTAIN"),
            }
        )

    by_epoch_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_epoch_rows[row["epoch"]].append(row)
        by_class_rows[row["class"]].append(row)
    totals = _aggregate(scored_rows)
    return {
        "schema_version": 1,
        "probe_count": len(scored_rows),
        "probes": scored_rows,
        "rows": scored_rows,
        "per_epoch": {key: _aggregate(value) for key, value in sorted(by_epoch_rows.items())},
        "per_class": {key: _aggregate(value) for key, value in sorted(by_class_rows.items())},
        "by_epoch": {key: _aggregate(value) for key, value in sorted(by_epoch_rows.items())},
        "by_class": {key: _aggregate(value) for key, value in sorted(by_class_rows.items())},
        "totals": totals,
        "three_way_totals": {
            "CORRECT": totals["correct"],
            "ABSTAIN": totals["abstain"],
            "HALLUCINATE": totals["hallucinate"],
        },
        "three_way": {
            "CORRECT": totals["correct"],
            "ABSTAIN": totals["abstain"],
            "HALLUCINATE": totals["hallucinate"],
        },
        "aggregates": {
            "per_epoch": {key: _aggregate(value) for key, value in sorted(by_epoch_rows.items())},
            "per_class": {key: _aggregate(value) for key, value in sorted(by_class_rows.items())},
        },
        "invalid_probe_lines": invalid_probe_lines,
        "invalid_result_lines": invalid_result_lines,
        "duplicate_result_probe_ids": sorted(duplicate_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--canaries", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = score(Path(args.results), Path(args.canaries), Path(args.probes))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"score_probes.py: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
