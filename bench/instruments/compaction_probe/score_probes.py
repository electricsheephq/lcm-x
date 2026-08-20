#!/usr/bin/env python3
"""Mechanically score compaction-probe answers."""

from __future__ import annotations

import argparse
import json
import re
import warnings
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# Expanded 2026-08-20 pre-official-scoring (disclosed in RUN-SHEET amendment):
# the original set missed the model's dominant abstention phrasings observed in
# smokes ("You didn't state …", "It isn't present in the conversation …"),
# which would bias traps toward false HALLUCINATE. CORRECT is checked before
# the abstain regex, so retention (CORRECT/canaries) is unaffected by
# construction; only the ABSTAIN/HALLUCINATE split moves. Applied uniformly to
# every arm.
ABSTAIN_PATTERNS = [
    r"(?i)\b(don'?t|do not|didn'?t|did not|doesn'?t|does not|can'?t|cannot|"
    r"couldn'?t|could not)\s+(know|recall|find|locate|see|have|remember|"
    r"mention|state|specify|provide|say|share|define|give|record|include)\b",
    r"(?i)\b(isn'?t|is not|wasn'?t|was not|aren'?t|are not|weren'?t|were not|"
    r"not)\s+(present|mentioned|specified|stated|provided|included|available|"
    r"recorded|defined|given|said|shared|sure|in (the|our|my))\b",
    r"(?i)\bno (record|memory|information|mention|such)\b",
    r"(?i)\bno\b.{0,60}\b(was|were|has been|have been)\s+(provided|"
    r"established|decided|set|given|specified|mentioned|recorded|defined|"
    r"chosen)\b",
    r"(?i)\bnever (mentioned|stated|specified|provided|said|defined|recorded|"
    r"shared|gave|included|decided|chose|set|established|agreed|picked)\b",
    r"(?i)\bunable to (find|locate|recall|see)\b",
    r"(?i)\banswer is:?\s*unknown\b",
    r"(?i)\bunsure\b",
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
                value = _canary_entry_value(row)
                result[str(key)] = value
            else:
                result[str(index)] = row
        return result
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                result[str(key)] = _canary_entry_value(value)
            else:
                result[str(key)] = value
        return result
    raise ValueError("canaries JSON must be a list or object")


def _canary_entry_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("value", "answer", "expected", "expected_value", "canary"):
            if value.get(key) is not None:
                return value[key]
        return None
    return value


def _canary_value(probe: Any, canaries: dict[str, Any]) -> Any:
    direct = _field(probe, "canary_value", "expected_value", "expected", "value")
    if direct is not None:
        return direct
    canary_id = _field(probe, "canary_id", "canary")
    if isinstance(canary_id, dict):
        return _canary_entry_value(canary_id)
    if canary_id is not None:
        return canaries.get(str(canary_id))
    # gen_material schema: probe rows carry no canary_id; the join key is the
    # probe's own id (probes.jsonl id == canaries.json id).
    own_id = _field(probe, "id", "probe_id")
    if own_id is not None:
        return canaries.get(str(own_id))
    return None


def _is_trap(probe: Any, canary_value: Any = None) -> bool:
    del canary_value  # Explicit trap metadata, never an unresolved-positive fallback.
    value = _field(probe, "trap", "is_trap", "negative", default=None)
    if value is None:
        # gen_material schema marks traps via kind: "trap".
        kind = _field(probe, "kind")
        return isinstance(kind, str) and kind.casefold() in {"trap", "negative"}
    if isinstance(value, str):
        value = value.casefold() in {"1", "true", "yes", "trap", "negative"}
    return bool(value)


def _raw_answer(result: Any) -> tuple[str, bool]:
    if not isinstance(result, dict):
        return "", True
    value = result.get("raw_answer", result.get("answer", result.get("output")))
    if not isinstance(value, str) or not value.strip():
        return "", True
    # Model output uses curly apostrophes ("don’t"); fold to ASCII so the
    # abstain patterns' '?  alternations match real phrasing.
    return value.replace("’", "'").replace("‘", "'"), False


def classify(
    raw_answer: Any,
    canary_value: Any,
    trap: bool,
    *,
    timed_out: bool = False,
) -> tuple[str, bool]:
    if timed_out or (isinstance(raw_answer, dict) and raw_answer.get("timed_out") is True):
        return "TIMEOUT", True
    answer, unparseable = _raw_answer(raw_answer if isinstance(raw_answer, dict) else {"raw_answer": raw_answer})
    expected = normalize(canary_value)
    # Explicit traps always use the abstention rule.  A trap's optional value
    # is diagnostic metadata only and must never make a fabricated answer CORRECT.
    if trap:
        if not unparseable and any(pattern.search(answer) for pattern in ABSTAIN_RE):
            return "ABSTAIN", False
        return "HALLUCINATE", unparseable
    if not unparseable and expected and expected in normalize(answer):
        return "CORRECT", False
    if not unparseable and any(pattern.search(answer) for pattern in ABSTAIN_RE):
        return "ABSTAIN", False
    return "HALLUCINATE", unparseable


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    traps = sum(bool(row["trap"]) for row in rows)
    timeouts = counts["TIMEOUT"]
    canaries = max(0, len(rows) - traps - timeouts)
    correct_negative = sum(row["classification"] == "ABSTAIN" and row["trap"] for row in rows)
    correct = counts["CORRECT"]
    total = len(rows)
    result: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "abstain": counts["ABSTAIN"],
        "hallucinate": counts["HALLUCINATE"],
        "timeout": timeouts,
        "timed_out": timeouts,
        "correct_negative": correct_negative,
        "canary_total": canaries,
        "trap_total": traps,
        "unparseable": sum(bool(row["unparseable"]) for row in rows),
        "retention": (correct / canaries) if canaries else None,
        "accuracy": ((correct + correct_negative) / total) if total else None,
    }
    return result


def score(results_path: Path, canaries_path: Path, probes_path: Path) -> dict[str, Any]:
    canary_payload = _load_json(canaries_path)
    canaries = _canary_map(canary_payload)
    canary_meta: dict[str, dict[str, Any]] = {}
    meta_rows = canary_payload.get("canaries") if isinstance(canary_payload, dict) else canary_payload
    if isinstance(meta_rows, list):
        for row in meta_rows:
            if isinstance(row, dict):
                key = row.get("canary_id", row.get("id", row.get("name")))
                if key is not None:
                    canary_meta[str(key)] = row
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
        trap = _is_trap(probe)
        if not trap and (value is None or not str(value).strip()):
            canary_id = _field(probe, "canary_id", "canary", default="<missing>")
            raise ValueError(
                f"positive probe {pid!r} has unresolved canary {canary_id!r}"
            )
        if trap and value is not None:
            warnings.warn(
                f"trap probe {pid!r} carries a value; ignoring it for scoring",
                UserWarning,
                stacklevel=2,
            )
        meta = canary_meta.get(pid, {})
        canary_id = _field(probe, "canary_id", "canary")
        if canary_id is None and pid in canary_meta:
            canary_id = pid  # resolved via the probe-id join (gen_material schema)
        scored = {
            "probe_id": pid,
            "epoch": str(
                _field(probe, "epoch", "epoch_id", default=None)
                or meta.get("epoch")
                or "unknown"
            ),
            "class": str(
                _field(probe, "class", "info_class", "category", default=None)
                or meta.get("class")
                or "unknown"
            ),
            "trap": trap,
            "canary_id": canary_id,
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
            timed_out = bool(isinstance(result, dict) and result.get("timed_out") is True)
            classification, classified_unparseable = classify(
                result or {},
                probe["canary_value"],
                probe["trap"],
                timed_out=timed_out,
            )
            unparseable = unparseable or classified_unparseable or result is None
        if duplicate:
            timed_out = False
        elif result is None:
            timed_out = False
        scored_rows.append(
            {
                **probe,
                "raw_answer": raw_answer,
                "classification": classification,
                "score": classification,
                "unparseable": bool(unparseable),
                "timed_out": bool(timed_out),
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
