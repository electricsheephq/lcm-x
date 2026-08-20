#!/usr/bin/env python3
"""Validate and render the disclosure-first benchmark scoreboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from html import escape as html_escape
from pathlib import Path
from typing import Any, Iterable


SCOREBOARD_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCOREBOARD_DIR / "results.jsonl"
OUTPUT_PATH = SCOREBOARD_DIR / "SCOREBOARD.md"

REQUIRED_FIELDS = (
    "id",
    "benchmark",
    "metric",
    "value",
    "display",
    "tier",
    "date",
    "system_commit",
    "harness_commit",
    "judge",
    "reader",
    "retrieval_config",
    "dataset_exposure",
    "breakdown",
    "variance",
    "failclose",
    "evidence",
    "caveats",
)
SCALAR_FIELDS = REQUIRED_FIELDS[:-2]


class ScoreboardError(ValueError):
    """A source row cannot safely be rendered."""


def _error(line_number: int, message: str) -> ScoreboardError:
    return ScoreboardError(f"line {line_number}: {message}")


def _validate_row(row: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise _error(line_number, "expected a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise _error(line_number, f"missing required field '{missing[0]}'")

    for field in SCALAR_FIELDS:
        value = row[field]
        if value is None:
            raise _error(line_number, f"field '{field}' must not be null")
        if field != "value" and not isinstance(value, str):
            raise _error(line_number, f"field '{field}' must be a string")
        if field != "value" and not value.strip():
            raise _error(line_number, f"field '{field}' must not be empty")

    value = row["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(line_number, "field 'value' must be a number")
    if not math.isfinite(value):
        raise _error(line_number, "field 'value' must be finite")

    if row["tier"] not in {"P", "F"}:
        raise _error(line_number, "field 'tier' must be 'P' or 'F'")
    try:
        date.fromisoformat(row["date"])
    except ValueError:
        raise _error(line_number, "field 'date' must be an ISO date") from None

    for field in ("evidence", "caveats"):
        values = row[field]
        if not isinstance(values, list):
            raise _error(line_number, f"field '{field}' must be a list")
        if any(not isinstance(item, str) for item in values):
            raise _error(line_number, f"field '{field}' items must be strings")

    if "superseded_by" in row:
        successor = row["superseded_by"]
        if not isinstance(successor, str) or not successor.strip():
            raise _error(line_number, "field 'superseded_by' must be a non-empty string")

    return row


def load_rows(path: str | Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Read and validate every non-empty JSONL row before any rendering."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise ScoreboardError(f"cannot read {source}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(line_number, f"malformed JSON ({exc.msg})") from None
            row = _validate_row(parsed, line_number)
            row_id = row["id"]
            if row_id in seen_ids:
                raise _error(
                    line_number,
                    f"duplicate id '{row_id}' (already used on line {seen_ids[row_id]})",
                )
            seen_ids[row_id] = line_number
            rows.append(row)

    for row in rows:
        successor = row.get("superseded_by")
        if successor is not None and successor not in seen_ids:
            raise ScoreboardError(
                f"row '{row['id']}' references missing successor '{successor}'"
            )
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inline(value: Any) -> str:
    """Keep table cells single-line and prevent pipes from changing the table."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _summary_cell(text: str, superseded: bool) -> str:
    return f"~~{text}~~" if superseded else text


def _render_summary(rows: Iterable[dict[str, Any]]) -> list[str]:
    lines = [
        "## Summary",
        "",
        "| Benchmark | Metric | Result | Tier | Date | Details |",
        "|---|---|---|---|---|---|",
    ]
    # Stable sorts make the requested benchmark-ascending/date-descending order
    # explicit while keeping same-day rows deterministic by id.
    ordered = sorted(rows, key=lambda item: item["id"])
    ordered = sorted(ordered, key=lambda item: date.fromisoformat(item["date"]), reverse=True)
    ordered = sorted(ordered, key=lambda item: item["benchmark"])
    for row in ordered:
        superseded = "superseded_by" in row
        details = _summary_cell(f"[details](#{row['id']})", superseded)
        if superseded:
            details += f" → [successor](#{row['superseded_by']})"
        lines.append(
            "| "
            + " | ".join(
                (
                    _summary_cell(_inline(row["benchmark"]), superseded),
                    _summary_cell(_inline(row["metric"]), superseded),
                    _summary_cell(_inline(row["display"]), superseded),
                    _summary_cell(_inline(row["tier"]), superseded),
                    _summary_cell(_inline(row["date"]), superseded),
                    details,
                )
            )
            + " |"
        )
    return lines


def _render_disclosure(row: dict[str, Any]) -> list[str]:
    lines = [f"### <a id=\"{html_escape(row['id'], quote=True)}\"></a>{_inline(row['id'])}", ""]
    fields = list(REQUIRED_FIELDS) + (["superseded_by"] if "superseded_by" in row else [])
    for field in fields:
        lines.append(f"**{field}:**")
        if field in ("evidence", "caveats"):
            values = row[field]
            lines.extend(f"- {_inline(item)}" for item in values or ["none"])
        else:
            lines.append(_inline(row[field]))
        lines.append("")
    return lines


def render(rows: Iterable[dict[str, Any]], source_sha256: str) -> str:
    """Render already-validated rows into deterministic Markdown."""
    values = list(rows)
    lines = [
        "# Scoreboard",
        "",
        "Every number ships with its full run config, variance, fail-close accounting, and known dataset defects — rows that cannot meet the standard do not render.",
        "",
        f"Generated from `results.jsonl` (sha256: `{source_sha256}`, rows: {len(values)})",
        "",
        *_render_summary(values),
        "",
        "## Row disclosures",
        "",
    ]
    for row in sorted(values, key=lambda item: item["id"]):
        lines.extend(_render_disclosure(row))
    return "\n".join(lines).rstrip() + "\n"


def generate(
    results_path: str | Path = RESULTS_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> str:
    """Validate, render, and write the complete scoreboard."""
    rows = load_rows(results_path)
    rendered = render(rows, sha256_file(results_path))
    Path(output_path).write_text(rendered, encoding="utf-8")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the disclosure-first scoreboard.")
    parser.add_argument("--check", action="store_true", help="validate only; do not write output")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_rows(args.results)
        if not args.check:
            rendered = render(rows, sha256_file(args.results))
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ScoreboardError) as exc:
        print(f"scoreboard: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
