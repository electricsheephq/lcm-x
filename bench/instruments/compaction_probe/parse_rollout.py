#!/usr/bin/env python3
"""Parse structural compaction and token telemetry from a Codex rollout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _timestamp_ms(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp() * 1000


def _content_chars(value: Any) -> int:
    """Count message text and encrypted replacement payload characters."""

    if isinstance(value, dict):
        total = 0
        if isinstance(value.get("encrypted_content"), str):
            total += len(value["encrypted_content"])
        content = value.get("content")
        if isinstance(content, (dict, list)):
            total += _content_chars(content)
        elif isinstance(content, str):
            total += len(content)
        if isinstance(value.get("text"), str):
            total += len(value["text"])
        return total
    if isinstance(value, list):
        return sum(_content_chars(item) for item in value)
    return 0


def _replacement_composition(history: Any) -> dict[str, int]:
    if not isinstance(history, list):
        history = []
    message_count = sum(
        1 for item in history if isinstance(item, dict) and item.get("type") == "message"
    )
    encrypted_blob_count = sum(
        1
        for item in history
        if isinstance(item, dict)
        and (isinstance(item.get("encrypted_content"), str) or item.get("type") == "encrypted_blob")
    )
    return {
        "message_count": message_count,
        "encrypted_blob_count": encrypted_blob_count,
        "total_chars": _content_chars(history),
    }


def _find_rollouts(session_id: str, sessions_root: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in sessions_root.rglob("*.jsonl")
        if session_id in path.name
    )
    if candidates:
        return candidates
    # The filename is normally sufficient, but older exports may be renamed.
    for path in sorted(sessions_root.rglob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as handle:
                first = json.loads(next(handle))
        except (OSError, StopIteration, json.JSONDecodeError):
            continue
        payload = first.get("payload", {}) if isinstance(first, dict) else {}
        if isinstance(payload, dict) and session_id in {
            payload.get("id"),
            payload.get("session_id"),
        }:
            candidates.append(path)
    return candidates


def _model_from_session(payload: dict[str, Any]) -> str | None:
    preferred = payload.get("base_instructions")
    if isinstance(preferred, dict):
        provenance = preferred.get("provenance")
        if isinstance(provenance, dict) and isinstance(provenance.get("model"), str):
            return provenance["model"]
    for node in _walk(payload):
        value = node.get("model")
        if isinstance(value, str) and value:
            return value
    return None


def _context_compaction_item(row: dict[str, Any]) -> dict[str, Any] | None:
    for node in _walk(row):
        if node.get("type") == "item_completed":
            item = node.get("item")
            if isinstance(item, dict) and item.get("type") == "ContextCompaction":
                return node
    return None


def _context_compacted(row: dict[str, Any]) -> bool:
    return any(node.get("type") == "context_compacted" for node in _walk(row))


def _parse_file(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append((line_no, row))

    session_payload: dict[str, Any] = {}
    cli_version: str | None = None
    model: str | None = None
    history_mode: str | None = None
    model_context_window: int | None = None
    token_series: list[dict[str, Any]] = []
    compacted_rows: list[tuple[int, dict[str, Any]]] = []
    context_rows: list[tuple[int, dict[str, Any]]] = []
    item_rows: list[tuple[int, dict[str, Any]]] = []

    for line_no, row in rows:
        if row.get("type") == "session_meta" and isinstance(row.get("payload"), dict):
            session_payload = row["payload"]
            cli_version = session_payload.get("cli_version")
            history_mode = session_payload.get("history_mode")
            model = _model_from_session(session_payload)
        if row.get("type") == "compacted":
            compacted_rows.append((line_no, row))
        if _context_compacted(row):
            context_rows.append((line_no, row))
        if _context_compaction_item(row) is not None:
            item_rows.append((line_no, row))
        for node in _walk(row):
            if node.get("type") != "token_count":
                continue
            info = node.get("info")
            if not isinstance(info, dict):
                continue
            last = info.get("last_token_usage")
            total = info.get("total_token_usage")
            context = info.get("model_context_window")
            if isinstance(context, int):
                model_context_window = context
            if not isinstance(last, dict) or not isinstance(total, dict):
                continue
            input_tokens = last.get("input_tokens")
            total_tokens = total.get("total_tokens")
            if not isinstance(input_tokens, (int, float)) or not isinstance(total_tokens, (int, float)):
                continue
            token_series.append(
                {
                    "ts": row.get("timestamp"),
                    "last_input": int(input_tokens),
                    "cumulative_total": int(total_tokens),
                }
            )
        if model is None:
            model = _model_from_session(row.get("payload", {})) if isinstance(row.get("payload"), dict) else None

    compactions: list[dict[str, Any]] = []
    for marker_index, (line_no, row) in enumerate(compacted_rows):
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        window_number = payload.get("window_number")
        if not isinstance(window_number, int):
            continue
        next_line = compacted_rows[marker_index + 1][0] if marker_index + 1 < len(compacted_rows) else None
        matching_item: dict[str, Any] | None = None
        for item_line, item_row in item_rows:
            if item_line > line_no and (next_line is None or item_line < next_line):
                matching_item = item_row
                break
        stall_ms: float | int | None = None
        if matching_item is not None:
            item_event = _context_compaction_item(matching_item)
            item = item_event.get("item", {}) if item_event else {}
            started = item_event.get("started_at_ms") if item_event else None
            completed = item_event.get("completed_at_ms") if item_event else None
            if started is None and isinstance(item, dict):
                started = item.get("started_at_ms")
            if completed is None and isinstance(item, dict):
                completed = item.get("completed_at_ms")
            if isinstance(started, (int, float)) and isinstance(completed, (int, float)):
                stall_ms = completed - started

        # Marker-to-context_compacted spacing is event-WRITE latency, not a
        # stall measurement -- reporting it as stall_ms would fabricate a
        # metric. When no ContextCompaction completion item exists, stall is
        # simply UNMEASURED for that boundary: stall_ms stays null and
        # stall_source says why (fail-closed metric semantics).
        stall_source = "item_completed" if stall_ms is not None else "unmeasured: no ContextCompaction item"
        compactions.append(
            {
                "window_number": window_number,
                "line_no": line_no,
                "prev_token_count_input": _previous_input(token_series, line_no, rows),
                "stall_ms": stall_ms,
                "stall_source": stall_source,
                "replacement_history_composition": _replacement_composition(
                    payload.get("replacement_history")
                ),
            }
        )

    return {
        "cli_version": cli_version,
        "model": model,
        "model_context_window": model_context_window,
        "compactions": compactions,
        "token_series": token_series,
        "history_mode": history_mode,
    }


def _previous_input(
    token_series: list[dict[str, Any]], marker_line: int, rows: list[tuple[int, dict[str, Any]]]
) -> int | None:
    """Return the last token input observed before a compaction marker."""

    marker_ts = _timestamp_ms(next(row.get("timestamp") for line, row in rows if line == marker_line))
    previous: int | None = None
    for point in token_series:
        point_ts = _timestamp_ms(point.get("ts"))
        if marker_ts is None or point_ts is None or point_ts <= marker_ts:
            previous = point["last_input"]
        else:
            break
    return previous


def parse_rollout(session_id: str, sessions_root: Path) -> dict[str, Any]:
    paths = _find_rollouts(session_id, sessions_root.expanduser())
    if not paths:
        raise FileNotFoundError(f"no rollout found for session {session_id} under {sessions_root}")
    reports = [_parse_file(path) for path in paths]
    if len(reports) == 1:
        return reports[0]
    # Multiple shards are uncommon, but concatenating their ordered telemetry
    # keeps the CLI useful for exports split at filesystem boundaries.
    result = reports[0]
    for report in reports[1:]:
        if result["cli_version"] is None:
            result["cli_version"] = report["cli_version"]
        if result["model"] is None:
            result["model"] = report["model"]
        if result["model_context_window"] is None:
            result["model_context_window"] = report["model_context_window"]
        result["compactions"].extend(report["compactions"])
        result["token_series"].extend(report["token_series"])
        if result["history_mode"] is None:
            result["history_mode"] = report["history_mode"]
    result["compactions"].sort(key=lambda row: (row["window_number"], row["line_no"]))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", type=Path, default=Path("~/.codex/sessions"))
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = parse_rollout(args.session_id, args.sessions_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "session_id": args.session_id,
                "compactions": len(report["compactions"]),
                "token_points": len(report["token_series"]),
                "model_context_window": report["model_context_window"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"parse_rollout.py: {exc}") from exc

