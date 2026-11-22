"""Read-only shape census of an LCM-X store: counts and histograms only, for fitting the kit (#2868).

    python -B -m bench.instruments.gate_kit.live_rows.census --db STOPPED_COPY/lcm.db [--payloads DIR] \
        [--top 5] --out census.json

Opened with `mode=ro&immutable=1` (never written; refused when a non-empty -wal sits beside it). The output holds
counts, length buckets and per-conversation shares; no content, no ids, no hashes of content. Copies are counted
in memory (rows of a conversation whose role and content equal an earlier row of the same conversation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

BUCKETS = [0, 100, 400, 1_600, 6_400, 12_000, 25_600, 102_400]  # characters
EXTERNALIZED_HEAD = "[Externalized tool output"


def bucket(length: int) -> str:
    index = bisect_right(BUCKETS, length) - 1
    return f">={BUCKETS[index]}"


def census(db: Path, payloads: Path | None, top: int) -> dict[str, Any]:
    wal = Path(str(db) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise SystemExit("a non-empty -wal is beside the store: checkpoint a stopped copy first")
    with closing(sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        per: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "rows": 0, "roles": Counter(), "sessions": set(), "lengths": Counter(), "copies": 0,
            "unstamped": 0, "tool_call_ids": 0, "externalized": 0, "chars": 0, "seen": set()})
        for conv, session, role, content, call, stamp in conn.execute(
                "SELECT conversation_id, session_id, role, content, tool_call_id, timestamp FROM messages "
                "ORDER BY store_id"):
            item = per[str(conv)]
            text = content if isinstance(content, str) else str(content or "")
            item["rows"] += 1
            item["roles"][str(role)] += 1
            item["sessions"].add(session)
            item["lengths"][bucket(len(text))] += 1
            item["chars"] += len(text)
            item["unstamped"] += stamp is None
            item["tool_call_ids"] += bool(call)
            item["externalized"] += text.startswith(EXTERNALIZED_HEAD)
            digest = hashlib.sha256(f"{role}\0{text}".encode()).digest()
            if digest in item["seen"]:
                item["copies"] += 1
            else:
                item["seen"].add(digest)
        counts = {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                  for name in ("summary_nodes", "occurrence_bindings", "host_uid_bindings") if name in tables}
    conversations = sorted(per.values(), key=lambda item: -item["rows"])
    shaped = [{"rows": c["rows"], "roles": dict(c["roles"]), "sessions": len(c["sessions"]),
               "lengths": dict(c["lengths"]), "copies": c["copies"], "unstamped": c["unstamped"],
               "tool_call_ids": c["tool_call_ids"], "externalized": c["externalized"],
               "mean_chars": round(c["chars"] / c["rows"], 1) if c["rows"] else 0} for c in conversations[:top]]
    return {
        "store_rows": sum(c["rows"] for c in conversations), "conversations": len(conversations),
        "tool_rows": sum(c["roles"].get("tool", 0) for c in conversations),
        "externalized_rows": sum(c["externalized"] for c in conversations),
        "copies": sum(c["copies"] for c in conversations), "table_rows": counts,
        "payload_files": sum(1 for p in payloads.rglob("*") if p.is_file()) if payloads and payloads.exists() else None,
        "largest_conversations": shaped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.resolve().is_relative_to(args.db.resolve().parent):
        raise SystemExit("--out must not be beside the store")
    result = census(args.db, args.payloads, args.top)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print(json.dumps({k: result[k] for k in ("store_rows", "conversations", "copies", "payload_files")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
