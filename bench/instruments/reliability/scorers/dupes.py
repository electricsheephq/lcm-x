"""SQL replay-duplicate counter (diagnostic), ported from the gauntlet ``sql_dup_counter.py``; uses no LCM-X code.

Per session, rows ordered by store_id:
- replay identity = (role, content, tool_calls, tool_call_id); a replay copy repeats an EARLIER row's identity;
- ingest batch = a maximal run of rows whose ingested_at gaps are all <= ``batch_gap`` seconds;
- burst = a maximal run of >= 2 replay copies inside one batch whose originals come from >= 2 earlier batches
  (a single-origin pair is a heartbeat, not a replay);
- counted = burst rows whose batch starts at or after the first compaction (summary_nodes.created_at, minus 1 s).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict


def count(db_path, batch_gap: float = 0.5, compaction_ts=()) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "select store_id, session_id, role, coalesce(content,''), coalesce(tool_calls,''), coalesce(tool_call_id,''),"
            " coalesce(ingested_at, 0) from messages order by session_id, store_id").fetchall()
        comp = sorted([float(r[0]) for r in db.execute("select created_at from summary_nodes where created_at is not null")]
                      + [float(x) for x in compaction_ts])
    finally:
        db.close()
    first_comp = comp[0] if comp else None
    by_session = defaultdict(list)
    for r in rows:
        by_session[r[1]].append(r)
    sessions, counted_total, any_total, key_extra, naive_extra = [], 0, 0, 0, 0
    for sid, srows in by_session.items():
        batch_of, batch_start, b, prev = [], [], -1, None
        for r in srows:
            t = float(r[6])
            if prev is None or t - prev > batch_gap:
                b += 1
                batch_start.append(t)
            batch_of.append(b)
            prev = t
        first_seen, origin, seen_content = {}, [], set()
        for i, r in enumerate(srows):
            key = (r[2], r[3], r[4], r[5])
            if key in first_seen:
                origin.append(batch_of[first_seen[key]])
                key_extra += 1
            else:
                first_seen[key] = i
                origin.append(None)
            naive_extra += (r[2], r[3]) in seen_content
            seen_content.add((r[2], r[3]))
        bursts, i = [], 0
        while i < len(srows):
            if origin[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < len(srows) and origin[j + 1] is not None and batch_of[j + 1] == batch_of[i]:
                j += 1
            run = range(i, j + 1)
            origins = {origin[k] for k in run}
            if len(run) >= 2 and len(origins) >= 2:
                start = batch_start[batch_of[i]]
                bursts.append({"first_store_id": srows[i][0], "last_store_id": srows[j][0], "rows": len(run),
                               "after_compaction": first_comp is not None and start >= first_comp - 1.0})
            i = j + 1
        counted = sum(x["rows"] for x in bursts if x["after_compaction"])
        counted_total += counted
        any_total += sum(x["rows"] for x in bursts)
        sessions.append({"session_id": sid, "rows": len(srows), "bursts": bursts[:20], "replayed_rows_after_compaction": counted})
    return {"messages_total": len(rows), "sessions": len(by_session), "first_compaction_ts": first_comp,
            "replayed_rows_after_compaction": counted_total, "burst_rows_any_time": any_total,
            "key_based_extra_rows_all_time": key_extra, "naive_content_only_extra_rows": naive_extra,
            "per_session": sessions}
