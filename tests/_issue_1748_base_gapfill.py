"""Frozen #1748 base f3cc8de918ba14c21ae246099652762a26255554 for selection proof."""
import json
from hermes_lcm.command import CHUNK_POLICY_REVISION, chunk_message


def _assert_frozen_eligibility():
    # This arm intentionally calls live chunking, but its eligibility must stay
    # at revision 2. Stop the comparison rather than silently move the baseline.
    assert CHUNK_POLICY_REVISION == 2, "frozen base revision changed"
    corpus = [("user", "short"), ("assistant", "x"), ("tool", "result"),
              ("system", "instruction"), (" USER ", "x"), ("user", "\n"),
              ("assistant", "\t"), ("tool", "\u00a0"), ("user", "\u3000"),
              ("user", None), ("user", [{"type": "text", "text": "fact"}])]
    expected = {
        "conversational": (True, True, False, False, True, False, False, False, False, False, True),
        "heads": (True, True, True, False, True, False, False, False, False, False, True),
        "full": (True, True, True, True, True, False, False, False, False, False, True),
    }
    for policy, eligibility in expected.items():
        assert tuple(any(c.char_end > c.char_start for c in chunk_message(i, role, content, policy=policy))
                     for i, (role, content) in enumerate(corpus, 1)) == eligibility, "frozen base eligibility changed"


def _chunk_gapfill_rows(conn, identity_hash, bounds, policy, limit, *, record_progress=False,
                        gap_page_size=None, gap_scan_state=None):
    """Scan new territory once; revisit unpublished candidates by id until covered."""
    _assert_frozen_eligibility()
    low, high = bounds
    scan_key = f"chunk_gapfill_scan:{identity_hash}"
    saved = conn.execute("SELECT value FROM metadata WHERE key = ?", (scan_key,)).fetchone()
    try:
        scan = json.loads(saved[0]) if saved else {}
        valid = (
            isinstance(scan, dict)
            and type(scan.get("revision")) is int and type(scan.get("high")) is int
            and isinstance(scan.get("policy"), str)
            and ("pending" not in scan or (isinstance(scan["pending"], list)
                 and all(type(i) is int for i in scan["pending"])))
            and ("ceiling" not in scan or type(scan["ceiling"]) is int)
        )
        if not valid:
            scan = {}
    except Exception:
        scan = {}  # Unusable progress metadata requires the full gap revalidation.
    if scan.get("revision") != CHUNK_POLICY_REVISION or scan.get("policy") != policy:
        scan = {}
    pending_ids = []
    ceiling = high + 1
    if "pending" in scan and scan["high"] <= high:
        pending_ids = sorted({i for i in scan["pending"] if i >= low}, reverse=True)
        low = max(low, scan["high"] + 1)
    elif scan.get("high") == high:
        # Only our own scan can certify a zero-chunk prefix.
        ceiling = scan.get("ceiling", ceiling)
    roles = "'assistant', 'user', 'tool'" if policy == "heads" else "'assistant', 'user'"
    page_size = max(1, min(gap_page_size or limit, limit))
    if gap_scan_state is not None:
        gap_scan_state["unproven"] = True
    candidates = []

    def record(progress, values=()):
        if not record_progress:
            return
        values = [*values, (scan_key, json.dumps(progress))]
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES " + ",".join("(?, ?)" for _ in values)
            + " ON CONFLICT(key) DO UPDATE SET value = excluded.value WHERE value != excluded.value",
            tuple(item for pair in values for item in pair),
        )

    def record_ceiling(*, pending_complete=False):
        # Keep unread pending ids. A fully replayed, empty list needs no skip
        # certificate; dropping it reopens the whole range below the ceiling.
        if ceiling < high + 1 and ("pending" not in scan or not pending_ids or pending_complete):
            record({"revision": CHUNK_POLICY_REVISION, "policy": policy, "high": high, "ceiling": ceiling})

    reached = 0
    page_ceiling = ceiling
    range_complete = low > high
    # Fetch each page completely: no SQLite cursor survives a yield to a writer.
    while not range_complete and reached < limit:
        take = min(page_size, limit - reached)
        rows = conn.execute(
            f"SELECT store_id, role, content FROM messages AS m "
            f"WHERE store_id >= ? AND store_id <= ? AND store_id < ? "
            f"AND lower(trim(role)) IN ({roles}) AND trim(COALESCE(content, '')) != '' "
            "AND NOT EXISTS (SELECT 1 FROM lcm_chunk_meta AS c "
            "WHERE c.store_id = m.store_id AND c.identity_hash = ?) "
            "ORDER BY store_id DESC LIMIT ?",
            (low, high, page_ceiling, identity_hash, take),
        ).fetchall()
        range_complete = len(rows) < take
        for row in rows:
            reached += 1
            if any(c.char_end > c.char_start for c in chunk_message(*row, policy=policy)):
                if not candidates:
                    record_ceiling()  # Deterministic, before the first candidate escapes.
                candidates.append(row[0])
                yield row
            elif not candidates:
                ceiling = min(ceiling, row[0])
        if rows:
            page_ceiling = rows[-1][0]
    # A saved pending range is older than the new range; replay it newest first.
    for start in range(0, len(pending_ids), min(500, page_size)):
        batch = pending_ids[start:start + min(500, page_size)]
        rows = conn.execute(
            "SELECT store_id, role, content FROM messages AS p "
            f"WHERE store_id IN ({','.join('?' for _ in batch)}) "
            "AND NOT EXISTS (SELECT 1 FROM lcm_chunk_meta AS c "
            "WHERE c.store_id = p.store_id AND c.identity_hash = ?) ORDER BY store_id DESC",
            (*batch, identity_hash),
        ).fetchall()
        for row in rows:
            if any(c.char_end > c.char_start for c in chunk_message(*row, policy=policy)):
                if not candidates:
                    record_ceiling()
                candidates.append(row[0])
                yield row
            elif not candidates and range_complete:
                ceiling = min(ceiling, row[0])
    if range_complete:
        if gap_scan_state is not None:
            gap_scan_state["unproven"] = False
        if len(candidates) < limit:
            # Only natural exhaustion certifies every range row and pending id.
            progress = {"revision": CHUNK_POLICY_REVISION, "policy": policy, "high": high,
                        "pending": candidates}
            values = [(f"chunk_policy_revision_through:{identity_hash}",
                       str(min(candidates, default=high + 1) - 1))]
            if not candidates:
                values.append((f"chunk_policy_revision:{identity_hash}", str(CHUNK_POLICY_REVISION)))
            record(progress, values)
    if not range_complete:
        record_ceiling(pending_complete=True)
