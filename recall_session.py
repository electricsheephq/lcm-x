"""Opt-in post-ranking session packs. No retrieval, writes or sidecar restores."""
from collections import deque
import copy
from functools import partial
import json
from pathlib import Path
import sqlite3
import time

from . import tokens
from .reasoning import resolve_occurrence_time
from .retrieval_context import RetrievalContext

TOKEN_BUDGET, CHAR_BUDGET = 16_000, 64_000


def session_schema(schema):
    if schema["name"] != "lcm_recall":
        return schema
    schema = copy.deepcopy(schema)
    schema["description"] = (
        "Search memory across all conversations using the existing ranked arms. "
        "Session delivery returns source messages around ranked seeds, possibly partial sessions, "
        "within 16,000 estimated tokens and 64,000 JSON characters. Both details return full source rows. "
        "limit bounds seeds; total_results can exceed limit. include selects seed arms; "
        "even summary seeds expand to raw source messages. Explicit exclusions remain hard filters."
    )
    props = schema["parameters"]["properties"]
    props["limit"]["description"] = "Ranked seed limit (default 8, maximum 25), not delivered message count."
    props["include"]["description"] = "Retrieval arms for seeds; session delivery returns raw source messages."
    return schema


def deliver_session(ordered, response, *, engine, deadline, excluded, allowed, seen_refs,
                    delta_requested, event_time_fields, occurrence, summary_leads):
    encoder = tokens._get_encoder()
    count = partial(tokens.count_tokens_with_encoder, encoder=encoder)
    delivery = dict(unit="session", seed_limit=response["limit"], requested_seed_limit=response.get(
        "limit_clamped_from", response["limit"]), token_budget=TOKEN_BUDGET, char_budget=CHAR_BUDGET,
        estimator="cl100k_base" if encoder is not None else "fallback", response_tokens=0,
        response_chars=0, session_count=0, message_count=0, truncated=False, sessions=[])
    delivery["query_truncated"] = len(response["query"]) > 4096
    delivery["summary_leads"] = [lead for lead in summary_leads[:response["limit"]]
        if lead.get("session_id") not in excluded and
        (allowed is None or lead.get("session_id") in allowed)]
    response["provenance"]["delivery"] = delivery
    response["query"] = response["query"][:4096]
    admitted, matched, rows, adjacency = [], {}, {}, {}
    conn = None

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError

    def encode():
        hits = [item for group in admitted for item in group]
        priority = {s["session_id"]: i for i, s in enumerate(delivery["sessions"])}
        hits.sort(key=lambda h: (priority[h["session_id"]], h["store_id"]))
        response["hits"], response["total_results"] = hits, len(hits)
        delivery.update(message_count=len(hits), session_count=len({h["session_id"] for h in hits}))
        for meta in delivery["sessions"]:
            ids = [h["store_id"] for h in hits if h["session_id"] == meta["session_id"]]
            # Ranges are consecutive source-order runs, never numeric-ID guesses
            # across interleaved sessions or gaps between anchor windows.
            ranges = []
            for sid in ids:
                if ranges and adjacency.get(ranges[-1][1], (None, None))[1] == sid:
                    ranges[-1][1] = sid
                else:
                    ranges.append([sid, sid])
            meta["returned_id_ranges"] = ranges
        if delta_requested:
            refs = [h["exact_ref"] for h in hits]
            response["delta"] = dict(protocol="exact-ref-delta-v1", seen_ref_count=len(seen_refs),
                novel_ref_count=len(refs), novel_refs=refs, progress=bool(refs),
                termination_reason=None if refs else "no_novel_exact_ref")
        for _ in range(10):
            raw = json.dumps(response, ensure_ascii=False)
            size = (count(raw), len(raw))
            if size == (delivery["response_tokens"], delivery["response_chars"]):
                return raw
            delivery["response_tokens"], delivery["response_chars"] = size
        raise ValueError("session size metadata did not converge")

    def fits():
        raw = encode()
        return len(raw) <= CHAR_BUDGET and count(raw) <= TOKEN_BUDGET

    def window(sid, anchor):
        check()
        page = store.load_session_window(sid, anchor_store_id=anchor, before=1, after=1)
        check()
        if any(r["session_id"] != sid for r in page):
            raise ValueError("session reader identity mismatch")
        rows.update({r["store_id"]: r for r in page})
        ids = [r["store_id"] for r in page]
        if anchor not in ids:
            return None
        i = ids.index(anchor)
        adjacency[anchor] = (ids[i - 1] if i else None, ids[i + 1] if i + 1 < len(ids) else None)
        return rows[anchor]

    def neighbor(sid, anchor, direction):
        if anchor not in adjacency:
            window(sid, anchor)
        return adjacency.get(anchor, (None, None))[direction]

    def group(sid, anchor):
        row = window(sid, anchor)
        if row is None:
            return [], None, None
        # Find the owning assistant before admitting a result. A malformed or
        # unreadable group is a lead only; bounded walk never guesses ownership.
        for _ in range(256):
            if row["role"] != "tool":
                break
            prior = neighbor(sid, row["store_id"], 0)
            row = window(sid, prior) if prior is not None else None
            if row is None:
                return [], None, None
        else:
            return [], None, None
        result = [row]
        calls = row.get("tool_calls") or []
        expected = [c.get("id") for c in calls]
        if calls and (row["role"] != "assistant" or None in expected or len(set(expected)) != len(expected)):
            return [], None, None
        if not calls and anchor != row["store_id"]:
            return [], None, None
        observed = []
        if calls:
            for _ in range(256):
                nxt = neighbor(sid, result[-1]["store_id"], 1)
                next_row = window(sid, nxt) if nxt is not None else None
                if next_row is None or next_row["role"] != "tool":
                    break
                result.append(next_row)
                observed.append(next_row.get("tool_call_id"))
            else:
                return [], None, None
            if sorted(observed, key=str) != sorted(expected, key=str):
                return [], None, None
        return result, neighbor(sid, row["store_id"], 0), neighbor(sid, result[-1]["store_id"], 1)

    def item(row):
        sid = row["store_id"]
        text = row.get("content") or ""
        seed = matched.get(sid)
        hit = dict(kind="message_excerpt", session_id=row["session_id"], store_id=sid,
            role=row["role"], source=row.get("source") or "", timestamp=row.get("timestamp") or 0,
            content=text, snippet=text[:300], content_offset=0, content_returned_chars=len(text),
            content_chars=len(text), content_truncated=False, evidence_span=dict(char_start=0, char_end=len(text)),
            exact_ref=f"lcm:{sid}:0-{len(text)}", expand_hint=f"lcm_expand(store_id={sid})",
            from_current_session=row["session_id"] == engine.current_session_id,
            score=round(float(seed["_final_score"]), 6) if seed else 0,
            arms=sorted(response["provenance"]["arms_run"][i] for i in seed["ranks"]) if seed else [],
            delivery_origin="ranked_seed" if seed else "session_neighbor", **event_time_fields(row))
        for key in ("tool_calls", "tool_call_id", "tool_name"):
            if row.get(key):
                hit[key] = row[key]
        if occurrence:
            dates = getattr(engine, "_session_occurrence_dates", {}) or {}
            hit["occurrence_time"] = resolve_occurrence_time(text, observed_at=row.get("observed_at") or 0,
                session_date=dates.get(row["session_id"]) or hit.get("event_time", "")[:10] or None)
            hit["occurrence_time"]["stored_at"] = row.get("ingested_at") or row.get("timestamp")
            hit["observation_time"] = dict(observed_at=row.get("observed_at") or None,
                ingested_at=row.get("ingested_at") or row.get("timestamp"),
                source="benchmark_session_date" if row["session_id"] in dates else
                       "host_message_timestamp" if row.get("observed_at") is not None else "ingest_fallback")
        return hit

    try:
        check()
        path = Path(engine._store.db_path).resolve()
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=max(.001, deadline - time.monotonic()))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute("BEGIN")
        io = RetrievalContext(engine._store, engine._dag).bind(conn, conn)
        store = io.store
        sessions = {}
        eligible = 0
        for rank, entry in enumerate(ordered):
            check()
            hit = entry["hit"]
            sid = hit.get("session_id")
            if sid in excluded or (allowed is not None and sid not in allowed):
                continue
            if hit.get("kind") == "summary":
                node = io.dag.get_node(int(hit["node_id"]))
                ids = io.dag.source_message_ids(node.node_id, limit=25) if node and node.session_id == sid else []
                sources = store.get_batch(ids) if ids else {}
                row = next((sources[i] for i in ids if i in sources and sources[i]["session_id"] == sid), None)
                if row is None and len(delivery["summary_leads"]) < response["limit"]:
                    delivery["summary_leads"].append(dict(node_id=hit["node_id"], session_id=sid,
                        expand_hint=f"lcm_expand(node_id={hit['node_id']}, session_id={json.dumps(sid)})"))
            else:
                row = store.get(int(hit["store_id"])) if hit.get("store_id") is not None else None
            check()
            if row is None or row["session_id"] != sid or row["store_id"] in matched:
                continue
            matched[row["store_id"]] = entry
            sessions.setdefault(sid, dict(session_id=sid, best_rank=rank, seed_ids=[],
                returned_id_ranges=[], complete=False, omission_reason=None))["seed_ids"].append(row["store_id"])
            eligible += 1
            if eligible >= response["limit"]:
                break
        delivery["sessions"] = list(sessions.values())
        for sid, meta in sessions.items():
            queue = deque(meta["seed_ids"])
            visited, emitted = set(), set()
            while queue:
                anchor = queue.popleft()
                if anchor in visited:
                    continue
                visited.add(anchor)
                batch, before, after = group(sid, anchor)
                if not batch:
                    meta["omission_reason"] = "unreadable_group"
                    meta["expand_hint"] = f"lcm_expand(store_id={anchor})"
                    continue
                visited.update(r["store_id"] for r in batch)
                pending = [item(r) for r in batch if r["store_id"] not in emitted]
                if delta_requested and any(h["exact_ref"] in seen_refs for h in pending):
                    meta["omission_reason"] = "no_novel_exact_ref"
                elif pending:
                    admitted.append(pending)
                    if fits():
                        emitted.update(h["store_id"] for h in pending)
                    else:
                        admitted.pop()
                        meta["omission_reason"] = "oversized_group" if not emitted else "budget"
                        meta["expand_hint"] = f"lcm_expand(store_id={anchor})"
                        # Try other ranked anchors/sessions, never repeatedly
                        # walk an over-budget neighborhood.
                        continue
                queue.extend(x for x in (before, after) if x is not None and x not in visited)
            meta["complete"] = meta["omission_reason"] is None
    except (TimeoutError, sqlite3.Error, OSError, ValueError) as exc:
        reason = "deadline" if time.monotonic() >= deadline else "read_error"
        # TimeoutError from a bounded reader is also an explicit deadline stop.
        if isinstance(exc, TimeoutError):
            reason = "deadline"
        if reason == "deadline":
            response["timeout"] = True
        for meta in delivery["sessions"]:
            if not meta["complete"]:
                meta["omission_reason"] = reason
        delivery["stop_reason"] = reason
    finally:
        if conn is not None:
            conn.close()
    delivery["truncated"] = bool(delivery.get("stop_reason")) or any(
        not s["complete"] for s in delivery["sessions"])
    # Final metadata and escaped JSON count too: refit whole groups and delta.
    while not fits() and admitted:
        removed = admitted.pop()
        for meta in delivery["sessions"]:
            if meta["session_id"] == removed[0]["session_id"]:
                meta.update(complete=False, omission_reason="budget")
        delivery["truncated"] = True
    while not fits():
        # Metadata is budgeted too, even for K=100 bench seeds and no citable
        # groups. Keep a bounded honest envelope rather than a first-item escape.
        delivery["truncated"] = True
        delivery["stop_reason"] = "budget"
        if delivery["summary_leads"]:
            delivery["summary_leads"].pop()
        elif delivery["sessions"]:
            delivery["sessions"].pop()
        elif response["query"]:
            response["query"] = response["query"][:len(response["query"]) // 2]
        else:
            break
    anchor = response["provenance"].get("fts_anchor")
    if anchor is not None:
        anchored_id = ordered[anchor["position"] - 1]["hit"].get("store_id") if anchor["fired"] else None
        anchor["delivered"] = any(h["store_id"] == anchored_id for h in response["hits"]) if anchor["fired"] else None
    return encode()
