"""Opt-in post-ranking session packs. No retrieval, writes or sidecar restores."""
from collections import deque
from contextlib import ExitStack
import copy
from functools import partial
import json
from pathlib import Path
import sqlite3
import time

from . import tokens
from .message_analysis import _tool_call_id
from .reasoning import resolve_occurrence_time
from .recall_delivery import _without_event_time
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
    resolved_summaries, fit_failure = set(), "budget"
    skipped, omissions = [], {}
    conn = None
    live_reads = ExitStack()
    last_encoded = ""

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError

    def encode():
        nonlocal last_encoded
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
            refs = [h["exact_ref"] for h in hits if h.get("exact_ref")]
            response["delta"] = dict(protocol="exact-ref-delta-v1", seen_ref_count=len(seen_refs),
                novel_ref_count=len(refs), novel_refs=refs, progress=bool(refs),
                termination_reason=None if refs else "no_novel_exact_ref")
        anchor = response["provenance"].get("fts_anchor")
        if anchor is not None:
            anchored_id = ordered[anchor["position"] - 1]["hit"].get("store_id") if anchor["fired"] else None
            anchor["delivered"] = any(h["store_id"] == anchored_id for h in hits) if anchor["fired"] else None
        for _ in range(10):
            raw = json.dumps(response, ensure_ascii=False)
            size = (count(raw), len(raw))
            if size == (delivery["response_tokens"], delivery["response_chars"]):
                return raw
            delivery["response_tokens"], delivery["response_chars"] = size
        # The fallback estimator has discontinuities. Reject this payload;
        # fitting only removes whole groups, never retries an oscillating one.
        delivery.update(stop_reason="size_nonconvergence", truncated=True)
        last_encoded = json.dumps(response, ensure_ascii=False)
        return None

    def fits():
        nonlocal fit_failure
        raw = encode()
        fit_failure = "size_nonconvergence" if raw is None else "budget"
        return raw is not None and len(raw) <= CHAR_BUDGET and count(raw) <= TOKEN_BUDGET

    def evidence_fits():
        # Pointers never outrank source evidence, during admission or refit.
        while not fits():
            uncertain = [h for g in admitted for h in g if h.get("time_uncertain")]
            if uncertain:
                hit = uncertain[-1]
                stripped = _without_event_time(hit)
                hit.clear()
                hit.update(stripped)
            elif delivery.get("seed_leads"):
                delivery["seed_leads"].pop()
            elif delivery["summary_leads"]:
                delivery["summary_leads"].pop()
            elif hints := [s for s in delivery["sessions"] if "expand_hint" in s]:
                hints[-1].pop("expand_hint")
            else:
                return False
        return True

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
        calls = row.get("tool_calls", [])
        calls = [] if calls is None else calls
        if not isinstance(calls, list):
            return [], None, None
        expected = [_tool_call_id(c) for c in calls]
        if calls and (row["role"] != "assistant" or "" in expected or len(set(expected)) != len(expected)):
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
                observed.append(str(next_row.get("tool_call_id") or "").strip())
            else:
                return [], None, None
            if sorted(observed, key=str) != sorted(expected, key=str):
                return [], None, None
        return result, neighbor(sid, row["store_id"], 0), neighbor(sid, result[-1]["store_id"], 1)

    def item(row):
        sid = row["store_id"]
        if sid in carried_ids:
            row = {**row, "time_uncertain": "carried"}
        text = row.get("content") or ""
        seed = matched.get(sid)
        hit = dict(kind="message_excerpt", session_id=row["session_id"], store_id=sid,
            role=row["role"], source=row.get("source") or "", timestamp=row.get("timestamp") or 0,
            content=text, snippet=text[:300], content_offset=0, content_returned_chars=len(text),
            content_chars=len(text), content_truncated=False, evidence_span=dict(char_start=0, char_end=len(text)),
            expand_hint=f"lcm_expand(store_id={sid})",
            from_current_session=row["session_id"] == engine.current_session_id,
            score=round(float(seed["_final_score"]), 6) if seed else 0,
            arms=sorted(response["provenance"]["arms_run"][i] for i in seed["ranks"]) if seed else [],
            delivery_origin="ranked_seed" if seed else "session_neighbor", **event_time_fields(row))
        if text:
            hit["exact_ref"] = f"lcm:{sid}:0-{len(text)}"
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
            if row.get("time_uncertain"):
                for key in ("observation_time", "occurrence_time"):
                    hit[key]["time_uncertain"] = "carried"
        return hit

    try:
        check()
        if str(engine._store.db_path) == ":memory:":
            # Borrow under the owners' locks; no transaction or PRAGMA changes.
            live_reads.enter_context(engine._store._write_lock)
            live_reads.enter_context(engine._dag._db_lock)
            io = RetrievalContext(engine._store, engine._dag)
        else:
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
                if row is not None:
                    resolved_summaries.add((sid, hit["node_id"]))
                if row is None and len(delivery["summary_leads"]) < response["limit"]:
                    delivery["summary_leads"].append(dict(node_id=hit["node_id"], session_id=sid,
                        expand_hint=f"lcm_expand(node_id={hit['node_id']}, session_id={json.dumps(sid)})"))
            else:
                row = store.get(int(hit["store_id"])) if hit.get("store_id") is not None else None
            if row is None or row["session_id"] != sid or row["store_id"] in matched:
                continue
            matched[row["store_id"]] = entry
            if sid not in sessions:
                sessions[sid] = dict(session_id=sid, best_rank=rank, seed_ids=[],
                    returned_id_ranges=[], complete=False, omission_reason=None)
                delivery["sessions"].append(sessions[sid])
            sessions[sid]["seed_ids"].append(row["store_id"])
            check()
            eligible += 1
            if eligible >= response["limit"]:
                break
        check()
        carried_ids = store.get_carried_stamp_ids([], session_ids=list(sessions))
        check()
        for sid, meta in sessions.items():
            queue = deque(meta["seed_ids"])
            visited, emitted = set(), set()
            omissions[sid] = []
            while queue:
                anchor = queue.popleft()
                if anchor in visited:
                    continue
                visited.add(anchor)
                batch, before, after = group(sid, anchor)
                if not batch:
                    meta["omission_reason"] = "unreadable_group"
                    meta["expand_hint"] = f"lcm_expand(store_id={anchor})"
                    omissions[sid].append((meta["omission_reason"], meta["expand_hint"]))
                    continue
                visited.update(r["store_id"] for r in batch)
                pending = [item(r) for r in batch if r["store_id"] not in emitted]
                if delta_requested and any(h.get("exact_ref") in seen_refs for h in pending):
                    meta["omission_reason"] = "no_novel_exact_ref"
                    omissions[sid].append((meta["omission_reason"], None))
                elif pending:
                    admitted.append(pending)
                    if evidence_fits():
                        emitted.update(h["store_id"] for h in pending)
                    else:
                        admitted.pop()
                        meta["omission_reason"] = (fit_failure if fit_failure == "size_nonconvergence"
                                                   else "oversized_group" if not emitted else "budget")
                        meta["expand_hint"] = f"lcm_expand(store_id={anchor})"
                        omission = (meta["omission_reason"], meta["expand_hint"])
                        omissions[sid].append(omission)
                        if fit_failure == "budget":
                            skipped.append((meta, pending, omission, (before, after), visited))
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
            known = {i for s in delivery["sessions"] for i in s["seed_ids"]}
            known.update(h["store_id"] for g in admitted for h in g)
            summaries = resolved_summaries | {(s["session_id"], s["node_id"]) for s in delivery["summary_leads"]}
            delivery["seed_leads"] = []
            for entry in ordered[:response["limit"]]:
                hit = entry["hit"]
                sid, seed = hit.get("session_id"), hit.get("store_id")
                if sid in excluded or (allowed is not None and sid not in allowed):
                    continue
                if seed is not None and seed not in known:
                    delivery["seed_leads"].append(dict(session_id=sid, seed_ids=[seed],
                        expand_hint=f"lcm_expand(store_id={seed})"))
                    known.add(seed)
                elif seed is None and hit.get("kind") == "summary" and (sid, hit["node_id"]) not in summaries:
                    delivery["summary_leads"].append(dict(session_id=sid, node_id=hit["node_id"],
                        expand_hint=f"lcm_expand(node_id={hit['node_id']}, session_id={json.dumps(sid)})"))
                    summaries.add((sid, hit["node_id"]))
        for meta in delivery["sessions"]:
            if not meta["complete"] and meta["omission_reason"] is None:
                meta["omission_reason"] = reason
                meta["expand_hint"] = f"lcm_expand(store_id={meta['seed_ids'][0]})"
        delivery["stop_reason"] = reason
    finally:
        if conn is not None:
            conn.close()
        live_reads.close()
    delivery["truncated"] = bool(delivery.get("stop_reason")) or any(
        not s["complete"] for s in delivery["sessions"])
    # Final metadata and escaped JSON count too: refit whole groups and delta.
    while not evidence_fits():
        if not admitted:
            break
        removed = admitted.pop()
        for meta in delivery["sessions"]:
            if meta["session_id"] == removed[0]["session_id"]:
                meta.update(complete=False, omission_reason=fit_failure)
                omissions.setdefault(meta["session_id"], []).append((fit_failure, None))
        delivery["truncated"] = True
    if delivery.get("stop_reason") != "deadline":
        # Only retry already-read budget skips, once, in admission order.
        # Refit removals and nonconverging groups never enter this list.
        for meta, pending, omission, frontier, visited in skipped:
            remaining = list(omissions[meta["session_id"]])
            unwalked = next((x for x in frontier if x is not None and x not in visited), None)
            if unwalked is None:
                remaining.remove(omission)
            else:
                remaining[remaining.index(omission)] = ("budget", f"lcm_expand(store_id={unwalked})")
            previous = dict(meta)
            meta.update(complete=not remaining, omission_reason=remaining[-1][0] if remaining else None)
            meta.pop("expand_hint", None)
            if remaining and remaining[-1][1]:
                meta["expand_hint"] = remaining[-1][1]
            delivery["truncated"] = bool(delivery.get("stop_reason")) or any(
                not s["complete"] for s in delivery["sessions"])
            admitted.append(pending)
            if not evidence_fits():
                admitted.pop()
                meta.clear()
                meta.update(previous)
            else:
                omissions[meta["session_id"]] = remaining
        delivery["truncated"] = bool(delivery.get("stop_reason")) or any(
            not s["complete"] for s in delivery["sessions"])
    while not fits():
        # Metadata is budgeted too, even for K=100 bench seeds and no citable
        # groups. Keep a bounded honest envelope rather than a first-item escape.
        delivery["truncated"] = True
        delivery["stop_reason"] = "budget"
        if delivery.get("seed_leads"):
            delivery["seed_leads"].pop()
        elif delivery["summary_leads"]:
            delivery["summary_leads"].pop()
        elif delivery["sessions"]:
            delivery["sessions"].pop()
        elif response["query"]:
            response["query"] = response["query"][:len(response["query"]) // 2]
        else:
            # An inherited envelope can exceed the cap without any evidence.
            # Return an explicit bounded failure, never silently overrun it.
            delivery.update(sessions=[], summary_leads=[], truncated=True, stop_reason="budget")
            delivery.pop("seed_leads", None)
            response.clear()
            response.update(error="session_base_envelope_over_budget", query="",
                            provenance={"delivery": delivery})
            break
    raw = encode()
    if raw is not None:
        return raw
    # Fixed-point failure must never escape as None or an oversized payload.
    # The last attempt retains size_nonconvergence from the failed refits.
    if len(last_encoded) <= CHAR_BUDGET and count(last_encoded) <= TOKEN_BUDGET:
        return last_encoded
    return json.dumps(dict(error="session_size_nonconvergence", hits=[], total_results=0,
        provenance={"delivery": dict(unit="session", estimator=delivery["estimator"],
                                    stop_reason="size_nonconvergence", truncated=True)}))
