"""Final recall selection and serialization, with host reads supplied by tools.

Ranking is complete before this stage. Strict verification, hydration and the
bounded event-time SQL read remain caller-owned helpers, preserving their read
order, deadline clock and logging namespace. This module owns response policy;
it neither resolves an engine nor reads a clock or opens a database.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from typing import Any

from .reasoning import resolve_occurrence_time
from .retrieval_core import _hit_identity
from .store import _normalize_observed_at

# Optional time fields the response cap strips before it drops a hit. The
# summary span keys appear only with ``read_summary_event_time`` (opt-in).
_EVENT_TIME_KEYS = ("event_time", "event_time_source", "time_uncertain",
                    "event_time_start", "event_time_end")


def deliver_recall(
    ordered: list[dict[str, Any]],
    *,
    query: str,
    limit: int,
    requested_limit: int,
    scope_bias: float,
    include: str,
    detail: str,
    reference_strict: bool,
    delta_requested: bool,
    seen_refs: set[str],
    include_occurrence_time: bool,
    deadline: float,
    fts_arm_index: int | None,
    arm_order: list[str],
    arm_weights: list[float],
    arm_hits: dict[str, list[dict[str, Any]]],
    coverage: dict[str, str],
    served_fts_index: str,
    rerank_status: str,
    rerank_scores: list,
    embedding_query_metrics: list[dict[str, Any]],
    degraded_reasons: list[str],
    timed_out: bool,
    fts_anchor_info: dict[str, Any] | None,
    fts_rerun: bool = False,
    summary_leads: list[dict[str, Any]],
    limit_cap: int,
    snippet_chars: int,
    response_char_cap: int,
    per_session_limit: int,
    expanded_hit_limit: int,
    content_chars: int,
    strict_selector_factory: Callable[..., Any],
    hydrate: Callable[..., dict[tuple, dict[str, Any]]],
    read_event_time: Callable[..., tuple[dict[int, dict[str, Any]], str | None]],
    event_time_fields: Callable[[dict[str, Any] | None], dict[str, str]],
    read_occurrence_source: Callable[[int], tuple[dict, dict | None]],
    rerank_enabled: Callable[[], bool],
    session_hint: Callable[[dict[str, Any]], str | None] | None = None,
    guidance_v2: bool = False,
    session_floor_info: dict[str, Any] | None = None,
    session_delivery: Callable[..., str] | None = None,
    delivery_order: str = "score",
    read_summary_event_time: Callable[..., tuple[dict[int, dict[str, str]], str | None]] | None = None,
    adjacent_reply: list[dict[str, int]] | None = None,
    adjacent_reply_unavailable: str | None = None,
    place_later: Callable[..., tuple[list, dict, str | None]] | None = None,
    snippet_window_hits: int = 0,
    read_snippet_windows: Callable[..., tuple[dict, str | None]] | None = None,
    extra_entries: int = 0,
    read_extra_entries: Callable[..., tuple[dict, str | None]] | None = None,
    _fit_snippets: bool = True,
) -> str:
    """Deliver ranked entries without changing retrieval or host-read policy."""
    if session_delivery is not None:
        # Reuse the existing empty envelope/metrics; only the opt-in branch pays
        # for it. The hits branch below retains its exact read/serialization order.
        options = dict(locals())
        options.pop("ordered")
        options.pop("session_delivery")
        options["_fit_snippets"] = False
        options["delivery_order"] = "score"
        options["read_summary_event_time"] = None
        response = json.loads(_deliver_hits([], **options))
        response["provenance"].pop("answer_ready", None)
        return session_delivery(ordered, response)
    # -- Response shaping (char-capped). The default snippets path retains the
    # historical order and serialized response exactly. answer_ready applies
    # stable post-rank diversity before bounded exact-ref hydration.
    diversity_dropped = 0
    if detail == "answer_ready":
        # Take only what the response can hold. Delta used to select the whole
        # 25-candidate cap up front and filter afterwards, which walked the
        # ranking to exhaustion while already-seen entries held session quota --
        # the refill then had nothing left to resume into. Selecting a wave at a
        # time lets a released slot be reused by the next wave.
        selection_limit = limit
        expanded_limit = (
            limit_cap
            if delta_requested
            else expanded_hit_limit
        )
        if reference_strict:
            strict_selector = strict_selector_factory(
                ordered,
                per_session_limit=per_session_limit,
                expanded_limit=expanded_limit,
            )
            selected_entries = strict_selector.take(selection_limit)
            strict_rows = strict_selector.rows
        else:
            selected_entries, diversity_dropped = _diverse_entries(
                ordered,
                limit=selection_limit,
                per_session_limit=per_session_limit,
            )
            strict_rows = None
        answer_ready_content = hydrate(
            selected_entries,
            query=query,
            expanded_limit=expanded_limit,
            rows_by_id=strict_rows,
            fts_arm_index=fts_arm_index,
        )
        if delta_requested:
            def _novel(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
                """Keep the entries carrying a reference the caller lacks.

                A discarded entry hands its session slot back: it is not part of
                the response, so it must not count against the density budget
                that decides which novel rows can still be delivered.
                """
                kept: list[dict[str, Any]] = []
                for entry in entries:
                    exact_ref = (
                        None
                        if entry["hit"].get("kind") == "summary"
                        else _exact_ref(
                            entry["hit"],
                            answer_ready_content.get(_hit_identity(entry["hit"])),
                        )
                    )
                    if exact_ref is not None and exact_ref not in seen_refs:
                        kept.append(entry)
                    elif reference_strict:
                        strict_selector.release(entry)
                return kept

            selected_entries = _novel(selected_entries)
            # Delta shaping discards entries the caller has already seen, so it
            # too must be able to draw on the ranked tail -- otherwise the mode
            # silently returns short while valid candidates remain, which is the
            # very underfill the resumable walk exists to prevent.
            while (
                reference_strict
                and len(selected_entries) < limit
                and not strict_selector.exhausted()
            ):
                more = strict_selector.take(limit)
                if not more:
                    break
                answer_ready_content.update(
                    hydrate(
                        more,
                        query=query,
                        expanded_limit=len(more),
                        rows_by_id=strict_selector.rows,
                        fts_arm_index=fts_arm_index,
                    )
                )
                selected_entries.extend(_novel(more))
            selected_entries = selected_entries[:limit]
    else:
        selected_entries = ordered
        diversity_dropped = 0
        answer_ready_content = {}
    later_statements, later_statements_unavailable = None, None
    if place_later is not None:
        later_originals = {_hit_identity(e["hit"]): e for e in selected_entries} if reference_strict else {}
        selected_entries, later_statements, later_statements_unavailable = place_later(
            selected_entries, limit=limit, seen_refs=seen_refs, reference_strict=reference_strict, delta_requested=delta_requested,
            hydrate=hydrate if detail == "answer_ready" else None,
            rows_by_id=strict_rows if detail == "answer_ready" else None,
            answer_ready_content=answer_ready_content,
        )
    snippet_window_content = {}
    snippet_window_skipped = None
    if detail == "snippets" and snippet_window_hits > 0:
        window_entries = [
            entry for entry in selected_entries[:limit]
            if entry["hit"].get("kind") == "message_excerpt"
            and entry["hit"].get("store_id") is not None
            and entry["hit"].get("adjacent_to") is None
        ][:snippet_window_hits]
        if window_entries:
            if read_snippet_windows is None:
                snippet_window_skipped = "read_error"
            else:
                snippet_window_content, snippet_window_skipped = read_snippet_windows(
                    window_entries, query=query, expanded_limit=len(window_entries),
                    fts_arm_index=fts_arm_index, deadline=deadline,
                )
    # Event time for message hits that answer_ready hydration did not already
    # cover (the snippets default, and answer_ready hits past the expansion
    # limit): one batched read by store_id of only the deliverable entries,
    # reusing rows reference-strict selection has already read. It runs after
    # selection and never feeds ranking, selection or the hit text (#995).
    event_time_rows: dict[int, dict[str, Any]] = (
        dict(strict_selector.rows) if reference_strict else {}
    )
    event_time_ids = [
        int(entry["hit"]["store_id"])
        for entry in selected_entries[:limit]
        if entry["hit"].get("kind") != "summary"
        and entry["hit"].get("store_id") is not None
    ]
    # A skipped or failed read never blocks past the deadline and never looks
    # like "no event time": a delivered hit it covered sets the response-level
    # ``event_time_unavailable`` marker instead.
    event_time_unavailable_reason: str | None = None
    event_time_unread_ids: set[int] = set()
    if event_time_ids:
        fetched_rows, event_time_unavailable_reason = read_event_time(
            event_time_ids, deadline=deadline
        )
        event_time_rows.update(fetched_rows)
        if event_time_unavailable_reason is not None:
            event_time_rows.clear()
            event_time_unread_ids = set(event_time_ids)
    # Opt-in summary span: one batched read of the deliverable summary nodes.
    # A failed read omits the span from every summary hit (atomic).
    summary_event_time: dict[int, dict[str, str]] = {}
    summary_event_time_unavailable: str | None = None
    summary_node_ids = list(dict.fromkeys(
        int(entry["hit"]["node_id"])
        for entry in selected_entries[:limit]
        if entry["hit"].get("kind") == "summary" and entry["hit"].get("node_id") is not None
    )) if read_summary_event_time is not None else []
    if summary_node_ids:
        summary_event_time, summary_event_time_unavailable = read_summary_event_time(
            summary_node_ids, deadline=deadline
        )
        if summary_event_time_unavailable is not None:
            summary_event_time = {}
    event_time_unavailable = False
    hits_out: list[dict[str, Any]] = []
    # Characters each delivered hit's optional event-time fields add on top of
    # its measured size (parallel to ``hits_out``).
    event_time_chars: list[int] = []
    response_chars = 0
    response_cap_truncated = False
    unreferenced_omitted = 0
    for consumed, entry in enumerate(selected_entries, 1):
        hit = entry["hit"]
        arms = sorted({arm_order[index] for index in entry["ranks"].keys()})
        item: dict[str, Any] = {
            "kind": hit.get("kind"),
            "session_id": hit.get("session_id"),
            "timestamp": hit.get("timestamp") or 0,
            "snippet": (hit.get("snippet") or "")[:snippet_chars],
            "score": round(float(entry["_final_score"]), 6),
            "expand_hint": hit.get("expand_hint"),
            "from_current_session": bool(hit.get("from_current_session")),
            "arms": arms,
        }
        if hit.get("adjacent_to") is not None:
            item.update(adjacent_to=hit["adjacent_to"], arms=[*arms, "adjacent"])
        if hit.get("later_statement_of") is not None:
            item["arms"] = [*arms, "later_statement"]
        for key in ("later_statement_of", "has_later_statements"):
            if key in hit:
                item[key] = hit[key]
        window = snippet_window_content.get(_hit_identity(hit))
        if window is not None:
            item["snippet"] = window["content"]
            item["snippet_window"] = {
                key: window[key] for key in (
                    "content_offset", "content_returned_chars", "content_chars", "content_truncated",
                )
            }
        if hit.get("lineage"):  # #1931: a delegated subagent's hit names its parent session
            item.update(parent_session_id=hit.get("parent_session_id"), lineage=hit["lineage"])
        if hit.get("kind") == "summary":
            item["node_id"] = hit.get("node_id")
            if hit.get("store_id") is not None:
                item["store_id"] = hit.get("store_id")
            if hit.get("node_id") is not None:
                item.update(summary_event_time.get(int(hit["node_id"]), {}))
        else:
            item["store_id"] = hit.get("store_id")
            if hit.get("duplicate_store_ids"):
                item["duplicate_store_ids"] = hit["duplicate_store_ids"]
            if hit.get("chunk_span"):
                item["chunk_span"] = hit["chunk_span"]
            if _hit_identity(hit) not in answer_ready_content and hit.get("store_id") is not None:
                item.update(event_time_fields(event_time_rows.get(int(hit.get("store_id") or 0))))
        if detail == "answer_ready":
            if session_hint and (hint := session_hint(hit)):
                item["session_hint"] = hint
            item["role"] = hit.get("role")
            item["source"] = hit.get("source") or (
                "summary" if hit.get("kind") == "summary" else ""
            )
            hydrated = answer_ready_content.get(_hit_identity(hit))
            if hydrated is not None:
                item.update({k: v for k, v in hydrated.items() if k not in _EVENT_TIME_KEYS})
                # Off: a summary sharing a message's store_id keeps taking that
                # message's event time (pinned). On: a summary never does.
                if read_summary_event_time is None or hit.get("kind") != "summary":
                    item.update(event_time_fields(event_time_rows.get(int(hit.get("store_id") or 0))))
            if delta_requested:
                exact_ref = _exact_ref(hit, hydrated)
                if exact_ref is not None:
                    item["exact_ref"] = exact_ref
            if include_occurrence_time and hit.get("kind") != "summary":
                session_dates, source_row = read_occurrence_source(int(hit.get("store_id") or 0))
                source_row = source_row or {}
                source_row = {**source_row, **event_time_rows.get(int(hit.get("store_id") or 0), {})}
                source_observed_at = source_row.get("observed_at")
                session_date = session_dates.get(str(hit.get("session_id")))
                if session_date is None and source_observed_at is not None:
                    try:
                        session_date = datetime.fromtimestamp(
                            float(source_observed_at), tz=timezone.utc
                        ).date().isoformat()
                    except (TypeError, ValueError, OverflowError, OSError):
                        session_date = None
                occurrence = resolve_occurrence_time(
                    (hydrated or {}).get("content") or hit.get("snippet") or "",
                    observed_at=source_observed_at or 0,
                    session_date=session_date,
                )
                occurrence["stored_at"] = source_row.get("ingested_at") or source_row.get("timestamp")
                item["occurrence_time"] = occurrence
                item["observation_time"] = {
                    "observed_at": occurrence.get("observed_at") or None,
                    "ingested_at": source_row.get("ingested_at") or source_row.get("timestamp"),
                    "source": (
                        "benchmark_session_date"
                        if str(hit.get("session_id")) in session_dates
                        else "host_message_timestamp"
                        if source_observed_at is not None
                        else "ingest_fallback"
                    ),
                }
                if source_row.get("time_uncertain") == "carried":
                    item["time_uncertain"] = "carried"
                    occurrence["time_uncertain"] = "carried"
                    item["observation_time"]["time_uncertain"] = "carried"
                if int(hit.get("store_id") or 0) in event_time_unread_ids:
                    for key in (*_EVENT_TIME_KEYS, "observation_time", "occurrence_time"):
                        item.pop(key, None)
        if reference_strict:
            # Selection already proved this candidate against its row, so there
            # is nothing left to re-check here -- only the proven span to
            # PUBLISH, so a consumer never has to guess an offset. (__init__.py's
            # _answer_ready_baseline substitutes 0 when the field is absent.)
            if hydrated is not None:
                span = (
                    int(hydrated["content_offset"]),
                    int(hydrated["content_returned_chars"]),
                )
            else:
                span = entry.get("_strict_span")
            if span is None:
                unreferenced_omitted += 1
                continue
            item["content_offset"], item["content_returned_chars"] = span
        # The cap sizes a hit without its optional event-time fields, so they
        # never evict a hit that fits without them (#995).
        item_chars = len(
            json.dumps(_without_event_time(item), ensure_ascii=window is not None)
        )
        if hits_out and response_chars + item_chars > response_char_cap:
            response_cap_truncated = True
            break
        response_chars += item_chars
        event_time_chars.append(
            len(json.dumps(item, ensure_ascii=window is not None)) - item_chars
        )
        if reference_strict:
            strict_selector.deliver(later_originals.get(_hit_identity(hit), entry) if place_later is not None else entry)
        if (
            hit.get("kind") != "summary"
            and hit.get("store_id") is not None
            and int(hit["store_id"]) in event_time_unread_ids
        ):
            event_time_unavailable = True
        hits_out.append(item)
        if len(hits_out) >= limit:
            break
    event_time_unavailable, event_time_unavailable_reason = _strip_hit_event_time(
        response_chars=response_chars,
        event_time_chars=event_time_chars,
        hits_out=hits_out,
        response_char_cap=response_char_cap,
        event_time_unavailable=event_time_unavailable,
        event_time_unavailable_reason=event_time_unavailable_reason,
    )

    degraded = bool(degraded_reasons)
    response: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "scope_bias": scope_bias,
        "include": include,
        "total_results": len(hits_out),
        "hits": hits_out,
        "provenance": {
            "arms_run": arm_order,
            "arm_weights": {name: arm_weights[i] for i, name in enumerate(arm_order)},
            "coverage": coverage,
            "fts_index": served_fts_index,
            "rerank": rerank_status,
            "ordering": (
                "rrf-fusion -> scope/recency prior -> rerank reorder (top window); "
                "the reported score is the scope/recency-adjusted RRF score, and "
                "rerank (when applied) only permutes the top window without "
                "replacing that score"
            ),
        },
        "metrics": {
            "embedding_query_calls": len(embedding_query_metrics),
            "embedding_query_tokens": sum(
                int(item["usage_tokens"] or 0) for item in embedding_query_metrics
            ),
            "embedding_query_tokens_complete": all(
                item["usage_tokens"] is not None for item in embedding_query_metrics
            ),
            "embedding_queries": embedding_query_metrics,
        },
        "degraded": degraded,
    }
    if session_floor_info is not None:
        response["provenance"]["session_floor"] = {**session_floor_info, "delivered": _sessions(hits_out)}
    if later_statements is not None:
        response["provenance"]["later_statements"] = later_statements
        _refresh_later_statements(response)
    if later_statements_unavailable is not None:
        response["provenance"]["later_statements_unavailable"] = later_statements_unavailable
    if adjacent_reply and detail != "answer_ready":
        # Snippets hits are final here; answer_ready publishes after fitting.
        receipts = _delivered_adjacent_receipts(hits_out, adjacent_reply)
        if receipts:
            response["provenance"]["adjacent_reply"] = receipts
    if adjacent_reply_unavailable is not None:
        response["provenance"]["adjacent_reply_unavailable"] = adjacent_reply_unavailable
    if rerank_enabled() and rerank_scores:
        response["provenance"]["rerank_scores"] = rerank_scores
    if fts_rerun:
        response["provenance"]["fts_rerun"] = True
    if fts_anchor_info is not None and len(arm_order) > 1:
        # The anchor only places the hit in the ranked window; answer_ready
        # selection (citation and per-session rules) and the response cap still
        # decide delivery, so report whether the anchored hit actually shipped.
        # ``delivered`` is null when the anchor did not fire (no anchored hit).
        fts_anchor_info["delivered"] = (
            _fts_anchor_delivered(response["hits"], arm_hits)
            if fts_anchor_info["fired"]
            else None
        )
        response["provenance"]["fts_anchor"] = fts_anchor_info
    if degraded:
        response["degraded_reason"] = "; ".join(dict.fromkeys(degraded_reasons))
    if timed_out:
        response["timeout"] = True
    if event_time_unavailable:
        response["event_time_unavailable"] = True
        response["event_time_unavailable_reason"] = event_time_unavailable_reason
    if requested_limit > limit_cap:
        response["limit_clamped_from"] = requested_limit
    if summary_event_time_unavailable is not None:
        response["provenance"]["summary_event_time_unavailable"] = summary_event_time_unavailable
    if snippet_window_skipped is not None:
        response["snippet_window_skipped"] = snippet_window_skipped
    if detail == "snippets" and _fit_snippets and response_cap_truncated:
        response["response_cap_truncated"] = True
    if detail == "snippets" and _fit_snippets and (
        (snippet_window_hits > 0 and snippet_window_skipped is None)
        or len(json.dumps(response)) > response_char_cap
    ):
        window_fit = snippet_window_hits > 0 and snippet_window_skipped is None
        # Every snippets path budgets the serialized envelope. Refresh receipts
        # to a fixed point; windows retain main's time-marker budget and fit order.
        def encoded_size(payload=response) -> int:
            survivors, provenance = payload["hits"], payload["provenance"]
            if "session_floor" in provenance:
                provenance["session_floor"]["delivered"] = _sessions(survivors)
            _refresh_later_statements(payload)
            if adjacent_reply:
                provenance.pop("adjacent_reply", None)
                receipts = _delivered_adjacent_receipts(survivors, adjacent_reply)
                if receipts:
                    provenance["adjacent_reply"] = receipts
            if (anchor := provenance.get("fts_anchor")) is not None:
                anchor["delivered"] = (
                    _fts_anchor_delivered(survivors, arm_hits) if anchor["fired"] else None
                )
            if window_fit and delivery_order == "time":
                return len(json.dumps({**payload, "provenance": {
                    **provenance, "delivery_order": "time",
                }}))
            return len(json.dumps(payload))

        if len(query) > 4_096:
            # Windows already fitted whole envelopes on main. Keep their query
            # unless the no-hit envelope, including fit markers, cannot fit.
            query_budget = len(json.dumps(response))
            if window_fit:
                empty = deepcopy(response)
                empty.update(hits=[], total_results=0)
                if hits_out:
                    empty["response_cap_truncated"] = True
                query_budget = encoded_size(empty)
            if query_budget > response_char_cap:
                response.update(query=query[:4_096], query_truncated=True)
        deferred_reason = None
        if encoded_size() > response_char_cap:
            if not window_fit:
                _, deferred_reason = _fit_event_time(
                    response, response_char_cap=response_char_cap, ensure_ascii=True,
                )
        while encoded_size() > response_char_cap and hits_out:
            hits_out.pop()
            response["total_results"] = len(hits_out)
            response["response_cap_truncated"] = True
        if deferred_reason is not None:
            response.update(event_time_unavailable=True, event_time_unavailable_reason=deferred_reason)
            if encoded_size() > response_char_cap:
                response.pop("event_time_unavailable")
                response.pop("event_time_unavailable_reason")
    if detail == "answer_ready":
        expansion = {
            "expanded_hit_count": sum("content" in hit for hit in hits_out),
            "expanded_hit_limit": expanded_hit_limit,
            "per_session_limit": per_session_limit,
            "diversity_dropped_count": diversity_dropped,
            "per_hit_char_cap": content_chars,
            "snippet_char_cap": snippet_chars,
            "response_char_cap": response_char_cap,
            "response_policy": (
                "rank-preserving session diversity, then exact-ref hydration; "
                "whole hits only when enforcing the response cap"
            ),
            "hydration_policy": "bounded exact reads only; no additional retrieval search",
            "response_truncated": response_cap_truncated,
        }
        if reference_strict:
            expansion["reference_strict"] = True
            expansion["diversity_dropped_count"] = strict_selector.diversity_dropped
            expansion["unreferenced_dropped_count"] = (
                strict_selector.unreferenced_dropped
            )
            expansion["unreferenced_omitted_count"] = unreferenced_omitted
            if session_hint:
                for lead in summary_leads:
                    if hint := session_hint(lead):
                        lead["session_hint"] = hint
            expansion["summary_leads"] = summary_leads
            expansion["reference_policy"] = (
                "every delivered hit publishes the (store_id, content_offset, "
                "content_returned_chars) span its text occupies in the current "
                "row; a candidate that fails that check is replaced by the "
                "next-ranked citable one. Summary nodes are never delivered as "
                "evidence -- their relevance reaches the ranking through the "
                "source messages beneath them, and the nodes themselves come "
                "back here as non-evidence drill-down leads"
            )
        if guidance_v2:
            expansion.update(sessions_delivered=_sessions(hits_out),
                             sessions_in_pool=_sessions([e["hit"] for e in ordered] + hits_out))
        response["detail"] = detail
        response["provenance"]["detail"] = detail
        response["provenance"]["answer_ready"] = expansion
        if delta_requested:
            novel_refs = [hit["exact_ref"] for hit in hits_out if hit.get("exact_ref")]
            response["delta"] = {
                "protocol": "exact-ref-delta-v1",
                "seen_ref_count": len(seen_refs),
                "novel_ref_count": len(novel_refs),
                "novel_refs": novel_refs,
                "progress": bool(novel_refs),
                "termination_reason": None if novel_refs else "no_novel_exact_ref",
            }
        if include_occurrence_time:
            response["provenance"]["occurrence_time"] = {
                "policy_version": "occurrence-time-v1",
                "anchor_source": "engine session metadata when available",
                "observation_is_not_occurrence": True,
            }

        encoded = _fit_answer_ready_response(
            fts_anchor_info=fts_anchor_info, response=response,
            response_char_cap=response_char_cap, expansion=expansion,
            delta_requested=delta_requested, arm_hits=arm_hits,
            adjacent_reply=adjacent_reply,
            ranked_hits=[e["hit"] for e in ordered] if guidance_v2 else None,
        )
        if delivery_order != "time":
            return encoded
    elif delivery_order != "time" and extra_entries <= 0:
        return json.dumps(response)

    # Both size-cap passes and event-time stripping run in score order first.
    # Sort only survivors, using the existing read even if their date was
    # stripped for space. Python's stable sort preserves score order on ties.
    def event_time_key(hit: dict[str, Any]) -> tuple[bool, float]:
        row = event_time_rows.get(int(hit.get("store_id") or 0), {})
        observed_at = (
            _normalize_observed_at(row.get("observed_at"))
            if hit.get("kind") != "summary" else None
        )
        return observed_at is None, observed_at if observed_at is not None else 0.0

    ensure_ascii = detail != "answer_ready"
    score_order = list(hits_out) if detail == "snippets" else None
    unmarked = json.dumps(response, ensure_ascii=ensure_ascii)
    if delivery_order == "time":
        hits_out.sort(key=event_time_key)
    if detail == "answer_ready":
        unmarked = json.dumps(response, ensure_ascii=ensure_ascii)
    # Reordering keeps the encoded length; the marker is extra, so it is added
    # only when it still fits the response cap (never by dropping a hit).
    if delivery_order == "time":
        response["provenance"]["delivery_order"] = "time"
    marked = json.dumps(response, ensure_ascii=ensure_ascii)
    if len(marked) > response_char_cap:
        response["provenance"].pop("delivery_order", None)
        marked = unmarked
        if detail == "answer_ready":
            return unmarked
        hits_out[:] = score_order
    if (detail == "snippets" and extra_entries > 0 and len(hits_out) == limit
            and not response_cap_truncated and not response.get("response_cap_truncated")):
        seen = {hit["store_id"] for hit in hits_out if hit.get("store_id") is not None}
        seen.update(sid for hit in hits_out for sid in hit.get("duplicate_store_ids", []))
        candidates = []
        for entry in selected_entries[consumed:consumed + min(extra_entries, limit // 2)]:
            sid = entry["hit"].get("store_id")
            if sid is not None and sid in seen:
                continue
            if sid is not None:
                seen.add(sid)
            candidates.append(entry)
        if candidates:
            try:
                content, skipped = read_extra_entries(
                    candidates, query=query, expanded_limit=len(candidates),
                    fts_arm_index=fts_arm_index, deadline=deadline, char_cap=300,
                ) if read_extra_entries is not None else ({}, "read_error")
                extras = []
                for entry in candidates if skipped is None else []:
                    hit = entry["hit"]
                    window = content[_hit_identity(hit)]
                    extras.append({
                        "kind": hit.get("kind"), "store_id": hit.get("store_id"),
                        "session_id": hit.get("session_id"), "event_time": window.get("event_time"),
                        "window": window["content"], "extra": True,
                    })
            except Exception as exc:  # noqa: BLE001 - optional extras must never fail recall
                logging.getLogger(__name__).warning("lcm_recall extra entries read failed: %s", type(exc).__name__)
                skipped = "read_error"
            if skipped is not None:
                response["extra_entries_skipped"] = skipped
                if len(json.dumps(response)) > response_char_cap:
                    response.pop("extra_entries_skipped")
            else:
                hits_out.extend(extras)
                while len(hits_out) > limit:
                    if len(json.dumps(response)) <= response_char_cap:
                        break
                    hits_out.pop()
            # Extras are not window hits: session_floor.delivered keeps the full-hit count.
            return json.dumps(response)
    return marked


def _fit_answer_ready_response(
    *,
    fts_anchor_info: dict[str, Any] | None,
    response: dict[str, Any],
    response_char_cap: int,
    expansion: dict[str, Any],
    delta_requested: bool,
    arm_hits: dict[str, list[dict[str, Any]]],
    adjacent_reply: list[dict[str, int]] | None = None,
    ranked_hits: list[dict[str, Any]] | None = None,
) -> str:
    anchor_fired = bool(fts_anchor_info and fts_anchor_info.get("fired"))
    if anchor_fired:
        # Size the cap check with the widest value (``false``) so the final
        # recompute below can never grow the encoded response past the cap.
        fts_anchor_info["delivered"] = False
    _refresh_later_statements(response)
    encoded = json.dumps(response, ensure_ascii=False)
    event_time_marker_deferred: str | None = None
    if len(encoded) > response_char_cap:
        encoded, event_time_marker_deferred = _fit_event_time(
            response, response_char_cap=response_char_cap
        )
    if len(encoded) > response_char_cap:
        original_query = response["query"]
        response["query"] = original_query[:4_096]
        expansion["query_truncated"] = len(response["query"]) < len(original_query)
        encoded = json.dumps(response, ensure_ascii=False)
    while len(encoded) > response_char_cap and (
        response["hits"] or expansion.get("summary_leads")
    ):
        if response["hits"]:
            response["hits"].pop()
        else:
            expansion["summary_leads"].pop()
        response["total_results"] = len(response["hits"])
        expansion["response_truncated"] = True
        _refresh_later_statements(response)
        if "session_floor" in response["provenance"]:
            response["provenance"]["session_floor"]["delivered"] = _sessions(response["hits"])
        expansion["expanded_hit_count"] = sum(
            "content" in hit for hit in response["hits"]
        )
        if "sessions_delivered" in expansion:
            expansion["sessions_delivered"] = _sessions(response["hits"])
            # The pool is the ranked sessions plus delivered children that survived fitting.
            expansion["sessions_in_pool"] = _sessions((ranked_hits or []) + response["hits"])
        encoded = json.dumps(response, ensure_ascii=False)
    if event_time_marker_deferred is not None:
        # Put the marker back with its original reason, only if it fits.
        marked = {
            **response,
            "event_time_unavailable": True,
            "event_time_unavailable_reason": event_time_marker_deferred,
        }
        marked_encoded = json.dumps(marked, ensure_ascii=False)
        if len(marked_encoded) <= response_char_cap:
            response.update(
                event_time_unavailable=True,
                event_time_unavailable_reason=event_time_marker_deferred,
            )
            encoded = marked_encoded
    if delta_requested:
        novel_refs = [
            hit["exact_ref"]
            for hit in response["hits"]
            if hit.get("exact_ref")
        ]
        response["delta"].update(
            {
                "novel_ref_count": len(novel_refs),
                "novel_refs": novel_refs,
                "progress": bool(novel_refs),
                "termination_reason": (
                    None if novel_refs else "no_novel_exact_ref"
                ),
            }
        )
        encoded = json.dumps(response, ensure_ascii=False)
    if anchor_fired:
        # The cap loop may have popped the anchored hit: report delivery
        # from the hits actually returned.
        fts_anchor_info["delivered"] = _fts_anchor_delivered(
            response["hits"], arm_hits
        )
        encoded = json.dumps(response, ensure_ascii=False)
    receipts = _delivered_adjacent_receipts(response["hits"], adjacent_reply or [])
    if receipts:
        # Receipts name only replies the final hits carry, and are published
        # only when they fit: a receipt never evicts a hit.
        marked = {
            **response,
            "provenance": {**response["provenance"], "adjacent_reply": receipts},
        }
        marked_encoded = json.dumps(marked, ensure_ascii=False)
        if len(marked_encoded) <= response_char_cap:
            response["provenance"]["adjacent_reply"] = receipts
            encoded = marked_encoded
    return encoded


def _refresh_later_statements(response):
    info = response["provenance"].get("later_statements")
    if info is None:
        return
    hits = response["hits"]
    by_id = {h.get("store_id"): h for h in hits if h.get("kind") != "summary"}
    for hit in hits:
        links = [sid for sid in hit.get("has_later_statements", []) if sid in by_id
                 and by_id[sid].get("later_statement_of") == hit.get("store_id")]
        if "has_later_statements" in hit:
            if links:
                hit["has_later_statements"] = links
            else:
                hit.pop("has_later_statements")
    info["delivered"] = sum("later_statement_of" in h for h in hits)


def _delivered_adjacent_receipts(
    hits: list[dict[str, Any]], pairs: list[dict[str, int]]
) -> list[dict[str, int]]:
    """Keep receipts whose reply ``hits`` deliver, also as a collapsed duplicate."""
    delivered: set[Any] = set()
    for hit in hits:
        if hit.get("kind") != "summary":
            delivered.add(hit.get("store_id"))
            delivered.update(hit.get("duplicate_store_ids", []))
    return [pair for pair in pairs if pair["store_id"] in delivered]


def _strip_hit_event_time(
    *,
    response_chars: int,
    event_time_chars: list[int],
    hits_out: list[dict[str, Any]],
    response_char_cap: int,
    event_time_unavailable: bool,
    event_time_unavailable_reason: str | None,
) -> tuple[bool, str | None]:
    # Over the cap only with the fields: strip them, lowest rank first, and
    # say so, instead of dropping any hit.
    event_time_total = response_chars + sum(event_time_chars)
    for index in reversed(range(len(hits_out))):
        if event_time_total <= response_char_cap:
            break
        if event_time_chars[index]:
            stripped = _without_event_time(hits_out[index])
            hits_out[index].clear()
            hits_out[index].update(stripped)
            event_time_total -= event_time_chars[index]
            # A ``deadline`` or ``read_error`` marker keeps its reason.
            if not event_time_unavailable:
                event_time_unavailable = True
                event_time_unavailable_reason = "response_cap"
    return event_time_unavailable, event_time_unavailable_reason


def _diverse_entries(
    ordered: list[dict[str, Any]],
    *,
    limit: int,
    per_session_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select a stable rank-preserving result set with bounded session density."""
    selected: list[dict[str, Any]] = []
    session_counts: dict[str, int] = {}
    dropped = 0
    for entry in ordered:
        hit = entry["hit"]
        raw_session_id = hit.get("session_id")
        # Missing session identities must not collapse into one synthetic session.
        # Exact refs remain independently eligible in their existing rank order.
        session_key = (
            str(raw_session_id)
            if raw_session_id not in {None, ""}
            else f"missing:{_hit_identity(hit)!r}"
        )
        if session_counts.get(session_key, 0) >= per_session_limit:
            dropped += 1
            continue
        session_counts[session_key] = session_counts.get(session_key, 0) + 1
        selected.append(entry)
        if len(selected) >= limit:
            break
    return selected, dropped


def _without_event_time(item: dict[str, Any]) -> dict[str, Any]:
    """Return ``item`` without its optional event-time fields.

    The response cap sizes hits by this view, so the optional fields can never
    evict a hit that fits without them (#995).
    """
    return {
        key: value
        for key, value in item.items()
        if key not in _EVENT_TIME_KEYS and not (
            item.get("time_uncertain") and key in ("observation_time", "occurrence_time"))
    }


def _fit_event_time(
    response: dict[str, Any], *, response_char_cap: int, ensure_ascii: bool = False,
) -> tuple[str, str | None]:
    """Strip hits' event-time fields, lowest rank first, until ``response`` fits.

    Runs before the cap touches anything else, so the optional fields never
    cost a hit, nor a query cut the response would not otherwise need. A
    stripped hit sets the ``event_time_unavailable`` marker with the
    ``response_cap`` reason, unless a ``deadline`` or ``read_error`` marker is
    already set: that reason is kept, never overwritten. If the response is
    still over the cap with every field stripped, the marker is taken out too,
    whatever its reason, so it is never what evicts a hit. Returns ``(encoded,
    deferred_reason)``: ``deferred_reason`` is the reason of a marker taken out,
    which the caller puts back if it fits once the cap has done the rest, and
    ``None`` otherwise (#995).
    """
    encoded = json.dumps(response, ensure_ascii=ensure_ascii)
    for hit in reversed(response["hits"]):
        if len(encoded) <= response_char_cap:
            return encoded, None
        if not any(key in hit for key in _EVENT_TIME_KEYS):
            continue
        stripped = _without_event_time(hit)
        hit.clear()
        hit.update(stripped)
        if not response.get("event_time_unavailable"):
            response["event_time_unavailable"] = True
            response["event_time_unavailable_reason"] = "response_cap"
        encoded = json.dumps(response, ensure_ascii=ensure_ascii)
    if len(encoded) > response_char_cap and response.get(
        "event_time_unavailable"
    ):
        del response["event_time_unavailable"]
        deferred_reason = response.pop("event_time_unavailable_reason")
        return json.dumps(response, ensure_ascii=ensure_ascii), deferred_reason
    return encoded, None


def _exact_ref(hit: dict[str, Any], hydrated: dict[str, Any] | None) -> str | None:
    """Return the exact identity for an opt-in delta item."""
    if hit.get("kind") == "summary":
        node_id = hit.get("node_id")
        return f"lcm-summary:{node_id}" if node_id is not None else None
    store_id = hit.get("store_id")
    if store_id is None or hydrated is None or "content" not in hydrated:
        return None
    start = int(hydrated.get("content_offset") or 0)
    end = start + len(str(hydrated.get("content") or ""))
    return f"lcm:{int(store_id)}:{start}-{end}"


def _fts_anchor_delivered(
    hits: list[dict[str, Any]], arm_hits: dict[str, list[dict[str, Any]]]
) -> bool:
    """Whether the FTS arm's anchored hit is among the returned ``hits`` (#950)."""
    anchor = _hit_identity(arm_hits["fts"][0])
    return any(
        _hit_identity(hit) == anchor
        or (anchor[0] == "message" and anchor[1] in hit.get("duplicate_store_ids", []))
        for hit in hits
    )


# The opt-in empty-envelope call must not re-enter observer wrappers installed
# on deliver_recall (FMP observes the real tools.py caller's effective K caps).
_deliver_hits = deliver_recall


def _sessions(hits: list[dict[str, Any]]) -> int:
    return len({h.get("session_id") for h in hits} - {None, ""})
