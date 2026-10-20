"""Final recall selection and serialization, with host reads supplied by tools.

Ranking is complete before this stage. Strict verification, hydration and the
bounded event-time SQL read remain caller-owned helpers, preserving their read
order, deadline clock and logging namespace. This module owns response policy;
it neither resolves an engine nor reads a clock or opens a database.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from typing import Any

from .reasoning import resolve_occurrence_time
from .retrieval_core import _hit_identity

_EVENT_TIME_KEYS = ("event_time", "event_time_source")


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
) -> str:
    """Deliver ranked entries without changing retrieval or host-read policy."""
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
        and _hit_identity(entry["hit"]) not in answer_ready_content
        and int(entry["hit"]["store_id"]) not in event_time_rows
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
            event_time_unread_ids = set(event_time_ids)
    event_time_unavailable = False
    hits_out: list[dict[str, Any]] = []
    # Characters each delivered hit's optional event-time fields add on top of
    # its measured size (parallel to ``hits_out``).
    event_time_chars: list[int] = []
    response_chars = 0
    response_cap_truncated = False
    unreferenced_omitted = 0
    for entry in selected_entries:
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
        if hit.get("kind") == "summary":
            item["node_id"] = hit.get("node_id")
            if hit.get("store_id") is not None:
                item["store_id"] = hit.get("store_id")
        else:
            item["store_id"] = hit.get("store_id")
            if hit.get("chunk_span"):
                item["chunk_span"] = hit["chunk_span"]
            if _hit_identity(hit) not in answer_ready_content and hit.get("store_id") is not None:
                item.update(event_time_fields(event_time_rows.get(int(hit["store_id"]))))
        if detail == "answer_ready":
            item["role"] = hit.get("role")
            item["source"] = hit.get("source") or (
                "summary" if hit.get("kind") == "summary" else ""
            )
            hydrated = answer_ready_content.get(_hit_identity(hit))
            if hydrated is not None:
                item.update(hydrated)
            if delta_requested:
                exact_ref = _exact_ref(hit, hydrated)
                if exact_ref is not None:
                    item["exact_ref"] = exact_ref
            if include_occurrence_time and hit.get("kind") != "summary":
                session_dates, source_row = read_occurrence_source(int(hit.get("store_id") or 0))
                source_row = source_row or {}
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
            json.dumps(_without_event_time(item), ensure_ascii=False)
        )
        if hits_out and response_chars + item_chars > response_char_cap:
            response_cap_truncated = True
            break
        response_chars += item_chars
        event_time_chars.append(
            len(json.dumps(item, ensure_ascii=False)) - item_chars
        )
        if reference_strict:
            strict_selector.deliver(entry)
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
    if rerank_enabled() and rerank_scores:
        response["provenance"]["rerank_scores"] = rerank_scores
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

        return _fit_answer_ready_response(
            fts_anchor_info=fts_anchor_info, response=response,
            response_char_cap=response_char_cap, expansion=expansion,
            delta_requested=delta_requested, arm_hits=arm_hits,
        )
    return json.dumps(response)


def _fit_answer_ready_response(
    *,
    fts_anchor_info: dict[str, Any] | None,
    response: dict[str, Any],
    response_char_cap: int,
    expansion: dict[str, Any],
    delta_requested: bool,
    arm_hits: dict[str, list[dict[str, Any]]],
) -> str:
    anchor_fired = bool(fts_anchor_info and fts_anchor_info.get("fired"))
    if anchor_fired:
        # Size the cap check with the widest value (``false``) so the final
        # recompute below can never grow the encoded response past the cap.
        fts_anchor_info["delivered"] = False
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
        expansion["expanded_hit_count"] = sum(
            "content" in hit for hit in response["hits"]
        )
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
    return encoded


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
            for key in _EVENT_TIME_KEYS:
                hits_out[index].pop(key, None)
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
        if key not in _EVENT_TIME_KEYS
    }


def _fit_event_time(response: dict[str, Any], *, response_char_cap: int) -> tuple[str, str | None]:
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
    encoded = json.dumps(response, ensure_ascii=False)
    for hit in reversed(response["hits"]):
        if len(encoded) <= response_char_cap:
            return encoded, None
        if not any(key in hit for key in _EVENT_TIME_KEYS):
            continue
        for key in _EVENT_TIME_KEYS:
            hit.pop(key, None)
        if not response.get("event_time_unavailable"):
            response["event_time_unavailable"] = True
            response["event_time_unavailable_reason"] = "response_cap"
        encoded = json.dumps(response, ensure_ascii=False)
    if len(encoded) > response_char_cap and response.get(
        "event_time_unavailable"
    ):
        del response["event_time_unavailable"]
        deferred_reason = response.pop("event_time_unavailable_reason")
        return json.dumps(response, ensure_ascii=False), deferred_reason
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
    return any(_hit_identity(hit) == anchor for hit in hits)
