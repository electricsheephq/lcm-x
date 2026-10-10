# Frozen verbatim from origin/main store.py MessageStore._search_like.
# Source SHA: 7cf309f65512c4d1adf3821399886e9e5a74a580
# Differential comparison only; production never imports this reference.
from __future__ import annotations

import json
import sqlite3
from typing import Any, Collection, Dict, List

from hermes_lcm.search_query import (
    AGE_DECAY_RATE,
    build_snippet,
    compute_like_fallback_fetch_limit,
    compute_search_candidate_cap,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    sanitize_like_query,
)
from hermes_lcm.store import (
    _MESSAGE_SELECT_COLUMNS,
    _conversation_filter_clause,
    _fallback_result_sort_key,
    _message_directness_score,
    _message_role_bias,
    _source_filter_clause,
)


class ReferenceLikeSearch:
    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None,
                     conversation_id: str | None = None,
                     role: str | None = None,
                     time_from: float | None = None,
                     time_to: float | None = None,
                     exclude_session_ids: Collection[str] | None = None) -> List[Dict[str, Any]]:
        # LIKE keeps every character the index cannot spell (emoji, punctuation)
        # because substring matching is the only way to find those rows.
        safe_query = sanitize_like_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["content IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        if exclude_session_ids:
            where.append("session_id NOT IN (SELECT value FROM json_each(?))")
            args.append(json.dumps(list(exclude_session_ids)))
        source_clause, source_args = _source_filter_clause("source", source)
        if source_clause:
            where.append(source_clause)
            args.extend(source_args)
        conversation_clause, conversation_args = _conversation_filter_clause("conversation_id", conversation_id)
        if conversation_clause:
            where.append(conversation_clause)
            args.extend(conversation_args)
        if role is not None:
            where.append("role = ?")
            args.append(role)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        like_clauses = []
        for term in terms:
            like_clauses.append("content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        normalized_sort = normalize_search_sort(sort)
        results: List[Dict[str, Any]] = []
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        order_by = ""
        order_args: list[Any] = []
        role_bias = "CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"

        def count_expr(term: str) -> tuple[str, list[Any]]:
            return (
                "((LENGTH(LOWER(content)) - LENGTH(REPLACE(LOWER(content), LOWER(?), ''))) "
                "/ NULLIF(LENGTH(?), 0))",
                [term, term],
            )

        if normalized_sort == "recency":
            score_exprs: list[str] = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"

            def build_unique_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(f"CASE WHEN ({expr}) > 0 THEN 1 ELSE 0 END")
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            def build_total_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(expr)
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            directness_args: list[Any] = []
            unique_score_expr, expr_args = build_unique_exprs(terms)
            directness_args.extend(expr_args)
            normalized_phrases = {(phrase or "").strip().lower() for phrase in phrases if (phrase or "").strip()}
            if phrases:
                phrase_hit_exprs: list[str] = []
                for phrase in phrases:
                    phrase_hit_exprs.append("CASE WHEN INSTR(LOWER(content), LOWER(?)) > 0 THEN 1 ELSE 0 END")
                    directness_args.append(phrase)
                phrase_hit_expr = " + ".join(phrase_hit_exprs) if phrase_hit_exprs else "0"
                non_phrase_terms = [term for term in terms if term.strip().lower() not in normalized_phrases]
                non_phrase_total_expr, expr_args = build_total_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                non_phrase_unique_expr, expr_args = build_unique_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({non_phrase_total_expr}) - ({non_phrase_unique_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) + (({phrase_hit_expr}) * 8.0) - MIN(({repetition_expr}), 6)"
            else:
                total_repetition_expr, expr_args = build_total_exprs(terms)
                directness_args.extend(expr_args)
                unique_repetition_expr, expr_args = build_unique_exprs(terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({total_repetition_expr}) - ({unique_repetition_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) - MIN(({repetition_expr}), 6)"
            order_args.extend(directness_args)
            order_by = (
                f"ORDER BY timestamp DESC, {role_bias} ASC, ({score_expr}) DESC, "
                f"({directness_expr}) DESC, store_id DESC"
            )

        def add_rows(rows: list[sqlite3.Row]) -> None:
            for row in rows:
                result = self._row_to_dict(row)
                content = result.get("content") or ""
                score = sum(
                    min(count_term_matches(content, term), 1) if collapse_risky_repeats else count_term_matches(content, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                result["search_rank"] = -float(score)
                result["snippet"] = build_snippet(content, terms)
                result["_fallback_score"] = float(score)
                result["_directness_score"] = _message_directness_score(result.get("role"), content, terms, phrases)
                results.append(result)

        if normalized_sort == "recency":
            candidate_cap = compute_search_candidate_cap(limit)
            offset = 0
            scanned_rows = 0
            while True:
                batch_limit = min(fetch_limit, candidate_cap - scanned_rows)
                if batch_limit <= 0:
                    break
                rows = self._conn.execute(
                    f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*base_args, *order_args, batch_limit, offset],
                ).fetchall()
                scanned_rows += len(rows)
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break
                if scanned_rows >= candidate_cap:
                    boundary_timestamp = rows[-1][8]
                    boundary_role_bias = _message_role_bias(rows[-1][3])
                    while True:
                        tie_rows = self._conn.execute(
                            f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                                FROM messages
                                WHERE {' AND '.join(where)}
                                {order_by}
                                LIMIT ? OFFSET ?""",
                            [*base_args, *order_args, fetch_limit, offset],
                        ).fetchall()
                        if not tie_rows:
                            break
                        matching_tie_rows = []
                        reached_next_primary_group = False
                        for tie_row in tie_rows:
                            if tie_row[8] == boundary_timestamp and _message_role_bias(tie_row[3]) == boundary_role_bias:
                                matching_tie_rows.append(tie_row)
                            else:
                                reached_next_primary_group = True
                                break
                        add_rows(matching_tie_rows)
                        if reached_next_primary_group or len(tie_rows) < fetch_limit:
                            break
                        offset += len(tie_rows)
                    break
        else:
            # Deterministic relevance/hybrid candidate scan for LIKE fallback.
            # Apply the same coarse score/directness ordering before the hard
            # candidate cap that Python uses below; otherwise a recent-biased
            # window can exclude older but materially better relevance matches.
            score_exprs: list[str] = []
            order_args = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"
            exact_query = (query or "").strip()
            exact_expr = "CASE WHEN LOWER(content) = LOWER(?) THEN 1 ELSE 0 END" if exact_query else "0"
            exact_args: list[Any] = [exact_query] if exact_query else []
            directness_expr = "0.0 + 0"

            if normalized_sort == "hybrid":
                primary_expr = (
                    f"(({score_expr}) / (1 + (MAX(0.0, "
                    f"((strftime('%s','now') - timestamp) / 3600.0)) * {AGE_DECAY_RATE})))"
                )
            else:
                primary_expr = f"({score_expr})"

            order_by = (
                f"ORDER BY {primary_expr} DESC, ({exact_expr}) DESC, ({directness_expr}) DESC, "
                f"{role_bias} ASC, timestamp DESC, store_id DESC"
            )
            candidate_cap = compute_search_candidate_cap(limit)
            offset = 0
            while offset < candidate_cap:
                batch_limit = min(fetch_limit, candidate_cap - offset)
                rows = self._conn.execute(
                    f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*base_args, *order_args, *exact_args, batch_limit, offset],
                ).fetchall()
                if not rows:
                    break
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break

        results.sort(key=lambda result: _fallback_result_sort_key(result, sort))
        for result in results:
            result.pop("_fallback_score", None)
        return results[:limit]

