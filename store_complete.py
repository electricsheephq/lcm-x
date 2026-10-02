"""#581 store-complete leaves: a leaf's summarizer input is a bounded, contiguous prefix of the owned
stored rows above the frontier, in store order.

The publication proof (lifecycle_state.py ``stage_compaction_publication``) asks every owned row in
(frontier, covered_end] to be covered or excluded. A leaf built only from the host view could never
prove a row the host no longer shows (a row the host compacted natively, an older duplicate copy),
so publication conflicted forever. Here rows the view shows keep their view content; rows it does
not show ("hidden") are read from the store and summarized like any other row. The proof itself is
unchanged: this only changes what is fed into it.

Owned = the bound conversation's (and blank-conversation) rows of the bound session plus the carry
set publication uses; another conversation's rows under the same session are not this leaf's. Every row counts against the leaf
budget; a leaf may be hidden rows only (it consumes no host row). The leaf ends before the first row
it cannot account for: a row a retained view row maps or shows (fresh tail, later host rows), or a
chunk row the budget cut. Ignore-matched rows keep their stored filter-exclusion proof; a row an
existing summary already covers, a stored system prompt (the anchor rule) and a hidden reply to an
ignored hidden prompt are excluded.
Duplicate copies are summarized like any other row (no duplicate-exclusion machinery).
Parity: with nothing hidden, claimed or excluded the leaf is today's input, except that one pass
reads ``_SCAN_LIMIT`` owned rows per source (up to ``_SCAN_EXTEND`` pages when only excluded rows
are read); a longer backlog is taken as a prefix and
drains over later passes (never past a row the scan did not read).
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

from .ingest_protection import quarantine_suspicious_assistant_messages
from .message_analysis import _tool_call_id
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)

_SCAN_LIMIT = 2000  # owned rows per source page; a larger eligible backlog drains over later passes
_SCAN_EXTEND = 4  # a first tool group open at the cap is re-read once with this many times the page


class HiddenBacklog(NamedTuple):
    """#597: bounded scan count; truthiness preserves the existing scheduling decision."""

    rows: int
    truncated: bool

    def __bool__(self) -> bool:
        return self.rows > 0


class StoreCompleteMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``self._lifecycle`` and the identity helpers."""

    def _store_complete_frontier(self) -> int:
        """The publication frontier (the lifecycle row), never below the in-process marker. A frontier
        moved by `/lcm rotate apply` (the backup-first operator command) is honoured by design."""
        state = self._lifecycle.get_by_conversation(self._conversation_id) if self._conversation_id else None
        return max(int(self._last_compacted_store_id or 0), int(getattr(state, "current_frontier_store_id", 0) or 0))

    def _store_complete_owned_rows(self, frontier: int, end: Optional[int], carry, scale: int = 1,
                                   excluded_ids=()) -> tuple[list, bool]:
        """Owned rows in (frontier, end] in store order, and whether a source was cut at ``_SCAN_LIMIT``
        x ``scale`` (the rows then stop at the lowest id every source reached). Excluded-only pages
        advance at most ``_SCAN_EXTEND`` pages; retain them for the leaf's contiguity checks."""
        limit = _SCAN_LIMIT * scale
        sources = [(str(self._session_id), frontier, end)] + [
            (source, max(start, frontier), stop if end is None else min(stop, end))
            for source, start, stop in carry if stop > frontier and (end is None or start < end)
        ]
        rows, reached = {}, None
        for source, start, stop in sources:
            page = []
            for _ in range(_SCAN_EXTEND if scale == 1 else 1):
                batch = self._store.get_range(source, start_id=start + 1, end_id=stop, limit=limit,
                                              conversation_id=self._conversation_id, include_blank_conversation=True)
                page.extend(batch)
                covered = self._store_complete_node_covered([int(row["store_id"]) for row in batch])
                if len(batch) < limit or any(
                    int(row["store_id"]) not in covered and int(row["store_id"]) not in excluded_ids
                    and row.get("role") != "system" and not self._matches_ignore_message_patterns(row, stored_row=True)
                    for row in batch
                ):
                    break
                start = int(batch[-1]["store_id"])
            if len(batch) >= limit:
                last = int(page[-1]["store_id"])
                reached = last if reached is None else min(reached, last)
            rows.update((int(row["store_id"]), row) for row in page if self._identity_anchor_owned(row, carry))
        return [rows[key] for key in sorted(rows) if reached is None or key <= reached], reached is not None

    def _store_complete_node_covered(self, store_ids) -> set:
        """Rows an existing message-sourced summary (any session) already covers: never reclaimed."""
        if not store_ids:
            return set()
        found = self._store.connection.execute(
            "SELECT DISTINCT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
            "WHERE node.source_type = 'messages' AND source.value IN (SELECT value FROM json_each(?))",
            (str(sorted(store_ids)),),
        ).fetchall()
        return {int(row[0]) for row in found}

    def _store_complete_messages(self, rows) -> list:
        """Hidden rows as summarizer input through the ingest replay protections (quarantine,
        redaction). Tool calls stay data: the summarizer reads them, nothing replays them."""
        messages = [self._store.to_openai_msg(row) for row in rows]
        messages = quarantine_suspicious_assistant_messages(
            messages, config=self._config, hermes_home=self._hermes_home, session_id=self._session_id,
            externalize=[False] * len(messages),
        )
        return self._redact_active_replay_messages(messages)

    def _hidden_backlog_label(self) -> int | str | None:
        """#597: None means not evaluated; a capped empty scan cannot prove zero backlog."""
        result = self._last_hidden_backlog
        if result is None:
            return None
        if not result.truncated:
            return result.rows
        return f"{result.rows}+" if result.rows else "unknown"

    def _hidden_backlog_status(self) -> dict:
        label = self._hidden_backlog_label()
        return {"hidden_rows": label} if label is not None else {}

    def _store_complete_backlog(self, working, leading: int) -> HiddenBacklog:
        """#581: no host raw chunk, but owned rows the view does not show wait above the frontier and
        below the first retained row: a hidden-only leaf is scheduled instead of a no-op."""
        from .identity_anchor import identity_anchor_enabled

        self._last_hidden_backlog = HiddenBacklog(0, False)
        if not identity_anchor_enabled() or not self._session_id or not self._conversation_id:
            return self._last_hidden_backlog
        full_map = self._get_store_id_map_for_messages(working[leading:])
        frontier = self._store_complete_frontier()
        mapped = set(full_map.values()) | set(self._get_store_ids_for_messages(working[:leading]) if leading else ())
        end = min((store_id - 1 for store_id in mapped if store_id > frontier), default=None)
        if end is not None and end <= frontier:
            return self._last_hidden_backlog
        rows, truncated = self._store_complete_owned_rows(frontier, end, self._load_compression_carry_ranges())
        conversation = {"", str(self._conversation_id or "")}
        loose = [int(row["store_id"]) for row in rows if int(row["store_id"]) not in mapped and row.get("role") != "system"
                 and str(row.get("conversation_id") or "").strip() in conversation
                 and not self._matches_ignore_message_patterns(row, stored_row=True)]
        result = self._last_hidden_backlog = HiddenBacklog(
            len(set(loose) - self._store_complete_node_covered(loose)), truncated)
        if result:
            self._current_compress_store_ids_by_message_id = full_map
        elif truncated:
            key = self._hold_conversation_key()
            if key not in self._hidden_backlog_unknown_warned:
                self._hidden_backlog_unknown_warned.add(key)
                last = max((int(row["store_id"]) for row in rows), default=frontier)
                logger.warning("LCM hidden backlog conversation=%s frontier=%d last_store_id=%d: backlog beyond "
                               "that point is unknown; pass scheduled as drained", key, frontier, last)
        return result

    def _store_complete_input(self, chunk, claims, full_map, view, raw_chunk, frontier, carry, budget,
                              accounted_ids=(), scale: int = 1, scanned=None) -> Optional[list]:
        """``[(input_row, [store_id, ...]), ...]`` in store order; ``[]`` when the leaf cannot start
        (the first owned row above the frontier is a retained occurrence); None: today's input."""
        from .identity_anchor import _match_occurrences

        self._store_complete_excluded, self._store_complete_cut = [], False
        given_claims = claims
        in_chunk = {id(message) for message in chunk}
        claims = {key: list(ids) for key, ids in claims.items()}
        # Rows this pass accounts for elsewhere: its exclusions (anchors, scaffold, committed replay)
        # and the dependent replies inside the raw chunk, which the consumed prefix covers.
        passive = set(accounted_ids) | {full_map[id(m)] for m in raw_chunk if id(m) in full_map and id(m) not in in_chunk}
        taken = {full_map[id(m)] for m in chunk if id(m) in full_map} | {s for ids in claims.values() for s in ids}
        top = max(taken, default=0)
        end = top - 1 if top > frontier else min(  # no host row: hidden backlog below the first retained row
            (store_id - 1 for store_id in set(full_map.values()) - passive if store_id > frontier), default=None)
        rows, truncated = scanned if scanned is not None else (
            self._store_complete_owned_rows(frontier, end, carry, scale, passive)
            if end is None or end > frontier else ([], False))
        loose = [row for row in rows if int(row["store_id"]) not in taken | passive | set(full_map.values())]
        ignored = {int(row["store_id"]) for row in loose if self._matches_ignore_message_patterns(row, stored_row=True)}
        covered = self._store_complete_node_covered([int(row["store_id"]) for row in loose if int(row["store_id"]) not in ignored])
        candidates = [row for row in loose if int(row["store_id"]) not in ignored | covered]
        occurrences = [(idx, self._message_replay_identity(message, strip_carrier=False))
                       for idx, message in enumerate(view) if id(message) not in full_map and not claims.get(id(message))]
        self._load_host_rewrite_overrides(candidates)
        shown = _match_occurrences(candidates, self._stored_row_forms, occurrences) if occurrences else {}
        blocked = set()
        for store_id, idx in shown.items():
            if id(view[idx]) in in_chunk:  # the chunk row's own text is this stored occurrence
                claims.setdefault(id(view[idx]), []).append(store_id)
                taken.add(store_id)
            else:  # a retained occurrence: never hidden, the leaf ends before it
                blocked.add(store_id)
        hidden = [row for row in candidates if int(row["store_id"]) not in taken | blocked]
        text = dict(zip((int(row["store_id"]) for row in hidden), self._store_complete_messages(hidden)))

        def ids_of(message, ids) -> list:
            return list(ids) + ([full_map[id(message)]] if id(message) in full_map else [])

        def position(message) -> int:
            return max(ids_of(message, claims.get(id(message), ())), default=0)

        out, used, excluded, dependent, index = [], 0, [], False, 0
        group: list = []  # the open tool group: [call ids, its index in out, used before it, answered ids]

        def take(message, ids) -> bool:  # the budget cuts only at a group boundary, never before the first
            nonlocal used, group          # row that carries a store id (the leaf's coverage)
            tokens = count_message_tokens(message)
            answers = message.get("role") == "tool" and group and \
                str(message.get("tool_call_id") or "").strip() in group[0]
            if message.get("role") != "tool":
                group = []
                if used + tokens > budget and any(ids_of(m, i) for m, i in out):
                    return False
                if message.get("tool_calls"):
                    group = [{_tool_call_id(call) for call in message["tool_calls"]}, len(out), used, set()]
            elif answers and used + tokens > budget and any(ids_of(m, i) for m, i in out[:group[1]]):
                # R6-5: a group is admitted whole or not at all, as _select_oldest_leaf_chunk and
                # tool_group_safe_end admit it: its results cross the budget and the leaf already
                # covers a stored row, so the leaf ends before the call (a first group is admitted whole).
                del out[group[1]:]
                used, group = group[2], []
                return False
            if answers:
                group[3].add(str(message.get("tool_call_id") or "").strip())
            out.append((message, ids))
            used += tokens
            return True

        def emit_chunk(limit) -> bool:
            nonlocal index, dependent
            while index < len(chunk) and (limit is None or position(chunk[index]) < limit):
                if not take(chunk[index], claims.get(id(chunk[index]), [])):
                    return False
                dependent = dependent and chunk[index].get("role") not in ("user", "system")
                index += 1
            return True

        complete = True
        for row in rows:
            store_id, role = int(row["store_id"]), row.get("role")
            if not emit_chunk(store_id):
                complete = False
                break
            if store_id in ignored:  # as compaction.py: an ignored row of any role makes the next replies dependent
                dependent = True
                continue
            if store_id in taken or store_id in passive:
                dependent = dependent and role not in ("user", "system")
                continue
            if store_id in covered or role == "system" or (dependent and role in ("assistant", "tool")):
                excluded.append(store_id)  # a skipped prompt still ends a dependent run (compaction.py:1524)
                dependent = dependent and role not in ("user", "system")
                continue
            if store_id not in text or not take(text[store_id], [store_id]):  # a retained row, or the budget
                complete = False
                break
            dependent = dependent and role not in ("user", "system")
        if complete and not truncated:
            complete = emit_chunk(None)
        elif complete and group and group[0] - group[3]:
            # B-GROUP-1: the scan stopped at _SCAN_LIMIT inside a tool group (a result not read): as R6-5,
            # the leaf ends before the call; the next leaf starts at it.
            if any(ids_of(m, i) for m, i in out[:group[1]]):
                del out[group[1]:]
                complete = False
            elif scale == 1:  # a first group is never deferred: one bounded larger scan reads its results
                call = min(ids_of(*out[group[1]]), default=frontier + 1)
                extended, truncated = self._store_complete_owned_rows(max(frontier, call - 1), end, carry, _SCAN_EXTEND)
                scanned = ([row for row in rows if int(row["store_id"]) < call] + extended, truncated)
                return self._store_complete_input(chunk, given_claims, full_map, view, raw_chunk, frontier, carry,
                                                  budget, accounted_ids, scale=_SCAN_EXTEND, scanned=scanned)
            else:  # the documented oversized-group exception: admitted as read
                logger.warning("LCM store-complete leaf: a first tool group is still open after a %d-row scan; "
                               "admitted as read (frontier=%d)", _SCAN_LIMIT * scale, frontier)
        # Contiguity: every owned row up to the leaf's last row is accounted for, else the leaf ends
        # before the first one that is not (a chunk row the budget cut, a claim left behind).
        while out:
            accounted = {s for m, ids in out for s in ids_of(m, ids)} | set(excluded) | ignored | passive
            last = max((s for m, ids in out for s in ids_of(m, ids)), default=0)
            hole = next((int(row["store_id"]) for row in rows
                         if int(row["store_id"]) < last and int(row["store_id"]) not in accounted), None)
            if hole is None:
                break
            out = out[:next(i for i, (m, ids) in enumerate(out) if any(s > hole for s in ids_of(m, ids)))]
            complete = False
        if not out or (not any(ids_of(m, ids) for m, ids in out)
                       and all(self._is_context_summary_content(m.get("content")) for m in chunk)):
            return []  # nothing to cover: at most a host summary of rows already at or below the frontier
        emitted = sum(1 for message, _ids in out if id(message) in in_chunk)
        if emitted == len(out) == len(chunk) and not excluded and not any(ids for _m, ids in out):
            return None  # nothing hidden, claimed or excluded: today's input and mapping
        last = max(s for m, ids in out for s in ids_of(m, ids)) if any(ids_of(m, ids) for m, ids in out) else frontier
        self._store_complete_excluded = [store_id for store_id in excluded if store_id < last]
        self._store_complete_cut = emitted < len(chunk) or not complete
        logger.info(
            "LCM store-complete leaf: frontier=%d last=%d rows=%d host_rows=%d/%d hidden=%d excluded=%d cut=%s",
            frontier, last, len(out), emitted, len(chunk), len(out) - emitted,
            len(self._store_complete_excluded), self._store_complete_cut,
        )
        return out
