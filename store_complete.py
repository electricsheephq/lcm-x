"""#581 store-complete leaves: a leaf's summarizer input is a bounded, contiguous prefix of the owned
stored rows above the frontier, in store order.

The publication proof (lifecycle_state.py ``stage_compaction_publication``) asks every owned row in
(frontier, covered_end] to be covered or excluded. A leaf built only from the host view could never
prove a row the host no longer shows (a row the host compacted natively, an older duplicate copy),
so publication conflicted forever. Here rows the view shows keep their view content; rows it does
not show ("hidden") are read from the store. The publication proof also rechecks copy exclusions.

Owned = the bound conversation's (and blank-conversation) rows of the bound session plus the carry
set publication uses; another conversation's rows under the same session are not this leaf's. Every summarized row counts against the leaf
budget; a leaf may be hidden rows only (it consumes no host row). The leaf ends before the first row
it cannot account for: a row a retained view row maps or shows (fresh tail, later host rows), or a
chunk row the budget cut. Ignore-matched rows keep their stored filter-exclusion proof; a row an
existing summary already covers, a stored system prompt (the anchor rule) and a hidden reply to an
ignored hidden prompt are excluded.
Proven hidden re-ingest copies are excluded only when an earlier owned byte-equal twin is summarized;
the publication transaction rechecks that proof. Equality alone never excludes an occurrence.
Parity: with nothing hidden, claimed or excluded the leaf is today's input, except that one pass
reads ``_SCAN_LIMIT`` owned rows per source (up to ``_SCAN_EXTEND`` pages when only excluded rows
are read); a longer backlog is taken as a prefix and
drains over later passes (never past a row the scan did not read).
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

from .config import host_message_uid_mode
from .fresh_tail import tool_group_safe_end
from .ingest_protection import quarantine_suspicious_assistant_messages
from .message_analysis import _tool_call_id
from .reconcile import _has_lossy_redacted_identity
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)

_SCAN_LIMIT = 2000  # owned rows per source page; a larger eligible backlog drains over later passes
_SCAN_EXTEND = 4  # a first tool group open at the cap is re-read once with this many times the page
_COPY_WINDOW = 8000  # store ids before the scanned prefix; a lookup miss leaves the row summarized
_CONTIGUITY = "leaf input cannot form a contiguous stored prefix"


class HiddenBacklog(NamedTuple):
    """#597: bounded scan count; truthiness preserves the existing scheduling decision."""

    rows: int
    truncated: bool

    def __bool__(self) -> bool:
        return self.rows > 0


class StoreCompleteMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``self._lifecycle`` and the identity helpers."""

    _retained_escape_enabled = True  # #1162: engine-only kill switch, never a config key

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

    def _store_complete_copy_twins(self, rows, carry) -> tuple[dict, set]:
        """One bounded batched read, keyed by exact stored columns, not normalized replay identity.
        Only hidden callers may use these candidates; coverage still requires a node or THIS leaf.
        Extending a tool-group scan reuses the lookup: misses safely remain summarizer input."""
        if not rows:
            return {}, set()
        ids = [int(row["store_id"]) for row in rows]
        first = min(ids)
        try:
            found = self._store.connection.execute(
                "SELECT store_id, session_id, conversation_id, role, content, tool_call_id, tool_calls, tool_name "
                "FROM messages WHERE store_id IN (SELECT value FROM json_each(?)) "
                "OR (store_id > ? AND store_id < ?) ORDER BY store_id",
                (str(ids), max(0, first - _COPY_WINDOW), first),
            ).fetchall()
        except Exception:
            logger.debug("LCM copy lookup unavailable", exc_info=True)
            return {}, set()
        owned = [row for row in found if self._identity_anchor_owned(
            {"store_id": row[0], "session_id": row[1]}, carry)
            and str(row[2] or "").strip() in {"", str(self._conversation_id or "")}]
        owned_ids = [int(row[0]) for row in owned]
        lineage, _reason = self._host_uid_lineage_key()
        bound, canonical, composite = {}, {}, set()
        # #1146 r2: absent uid bindings prove E2 copies only while the shadow writer runs.
        if lineage and host_message_uid_mode() != "off":
            try:
                bound = self._store.host_uid_uids_of_stores(lineage, owned_ids)
                canonical = self._store.host_uid_uids_of_stores(lineage, owned_ids, kinds=("canonical",), require_disagree=True)
                composite = {int(r["related_store_id"]) for r in self._store.get_message_relations(owned_ids, "composite", related=True)}
            except Exception:  # optional uid evidence unavailable: E2 is false, never a guessed binding
                bound, canonical = {}, {}
                logger.debug("LCM copy uid evidence unavailable", exc_info=True)
        by_key, twins = {}, {}
        stored = {int(row["store_id"]): row for row in rows}
        for raw in owned:
            sid, key = int(raw[0]), tuple(raw[3:])
            row = stored.get(sid)
            if row and by_key.get(key) and not _has_lossy_redacted_identity(self._message_replay_identity(row, strip_carrier=False)):
                e1 = (row.get("role") == "assistant" and any(_tool_call_id(call) for call in (row.get("tool_calls") or []))
                      or row.get("role") == "tool" and bool(str(row.get("tool_call_id") or "").strip()))
                # #1146 r3: never E2 for user rows; the host may ingest the current turn before it mints its uid.
                # Disclosed residual: an assistant row left unbound by a failed shadow write and also hidden can be excluded; its bytes stay summarized through the twin.
                matches = [t for t in by_key.get(key, ()) if (e1 and not bound.get(sid)) or (row.get("role") != "user" and canonical.get(t) and not bound.get(sid) and sid not in composite)]
                if matches:
                    twins[sid] = matches
            by_key.setdefault(key, []).append(sid)
        covered = self._store_complete_node_covered({t for matches in twins.values() for t in matches})
        return twins, covered

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

    def _store_complete_backlog(self, working, leading: int, copy_lookup_out=None) -> HiddenBacklog:
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
        carry = self._load_compression_carry_ranges()
        rows, truncated = self._store_complete_owned_rows(frontier, end, carry)
        conversation = {"", str(self._conversation_id or "")}
        loose = [int(row["store_id"]) for row in rows if int(row["store_id"]) not in mapped and row.get("role") != "system"
                 and str(row.get("conversation_id") or "").strip() in conversation
                 and not self._matches_ignore_message_patterns(row, stored_row=True)]
        twins, twin_covered = self._store_complete_copy_twins(rows, carry) if loose else ({}, set())
        if copy_lookup_out is not None:
            copy_lookup_out.append((twins, twin_covered))  # this leaf reuses the scheduler's bounded lookup
        # No current leaf yet: only node-covered twins prove a drained backlog. Retained unmapped
        # occurrences are never subtracted (the walk will block them).
        from .identity_anchor import _match_occurrences
        candidates = [row for row in rows if int(row["store_id"]) in loose]
        occurrences = [(i, self._message_replay_identity(m, strip_carrier=False))
                       for i, m in enumerate(working) if id(m) not in full_map]
        self._load_host_rewrite_overrides(candidates)
        blocked = set(_match_occurrences(candidates, self._stored_row_forms, occurrences)) if occurrences else set()
        copies = {sid for sid, matches in twins.items() if not truncated and sid not in blocked and any(t in twin_covered for t in matches)}
        result = self._last_hidden_backlog = HiddenBacklog(
            len(set(loose) - self._store_complete_node_covered(loose) - copies), truncated)
        if result:
            self._compress_run.store_ids_by_message_id = full_map
        elif truncated:
            key = self._hold_conversation_key()
            if key not in self._hidden_backlog_unknown_warned:
                self._hidden_backlog_unknown_warned.add(key)
                last = max((int(row["store_id"]) for row in rows), default=frontier)
                logger.warning("LCM hidden backlog conversation=%s frontier=%d last_store_id=%d: backlog beyond "
                               "that point is unknown; pass scheduled as drained", key, frontier, last)
        return result

    def _store_complete_input(self, chunk, claims, full_map, view, raw_chunk, frontier, carry, budget,
                              accounted_ids=(), scale: int = 1, scanned=None, copy_lookup=None,
                              escape_retained: bool = False, allow_retained_escape: bool = False,
                              lift_allowed: bool = True, prefix_retry: bool = True) -> Optional[list]:
        """``[(input_row, [store_id, ...]), ...]`` in walk order (when the host reordered the leaf,
        ``_store_complete_order`` holds each input row's store-order rank); ``[]`` when the leaf cannot start
        (a retained occurrence, no source rows, or a contiguity cut); None: today's input."""
        from .identity_anchor import _match_occurrences

        self._store_complete_excluded, self._store_complete_cut = [], False
        self._store_complete_copy_exclusion_proofs = {}
        self._store_complete_blocked = None  # #1162: (store_id, role, kind, chunk head position) of a [] return
        self._store_complete_empty_reason = ""
        self._store_complete_escaped = []
        self._store_complete_order = {}  # #2007: id(input row) -> store-order rank, set only for a host-reordered leaf
        given_claims = claims
        in_chunk = {id(message) for message in chunk}
        claims = {key: list(ids) for key, ids in claims.items()}
        # Rows this pass accounts for elsewhere: its exclusions (anchors, scaffold, committed replay)
        # and the dependent replies inside the raw chunk, which the consumed prefix covers.
        passive = set(accounted_ids) | {full_map[id(m)] for m in raw_chunk if id(m) in full_map and id(m) not in in_chunk}
        taken = {full_map[id(m)] for m in chunk if id(m) in full_map} | {s for ids in claims.values() for s in ids}
        top = max(taken, default=0)
        end = top - 1 if top > frontier else min(  # no host row: hidden backlog below the first retained row
            (store_id - 1 for store_id in set(full_map.values()) - passive
             if store_id > frontier and not escape_retained), default=None)
        rows, truncated = scanned if scanned is not None else (
            self._store_complete_owned_rows(frontier, end, carry, scale, passive)
            if end is None or end > frontier else ([], False))
        loose = [row for row in rows if int(row["store_id"]) not in
                 taken | passive | (set() if escape_retained else set(full_map.values()))]
        ignored = {int(row["store_id"]) for row in loose if self._matches_ignore_message_patterns(row, stored_row=True)}
        covered = self._store_complete_node_covered([int(row["store_id"]) for row in loose if int(row["store_id"]) not in ignored])
        candidates = [row for row in loose if int(row["store_id"]) not in ignored | covered]
        occurrences = [(idx, self._message_replay_identity(message, strip_carrier=False))
                       for idx, message in enumerate(view) if id(message) not in full_map and not claims.get(id(message))]
        self._load_host_rewrite_overrides(candidates)
        shown = _match_occurrences(candidates, self._stored_row_forms, occurrences) if occurrences else {}
        blocked = set()
        escaped = {int(row["store_id"]): "mapped" for row in candidates
                   if escape_retained and int(row["store_id"]) in full_map.values()}
        for store_id, idx in shown.items():
            if id(view[idx]) in in_chunk:  # the chunk row's own text is this stored occurrence
                claims.setdefault(id(view[idx]), []).append(store_id)
                taken.add(store_id)
            elif escape_retained:
                escaped[store_id] = "shown"
            else:  # a retained occurrence: never hidden, the leaf ends before it
                blocked.add(store_id)
        hidden = [row for row in candidates if int(row["store_id"]) not in taken | blocked]
        if copy_lookup is None and hidden:
            copy_lookup = self._store_complete_copy_twins(rows, carry)
        twins, twin_covered = copy_lookup if copy_lookup is not None else ({}, set())
        copy_proofs, admitted, group_twins = {}, set(), {}
        text = dict(zip((int(row["store_id"]) for row in hidden), self._store_complete_messages(hidden)))

        def ids_of(message, ids) -> list:
            return list(ids) + ([full_map[id(message)]] if id(message) in full_map else [])

        def position(message, ids=None) -> int:
            return max(ids_of(message, claims.get(id(message), ()) if ids is None else ids), default=0)

        def calls(messages):
            return {_tool_call_id(call) for m in messages for call in m.get("tool_calls") or ()} - {""}

        def results(messages):
            return {str(m.get("tool_call_id") or "").strip() for m in messages if m.get("role") == "tool"} - {""}

        retained = ([text[sid] for sid in escaped if sid in text] +
                    [m for m in view if id(m) not in in_chunk]) if escape_retained else []
        cross = (calls(chunk) & results(retained)) | (calls(retained) & results(chunk))
        group_cut = min((sid for sid, m in [*[(int(r["store_id"]), r) for r in rows],
                                           *[(position(m), m) for m in chunk]]
                         if sid > frontier and (calls([m]) & cross or results([m]) & cross)), default=None)

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
                admitted.intersection_update(s for m, i in out for s in ids_of(m, i))
                used, group = group[2], []
                return False
            if answers:
                group[3].add(str(message.get("tool_call_id") or "").strip())
            out.append((message, ids))
            admitted.update(ids_of(message, ids))
            used += tokens
            return True

        def emit_chunk(limit) -> bool:
            nonlocal index, dependent
            while index < len(chunk) and (limit is None or position(chunk[index]) < limit):
                if group_cut is not None and position(chunk[index]) >= group_cut:
                    return False  # I9: no host half of an escaped tool group is consumed
                if not take(chunk[index], claims.get(id(chunk[index]), [])):
                    return False
                dependent = dependent and chunk[index].get("role") not in ("user", "system")
                index += 1
            return True

        complete = True
        for row_index, row in enumerate(rows):
            store_id, role = int(row["store_id"]), row.get("role")
            if group_cut is not None and store_id >= group_cut:
                complete = False
                break
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
            twin = next((t for t in twins.get(store_id, ()) if t in twin_covered or t in admitted), None)
            if store_id in text and role == "assistant" and row.get("tool_calls"):
                # #1154: decide copy exclusion once for the caller and its whole tool group.
                call_ids = {_tool_call_id(call) for call in row["tool_calls"]}
                members, group_dependent = [store_id], False
                for result in rows[row_index + 1:]:
                    if result.get("role") != "tool":
                        break
                    sid = int(result["store_id"])
                    if sid in ignored:
                        group_dependent = True
                    if (str(result.get("tool_call_id") or "").strip() in call_ids
                            and sid not in taken | passive | covered | ignored and not group_dependent):
                        members.append(sid)
                choices = {sid: next((t for t in twins.get(sid, ())
                                      if t in twin_covered or t in admitted), None) for sid in members}
                can_exclude = all(sid in text and choices[sid] is not None for sid in members)
                group_twins.update({sid: choices[sid] if can_exclude else None for sid in members})
            if store_id in group_twins:
                twin = group_twins[store_id]
            if store_id in text and twin is not None:
                copy_proofs[store_id] = twin
                excluded.append(store_id)
                if role == "assistant" and row.get("tool_calls"):
                    group = []
                dependent = dependent and role not in ("user", "system")
                continue
            if store_id not in text or not take(text[store_id], [store_id]):  # a retained row, or the budget
                if store_id not in text:  # #1162: shown by an unmapped view occurrence, or mapped by a view row
                    self._store_complete_blocked = (store_id, role, "shown" if store_id in blocked else "mapped",
                                                    position(chunk[0]) if chunk else 0)
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
                                                  budget, accounted_ids, scale=_SCAN_EXTEND, scanned=scanned,
                                                  copy_lookup=copy_lookup, escape_retained=escape_retained,
                                                  allow_retained_escape=allow_retained_escape,
                                                  lift_allowed=lift_allowed, prefix_retry=prefix_retry)
            else:  # the documented oversized-group exception: admitted as read
                logger.warning("LCM store-complete leaf: a first tool group is still open after a %d-row scan; "
                               "admitted as read (frontier=%d)", _SCAN_LIMIT * scale, frontier)
        # Contiguity: every owned row up to the leaf's last row is accounted for, else the leaf ends
        # before the first one that is not (a chunk row the budget cut, a claim left behind).
        hole_stop = None  # #1998: (hole, taken rows not emitted) of the cut that empties the leaf
        while out:
            accounted = {s for m, ids in out for s in ids_of(m, ids)} | set(excluded) | ignored | passive
            last = max((s for m, ids in out for s in ids_of(m, ids)), default=0)
            hole = next((int(row["store_id"]) for row in rows
                         if int(row["store_id"]) < last and int(row["store_id"]) not in accounted), None)
            if hole is None:
                break
            out = out[:next(i for i, (m, ids) in enumerate(out) if any(s > hole for s in ids_of(m, ids)))]
            complete = False
            if not out:
                self._store_complete_empty_reason = "leaf input cannot form a contiguous stored prefix"
                hole_stop = (hole, len(taken - accounted))
        if copy_proofs and not any(ids_of(m, ids) for m, ids in out):
            return self._store_complete_input(chunk, given_claims, full_map, view, raw_chunk, frontier, carry,
                                              budget, accounted_ids, scale=scale, scanned=(rows, truncated), copy_lookup=({}, set()),
                                              escape_retained=escape_retained, allow_retained_escape=allow_retained_escape,
                                              lift_allowed=lift_allowed, prefix_retry=prefix_retry)
        if not out and allow_retained_escape and self._retained_escape_enabled:
            blocked_row = self._store_complete_blocked
            lifted = self._context_aware_leaf_cap() or max(budget, 1) * _SCAN_EXTEND
            if (blocked_row and not escape_retained) or (not blocked_row and lift_allowed and
                    self._store_complete_empty_reason == "leaf input cannot form a contiguous stored prefix"
                    and lifted > budget):
                result = self._store_complete_input(
                    chunk, given_claims, full_map, view, raw_chunk, frontier, carry,
                    budget if blocked_row else lifted, accounted_ids, scale=scale, scanned=(rows, truncated),
                    copy_lookup=copy_lookup, escape_retained=bool(blocked_row) or escape_retained,
                    allow_retained_escape=not escape_retained, lift_allowed=lift_allowed and bool(blocked_row),
                    prefix_retry=prefix_retry)
                if blocked_row is not None:
                    self._store_complete_blocked = blocked_row
                return result
        if hole_stop is not None and not out:
            # #1998: a chunk row stored above a later chunk row (a claim on an unmapped row) holds back
            # every row behind it; when the walk stops first (the budget, a truncated scan), their claims
            # become a hole. Once, retry on the tool-group-safe chunk prefix before the blocking row: the
            # rows cut off stay retained host rows, which the #1162 escape summarises from the store.
            hole, pending = hole_stop
            holder = next((k for k in range(index, len(chunk)) if hole in claims.get(id(chunk[k]), ())), None)
            end = tool_group_safe_end(chunk, index) if holder is not None else 0
            retry = (prefix_retry and holder is not None and not escape_retained and allow_retained_escape
                     and self._retained_escape_enabled)
            logged = self.__dict__.setdefault("_contiguity_hole_logged", set())
            if str(self._conversation_id or "") not in logged:
                logged.add(str(self._conversation_id or ""))
                logger.info(
                    "LCM contiguity hole: conversation=%s hole_store_id=%d role=%s taken_not_emitted=%d "
                    "blocking_chunk_index=%d/%d blocking_position=%d truncated=%s budget=%d used=%d prefix_retry=%s",
                    self._conversation_id, hole, next((row.get("role") for row in rows if int(row["store_id"]) == hole), None),
                    pending, index, len(chunk), position(chunk[index]) if index < len(chunk) else 0, truncated,
                    budget, used, retry)
            if retry:
                prefix = chunk[:end if end <= index else 0]  # never extended through the blocking row
                cut = {id(message) for message in chunk[len(prefix):]}
                raw_end = next((i for i, message in enumerate(raw_chunk) if id(message) in cut), len(raw_chunk))
                result = self._store_complete_input(
                    prefix, {id(m): given_claims[id(m)] for m in prefix if id(m) in given_claims}, full_map, view,
                    raw_chunk[:raw_end] if prefix else [], frontier, carry, budget, accounted_ids,
                    allow_retained_escape=True, lift_allowed=False, prefix_retry=False)
                if result is None:  # the prefix is today's input; the caller would read None as the whole chunk
                    result = [(message, []) for message in prefix]
                if result:
                    self._store_complete_cut = True
                    return result
                self._store_complete_empty_reason, self._store_complete_blocked = _CONTIGUITY, None
                self._store_complete_escaped, self._store_complete_order = [], {}
        if not out or (not any(ids_of(m, ids) for m, ids in out)
                       and all(self._is_context_summary_content(m.get("content")) for m in chunk)):
            if not self._store_complete_blocked and not self._store_complete_empty_reason:
                # This branch already established that no output row carries a source.
                self._store_complete_empty_reason = (
                    f"leaf input has no stored source rows (input_rows_without_source={len(out)})")
            return []  # nothing to cover: at most a host summary of rows already at or below the frontier
        emitted = sum(1 for message, _ids in out if id(message) in in_chunk)
        self._store_complete_escaped = [(sid, m.get("role"), escaped[sid])
                                        for m, ids in out for sid in ids if sid in escaped]
        # #2007: the walk above decides WHICH rows the leaf takes and their order (a host prefix of the chunk,
        # which _leaf_consumption and a rescue prefix rely on); the summarizer reads them in store order, each
        # tool group whole, through the rank published in _store_complete_order.
        # A call group is an assistant row with tool calls plus every tool row that answers it: the call with
        # that id stored nearest before the result, else the nearest preceding one in the walk (a repeated id
        # follows the store, #2011). Every other row with a store id is its own unit; a row
        # without one (position 0) stays behind the row before it, a leading one first. A group sorts at
        # its call's store position or, when the call has none, at its earliest stored result; inside it
        # the call stays first and its results follow in store order. A leaf whose stored rows are already
        # in store order gets no rank, even when the store holds a row inside a group; in a regrouped
        # leaf such a row follows the group. The sorts are stable.
        def call_ids(message) -> set:
            return calls([message]) if message.get("role") == "assistant" else set()

        def tool_id(message) -> str:
            return str(message.get("tool_call_id") or "").strip() if message.get("role") == "tool" else ""

        answered = {call_id for message, _ids in out for call_id in call_ids(message)}
        parts, part = [], None  # part = [row, *id-less rows behind it], in walk order
        for message, ids in out:
            if tool_id(message) in answered or call_ids(message) or position(message, ids) or part is None:
                part = [(message, ids)]
                parts.append(part)
            else:
                part.append((message, ids))
        heads, owners = {}, {}  # walk index of a unit's head -> unit; call id -> [(store position, walk index)]
        for index, part in enumerate(parts):
            if call_ids(part[0][0]):
                heads[index] = [part]  # unit = [[head, *id-less], [result, *id-less], ...]
                for call_id in call_ids(part[0][0]):
                    owners.setdefault(call_id, []).append((position(*part[0]), index))
        for index, part in enumerate(parts):
            if index in heads:
                continue
            at, mine = position(*part[0]), owners.get(tool_id(part[0][0]), [])
            owner = max([o for o in mine if at and 0 < o[0] < at], default=None) or max(
                [o for o in mine if o[1] < index], key=lambda o: o[1], default=None)
            if owner is None:
                heads[index] = [part]
            else:
                heads[owner[1]].append(part)
        units = [heads[index] for index in sorted(heads)]

        def by_position(parts) -> list:  # stable; a part without a store id keeps its predecessor's key
            keys, key = [], 0
            for unit_parts in parts:
                key = position(*unit_parts[0][0]) or min((p for p in (position(*r[0]) for r in unit_parts[1:]) if p),
                                                        default=0) or key
                keys.append(key)
            return [item for _key, item in sorted(zip(keys, parts), key=lambda pair: pair[0])]
        stored = [p for p in (position(m, ids) for m, ids in out) if p]
        if any(a > b for a, b in zip(stored, stored[1:])):
            ordered = [pair for unit in by_position(units)
                       for part in [unit[0], *(p[0] for p in by_position([[r] for r in unit[1:]]))] for pair in part]
            self._store_complete_order = {id(m): i for i, (m, _ids) in enumerate(ordered)}
        if emitted == len(out) == len(chunk) and not excluded and not any(ids for _m, ids in out):
            return None  # nothing hidden, claimed or excluded: today's input and mapping
        last = max(s for m, ids in out for s in ids_of(m, ids)) if any(ids_of(m, ids) for m, ids in out) else frontier
        self._store_complete_excluded = [store_id for store_id in excluded if store_id < last]
        self._store_complete_copy_exclusion_proofs = {c: t for c, t in copy_proofs.items() if c < last}
        self._store_complete_cut = emitted < len(chunk) or not complete
        logger.info(
            "LCM store-complete leaf: frontier=%d last=%d rows=%d host_rows=%d/%d hidden=%d excluded=%d copies=%d cut=%s",
            frontier, last, len(out), emitted, len(chunk), len(out) - emitted,
            len(self._store_complete_excluded), len(self._store_complete_copy_exclusion_proofs), self._store_complete_cut,
        )
        return out
