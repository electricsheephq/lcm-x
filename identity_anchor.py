"""#436: host-timestamp-anchored, occurrence-bound message identity (design REVISION 1).

Hermes re-issues every durable row on every copy path (compaction generations, rotation, restore,
``replace_messages``) and rewrites, merges and re-orders scaffolding on the way. The message
``timestamp`` survives all of them and LCM stores it as ``observed_at``. Each incoming row with a host
stamp is recognised INDIVIDUALLY against the stored occurrences of the proven lineage instead of as part
of an ordered prefix, so one unmatched row no longer re-appends the rest of the list.

Rules (REVISION 1):
R1 the full replay payload identity plus ``observed_at`` finds CANDIDATES in this session and its
verified compression ancestors (host state.db ``parent_session_id``), consumed once per host
occurrence (multiset); R8 LCM's own carriers and summaries keep their DAG-verified identity.
R5 a NULL stamp is backfilled only for a uniquely proven occurrence, and H1's unstable current-turn
stamp attaches only to the one occurrence proven this turn.
R2 a merge survivor is recognised only by an exact, unique, ordered decomposition into stored
occurrences, or by a relation LCM itself recorded when it saw that composite; R3 a survivor holding
a stored head and a new remainder stores the remainder once (its own stamp unknown) with the relation.
R4 (coverage bound to the summarizer input) lives in compaction's input loop.
R7 carry comes only from the verified compression-ancestor chain, never from a sibling session.
R6 (supersede on positive evidence) hooks the host-rewrite capture.

``LCM_IDENTITY_ANCHOR`` (default on): ``0``/``false``/``no``/``off`` restores the pre-#436 ingest exactly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from .recognition_matching import (
    _MAX_DECOMPOSITIONS as _MAX_DECOMPOSITIONS,
    _DECOMPOSE_BUDGET as _DECOMPOSE_BUDGET,
    _DECOMPOSE_MAX_PARTS as _DECOMPOSE_MAX_PARTS,
    _MATCH_WORK_PER_ITEM as _MATCH_WORK_PER_ITEM,
    _MATCH_WORK_FLOOR as _MATCH_WORK_FLOOR,
    _decompositions as _decompositions,
    _match_occurrences as _match_occurrences,
    _reserve_shown as _reserve_shown,
    _reroute_for_null as _reroute_for_null,
)
from .config import _parse_bool_env
from .fresh_tail import tool_group_safe_end
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .store import _normalize_observed_at
from .tokens import count_messages_tokens

logger = logging.getLogger(__name__)

_CARRY_PREFIX = "identity_anchor_carry"
_RECENT_CAP = 16  # rows this process stored lately: the R5 current-turn window
_POOL_WINDOW = 256  # store ids either side of a stamp donor searched for a composite's constituents
_INSERT_DIFF_BUDGET = 1_000_000  # #633: old x new rows compared per insertion diff (quadratic worst case ~40 ms)
_CARRIER_LEDGER_CAP = 256  # #1000: carrier emission keys remembered as seen / retired (process-local)


def identity_anchor_enabled() -> bool:
    return _parse_bool_env("LCM_IDENTITY_ANCHOR", True, unknown=True, true_values=())


class IdentityAnchorMixin:
    """Mixed into LCMEngine; reads ``self._store``, the reconcile identity helpers and ``_state_db_path``."""

    # -- lineage (R7) --------------------------------------------------------

    def _identity_anchor_chain(self) -> list[str]:
        """The bound session's verified compression ancestors, nearest first: host state.db
        ``sessions.parent_session_id`` while the parent ended with ``end_reason='compression'``
        (written in one txn by ``publish_compression_child``). Read-only; empty when unreadable."""
        session_id = str(self._session_id or "")
        cached = getattr(self, "_identity_anchor_chain_cache", None)
        if cached is not None and cached[0] == session_id:
            return cached[1]
        chain: list[str] = []
        read = False
        try:
            from .host_uid import _is_fork_child

            path = self._state_db_path()
            if session_id and path.exists():
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
                conn.row_factory = sqlite3.Row
                try:
                    current, seen = session_id, {session_id}
                    for _ in range(256):
                        child = conn.execute("SELECT * FROM sessions WHERE id = ?", (current,)).fetchone()
                        if not child or _is_fork_child(dict(child)):
                            break
                        row = conn.execute(
                            "SELECT id, end_reason FROM sessions WHERE id = ? LIMIT 1",
                            (child["parent_session_id"],),
                        ).fetchone()
                        if not row or str(row[1] or "") != "compression" or str(row[0]) in seen:
                            break
                        current = str(row[0])
                        seen.add(current)
                        chain.append(current)
                    read = True
                finally:
                    conn.close()
        except Exception as exc:  # host DB drift or absence: no ancestry, session scope only
            logger.debug("LCM identity-anchor ancestry read failed: %s", exc)
            chain = []
        if read:  # T4: only a completed read is cached; a missing or failing state.db is asked again
            self._identity_anchor_chain_cache = (session_id, chain)
        return chain

    def _identity_anchor_carry_key(self) -> str:
        return f"{_CARRY_PREFIX}:{self._session_id}"

    def _identity_anchor_carry_ranges(self) -> list:
        """Durable anchor carry, revalidated on every load: a source must still be a verified ancestor,
        and only rows above the frontier (not yet covered) stay publishable."""
        if not identity_anchor_enabled() or not self._session_id:
            return []
        try:
            payload = self._store.read_metadata_json(self._identity_anchor_carry_key())
        except Exception:
            return []
        if not isinstance(payload, list) or not payload:
            return []
        chain, frontier = set(self._identity_anchor_chain()), int(self._last_compacted_store_id or 0)
        return [
            (str(source), max(int(start), frontier), int(end))
            for source, start, end in (item for item in payload if isinstance(item, list) and len(item) == 3)
            if str(source) in chain and int(end) > frontier
        ]

    def _register_identity_anchor_carry(self, rows) -> None:
        """R7: publish eligibility for parent rows this session replays: only verified ancestors' rows
        above the frontier, in ranges that never span a row that was not recognised. Durable first."""
        chain, frontier = set(self._identity_anchor_chain()), int(self._last_compacted_store_id or 0)
        wanted: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            owner, store_id = str(row.get("session_id") or ""), int(row["store_id"])
            if owner in chain and store_id > frontier:
                wanted[owner].add(store_id)
        if not wanted:
            return
        current = self._load_compression_carry_ranges()
        added = []
        for owner, ids in wanted.items():
            ids = {i for i in ids if not any(s == owner and a < i <= b for s, a, b in current)}
            if not ids:
                continue
            run = None
            for row in self._store.get_range(owner, start_id=min(ids), end_id=max(ids), limit=100000):
                store_id = int(row["store_id"])
                if store_id in ids:
                    run = (store_id - 1 if run is None else run[0], store_id)
                elif run is not None:
                    added.append((owner, *run))
                    run = None
            if run is not None:
                added.append((owner, *run))
        if added:
            try:
                stored = self._store.read_metadata_json(self._identity_anchor_carry_key())
            except Exception:
                stored = None
            previous = [tuple(item) for item in stored if isinstance(item, list)] if isinstance(stored, list) else []
            merged = self._coalesce_compression_carry_ranges(previous + added)
            self._store.write_metadata_json([self._identity_anchor_carry_key()], json.dumps([list(i) for i in merged]))
            logger.info("LCM identity-anchor carry from verified ancestors: session=%s ranges=%s", self._session_id, added)

    # -- R1-R5 pre-match -----------------------------------------------------

    def _identity_is_lcm_scaffold(self, message, *, verified: bool = False) -> bool:
        """R8: LCM's own summary/carrier rows keep their DAG-verified identity (``verified``: only a
        DAG-verified pure LCM scaffold; else any summary-shaped row, which the anchor leaves alone)."""
        if self._is_verified_replay_scaffold_message(message):
            return True
        text = text_content_for_pattern_matching(message.get("content")) or ""
        return not verified and bool(self._is_context_summary_content(text))

    def _identity_texts(self, row) -> set[str]:
        memo = self._identity_anchor_text_memo
        store_id = int(row.get("store_id") or 0)
        if store_id not in memo:
            forms = (self._stored_row_forms(row) if row.get("observed_at") is not None  # #821: an override form
                     else {self._message_replay_identity(row, stored_row=True)})  # needs a stamp-bound row
            memo[store_id] = {form[1] for form in forms if not _lossy(form)}
        return memo[store_id]

    def _identity_anchor_prematch(self, messages, identity_messages, cursor: int, audit_from: Optional[int] = None,
                                  *, accepted_prefix: int = 0) -> Dict[str, Any]:
        """Rows at or after ``cursor`` recognised as replays of stored occurrences, plus the R3
        remainders to store, the relations to record and the R5 backfills. ``audit_from``: the host
        changed its list before the cursor from there (a positional cursor no longer proves those rows
        stored): a stamped row -- or an unstamped user row (#633) -- there that no stored occurrence
        explains moves ``plan["cursor"]`` back. ``accepted_prefix`` is the compaction boundary
        already verified in this ingest call, even if the cached pre-compaction list differs."""
        accepted_prefix = max(0, min(accepted_prefix, cursor, len(messages)))
        plan: Dict[str, Any] = {"replayed": set(), "remainders": {}, "relations": [], "carry": [], "backfill": [],
                                "cursor": cursor, "accepted_prefix": accepted_prefix}
        self._identity_anchor_text_memo: dict[int, set[str]] = {}
        n = len(messages)
        start = cursor if audit_from is None else max(0, min(audit_from, cursor))
        rewritten = self._identity_anchor_rewritten(messages, cursor)
        if rewritten:  # R6: a rewritten host object is examined like a new row (a merge survivor, ...)
            start = min(start, min(rewritten))
        start = max(start, accepted_prefix)
        if not identity_anchor_enabled() or not self._session_id or start >= n:
            return plan
        stamps = {}
        for idx in range(n):
            observed_at = _normalize_observed_at(messages[idx].get("timestamp"))
            if observed_at is not None:
                stamps[idx] = observed_at
        wanted = {stamps[idx] for idx in range(start, n) if idx in stamps}
        if not wanted and start >= cursor:
            return plan
        # #633: nothing stamped from ``start`` on still leaves a changed prefix to audit (an unstamped row).
        chain = self._identity_anchor_chain()
        rows = self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *chain], sorted(wanted)
        ) if wanted else []
        self._load_host_rewrite_overrides(rows)
        plan["stamped_users"] = {int(row["store_id"]) for row in rows if row.get("role") == "user"}
        by_stamp: dict[float, list] = defaultdict(list)
        for row in rows:
            by_stamp[float(row["observed_at"])].append((row, self._stored_row_forms(row)))
        identities: dict[int, Optional[tuple]] = {}
        recovery_identities = {}
        proof = self._active_emission_proof()
        if proof and any(d.get("kind") == "recovery" for d in proof.get("emissions") or ()):
            projection, projected = self._occurrence_replay_identities(identity_messages, proof)
            recovery_identities = {i: identity for i, (entry, identity) in enumerate(zip(projection.entries, projected))
                                   if entry.kind in {"recovery", "recovery_base"}}
            plan["replayed"].update(i for i, identity in recovery_identities.items() if identity is None)

        def identity_at(idx: int) -> Optional[tuple]:
            if idx not in identities:
                message = identity_messages[idx]
                identity = None if self._identity_is_lcm_scaffold(message) else self._message_replay_identity(
                    message, strip_carrier=False
                )
                identity = recovery_identities.get(idx, identity)
                identities[idx] = None if identity is None or _lossy(identity) else identity
            return identities[idx]

        view_counts: dict = {}

        def view_count(identity) -> int:  # occurrences in the WHOLE host view (multiplicity evidence)
            if not view_counts:
                for i, message in enumerate(identity_messages):
                    key = identities.get(i) if i in identities else self._message_replay_identity(message, strip_carrier=False)
                    view_counts[key] = view_counts.get(key, 0) + 1
            return view_counts.get(identity, 0)

        def shown(idx: int, matched_too: bool = False) -> Counter:  # B-ID-1: the view's own occurrences no
            # stored row has matched yet (matched_too: every other occurrence in the view)
            return Counter((stamps.get(i), identity_at(i)) for i in range(n)  # unstamped: (None, form)
                           if i != idx and (matched_too or i not in matched) and identity_at(i) is not None)

        consumed: set[int] = set()
        matched: dict[int, list] = {}
        emitted = None  # #1000: the view's carriers LCM emitted, still where it emitted them; read once
        carriers = [d for d in (proof.get("emissions") or ()) if isinstance(d, dict) and d.get("kind") == "carrier"
                    ] if proof else []

        def emitted_carrier(idx: int) -> Optional[list]:  # #1000: a carrier LCM emitted around a stored user row
            nonlocal emitted
            if not carriers or identity_messages[idx].get("role") != "user":
                return None
            if emitted is None:
                emitted = self._identity_anchor_emitted_carriers(identity_messages, proof, carriers)
            return self._identity_anchor_emitted_carrier(identity_messages[idx], emitted.get(idx), chain)

        # R1: per key, the host view's occurrences consume the stored ones in order; the rest are new.
        forms_of = {id(r): forms for pairs in by_stamp.values() for r, forms in pairs}
        occurrences = [(idx, (stamps[idx], identity_at(idx))) for idx in sorted(stamps)
                       if stamps[idx] in wanted and identity_at(idx) is not None]
        for store_id, idx in _match_occurrences(
                rows, lambda r: {(float(r["observed_at"]), form) for form in forms_of[id(r)]}, occurrences).items():
            consumed.add(store_id)
            matched[idx] = [next(r for r, _forms in by_stamp[stamps[idx]] if int(r["store_id"]) == store_id)]
            if idx >= start:
                plan["replayed"].add(idx)
        for idx in range(start, n if wanted else start):  # nothing stamped: only the #633 audit below
            identity = identity_at(idx) if idx in stamps and idx not in plan["replayed"] else None
            if identity is not None and identity[0] == "user":
                group = emitted_carrier(idx)  # #1000: Hermes 0.21.5 stamps the adopted carrier with its own clock;
                if group is not None:  # the claim is the descriptor's and the stored text's, never the stamp's
                    self._identity_anchor_take(idx, group, consumed, matched, plan)
                    continue
                row = self._identity_anchor_ws_row(identity, stamps[idx], [r for r, _f in by_stamp[stamps[idx]]], consumed)
                if row is not None:  # R1-ws: the host persisted this occurrence without its edge whitespace
                    plan.setdefault("ws", []).append((row, identity_messages[idx]))
                    self._identity_anchor_take(idx, [row], consumed, matched, plan)
                    continue
                self._capture_merged_user_head(identity_messages[idx], by_stamp[stamps[idx]], consumed)
                self._identity_anchor_user_row(idx, identity, stamps[idx], by_stamp, consumed, matched, plan, view_count,
                                               shown)
            elif idx not in stamps and idx not in plan["replayed"]:  # D-D plan (ii): H1 merged LCM's carrier
                group = self._identity_anchor_carrier_group(identity_messages[idx], consumed)
                group = group if group is not None else emitted_carrier(idx)  # #1000: a survival fit's carrier
                if group is not None:
                    self._identity_anchor_take(idx, group, consumed, matched, plan)
        for idx in sorted(plan["replayed"]):  # a replayed survival-fit projection: its stored replies follow it
            if len(matched.get(idx, ())) == 1 and callable(getattr(self, "_survival_projection_followers", None)):
                for k, row in self._survival_projection_followers(identity_messages, idx, matched[idx][0], stamps):
                    if k in plan["replayed"] or int(row["store_id"]) in consumed:
                        break
                    self._identity_anchor_take(k, [row], consumed, matched, plan)
        for idx in range(cursor, n):  # a cold restart can have no positional prefix to audit
            if idx in plan["replayed"] or idx not in stamps or messages[idx].get("role") != "user":
                continue
            copies = [row for row, forms in by_stamp.get(stamps.get(idx - 1), ())
                      if int(row["store_id"]) not in consumed and identity_at(idx) in forms]
            carried = self._identity_anchor_carried_stamp(messages, idx, stamps, copies, matched.get(idx - 1, ()), chain,
                                                          distinct=_distinct_from_predecessor(identity_at, idx))
            carried = self._identity_anchor_durable_stamp(carried, matched.get(idx - 1, ()), stamps.get(idx - 1))
            if carried:
                self._identity_anchor_take(idx, carried, consumed, matched, plan)
            elif any(r["row"] is messages[idx] for r in getattr(self, "_identity_anchor_recoveries", ())):
                # No intermediate ingest: the rewritten recovery may be after the cursor already.
                from .reconcile import _proof_user_identity

                copies = [{"store_id": entry[2]} for entry in getattr(self, "_identity_anchor_recent", ())
                          if entry[0] in [str(self._session_id), *chain] and identity_at(idx) is not None
                          and _proof_user_identity(entry[1]) == _proof_user_identity(identity_at(idx))
                          and entry[2] not in consumed]
                recovery = self._identity_anchor_recovery_stamp(messages, idx, stamps, copies, chain)
                if recovery:
                    plan["relations"].append(("alt_stamp", stamps[idx], recovery, None))
                    self._identity_anchor_take(idx, recovery, consumed, matched, plan)
        if start < cursor:
            self._identity_anchor_audit(messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan,
                                        matched)
        self._identity_anchor_tool_segments(messages, plan["cursor"], plan["replayed"], matched, plan.get("positional", ()))
        for idx in list(plan["remainders"]):
            if idx in plan["replayed"]:
                del plan["remainders"][idx]
        plan["explained"] = plan["replayed"] - plan.get("positional", set())
        plan["replayed"] = {idx for idx in plan["replayed"] if idx >= plan["cursor"]}
        plan["carry"] = [row for idx in plan["replayed"] | set(plan["remainders"]) for row in matched.get(idx, ())]
        plan["carry"] += [row for row, _message in plan.get("ws", ())]  # R1-ws matches the audit proved too
        if plan["replayed"] and any(str(row.get("session_id")) in chain for row in plan["carry"]):
            # A rotation child resuming onto its ancestors' rows: LCM's own DAG-verified carrier heading
            # the list is compress() output, not a host message (R8).
            for idx in range(plan["cursor"], min(plan["replayed"])):
                if self._identity_is_lcm_scaffold(identity_messages[idx], verified=True):
                    plan["replayed"].add(idx)
        plan["matched"] = {**plan.get("matched", {}), **matched}  # read only by the v0.26.0 host-uid shadow
        return plan

    def _identity_anchor_audit(self, messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan,
                               matched) -> None:
        """Rows in ``[start, cursor)`` of a list the host changed before the cursor: a stamped host row,
        or an unstamped user row the host inserted there (#633: a /steer row), that
        no stored occurrence explains (key, witness, alias, or a stored copy of its content -- up to
        edge whitespace, with the same stamp or a canonical host-uid binding for a NULL stamp --
        in the tail of this session or a verified ancestor, each copy used once) and no ignore pattern drops was never stored. The cursor moves back
        to the first such row; every other row of that range stays a replay (today's positional proof)."""
        from .reconcile import _proof_user_identity

        span = max(64, 2 * (cursor - start))
        held: dict = defaultdict(list)
        chain = self._identity_anchor_chain()
        for session in [str(self._session_id), *chain]:
            for row in self._store.get_session_tail(session, limit=span):
                if int(row["store_id"]) not in consumed:
                    held[_proof_user_identity(self._message_replay_identity(row, stored_row=True))].append(row)
        bindings = None

        def bound_null(idx, row):
            # #1403: content (even unique content) cannot identify a NULL-stamp occurrence.
            # Read only existing canonical host bindings in this verified home/lineage.
            from .host_uid import _valid_uid

            nonlocal bindings
            uid = messages[idx].get("message_uid")
            if not _valid_uid(uid):
                return False
            if bindings is None:
                bindings = {}
                try:
                    lineage, _problem = self._host_uid_lineage_key()
                    if lineage is not None:
                        uids = {m.get("message_uid") for m in messages[start:cursor]
                                if _valid_uid(m.get("message_uid"))}
                        bindings = self._store.host_uid_bindings_for(lineage, uids, ("canonical",))
                except Exception as exc:
                    # Do not let the outer pre-match fallback restore an unproven cursor.
                    logger.warning("LCM audit uid proof unavailable (%s); retaining unproven rows", type(exc).__name__)
            return (int(row["store_id"]), "canonical") in bindings.get(uid, ())

        missed, deferred, audited = [], [], set()
        inserted = None
        for idx in range(start, cursor):
            identity = identity_at(idx)
            if idx not in stamps:  # #633: an unstamped user row the host inserted (a /steer), not a rewrite
                if identity is None or identity[0] != "user":
                    continue
                if inserted is None:
                    inserted = self._identity_anchor_inserted(messages, start, cursor)
                if idx in inserted:
                    deferred.append(idx)
                continue
            if idx in plan["remainders"]:  # a held head plus a new remainder: the remainder is unstored
                missed.append(idx)
                continue
            if (identity is None or idx in plan["replayed"] or identity[0] not in ("user", "assistant")
                    or identity_messages[idx].get("tool_calls")  # tool rows: the host rewrites them in place
                    or self._message_replay_identity(identity_messages[idx]) != identity  # carries LCM's carrier (R8)
                    or self._matches_ignore_message_patterns(messages[idx])):
                continue
            audited.add(idx)
            copies = [row for row in held.get(_proof_user_identity(identity), ()) if int(row["store_id"]) not in consumed]
            ws = self._identity_anchor_ws_row(identity, stamps[idx], copies, consumed)
            if ws is not None:  # R1-ws: the same occurrence (same stamp, edge whitespace only), recorded
                consumed.add(int(ws["store_id"]))
                plan.setdefault("matched", {})[idx] = [ws]  # host-uid shadow only
                plan.setdefault("ws", []).append((ws, identity_messages[idx]))
                continue
            previous = matched.get(idx - 1, plan.get("matched", {}).get(idx - 1, ()))
            carried = self._identity_anchor_carried_stamp(
                messages, idx, stamps, copies, previous, chain, distinct=_distinct_from_predecessor(identity_at, idx))
            carried = self._identity_anchor_durable_stamp(carried, previous, stamps.get(idx - 1))
            recovery_candidates = copies
            if not copies and any(r["row"] is messages[idx] for r in getattr(self, "_identity_anchor_recoveries", ())):
                # A recovery can drop more orphan followers than the audit tail holds.
                recovery_candidates = [{"store_id": entry[2]} for entry in getattr(self, "_identity_anchor_recent", ())
                                       if entry[0] in [str(self._session_id), *chain] and entry[2] not in consumed
                                       and _proof_user_identity(entry[1]) == _proof_user_identity(identity)]
            # #1328: equal content at a different known stamp is a different occurrence.
            # A NULL copy needs independent occurrence proof before consumption or unique backfill.
            copies = [row for row in copies if row.get("observed_at") == stamps[idx]
                      or row.get("observed_at") is None and bound_null(idx, row)]
            if not copies:
                copies = carried or self._identity_anchor_recovery_stamp(
                    messages, idx, stamps, recovery_candidates, chain)
                if copies and not carried:
                    plan["relations"].append(("alt_stamp", stamps[idx], copies, None))
            if copies:
                consumed.add(int(copies[0]["store_id"]))
                plan.setdefault("matched", {})[idx] = [copies[0]]  # host-uid shadow only
                if (len(copies) == 1 and copies[0].get("observed_at") is None
                        and self._message_replay_identity(copies[0], stored_row=True) == identity):
                    plan["backfill"].append((int(copies[0]["store_id"]), stamps[idx]))
                continue
            missed.append(idx)
        if deferred:  # #633: an inserted row is explained only by a stored copy no other row of the view holds
            reserved = Counter(_proof_user_identity(identity_at(i)) for i in range(cursor)
                               if i not in inserted and i not in plan["replayed"] and i not in audited
                               and identity_at(i) is not None)
            for idx in deferred:
                identity = identity_at(idx)
                if (idx in plan["replayed"] or identity_messages[idx].get("tool_calls")
                        or self._message_replay_identity(identity_messages[idx]) != identity  # carries LCM's carrier (R8)
                        or self._matches_ignore_message_patterns(messages[idx])):
                    continue
                key = _proof_user_identity(identity)
                copies = [row for row in held.get(key, ()) if int(row["store_id"]) not in consumed
                          and (row.get("observed_at") is not None or bound_null(idx, row))]
                if len(copies) > reserved[key]:
                    consumed.add(int(copies[-1]["store_id"]))
                    plan.setdefault("matched", {})[idx] = [copies[-1]]  # host-uid shadow only
                    continue
                missed.append(idx)
        if missed:
            plan["cursor"] = min(missed)
            plan["positional"] = {idx for idx in range(min(missed), cursor) if idx not in missed}
            plan["replayed"].update(plan["positional"])
            logger.info("LCM identity-anchor: host changed its list before the cursor; %d unstored rows from %d: session=%s",
                        len(missed), min(missed), self._session_id)

    def _identity_anchor_record_recovery(self, result, emissions) -> None:
        """#1505: keep the actual published recovery object, never a displaced-occurrence mapping."""
        entries = [d for d in emissions if d.get("kind") == "recovery"]
        if len(entries) != 1:
            return
        idx = entries[0]["output_occurrence"]["index"]
        row = result[idx]
        receipt = {"row": row, "content": row["content"], "index": idx,
                   "session": str(self._session_id), "keys": self._identity_anchor_occurrence_keys(row),
                   "store": self._store, "home": str(self._hermes_home), "conversation": str(self._conversation_id),
                   "generation": getattr(self, "_identity_anchor_user_generation", 0), "stamp": None, "used": False,
                   "max_store_id": self._store.max_store_id()}  # the recovered turn was stored before its recovery
        receipts = list(getattr(self, "_identity_anchor_recoveries", ()))
        for previous in receipts:
            if (previous["row"] is row or previous["store"] is self._store
                    and previous["generation"] == receipt["generation"]):
                previous["blocked"] = receipt["blocked"] = True
        self._identity_anchor_recoveries = (receipts + [receipt])[-4:]
        self._identity_anchor_recovery = receipt

    @staticmethod
    def _identity_anchor_occurrence_keys(row) -> dict:
        # _row_id is the host's insertion address, reassigned during compaction, not an event id.
        return {key: value for key, value in row.items()
                if (key.endswith("_id") or key.endswith("_uid")) and key != "_row_id"}

    def _identity_anchor_observe_recovery(self) -> None:
        """Capture the emitted object's own host stamp while its generated bytes still survive."""
        for receipt in getattr(self, "_identity_anchor_recoveries", ()):
            if receipt.get("blocked") or receipt["used"] or receipt["stamp"] is not None:
                continue
            row = receipt["row"]
            if (row.get("role") == "user" and row.get("content") == receipt["content"]
                    and self._identity_anchor_occurrence_keys(row) == receipt["keys"]):
                receipt["stamp"] = _normalize_observed_at(row.get("timestamp"))

    def _identity_anchor_recovery_stamp(self, messages, idx, stamps, copies, chain) -> list:
        """One live positional replacement of LCM's own recovery row; every uncertainty stores.

        Equal stamps, equal normalized text, alternate stamps and copied dicts cannot stand in for
        the exact emitted object. Only the stamp observed while its generated bytes survived proves the
        rewrite; a stamp first seen after it could be a distinct resend, so it stores. Persist the alias
        through the caller's plan; cold replays use R1.
        """
        from .reconcile import _proof_user_identity
        from .store import _normalize_conversation_id_value

        receipts = getattr(self, "_identity_anchor_recoveries", ())
        own = [r for r in receipts if r["row"] is messages[idx]]
        if len(own) != 1:
            return []
        receipt, stamp = own[0], stamps.get(idx)
        if (receipt.get("blocked") or receipt["used"] or stamp is None
                or receipt["store"] is not self._store or receipt["home"] != str(self._hermes_home)
                or receipt["conversation"] != str(self._conversation_id)
                or idx != receipt["index"] or messages[idx].get("role") != "user"
                or self._identity_anchor_occurrence_keys(messages[idx]) != receipt["keys"]
                or receipt["stamp"] is None or stamp != receipt["stamp"] or len(copies) != 1):
            return []
        scope = [str(self._session_id), *chain]
        if receipt["session"] not in scope:
            return []
        store_id = int(copies[0]["store_id"])
        if any(r.get("store_id") == store_id for r in receipts) or store_id > receipt["max_store_id"]:
            return []
        row = self._store.get_batch([store_id]).get(store_id)
        conversation = _normalize_conversation_id_value(self._conversation_id)
        if (not row or row.get("session_id") not in scope or row.get("role") != "user"
                or _normalize_conversation_id_value(row.get("conversation_id")) != conversation
                or row.get("observed_at") is None or row["observed_at"] == stamp
                or _proof_user_identity(self._message_replay_identity(row, stored_row=True))
                != _proof_user_identity(self._message_replay_identity(messages[idx], strip_carrier=False))):
            return []
        for session in scope:
            followers = self._store.get_range(session, start_id=store_id + 1, limit=257,
                                              conversation_id=conversation, include_blank_conversation=True)
            if len(followers) == 257 or any(r.get("role") not in {"assistant", "tool"} for r in followers):
                return []
        receipt.update(used=True, store_id=store_id)
        return [row]

    @staticmethod
    def _identity_anchor_carried_stamp(messages, idx, stamps, copies, previous, chain, *, distinct=False) -> list:
        """#1501: exact adjacent occurrence proof for the host's carried predecessor stamp.

        Candidates already have today's content identity, verified scope and single-consumption filter.
        A tolerated copy keeps its stamp, so it cannot independently prove the next carried copy.
        The host folds consecutive user content only: the predecessor and its match must be user rows,
        the stamp must move forward, and an identical predecessor text gives no positional proof
        (``distinct``: the caller compares both rows in the matching identity form).
        """
        if (idx < 1 or not distinct or not chain or messages[idx].get("message_uid") or messages[idx].get("role") != "user"
                or len(previous) != 1 or stamps.get(idx - 1) is None or stamps.get(idx) is None
                or stamps[idx] <= stamps[idx - 1] or messages[idx - 1].get("role") != "user"
                or messages[idx - 1].get("content") == messages[idx].get("content")):
            return []
        prior = previous[0]
        if prior.get("role") != "user" or prior.get("observed_at") != stamps[idx - 1]:
            return []
        return [row for row in copies if int(row["store_id"]) == int(prior["store_id"]) + 1
                and row.get("observed_at") == stamps[idx - 1]]

    def _identity_anchor_durable_stamp(self, rows, previous, stamp) -> list:
        """#1501: the carried candidates whose own stored row holds ``stamp``, when the proven predecessor's
        own stored row holds it too, with store-time positional-replacement provenance. A recorded
        alternate stamp (``find_rows_by_observed_at``'s R5 projection) proves another occurrence, never
        a carried stamp, on either row."""
        if not rows or len(previous) != 1:
            return []
        prior = int(previous[0]["store_id"])
        stored = self._store.get_batch(sorted({prior, *(int(row["store_id"]) for row in rows)}))
        if prior not in stored or stored[prior].get("observed_at") != stamp:
            return []
        marked = {int(rel["store_id"]) for rel in self._store.get_message_relations(
            [int(row["store_id"]) for row in rows], "carried_stamp")
            if rel["related_store_id"] == prior and rel["observed_at"] == stamp}
        return [row for row in rows if (stored.get(int(row["store_id"])) or {}).get("observed_at") == stamp
                and int(row["store_id"]) in marked]

    def _identity_anchor_mark_carried(self, messages, stored_at, plan) -> None:
        """Record a new user row taking its absent predecessor's slot at that predecessor's stamp.

        The preceding list row must still be the stored predecessor's predecessor, in this session
        or its verified compression chain. Absence alone (trim, compaction, /new) is not proof.
        Read stored rows right after R is stored; an R3 remainder is stored with a NULL stamp. Only a
        predecessor the prematch's stamp lookup returned can qualify, so other rows cost no read.
        """
        stamped_users = plan.get("stamped_users", ())
        candidates = {idx: store_id for idx, store_id in stored_at.items()
                      if idx > 0 and messages[idx].get("role") == "user" and not messages[idx].get("message_uid")
                      and store_id - 1 in stamped_users}
        if not candidates:
            return
        scope = [str(self._session_id), *self._identity_anchor_chain()]
        stored = self._store.get_batch(sorted({sid for store_id in candidates.values() for sid in (store_id - 1, store_id)}))
        stamps = [_normalize_observed_at(message.get("timestamp")) for message in messages]
        identities = [self._message_replay_identity(message, strip_carrier=False) for message in messages]
        groups = []
        for idx, store_id in candidates.items():
            row, prior = stored.get(store_id) or {}, stored.get(store_id - 1) or {}
            stamp = row.get("observed_at")
            if (row.get("role") != "user" or stamp is None or prior.get("role") != "user"
                    or prior.get("session_id") not in scope or prior.get("observed_at") != stamp):
                continue
            if any(at == stamp and self._anchor_row_admits(identity, prior)
                   for at, identity in zip(stamps, identities)):
                continue
            predecessors = [r for session in scope
                            for r in self._store.get_session_rows_through(session, store_id - 2, 1)]
            before = max(predecessors, key=lambda r: int(r["store_id"]), default=None)
            if (before is None or before.get("observed_at") is None or stamps[idx - 1] != before.get("observed_at")
                    or self._identity_is_lcm_scaffold(messages[idx - 1])
                    or not self._anchor_row_admits(identities[idx - 1], before)):
                continue
            groups.append([(store_id, "carried_stamp", store_id - 1, None, stamp)])
        if groups:
            self._store.add_message_relations(groups)

    def _identity_anchor_inserted(self, messages, start: int, cursor: int) -> set:
        """#633: indexes in ``[start, cursor)`` the host inserted into the last list LCM ingested (Hermes
        0.21.2+ inserts a /steer row after the newest tool result), by an identity diff of the two lists
        from ``start``. A row that replaces, rewrites or re-merges an ingested one is not inserted. Over
        the diff budget nothing counts as inserted (the v0.24.6 behaviour: the row is not audited)."""
        before = getattr(self, "_last_active_replay_source_identities", None)
        if not before or max(0, len(before) - start) * (len(messages) - start) > _INSERT_DIFF_BUDGET:
            return set()
        now = [self._message_replay_identity(message, strip_carrier=False) for message in messages[start:]]
        opcodes = SequenceMatcher(None, list(before[start:]), now, autojunk=False).get_opcodes()
        return {start + j for tag, _i1, _i2, j1, j2 in opcodes if tag == "insert" for j in range(j1, j2)
                if start + j < cursor}

    def _capture_merged_user_head(self, message, stamped, consumed) -> None:
        """#821: an occurrence-bound host merge marker records the head before R2/R3 matching."""
        from .reconcile import _proof_user_identity

        marker, content = message.get("_merged_turn_prefix"), message.get("content")
        if not isinstance(marker, str) or not isinstance(content, str):
            return
        upstream = content == marker or content.startswith(marker + "\n\n")
        r34 = marker.endswith("\n\n") and content.startswith(marker)
        if upstream == r34:  # ambiguous or absent host form: nothing inferred
            return
        donors = [r for r, _forms in stamped if r.get("role") == "user" and int(r["store_id"]) not in consumed]
        if len(donors) != 1:
            return
        row = donors[0]
        head = {**message, "content": marker[:-2] if r34 else marker}
        live, stored = self._message_replay_identity(head, strip_carrier=False), self._message_replay_identity(row, stored_row=True)
        if _lossy(live) or _lossy(stored) or _proof_user_identity(live) != _proof_user_identity(stored):
            return
        override = self._host_rewrite_override_content(row)
        if override is not None and override != head["content"]:
            return
        store_id = int(row["store_id"])
        try:
            self._record_ws_host_rewrite(row, head)  # existing payload, protection and skip_unchanged writer
        except Exception as exc:
            logger.warning("LCM merge-head capture for store_id %s failed (%s)", store_id, type(exc).__name__)
            return
        self._identity_anchor_text_memo.pop(store_id, None)

    def _identity_anchor_user_row(self, idx, identity, stamp, by_stamp, consumed, matched, plan, view_count, shown) -> None:
        """R2/R3/R5 for one unmatched user row: a recorded witness, an exact unique decomposition,
        the H1 unstable current-turn stamp, else a held head plus a new remainder. Else: new."""
        content = identity[1]
        stamped = [row for row, _forms in by_stamp.get(stamp, ()) if row.get("role") == "user"]
        donors = [row for row in stamped if int(row["store_id"]) not in consumed]
        # R2 witness: a composite LCM itself saw at this stamp, or its rewritten survivor (U alone).
        for group in self._identity_anchor_witnesses(stamped, stamp):
            texts = [self._identity_texts(row) for row in group]
            if self._identity_anchor_group_matches(content, group):
                view = group
            else:
                view = [row for row, text in zip(group, texts) if content in text and row.get("observed_at") != stamp]
                if len(view) != 1 or sum(content in forms for forms in texts) != 1:
                    continue
            if all(int(row["store_id"]) not in consumed for row in view):
                return self._identity_anchor_take(idx, view, consumed, matched, plan)
            # #563: the witnessed composite (or its survivor) is a VIEW of stored occurrences even where
            # the host view also carries them on their own (H2 re-flushes the survivor as a new row):
            # a replay, never a new occurrence; each witnessed form explains one view occurrence.
            form = (tuple(int(row["store_id"]) for row in group), tuple(int(row["store_id"]) for row in view))
            if form not in plan.setdefault("witnessed", set()):
                plan["witnessed"].add(form)
                matched[idx] = list(view)
                plan["replayed"].add(idx)
                return None
        if not donors:
            return self._identity_anchor_constituent_copy(idx, identity, stamp, consumed, matched, plan)
        if "\n\n" not in content:
            return
        # B-ID-1: a row the host view shows as its own occurrence is reserved by it, never a constituent.
        pool = self._identity_anchor_pool(donors, consumed)
        reserved = self._identity_anchor_reserved(pool, shown(idx))
        pool = [row for row in pool if int(row["store_id"]) not in reserved]
        donors = [row for row in donors if int(row["store_id"]) not in reserved]
        pool = self._identity_anchor_eligible(pool, donors, content, stamp)
        pool, runs = self._identity_anchor_scope_unstamped(pool, donors)
        texts = {text for row in pool for text in self._identity_texts(row)}
        donor_texts = {text for row in donors for text in self._identity_texts(row)}
        group, ambiguous = self._identity_anchor_compose(content, texts, pool, donors, consumed, runs=runs)
        if group is not None:
            plan["relations"].append(("composite", stamp, group, None))
            return self._identity_anchor_take(idx, group, consumed, matched, plan)
        if ambiguous:
            return  # several decompositions fit: the composite stays a pending (new) occurrence
        # R5: H1 keeps the absorbing row's stamp on a survivor the persist override rewrote to a row this
        # process stored in the current turn under the host's other stamp (in this session, or in the
        # verified ancestor a mid-turn rotation just closed): that ONE occurrence, aliased.
        scope = {self._session_id, *self._identity_anchor_chain()}
        recent = [entry for entry in getattr(self, "_identity_anchor_recent", ())
                  if entry[0] in scope and entry[1] == identity and entry[3] not in (None, stamp)]
        if (len(recent) == 1 and recent[0][2] not in consumed and view_count(identity) == 1
                and any(content.startswith(text + "\n\n") for text in donor_texts)):
            row = self._store.get_batch([recent[0][2]]).get(recent[0][2])
            # B-ID-2: the cached id names this occurrence only while the row still is it (same store, scope).
            if (row is not None and row.get("role") == "user" and str(row.get("session_id") or "") in scope
                    and recent[0][1] in self._stored_row_forms(row)):
                plan["relations"].append(("alt_stamp", stamp, [row], None))
                return self._identity_anchor_take(idx, [row], consumed, matched, plan)
        # R3: held constituents then a new remainder, stored once with its own stamp unknown.
        partials = [(parts, rest) for parts, rest in _decompositions(content, texts, partial=True) or ()
                    if rest and donor_texts & set(parts) and rest not in texts]
        longest = max((len(parts) for parts, _rest in partials), default=0)
        partials = [(parts, rest) for parts, rest in partials if len(parts) == longest]  # every held part accounted
        if len(partials) == 1:
            parts, rest = partials[0]
            # The host kept the absorbed turn beside its in-place merge, in this same view: the remainder's own
            # occurrence, stamped AFTER the held head, is here (stored already or not), so it is not new. No cut;
            # the composite is stored whole, as before #845. An older row with the same text is another turn
            # (B-ID-1) and does not count.
            if any(form[0] == "user" and at is not None and stamp is not None and at > stamp
                   and form[1].strip() == rest.strip() for at, form in shown(idx, matched_too=True)):
                return
            group = self._identity_anchor_assign(parts, pool, donors, consumed, runs=runs)
            if group is not None:
                consumed.update(int(row["store_id"]) for row in group)
                matched[idx] = group
                plan["remainders"][idx] = (rest, stamp, group, parts)

    def _identity_anchor_scope_unstamped(self, pool, donors) -> tuple:
        """#851: ``(pool, runs)``: an unstamped row stays only inside some donor's run; ``runs`` later binds it
        to the donor its group actually uses (``_identity_anchor_assign``). No unstamped row: no read."""
        if all(row.get("observed_at") is not None for row in pool):
            return pool, {}
        runs = self._identity_anchor_runs(donors)
        return [row for row in pool if row.get("observed_at") is not None
                or any(self._identity_anchor_in_run(row, donor, runs) for donor in donors)], runs

    def _identity_anchor_runs(self, donors) -> dict:
        """#851: store id -> run for the stored rows near each donor, one read per merged window. A run is a
        stretch of a session's user rows of one conversation (a blank legacy id joins any) with no other stored
        row between them: the host merges only back-to-back user messages (#583)."""
        windows: dict[str, list] = defaultdict(list)
        for donor in donors:
            store_id = int(donor["store_id"])
            windows[str(donor["session_id"])].append([max(0, store_id - _POOL_WINDOW), store_id + _POOL_WINDOW])
        runs: dict[int, tuple] = {}
        for session, spans in windows.items():
            merged: list = []
            for lo, hi in sorted(spans):
                if merged and lo <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                run = conversation = None
                for row in self._store.get_range(session, start_id=lo, end_id=hi, limit=hi - lo + 1):
                    if row.get("role") != "user":
                        run = conversation = None
                        continue
                    own = str(row.get("conversation_id") or "").strip() or None
                    if run is None or own and conversation and own != conversation:
                        run, conversation = (session, int(row["store_id"])), None
                    conversation = conversation or own
                    runs[int(row["store_id"])] = run
        return runs

    @staticmethod
    def _identity_anchor_in_run(row, donor, runs) -> bool:
        """#851: an unstamped constituent is proven only inside the run of the donor it is composed with."""
        run = runs.get(int(row["store_id"]))
        return (run is not None and run == runs.get(int(donor["store_id"]))
                and abs(int(row["store_id"]) - int(donor["store_id"])) <= _POOL_WINDOW)

    def _identity_anchor_ws_row(self, identity, stamp, rows, consumed) -> Optional[dict]:
        """R1-ws (D-D'): the first unconsumed stored user row at the SAME host stamp whose content differs from
        the view's by edge whitespace only (H1 persists ``prompt.strip()``); stripped texts equal, non-empty."""
        text = identity[1].strip()
        if identity[0] != "user" or not text or stamp is None:
            return None
        for row in rows:
            stored = self._message_replay_identity(row, stored_row=True)
            if (row.get("role") == "user" and int(row["store_id"]) not in consumed
                    and _normalize_observed_at(row.get("observed_at")) == stamp and stored != identity
                    and stored[1].strip() == text and tuple(stored[2:]) == tuple(identity[2:])):
                return row
        return None

    def _identity_anchor_carrier_group(self, message, consumed) -> Optional[list]:
        """D-D plan (ii): H1 merged LCM's DAG-verified carrier (R8) with user rows: the remainder's one exact
        R2 decomposition into the stored user run right after the carrier's coverage end, else None (bytes)."""
        if (message.get("role") != "user" or message.get("tool_calls")
                or self._generated_context_carrier_remainder(message) is None):
            return None
        rest = self._message_replay_identity(message)[1]  # the carrier stripped (DAG-verified, #483)
        parts = self._verified_lcm_summary_prefix(normalize_content_value(message.get("content")) or "")[1]
        covered = self._dag.coverage_end(parts) if parts and "\n\n" in rest else None
        if not covered:
            return None
        rows = sorted((row for session in [str(self._session_id), *self._identity_anchor_chain()]
                       for row in self._store.get_range(session, start_id=covered + 1, end_id=covered + _POOL_WINDOW,
                                                        limit=_POOL_WINDOW)), key=lambda row: int(row["store_id"]))
        run = []
        for row in rows:  # the host merged consecutive user rows: the run ends at the first other row
            if row.get("role") != "user" or int(row["store_id"]) in consumed:
                break
            run.append(row)
        self._load_host_rewrite_overrides(run)
        forms = [{form[1] for form in self._stored_row_forms(row)} for row in run]
        found = _decompositions(rest, set().union(*forms), partial=False) if run else None
        full = [parts for parts, remainder in found or () if not remainder]
        if len(full) != 1 or len(full[0]) > len(run) or any(text not in forms[i] for i, text in enumerate(full[0])):
            return None
        return run[:len(full[0])]

    def _identity_anchor_emitted_carriers(self, identity_messages, proof, carriers) -> dict:
        """#1000: ``{idx: entry}`` for each carrier descriptor of the active proof that the view still holds as the
        occurrence LCM emitted: projected at its own recorded output index (continuity), and not retired. A
        prefix and ordinal alone also bind a later byte-equal row (a NEW turn); the index does not. A carrier
        once held this way and now absent from that index is retired: no later pass binds it (process-local)."""
        from .reconcile import _carrier_emission_key, _project_emitted_occurrences

        held = {}
        for idx, entry in enumerate(_project_emitted_occurrences(identity_messages, proof=proof).entries):
            if entry.kind == "carrier" and entry.output_index == idx:
                held[_carrier_emission_key(entry)] = (idx, entry)
        seen, retired = self._emitted_carrier_ledger()
        out = {}
        for descriptor in carriers:
            key = _carrier_emission_key(descriptor)
            if key in retired:
                continue
            if key in held:
                seen[key] = None
                out[held[key][0]] = held[key][1]
            elif key in seen:
                retired[key] = None
        for ledger in (seen, retired):  # bounded: the oldest keys go first
            while len(ledger) > _CARRIER_LEDGER_CAP:
                del ledger[next(iter(ledger))]
        return out

    def _emitted_carrier_ledger(self) -> tuple[dict, dict]:
        """#1000: carrier keys held at their emitted index (seen), and those since absent (retired)."""
        if getattr(self, "_emitted_carriers_seen", None) is None:
            self._emitted_carriers_seen, self._emitted_carriers_retired = {}, {}
        return self._emitted_carriers_seen, self._emitted_carriers_retired

    def _identity_anchor_emitted_carrier(self, message, entry, chain) -> Optional[list]:
        """#1000: ``[U]`` when ``entry`` (the view's occurrence LCM emitted, see above) is a carrier formed around
        its stored user row U (``retained_source``) and the carrier's remainder is exactly U's stored text: a
        replay of U, generated content whether or not another occurrence in the view already matched U. U must
        be a user row of this session or a verified ancestor, in the bound conversation. Any failed check
        returns None and the row is stored, as before (a host row merged behind the carrier makes the
        remainder longer than U)."""
        source = entry.retained_source if entry is not None and entry.kind == "carrier" else None
        store_id = source.get("store_id") if source is not None else None
        if type(store_id) is not int or message.get("role") != "user" or message.get("tool_calls"):
            return None
        row = self._store.get_batch([store_id]).get(store_id)
        if (row is None or row.get("role") != "user" or int(row["store_id"]) != store_id
                or str(row.get("session_id") or "") not in {str(self._session_id), *chain}
                or str(row.get("conversation_id") or "") != str(self._conversation_id or "")):
            return None
        self._load_host_rewrite_overrides([row])
        return [row] if entry.effective_identity[1] in self._identity_texts(row) else None

    def _identity_anchor_constituent_copy(self, idx, identity, stamp, consumed, matched, plan) -> None:
        """R3 + R5: a later timestamped copy of a remainder U whose own stamp was unknown: the host view
        carries it right after the occurrence of its composite's stamp donor R (the recorded relation
        orders R before U). That ONE NULL-stamped U is this occurrence; its stamp is backfilled."""
        previous = max((i for i in matched if i < idx), default=None)
        if previous is None or len(matched[previous]) != 1:
            return
        head = matched[previous][0]
        for group in self._identity_anchor_witnesses([head], head.get("observed_at")):
            members = [row for row in group[group.index(next(r for r in group if int(r["store_id"]) == int(head["store_id"]))) + 1:]
                       if row.get("observed_at") is None and int(row["store_id"]) not in consumed
                       and not _lossy(identity) and identity in self._stored_row_forms(row)]
            if len(members) == 1:
                plan["backfill"].append((int(members[0]["store_id"]), stamp))
                return self._identity_anchor_take(idx, members, consumed, matched, plan)

    def _identity_anchor_compose(self, content, texts, pool, donors, consumed, runs=None) -> tuple:
        """R2 form (i): ``(group, ambiguous)``; ``group`` is the one exact, unique, ordered decomposition
        of ``content`` into stored occurrences (a stamp donor among them), each used once."""
        donor_texts = {text for row in donors for text in self._identity_texts(row)}
        found = _decompositions(content, texts, partial=False)
        if found is None:  # T3: search budget spent: ambiguous, the composite is stored whole
            return None, True
        full = [parts for parts, rest in found if not rest and donor_texts & set(parts)]
        group = self._identity_anchor_assign(full[0], pool, donors, consumed, runs=runs) if len(full) == 1 else None
        return group, len(full) > 1 or bool(full) and group is None

    def _identity_anchor_group_matches(self, content, group) -> bool:
        """A witnessed composite in one unique ordered choice of admissible constituent forms."""
        forms = [self._identity_texts(row) for row in group]
        raw = [self._message_replay_identity(row, stored_row=True)[1] for row in group]
        if content == "\n\n".join(raw) and all(text in forms[i] for i, text in enumerate(raw)):
            return True  # keep the existing occurrence-bound raw witness, including repeated texts
        found = _decompositions(content, set().union(*forms), partial=False) if forms else None
        full = [parts for parts, rest in found or () if not rest]
        return (len(full) == 1 and len(full[0]) == len(group)
                and all(text in forms[i] and sum(text in f for f in forms) == 1
                        for i, text in enumerate(full[0])))

    def _identity_anchor_eligible(self, pool, donors, content, stamp) -> list:
        """#821: older rows replay only as one whole recorded composite behind the donor head."""
        newer = [row for row in pool if row.get("observed_at") is None or float(row["observed_at"]) >= stamp]
        older = {int(row["store_id"]): row for row in pool if row not in newer}
        tails = {content[len(text) + 2:] for row in donors for text in self._identity_texts(row)
                 if content.startswith(text + "\n\n") and "\n\n" in content[len(text) + 2:]}
        if not older or not tails:
            return newer
        groups = {tuple(int(row["store_id"]) for row in group): group
                  for group in self._identity_anchor_witnesses(list(older.values()), stamp, older=True)
                  if len(group) >= 2 and all(int(row["store_id"]) in older for row in group)
                  and any(self._identity_anchor_group_matches(tail, group) for tail in tails)}
        return newer + next(iter(groups.values())) if len(groups) == 1 else newer

    def _identity_anchor_take(self, idx, rows, consumed, matched, plan) -> None:
        consumed.update(int(row["store_id"]) for row in rows)
        matched[idx] = list(rows)
        plan["replayed"].add(idx)

    def _identity_anchor_witnesses(self, donors, stamp, *, older=False) -> list:
        """Recorded composite groups at ``stamp`` (or older), headed by a donor, in order."""
        groups: dict[tuple, list] = defaultdict(list)
        for rel in self._store.get_message_relations([int(row["store_id"]) for row in donors], "composite"):
            eligible = (rel["observed_at"] is not None and float(rel["observed_at"]) < stamp) if older else rel["observed_at"] == stamp
            if eligible and rel["related_store_id"] is not None:
                groups[(rel["store_id"], rel["created_at"])].append((int(rel["ordinal"] or 0), int(rel["related_store_id"])))
        out = []
        for members in groups.values():
            ids = [store_id for _ordinal, store_id in sorted(members)]
            found = self._store.get_batch(ids)
            if len(found) == len(ids):
                out.append([found[store_id] for store_id in ids])
        return out

    def _identity_anchor_pool(self, donors, consumed) -> list:
        """Candidate constituents: unconsumed user rows of the lineage near each stamp donor, plus the
        bound session's recent rows (bounded). The caller drops rows the host view shows as their own
        occurrences (B-ID-1) before it decomposes."""
        rows: dict[int, dict] = {}
        for donor in donors:
            store_id = int(donor["store_id"])
            for row in self._store.get_range(str(donor["session_id"]), start_id=max(0, store_id - _POOL_WINDOW),
                                             end_id=store_id + _POOL_WINDOW, limit=2 * _POOL_WINDOW + 1):
                rows[int(row["store_id"])] = row
        for row in self._store.get_session_tail(str(self._session_id), limit=64):
            rows[int(row["store_id"])] = row
        return [row for store_id, row in sorted(rows.items())
                if row.get("role") == "user" and store_id not in consumed]

    def _identity_anchor_reserved(self, pool, shown: Counter) -> set:
        """B-ID-1: store ids of ``pool`` rows the view's ``shown`` occurrences ``{(stamp, form): n}`` hold as their
        own (``_reserve_shown``), a stamped row also under its recorded R5 alias stamps (B-ID-3)."""
        aliases: dict = defaultdict(set)
        for rel in self._store.get_message_relations([int(row["store_id"]) for row in pool], "alt_stamp"):
            if _normalize_observed_at(rel["observed_at"]) is not None:
                aliases[int(rel["store_id"])].add(_normalize_observed_at(rel["observed_at"]))
        return _reserve_shown(pool, self._stored_row_forms, list(enumerate(shown.elements())), aliases) if pool else set()

    def _identity_anchor_assign(self, parts, pool, donors, consumed, runs=None) -> Optional[list]:
        """Bind each part to one stored occurrence (a donor first), each used once;
        an override collision cannot choose between rows whose raw forms differ from the part.
        ``pool`` holds no row the host view shows as its own occurrence (B-ID-1).
        #851: an unstamped row binds only inside the run of a donor in the same group: one attempt per donor;
        distinct groups from different donors are ambiguous (None)."""
        if not runs:
            return self._identity_anchor_bind(parts, pool, donors, consumed)
        found: dict[tuple, list] = {}
        for donor in donors:
            scoped = [row for row in pool
                      if row.get("observed_at") is not None or self._identity_anchor_in_run(row, donor, runs)]
            group = self._identity_anchor_bind(parts, scoped, donors, consumed)
            if group is not None and int(donor["store_id"]) in {int(row["store_id"]) for row in group}:
                found[tuple(int(row["store_id"]) for row in group)] = group
        return next(iter(found.values())) if len(found) == 1 else None

    def _identity_anchor_bind(self, parts, pool, donors, consumed) -> Optional[list]:
        taken: set[int] = set(consumed)
        donor_ids = {int(row["store_id"]) for row in donors}
        group = []
        for text in parts:
            options = sorted((row for row in pool if text in self._identity_texts(row)
                              and int(row["store_id"]) not in taken),
                             key=lambda row: (int(row["store_id"]) not in donor_ids, int(row["store_id"])))
            if not options or len(options) > 1 and any(
                self._message_replay_identity(row, stored_row=True)[1] != text for row in options
            ):
                return None
            taken.add(int(options[0]["store_id"]))
            group.append(options[0])
        return group

    def _identity_anchor_tool_segments(self, messages, cursor: int, hits: set, matched, proven=()) -> None:
        """An assistant tool-call segment is replayed whole or not at all, except a stored segment whose
        only new rows are trailing results resuming in place on its session's newest row. ``proven``
        rows (the audited prefix's positional replays) stay replayed."""
        index, n = max(cursor, 0), len(messages)
        while index < n:
            if str(messages[index].get("role") or "") == "assistant" and messages[index].get("tool_calls"):
                end = index + 1
                while end < n and str(messages[end].get("role") or "") == "tool":
                    end += 1
                held = [k for k in range(index, end) if k in hits]
                resumes = bool(held) and held == list(range(index, index + len(held))) and len(held) < end - index
                if resumes:
                    last = max((row for k in held for row in matched.get(k, ())), key=lambda r: int(r["store_id"]),
                               default=None)
                    tail = self._store.get_session_tail(str(last["session_id"]), limit=1) if last else None
                    resumes = bool(tail) and int(tail[-1]["store_id"]) == int(last["store_id"])
                if len(held) != end - index and not resumes:
                    hits.difference_update(k for k in range(index, end) if k not in proven)
                index = end
            else:
                index += 1

    def _identity_anchor_backfill_prefix(self, messages, identity_messages, cursor: int) -> list:
        """R5 NULL backfill for a reconciled or same-call accepted compaction prefix: a stamped row whose occurrence-bound mapped row
        (#488 mapper) has ``observed_at`` NULL and the exact identity, when no other mapped NULL row shares
        that identity and no stored row already holds it at that stamp."""
        pairs = [(idx, _normalize_observed_at(messages[idx].get("timestamp"))) for idx in range(min(cursor, len(messages)))]
        pairs = [(idx, stamp) for idx, stamp in pairs if stamp is not None]
        if not identity_anchor_enabled() or not pairs:
            return []
        saved = getattr(self, "_current_compress_placeholder_identity_counts", None)
        try:
            mapping = self._get_store_id_map_for_messages(messages[:cursor])
        finally:
            self._current_compress_placeholder_identity_counts = saved
        rows = self._store.get_batch(sorted({mapping[id(messages[i])] for i, _s in pairs if id(messages[i]) in mapping}))
        null = {store_id: row for store_id, row in rows.items() if row.get("observed_at") is None}
        if not null:
            return []
        identity_of = {store_id: self._message_replay_identity(row, stored_row=True) for store_id, row in null.items()}
        counts = defaultdict(int)
        for identity in identity_of.values():
            counts[identity] += 1
        anchored = self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *self._identity_anchor_chain()],
            [stamp for _idx, stamp in pairs],
        )
        held = {(float(row["observed_at"]), self._message_replay_identity(row, stored_row=True)) for row in anchored}
        out = []
        for idx, stamp in pairs:
            store_id = mapping.get(id(messages[idx]))
            identity = self._message_replay_identity(identity_messages[idx], strip_carrier=False)
            if (store_id in null and not _lossy(identity) and identity == identity_of[store_id]
                    and counts[identity] == 1 and (stamp, identity) not in held):
                out.append((store_id, stamp))
        return out

    # -- R4: coverage bound to the summarizer input ----------------------------

    def _identity_anchor_summary_input(self, chunk, full_map, view=(), raw_chunk=(), budget=None,
                                       accounted_ids=(), copy_lookup=None) -> Optional[list]:
        """The summarizer input for ``chunk`` built TOGETHER with what each input row may claim:
        ``[(input_row, [store_id, ...]), ...]``, ``[]`` (no leaf can start), or None (today's input).
        - A live composite LCM recorded (R2 witness) claims its constituents: their text is in it.
        - #581: every other owned row above the frontier that no live row maps, and that no unmapped
          occurrence of the host ``view`` accounts for (occurrence for occurrence: one live row never
          hides two stored ones), is REHYDRATED from its stored bytes, in store order, and claims only
          itself; the leaf is a bounded contiguous prefix of the owned rows (store_complete.py).
        Nothing is claimed without its text in the input; rows with valid coverage (<= frontier) keep it."""
        if not identity_anchor_enabled() or not isinstance(full_map, dict) or not (chunk or budget) or (raw_chunk and not chunk):
            return None  # an empty chunk is the #581 hidden-backlog leaf, never a chunk of dependent replies
        frontier, mapped = self._store_complete_frontier(), set(full_map.values())
        carry = self._load_compression_carry_ranges()

        def owned(row) -> bool:
            return self._identity_anchor_owned(row, carry)

        claimed: set[int] = set()
        claims: dict[int, list] = {}
        shown: Optional[Counter] = None  # the view's unmapped occurrences, built once on first need
        in_view: set[int] = set()  # id() of the view objects, built with ``shown``
        self._identity_anchor_text_memo = {}
        scope = [str(self._session_id), *self._identity_anchor_chain()]
        for message in chunk:
            stamp = _normalize_observed_at(message.get("timestamp"))
            if id(message) in full_map:
                continue
            if stamp is None:  # D-D plan (ii): LCM's carrier merged with stored rows claims them (text in it)
                group = self._identity_anchor_carrier_group(message, set()) or ()
                ids = [int(row["store_id"]) for row in group if owned(row) and int(row["store_id"]) > frontier
                       and int(row["store_id"]) not in mapped | claimed]
                if ids:
                    claimed.update(ids)
                    claims[id(message)] = ids
                continue
            # #563: a live row the ordered mapper left unmapped (the host put it after a row stored
            # later) claims its R1-key occurrence; its own bytes are this input row.
            store_id = self._identity_anchor_key_occurrence(message, mapped | claimed, carry)
            if store_id is not None:
                claimed.add(store_id)
                claims[id(message)] = [store_id]
                continue
            if message.get("role") != "user":
                continue
            donors = [row for row in self._store.find_rows_by_observed_at(str(self._conversation_id or ""), scope, [stamp])
                      if row.get("role") == "user"]
            content = self._message_replay_identity(message, strip_carrier=False)[1]
            groups = [group for group in self._identity_anchor_witnesses(donors, stamp)
                      if self._identity_anchor_group_matches(content, group)] if donors else []
            if not groups and donors and "\n\n" in content:  # LCM observes the exact composite here (form i)
                # B-ID-1 (PR #590 r2): a row a live view row maps, or that the view's unmapped occurrences hold
                # as their own (the ingest site's reservation), is never a constituent. Over-reservation only
                # leaves the composite uncomposed: its text claims nothing (duplication, never a claim).
                if shown is None:
                    shown = Counter(self._identity_anchor_view_key(m) for m in view if id(m) not in full_map)
                    shown.pop(None, None)
                    in_view = {id(m) for m in view}
                pool = self._identity_anchor_pool(donors, mapped)
                # Only a view object's own occurrence is in ``shown``: a row that is not one (a stored row the
                # #581 input carries) subtracting its key would release another view row's reservation.
                own = self._identity_anchor_view_key(message) if id(message) in in_view else None
                reserved = self._identity_anchor_reserved(pool, shown - Counter([own] if own else []))
                pool = [row for row in pool if int(row["store_id"]) not in reserved]
                donors = [row for row in donors if int(row["store_id"]) not in mapped | reserved]
                pool = self._identity_anchor_eligible(pool, donors, content, stamp)
                pool, runs = self._identity_anchor_scope_unstamped(pool, donors)
                group, _ambiguous = self._identity_anchor_compose(
                    content, {text for row in pool for text in self._identity_texts(row)}, pool, donors, set(), runs=runs
                ) if donors else (None, False)
                if group is not None:
                    self._store.add_message_relations([_composite_relation(group, stamp)])
                    groups = [group]
            if len(groups) == 1:
                ids = [int(row["store_id"]) for row in groups[0]
                       if owned(row) and int(row["store_id"]) > frontier and int(row["store_id"]) not in mapped | claimed]
                claimed.update(ids)
                claims[id(message)] = ids
        # #563: a claim never jumps a row a live row after this chunk maps (a later pass publishes that
        # row; the claim would leave a hole below it). The claim stays pending: gap-fill rehydrates it.
        inside = {id(message) for message in [*chunk, *raw_chunk]}
        bound = min((full_map[id(message)] for message in view if id(message) in full_map
                     and id(message) not in inside and full_map[id(message)] > frontier), default=None)
        if bound is not None and any(store_id > bound for store_id in claimed):
            claims = {key: [store_id for store_id in ids if store_id < bound] for key, ids in claims.items()}
            claimed = {store_id for ids in claims.values() for store_id in ids}
        if budget is None:
            budget = count_messages_tokens(chunk)
        return self._store_complete_input(chunk, claims, full_map, view, raw_chunk, frontier, carry,
                                          max(budget, count_messages_tokens(chunk)), accounted_ids,
                                          copy_lookup=copy_lookup)

    def _identity_anchor_view_key(self, message) -> Optional[tuple]:
        """A view row's occurrence key ``(stamp, form)`` as the ingest site keys it; None for LCM's own scaffold
        or a lossy identity."""
        if self._identity_is_lcm_scaffold(message):
            return None
        identity = self._message_replay_identity(message, strip_carrier=False)
        return None if identity is None or _lossy(identity) else (_normalize_observed_at(message.get("timestamp")), identity)

    def _identity_anchor_owned(self, row, carry) -> bool:
        store_id, owner = int(row["store_id"]), str(row.get("session_id") or "")
        return owner == self._session_id or any(owner == s and a < store_id <= b for s, a, b in carry)

    def _identity_anchor_key_occurrence(self, message, taken, carry) -> Optional[int]:
        """#563: the owned occurrence above the frontier, in store order, that carries ``message``'s R1
        key (host stamp + full payload identity) and that is not ``taken`` (mapped or claimed)."""
        stamp = _normalize_observed_at(message.get("timestamp"))
        identity = self._message_replay_identity(message, strip_carrier=False) if stamp is not None else None
        if identity is None or _lossy(identity):
            return None
        frontier = int(self._last_compacted_store_id or 0)
        for row in self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *self._identity_anchor_chain()], [stamp]
        ):
            store_id = int(row["store_id"])
            if (store_id > frontier and store_id not in taken and self._identity_anchor_owned(row, carry)
                    and identity in self._stored_row_forms(row)):
                return store_id
        return None

    def _identity_anchor_covered_view(self, message, full_map) -> bool:
        """#563: a live user row that no store row maps and whose text is exactly a recorded composite
        (R2 witness) or its survivor, every constituent already covered (at or below the frontier)."""
        stamp = _normalize_observed_at(message.get("timestamp"))
        if message.get("role") != "user" or id(message) in full_map:
            return False
        frontier = int(self._last_compacted_store_id or 0)
        if stamp is None:  # D-D plan (ii): a carrier-headed composite of covered rows
            group = self._identity_anchor_carrier_group(message, set())
            return bool(group) and all(int(row["store_id"]) <= frontier for row in group)
        donors = [row for row in self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *self._identity_anchor_chain()], [stamp]
        ) if row.get("role") == "user"]
        content = self._message_replay_identity(message, strip_carrier=False)[1]
        for group in self._identity_anchor_witnesses(donors, stamp) if donors else ():
            texts = [self._identity_texts(row) for row in group]
            view = group if self._identity_anchor_group_matches(content, group) else [
                row for row, text in zip(group, texts) if content in text]
            if view and all(int(row["store_id"]) <= frontier for row in view):
                return True
        return False

    def _identity_anchor_extend_chunk(self, chunk, candidates) -> list:
        """#563: two leaf-chunk ends that can never publish, on H2's merge-turn views and re-flushes.
        - Only live views of covered history (a witnessed composite or its survivor, constituents at
          or below the frontier): they claim nothing, and oldest-first selection picks them again on
          every pass. The chunk takes the views after them and the oldest raw row (or its tool group);
          the views' text stays in the input and claims nothing (their coverage exists).
        - A following live row whose stored occurrence precedes one the chunk covers (the host put a
          row it re-flushed later before it): coverage would skip that occurrence. The chunk takes it."""
        if not chunk or len(chunk) >= len(candidates) or not identity_anchor_enabled():
            return chunk
        full_map, self._identity_anchor_text_memo = self._compress_run.store_ids_by_message_id or {}, {}
        end = len(chunk)
        if all(self._identity_anchor_covered_view(message, full_map) for message in chunk):
            while end < len(candidates) and self._identity_anchor_covered_view(candidates[end], full_map):
                end += 1
            if end >= len(candidates):
                return chunk
            end += len(self._select_oldest_leaf_chunk(list(candidates[end:]), 1))
        taken, carry = set(full_map.values()), self._load_compression_carry_ranges()
        top = max((full_map[id(message)] for message in candidates[:end] if id(message) in full_map), default=0)
        while end < len(candidates):
            store_id = full_map.get(id(candidates[end]))
            if store_id is None:
                store_id = self._identity_anchor_key_occurrence(candidates[end], taken, carry)
                taken.add(store_id or 0)
            if store_id is None or store_id >= top:
                break
            end += 1
        end = max(len(chunk), tool_group_safe_end(candidates, end)) if end > len(chunk) else end
        return list(candidates[:end])

    # -- writes ----------------------------------------------------------------

    def _identity_anchor_commit(self, plan, remainder_ids: Optional[dict] = None) -> None:
        """Record what the pre-match proved: relation groups (witnesses, alternate stamps, remainders),
        the R5 backfills and the R7 carry. Relations are durable before the carry that relies on them."""
        if remainder_ids is None:
            groups = [[(int(group[0]["store_id"]), "alt_stamp", None, None, stamp)] if kind == "alt_stamp"
                      else _composite_relation(group, stamp) for kind, stamp, group, _extra in plan["relations"]]
        else:  # after the store: only the remainders' groups, which need the new ids
            groups = [_composite_relation(plan["remainders"][idx][2] + [{"store_id": store_id}], plan["remainders"][idx][1])
                      for idx, store_id in remainder_ids.items()]
        if groups:
            self._store.add_message_relations(groups)
        for row, message in plan.get("ws", ()) if remainder_ids is None else ():
            self._record_ws_host_rewrite(row, message)
        self._store.backfill_observed_at_many(plan["backfill"] if remainder_ids is None else ())
        if remainder_ids is None and plan["carry"]:
            self._register_identity_anchor_carry(plan["carry"])

    def _record_ws_host_rewrite(self, row, message) -> None:
        """R1-ws: record the view's form as a #498 host-rewrite override; the stored row is never modified."""
        store_id, stored = int(row["store_id"]), normalize_content_value(row.get("content")) or ""
        identity = self._message_replay_identity(row, stored_row=True)
        self._host_rewrite_state()[0][store_id] = (identity, stored, [message])  # the capture consumes it
        self._capture_host_rewrite(store_id, identity, stored, message)

    def _note_identity_anchor_version(self, store_id, message) -> None:
        """R6: this process stored ``store_id`` from ``message`` and now observes the SAME host object
        with other content: positive evidence of an in-place rewrite. Held until the new version's store."""
        if identity_anchor_enabled():
            versions = [entry for entry in getattr(self, "_identity_anchor_versions", ()) if entry[0] is not message]
            self._identity_anchor_versions = (versions + [(message, int(store_id))])[-_RECENT_CAP:]

    def _identity_anchor_rewritten(self, messages, cursor: int) -> list:
        """Indexes before ``cursor`` holding a host object this process stored and has since seen
        rewritten (R6 before/after evidence, stamped or not)."""
        versions = getattr(self, "_identity_anchor_versions", ()) if identity_anchor_enabled() else ()
        return [idx for idx in range(min(cursor, len(messages)))
                if any(message is messages[idx] for message, _store_id in versions)]

    def _identity_anchor_version_rewind(self, messages, plan) -> None:
        """R6: a rewritten host object whose new bytes the pre-match did not explain as stored
        occurrences (a composite, an alias, a replay) is an occurrence not stored yet, stamped or not:
        the cursor moves back to it; every other row of that range stays a replay (today's positional
        proof)."""
        cursor = plan["cursor"]
        rewritten = [idx for idx in self._identity_anchor_rewritten(messages, cursor)
                     if idx >= plan.get("accepted_prefix", 0) and idx not in plan.get("explained", ())]
        if not rewritten:
            return
        for idx in rewritten:  # a rewound row is a new store now; the host-uid shadow sees it via its store id
            plan.get("matched", {}).pop(idx, None)
        plan["cursor"] = min(rewritten)
        plan["replayed"].update(idx for idx in range(min(rewritten), cursor) if idx not in rewritten)
        plan["replayed"].difference_update(rewritten)

    def _identity_anchor_record_versions(self, stored) -> None:
        """R6: a rewritten host object that this ingest stored as a row of its own is a new version of
        the row it was stored as before: ``supersedes`` relation, both versions' bytes kept. A version
        this ingest recognised as stored occurrences (a composite, a replay) records nothing here."""
        versions = list(getattr(self, "_identity_anchor_versions", ()))
        groups = []
        for message, store_id in stored:
            old = next((entry for entry in versions if entry[0] is message), None)
            if old is not None:
                versions.remove(old)
                groups.append([(int(store_id), "supersedes", old[1], None, _normalize_observed_at(message.get("timestamp")))])
        self._identity_anchor_versions = versions
        if groups:
            self._store.add_message_relations(groups)

    def _identity_anchor_remember(self, stored) -> None:
        """R5 current-turn window: (session, identity, store_id, observed_at) of user rows just stored."""
        recent = list(getattr(self, "_identity_anchor_recent", ()))
        before = len(recent)
        for identity, store_id, message in stored:
            if identity is not None and identity[0] == "user":
                recent.append((self._session_id, identity, int(store_id), _normalize_observed_at(message.get("timestamp"))))
        if len(recent) > before:
            self._identity_anchor_user_generation = getattr(self, "_identity_anchor_user_generation", 0) + 1
        self._identity_anchor_recent = recent[-_RECENT_CAP:]


def _composite_relation(group, stamp) -> list:
    """One composite relation group, keyed by the constituent whose host stamp the composite carries."""
    head = next((row for row in group if row.get("observed_at") == stamp), group[0])
    return [(int(head["store_id"]), "composite", int(row["store_id"]), ordinal, stamp) for ordinal, row in enumerate(group)]


def _distinct_from_predecessor(identity_at, idx: int) -> bool:
    """#1501: the view rows ``idx - 1`` and ``idx`` differ in the form occurrence matching compares
    (edge whitespace stripped, structured content canonical); an unknown identity is never distinct."""
    from .reconcile import _proof_user_identity

    if idx < 1:
        return False
    previous, current = identity_at(idx - 1), identity_at(idx)
    return previous is not None and current is not None and _proof_user_identity(previous) != _proof_user_identity(current)


def _lossy(identity) -> bool:
    from .reconcile import _has_lossy_redacted_identity

    return _has_lossy_redacted_identity(identity)


def _raw_remainder(message, remainder) -> Optional[str]:
    """R3's stored bytes: the survivor's raw content must be EXACTLY the matched constituent forms,
    each joined by the host's ``"\n\n"``, then ``"\n\n"`` and the remainder; else None (the
    survivor is stored whole: visible duplication, never a lost or altered separator)."""
    rest, _stamp, _group, parts = remainder
    raw = normalize_content_value(message.get("content")) or ""
    head = "\n\n".join(parts) + "\n\n"
    return raw[len(head):] if raw.startswith(head) and raw[len(head):] == rest else None
