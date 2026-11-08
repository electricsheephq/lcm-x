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

import hashlib
import json
import os
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
from .config import identity_anchor_enabled
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
_PRESENCE_WINDOW_ROWS = 10000  # total stored rows across the presence scope
_PRESENCE_WINDOW_BYTES = 15_000_000  # UTF-8 bytes of stored content and tool-call text across the scope
_PRESENCE_WINDOW_CALL_IDS = 10_000  # stored tool-call entries across the scope (#2466 wait bar)
_PRESENCE_PARSE_BYTES = 256_000  # call-row text parsed as JSON when an id's raw forms differ (#2466 wait bar)
_PRESENCE_WINDOW_LOGGED: set[tuple[str, int, str]] = set()  # store file and session, once per process
_PRESENCE_SHAPE_LOGGED: set[tuple[str, int, str]] = set()  # non-standard tool shapes, once per store and session
_UNREADABLE_CALLS = (object(),)  # an unreadable tool_calls value: no host id can equal it (#2466 review)

_ROTATION_LINEAGE_UNRECORDED: set[tuple[str, str]] = set()  # store file and session whose predecessor write failed, per process


def _ws_equivalent(stored, identity) -> bool:
    return (identity[0] == "user" and bool(identity[1].strip()) and stored != identity
            and stored[1].strip() == identity[1].strip() and tuple(stored[2:]) == tuple(identity[2:]))


class IdentityAnchorMixin:
    """Mixed into LCMEngine; reads ``self._store``, the reconcile identity helpers and ``_state_db_path``."""

    # -- lineage (R7) --------------------------------------------------------

    def _identity_anchor_chain_cached(self) -> list[str]:
        """Verified ancestors already cached for this binding; never performs I/O."""
        cached = getattr(self, "_identity_anchor_chain_cache", None)
        return cached[1] if cached is not None and cached[0] == str(self._session_id or "") else []

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

    def _observe_alias_uid_conflicts(self, rows, canonical_stamps, keys_of, occurrences, messages):
        """Count conflicting alias edges; never change matching or persisted state (#1519)."""
        try:
            self._count_alias_uid_conflicts(rows, canonical_stamps, keys_of, occurrences, messages)
        except Exception as exc:
            logger.debug("LCM identity-anchor host UID observation unavailable (%s)", type(exc).__name__)

    def _count_alias_uid_conflicts(self, rows, canonical_stamps, keys_of, occurrences, messages):
        from .config import host_message_uid_mode
        from .host_uid import _valid_uid

        mode = host_message_uid_mode()
        if mode == "off":
            return
        incoming = {idx: messages[idx].get("message_uid") for idx, _key in occurrences
                    if _valid_uid(messages[idx].get("message_uid"))}
        incoming_keys = {key for idx, key in occurrences if idx in incoming}
        row_keys = {id(r): set(keys_of(r)) for r in rows} if incoming_keys else {}
        if canonical_stamps is None:  # discarded provenance cannot justify extra diagnostic reads
            return
        aliases = [r for r in rows
                   if int(r["store_id"]) in canonical_stamps
                   and _normalize_observed_at(canonical_stamps[int(r["store_id"])]) != float(r["observed_at"])
                   and incoming_keys.intersection(row_keys.get(id(r), ()))]
        if not aliases:
            return
        lineage, _problem = self._host_uid_lineage_key()
        if lineage is None:
            return
        if not self._store._host_uid_table_exists():
            return
        # JSON key sets keep these two indexed reads bounded even above SQLite's bind-variable limit.
        canonical = defaultdict(set)
        for sid, uid in self._store._conn.execute(
            "SELECT store_id, uid FROM host_uid_bindings INDEXED BY idx_host_uid_store "
            "WHERE store_id IN (SELECT value FROM json_each(?)) AND lineage_key = ? AND kind = 'canonical'",
            (json.dumps(sorted({int(r["store_id"]) for r in aliases})), lineage),
        ):
            canonical[int(sid)].add(str(uid))
        if not canonical:
            return
        uids = set(incoming.values()) | {uid for values in canonical.values() for uid in values if _valid_uid(uid)}
        engine = {str(uid) for (uid,) in self._store._conn.execute(
            "SELECT uid FROM host_uid_bindings WHERE lineage_key = ? AND kind = 'engine' "
            "AND uid IN (SELECT value FROM json_each(?))", (lineage, json.dumps(sorted(uids))))}
        incoming = {idx: uid for idx, uid in incoming.items() if uid not in engine}
        canonical = {sid: {uid for uid in values if _valid_uid(uid) and uid not in engine}
                     for sid, values in canonical.items()}
        by_key = defaultdict(Counter)
        for idx, key in occurrences:
            if idx in incoming:
                by_key[key][incoming[idx]] += 1
        totals = {key: sum(uids.values()) for key, uids in by_key.items()}
        # Count via the key index without materialising the potentially quadratic edge set.
        count = 0
        for row in aliases:
            known = canonical.get(int(row["store_id"]), ())
            if len(known) == 1:
                uid = next(iter(known))
                count += sum(totals.get(key, 0) - by_key[key][uid] for key in row_keys[id(row)])
        if count:
            logger.debug("LCM identity-anchor host UID shadow would reject alias edges: count=%d", count)

    def _identity_anchor_prematch(self, messages, identity_messages, cursor: int, audit_from: Optional[int] = None,
                                  *, accepted_prefix: int = 0, ledger_pairs=None, ledger_today=None) -> Dict[str, Any]:
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
        ledger_pairs = ledger_pairs or {}
        fetch_start = min(start, ledger_today) if ledger_today is not None else start
        if not identity_anchor_enabled() or not self._session_id or start >= n:
            return plan
        stamps = {}
        for idx in range(n):
            observed_at = _normalize_observed_at(messages[idx].get("timestamp"))
            if observed_at is not None:
                stamps[idx] = observed_at
        wanted = {stamps[idx] for idx in range(fetch_start, n) if idx in stamps}
        if not wanted and start >= cursor:
            return plan
        # #633: nothing stamped from ``start`` on still leaves a changed prefix to audit (an unstamped row).
        chain = self._identity_anchor_chain()
        rows = self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *chain], sorted(wanted)
        ) if wanted else []
        self._load_host_rewrite_overrides(rows)
        plan["stamped_users"] = {int(row["store_id"]) for row in rows if row.get("role") == "user"}
        canonical_stamps = getattr(rows, "canonical_stamps", None)
        consumed = {int(row["store_id"]) for row in ledger_pairs.values()}
        rows = [row for row in rows if int(row["store_id"]) not in consumed]
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

        matched: dict[int, list] = {idx: [row] for idx, row in ledger_pairs.items()}
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
                       if idx not in ledger_pairs and stamps[idx] in wanted and identity_at(idx) is not None]
        def keys_of(r):
            return {(float(r["observed_at"]), form) for form in forms_of[id(r)]}
        ws_index = None

        def diag_keys_of(r):
            nonlocal ws_index
            if ws_index is None:  # Only reached after the observer's mode/valid-UID gates.
                from .host_uid import _valid_uid

                ws_index = defaultdict(set)
                for idx, key in occurrences:
                    if not _valid_uid(identity_messages[idx].get("message_uid")):
                        continue
                    ident = key[1]
                    if ident[0] == "user" and ident[1].strip():
                        ws_index[(key[0], ident[1].strip(), tuple(ident[2:]))].add(key)
            keys = keys_of(r)
            if r.get("role") == "user":
                forms = forms_of[id(r)]  # A singleton is necessarily the exact stored identity.
                stored = next(iter(forms)) if len(forms) == 1 else self._message_replay_identity(r, stored_row=True)
                bucket = (float(r["observed_at"]), stored[1].strip(), tuple(stored[2:]))
                keys.update(key for key in ws_index.get(bucket, ()) if _ws_equivalent(stored, key[1]))
            return keys

        self._observe_alias_uid_conflicts(
            rows, canonical_stamps, diag_keys_of, occurrences, identity_messages)
        for store_id, idx in _match_occurrences(rows, keys_of, occurrences).items():
            consumed.add(store_id)
            matched[idx] = [next(r for r, _forms in by_stamp[stamps[idx]] if int(r["store_id"]) == store_id)]
            if idx >= start:
                plan["replayed"].add(idx)
        for idx in range(start, n if wanted else start):  # nothing stamped: only the #633 audit below
            if idx in ledger_pairs:
                continue
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
            if idx in ledger_pairs or idx in plan["replayed"] or idx not in stamps or messages[idx].get("role") != "user":
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
                                        matched, excluded=ledger_pairs)
        self._identity_anchor_tool_segments(messages, plan["cursor"], plan["replayed"], matched, plan.get("positional", ()))
        for idx in list(plan["remainders"]):
            if idx in plan["replayed"]:
                del plan["remainders"][idx]
        plan["explained"] = (plan["replayed"] - plan.get("positional", set())) | ledger_pairs.keys()
        plan["replayed"] = {idx for idx in plan["replayed"] if idx >= plan["cursor"]}
        plan["carry"] = [row for idx in plan["replayed"] | set(plan["remainders"]) for row in matched.get(idx, ())]
        plan["carry"] += [row for row, _message in plan.get("ws", ())]  # R1-ws matches the audit proved too
        if plan["replayed"] and any(str(row.get("session_id")) in chain for row in plan["carry"]):
            # A rotation child resuming onto its ancestors' rows: LCM's own DAG-verified carrier heading
            # the list is compress() output, not a host message (R8).
            for idx in range(plan["cursor"], min(plan["replayed"])):
                if self._identity_is_lcm_scaffold(identity_messages[idx], verified=True):
                    plan["replayed"].add(idx)
        # Seeded run pairs support local carried-stamp proof only: the ledger
        # writer and uid shadow must still see an ordinary unbound prefix.
        plan["matched"] = {idx: rows for idx, rows in {**plan.get("matched", {}), **matched}.items()
                           if idx not in ledger_pairs}
        return plan

    def _identity_anchor_audit(self, messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan,
                               matched, *, excluded=()) -> None:
        """Rows in ``[start, cursor)`` of a list the host changed before the cursor: a stamped host row,
        or an unstamped user row the host inserted there (#633: a /steer row), that
        no stored occurrence explains (key, witness, alias, or a stored copy of its content -- up to
        edge whitespace, with the same stamp or a canonical host-uid binding for a NULL stamp --
        in the tail of this session or a verified ancestor, each copy used once) and no ignore pattern drops was never stored. When there
        is such a row, every other row of the range must be accounted for too (#2071's rule, here for the parent
        list): ``_identity_anchor_unaccounted`` adds the rows provably never stored, and keeps its unsure cases as
        replays. The cursor moves back to the first row to store; every other row from there stays a replay."""
        from .reconcile import _proof_user_identity

        span = max(64, 2 * (cursor - start))
        held: dict = defaultdict(list)
        chain = self._identity_anchor_chain()
        capped = False  # a session holds more rows than the audited tail: its older copies are not in ``held``
        for session in [str(self._session_id), *chain]:
            tail = self._store.get_session_tail(session, limit=span)
            capped = capped or len(tail) >= span
            for row in tail:
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
            if idx in excluded:
                continue
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
            # Class A (#2071 rule): the cursor resumes only behind rows that are accounted for. A position no longer
            # proves a row stored once the host changed the list, and the audit above never examines tool rows.
            # A host object this process stored and now sees rewritten in place (R6) is never credited as present nor
            # made a positional replay here: the version rewind still stores its new version, as on main (#2466 review).
            rewritten = {idx for idx in self._identity_anchor_rewritten(messages, cursor)
                         if idx >= plan.get("accepted_prefix", 0) and idx not in plan["replayed"] and idx not in excluded}
            found = self._identity_anchor_unaccounted(
                messages, identity_messages, cursor, start, identity_at, held, capped, consumed, plan, matched,
                set(excluded) | rewritten, missed)
            if found is None:  # the presence check tripped (logged there): main's plan exactly (#2466 review)
                plan["cursor"] = min(missed)
                plan["positional"] = {idx for idx in range(min(missed), cursor) if idx not in missed}
                plan["replayed"].update(plan["positional"])
                logger.info("LCM identity-anchor: host changed its list before the cursor; %d unstored rows from %d: "
                            "session=%s", len(missed), min(missed), self._session_id)
                return
            missed = sorted({*missed, *found})
            plan["cursor"] = min(missed)
            plan["positional"] = {idx for idx in range(min(missed), cursor) if idx not in missed and idx not in rewritten}
            plan["replayed"].update(plan["positional"])
            logger.warning("LCM identity-anchor: host changed its list before the cursor; %d row(s) from %d are neither "
                           "stored nor compaction outputs; storing them: session=%s",
                           len(missed), min(missed), self._session_id)

    def _identity_anchor_unaccounted(self, messages, identity_messages, cursor, start, identity_at, held, capped,
                                     consumed, plan, matched, excluded, missed) -> Optional[list[int]]:
        """Call and result rows in ``[start, cursor)`` that are neither bound, matched, a compaction output (LCM's
        own scaffold) nor ignored, and whose id is stored nowhere in the scope: this session, its verified ancestors
        and its recorded predecessors, within the row and stored-byte budgets. One id is one invocation, so an id
        stored anywhere accounts for its rows, including an upgraded store's pre-ledger rows. Pairs stay whole and a
        reused id keeps main's result. Text rows are not checked (#2116, v0.30.x). ``capped``, ``held`` and
        ``consumed`` are retained for callers but unused. Past the window, on a non-standard stored tool shape, or on a
        failed read, it returns None (a trip): the caller then keeps main's plan exactly, adding nothing beyond its own finds,
        as on main. Unsure is accounted (the positional replay, as before): a call row whose ids cannot be read."""
        calls_memo: dict = {}  # view rows are read several times below

        def calls(msg, stored=False):
            if not stored:
                if id(msg) not in calls_memo:
                    calls_memo[id(msg)] = read_calls(msg, False)
                return calls_memo[id(msg)]
            return read_calls(msg, True)

        def read_calls(msg, stored):
            value = msg.get("tool_calls") or ()
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    return _UNREADABLE_CALLS  # unreadable: equal to nothing, so never proof of a copy
            if not stored and isinstance(value, list) and any(
                    not isinstance(c.get("id"), str) for c in value if isinstance(c, dict)):
                return None  # non-string host ids cannot prove absence against SQLite's JSON casts
            return tuple(sorted(str(c.get("id") or "") for c in value if isinstance(c, dict))) if isinstance(
                value, list) else ()

        def tool_row(msg):
            return calls(msg) != () or msg.get("role") == "tool" and bool(msg.get("tool_call_id"))

        def key(msg, stored=False):
            ids = calls(msg, stored=stored)
            if ids is None or ids is _UNREADABLE_CALLS:
                return None
            if ids:
                return ("call", ids)
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                return ("tool", str(msg["tool_call_id"])) if stored or isinstance(msg["tool_call_id"], str) else None
            identity = self._message_replay_identity(msg, stored_row=stored)
            return None if _lossy(identity) else (identity[0], identity[1].strip())

        explained = set(excluded) | set(matched) | set(plan.get("matched", {})) | plan["replayed"] | set(missed)
        try:
            # Only call and result rows are presence-checked (class A). A replaced reply (#2116) keeps main's
            # result: its text identity can load payload files the window cannot bound (#2466 review, split to
            # v0.30.x under #2467).
            candidates = {idx: key(identity_messages[idx]) for idx in range(start, cursor)
                          if idx not in explained and tool_row(identity_messages[idx])
                          and not (self._identity_is_lcm_scaffold(identity_messages[idx])
                                   and identity_at(idx) is None)  # scaffold first: identity_at parses arguments
                          and not self._matches_ignore_message_patterns(messages[idx])}
            session = str(self._session_id)
            chain = self._identity_anchor_chain()
            cached = getattr(self, "_identity_anchor_chain_cache", None)
            if self._state_db_path().exists() and (cached is None or cached[0] != session):
                raise ValueError("incomplete ancestry read")
            predecessors = self._rotation_predecessor_session_ids(session)
            # The helper swallows read errors. Verify its reads before trusting the scope as complete.
            for hop, source in enumerate([session, *predecessors]):
                value = self._store.read_metadata_json(self._rotation_predecessor_metadata_key(source))
                next_source = value if isinstance(value, str) and value and value not in [session, *predecessors[:hop]] else None
                if next_source != (predecessors[hop] if hop < len(predecessors) else None):
                    raise ValueError("incomplete rotation predecessor scope")
            scope = list(dict.fromkeys([session, *chain, *predecessors]))
            failed = {session_id for path, session_id in _ROTATION_LINEAGE_UNRECORDED
                      if path == os.path.realpath(str(self._db_path))}
            if set(scope) & (getattr(self, "_rotation_lineage_unrecorded", set()) | failed):
                # Any session of the scope whose own predecessor write failed ends the scope early (#2466 review).
                raise ValueError("unrecorded rotation lineage")
            rows = {int(row["store_id"]): row for group in held.values() for row in group}
            budget, byte_budget, id_budget = _PRESENCE_WINDOW_ROWS, _PRESENCE_WINDOW_BYTES, _PRESENCE_WINDOW_CALL_IDS
            for source in scope:
                # Size the window in SQL before loading any row (#2466 review, wait bar): stored UTF-8 bytes per row
                # (SQLite's length() of text stops at the first NUL, and Python loads the whole value), newest first,
                # stopping at the first budget it passes, so an over-budget scope costs at most its budget plus one
                # row of reading and decodes nothing. Only call and result rows are compared, by their stored text,
                # so the audit loads no payload file. Stored tool-call entries are counted too: the audit's work grows
                # with them, so a window of wide parallel call rows is bounded as well. Only the standard tool shape is
                # admitted (#2466 review, scope cut): tool_calls NULL, or a JSON array of objects whose id and
                # function.arguments are text. Any other row gives main's result for the whole audit, before any row
                # loads, so SQL-visible limits bound the work on the admitted path.
                sized = self._store._conn.execute(
                    "SELECT length(CAST(content AS BLOB)), length(CAST(tool_calls AS BLOB)), "
                    "CASE WHEN tool_calls IS NULL THEN 0 WHEN length(CAST(tool_calls AS BLOB)) > ? THEN -1 "
                    "WHEN NOT json_valid(tool_calls) THEN -1 "
                    "WHEN json_type(tool_calls) <> 'array' THEN -1 "
                    "WHEN EXISTS (SELECT 1 FROM json_each(tool_calls) AS e WHERE e.type <> 'object' "
                    "OR json_type(tool_calls, '$[' || e.key || '].id') IS NOT 'text' "
                    "OR json_type(tool_calls, '$[' || e.key || '].function.arguments') IS NOT 'text') THEN -1 "
                    "ELSE json_array_length(tool_calls) END "
                    "FROM messages WHERE session_id=? ORDER BY store_id DESC LIMIT ?",
                    (_PRESENCE_WINDOW_BYTES, source, budget + 1))
                count, shape_ok = 0, True
                for content_len, calls_len, call_ids in sized:
                    count += 1
                    if call_ids < 0:
                        shape_ok = False
                        break
                    byte_budget -= (content_len or 0) + (calls_len or 0)
                    id_budget -= call_ids
                    if count > budget or byte_budget < 0 or id_budget < 0:
                        break
                sized.close()
                if not shape_ok:
                    notice = (str(self._db_path), self._db_path.stat().st_ino, session)
                    if notice not in _PRESENCE_SHAPE_LOGGED:
                        logger.info("LCM identity-anchor presence check skipped: non-standard tool-call shape: "
                                    "session=%s scope=%d", session, len(scope))
                        _PRESENCE_SHAPE_LOGGED.add(notice)
                    return None  # a trip: the caller restores main's plan
                if count > budget or byte_budget < 0 or id_budget < 0:
                    notice = (str(self._db_path), self._db_path.stat().st_ino, session)
                    if notice not in _PRESENCE_WINDOW_LOGGED:
                        logger.info("LCM identity-anchor presence window incomplete: session=%s scope=%d "
                                    "budget=%d bytes=%d call_ids=%d", session, len(scope), budget, byte_budget, id_budget)
                        _PRESENCE_WINDOW_LOGGED.add(notice)
                    return None  # a trip: the caller restores main's plan
                end, pages = None, []
                while True:
                    limit = min(256, budget + 1)
                    tail = (self._store.get_session_tail(source, limit=limit) if end is None else
                            self._store.get_session_rows_through(source, end, limit))
                    budget -= len(tail)
                    pages.append(tail)
                    if len(tail) < limit:
                        break
                    end = int(tail[0]["store_id"]) - 1
                for page in reversed(pages):  # Preserve the previous chronological copy-consumption order.
                    rows.update((int(row["store_id"]), row) for row in page)
            # One id, one invocation (#2466 review, plain rule). Every call id and result id stored anywhere in the
            # scope counts as stored. An id whose call rows differ between the view and the window (the row's ids,
            # tool names, or arguments compared as parsed JSON) is reused: its rows keep main's result (#2172).
            # Result text is not compared: the store can hold one result in another form (an externalized or a
            # host-pruned output).
            call_sig: dict = defaultdict(set)
            rows_by_id: dict = defaultdict(list)
            form_token: dict = {}
            stored_calls, stored_results = set(), set()

            def canonical(arguments):
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        return arguments.strip()
                return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)

            def shape(msg, parse=False):
                value = msg.get("tool_calls") or ()
                if isinstance(value, str):
                    value = json.loads(value)
                parts = []
                for c in value if isinstance(value, list) else ():
                    function = (c.get("function") or {}) if isinstance(c, dict) else {}
                    if isinstance(c, dict) and isinstance(function, dict):
                        arguments = function.get("arguments")
                        if parse:
                            arguments = canonical(arguments)
                        elif not isinstance(arguments, str):
                            arguments = object()  # equal to nothing as stored: parsed within the budget
                        parts.append((str(c.get("id") or ""), str(function.get("name") or ""), arguments))
                return tuple(sorted(parts))

            def note(msg, stored):
                ids = calls(msg, stored=stored)
                if ids:
                    if ids is _UNREADABLE_CALLS:
                        return
                    # One lookup per row: a row with many ids would otherwise rehash its whole form per id.
                    form = form_token.setdefault(shape(msg), len(form_token))
                    for call_id in ids:
                        call_sig[call_id].add(form)
                        rows_by_id[call_id].append(msg)
                        if stored:
                            stored_calls.add(call_id)
                elif stored and msg.get("role") == "tool" and msg.get("tool_call_id"):
                    stored_results.add(str(msg["tool_call_id"]))

            for store_id in sorted(rows):
                note(rows[store_id], True)
            for msg in identity_messages:
                note(msg, False)
            # Raw forms first; arguments are parsed only for an id whose raw forms differ, within a byte budget (the
            # wait bar). Past it the id stays reused, which is main's result.
            reused, parse_budget, parsed, parsed_token = set(), _PRESENCE_PARSE_BYTES, {}, {}
            for call_id, forms in call_sig.items():
                if len(forms) > 1:
                    group = rows_by_id[call_id]
                    pending = [m for m in group if id(m) not in parsed]
                    if pending:
                        need = sum(len(repr(m.get("tool_calls"))) for m in pending) if parse_budget > 0 else 1
                        if need > parse_budget:
                            parse_budget = 0  # spent: every later differing id stays reused
                            reused.add(call_id)
                            continue
                        parse_budget -= need
                        for m in pending:
                            parsed[id(m)] = parsed_token.setdefault(shape(m, parse=True), len(parsed_token))
                    if len({parsed[id(m)] for m in group}) > 1:
                        reused.add(call_id)
        except Exception as exc:
            logger.warning("LCM identity-anchor presence unavailable (%s); keeping positional replays", type(exc).__name__)
            return None  # a trip: the caller restores main's plan
        results_of: dict = defaultdict(list)
        callers: dict = defaultdict(list)
        for k, msg in enumerate(identity_messages):
            ids = calls(msg) or ()
            if ids:
                for call_id in ids:
                    callers[call_id].append(k)
            elif msg.get("role") == "tool" and msg.get("tool_call_id"):
                results_of[str(msg["tool_call_id"])].append(k)
        # A view that shows one id on more than one call row, or more than one result, is a reused id too.
        reused |= {call_id for rows_of in (callers, results_of) for call_id, at in rows_of.items() if len(at) > 1}
        # A tool segment (a call row and the results right after it) is replayed whole or not at all
        # (_identity_anchor_tool_segments): storing one of its rows makes its stored rows at or past the cursor
        # stored again. So no row of a segment that holds such a stored row is stored here (#2466 review).
        blocked, k = set(), 0
        while k < len(identity_messages):
            if identity_messages[k].get("role") == "assistant" and identity_messages[k].get("tool_calls"):
                end = k + 1
                while end < len(identity_messages) and identity_messages[end].get("role") == "tool":
                    end += 1
                if any(j >= cursor and (set(calls(identity_messages[j]) or _UNREADABLE_CALLS) <= stored_calls if j == k else
                                        str(identity_messages[j].get("tool_call_id") or "") in stored_results)
                       for j in range(k, end)):
                    blocked.update(range(k, end))
                k = end
            else:
                k += 1
        new = set()
        for idx, wanted in candidates.items():
            if wanted is None:
                continue  # unsure: accounted, as on main
            ids = set(wanted[1]) if wanted[0] == "call" else {wanted[1]}
            if idx in blocked or ids & reused or ids & (stored_calls if wanted[0] == "call" else stored_results):
                continue  # reused (main's result), or stored: accounted (a partly stored call row keeps main's result)
            new.add(idx)

        # Groups stay whole (#2466 review). A new call is stored only if each result of its ids in the view is stored
        # now, or sits at or past the cursor and is not replayed (the view's own later rows). A new result is stored
        # only if its call is stored, or a call row carrying its id is stored now and not dropped by the ignore
        # patterns. Anything else keeps main's result, never a split pair.
        stored_now = set(missed) | new
        ignored: dict = {}

        def dropped(k):
            if k not in ignored:
                ignored[k] = self._matches_ignore_message_patterns(messages[k])
            return ignored[k]

        changed = True
        while changed:
            changed = False
            for idx in sorted(new):
                wanted = candidates[idx]
                if wanted[0] == "call":
                    ok = all(j in stored_now or j < cursor and call_id in stored_results
                             or j >= cursor and j not in plan["replayed"] and call_id not in stored_results
                             and not dropped(j)
                             for call_id in wanted[1] for j in results_of.get(call_id, ()))
                else:
                    ok = wanted[1] in stored_calls or any(k in stored_now and not dropped(k)
                                                          for k in callers.get(wanted[1], ()))
                if not ok:
                    new.discard(idx)
                    stored_now.discard(idx)
                    changed = True
        return sorted(new)

    def _identity_anchor_record_recovery(self, result, emissions, messages=()) -> None:
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
        # Bind the note to the latest input user occurrence, never to a later equal-text row.
        source = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        receipt["source_stamp"] = _normalize_observed_at(source.get("timestamp")) if source is not None else None
        receipts = list(getattr(self, "_identity_anchor_recoveries", ()))
        for previous in receipts:
            if (previous["row"] is row or previous["store"] is self._store
                    and previous["generation"] == receipt["generation"]):
                previous["blocked"] = receipt["blocked"] = True
        self._identity_anchor_recoveries = (receipts + [receipt])[-4:]
        self._identity_anchor_recovery = receipt

    def _identity_anchor_link_durable_recovery(self, attempt, lineage) -> bool | None:
        """#1523: durable doubt is a visible link, never authority to skip a cold user row."""
        from . import ingest_ledger
        from .store import _normalize_conversation_id_value

        conn, salt = self._store._conn, attempt["salt"]
        view, _reason = ingest_ledger.emission_candidate(conn, lineage)
        if view is None:
            return
        emitted = ingest_ledger.decode_entries(view["entries"])
        entries = attempt["entries"]
        recoveries = list(conn.execute(
            "SELECT emission_id,output_index,descriptor_digest,observed_stamp,recovered_store_id "
            "FROM ledger_emissions WHERE lineage=? AND view_seq=? AND kind='recovery' "
            "AND recovered_store_id IS NOT NULL AND state IN ('proposed','adopted')",
            (lineage, view["view_seq"])))
        if len(recoveries) != 1:
            return
        emission, idx, digest, observed, target = recoveries[0]
        if (idx != len(emitted) - 1 or idx >= len(entries)
                or any(a.identity_digest != b.identity_digest or a.stamp_digest != b.stamp_digest
                       or a.uidset_digest != b.uidset_digest for a, b in zip(emitted[:idx], entries[:idx]))
                or entries[idx].uidset_digest != emitted[idx].uidset_digest):
            return
        stamp = attempt["stamps"][idx]
        generated = entries[idx].identity_digest == digest
        stored_id = attempt["stored"].get(idx)
        if stamp is None or observed not in (None, stamp):
            return
        if stored_id is None:
            # The cold cursor can recognize the descriptor without storing the note.
            # Admit its later warm rewrite to reads only; durable writes still need both rows.
            return True if generated else None
        if target >= stored_id:
            return
        rows = self._store.get_batch([target, stored_id])
        prior, current = rows.get(target), rows.get(stored_id)
        conversation = _normalize_conversation_id_value(self._conversation_id)
        if (not prior or not current or prior.get("role") != "user" or current.get("role") != "user"
                or any(r.get("session_id") != self._session_id for r in (prior, current))
                # A legacy recovered row may carry no conversation id (NULL or blank); the new row never may.
                or _normalize_conversation_id_value(prior.get("conversation_id")) not in (conversation, "")
                or _normalize_conversation_id_value(current.get("conversation_id")) != conversation
                or prior.get("observed_at") is None or prior["observed_at"] == current.get("observed_at")
                or not generated and self._message_replay_identity(prior, stored_row=True)
                != self._message_replay_identity(current, stored_row=True)):
            return
        # Only assistant/tool followers or the observed generated note may intervene.
        # Page through every row between them: a long tool run must not hide the link (review: no follower cap).
        start = target + 1
        while start < stored_id:
            followers = self._store.get_range(str(self._session_id), start_id=start, end_id=stored_id - 1,
                                              limit=256, conversation_id=conversation, include_blank_conversation=True)
            if any(r.get("role") not in {"assistant", "tool"} and not (
                    observed is not None and ingest_ledger.stamp_text(r.get("observed_at")) == observed
                    and ingest_ledger.identity_digest(salt, r) == digest) for r in followers):
                return
            if len(followers) < 256:
                break
            start = followers[-1]["store_id"] + 1
        if self._store.get_message_relations([target], "alt_stamp"):
            return  # the live receipt already consumed this occurrence
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone() and conn.execute(
                "SELECT 1 FROM message_relations WHERE kind='possible_replay_of' AND related_store_id=? "
                "AND observed_at IS NULL", (target,)).fetchone():  # a borrowed-stamp link never consumes its target
            return
        if generated:
            if observed is None:
                conn.execute("UPDATE ledger_emissions SET observed_stamp=? WHERE emission_id=? AND observed_stamp IS NULL",
                             (stamp, emission))
            return True  # Keep admitting the later warm rewrite; row guards still own every write.
        else:
            # A source keeps its one link (the borrowed-stamp linker runs first); the alt_stamp read above
            # ensured the relation table.
            if not conn.execute("SELECT 1 FROM message_relations WHERE kind='possible_replay_of' AND store_id=? LIMIT 1",
                                (stored_id,)).fetchone():
                self._store.add_message_relations([[(stored_id, "possible_replay_of", target, None, None)]])
            return False

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
            if getattr(self, "_ingest_ledger_observations", None) is not None:
                self._ingest_ledger_observations.extend((sid, kind, stamp) for group in groups
                                                       for sid, kind, _related, _ordinal, stamp in group)

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
            with self._ingest_evidence():
                self._record_ws_host_rewrite(row, head)  # existing payload, protection and skip_unchanged writer
        except Exception as exc:
            self._raise_ingest_busy(exc)
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
                    and _normalize_observed_at(row.get("observed_at")) == stamp and _ws_equivalent(stored, identity)):
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
                                          copy_lookup=copy_lookup, allow_retained_escape=self._retained_escape_enabled)

    def _retained_escape_memo(self) -> dict:
        """Escape-covered identities for the session and its compression ancestors; never a source claim."""
        cache = self.__dict__.setdefault("_retained_escape_memos", {})
        merged = {}
        for session_id in [self._session_id, *self._identity_anchor_chain()]:
            key = self._replay_snapshot_metadata_key("retained_escape", session_id)
            if key in cache:
                for digest, sid in cache[key].items():
                    merged.setdefault(digest, sid)
                continue
            memo = {}
            try:
                payload = self._store.read_metadata_json(key)
                for item in payload if isinstance(payload, list) else ():
                    if not isinstance(item, dict) or not isinstance(item.get("identity_sha256"), str):
                        continue
                    try:
                        sid = int(item.get("store_id"))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if sid > 0:
                        memo[item["identity_sha256"]] = sid
            except (ValueError, TypeError):  # includes JSONDecodeError from the store's decoder
                logger.debug("LCM retained escape metadata corrupt; caching empty memo", exc_info=True)
                memo = {}
            except Exception:
                logger.debug("LCM retained escape metadata load failed", exc_info=True)
                continue  # retry later; leave this key uncached to prevent a partial write
            cache[key] = memo
            for digest, sid in memo.items():
                merged.setdefault(digest, sid)
        return merged

    @staticmethod
    def _retained_escape_digest(identity) -> str:
        return hashlib.sha256(json.dumps(list(identity), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _record_retained_escape(self, escaped) -> None:
        if not escaped:
            return
        try:
            memo = dict(self._retained_escape_memo())
            key = self._replay_snapshot_metadata_key("retained_escape")
            if key not in self._retained_escape_memos:
                return  # this call's metadata load failed
            for sid, row in self._store.get_batch([sid for sid, _role, _kind in escaped]).items():
                for form in self._stored_row_forms(row):
                    memo[self._retained_escape_digest(form)] = sid
            recent = set(sorted(set(memo.values()))[-256:])  # cap rows, not their alternate forms
            memo = {digest: sid for digest, sid in memo.items() if sid in recent}
            rows = [{"store_id": sid, "identity_sha256": digest} for digest, sid in sorted(memo.items())]
            self._store.write_metadata_json([key], json.dumps(rows, sort_keys=True), skip_unchanged=True)
            self._retained_escape_memos[key] = memo
        except Exception:
            logger.debug("LCM retained escape metadata write failed", exc_info=True)

    def _retained_escape_covered_view(self, message, frontier=None) -> bool:
        if not self._retained_escape_enabled:
            return False
        memo = self._retained_escape_memo()
        if not memo:
            return False
        frontier = self._store_complete_frontier() if frontier is None else frontier
        for strip_carrier in (False, True):
            identity = self._message_replay_identity(message, strip_carrier=strip_carrier)
            if 0 < memo.get(self._retained_escape_digest(identity), 0) <= frontier:
                return True
        return False

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

    def _identity_anchor_covered_view(self, message, full_map, frontier=None) -> bool:
        """#563: a live user row that no store row maps and whose text is exactly a recorded composite
        (R2 witness) or its survivor, every constituent already covered (at or below the frontier)."""
        if id(message) not in full_map and self._retained_escape_covered_view(message, frontier):
            return True  # #1162: any role, including an unmapped host copy after a cold resume
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
        frontier = self._store_complete_frontier()
        end = len(chunk)
        if all(self._identity_anchor_covered_view(message, full_map, frontier) for message in chunk):
            while end < len(candidates) and self._identity_anchor_covered_view(candidates[end], full_map, frontier):
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
            if getattr(self, "_ingest_ledger_observations", None) is not None:
                self._ingest_ledger_observations.extend((sid, kind, stamp) for group in groups
                                                       for sid, kind, _related, _ordinal, stamp in group if kind == "alt_stamp")
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

    def _identity_anchor_record_versions(self, stored, *, after_commit=None) -> None:
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
        if after_commit is None:
            self._identity_anchor_versions = versions
        else:
            after_commit.append(lambda: setattr(self, "_identity_anchor_versions", versions))
        if groups:
            self._store.add_message_relations(groups)

    def _identity_anchor_record_composite_of(self, stored_ids) -> None:
        """Provenance for stored todo composites; never participates in replay decisions.

        A stored user row gets ``composite_of`` rows only when exactly one decomposition of its text fits
        either rule, counted across both, with every part an earlier stored user row of its own session:
        (i) the LEADING rendered todo block (the #516 span: header, item lines, optional reload notice),
        unstored, then whole stored rows; (ii) whole stored rows only, the first a stored todo row. Parts join with exactly ``"\\n\\n"``. Ambiguity or an over-budget search
        writes nothing. A part text held by more than one row resolves by the host's consecutive-turn merge: several
        held parts link only to a unique run of consecutive session rows in text order, one held part only to the row
        immediately before C (``_composite_of_holders``); anything else writes nothing."""
        from .reconcile import _PRESERVED_TODO_CONTEXT_PREFIX as header, _todo_annotation_span

        groups = []
        for store_id, row in self._store.get_batch(stored_ids).items():
            content = row.get("content")
            if (row.get("role") != "user" or not isinstance(content, str)
                    or header not in content or "\n\n" not in content):
                continue
            by_text = defaultdict(list)
            for sid, text in self._store._conn.execute(
                "SELECT store_id, content FROM messages WHERE conversation_id = ? AND session_id = ? "
                "AND role = 'user' AND store_id < ? AND content != '' AND instr(?, content) > 0",
                (row["conversation_id"], row["session_id"], store_id, content),
            ):
                by_text[text].append(sid)
            fits = _composite_of_splits(content, by_text, header, _todo_annotation_span)
            if fits is None or len(fits) != 1:
                continue
            choices = [by_text[part] for part, held in fits[0] if held]
            if all(len(holders) == 1 for holders in choices):
                chosen = [holders[0] for holders in choices]
                if len(set(chosen)) != len(chosen):
                    continue  # one stored occurrence never stands for two parts
            else:  # identical texts held by several rows: the host merged CONSECUTIVE turns (ruling)
                order = [sid for (sid,) in self._store._conn.execute(
                    "SELECT store_id FROM messages WHERE session_id = ? AND store_id < ? ORDER BY store_id",
                    (row["session_id"], store_id))]
                chosen = _composite_of_holders(choices, order)
                if chosen is None:
                    continue
            held_ids = iter(chosen)
            groups.append([(store_id, "composite_of", next(held_ids) if held else None,
                            ordinal, row.get("observed_at")) for ordinal, (part, held) in enumerate(fits[0])])
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


def _composite_of_splits(content: str, by_text, header: str, todo_span) -> list[list[tuple[str, bool]]] | None:
    """The decompositions of ``content`` that composite_of rule (i) or (ii) allows, as ``[(part, stored)]``.
    None when a search is over budget or capped, so more splits may exist. A part whose text several rows
    hold still yields its split here; the caller resolves its holder (``_composite_of_holders``) or writes nothing."""
    texts = set(by_text)
    whole = _decompositions(content, texts, partial=False)
    if whole is None or len(whole) >= _MAX_DECOMPOSITIONS:
        return None
    found = [[(part, True) for part in parts] for parts, _rest in whole if parts[0].startswith(header)]
    if not content.startswith(header):
        return found
    # Rule (i): the scaffold is exactly the rendered todo block the #516 identity cut strips; the header
    # alone (no item line) is not a rendered block. It never holds a whole stored row (a stored todo row is
    # rule (ii)), and the host's merge puts exactly one "\n\n" behind it.
    end = todo_span(content, 0)
    scaffold, rest = content[:end], content[end + 2:]
    if (scaffold.split("\n", 1)[0] != header or "\n" not in scaffold or not content.startswith("\n\n", end) or not rest
            or any(scaffold == text or scaffold.startswith(text + "\n\n") or scaffold.endswith("\n\n" + text)
                   or "\n\n" + text + "\n\n" in scaffold for text in texts)):
        return found
    if rest in texts:
        found.append([(scaffold, False), (rest, True)])
    tails = _decompositions(rest, texts, partial=False)
    if tails is None or len(tails) >= _MAX_DECOMPOSITIONS:
        return None
    return found + [[(scaffold, False)] + [(part, True) for part in parts] for parts, _rest in tails]


def _composite_of_holders(choices: list[list[int]], order: list[int]) -> list[int] | None:
    """One stored row per held part when some part's text is held by several rows of C's session
    (``order``: the session's store ids before C, every role). Several parts: exactly one choice whose
    rows are consecutive in store order with no row of any role between them, in text order. One part:
    the row immediately before C, when it holds that text. Anything else: None (no relation)."""
    if len(choices) == 1:
        return [order[-1]] if order and order[-1] in choices[0] else None
    position = {sid: index for index, sid in enumerate(order)}
    rest = [set(holders) for holders in choices[1:]]  # O(1) membership: the walk stays linear in history
    runs = []
    for first in choices[0]:
        run, at = [first], position[first]
        for holders in rest:
            nxt = order[at + 1] if at + 1 < len(order) else None
            if nxt not in holders:
                break
            run.append(nxt)
            at += 1
        else:
            runs.append(run)
            if len(runs) > 1:
                return None
    return runs[0] if len(runs) == 1 else None
