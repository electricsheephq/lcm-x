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
import os
import sqlite3
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from .fresh_tail import tool_group_safe_end
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .store import _normalize_observed_at
from .tokens import count_messages_tokens

logger = logging.getLogger(__name__)

_CARRY_PREFIX = "identity_anchor_carry"
_RECENT_CAP = 16  # rows this process stored lately: the R5 current-turn window
_POOL_WINDOW = 256  # store ids either side of a stamp donor searched for a composite's constituents
_MAX_DECOMPOSITIONS = 3
_DECOMPOSE_BUDGET = 2048  # T3: prefixes visited per decomposition (ingest thread)
_DECOMPOSE_MAX_PARTS = 64
_MATCH_WORK_PER_ITEM, _MATCH_WORK_FLOOR = 32, 4096  # matching search budget per row + occurrence + key (see below)


def identity_anchor_enabled() -> bool:
    return (os.environ.get("LCM_IDENTITY_ANCHOR") or "").strip().lower() not in {"0", "false", "no", "off"}


def _decompositions(content: str, texts: set, *, partial: bool) -> Optional[list]:
    """Ordered splits of ``content`` into ``"\\n\\n"``-joined whole ``texts`` (at least two parts).
    ``partial`` also returns ``(parts, remainder)`` for a held prefix followed by a new remainder.
    Stops after ``_MAX_DECOMPOSITIONS`` (the caller only needs to know whether one is unique).
    Iterative and bounded (T3): more than ``_DECOMPOSE_BUDGET`` visited prefixes or
    ``_DECOMPOSE_MAX_PARTS`` parts returns None, "ambiguous": the row is stored whole."""
    out: list = []
    by_head: dict[str, list] = defaultdict(list)
    short = [text for text in texts if 0 < len(text) < 16]
    for text in texts:
        if len(text) >= 16:
            by_head[text[:16]].append(text)
    stack: list = []  # (pos, parts, texts left to try): the recursive walk, iteratively (T3)
    visited = 0

    def enter(pos: int, parts: list) -> bool:
        """One walk() call: False once the budget is spent."""
        nonlocal visited
        if len(out) >= _MAX_DECOMPOSITIONS:
            return True
        visited += 1
        if visited > _DECOMPOSE_BUDGET or len(stack) >= _DECOMPOSE_MAX_PARTS:
            return False
        if partial and parts and pos < len(content):
            out.append((list(parts), content[pos:]))
        stack.append((pos, parts, iter(by_head.get(content[pos:pos + 16], []) + short)))
        return True

    if not enter(0, []):
        return None
    while stack:
        pos, parts, pending = stack[-1]
        text = next(pending, None)
        if text is None:
            stack.pop()
            continue
        end = pos + len(text)
        if not content.startswith(text, pos):
            continue
        if end == len(content) and len(parts) >= 1:
            out.append((parts + [text], ""))
        elif content.startswith("\n\n", end) and end + 2 < len(content) and not enter(end + 2, parts + [text]):
            return None
    return out


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
            path = self._state_db_path()
            if session_id and path.exists():
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
                try:
                    current, seen = session_id, {session_id}
                    for _ in range(256):
                        row = conn.execute(
                            "SELECT p.id, p.end_reason FROM sessions c JOIN sessions p ON p.id = c.parent_session_id "
                            "WHERE c.id = ? LIMIT 1", (current,),
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

    def _identity_text(self, row) -> str:
        memo = self._identity_anchor_text_memo
        store_id = int(row.get("store_id") or 0)
        if store_id not in memo:
            memo[store_id] = self._message_replay_identity(row, stored_row=True)[1]
        return memo[store_id]

    def _identity_anchor_prematch(self, messages, identity_messages, cursor: int, audit_from: Optional[int] = None) -> Dict[str, Any]:
        """Rows at or after ``cursor`` recognised as replays of stored occurrences, plus the R3
        remainders to store, the relations to record and the R5 backfills. ``audit_from``: the host
        changed its list before the cursor from there (a positional cursor no longer proves those rows
        stored): a stamped row -- or an unstamped user row (#633) -- there that no stored occurrence
        explains moves ``plan["cursor"]`` back."""
        plan: Dict[str, Any] = {"replayed": set(), "remainders": {}, "relations": [], "carry": [], "backfill": [],
                                "cursor": cursor}
        self._identity_anchor_text_memo: dict[int, str] = {}
        n = len(messages)
        start = cursor if audit_from is None else max(0, min(audit_from, cursor))
        rewritten = self._identity_anchor_rewritten(messages, cursor)
        if rewritten:  # R6: a rewritten host object is examined like a new row (a merge survivor, ...)
            start = min(start, min(rewritten))
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
        by_stamp: dict[float, list] = defaultdict(list)
        for row in rows:
            by_stamp[float(row["observed_at"])].append((row, self._stored_row_forms(row)))
        identities: dict[int, Optional[tuple]] = {}

        def identity_at(idx: int) -> Optional[tuple]:
            if idx not in identities:
                message = identity_messages[idx]
                identity = None if self._identity_is_lcm_scaffold(message) else self._message_replay_identity(
                    message, strip_carrier=False
                )
                identities[idx] = None if identity is None or _lossy(identity) else identity
            return identities[idx]

        view_counts: dict = {}

        def view_count(identity) -> int:  # occurrences in the WHOLE host view (multiplicity evidence)
            if not view_counts:
                for i, message in enumerate(identity_messages):
                    key = identities.get(i) if i in identities else self._message_replay_identity(message, strip_carrier=False)
                    view_counts[key] = view_counts.get(key, 0) + 1
            return view_counts.get(identity, 0)

        def shown(idx: int) -> Counter:  # B-ID-1: the view's own occurrences no stored row has matched yet
            return Counter((stamps.get(i), identity_at(i)) for i in range(n)  # unstamped: (None, form)
                           if i != idx and i not in matched and identity_at(i) is not None)

        consumed: set[int] = set()
        matched: dict[int, list] = {}
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
                row = self._identity_anchor_ws_row(identity, stamps[idx], [r for r, _f in by_stamp[stamps[idx]]], consumed)
                if row is not None:  # R1-ws: the host persisted this occurrence without its edge whitespace
                    plan.setdefault("ws", []).append((row, identity_messages[idx]))
                    self._identity_anchor_take(idx, [row], consumed, matched, plan)
                    continue
                self._identity_anchor_user_row(idx, identity, stamps[idx], by_stamp, consumed, matched, plan, view_count,
                                               shown)
            elif idx not in stamps and idx not in plan["replayed"]:  # D-D plan (ii): H1 merged LCM's carrier
                group = self._identity_anchor_carrier_group(identity_messages[idx], consumed)
                if group is not None:
                    self._identity_anchor_take(idx, group, consumed, matched, plan)
        for idx in sorted(plan["replayed"]):  # a replayed survival-fit projection: its stored replies follow it
            if len(matched.get(idx, ())) == 1 and callable(getattr(self, "_survival_projection_followers", None)):
                for k, row in self._survival_projection_followers(identity_messages, idx, matched[idx][0], stamps):
                    if k in plan["replayed"] or int(row["store_id"]) in consumed:
                        break
                    self._identity_anchor_take(k, [row], consumed, matched, plan)
        if start < cursor:
            self._identity_anchor_audit(messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan)
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
        return plan

    def _identity_anchor_audit(self, messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan) -> None:
        """Rows in ``[start, cursor)`` of a list the host changed before the cursor: a stamped host row,
        or an unstamped user row the host inserted there (#633: a /steer row; by content alone), that
        no stored occurrence explains (key, witness, alias, or a stored copy of its content -- up to
        edge whitespace, under another stamp or none -- in the tail of this session or a verified
        ancestor, each copy used once) and no ignore pattern drops was never stored. The cursor moves back
        to the first such row; every other row of that range stays a replay (today's positional proof)."""
        from .reconcile import _proof_user_identity

        span = max(64, 2 * (cursor - start))
        held: dict = defaultdict(list)
        for session in [str(self._session_id), *self._identity_anchor_chain()]:
            for row in self._store.get_session_tail(session, limit=span):
                if int(row["store_id"]) not in consumed:
                    held[_proof_user_identity(self._message_replay_identity(row, stored_row=True))].append(row)
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
                plan.setdefault("ws", []).append((ws, identity_messages[idx]))
                continue
            if copies:
                consumed.add(int(copies[0]["store_id"]))
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
                copies = [row for row in held.get(key, ()) if int(row["store_id"]) not in consumed]
                if len(copies) > reserved[key]:
                    consumed.add(int(copies[-1]["store_id"]))
                    continue
                missed.append(idx)
        if missed:
            plan["cursor"] = min(missed)
            plan["positional"] = {idx for idx in range(min(missed), cursor) if idx not in missed}
            plan["replayed"].update(plan["positional"])
            logger.info("LCM identity-anchor: host changed its list before the cursor; %d unstored rows from %d: session=%s",
                        len(missed), min(missed), self._session_id)

    def _identity_anchor_inserted(self, messages, start: int, cursor: int) -> set:
        """#633: indexes in ``[start, cursor)`` the host inserted into the last list LCM ingested (Hermes
        0.21.2+ inserts a /steer row after the newest tool result), by an identity diff of the two lists
        from ``start``. A row that replaces, rewrites or re-merges an ingested one is not inserted."""
        before = getattr(self, "_last_active_replay_source_identities", None)
        if not before:
            return set()
        now = [self._message_replay_identity(message, strip_carrier=False) for message in messages[start:]]
        opcodes = SequenceMatcher(None, list(before[start:]), now, autojunk=False).get_opcodes()
        return {start + j for tag, _i1, _i2, j1, j2 in opcodes if tag == "insert" for j in range(j1, j2)
                if start + j < cursor}

    def _identity_anchor_user_row(self, idx, identity, stamp, by_stamp, consumed, matched, plan, view_count, shown) -> None:
        """R2/R3/R5 for one unmatched user row: a recorded witness, an exact unique decomposition,
        the H1 unstable current-turn stamp, else a held head plus a new remainder. Else: new."""
        content = identity[1]
        stamped = [row for row, _forms in by_stamp.get(stamp, ()) if row.get("role") == "user"]
        donors = [row for row in stamped if int(row["store_id"]) not in consumed]
        # R2 witness: a composite LCM itself saw at this stamp, or its rewritten survivor (U alone).
        for group in self._identity_anchor_witnesses(stamped, stamp):
            texts = [self._identity_text(row) for row in group]
            if content == "\n\n".join(texts):
                view = group
            else:
                view = [row for row, text in zip(group, texts) if text == content and row.get("observed_at") != stamp]
                if len(view) != 1 or texts.count(content) != 1:
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
        texts = {self._identity_text(row) for row in pool}
        donor_texts = {self._identity_text(row) for row in donors}
        group, ambiguous = self._identity_anchor_compose(content, texts, pool, donors, consumed)
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
            group = self._identity_anchor_assign(parts, pool, donors, consumed)
            if group is not None:
                consumed.update(int(row["store_id"]) for row in group)
                matched[idx] = group
                plan["remainders"][idx] = (rest, stamp, group)

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
                       and self._message_replay_identity(row, stored_row=True) == identity]
            if len(members) == 1:
                plan["backfill"].append((int(members[0]["store_id"]), stamp))
                return self._identity_anchor_take(idx, members, consumed, matched, plan)

    def _identity_anchor_compose(self, content, texts, pool, donors, consumed) -> tuple:
        """R2 form (i): ``(group, ambiguous)``; ``group`` is the one exact, unique, ordered decomposition
        of ``content`` into stored occurrences (a stamp donor among them), each used once."""
        donor_texts = {self._identity_text(row) for row in donors}
        found = _decompositions(content, texts, partial=False)
        if found is None:  # T3: search budget spent: ambiguous, the composite is stored whole
            return None, True
        full = [parts for parts, rest in found if not rest and donor_texts & set(parts)]
        group = self._identity_anchor_assign(full[0], pool, donors, consumed) if len(full) == 1 else None
        return group, len(full) > 1

    def _identity_anchor_take(self, idx, rows, consumed, matched, plan) -> None:
        consumed.update(int(row["store_id"]) for row in rows)
        matched[idx] = list(rows)
        plan["replayed"].add(idx)

    def _identity_anchor_witnesses(self, donors, stamp) -> list:
        """Recorded composite groups at ``stamp`` headed by a donor, constituents in order."""
        groups: dict[tuple, list] = defaultdict(list)
        for rel in self._store.get_message_relations([int(row["store_id"]) for row in donors], "composite"):
            if rel["observed_at"] == stamp and rel["related_store_id"] is not None:
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

    def _identity_anchor_assign(self, parts, pool, donors, consumed) -> Optional[list]:
        """Bind each part to one stored occurrence (a donor for a donor's text first), each used once;
        ``pool`` holds no row the host view shows as its own occurrence (B-ID-1)."""
        taken: set[int] = set(consumed)
        donor_ids = {int(row["store_id"]) for row in donors}
        group = []
        for text in parts:
            options = sorted((row for row in pool if self._identity_text(row) == text
                              and int(row["store_id"]) not in taken),
                             key=lambda row: (int(row["store_id"]) not in donor_ids, int(row["store_id"])))
            if not options:
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
        """R5 NULL backfill for the reconciled prefix: a stamped row whose occurrence-bound mapped row
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
                                       accounted_ids=()) -> Optional[list]:
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
                      if "\n\n".join(self._identity_text(row) for row in group) == content] if donors else []
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
                group, _ambiguous = self._identity_anchor_compose(
                    content, {self._identity_text(row) for row in pool}, pool, donors, set()
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
                                          max(budget, count_messages_tokens(chunk)), accounted_ids)

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
            texts = [self._identity_text(row) for row in group]
            view = group if content == "\n\n".join(texts) else [row for row, text in zip(group, texts) if text == content]
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
        full_map, self._identity_anchor_text_memo = self._current_compress_store_ids_by_message_id or {}, {}
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
        for store_id, stamp in plan["backfill"] if remainder_ids is None else ():
            self._store.backfill_observed_at(store_id, stamp)
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
                     if idx not in plan.get("explained", ())]
        if not rewritten:
            return
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
        for identity, store_id, message in stored:
            if identity is not None and identity[0] == "user":
                recent.append((self._session_id, identity, int(store_id), _normalize_observed_at(message.get("timestamp"))))
        self._identity_anchor_recent = recent[-_RECENT_CAP:]


def _match_occurrences(rows, keys_of, occurrences) -> dict:
    """``{store_id: occurrence}``: a maximum matching of stored ``rows`` (store order; a row listed twice, e.g.
    under an alternate stamp, is one row) to view ``occurrences`` ``[(occurrence, key)]`` (view order) by any key
    of ``keys_of(row)``, each side once. A row takes its first key with a free occurrence, else an augmenting
    path frees one; each key's rows then take its occurrences in store order <-> view order, so rows with one
    admissible key get what an in-order walk gave them. Deterministic; O(rows + occurrences + keys) memory.
    Rows with the same keys are one type and a search walks types, not rows. The searches of one call share a
    work budget (key and type visits); once it is spent the remaining rows take only a free key of their own, the
    result stays a valid deterministic matching that may fall short of maximum, and one WARNING (counts) is logged."""
    slots: dict = defaultdict(list)
    for occurrence, key in occurrences:
        slots[key].append(occurrence)
    merged: dict = {}
    for row in rows:
        merged.setdefault(int(row["store_id"]), set()).update(keys_of(row))
    kinds = {sid: tuple(sorted((k for k in ks if k in slots), key=repr)) for sid, ks in merged.items()}
    load: Counter = Counter()
    units: dict = defaultdict(Counter)  # type -> key -> its rows on that key
    matched: set = set()
    movable: dict = defaultdict(dict)  # key -> multi-key types with a unit on it (ordered set); nothing else moves
    dead: set = set()  # never an endpoint or a waypoint again (load never falls; Kuhn's lemma for failed searches)
    budget, work, spent = _MATCH_WORK_PER_ITEM * (len(kinds) + len(occurrences) + len(slots)) + _MATCH_WORK_FLOOR, 0, False
    for sid, own in kinds.items():
        found = next((k for k in own if load[k] < len(slots[k])), None)
        parent: dict = {}
        if found is None and own and not spent:
            queue = []
            for k in own:
                if k not in dead:
                    parent[k] = (None, None)
                    (queue.append if movable[k] else dead.add)(k)  # full with nothing that can leave: dead
            seen: set = set()
            for k in queue:  # BFS over keys; the queue grows while it is walked
                gone = []
                for kind in movable[k]:
                    spent = work == budget
                    if spent:
                        break
                    work += 1
                    others = [other for other in kind if other != k and other not in dead]
                    if not others:
                        gone.append(kind)  # stuck on k for good
                    elif kind not in seen:
                        seen.add(kind)
                        for other in others:
                            if other in parent:
                                continue
                            spent = work == budget
                            if spent:
                                break
                            work += 1
                            parent[other] = (kind, k)
                            if load[other] < len(slots[other]):
                                found = other
                                break
                            (queue.append if movable[other] else dead.add)(other)
                    if found is not None or spent:
                        break
                for kind in gone:
                    del movable[k][kind]
                if found is not None or spent:
                    break
            if found is None and not spent:
                dead.update(parent)
        if found is None:
            continue
        load[found] += 1
        kind, previous = parent.get(found, (None, None))
        while previous is not None:  # one unit of each type on the path moves one key along
            units[kind][previous] -= 1
            if not units[kind][previous]:
                movable[previous].pop(kind, None)
            units[kind][found] += 1
            movable[found][kind] = None
            found = previous
            kind, previous = parent[found]
        matched.add(sid)
        units[own][found] += 1
        if len(own) > 1:
            movable[found][own] = None
    given: dict = defaultdict(Counter)  # type -> key -> its rows given that key so far
    taken: Counter = Counter()
    result = {}
    for sid, kind in kinds.items():  # store order: a type's rows fill its keys by units, keys in order
        if sid in matched:
            k = next(k for k in kind if given[kind][k] < units[kind][k])
            given[kind][k] += 1
            result[sid] = slots[k][taken[k]]
            taken[k] += 1
    if spent:
        logger.warning("LCM identity-anchor matching spent its work budget %d: rows=%d occurrences=%d keys=%d matched=%d",
                       budget, len(kinds), len(occurrences), len(slots), len(result))
    return result


def _reserve_shown(pool, forms_of, occurrences, aliases=None) -> set:
    """B-ID-1: store ids of ``pool`` rows (store order) the host view shows as their own occurrences
    ``[(occurrence, (stamp, form))]``. A stamped row answers only its own stamp and its recorded alias stamps
    (``aliases``: store id -> stamps, B-ID-3), a NULL-stamped (legacy) row an unstamped occurrence. #583: then a
    NULL row LCM stored before the host stamped it answers a stamped occurrence of its form still unmatched,
    keyed by form alone (a row's keys are its forms, never its stamps: linear), in store order."""
    aliases = aliases or {}

    def keys_of(row) -> set:
        stamps = {_normalize_observed_at(row.get("observed_at")), *aliases.get(int(row["store_id"]), ())}
        return {(stamp, form) for stamp in stamps for form in forms_of(row)}

    first = _match_occurrences(pool, keys_of, occurrences)
    null = [row for row in pool if _normalize_observed_at(row.get("observed_at")) is None and int(row["store_id"]) not in first]
    taken = set(first.values())
    left = [(occurrence, ("null", key[1])) for occurrence, key in occurrences if key[0] is not None and occurrence not in taken]
    reserved = set(first)
    if null and left:  # over-reservation only stores a composite whole: duplication, never loss
        second = _match_occurrences(null, lambda row: {("null", form) for form in forms_of(row)}, left)
        reserved.update(second)
        rest = [row for row in null if int(row["store_id"]) not in second]
        if rest:
            reserved |= _reroute_for_null(pool, rest, first, taken | set(second.values()), dict(occurrences), keys_of, forms_of)
    return reserved


def _reroute_for_null(pool, rest, first, taken, key_of, keys_of, forms_of) -> set:
    """PR #590: augmentation from pass one for the NULL rows ``rest`` still unreserved. One takes an occurrence of
    its form a stamped row holds only when that row can move to a free occurrence of another of its OWN keys, so a
    stamped row is never unmatched. Free occurrences only fall here: a holder that cannot move never can, and
    leaves the index after one look. Bounded like ``_match_occurrences``; a spent budget keeps what was found."""
    free = Counter(key for occurrence, key in key_of.items() if key[0] is not None and occurrence not in taken)
    current = {int(row["store_id"]): key_of[first[int(row["store_id"])]] for row in pool
               if int(row["store_id"]) in first and key_of[first[int(row["store_id"])]][0] is not None}
    holders: dict = defaultdict(deque)  # form -> stamped rows holding an occurrence of it, store order
    rows = {int(row["store_id"]): row for row in pool}
    for sid, key in current.items():
        holders[key[1]].append(sid)
    out: set = set()
    budget, work = _MATCH_WORK_PER_ITEM * (len(pool) + len(key_of)) + _MATCH_WORK_FLOOR, 0
    for row in rest:
        for form in sorted(forms_of(row), key=repr):
            queue = holders.get(form)
            while queue and int(row["store_id"]) not in out and work < budget:
                sid = queue.popleft()
                work += 1
                target = next((k for k in sorted(keys_of(rows[sid]), key=repr) if k != current[sid] and free[k] > 0), None)
                if target is not None:  # the holder moves; this NULL row takes the occurrence it left
                    free[target] -= 1
                    current[sid] = target
                    holders[target[1]].append(sid)
                    out.add(int(row["store_id"]))
            if int(row["store_id"]) in out:
                break
        if work >= budget:
            logger.warning("LCM identity-anchor NULL re-route spent its work budget %d: rows=%d occurrences=%d reserved=%d",
                           budget, len(pool), len(key_of), len(out))
            break
    return out


def _composite_relation(group, stamp) -> list:
    """One composite relation group, keyed by the constituent whose host stamp the composite carries."""
    head = next((row for row in group if row.get("observed_at") == stamp), group[0])
    return [(int(head["store_id"]), "composite", int(row["store_id"]), ordinal, stamp) for ordinal, row in enumerate(group)]


def _lossy(identity) -> bool:
    from .reconcile import _has_lossy_redacted_identity

    return _has_lossy_redacted_identity(identity)


def _raw_remainder(message, remainder) -> Optional[str]:
    """R3's stored bytes: the survivor's raw content must be EXACTLY the held constituents' stored
    bytes, each joined by the host's ``"\n\n"``, then ``"\n\n"`` and the remainder; else None (the
    survivor is stored whole: visible duplication, never a lost or altered separator)."""
    rest, _stamp, group = remainder
    raw = normalize_content_value(message.get("content")) or ""
    head = "\n\n".join(normalize_content_value(row.get("content")) or "" for row in group) + "\n\n"
    return raw[len(head):] if raw.startswith(head) and raw[len(head):] == rest else None
