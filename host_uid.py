"""v0.26.0 slice A: host ``message_uid`` SHADOW bindings (design REVISION 3). After #436 decides an ingest,
each uid-bearing host dict gets one R3-3 class and the row #436 chose goes to the droppable ``host_uid_bindings``
table. Nothing feeds back into ingest, replay, emission or commit; every exception is caught, counted and fails
open. Event counter keys are ``[replay.]class.outcome[.reason]``; ``skipped.no_uid`` stays in memory, so a no-uid
host writes nothing. The shadow GATE is counted per binding in the table (``first_check`` / ``disagree_seen``),
so a replayed prefix never inflates it. ``LCM_HOST_MESSAGE_UID=off`` does nothing; ``on`` runs as ``shadow``.
From the first release, a persisted lineage key is ``<home_tag>:<root>`` (#836): one database shared by several
Hermes homes never mixes two profiles' roots of the same session id."""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from .config import host_message_uid_mode
from .host_uid_emit import engine_uid, identity_emit_enabled

logger = logging.getLogger(__name__)

HOST_UID_COUNTER_KEY = "host_uid:counters"
_MAX_UID_CHARS = 256
_MAX_LINEAGE_HOPS = 256
_FORK_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")
_AGREE_OUTCOMES = {"agree", "version_new", "agree_new", "agree_bind"}
_NO_STATE_DB = (None, "unresolved")  # cached by identity: a host without state.db is not re-read
_LINEAGE_CACHE_CAP = 512  # entries; the oldest is dropped first


def _valid_uid(value) -> bool:
    return isinstance(value, str) and 0 < len(value) <= _MAX_UID_CHARS


def _is_fork_child(row: dict) -> bool:
    """Hermes ``_is_explicit_fork_child_row(include_reset=True)``; a marker counts only when it names the parent."""
    if row.get("source") == "tool":
        return True
    cfg = row.get("model_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            return False
    if not isinstance(cfg, dict):
        return False
    markers = tuple(cfg.get(key) for key in _FORK_MARKERS)
    parent_id = row.get("parent_session_id")
    return parent_id in markers if parent_id else any(marker is not None for marker in markers)


def _outcome(key: str) -> Optional[str]:
    outcome = (key.removeprefix("replay.").split(".") + [""])[1]
    return "agree" if outcome in _AGREE_OUTCOMES else "disagree" if outcome == "disagree" else None


def _around(sorted_values: list, value) -> tuple:
    """The nearest entries strictly before and after ``value`` in a sorted list (None: none)."""
    lo, hi = bisect.bisect_left(sorted_values, value), bisect.bisect_right(sorted_values, value)
    return sorted_values[lo - 1] if lo else None, sorted_values[hi] if hi < len(sorted_values) else None


def _fmt_counts(counts: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items())
                    if isinstance(value, int) and value) or "(none)"


class HostUidShadowMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``_state_db_path`` and the reconcile identity helpers."""

    def _host_uid_lineage_key(self, session_id=None) -> tuple:
        """R3-4: ``("<home_tag>:<root>", None)``, the root by Hermes' own walk (``_session_turn_lease_key_on_conn``:
        up while the parent ended by compression and the row is no fork child), else ``(None, "read_error" |
        "unresolved")`` (no state.db, missing row, cycle, over 256 hops). ``home_tag`` (#836) = 16 hex of the
        sha256 of the resolved state.db path. Cached per (state.db, session id): a key, or a missing state.db."""
        session_id = str(self._session_id if session_id is None else session_id or "")
        cache = self.__dict__.setdefault("_host_uid_lineage_cache", {})
        try:
            path = Path(self._state_db_path())
            key = (str(path), session_id)  # keyed by the home too: a profile switch changes the state.db
            if key in cache:
                return cache[key]
            found = self._host_uid_read_lineage(path, session_id)
            if found[0] is not None:  # a resolve error is a read error below, never an exception into ingest
                tag = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
                found = (f"{tag}:{found[0]}", None)
        except Exception as exc:  # host DB drift: no root, no binding; counted as an error
            self._host_uid_read_error = exc
            return None, "read_error"
        if found[0] is not None or found is _NO_STATE_DB:
            cache[key] = found
            while len(cache) > _LINEAGE_CACHE_CAP:
                cache.pop(next(iter(cache)))
        return found

    def _host_uid_read_lineage(self, path: Path, session_id: str) -> tuple:
        """One uncached read of the lineage root. Only FileNotFoundError means a missing state.db (unresolved,
        never an error); PermissionError and every other OSError propagate as a read error, never cached."""
        try:
            os.stat(path)
        except FileNotFoundError:
            return _NO_STATE_DB
        root = None
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            def read(sid: str) -> Optional[dict]:
                row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
                return dict(row) if row else None

            current, seen = read(session_id) if session_id else None, {session_id}
            for _ in range(_MAX_LINEAGE_HOPS + 1):
                parent_id = str(current.get("parent_session_id") or "") if current else ""
                if current is None or parent_id in seen:  # missing row or a cycle: unresolved
                    break
                parent = read(parent_id) if parent_id and not _is_fork_child(current) else None
                if parent is None or parent.get("end_reason") != "compression":
                    root = str(current["id"])
                    break
                seen.add(parent_id)
                current = parent
        finally:
            conn.close()
        return (root, None) if root is not None else (None, "unresolved")

    def _mint_engine_uids(self, generated) -> None:
        """R4-1 (B1's gate, a resolved lineage): ``[(row, kind, basis | None, proof_kind)]`` in emitted order gets
        deterministic engine uids, the ordinal counted per (kind, basis); ``basis`` None = the content's sha256.
        Pending until the compress that returns them records them; nothing else about the row changes."""
        try:
            if not generated or not identity_emit_enabled():
                return
            lineage = self._host_uid_lineage_key()[0]
            if lineage is None:
                return
            seen, pending = Counter(), self.__dict__.setdefault("_engine_uids_pending", {})
            for row, kind, basis, proof_kind in generated:
                basis = basis or hashlib.sha256(str(row.get("content") or "").encode("utf-8")).hexdigest()
                row["message_uid"] = uid = engine_uid(lineage, kind, basis, seen[(kind, basis)])
                seen[(kind, basis)] += 1
                pending[uid] = (lineage, proof_kind)
        except Exception as exc:  # fail open: an unminted row is a no-uid row, as at the base
            self._host_uid_count(Counter(errors=1), exc)

    def _host_uid_record_engine(self, emitted) -> None:
        """R4-2: the engine uids ``emitted`` carries (top level or absorbed) go to ``host_uid_bindings`` as
        ``kind='engine'``, ``store_id=0``, once per process; the shadow write path, fail-open and counted. A
        bypassed session writes no LCM state, so its marker's uid is minted but never recorded."""
        pending = self.__dict__.get("_engine_uids_pending")
        if not pending:
            return
        try:
            if self._bypasses_lcm_context_management():
                pending.clear()
                return
            done = self.__dict__.setdefault("_engine_uids_recorded", set())
            present = {uid for row in emitted if isinstance(row, dict)
                       for uid in [row.get("message_uid"), *(row.get("_absorbed_message_uids") or ())]
                       if isinstance(uid, str)}
            records: dict = defaultdict(list)
            for uid in (present & pending.keys()) - done:
                lineage, proof_kind = pending[uid]
                records[lineage].append((0, uid, "engine", proof_kind))
            pending.clear()
            for lineage, rows in records.items():
                self._store.add_host_uid_bindings(lineage, rows)
                done.update(uid for _sid, uid, _kind, _proof in rows)
        except Exception as exc:
            self._host_uid_count(Counter(errors=1), exc)

    def _host_uid_engine_uids(self, uids) -> set:
        """The ``uids`` that are engine uids of this lineage: the durable record (one batched lookup) plus the
        uids this engine minted and has not recorded yet. Fails open to the in-memory set."""
        uids = {uid for uid in uids if isinstance(uid, str)}
        known = uids & (set(self.__dict__.get("_engine_uids_pending") or ()) | self.__dict__.get(
            "_engine_uids_recorded", set()))
        try:
            lineage = self._host_uid_lineage_key()[0]
            if lineage is not None and uids - known:
                known |= set(self._store.host_uid_bindings_for(lineage, uids - known, ("engine",)))
        except Exception as exc:
            self._host_uid_count(Counter(errors=1), exc)
        return known

    def _host_uid_capture(self, messages, identity_messages, start: int, cursor: int, plan, tool_segment,
                          session_id=None):
        """Read the uids off the HOST dicts after #436 decided and before the INSERT drops unknown keys."""
        if host_message_uid_mode() == "off":
            return None
        try:
            decided = set(range(max(0, start), len(messages))) | set((plan or {}).get("matched") or ())  # F3: + prefix
            values = {idx: messages[idx].get("message_uid") for idx in sorted(decided) if idx < len(messages)}
            entries = {idx: value for idx, value in values.items() if value is not None}
            return {"entries": entries, "no_uid": len(values) - len(entries), "messages": messages,
                    "identity": identity_messages, "cursor": cursor, "plan": plan or {},
                    "tool_segment": set(tool_segment or ()), "session_id": session_id}
        except Exception as exc:
            self._host_uid_count(Counter(errors=1), exc)
            return None

    def _host_uid_shadow(self, capture, stored_at=None, remainders=()) -> None:
        """Classify and bind one ingest's decided dicts; ``stored_at`` = {index: new store id}."""
        if capture is None:
            return
        delta, error = Counter({"skipped.no_uid": capture["no_uid"]}), None
        try:
            error = self._host_uid_classify(capture, stored_at or {}, set(remainders or ()), delta)
        except Exception as exc:
            delta["errors"] += 1
            error = exc
        try:
            self._host_uid_count(delta, error)
        except Exception:  # pragma: no cover - counting never blocks ingest
            logger.debug("LCM host-uid count failed")

    def _host_uid_classify(self, capture, stored_at: dict, remainders: set, delta: Counter):
        valid = {idx: uid for idx, uid in capture["entries"].items() if _valid_uid(uid)}
        delta["skipped.invalid_uid"] += len(capture["entries"]) - len(valid)
        if not valid:
            return None
        lineage, problem = self._host_uid_lineage_key(capture["session_id"])
        if lineage is None:
            delta[f"skipped.no_lineage_root.{problem}"] += len(valid)
            if problem == "read_error":
                delta["errors"] += 1
                return getattr(self, "_host_uid_read_error", None)
            return None
        store, messages, identity = self._store, capture["messages"], capture["identity"]
        plan, cursor = capture["plan"], capture["cursor"]
        replayed, matched = plan.get("replayed") or set(), plan.get("matched") or {}

        def absorbed(message) -> list:
            values = message.get("_absorbed_message_uids")
            return [uid for uid in values if _valid_uid(uid)] if isinstance(values, (list, tuple)) else []

        present = {uid for idx, uid in valid.items() for uid in [uid, *absorbed(messages[idx])]}
        bindings, engine = store.host_uid_bindings_for(lineage, present), set(
            store.host_uid_bindings_for(lineage, present, ("engine",)))
        by_store: dict = defaultdict(set)  # store_id -> uids bound to it (F6: built once, kept current)
        for uid, items in bindings.items():
            for sid, _kind in items:
                by_store[sid].add(uid)
        fetched: dict = {}
        self._host_uid_fetch(by_store, fetched)  # one batched read for the whole decided range
        targets = {idx: int(matched[idx][0]["store_id"]) for idx in valid
                   if idx not in stored_at and len(matched.get(idx) or ()) == 1}
        held = store.host_uid_uids_of_stores(lineage, set(targets.values()))
        writes, checks, aliases = [], [], []

        def bind(store_id, uid, kind, proof) -> None:
            writes.append((store_id, uid, kind, proof))
            bindings.setdefault(uid, []).append((int(store_id), kind))
            by_store[int(store_id)].add(uid)

        for idx, uid in sorted(valid.items()):
            stored = stored_at.get(idx)
            rows = matched.get(idx) if stored is None or idx in remainders else None
            if engine & {uid, *absorbed(messages[idx])}:  # GENERATED (R3-3 class 2): events only, no gate check
                delta[self._host_uid_generated(uid, absorbed(messages[idx]), engine, bindings, rows, stored,
                                               idx in remainders)] += 1
                continue
            if idx in remainders or (rows and len(rows) > 1):  # COMPOSITE / REMAINDER (F3: prefix included)
                ids = {int(row["store_id"]) for row in rows or ()} | ({int(stored)} if stored is not None else set())
                agree = all(not bindings.get(present) or ids & {sid for sid, _k in bindings[present]}
                            for present in [uid, *absorbed(messages[idx])])
                outcome = "agree" if agree else "disagree"
                delta[f"composite.{outcome}.remainder" if idx in remainders else f"replay.composite.{outcome}"] += 1
                continue
            bound = [sid for sid, _kind in bindings.get(uid, ())]
            canonical = next((sid for sid, kind in bindings.get(uid, ()) if kind == "canonical"), bound[0] if bound else None)
            unmapped = not rows and stored is None and (idx < cursor or idx in capture["tool_segment"] or idx in replayed)
            if bound:  # BOUND: compared against the payload-matching version (R1-1); the gate marks that binding
                hit = self._host_uid_bytes_match(bound, identity[idx], fetched) if not rows else None
                if stored is not None:  # VERSION_NEW is an event only; a stored duplicate is a replay check
                    delta["bound.disagree.stored_despite_match" if hit is not None else "bound.version_new"] += 1
                    if hit is not None:
                        checks.append((uid, hit, False))
                    else:
                        bind(stored, uid, "version", "version_new")
                elif rows:  # #436 replayed onto one row: AGREE when it is a row bound to this uid
                    target = int(rows[0]["store_id"])
                    delta["replay.bound.agree.replay" if target in bound else "replay.bound.disagree.replay_other_row"] += 1
                    checks.append((uid, target if target in bound else canonical, target in bound))
                elif unmapped and hit is None and (merge := self._host_uid_lcm_merge(
                        messages[idx], [uid, *absorbed(messages[idx])], bindings, fetched)):
                    delta["replay.composite.agree.lcm_merge"] += 1  # LCM's own site-1 merge (R3-2)
                    checks.extend((part, sid, True) for part, sid in merge)
                elif unmapped:  # a replay with no row map: AGREE when a bound row holds these bytes
                    delta["replay.bound.agree.prefix_replay" if hit is not None else "replay.bound.disagree.prefix_replay"] += 1
                    checks.append((uid, canonical if hit is None else hit, hit is not None))
                else:
                    delta["skipped.not_stored"] += 1
            elif stored is not None:  # UNBOUND
                delta["unbound.agree_new"] += 1
                bind(stored, uid, "canonical", "stored_new")
            elif rows:
                target = targets[idx]
                if (held.get(target, set()) | by_store.get(target, set())) - {uid}:
                    aliases.append((idx, uid, target))
                else:
                    delta["replay.unbound.agree_bind"] += 1
                    bind(target, uid, "canonical", "anchor_replay")
            else:
                delta["replay.skipped.unmapped_replay" if unmapped else "skipped.not_stored"] += 1
        store.add_host_uid_bindings(lineage, writes)
        store.record_host_uid_checks(lineage, checks)
        if aliases:
            self._host_uid_alias_candidates(lineage, capture, stored_at, matched, bindings, fetched, aliases, delta)
        return None

    @staticmethod
    def _host_uid_generated(uid, absorbed, engine, bindings, rows, stored, remainder) -> str:
        """A dict naming an engine uid: AGREE when #436 stored and mapped nothing for it, ``.remainder`` for a
        carrier whose user remainder is the row its absorbed host uids are bound to (already stored, or the one
        #436 stored; every mapped row a row of those uids, else ``mapped_extra``); else DISAGREE. Nothing stored
        or mapped is the correct outcome for a generated row: the metric measures LCM storing or mapping its own
        rows wrongly, not recognition alone."""
        hosts = [u for u in [uid, *absorbed] if u not in engine]
        host_rows = {sid for h in hosts for sid, _k in bindings.get(h, ())}
        mapped = {int(row["store_id"]) for row in rows or ()}
        if stored is None and not mapped:
            return "generated.agree.remainder" if hosts and all(bindings.get(h) for h in hosts) else "generated.agree"
        if stored is not None and not remainder:
            return "generated.disagree.stored"
        if mapped - host_rows:  # #436 mapped a row no absorbed host uid names (a stored generated row: #534)
            return "generated.disagree.mapped_extra"
        ids = mapped | ({int(stored)} if stored is not None else set())
        if hosts and all(not bindings.get(h) or ids & {sid for sid, _k in bindings[h]} for h in hosts):
            return "generated.agree.remainder"
        return "generated.disagree.remainder"

    def _host_uid_fetch(self, store_ids, fetched: dict) -> None:
        """Rows cached across one ingest: one batched, chunked read of the ids not fetched yet."""
        missing = sorted({int(sid) for sid in store_ids if int(sid) not in fetched})
        fetched.update({sid: None for sid in missing})
        for start in range(0, len(missing), 500):
            fetched.update(self._store.get_batch(missing[start:start + 500]))

    def _host_uid_bytes_match(self, store_ids, identity_message, fetched: dict) -> Optional[int]:
        """The first bound row holding this dict's payload (its stored or host-rewrite form), else None."""
        if any(int(sid) not in fetched for sid in store_ids):
            self._host_uid_fetch(store_ids, fetched)
        identity = self._message_replay_identity(identity_message, strip_carrier=False)
        return next((int(sid) for sid in store_ids if fetched.get(int(sid)) is not None
                     and identity in self._stored_row_forms(fetched[int(sid)])), None)

    @staticmethod
    def _host_uid_lcm_merge(message, uids, bindings, fetched) -> Optional[list]:
        """Site 1: one bound row per uid, in order, whose stripped non-empty contents newline-join to this
        assistant dict's content (only ``content`` compared) -> [(uid, store_id), ...], else None."""
        content = message.get("role") == "assistant" and message.get("content")
        if len(uids) < 2 or not isinstance(content, str) or not all(uid in bindings for uid in uids):
            return None

        def walk(pos: int, joined: str, picked: list) -> Optional[list]:
            if pos == len(uids):
                return picked if joined == content else None
            for sid, _kind in bindings[uids[pos]]:
                part = (fetched.get(sid) or {}).get("content")
                part = part.strip() if isinstance(part, str) else None
                nxt = joined if not part else f"{joined}\n{part}" if joined else part
                if part is not None and content.startswith(nxt) and (
                        found := walk(pos + 1, nxt, picked + [(uids[pos], sid)])):
                    return found
            return None

        return walk(0, "", [])

    def _host_uid_alias_candidates(self, lineage, capture, stored_at, matched, bindings, fetched, aliases, delta):
        """R3-1 (shadow, observational): ``position_proof`` when the rows the nearest bound view neighbours map to
        IN THIS INGEST (stored now, the #436 row, the bound version whose bytes match, else the canonical) are the
        target row's nearest rows bound to a uid of the view (none on a side in both matches; one side bound)."""
        messages, identity = capture["messages"], capture["identity"]
        view = [(idx, message.get("message_uid")) for idx, message in enumerate(messages)
                if _valid_uid(message.get("message_uid"))]
        bindings.update(self._store.host_uid_bindings_for(lineage, {uid for _i, uid in view} - set(bindings)))
        bound_pos = [pos for pos, (_idx, uid) in enumerate(view) if bindings.get(uid)]
        rows = sorted({sid for _i, uid in view for sid, _kind in bindings.get(uid, ())})
        position = {idx: pos for pos, (idx, _uid) in enumerate(view)}
        resolved: dict = {}

        def resolve(pos):
            if pos is None or pos in resolved:
                return resolved.get(pos)
            idx, uid = view[pos]
            if stored_at.get(idx) is not None:
                return int(stored_at[idx])
            if len(matched.get(idx) or ()) == 1:
                return int(matched[idx][0]["store_id"])
            items = bindings[uid]
            hit = self._host_uid_bytes_match([sid for sid, _kind in items], identity[idx], fetched)
            resolved[pos] = hit if hit is not None else next((s for s, kind in items if kind == "canonical"), items[0][0])
            return resolved[pos]

        records = []
        for idx, uid, target in aliases:
            before, after = (resolve(pos) for pos in _around(bound_pos, position[idx]))
            proof = (before, after) == _around(rows, target) and (before is not None or after is not None)
            reason = "position_proof" if proof else "unknown"
            delta[f"replay.unbound.alias_candidate.{reason}"] += 1
            records.append((target, uid, "alias_candidate", reason))
        self._store.add_host_uid_bindings(lineage, records)

    def _host_uid_count(self, delta: Counter, error: Optional[BaseException] = None) -> None:
        delta = +delta
        if error is not None:  # one WARNING per engine, then DEBUG
            logged, self._host_uid_error_logged = getattr(self, "_host_uid_error_logged", False), True
            (logger.debug if logged else logger.warning)("LCM host-uid shadow failed (%s); ingest unaffected",
                                                         type(error).__name__)
        if not delta:
            return
        counts = self._host_uid_counters = getattr(self, "_host_uid_counters", None) or Counter()
        counts.update(delta)
        durable = {key: value for key, value in delta.items() if key != "skipped.no_uid"}
        if not durable:
            return

        def merged(record):
            record = dict(record) if isinstance(record, dict) else {}
            for key, value in durable.items():
                try:
                    record[key] = int(record.get(key) or 0) + value
                except (TypeError, ValueError, OverflowError):
                    record[key] = value
            return record

        try:  # never inside someone else's transaction: skipped and counted
            if getattr(self._store._conn, "in_transaction", False):
                raise sqlite3.OperationalError("host-uid tally skipped: the connection is in a transaction")
            self._store.update_metadata_json(HOST_UID_COUNTER_KEY, merged)
        except Exception as exc:
            counts["errors"] += 1
            logger.debug("LCM host-uid counter write failed (%s)", type(exc).__name__)

    def _host_uid_log_compaction_summary(self) -> None:
        """One INFO line per compaction, counts only; silent on a host that sent no uid."""
        try:
            counts = {k: v for k, v in (getattr(self, "_host_uid_counters", None) or {}).items()
                      if v and k != "skipped.no_uid"}
            if not counts or host_message_uid_mode() == "off":
                return
            agree = sum(v for k, v in counts.items() if _outcome(k) == "agree")
            disagree = sum(v for k, v in counts.items() if _outcome(k) == "disagree")
            logger.info("LCM host-uid shadow: events agree=%d disagree=%d counts=%s", agree, disagree,
                        _fmt_counts(counts))
        except Exception as exc:
            logger.debug("LCM host-uid summary failed (%s)", type(exc).__name__)


def host_uid_doctor_lines(engine: Any) -> list[str]:
    """Doctor ``host_uid`` section: counts only (gate per binding, event counts, errors, table size)."""
    try:
        durable, rows = engine._store.read_metadata_json(HOST_UID_COUNTER_KEY), engine._store.count_host_uid_bindings()
        gate, engine_rows = engine._store.host_uid_gate(), engine._store.count_host_uid_bindings(("engine",))
    except Exception as exc:
        durable, rows, gate = None, f"error: {type(exc).__name__}", []
        engine_rows = rows
    durable = durable if isinstance(durable, dict) else {}
    process = dict(getattr(engine, "_host_uid_counters", None) or {})
    errors = durable.get("errors") if isinstance(durable.get("errors"), int) else 0
    checked, agree = sum(c for c, _a in gate), sum(a for _c, a in gate)
    return [
        f"host_uid_mode: {host_message_uid_mode()}",
        f"host_uid_gate_checked: {checked}",
        f"host_uid_gate_agree: {agree}",
        f"host_uid_gate_disagree: {checked - agree}",
        "host_uid_gate_per_lineage: " + (" ".join(f"{c}/{a}/{c - a}" for c, a in gate[:10]) or "(none)")
        + " (checked/agree/disagree)",
        f"host_uid_event_counts: {_fmt_counts({k: v for k, v in durable.items() if k != 'errors'})}",
        f"host_uid_process_event_counts: {_fmt_counts({k: v for k, v in process.items() if k != 'errors'})}",
        f"host_uid_errors: {errors}" + (f" (process {process['errors']})" if process.get("errors") else ""),
        f"host_uid_bindings_rows: {'absent' if rows is None else rows}",
        f"host_uid_engine_rows: {'absent' if engine_rows is None else engine_rows}",
    ]
