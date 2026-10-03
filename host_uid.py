"""v0.26.0 slice A: host ``message_uid`` SHADOW bindings (design REVISION 3). After #436 decides an ingest,
each uid-bearing host dict gets one R3-3 class and the row #436 chose goes to the droppable ``host_uid_bindings``
table. Nothing feeds back into ingest, replay, emission or commit; every exception is caught, counted and fails
open. Counter keys are ``class.outcome[.reason]``; ``skipped.no_uid`` stays in memory, so a no-uid host writes
nothing. ``LCM_HOST_MESSAGE_UID=off`` does nothing; ``on`` is reserved (slice C) and runs as ``shadow``."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from typing import Any, Optional

from .config import host_message_uid_mode

logger = logging.getLogger(__name__)

HOST_UID_COUNTER_KEY = "host_uid:counters"
_MAX_UID_CHARS = 256
_MAX_LINEAGE_HOPS = 256
_FORK_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")
_AGREE_OUTCOMES = {"agree", "version_new", "agree_new", "agree_bind"}


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
    outcome = (key.split(".") + [""])[1]
    return "agree" if outcome in _AGREE_OUTCOMES else "disagree" if outcome == "disagree" else None


def _fmt_counts(counts: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items())
                    if isinstance(value, int) and value) or "(none)"


class HostUidShadowMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``_state_db_path`` and the reconcile identity helpers."""

    def _host_uid_lineage_key(self) -> Optional[str]:
        """R3-4: the lineage root by Hermes' own walk (``_session_turn_lease_key_on_conn``): up while the parent
        ended by compression and the row is no fork child. None when unreadable, missing or over 256 hops."""
        session_id = str(self._session_id or "")
        cached = getattr(self, "_host_uid_lineage_cache", None)
        if cached is not None and cached[0] == session_id:
            return cached[1]
        root = None
        try:
            path = self._state_db_path()
            if not session_id or not path.exists():
                return None
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row
            try:
                def read(sid: str) -> Optional[dict]:
                    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
                    return dict(row) if row else None

                current, seen = read(session_id), {session_id}
                for _ in range(_MAX_LINEAGE_HOPS + 1):
                    if current is None:
                        break
                    parent_id = current.get("parent_session_id")
                    parent = None
                    if parent_id and parent_id not in seen and not _is_fork_child(current):
                        parent = read(str(parent_id))
                    if parent is None or parent.get("end_reason") != "compression":
                        root = str(current["id"])
                        break
                    seen.add(str(parent_id))
                    current = parent
            finally:
                conn.close()
        except Exception as exc:  # host DB drift or absence: no root, no binding
            logger.debug("LCM host-uid lineage read failed: %s", type(exc).__name__)
            return None
        if root is not None:  # only a completed read is cached
            self._host_uid_lineage_cache = (session_id, root)
        return root

    def _host_uid_capture(self, messages, identity_messages, start: int, cursor: int, plan, tool_segment):
        """Read the uids off the HOST dicts after #436 decided and before the INSERT drops unknown keys."""
        if host_message_uid_mode() == "off":
            return None
        try:
            values = {idx: messages[idx].get("message_uid") for idx in range(max(0, start), len(messages))}
            entries = {idx: value for idx, value in values.items() if value is not None}
            return {"entries": entries, "no_uid": len(values) - len(entries), "messages": messages, "identity": identity_messages,
                    "cursor": cursor, "plan": plan or {}, "tool_segment": set(tool_segment or ())}
        except Exception as exc:
            self._host_uid_count(Counter(errors=1), exc)
            return None

    def _host_uid_shadow(self, capture, stored_at=None, remainders=()) -> None:
        """Classify and bind one ingest's decided dicts; ``stored_at`` = {index: new store id}."""
        if capture is None:
            return
        delta, error = Counter({"skipped.no_uid": capture["no_uid"]}), None
        try:
            self._host_uid_classify(capture, stored_at or {}, set(remainders or ()), delta)
        except Exception as exc:
            delta["errors"] += 1
            error = exc
        try:
            self._host_uid_count(delta, error)
        except Exception:  # pragma: no cover - counting never blocks ingest
            logger.debug("LCM host-uid count failed")

    def _host_uid_classify(self, capture, stored_at: dict, remainders: set, delta: Counter) -> None:
        valid = {idx: uid for idx, uid in capture["entries"].items() if _valid_uid(uid)}
        delta["skipped.invalid_uid"] += len(capture["entries"]) - len(valid)
        if not valid:
            return
        lineage = self._host_uid_lineage_key()
        if lineage is None:
            delta["skipped.no_lineage_root"] += len(valid)
            return
        store, messages, plan = self._store, capture["messages"], capture["plan"]
        replayed, matched = plan.get("replayed") or set(), plan.get("matched") or {}

        def absorbed(message) -> list:
            values = message.get("_absorbed_message_uids")
            return [uid for uid in values if _valid_uid(uid)] if isinstance(values, (list, tuple)) else []

        bindings = store.host_uid_bindings_for(
            lineage, {uid for idx, uid in valid.items() for uid in [uid, *absorbed(messages[idx])]})
        writes, aliases, fetched = [], [], {}
        self._host_uid_fetch({sid for items in bindings.values() for sid, _kind in items}, fetched)  # one read

        def bind(store_id, uid, kind, proof) -> None:
            writes.append((store_id, uid, kind, proof))
            bindings.setdefault(uid, []).append((int(store_id), kind))

        for idx, uid in sorted(valid.items()):
            rows = matched.get(idx) if idx in replayed or idx in remainders else None
            stored = stored_at.get(idx)
            if idx in remainders or (rows and len(rows) > 1):  # COMPOSITE / REMAINDER
                ids = {int(row["store_id"]) for row in rows or ()} | ({int(stored)} if stored is not None else set())
                agree = all(not bindings.get(present) or ids & {sid for sid, _k in bindings[present]}
                            for present in [uid, *absorbed(messages[idx])])
                kind = "remainder" if idx in remainders else "composite"
                delta[f"composite.{'agree' if agree else 'disagree'}.{kind}"] += 1
                continue
            bound = [sid for sid, _kind in bindings.get(uid, ())]
            replay_unmapped = not rows and (idx < capture["cursor"] or idx in capture["tool_segment"] or idx in replayed)
            if bound:  # BOUND: compared against the payload-matching version (R1-1)
                if stored is not None:
                    if self._host_uid_bytes_match(bound, capture["identity"][idx], fetched):
                        delta["bound.disagree.stored_despite_match"] += 1
                    else:
                        delta["bound.version_new"] += 1
                        bind(stored, uid, "version", "version_new")
                elif rows:
                    delta["bound.agree.replay" if int(rows[0]["store_id"]) in bound
                          else "bound.disagree.replay_other_row"] += 1
                elif replay_unmapped:
                    delta["bound.agree.prefix_replay" if self._host_uid_bytes_match(bound, capture["identity"][idx], fetched)
                          else "bound.disagree.prefix_replay"] += 1
                else:
                    delta["skipped.not_stored"] += 1
            elif stored is not None:  # UNBOUND
                delta["unbound.agree_new"] += 1
                bind(stored, uid, "canonical", "stored_new")
            elif rows:
                target = int(rows[0]["store_id"])
                others = store.host_uid_uids_of_store(lineage, target) | {
                    u for u, items in bindings.items() if any(sid == target for sid, _k in items)}
                if others - {uid}:
                    aliases.append((idx, uid, target))
                else:
                    delta["unbound.agree_bind"] += 1
                    bind(target, uid, "canonical", "anchor_replay")
            else:
                delta["skipped.unmapped_replay" if replay_unmapped else "skipped.not_stored"] += 1
        store.add_host_uid_bindings(lineage, writes)
        if aliases:
            self._host_uid_alias_candidates(lineage, messages, aliases, delta)

    def _host_uid_fetch(self, store_ids, fetched: dict) -> None:
        """Rows cached across one ingest (a restart replays the whole prefix): one batched read, chunked."""
        missing = sorted({int(sid) for sid in store_ids} - set(fetched))
        fetched.update({sid: None for sid in missing})
        for start in range(0, len(missing), 500):
            fetched.update(self._store.get_batch(missing[start:start + 500]))

    def _host_uid_bytes_match(self, store_ids, identity_message, fetched: dict) -> bool:
        """A bound canonical or version row holds this dict's payload (its stored or host-rewrite form)."""
        self._host_uid_fetch(store_ids, fetched)
        identity = self._message_replay_identity(identity_message, strip_carrier=False)
        return any(identity in self._stored_row_forms(fetched[int(sid)]) for sid in store_ids
                   if fetched.get(int(sid)) is not None)

    def _host_uid_alias_candidates(self, lineage: str, messages, aliases, delta: Counter) -> None:
        """R3-1 (shadow, observational): ``position_proof`` when the occurrence's nearest canonically bound view
        neighbours are the row's own nearest bound rows (none on a side in both matches; one side bound)."""
        view = [(idx, message.get("message_uid")) for idx, message in enumerate(messages)
                if _valid_uid(message.get("message_uid"))]
        canonical = {uid: next((sid for sid, kind in items if kind == "canonical"), None)
                     for uid, items in self._store.host_uid_bindings_for(lineage, {uid for _i, uid in view}).items()}
        records = []
        for idx, uid, target in aliases:
            before = next((canonical[u] for i, u in reversed(view) if i < idx and canonical.get(u)), None)
            after = next((canonical[u] for i, u in view if i > idx and canonical.get(u)), None)
            proof = (before, after) == self._store.host_uid_canonical_neighbours(lineage, target) and (
                before is not None or after is not None)
            reason = "position_proof" if proof else "unknown"
            delta[f"unbound.alias_candidate.{reason}"] += 1
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

        try:
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
            logger.info("LCM host-uid shadow: agree=%d disagree=%d counts=%s", agree, disagree, _fmt_counts(counts))
        except Exception as exc:
            logger.debug("LCM host-uid summary failed (%s)", type(exc).__name__)


def host_uid_doctor_lines(engine: Any) -> list[str]:
    """Doctor ``host_uid`` section: counts only (per class and reason, errors, table size)."""
    try:
        durable, rows = engine._store.read_metadata_json(HOST_UID_COUNTER_KEY), engine._store.count_host_uid_bindings()
    except Exception as exc:
        durable, rows = None, f"error: {type(exc).__name__}"
    durable = durable if isinstance(durable, dict) else {}
    process = dict(getattr(engine, "_host_uid_counters", None) or {})
    errors = durable.get("errors") if isinstance(durable.get("errors"), int) else 0
    return [
        f"host_uid_mode: {host_message_uid_mode()}",
        f"host_uid_counts: {_fmt_counts({k: v for k, v in durable.items() if k != 'errors'})}",
        f"host_uid_process_counts: {_fmt_counts({k: v for k, v in process.items() if k != 'errors'})}",
        f"host_uid_errors: {errors}" + (f" (process {process['errors']})" if process.get("errors") else ""),
        f"host_uid_bindings_rows: {'absent' if rows is None else rows}",
    ]
