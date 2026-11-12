"""Proof-gated, off-turn recovery of stranded host rows (#2624)."""

import copy
import hashlib
import inspect
import json
import logging
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

from .host_uid import _is_fork_child
from .ingest_protection import protect_messages_for_ingest
from .maintenance_scheduler import _HOST_ROW_RECOVERY_SCHEDULER
from .message_analysis import _tool_call_id, _tool_result_names
from .reconcile import _commit_proof_identity_digest, _proof_user_identity
from .session_patterns import build_session_match_keys, compile_session_patterns, matches_session_pattern
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_SEEN = set()
_WARNED = set()
_COUNTS = Counter(dict.fromkeys((
    "candidates", "recovered_candidates", "recovered_rows", "unproven", "reader_error",
    "handle_missing", "signature_mismatch", "suffix_rejected", "ignored_in_suffix", "raced",
    "no_provenance", "walk_capped", "empty_suffix", "on_tip_path", "session_filtered",
), 0))
_MAX_SESSIONS = 4096


def host_row_recovery_counts() -> dict:
    with _LOCK:
        return dict(_COUNTS)


def _count(key, n=1):
    with _LOCK:
        _COUNTS[key] += n


def _unproven(sid, reason, n):
    _count("unproven")
    if reason in _COUNTS:
        _count(reason)
    with _LOCK:
        first = sid not in _WARNED
        _WARNED.add(sid)
    if first:
        logger.warning("LCM host-row recovery unproven: candidate=%s reason=%s host_rows=%d",
                       hashlib.sha256(sid.encode()).hexdigest()[:12], reason, n)


def _handle_reason(handle):
    for name in ("get_messages_as_conversation", "get_compression_tip"):
        method = getattr(handle, name, None)
        if not callable(method):
            return "handle_missing"
        try:
            params = list(inspect.signature(method).parameters.values())
            if not params or params[0].name != "session_id" or params[0].kind not in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return "signature_mismatch"
            if name == "get_messages_as_conversation":
                keywords = {p.name for p in params[1:] if p.kind in (
                    inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )}
                if not {"include_ancestors", "include_inactive", "repair_alternation",
                        "include_row_ids", "include_compacted"} <= keywords:
                    return "signature_mismatch"
            inspect.signature(method).bind("session")
        except (TypeError, ValueError):
            return "signature_mismatch"
    return None


def _tip_path(path, bound, handle):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
    try:
        sid, seen = handle.get_compression_tip(bound), set()
        while sid != bound:
            if not sid or sid in seen or len(seen) >= _MAX_SESSIONS:
                _count("walk_capped")
                logger.warning("LCM host-row recovery tip walk unproven")
                return None
            seen.add(sid)
            row = conn.execute("SELECT parent_session_id FROM sessions WHERE id=?", (sid,)).fetchone()
            sid = row[0] if row else None
        return seen | {bound}
    finally:
        conn.close()


def _skip_on_tip_path(path, bound, handle, sid):
    tip_path = _tip_path(path, bound, handle)
    if tip_path is None:
        return True
    if sid in tip_path:
        _count("on_tip_path")
        logger.debug("LCM host-row recovery skipped on tip path")
        return True
    return False


def _candidates(path, bound, handle, handle_reason, filtered=lambda sid, source: False):
    excluded = {bound} if handle_reason else _tip_path(path, bound, handle)
    if excluded is None:
        return []
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
    conn.row_factory = sqlite3.Row
    try:
        pending, visited, leaves = [bound], {bound}, []
        while pending:
            parent = pending.pop()
            for raw in conn.execute("SELECT * FROM sessions WHERE parent_session_id=?", (parent,)):
                row = dict(raw)
                if _is_fork_child(row):
                    continue
                sid = row["id"]
                if sid in visited or len(visited) >= _MAX_SESSIONS:
                    _count("walk_capped")
                    logger.warning("LCM host-row recovery descendant walk capped")
                    return []
                visited.add(sid)
                if filtered(sid, row.get("source")):  # an operator-ignored or stateless session and its subtree
                    _count("session_filtered")
                    continue
                if row["ended_at"] is None:
                    if sid not in excluded:
                        leaves.append(sid)
                elif row["end_reason"] == "compression":
                    pending.append(sid)
        return leaves
    finally:
        conn.close()


def _strict_proof_cursor(det, messages) -> Optional[int]:
    try:
        payload = det._durable_commit_proof_payload()
        if payload is None or payload.get("native"):
            return None
        target = list(payload.get("effective_sha256") or [])
        if not target:
            return None
        cursor = det._cursor_from_durable_commit_proof(messages)
        if cursor is None:
            return None
        projection, occ = det._occurrence_replay_identities(messages, payload)
        digests, p = [], 0
        for p, identity in enumerate(occ, 1):
            if identity is not None:
                digests.append(_commit_proof_identity_digest(
                    _proof_user_identity(identity) if payload.get("version") != 2 else identity))
                if len(digests) == len(target):
                    break
        if digests != target:
            return None
        while p < len(occ) and occ[p] is None and projection.entries[p].kind == "recovery":
            p += 1
        n_after = det._store.connection.execute(
            "SELECT count(*) FROM messages WHERE session_id=? AND store_id>?",
            (det._session_id, payload["last_store_id"]),
        ).fetchone()[0]
        return cursor if cursor == p + n_after else None
    except Exception:
        return None


def _valid_suffix(messages, cursor):
    calls, answered = set(), set()
    for index, row in enumerate(messages):
        if not isinstance(row, dict):
            return False
        role = row.get("role")
        if index >= cursor and role not in ("user", "assistant", "tool"):
            return False
        if role == "assistant" and "tool_calls" in row:
            tc = row["tool_calls"]
            if not isinstance(tc, list) or any(not isinstance(c, dict) or not _tool_call_id(c) for c in tc):
                return False
            ids = [_tool_call_id(c) for c in tc]
            if index >= cursor and (not ids or len(ids) != len(set(ids))):
                return False
            calls.update(ids)
            answered.difference_update(ids)
        if role == "tool":
            cid = row.get("tool_call_id")
            if index >= cursor and (not cid or cid not in calls or cid in answered):
                return False
            answered.add(cid)
    return True


def _raw_proof(conn, sid):
    row = conn.execute("SELECT value FROM metadata WHERE key=?", ("compaction_commit_proof:" + sid,)).fetchone()
    return row[0] if row else None


def _recover_candidate(engine_cls, config, home, db_path, state_path, conv, bound, sid, platform, window, handle, reason):
    det = engine_cls(config=copy.deepcopy(config), hermes_home=home)
    det._discard_prepared_leaves = lambda: None
    try:
        store = det._store  # R2: open before installing ANY candidate scope
        if Path(store.db_path).resolve() != Path(db_path):
            return _unproven(sid, "store_mismatch", 0)
        det._session_id, det._conversation_id = sid, conv
        det._hermes_home, det._session_platform = home, platform
        conn = store.connection
        raw = _raw_proof(conn, sid)
        payload = json.loads(raw) if raw else {}
        provenance = (payload.get("conversation_id") == conv and payload.get("hermes_home") == home)
        if not provenance and not conn.execute(
            "SELECT 1 FROM messages WHERE session_id=? AND conversation_id=? LIMIT 1", (sid, conv),
        ).fetchone():
            _count("no_provenance")
            logger.debug("LCM host-row recovery skipped without provenance")
            return
        _count("candidates")
        if reason:
            return _unproven(sid, reason, 0)
        if _skip_on_tip_path(state_path, bound, handle, sid):
            return
        try:
            messages = handle.get_messages_as_conversation(sid)
        except Exception:
            return _unproven(sid, "reader_error", 0)
        if _skip_on_tip_path(state_path, bound, handle, sid):
            return
        cursor = _strict_proof_cursor(det, messages)
        if cursor is None:
            return _unproven(sid, "proof", len(messages))
        suffix = messages[cursor:]
        if not suffix:
            _count("empty_suffix")
            logger.debug("LCM host-row recovery empty suffix")
            return
        if not _valid_suffix(messages, cursor):
            return _unproven(sid, "suffix_rejected", len(messages))
        if any(det._matches_ignore_message_patterns(m) for m in suffix):
            return _unproven(sid, "ignored_in_suffix", len(messages))
        uids = det._host_uid_capture(suffix, suffix, 0, 0, None, (), session_id=sid, store=store)
        lineage = None
        if uids is not None:
            uids["conversation_id"] = conv
            lineage = det._host_uid_lineage_key(sid, hermes_home=home, store=store)[0]
        names = _tool_result_names(messages)  # file reads and payload writes stay outside the write lock
        protected = protect_messages_for_ingest(
            suffix, config=det._config, hermes_home=home, session_id=sid,
            tool_name_hints=[names.get(cursor + i, "") for i in range(len(suffix))],
            context_window_tokens=window,
        )
        with store.atomic(immediate=True):
            reset = conn.execute(
                "SELECT last_reset_at FROM lcm_lifecycle_state WHERE conversation_id=?", (conv,),
            ).fetchone()
            if (_raw_proof(conn, sid) != raw or (reset and reset[0] is not None
                    and reset[0] >= float(payload.get("created_at") or 0))):
                return _unproven(sid, "raced", len(messages))
            checked = _strict_proof_cursor(det, messages)
            if checked != cursor or len(messages[checked:]) != len(suffix):
                return _unproven(sid, "raced", len(messages))
            ids = store._append_protected_batch(
                sid, protected, [count_message_tokens(m) for m in protected], source=platform,
                conversation_id=conv, metadata_messages=suffix,
                metadata_factory=lambda msg, store_id: det._real_user_scaffold_metadata_rows(msg, store_id, bound=False)
                + [(f"host_row_recovered:{store_id}", json.dumps({"version": 1, "issue": 2624}, sort_keys=True))],
            )
            if lineage is not None:
                det._host_uid_shadow(uids, dict(enumerate(ids)), store=store, hermes_home=home)
        _count("recovered_candidates")
        _count("recovered_rows", len(ids))
        roles = Counter(m["role"] for m in suffix)
        logger.info("LCM host-row recovery stored: candidate=%s rows=%d roles=%s",
                    hashlib.sha256(sid.encode()).hexdigest()[:12], len(ids),
                    " ".join(f"{r}={n}" for r, n in sorted(roles.items())))
    except Exception:
        _unproven(sid, "worker_error", 0)
    finally:
        det._close_storage()


def _recover(engine_cls, config, home, db_path, state_path, conv, bound, platform, window, handle):
    try:
        reason = _handle_reason(handle)
        patterns = [compile_session_patterns(config.ignore_session_patterns),
                    compile_session_patterns(config.stateless_session_patterns)]
        def filtered(sid, source):  # fail closed: the bound platform or the session's own host source
            return any(matches_session_pattern(build_session_match_keys(sid, platform=pf), p)
                       for pf in {platform, source or platform} for p in patterns)
        for sid in _candidates(state_path, bound, handle, reason, filtered):
            _recover_candidate(engine_cls, config, home, db_path, state_path, conv, bound, sid,
                               platform, window, handle, reason)
    except Exception:
        logger.debug("LCM host-row recovery unavailable")


class HostRowRecoveryMixin:
    def _enqueue_host_row_recovery(self, session_id, kwargs):
        try:
            if (not self._config.host_row_recovery_enabled or kwargs.get("boundary_reason")
                    or not self._session_id or not self._conversation_id
                    or self._session_id != session_id
                    or self._session_ignored or self._session_stateless or self.side_channel_active
                    or self._thread_context_stateless()
                    or self._thread_context_has_auxiliary_session(self._session_id)
                    or getattr(self, "_host_session_db", None) is None):
                return
            key = (self._store.db_identity, self._hermes_home, self._conversation_id, self._session_id)
            with _LOCK:
                if key in _SEEN:
                    return
                config = copy.deepcopy(self._config)
                db_path = str(self._db_path.resolve())
                config.database_path = db_path
                args = (type(self), config, self._hermes_home, db_path, str(self._state_db_path()),
                        self._conversation_id, self._session_id, self._session_platform,
                        self.context_length, self._host_session_db)
                scheduler_key = (
                    "host-row-recovery-db:" + hashlib.sha256(key[0].encode()).hexdigest()[:16],
                    "host-row-recovery:" + hashlib.sha256(repr(key[1:]).encode()).hexdigest()[:16],
                )
                if _HOST_ROW_RECOVERY_SCHEDULER.schedule(scheduler_key, lambda: _recover(*args)):
                    _SEEN.add(key)
        except Exception:
            logger.debug("LCM host-row recovery enqueue unavailable")
