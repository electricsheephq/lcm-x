"""Off-turn, forward-only recovery of complete failed host turns (#2042)."""
import copy
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from .engine_registry import live_bound_session_ids
from .host_row_recovery import _valid_suffix
from .host_uid import _is_fork_child
from .ingest_protection import protect_messages_for_ingest
from .maintenance_scheduler import _HOST_ROW_RECOVERY_SCHEDULER
from .message_analysis import _tool_call_id, _tool_result_names
from .session_patterns import build_session_match_keys, compile_session_patterns, matches_session_pattern
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)
_now = time.time
_LOCK = threading.Lock()
_SEEN, _MARKERS = {}, {}
_COUNTS = Counter()


def unended_catchup_counts():
    with _LOCK:
        return dict(_COUNTS)


def _count(reason, n=1):
    with _LOCK:
        _COUNTS[reason] += n


def _complete_groups(messages):
    pending = set()
    for row in messages:
        if row.get('role') in ('user', 'assistant'):
            if pending:
                return False
            pending = {_tool_call_id(call) for call in row.get('tool_calls', [])}
        elif row.get('role') == 'tool':
            cid = row.get('tool_call_id')
            if cid not in pending:
                return False
            pending.remove(cid)
    return not pending


def _fingerprint(conn, sid, archived, revision):
    return tuple(conn.execute(f"""SELECT count(CASE WHEN active=1 THEN 1 END),
        max(CASE WHEN active=1 THEN id END),
        max(CASE WHEN active=1 AND role NOT IN ('session_meta','system') THEN id END),
        sum(CASE WHEN active<>1 OR active IS NULL OR ({archived}) THEN 1 ELSE 0 END),
        (SELECT {revision} FROM sessions WHERE id=?) FROM messages WHERE session_id=?""", (sid, sid)).fetchone())


def _store_candidate(engine_cls, config, home, db_path, row, messages, window):
    det = engine_cls(config=copy.deepcopy(config), hermes_home=home)
    det._discard_prepared_leaves = lambda: None
    try:
        store = det._store
        if Path(store.db_path).resolve() != Path(db_path):
            return _count('store_mismatch')
        sid, conv = row['id'], row.get('session_key') or row['id']
        det._session_id, det._conversation_id = sid, conv
        det._hermes_home, det._session_platform = home, row.get('source') or ''
        if any(det._matches_ignore_message_patterns(m) for m in messages):
            return _count('ignored_in_suffix')
        uids = det._host_uid_capture(messages, messages, 0, 0, None, (), session_id=sid, store=store)
        lineage = None
        if uids is not None:
            uids['conversation_id'] = conv
            lineage = det._host_uid_lineage_key(sid, hermes_home=home, store=store)[0]
        names = _tool_result_names(messages)
        protected = protect_messages_for_ingest(messages, config=det._config, hermes_home=home, session_id=sid,
            tool_name_hints=[names.get(i, '') for i in range(len(messages))], context_window_tokens=window)
        with store.atomic(immediate=True):
            if store.has_session_messages(sid):
                return _count('raced')
            ids = store._append_protected_batch(sid, protected, [count_message_tokens(m) for m in protected],
                source=det._session_platform, conversation_id=conv, metadata_messages=messages,
                metadata_factory=lambda msg, store_id: det._real_user_scaffold_metadata_rows(msg, store_id, bound=False)
                + [(f'unended_turn_recovered:{store_id}', json.dumps({'version': 1, 'issue': 2042}, sort_keys=True))])
            if lineage is not None:
                det._host_uid_shadow(uids, dict(enumerate(ids)), store=store, hermes_home=home)
        _count('recovered_candidates')
        _count('recovered_rows', len(ids))
        logger.info('LCM unended catch-up stored: candidate=%s rows=%d roles=%s',
                    hashlib.sha256(sid.encode()).hexdigest()[:12], len(ids), dict(Counter(m['role'] for m in messages)))
        return True
    finally:
        det._close_storage()


def _catch_up(engine_cls, config, home, db_path, state_path, bound, platform, window, handle, profile, bind_time):
    try:
        with closing(sqlite3.connect(db_path, timeout=1.0)) as store, closing(sqlite3.connect(
                Path(state_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=1.0)) as host:
            marker_key = 'unended_catchup_since:' + hashlib.sha256(home.encode()).hexdigest()[:16]
            key = (str(Path(db_path).resolve()), home)
            with _LOCK:
                if key not in _MARKERS:
                    store.execute('INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)', (marker_key, str(bind_time)))
                    store.commit()
                    _MARKERS[key] = float(store.execute('SELECT value FROM metadata WHERE key=?', (marker_key,)).fetchone()[0])
                since = _MARKERS[key]
            sc = {r[1] for r in host.execute('PRAGMA table_info(sessions)')}
            mc = {r[1] for r in host.execute('PRAGMA table_info(messages)')}
            if not {'id', 'source', 'parent_session_id', 'started_at', 'ended_at', 'end_reason', 'model_config'} <= sc or not {
                    'id', 'session_id', 'role', 'display_kind', 'active', 'timestamp'} <= mc:
                return _count('missing_columns')
            archived = ' OR '.join(f'COALESCE({c},0)<>0' for c in ('compacted', '_compressed_summary') if c in mc) or '0'
            revision = 'transcript_revision' if 'transcript_revision' in sc else 'NULL'
            profile_filter = 'AND COALESCE(s.profile_name,?)=?' if 'profile_name' in sc else ''
            host.row_factory = sqlite3.Row
            candidates = host.execute(f"""SELECT s.*, b.id AS boundary_id FROM sessions s JOIN messages b ON b.id=(
                SELECT max(id) FROM messages WHERE session_id=s.id AND active=1 AND role NOT IN ('session_meta','system'))
                WHERE s.parent_session_id IS NULL AND COALESCE(s.source,'')<>'subagent'
                AND COALESCE(s.end_reason,'')<>'compression' AND s.id<>? {profile_filter}
                AND NOT EXISTS (SELECT 1 FROM messages WHERE session_id=s.id AND ({archived}))
                AND b.display_kind='failed_turn' AND b.timestamp>=? AND b.timestamp>=? AND b.timestamp<=?
                ORDER BY b.timestamp DESC, b.id DESC LIMIT 64""",
                (bound, *((profile, profile) if profile_filter else ()), since, _now()-7*86400, _now()-600)).fetchall()
            patterns = [compile_session_patterns(config.ignore_session_patterns), compile_session_patterns(config.stateless_session_patterns)]
            stored = 0
            for raw in candidates:
                row, reason = dict(raw), None
                sid = row['id']
                _count('candidates')
                if store.execute('SELECT 1 FROM messages WHERE session_id=? LIMIT 1', (sid,)).fetchone():
                    reason = 'existing_rows'
                elif _is_fork_child(row):
                    reason = 'fork_child'
                elif any(matches_session_pattern(build_session_match_keys(sid, platform=pf), p)
                         for pf in {platform, row.get('source') or platform} for p in patterns):
                    reason = 'session_filtered'
                elif sid in live_bound_session_ids():
                    reason = 'live_bound'
                if reason:
                    _count(reason)
                    continue
                f1 = _fingerprint(host, sid, archived, revision)
                messages = handle.get_messages_as_conversation(sid)
                if (f1 != _fingerprint(host, sid, archived, revision) or f1[3] or f1[2] != row['boundary_id']
                        or len(messages) != f1[0] or not messages or messages[-1].get('display_kind') != 'failed_turn'):
                    _count('generation_changed')
                elif any(m.get('role') in ('session_meta', 'system') for m in messages):
                    _count('non_turn_role')
                elif not _valid_suffix(messages, 0) or len(messages)>500 or messages[-1].get('role') != 'assistant':
                    _count('suffix_rejected')
                elif not _complete_groups(messages):
                    _count('incomplete_tool_group')
                else:
                    stored += bool(_store_candidate(engine_cls, config, home, db_path, row, messages, window))
                    if stored >= 16:
                        break
    except Exception:
        _count('worker_error')
        logger.debug('LCM unended catch-up unavailable', exc_info=True)


class UnendedCatchupMixin:
    def _enqueue_unended_catchup(self, session_id, kwargs):
        try:
            if (not self._config.host_row_recovery_enabled or kwargs.get('boundary_reason')
                    or not self._session_id or not self._conversation_id or self._session_id != session_id
                    or self._session_ignored or self._session_stateless or self.side_channel_active
                    or self._thread_context_stateless() or self._thread_context_has_auxiliary_session(self._session_id)
                    or getattr(self, '_host_session_db', None) is None):
                return
            home, now = str(Path(self._hermes_home).resolve()), _now()
            key = (self._store.db_identity, home)
            with _LOCK:
                if now - _SEEN.get(key, float('-inf')) < 600:
                    return
                config = copy.deepcopy(self._config)
                config.database_path = db_path = str(self._db_path.resolve())
                args = (type(self), config, home, db_path, str(self._state_db_path()), session_id,
                        self._session_platform, self.context_length, self._host_session_db, self._entity_profile_key(db_path), now)
                scheduler_key = tuple('unended-catchup:' + hashlib.sha256(k.encode()).hexdigest()[:16] for k in key)
                if _HOST_ROW_RECOVERY_SCHEDULER.schedule(scheduler_key, lambda: _catch_up(*args)):
                    _SEEN[key] = now
        except Exception:
            logger.debug('LCM unended catch-up enqueue unavailable')
