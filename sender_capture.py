"""Capture-only, process-local current-occurrence associations (S0)."""
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import re
import threading
import time
import unicodedata

from . import config
from .host_uid import _valid_uid

logger = logging.getLogger(__name__)
SENDER_SOURCES = frozenset({"gateway_principal", "gateway"})  # gateway reserved, never emitted in S0
_LOCK = threading.Lock()
_PENDING = {}
_RETIRED = OrderedDict()
_EXCLUDED = OrderedDict()
_TTL, _CAP, _RETIRED_CAP = 900, 256, 1024


@dataclass
class Pending:
    turn_id: str
    obj: dict
    identity: tuple
    conversation_id: str
    session_id: str
    sender: tuple[str, str]
    created: float
    store_ids: set[int] = field(default_factory=set)


def _key(engine):
    return str(engine._store.db_path.resolve()), str(engine._session_id)


def _retire(key, turn_id):
    old = _PENDING.get(key)
    if old is not None and old.turn_id == turn_id:
        _PENDING.pop(key)
    _RETIRED[(*key, turn_id)] = None
    while len(_RETIRED) > _RETIRED_CAP:
        _RETIRED.popitem(last=False)


def retire(engine, turn_id=None):
    try:
        key = _key(engine)
        with _LOCK:
            old = _PENDING.get(key)
            if turn_id or old is not None:
                _retire(key, str(turn_id or old.turn_id))
    except Exception:
        logger.debug("sender_capture:retire_error")


def snapshot(engine):
    key = _key(engine)
    with _LOCK:
        return _PENDING.get(key)


def create_pending(engine, payload):
    try:
        key, now = _key(engine), time.monotonic()
        turn = str(payload.get("turn_id") or "")
        with _LOCK:
            if (*key, turn) in _RETIRED:
                logger.debug("sender_capture:refused_retired")
                return None
            for binding, old in list(_PENDING.items()):
                if now - old.created >= _TTL or (binding == key and old.turn_id != turn):
                    _retire(binding, old.turn_id)
        raw = payload.get("sender_id")
        platform = str(payload.get("platform") or "").lower()
        history = payload.get("conversation_history")
        if (not config._parse_bool_env("LCM_SENDER_CAPTURE", True) or not turn or not key[1]
                or payload.get("session_id") != key[1] or not isinstance(raw, str)
                or not re.fullmatch(r"[a-z0-9_.-]{1,32}", platform)):
            return None
        sender = raw.strip()
        if not sender or len(sender) > 200 or any(unicodedata.category(c) in ("Cc", "Cs") for c in sender):
            return None
        obj = history[-1] if isinstance(history, list) and history else None
        author = payload.get("turn_author")
        # A tail object the host reused from an earlier occurrence cannot name one row.
        if (not isinstance(obj, dict) or any(o is obj for o in history[:-1])
                or obj.get("role") != "user" or "display_kind" in obj
                or (author is not None and (not isinstance(author, dict) or author.get("is_bot")
                                           or author.get("id") != raw))):
            return None
        pending = Pending(turn, obj, engine._message_replay_identity(obj, strip_carrier=False),
                          str(engine._conversation_id or ""), key[1], (f"{platform}:{sender}", "gateway_principal"), now)
        with _LOCK:
            if (*key, turn) in _RETIRED:
                logger.debug("sender_capture:refused_retired")
                return None
            old = _PENDING.get(key)
            if old is not None and old.turn_id != turn:
                _retire(key, old.turn_id)
            elif old is not None:
                return old
            while len(_PENDING) >= _CAP:
                binding = next(iter(_PENDING))
                _retire(binding, _PENDING[binding].turn_id)
            _PENDING[key] = pending
        try_update(engine)
        return pending
    except Exception:
        logger.debug("sender_capture:create_error")
        return None


def _watch(engine):
    if getattr(engine, "_host_rewrite_store", None) is engine._store:
        return list(getattr(engine, "_host_rewrite_watch", {}).items())
    return []


def matches(engine, pending, message):
    if (message.get("role") != "user" or "display_kind" in message
            or str(engine._session_id) != pending.session_id
            or str(engine._conversation_id or "") != pending.conversation_id
            or message.get("_absorbed_message_uids") or message.get("_merged_turn_prefix")):
        return False
    uid, other = pending.obj.get("message_uid"), message.get("message_uid")
    if _valid_uid(uid) and _valid_uid(other):
        candidate = uid == other
    else:
        candidate = message is pending.obj or any(
            any(obj is pending.obj for obj in entry[2]) and any(obj is message for obj in entry[2])
            for _sid, entry in _watch(engine))
    return candidate and engine._message_replay_identity(message, strip_carrier=False) == pending.identity


def batch_senders(engine, originals, excluded=()):
    try:
        pending = snapshot(engine)
        if pending is None or pending.store_ids:
            return None, None
        values = [pending.sender if i not in excluded and matches(engine, pending, obj) else None
                  for i, obj in originals]
        return (pending, values) if sum(value is not None for value in values) == 1 else (None, None)
    except Exception:
        logger.debug("sender_capture:match_error")
        return None, None


def record(engine, pending, ids, values):
    try:
        key = _key(engine)
        with _LOCK:
            if _PENDING.get(key) is pending:
                pending.store_ids.update(sid for sid, sender in zip(ids, values) if sender is not None)
    except Exception:
        logger.debug("sender_capture:record_error")


def exclude(engine, store_ids):
    try:
        db = _key(engine)[0]
        with _LOCK:
            for sid in store_ids:
                _EXCLUDED.setdefault((db, sid), None)
            while len(_EXCLUDED) > _RETIRED_CAP:
                _EXCLUDED.popitem(last=False)
    except Exception:
        logger.debug("sender_capture:exclude_error")


def try_update(engine):
    try:
        pending = snapshot(engine)
        if pending is None or pending.store_ids or not matches(engine, pending, pending.obj):
            return
        candidates = [(sid, obj) for sid, entry in _watch(engine)
                      if any(obj is pending.obj for obj in entry[2]) for obj in entry[2]
                      if matches(engine, pending, obj)
                      and engine._message_replay_identity(obj) == entry[0]]
        ids = {sid for sid, _obj in candidates}
        db = _key(engine)[0]
        with _LOCK:
            if any((db, sid) in _EXCLUDED for sid in ids):
                return
        if len(ids) != 1:
            return
        sid = next(iter(ids))
        count = engine._store.set_sender_if_null(sid, *pending.sender,
                                                [engine._session_id, *engine._identity_anchor_chain_cached()])
        if count:
            record(engine, pending, [sid], [pending.sender])
    except Exception:
        logger.debug("sender_capture:update_error")
