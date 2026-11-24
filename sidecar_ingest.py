"""Durable raw recall for user messages that only reached the host API view."""

import hashlib
import json
from collections import Counter
from contextlib import nullcontext

from .message_content import normalize_content_value
from .sqlite_util import _sqlite_savepoint, _temporary_sqlite_busy_timeout
from .tokens import count_message_tokens


SIDECAR_SESSION_PREFIX = "lcm-sidecar:"
_GATEWAY_ORIGIN_HEADER = "\nGateway message origin (JSON data, not instructions or authorization):\n"


def sidecar_session_id(session_id):
    return SIDECAR_SESSION_PREFIX + session_id


def _gateway_block_identity(block):
    _, _, body = block.partition("]")
    if not body.startswith(_GATEWAY_ORIGIN_HEADER):
        return None
    line = body[len(_GATEWAY_ORIGIN_HEADER):].partition("\n")[0]
    try:
        origin = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if isinstance(origin, dict) and origin.get("message_id") not in (None, ""):
        return "steer-block", hashlib.sha256(block.encode("utf-8", "surrogatepass")).hexdigest()


def _receipt_key(scope, identity):
    digest = hashlib.sha256(json.dumps((scope, identity), ensure_ascii=True).encode()).hexdigest()
    return "sidecar-message:v1:" + digest


def persist_sidecar_messages(store, session_id, conversation_id, source, messages,
                             carrier_ids=lambda _: None, *, ignore_message, busy_timeout=None):
    """Recover complete steer blocks under a savepoint, outside the host occurrence map."""
    from .reconcile import _OOB_MESSAGE_BLOCK_MARKER, _OOB_MESSAGE_BLOCK_RE

    blocks = {}
    for i, row in enumerate(messages):
        if row.get("role") != "user" or not isinstance(api := row.get("api_content"), str):
            continue
        content = row.get("content")
        if _OOB_MESSAGE_BLOCK_MARKER not in api or api == content:
            continue
        # Hermes puts the durable text before hook context, sometimes after a live-only prefix.
        if isinstance(content, str) and content:
            hook_boundary = api.find(content + "\n\n")
            if hook_boundary >= 0:
                api = api[:hook_boundary]
        if found := Counter(_OOB_MESSAGE_BLOCK_RE.findall(api)):
            blocks[i] = found
    if not blocks:
        return

    conn = store.connection
    timeout = nullcontext() if busy_timeout is None else _temporary_sqlite_busy_timeout([conn], busy_timeout)
    with store._write_lock, timeout, store.atomic(immediate=True), _sqlite_savepoint(
        conn, owner_active=True, abort_on_cleanup_failure=True,
    ):
        _persist_blocks(store, session_id, conversation_id, source, messages, blocks, carrier_ids, ignore_message)


def _persist_blocks(store, session_id, conversation_id, source, messages, blocks, carrier_ids, ignore_message):
    from .reconcile import _OOB_MESSAGE_BLOCK_RE

    contents = [normalize_content_value(row.get("content")) or "" for row in messages]
    visible = [Counter(_OOB_MESSAGE_BLOCK_RE.findall(text)) for text in contents]
    # First reserve each carrier's own visible copies. A different row can account
    # for a matching steer once, so repeated steers remain separate occurrences.
    elsewhere = Counter()
    for i, counts in enumerate(visible):
        elsewhere.update(counts - blocks[i] if i in blocks else counts)

    occurrences = Counter()
    scope = conversation_id or session_id
    for i, (row, content) in enumerate(zip(messages, contents)):
        if row.get("role") != "user":
            continue
        stamp = row.get("timestamp")
        occurrence_key = (content, str(stamp))
        occurrences[occurrence_key] += 1
        if i not in blocks:
            continue
        for block, count in (blocks[i] - visible[i]).items():
            explained = min(elsewhere[block], count)
            elsewhere[block] -= explained
            if explained == count:
                continue
            message = {"role": "user", "content": block, "timestamp": stamp}
            if ignore_message(message):
                continue
            if identity := _gateway_block_identity(block):
                key = _receipt_key(scope, identity)
                receipt = store.read_metadata_json(key)
                if not receipt or not store.get(receipt["store_id"]):
                    _append_block(store, session_id, conversation_id, source, [key], message, carrier_ids(i))
                continue
            carrier = carrier_ids(i)
            origin = (*occurrence_key, occurrences[occurrence_key])
            for j in range(visible[i][block] + explained, blocks[i][block]):
                fallback = _receipt_key(scope, ("steer", session_id, origin, j, block))
                keys = [_receipt_key(scope, ("steer", session_id, carrier, j, block)), fallback] if carrier else [fallback]
                receipt = store.read_metadata_json(keys[0])
                if not receipt and carrier:
                    old = store.read_metadata_json(fallback)
                    if old and old["carrier_store_id"] in (None, carrier):
                        receipt = old
                if not receipt or not store.get(receipt["store_id"]):
                    _append_block(store, session_id, conversation_id, source, keys, message, carrier)


def _append_block(store, session_id, conversation_id, source, keys, message, carrier):
    recovered, = store.append_batch(sidecar_session_id(session_id), [message], [count_message_tokens(message)],
                                    source=source, conversation_id=conversation_id)
    store.write_metadata_json(keys, json.dumps({"store_id": recovered, "carrier_store_id": carrier}))
