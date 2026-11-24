"""Durable raw recall for user messages that only reached the host API view."""

import hashlib
import json
import re
from collections import Counter
from functools import cache

from .message_content import normalize_content_value
from .tokens import count_message_tokens


SIDECAR_SESSION_PREFIX = "lcm-sidecar:"
_DISCORD_ENVELOPE = re.compile(
    r"\[Triggering message id: `(\d+)` — use as `message_id` for reply/react/pin via the discord tools\.\]\n\n"
)
_PLATFORM_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:@-]{0,127}")


def sidecar_session_id(session_id):
    return SIDECAR_SESSION_PREFIX + session_id


def _platform_message(api, content):
    if match := _DISCORD_ENVELOPE.match(api):
        body = api[match.end():]
        # The gateway separates both its note and appended hook context with blank lines.
        if body.strip() and "\n\n" not in body and body != content:
            return match[1], body
    api = api.removeprefix(content + "\n\n").lstrip()
    if api.startswith("{") and '"message_id"' in api:
        try:
            envelope, _ = json.JSONDecoder().raw_decode(api)
        except (ValueError, RecursionError):
            return None
        message_id, body = envelope.get("message_id"), envelope.get("text", envelope.get("content"))
        if type(message_id) is int:
            message_id = str(message_id)
        if (isinstance(message_id, str) and _PLATFORM_ID.fullmatch(message_id)
                and isinstance(body, str) and body.strip()):
            return message_id, body


def _platform_id_is_visible(store, contents, message_id):
    return any(message_id in text for text in contents) or store.connection.execute(
        "SELECT 1 FROM messages_fts JOIN messages ON messages.store_id = messages_fts.rowid "
        "WHERE messages_fts MATCH ? AND instr(messages.content, ?) > 0 LIMIT 1",
        (f'"{message_id}"', message_id),
    ).fetchone() is not None


def _receipt_key(scope, identity):
    digest = hashlib.sha256(json.dumps((scope, identity), ensure_ascii=True).encode()).hexdigest()
    return "sidecar-message:v1:" + digest


def persist_sidecar_messages(store, session_id, conversation_id, source, messages, carrier_ids):
    """Append inside the normal ingest transaction, outside its host occurrence map."""
    from .reconcile import _OOB_MESSAGE_BLOCK_MARKER, _OOB_MESSAGE_BLOCK_RE

    sidecars = {i: api for i, row in enumerate(messages) if row.get("role") == "user"
                and isinstance(api := row.get("api_content"), str)
                and (_OOB_MESSAGE_BLOCK_MARKER in api or "message_id" in api)
                and api != row.get("content")}
    if not sidecars:
        return

    contents = [normalize_content_value(row.get("content")) or "" for row in messages]
    visible = [Counter(_OOB_MESSAGE_BLOCK_RE.findall(text)) for text in contents]
    blocks = {i: Counter(_OOB_MESSAGE_BLOCK_RE.findall(api)) for i, api in sidecars.items()}
    # First reserve each carrier's own visible copies. A different row can account
    # for a matching steer once, so repeated steers remain separate occurrences.
    elsewhere = Counter()
    for i, counts in enumerate(visible):
        elsewhere.update(counts - blocks[i] if i in blocks else counts)
    occurrences = Counter()
    scope = conversation_id or session_id

    @cache
    def id_is_visible(message_id):
        return _platform_id_is_visible(store, contents, message_id)

    for i, (row, content) in enumerate(zip(messages, contents)):
        if row.get("role") != "user":
            continue
        stamp = row.get("timestamp")
        occurrence_key = (content, str(stamp))
        occurrences[occurrence_key] += 1
        if i not in sidecars:
            continue
        candidates = []
        carrier = None
        for block, count in (blocks[i] - visible[i]).items():
            explained = min(elsewhere[block], count)
            elsewhere[block] -= explained
            if explained == count:
                continue
            carrier = carrier or carrier_ids(i)
            origin = (*occurrence_key, occurrences[occurrence_key])
            candidates.extend((("steer", session_id, origin, j, block), block)
                              for j in range(visible[i][block] + explained, blocks[i][block]))
        if envelope := _platform_message(sidecars[i], content):
            message_id, body = envelope
            candidates.append((("platform", message_id), body))
        for identity, body in candidates:
            keys = [_receipt_key(scope, identity)]
            if identity[0] == "steer" and carrier:
                keys.insert(0, _receipt_key(scope, (*identity[:2], carrier, *identity[3:])))
            receipt = store.read_metadata_json(keys[0])
            if not receipt and len(keys) > 1:
                fallback = store.read_metadata_json(keys[1])
                if fallback and fallback["carrier_store_id"] in (None, carrier):
                    receipt = fallback
            if receipt and store.get(receipt["store_id"]):
                continue
            if identity[0] == "platform" and id_is_visible(identity[1]):
                continue
            message = {"role": "user", "content": body, "timestamp": stamp}
            recovered, = store.append_batch(sidecar_session_id(session_id), [message], [count_message_tokens(message)],
                                            source=source, conversation_id=conversation_id)
            store.write_metadata_json(keys, json.dumps({"store_id": recovered,
                                                         "carrier_store_id": carrier or carrier_ids(i)}))
