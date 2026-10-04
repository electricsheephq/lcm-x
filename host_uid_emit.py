"""Host-owned identity carry, plus slice B2's deterministic ENGINE uids for LCM-generated rows (REVISION 4).
No storage or replay decisions here."""

from __future__ import annotations

import hashlib
from collections import Counter
from importlib import import_module
from typing import Any, Callable

from .config import host_message_uid_mode

IDENTITY_KEYS = ("message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid")
ADDRESS_KEYS = ("_row_id", "_db_row_snapshot", "_canonical_row")
_host_uid_capability: bool | None = None


def host_uid_capable() -> bool:
    """Probe the host persistence-only contract once per process, failing closed."""
    global _host_uid_capability
    if _host_uid_capability is None:
        try:
            fields = import_module("agent.message_metadata").PERSISTENCE_ONLY_MESSAGE_FIELDS
            _host_uid_capability = "message_uid" in fields
        except Exception:
            _host_uid_capability = False
    return _host_uid_capability


def identity_emit_enabled() -> bool:
    """B1's gate for every key LCM ADDS: carry mode on and a capable host."""
    return host_message_uid_mode() != "off" and host_uid_capable()


def engine_uid(lineage_key: str, kind: str, basis: str, ordinal: int) -> str:
    """R4-1: the same generated row in the same lineage keeps one 32-hex uid across assemblies and replays."""
    return hashlib.sha256(f"lcmx-engine-uid\0{lineage_key}\0{kind}\0{basis}\0{ordinal}".encode()).hexdigest()[:32]


def carry_identity(source: dict, target: dict, keys: tuple[str, ...] = IDENTITY_KEYS) -> dict:
    """Copy only specified host keys already present; carry mode alone permits copies."""
    if host_message_uid_mode() != "off":
        for key in keys:
            if key in source:
                target[key] = source[key]
    return target


def record_absorbed_message(survivor: dict, dropped: dict) -> None:
    """Mirror agent.message_metadata.record_absorbed_message with survivor text first."""
    def uid_list(value: Any) -> list[str]:
        return list(dict.fromkeys(u for u in (value if isinstance(value, list) else ())
                                  if isinstance(u, str) and u))

    uid = dropped.get("message_uid")
    ordered = (uid_list(survivor.get("_absorbed_message_uids"))
               + ([uid] if isinstance(uid, str) and uid else [])
               + uid_list(dropped.get("_absorbed_message_uids")))
    absorbed = [u for u in dict.fromkeys(ordered) if u != survivor.get("message_uid")]
    if absorbed:
        survivor["_absorbed_message_uids"] = absorbed


def host_tool_call_key(tool_call: Any) -> str:
    """Return the host's UID-map key, with LCM's tool_call_id fallback last."""
    # Mirror agent.message_sanitization.coalesce_tool_call_id.
    if not isinstance(tool_call, dict):
        return ""
    for raw in (tool_call.get("call_id"), tool_call.get("id")):
        value = raw.strip() if isinstance(raw, str) else ""
        if value:
            return value.split("|", 1)[0].strip() or value
    value = tool_call.get("tool_call_id")
    key = str(value).strip() if value else ""
    return key.split("|", 1)[0].strip() or key


def per_occurrence_tool_call_uids(uids: dict, calls: list, call_id: Callable[[Any], str]) -> dict:
    """Mirror agent.message_metadata.per_occurrence_tool_call_uids using the supplied key."""
    counts = Counter(cid for call in calls if (cid := call_id(call)))
    expanded = dict(uids)
    for cid, count in counts.items():
        if count > 1 and isinstance(uid := uids.get(cid), str):
            expanded[cid] = [uid] * count
    return expanded


def merge_tool_call_uids(into: dict, extra: dict) -> dict:
    """Mirror agent.message_metadata.merge_tool_call_uids without losing occurrences."""
    merged = dict(into)
    for cid, uid in extra.items():
        if cid in merged:
            own = merged[cid]
            merged[cid] = (list(own) if isinstance(own, list) else [own]) + (
                list(uid) if isinstance(uid, list) else [uid])
        else:
            merged[cid] = uid
    return merged


def sync_cached_host_metadata(host: list[dict], cached: list[dict], generated_ids: set[int]) -> None:
    """Refresh positional host identity and addressing together; misalignment is a no-op."""
    if host_message_uid_mode() == "off" or len(host) != len(cached):
        return
    for source, target in zip(host, cached):
        generated = id(target) in generated_ids
        for key in IDENTITY_KEYS + ADDRESS_KEYS:
            if key in source and not (generated and key in ADDRESS_KEYS + ("_tool_call_uids",)):
                target[key] = source[key]
            else:
                target.pop(key, None)
