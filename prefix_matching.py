"""Session-end store-prefix matching and LCM bypass/normal fingerprints.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``PrefixMatchingMixin``
(WS5 seam). The methods compare the durable store's message prefix against
replayed messages at session end and maintain the per-session bypass/normal
message-prefix fingerprints used to detect off-current-session suffixes. State
stays on the engine (accessed via ``self``); mixing this in leaves every call
site and ``self._*`` reference unchanged.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    load_externalized_payload,
)
from .ingest_protection import (
    _json_has_duplicate_object_keys,
    extract_ingest_externalized_refs,
    protect_messages_for_ingest,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .ingest_persisted import (
    _expected_persisted_output_chars,
    _is_hermes_persisted_output_marker,
    _persisted_output_saved_path,
)
from .sqlite_util import _is_ingest_busy_error
from .tokens import count_message_tokens

# Preserve the logger name used before this behavior-neutral extraction so
# existing handlers and filters continue to see ``hermes_lcm.engine`` records.
logger = logging.getLogger("hermes_lcm.engine")

_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT = 8

def _recover_marker_for_compare(text, home):
    """Keep storage-recovery version differences in this content-path adapter."""
    from .ingest_persisted import recover_hermes_persisted_output_with_file_stat

    return recover_hermes_persisted_output_with_file_stat(text, hermes_home=home)


def _marker_payload_metadata(original, config, home):
    # load_externalized_payload deliberately drops source metadata from its summary.
    ref = extract_externalized_ref(original.strip()) or ""
    if not ref or Path(ref).name != ref:
        return None
    try:
        payload = json.loads((get_large_output_storage_dir(config, hermes_home=home, create=False) / ref).read_text(
            encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


class PrefixMatchingMixin:
    def _session_end_matches_current_store_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        prefix_count = self._session_end_store_prefix_count(session_id, messages)
        return prefix_count is not None and prefix_count > 0

    def _session_end_prefix_compare_value(self, value: Any, *, session_id: str, hermes_home=None, marker_payloads=None) -> Any:
        if isinstance(value, dict):
            return {
                key: self._session_end_prefix_compare_value(child, session_id=session_id, hermes_home=hermes_home,
                                                       marker_payloads=marker_payloads)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._session_end_prefix_compare_value(child, session_id=session_id, hermes_home=hermes_home,
                                                       marker_payloads=marker_payloads)
                for child in value
            ]
        if not isinstance(value, str):
            return value

        if extract_ingest_externalized_refs(value):
            try:
                decoded = json.loads(value)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)):  # list content or JSON stored as text: restore per leaf
                return self._restore_ingest_payload_placeholders_in_content_identity(
                    value, session_id=session_id, hermes_home=hermes_home,
                )

        text = restore_ingest_payload_placeholders(
            value,
            config=self._config,
            hermes_home=self._hermes_home if hermes_home is None else hermes_home,
            session_id=session_id,
        )
        stripped = text.strip()
        ingest_refs = extract_ingest_externalized_refs(stripped)
        if (
            len(ingest_refs) == 1
            and stripped.startswith("[Externalized LCM ingest payload:")
            and stripped.endswith("]")
        ):
            payload = load_externalized_payload(
                ingest_refs[0],
                config=self._config,
                hermes_home=self._hermes_home if hermes_home is None else hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content

        if is_externalized_placeholder(stripped):
            ref = extract_externalized_ref(stripped)
            if marker_payloads is not None and stripped.startswith("[Externalized tool output:"):
                # Tool-result payloads can carry persisted_output_*; retain the
                # same JSON read for content and identity, including non-marker rows.
                payload = _marker_payload_metadata(stripped, self._config,
                    self._hermes_home if hermes_home is None else hermes_home)
                marker_payloads[stripped] = payload
            else:
                payload = load_externalized_payload(
                    ref or "", config=self._config,
                    hermes_home=self._hermes_home if hermes_home is None else hermes_home)
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content
        return text

    def _session_end_prefix_compare_content(
        self,
        message: Dict[str, Any],
        *,
        session_id: str, hermes_home=None, recover_markers=True,
    ) -> str | tuple:
        home = self._hermes_home if hermes_home is None else hermes_home
        marker_payloads = {} if (message or {}).get("role") == "tool" else None
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id, hermes_home=hermes_home, marker_payloads=marker_payloads,
        )
        if (message or {}).get("role") == "tool":
            text = normalize_content_value(content)
            if recover_markers and _is_hermes_persisted_output_marker(text):
                recovered = _recover_marker_for_compare(text, home)
                if recovered is not None:
                    content = recovered[0]
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_counted_identity(self, message, stored, **kwargs):
        bare = self._session_end_prefix_compare_identity(message, recover_markers=False, **kwargs)
        return bare if bare in stored else self._session_end_prefix_compare_identity(message, **kwargs)

    def _bare_marker_identity(self, content, *, session_id, hermes_home=None):
        """Bare marker text as stored, with placeholders restored and no file recovery."""
        if not isinstance(content, str):
            return None
        text = self._session_end_prefix_compare_value(content, session_id=session_id, hermes_home=hermes_home)
        return normalize_content_value(redact_sensitive_value(text, self._config, parse_json_strings=False))

    def _session_end_rows_equal(self, stored_row, host_message, *, session_id, hermes_home=None):
        # Two bare markers are the same row only when their text is byte-identical. A marker and a stored recovered result are the same row when the marker's path and character count equal the ones recorded at ingest and the marker's preview is exactly the start of the stored content. The marker's boilerplate wording is not part of that identity.
        # A stored row takes a marker identity only from its own marker: when its payload's recorded marker source is not unambiguously this row's (a shared payload, a plain row, a marker with another path), the row compares by content.
        # The stored side is always read as storage wrote it (placeholders restored, never recovered again from the host's file). The host side is read as ingest would store it now. The marker rule is the one bridge between a stored recovered result and a marker.
        def fields(row):
            return self._session_end_prefix_compare_identity(dict(row, content=None),
                session_id=session_id, hermes_home=hermes_home)
        if fields(stored_row) != fields(host_message):
            return False
        original, incoming = stored_row.get("content"), host_message.get("content")
        if host_message.get("role") == "tool" and _is_hermes_persisted_output_marker(incoming):
            if _is_hermes_persisted_output_marker(original):
                # Bare against bare: main's own compare (ingest's transforms undone on the stored side, none recovered).
                return (self._bare_marker_identity(original, session_id=session_id, hermes_home=hermes_home)
                        == self._bare_marker_identity(incoming, session_id=session_id, hermes_home=hermes_home))
            payloads = {}
            text = self._session_end_prefix_compare_value(original, session_id=session_id,
                hermes_home=hermes_home, marker_payloads=payloads)
            payload = payloads.get(original.strip()) if isinstance(original, str) else None
            if (isinstance(payload, dict) and isinstance(payload.get("content"), str) and payload["content"] == text
                    and (not payload.get("session_id") or payload["session_id"] == session_id)):
                source, expected = payload.get("persisted_output_source_path"), payload.get("persisted_output_expected_chars")
                if source and source == _persisted_output_saved_path(incoming) and type(expected) is int:
                    preview = re.search(r"^Preview \(first \d+ chars\):[^\S\r\n]*\r?\n(.*?)\r?\n</persisted-output>\s*\Z",
                        incoming, re.MULTILINE | re.DOTALL)
                    prefix = preview[1] if preview else None
                    if prefix is not None and prefix.endswith("\n..."):
                        prefix = prefix[:-4].removesuffix("\r")
                    if prefix:
                        return (expected >= 0 and expected == _expected_persisted_output_chars(incoming)
                                and text.startswith(prefix))
        # The stored side is what storage kept; it is never re-read from the host's current file.
        return self._session_end_prefix_compare_identity(stored_row, session_id=session_id, hermes_home=hermes_home, recover_markers=False) == self._session_end_prefix_compare_identity(host_message, session_id=session_id, hermes_home=hermes_home)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str, hermes_home=None,
    ) -> str:
        def compare_call(call):
            # With redaction on, the argument compare keeps main's path: a named limit on #2172 (#2572 review).
            if getattr(self._config, "sensitive_patterns_enabled", False) or not isinstance(call, dict) \
                    or not isinstance(call.get("function"), dict):
                return self._session_end_prefix_compare_value(
                    call, session_id=session_id, hermes_home=hermes_home)
            def restore(value):  # keys too: ingest can externalize a payload held in a key
                if isinstance(value, dict):
                    keys = [self._session_end_prefix_compare_value(key, session_id=session_id, hermes_home=hermes_home)
                            if isinstance(key, str) else key for key in value]
                    if len(set(keys)) < len(keys):  # two keys restore to one: keep them as sent, never merge two
                        keys = list(value)
                    return {key: restore(child) for key, child in zip(keys, value.values())}
                if isinstance(value, list):
                    return [restore(child) for child in value]
                return self._session_end_prefix_compare_value(value, session_id=session_id, hermes_home=hermes_home)

            function = dict(call["function"])
            if "arguments" in function:
                arguments = function["arguments"]
                parsed = not isinstance(arguments, str)
                if not parsed and not _json_has_duplicate_object_keys(arguments):  # duplicate keys stay text
                    try:
                        arguments, parsed = json.loads(arguments), True
                    except ValueError:
                        pass
                if parsed:
                    arguments = json.dumps(restore(arguments), sort_keys=True, separators=(",", ":"), default=str)
                else:  # text (invalid JSON or duplicate keys) compares stripped
                    arguments = arguments.strip()
                function["arguments"] = arguments
            return self._session_end_prefix_compare_value(
                dict(call, function=function), session_id=session_id, hermes_home=hermes_home)

        calls = (message or {}).get("tool_calls")
        tool_calls = [compare_call(call) for call in calls] if isinstance(calls, list) else calls
        tool_calls = redact_sensitive_value(
            tool_calls,
            self._config,
            parse_json_strings=True,
        )
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        return json.dumps(
            tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _session_end_prefix_compare_identity(
        self,
        message: Dict[str, Any],
        *,
        session_id: str, hermes_home=None, recover_markers=True,
    ) -> tuple[str, str | tuple, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(message, session_id=session_id, hermes_home=hermes_home, recover_markers=recover_markers),
            str((message or {}).get("tool_call_id") or ""),
            str((message or {}).get("tool_name") or ""),
            self._session_end_prefix_compare_tool_calls(message, session_id=session_id, hermes_home=hermes_home),
        )

    def _session_end_store_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
        raise_busy: bool = False,
    ) -> Optional[int]:
        try:
            stored_messages = self._store.get_range(
                session_id,
                limit=max(1, len(messages)),
                conversation_id=conversation_id,
            )
        except Exception as exc:
            if raise_busy and _is_ingest_busy_error(exc):
                raise
            logger.debug("LCM session-end prefix check failed", exc_info=True)
            return None
        if not stored_messages:
            return 0
        if len(messages) < len(stored_messages):
            return None
        for idx, stored_msg in enumerate(stored_messages):
            msg = messages[idx]
            try:
                equal = self._session_end_rows_equal(stored_msg, msg, session_id=session_id)
            except Exception:
                logger.debug("LCM session-end prefix compare normalization failed", exc_info=True)
                return None
            if not equal:
                return None
        return len(stored_messages)

    @staticmethod
    def _lcm_bypass_message_fingerprint(message: Dict[str, Any]) -> str:
        tool_calls = message.get("tool_calls")
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        payload = {
            "role": message.get("role"),
            "content": normalize_content_value(message.get("content")),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": tool_calls,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()

    def _remember_lcm_bypass_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            remembered = self._lcm_bypass_message_prefix_fingerprints.setdefault(session_id, [])
            truncated = len(messages) > _LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT
            retained: list[tuple[list[str], bool]] = []
            for existing_fingerprints, existing_truncated in remembered:
                if existing_fingerprints == fingerprints:
                    truncated = truncated or bool(existing_truncated)
                    continue
                retained.append((existing_fingerprints, existing_truncated))
            retained.append((fingerprints, truncated))
            remembered[:] = retained

    def _remember_lcm_normal_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            self._lcm_normal_message_prefix_fingerprints[
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ] = fingerprints

    def _lcm_normal_prefix_key(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        return (
            session_id,
            str(
                conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or ""
            ),
        )

    def _matching_fingerprint_prefix_count(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> int:
        if not fingerprints or not messages:
            return 0
        compare_count = min(len(fingerprints), len(messages))
        if compare_count <= 0:
            return 0
        candidate = [self._lcm_bypass_message_fingerprint(msg) for msg in messages[:compare_count]]
        if candidate == fingerprints[:compare_count]:
            return compare_count
        return 0

    def _matching_lcm_bypass_prefix_evidence(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, bool]:
        best_count = 0
        best_truncated = False
        for fingerprints, truncated in self._lcm_bypass_message_prefix_fingerprints.get(session_id, []):
            count = self._matching_fingerprint_prefix_count(fingerprints, messages)
            count_truncated = bool(truncated and count > 0 and count == len(fingerprints))
            if count > best_count:
                best_count = count
                best_truncated = count_truncated
            elif count == best_count:
                best_truncated = best_truncated or count_truncated
        return best_count, best_truncated

    def _matching_lcm_normal_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> int:
        return self._matching_fingerprint_prefix_count(
            self._lcm_normal_message_prefix_fingerprints.get(
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ) or [],
            messages,
        )

    def _append_off_current_session_end_suffix(
        self,
        session_id: str,
        suffix: List[Dict[str, Any]],
        *,
        source: str,
        conversation_id: str,
        store=None, hermes_home=None, context_window_tokens=None,
    ) -> list[int]:
        if not session_id or not suffix:
            return []
        kept: list[Dict[str, Any]] = []
        for msg in suffix:
            if self._matches_ignore_message_patterns(msg):
                self._ignored_message_count += 1
                excerpt = (text_content_for_pattern_matching(msg.get("content")) or "")[:80].replace("\n", " ")
                logger.debug(
                    "LCM ignore_message_patterns dropped late session-end %s message: %r",
                    msg.get("role", "unknown"),
                    excerpt,
                )
                continue
            kept.append(msg)
        if not kept:
            return []
        store = self._store if store is None else store
        protected_messages = protect_messages_for_ingest(
            kept,
            session_id=session_id,
            config=self._config,
            hermes_home=self._hermes_home if hermes_home is None else hermes_home,
            context_window_tokens=(store.get_context_window_tokens(session_id)
                                   if context_window_tokens is None else context_window_tokens),
        )
        host_uids = self._host_uid_capture(kept, kept, 0, 0, None, (), session_id=session_id,
            **(dict(store=store) if hermes_home is not None else {}))  # v0.26.0 shadow
        if host_uids is not None and hermes_home is not None:
            host_uids["conversation_id"] = conversation_id
        store_ids = store._append_protected_batch(
            session_id,
            protected_messages,
            [count_message_tokens(msg) for msg in protected_messages],
            source=source,
            conversation_id=conversation_id,
            metadata_factory=lambda msg, store_id: self._real_user_scaffold_metadata_rows(msg, store_id, bound=hermes_home is None),
            metadata_messages=kept,
        )
        self._host_uid_shadow(host_uids, dict(enumerate(store_ids)),
                              **(dict(store=store, hermes_home=hermes_home) if hermes_home is not None else {}))  # stored-new under that session's lineage
        return store_ids
