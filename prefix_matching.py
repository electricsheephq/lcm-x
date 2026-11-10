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
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    load_externalized_payload,
)
from .ingest_protection import (
    extract_ingest_externalized_refs,
    protect_messages_for_ingest,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .ingest_persisted import (
    _expected_persisted_output_chars,
    _is_hermes_persisted_output_marker,
    _persisted_output_preview_prefix_digest,
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
        session_id: str, hermes_home=None,
    ) -> str | tuple:
        home = self._hermes_home if hermes_home is None else hermes_home
        marker_payloads = {} if (message or {}).get("role") == "tool" else None
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id, hermes_home=hermes_home, marker_payloads=marker_payloads,
        )
        if (message or {}).get("role") == "tool":
            text = normalize_content_value(content)
            if not bool(getattr(self._config, "large_output_externalization_enabled", True)):
                # Without durable marker metadata, retain the recovered-content compare.
                if _is_hermes_persisted_output_marker(text):
                    recovered = _recover_marker_for_compare(text, home)
                    if recovered is not None:
                        content = recovered[0]
            else:
                identity = self._session_end_marker_identity(
                    message, text, session_id=session_id, payloads=marker_payloads)
                if identity is not None:
                    return identity
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_marker_identity(self, message, text, *, session_id, payloads):
        # A marker row equals a stored row exactly when the host's marker text would be byte-identical (path, character count, preview), whether or not the file still exists. A stored row without recorded marker metadata compares by recovered content.
        if _is_hermes_persisted_output_marker(text):
            source = _persisted_output_saved_path(text)
            expected = _expected_persisted_output_chars(text)
            redacted = normalize_content_value(redact_sensitive_value(
                text, self._config, parse_json_strings=False))
            digest = _persisted_output_preview_prefix_digest(redacted)
        else:
            original = (message or {}).get("content")
            if not isinstance(original, str) or not is_externalized_placeholder(original.strip()):
                return None
            payload = payloads.get(original.strip())
            if (not isinstance(payload, dict) or not isinstance(payload.get("content"), str)
                    or payload["content"] != text
                    or (payload.get("session_id") and payload["session_id"] != session_id)):
                return None
            source = payload.get("persisted_output_source_path")
            expected = payload.get("persisted_output_expected_chars")
            digest = payload.get("persisted_output_redacted_preview_sha256")
        if not (isinstance(source, str) and source and type(expected) is int and expected >= 0
                and isinstance(digest, str) and len(digest) == 64):
            return None
        # A tuple cannot collide with literal user/tool text. All three fields must match.
        return ("persisted-output-source", source, expected, digest)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str, hermes_home=None,
    ) -> str:
        tool_calls = self._session_end_prefix_compare_value(
            (message or {}).get("tool_calls"),
            session_id=session_id, hermes_home=hermes_home,
        )
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
        session_id: str, hermes_home=None,
    ) -> tuple[str, str | tuple, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(message, session_id=session_id, hermes_home=hermes_home),
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
                message_identity = self._session_end_prefix_compare_identity(
                    msg,
                    session_id=session_id,
                )
                stored_identity = self._session_end_prefix_compare_identity(
                    stored_msg,
                    session_id=session_id,
                )
            except Exception:
                logger.debug("LCM session-end prefix compare normalization failed", exc_info=True)
                return None
            if message_identity != stored_identity:
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
