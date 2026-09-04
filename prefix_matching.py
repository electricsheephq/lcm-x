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
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
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
from .tokens import count_message_tokens

# Preserve the logger name used before this behavior-neutral extraction so
# existing handlers and filters continue to see ``hermes_lcm.engine`` records.
logger = logging.getLogger("hermes_lcm.engine")

_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT = 8


class PrefixMatchingMixin:
    def _session_end_matches_current_store_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        prefix_count = self._session_end_store_prefix_count(session_id, messages)
        return prefix_count is not None and prefix_count > 0

    def _session_end_prefix_compare_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                key: self._session_end_prefix_compare_value(child, session_id=session_id)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._session_end_prefix_compare_value(child, session_id=session_id)
                for child in value
            ]
        if not isinstance(value, str):
            return value

        text = restore_ingest_payload_placeholders(
            value,
            config=self._config,
            hermes_home=self._hermes_home,
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
                hermes_home=self._hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content

        if is_externalized_placeholder(stripped):
            ref = extract_externalized_ref(stripped)
            payload = load_externalized_payload(
                ref or "",
                config=self._config,
                hermes_home=self._hermes_home,
            )
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
        session_id: str,
    ) -> str:
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id,
        )
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> str:
        tool_calls = self._session_end_prefix_compare_value(
            (message or {}).get("tool_calls"),
            session_id=session_id,
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
        session_id: str,
    ) -> tuple[str, str, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(message, session_id=session_id),
            str((message or {}).get("tool_call_id") or ""),
            str((message or {}).get("tool_name") or ""),
            self._session_end_prefix_compare_tool_calls(message, session_id=session_id),
        )

    def _session_end_store_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> Optional[int]:
        try:
            stored_messages = self._store.get_range(
                session_id,
                limit=max(1, len(messages)),
                conversation_id=conversation_id,
            )
        except Exception:
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

    def _messages_match_fingerprint_prefix(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_fingerprint_prefix_count(fingerprints, messages) > 0

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

    def _messages_match_lcm_bypass_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_lcm_bypass_prefix_count(session_id, messages) > 0

    def _matching_lcm_bypass_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        count, _truncated = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        return count

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

    def _messages_match_lcm_normal_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> bool:
        return self._matching_lcm_normal_prefix_count(
            session_id,
            messages,
            conversation_id=conversation_id,
        ) > 0

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
        protected_messages = protect_messages_for_ingest(
            kept,
            session_id=session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return self._store._append_protected_batch(
            session_id,
            protected_messages,
            [count_message_tokens(msg) for msg in protected_messages],
            source=source,
            conversation_id=conversation_id,
            metadata_factory=self._real_user_scaffold_metadata_rows,
            metadata_messages=kept,
        )
