"""Active-replay identity, ignore-pattern matching, and payload restoration.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``ReplayIdentityMixin``
(WS5 seam). The methods match stored rows against ignore patterns, cache and
copy active-replay messages, restore externalized ingest-payload placeholders,
and prove/redact persisted-output identities during replay. State stays on the
engine (accessed via ``self``); mixing this in leaves every call site and
``self._*`` reference unchanged.
"""

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    find_externalized_tool_result_content_for_call,
    load_externalized_payload,
)
from .ingest_protection import (
    _expected_persisted_output_chars,
    extract_all_externalized_payload_refs,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _persisted_output_inline_preview_sha256,
    _persisted_output_preview_prefix_digest,
    _persisted_output_saved_path,
    assistant_output_quarantine_reason,
    extract_ingest_externalized_refs,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
)
from .message_content import (
    normalize_content_value,
    stored_text_content_for_pattern_matching,
    text_content_for_pattern_matching,
)
from .message_patterns import matches_message_pattern
from .reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from .sanitize import _should_drop_active_assistant_message

logger = logging.getLogger(__name__)


class ReplayIdentityMixin:
    def _stored_row_externalized_text_parts_for_pattern_matching(self, msg: Dict[str, Any]) -> list[str]:
        ref_sources: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            ref_sources.append(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                ref_sources.append(json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                ref_sources.append(str(tool_calls))
        refs: list[str] = []
        for source in ref_sources:
            for ref in extract_all_externalized_payload_refs(source):
                if ref not in refs:
                    refs.append(ref)
        parts: list[str] = []
        session_id = str(msg.get("session_id") or self._session_id or "")
        for ref in refs:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if not payload:
                continue
            payload_session_id = str(payload.get("session_id") or "")
            if session_id and payload_session_id and payload_session_id != session_id:
                continue
            payload_content = payload.get("content")
            if isinstance(payload_content, str):
                parts.append(payload_content)
        return parts

    def _stored_row_externalized_text_for_pattern_matching(self, msg: Dict[str, Any]) -> str:
        return "\n".join(self._stored_row_externalized_text_parts_for_pattern_matching(msg))

    def _is_cached_active_replay_message_at_index(self, idx: int, msg: Dict[str, Any]) -> bool:
        if idx < 0 or idx >= len(self._last_active_replay_messages):
            return False
        return self._message_replay_identity(msg) == self._message_replay_identity(
            self._last_active_replay_messages[idx]
        )

    def _matches_ignore_message_patterns(self, msg: Dict[str, Any], *, stored_row: bool = False) -> bool:
        if not self._compiled_ignore_message_patterns:
            return False
        content = msg.get("content")
        text = (
            stored_text_content_for_pattern_matching(content)
            if stored_row
            else text_content_for_pattern_matching(content)
        ) or ""
        if matches_message_pattern(text, self._compiled_ignore_message_patterns):
            return True
        if stored_row:
            externalized_parts = self._stored_row_externalized_text_parts_for_pattern_matching(msg)
            for externalized_text in externalized_parts:
                if externalized_text and matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns):
                    return True
            externalized_text = "\n".join(externalized_parts)
            if externalized_text and externalized_text != text:
                return matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns)
        return False

    def _content_has_externalized_placeholder_ref(self, content: str) -> bool:
        return bool(extract_externalized_ref(content) or extract_ingest_externalized_refs(content))

    def _has_prior_raw_externalized_placeholder_row(self, store_id: int, msg: Dict[str, Any]) -> bool:
        if not self._session_id:
            return False
        raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
        after_store_id = 0
        while True:
            rows = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=after_store_id,
                limit=1000,
            )
            if not rows:
                return False
            for row in rows:
                row_store_id = int(row.get("store_id") or 0)
                if row_store_id >= store_id:
                    return False
                if self._raw_externalized_placeholder_replay_identity(row) == raw_identity:
                    return True
                after_store_id = max(after_store_id, row_store_id)

    def _mapped_stored_row_matches_ignore_message_patterns(self, msg: Dict[str, Any]) -> bool:
        store_id = msg.get("store_id")
        content = normalize_content_value(msg.get("content")) or ""
        has_externalized_placeholder = self._content_has_externalized_placeholder_ref(content)
        mapped_from_active_placeholder = False
        if store_id is None:
            store_id = self._current_compress_store_ids_by_message_id.get(id(msg))
            mapped_from_active_placeholder = has_externalized_placeholder and store_id is not None
        if store_id is None:
            return False
        if mapped_from_active_placeholder and self._has_prior_raw_externalized_placeholder_row(int(store_id), msg):
            raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
            if self._current_compress_placeholder_identity_counts.get(raw_identity, 0) <= 1:
                return False
        try:
            stored = self._store.get(int(store_id))
        except Exception:
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        return bool(stored and self._matches_ignore_message_patterns(stored, stored_row=True))

    def _copy_active_replay_messages_preserving_generated_ids(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        copied_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in active_replay_messages:
            copied_message = dict(message)
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(copied_message))
            copied_replay_messages.append(copied_message)
        return copied_replay_messages

    def _remember_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in original_messages
        ]
        self._last_active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
            active_replay_messages
        )
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(active_replay_messages)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(active_replay_messages)
        )
        return active_replay_messages

    def _cached_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        identities = [self._message_replay_identity(message) for message in original_messages]
        if identities == getattr(self, "_last_active_replay_source_identities", None):
            cached = getattr(self, "_last_active_replay_messages", None)
            if cached is not None:
                return self._copy_active_replay_messages_preserving_generated_ids(cached)
        return None

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    def _restore_ingest_payload_placeholders_in_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(key, session_id=session_id)
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(val, session_id=session_id)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._restore_ingest_payload_placeholders_in_value(item, session_id=session_id) for item in value]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(self, content: str, *, session_id: str) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(decoded, session_id=session_id)
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )

    def _recovered_content_matches_durable_identity(self, recovered_content: str, durable_content: str) -> bool:
        recovered_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                self._config,
                parse_json_strings=False,
            )
        )
        if recovered_identity_content == durable_content:
            return True
        redaction_names = sorted(set(re.findall(r"\[LCM sensitive redaction: name=([^;\]]+)", durable_content)))
        if not redaction_names or bool(getattr(self._config, "sensitive_patterns_enabled", False)):
            return False
        compat_config = copy.copy(self._config)
        compat_config.sensitive_patterns_enabled = True
        compat_config.sensitive_patterns = redaction_names
        compat_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                compat_config,
                parse_json_strings=False,
            )
        )
        return compat_identity_content == durable_content

    @staticmethod
    def _persisted_output_marker_replay_proof(content: str) -> tuple[str | None, bool]:
        inline_preview_sha256 = _persisted_output_inline_preview_sha256(content)
        preview_sha256 = inline_preview_sha256 or _persisted_output_preview_prefix_digest(content)
        if not preview_sha256:
            return None, False
        allow_redacted_preview_match = inline_preview_sha256 is None and not _has_lossy_sensitive_redaction(content)
        return preview_sha256, allow_redacted_preview_match

    def _has_any_durable_persisted_output_payload_for_marker(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if expected_chars is None or not persisted_output_source_path or not persisted_output_preview_sha256:
            return False
        if recover_hermes_persisted_output_with_file_stat(content) is None:
            return False
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return durable_content is not None

    @classmethod
    def _is_active_context_droppable_identity(cls, identity: tuple[str, str, str, str]) -> bool:
        """Return true for durable rows sanitized out of active replay only."""
        role, content, _tool_call_id, tool_calls = identity
        if role != "assistant" or tool_calls:
            return False
        return _should_drop_active_assistant_message({
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        })

    def _ignored_message_is_quarantinable_assistant(self, msg: Dict[str, Any]) -> bool:
        if self._is_volatile_ignored_quarantine_placeholder(
            msg,
            text_content_for_pattern_matching(msg.get("content")) or "",
        ):
            return True
        identity = self._message_replay_identity(msg)
        if self._is_quarantined_assistant_replay_identity(identity):
            return True
        if not self._matches_ignore_message_patterns(msg):
            return False
        if identity[0] != "assistant":
            return False
        content = normalize_content_value(msg.get("content")) or ""
        return assistant_output_quarantine_reason(content) is not None

    def _redact_active_replay_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        redacted_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in messages:
            redacted_message = dict(message)
            if "content" in redacted_message:
                redacted_content = redact_sensitive_value(
                    redacted_message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
                redacted_message["content"] = redacted_content

            if "tool_calls" in redacted_message:
                redacted_message["tool_calls"] = redact_sensitive_value(
                    redacted_message.get("tool_calls"),
                    self._config,
                    parse_json_strings=True,
                )
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(redacted_message))
            redacted_replay_messages.append(redacted_message)
        return redacted_replay_messages
