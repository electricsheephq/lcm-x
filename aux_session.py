"""Auxiliary / subagent session tracking for the LCM engine (WS5 Seam 5).

The ``AuxiliarySessionMixin`` holds the auxiliary-session lifecycle tracking:
the per-thread auxiliary stack, active/lineage session-id sets, register /
deactivate / handoff, and the host-coupled caller-frame + state.db ancestor
detection. These methods were lifted verbatim out of ``LCMEngine`` and continue
to run bound to the engine instance (``self`` is the ``LCMEngine``), so the
auxiliary state (initialised in ``__init__``, cleared in
``_reset_profile_runtime_state``, guarded by ``_auxiliary_session_lock``) stays
owned by the engine and resolves via ``self`` — as do the callbacks
(``_end_host_fallback_compressor_for_session``, ``_state_db_path``). ``LCMEngine``
mixes this in, so no call site and no test changes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
import threading
import time
import weakref
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

from .sqlite_util import _is_ingest_busy_error

logger = logging.getLogger(__name__)


# --- Explicit subagent lineage (WS5.7) --------------------------------------
#
# Hosts that expose the plugin hook bus fire ``subagent_start`` / ``subagent_stop``
# with the explicit ``child_session_id`` / ``parent_session_id`` linkage when a
# delegate/subagent is spawned (see ``__init__.py`` registration). We record that
# linkage here, keyed by child session id, so auxiliary-session detection can use
# the host's own explicit signal instead of walking the call stack and reading
# private agent attributes. The ``inspect.currentframe()`` frame walk below stays
# as a legacy fallback for hosts that do not fire these hooks.
_SUBAGENT_LINEAGE_LOCK = threading.RLock()
_SUBAGENT_LINEAGE_BY_SESSION_ID: "dict[str, dict[str, str]]" = {}
_SUBAGENT_LINEAGE_MAX = 4096


def record_subagent_start(payload: Dict[str, Any]) -> None:
    """Record an explicit child→parent subagent linkage from a ``subagent_start`` hook."""
    child_session_id = str((payload or {}).get("child_session_id") or "")
    if not child_session_id:
        return
    record = {
        "parent_session_id": str(payload.get("parent_session_id") or ""),
        "child_subagent_id": str(payload.get("child_subagent_id") or ""),
        "parent_subagent_id": str(payload.get("parent_subagent_id") or ""),
        "role": str(payload.get("child_role") or ""),
    }
    with _SUBAGENT_LINEAGE_LOCK:
        _SUBAGENT_LINEAGE_BY_SESSION_ID[child_session_id] = record
        # Bound memory: the child's on_session_start consumes the record long
        # before subagent_stop, so evicting the oldest entries is safe.
        overflow = len(_SUBAGENT_LINEAGE_BY_SESSION_ID) - _SUBAGENT_LINEAGE_MAX
        if overflow > 0:
            for stale in list(_SUBAGENT_LINEAGE_BY_SESSION_ID)[:overflow]:
                _SUBAGENT_LINEAGE_BY_SESSION_ID.pop(stale, None)


def record_subagent_stop(payload: Dict[str, Any]) -> None:
    """Drop an explicit subagent linkage when its ``subagent_stop`` hook fires."""
    child_session_id = str((payload or {}).get("child_session_id") or "")
    if not child_session_id:
        return
    with _SUBAGENT_LINEAGE_LOCK:
        _SUBAGENT_LINEAGE_BY_SESSION_ID.pop(child_session_id, None)
    with _LIVE_CHILD_LOCK:
        for session_id in [sid for sid, root in _LIVE_DELEGATE_CHILD_ROOTS.items() if root == child_session_id]:
            _LIVE_DELEGATE_CHILD_ROOTS.pop(session_id, None)
            _LIVE_DELEGATE_CHILD_ENGINES.pop(session_id, None)
            _LIVE_CHILD_MOVED_WARNED.discard(session_id)


def explicit_subagent_lineage(session_id: str) -> Dict[str, str]:
    """Return the recorded explicit subagent linkage for a session id, or ``{}``."""
    key = str(session_id or "")
    if not key:
        return {}
    with _SUBAGENT_LINEAGE_LOCK:
        record = _SUBAGENT_LINEAGE_BY_SESSION_ID.get(key)
        return dict(record) if record else {}


class AuxiliarySessionMixin:
    def _log_auxiliary_session_decision(self, session_id, *, platform="", parent="", reason="inherited_bypass"):
        try:
            agent = self._delegate_child_agent(session_id)
            auxiliary = agent is not None and _is_auxiliary_agent(agent)
            record = explicit_subagent_lineage(session_id)
            decision = "delegate_child" if not auxiliary and agent is not None and _is_delegate_agent(agent) else "auxiliary"
            reason = ("background_review" if "background_review" in (
                getattr(agent, "_memory_write_origin", ""), getattr(agent, "_memory_write_context", "")
            ) else "toolsets_memory_only") if auxiliary else "subagent_start" if record else (
                "frame_marker" if agent is not None else reason
            )
            parent = parent or record.get("parent_session_id", "") or str(getattr(agent, "_parent_session_id", "") or "")
            self._log_session_decision(session_id, decision, reason, platform=platform, parent=parent)
            return reason
        except Exception:
            return "unknown"  # Diagnostics cannot change bind or ingest outcomes.

    def _thread_context_auxiliary_stack(self) -> list[str]:
        stack = getattr(self._thread_context, "auxiliary_session_stack", None)
        if stack is None:
            current = str(getattr(self._thread_context, "current_auxiliary_session_id", "") or "")
            stack = [current] if current else []
            self._thread_context.auxiliary_session_stack = stack
        return stack

    def _sync_thread_context_current_auxiliary(self) -> list[str]:
        stack = self._thread_context_auxiliary_stack()
        active_ids = self._active_auxiliary_session_ids()
        stack[:] = [session_id for session_id in stack if session_id in active_ids]
        self._thread_context.current_auxiliary_session_id = stack[-1] if stack else ""
        return stack

    def _thread_context_session_id(self) -> str:
        stack = self._sync_thread_context_current_auxiliary()
        stack_session_id = self._in_process_auxiliary_session_id_from_stack()
        if stack_session_id:
            return stack_session_id
        if stack:
            return stack[-1]
        return ""

    def _current_auxiliary_prompt_tokens(self, session_id: str) -> int:
        if not session_id:
            return 0
        caller_generation = self._in_process_auxiliary_caller_generation(session_id)
        with self._auxiliary_session_lock:
            if session_id not in self._auxiliary_session_ids:
                return 0
            active_generation = self._auxiliary_session_generations.get(session_id)
            stack = self._thread_context_auxiliary_stack()
            stack_marks_current_session = bool(
                active_generation is not None
                and caller_generation == 0
                and stack
                and stack[-1] == session_id
            )
            generation_matches = (
                caller_generation == 0
                if active_generation is None
                else caller_generation == active_generation or stack_marks_current_session
            )
            if not generation_matches:
                return 0
            return self._auxiliary_last_prompt_tokens.get(session_id, 0)

    def _thread_context_has_auxiliary_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_session_ids

    def _active_auxiliary_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_session_ids)

    def _known_auxiliary_lineage_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_lineage_session_ids)

    def _known_auxiliary_parent_lineage_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_lineage_session_ids) - set(
                self._auxiliary_foreground_reused_session_ids
            )

    def _has_auxiliary_lineage_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_lineage_session_ids

    def _auxiliary_lineage_suppressed_as_foreground(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_foreground_reused_session_ids

    def _thread_context_stateless(self) -> bool:
        return bool(self._thread_context_session_id())

    def _register_auxiliary_session(
        self,
        session_id: str,
        *,
        preserve_foreground_reuse_marker: bool = False,
    ) -> bool:
        generation = self._in_process_auxiliary_caller_generation(session_id)
        with self._auxiliary_session_lock:
            previous_generation = self._auxiliary_session_generations.get(session_id)
            had_active_session = session_id in self._auxiliary_session_ids
            if generation and self._auxiliary_generation_is_retired(session_id, generation):
                if previous_generation is not None and previous_generation != generation:
                    return False
                if had_active_session and previous_generation is None:
                    return False
                generation = self._in_process_auxiliary_caller_generation(
                    session_id,
                    refresh_retired=True,
                )
            had_cached_usage = session_id in self._auxiliary_last_prompt_tokens
            self._auxiliary_session_ids.add(session_id)
            self._auxiliary_lineage_session_ids.add(session_id)
            if not preserve_foreground_reuse_marker:
                self._auxiliary_foreground_reused_session_ids.discard(session_id)
            if generation:
                if previous_generation != generation:
                    if previous_generation is not None:
                        self._retire_auxiliary_generation(session_id, previous_generation)
                    if self._host_fallback_state_for_session(session_id).get("_host_fallback_session_id") == session_id:
                        self._end_host_fallback_compressor_for_session(
                            session_id,
                            [],
                            current_session_bypasses=True,
                            require_session_match=True,
                        )
                    if (
                        had_active_session
                        or had_cached_usage
                        or previous_generation is not None
                    ):
                        self._auxiliary_direct_end_guard_session_ids.add(session_id)
                    self._auxiliary_last_prompt_tokens.pop(session_id, None)
                self._auxiliary_session_generations[session_id] = generation
                self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            else:
                if previous_generation is not None and had_active_session:
                    return True
                if previous_generation is not None:
                    self._retire_auxiliary_generation(session_id, previous_generation)
                self._auxiliary_last_prompt_tokens.pop(session_id, None)
                self._auxiliary_session_generations.pop(session_id, None)
                self._auxiliary_direct_end_guard_session_ids.discard(session_id)
                self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            return True

    def _retire_auxiliary_generation(self, session_id: str, generation: int | None) -> None:
        if session_id and generation:
            self._auxiliary_retired_session_generations.setdefault(session_id, set()).add(generation)

    def _drop_auxiliary_generation_token(
        self,
        object_id: int,
        token: int,
        token_ref: weakref.ReferenceType[Any],
    ) -> None:
        with self._auxiliary_session_lock:
            existing = self._auxiliary_generation_tokens.get(object_id)
            if existing is not None and existing == (token_ref, token):
                self._auxiliary_generation_tokens.pop(object_id, None)

    def _auxiliary_generation_is_retired(self, session_id: str, generation: int) -> bool:
        return bool(generation) and generation in self._auxiliary_retired_session_generations.get(
            session_id,
            set(),
        )

    def _deactivate_auxiliary_session(self, session_id: str, *, generation: int = 0) -> bool:
        if not session_id:
            return False
        with self._auxiliary_session_lock:
            if self._auxiliary_generation_is_retired(session_id, generation):
                return False
            active_generation = self._auxiliary_session_generations.get(session_id)
            if active_generation is None:
                expected_parent = self._auxiliary_handoff_parent_session_ids.get(session_id)
                caller_parent = self._in_process_parent_session_id(
                    {},
                    session_id=session_id,
                    include_explicit=False,
                )
                if session_id in self._auxiliary_direct_end_guard_session_ids:
                    has_usage = session_id in self._auxiliary_last_prompt_tokens
                    if not generation and not has_usage:
                        return False
                    if expected_parent and caller_parent and caller_parent != expected_parent:
                        return False
            elif generation != active_generation:
                expected_parent = self._auxiliary_handoff_parent_session_ids.get(session_id)
                has_cached_usage = session_id in self._auxiliary_last_prompt_tokens
                if generation != 0 or (
                    session_id in self._auxiliary_direct_end_guard_session_ids
                    and not has_cached_usage
                ):
                    return False
            self._auxiliary_session_ids.discard(session_id)
            self._auxiliary_last_prompt_tokens.pop(session_id, None)
            self._retire_auxiliary_generation(session_id, active_generation or generation)
            self._auxiliary_session_generations.pop(session_id, None)
            self._auxiliary_direct_end_guard_session_ids.discard(session_id)
            self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            return True

    def _mark_thread_context_stateless(
        self,
        session_id: str,
        *,
        preserve_foreground_reuse_marker: bool = False,
    ) -> None:
        if not self._register_auxiliary_session(
            session_id,
            preserve_foreground_reuse_marker=preserve_foreground_reuse_marker,
        ):
            return
        stack = self._thread_context_auxiliary_stack()
        stack[:] = [existing for existing in stack if existing != session_id]
        stack.append(session_id)
        self._thread_context.current_auxiliary_session_id = session_id

    def _clear_thread_context_stateless(self, session_id: str = "") -> None:
        stack = self._thread_context_auxiliary_stack()
        if session_id:
            stack[:] = [existing for existing in stack if existing != session_id]
        else:
            stack.clear()
        self._sync_thread_context_current_auxiliary()
        if not stack:
            for key in ("_last_compression_status", "_last_compression_noop_reason", "_last_compress_aborted"):
                vars(self._thread_context).pop(key, None)

    def _handoff_auxiliary_session(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        preserve_old_session: bool = False,
    ) -> None:
        generation = self._in_process_auxiliary_caller_generation(new_session_id)
        if generation and self._auxiliary_generation_is_retired(new_session_id, generation):
            with self._auxiliary_session_lock:
                if new_session_id:
                    self._auxiliary_lineage_session_ids.add(new_session_id)
            return
        stack = self._thread_context_auxiliary_stack()
        handoff_from_active_thread_marker = old_session_id in stack
        with self._auxiliary_session_lock:
            old_session_was_active = old_session_id in self._auxiliary_session_ids
            if old_session_id:
                if not preserve_old_session:
                    self._auxiliary_last_prompt_tokens.pop(old_session_id, None)
                    self._retire_auxiliary_generation(
                        old_session_id,
                        self._auxiliary_session_generations.get(old_session_id),
                    )
                    self._auxiliary_session_generations.pop(old_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(old_session_id)
                    self._auxiliary_handoff_parent_session_ids.pop(old_session_id, None)
                    self._auxiliary_session_ids.discard(old_session_id)
                self._auxiliary_lineage_session_ids.add(old_session_id)
            if new_session_id:
                previous_new_generation = self._auxiliary_session_generations.get(new_session_id)
                had_new_runtime_state = (
                    new_session_id in self._auxiliary_session_ids
                    or new_session_id in self._auxiliary_last_prompt_tokens
                    or previous_new_generation is not None
                    or new_session_id in self._auxiliary_direct_end_guard_session_ids
                )
                had_new_session_state = (
                    had_new_runtime_state
                    or new_session_id in self._auxiliary_lineage_session_ids
                )
                had_direct_end_guard = new_session_id in self._auxiliary_direct_end_guard_session_ids
                expected_new_parent = self._auxiliary_handoff_parent_session_ids.get(new_session_id)
                stale_generationless_parent_handoff = bool(
                    previous_new_generation is not None
                    and not generation
                    and old_session_id
                    and not old_session_was_active
                    and self._auxiliary_retired_session_generations.get(old_session_id)
                    and old_session_id != expected_new_parent
                )
                if stale_generationless_parent_handoff:
                    self._auxiliary_lineage_session_ids.add(new_session_id)
                    return
                new_generation_replaces_old = (
                    previous_new_generation is not None
                    and (
                        (bool(generation) and previous_new_generation != generation)
                        or (
                            not generation
                            and had_direct_end_guard
                            and (
                                new_session_id not in self._auxiliary_last_prompt_tokens
                                or not expected_new_parent
                                or old_session_id == expected_new_parent
                            )
                        )
                    )
                )
                if had_new_session_state and new_generation_replaces_old:
                    self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                boundary_from_live_new_generation = bool(generation) and previous_new_generation == generation
                if (
                    previous_new_generation is not None
                    and not generation
                    and had_direct_end_guard
                    and new_session_id in self._auxiliary_last_prompt_tokens
                    and expected_new_parent
                    and old_session_id
                    and not old_session_was_active
                    and old_session_id != expected_new_parent
                ):
                    self._auxiliary_lineage_session_ids.add(new_session_id)
                    return
                preserve_new_foreground_marker = (
                    new_session_id in self._auxiliary_foreground_reused_session_ids
                    and not boundary_from_live_new_generation
                )
                if not preserve_new_foreground_marker:
                    self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                    if had_new_session_state and (
                        new_generation_replaces_old
                        or had_direct_end_guard
                        or (bool(generation) and previous_new_generation is None and had_new_runtime_state)
                        or (
                            previous_new_generation is None
                            and not generation
                            and new_session_id in self._auxiliary_lineage_session_ids
                            and (
                                handoff_from_active_thread_marker
                                or (
                                    old_session_id
                                    and had_new_runtime_state
                                    and (
                                        old_session_id in self._auxiliary_lineage_session_ids
                                        or old_session_id in self._auxiliary_session_ids
                                    )
                                )
                            )
                        )
                    ):
                        self._auxiliary_direct_end_guard_session_ids.add(new_session_id)
                        if old_session_id:
                            self._auxiliary_handoff_parent_session_ids[new_session_id] = old_session_id
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_foreground_reused_session_ids.discard(new_session_id)
                    if generation:
                        if (previous_new_generation != generation
                                and self._host_fallback_state_for_session(new_session_id).get(
                                    "_host_fallback_session_id") == new_session_id):
                            self._end_host_fallback_compressor_for_session(
                                new_session_id,
                                [],
                                current_session_bypasses=True,
                                require_session_match=True,
                            )
                        self._auxiliary_session_generations[new_session_id] = generation
                        self._auxiliary_handoff_parent_session_ids.pop(new_session_id, None)
                    elif previous_new_generation is None or new_generation_replaces_old:
                        self._auxiliary_session_generations.pop(new_session_id, None)
                elif generation:
                    if previous_new_generation != generation:
                        if previous_new_generation is not None:
                            self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                        self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                        self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                        if self._host_fallback_state_for_session(new_session_id).get("_host_fallback_session_id") == new_session_id:
                            self._end_host_fallback_compressor_for_session(
                                new_session_id,
                                [],
                                current_session_bypasses=True,
                                require_session_match=True,
                            )
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_session_generations[new_session_id] = generation
                    self._auxiliary_handoff_parent_session_ids.pop(new_session_id, None)
                else:
                    if previous_new_generation is not None:
                        self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                    self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                    if self._host_fallback_state_for_session(new_session_id).get("_host_fallback_session_id") == new_session_id:
                        self._end_host_fallback_compressor_for_session(
                            new_session_id,
                            [],
                            current_session_bypasses=True,
                            require_session_match=True,
                        )
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_session_generations.pop(new_session_id, None)
                    if old_session_id:
                        self._auxiliary_handoff_parent_session_ids[new_session_id] = old_session_id
                self._auxiliary_lineage_session_ids.add(new_session_id)
        stack = self._thread_context_auxiliary_stack()
        had_thread_marker = old_session_id in stack or new_session_id in stack
        stack[:] = [
            existing
            for existing in stack
            if existing not in {old_session_id, new_session_id}
        ]
        if had_thread_marker and new_session_id:
            stack.append(new_session_id)
        self._sync_thread_context_current_auxiliary()

    def _unmark_thread_context_auxiliary_session(
        self,
        session_id: str,
        *,
        suppress_as_foreground_reuse: bool = True,
    ) -> None:
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.discard(session_id)
            if suppress_as_foreground_reuse and session_id in self._auxiliary_lineage_session_ids:
                self._auxiliary_foreground_reused_session_ids.add(session_id)
            self._auxiliary_last_prompt_tokens.pop(session_id, None)
            self._retire_auxiliary_generation(
                session_id,
                self._auxiliary_session_generations.pop(session_id, None),
            )
            self._auxiliary_direct_end_guard_session_ids.discard(session_id)
            self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
        self._clear_thread_context_stateless(session_id)

    def _caller_is_auxiliary_agent_frame(self, caller_self: Any) -> bool:
        if caller_self is None:
            return False
        if getattr(caller_self, "_subagent_id", None):
            return True
        if getattr(caller_self, "_parent_subagent_id", None):
            return True
        try:
            if int(getattr(caller_self, "_delegate_depth", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        memory_origin = str(getattr(caller_self, "_memory_write_origin", "") or "")
        memory_context = str(getattr(caller_self, "_memory_write_context", "") or "")
        if memory_origin == "background_review" or memory_context == "background_review":
            return True
        log_prefix = str(getattr(caller_self, "log_prefix", "") or "").strip()
        if log_prefix.startswith("[subagent-"):
            return True
        enabled_toolsets = getattr(caller_self, "enabled_toolsets", None)
        if enabled_toolsets is not None:
            try:
                toolsets = {str(toolset) for toolset in enabled_toolsets}
            except TypeError:
                toolsets = set()
            if toolsets and toolsets <= {"memory", "skills"}:
                return True
        if getattr(caller_self, "ephemeral_system_prompt", None) and log_prefix.startswith("[subagent-"):
            return True
        return False

    def _auxiliary_generation_token_for(self, caller_self: object, *, force_new: bool = False) -> int:
        """Return a process-local, non-reusable token for an auxiliary frame.

        ``id(obj)`` can be reused after GC, which makes retired-generation
        checks incorrectly reject later live frames. Key by id for lookup speed,
        but verify object identity with ``is`` before reusing a token. Weak refs
        avoid retaining ordinary frames; objects that cannot be weak-referenced
        are held strongly so their id cannot be recycled while the token remains
        live in this engine runtime.
        """
        with self._auxiliary_session_lock:
            object_id = id(caller_self)
            existing = self._auxiliary_generation_tokens.get(object_id)
            if existing is not None and not force_new:
                existing_ref, token = existing
                if isinstance(existing_ref, weakref.ReferenceType):
                    existing_obj = existing_ref()
                else:
                    existing_obj = existing_ref
                if existing_obj is caller_self:
                    return token

            self._auxiliary_next_generation_token += 1
            token = self._auxiliary_next_generation_token
            try:
                token_ref: Any = weakref.ref(
                    caller_self,
                    lambda ref, object_id=object_id, token=token: self._drop_auxiliary_generation_token(
                        object_id,
                        token,
                        ref,
                    ),
                )
            except TypeError:
                token_ref = caller_self
            self._auxiliary_generation_tokens[object_id] = (token_ref, token)
            return token

    def _in_process_auxiliary_caller_generation(
        self,
        session_id: str,
        *,
        refresh_retired: bool = False,
    ) -> int:
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return 0
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                caller_session = str(getattr(caller_self, "session_id", "") or "")
                if not session_id or caller_session == session_id:
                    token = self._auxiliary_generation_token_for(caller_self)
                    if refresh_retired and self._auxiliary_generation_is_retired(
                        session_id,
                        token,
                    ):
                        token = self._auxiliary_generation_token_for(caller_self, force_new=True)
                    return token
                frame = frame.f_back
        finally:
            del frame
        return 0

    def _in_process_parent_session_id(
        self,
        kwargs: Dict[str, Any],
        session_id: str = "",
        include_explicit: bool = True,
        require_auxiliary_frame: bool = True,
    ) -> str:
        explicit = str(kwargs.get("parent_session_id") or "")
        if include_explicit and explicit:
            return explicit
        target_session_id = str(session_id or kwargs.get("session_id") or "")
        if include_explicit and target_session_id:
            explicit_parent = str(
                explicit_subagent_lineage(target_session_id).get("parent_session_id") or ""
            )
            if explicit_parent:
                return explicit_parent
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if require_auxiliary_frame and not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                parent = str(getattr(caller_self, "_parent_session_id", "") or "")
                caller_session = str(getattr(caller_self, "session_id", "") or "")
                if parent and caller_session and (
                    not target_session_id or caller_session == target_session_id
                ):
                    return parent
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _in_process_auxiliary_session_id_from_stack(self) -> str:
        active_ids = self._active_auxiliary_session_ids()
        lineage_ids = self._known_auxiliary_lineage_session_ids()
        if not active_ids and not lineage_ids and not self._session_id:
            return ""
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                session_id = str(getattr(caller_self, "session_id", "") or "")
                parent_id = str(getattr(caller_self, "_parent_session_id", "") or "")
                if session_id and parent_id and (
                    session_id in active_ids
                    or session_id in lineage_ids
                    or parent_id == self._session_id
                    or parent_id in lineage_ids
                ):
                    return session_id
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _is_live_auxiliary_child_session(
        self,
        session_id: str,
        parent_session_id: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Return True when a same-process child agent should not rebind LCM.

        Detect Hermes auxiliary/background child sessions without treating real
        foreground branches as stateless. In-process auxiliary agent frames are
        trusted even when this engine is fresh and has no bound foreground yet.
        Explicit parent metadata by itself is not enough, because legitimate
        foreground branches can also carry parent ids before their state.db row
        is visible to the plugin.
        """
        if not session_id or session_id == parent_session_id:
            return False
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        known_auxiliary_parent_ids = self._known_auxiliary_parent_lineage_session_ids()
        explicit_parent_id = str(kwargs.get("parent_session_id") or "")
        in_process_parent_id = self._in_process_parent_session_id(
            kwargs,
            session_id,
            include_explicit=False,
        )
        if in_process_parent_id:
            if not parent_session_id or in_process_parent_id == parent_session_id:
                return True
            if in_process_parent_id in known_auxiliary_ids:
                return True
        if explicit_parent_id:
            if self._thread_context_has_auxiliary_session(explicit_parent_id):
                return True
            if explicit_parent_id in known_auxiliary_parent_ids and explicit_parent_id != self._session_id:
                return True
            if (
                explicit_parent_id in known_auxiliary_ids
                and self._lcm_session_last_bypassed.get(explicit_parent_id)
            ):
                return True
            return False
        if not parent_session_id:
            return False

        path = self._state_db_path(kwargs)
        if not path.exists():
            return False
        try:
            uri = path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT
                        child.parent_session_id,
                        child.started_at,
                        child.ended_at,
                        parent.id,
                        parent.ended_at
                    FROM sessions AS child
                    LEFT JOIN sessions AS parent
                        ON parent.id = child.parent_session_id
                    WHERE child.id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary child session probe failed: %s", exc)
            return False
        if not row:
            return False
        child_parent_id, child_started_at, child_ended_at, actual_parent_id, parent_ended_at = row
        if child_ended_at is not None or actual_parent_id is None:
            return False

        active_auxiliary_ids = self._active_auxiliary_session_ids()
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        known_auxiliary_parent_ids = self._known_auxiliary_parent_lineage_session_ids()
        bypassed_auxiliary_ids = {
            str(auxiliary_id or "")
            for auxiliary_id in known_auxiliary_ids
            if self._lcm_session_last_bypassed.get(str(auxiliary_id or ""))
        }
        if child_parent_id in active_auxiliary_ids:
            return True
        if child_parent_id in known_auxiliary_parent_ids and child_parent_id != self._session_id:
            return True
        if child_parent_id in bypassed_auxiliary_ids:
            return True
        if child_parent_id != parent_session_id:
            return self._session_has_auxiliary_ancestor(
                str(child_parent_id or ""),
                known_auxiliary_parent_ids | active_auxiliary_ids | bypassed_auxiliary_ids,
                path,
            )
        return False

    def _session_has_auxiliary_ancestor(
        self,
        session_id: str,
        auxiliary_lineage_ids: set[str],
        state_db_path: Path,
    ) -> bool:
        if not session_id or not auxiliary_lineage_ids or not state_db_path.exists():
            return False
        visited: set[str] = set()
        current = session_id
        try:
            uri = state_db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                for _ in range(32):
                    if not current or current in visited:
                        return False
                    if current in auxiliary_lineage_ids:
                        return True
                    visited.add(current)
                    row = conn.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                    if not row:
                        return False
                    current = str(row[0] or "")
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary ancestor probe failed: %s", exc)
            return False
        return False


# --- Delegated subagent sessions (#1931) ------------------------------------
#
# A session the host's delegate tool started (a DELEGATE child) stays bypassed for context management, but
# its rows are stored under the child's own session id and conversation id, with a ``session_lineage`` link
# to its parent. AUXILIARY sessions (background review, memory/skills-only toolsets, auxiliary model calls)
# are never stored; when both classes match, AUXILIARY wins. Rows reach the store only through the
# non-current-session route (a stored-prefix match, or a unique tail anchor, plus
# ``_append_off_current_session_end_suffix``), so an engine bound to another session keeps its binding,
# cursor, frontier, lifecycle state and DAG. Classification and resume boundaries are shared by clones.

_DELEGATE_SOURCE = "subagent"  # the host's platform (and state.db ``sessions.source``) for a delegate child
_ANCHOR_ROWS = 3
_FRAME_DEPTH = 48
_COUNTS_LOCK = threading.Lock()
_COUNTS: Counter = Counter()
# Shared by clones and hook engines: compare and append form one process-local operation.
_DELEGATE_STORE_LOCK = threading.RLock()
_LIVE_CHILD_STORE_WAIT_S = 1.0
# #2482: live delegated children, by session id -> the root child id that subagent_start named. Filled when a child
# binds its own engine on its own thread as a delegate (its frames are visible there, not on the host's hook worker
# thread) and carried across its compression handoff; a root's subagent_stop ends its entries.
_LIVE_CHILD_LOCK = threading.Lock()
_LIVE_DELEGATE_CHILD_ROOTS: dict[str, str] = {}
# The engine the child bound and the storage it served then, by session id (weak reference, identity): the early
# store writes only through that engine and only while it still serves that storage, so a child's rows never land in
# another profile's database (the plugin prototype's, or a later rebind of the child's engine).
_LIVE_DELEGATE_CHILD_ENGINES: dict[str, tuple] = {}
_LIVE_CHILD_MOVED_WARNED: set[str] = set()
_LIVE_CHILD_STORES_IN_FLIGHT: set[str] = set()
_LIVE_CHILD_PENDING: dict[str, Any] = {}  # the newest store handed over while one was in flight, per child
_LIVE_CHILD_HANDED_ROWS: dict[str, int] = {}  # rows in the newest list handed over, per child with a store in flight
_LIVE_CHILD_STORED_ROWS: dict[str, int] = {}  # rows in the newest list whose store succeeded, per such child
_LIVE_CHILD_WARNED: set[str] = set()  # children whose stuck store a shutdown drain already reported
_LIVE_CHILD_STORE_JOIN_S = 2.0  # the most a host hook waits for a live-child store; the store itself may finish later
_LIVE_CHILD_SHUTDOWN_DRAIN_S = 5.0  # the most an engine shutdown waits for in-flight or queued live-child stores
LIVE_STORE_RETRY = "retry"  # a live-child store that found the database busy: its worker tries the same list again
_LIVE_CHILD_STORE_RETRIES = 3  # retries per list, unless a newer list is handed over first
_LIVE_CHILD_RETRY_DELAY_S = 0.25  # times the retry number
_DELEGATE_AUXILIARY_IDS: set[tuple[str, str]] = set()
_DELEGATE_RESUME_BOUNDARIES: dict[tuple[str, str], list[bytes]] = {}
_DELEGATE_COMPRESSION_PARENTS: dict[tuple[str, str], str] = {}
# #2071: digests of the rows the child's last bypassed compaction produced (absent from its input). They are the only
# rows the stored-tail anchor may pass over without finding them stored.
_DELEGATE_COMPACTION_OUTPUTS: dict[tuple[str, str], list[bytes]] = {}


def live_delegate_child_root(session_id: str) -> str:
    """The root child id of a live delegated child bound in this process, or ``""``."""
    with _LIVE_CHILD_LOCK:
        return _LIVE_DELEGATE_CHILD_ROOTS.get(str(session_id or ""), "")


def live_child_storage_identity(engine: Any) -> tuple[str, str]:
    """The profile home and database an engine serves now."""
    return str(getattr(engine, "_hermes_home", "") or ""), str(getattr(engine, "_db_path", "") or "")


def live_delegate_child_engine(session_id: str) -> tuple[Any, tuple]:
    """The engine a live delegated child bound in this process (if still alive) and the storage it served then."""
    with _LIVE_CHILD_LOCK:
        entry = _LIVE_DELEGATE_CHILD_ENGINES.get(str(session_id or ""))
    if entry is None:
        return None, ()
    return entry[0](), entry[1]


def warn_live_child_moved(session_id: str) -> None:
    """One WARNING per child whose engine now serves another profile; its rows are left to the end store."""
    with _LIVE_CHILD_LOCK:
        if session_id in _LIVE_CHILD_MOVED_WARNED:
            return
        _LIVE_CHILD_MOVED_WARNED.add(session_id)
    logger.warning(
        "LCM live delegated-child store skipped: the child's engine now serves another profile than when the child "
        "started; child=%s is left to its end store (#2482)", hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
    )


def run_live_child_store(session_id: str, operation, rows: int = 0) -> None:
    """Run ``operation`` on a worker thread and wait at most ``_LIVE_CHILD_STORE_JOIN_S`` (#2482); never raises.

    A host hook must stay far below its own timeout, and a store can wait on SQLite past the lock bounds. One store
    per child runs at a time: a list handed over meanwhile replaces any older pending one and runs right after it
    (each list holds every earlier row, and the store appends only what is not stored yet).
    """
    with _LIVE_CHILD_LOCK:
        _LIVE_CHILD_HANDED_ROWS[session_id] = rows
        if session_id in _LIVE_CHILD_STORES_IN_FLIGHT:
            _LIVE_CHILD_PENDING[session_id] = (operation, rows)
            _count("live_store_queued")
            return
        _LIVE_CHILD_STORES_IN_FLIGHT.add(session_id)

    def finish():  # under _LIVE_CHILD_LOCK
        _LIVE_CHILD_STORES_IN_FLIGHT.discard(session_id)
        _LIVE_CHILD_PENDING.pop(session_id, None)
        _LIVE_CHILD_HANDED_ROWS.pop(session_id, None)
        _LIVE_CHILD_STORED_ROWS.pop(session_id, None)
        _LIVE_CHILD_WARNED.discard(session_id)

    def runner():
        current = (operation, rows)
        try:
            while current is not None:
                for attempt in range(_LIVE_CHILD_STORE_RETRIES + 1):
                    try:
                        stored = current[0]()
                    except Exception as exc:
                        logger.debug("LCM live delegated-child store error: %s", exc)
                        break
                    if stored is True:  # only a store that really ran counts its rows as stored
                        with _LIVE_CHILD_LOCK:
                            _LIVE_CHILD_STORED_ROWS[session_id] = current[1]
                        break
                    with _LIVE_CHILD_LOCK:
                        superseded = session_id in _LIVE_CHILD_PENDING
                    if stored != LIVE_STORE_RETRY or superseded or attempt == _LIVE_CHILD_STORE_RETRIES:
                        break
                    time.sleep(_LIVE_CHILD_RETRY_DELAY_S * (attempt + 1))  # busy: the same list again, shortly
                with _LIVE_CHILD_LOCK:
                    current = _LIVE_CHILD_PENDING.pop(session_id, None)
                    if current is None:
                        finish()
        finally:
            if current is not None:  # something beyond Exception escaped: never leave the child marked in flight
                with _LIVE_CHILD_LOCK:
                    finish()

    try:
        worker = threading.Thread(target=runner, name="lcm-live-child-store", daemon=True)
        worker.start()
    except Exception as exc:
        with _LIVE_CHILD_LOCK:
            finish()
        logger.debug("LCM live delegated-child store not started: %s", exc)
        return
    worker.join(_LIVE_CHILD_STORE_JOIN_S)


def drain_live_child_stores(timeout: float) -> None:
    """Wait at most ``timeout`` seconds for in-flight live-child stores, queued ones included (engine shutdown)."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _LIVE_CHILD_LOCK:
            if not _LIVE_CHILD_STORES_IN_FLIGHT - _LIVE_CHILD_WARNED:
                return  # a child an earlier drain already reported stuck is not waited for again
        if time.monotonic() >= deadline:
            with _LIVE_CHILD_LOCK:
                waiting = {sid: max(0, _LIVE_CHILD_HANDED_ROWS.get(sid, 0) - _LIVE_CHILD_STORED_ROWS.get(sid, 0))
                           for sid in _LIVE_CHILD_STORES_IN_FLIGHT if sid not in _LIVE_CHILD_WARNED}
                _LIVE_CHILD_WARNED.update(waiting)
            for sid, unstored in sorted(waiting.items()):
                # NOT SILENT (#2482): this child's newest rows are lost if the process ends before the store lands.
                logger.warning(
                    "LCM shutdown drain timed out after %.1fs: a live delegated-child store is still waiting "
                    "(likely a busy database); child=%s has up to %d handed-over row(s) not yet stored, lost if "
                    "the process ends first (#2483)",
                    timeout, hashlib.sha256(sid.encode("utf-8")).hexdigest()[:12], unstored,
                )
            return
        time.sleep(0.01)


def delegate_child_store_counts() -> Dict[str, int]:
    """Process-lifetime counts: ``rows_stored`` and ``fallbacks`` (rows a delegated child could not store)."""
    with _COUNTS_LOCK:
        return dict(_COUNTS)


def _count(key: str, amount: int = 1) -> None:
    with _COUNTS_LOCK:
        _COUNTS[key] += amount


def _is_auxiliary_agent(agent: Any) -> bool:
    origin = str(getattr(agent, "_memory_write_origin", "") or "")
    context = str(getattr(agent, "_memory_write_context", "") or "")
    if "background_review" in (origin, context):
        return True
    toolsets = getattr(agent, "enabled_toolsets", None)
    try:
        names = {str(toolset) for toolset in toolsets} if toolsets is not None else set()
    except TypeError:
        names = set()
    return bool(names) and names <= {"memory", "skills"}


def _is_delegate_agent(agent: Any) -> bool:
    if getattr(agent, "_subagent_id", None) or getattr(agent, "_parent_subagent_id", None):
        return True
    try:
        if int(getattr(agent, "_delegate_depth", 0) or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return str(getattr(agent, "log_prefix", "") or "").strip().startswith("[subagent-")


class DelegateChildMixin:
    def _delegate_child_key(self, session_id: str) -> tuple[str, str]:
        return (str(self._store.db_path), session_id)

    def _remember_delegate_child_auxiliary(self, session_id: str) -> None:
        agent = self._delegate_child_agent(session_id)
        if agent is not None and _is_auxiliary_agent(agent):
            with _DELEGATE_STORE_LOCK:
                _DELEGATE_AUXILIARY_IDS.add(self._delegate_child_key(session_id))
        elif agent is not None and _is_delegate_agent(agent):
            with _LIVE_CHILD_LOCK:
                _LIVE_DELEGATE_CHILD_ROOTS.setdefault(session_id, session_id)
                _LIVE_DELEGATE_CHILD_ENGINES[session_id] = (weakref.ref(self), live_child_storage_identity(self))

    def _stored_delegate_child_link(self, session_id: str) -> Dict[str, str]:
        if not self._store._session_lineage_table_exists():
            return {}
        columns = ("parent_session_id", "source", "child_subagent_id", "parent_subagent_id", "child_role")
        row = self._store.connection.execute(
            "SELECT parent_session_id, source, child_subagent_id, parent_subagent_id, child_role "
            "FROM session_lineage WHERE child_session_id = ?", (session_id,)).fetchone()
        return dict(zip(columns, (str(value or "") for value in row))) if row else {}

    def _is_known_delegate_child(self, session_id: str) -> bool:
        key = self._delegate_child_key(session_id)
        return bool(self._stored_delegate_child_link(session_id)
                    or key in _DELEGATE_RESUME_BOUNDARIES or key in _DELEGATE_COMPRESSION_PARENTS)

    def _delegate_child_store_sessions(self, session_id: str) -> list[str]:
        sessions = []
        while session_id and session_id not in sessions:
            sessions.append(session_id)
            link = self._stored_delegate_child_link(session_id)
            parent = _DELEGATE_COMPRESSION_PARENTS.get(self._delegate_child_key(session_id), "")
            session_id = link["parent_session_id"] if link.get("source") == "compression" else parent
        return sessions

    @staticmethod
    def _delegate_child_boundary_digest(identity: tuple) -> bytes:
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8", "surrogatepass")).digest()

    @staticmethod
    def _delegate_child_unique_end(identities: list[tuple] | list[bytes],
                                   anchor: list[tuple] | list[bytes]) -> Optional[int]:
        if not anchor:
            return None
        width = len(anchor)
        ends = [index + width for index in range(len(identities) - width + 1)
                if identities[index:index + width] == anchor]
        return ends[0] if len(ends) == 1 else None

    def _remember_delegate_child_boundary(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        identities = [self._session_end_prefix_compare_identity(msg, session_id=session_id) for msg in messages
                      if isinstance(msg, dict) and not self._matches_ignore_message_patterns(msg)]
        tail = identities[-_ANCHOR_ROWS:]
        # Repeated tails need the full output to identify the actual end uniquely.
        boundary = tail if self._delegate_child_unique_end(identities, tail) is not None else identities
        _DELEGATE_RESUME_BOUNDARIES[self._delegate_child_key(session_id)] = [
            self._delegate_child_boundary_digest(identity) for identity in boundary]

    def _remember_delegate_child_outputs(self, session_id: str, messages, result) -> None:
        def identities(rows):
            return [self._session_end_prefix_compare_identity(msg, session_id=session_id) for msg in rows
                    if isinstance(msg, dict) and not self._matches_ignore_message_patterns(msg)]
        produced = Counter(identities(result)) - Counter(identities(messages))
        _DELEGATE_COMPACTION_OUTPUTS[self._delegate_child_key(session_id)] = [
            self._delegate_child_boundary_digest(identity) for identity in produced.elements()]

    def _compress_delegate_child_bypassed(self, messages, current_tokens, focus_topic, force):
        session_id = self._bypass_lcm_session_id()
        stored_tail, covered = None, False
        with _DELEGATE_STORE_LOCK:
            # Closing may have begun while this pass waited for the child-store lock.
            if self._compaction_is_closing():
                return messages
            store_outcome: Dict[str, Any] = {}
            pre_store = self._store_delegate_child_turn(
                session_id, messages, platform=_DELEGATE_SOURCE, require_agent=True, outcome=store_outcome)
            if pre_store is None and _is_ingest_busy_error(store_outcome.get("error") or Exception()):
                # #2306: compacting now would drop rows that were never stored. A lock clears, so keep the view;
                # the next compression or turn retries the store. Other failures compact as before (a store
                # that keeps failing saves nothing by stalling the child) and stay visible below.
                self._note_delegate_child_fallback(session_id, "compaction skipped: input not stored")
                state = self._bypass_compress_state()
                state["_last_compression_status"] = "noop"
                state["_last_compression_noop_reason"] = "compaction skipped: input not stored"
                return messages
            try:
                if self._delegate_child_link(session_id, platform=_DELEGATE_SOURCE, require_agent=True):
                    tail = self._store.get_session_tail(session_id, limit=1)
                    view = [msg for msg in messages
                            if isinstance(msg, dict) and not self._matches_ignore_message_patterns(msg)]
                    # The input is covered only when this pass stored it and its last row is the stored tail. After a
                    # failed store, a matching tail can be a copy of the last stored row behind an unstored one (#2071),
                    # so the summary is kept: a surplus head, never a lost one.
                    covered = bool(pre_store is not None and tail and view and self._session_end_prefix_compare_identity(
                        tail[-1], session_id=session_id,
                    ) == self._session_end_prefix_compare_identity(view[-1], session_id=session_id))
                    if covered:
                        stored_tail = tail
                    else:
                        self._note_delegate_child_fallback(
                            session_id, "compression boundary not recorded: input not stored")
            except Exception as exc:
                self._note_delegate_child_fallback(session_id, f"compression boundary not recorded: {exc}")
        result = self._compress_bypassed(messages, current_tokens, focus_topic, force)
        if self._compaction_is_closing():
            return messages
        try:
            with _DELEGATE_STORE_LOCK:
                # An earlier compaction's outputs never exempt a row once this one has replaced the view.
                _DELEGATE_COMPACTION_OUTPUTS.pop(self._delegate_child_key(session_id), None)
                if self._delegate_child_link(session_id, platform=_DELEGATE_SOURCE, require_agent=True):
                    try:  # recorded apart from the boundary: it must survive a failed or dropped boundary
                        # Only over a covered input: a summary of a row that was never stored is that row's only
                        # trace, so it must be stored like any other row.
                        if covered:
                            self._remember_delegate_child_outputs(session_id, messages, result)
                    except Exception as exc:
                        self._note_delegate_child_fallback(session_id, f"compaction outputs not recorded: {exc}")
                    if stored_tail and self._store.get_session_tail(session_id, limit=1) == stored_tail:
                        self._remember_delegate_child_boundary(session_id, result)
                    else:
                        # A concurrent store advanced the tail: its rows must not be replayed from an older boundary.
                        _DELEGATE_RESUME_BOUNDARIES.pop(self._delegate_child_key(session_id), None)
        except Exception as exc:
            self._note_delegate_child_fallback(session_id, f"compression boundary not recorded: {exc}")
        return result

    def _handoff_delegate_child_compression(self, old_session_id: str, new_session_id: str) -> None:
        try:
            with _LIVE_CHILD_LOCK:
                root = _LIVE_DELEGATE_CHILD_ROOTS.get(old_session_id)
                if root and new_session_id:
                    _LIVE_DELEGATE_CHILD_ROOTS[new_session_id] = root
                    if old_session_id in _LIVE_DELEGATE_CHILD_ENGINES:
                        _LIVE_DELEGATE_CHILD_ENGINES[new_session_id] = _LIVE_DELEGATE_CHILD_ENGINES[old_session_id]
            with _DELEGATE_STORE_LOCK:
                old_key, new_key = self._delegate_child_key(old_session_id), self._delegate_child_key(new_session_id)
                if old_key in _DELEGATE_AUXILIARY_IDS:
                    _DELEGATE_AUXILIARY_IDS.add(new_key)
                    return
                if not self._is_known_delegate_child(old_session_id):
                    return
                if not self._stored_delegate_child_link(old_session_id):
                    try:
                        link = self._delegate_child_link(old_session_id, platform=_DELEGATE_SOURCE)
                        if link:
                            self._store.record_session_lineage(old_session_id, **link)
                        else:
                            self._note_delegate_child_fallback(
                                old_session_id, "parent link not recorded during compression handoff: no parent identity resolves")
                    except Exception as exc:
                        self._note_delegate_child_fallback(
                            old_session_id, f"parent link not recorded during compression handoff: {exc}")
                _DELEGATE_COMPRESSION_PARENTS[new_key] = old_session_id
                if old_key in _DELEGATE_RESUME_BOUNDARIES:
                    _DELEGATE_RESUME_BOUNDARIES[new_key] = _DELEGATE_RESUME_BOUNDARIES[old_key]
                if old_key in _DELEGATE_COMPACTION_OUTPUTS:
                    _DELEGATE_COMPACTION_OUTPUTS[new_key] = _DELEGATE_COMPACTION_OUTPUTS[old_key]
                self._store.record_session_lineage(new_session_id, old_session_id, source="compression")
                try:
                    stored_link = self._stored_delegate_child_link(new_session_id)
                    if stored_link.get("source") == "compression" and stored_link.get("parent_session_id") == old_session_id:
                        _DELEGATE_COMPRESSION_PARENTS.pop(new_key, None)
                except Exception:
                    pass  # Cleanup alone must not change storage or fallback counts.
        except Exception as exc:
            self._note_delegate_child_fallback(new_session_id, f"compression parent link not recorded: {exc}")

    def _delegate_child_agent(self, session_id: str) -> Any:
        """The running agent object for ``session_id`` on this call stack, or None."""
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(_FRAME_DEPTH):
                if frame is None:
                    return None
                caller = frame.f_locals.get("self")
                if caller is not None and caller is not self and \
                        str(getattr(caller, "session_id", "") or "") == session_id:
                    return caller
                frame = frame.f_back
        finally:
            del frame
        return None

    def _host_session_parent(self, session_id: str) -> tuple[str, str]:
        """``(parent_session_id, source)`` of the host state.db row, read-only; ``("", "")`` when unknown."""
        try:
            path = self._state_db_path({})
            if not path.exists():
                return "", ""
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                row = conn.execute("SELECT parent_session_id, source FROM sessions WHERE id = ? LIMIT 1",
                                   (session_id,)).fetchone()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM delegated-child state.db probe failed: %s", exc)
            return "", ""
        return (str(row[0] or ""), str(row[1] or "")) if row else ("", "")

    def _delegate_child_link(self, session_id: str, *, platform: str = "",
                             require_agent: bool = False) -> Optional[Dict[str, str]]:
        """The parent link of a DELEGATE child, or None for an AUXILIARY or unproven session."""
        agent = self._delegate_child_agent(session_id)
        if agent is not None and _is_auxiliary_agent(agent):
            self._log_auxiliary_session_decision(session_id, platform=platform)
            _DELEGATE_AUXILIARY_IDS.add(self._delegate_child_key(session_id))
            self._log_session_ingest_outcome(session_id, reason="auxiliary")
            return None  # AUXILIARY wins over every delegate signal
        if self._delegate_child_key(session_id) in _DELEGATE_AUXILIARY_IDS:
            self._log_session_ingest_outcome(session_id, reason="auxiliary_lineage")
            return None
        platform = platform or self._lcm_session_last_platform.get(session_id, "") or _DELEGATE_SOURCE
        if (agent is None and require_agent) or self._session_id_matches_lcm_bypass_filters(
            session_id, platform=platform,
        ):
            self._log_session_ingest_outcome(session_id, reason="no_agent" if agent is None and require_agent else "bypass_pattern")
            return None
        record = explicit_subagent_lineage(session_id)
        stored_link = self._stored_delegate_child_link(session_id)
        compression_parent = _DELEGATE_COMPRESSION_PARENTS.get(self._delegate_child_key(session_id), "")
        delegate = bool(record or stored_link or compression_parent) or (agent is not None and _is_delegate_agent(agent))
        with self._auxiliary_session_lock:
            bind_parent = self._delegate_child_bind_parent_ids.get(session_id, "")
        candidates = [("subagent_start", record.get("parent_session_id", "")), ("bind_kwargs", bind_parent),
                      ("parent_frame", str(getattr(agent, "_parent_session_id", "") or ""))]
        if not delegate or not any(parent for _, parent in candidates):
            host_parent, host_source = self._host_session_parent(session_id)
            delegate = delegate or (bool(host_parent) and host_source == _DELEGATE_SOURCE)
            candidates.append(("state_db", host_parent))
        source, parent = (stored_link["source"], stored_link["parent_session_id"]) if stored_link else next(
            ((name, value) for name, value in [("compression", compression_parent), *candidates] if value), ("", ""))
        if not delegate:
            # Not a skip: the caller still decides this session's route (#2042).
            return None
        if not parent or parent == session_id:
            self._note_delegate_child_fallback(session_id, "rows not stored: no parent identity resolves")
            self._log_session_ingest_outcome(session_id, reason="no_parent")
            return None
        if parent == self._session_id:
            parent_platform = self._session_platform
            if self._session_ignored or (self._session_stateless and not self._is_known_delegate_child(parent)):
                self._log_session_ingest_outcome(session_id, reason="ignored_parent")
                return None
        else:
            parent_platform = self._lcm_session_last_platform.get(parent, "") or self._host_session_parent(parent)[1]
        if self._session_id_matches_lcm_bypass_filters(parent, platform=parent_platform):
            self._log_session_ingest_outcome(session_id, reason="parent_bypass_pattern")
            return None
        self._log_session_decision(session_id, "delegate_child",
                                   "state_db_subagent" if source == "state_db" else "frame_marker" if source == "parent_frame" else source,
                                   platform=platform, parent=parent)
        return {
            "parent_session_id": parent, "source": source,
            "child_subagent_id": record.get("child_subagent_id") or str(getattr(agent, "_subagent_id", "") or ""),
            "parent_subagent_id": record.get("parent_subagent_id")
            or str(getattr(agent, "_parent_subagent_id", "") or ""),
            "child_role": record.get("role") or str(getattr(agent, "_delegate_role", "") or ""),
        }

    def _store_delegate_child_turn(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        platform: str = "",
        after_ingest: bool = False,
        require_agent: bool = False,
        outcome: Optional[dict] = None,
        lock_wait: Optional[float] = None,
    ) -> Optional[int]:
        """Store a delegated child's new rows; return their count, or None on failure. Never raises."""
        try:
            session_id = str(session_id or "")
            if not session_id or not messages:
                return 0
            if self._session_id and not self._bypasses_lcm_context_management() and (
                after_ingest or session_id == self._session_id
            ):
                return 0  # this engine's own ingest owns the view
            def store_locked():
                link = self._delegate_child_link(session_id, platform=platform, require_agent=require_agent)
                if link is None:
                    return 0
                if isinstance(outcome, dict):
                    outcome["linked"] = True
                with self._attribution_span("delegate_child_store", len(messages)):
                    return self._append_delegate_child_rows(session_id, messages, link)

            if lock_wait is None:
                with _DELEGATE_STORE_LOCK:
                    return store_locked()
            if not _DELEGATE_STORE_LOCK.acquire(timeout=lock_wait):
                if isinstance(outcome, dict):
                    outcome["skipped"] = True
                _count("live_store_skipped")
                logger.debug(
                    "LCM delegated child %s: store lock busy, left to the next hook or the end store",
                    session_id,
                )
                return 0
            try:
                return store_locked()
            finally:
                _DELEGATE_STORE_LOCK.release()
        except Exception as exc:
            if isinstance(outcome, dict):
                outcome["error"] = exc
            self._note_delegate_child_fallback(session_id, f"rows not stored: {type(exc).__name__}: {exc}")
            return None

    def _append_delegate_child_rows(self, session_id: str, messages: List[Dict[str, Any]], link: Dict[str, str]) -> int:
        view = [msg for msg in messages if isinstance(msg, dict) and not self._matches_ignore_message_patterns(msg)]
        if not view:
            return 0
        try:
            stored_link = self._stored_delegate_child_link(session_id)
            if not stored_link:
                self._store.record_session_lineage(session_id, link["parent_session_id"], source=link["source"],
                                                   child_subagent_id=link["child_subagent_id"],
                                                   parent_subagent_id=link["parent_subagent_id"],
                                                   child_role=link["child_role"])
                try:
                    stored_link = self._stored_delegate_child_link(session_id)
                except Exception:
                    pass  # Retain the bind parent if cleanup cannot confirm the link.
            if stored_link:
                with self._auxiliary_session_lock:
                    self._delegate_child_bind_parent_ids.pop(session_id, None)
        except Exception as exc:  # the rows still go in; they stay reachable by session id
            self._note_delegate_child_fallback(session_id, f"parent link not recorded: {type(exc).__name__}: {exc}")
        key = self._delegate_child_key(session_id)
        boundary = _DELEGATE_RESUME_BOUNDARIES.get(key)
        identities = [self._session_end_prefix_compare_identity(msg, session_id=session_id) for msg in view]
        start = self._delegate_child_unique_end(
            [self._delegate_child_boundary_digest(identity) for identity in identities], boundary) if boundary else None
        anchored, rejected = start is not None, []
        if start is not None:
            logger.debug("LCM delegated child %s: resuming after compression boundary", session_id)
        else:
            start = self._session_end_store_prefix_count(session_id, view)
            if start == 0 and len(self._delegate_child_store_sessions(session_id)) > 1:
                start = None  # an empty continuation is not an empty logical child history
        if start is None:
            sessions = self._delegate_child_store_sessions(session_id)
            stored_tail = []
            for sid in sessions:
                stored_tail = [(sid, row) for row in self._store.get_session_tail(
                    sid, limit=_ANCHOR_ROWS - len(stored_tail))] + stored_tail
                if len(stored_tail) == _ANCHOR_ROWS:
                    break
            outputs = Counter(_DELEGATE_COMPACTION_OUTPUTS.get(key) or ())
            lineage: list[Counter] = []  # the lineage's stored identities, read once per append (under the store lock)

            def unaccounted(front: list, end: int) -> list[int]:
                # #2071: each row in front of the match must be stored in the lineage (apart from the anchor rows,
                # the lineage's last stored rows) or be one of the last compaction's outputs, one occurrence each.
                if not front:
                    return []
                if not lineage:
                    lineage.append(Counter(self._session_end_prefix_compare_identity(row, session_id=sid)
                                           for sid in sessions for row in self._store.get_session_messages(sid, limit=-1)))
                stored_ids = lineage[0].copy()
                stored_ids.subtract(identities[len(front):end])
                return self._delegate_child_unaccounted(front, stored_ids, outputs)

            start = self._delegate_child_anchor_end(session_id, view, stored_tail, unaccounted, rejected)
            anchored = start is not None or bool(rejected)
            if rejected:
                end, missing = rejected[0]
                self._note_delegate_child_fallback(
                    session_id, f"{len(missing)} row(s) in front of the stored-tail anchor are neither stored nor "
                    "compaction outputs; storing them with the rows after the anchor")
                suffix = [view[index] for index in missing] + view[end:]
            elif start is None:
                self._note_delegate_child_fallback(
                    session_id, "stored rows no longer match the child's view; resuming with counted identities")
                remaining = Counter(self._session_end_prefix_compare_identity(row, session_id=sid)
                                    for sid in sessions for row in self._store.get_session_messages(sid, limit=-1))
                suffix = []
                for msg, identity in zip(view, identities):
                    if remaining[identity] > 0:
                        remaining[identity] -= 1
                    elif outputs[self._delegate_child_boundary_digest(identity)] > 0:  # a compacted head
                        outputs[self._delegate_child_boundary_digest(identity)] -= 1
                    else:
                        suffix.append(msg)
            else:
                logger.debug("LCM delegated child %s: resuming after stored tail anchor", session_id)
                suffix = view[start:]
        else:
            suffix = view[start:]
        if not suffix:
            return 0
        stored = self._append_off_current_session_end_suffix(
            session_id, suffix, source=_DELEGATE_SOURCE, conversation_id=session_id)
        if (boundary and anchored) or rejected:  # a rejection stores rows out of view order: always mark the end
            self._remember_delegate_child_boundary(session_id, view)
        _count("rows_stored", len(stored))
        logger.debug("LCM stored %d delegated-child rows: session=%s parent=%s",
                     len(stored), session_id, link["parent_session_id"])
        return len(stored)

    def _delegate_child_unaccounted(self, front: list, stored: Counter, outputs: Counter) -> list[int]:
        """Indexes of ``front`` rows that are neither stored nor a compaction output (one occurrence each)."""
        stored, outputs, missing = stored.copy(), outputs.copy(), []
        for index, identity in enumerate(front):
            digest = self._delegate_child_boundary_digest(identity)
            if stored[identity] > 0:
                stored[identity] -= 1
            elif outputs[digest] > 0:
                outputs[digest] -= 1
            else:
                missing.append(index)
        return missing

    def _delegate_child_anchor_end(self, session_id: str, view: List[Dict[str, Any]],
                                  stored_rows: list[tuple[str, Dict[str, Any]]],
                                  unaccounted=None, rejected: Optional[list] = None) -> Optional[int]:
        """After the host rewrote the child's head (native compaction), resume after the stored tail when
        it occurs exactly once in the view, and only when every row in front of it is accounted for (#2071):
        ``unaccounted(front, end)`` lists the front rows that are neither stored nor compaction outputs; when it lists
        any, ``(end, those indexes)`` goes to ``rejected`` and None is returned."""
        anchor = [self._session_end_prefix_compare_identity(row, session_id=sid)
                  for sid, row in stored_rows]
        identities = [self._session_end_prefix_compare_identity(msg, session_id=session_id) for msg in view]
        for width in range(len(anchor), 0, -1):
            end = self._delegate_child_unique_end(identities, anchor[-width:])
            if end is not None:
                missing = unaccounted(identities[:end - width], end) if unaccounted is not None else []
                if missing:  # a narrower match would sit behind the same rows
                    if rejected is not None:
                        rejected.append((end, missing))
                    return None
                return end
        return None

    def _note_delegate_child_fallback(self, session_id: str, reason: str) -> None:
        _count("fallbacks")
        logger.warning("LCM delegated child %s: %s", session_id or "(unknown)", reason)
