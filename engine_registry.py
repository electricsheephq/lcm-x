"""Process-wide registry of active LCM runtime clones by session/lane.

Isolated from ``engine.py`` (WS5 seam): LCM clones register their own
session/conversation binding so post-turn ingest can follow the active clone
instead of the process-wide plugin singleton. The lock and the two weak
registries live here alongside the pure resolver/matcher helpers that read
them. ``engine.py`` imports the shared lock, the two registries, the removal
helper, and the public ``resolve_active_lcm_engine`` entry point; the binding
methods on ``LCMEngine`` mutate the same shared objects by reference.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

_ACTIVE_ENGINE_REGISTRY_LOCK = threading.RLock()
_ACTIVE_ENGINE_COLD_START_LOCK = threading.RLock()
_ACTIVE_ENGINES_BY_SESSION_ID = weakref.WeakValueDictionary()
_ACTIVE_ENGINES_BY_CONVERSATION_ID = weakref.WeakValueDictionary()


class ActiveEngineUseStatus(str, Enum):
    """Terminal result of one stable active-engine use attempt."""

    USED = "used"
    ENGINE_NOT_RESIDENT = "engine_not_resident"
    RESIDENT_ENGINE_CONFLICT = "resident_engine_conflict"
    BINDING_CHANGED = "binding_changed"
    CLOSED_ENGINE = "closed_engine"
    BUSY = "busy"
    UNSUPPORTED_ENGINE = "unsupported_engine"
    REENTRANT_LIFECYCLE = "reentrant_lifecycle"


@dataclass(frozen=True)
class ActiveEngineUseResult:
    status: ActiveEngineUseStatus
    value: Any = None

    @property
    def used(self) -> bool:
        return self.status is ActiveEngineUseStatus.USED


def _is_usable_lcm_engine(engine: Any) -> bool:
    return bool(
        engine is not None
        and getattr(engine, "name", None) == "lcm"
        and hasattr(engine, "ingest")
    )


def _engine_matches_session_binding(engine: Any, session_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and session_id
        and str(getattr(engine, "_session_id", "") or "") == session_id
    )


def _engine_matches_conversation_binding(engine: Any, conversation_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and conversation_id
        and str(getattr(engine, "_conversation_id", "") or "") == conversation_id
    )


def _engine_matches_foreground_binding(
    engine: Any,
    session_id: str,
    conversation_id: str,
) -> bool:
    """Return whether an engine's operator-facing foreground matches the lane.

    A stateless/ignored side channel can temporarily own the engine's bound
    session while ``current_session_id`` and ``current_conversation_id`` keep
    pointing at the foreground conversation. The direct registries intentionally
    follow the bound session for ingest dispatch, so operator commands need this
    bounded fallback scan to recover the same clone without rebinding it.
    """
    if not _is_usable_lcm_engine(engine) or not (session_id or conversation_id):
        return False
    try:
        foreground_session_id = str(
            getattr(engine, "current_session_id", "") or ""
        )
        foreground_conversation_id = str(
            getattr(engine, "current_conversation_id", "") or ""
        )
    except Exception:
        return False
    return bool(
        (not session_id or foreground_session_id == session_id)
        and (
            not conversation_id
            or foreground_conversation_id == conversation_id
        )
    )


def _remove_registry_entries_for_engine(
    engine: Any,
    *,
    keep_session_id: str = "",
    keep_conversation_id: str = "",
) -> None:
    for registered_session_id, registered_engine in list(_ACTIVE_ENGINES_BY_SESSION_ID.items()):
        if registered_engine is engine and registered_session_id != keep_session_id:
            _ACTIVE_ENGINES_BY_SESSION_ID.pop(registered_session_id, None)
    for registered_conversation_id, registered_engine in list(
        _ACTIVE_ENGINES_BY_CONVERSATION_ID.items()
    ):
        if registered_engine is engine and registered_conversation_id != keep_conversation_id:
            _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(registered_conversation_id, None)


def resolve_active_lcm_engine(session_id: str = "", conversation_id: str = "") -> Any:
    """Return a point-in-time LCM runtime clone bound to a session/lane.

    Newer Hermes Agent hosts pass the active per-agent context engine directly
    to ``post_llm_call`` hooks. Older hosts may only pass session/lane ids. LCM
    clones register their own session binding when ``on_session_start`` runs so
    post-turn ingest can still follow the active clone instead of rebinding the
    process-wide plugin singleton. Mutation callers must use
    ``use_active_lcm_engine`` so the binding is revalidated while the selected
    engine cannot be rebound or closed.
    """
    session_id = str(session_id or "")
    conversation_id = str(conversation_id or "")
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        if session_id:
            engine = _ACTIVE_ENGINES_BY_SESSION_ID.get(session_id)
            if _engine_matches_session_binding(engine, session_id):
                return engine
            if engine is not None:
                _ACTIVE_ENGINES_BY_SESSION_ID.pop(session_id, None)
        if conversation_id:
            engine = _ACTIVE_ENGINES_BY_CONVERSATION_ID.get(conversation_id)
            conversation_matches = _engine_matches_conversation_binding(
                engine,
                conversation_id,
            )
            session_matches = not session_id or _engine_matches_session_binding(
                engine,
                session_id,
            )
            if conversation_matches and session_matches:
                return engine
            if engine is not None and not conversation_matches:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(conversation_id, None)
        # Direct lookups above follow the engine's actively-bound ingest lane.
        # If that lane is a side channel, find the same engine by its stable
        # operator-facing foreground view. Registry size is bounded by live
        # AIAgent clones and values are weak, so this scan does not retain stale
        # runtimes or grow with historical sessions.
        seen: set[int] = set()
        for engine in (
            list(_ACTIVE_ENGINES_BY_SESSION_ID.values())
            + list(_ACTIVE_ENGINES_BY_CONVERSATION_ID.values())
        ):
            engine_id = id(engine)
            if engine_id in seen:
                continue
            seen.add(engine_id)
            if _engine_matches_foreground_binding(
                engine,
                session_id,
                conversation_id,
            ):
                return engine
    return None


def has_resident_lcm_engine() -> bool:
    """Return whether any live LCM runtime is currently registered."""
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        return bool(
            list(_ACTIVE_ENGINES_BY_SESSION_ID.values())
            or list(_ACTIVE_ENGINES_BY_CONVERSATION_ID.values())
        )


def _binding_still_selects(
    engine: Any,
    *,
    session_id: str,
    conversation_id: str,
) -> bool:
    """Revalidate with the resolver's established session-first semantics."""
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        if session_id:
            return bool(
                _ACTIVE_ENGINES_BY_SESSION_ID.get(session_id) is engine
                and _engine_matches_session_binding(engine, session_id)
            )
        if conversation_id:
            return bool(
                _ACTIVE_ENGINES_BY_CONVERSATION_ID.get(conversation_id) is engine
                and _engine_matches_conversation_binding(engine, conversation_id)
            )
    return False


def use_active_lcm_engine(
    operation,
    *,
    session_id: str = "",
    conversation_id: str = "",
    timeout: float | None = 5.0,
) -> ActiveEngineUseResult:
    """Resolve and use one engine while its validated binding stays stable."""
    session_id = str(session_id or "")
    conversation_id = str(conversation_id or "")
    engine = resolve_active_lcm_engine(
        session_id=session_id,
        conversation_id=conversation_id,
    )
    if engine is None:
        return ActiveEngineUseResult(ActiveEngineUseStatus.ENGINE_NOT_RESIDENT)
    run_stably = getattr(engine, "_run_stably", None)
    if not callable(run_stably):
        return ActiveEngineUseResult(ActiveEngineUseStatus.UNSUPPORTED_ENGINE)
    return run_stably(
        operation,
        validate=lambda selected: _binding_still_selects(
            selected,
            session_id=session_id,
            conversation_id=conversation_id,
        ),
        timeout=timeout,
    )


def use_lcm_engine(
    engine: Any,
    operation,
    *,
    timeout: float | None = 5.0,
) -> ActiveEngineUseResult:
    """Use a host-supplied or cold prototype engine with stable lifetime."""
    if not _is_usable_lcm_engine(engine):
        return ActiveEngineUseResult(ActiveEngineUseStatus.UNSUPPORTED_ENGINE)
    run_stably = getattr(engine, "_run_stably", None)
    if not callable(run_stably):
        return ActiveEngineUseResult(ActiveEngineUseStatus.UNSUPPORTED_ENGINE)
    return run_stably(operation, timeout=timeout)


def use_cold_lcm_engine(
    engine: Any,
    operation,
    *,
    timeout: float | None = 5.0,
) -> ActiveEngineUseResult:
    """Use an unbound prototype only while no other runtime can register."""
    def claim_and_use(selected):
        with _ACTIVE_ENGINE_COLD_START_LOCK:
            if has_resident_lcm_engine():
                return ActiveEngineUseResult(
                    ActiveEngineUseStatus.RESIDENT_ENGINE_CONFLICT
                )
            if str(getattr(selected, "_session_id", "") or ""):
                return ActiveEngineUseResult(ActiveEngineUseStatus.BINDING_CHANGED)
            return ActiveEngineUseResult(
                ActiveEngineUseStatus.USED,
                operation(selected),
            )

    outer = use_lcm_engine(engine, claim_and_use, timeout=timeout)
    return outer.value if outer.used else outer
