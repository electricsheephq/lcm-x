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

import logging
import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .plugin_identity import is_lcm_engine_name

_ACTIVE_ENGINE_REGISTRY_LOCK = threading.RLock()
_ACTIVE_ENGINE_COLD_START_LOCK = threading.RLock()
_ACTIVE_ENGINES_BY_SESSION_ID = weakref.WeakValueDictionary()
_ACTIVE_ENGINES_BY_CONVERSATION_ID = weakref.WeakValueDictionary()
logger = logging.getLogger(__name__)


class EngineUnloadGroup:
    """One plugin registration owns its prototype, clones and SQLite work."""

    def __init__(self, prototype: Any) -> None:
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.engines = weakref.WeakSet((prototype,))
        self.databases: set[str] = set()
        self._workers: set[threading.Thread] = set()
        self._rollup_owners: set[object] = set()
        self._retired_embedding_owners: set[object] = set()
        self._deferred = weakref.WeakSet()

    def start_worker(self, worker: threading.Thread) -> bool:
        with self.lock:
            if self.closing.is_set():
                return False
            self._workers = {active for active in self._workers if active.is_alive()}
            worker.start()
            self._workers.add(worker)
            return True

    def schedule_rollup(self, scheduler, key, work, *, owner) -> None:
        with self.lock:
            if self.closing.is_set():
                return
            self._rollup_owners = {
                pending for pending in self._rollup_owners
                if not scheduler.drain_owner(pending, timeout=0)
            }
            if scheduler.schedule(key, work, owner=owner):
                self._rollup_owners.add(owner)

    def remember_embedding_retirement(self, scheduler, owner, *, drained: bool) -> None:
        with self.lock:
            prior_owners = tuple(self._retired_embedding_owners)
        finished = {prior for prior in prior_owners if scheduler.retire_owner(prior, timeout=0)}
        with self.lock:
            self._retired_embedding_owners.difference_update(finished)
            if not drained:
                self._retired_embedding_owners.add(owner)

    def _defer_shutdown(self, engine, *, wait_for_children: bool = False) -> None:
        from .aux_session import drain_live_child_stores
        from .config import profile_bound

        def finish():
            try:
                if wait_for_children:
                    drain_live_child_stores(None, owner=self)
                with engine._compaction_gate:
                    engine._compaction_gate.wait_for(lambda: engine._compaction_inflight == 0)
                # A terminal tail cannot consume every retry budget while a
                # later tail waits. Keep it for the final warning and close.
                terminal_tails: set[int] = set()
                while not engine._stable_use_closed:
                    engine.shutdown(terminal_tails=terminal_tails)
                    if not engine._stable_use_closed:
                        time.sleep(0.1)
            except Exception:
                logger.exception("LCM deferred engine cleanup failed after plugin unload")

        with self.lock:
            if engine in self._deferred:
                return
            self._deferred.add(engine)
        threading.Thread(target=profile_bound(finish), name="lcm-deferred-unload", daemon=True).start()

    def shutdown(self) -> None:
        from . import leaf_prep
        from .aux_session import drain_live_child_stores
        from .db_bootstrap import drain_background_database_workers
        from .engine import _shutdown_compaction_wait_s
        from .maintenance_scheduler import _EMBEDDING_MAINTENANCE_SCHEDULER, _ROLLUP_MAINTENANCE_SCHEDULER
        from .retrieval_core import _retire_owned_vector_stores

        with self.lock:
            self.closing.set()
            engines = list(self.engines)
            workers = list(self._workers)
            rollup_owners = list(self._rollup_owners)
        deadline = time.monotonic() + _shutdown_compaction_wait_s()
        children_drained = drain_live_child_stores(max(0.0, deadline - time.monotonic()), owner=self)

        # Abandoned deadline workers may still hold private database handles.
        drained = True
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(max(0.0, deadline - time.monotonic()))
            drained = not worker.is_alive() and drained

        for engine in engines:
            try:
                if not children_drained:
                    self._defer_shutdown(engine, wait_for_children=True)
                    continue
                engine.shutdown(deadline=deadline)
                if not engine._stable_use_closed:
                    self._defer_shutdown(engine)
            except Exception:
                logger.exception("LCM engine cleanup failed during plugin unload")

        for owner in rollup_owners:
            drained = _ROLLUP_MAINTENANCE_SCHEDULER.drain_owner(
                owner, timeout=max(0.0, deadline - time.monotonic()),
            ) and drained
        with self.lock:
            retired_embedding_owners = tuple(self._retired_embedding_owners)
        for owner in retired_embedding_owners:
            drained = _EMBEDDING_MAINTENANCE_SCHEDULER.retire_owner(
                owner, timeout=max(0.0, deadline - time.monotonic()),
            ) and drained
        drained = leaf_prep.retire_owner(self, timeout=max(0.0, deadline - time.monotonic())) and drained
        with self.lock:
            databases = self.databases.copy()
        drained = drain_background_database_workers(databases, deadline=deadline) and drained
        drained = _retire_owned_vector_stores(self, deadline=deadline) and drained
        if not drained:
            logger.warning("LCM plugin unload timed out waiting for SQLite work to finish")


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
        and is_lcm_engine_name(getattr(engine, "name", None))
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


def live_bound_session_ids() -> set[str]:
    """Session ids the active-engine registry lists for this process (#753).

    Empty-lifecycle GC must not delete the row of a session another clone in
    this process has bound but not ingested into yet. The registry holds only
    live engines (weak values; shutdown and profile switches unregister), so
    rows left by dead processes or shut-down engines stay collectable. It keeps
    one engine per session id (the last to bind): if two engines bind the same
    id and the later one moves on, the earlier one's id is not listed.
    """
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        return {
            str(session_id)
            for session_id, engine in list(_ACTIVE_ENGINES_BY_SESSION_ID.items())
            if session_id and engine is not None
        }


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
    allow_unloading: bool = False,
) -> ActiveEngineUseResult:
    """Use a host-supplied or cold prototype engine with stable lifetime."""
    if not _is_usable_lcm_engine(engine):
        return ActiveEngineUseResult(ActiveEngineUseStatus.UNSUPPORTED_ENGINE)
    run_stably = getattr(engine, "_run_stably", None)
    if not callable(run_stably):
        return ActiveEngineUseResult(ActiveEngineUseStatus.UNSUPPORTED_ENGINE)
    if allow_unloading:
        return run_stably(operation, timeout=timeout, allow_unloading=True)
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
