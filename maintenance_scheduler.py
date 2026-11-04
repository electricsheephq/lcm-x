"""Process-wide maintenance schedulers and the process-exit drain of pending session tails."""

import asyncio
import atexit
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from .embedding_maintenance import _STOP_EMBEDDING_MAINTENANCE

# Preserve the engine logger under the package name used by the host.
logger = logging.getLogger(f"{__package__}.engine" if __package__ else "engine")


class _RollupMaintenanceScheduler:
    """Run deduplicated rollup jobs on one process-wide worker.

    Session binding only enqueues work. The worker is shared by every engine so
    rapid gateway binds cannot create one thread per session. A key that is
    already queued is ignored; a key requested while active gets at most one
    follow-up pass, which preserves eventual progress when new staleness arrives
    during an in-flight build without allowing concurrent duplicate builds.
    """

    def __init__(
        self,
        max_pending_jobs: int = 64,
        *,
        kind: str = "temporal rollup",
        thread_name: str = "lcm-rollup-maintenance",
    ) -> None:
        self._kind = kind
        self._thread_name = thread_name
        self._condition = threading.Condition()
        self._max_pending_jobs = max(1, int(max_pending_jobs))
        self._jobs: deque[
            tuple[tuple[str, str], Callable[[], None]]
        ] = deque()
        self._queued_keys: set[tuple[str, str]] = set()
        # Keep one equally bounded overflow bank; admission never grows the ready queue.
        self._deferred_jobs: dict[tuple[str, str], Callable[[], None]] = {}
        self._not_before: dict[tuple[str, str], float] = {}
        self._active_keys: set[tuple[str, str]] = set()
        # One worker means at most one active key. Keep its requested rerun in
        # a literal single slot so queue saturation cannot discard the
        # documented active-pass follow-up guarantee or grow hidden state.
        self._follow_up_key: tuple[str, str] | None = None
        self._follow_up_job: Callable[[], None] | None = None
        self._running_follow_up = False
        self._exclusive_keys: set[tuple[str, str]] = set()
        self._owned_keys: dict[object, set[tuple[str, str]]] = {}
        self._key_owners: dict[tuple[str, str], set[object]] = {}
        self._handoff_jobs: dict[tuple[str, str], dict[object, Callable[[], None]]] = {}
        self._retired_active_keys: dict[object, set[tuple[str, str]]] = {}
        self._worker: threading.Thread | None = None
        self._closed = False
        self._closed_logged = False

    def _track_owner_locked(
        self,
        key: tuple[str, str],
        owner: object | None,
        job: Callable[[], None] | None = None,
    ) -> None:
        if owner is None:
            return
        self._owned_keys.setdefault(owner, set()).add(key)
        self._key_owners.setdefault(key, set()).add(owner)
        if job is not None:
            self._handoff_jobs.setdefault(key, {})[owner] = job

    def _release_key_owners_if_idle_locked(self, key: tuple[str, str]) -> None:
        if (
            key in self._queued_keys
            or key in self._deferred_jobs
            or key in self._active_keys
            or key in self._exclusive_keys
            or self._follow_up_key == key
        ):
            return
        self._handoff_jobs.pop(key, None)
        for owner in self._key_owners.pop(key, set()):
            owned = self._owned_keys.get(owner)
            if owned is None:
                continue
            owned.discard(key)
            if not owned:
                self._owned_keys.pop(owner, None)

    def schedule(
        self,
        key: tuple[str, str],
        job: Callable[[], None],
        *,
        owner: object | None = None,
        delay: float = 0.0,
        handoff: bool = False,
    ) -> bool:
        global _EMBEDDING_EXIT_QUIESCE_REGISTERED
        with self._condition:
            if self._closed:
                if not self._closed_logged:
                    logger.info("LCM %s maintenance scheduler is closed", self._kind)
                    self._closed_logged = True
                return False
            if key in self._exclusive_keys:
                logger.info(
                    "LCM temporal rollup maintenance deferred while an operator "
                    "rebuild owns database=%s scope=%s",
                    key[0],
                    key[1],
                )
                return False
            if key in self._queued_keys or key in self._deferred_jobs:
                if delay > 0:
                    self._not_before[key] = max(
                        self._not_before.get(key, 0), time.monotonic() + delay,
                    )
                self._track_owner_locked(key, owner, job if handoff else None)
                return True
            if delay > 0:
                self._not_before[key] = time.monotonic() + delay
                if self._follow_up_key == key:
                    self._follow_up_key = self._follow_up_job = None
            if key in self._active_keys and not self._running_follow_up and delay <= 0:
                self._follow_up_key = key
                self._follow_up_job = job
                self._track_owner_locked(key, owner, job if handoff else None)
                self._condition.notify_all()
                return True
            if len(self._jobs) >= self._max_pending_jobs:
                if self._kind == "incremental embedding" and len(self._deferred_jobs) < self._max_pending_jobs:
                    self._deferred_jobs[key] = job
                    self._track_owner_locked(key, owner, job if handoff else None)
                    return True
                logger.warning(
                    "LCM %s maintenance queue is full; " + (
                        "bounded deferred capacity exhausted; refusing database=%s scope=%s"
                        if self._kind == "incremental embedding" else "deferring database=%s scope=%s until a later bind"),
                    self._kind,
                    key[0],
                    key[1],
                )
                if self._kind == "incremental embedding":
                    from .embedding_maintenance import _record
                    _record(key[0], "queue_capacity_exhausted", "ready and deferred capacities exhausted")
                self._not_before.pop(key, None)
                return False
            if self._worker is None or not self._worker.is_alive():
                if self is _EMBEDDING_MAINTENANCE_SCHEDULER and not _EMBEDDING_EXIT_QUIESCE_REGISTERED:
                    atexit.register(_quiesce_embedding_maintenance)
                    _EMBEDDING_EXIT_QUIESCE_REGISTERED = True
                worker = threading.Thread(
                    target=self._run,
                    name=self._thread_name,
                    daemon=True,
                )
                worker.start()
                self._worker = worker
            self._jobs.append((key, job))
            self._queued_keys.add(key)
            self._track_owner_locked(key, owner, job if handoff else None)
            self._condition.notify_all()
            return True

    def handoff(self, key: tuple[str, str], retired_owner: object, *, delay: float = 0.0) -> bool:
        """Continue LOCAL work through a surviving owner, preserving its idle delay."""
        with self._condition:
            for owner, job in self._handoff_jobs.get(key, {}).items():
                if owner is not retired_owner:
                    return self.schedule(key, job, owner=owner, delay=delay, handoff=True)
        return False

    def try_acquire_exclusive(self, key: tuple[str, str]) -> bool:
        """Reserve one idle key for a synchronous operator rebuild.

        This is intentionally non-blocking: a manual rebuild must not race a
        provider-backed maintenance pass, but it also must not wait behind one
        on the gateway thread. The caller can ask the operator to retry once the
        background pass completes.
        """
        with self._condition:
            busy_keys = self._queued_keys | self._active_keys | self._exclusive_keys | set(self._deferred_jobs)
            if key in busy_keys or self._follow_up_key == key:
                return False
            if len(self._exclusive_keys) >= self._max_pending_jobs:
                return False
            self._exclusive_keys.add(key)
            return True

    def release_exclusive(self, key: tuple[str, str]) -> None:
        with self._condition:
            self._exclusive_keys.discard(key)
            self._condition.notify_all()

    def drain(
        self,
        keys: set[tuple[str, str]],
        timeout: float | None = None,
    ) -> bool:
        """Wait until none of ``keys`` is queued or active."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while keys & (
                self._queued_keys
                | set(self._deferred_jobs)
                | self._active_keys
                | self._exclusive_keys
                | ({self._follow_up_key} if self._follow_up_key else set())
            ):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def drain_owner(
        self,
        owner: object,
        timeout: float | None = None,
    ) -> bool:
        """Wait until every outstanding key accepted for ``owner`` is idle."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._owned_keys.get(owner):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def retire_owner(self, owner: object, timeout: float) -> bool:
        """Drop an owner's unshared pending work and wait for its active keys."""
        try:
            deadline = time.monotonic() + max(0.0, timeout)
            with self._condition:
                keys = self._owned_keys.pop(owner, set())
                for key in keys:
                    owners = self._key_owners.get(key, set())
                    owners.discard(owner)
                    handoffs = self._handoff_jobs.get(key, {})
                    handoffs.pop(owner, None)
                    if owners:
                        if handoffs:
                            # LOCAL dedup retains each live owner's profile-bound job.
                            job = next(iter(handoffs.values()))
                            self._jobs = deque((k, job if k == key else j) for k, j in self._jobs)
                            if key in self._deferred_jobs:
                                self._deferred_jobs[key] = job
                            if self._follow_up_key == key:
                                self._follow_up_job = job
                        continue
                    self._handoff_jobs.pop(key, None)
                    self._key_owners.pop(key, None)
                    self._deferred_jobs.pop(key, None)
                    self._not_before.pop(key, None)
                    if key in self._queued_keys:
                        self._jobs = deque(
                            (queued_key, job) for queued_key, job in self._jobs
                            if queued_key != key
                        )
                        self._queued_keys.discard(key)
                    if self._follow_up_key == key:
                        self._follow_up_key = None
                        self._follow_up_job = None
                keys |= self._retired_active_keys.pop(owner, set()) & self._active_keys
                self._condition.notify_all()
                if timeout <= 0:
                    active = keys & self._active_keys
                    if active:
                        self._retired_active_keys[owner] = active
                    return not bool(active)
                while keys & self._active_keys:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        for key in sorted(keys & self._active_keys):
                            logger.warning(
                                "LCM %s maintenance retirement timeout for database=%s scope=%s",
                                self._kind, key[0], key[1],
                            )
                        return False
                    self._condition.wait(remaining)
                return True
        except Exception:
            logger.warning("LCM could not retire %s maintenance owner", self._kind, exc_info=True)
            return False

    def quiesce(self, timeout: float) -> bool:
        """Stop accepting work, discard pending passes, and bound the active wait."""
        try:
            if self is _EMBEDDING_MAINTENANCE_SCHEDULER:
                _STOP_EMBEDDING_MAINTENANCE.set()
            deadline = time.monotonic() + max(0.0, timeout)
            with self._condition:
                self._closed = True
                pending_keys = self._queued_keys.copy()
                pending_keys.update(self._deferred_jobs)
                self._deferred_jobs.clear()
                self._not_before.clear()
                if self._follow_up_key is not None:
                    pending_keys.add(self._follow_up_key)
                self._jobs.clear()
                self._queued_keys.clear()
                self._follow_up_key = None
                self._follow_up_job = None
                for key in pending_keys:
                    self._release_key_owners_if_idle_locked(key)
                self._condition.notify_all()
                while self._active_keys:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        for key in sorted(self._active_keys):
                            logger.warning(
                                "LCM %s maintenance quiesce timeout for database=%s scope=%s",
                                self._kind, key[0], key[1],
                            )
                        return False
                    self._condition.wait(remaining)
            if self is _EMBEDDING_MAINTENANCE_SCHEDULER:
                from .embedding_provider import _drain_local_embedding_workers
                return _drain_local_embedding_workers(deadline)
            return True
        except Exception:
            logger.warning("LCM could not quiesce %s maintenance", self._kind, exc_info=True)
            return False

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._jobs:
                    # Retirement may empty the ready queue while retained work survives.
                    if self._deferred_jobs:
                        deferred_key = next(iter(self._deferred_jobs))
                        self._jobs.append((deferred_key, self._deferred_jobs.pop(deferred_key)))
                        self._queued_keys.add(deferred_key)
                        break
                    if self._closed:
                        return
                    self._condition.wait()
                while self._jobs and all(self._not_before.get(k, 0) > time.monotonic() for k, _ in self._jobs):
                    if self is _EMBEDDING_MAINTENANCE_SCHEDULER and _STOP_EMBEDDING_MAINTENANCE.is_set():
                        self.quiesce(0)
                        return
                    self._condition.wait(min(0.5, min(self._not_before[k] for k, _ in self._jobs) - time.monotonic()))
                if not self._jobs:
                    continue
                while self._not_before.get(self._jobs[0][0], 0) > time.monotonic():
                    self._jobs.rotate(-1)
                key, job = self._jobs.popleft()
                self._not_before.pop(key, None)
                if self._deferred_jobs:
                    deferred_key = next(iter(self._deferred_jobs))
                    self._jobs.append((deferred_key, self._deferred_jobs.pop(deferred_key)))
                    self._queued_keys.add(deferred_key)
                self._queued_keys.discard(key)
                self._active_keys.add(key)
                self._running_follow_up = False
            while True:
                try:
                    job()
                except (Exception, asyncio.CancelledError):
                    logger.warning(
                        "LCM background %s maintenance failed for database=%s scope=%s",
                        self._kind,
                        key[0],
                        key[1],
                        exc_info=True,
                    )
                with self._condition:
                    if self._follow_up_key == key and self._follow_up_job is not None:
                        job = self._follow_up_job
                        self._follow_up_key = None
                        self._follow_up_job = None
                        self._running_follow_up = True
                        self._condition.notify_all()
                        continue
                    self._active_keys.discard(key)
                    for owner, keys in list(self._retired_active_keys.items()):
                        keys.discard(key)
                        if not keys:
                            self._retired_active_keys.pop(owner, None)
                    self._running_follow_up = False
                    self._release_key_owners_if_idle_locked(key)
                    self._condition.notify_all()
                    break


_ROLLUP_MAINTENANCE_SCHEDULER = _RollupMaintenanceScheduler()
# #1014: incremental embedding passes get their own worker, so a slow local
# model load never delays a rollup pass and draining one never waits on the other.
_EMBEDDING_MAINTENANCE_SCHEDULER = _RollupMaintenanceScheduler(
    kind="incremental embedding", thread_name="lcm-embedding-maintenance"
)
# One in-flight provider batch plus publication, with headroom for a 2-vCPU host.
# Session retirement keeps its separate budget and never signals process exit.
_EMBEDDING_EXIT_TIMEOUT_S = 10.0
_EMBEDDING_EXIT_QUIESCE_REGISTERED = False


def _quiesce_embedding_maintenance() -> bool:
    _flush_pending_tails_at_exit()
    return _EMBEDDING_MAINTENANCE_SCHEDULER.quiesce(_EMBEDDING_EXIT_TIMEOUT_S)


_SESSION_END_SHUTDOWN_TIMEOUT_MS = 5000
# Strong references keep an ended engine's only pending copy alive until drained.
_PENDING_TAIL_ENGINES: set[Any] = set()


def _flush_pending_tails_at_exit() -> None:
    deadline = time.monotonic() + _SESSION_END_SHUTDOWN_TIMEOUT_MS / 1000
    for engine in tuple(_PENDING_TAIL_ENGINES):
        with engine._exclusive_lifecycle("exit flush", deadline=deadline) as acquired:
            if acquired:
                engine._retry_pending_session_tails(deadline=deadline, exiting=True)
            else:
                engine._warn_unflushed_session_tails("lifecycle lock")


atexit.register(_flush_pending_tails_at_exit)
