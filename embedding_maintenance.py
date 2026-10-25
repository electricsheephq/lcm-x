"""Incremental embedding maintenance (#1014).

`/lcm embed backfill` was the only writer of vectors, so a store started with
none and every summary or chunk published after a manual backfill stayed
unembedded. The engine now schedules one bounded pass in the background after a
leaf summary is published and on session bind. The pass runs the backfill core
itself (lease, in-flight markers, privacy policy, dtype, budget, provider
timeout), one batch per corpus, so it can never publish a row twice alongside a
manual backfill or another process.

Message chunks are embedded only for FastEmbed or an Ollama endpoint on a
loopback host: the raw-text authorization (`--confirm-raw-text`) is per
invocation and never persisted, so chunk embedding for a cloud provider or a
remote Ollama stays operator-initiated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import effective_embedding_model, embedding_provider_id, profile_bound
from .embedding_provider import FastembedProvider, _is_loopback_url

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_PASS: dict[str, tuple[float, str]] = {}
_FAILURES: dict[str, int] = {}
_FAILURE_WARNED: dict[str, bool] = {}
_STOP_EMBEDDING_MAINTENANCE = threading.Event()
_PREPARE: dict[tuple[str, str, int], tuple[str, str]] = {}
_PREPARE_THREAD: threading.Thread | None = None
# model key -> (database, owner token) -> enqueue; one entry per live owner, so a retiring engine
# never displaces another engine's callback for the same database.
_PREPARE_READY: dict[tuple[str, str, int], dict[tuple[str, object], Callable[[], None]]] = {}
_PREPARE_RETRY: dict[tuple[str, str, int], tuple[int, float]] = {}
# #1586: 0.82 passes the coverage/CPU bars in the 2-vCPU floor-surrogate probe.
_LOCAL_IDLE_MULTIPLIER = 0.82
_SUCCESS_OUTCOMES = frozenset({"complete"})


def database_identity(db_path: str | Path) -> str:
    return str(Path(db_path).resolve())


def incremental_embedding_state(db_path: str | Path) -> tuple[tuple[float, str] | None, int]:
    """Return this database's last pass ``(time, outcome)`` and its failure count."""
    with _STATE_LOCK:
        identity = database_identity(db_path)
        return _LAST_PASS.get(identity), _FAILURES.get(identity, 0)


def _record(db_path: str | Path, outcome: str, failure: str | None) -> None:
    with _STATE_LOCK:
        identity = database_identity(db_path)
        _LAST_PASS[identity] = (time.time(), outcome)
        if failure is None:
            parts = [part.split("=", 1)[-1].split(":") for part in outcome.split()]
            wrote = any(len(part) == 2 and part[1].isdigit() and int(part[1]) > 0 for part in parts)
            completed = parts and all(part[0] in _SUCCESS_OUTCOMES for part in parts)
            if wrote or completed:
                _FAILURES.pop(identity, None)
                _FAILURE_WARNED.pop(identity, None)
            return
        _FAILURES[identity] = _FAILURES.get(identity, 0) + 1
        first = not _FAILURE_WARNED.get(identity, False)
        _FAILURE_WARNED[identity] = True
    (logger.warning if first else logger.debug)(
        "LCM incremental embedding pass skipped (%s): %s; it retries after the "
        "next summary or session bind",
        outcome,
        failure,
    )


def _local_model_key(config: Any) -> tuple[str, str, int] | None:
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    if provider not in {"fastembed", "fast-embed"}:
        return None
    model = effective_embedding_model(getattr(config, "embedding_provider", ""), getattr(config, "embedding_model", ""))
    return FastembedProvider(model)._cache_key() if model else None


def local_model_state(config: Any) -> tuple[str, str] | None:
    """This process's ``(state, reason)`` for the configured local model, or None."""
    key = _local_model_key(config)
    with _STATE_LOCK:
        return _PREPARE.get(key) if key else None


def note_local_model_present(config: Any) -> None:
    """A successful explicit warmup makes the local model present for this process."""
    key = _local_model_key(config)
    if key:
        with _STATE_LOCK:
            _PREPARE[key] = ("present", "")


def start_local_model_prepare(
    db_path: str | Path, config: Any, *, on_ready: Callable[[], None] | None = None,
    owner: object | None = None,
) -> bool:
    """Start this process's one background prepare of the local model (#1186).

    The thread runs the warmup path, the only one allowed to download, and
    leaves profile registration to owner-tracked jobs. Recall never waits on it:
    until the model is ready it serves full text and reports degraded. Failures
    retry at ten-minute intervals for six attempts, then background triggers retry
    after five minutes, doubling to a thirty-minute cap. A model requested
    while another is preparing is admitted by a later trigger.
    ``on_ready`` only enqueues maintenance; it must never run a pass inline.
    """
    global _PREPARE_THREAD
    key = _local_model_key(config)
    if key is None or not getattr(config, "embeddings_enabled", False):
        return False
    start_error = None
    with _STATE_LOCK:
        state = _PREPARE.get(key)
        attempts, retry_at = _PREPARE_RETRY.get(key, (0, 0.0))
        exhausted = attempts >= 6 and (state is None or state[0] != "preparing")
        if on_ready is not None and (state is None or state[0] != "present"):
            if state is not None or _PREPARE_THREAD is None or not _PREPARE_THREAD.is_alive():
                token = on_ready if owner is None else owner
                _PREPARE_READY.setdefault(key, {})[(database_identity(db_path), token)] = on_ready
        if ((state is not None and state[0] != "unavailable") or (exhausted and on_ready is None)
                or time.monotonic() < retry_at):
            return False
        if _PREPARE_THREAD is not None and _PREPARE_THREAD.is_alive():
            return False
        _PREPARE_RETRY[key] = (attempts + 1, time.monotonic() + 600.0)
        _PREPARE[key] = ("preparing", "")
        # Started under the lock, so a concurrent caller sees it alive.
        _PREPARE_THREAD = threading.Thread(
            target=profile_bound(_prepare_local_model), args=(key, db_path, config),
            name="lcm-embedding-prepare", daemon=True,
        )
        try:
            _PREPARE_THREAD.start()
        except Exception as exc:
            _PREPARE_THREAD = None
            if state is None:
                _PREPARE.pop(key, None)
            else:
                _PREPARE[key] = state
            start_error = exc
    if start_error is not None:
        logger.warning("LCM local model prepare could not start; retrieval uses full text", exc_info=start_error)
        return False
    return True


def _prepare_local_model(key: tuple[str, str, int], db_path: str | Path, config: Any) -> None:
    try:
        from . import command  # lazy: command imports most of the plugin

        # Only warm the shared model here. Store registration belongs to the
        # owner-tracked maintenance job; retirement fences its enqueue callback.
        provider = command.resolve_provider(config)
        provider.warmup()
        result = {}
    except Exception as exc:
        result = f"error: {exc}"
    if isinstance(result, dict):
        with _STATE_LOCK:
            _PREPARE[key] = ("present", "")
            _PREPARE_RETRY.pop(key, None)
            callbacks = _PREPARE_READY.pop(key, {})
        for enqueue in list(callbacks.values()):
            try:
                enqueue()
            except Exception:
                logger.warning("LCM could not enqueue embedding catch-up after preparation", exc_info=True)
        return
    reason = str(result).rsplit("error: ", 1)[-1].strip()
    with _STATE_LOCK:
        _PREPARE[key] = ("unavailable", reason)
        attempts = _PREPARE_RETRY.get(key, (0, 0.0))[0]
        if attempts >= 6:
            _PREPARE_RETRY[key] = (attempts, time.monotonic() + min(1800.0, 300.0 * 2 ** min(attempts - 6, 3)))
        identity = database_identity(db_path)
        first = not _FAILURE_WARNED.get(identity, False)
        _FAILURE_WARNED[identity] = True
    (logger.warning if first else logger.debug)(
        "LCM local embedding model %s is unavailable (%s); recall uses full text. "
        "Check `/lcm doctor` (embedding_provider_health); background retries use bounded backoff",
        key,
        reason,
    )


def retire_local_model_prepare(owner) -> None:
    with _STATE_LOCK:
        for callbacks in _PREPARE_READY.values():
            for callback_key in list(callbacks):
                if callback_key[1] is owner:
                    callbacks.pop(callback_key, None)


def run_incremental_embedding_pass(
    db_path: str | Path, config: Any, *, breaker: Any = None, slice_state: dict | None = None,
) -> str | dict[str, Any]:
    """Run one bounded pass over both corpora and record its outcome. Never raises."""
    started = time.monotonic()
    if slice_state is not None:
        slice_state.update(vectors_written=0, scan_frontier_advance=False, remaining=False, **{"continue": False})
        slice_state["deadline"] = started + 1.0
    try:
        outcome, failure = _run_pass(db_path, config, breaker, slice_state)
    except Exception as exc:
        outcome, failure = "error", str(exc)
    if slice_state is not None:
        active = time.monotonic() - started
        deferred = outcome in {"lease_held", "circuit_open", "model_preparing", "model_unavailable", "model_prepare_exhausted"}
        retry_at = _PREPARE_RETRY.get(_local_model_key(config), (0, 0.0))[1]
        slice_state.update(active_s=active, overrun_s=max(0.0, active - 1.0),
                           delay=max(_LOCAL_IDLE_MULTIPLIER * active, (max(1.0, retry_at - time.monotonic()) if outcome in {"model_unavailable", "model_prepare_exhausted"} else 1.0) if deferred else 0.0))
        if slice_state.pop("retry_wait", False):
            slice_state["delay"] = max(600.0, slice_state["delay"])
        slice_state["continue"] = deferred or (not failure and slice_state.get("continue", False))
        slice_state["stop_reason"] = outcome if not slice_state["continue"] else ""
        slice_state["rows_written"] = slice_state["vectors_written"]
    if outcome == "stopped":
        logger.info("LCM embedding maintenance stopped for database=%s scope=embeddings", db_path)
    _record(db_path, outcome, failure)
    return dict(slice_state, outcome=outcome) if slice_state is not None else outcome


def _automatic_chunks_allowed(config: Any, command: Any) -> bool:
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    provider = embedding_provider_id(provider)
    if not command._is_local_embedding_provider(provider):
        return False
    if provider != "ollama":
        return True  # FastEmbed runs in this process
    # Read the base URL exactly as resolve_provider/OllamaProvider do.
    base_url = str(
        getattr(config, "ollama_base_url", "http://localhost:11434")
    ).strip().rstrip("/") or "http://localhost:11434"
    return _is_loopback_url(base_url)


def _corpus_failure(result: dict[str, Any]) -> str | None:
    if result["status"] in {"error", "failed"}:
        if result["error"]:
            return str(result["error"])
        return str(result["failed"][0][1]) if result["failed"] else str(result["status"])
    # A provider that fails before dispatch leaves no failed row: it shows only
    # as selected work that embedded nothing without a stop reason.
    attempted = result["selected"] - result["privacy_withheld"] - len(result["skipped"])
    if attempted > 0 and not result["embedded"] and not result["stop_reason"]:
        return "provider embedded no selected document"
    return None


def _run_pass(db_path: str | Path, config: Any, breaker: Any, slice_state=None) -> tuple[str, str | None]:
    from . import command  # lazy: command imports most of the plugin
    from .embedding_provider import probe_provider_availability

    if _STOP_EMBEDDING_MAINTENANCE.is_set():
        return "stopped", None
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    local = embedding_provider_id(provider) == "fastembed"
    sliced = local and slice_state is not None
    read_conn = command._embedding_read_connection(db_path)
    try:
        profile = command._embedding_current_profile(read_conn)
        if (local and not getattr(config, "embeddings_enabled_explicit", True) and profile is not None
                and embedding_provider_id(str(profile["provider"])) != "fastembed"):
            return f"profile_held:{profile['provider']}", None
        no_profile = profile is None or (local and command._chunk_current_profile(read_conn) is None)
    finally:
        read_conn.close()
    state = local_model_state(config) if local else None
    if state is not None and state[0] == "preparing":
        return "model_preparing", None
    if state is not None and state[0] == "unavailable":
        attempts = _PREPARE_RETRY.get(_local_model_key(config), (0, 0.0))[0]
        return ("model_prepare_exhausted" if attempts >= 6 else "model_unavailable"), None
    if no_profile:
        if not local:
            return "no_profile", None  # `/lcm embed warmup` has not run yet
        result = command._embedding_register_profiles(config, db_path, allow_download=False)
        if isinstance(result, str):
            return "auto_register_unavailable", result
        del result  # providers are disposable; FastEmbed shares its model
    if breaker is not None and breaker.is_open():
        return "circuit_open", None  # the query path's breaker is cooling down
    if not _automatic_chunks_allowed(config, command):
        try:
            command._validate_cloud_chunk_metadata(config, db_path)
        except Exception as exc:
            logger.debug("Cloud chunk metadata walk remains unproven: %s", exc)
    probe = probe_provider_availability(config)
    if not probe["available"]:
        return "provider_unavailable", str(probe["detail"])

    limit = command._embedding_backfill_batch_size(
        getattr(config, "embedding_max_batch_items", None)
    )
    if sliced:
        limit = min(32, limit)
    corpora = [("summaries", lambda retry: command._embedding_backfill_summary_run(
        config, db_path, apply=True, limit=limit, retry_uncertain=retry,
        stop_event=_STOP_EMBEDDING_MAINTENANCE, **({"maintenance_slice": max(0.001, slice_state["deadline"] - time.monotonic())} if sliced else {}),
    ))]
    if _automatic_chunks_allowed(config, command):
        corpora.append(("chunks", lambda retry: command._chunk_backfill_run(
            config, db_path, apply=True, limit=limit, retry_uncertain=retry, policy="",
            incremental=True,
            stop_event=_STOP_EMBEDDING_MAINTENANCE, **({"maintenance_slice": max(0.001, slice_state["deadline"] - time.monotonic())} if sliced else {}),
        )))
    if sliced:
        stage = slice_state.get("stage", 0)
        if stage == 0:
            slice_state["uncertain"] = False
            slice_state["cycle_remaining"] = False
        corpora = [corpora[stage % len(corpora)]]
        before = _scan_frontier(db_path, command)
    outcomes: list[str] = []
    failures: list[str] = []
    for name, run in corpora:
        if _STOP_EMBEDDING_MAINTENANCE.is_set():
            return "stopped", None
        embedded = 0
        if sliced and time.monotonic() >= slice_state["deadline"]:
            slice_state["continue"] = True
            return "slice_budget", None
        retries = (stage >= 2,) if sliced else ((False, True) if local else (False,))
        for retry in retries:
            if _STOP_EMBEDDING_MAINTENANCE.is_set():
                return "stopped", None
            result = run(retry)
            if isinstance(result, str) or _corpus_failure(result) is not None:
                break
            embedded += result["embedded"]
            # Only in-process FastEmbed may retry uncertainty without operator consent.
            if result["stop_reason"]:
                break
        if isinstance(result, str):
            if getattr(result, "reason", "") == "lease_held":
                if not outcomes:
                    return "lease_held", None  # a manual backfill or another process
                outcomes.append(f"{name}=lease_held")
                break
            outcomes.append(f"{name}=refused")
            continue
        if result["stop_reason"] == "process_exit":
            return "stopped", None
        failure = _corpus_failure(result)
        if sliced:
            slice_state["vectors_written"] += embedded
            slice_state["remaining"] = bool(result.get("remaining"))
            slice_state["cycle_remaining"] = slice_state.get("cycle_remaining", False) or slice_state["remaining"]
            slice_state["uncertain"] |= bool(result.get("uncertain"))
        if failure is None:
            outcomes.append(f"{name}={result['status']}:{embedded}")
            continue
        outcomes.append(f"{name}={result['status']}")
        failures.append(f"{name}: {failure}")
        # A whole-run error (provider resolution, profile mismatch, auth,
        # privacy refusal) is shared by both corpora; a row-level failure of
        # one corpus must not starve the other.
        if result["status"] == "error":
            break
    if sliced:
        advanced = before != _scan_frontier(db_path, command)
        slice_state["scan_frontier_advance"] = advanced
        progress = slice_state.get("cycle_progress", 0) + slice_state["vectors_written"] + int(advanced)
        slice_state.update(stage=(stage + 1) % 4, cycle_progress=progress if stage < 3 else 0)
        slice_state["continue"] = not failures and not any("refused" in s for s in outcomes) and (stage < 3 or progress > 0)
        if not failures and stage == 3 and not progress and slice_state["uncertain"]:
            slice_state.update(retry_wait=True, **{"continue": True})
            outcomes.append("row_backoff")
        if not slice_state["continue"] and not failures:
            outcomes.append("refused" if progress or any("refused" in s for s in outcomes) else
                            "no_progress" if slice_state["cycle_remaining"] else "complete")
    return " ".join(outcomes), "; ".join(failures) or None


def _scan_frontier(db_path, command):
    conn = command._embedding_read_connection(db_path)
    try:
        profile = command._chunk_current_profile(conn)
        if profile is None:
            return None
        identity = str(profile["identity_hash"])
        return (command._chunk_cursor(conn, identity), tuple(conn.execute(
            "SELECT key, value FROM metadata WHERE key IN (?, ?, ?) ORDER BY key",
            tuple(f"{prefix}:{identity}" for prefix in (
                "chunk_gapfill_scan", "chunk_policy_revision", "chunk_policy_revision_through")),
        ).fetchall()))
    finally:
        conn.close()
