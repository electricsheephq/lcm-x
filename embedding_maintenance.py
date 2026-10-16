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

#1186: a pass drains the backlog in successive batches until it is empty or
the per-trigger time budget runs out; the next trigger resumes where it
stopped. A local FastEmbed model that is not cached yet is prepared once per
process on a background thread, through the explicit warmup path, so no user
turn waits on the download.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .embedding_provider import _is_loopback_url

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_PASS: dict[str, tuple[float, str]] = {}
_FAILURES = 0
_FAILURE_WARNED = False
# #1186: per-trigger wall-clock budget for the catch-up drain (checked before each corpus and batch).
_CATCH_UP_BUDGET_S = 20.0
# #1186: local model -> (present | preparing | unavailable, reason) for this process.
_PREPARE: dict[str, tuple[str, str]] = {}
# #1186: this process's prepare thread; while it is alive no other prepare starts, and another
# model is admitted by a later trigger.
_PREPARE_THREAD: threading.Thread | None = None


def database_identity(db_path: str | Path) -> str:
    return str(Path(db_path).resolve())


def incremental_embedding_state(db_path: str | Path) -> tuple[tuple[float, str] | None, int]:
    """Return this database's last pass ``(time, outcome)`` and the process failure count."""
    with _STATE_LOCK:
        return _LAST_PASS.get(database_identity(db_path)), _FAILURES


def _record(db_path: str | Path, outcome: str, failure: str | None) -> None:
    global _FAILURES, _FAILURE_WARNED
    with _STATE_LOCK:
        _LAST_PASS[database_identity(db_path)] = (time.time(), outcome)
        if failure is None:
            return
        _FAILURES += 1
        first = not _FAILURE_WARNED
        _FAILURE_WARNED = True
    (logger.warning if first else logger.debug)(
        "LCM incremental embedding pass skipped (%s): %s; it retries after the "
        "next summary or session bind",
        outcome,
        failure,
    )


def _local_model_key(config: Any) -> str | None:
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    if provider not in {"fastembed", "fast-embed"}:
        return None
    return str(getattr(config, "embedding_model", "") or "").strip() or None


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


def start_local_model_prepare(db_path: str | Path, config: Any) -> bool:
    """Start this process's one background prepare of the local model (#1186).

    The thread runs the warmup path, the only one allowed to download, and
    registers the profiles. Recall never waits on it: until the model is ready
    it serves full text and reports degraded. A failure warns once; the next
    process start retries. Only one prepare runs per process: a model requested
    while another is preparing is admitted by a later trigger.
    """
    global _PREPARE_THREAD
    key = _local_model_key(config)
    if key is None:
        return False
    with _STATE_LOCK:
        if key in _PREPARE or (_PREPARE_THREAD is not None and _PREPARE_THREAD.is_alive()):
            return False
        _PREPARE[key] = ("preparing", "")
        # Started under the lock, so a concurrent caller sees it alive.
        _PREPARE_THREAD = threading.Thread(
            target=_prepare_local_model, args=(key, db_path, config),
            name="lcm-embedding-prepare", daemon=True,
        )
        _PREPARE_THREAD.start()
    return True


def _prepare_local_model(key: str, db_path: str | Path, config: Any) -> None:
    global _FAILURE_WARNED
    try:
        from . import command  # lazy: command imports most of the plugin

        result = command._embedding_register_profiles(config, db_path, allow_download=True)
    except Exception as exc:
        result = f"error: {exc}"
    if isinstance(result, dict):
        with _STATE_LOCK:
            _PREPARE[key] = ("present", "")
        return
    reason = str(result).rsplit("error: ", 1)[-1].strip()
    with _STATE_LOCK:
        _PREPARE[key] = ("unavailable", reason)
        first = not _FAILURE_WARNED
        _FAILURE_WARNED = True
    (logger.warning if first else logger.debug)(
        "LCM local embedding model %s is unavailable (%s); recall uses full text. "
        "Check `/lcm doctor` (embedding_provider_health); it retries on the next start",
        key,
        reason,
    )


def run_incremental_embedding_pass(db_path: str | Path, config: Any, *, breaker: Any = None) -> str:
    """Run one bounded pass over both corpora and record its outcome. Never raises."""
    try:
        outcome, failure = _run_pass(db_path, config, breaker)
    except Exception as exc:
        outcome, failure = "error", str(exc)
    _record(db_path, outcome, failure)
    return outcome


def _automatic_chunks_allowed(config: Any, command: Any) -> bool:
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    # resolve_provider accepts `fast-embed` for the provider whose id is `fastembed`.
    provider = {"fast-embed": "fastembed"}.get(provider, provider)
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


def _run_pass(db_path: str | Path, config: Any, breaker: Any) -> tuple[str, str | None]:
    from . import command  # lazy: command imports most of the plugin
    from .embedding_provider import probe_provider_availability

    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    local = provider in {"fastembed", "fast-embed"}
    read_conn = command._embedding_read_connection(db_path)
    try:
        no_profile = command._embedding_current_profile(read_conn) is None
        # Registration commits each profile separately; finish a partial one.
        no_profile = no_profile or (local and command._chunk_current_profile(read_conn) is None)
    finally:
        read_conn.close()
    if no_profile:
        if not local:
            return "no_profile", None  # `/lcm embed warmup` has not run yet
        state = local_model_state(config)
        if state is not None and state[0] == "preparing":
            return "model_preparing", None  # the background prepare registers the profiles
        result = command._embedding_register_profiles(config, db_path, allow_download=False)
        if isinstance(result, str):
            return "auto_register_unavailable", result
        del result  # release the probe models before the backfill loads its own
    if breaker is not None and breaker.is_open():
        return "circuit_open", None  # the query path's breaker is cooling down
    probe = probe_provider_availability(config)
    if not probe["available"]:
        return "provider_unavailable", str(probe["detail"])

    limit = command._embedding_backfill_batch_size(
        getattr(config, "embedding_max_batch_items", None)
    )
    corpora = [("summaries", lambda: command._embedding_backfill_summary_run(
        config, db_path, apply=True, limit=limit, retry_uncertain=False,
    ))]
    if _automatic_chunks_allowed(config, command):
        corpora.append(("chunks", lambda: command._chunk_backfill_run(
            config, db_path, apply=True, limit=limit, retry_uncertain=False, policy="",
            incremental=True,
        )))
    outcomes: list[str] = []
    failures: list[str] = []
    deadline = time.monotonic() + _CATCH_UP_BUDGET_S
    for name, run in corpora:
        # No corpus starts after the budget; the next trigger resumes it.
        if time.monotonic() >= deadline:
            outcomes.append(f"{name}=deferred")
            continue
        embedded = 0
        while True:
            result = run()
            if isinstance(result, str) or _corpus_failure(result) is not None:
                break
            embedded += result["embedded"]
            # Drain batch by batch; no batch starts after the budget, and whatever is
            # left resumes on the next trigger.
            if (not result["embedded"] or not result["remaining"] or result["stop_reason"]
                    or time.monotonic() >= deadline):
                break
        if isinstance(result, str):
            if getattr(result, "reason", "") == "lease_held":
                if not outcomes:
                    return "lease_held", None  # a manual backfill or another process
                outcomes.append(f"{name}=lease_held")
                break
            outcomes.append(f"{name}=refused")
            continue
        failure = _corpus_failure(result)
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
    return " ".join(outcomes), "; ".join(failures) or None
