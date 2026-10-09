"""Incremental embedding maintenance (#1014).

`/lcm embed backfill` was the only writer of vectors, so a store started with
none and every summary or chunk published after a manual backfill stayed
unembedded. The engine now schedules one bounded pass in the background after a
summary is published and on session bind. The pass runs the backfill core
itself (lease, in-flight markers, privacy policy, dtype, budget, provider
timeout), one batch per corpus, so it can never publish a row twice alongside a
manual backfill or another process.

Message chunks are embedded only for a local provider: the cloud raw-text
authorization (`--confirm-raw-text`) is per invocation and never persisted, so
cloud chunk embedding stays operator-initiated.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_PASS: dict[str, tuple[float, str]] = {}
_FAILURES = 0
_FAILURE_WARNED = False


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


def run_incremental_embedding_pass(db_path: str | Path, config: Any, *, breaker: Any = None) -> str:
    """Run one bounded pass over both corpora and record its outcome. Never raises."""
    try:
        outcome, failure = _run_pass(db_path, config, breaker)
    except Exception as exc:
        outcome, failure = "error", str(exc)
    _record(db_path, outcome, failure)
    return outcome


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

    read_conn = command._embedding_read_connection(db_path)
    try:
        if command._embedding_current_profile(read_conn) is None:
            return "no_profile", None  # `/lcm embed warmup` has not run yet
    finally:
        read_conn.close()
    if breaker is not None and not breaker.allows():
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
    if command._is_local_embedding_provider(getattr(config, "embedding_provider", "")):
        corpora.append(("chunks", lambda: command._chunk_backfill_run(
            config, db_path, apply=True, limit=limit, retry_uncertain=False, policy="",
        )))
    outcomes = []
    for name, run in corpora:
        result = run()
        if isinstance(result, str):
            if getattr(result, "reason", "") == "lease_held":
                return "lease_held", None  # a manual backfill or another process
            outcomes.append(f"{name}=refused")
            continue
        failure = _corpus_failure(result)
        if failure is not None:
            return f"{name}={result['status']}", failure
        outcomes.append(f"{name}={result['status']}:{result['embedded']}")
    return " ".join(outcomes), None
