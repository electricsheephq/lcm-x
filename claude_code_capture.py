"""Claude Code transcript capture into an LCM-X store that Claude Code owns.

Design: docs/claude-code-adapter.md. The transcript file is the capture source; hooks
only say which transcript changed. One store has one owning harness: the writer opens
only stores stamped ``capture:owner`` for Claude Code, never a Hermes profile's store.
Nothing here compacts, blocks or injects context.

Skeleton: the classification below is final for projection v1; the projection, the
writer and the worker land in later commits of this PR.
"""
from __future__ import annotations

from typing import Any, Mapping

PROJECTION_VERSION = "claude-code-v1"
OWNER_KEY = "capture:owner"
OWNER_STAMP = {"host": "claude-code", "format": 1}
SOURCE = "claude-code"

# Entry classes. The audit counts every transcript entry by class, so nothing is
# dropped silently: an unknown type keeps its raw line and is counted.
PROJECTED = "projected"      # becomes message rows; raw line kept in the ledger
RAW_ONLY = "raw_only"        # no rows; protected raw line kept in the ledger
SHARED_RAW = "shared_raw"    # no rows; protected raw line kept once per content digest
UNKNOWN = "unknown_type"     # no rows; raw line kept and counted

PROJECTED_TYPES = frozenset({"user", "assistant"})
# Request snapshots repeat content-addressed blobs (system prompt, tools, request bodies) many
# times per session; their unique content is kept once, never reduced to a digest.
SHARED_RAW_TYPES = frozenset({"api-request", "api-request-blob", "api-request-shape"})
RAW_ONLY_TYPES = frozenset({
    "system", "attachment", "summary", "queue-operation", "last-prompt", "mode", "custom-title",
    "agent-name", "cost-state", "pr-link", "file-history-snapshot", "file-history-delta", "tools-used",
})


def classify_entry(entry: Mapping[str, Any]) -> str:
    """The projection-v1 class of one decoded transcript entry."""
    kind = entry.get("type")
    if kind in PROJECTED_TYPES and isinstance(entry.get("message"), Mapping):
        return PROJECTED
    if kind in SHARED_RAW_TYPES:
        return SHARED_RAW
    if kind in RAW_ONLY_TYPES or kind in PROJECTED_TYPES:
        return RAW_ONLY
    return UNKNOWN


def lineage_key(session_id: str, agent_id: str | None = None) -> str:
    """``host_uid_bindings`` lineage: the session, or a subagent inside it."""
    if not session_id:
        raise ValueError("session_id is required")
    base = f"{SOURCE}:{session_id}"
    return f"{base}/agent:{agent_id}" if agent_id else base


def project_entry(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Message rows for one projected entry, each with its uid (``uuid`` or ``uuid#n``).

    Contract: user text and images give one ``user`` row; each ``tool_result`` part gives
    one ``tool`` row with ``tool_call_id``; assistant text and ``tool_use`` parts give one
    ``assistant`` row with ``tool_calls``. Meta entries, compact summaries and API-error
    entries become ``system`` rows. Thinking and unknown parts stay in the ledger only.
    """
    raise NotImplementedError("projection v1 lands in a later commit of this PR")


def content_digest(entry: Mapping[str, Any]) -> str:
    """Digest of an entry's message content, excluding envelope fields that change when
    Claude Code copies the entry into another session (``sessionId`` on ``--fork-session``)."""
    raise NotImplementedError("lands with the projection")


def run_worker(home: str, *, deadline_s: float | None = None) -> dict[str, Any]:
    """Drain the spool: read each changed transcript from its cursor and commit rows,
    ``host_uid_bindings``, ``capture_events`` and the cursor per batch in one
    ``ingest_atomic()`` transaction; then one bounded embedding catch-up pass.
    One worker per store (exclusive lock); it re-reads the spool before exiting."""
    raise NotImplementedError("the writer and worker land in a later commit of this PR")
