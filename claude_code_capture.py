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
    # Desktop bookkeeping seen in real transcripts. Types that may carry content (``result``,
    # ``started``) stay unknown, so the audit shows them until a projection version handles them.
    "atis-latch", "bridge-session", "frame-link", "history-suppression",
    "artifact-autoreact-ledger", "artifact-comment-monitor",
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


def row_session_id(session_id: str, agent_id: str | None = None) -> str:
    """The rows' ``session_id``. A subagent's entries carry the parent's ``sessionId``; its
    rows get their own id, so the parent's chronology stays clean and the child has its own
    ``session_lineage`` link."""
    if not session_id:
        raise ValueError("session_id is required")
    return f"{session_id}/agent:{agent_id}" if agent_id else session_id


def lineage_key(session_id: str, agent_id: str | None = None) -> str:
    """``host_uid_bindings`` lineage: the session, or a subagent inside it."""
    return f"{SOURCE}:{row_session_id(session_id, agent_id)}"


def project_entry(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Message rows for one projected entry, each with its uid (``uuid`` or ``uuid#n``).

    Contract: user text and images give one ``user`` row; each ``tool_result`` part gives
    one ``tool`` row with ``tool_call_id``, holding the full text of a ``tool-results`` file
    when Claude Code saved the output there; an assistant entry gives one row with its text
    and ``tool_use`` parts (none for a thinking-only entry). Meta entries, compact summaries
    and API-error entries become ``system`` rows. Thinking stays in the ledger only.
    """
    raise NotImplementedError("projection v1 lands in a later commit of this PR")


def content_digest(entry: Mapping[str, Any]) -> str:
    """Digest of an entry without the envelope fields that change when Claude Code copies
    it into another session (``sessionId`` and ``promptId``, observed on ``--fork-session``)."""
    raise NotImplementedError("lands with the projection")


def run_worker(home: str, *, deadline_s: float | None = None) -> dict[str, Any]:
    """Drain the spool under the store lock: claim each record by rename, read its transcript
    from the cursor and commit rows, bindings, relations, ``capture_events`` and the cursor
    per batch in one ``ingest_atomic()`` transaction; settle; sweep registered transcripts;
    run one bounded embedding pass; switch the store to rollback mode; release the lock and
    rescan the spool, looping if a record arrived (no lost wake-up)."""
    raise NotImplementedError("the writer and worker land in a later commit of this PR")
