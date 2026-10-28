"""#787 slice 1: prepare leaf summaries off-turn, publish them at the threshold crossing (default off).

Turn end (the post_llm_call hook, after its ingest) only enqueues a snapshot: the binding, the durable frontier,
a copy of the config, the host token count and a ``contextvars.copy_context()``. One daemon thread per process
runs the jobs, each inside its enqueue-time context. A job never touches the live engine: it opens its own
read-only SQLite connection, reads the owned rows above the durable frontier (never the fresh tail, never a row
the foreground would exclude: it stops before one), and summarises one chunk per leaf through the foreground's
summariser with that snapshot, its own breaker keys (``prep:``), its own spend guard and no foreground budget.

Prepared leaves live in a bounded in-memory cache. At the crossing, compress() promotes the longest chain that
starts at the durable frontier through the fenced publication, re-checking each leaf's coverage digest and route
fingerprint inside the publication transaction; a rejected leaf ends the chain and the foreground path runs as
before. A restart, a shutdown or a storage rebind loses the cache; nothing durable is written off-turn.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import logging
import re
import sqlite3
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import hermes_auxiliary_compression_max_concurrency
from .db_bootstrap import select_conversation_range
from .fresh_tail import resolve_fresh_tail_boundary
from .message_content import text_content_for_pattern_matching
from .ingest_protection import (
    quarantine_suspicious_assistant_messages,
    redact_sensitive_value,
)
from .tokens import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

REJECT_REASONS = (
    "frontier_moved",
    "binding_changed",
    "coverage_digest_mismatch",
    "route_fingerprint_mismatch",
    "host_unit_split",
    "fence_conflict",
    "stale_evicted",
)
ROUTE_KEY_PREFIX = "prep:"
_PER_CONVERSATION = 16
_PER_PROCESS = 64
_SCAN_LIMIT = 2000
_MIN_FOREGROUND_SLOTS = 2
_ROW_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned, conversation_id, "
    "ingested_at, observed_at, observed_at_source"
)


class PromotionRejected(RuntimeError):
    """A prepared leaf failed its re-validation inside the publication transaction."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PreparedLeaf:
    session_id: str
    conversation_id: str
    start_frontier: int
    covered_store_ids: tuple
    coverage_digest: str
    route_fingerprint: str
    summary: str
    level: int
    model: str
    source_tokens: int
    summary_tokens: int
    excluded_store_ids: tuple = ()  # stored system prompts between the start frontier and the first covered row

    @property
    def covered_end(self) -> int:
        return max(self.start_frontier, max(self.covered_store_ids))


@dataclass
class _Job:
    db_path: str
    hermes_home: str
    session_id: str
    conversation_id: str
    frontier: int
    generation: int
    config: Any
    fingerprint: str
    tokens: int
    threshold_tokens: int
    context_length: int
    fresh_tail_max_tokens: int
    focus_topic: Optional[str]
    breaker: Any
    foreground_guard: Any
    context: contextvars.Context = field(repr=False)
    dependent_records: tuple = ()  # the session's marked ignored-dependent replies, read at enqueue

    @property
    def key(self) -> tuple:
        return (self.db_path, self.conversation_id)


def route_fingerprint(config) -> str:
    """Everything that shapes a leaf's summary except the focus topic, which every pass re-derives (D-4)."""
    values = [
        getattr(config, name, None)
        for name in (
            "summary_model",
            "summary_fallback_models",
            "summary_reasoning_effort",
            "custom_instructions",
            "summary_prompt_version",
            "l2_budget_ratio",
            "l3_truncate_tokens",
            "leaf_target_ratio",
            "leaf_target_min_tokens",
            "leaf_target_max_tokens",
            "leaf_chunk_tokens",
            "large_output_externalization_enabled",
            "large_output_externalization_threshold_chars",
            "large_output_externalization_path",
            "sensitive_patterns_enabled",
            "sensitive_patterns",
        )
    ]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_digest(rows) -> str:
    """sha256 over the covered rows' (store_id, content) in store order (the reconcile.py:439 idiom)."""
    pairs = sorted([int(store_id), content] for store_id, content in rows)
    return hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def committed_coverage_digest(conn: sqlite3.Connection, store_ids) -> Optional[str]:
    """The digest of the rows as ``conn`` sees them; None when one is missing."""
    ids = sorted({int(store_id) for store_id in store_ids})
    rows = conn.execute(
        "SELECT m.store_id, m.content FROM json_each(?) AS source CROSS JOIN messages AS m "
        "WHERE m.store_id = source.value ORDER BY m.store_id",
        (json.dumps(ids),),
    ).fetchall()
    if [int(row[0]) for row in rows] != ids:
        return None
    return coverage_digest(rows)


def working_leaf_chunk_tokens(config, context_length: int, raw_tokens: int) -> int:
    """The foreground's sweep chunk (compaction.py ``_working_leaf_chunk_tokens``) from a snapshot."""
    base = max(1, config.leaf_chunk_tokens)
    ctx_cap = max(1, int(context_length * 0.4)) if context_length >= 50_000 else None
    if ctx_cap is not None and base > ctx_cap:
        base = ctx_cap
    if not config.dynamic_leaf_chunk_enabled:
        return base
    ceiling = max(base, config.dynamic_leaf_chunk_max)
    if ctx_cap is not None and ceiling > ctx_cap:
        ceiling = ctx_cap
    working = base
    while working < ceiling and raw_tokens > working * 2:
        working = min(ceiling, working * 2)
    return working


_SUMMARY_SCAFFOLD = re.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
)


def excluded_from_summary_input(
    row: Dict[str, Any], session_id: str, dependent_records
) -> bool:
    """Review r1 F1: a stored row the leaf loop keeps out of its summary input (a marked ignored-dependent reply,
    an active-replay placeholder, replay scaffold). Preparation stops before it; read-only and conservative: the
    foreground decides such rows."""
    from .compress_run import CompressRun
    from .placeholder_ledger import PlaceholderLedgerMixin
    from .reconcile import (
        _PRESERVED_OBJECTIVE_CONTEXT_PREFIX,
        _PRESERVED_TODO_CONTEXT_PREFIX,
        ReconcileMixin,
    )

    role = str(row.get("role") or "")
    text = text_content_for_pattern_matching(row.get("content")) or ""
    if text.strip().startswith("[LCM active replay placeholder:"):
        return True
    if role != "tool" and (
        text.lstrip().startswith(
            (_PRESERVED_OBJECTIVE_CONTEXT_PREFIX, _PRESERVED_TODO_CONTEXT_PREFIX)
        )
        or ("[Expand for details:" in text and _SUMMARY_SCAFFOLD.search(text))
    ):
        return True
    if role not in ("assistant", "tool") or not dependent_records:
        return False
    probe = type(
        "_Probe",
        (),
        {
            "_session_id": session_id,
            "_compress_run": CompressRun(),
            "_stable_tool_calls_identity": staticmethod(
                ReconcileMixin._stable_tool_calls_identity
            ),
        },
    )()  # the foreground's own fingerprints, without the engine
    store = PlaceholderLedgerMixin._ignored_dependent_reply_store_fingerprint(
        probe, row
    )
    content = PlaceholderLedgerMixin._ignored_dependent_reply_content_fingerprint(
        probe, row, text
    )
    return any(
        (store and record.get("store") == store)
        or (content and record.get("content") == content)
        for record in dependent_records
    )


def host_unit_boundary(
    previous: Optional[Dict[str, Any]], following: Optional[Dict[str, Any]]
) -> bool:
    """Amendment M-1 item 5: a prepared chunk ends between two stored rows only where the host cannot merge
    them into one message: before a user or system row, or after a tool result that no further result follows.
    Never inside one assistant emission (its content and tool-call rows) or a tool group."""
    if previous is None or following is None:
        return False
    role = following.get("role")
    return role in ("user", "system") or (
        previous.get("role") == "tool" and role != "tool"
    )


def levels(
    threshold_tokens: int, context_length: int, chunk_tokens: int
) -> Optional[tuple[float, float]]:
    """(P, L) in host units, or None when preparation is off for this window (D-10)."""
    window, threshold = int(context_length or 0), int(threshold_tokens or 0)
    if window <= 0 or threshold <= 0:
        return None
    prepare = threshold - 0.15 * window
    low_water = threshold - max(2 * chunk_tokens, 0.1 * window)
    if prepare <= 0.30 * window or low_water <= 0:
        return None
    return prepare, low_water


def _decode_row(row) -> Dict[str, Any]:
    from .store import MessageStore

    return MessageStore._row_to_dict(
        None, row
    )  # stateless decoder (store.py); no live store is touched


def _summary_input(
    rows, config, hermes_home: str, session_id: str
) -> List[Dict[str, Any]]:
    """Stored rows as summariser input through the replay protections (store_complete.py's path), no writes."""
    from .store import MessageStore

    messages = [MessageStore.to_openai_msg(None, row) for row in rows]
    messages = quarantine_suspicious_assistant_messages(
        messages,
        config=config,
        hermes_home=hermes_home,
        session_id=session_id,
        externalize=[False] * len(messages),
    )
    redacted = []
    for message in messages:
        message = dict(message)
        if "content" in message:
            message["content"] = redact_sensitive_value(
                message.get("content"), config, parse_json_strings=False
            )
        if "tool_calls" in message:
            message["tool_calls"] = redact_sensitive_value(
                message.get("tool_calls"), config, parse_json_strings=True
            )
        redacted.append(message)
    return redacted


class _LeafPrep:
    """The process-wide queue, worker, cache and counters."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jobs: "OrderedDict[tuple, _Job]" = OrderedDict()
        self._cache: "OrderedDict[tuple, List[PreparedLeaf]]" = OrderedDict()
        self._generations: Dict[str, int] = {}
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._call_slot = threading.BoundedSemaphore(1)
        # one worker spend guard per profile (Hermes home, store), kept across other profiles' jobs
        self._spend_guards: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.counters: Counter = Counter()

    # -- turn thread ---------------------------------------------------------------------------------------

    def generation(self, db_path: str) -> int:
        with self._condition:
            return self._generations.get(db_path, 0)

    def enqueue(self, job: _Job) -> bool:
        with self._condition:
            if job.generation != self._generations.get(job.db_path, 0):
                return False
            if self._jobs.pop(job.key, None) is not None:
                self.counters["jobs_superseded"] += 1
            self._jobs[job.key] = job
            self.counters["jobs_enqueued"] += 1
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name="lcm-leaf-prep", daemon=True
                )
                self._worker.start()
            self._condition.notify_all()
            return True

    def discard(self, db_path: str) -> None:
        """Shutdown or storage rebind: drop the queued jobs and prepared leaves; a running job is discarded."""
        db_path = str(db_path)
        with self._condition:
            self._generations[db_path] = self._generations.get(db_path, 0) + 1
            for key in [key for key in self._jobs if key[0] == db_path]:
                del self._jobs[key]
            for key in [key for key in self._cache if key[0] == db_path]:
                self.counters["wasted"] += len(self._cache.pop(key))
            self._condition.notify_all()

    def ready_chain(
        self, db_path: str, conversation_id: str, frontier: int
    ) -> List[PreparedLeaf]:
        """The cached chain starting at ``frontier``; leaves no longer reachable from it are dropped."""
        key = (str(db_path), conversation_id)
        with self._condition:
            leaves = self._cache.get(key)
            if not leaves:
                return []
            self._cache.move_to_end(key)
            by_start = {leaf.start_frontier: leaf for leaf in leaves}
            chain, cursor = [], int(frontier)
            while cursor in by_start:
                chain.append(by_start[cursor])
                cursor = by_start[cursor].covered_end
            stale = [leaf for leaf in leaves if leaf not in chain]
            if stale:
                self._remove_locked(key, stale, "frontier_moved")
            return chain

    def reject(self, db_path: str, conversation_id: str, leaves, reason: str) -> None:
        with self._condition:
            self._remove_locked((str(db_path), conversation_id), list(leaves), reason)

    def promoted(self, db_path: str, conversation_id: str, leaf: PreparedLeaf) -> None:
        with self._condition:
            self._remove_locked((str(db_path), conversation_id), [leaf], None)
            self.counters["promoted"] += 1

    def status(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "queued_jobs": len(self._jobs),
                "cached_leaves": sum(len(leaves) for leaves in self._cache.values()),
                **{
                    name: int(self.counters.get(name, 0))
                    for name in (
                        "jobs_enqueued",
                        "jobs_superseded",
                        "prepared",
                        "promoted",
                        "wasted",
                        "failed",
                        "prep_guard_full",
                    )
                },
                "rejected": {
                    reason: int(self.counters.get(f"rejected_{reason}", 0))
                    for reason in REJECT_REASONS
                },
            }

    def _count(self, name: str) -> None:
        with self._condition:
            self.counters[name] += 1

    def _remove_locked(
        self, key: tuple, leaves: List[PreparedLeaf], reason: Optional[str]
    ) -> None:
        cached = self._cache.get(key)
        if not cached:
            return
        drop = [leaf for leaf in cached if leaf in leaves]
        if reason is not None:
            self.counters[f"rejected_{reason}"] += len(drop)
        remaining = [leaf for leaf in cached if leaf not in leaves]
        if remaining:
            self._cache[key] = remaining
        else:
            del self._cache[key]

    # -- worker --------------------------------------------------------------------------------------------

    def wait_idle(self, timeout: float) -> bool:
        """True once no job is queued or running (tests and diagnostics)."""
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._jobs and not self._running, timeout
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                self._running = False
                self._condition.notify_all()
                while not self._jobs:
                    self._condition.wait()
                _key, job = self._jobs.popitem(last=False)
                self._running = True
            try:
                job.context.run(
                    self._prepare, job
                )  # D-11: the enqueue-time context (secret scope)
            except Exception:
                self._count("failed")
                logger.warning(
                    "LCM background leaf preparation failed for conversation=%s",
                    job.conversation_id,
                    exc_info=True,
                )

    def _current(self, job: _Job, conn: sqlite3.Connection) -> bool:
        """The job's binding and frontier still hold, and no shutdown/rebind intervened."""
        if self.generation(job.db_path) != job.generation:
            return False
        row = conn.execute(
            "SELECT current_session_id, current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (job.conversation_id,),
        ).fetchone()
        return (
            row is not None
            and str(row[0] or "") == job.session_id
            and int(row[1] or 0) == job.frontier
        )

    def _guard(self, job: _Job):
        config = job.config
        shape = (
            int(config.background_leaf_prep_max_calls),
            float(config.background_leaf_prep_window_s),
            float(config.summary_spend_backoff_seconds),
        )
        identity = (job.hermes_home, job.db_path)
        if (
            identity not in self._spend_guards
            and len(self._spend_guards) >= _PER_PROCESS
        ):
            free = next(  # only a guard with no call in its window and no backoff left goes
                (
                    key
                    for key, (_shape, guard) in self._spend_guards.items()
                    if guard.max_calls <= 0 or guard.remaining() >= guard.max_calls
                ),
                None,
            )
            if free is None:
                self._count("prep_guard_full")
                return (
                    None  # every kept guard still carries a spend or backoff obligation
                )
            del self._spend_guards[free]
        kept = self._spend_guards.get(identity)
        if kept is None or kept[0] != shape:  # a new profile, or its own limits changed
            from .escalation import SummarySpendGuard

            kept = (
                shape,
                SummarySpendGuard(
                    max_calls=shape[0],
                    window_seconds=shape[1],
                    backoff_seconds=shape[2],
                ),
            )
            self._spend_guards[identity] = kept
        self._spend_guards.move_to_end(identity)
        return kept[1]

    def _prepare(self, job: _Job) -> None:
        if self.generation(job.db_path) != job.generation:
            return
        if (
            hermes_auxiliary_compression_max_concurrency(job.hermes_home) == 1
        ):  # D-7: the host's single aux slot
            return
        uri = Path(job.db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
        try:
            self._prepare_on(job, conn)
        finally:
            conn.close()

    def _prepare_on(self, job: _Job, conn: sqlite3.Connection) -> None:
        from .compaction import select_oldest_leaf_chunk
        from .engine import (
            LeafPrepUnavailable,
            _is_retry_worthy_leaf_summary_error,
            _leaf_target_tokens_for,
            _next_leaf_rescue_chunk_for,
            _primary_summary_route_for,
            _serialize_leaf_input,
            summarize_leaf_chunk_with_rescue,
        )
        from .escalation import summary_route_available

        config = job.config
        if not self._current(job, conn):
            return
        raw = select_conversation_range(
            conn,
            _ROW_COLUMNS,
            job.session_id,
            job.conversation_id,
            job.frontier,
            None,
            _SCAN_LIMIT,
        )
        if len(raw) >= _SCAN_LIMIT:
            return  # the newest rows (the fresh tail) were not read
        rows = [_decode_row(row) for row in raw]
        following = {  # the next stored row, tail included: what the host could merge a chunk end with
            int(row["store_id"]): rows[index + 1] if index + 1 < len(rows) else None
            for index, row in enumerate(rows)
        }
        messages = _summary_input(rows, config, job.hermes_home, job.session_id)
        tail = resolve_fresh_tail_boundary(
            messages,
            fresh_tail_count=config.fresh_tail_count,
            fresh_tail_max_tokens=job.fresh_tail_max_tokens,
        )
        rows, messages = rows[: tail.start], messages[: tail.start]
        anchors, rows, messages = self._eligible_prefix(conn, job, rows, messages)
        if not rows:
            return
        chunk_tokens = working_leaf_chunk_tokens(
            config, job.context_length, count_messages_tokens(messages)
        )
        bounds = levels(job.threshold_tokens, job.context_length, chunk_tokens)
        if bounds is None or job.tokens < bounds[0]:
            return
        need = (
            job.tokens - bounds[1] + chunk_tokens
        )  # D-10: down to L, plus one chunk of lookahead

        position, freed = 0, 0
        chain = self.ready_chain(job.db_path, job.conversation_id, job.frontier)
        for index, leaf in enumerate(chain):
            covered = [
                int(row["store_id"])
                for row in rows[position : position + len(leaf.covered_store_ids)]
            ]
            current = coverage_digest(
                (row["store_id"], row.get("content"))
                for row in rows[position : position + len(covered)]
            )
            if (
                tuple(covered) != leaf.covered_store_ids
                or current != leaf.coverage_digest
                or leaf.route_fingerprint != job.fingerprint
            ):
                with self._condition:
                    self.counters["wasted"] += len(chain) - index
                    self._remove_locked(job.key, chain[index:], None)
                chain = chain[:index]
                break
            position += len(covered)
            freed += leaf.source_tokens - leaf.summary_tokens
        guard = self._guard(job)
        if guard is None:
            return
        while freed < need and len(chain) < _PER_CONVERSATION and position < len(rows):
            remaining = messages[position:]
            if count_messages_tokens(remaining) < chunk_tokens:
                return  # never a partial chunk: the next turn brings more rows
            end = position + len(select_oldest_leaf_chunk(remaining, chunk_tokens))

            def unit_end(end: int) -> bool:
                return host_unit_boundary(
                    rows[end - 1], following.get(int(rows[end - 1]["store_id"]))
                )

            while end > position and not unit_end(end):  # back to a host-unit boundary,
                end -= 1
            if end == position:  # else forward to the next one inside the eligible rows
                end = position + 1
                while end < len(rows) and not unit_end(end):
                    end += 1
                if not unit_end(end):
                    return
            chunk = messages[position:end]
            # D-6: skip while the live route is open, the foreground guard is short or the own guard is spent
            if not summary_route_available(
                config.summary_model,
                config.summary_fallback_models,
                job.breaker,
                prune=False,  # a read of the live breaker, never a write
            ):
                return
            if (
                job.foreground_guard.remaining() < _MIN_FOREGROUND_SLOTS
                or not guard.allows()
            ):
                return
            if not self._call_slot.acquire(
                blocking=False
            ):  # D-7: one worker model call per process
                return
            try:
                if self.generation(job.db_path) != job.generation:
                    return
                result = summarize_leaf_chunk_with_rescue(
                    chunk,
                    config=config,
                    focus_topic=job.focus_topic,
                    circuit_breaker=job.breaker,
                    spend_guard=guard,
                    serialize=lambda batch, budget_tokens=None: _serialize_leaf_input(
                        batch,
                        config=config,
                        hermes_home=job.hermes_home,
                        session_id=job.session_id,
                        externalize_write=False,
                        budget_tokens=budget_tokens,
                    ),
                    leaf_target_tokens=lambda tokens: _leaf_target_tokens_for(
                        config, tokens
                    ),
                    next_rescue_chunk=lambda batch, tokens, final=False: (
                        _next_leaf_rescue_chunk_for(
                            config, select_oldest_leaf_chunk, batch, tokens, final=final
                        )
                    ),
                    primary_route=lambda: _primary_summary_route_for(
                        config, job.breaker
                    ),
                    retry_worthy=_is_retry_worthy_leaf_summary_error,
                    route_key_prefix=ROUTE_KEY_PREFIX,
                )
            except LeafPrepUnavailable:
                return
            except Exception:
                self._count("failed")
                logger.debug(
                    "LCM background leaf preparation summary failed", exc_info=True
                )
                return
            finally:
                self._call_slot.release()
            if result.level == 3 and not result.level_3_verbatim:
                return  # a truncation is never prepared; the foreground decides about level 3 (#652)
            covered_rows = rows[position : position + len(result.chunk)]
            if not unit_end(position + len(covered_rows)):
                return  # a rescue prefix that ends inside a host unit could never be promoted
            start = chain[-1].covered_end if chain else job.frontier
            leaf = PreparedLeaf(
                session_id=job.session_id,
                conversation_id=job.conversation_id,
                start_frontier=start,
                covered_store_ids=tuple(int(row["store_id"]) for row in covered_rows),
                excluded_store_ids=() if chain else anchors,
                coverage_digest=coverage_digest(
                    (row["store_id"], row.get("content")) for row in covered_rows
                ),
                route_fingerprint=job.fingerprint,
                summary=result.summary,
                level=result.level,
                model=result.model,
                source_tokens=result.source_tokens,
                summary_tokens=count_tokens(result.summary),
            )
            if not self._current(job, conn):
                return  # D-9: the binding or frontier moved while the call ran
            if not self._cache_leaf(job, leaf):
                return
            chain.append(leaf)
            position += len(covered_rows)
            freed += leaf.source_tokens - leaf.summary_tokens

    def _eligible_prefix(self, conn, job: _Job, rows, messages):
        """(leading system prompt ids, rows, messages): the rows up to the first one the foreground would exclude
        or this slice cannot account for: a later stored system prompt, a row a summary already covers, or another
        session's row of this conversation (carry)."""
        if not rows:
            return (), rows, messages
        ids = [int(row["store_id"]) for row in rows]
        covered = {
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
                "WHERE node.source_type = 'messages' AND source.value IN (SELECT value FROM json_each(?))",
                (json.dumps(ids),),
            ).fetchall()
        }
        foreign = conn.execute(
            "SELECT MIN(store_id) FROM messages WHERE store_id > ? AND store_id <= ? AND session_id != ? "
            "AND conversation_id = ?",
            (job.frontier, ids[-1], job.session_id, job.conversation_id),
        ).fetchone()
        foreign_id = int(foreign[0]) if foreign and foreign[0] is not None else None
        lead = 0  # a stored system prompt right above the frontier is the host's anchor: excluded, as there
        while lead < len(rows) and rows[lead].get("role") == "system":
            lead += 1
        end = lead
        for row in rows[lead:]:
            store_id = int(row["store_id"])
            if (
                row.get("role") == "system"
                or store_id in covered
                or (foreign_id is not None and store_id > foreign_id)
                or excluded_from_summary_input(
                    row, job.session_id, job.dependent_records
                )
            ):
                break
            end += 1
        return tuple(ids[:lead]), rows[lead:end], messages[lead:end]

    def _cache_leaf(self, job: _Job, leaf: PreparedLeaf) -> bool:
        with self._condition:
            if self._generations.get(job.db_path, 0) != job.generation:
                return False
            leaves = self._cache.setdefault(job.key, [])
            leaves.append(leaf)
            self._cache.move_to_end(job.key)
            self.counters["prepared"] += 1
            while sum(len(cached) for cached in self._cache.values()) > _PER_PROCESS:
                oldest = next(iter(self._cache))
                if oldest == job.key:
                    break
                self._remove_locked(oldest, list(self._cache[oldest]), "stale_evicted")
            return True


_LEAF_PREP = _LeafPrep()


def enqueue_after_turn(engine, history) -> bool:
    """Turn thread, after a successful ingest: capture the snapshot and enqueue one job (D-1, D-11)."""
    if (
        engine._bypasses_lcm_context_management()
        or not engine._session_id
        or not engine._conversation_id
        or getattr(engine, "_consecutive_ingest_failures", 0)
    ):
        return False
    if engine._compiled_ignore_message_patterns:
        return False  # ignored rows need exclusion proofs and dependent-reply handling: foreground only
    threshold, window = (
        int(engine.threshold_tokens or 0),
        int(engine.context_length or 0),
    )
    tokens = int(engine.last_prompt_tokens or 0)
    if threshold <= 0 or window <= 0 or tokens < threshold - 0.15 * window:
        return False  # below P: nothing to prepare yet
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    if (
        state is None
        or str(getattr(state, "current_session_id", "") or "") != engine._session_id
    ):
        return False
    db_path = str(engine._store.db_path)
    config = copy.copy(engine._config)
    recent = engine._redact_active_replay_messages(list(history or [])[-64:])
    job = _Job(
        db_path=db_path,
        hermes_home=str(engine._hermes_home or ""),
        session_id=engine._session_id,
        conversation_id=engine._conversation_id,
        frontier=int(getattr(state, "current_frontier_store_id", 0) or 0),
        generation=_LEAF_PREP.generation(db_path),
        config=config,
        fingerprint=route_fingerprint(config),
        tokens=tokens,
        threshold_tokens=threshold,
        context_length=window,
        fresh_tail_max_tokens=int(engine._effective_fresh_tail_max_tokens()),
        # the foreground's own steering hint from recent turns (it steers; it is not summarised content)
        focus_topic=engine._derive_auto_focus_topic(recent),
        breaker=engine._summary_circuit_breaker,
        foreground_guard=engine._summary_spend_guard,
        context=contextvars.copy_context(),
        dependent_records=tuple(
            engine._load_generated_ignored_dependent_reply_records()
        ),
    )
    return _LEAF_PREP.enqueue(job)


def discard(db_path) -> None:
    _LEAF_PREP.discard(str(db_path))


def ready_chain(db_path, conversation_id: str, frontier: int) -> List[PreparedLeaf]:
    return _LEAF_PREP.ready_chain(str(db_path), conversation_id, frontier)


def reject(db_path, conversation_id: str, leaves, reason: str) -> None:
    _LEAF_PREP.reject(str(db_path), conversation_id, leaves, reason)


def promoted(db_path, conversation_id: str, leaf: PreparedLeaf) -> None:
    _LEAF_PREP.promoted(str(db_path), conversation_id, leaf)


def status() -> Dict[str, Any]:
    return _LEAF_PREP.status()
