"""Ordered occurrence checks for append, compress, replay and cold resume.

RecoveryMachine also generates host recovery fits, with larger delivered rows
and a small context window; the original machine keeps its input distribution.
RotationMachine adds the host's compress/end/start commit with fresh child
sessions, checking the ledger across the whole conversation.

The ledger records deliveries, not distinct text. The host view can archive its
last exchange on retry; the ledger must retain both exchanges. Compression
changes only the host view. No host sanitization or arbitrary prefix
rewrite is generated here. Cold resume excludes only a changed, sanitized
compression output whose durable proof is stale (#1286), until a compacted pass.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytest.importorskip("hypothesis")
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from hermes_lcm import engine as engine_module
from hermes_lcm import escalation, tokens
from tests.conftest import _MAINTENANCE_SCHEDULERS, _wait_scheduler_idle
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value


for name, examples in (("ci", 25), ("nightly", 1000)):
    settings.register_profile(name, max_examples=examples, stateful_step_count=12,
                              derandomize=True, deadline=None, database=None)
# Applied to these machines only, never session-wide: LCM_HYPOTHESIS_PROFILE=nightly runs 1,000 examples.
PROFILE = settings.get_profile(os.environ.get("LCM_HYPOTHESIS_PROFILE", "ci"))


def identity(message):
    """Independent normalization, including tool metadata on assistant rows."""
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    return (message["role"], content or "", message.get("tool_call_id") or "",
            tuple((call["id"], call["function"]["name"]) for call in calls))


def assert_occurrences(expected, stored):
    wanted, actual = list(map(identity, expected)), list(map(identity, stored))
    duplicates, collapses = Counter(actual) - Counter(wanted), Counter(wanted) - Counter(actual)
    assert actual == wanted, (
        f"duplicates={dict(duplicates)}; collapses={dict(collapses)}; "
        f"order_mismatch={not duplicates and not collapses}; "
        f"expected={wanted}; stored={actual}"
    )


def retry_resend_1179(host, replacement):
    """Only a new equal-text user occurrence replacing a complete last exchange."""
    return (len(host) >= 4 and host[-2]["role"] == "user"
            and host[-1]["role"] == "assistant" and not host[-1].get("tool_calls")
            and identity(host[-2]) == identity(replacement)
            and host[-2].get("message_uid") != replacement.get("message_uid"))


# Each entry excludes an input shape only while its issue's strict xfail stands in test_identity_seeds.py
# (enforced there by test_known_open_exclusions_track_their_strict_xfails).
def cold_stale_proof(stale_proof, operation):
    """#1286: a changed sanitized output leaves the durable commit proof stale.

    A compacted pass refreshes the proof; a new turn does not. Carrier depth
    alone never excludes a cold resume. Strict seeds retain the failing paths.
    """
    return operation == "cold_resume" and stale_proof


KNOWN_OPEN = {1179: retry_resend_1179, 1286: cold_stale_proof}

REPROJECTED_USER = "#503: repeated recovery fit maps a projected user to an older equal-text source"


def reprojected_user_input(aliases, host, operation):
    """Only the still-present output of a recovery that misbound a proven projection."""
    return operation in ("new_turn", "replay", "compress", "recover", "cold_resume") and any(
        (row.get("message_uid"), identity(row)) in aliases for row in host)


KNOWN_OPEN[503] = reprojected_user_input

def ambiguous_unstamped_tool_window_259(host, ledger):
    """#259 permits duplication of an unproven, truncated tool-only window.

    Full-view append/replay never matches. This is a domain exclusion, never a
    waiver for an oracle failure or for any missing occurrence.
    """
    return (bool(host) and len(host) < len(ledger)
            and all(row["role"] in ("assistant", "tool") for row in host)
            and any(row["role"] == "tool" for row in host)
            and all("timestamp" not in row and "message_uid" not in row for row in host))


def unstamped_user_beside_carrier_923(host, ledger):
    """#923 strict-xfail shape: unstamped U immediately beside its carrier.

    Requires the exact emitted carrier/U supplied by a survival-fit fixture;
    this machine never emits carriers. See test_issue_650_survival_fit_keeps_summary.
    """
    if len(host) < 2 or "timestamp" in host[1] or "message_uid" in host[1]:
        return False
    carrier, user = host[:2]
    return (carrier["role"] == user["role"] == "user"
            and isinstance(carrier.get("content"), str)
            and carrier["content"].startswith("[LCM Summary (")
            and bool(user.get("content"))
            and carrier["content"].endswith("\n\n" + str(user["content"]))
            and identity(user) in list(map(identity, ledger)))


EXCEPTIONS = {259: ambiguous_unstamped_tool_window_259, 923: unstamped_user_beside_carrier_923}
TEXT = st.sampled_from(("alpha", "same text", "βeta", "line one\nline two"))


class ExactlyOnceMachine(RuleBasedStateMachine):
    include_known_open = False

    def __init__(self):
        super().__init__()
        self.temp = TemporaryDirectory(prefix="lcm-identity-")
        self.home = Path(self.temp.name)
        self.patch = pytest.MonkeyPatch()
        # Force the existing char fallback, including functions imported by value.
        self.patch.setattr(tokens, "_get_encoder", lambda: None)
        tokens._count_tokens_cached.cache_clear()
        self.patch.setattr(escalation, "_call_llm_for_summary",
                           lambda *args, **kwargs: "Earlier turns retained.\nExpand for details about: turns")
        self.host, self.occurrences = [], []
        self.stale_proof = False
        self.sequence = 0
        self.retry_occurrence_index = None
        self.engine = self.start_engine()

    def start_engine(self):
        engine = LCMEngine(config=LCMConfig(database_path=str(self.home / "lcm.db"),
                                          fresh_tail_count=2, leaf_chunk_tokens=16),
                           hermes_home=str(self.home / "home"))
        engine.on_session_start("identity-session", platform="cli", conversation_id="identity-conversation",
                                context_length=200_000)
        return engine

    def stamp(self, row):
        self.sequence += 1
        return dict(row, timestamp=float(self.sequence), message_uid=f"occurrence-{self.sequence}")

    def rows(self):
        cursor = self.engine._store.connection.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id FROM messages "
            "WHERE conversation_id = ? ORDER BY store_id", ("identity-conversation",))
        rows = [dict(zip((column[0] for column in cursor.description), row)) for row in cursor.fetchall()]
        assert all(row["session_id"] == "identity-session" for row in rows)
        return rows

    @rule(text=TEXT, structured=st.booleans(), tool_count=st.integers(0, 3),
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")))
    def new_turn(self, text, structured, tool_count, retry, tool_name):
        user = self.stamp({"role": "user", "content": (
            [{"type": "text", "text": text}, {"type": "text", "text": "part two"}]
            if structured else text)})
        if retry and len(self.host) >= 4 and self.host[-2]["role"] == "user":
            user["content"] = deepcopy(self.host[-2]["content"])
        is_retry = retry and retry_resend_1179(self.host, user)
        if is_retry and not self.include_known_open:
            if KNOWN_OPEN[1179](self.host, user):
                return  # exclude this input shape, never an observed oracle failure
        if is_retry:
            self.host = self.host[:-2]
        batch = [user]
        calls = [{"id": f"call-{self.sequence}-{i}", "type": "function",
                  "function": {"name": tool_name, "arguments": "{}"}} for i in range(tool_count)]
        if calls:
            batch.append(self.stamp({"role": "assistant", "content": "calling", "tool_calls": calls}))
            batch.extend(self.stamp({"role": "tool", "content": f"result {i}", "tool_call_id": call["id"]})
                         for i, call in enumerate(calls))
        batch.append(self.stamp({"role": "assistant", "content": f"answer {text} occurrence {self.sequence}"}))
        self.retry_occurrence_index = len(self.occurrences) if is_retry else None
        self.host.extend(batch)
        self.occurrences.extend(deepcopy(batch))
        self.engine.ingest(self.host)

    @precondition(lambda self: bool(self.host))
    @rule()
    def replay(self):
        self.engine.ingest(self.host)

    @precondition(lambda self: bool(self.host))
    @rule()
    def compress(self):
        # Always force: sample actual compaction, without adding cooldown policy
        # to the occurrence model. Compression delivers no new occurrences.
        previous = deepcopy(self.host)
        self.host = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
        if self.engine._last_compression_status == "sanitized" and self.host != previous:
            self.stale_proof = True
        elif self.engine._last_compression_status == "compacted":
            self.stale_proof = False

    @precondition(lambda self: bool(self.host))
    @rule()
    def cold_resume(self):
        if KNOWN_OPEN[1286](self.stale_proof, "cold_resume"):
            return  # exclude the input shape, never an observed oracle failure
        self.engine.shutdown()
        self.engine = self.start_engine()
        self.host = deepcopy(self.host)
        self.engine.ingest(self.host)

    @invariant()
    def exactly_once(self):
        # Exceptions cannot mask failures: assert that generated inputs stay out
        # of those domains, then compare the complete ordered occurrence ledger.
        assert not any(predicate(self.host, self.occurrences) for predicate in EXCEPTIONS.values())
        assert_occurrences(self.occurrences, self.rows())

    def teardown(self):
        try:
            self.engine.shutdown()
            # Product-default maintenance is asynchronous; keep the temporary
            # store and fakes alive until the existing test drain says idle.
            for name in _MAINTENANCE_SCHEDULERS:
                scheduler = getattr(engine_module, name, None)
                if scheduler is not None:
                    assert _wait_scheduler_idle(scheduler, 30.0), name
        finally:
            self.patch.undo()
            tokens._count_tokens_cached.cache_clear()
            self.temp.cleanup()


TestExactlyOnce = ExactlyOnceMachine.TestCase
TestExactlyOnce.settings = PROFILE


class RecoveryMachine(ExactlyOnceMachine):
    """The same ledger and rules, plus the host's bypass-cooldown recovery."""

    recovery_aliases = ()

    def start_engine(self):
        engine = LCMEngine(config=LCMConfig(database_path=str(self.home / "lcm.db"),
                                          fresh_tail_count=4, leaf_chunk_tokens=16),
                           hermes_home=str(self.home / "home"))
        engine.on_session_start("identity-session", platform="cli", conversation_id="identity-conversation",
                                context_length=2048)
        return engine

    def stamp(self, row):
        # Keep plain alpha exchanges short for carrier/kept-row coverage; the
        # other inputs force real projections with the char-token fallback.
        # Expand before stamping/ledger delivery, never after ingest.
        row = deepcopy(row)
        if isinstance(row.get("content"), str):
            if "alpha" not in row["content"]:
                row["content"] *= 384
        elif isinstance(row.get("content"), list):
            for part in row["content"]:
                if part.get("type") == "text":
                    part["text"] *= 384
        return super().stamp(row)

    @precondition(lambda self: bool(self.host))
    @rule()
    def recover(self):
        self._compress_view(recovery=True)

    @precondition(lambda self: bool(self.host))
    @rule()
    def compress(self):
        self._compress_view(recovery=False)

    def _compress_view(self, recovery):
        operation = "recover" if recovery else "compress"
        if KNOWN_OPEN[503](self.recovery_aliases, self.host, operation):
            return
        # Remember the trajectory, not marker text: a proven stored occurrence
        # is about to be projected from a DIFFERENT occurrence of that text.
        mapped = self.engine._get_store_id_map_for_messages(self.host)
        misbound = set()
        for row in self.host:
            source = self.engine._survival_projection_source(
                row, row["role"], normalize_content_value(row.get("content")) or "")
            if source is None and row["role"] == "user" and row.get("timestamp"):
                candidates = self.engine._store.connection.execute(
                    "SELECT store_id FROM messages WHERE session_id = ? AND conversation_id = ? "
                    "AND observed_at = ?", ("identity-session", "identity-conversation", row["timestamp"]))
                for (store_id,) in candidates.fetchall():
                    stored = self.engine._store.get(store_id)
                    if identity(row) == identity(stored):
                        source = stored
                        break
            if (row["role"] == "user" and source is not None
                    and mapped.get(id(row)) not in (None, source["store_id"])):
                misbound.add(row.get("message_uid"))
        previous = deepcopy(self.host)
        self.host = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1,
                                         bypass_cooldown=recovery, force=not recovery)
        self.recovery_aliases = tuple(
            (row.get("message_uid"), identity(row)) for row in self.host
            if (row.get("message_uid"), identity(row)) in self.recovery_aliases
            or (row.get("message_uid") in misbound
                and "[LCM survival fit: this user message " in str(row.get("content", ""))
                and self.engine._survival_projection_source(
                row, row["role"], normalize_content_value(row.get("content")) or "") is None))
        if self.engine._last_compression_status == "sanitized" and self.host != previous:
            self.stale_proof = True
        elif self.engine._last_compression_status == "compacted":
            self.stale_proof = False

    @precondition(lambda self: bool(self.host))
    @rule()
    def cold_resume(self):
        if KNOWN_OPEN[503](self.recovery_aliases, self.host, "cold_resume"):
            return
        super().cold_resume()

    @precondition(lambda self: bool(self.host))
    @rule()
    def replay(self):
        if not KNOWN_OPEN[503](self.recovery_aliases, self.host, "replay"):
            super().replay()

    @rule(text=TEXT, structured=st.booleans(), tool_count=st.integers(0, 3),
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")))
    def new_turn(self, text, structured, tool_count, retry, tool_name):
        if not KNOWN_OPEN[503](self.recovery_aliases, self.host, "new_turn"):
            super().new_turn(text, structured, tool_count, retry, tool_name)


TestRecovery = RecoveryMachine.TestCase
TestRecovery.settings = PROFILE


class RotationMachine(ExactlyOnceMachine):
    """The same occurrence ledger across the host's compression session chain."""

    def __init__(self):
        self.session_id = "identity-session"
        self.session_ids = [self.session_id]
        super().__init__()

    def _publish_host_session(self, child, parent=None):
        """Publish the host ancestry atomically, before LCM sees the child."""
        home = self.home / "home"
        home.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(home / "state.db")
        try:
            with connection:
                connection.execute("CREATE TABLE IF NOT EXISTS sessions "
                                   "(id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
                if parent is not None:
                    connection.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = ?", (parent,))
                connection.execute("INSERT INTO sessions VALUES (?, ?, NULL)", (child, parent))
        finally:
            connection.close()

    def start_engine(self):
        if not (self.home / "home" / "state.db").exists():
            self._publish_host_session(self.session_id)
        engine = LCMEngine(config=LCMConfig(database_path=str(self.home / "lcm.db"),
                                          fresh_tail_count=2, leaf_chunk_tokens=16),
                           hermes_home=str(self.home / "home"))
        engine.on_session_start(self.session_id, platform="cli", conversation_id="identity-conversation",
                                context_length=200_000)
        return engine

    def rows(self):
        cursor = self.engine._store.connection.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id FROM messages "
            "WHERE conversation_id = ? ORDER BY store_id", ("identity-conversation",))
        rows = [dict(zip((column[0] for column in cursor.description), row)) for row in cursor.fetchall()]
        assert all(row["session_id"] in self.session_ids for row in rows)
        return rows

    @precondition(lambda self: bool(self.host))
    @rule()
    def rotate(self):
        previous = deepcopy(self.host)
        compressed = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
        if self.engine._last_compression_status == "sanitized" and compressed != previous:
            self.stale_proof = True
        elif self.engine._last_compression_status == "compacted":
            self.stale_proof = False
        # Hermes rejects candidates unchanged even after removing its DB marker.
        def without_persisted(rows):
            return [{key: value for key, value in row.items() if key != "_db_persisted"} for row in rows]

        if compressed == previous or without_persisted(compressed) == without_persisted(previous):
            self.host = previous
            return
        self.engine.on_session_end(self.session_id, previous)
        child = f"identity-session-{len(self.session_ids)}"
        self._publish_host_session(child, self.session_id)
        self.engine.on_session_start(child, platform="cli", conversation_id="identity-conversation",
                                     boundary_reason="compression", old_session_id=self.session_id)
        self.session_id = child
        self.session_ids.append(child)
        self.host = compressed


TestRotation = RotationMachine.TestCase
TestRotation.settings = PROFILE
