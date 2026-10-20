"""Ordered occurrence checks for append, compress, replay and cold resume.

RecoveryMachine also generates host recovery fits, with larger delivered rows
and a small context window; the original machine keeps its input distribution.
RotationMachine adds the host's compress/end/start commit with fresh child
sessions, checking the ledger across the whole conversation. HostMergeMachine
adds post-adoption role repair; GatewayResetMachine adds gateway /new with
a fresh engine and the same conversation key. CLI bare /new has a seed only.
RestartMachine loses process state at durable transcript/publication boundaries.

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
from hypothesis import Phase, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from hermes_lcm import engine as engine_module
from hermes_lcm import escalation, tokens
from tests.conftest import _MAINTENANCE_SCHEDULERS, _wait_scheduler_idle
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value


# No explain phase: after a failing example (every strict xfail here fails) its line tracer leaves
# sys.monitoring LINE events set on Python 3.12+, which slows every later test in the session.
NO_EXPLAIN = tuple(phase for phase in Phase if phase is not Phase.explain)
for name, examples in (("ci", 25), ("nightly", 500)):
    settings.register_profile(name, max_examples=examples, stateful_step_count=12,
                              derandomize=True, deadline=None, database=None, phases=NO_EXPLAIN)
# Applied to these machines only, never session-wide. Halve each nightly
# allocation so the saved work scales with its measured share; keep 12 steps.
PROFILE = settings.get_profile(os.environ.get("LCM_HYPOTHESIS_PROFILE", "ci"))


def identity(message):
    """Independent normalization, including tool metadata on assistant rows."""
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON object key")
        return result

    def arguments(call):
        value = call["function"].get("arguments")
        if isinstance(value, str):
            try:
                return json.dumps(json.loads(value, object_pairs_hook=unique_object),
                                  sort_keys=True, separators=(",", ":"))
            except ValueError:
                pass
        return value

    return (message["role"], content or "", message.get("tool_call_id") or "",
            tuple((call["id"], call.get("type"), call["function"]["name"], arguments(call)) for call in calls))


def ingest_diagnostics(engine):
    """Read recorded state only; the legacy cursor property delegates to FrontierState."""
    missing = "not recorded" if engine is not None else "engine not supplied"
    return "; ".join(
        f"{name}={getattr(engine, name, missing)!r}"
        for name in ("_ingest_cursor", "_last_ingest_reconciliation",
                     "_ingest_failure_count", "_last_ingest_error")
    )


def assert_occurrences(expected, stored, *, engine=None):
    wanted, actual = list(map(identity, expected)), list(map(identity, stored))
    duplicates, collapses = Counter(actual) - Counter(wanted), Counter(wanted) - Counter(actual)
    assert actual == wanted, (
        f"duplicates={dict(duplicates)}; collapses={dict(collapses)}; "
        f"order_mismatch={not duplicates and not collapses}; "
        f"expected={wanted}; stored={actual}; {ingest_diagnostics(engine)}"
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


KNOWN_OPEN = {1286: cold_stale_proof}

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
TOOL_ARGUMENTS = st.permutations((' {"query": 0}', '{"query":1}', '{"query":2}'))
DEFAULT_ARGUMENTS = (' {"query": 0}', '{"query":1}', '{"query":2}')


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
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")), arguments=TOOL_ARGUMENTS)
    def new_turn(self, text, structured, tool_count, retry, tool_name, arguments=DEFAULT_ARGUMENTS):
        user = self.stamp({"role": "user", "content": (
            [{"type": "text", "text": text}, {"type": "text", "text": "part two"}]
            if structured else text)})
        if retry and len(self.host) >= 4 and self.host[-2]["role"] == "user":
            user["content"] = deepcopy(self.host[-2]["content"])
        is_retry = retry and retry_resend_1179(self.host, user)
        if is_retry:
            self.host = self.host[:-2]
        batch = [user]
        calls = [{"id": f"call-{self.sequence}-{i}", "type": "function",
                  "function": {"name": tool_name, "arguments": arguments[i]}} for i in range(tool_count)]
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
        assert_occurrences(self.occurrences, self.rows(), engine=self.engine)

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
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")), arguments=TOOL_ARGUMENTS)
    def new_turn(self, text, structured, tool_count, retry, tool_name, arguments=DEFAULT_ARGUMENTS):
        if not KNOWN_OPEN[503](self.recovery_aliases, self.host, "new_turn"):
            super().new_turn(text, structured, tool_count, retry, tool_name, arguments)


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

    @rule(text=TEXT, structured=st.booleans(), tool_count=st.integers(0, 3),
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")), arguments=TOOL_ARGUMENTS)
    def new_turn(self, text, structured, tool_count, retry, tool_name, arguments=DEFAULT_ARGUMENTS):
        delivered = len(self.occurrences)
        super().new_turn(text, structured, tool_count, retry, tool_name, arguments)
        for occurrence in self.occurrences[delivered:]:
            occurrence["session_id"] = self.session_id

    def rows(self):
        cursor = self.engine._store.connection.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id FROM messages "
            "WHERE conversation_id = ? ORDER BY store_id", ("identity-conversation",))
        rows = [dict(zip((column[0] for column in cursor.description), row)) for row in cursor.fetchall()]
        # Let the occurrence oracle diagnose missing or extra rows first.
        if len(rows) == len(self.occurrences):
            assert [row["session_id"] for row in rows] == [
                occurrence["session_id"] for occurrence in self.occurrences
            ], "session ownership"
        return rows

    @precondition(lambda self: bool(self.host))
    @rule()
    def rotate(self):
        previous = deepcopy(self.host)
        compressed = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
        # Hermes rejects candidates unchanged even after removing its DB marker.
        def without_persisted(rows):
            return [{key: value for key, value in row.items() if key != "_db_persisted"} for row in rows]

        if compressed == previous or without_persisted(compressed) == without_persisted(previous):
            self.host = previous
            return
        if self.engine._last_compression_status == "sanitized":
            self.stale_proof = True
        elif self.engine._last_compression_status == "compacted":
            self.stale_proof = False
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


def merge_host_users(messages):
    """Fleet role repair, restricted to this generator's LCM (not native) rows.

    agent/agent_runtime_helpers.py:553-594: LCM's summary prefixes are not
    native handoffs recognized by split_user_originated_turn. This generator
    emits no native handoffs, steer rows, or API sidecars.
    """
    merged = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (previous is not None and previous.get("role") == message.get("role") == "user"
                and isinstance(previous.get("content", ""), str)
                and isinstance(message.get("content", ""), str)):
            prefix, incoming = previous.get("content", ""), message.get("content", "")
            content = prefix + "\n\n" + incoming if prefix and incoming else prefix or incoming
            previous["content"] = content
            if prefix and incoming:
                previous["_merged_turn_prefix"] = prefix + "\n\n"
            if content != prefix:
                previous.pop("_db_persisted", None)
        else:
            merged.append(message)
    return merged


class HostMergeMachine(RotationMachine):
    """Repair role alternation after adoption and before the next ingest.

    Fleet: conversation_compression.py:3673-3683, 3445-3451, 2555-2559;
    turn_iteration_prep.py:203 -> agent_runtime_helpers.py:553-594.
    """

    @precondition(lambda self: bool(self.host))
    @rule(in_place=st.booleans())
    def host_merge(self, in_place):
        if not in_place:
            before = len(self.session_ids)
            self.rotate()
            if len(self.session_ids) == before:
                return
        else:
            previous = deepcopy(self.host)
            compressed = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
            if self.engine._last_compression_status == "sanitized" and compressed != previous:
                self.stale_proof = True
            elif self.engine._last_compression_status == "compacted":
                self.stale_proof = False
            def stripped(rows):
                return [{k: v for k, v in row.items() if k != "_db_persisted"} for row in rows]

            if compressed == previous or stripped(compressed) == stripped(previous):
                self.host = previous
                return
            self.engine.on_session_end(self.session_id, previous)
            self.engine.on_session_start(self.session_id, platform="cli", conversation_id="identity-conversation",
                                         boundary_reason="compression", old_session_id=self.session_id)
            self.host = compressed
        self.host = merge_host_users(self.host)


TestHostMerge = HostMergeMachine.TestCase
TestHostMerge.settings = PROFILE


class GatewayResetMachine(ExactlyOnceMachine):
    """Gateway /new ends/shuts down the old engine, then builds a fresh one.

    Fleet: gateway/slash_commands_session.py:174-184; run_agent.py:906-917,
    977-989; agent_init.py:1886,2106-2111. The gateway key survives /new,
    so every delivery remains in the same conversation's occurrence ledger.
    """

    def __init__(self):
        self.session_id = "identity-session"
        self.session_ids = [self.session_id]
        super().__init__()

    def start_engine(self):
        engine = LCMEngine(config=LCMConfig(database_path=str(self.home / "lcm.db"),
                                            fresh_tail_count=2, leaf_chunk_tokens=16),
                           hermes_home=str(self.home / "home"))
        engine.on_session_start(self.session_id, platform="telegram", conversation_id="identity-conversation", context_length=200_000)
        return engine

    @rule(text=TEXT, structured=st.booleans(), tool_count=st.integers(0, 3),
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")), arguments=TOOL_ARGUMENTS)
    def new_turn(self, text, structured, tool_count, retry, tool_name, arguments=DEFAULT_ARGUMENTS):
        delivered = len(self.occurrences)
        super().new_turn(text, structured, tool_count, retry, tool_name, arguments)
        for occurrence in self.occurrences[delivered:]:
            occurrence["session_id"] = self.session_id

    def rows(self):
        cur = self.engine._store.connection.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id FROM messages "
            "WHERE conversation_id = ? ORDER BY store_id", ("identity-conversation",))
        rows = [dict(zip((c[0] for c in cur.description), r)) for r in cur.fetchall()]
        if len(rows) == len(self.occurrences):  # a lost or extra row is the oracle's to report, with its diagnosis
            assert [r["session_id"] for r in rows] == [r["session_id"] for r in self.occurrences], "session ownership"
        others = self.engine._store.connection.execute(
            "SELECT count(*) FROM messages WHERE conversation_id IS NULL OR conversation_id != ?",
            ("identity-conversation",)).fetchone()[0]
        assert others == 0, others
        return rows

    @precondition(lambda self: bool(self.host))
    @rule()
    def compress(self):
        previous = deepcopy(self.host)
        compressed = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
        if self.engine._last_compression_status == "sanitized" and compressed != previous:
            self.stale_proof = True
        elif self.engine._last_compression_status == "compacted":
            self.stale_proof = False
        def strip(rows):
            return [{k: v for k, v in row.items() if k != "_db_persisted"} for row in rows]

        if compressed == previous or strip(compressed) == strip(previous):
            self.host = previous
            return
        self.engine.on_session_end(self.session_id, previous)
        self.engine.on_session_start(self.session_id, platform="telegram", conversation_id="identity-conversation",
                                     boundary_reason="compression", old_session_id=self.session_id)
        self.host = compressed

    @rule()
    def reset(self):
        self.engine.on_session_end(self.session_id, deepcopy(self.host))
        self.engine.shutdown()
        self.session_id = f"identity-reset-{len(self.session_ids)}"
        self.session_ids.append(self.session_id)
        self.host = []
        self.stale_proof = False
        self.engine = self.start_engine()


TestGatewayReset = GatewayResetMachine.TestCase
TestGatewayReset.settings = PROFILE


RESTART_CHILD = "#1286: cold published child lacks its parent emission proof"


def interrupted_child_start(point, host, prior_emission, include_known_open):
    """The host published the child but no compression start re-keyed its proof.

    Two delivered turns can emit the first head; an adopted prior emission
    can be re-rendered. Both need the missing child-start re-key. Check that
    trajectory BEFORE compress, so an excluded restart changes no live proof.
    """
    return (point == "rotation-published" and not include_known_open
            and (prior_emission or sum(row.get("role") == "user" for row in host) >= 2))


KNOWN_OPEN[RESTART_CHILD] = interrupted_child_start

RESTART_SUPERSEDED = "#1286: unadopted compaction supersedes the durable input emission"


def interrupted_recompression(point, prior_emission, include_known_open):
    """A second compress may replace the only proof of the still-durable input.

    Keyed on the prior adopted emission, before starting the interrupted
    compaction. No generated text or observed replay result selects this path.
    """
    return point in ("in-place", "rotation-ended") and prior_emission and not include_known_open


KNOWN_OPEN[RESTART_SUPERSEDED] = interrupted_recompression

RESTART_PENDING = "#849: unanswered persisted user merges with the next completed turn"


def unanswered_before_next_turn(point, host, include_known_open):
    """A fresh pending user has no ingest donor before the next prompt's repair.

    The restart rule observes the next completed turn, not a fabricated cold
    ingest. Its two plain prompts become one row, including without a duplicate.
    """
    return (point == "user-persisted" and bool(host) and host[-1].get("role") == "assistant"
            and not include_known_open)


KNOWN_OPEN[RESTART_PENDING] = unanswered_before_next_turn


RETRY_MERGED_SCAFFOLD = "#1179: retry beside a host-merged scaffold after compaction"


def retry_beside_merged_scaffold(engine, host, user):
    """The retry repeats the suffix of an earlier proof-bound merged row."""
    projection, identities = engine._occurrence_replay_identities(host, engine._active_emission_proof())
    return any(entry.generated_span is not None and entry.kind == "summary"
               and projected is not None and projected[1] == user["content"]
               for entry, projected in zip(projection.entries[:-2], identities[:-2]))


KNOWN_OPEN[RETRY_MERGED_SCAFFOLD] = retry_beside_merged_scaffold


def repeated_user_after_proof_loss(proof_lost, proof, host, occurrences, user):
    """#1328: the unchanged emitted view lost its proof before a fresh user.

    No-progress recompression or unchanged replay consumes the proof without
    a delivery. The next turn's role repair must still change that prefix:
    the four-row control is already repaired and passes after proof loss.
    Tool rows do not select the input; the earlier occurrence must have a
    different timestamp as well as equal identity.
    """
    return (proof_lost and proof is None
            and merge_host_users(deepcopy(host)) != host
            and any(row["role"] == "user" and identity(row) == identity(user)
                    and row.get("timestamp") != user.get("timestamp") for row in occurrences))


class RestartMachine(RotationMachine):
    """Restart from the host's durable view, never the interrupted live list.

    Fleet conversation_compression.py:3683, 3712, 3329-3351 separates the
    end hook, parent flush and atomic child publication. Restart follows the
    published tip (cli_agent_setup_mixin.py:534-617) and calls ordinary start
    (agent_init.py:2106-2111), without a compression boundary notification.
    turn_context.py:963-977 persists the user before post_llm_call ingestion.
    """

    def __init__(self):
        self.host_clock = 1_000_000
        self.proof_lost_view = None
        super().__init__()

    @precondition(lambda self: bool(self.host))
    @rule()
    def compress(self):
        previous = deepcopy(self.host)
        prior_proof = self.engine._compress_commit_proof
        super().compress()
        self._record_proof_loss(previous, prior_proof)

    @precondition(lambda self: bool(self.host))
    @rule()
    def replay(self):
        previous = deepcopy(self.host)
        prior_proof = self.engine._compress_commit_proof
        super().replay()
        self._record_proof_loss(previous, prior_proof)

    @precondition(lambda self: bool(self.host))
    @rule()
    def rotate(self):
        previous = deepcopy(self.host)
        prior_proof = self.engine._compress_commit_proof
        super().rotate()
        self._record_proof_loss(previous, prior_proof)

    def _record_proof_loss(self, previous, prior_proof):
        # rotate() also compresses, including the host-rejected no-progress
        # path. Track the resulting state regardless of the entry point.
        if (prior_proof and prior_proof.get("emissions") and previous == self.host
                and self.engine._compress_commit_proof is None):
            self.proof_lost_view = previous
        elif self.host != self.proof_lost_view or self.engine._compress_commit_proof is not None:
            self.proof_lost_view = None

    def _persist_host_view(self, session, messages, parent=None):
        # Model only the host's persisted columns, not LCM's private metadata
        # or the generator's message_uid. Late CLI resume has no _row_id.
        fields = ("role", "content", "timestamp", "tool_calls", "tool_call_id", "_compressed_summary")
        durable = []
        for message in messages:
            row = {key: deepcopy(message[key]) for key in fields if key in message}
            if not row.get("timestamp"):
                self.host_clock += 1
                row["timestamp"] = float(self.host_clock)
            # _insert_message_rows stamps the live dict as well as SQLite.
            message["timestamp"] = row["timestamp"]
            durable.append(row)
        with sqlite3.connect(self.home / "home" / "state.db") as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS host_views "
                               "(session_id TEXT PRIMARY KEY, messages TEXT NOT NULL)")
            if parent is not None:
                # Host publication commits ancestry AND handoff in one txn.
                connection.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = ?", (parent,))
                connection.execute("INSERT INTO sessions VALUES (?, ?, NULL)", (session, parent))
            connection.execute("INSERT OR REPLACE INTO host_views VALUES (?, ?)",
                               (session, json.dumps(durable, ensure_ascii=False)))

    def _restart_from_host(self):
        # All modeled calls have returned with their transactions committed.
        # shutdown closes handles only; no end/reset hook or in-memory proof
        # is transferred. This models process-state loss, not power-loss IO.
        self.engine.shutdown()
        with sqlite3.connect(self.home / "home" / "state.db") as connection:
            session = self.session_id
            while child := connection.execute(
                    "SELECT child.id FROM sessions child JOIN sessions parent "
                    "ON child.parent_session_id = parent.id "
                    "WHERE parent.id = ? AND parent.end_reason = 'compression'", (session,)).fetchone():
                session = child[0]
            payload = connection.execute("SELECT messages FROM host_views WHERE session_id = ?",
                                         (session,)).fetchone()[0]
        self.session_id = session
        self.host = merge_host_users([dict(row, _db_persisted=True) for row in json.loads(payload)])
        self.engine = self.start_engine()
        self.proof_lost_view = None
        # r34.10 cold start only binds; post_llm_call delivers the next
        # completed turn, after per-request role repair. Never ingest here.
        self.stale_proof = False

    @rule(text=st.one_of(TEXT, st.just("alpha 2")), structured=st.booleans(), tool_count=st.integers(0, 3),
          retry=st.booleans(), tool_name=st.sampled_from(("lookup", "inspect")), arguments=TOOL_ARGUMENTS)
    def new_turn(self, text, structured, tool_count, retry, tool_name, arguments=DEFAULT_ARGUMENTS):
        user = self.stamp({"role": "user", "content": (
            [{"type": "text", "text": text}, {"type": "text", "text": "part two"}]
            if structured else text)})
        if retry and len(self.host) >= 4 and self.host[-2]["role"] == "user":
            user["content"] = deepcopy(self.host[-2]["content"])
        is_retry = retry and retry_resend_1179(self.host, user)
        if (is_retry and not self.include_known_open
                and KNOWN_OPEN[RETRY_MERGED_SCAFFOLD](self.engine, self.host, user)):
            # #1179 sibling S1 (#1446): retry beside a host-merged scaffold after compaction.
            # This replay retries its plain suffix; the earlier merged row fails on the next cold resume.
            return
        self.proof_lost_view = None
        if is_retry:
            self.host = self.host[:-2]
        self.retry_occurrence_index = len(self.occurrences) if is_retry else None
        self.host.append(user)
        self.occurrences.append(dict(deepcopy(user), session_id=self.session_id))
        self.host = merge_host_users(self.host)
        batch = []
        calls = [{"id": f"call-{self.sequence}-{i}", "type": "function",
                  "function": {"name": tool_name, "arguments": arguments[i]}} for i in range(tool_count)]
        if calls:
            batch.append(self.stamp({"role": "assistant", "content": "calling", "tool_calls": calls}))
            batch.extend(self.stamp({"role": "tool", "content": f"result {i}", "tool_call_id": call["id"]})
                         for i, call in enumerate(calls))
        batch.append(self.stamp({"role": "assistant", "content": f"answer {text} occurrence {self.sequence}"}))
        self.host.extend(batch)
        self.occurrences.extend(dict(deepcopy(row), session_id=self.session_id) for row in batch)
        self.engine.ingest(self.host)

    @precondition(lambda self: bool(self.host))
    @rule()
    def cold_resume(self):
        if KNOWN_OPEN[1286](self.stale_proof, "cold_resume"):
            return
        self._persist_host_view(self.session_id, self.host)
        self._restart_from_host()
        self.new_turn("next prompt", False, 0, False, "lookup")

    @precondition(lambda self: bool(self.host))
    @rule(point=st.sampled_from(("in-place", "rotation-ended", "rotation-published", "user-persisted")),
          text=TEXT)
    def restart(self, point, text):
        # The same cold-replay trajectory as the inherited #1286 guard, keyed
        # BEFORE the transition; never exclude from the observed store rows.
        if KNOWN_OPEN[1286](self.stale_proof, "cold_resume"):
            return
        if KNOWN_OPEN[RESTART_PENDING](point, self.host, self.include_known_open):
            return
        prior_proof = self.engine._durable_commit_proof_payload() or {}
        if KNOWN_OPEN[RESTART_CHILD](point, self.host, bool(prior_proof.get("emissions")),
                                     self.include_known_open):
            return
        if KNOWN_OPEN[RESTART_SUPERSEDED](point, bool(prior_proof.get("emissions")), self.include_known_open):
            return
        self._persist_host_view(self.session_id, self.host)
        if point == "user-persisted":
            user = self.stamp({"role": "user", "content": text})
            self.host.append(user)
            self.occurrences.append(dict(deepcopy(user), session_id=self.session_id))
            self._persist_host_view(self.session_id, self.host)
        else:
            previous = deepcopy(self.host)
            compressed = self.engine.compress(self.host, current_tokens=self.engine.threshold_tokens + 1, force=True)
            if point != "in-place":
                def stripped(rows):
                    return [{k: v for k, v in row.items() if k != "_db_persisted"} for row in rows]

                if compressed == previous or stripped(compressed) == stripped(previous):
                    return  # host rejects no progress, so no end/start window
                self.engine.on_session_end(self.session_id, previous)
                if point == "rotation-published":
                    child = f"identity-session-{len(self.session_ids)}"
                    self._persist_host_view(child, compressed, parent=self.session_id)
                    self.session_ids.append(child)
        self._restart_from_host()
        # One generated transition includes the next completed turn. The
        # unchanged oracle then checks all persisted deliveries after their
        # first host-reachable ingest, never an unanswered intermediate state.
        self.new_turn("next prompt", False, 0, False, "lookup")


TestRestart = RestartMachine.TestCase
TestRestart.settings = PROFILE
