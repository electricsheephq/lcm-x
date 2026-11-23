"""T1-555 (b): publish committed leaves in the compress call that stores them."""

import hashlib
import sqlite3
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import engine_context, pin_char_counter
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_host_uid_shadow import _state_db

SID = "deferred-publication"
FRESH_ROWS = 8


class DeferredPublication(AssertionError):
    pass


def covered_nodes(nodes, messages):
    """Match rendered spans in the returned prefix and follow condensation sources."""
    visible = set()
    for index, row in enumerate(messages):
        if row["role"] == "system":
            continue
        content = row.get("content")
        if isinstance(content, str):
            matches = {node_id for node_id, node in nodes.items()
                       if engine_module._summary_part_text(node) in content}
            if matches:
                visible.update(matches)
                continue
        if index != 0 or row["role"] != "user" or visible:
            break

    pending = list(visible)
    while pending:
        node = nodes[pending.pop()]
        if node.source_type == "nodes":
            added = set(node.source_ids) - visible
            pending.extend(added)
            visible.update(added)
    return visible


class Host:
    def __init__(self, engine, record_property):
        self.engine = engine
        self.record_property = record_property
        self.flush = _host_flush()
        self.rows = []
        self.stored_at = {}
        self.published_at = {}
        self.calls = []

    def rows_for_turns(self, start, stop):
        rows = self.flush([{"role": role, "content": f"Synthetic {role} {turn}. " + "alpha beta " * 110}
                           for turn in range(start, stop) for role in ("user", "assistant")])
        if host_message_uid_mode() == "on":
            for row in rows:
                row["message_uid"] = f"{row['_row_id']:032x}"
        self.rows.extend(deepcopy(rows))
        return rows

    def _census(self, stage):
        expected = Counter((SID, row["role"], row["content"]) for row in self.rows)
        actual = Counter(self.engine._store.connection.execute("SELECT session_id, role, content FROM messages"))
        lost = (expected - actual).total()
        extra = actual - expected
        duplicated = sum(count for key, count in extra.items() if key in expected)
        assert not (lost or extra), (
            f"{stage}: {lost} lost, {duplicated} duplicated, "
            f"{extra.total() - duplicated} unexpected; expected {expected.total()}, stored {actual.total()}"
        )
        return actual.total()

    def ingest(self, messages):
        self.engine.ingest(messages)
        self._census("after ingest")

    def call(self, messages):
        number = len(self.calls) + 1
        self.ingest(messages)
        self.engine.note_turn_complete()
        observed = count_messages_tokens(messages) + 2_000  # provider system and tool schemas
        assert self.engine.should_compress(observed)
        assert not self.engine._automatic_compression_blocked()
        returned = self.engine.compress(messages, current_tokens=observed)
        rows = self._census(f"after compress call {number}")
        assert isinstance(returned, list)
        nodes = {node.node_id: node for node in self.engine._dag.get_session_nodes(SID)}
        new = sorted(nodes.keys() - self.stored_at.keys())
        assert all(nodes[node_id].source_ids for node_id in new), "a stored summary lacks sources"
        self.stored_at.update(dict.fromkeys(new, number))
        for node_id in covered_nodes(nodes, returned):
            self.published_at.setdefault(node_id, number)
        first = sorted(node_id for node_id, call in self.published_at.items() if call == number)
        self.calls.append((new, first, rows))
        self.record_property(f"call_{number}", f"stored={new}, first returned={first}, rows={rows}, lost=0, duplicated=0")
        return returned

    def assert_no_deferred(self):
        deferred = {node_id: (self.stored_at[node_id], published)
                    for node_id, published in self.published_at.items() if self.stored_at[node_id] < published}
        if deferred:
            raise DeferredPublication(
                f"node: (stored call, first returned call) = {deferred}; "
                f"per call (stored nodes, first returned nodes, stored rows) = {self.calls}"
            )


@pytest.fixture
def hosts(tmp_path, monkeypatch, record_property):
    pin_char_counter(monkeypatch)

    def summarize(*, text, **_kwargs):
        digest = hashlib.sha256(text.encode()).hexdigest()[:20]
        return f"Synthetic summary {digest}.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)

    @contextmanager
    def open_host(*, with_uids=False):
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on" if with_uids else "off")
        if with_uids:
            monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
            home = tmp_path / "home"
            home.mkdir()
            _state_db(home, [(SID, None, None)])
        with engine_context(
            tmp_path, window=128_000, session_id=SID, conversation_id=SID,
            context_threshold=0.01, fresh_tail_count=FRESH_ROWS, fresh_tail_max_tokens=0,
            leaf_chunk_tokens=4_000, dynamic_leaf_chunk_enabled=False,
            threshold_full_sweep_enabled=False, incremental_max_depth=0, l3_truncate_tokens=1,
            fresh_tail_pressure_yield_enabled=False, deferred_maintenance_enabled=False,
            embeddings_enabled=False, temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
        ) as engine:
            yield Host(engine, record_property)

    return open_host


@pytest.mark.xfail(strict=True, reason="deferred publication F3, swap rule (b)", raises=DeferredPublication)
def test_f3_hidden_only_leaf(hosts):
    with hosts() as host:
        history = host.rows_for_turns(0, 6)
        host.ingest(history)
        # The host's native context fit keeps the fresh tail and leaves older rows in LCM.
        tail = history[-FRESH_ROWS:]
        assert host.engine._fresh_tail_start(tail) == 0
        returned = host.call(tail)
        assert host.calls[0][0], "the hidden backlog must produce a leaf"
        assert host.engine._compress_hidden_rows_consumed > 0
        assert host.engine._compress_host_rows_consumed == 0

        host.call(returned + host.rows_for_turns(6, 8))
        host.assert_no_deferred()


def test_f4_uid_guard(hosts):
    with hosts(with_uids=True) as host:
        history = host.rows_for_turns(0, 6)
        assert host_message_uid_mode() == "on" and host_uid_emit.identity_emit_enabled()
        host.call(history)
        assert host.calls[0][0], "the first pass must establish a committed prefix"
        assert host.engine._store.count_host_uid_bindings(("canonical",)) == len(history)

        # Replay the covered UID prefix alongside a new unsummarized turn.
        replay = history + host.rows_for_turns(6, 8)
        returned = host.call(replay)
        assert host.calls[1][0], "the UID replay must commit a new leaf"
        host.call(returned)
        host.assert_no_deferred()


def test_f5_refused_pass(hosts):
    with hosts() as host:
        history = host.rows_for_turns(0, 6)
        host.call(history)
        assert host.calls[0][0], "the rejected result must have a committed leaf"
        host.engine.record_rejected_compaction()

        # The host declined the result and replays its prior list. #457 can adopt
        # the committed prefix without summarizing those rows a second time.
        adopted = host.call(history)
        assert host.engine.last_compression_status == "compacted"
        assert len(adopted) < len(history)
        assert not host.calls[1][0], "the retry must use the committed prefix"
        host.assert_no_deferred()


@pytest.mark.xfail(strict=True, reason="deferred publication F6, swap rule (b)", raises=DeferredPublication)
def test_f6_exception_after_leaf_commit(hosts, monkeypatch):
    with hosts() as host:
        history = host.rows_for_turns(0, 6)

        def locked_after_leaf(*_args, **_kwargs):
            assert host.engine._dag.get_session_nodes(SID), "the injected fault must follow a durable leaf"
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as patch:
            patch.setattr(host.engine, "_assemble_committed_compaction_context", locked_after_leaf)
            returned = host.call(history)
        assert host.engine.last_compression_status == "error"
        assert host.calls[0][0], "the fail-open pass must have kept its committed leaf"

        host.call(returned)
        assert not host.calls[1][0], "the retry must use the committed prefix"
        host.assert_no_deferred()
