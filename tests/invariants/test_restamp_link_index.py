"""Synthetic warm-engine shared-stamp linking: behavior and identity-build cost."""

import json
import time
from collections import Counter
from copy import deepcopy

import pytest

from tests.harness import make_engine

BASE = 1_700_000_000.0
PAD = " synthetic tool output" * 12
# Captured on unmodified main 9f7f0bd3c666479c983b294a147525bfc46270dd,
# independently for full and host-pruned tools: exactly the same 349 pairs.
# Ranges encode the consecutive pairs, not a fixture or an identity-based oracle.
EXPECTED_LINKS = ({(581, 201), (938, 211)}
                  | {(sid, sid - 352) for sid in range(762, 933)}
                  | {(sid, sid - 347) for sid in range(933, 938)}
                  | {(sid, sid - 348) for sid in range(939, 1110)})


def _view():
    rows = []
    for turn in range(40):
        stamp = BASE + 100 * turn
        rows.append({"role": "user", "content": f"user {turn}{PAD}", "timestamp": stamp})
        for chain in range(4):
            call_id = f"call_{turn}_{chain}"
            rows.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": "probe", "arguments": json.dumps({"turn": turn, "chain": chain})}}],
                "timestamp": stamp + 1 + 2 * chain})
            rows.append({"role": "tool", "tool_call_id": call_id,
                         "content": f"result {turn} {chain}{PAD}", "timestamp": stamp + 2 + 2 * chain})
        rows.append({"role": "assistant", "content": f"final {turn}{PAD}", "timestamp": stamp + 11})
    for index, row in enumerate(rows):
        row["message_uid"] = f"{index + 1:032x}"
    return rows


def _reload(view, stamps):
    """Shared borrowed stamps and no assistant/tool uids, as on a host reload."""
    result = deepcopy(view)
    changed_user = False
    for index, row in enumerate(result):
        if row["role"] == "user":
            if len(stamps) == 1 or index < len(result) // 2 or changed_user:
                continue
            changed_user = True
        row["timestamp"] = stamps[0 if index < len(result) // 2 else -1]
        row.pop("message_uid", None)
    return result


def _links(engine):
    return set(engine._store._conn.execute(
        "SELECT store_id, related_store_id FROM message_relations WHERE kind='possible_replay_of'"))


@pytest.mark.parametrize("pruned", [False, True], ids=["full-tools", "pruned-tools"])
def test_two_warm_restamp_bursts(tmp_path, monkeypatch, pruned):
    from hermes_lcm import host_uid_emit
    from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", False)
    engine = make_engine(tmp_path, window=200_000, session_id="S", conversation_id="conv",
                         fresh_tail_count=390, leaf_chunk_tokens=1)
    try:
        view = _view()
        assert len(view) == 400
        engine.ingest(deepcopy(view))
        stamps = tuple(row["timestamp"] for row in view if row["role"] == "user")[-2:]
        first = _reload(view, stamps)
        if pruned:
            for row in first[:-10]:
                if row["role"] == "tool":
                    row["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
        engine.ingest(deepcopy(first))
        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation",
                            lambda **kwargs: ("Earlier synthetic turns.\nExpand for details about: turns", 1))
        compacted = engine.compress(deepcopy(first), force=True)
        assert engine._last_compression_status == "compacted"
        # The warm engine's output proof names the first burst's shared stamps.
        # Exchanging them on the next reload forces linking against large buckets.
        before = engine._store.get_session_messages("S")

        original_link = engine._link_restamped_occurrences
        original_identity = engine._message_replay_identity
        original_find = engine._store.find_rows_by_observed_at
        original_tool_rows = engine._store.find_session_tool_rows
        active, counts, target_ids, calls = [], Counter(), set(), Counter()

        def identity(message, **kwargs):
            if active:
                counts["identity_calls"] += 1
                calls[message.get("store_id")] += 1
            return original_identity(message, **kwargs)

        def find(*args, **kwargs):
            targets = original_find(*args, **kwargs)
            if active:
                target_ids.update(row["store_id"] for row in targets)
            return targets

        def tool_rows(*args, **kwargs):
            if active:
                counts["tool_queries"] += 1
            return original_tool_rows(*args, **kwargs)

        def link(messages, stored_at, previous, recovery_fallback):
            counts["rows"] += sum(bool(stored_at.get(index)) for index in previous)
            counts["link_calls"] += 1
            active.append(True)
            try:
                return original_link(messages, stored_at, previous, recovery_fallback)
            finally:
                active.pop()

        monkeypatch.setattr(engine, "_message_replay_identity", identity)
        monkeypatch.setattr(engine._store, "find_rows_by_observed_at", find)
        monkeypatch.setattr(engine._store, "find_session_tool_rows", tool_rows)
        monkeypatch.setattr(engine, "_link_restamped_occurrences", link)
        second = _reload(compacted, stamps[::-1])
        second.append({"role": "user", "content": "a new synthetic question", "timestamp": BASE + 1_000_000})
        started = time.perf_counter()
        engine.ingest(deepcopy(second))
        seconds = time.perf_counter() - started
        links = _links(engine)
        evidence = dict(pruned=pruned, seconds=seconds, targets=len(target_ids),
                        **counts, relations=len(links), max_calls=max(calls.values(), default=0))
        print("RESTAMP_INDEX", json.dumps(evidence, sort_keys=True))
        assert links == EXPECTED_LINKS, evidence
        assert len(links) == 349
        assert engine._store.get_session_messages("S")[:len(before)] == before
        assert counts["link_calls"] == 1 and counts["rows"] == 348, evidence
        assert len(target_ids) == 712, evidence  # keep the cost check non-vacuous
        # Wall time is diagnostic only: query/build counts are stable under load.
        if pruned:
            assert counts["tool_queries"] < counts["rows"] + len(target_ids), evidence
        assert counts["identity_calls"] < 3 * (counts["rows"] + len(target_ids)), evidence
        assert max(calls.values(), default=0) <= (2 if pruned else 1), evidence
    finally:
        engine.shutdown()


def test_predecessor_alias_and_decreasing_source_ids(tmp_path, monkeypatch):
    """Drive the ingest-owned linker directly, as the existing rotation-tier cells do."""
    from hermes_lcm import host_uid_emit
    from hermes_lcm.engine import _commit_proof_identity_digest, _proof_user_identity

    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", False)
    engine = make_engine(tmp_path, window=200_000, session_id="S", conversation_id="conv",
                         embeddings_enabled=False, deferred_maintenance_enabled=False)
    try:
        def user(content, stamp):
            return {"role": "user", "content": content, "timestamp": stamp}

        # Real R5/H1 alias path: a merged survivor inherits its absorbed head's stamp.
        merged = "synthetic head\n\nsynthetic tail"
        engine.ingest([user("synthetic head", 10.0), user(merged, 20.0)])
        view = [user(merged, 10.0), {"role": "assistant", "content": "synthetic reply", "timestamp": 21.0},
                user("synthetic pending", 30.0)]
        engine.ingest(deepcopy(view))
        store = engine._store
        alias = next(row for row in store.get_session_messages("S") if row["content"] == merged)
        assert alias["observed_at"] == 20.0
        assert [row["observed_at"] for row in store.get_message_relations(
            [alias["store_id"]], "alt_stamp")] == [10.0]

        def append(session, content, stamp):
            return store.append(session, user(content, stamp), conversation_id="conv")

        early = append("S", "synthetic beta", 10.0)
        # Real engine compression-boundary rotation records the one-hop predecessor.
        engine.on_session_end("S", deepcopy(view))
        engine.on_session_start("rotated", boundary_reason="compression", old_session_id="S",
                                conversation_id="conv", platform="cli")
        assert engine._rotation_predecessor_session_ids("rotated", max_hops=1) == ["S"]
        low = append("rotated", "synthetic beta", 300.0)
        future = append("S", "synthetic beta", 10.0)
        high = append("rotated", merged, 301.0)
        assert early < low < future < high

        finds = []
        original_find = store.find_rows_by_observed_at

        def find(conversation, sessions, stamps):
            rows = original_find(conversation, sessions, stamps)
            finds.append((sessions, rows))
            return rows

        monkeypatch.setattr(store, "find_rows_by_observed_at", find)
        messages, stored_at, previous = [], {}, {}
        stored = store.get_batch([high, low])
        # Same predecessor bucket: high indexes the future beta; low must filter it out.
        for idx, sid in enumerate((high, low)):
            row = stored[sid]
            digest = _commit_proof_identity_digest(_proof_user_identity(
                engine._message_replay_identity(row, stored_row=True)))
            messages.append(user(row["content"], row["observed_at"]))
            stored_at[idx] = sid
            previous[idx] = ("rotated", {"timestamp": 10.0, "identity_sha256": digest})
        with store.atomic(immediate=True):
            engine._link_restamped_occurrences(messages, stored_at, previous, False)
        assert [sessions for sessions, _rows in finds] == [["rotated"], ["S"]]
        alias_copy = next(row for row in finds[1][1] if row["store_id"] == alias["store_id"])
        assert alias_copy == {**alias, "observed_at": 10.0}
        links = _links(engine)
        print("RESTAMP_PREDECESSOR", json.dumps({"pairs": sorted(links), "source_order": [high, low],
                                                 "early": early, "future": future}, sort_keys=True))
        # Captured first on unmodified main 9f7f0bd3c (r2-old-record): full pair SET,
        # including absence of a link to future=7 despite visiting source=8 first.
        assert links == {(6, 5), (8, 2)}
    finally:
        engine.shutdown()
