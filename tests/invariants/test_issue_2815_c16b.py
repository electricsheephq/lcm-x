"""Regression cells for tool-group closure, warm UID replay and carrier selection."""

import sqlite3
from copy import deepcopy

import pytest

from hermes_lcm.config import uid_recognition_enabled
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.invariants.test_issue_2781_placeholder_walk import (
    LOG,
    PAD,
    PRUNED,
    _bindings,
    _canonical,
    _copy,
    _covered,
    _cross_copy,
    _far_store,
    _frontier,
    _group,
    _lineage,
    _mode,
    _msg,
    _no_uid,
    _open,
    _opening,
    _pass,
    _present_node_sources,
    _rows,
    summaries as summaries,
)

pytestmark = pytest.mark.issue(2815)
CONTIGUITY = "leaf input cannot form a contiguous stored prefix"


def _words(tag, size):
    return " ".join(f"{tag}w{i}" for i in range(size))


def _resolved_ids(engine, rows):
    assert uid_recognition_enabled(engine)
    return [hit["store_id"] for hit in engine._host_uid_resolve(rows).values()]


@pytest.mark.parametrize("reordered", [True, False], ids=["reordered", "contiguous-control"])
def test_b1_closure_preserves_a_publishable_prefix(tmp_path, monkeypatch, summaries, caplog, reordered):
    _mode(monkeypatch, "on")
    caplog.set_level("INFO", logger=LOG)
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [])
        frontier = _frontier(engine)
        x = _msg("x", "user", _words("xray", 120))
        p = _msg("p", "assistant", _words("papa", 120))
        caller, answer = _group("hidden", 1)
        q = _msg("q", "user", _words("quebec", 120))
        budget = sum(count_message_tokens(row) for row in (x, p, caller, q)) + 40
        big = _msg("big", content=_words("bigrow", budget * 6))
        t = _msg("t", "user", _words("tango", 120))
        tail = [_msg("tail1"), _msg("tail2", "user")]
        stored = [x, p, caller, q, big, answer, t, *tail]
        engine.ingest([*prefix, *stored])
        ids = [_canonical(engine, row) for row in stored]
        assert _resolved_ids(engine, stored) == ids
        assert engine._context_aware_leaf_cap() is None
        assert count_message_tokens(big) > budget * 4
        engine._config.leaf_chunk_tokens = budget
        view = [summary, *([p, q, x] if reordered else [x, p, q]), t, *tail]
        output = _pass(engine, deepcopy(view))
        assert _frontier(engine) > frontier, engine._last_compression_noop_reason
        assert engine._last_compression_noop_reason != CONTIGUITY
        expected = ids[:4] if reordered else ids[:2]
        assert _present_node_sources(engine, output) - set(range(1, frontier + 1)) == set(expected)
        if reordered:
            _pass(engine, deepcopy(view))
            assert engine._last_compression_noop_reason != CONTIGUITY
            skipped = [record.message for record in caplog.records if "reason=contiguity" in record.message]
            assert len(skipped) == 1 and "first tool group remains open" in skipped[0]


def test_b1_store_ordered_caller_with_lower_claim(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, _ = _opening(engine, [])
        caller, answer = _group("caller", 1)
        prior = _msg("prior", "user")
        blocker = _msg("blocker", "user")
        later = _msg("later", "user")
        engine.ingest([*prefix, _no_uid(caller), prior, caller, blocker, answer, later])
        store_ids = [row["store_id"] for row in _rows(engine)[len(prefix):]]
        low_claim, prior_id, caller_id, _blocker_id, answer_id, later_id = store_ids
        chunk = [prior, caller, later]
        assert _resolved_ids(engine, chunk) == [prior_id, caller_id, later_id]
        assert _resolved_ids(engine, [answer]) == [answer_id]
        assert engine._stored_row_forms(engine._store.get(low_claim)) == engine._stored_row_forms(
            engine._store.get(caller_id))
        view = [*chunk, blocker]
        mapping = engine._get_store_id_map_for_messages(view)
        # The walk accepts both a map and claims. Supply the lower, byte-equal
        # claim explicitly to exercise their interaction at this boundary.
        leaf = engine._store_complete_input(
            chunk, {id(caller): [low_claim]}, mapping, view, chunk, _frontier(engine), (),
            count_messages_tokens(chunk))
        assert [mapping[id(row)] for row, _claims in leaf] == [prior_id, caller_id]
        assert engine._store_complete_empty_reason != CONTIGUITY


def test_b2_older_excluded_answer_does_not_close_new_caller(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        old = _group("old", ids=["reused"])
        earlier = _group("distinct", ids=["reused"])
        prefix, summary = _opening(engine, [*earlier, *old])
        old_ids = [_canonical(engine, row) for row in old]
        assert len(earlier[1]["content"]) > 200 and earlier[1]["content"] != old[1]["content"]
        assert {*old_ids, _canonical(engine, earlier[1])} <= set(_covered(engine))
        prior = _msg("prior", "user")
        caller, result = _group("new", ids=["reused"])
        retained, later, tail = _msg("retained", "user"), _msg("later"), _msg("tail")
        copies = _copy(old, 1)
        engine.ingest([*prefix, *copies, prior, caller, retained, result, later, tail])
        caller_copy, version = [row["store_id"] for row in _rows(engine)[len(prefix):len(prefix) + 2]]
        engine._store.add_host_uid_bindings(_lineage(engine), [
            (version, old[1]["message_uid"], "version", "version_new"),
        ])
        assert _bindings(engine, old)[old[1]["message_uid"]] == [(old_ids[1], "canonical"), (version, "version")]
        assert _resolved_ids(engine, [caller, result]) == [_canonical(engine, row) for row in (caller, result)]
        output = _pass(engine, [summary, prior, caller, result, later, retained, tail])
        assert caller in output and result in output, engine._last_compression_noop_reason
        assert engine._store_complete_copy_exclusion_proofs == dict(zip((caller_copy, version), old_ids))


def test_b2_later_reuse_does_not_make_unanswered_call_pending(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [])
        prior = _msg("prior", "user")
        caller, _ = _group("unanswered", ids=["reused"])
        newer, result = _group("newer", ids=["reused"])
        blocker, later, tail = _msg("blocker", "user"), _msg("later", "user"), _msg("tail")
        stored = [prior, caller, blocker, newer, result, later, tail]
        engine.ingest([*prefix, *stored])
        ids = [_canonical(engine, row) for row in stored]
        assert _resolved_ids(engine, stored) == ids
        output = _pass(engine, [summary, prior, caller, later, blocker, tail])
        assert ids[1] in _present_node_sources(engine, output), engine._last_compression_noop_reason
        assert ids[4] not in _present_node_sources(engine, output)


def test_b2_later_live_answer_is_bounded_when_caller_has_no_store_id(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, _ = _opening(engine, [])
        prior, blocker, later = [_msg(name, "user") for name in ("prior", "blocker", "later")]
        caller, _ = _group("unbound", ids=["reused"])
        newer, result = _group("newer", ids=["reused"])
        stored = [prior, blocker, newer, result, later]
        engine.ingest([*prefix, *stored])
        assert _resolved_ids(engine, stored) == [_canonical(engine, row) for row in stored]
        assert _bindings(engine, [caller]) == {}
        chunk = [prior, caller, newer, result, later]
        view = [*chunk, blocker]
        mapping = engine._get_store_id_map_for_messages(view)
        assert id(caller) not in mapping
        leaf = engine._store_complete_input(
            chunk, {}, mapping, view, chunk, _frontier(engine), (), count_messages_tokens(chunk))
        assert [row for row, _ids in leaf] == [prior, caller]


def test_b3_warm_timestamp_claim_leaves_placeholder_version_available(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        caller, full = _group("call", 1)
        caller = dict(caller, timestamp=1_790_000_000.0)
        full = dict(full, timestamp=1_790_000_001.0)
        earlier = _group("earlier", ids=[full["tool_call_id"]])
        prefix, summary = _opening(engine, earlier)
        assert len(earlier[1]["content"]) > 200 and earlier[1]["content"] != full["content"]
        assert _canonical(engine, earlier[1]) in _covered(engine)
        retained, after, tail = _msg("retained", "user"), _msg("after"), _msg("tail")
        later = [_msg("later", "user"), _msg("latest")]
        copy_caller, version_row = _copy([caller, full], 1)
        copy_caller.pop("timestamp")
        version_row.pop("timestamp")
        replay_caller, _ = _group("replay-caller", ids=[full["tool_call_id"]])
        engine.ingest([*prefix, caller, full, retained, copy_caller, version_row, replay_caller, after, tail, *later])
        canonical = _canonical(engine, full)
        version = _rows(engine)[len(prefix) + 4]["store_id"]
        assert engine._store.get(version)["content"] == PRUNED
        engine._store.add_host_uid_bindings(_lineage(engine), [(version, full["message_uid"], "version", "version_new")])
        assert _bindings(engine, [full])[full["message_uid"]] == [(canonical, "canonical"), (version, "version")]
        placeholder = dict(full, content=PRUNED)
        placeholder.pop("timestamp")
        assert _resolved_ids(engine, [placeholder]) == [canonical]
        replay_caller_id = _canonical(engine, replay_caller)
        assert _resolved_ids(engine, [replay_caller]) == [replay_caller_id]
        first = _pass(engine, [summary, caller, placeholder, after, retained, tail])
        assert canonical in _present_node_sources(engine, first)
        assert not any(row.get("message_uid") == full["message_uid"] for row in first)
        assert engine._ingest_cursor == len(first)
        live = [*first, _no_uid(caller), _no_uid(full), replay_caller, placeholder, *later]
        placeholder_index = len(first) + 3
        prematch = engine._identity_anchor_prematch
        resolve = engine._host_uid_resolve
        observed_claims, observed_hits = [], []

        def observe_prematch(*args, **kwargs):
            plan = prematch(*args, **kwargs)
            observed_claims.append({i: [row["store_id"] for row in rows] for i, rows in plan["matched"].items()})
            return plan

        def observe_resolve(messages, identities=None, **kwargs):
            hits = resolve(messages, identities, **kwargs)
            if identities is not None and not kwargs.get("prefix_only") and len(messages) == len(live):
                observed_hits.append({i: row["store_id"] for i, row in hits.items() if row is not None})
            return hits

        before = [row["store_id"] for row in _rows(engine)]
        with monkeypatch.context() as subject:
            subject.setattr(engine, "_identity_anchor_prematch", observe_prematch)
            subject.setattr(engine, "_host_uid_resolve", observe_resolve)
            engine.ingest(live)
        assert observed_claims[-1][len(first) + 1] == [canonical]
        assert observed_hits[-1].get(placeholder_index) == version, observed_hits
        assert [row["store_id"] for row in _rows(engine)] == before
        frontier = _frontier(engine)
        engine.note_turn_complete()
        output = engine.compress(live, current_tokens=count_messages_tokens(live), force=True)
        assert _frontier(engine) > frontier and _present_node_sources(engine, output), (
            engine._last_compression_noop_reason)


def test_b4_twin_lookup_failure_allows_publication(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, _, canonical, cases, copies, after = _far_store(engine)
        assert _resolved_ids(engine, [cases[0][0]]) == [cases[0][1]]
        extra = _msg("extra", "user", PAD * 4)
        lookup = engine._store_complete_placeholder_twins
        failures = []

        class FailingConnection:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, *args):
                if "FROM host_uid_bindings AS version" in sql:
                    failures.append(sql)
                    raise sqlite3.OperationalError("synthetic twin lookup failure")
                return self.conn.execute(sql, *args)

        def fail_twins(conn, rows):
            return lookup(FailingConnection(conn), rows)

        monkeypatch.setattr(engine, "_store_complete_placeholder_twins", fail_twins)
        error = None
        try:
            output = _cross_copy(engine, view, after, extra=[extra])
        except sqlite3.OperationalError as caught:
            error = caught
            output = []
        assert set(canonical[0]) <= _present_node_sources(engine, output), error or engine._last_compression_noop_reason
        assert failures
        assert set(copies) <= {row["store_id"] for row in _rows(engine)}


def test_b5_held_version_is_unavailable_to_store_map_fallback(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        _, _, _, cases, _, _ = _far_store(engine, count=1)
        live, canonical, version = cases[0]
        unbound = _no_uid(live)
        assert _resolved_ids(engine, [live]) == [canonical]
        assert engine._get_store_id_map_for_messages([unbound], uid_first=False)[id(unbound)] == version
        mapping = engine._get_store_id_map_for_messages([live, unbound])
        assert dict(mapping) == {id(live): canonical}
        assert mapping.held == {version: canonical}


@pytest.mark.parametrize("system", [False, True], ids=["glued", "leading-system"])
def test_b6_first_leaf_charges_carrier_remainder(tmp_path, monkeypatch, summaries, system):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        from hermes_lcm import engine as engine_module
        from hermes_lcm.engine import _summary_part_text

        system_rows = [{"role": "system", "content": "Synthetic system message."}] if system else []
        opening = [_msg("opening", "user"), _msg("seed", content=_words("seed", 1800))]
        engine.ingest(opening)
        engine._config.fresh_tail_count = 0
        engine._config.leaf_chunk_tokens = count_messages_tokens(opening)
        with monkeypatch.context() as setup:
            setup.setattr(engine_module, "summarize_with_escalation",
                          lambda **_kwargs: (_words("summary", 370) + "\nExpand for details about: turns", 1))
            first = _pass(engine, opening)
        assert _canonical(engine, opening[-1]) in _present_node_sources(engine, first)
        seed, = engine._dag.get_session_nodes("S")
        summary = {"role": "user", "content": _summary_part_text(seed)}
        engine._config.fresh_tail_count = 2
        turns = [_msg(f"turn-{i}", "user" if i % 2 == 0 else "assistant") for i in range(18)]
        engine.ingest([*first, *turns])
        ids = [_canonical(engine, row) for row in turns]
        assert _resolved_ids(engine, turns) == ids
        assert count_message_tokens(summary) > count_messages_tokens(turns[:6])
        budget = sum(count_message_tokens(row) for row in turns[:6]) + 12
        engine._config.leaf_chunk_tokens = engine._config.dynamic_leaf_chunk_max = budget
        carrier = dict(turns[0], content=summary["content"] + "\n\n" + turns[0]["content"])
        assert engine._generated_context_carrier_remainder(carrier) == turns[0]["content"]
        view = [*system_rows, summary, *turns] if system else [carrier, *turns[1:]]
        before = len(_rows(engine))
        output = _pass(engine, view)
        leaf = next((node for node in engine._dag.get_session_nodes("S") if node.source_type == "messages"
                     and ids[0] in node.source_ids), None)
        assert leaf is not None and list(leaf.source_ids) in (
            [ids[:6]] if system else [ids[:5], ids[:6]]), engine._last_compression_noop_reason
        assert set(leaf.source_ids) <= _present_node_sources(engine, output)
        assert _frontier(engine) == max(leaf.source_ids)
        assert len(_rows(engine)) == before
