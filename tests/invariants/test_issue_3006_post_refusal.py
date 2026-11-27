"""Issues #3001 and #3006: refused-pass replay and clean restart (I1, I2, I11)."""

import json
import logging
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
from itertools import count
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm import rollup_builder
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_analysis import _merge_adjacent_assistant_messages
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_messages_tokens
from tests.test_compression_boundary import _config, _stub_summarizer, _turn
from tests.test_host_uid_shadow import _state_db
from tests.test_refused_compaction_rollback import (
    _compaction_records,
    _defer_rewriting_pass_refusal,
    _drain_session_end_maintenance,
    committed_host,  # noqa: F401
)

SID = "S0"
LINEAGE_STOP = "selected leaf chunk lacks raw store lineage"


def _verified_user_heads(engine):
    return {
        store_id: content
        for store_id, content in engine._store._conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id = ? AND role = 'user'", (SID,)
        )
        if engine._verified_lcm_summary_prefix_end(content)
    }


def _host_flush():
    row_ids = count(1)

    def flush(rows):
        result = []
        for row in rows:
            if "timestamp" in row and "_row_id" in row:
                result.append(row)
                continue
            row_id = next(row_ids)
            stamped = dict(row)
            stamped.setdefault("timestamp", 1_760_000_000.0 + row_id * 0.5)
            stamped.setdefault("_row_id", row_id)
            result.append(stamped)
        return result

    return flush


def _restart(engine, binding):
    config, home = engine._config, str(engine._hermes_home)
    engine.shutdown()
    resumed = LCMEngine(config=config, hermes_home=home)
    resumed.on_session_start(SID, **binding)
    return resumed


R3_WINDOW = 40_000
R3_OVERHEAD = 20_000


def _r3_frontier(engine):
    return int(engine._lifecycle.get_by_session(SID).current_frontier_store_id)


def _r3_raw_key(engine, row):
    if engine._is_verified_replay_scaffold_message(row):
        return None
    content = engine._generated_context_carrier_remainder(row) or normalize_content_value(row["content"])
    return _r3_literal_key(row, content)


def _r3_literal_key(row, content):
    return (
        row["role"], content, row.get("tool_call_id") or "", normalize_content_value(row.get("tool_calls")),
        row.get("tool_name") or row.get("name") or "",
    )


def _r3_fit_coverage(engine, before, plan):
    # The planner can shorten an ingest-proven list even where its per-row map
    # is incomplete. Match the cut against literal stored rows. Repeated bytes
    # certify coverage only when all of their stored copies agree.
    def raw_counts(rows):
        return Counter(key for row in rows if (key := _r3_raw_key(engine, row)) is not None)

    dropped = raw_counts(before) - raw_counts(plan.fitted)
    if dropped.total() != plan.count:
        return 0, plan.count
    stored = defaultdict(set)
    for row in engine._store.get_session_messages(SID):
        stored[_r3_literal_key(row, row["content"])].add(row["store_id"])
    covered = engine._store_complete_node_covered({sid for key in dropped for sid in stored[key]})
    uncovered, unknown = 0, 0
    for key, copies in dropped.items():
        ids = stored[key]
        if not ids or (ids & covered and ids - covered):
            unknown += copies
        elif not ids & covered:
            uncovered += copies
    return uncovered, unknown


def _r3_threshold_pass(engine, live, caplog, summarizer, monkeypatch, *, overhead=R3_OVERHEAD):
    estimate = count_messages_tokens(live)
    prompt_tokens = estimate + overhead
    frontier = _r3_frontier(engine)
    record = {
        "gate": "below threshold", "stop": "—", "noop": "", "leaves": 0, "condensations": 0,
        "calls": 0, "rows_in": len(live), "rows_out": len(live),
        "tokens_in": estimate, "tokens_out": estimate, "frontier_in": frontier, "frontier_out": frontier,
        "fit_rows": 0, "uncovered": 0, "unknown_coverage": 0, "fit_reasons": [], "adoption": "",
        "hidden_only": False,
    }
    # The token-based threshold route and the host wrapper's automatic hold gate.
    if not engine.should_compress(prompt_tokens) or engine._automatic_compression_blocked():
        if prompt_tokens >= engine.threshold_tokens:
            record["gate"] = engine._compression_block_reason() or "threshold gate declined"
        return live, record

    caplog.clear()
    before_calls = summarizer.call_count
    before_sweep = engine._last_threshold_full_sweep
    with monkeypatch.context() as patch:
        apply_fit = Mock(wraps=engine._apply_survival_fit)
        patch.setattr(engine, "_apply_survival_fit", apply_fit)
        returned = engine.compress(live, current_tokens=prompt_tokens)
    stop, = [
        entry.getMessage() for entry in caplog.records if entry.name == "hermes_lcm.compaction"
        and entry.getMessage().startswith("LCM compaction stop: reason=")
    ]
    reason, leaf_stats = stop.split("reason=", 1)[1].split(" leaves=", 1)
    fits = [
        re.search(r"reason=(.*?), conversation=.*?, dropped_rows=(\d+), .*?uncovered_rows=(\d+|unknown)", line)
        for line in caplog.messages if line.startswith("LCM survival fit applied (")
    ]
    fit_rows = sum(int(fit[2]) for fit in fits)
    coverage = [fit[3] for fit in fits]
    unknown = 0
    if "unknown" in coverage:
        applied = [(before, plan) for call in apply_fit.call_args_list for before, plan in [call.args[:2]]
                   if count_messages_tokens(plan.fitted) < count_messages_tokens(before)]
        assert len(applied) == len(fits)
        proven = [
            _r3_fit_coverage(engine, before, plan) if fit[3] == "unknown" else (int(fit[3]), 0)
            for (before, plan), fit in zip(applied, fits)
        ]
        uncovered, unknown = (sum(value for value, _ in proven), sum(value for _, value in proven))
    else:
        uncovered = sum(map(int, coverage))
    sweep = engine._last_threshold_full_sweep
    record.update(
        gate="ran", stop=reason, noop=engine.last_compression_noop_reason, leaves=int(leaf_stats.split()[0]),
        condensations=0 if sweep is before_sweep else sweep.get("condensation_passes", 0),
        calls=summarizer.call_count - before_calls, rows_out=len(returned), tokens_out=count_messages_tokens(returned),
        frontier_out=_r3_frontier(engine), fit_rows=fit_rows, uncovered=uncovered, unknown_coverage=unknown,
        fit_reasons=[fit[1] for fit in fits],
        hidden_only=any(line.startswith("LCM hidden-only compaction kept host list unchanged")
                        for line in caplog.messages),
    )
    return returned, record


def _r3_adopt(engine, binding, live, returned, flush, record):
    # Mirror #922's token anti-growth guard and #1116 Host.commit_in_place.
    if returned != live and record["tokens_out"] <= record["tokens_in"]:
        _drain_session_end_maintenance(engine)
        engine.on_session_end(SID, live)
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, **binding)
        record["adoption"] = "adopted"
        return flush(returned)
    if record["tokens_out"] > record["tokens_in"]:
        # Keep the history. The experiment injects a single refusal signal.
        record["adoption"] = "kept (would grow)"
    return live


def _r3_grow_to_leaf(engine, live, turn_ids, flush, end_turn, caplog, summarizer, monkeypatch, *,
                     finish=False, make_turn=_turn):
    attempts = []
    for _ in range(160):
        turn = list(make_turn(next(turn_ids)))
        live = live + flush(turn[1:] if finish else turn)
        finish = False
        engine.ingest(live)
        returned, record = _r3_threshold_pass(engine, live, caplog, summarizer, monkeypatch)
        if record["gate"] == "ran":
            attempts.append(record)
            if record["leaves"]:
                return live, returned, record
        end_turn(engine)
    pytest.fail(f"no automatic leaf after 160 ordinary turns, threshold={engine.threshold_tokens}, "
                f"last gate={record['gate']}, last list estimate={record['tokens_in']}, "
                f"admitted passes={json.dumps(attempts)}")


def _r3_repeated_seed_turn(_i):
    # Model a repeated request with a short, repeated answer. A structured
    # text user row keeps LCM's preceding summary in its own row.
    user, reply = _turn(1)
    return {**user, "content": [{"type": "text", "text": user["content"]}]}, reply


def _r4_departures(engine, before, adopted):
    """Check raw rows leaving any adopted view against durable DAG leaf sources."""
    def occurrences(view):
        return Counter(
            (key, row.get("timestamp"))
            for row in view if (key := _r3_raw_key(engine, row)) is not None
        )

    previous, remaining = occurrences(before), occurrences(adopted)
    departed = previous - remaining
    # Assembly can attach a retained user row to a new summary carrier without
    # copying its stamp. Recognise that row by unique content in the fixture.
    unpaired = remaining - previous
    for key, stamp in tuple(departed):
        if unpaired[key, None] == 1 and sum(n for (k, _), n in departed.items() if k == key) == 1:
            del departed[key, stamp]
    if not departed:
        return (0, 0, 0), []
    rows = engine._store.get_session_messages(SID)
    stored = defaultdict(set)
    stamped = defaultdict(set)
    for row in rows:
        key = _r3_literal_key(row, row["content"])
        stored[key].add(row["store_id"])
        stamped[key, row["observed_at"]].add(row["store_id"])

    # The committed fixture emits one merged plain-text assistant reply.
    # When there is no literal copy, require one exact adjacent pair and
    # use both of its source rows to decide whether the merged row is covered.
    unmatched = {key for key, _ in departed} - stored.keys()
    merged = defaultdict(list)
    for left, right in zip(rows, rows[1:]):
        if left["role"] == right["role"] == "assistant":
            left_key = _r3_literal_key(left, left["content"])
            right_key = _r3_literal_key(right, right["content"])
            if not any(left_key[2:] + right_key[2:]):
                content = f"{left['content'].strip()}\n{right['content'].strip()}"
                key = _r3_literal_key(left, content)
                if key in unmatched:
                    merged[key].append((left["store_id"], right["store_id"]))
    merged = {key: pairs[0] for key, pairs in merged.items() if len(pairs) == 1}
    source_ids = {sid for ids in stored.values() for sid in ids}
    source_ids.update(sid for pair in merged.values() for sid in pair)
    covered = engine._store_complete_node_covered(source_ids)
    uncovered, unknown = 0, 0
    for (key, stamp), copies in departed.items():
        ids = stamped[key, stamp] or stored[key]
        if key in merged:
            uncovered += copies if set(merged[key]) - covered else 0
        elif not ids or (ids & covered and ids - covered):
            unknown += copies
        elif not ids & covered:
            uncovered += copies
    return (departed.total(), uncovered, unknown), sorted(merged.values())


@pytest.mark.parametrize(
    ("awkward", "deferred", "restart", "second_refusal", "turns"),
    [
        pytest.param(False, True, True, True, 100, id="way3-second-refusal"),
        pytest.param(False, True, True, False, 100, id="way3-single-refusal"),
        pytest.param(True, True, True, True, 100, id="way4-second-refusal"),
        pytest.param(True, True, True, False, 100, id="way4-single-refusal"),
        pytest.param(True, False, False, False, 60, id="way5-applied-warm"),
        pytest.param(True, False, True, False, 60, id="way6-applied-restart"),
    ],
)
def test_r4_automatic_refusal_recovery(
    awkward, deferred, restart, second_refusal, turns, request, tmp_path, monkeypatch, caplog,
):
    flush = _host_flush()
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    with ExitStack() as cleanup:
        if awkward:
            engine, kept, original_binding = request.getfixturevalue("committed_host")
            binding = dict(original_binding, context_length=R3_WINDOW)
            engine.on_session_start(SID, **binding)
            live, turn_ids = flush(kept), count(20)
            stub = engine_module.summarize_with_escalation
        else:
            binding = dict(platform="acp", context_length=R3_WINDOW, conversation_id="refusal-test")
            engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
            cleanup.callback(engine.shutdown)
            engine.on_session_start(SID, **binding)
            live, turn_ids = [], count(1)
            stub = _stub_summarizer()

        summarizer = Mock(wraps=stub)
        monkeypatch.setattr(engine_module, "summarize_with_escalation", summarizer)
        caplog.set_level(logging.INFO, logger="hermes_lcm")
        real_mono, real_wall = engine_module.time.monotonic, engine_module.time.time
        clock_offset = 0
        monkeypatch.setattr(engine_module.time, "monotonic", lambda: real_mono() + clock_offset)
        monkeypatch.setattr(engine_module.time, "time", lambda: real_wall() + clock_offset)

        def end_turn(active):
            nonlocal clock_offset
            active.note_turn_complete()
            clock_offset += 60

        make_turn = _turn if awkward else _r3_repeated_seed_turn
        if not awkward:
            live, seed, seed_record = _r3_grow_to_leaf(
                engine, live, turn_ids, flush, end_turn, caplog, summarizer, monkeypatch, make_turn=make_turn,
            )
            live = _r3_adopt(engine, binding, live, seed, flush, seed_record)
            assert engine._verified_lcm_summary_prefix_end(live[0]["content"])
            end_turn(engine)

        live, refused, refusal = _r3_grow_to_leaf(
            engine, live, turn_ids, flush, end_turn, caplog, summarizer, monkeypatch,
            finish=awkward, make_turn=make_turn,
        )
        snapshot = engine._compress_refusal
        assert refused != live and snapshot is not None, refusal
        heads_before = _verified_user_heads(engine)
        refusal_records = _compaction_records(engine)
        _drain_session_end_maintenance(engine)
        with monkeypatch.context() as patch:
            if deferred:
                patch.setattr(engine, "_compaction_closing", True)
            engine.record_rejected_compaction()
        if deferred:
            assert engine._compress_refusal is snapshot and snapshot["refused"]
            assert _compaction_records(engine) == refusal_records
        else:
            assert engine._compress_refusal is None and _compaction_records(engine) != refusal_records
        end_turn(engine)
        if restart:
            engine = _restart(engine, binding)
            cleanup.callback(engine.shutdown)

        passes, problems = [], []
        second_refusal_turn = None
        stored_before = engine._store.get_session_messages(SID)
        expected_rows = [
            (_r3_literal_key(row, row["content"]), row["observed_at"]) for row in stored_before
        ]
        refused_sources = engine._store_complete_node_covered(
            {row["store_id"] for row in stored_before}
        )
        for turn in range(1, turns + 1):
            new_rows = flush(list(_turn(next(turn_ids))))
            expected_rows.extend((_r3_raw_key(engine, row), row["timestamp"]) for row in new_rows)
            live = live + new_rows
            before_ingest = engine._store.get_session_count(SID)
            engine.ingest(live)
            added = engine._store.get_session_count(SID) - before_ingest
            if added != len(new_rows):
                problems.append(f"turn {turn}: ingest stored {added} rows for {len(new_rows)} new rows")
            returned, record = _r3_threshold_pass(engine, live, caplog, summarizer, monkeypatch)
            reject = (
                second_refusal and second_refusal_turn is None and turn > 1 and returned != live
                and record["leaves"] > 0 and record["frontier_out"] > record["frontier_in"]
            )
            if reject:
                _drain_session_end_maintenance(engine)
                engine.record_rejected_compaction()
                second_refusal_turn = turn
                refused_sources.update(
                    sid for _, _, sid in engine._dag.get_leaf_sources_through(SID, _r3_frontier(engine))
                )
            proposed = returned if not reject and record["tokens_out"] <= record["tokens_in"] else live
            departed, merged_sources = _r4_departures(engine, live, proposed)
            if (record["unknown_coverage"] and merged_sources and record["fit_reasons"] == ["noop"]
                    and record["leaves"] == 0 and record["fit_rows"] == departed[0]):
                record["uncovered"], record["unknown_coverage"] = departed[1:]
            if not reject:
                live = _r3_adopt(engine, binding, live, returned, flush, record)
            if record["gate"] == "ran":
                passes.append(dict(
                    record, turn=turn, kept_rows=len(live), departed=departed, merged_sources=merged_sources,
                ))
            end_turn(engine)

        assert (second_refusal_turn is not None) == second_refusal
        assert _verified_user_heads(engine) == heads_before
        stored_after = engine._store.get_session_messages(SID)
        assert [
            (_r3_literal_key(row, row["content"]), row["observed_at"]) for row in stored_after
        ] == expected_rows
        leaf_sources = Counter(sid for _, _, sid in engine._dag.get_leaf_sources_through(SID, _r3_frontier(engine)))
        assert all(leaf_sources[sid] == 1 for sid in refused_sources), "a refused pass's leaf was paid for again"
        first = passes[0] if passes else {}
        if not first or not (first["leaves"] and first["frontier_out"] > first["frontier_in"]
                             and first["kept_rows"] < first["rows_in"]):
            problems.append(f"first admitted pass did not advance and shrink: {first}")
        if awkward and first and not first["merged_sources"]:
            problems.append("first admitted pass kept the merged reply from the refused pass")
        if second_refusal_turn is not None:
            after = next((row for row in passes if row["turn"] > second_refusal_turn), None)
            if not after or not (after["leaves"] and after["kept_rows"] < after["rows_in"]):
                problems.append(f"first pass after second refusal did not advance and shrink: {after}")
        for row in passes:
            if (LINEAGE_STOP in (row["stop"], row["noop"])
                    or row["uncovered"] or row["unknown_coverage"] or any(row["departed"][1:])):
                problems.append(f"turn {row['turn']}: {row}")
        assert not problems, "\n".join(problems)


def test_crash_without_shutdown_leaves_deferred_refusal_residual(committed_host, monkeypatch):  # noqa: F811
    """Without the shutdown hook, restart can still store the kept summary head again."""
    engine, kept, binding = committed_host
    _defer_rewriting_pass_refusal(engine, kept, monkeypatch)
    before = _verified_user_heads(engine)
    # Open a new engine on the same durable state, bypassing shutdown of the old one.
    resumed = LCMEngine(config=engine._config, hermes_home=str(engine._hermes_home))
    try:
        resumed.on_session_start(SID, **binding)
        resumed.ingest(kept + [_turn(20)[1]])
        added = {key: value for key, value in _verified_user_heads(resumed).items() if key not in before}
        assert list(added.values()) == [kept[0]["content"]]
    finally:
        resumed.shutdown()


def test_shutdown_closes_store_after_busy_refusal_restore(committed_host, monkeypatch, caplog):  # noqa: F811
    engine, kept, _ = committed_host
    _defer_rewriting_pass_refusal(engine, kept, monkeypatch)
    monkeypatch.setattr(engine_module.maintenance_scheduler, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 10)
    store = engine._store
    with closing(sqlite3.connect(store.db_path)) as locker:
        locker.execute("BEGIN IMMEDIATE")
        engine.shutdown()
    assert store.connection is None
    assert "deferred compaction refusal restore failed" in caplog.text


def test_shutdown_flushes_pending_tails_after_busy_refusal_restore(
    committed_host, monkeypatch, caplog,  # noqa: F811
):
    engine, kept, _ = committed_host
    _defer_rewriting_pass_refusal(engine, kept, monkeypatch)
    monkeypatch.setattr(engine_module.maintenance_scheduler, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 200)
    now = 1.0
    clock = Mock(wraps=engine_module.time)
    clock.monotonic.side_effect = lambda: now
    monkeypatch.setattr(engine_module, "time", clock)
    store = engine._store
    restore = engine._restore_refused_compaction
    expected = [("ended-1", "First pending tail"), ("ended-2", "Second pending tail")]
    with closing(sqlite3.connect(store.db_path)) as locker:
        locker.execute("BEGIN IMMEDIATE")
        for session_id, content in expected:
            engine._queue_cold_session_tail(
                [{"role": "user", "content": content}], session_id=session_id,
                conversation_id=session_id, source="acp", error=sqlite3.OperationalError("database is locked"),
            )

        def restore_while_busy(*args, **kwargs):
            nonlocal now
            (timeout_ms,) = store._conn.execute("PRAGMA busy_timeout").fetchone()
            try:
                return restore(*args, **kwargs)
            finally:
                # Account for the SQLite wait and clock granularity, then let the tails write.
                now += (timeout_ms + 2) / 1000
                locker.rollback()

        monkeypatch.setattr(engine, "_restore_refused_compaction", restore_while_busy)
        engine.shutdown()
        rows = locker.execute(
            "SELECT session_id, content FROM messages WHERE session_id IN ('ended-1', 'ended-2') ORDER BY store_id"
        ).fetchall()

    assert "deferred compaction refusal restore failed" in caplog.text
    assert rows == expected


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("case", [
    "covered", "json-join", "no-refusal", "one-uncovered", "straddles-frontier", "missing-frontier",
    "different-join", "tool-calls", "structured", "deeply-nested-left", "deeply-nested-right",
])
def test_merged_replay_requires_exact_covered_pair(tmp_path, monkeypatch, uid_mode, case):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(SID, platform="acp", conversation_id="refusal-test")
        left = {"role": "assistant", "content": " First reply. ", "store_id": 11}
        right = {"role": "assistant", "content": "Second reply.", "store_id": 12}
        successor = {"role": "user", "content": "Next request.", "store_id": 14}
        frontier = {"straddles-frontier": 11, "missing-frontier": 13}.get(case, 12)
        if case == "structured":
            left["content"] = '[{"text": "First reply.", "type": "text"}]'
        elif case == "json-join":
            left["content"], right["content"] = "[1,", "2]"
        elif case.startswith("deeply-nested-"):
            depth = max(10_000, sys.getrecursionlimit() + 1)
            nested = left if case == "deeply-nested-left" else right
            nested["content"] = "[" * depth + "0" + "]" * depth
        elif case == "tool-calls":
            right["tool_calls"] = [{"id": "synthetic-call", "type": "function"}]
        joined = left["content"].strip() + ("\n\n" if case == "different-join" else "\n") + right["content"]
        messages = [{"role": "assistant", "content": joined}, successor]
        rows = [left, right, successor]
        engine._last_compacted_store_id = frontier
        monkeypatch.setattr(engine._lifecycle, "get_by_conversation", lambda _: SimpleNamespace(
            current_session_id=SID, current_frontier_store_id=frontier,
        ))
        monkeypatch.setattr(engine._store, "get_session_rows_through", lambda _, end, limit:
                            [row for row in rows if row["store_id"] <= end][-limit:] if limit else [])
        monkeypatch.setattr(engine._store, "get_session_messages_after", lambda _, end, limit:
                            [row for row in rows if row["store_id"] > end][:limit])
        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda _: {id(successor): 14})
        monkeypatch.setattr(engine, "_latest_compaction_was_refused", lambda: case != "no-refusal")
        sources = [(3, 7, 12)]
        if case != "one-uncovered":
            sources.append((2, 5, 11))
        monkeypatch.setattr(engine._dag, "get_leaf_sources_through", lambda *_: sources)

        drops, _, tokens, kept = engine._committed_replay_drops(messages, 0)
        assert (drops, tokens, kept) == (([0], 12, 0) if case in ("covered", "json-join") else ([], 0, 0))
    finally:
        engine.shutdown()


def _tool_reply(turn, suffix=""):
    call_id = f"call-{turn}{suffix}"
    return [
        {"role": "assistant", "content": f"Looking up turn {turn}{suffix}. ", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "content": f"Result for turn {turn}{suffix}.", "tool_call_id": call_id},
    ]


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("shape,uncovered", [
    ("three-way", None), ("three-way", 0),
    ("three-way-with-exact", None), ("three-way-with-exact", 3),
    ("tool-separated", None), ("tool-separated", 1),
    ("two-tool-separated", None), ("two-tool-separated", 4),
    ("cleanup-middle", None), ("cleanup-middle", 1),
    ("ignored-marker-middle", None), ("ignored-marker-middle", 1),
], ids=[
    "three-way-covered", "three-way-uncovered-end",
    "following-exact-covered", "following-exact-uncovered",
    "tool-separated-covered", "tool-separated-uncovered-tool",
    "two-tool-separated-covered", "two-tool-separated-uncovered-tool",
    "cleanup-middle-covered", "cleanup-middle-uncovered-middle",
    "ignored-marker-middle-covered-negative", "ignored-marker-middle-uncovered-negative",
])
def test_refused_replay_requires_coverage_for_every_absorbed_row(tmp_path, monkeypatch, uid_mode, shape, uncovered):
    """I2, I11. A merged reply replays only when its entire durable run was summarised."""
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    left = {"role": "assistant", "content": " First reply. "}
    middle = {"role": "assistant", "content": " Second reply. "}
    right = {"role": "assistant", "content": "Last reply."}
    by_shape = {
        "three-way": [left, middle, right],
        "three-way-with-exact": [left, middle, right],
        "tool-separated": [*_tool_reply(1), right],
        "two-tool-separated": [*_tool_reply(1), *_tool_reply(1, "b"), right],
        "cleanup-middle": [left, {"role": "assistant", "content": "<think>Internal-only narration.</think>"}, right],
        "ignored-marker-middle": [left, {"role": "assistant", "content": "Visible narration."}, right],
    }
    run = [{**row, "store_id": 11 + i} for i, row in enumerate(by_shape[shape])]
    joined = "\n".join(
        row["content"].strip() for i, row in enumerate(run)
        if row["role"] == "assistant" and not (i == 1 and shape.endswith("-middle"))
    )
    messages = [{"role": "assistant", "content": joined}]
    store_ids = {}
    if shape == "three-way-with-exact":
        exact = {"role": "assistant", "content": "Separate exact reply."}
        messages.append(exact)
        run.append({**exact, "store_id": 14})
        store_ids[id(exact)] = 14
    frontier = run[-1]["store_id"]
    successor = {"role": "user", "content": "Next request.", "store_id": frontier + 1}
    messages.append(successor)
    store_ids[id(successor)] = successor["store_id"]
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(SID, platform="acp", conversation_id="refusal-test")
        if shape == "cleanup-middle":
            assert [row["content"] for row in engine._sanitize_active_context_messages(run)] == [joined]
        if shape == "ignored-marker-middle":
            engine._remember_generated_ignored_dependent_reply(run[1], run[1]["content"])
            # The store-scoped marker can prove EOF cleanup. Between replies
            # it does not prove that the visible narration was omitted.
            assert engine._drop_preexisting_generated_ignored_dependent_eof_replies(
                run, engine._load_generated_ignored_dependent_reply_records(),
            ) == run
        engine._last_compacted_store_id = frontier
        monkeypatch.setattr(engine._lifecycle, "get_by_conversation", lambda _: SimpleNamespace(
            current_session_id=SID, current_frontier_store_id=frontier,
        ))
        monkeypatch.setattr(engine._store, "get_session_rows_through", lambda _, end, limit:
                            [row for row in run if row["store_id"] <= end][-limit:] if limit else [])
        monkeypatch.setattr(engine._store, "get_session_messages_after", lambda *_args, **_kwargs: [successor])
        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda *_args, **_kwargs: store_ids)
        monkeypatch.setattr(engine, "_latest_compaction_was_refused", lambda: True)
        sources = [(i + 1, 3, row["store_id"]) for i, row in enumerate(run) if i != uncovered]
        monkeypatch.setattr(engine._dag, "get_leaf_sources_through", lambda *_: sources)

        drops, _, tokens, kept = engine._committed_replay_drops(messages, 0)
        if shape == "three-way-with-exact":
            remaining = [message for i, message in enumerate(messages) if i not in drops]
            if uncovered is None:
                assert (remaining, tokens, kept) == ([successor], 12, 0)
            else:
                assert (remaining, tokens, kept) == ([exact, successor], 9, 1)
        else:
            covered = uncovered is None and shape != "ignored-marker-middle"
            assert (drops, tokens, kept) == (([0], 3 * len(run), 0) if covered else ([], 0, 0))
    finally:
        engine.shutdown()


@pytest.fixture
def tool_refusal_host(tmp_path, monkeypatch, request):
    """I1, I2, I11. Seed a kept host view with one real stored tool group projected to text."""
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    config = _config(tmp_path)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    binding = dict(platform="acp", context_length=R3_WINDOW, conversation_id="refusal-test")
    try:
        engine.on_session_start(SID, **binding)
        pre = [row for i in range(1, 12) for row in _turn(i)]
        group = [*_tool_reply(12), _turn(12)[1]]
        pre.extend([_turn(12)[0], *group, _turn(13)[0]])
        left = {key: value for key, value in group[0].items() if key != "tool_calls"}
        merged = left["content"].strip() + "\n" + group[2]["content"]
        stored_projection = getattr(request, "param", "projected") == "stored-projection"
        sanitize = engine._sanitize_active_context_messages

        def project_stored_group(messages, **kwargs):
            # Exercise the host's text-only assembly shape before the engine
            # records the initial output, so the refused input stays that shape.
            if kwargs.get("generated_rows") is not None and all(row in messages for row in group):
                group_text = {row["content"] for row in group}
                stored = engine._store.get_session_messages(SID)
                raw = [row for row in stored if row["content"] in group_text]
                assert [row["role"] for row in raw] == ["assistant", "tool", "assistant"]
                assert raw[0]["tool_calls"] and all(row["content"] != merged for row in stored)
                projected, = _merge_adjacent_assistant_messages([
                    {"role": row["role"], "content": row["content"]}
                    for row in raw if row["role"] != "tool"
                ])
                start = messages.index(group[0])
                assert messages[start:start + len(group)] == group
                messages = [*messages[:start], projected, *messages[start + len(group):]]
            return sanitize(messages, **kwargs)

        with monkeypatch.context() as patch:
            if stored_projection:
                patch.setattr(engine, "_sanitize_active_context_messages", project_stored_group)
            emitted = engine.compress(pre, force=True)
        assert engine._last_compress_leaves[1] == 1
        # The kept host list carries narration after it discards the tool payload
        # and call metadata. Send that projection through the real merge funnel.
        kept = emitted
        if not stored_projection:
            assert all(row in emitted for row in group)
            kept = engine._sanitize_active_context_messages([
                left if row == group[0] else row for row in emitted if row != group[1]
            ])
        assert any(row["content"] == merged for row in kept)
        _drain_session_end_maintenance(engine)
        engine.on_session_end(SID, pre)
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, **binding)
        kept += [_turn(13)[1], *(row for i in range(14, 20) for row in _turn(i)), _turn(20)[0]]
        yield engine, kept, binding, group, merged
    finally:
        engine.shutdown()


def _emission_state(engine, handle=None):
    handle = handle or engine._emission_handle
    if handle is None:
        return None
    conn = engine._store._conn
    lineage = engine_module.ingest_ledger.persisted_lineage(conn, handle.session_id, handle.lineage)
    return conn.execute(
        "SELECT state FROM ingest_views WHERE lineage IN (?,?) AND view_seq=? AND kind='emission'",
        (lineage, handle.lineage, handle.view_seq),
    ).fetchone()


def _real_leaf_refusal(engine, kept, binding):
    engine.on_session_start(SID, **binding)
    before = _r3_frontier(engine)
    refused = engine.compress(kept, force=True)
    frontier = _r3_frontier(engine)
    assert engine._last_compress_leaves[1] == 1 and frontier > before and refused != kept
    _drain_session_end_maintenance(engine)
    return refused, frontier


@pytest.mark.parametrize("uid_mode", ["off", "on"], ids=["uid-off", "uid-on"])
def test_post_refusal_host_rows_are_append_only_until_next_compaction(request, tmp_path, monkeypatch, uid_mode):
    """I1, I2, I11. Three completed host turns keep the refused view's prefix byte-stable."""
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    home = tmp_path / "home"
    home.mkdir()
    _state_db(home, [(SID, None, None)])
    engine, kept, binding = request.getfixturevalue("committed_host")
    flush = _host_flush()
    live = flush([*kept, _turn(20)[1]])

    def snapshot(rows):
        return [json.dumps(row, sort_keys=True) for row in rows]

    def stored_rows():
        cursor = engine._store.connection.execute("SELECT * FROM messages WHERE session_id=? ORDER BY store_id", (SID,))
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor]

    def assert_bytes(actual, expected, where):
        for index, (row, before) in enumerate(zip(actual, expected)):
            if json.dumps(row, sort_keys=True) != before:
                previous = json.loads(before)
                changed = sorted(key for key in previous.keys() | row.keys()
                                 if json.dumps({key: previous[key]} if key in previous else {}, sort_keys=True)
                                 != json.dumps({key: row[key]} if key in row else {}, sort_keys=True))
                pytest.fail(f"{where}: row {index} ({previous['role']}, store_id={previous.get('store_id')}) "
                            f"changed fields {changed}")
        assert len(actual) == len(expected), f"{where}: row count {len(expected)} -> {len(actual)}"

    visible = snapshot(live)
    _, frontier = _real_leaf_refusal(engine, live, binding)
    refused_handle = engine._emission_handle
    engine.record_rejected_compaction()
    assert engine._compress_refusal is None
    assert _emission_state(engine, refused_handle) == ("refused",)
    engine.note_turn_complete()
    assert_bytes(live, visible, "refused compaction input")
    stored = snapshot(stored_rows())
    calls, commit_records = engine._compress_calls, _compaction_records(engine)

    # Capture the actual plugin callbacks from a minimal host registry.
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "registry.db"))
    hooks, registered = {}, []
    plugin = run_path(str(Path(__file__).resolve().parents[2] / "__init__.py"), run_name="hermes_lcm._post_refusal")
    with ExitStack() as cleanup:
        def register_engine(prototype):
            registered.append(prototype)
            cleanup.callback(prototype.shutdown)

        plugin["register"](SimpleNamespace(register_context_engine=register_engine, register_hook=hooks.__setitem__))
        assert len(registered) == 1
        for call in range(1, 4):
            user, reply = flush(_turn(20 + call))
            for row in (user, reply):
                row["message_uid"] = f"{row['_row_id']:032x}"
            user_bytes, reply_bytes = snapshot([user, reply])
            live.append(user)
            payload = dict(session_id=SID, conversation_id=binding["conversation_id"], platform=binding["platform"],
                           turn_id=f"append-{call}", user_message=user["content"])

            def check(phase, expected_visible):
                where = f"ordinary call {call} {phase}"
                assert_bytes(live, expected_visible, f"{where} host list")
                assert_bytes(stored_rows()[:len(stored)], stored, f"{where} stored prefix")
                assert engine._compress_calls == calls and _r3_frontier(engine) == frontier, where
                assert _compaction_records(engine) == commit_records, where
                assert _emission_state(engine, refused_handle) == ("refused",), where

            prompt = visible + [user_bytes]
            tokens = count_messages_tokens(live)
            assert not engine.should_compress(tokens), f"ordinary call {call} threshold requested compaction"
            preflight = engine.should_compress_preflight(live)
            check("preflight", prompt)
            assert not preflight, f"ordinary call {call} preflight requested compaction"
            assert hooks["pre_llm_call"](**payload, conversation_history=list(live)) is None
            assert hooks["pre_api_request"](**payload, conversation_history=list(live)) is None
            request_rows = deepcopy(live)
            if select := getattr(engine, "select_context", None):
                selected = select(request_rows, conversation_messages=deepcopy(live),
                                  incoming_message=request_rows[-1], budget_tokens=binding["context_length"])
                assert_bytes(request_rows if selected is None else selected, prompt, f"ordinary call {call} selection")
            check("request hooks", prompt)

            usage = dict(prompt_tokens=tokens, completion_tokens=count_messages_tokens([reply]))
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            engine.update_from_response(usage)
            assert not engine.should_compress_info()[0], f"ordinary call {call} response requested compaction"
            if prune := getattr(engine, "prune_tool_results_only", None):
                pruned, count_pruned = prune(live, current_tokens=tokens)
                assert_bytes(pruned, prompt, f"ordinary call {call} tool pruning")
                assert count_pruned == 0
            check("response", prompt)
            live.append(reply)
            visible = prompt + [reply_bytes]
            assert hooks["post_llm_call"](**payload, conversation_history=list(live), context_compressor=engine) is None
            check("post_llm_call ingest", visible)
            engine.on_turn_complete(deepcopy(live), usage=usage, turn_id=payload["turn_id"], interrupted=False)
            check("on_turn_complete", visible)
            durable = stored_rows()
            assert [(row["role"], row["content"]) for row in durable[len(stored):]] == [
                (user["role"], user["content"]), (reply["role"], reply["content"]),
            ], f"ordinary call {call} appended different stored rows"
            stored = snapshot(durable)

    returned = engine.compress(live, force=True)
    assert_bytes(live, visible, "next compaction input")
    assert_bytes(stored_rows(), stored, "next compaction stored rows")
    assert snapshot(returned) != visible and len(returned) < len(live)
    assert engine._compress_calls == calls + 1


def test_tool_separated_refusal_recovers_with_covered_survival_fit(tool_refusal_host, monkeypatch, caplog):
    """I1, I2, I11. A refused leaf already covering a merged tool reply lets the next pass advance."""
    engine, kept, binding, group, merged = tool_refusal_host
    _, frontier = _real_leaf_refusal(engine, kept, binding)
    engine.record_rejected_compaction()
    assert _emission_state(engine) == ("refused",)
    engine.note_turn_complete()
    live = kept + [_turn(20)[1]]
    engine.ingest(live)
    sources = {sid for _, _, sid in engine._dag.get_leaf_sources_through(SID, frontier)}
    group_ids = {row["store_id"] for row in engine._store.get_session_messages(SID)
                 if row["content"] in {msg["content"] for msg in group}}
    assert len(group_ids) == len(group) and group_ids <= sources
    before = engine._store.get_session_messages(SID)
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    summarizer = Mock(wraps=engine_module.summarize_with_escalation)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarizer)
    replay = engine._committed_replay_drops
    replayed = []

    def observe_replay(working, *args, **kwargs):
        result = replay(working, *args, **kwargs)
        replayed.extend(working[position]["content"] for position in result[0])
        return result

    monkeypatch.setattr(engine, "_committed_replay_drops", observe_replay)
    returned, record = _r3_threshold_pass(
        engine, live, caplog, summarizer, monkeypatch, overhead=engine.threshold_tokens - count_messages_tokens(live),
    )
    assert record["gate"] == "ran" and LINEAGE_STOP not in (record["stop"], record["noop"]), record
    assert record["leaves"] and _r3_frontier(engine) > frontier and len(returned) < len(live), record
    assert record["uncovered"] == record["unknown_coverage"] == 0, record
    assert all(row["content"] != merged for row in returned)
    assert engine._store.get_session_messages(SID) == before
    assert merged in replayed, record


@pytest.mark.parametrize("tool_refusal_host", ["stored-projection"], indirect=True)
def test_refused_tool_reply_reconstructs_original_stored_rows(tool_refusal_host, monkeypatch, caplog):
    """I1, I2, I11. A refused leaf can cover a host merge that has no exact stored row."""
    engine, kept, binding, group, merged = tool_refusal_host
    _, frontier = _real_leaf_refusal(engine, kept, binding)
    engine.record_rejected_compaction()
    assert _emission_state(engine) == ("refused",)
    engine.note_turn_complete()

    stored = engine._store.get_session_messages(SID)
    group_text = {row["content"] for row in group}
    stored_group = [row for row in stored if row["content"] in group_text]
    assert [row["role"] for row in stored_group] == ["assistant", "tool", "assistant"]
    assert stored_group[0]["tool_calls"] and all(row["content"] != merged for row in stored)
    covered = {store_id for _, _, store_id in engine._dag.get_leaf_sources_through(SID, frontier)}
    assert {row["store_id"] for row in stored_group} <= covered
    live = kept + [_turn(20)[1]]
    replay = engine._committed_replay_drops
    replayed = []

    def observe_replay(working, start, **kwargs):
        result = replay(working, start, **kwargs)
        replayed.extend(working[position]["content"] for position in result[0])
        return result

    monkeypatch.setattr(engine, "_committed_replay_drops", observe_replay)
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    summarizer = Mock(wraps=engine_module.summarize_with_escalation)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarizer)
    returned, record = _r3_threshold_pass(
        engine, live, caplog, summarizer, monkeypatch, overhead=engine.threshold_tokens - count_messages_tokens(live),
    )
    assert LINEAGE_STOP not in (record["stop"], record["noop"]), record
    assert record["leaves"] and _r3_frontier(engine) > frontier and len(returned) < len(live), record
    assert record["uncovered"] == record["unknown_coverage"] == 0, record
    assert merged in replayed and all(row["content"] != merged for row in returned)
    after = engine._store.get_session_messages(SID)
    assert after[:-1] == stored and after[-1]["content"] == live[-1]["content"]


@pytest.mark.parametrize("adopted", [False, True], ids=["retry-before-gate", "already-adopted"])
def test_refused_emission_write_retries_original_view(committed_host, monkeypatch, adopted):  # noqa: F811
    """I1, I2, I11. A skipped refusal follows the original proposal, even across compress calls."""
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    engine, kept, binding = committed_host
    _, frontier = _real_leaf_refusal(engine, kept, binding)
    handle = engine._emission_handle
    assert _emission_state(engine, handle) == ("proposed",)

    @contextmanager
    def unavailable(_salt):
        yield None, "in_transaction"

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_emission_owner", unavailable)
        engine.record_rejected_compaction()
    assert _emission_state(engine, handle) == ("proposed",)
    if adopted:
        with engine._emission_owner(engine._store.ingest_ledger_salt()) as (conn, _):
            moved, _ = engine_module.ingest_ledger.set_emission_state(conn, handle.lineage, handle.view_seq, "adopted")
        assert moved > 0
    engine.note_turn_complete()

    gate_outcomes = []
    gate = engine._latest_compaction_was_refused

    def observe_gate():
        result = gate()
        gate_outcomes.append((result, _emission_state(engine, handle)))
        return result

    monkeypatch.setattr(engine, "_latest_compaction_was_refused", observe_gate)
    # Leave compactable input beyond the fresh tail after the retry drops the covered prefix.
    live = kept + [_turn(20)[1], *(row for i in range(21, 28) for row in _turn(i))]
    returned = engine.compress(live, force=True)
    assert engine._compress_calls > handle.compress_call and gate_outcomes
    assert gate_outcomes[0] == ((False, ("adopted",)) if adopted else (True, ("refused",)))
    if not adopted:
        assert _r3_frontier(engine) > frontier and len(returned) < len(live)


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("next_session", [SID, "S1"], ids=["same-session", "session-switch"])
def test_refused_emission_retry_respects_session_binding(request, monkeypatch, uid_mode, next_session):
    """I1, I2, I11. An ordinary bind preserves a retry only for its own session."""
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    engine, kept, binding = request.getfixturevalue("committed_host")
    _real_leaf_refusal(engine, kept, binding)
    handle = engine._emission_handle
    assert _emission_state(engine, handle) == ("proposed",)

    @contextmanager
    def unavailable(_salt):
        yield None, "in_transaction"

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_emission_owner", unavailable)
        engine.record_rejected_compaction()
    assert _emission_state(engine, handle) == ("proposed",)
    assert (handle, engine._emission_handle_salt) in engine._pending_refused_emissions
    engine.note_turn_complete()

    write = Mock(wraps=engine._write_emission_state)
    monkeypatch.setattr(engine, "_write_emission_state", write)
    engine.on_session_start(next_session, **binding)
    assert (engine._session_id, engine._conversation_id) == (next_session, binding["conversation_id"])
    if next_session == SID:
        gate = Mock(wraps=engine._latest_compaction_was_refused)
        monkeypatch.setattr(engine, "_latest_compaction_was_refused", gate)
        drops, _, _, _ = engine._committed_replay_drops(kept, 0)
        assert _emission_state(engine, handle) == ("refused",)
        gate.assert_called_once_with()
        assert drops
        write.assert_called_once_with(handle, engine._emission_handle_salt, "refused")
    else:
        fresh = [row for i in range(30, 44) for row in _turn(i)]
        assert engine.compress(fresh, force=True) != fresh
        new_handle = engine._emission_handle
        assert new_handle is not None and new_handle.session_id == next_session
        assert _emission_state(engine, new_handle) == ("proposed",)
        assert not engine._latest_compaction_was_refused()
        assert _emission_state(engine, new_handle) == _emission_state(engine, handle) == ("proposed",)
        write.assert_not_called()
    assert not engine._pending_refused_emissions


@pytest.mark.parametrize("case", ["later-accepted", "session-switch", "ledger-off"])
def test_real_refusal_gate_boundaries(request, monkeypatch, caplog, case):
    """I1, I2, I11. The replay gate is scoped to a genuine host refusal and the bound frontier."""
    monkeypatch.setattr(rollup_builder, "summarize_with_escalation", _stub_summarizer())
    if case == "ledger-off":
        monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine, kept, binding = request.getfixturevalue("committed_host")
    _, frontier = _real_leaf_refusal(engine, kept, binding)
    handle = engine._emission_handle
    engine.record_rejected_compaction()
    if case == "ledger-off":
        assert handle is None and not engine._latest_compaction_was_refused()
    else:
        assert _emission_state(engine, handle) == ("refused",)
        assert engine._latest_compaction_was_refused()
        assert engine._committed_replay_drops(kept, 0)[0]
    engine.note_turn_complete()
    if case == "later-accepted":
        # Route backoff permits a no-leaf replay to re-emit the summaries.
        # Carry out the later host commit in its normal end/start order.
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_summary_route_stop_applies", lambda *_args, **_kwargs: True)
            # This synthetic route refusal has no real circuit cooldown.
            patch.setattr(engine, "_start_sweep_budget_hold", lambda *_args, **_kwargs: None)
            accepted = engine.compress(kept, force=True)
        assert _r3_frontier(engine) == frontier and engine._last_compress_leaves[1] == 0
        assert len(accepted) < len(kept) and engine._emission_handle.view_seq > handle.view_seq
        _drain_session_end_maintenance(engine)
        engine.on_session_end(SID, kept)
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, **binding)
        assert _emission_state(engine) == ("adopted",)
        assert _emission_state(engine, handle) == ("refused",)
        assert not engine._latest_compaction_was_refused()
        covered = {store_id for _, _, store_id in engine._dag.get_leaf_sources_through(SID, frontier)}
        ids = engine._get_store_id_map_for_messages(accepted)
        scaffold_end, carrier = engine._replay_head(accepted, 0, frontier)
        for i, row in enumerate(accepted):
            if i < scaffold_end or (i == scaffold_end and carrier):
                continue
            store_id = ids.get(id(row))
            assert store_id and (store_id > frontier or store_id in covered), row
        assert engine._committed_replay_drops(accepted, 0)[0] == []
        caplog.set_level(logging.INFO, logger="hermes_lcm")
        summarizer = Mock(wraps=engine_module.summarize_with_escalation)
        monkeypatch.setattr(engine_module, "summarize_with_escalation", summarizer)
        live = accepted + [_turn(20)[1], *_turn(21)]
        _, record = _r3_threshold_pass(
            engine, live, caplog, summarizer, monkeypatch, overhead=engine.threshold_tokens - count_messages_tokens(live),
        )
        assert record["gate"] == "ran" and LINEAGE_STOP not in (record["stop"], record["noop"]), record
        return
    elif case == "session-switch":
        engine.on_session_start("S1", **binding)
        assert engine._lifecycle.get_by_conversation(engine._conversation_id).current_session_id != SID

    drops, store_map, _, _ = engine._committed_replay_drops(kept, 0)
    assert drops == []
    assert (store_map is None) == (case == "session-switch")
