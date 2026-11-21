"""Issues #3001 and #3006: refused-pass replay and clean restart (I1, I2, I11)."""

import json
import logging
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm import rollup_builder
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_messages_tokens
from tests.test_compression_boundary import _config, _stub_summarizer, _turn
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


def _r3_threshold_pass(engine, live, caplog, summarizer, monkeypatch):
    estimate = count_messages_tokens(live)
    prompt_tokens = estimate + R3_OVERHEAD
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
            reject = second_refusal and second_refusal_turn is None and turn > 1 and returned != live
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


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("case", [
    "covered", "no-refusal", "one-uncovered", "straddles-frontier", "missing-frontier",
    "different-join", "tool-calls", "structured",
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
        if case == "tool-calls":
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
        assert (drops, tokens, kept) == (([0], 12, 0) if case == "covered" else ([], 0, 0))
    finally:
        engine.shutdown()
