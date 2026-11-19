"""#2815: no-leaf replay preserves the cached head and returns valid tool pairs."""

import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_lcm import host_uid_emit, store
from hermes_lcm.engine import _model_only_display_metadata_key, _model_only_display_metadata_keys
from hermes_lcm.store_complete import _UID_DROP_INPUT
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_c8 import (
    _alter_guard_input,
    _output_sources,
    _seed,
    _set_frontier,
    _summary_head,
)
from tests.invariants.test_issue_2815_uid_restart import _view
from tests.test_issue_1201_summary_model_only import _host
from tests.test_issue_581_582_store_complete_survival import _rows, summaries as summaries
from tests.test_issue_671_stub_first_exit import tool_pair

pytestmark = pytest.mark.issue(2815)
SHAPES = ["plain", "carrier", "system-plain"]


@pytest.fixture(autouse=True)
def host_uid(monkeypatch):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)


def _mode(monkeypatch, mode):
    if mode == "unset":
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)


def _engine(path):
    path.mkdir()
    engine = _seed(path, leaf_chunk_tokens=50_000, dynamic_leaf_chunk_enabled=False, survival_fit=False,
                   embeddings_enabled=False, temporal_rollups_enabled=False)
    engine.threshold_tokens = 8_000  # Keep turn-paced holds out of the three-pass comparison.
    return engine


def _unanswered():
    call, _ = tool_pair("c15-unanswered", "unused")
    return [
        {"role": "user", "content": "Run the synthetic lookup.", "message_uid": "lookup-user", "timestamp": 2000.0},
        dict(call, content="", message_uid="lookup-call", timestamp=2001.0),
    ]


def _reassembly_input(engine, monkeypatch, shape, *, resumed=True):
    rows = _view(7)
    covered = 8 if shape == "carrier" else 9
    engine.ingest(rows)
    ids = [row["store_id"] for row in _rows(engine)]
    assert len(ids) == len(rows)
    _set_frontier(engine, ids[:covered])
    head = dict(_summary_head(engine), message_uid="engine-head", display_metadata={"host_model_only": True})
    engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0],
                                        [(0, "engine-head", "engine", "summary")])
    if shape == "carrier":
        head.update(content=head["content"] + "\n\n" + rows[0]["content"], _absorbed_message_uids=["uid-0"])
        view = [head, *rows[1:], *_unanswered()]
    else:
        view = [head, *rows, *_unanswered()]

    replay = engine._committed_replay_drops

    def committed(working, start, *, uid_first=True):
        if working[0].get("message_uid") == "engine-head" and (not uid_first or not engine._uid_recognition_enabled()):
            end = covered if shape == "carrier" else covered + 1
            return list(range(1, end)) if resumed else [], None, 0, int(shape == "carrier")
        return replay(working, start, uid_first=uid_first)

    monkeypatch.setattr(engine, "_committed_replay_drops", committed)
    return view, ids


def _compress(engine, view, *, on):
    recognition = []
    compress_impl = engine._compress_impl

    def observe(messages, *args, **kwargs):
        recognition.append(engine._uid_recognition_enabled() and any(row.get("message_uid") for row in messages))
        return compress_impl(messages, *args, **kwargs)

    with patch.object(engine, "_compress_impl", observe):
        output = engine.compress(view, current_tokens=count_messages_tokens(view))
    assert recognition == [on]
    assert engine._last_compress_leaves[1] == 0
    return output


def test_carrier_head_reassembles_with_stub(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _closing(_engine(tmp_path / "on")) as engine:
        view, _ = _reassembly_input(engine, monkeypatch, "carrier")
        remainder = engine._generated_context_carrier_remainder(view[0])
        digest = hashlib.sha256(remainder.encode()).hexdigest()
        guard = engine._guard_uid_covered_drop
        assembled = []

        def observe_guard(result):
            guarded = guard(result)
            assert guarded is result
            assembled.append(result)
            return guarded

        with patch.object(engine, "_guard_uid_covered_drop", observe_guard):
            output = _compress(engine, view, on=True)
        assert engine._missing_tool_result_stub("c15-unanswered") in output
        texts = [engine._generated_context_carrier_remainder(row) or row["content"] for row in output]
        assert sum(hashlib.sha256(content.encode()).hexdigest() == digest for content in texts) == 1
        assert all(engine._generated_context_carrier_remainder({"role": "user", "content": text}) is None
                   for text in texts)
        assert assembled == [output]
        assert engine._ingest_cursor == len(output)
        assert not summaries


def _cached_input(engine, shape):
    rows = _view(7)
    system = {"role": "system", "content": "synthetic system prompt", "timestamp": 900.0}
    raw = [system, *rows] if shape == "system-plain" else rows
    covered = {"plain": 5, "carrier": 8, "system-plain": 9}[shape]
    engine.ingest(raw)
    _set_frontier(engine, [row["store_id"] for row in _rows(engine)][:covered])
    view = engine._assemble_context(system if shape == "system-plain" else None, raw[covered:])
    head = view[int(shape == "system-plain")]
    assert bool(engine._generated_context_carrier_remainder(head)) == (shape == "carrier")
    assert head.get("message_uid")
    engine._ingest_cursor = len(view)
    _, emissions, reason = engine._record_compress_commit_proof(raw, view)
    assert emissions and reason is None
    return view


def _observe_replay_exit(engine, monkeypatch):
    entries, kept = [], []
    finish = engine._finish_without_leaf
    covers = engine._replayed_head_covers_frontier

    def capture(state, *args, **kwargs):
        scope = _UID_DROP_INPUT.get()
        entry = dict(trimmed=bool(scope and scope.uid_trimmed), legacy=scope.legacy_replay if scope else None,
                     working=state.working_messages, dropped=state.dropped_replayed_scaffold_messages,
                     resumed=state.resumed_prefix, ahead=state.resumed_ahead)
        result = finish(state, *args, **kwargs)
        entry["adopted"] = bool(entry["legacy"] and state.working_messages is entry["legacy"][0])
        entries.append(entry)
        return result

    def capture_keep(state):
        result = covers(state)
        kept.append(result)
        return result

    monkeypatch.setattr(engine, "_finish_without_leaf", capture)
    monkeypatch.setattr(engine, "_replayed_head_covers_frontier", capture_keep)
    return entries, kept


def _assert_keep_branch(shape, entry, kept, decisions):
    assert entry["adopted"], entry
    if shape == "carrier":
        assert not entry["trimmed"] and not decisions
    else:
        assert entry["trimmed"] and len(decisions) == 1 and decisions[0] is entry["legacy"]
        assert kept == [True]


def _keep_fixture(engine, view, shape, monkeypatch):
    if shape != "carrier":
        return
    # The carrier itself is retained by the legacy replay decision. Exercise a
    # UID-first replay that also considers its position while it has no raw ID.
    committed = engine._committed_replay_drops

    def replay(working, start, *, uid_first=True):
        if working[0].get("message_uid") == view[0]["message_uid"]:
            if uid_first and engine._uid_recognition_enabled():
                return [0], {}, 0, 0
            return [], {}, 0, 0
        return committed(working, start, uid_first=uid_first)

    monkeypatch.setattr(engine, "_committed_replay_drops", replay)


def _head_bytes(view, shape):
    return json.dumps(view[int(shape == "system-plain")], ensure_ascii=False).encode()


def _pin_uid_lineage(engine, monkeypatch):
    # Engine UIDs otherwise incorporate the path of each isolated state.db.
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda *args, **kwargs: ("c15-synthetic:S", None))


def _observe_helper(engine):
    replay = getattr(engine, "_uid_trimmed_base_replay", None)
    return patch.object(engine, "_uid_trimmed_base_replay", wraps=replay, create=True)


@pytest.mark.parametrize("shape", SHAPES)
def test_keep_head_without_assembly(tmp_path, monkeypatch, summaries, shape):
    _mode(monkeypatch, "on")
    with _closing(_engine(tmp_path / "on")) as engine:
        view = _cached_input(engine, shape)
        _keep_fixture(engine, view, shape, monkeypatch)
        head = _head_bytes(view, shape)
        entries, kept = _observe_replay_exit(engine, monkeypatch)
        with _observe_helper(engine) as helper, patch.object(
            engine, "_assemble_context", wraps=engine._assemble_context,
        ) as assemble, patch.object(engine, "_guard_uid_covered_drop", wraps=engine._guard_uid_covered_drop) as guard:
            output = _compress(engine, view, on=True)

        assert output is view
        assert assemble.call_count == guard.call_count == 0
        assert _head_bytes(output, shape) == head
        assert engine._last_compression_status == "noop"
        assert engine._ingest_cursor == len(output)
        [entry] = entries
        decisions = [entry["legacy"]] if helper.call_count == 1 else []
        _assert_keep_branch(shape, entry, kept, decisions)
        print(f"B2 {shape}: trimmed={entry['trimmed']} legacy_adopted={entry['adopted']} keep={kept}")
        assert not summaries


@pytest.mark.parametrize("shape", SHAPES)
def test_three_pass_head_stability(tmp_path, monkeypatch, summaries, shape):
    paired = {}
    for mode in ("unset", "on"):
        _mode(monkeypatch, mode)
        with _closing(_engine(tmp_path / mode)) as engine:
            _pin_uid_lineage(engine, monkeypatch)
            output = _cached_input(engine, shape)
            _keep_fixture(engine, output, shape, monkeypatch)
            entries, kept = _observe_replay_exit(engine, monkeypatch)
            snapshots, branches = [], []
            with _observe_helper(engine) as helper, patch.object(
                engine, "_assemble_context", wraps=engine._assemble_context,
            ) as assemble:
                for index in range(3):
                    view = [*output, {
                        "role": "user", "content": f"Synthetic new user {index}.",
                        "message_uid": f"new-user-{index}", "timestamp": 2100.0 + index,
                    }]
                    old_entries, old_kept, old_decisions = len(entries), len(kept), helper.call_count
                    output = _compress(engine, view, on=mode == "on")
                    head_uid = output[int(shape == "system-plain")].get("message_uid")
                    snapshots.append((output is view, _head_bytes(output, shape), head_uid, engine._ingest_cursor,
                                      engine._last_compression_status))
                    [entry] = entries[old_entries:]
                    branches.append((entry, kept[old_kept:], helper.call_count - old_decisions))
                    engine.note_turn_complete()
            assert assemble.call_count == 0
            paired[mode] = snapshots, branches
            print(f"B3 {shape} {mode}: "
                  f"{[(entry['trimmed'], entry['adopted'], keep, calls) for entry, keep, calls in branches]}")

    assert paired["on"][0] == paired["unset"][0]
    for entry, kept, calls in paired["on"][1]:
        decisions = [entry["legacy"]] if calls == 1 else []
        _assert_keep_branch(shape, entry, kept, decisions)
    assert not summaries


@pytest.mark.parametrize("shape", ["plain", "carrier"])
def test_guard_restores_before_with_tool_stub(tmp_path, monkeypatch, summaries, shape):
    _mode(monkeypatch, "on")
    with _closing(_engine(tmp_path / "on")) as engine:
        view, _ = _reassembly_input(engine, monkeypatch, shape)
        seen = []

        def without_header(result, _patch):
            scope = _UID_DROP_INPUT.get()
            emitted = {row.get("message_uid") for row in result}
            checked_missing = [sid for row in scope.before
                               if (record := scope.removed.get(id(row))) and record[1]
                               and row.get("message_uid") not in emitted and (sid := record[0])]
            assert checked_missing and set(checked_missing) <= _output_sources(engine, result)
            altered = [row for row in result if not _output_sources(engine, [row])]
            assert not _output_sources(engine, altered)
            seen.append(scope.before)
            return altered

        _alter_guard_input(engine, monkeypatch, without_header)
        output = _compress(engine, view, on=True)
        assert len(seen) == 1, "no-leaf assembly did not call the UID guard"
        assert output == engine._sanitize_active_context_messages(seen[0], insert_missing_tool_stubs=True)
        assert engine._missing_tool_result_stub("c15-unanswered") in output
        assert engine._compress_host_rows_consumed == 0
        assert engine._ingest_cursor == len(output)
        assert engine._last_compression_status == "sanitized"
        assert not summaries


@pytest.mark.parametrize(("shape", "resumed"), [
    pytest.param("plain", False, id="plain"),
    pytest.param("carrier", True, id="carrier"),
    pytest.param("plain", True, id="resumed"),
])
def test_helper_matches_unset_replay_on_entry_to_no_leaf_exit(tmp_path, monkeypatch, summaries, shape, resumed):
    entries, progress = {}, {}
    for mode in ("unset", "on"):
        _mode(monkeypatch, mode)
        with _closing(_engine(tmp_path / mode)) as engine:
            view, _ = _reassembly_input(engine, monkeypatch, shape, resumed=resumed)
            captured, _ = _observe_replay_exit(engine, monkeypatch)
            _compress(engine, view, on=mode == "on")
            [entries[mode]] = captured
            progress[mode] = engine._no_progress_proof_complete
            if mode == "unset" and shape == "carrier":
                remainder = engine._generated_context_carrier_remainder(view[0])
                assert entries[mode]["working"][0]["content"] == view[0]["content"]
                entries[mode]["working"] = [
                    {"role": "user", "content": remainder, "message_uid": view[0]["_absorbed_message_uids"][0]},
                    *entries[mode]["working"][1:],
                ]

    base = entries["unset"]
    working, dropped, resumed_prefix, ahead = entries["on"]["legacy"]
    assert [row["content"] for row in working] == [row["content"] for row in base["working"]]
    if shape == "carrier":
        assert working[0] == base["working"][0]
    assert (dropped, resumed_prefix, ahead) == (base["dropped"], base["resumed"], base["ahead"])
    print(f"B5 {shape} resumed={resumed}: no-progress-proof-complete={progress}")
    assert not summaries


@pytest.mark.parametrize("mode", ["unset", "shadow"])
@pytest.mark.parametrize("shape", ["plain", "carrier"])
def test_mode_controls(tmp_path, monkeypatch, summaries, mode, shape):
    _mode(monkeypatch, mode)
    # Store rows contain ingestion time even when each host timestamp is fixed.
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: 3000.0, perf_counter=time.perf_counter))
    with _closing(_engine(tmp_path / mode)) as engine:
        _pin_uid_lineage(engine, monkeypatch)
        view, _ = _reassembly_input(engine, monkeypatch, shape)
        with _observe_helper(engine) as helper, patch.object(
            engine, "_guard_uid_covered_drop", wraps=engine._guard_uid_covered_drop,
        ) as guard:
            output = _compress(engine, view, on=False)
        assert helper.call_count == guard.call_count == 0
        payload = [output, _rows(engine), engine._ingest_cursor, engine._last_compression_status,
                   engine._last_compression_noop_reason]
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if receipt := os.environ.get("C15_CONTROL_RECEIPT"):
            path = Path(receipt)
            record = json.loads(path.read_text()) if path.exists() else {}
            record[f"{mode}-{shape}"] = digest
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def display_host(monkeypatch):
    _host(monkeypatch, "host_model_only")
    yield
    _model_only_display_metadata_key.cache_clear()


@pytest.mark.parametrize("shape", ["plain", "carrier"])
@pytest.mark.parametrize("project", [False, True], ids=["unprojected", "projected"])
def test_reassembled_head_display_keys(tmp_path, monkeypatch, summaries, display_host, shape, project):
    keys = []
    for mode in ("unset", "on"):
        _mode(monkeypatch, mode)
        with _closing(_engine(tmp_path / mode)) as engine:
            view, _ = _reassembly_input(engine, monkeypatch, shape)
            if project:
                view = deepcopy(view)
                for row in view:
                    if row["role"] == "user" and (metadata := row.get("display_metadata")):
                        for key in _model_only_display_metadata_keys():
                            metadata.pop(key, None)
                        if not metadata:
                            del row["display_metadata"]
                    if row["role"] in {"assistant", "tool"}:
                        row.pop("timestamp", None)
            with patch.object(engine, "_assemble_context", wraps=engine._assemble_context) as assemble:
                output = _compress(engine, view, on=mode == "on")
            assert assemble.call_count == 1
            assert output[0]["content"].startswith("[Recent Summary")
            assert engine._missing_tool_result_stub("c15-unanswered") in output
            assert all(row.get("message_uid") != "uid-1" for row in output)
            keys.append(set(output[0].get("display_metadata", {})))
    print(f"B7 {shape} projected={project}: unset={sorted(keys[0])} on={sorted(keys[1])}")
    assert keys[1] == keys[0]
    assert not summaries
