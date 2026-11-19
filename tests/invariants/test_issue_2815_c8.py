"""#2815: only rows removed as already covered can trigger the covered-drop restore."""

import logging
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hermes_lcm import externalize, host_uid_emit
from hermes_lcm.engine import _summary_part_text
from hermes_lcm.reconcile import _has_lossy_redacted_identity
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_uid_restart import _open, _view
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows, _stage, _turn, summaries  # noqa: F401
from tests.test_issue_671_stub_first_exit import tool_view

CALL_INDEX = 5


@pytest.fixture(autouse=True)
def uid_on(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", False)


def _seed(path, **settings):
    _state_db(path, [("S", None, None)])
    return _open(path, sensitive_patterns_enabled=True, **settings)


def _set_frontier(engine, ids):
    _stage(engine, 0, ids)
    engine._last_compacted_store_id = ids[-1]


def _lossy_view(engine, *, replay=False):
    raw = _view(10)
    call = raw[CALL_INDEX]
    call["_tool_call_uids"] = ["synthetic-tool-call-uid"]
    call["tool_calls"][0]["function"]["arguments"] = '{"password":"synthetic-password"}'
    engine.ingest(raw)
    ids = [row["store_id"] for row in _rows(engine)]
    assert len(ids) == len(raw)
    _set_frontier(engine, ids[:8 if replay else 4])
    view = engine._redact_active_replay_messages(raw)
    call = view[CALL_INDEX]
    assert call["content"] == "" and call["tool_calls"]
    assert _has_lossy_redacted_identity(engine._message_replay_identity(call, strip_carrier=False))
    assert CALL_INDEX not in engine._host_uid_resolve(view)
    if replay:
        # Keep the second turn intact until replay admission. Its tool pair is
        # already indivisible because the assistant call cannot resolve.
        for i in (4, 7):
            view[i].pop("message_uid")
        return view, ids

    twin = {key: deepcopy(call[key]) for key in ("role", "content", "tool_calls")}
    twin_id = engine._store.append("S", twin, conversation_id="conv")
    return view, ids, twin_id


def _summary_head(engine):
    return {"role": "user", "content": _summary_part_text(engine._dag.get_node(1))}


def _output_sources(engine, output):
    return {
        store_id
        for row in output
        if row.get("role") in {"user", "assistant"} and isinstance(content := row.get("content"), str)
        for header in engine._LCM_SUMMARY_PART_HEADER_RE.finditer(content)
        for node_id in engine._verified_lcm_summary_prefix(content[header.start():])[1]
        for store_id in engine._dag.source_message_ids(node_id, limit=2**63 - 1)
    }


def _guard_counts(caplog):
    records = [record for record in caplog.records
               if record.name == "hermes_lcm.store_complete" and record.msg.startswith("LCM UID covered drop:")]
    assert len(records) == 1
    return records[0].args


def test_lossy_leaf_row_does_not_block_shrink(tmp_path, caplog, summaries):  # noqa: F811
    with _closing(_seed(tmp_path)) as engine, caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
        view, ids, twin = _lossy_view(engine)
        output = engine.compress(view, current_tokens=24000)
        nodes = engine._dag.get_session_nodes("S")
        assert len(summaries) == 1 and len(nodes) == 2 and ids[CALL_INDEX] in nodes[-1].source_ids
        assert len(output) < len(view)
        emitted = {row.get("message_uid") for row in output}
        removed = {sid for row, sid in zip(view, ids) if row["message_uid"] not in emitted}
        assert set(ids[:4]) | set(nodes[-1].source_ids) <= removed
        covered = _output_sources(engine, output)
        assert removed <= covered and twin not in covered
        assert _guard_counts(caplog) == (4, 0, len(removed) - 4, 1)


@pytest.mark.parametrize("shape", ["A", "B", "C"])
def test_carrier_does_not_block_shrink(tmp_path, monkeypatch, summaries, shape):  # noqa: F811
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", shape != "A")
    with _closing(_seed(tmp_path)) as engine:
        rows = _view(7)
        engine.ingest(rows[:8])
        ids = [row["store_id"] for row in _rows(engine)]
        for i in range(12):
            for row in _turn(f"hidden-{i}", 500.0 + i):
                engine._store.append("S", row, conversation_id="conv")
        engine.ingest(rows)
        _set_frontier(engine, ids)
        head = dict(_summary_head(engine), message_uid="engine-head")
        engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0],
                                            [(0, "engine-head", "engine", "summary")])
        view = [head, *rows]
        if shape == "B":
            view[9] = dict(view[9], _absorbed_message_uids=["uid-host-merged"])
        elif shape == "C":
            head.update(content=head["content"] + "\n\n" + rows[0]["content"], _absorbed_message_uids=["uid-0"])
            view = [head, *rows[1:]]
        covered_uids = {row["message_uid"] for row in rows[1 if shape == "C" else 0:8]}
        output = engine.compress(view, current_tokens=24000)
        assert engine._last_compress_leaves[1] == 1
        assert engine._compress_hidden_rows_consumed > 0
        assert len(output) < len(view)
        assert not (covered_uids & {row.get("message_uid") for row in output})
        assert engine._generated_context_carrier_remainder(output[0]) == rows[8]["content"]
        assert len(output[0].get("_absorbed_message_uids") or ()) == {"A": 0, "B": 2, "C": 1}[shape]
        assert set(ids) <= _output_sources(engine, output)


def _alter_guard_input(engine, monkeypatch, alter):
    guard = engine._guard_uid_covered_drop

    def check(result):
        with monkeypatch.context() as patch:
            return guard(alter(result, patch))

    monkeypatch.setattr(engine, "_guard_uid_covered_drop", check)


@pytest.mark.parametrize("fault", ["missing-header", "changed-binding"])
def test_uid_drop_with_failed_proof_restores(tmp_path, monkeypatch, caplog, summaries, fault):  # noqa: F811
    with _closing(_seed(tmp_path)) as engine, caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
        view = _view(7)
        engine.ingest(view)
        ids = [row["store_id"] for row in _rows(engine)]
        _set_frontier(engine, ids[:1])
        uid = view[0]["message_uid"]
        resolve = engine._host_uid_resolve

        def alter(result, patch):
            assert not any(row.get("message_uid") == uid for row in result)
            assert ids[0] in _output_sources(engine, result)
            if fault == "missing-header":
                without_header = [row for row in result if ids[0] not in _output_sources(engine, [row])]
                assert ids[0] not in _output_sources(engine, without_header)
                return without_header

            def shifted(rows, **kwargs):
                matches = resolve(rows, **kwargs)
                for i, match in matches.items():
                    if rows[i].get("message_uid") == uid:
                        # Both IDs are covered. Restore only if the original binding is checked.
                        assert match["store_id"] == ids[0] and ids[1] in _output_sources(engine, result)
                        matches[i] = {**match, "store_id": ids[1]}
                return matches

            patch.setattr(engine, "_host_uid_resolve", shifted)
            return result

        _alter_guard_input(engine, monkeypatch, alter)
        output = engine.compress(view, current_tokens=24000)
        assert engine._last_compress_leaves[1] == 1
        assert output == engine._sanitize_active_context_messages(view, insert_missing_tool_stubs=False)
        assert engine._compress_host_rows_consumed == 0 and engine._ingest_cursor == len(output)
        assert _guard_counts(caplog)[:2] == (1, 1)


def _replay_with_ids(engine, monkeypatch, replay_ids, *, base_adopts=True):
    replay = engine._committed_replay_drops
    observed = []

    def record(working, start, **kwargs):
        drops, ids, tokens, kept = replay(working, start, **kwargs)
        if kwargs.get("uid_first") is False:
            return (drops, ids, tokens, kept) if base_adopts else ([], ids, 0, 0)
        if drops:
            assert len(drops) == len(replay_ids) and not kept
            observed.append(tuple(working[i].get("role") for i in drops))
            # Supply the replay map at the caller seam. The durable replay proof
            # above has still compared the actual rows and the frontier successor.
            ids = {**ids, **{id(working[i]): sid for i, sid in zip(drops, replay_ids)}}
        return drops, ids, tokens, kept

    monkeypatch.setattr(engine, "_committed_replay_drops", record)
    return observed


@pytest.mark.parametrize("uncovered", [False, True], ids=["lossy-covered", "uncovered-id"])
def test_committed_replay_checks_the_id_it_used(tmp_path, monkeypatch, caplog, summaries, uncovered):  # noqa: F811
    with _closing(_seed(tmp_path)) as engine, caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
        view, ids = _lossy_view(engine, replay=True)
        replay_ids = ids[4:8]
        if uncovered:
            replay_ids[1] = ids[-1]
        observed = _replay_with_ids(engine, monkeypatch, replay_ids)
        output = engine.compress(view, current_tokens=24000)
        assert observed == [("user", "assistant", "tool", "assistant")]
        assert engine._last_compress_leaves[1] == 1
        if uncovered:
            assert output == view
            assert _guard_counts(caplog)[:2] == (8, 1)
        else:
            assert len(output) < len(view)
            assert set(replay_ids) <= _output_sources(engine, output)
            assert all(row.get("tool_call_id") != view[6]["tool_call_id"] for row in output)
            assert not any(record.name == "hermes_lcm.store_complete" and "recorded_failing=" in record.msg
                           for record in caplog.records)


@pytest.mark.parametrize("base_adopts", [True, False], ids=["base-removes", "base-keeps"])
def test_unmapped_replay_row_uses_the_base_decision(tmp_path, monkeypatch, caplog, summaries, base_adopts):  # noqa: F811
    with _closing(_seed(tmp_path)) as engine, caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
        view, ids = _lossy_view(engine, replay=True)
        for row in view[:8]:
            row.pop("message_uid", None)  # Replay is the sole path that can remove these rows.
        replay_ids = ids[:8]
        replay_ids[CALL_INDEX] = None
        observed = _replay_with_ids(engine, monkeypatch, replay_ids, base_adopts=base_adopts)
        output = engine.compress(view, current_tokens=24000)
        assert len(observed) == 1 and len(observed[0]) == 8
        assert engine._last_compress_leaves[1] == 1
        if base_adopts:
            assert len(output) < len(view)
        else:
            assert output == view
            assert _guard_counts(caplog)[:2] == (8, 1)


def test_stub_first_exit_uses_the_base_replay_boundary(tmp_path, monkeypatch, summaries):  # noqa: F811
    # Keep generated payload references byte-identical across the two isolated homes.
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        time=lambda: 1_000, time_ns=lambda: 1_000_000_000_000,
        gmtime=lambda: time.gmtime(1_000), strftime=time.strftime,
    ))

    def run(mode):
        path = tmp_path / str(mode)
        path.mkdir()
        with _closing(_seed(path, leaf_chunk_tokens=1000, large_output_externalization_enabled=True,
                            large_output_externalization_threshold_chars=1_000_000,
                            large_output_active_replay_stubbing_enabled=True,
                            fresh_tail_pressure_yield_enabled=False)) as engine:
            engine.threshold_tokens = 10_000
            view = tool_view()
            for i, row in enumerate(view):
                row["message_uid"] = f"stub-{i}"
            engine.ingest(view)
            ids = [row["store_id"] for row in _rows(engine)]
            _set_frontier(engine, ids[:4])
            store_map = engine._get_store_id_map_for_messages

            def boundary(rows, *args, uid_first=True, **kwargs):
                result = store_map(rows, *args, uid_first=uid_first, **kwargs)
                if len(rows) > 3 and rows[0]["content"] == view[1]["content"]:
                    # A base mapping to a different successor prevents adoption.
                    # The UID-first map can instead confirm the actual successor.
                    result[id(rows[3])] = ids[4 if uid_first and mode == "on" else 5]
                return result

            with monkeypatch.context() as patch:
                if mode is None:
                    patch.delenv("LCM_HOST_MESSAGE_UID")
                patch.setattr(engine, "_get_store_id_map_for_messages", boundary)
                return engine.compress(view, current_tokens=count_messages_tokens(view)), engine._last_stub_first_exit

    base, base_exit = run(None)
    output, on_exit = run("on")
    assert base_exit is not None
    assert on_exit == base_exit
    assert output == base
    assert not summaries


def test_restored_list_runs_through_survival_fit(tmp_path, monkeypatch, caplog, summaries):  # noqa: F811
    def run(path, *, restore):
        path.mkdir()
        with _closing(_seed(path, context_length=3300, context_threshold=0.6)) as engine:
            view = _view(10)
            engine.ingest(view)
            ids = [row["store_id"] for row in _rows(engine)]
            _set_frontier(engine, ids[:4])
            tokens = count_messages_tokens(view)
            ceiling = engine._survival_fit_budget(view, tokens)
            assert engine._survival_measure(view) > ceiling
            if restore:
                # Remove just the old summary header at the proof seam.
                _alter_guard_input(engine, monkeypatch, lambda output, _: [
                    row for row in output if ids[0] not in _output_sources(engine, [row])
                ])
            output = engine.compress(view, current_tokens=tokens)
            assert engine._last_compress_leaves[1] >= 1
            assert engine._survival_measure(output) <= ceiling
            assert output[-2:] == view[-2:]

    with monkeypatch.context() as patch:
        patch.delenv("LCM_HOST_MESSAGE_UID")
        run(tmp_path / "unset", restore=False)
    with caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
        run(tmp_path / "on", restore=True)
        assert _guard_counts(caplog)[:2] == (4, 4)


def _stored_content(engine):
    return [(row["role"], row["content"], row["tool_call_id"], row["tool_calls"]) for row in _rows(engine)]


def _leaf_ids(engine):
    return {node.node_id for node in engine._dag.get_session_nodes("S") if node.source_type == "messages"}


def test_three_leaf_publishing_passes_shrink_without_store_changes(tmp_path, monkeypatch, summaries):  # noqa: F811
    def run(name, *, expect_shrink):
        path = tmp_path / name
        path.mkdir()
        with _closing(_seed(path, dynamic_leaf_chunk_max=900)) as engine:
            view, _, _ = _lossy_view(engine)
            original = _stored_content(engine)
            lengths = [len(view)]
            for _ in range(3):
                leaves = _leaf_ids(engine)
                output = engine.compress(view, current_tokens=24000)
                assert _stored_content(engine) == original
                if expect_shrink:
                    assert _leaf_ids(engine) - leaves, engine.last_compression_noop_reason
                    assert len(output) < len(view)
                lengths.append(len(output))
                view = output
                engine.note_turn_complete()
            print(f"{name}: view lengths {lengths}; stored rows {len(original)}")
            return original

    on = run("on", expect_shrink=True)
    with monkeypatch.context() as patch:
        patch.delenv("LCM_HOST_MESSAGE_UID")
        assert run("unset", expect_shrink=False) == on
