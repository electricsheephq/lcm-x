"""#2815 S0: compare host replay after no-leaf passes in both UID modes."""

import time
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import externalize, host_uid_emit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine, _summary_part_text
from hermes_lcm.tokens import count_messages_tokens, count_tokens
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_uid_restart import _store_copies, _view
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _frontier, _rows
from tests.test_issue_628_summary_circuit import _open_circuit
from tests.test_issue_671_stub_first_exit import STUB, payload_of

_HEAD_SEPARATOR = "\n\n---\n\n"
_HARD_STOP = "time_budget_exhausted"
_EXITS = {
    "hard-budget": _HARD_STOP,
    "soft-after-condensation": "soft_target_reached",
    "stub-first": "stub_first_exit",
    "below-threshold-cleanup": None,
    "route-refused": "summary_route_unavailable",
    "no-progress-hold": None,
    "cold-bind": _HARD_STOP,
}


@dataclass
class Pass:
    input_rows: list[dict]
    output: list[dict]
    same_object: bool
    stored: list[dict]
    frontier: int
    status: str
    noop_reason: str
    stop: str | None
    leaves: int
    condensations: int
    summary_calls: list[int]
    cold_reconciliations: int


def _seed(engine, cell):
    rows = _view(7)
    rows[2]["name"] = "read_file"
    rows[-4]["api_content"] = [{"type": "text", "text": "Synthetic provider-side user text"}]
    if cell == "stub-first":
        # Keep the first user row distinct from the generated summary carrier.
        rows[0]["content"] = [{"type": "text", "text": rows[0]["content"]}]
        for index in range(2, len(rows), 4):
            rows[index]["content"] = payload_of(3000, f"tool{index}x")

    engine.ingest(deepcopy(rows))
    ids = [row["store_id"] for row in _rows(engine)]
    assert len(ids) == len(rows)
    groups = (ids[:4], ids[4:8]) if cell == "soft-after-condensation" else (ids[:8],)
    frontier = 0
    nodes = []
    for group in groups:
        text = f"Synthetic covered turns through store row {group[-1]}."
        if len(groups) == 2:
            text += " Earlier synthetic conversation detail." * 150
        node = SummaryNode(session_id="S", summary=text, token_count=count_tokens(text),
                           source_ids=group, created_at=float(group[-1]))

        def stage(conn, node_id):
            engine._lifecycle.stage_compaction_publication(conn, "conv", "S", node_id, frontier, group)

        nodes.append(engine._dag.get_node(engine._dag.add_node(node, before_commit=stage)))
        frontier = group[-1]
    engine._last_compacted_store_id = frontier
    _store_copies(engine, rows)

    head = {"role": "user", "content": _HEAD_SEPARATOR.join(_summary_part_text(node) for node in reversed(nodes))}
    engine._mint_engine_uids([(head, "summary", None, "summary")])
    engine._host_uid_record_engine([head])
    lineage, _ = engine._host_uid_lineage_key()
    uids = [row["message_uid"] for row in rows]
    bindings = engine._store.host_uid_bindings_for(
        lineage, [head["message_uid"], *uids], kinds=("canonical", "engine"),
    )
    assert all((sid, "canonical") in bindings[uid] for uid, sid in zip(uids, ids))
    assert (0, "engine") in bindings[head["message_uid"]]
    assert _frontier(engine) == ids[7]
    return [head, *rows]


def _run(home, host_state, patch, cell, mode):
    home.mkdir()
    # Two LCM stores and payload homes simulate the same host profile, so the
    # engine's deterministic UID derivation sees one synthetic state.db path.
    (home / "state.db").symlink_to(host_state)
    if mode == "on":
        patch.setenv("LCM_HOST_MESSAGE_UID", mode)
    else:
        patch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    stub = cell == "stub-first"
    config = LCMConfig(
        database_path=str(home / "lcm.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1000 if stub else 400,
        context_threshold=0.1,
        threshold_full_sweep_enabled=cell != "publish-leaf",
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        condensation_fanin=2,
        summary_prefix_target_tokens=400,
        l3_truncate_tokens=2,
        temporal_rollups_enabled=False,
        background_leaf_prep_enabled=False,
        fresh_tail_pressure_yield_enabled=False,
        large_output_externalization_enabled=stub,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=stub,
    )
    offset = 0
    clock = time.monotonic
    calls = []
    reconciliations = []

    def summarize(**kwargs):
        nonlocal offset
        calls.append(kwargs["depth"])
        if cell == "soft-after-condensation":
            offset += 50
        return "Earlier synthetic turns.\nExpand for details about: turns", 1

    def bind(stack):
        engine = stack.enter_context(_closing(LCMEngine(config=config, hermes_home=str(home))))
        engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
        return engine

    with ExitStack() as seed_engine, ExitStack() as restarted_engine:
        engine = bind(seed_engine)
        view = _seed(engine, cell)
        original = deepcopy(view)
        if cell == "cold-bind":
            seed_engine.close()
            engine = bind(restarted_engine)
            assert engine._compress_calls == 0 and engine._ingest_cursor == 0
            reconcile = engine._reconcile_ingest_cursor_from_store

            def observe_reconcile(*args, **kwargs):
                reconciliations.append(True)
                return reconcile(*args, **kwargs)

            patch.setattr(engine, "_reconcile_ingest_cursor_from_store", observe_reconcile)

        current_tokens = 24_000
        if stub:
            engine.threshold_tokens = 14_000
            current_tokens = count_messages_tokens(view)
            assert current_tokens > engine.threshold_tokens
        elif cell == "below-threshold-cleanup":
            current_tokens = count_messages_tokens(view)
            assert current_tokens < engine.threshold_tokens
            engine._compaction_request.below_threshold_cleanup_only = True
        elif cell == "no-progress-hold":
            engine._start_no_progress_hold("no_progress")
            assert engine._no_progress_hold_blocks(current_tokens)
        elif cell == "route-refused":
            _open_circuit(engine)
            assert not engine._summary_route_available()
        if cell in {"hard-budget", "cold-bind"}:
            prepare = engine._prepare_retained_user_anchor

            def spend_budget(*args, **kwargs):
                nonlocal offset
                result = prepare(*args, **kwargs)
                offset += 121
                return result

            patch.setattr(engine, "_prepare_retained_user_anchor", spend_budget)

        patch.setattr(time, "monotonic", lambda: clock() + offset)
        patch.setattr(engine_module, "summarize_with_escalation", summarize)
        before = {node.node_id for node in engine._dag.get_session_nodes("S")}
        output = engine.compress(view, current_tokens=current_tokens)
        new_nodes = [node for node in engine._dag.get_session_nodes("S") if node.node_id not in before]
        return Pass(
            original, output, output is view, _rows(engine), _frontier(engine),
            engine.last_compression_status, engine.last_compression_noop_reason,
            (engine._last_threshold_full_sweep or {}).get("stop_reason"),
            sum(node.source_type == "messages" for node in new_nodes),
            sum(node.source_type == "nodes" for node in new_nodes),
            calls, len(reconciliations),
        )


@pytest.fixture
def paired_passes(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    # Persisted row timestamps and externalized references must be reproducible.
    monkeypatch.setattr(time, "time", lambda: 2_000.0)
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        time=time.time, time_ns=lambda: 2_000_000_000_000,
        gmtime=lambda: time.gmtime(2_000), strftime=time.strftime,
    ))

    def run(cell):
        results = []
        for mode in ("unset", "on"):
            with monkeypatch.context() as patch:
                results.append(_run(tmp_path / mode, tmp_path / "state.db", patch, cell, mode))
        return results

    return run


def _value_shape(value):
    shape = type(value).__name__
    if isinstance(value, (str, list, dict)):
        shape += f" len={len(value)}"
    return shape


def _assert_same_output(unset, on):
    engine_uid = unset.input_rows[0]["message_uid"]
    for index, (left, right) in enumerate(zip_longest(unset.output, on.output)):
        if left != right:
            left_keys, right_keys = left or {}, right or {}
            differences = {
                key: tuple(_value_shape(row[key]) if key in row else "missing" for row in (left_keys, right_keys))
                for key in sorted(left_keys.keys() | right_keys.keys())
                if key not in left_keys or key not in right_keys or left_keys[key] != right_keys[key]
            }
            origin = tuple("missing" if row is None else
                           "engine-made head" if row.get("message_uid") == engine_uid else "host row"
                           for row in (left, right))
            pytest.fail(f"first differing index {index}, lengths {len(unset.output)}/{len(on.output)}, "
                        f"row origin unset/on {origin}, differing keys unset/on {differences}")
    assert len(unset.output) == len(on.output)


@pytest.mark.parametrize("cell", _EXITS)
def test_no_leaf_exit_preserves_host_rows(cell, paired_passes):
    unset, on = paired_passes(cell)
    assert unset.input_rows == on.input_rows
    _assert_same_output(unset, on)
    assert unset.same_object == on.same_object
    assert len(unset.stored) == len(on.stored)
    assert unset.stored == on.stored
    assert unset.frontier == on.frontier == 8
    assert (unset.status, unset.noop_reason) == (on.status, on.noop_reason)

    for mode, result in (("unset", unset), ("on", on)):
        assert result.status != "error", (mode, result.noop_reason)
        assert result.leaves == 0, mode
        # An exit that never reaches the sweep leaves its stop reason at the initial empty string.
        assert (result.stop or None) == _EXITS[cell], (mode, result.status, result.noop_reason)
        assert result.condensations == (1 if cell == "soft-after-condensation" else 0)
        assert result.summary_calls == ([1] if cell == "soft-after-condensation" else [])
        assert [row["message_uid"] for row in result.output] == [row["message_uid"] for row in result.input_rows]
        if cell in {"below-threshold-cleanup", "no-progress-hold", "stub-first"}:
            assert result.status == "sanitized"
        if cell == "stub-first":
            assert any(row["role"] == "tool" and row["content"].startswith(STUB) for row in result.output)
        if cell == "cold-bind":
            assert result.cold_reconciliations > 0
    print(f"equal rows={len(on.output)}, input identity={on.same_object}, stored={len(on.stored)}, "
          f"frontier={on.frontier}, status={on.status!r}, no-op reason={on.noop_reason!r}, "
          f"stop={on.stop!r}, condensations={on.condensations}")


def test_published_leaf_makes_the_two_modes_differ(paired_passes):
    unset, on = paired_passes("publish-leaf")
    assert unset.input_rows == on.input_rows
    assert unset.leaves > 0 and on.leaves > 0
    assert unset.status == on.status == "compacted"
    assert len(on.output) < len(unset.output)
    assert {row["message_uid"] for row in on.input_rows[1:9]}.isdisjoint(
        row.get("message_uid") for row in on.output
    )
    print(f"published leaves unset/on={unset.leaves}/{on.leaves}, "
          f"output rows unset/on={len(unset.output)}/{len(on.output)}")
