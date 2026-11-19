"""#2815: prove no-leaf hand-backs against independent mode-unset passes."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from hermes_lcm.store_complete import _UID_DROP_INPUT
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_c18 import (
    CALL,
    SHAPES,
    STUB,
    _adoptable_plain,
    _assert_bound,
    _binding_records,
    _carrier_replay,
    _closing,
    _compare_no_leaf,
    _missing,
    _new_user,
    _observe_guard,
    _open,
    _publish,
    _refuse_leaf,
    _run,
    _seed,
    _spy_decisions,
    _spy_finish,
    _view,
    summaries as summaries,
    tool_pair,
    uid_on as uid_on,
)

pytestmark = pytest.mark.issue(2815)


def _subject(engine, monkeypatch, view, *, leaf=False):
    _assert_bound(engine, view)
    with monkeypatch.context() as active:
        active.setattr(engine, "_c20_subject_pass", True, raising=False)
        if leaf:
            return engine.compress(view, current_tokens=count_messages_tokens(view))
        return _run(engine, view)


@pytest.mark.parametrize(("shape", "case"), [
    ("plain", "plain-conflict"), ("carrier", "carrier-conflict"), ("plain", "missing-uid"),
])
def test_no_leaf_wrapper_and_adopted_restore(tmp_path, monkeypatch, summaries, shape, case):
    _compare_no_leaf(tmp_path, monkeypatch, shape, case)


def test_leaf_guard_compares_stub_free_restore(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine, covered_replay=True)
        call, raw_tool = tool_pair("c20-large", "synthetic tool text " * 140)
        view[-1:] = [call, raw_tool, _missing(calls=35)]
        engine._config.leaf_chunk_tokens = 1
        engine._config.large_output_externalization_enabled = True
        engine._config.large_output_active_replay_stubbing_enabled = True
        engine._config.large_output_active_replay_stub_threshold_tokens = 350
        verdicts = _observe_guard(engine, monkeypatch, deny_coverage=True)
        with patch.object(engine, "_start_no_progress_hold", wraps=engine._start_no_progress_hold) as hold:
            output = _subject(engine, monkeypatch, view, leaf=True)
        assert engine._last_compress_leaves[1] == 1
        [(assembly, guarded)] = verdicts
        assert guarded is not assembly
        assert guarded[:-2] == view[:-2] and guarded[-1] == view[-1]
        assert guarded[-2]["content"].startswith(STUB) and guarded[-2]["tool_call_id"] == raw_tool["tool_call_id"]
        calls = [item["id"] for item in view[-1]["tool_calls"]]
        expected = guarded + [engine._missing_tool_result_stub(call) for call in calls]
        assert len(guarded) == len(view) <= len(expected)
        assert count_messages_tokens(guarded) < count_messages_tokens(view) <= count_messages_tokens(expected)
        assert output == expected
        assert not any(call.args == ("hidden_only",) for call in hold.call_args_list)
        assert engine._ingest_cursor == len(output)


@pytest.mark.parametrize("absorbed", ["self", "dict", "string", "distinct"])
def test_carrier_absorbed_uid_shape(tmp_path, monkeypatch, summaries, absorbed):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _carrier_replay(engine, "engine", re_glue=False)
        head = view[0]
        own = head["message_uid"]
        original, = head["_absorbed_message_uids"]
        remainder = engine._generated_context_carrier_remainder(head)
        assert engine._host_uid_engine_uids({own, original}) == {own}
        head["_absorbed_message_uids"] = {"self": [own], "dict": [{"a": 1}],
                                         "string": "x", "distinct": [original]}[absorbed]
        decisions = _spy_decisions(engine, monkeypatch)
        try:
            output = _subject(engine, monkeypatch, view)
        except TypeError as exc:
            print(f"compress raised {exc!r}")
            output = None
        if absorbed == "distinct":
            assert output is not None and output[1] == {"role": "user", "content": remainder, "message_uid": original}
            assert len(decisions) == 1 and decisions[0][1] is not None
        else:
            assert output is not None and output[0] == head
            assert all(row.get("message_uid") != own for row in output[1:])
            assert decisions == [(False, None)]
        assert engine._ingest_cursor == len(output)


@pytest.mark.parametrize("shape", SHAPES)
def test_aged_replay_result_is_stubbed(tmp_path, monkeypatch, summaries, shape):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _carrier_replay(engine, "engine") if shape == "carrier" else _adoptable_plain(engine)
        resolved = engine._host_uid_resolve(view)
        first_tool = next(
            row for index, row in enumerate(view) if row["role"] == "tool"
            and (index not in resolved or resolved[index]["store_id"] > engine._last_compacted_store_id)
        )
        engine._config.large_output_active_replay_stubbing_enabled = True
        engine._config.large_output_externalization_enabled = True
        engine._config.large_output_active_replay_stub_aged_threshold_tokens = 10
        engine._config.large_output_active_replay_stub_threshold_tokens = 10_000
        entries = _spy_finish(engine, monkeypatch)
        output = _subject(engine, monkeypatch, view)
        assert any(row.get("tool_call_id") == first_tool["tool_call_id"]
                   and row.get("message_uid") == first_tool["message_uid"]
                   and row["content"].startswith(STUB) for row in output)
        [(scope, forced, _)] = entries
        assert scope.legacy_replay and not forced


def test_pre_leaf_condensation_returns_carrier_assembly(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _carrier_replay(engine, "engine", two_leaves=True)
        assert len(engine._dag.get_uncondensed_at_depth("S", 0)) == 2
        engine._config.threshold_full_sweep_enabled = True
        engine._config.condensation_fanin = 2
        engine._config.summary_prefix_target_tokens = 1
        engine.threshold_tokens = 100
        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _refuse_leaf)
        entries = _spy_finish(engine, monkeypatch)
        output = _subject(engine, monkeypatch, view)
        assert "[Session Arc Summary (d1," in output[0]["content"]
        assert engine._missing_tool_result_stub(CALL) in output
        [(scope, forced, condensations)] = entries
        assert scope.legacy_replay and not forced and condensations == 1


def test_binding_change_returns_original(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine)
        finish = engine._finish_without_leaf
        exits = []

        def rebind(state, *args):
            scope = _UID_DROP_INPUT.get()
            assert scope.uid_trimmed and scope.legacy_replay is not None
            engine.on_session_start("B", platform="telegram", context_length=200_000, conversation_id="B-conv")
            result = finish(state, *args)
            exits.append(result)
            return result

        monkeypatch.setattr(engine, "_finish_without_leaf", rebind)
        returned = _subject(engine, monkeypatch, view)
        [(action, output)] = exits
        assert output is view
        assert action == "return"
        assert returned is view


@pytest.mark.parametrize("fits", [True, False], ids=["fits", "over-cap"])
def test_adopted_assembly_cap(tmp_path, monkeypatch, summaries, fits):
    with _closing(_seed(tmp_path / "store")) as engine:
        if fits:
            view = _adoptable_plain(engine, covered_replay=True)
        else:
            _, published = _publish(engine, "plain")
            view = [*published, _new_user(), _missing(calls=3)]
        engine.ingest(view)
        engine._config.fresh_tail_count = 1
        observed = count_messages_tokens(view)
        cap = observed + (1000 if fits else 1)
        engine._config.max_assembly_tokens = cap
        assert engine._effective_assembly_token_cap() == cap > observed
        entries = _spy_finish(engine, monkeypatch)
        assembled = []
        assemble = engine._assemble_context

        def record(*args, **kwargs):
            output = assemble(*args, **kwargs)
            assembled.append(output)
            return output

        monkeypatch.setattr(engine, "_assemble_context", record)
        output = _subject(engine, monkeypatch, view)
        if fits:
            assert len(assembled) == 1 and output is assembled[0]
        else:
            assert output == view + [engine._missing_tool_result_stub(call) for call in (CALL, f"{CALL}-1", f"{CALL}-2")]
        [candidate] = assembled
        [(scope, forced, _)] = entries
        assert scope.uid_trimmed and scope.legacy_replay is not None and not forced
        assert (count_messages_tokens(candidate) <= cap) == fits
        assert candidate != engine._sanitize_active_context_messages(view, insert_missing_tool_stubs=True)
        assert engine._ingest_cursor == len(output)


@pytest.mark.parametrize("origin", ["cold-host", "canonical-only"])
def test_carrier_without_warm_adoption_or_engine_uid(tmp_path, monkeypatch, summaries, origin):
    path = tmp_path / "store"
    with _closing(_seed(path)) as warm:
        if origin == "cold-host":
            view = _carrier_replay(warm, "host")
        else:
            view = _carrier_replay(warm, "engine", re_glue=False, coalesce_replies=True)
            raw = _view(6)
            own = raw[4]["message_uid"]
            assert [tuple(row) for row in _binding_records(warm, own)] == [("canonical", "stored_new", 1)]
            view[0] = {**view[0], "message_uid": own}
            assert warm._host_uid_engine_uids({own}) == set()
            _assert_carrier_decline(warm, monkeypatch, view, origin)
    if origin == "cold-host":
        with _closing(_open(path)) as cold:
            cold._config.leaf_chunk_tokens = 50_000
            cold._config.fresh_tail_count = 3
            _assert_carrier_decline(cold, monkeypatch, view, origin)


def _assert_carrier_decline(engine, monkeypatch, view, origin):
    # This host has already coalesced adjacent assistant replies before calling
    # compress. Keep the unanswered call to distinguish the two hand-backs.
    view = engine._sanitize_active_context_messages(view, insert_missing_tool_stubs=False)
    expected = deepcopy(view)
    decisions = _spy_decisions(engine, monkeypatch)
    with (
        patch.object(engine, "_host_uid_engine_uids", wraps=engine._host_uid_engine_uids) as lookup,
        patch.object(engine, "_reconcile_ingest_cursor_from_store",
                     wraps=engine._reconcile_ingest_cursor_from_store) as reconcile,
    ):
        output = _subject(engine, monkeypatch, view)
    own = view[0]["message_uid"]
    absorbed, = view[0]["_absorbed_message_uids"]
    split_lookups = [call for call in lookup.call_args_list if call.args == ({own, absorbed},) and not call.kwargs]
    print(f"{origin}: replay={decisions}; caller-list={output is view}; split-lookups={split_lookups}")
    if origin == "cold-host":
        assert output is view and output == expected
        assert reconcile.call_count >= 1
        assert not split_lookups
    else:
        assert output == expected + [engine._missing_tool_result_stub(CALL)]
        assert len(split_lookups) == 1
    assert decisions == [(False, None)]
    assert engine._ingest_cursor == len(output)
