"""#2815: replay exit parity and carrier-led publication with bound host UIDs."""

import sqlite3
from copy import deepcopy
from unittest.mock import patch

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.compaction import SweepBudgetExhausted, select_oldest_leaf_chunk
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER as PRUNED
from hermes_lcm.store_complete import _UID_DROP_INPUT
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.invariants.test_issue_2815_c18 import (
    CALL,
    _adoptable_plain,
    _assert_bound,
    _carrier_replay,
    _closing,
    _missing,
    _new_user,
    _observe_guard,
    _open,
    _pairs,
    _publish,
    _run,
    _view,
    summaries as summaries,
    tool_pair,
    uid_on as uid_on,
)
from tests.test_host_uid_shadow import _state_db

pytestmark = pytest.mark.issue(2815)


@pytest.fixture
def lineage(tmp_path):
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    return tmp_path


def _arm_engine(home, arm):
    path = home / arm
    path.mkdir()
    return _open(path, hermes_home=home)


def _bound_backlog(engine, view):
    _assert_bound(engine, view)
    assert any(row["store_id"] > engine._last_compacted_store_id
               for row in engine._host_uid_resolve(view).values())


def _subject(engine, fault, *, mode):
    if mode == "unset":
        fault.delenv("LCM_HOST_MESSAGE_UID")
    else:
        fault.setenv("LCM_HOST_MESSAGE_UID", mode)
    fault.setattr(engine, "_c20b_subject_pass", mode == "on", raising=False)


def _outcome(engine, view):
    output = _run(engine, view)
    assert engine._ingest_cursor == len(output)
    return output, engine._last_compression_status, engine._last_compression_noop_reason


def _early_exit(home, monkeypatch, mode, exit_case):
    with _closing(_arm_engine(home, mode)) as engine, monkeypatch.context() as fault:
        # The same published plain head and bound raw backlog are built in each store.
        view = _adoptable_plain(engine)
        _bound_backlog(engine, view)
        timed = exit_case in {"sweep", "anchor_ids", "replay_drops"}
        if timed:
            engine._config.threshold_full_sweep_enabled = True
            engine.threshold_tokens = 1

        admitted = []
        admit_leaf = engine._admit_leaf_and_adopt_replay

        def admit(state, leaf, passes, step_done):
            if exit_case == "sweep":
                state.sweep_deadline = 0
            if exit_case == "foreground":
                state.budget.progress = "condensation"

                def refuse(route):
                    raise SweepBudgetExhausted("foreground soft target reached", reason="soft_target_reached")

                fault.setattr(state.budget, "admit", refuse)

            def clock(step, started):
                now = step_done(step, started)
                return state.sweep_deadline if step == exit_case else now

            out = admit_leaf(state, leaf, passes, clock)
            admitted.append((passes, out[0], state.sweep_stop_reason))
            return out

        fault.setattr(engine, "_admit_leaf_and_adopt_replay", admit)
        if exit_case == "condensation":
            condense = engine._condense_before_leaves

            def spend_passes(state, *args):
                result = condense(state, *args)
                state.max_leaf_passes = 0
                return result

            fault.setattr(engine, "_condense_before_leaves", spend_passes)

        _subject(engine, fault, mode=mode)
        result = _outcome(engine, view)
        expected = [] if exit_case == "condensation" else [
            (0, "break", "soft_target_reached" if exit_case == "foreground" else "time_budget_exhausted")
        ]
        assert admitted == expected
        return view, result


@pytest.mark.parametrize("exit_case", ["foreground", "sweep", "condensation", "anchor_ids", "replay_drops"])
def test_d1_exit_before_replay_drop(lineage, monkeypatch, summaries, exit_case):
    base_view, (base, base_status, base_reason) = _early_exit(lineage, monkeypatch, "unset", exit_case)
    view, (output, status, reason) = _early_exit(lineage, monkeypatch, "on", exit_case)
    assert view == base_view and base is base_view
    assert output is view
    assert (status, reason) == (base_status, base_reason)


def _conflict_arm(home, monkeypatch, mode, variant):
    with _closing(_arm_engine(home, mode)) as engine, monkeypatch.context() as fault:
        if variant == "moved-frontier":
            view = _carrier_replay(engine, "engine")
            view[0]["_absorbed_message_uids"].append("c18-unbound")
        elif variant.startswith("quote"):
            _, published = _publish(engine, "plain")
            content = "I pasted [Recent Summary (d0, node 7)] about sprockets. What does it mean?"
            if variant == "quote-expand":
                content += " [Expand for details: lcm_expand(node_id=7)]"
            quote = {"role": "user", "content": content, "message_uid": "c20b-quote"}
            view = [quote, *published, *_view(7)[21:], _new_user(), _missing()]
        else:
            view = _adoptable_plain(engine)
        _bound_backlog(engine, view)
        engine._config.leaf_chunk_tokens = 1
        if variant == "route-refused":
            engine._config.survival_fit = True
            # Keep the small selected source whole so a route-free publication is possible.
            engine._config.l3_truncate_tokens = 4000
            fault.setattr(engine, "_summary_route_available", lambda: False)

        published, quote_identities = [], []
        publish = engine._publish_leaf_node
        if variant.startswith("quote"):
            replay = engine._committed_replay_drops

            def observe_replay(rows, start, *args, **kwargs):
                if not published and rows[0].get("message_uid") == view[0]["message_uid"]:
                    quote_identities.append(engine._compress_occurrences[id(rows[0])][1])
                return replay(rows, start, *args, **kwargs)

            fault.setattr(engine, "_committed_replay_drops", observe_replay)

        def lock(*args, **kwargs):
            scope = _UID_DROP_INPUT.get()
            published.append(scope)
            if variant == "moved-frontier":
                frontier = engine._last_compacted_store_id
                with sqlite3.connect(engine._config.database_path) as writer:
                    writer.execute("UPDATE lcm_lifecycle_state SET current_frontier_store_id=? WHERE conversation_id=?",
                                   (frontier + 1, "conv"))
                writer.close()
                with pytest.raises(LifecyclePublicationConflictError) as exc:
                    publish(*args, **kwargs)
                raise exc.value
            raise sqlite3.OperationalError("database is locked")

        fault.setattr(engine, "_publish_leaf_node", lock)
        _subject(engine, fault, mode=mode)
        with patch.object(engine, "_settle_leaf_stop", wraps=engine._settle_leaf_stop) as settle:
            result = _outcome(engine, view)
        if mode == "unset" and variant == "moved-frontier":
            assert not published  # Unset adopts the committed carrier and assembles.
        else:
            assert published and not settle.called
        if variant.startswith("quote"):
            assert quote_identities and all(identity is not None for identity in quote_identities)
        if mode == "on":
            scope = published[0]
            assert scope.uid_trimmed
            if variant == "route-refused":
                assert scope.adopt_before_stop and scope.legacy_replay is None
        return view, result, engine._missing_tool_result_stub(CALL)


@pytest.mark.xfail(strict=True, reason="R2, accepted in Round c20p: post-pass read sees the moved frontier")
def test_d2_second_writer_moves_frontier(lineage, monkeypatch, summaries):
    _, (base, _, _), _ = _conflict_arm(lineage, monkeypatch, "unset", "moved-frontier")
    _, (output, _, _), _ = _conflict_arm(lineage, monkeypatch, "on", "moved-frontier")
    assert _pairs(base)[0][-1] == _pairs(base)[1][-1] == CALL
    assert (_pairs(output)[0][-1], _after_final_call(output)) == (_pairs(base)[0][-1], _after_final_call(base))


def _after_final_call(rows):
    index = next(i for i, row in enumerate(rows) if any(call["id"] == CALL for call in row.get("tool_calls", ())))
    return rows[index + 1:index + 2]


@pytest.mark.parametrize("variant", ["quote-expand", "quote", "plain"])
def test_d3_quotation_publication_conflict(lineage, monkeypatch, summaries, variant):
    base_view, (base, base_status, base_reason), stub = _conflict_arm(lineage, monkeypatch, "unset", variant)
    view, (output, status, reason), _ = _conflict_arm(lineage, monkeypatch, "on", variant)
    assert view == base_view
    assert _after_final_call(base) == ([stub] if variant == "plain" else [])
    assert _after_final_call(output) == _after_final_call(base)
    if variant.startswith("quote"):
        assert output == base
    assert [row for row in output if row in view] == view
    assert (status, reason) == (base_status, base_reason)


@pytest.mark.parametrize("variant", ["route-refused", "plain"])
def test_d4_route_refused_publication_conflict(lineage, monkeypatch, summaries, variant):
    base_view, (base, base_status, base_reason), stub = _conflict_arm(lineage, monkeypatch, "unset", variant)
    view, (output, status, reason), _ = _conflict_arm(lineage, monkeypatch, "on", variant)
    assert view == base_view and _after_final_call(base) == [stub]
    assert _after_final_call(output) == _after_final_call(base)
    assert (status, reason) == (base_status, base_reason)


LEAF = 2400
PAD = " alpha beta gamma delta"


def _walk_rows(restored, *, system=False, repeated=False):
    call, tool = tool_pair("c20b-walk", "RMARK" + PAD * restored)
    call["tool_calls"][0]["function"]["arguments"] = '{"q":"CMARK"}'
    prefix = [{"role": "system", "content": "Synthetic system prompt."}] if system else []
    rows = [
        *prefix,
        {"role": "user", "content": "OMARK" + PAD * 1250},
        {"role": "assistant", "content": "OAMARK" + PAD * 1250},
        {"role": "user", "content": "UMARK" + PAD * 7},
        call, tool,
    ]
    if repeated:
        rows.extend({"role": "user", "content": f"raw {i}" + PAD * 41} for i in range(48))
    else:
        rows.extend([
            {"role": "user", "content": "BMARK" + PAD * 760},
            {"role": "assistant", "content": "BA short"},
            {"role": "user", "content": "T1 tail"},
            {"role": "assistant", "content": "T2 tail"},
        ])
    return [{**row, "message_uid": f"c20b-walk-{i}"} for i, row in enumerate(rows)]


def _walk_engine(home, arm, *, default_max=False):
    path = home / arm
    path.mkdir()
    config = LCMConfig(
        database_path=str(path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=LEAF,
        dynamic_leaf_chunk_enabled=False, threshold_full_sweep_enabled=False,
        temporal_rollups_enabled=False, survival_fit=False, embeddings_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        large_output_externalization_threshold_chars=200_000,
    )
    if not default_max:
        config.dynamic_leaf_chunk_max = LEAF
    engine = LCMEngine(config=config, hermes_home=str(home))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _seed_walk(engine, fault, raw, summary):
    texts = []

    def summarize(**kwargs):
        texts.append(kwargs["text"])
        return summary, 1

    fault.setattr(engine_module, "summarize_with_escalation", summarize)
    start = int(raw[0]["role"] == "system")
    source, tail = raw[start:start + 2], raw[start + 2:]
    engine.ingest(raw)
    engine._config.leaf_chunk_tokens = count_messages_tokens(source)
    engine._config.fresh_tail_count = len(tail)
    tokens = count_messages_tokens(raw)
    engine.threshold_tokens = tokens - 100
    emitted = engine.compress(raw, force=True, current_tokens=tokens)
    assert engine._last_compress_leaves[1] == 1
    engine._config.leaf_chunk_tokens = LEAF
    engine._config.fresh_tail_count = 2
    engine.threshold_tokens = 100
    return emitted, tail, texts


def _prune(rows):
    return [
        {**row, "content": PRUNED} if row.get("tool_call_id") == "c20b-walk" else deepcopy(row)
        for row in rows
    ]


def _tool_group_source(engine, view, raw):
    _bound_backlog(engine, view)
    index = next(i for i, row in enumerate(view) if row.get("content") == PRUNED)
    resolved = engine._host_uid_resolve(view)
    tool = next(row for row in raw if row["role"] == "tool")
    assert resolved[index]["content"] == tool["content"]
    assert resolved[index - 1]["store_id"] < resolved[index]["store_id"]
    return tool


def _d5_walk(home, monkeypatch, shape, restored, *, mode="on", default_max=False):
    with _closing(_walk_engine(home, shape, default_max=default_max)) as engine, monkeypatch.context() as fault:
        raw = _walk_rows(restored)
        summary = "Earlier turns." + " summary words here" * 1500 + "\nExpand for details about: turns"
        emitted, tail, texts = _seed_walk(engine, fault, raw, summary)
        head = emitted[0]
        remainder = engine._generated_context_carrier_remainder(head)
        assert remainder == tail[0]["content"]
        assert head["_absorbed_message_uids"] == [tail[0]["message_uid"]]
        if shape == "user":
            pure = {key: value for key, value in head.items() if key != "_absorbed_message_uids"}
            pure["content"] = head["content"][:-(len(remainder) + 2)]
            view = _prune([pure, *tail])
        else:
            view = _prune([head, *tail[1:]])
        tool = _tool_group_source(engine, view, raw)
        assert count_message_tokens(tail[0]) < LEAF / 40
        assert 2.9 * LEAF < count_message_tokens(head) < 3.1 * LEAF
        assert (count_messages_tokens(tail[:2] + [tool]) < LEAF) == (restored == 70)
        assert count_message_tokens(tail[3]) > LEAF

        budgets = []
        walk = engine._store_complete_input

        def observe(chunk, claims, full_map, active, raw_chunk, frontier, carry, budget, *args, **kwargs):
            budgets.append((budget, count_messages_tokens(chunk)))
            if not default_max:
                assert all("BMARK" not in row.get("content", "") for row in chunk)
            else:
                assert any("BMARK" in row.get("content", "") for row in chunk)
            return walk(chunk, claims, full_map, active, raw_chunk, frontier, carry, budget, *args, **kwargs)

        fault.setattr(engine, "_store_complete_input", observe)
        _subject(engine, fault, mode=mode)
        engine._ingest_cursor = len(view)
        engine.note_turn_complete()
        output = engine.compress(view, force=True, current_tokens=count_messages_tokens(view))
        assert engine._ingest_cursor == len(output) and engine._last_compress_leaves[1] == 1
        _, sent = texts
        markers = tuple(mark for mark in ("UMARK", "CMARK", "RMARK", "BMARK") if mark in sent)
        return markers, budgets[0]


@pytest.mark.parametrize("restored", [70, 700], ids=["fits", "over"])
def test_d5_carrier_and_user_walk_extent(lineage, monkeypatch, restored):
    user, _ = _d5_walk(lineage, monkeypatch, "user", restored)
    carrier, _ = _d5_walk(lineage, monkeypatch, "carrier", restored)
    expected = ("UMARK", "CMARK", "RMARK") if restored == 70 else ("UMARK",)
    assert user == expected
    assert carrier == user


@pytest.mark.parametrize("mode", ["unset", "shadow"])
def test_d5_recognition_off_keeps_head_budget(lineage, monkeypatch, mode):
    extent, (budget, full_chunk) = _d5_walk(lineage, monkeypatch, "carrier", 70, mode=mode)
    assert (extent, budget) == (("UMARK", "CMARK"), full_chunk)


def test_d5_default_max_keeps_head_extent(lineage, monkeypatch):
    extent, _ = _d5_walk(lineage, monkeypatch, "carrier", 70, default_max=True)
    assert extent == ("UMARK", "CMARK", "RMARK")


def _d6_passes(home, monkeypatch, arm, *, system, plain):
    with _closing(_walk_engine(home, arm)) as engine, monkeypatch.context() as fault:
        raw = _walk_rows(70, system=system, repeated=True)
        summary = "Earlier turns." + PAD * 395 + "\nExpand for details about: turns"
        view, _, texts = _seed_walk(engine, fault, raw, summary)
        view = _prune(view)
        _tool_group_source(engine, view, raw)
        if not system:
            assert engine._generated_context_carrier_remainder(view[0]) is not None
            assert 0.9 * LEAF < count_message_tokens(view[0]) < 1.1 * LEAF
        engine._config.survival_fit = True
        fit = engine._survival_fit
        records = []

        for _ in range(4):
            with monkeypatch.context() as subject:
                _subject(engine, subject, mode="on")
                subject.setattr(engine, "_c20b_plain_selector", plain, raising=False)
                if system and plain:
                    subject.setattr(engine, "_select_oldest_leaf_chunk", select_oldest_leaf_chunk)
                statuses = []

                def observe_fit(original, candidate, *args, **kwargs):
                    output = fit(original, candidate, *args, **kwargs)
                    statuses.append(output is not candidate)
                    return output

                subject.setattr(engine, "_survival_fit", observe_fit)
                carrier = any(engine._generated_context_carrier_remainder(row) is not None for row in view[:2])
                before = engine._last_compacted_store_id
                rows = engine._store.get_session_messages("S", limit=1000)
                backlog = [row for row in rows if row["store_id"] > before][:-2]
                assert count_messages_tokens(backlog) > LEAF
                nodes = {node.node_id for node in engine._dag.get_session_nodes("S")}
                engine.note_turn_complete()
                output = engine.compress(view, current_tokens=count_messages_tokens(view))
                leaves = engine._last_compress_leaves[1] if engine._last_compress_leaves else 0
                extents = tuple(tuple(node.source_ids) for node in engine._dag.get_session_nodes("S")
                                if node.node_id not in nodes and node.source_type == "messages")
                records.append((carrier, before, engine._last_compacted_store_id, leaves, extents, tuple(statuses)))
                assert engine._ingest_cursor == len(output)
                view = _prune(output)
        return records, texts


def test_d6_carrier_led_passes_advance(lineage, monkeypatch):
    records, texts = _d6_passes(lineage, monkeypatch, "carrier", system=False, plain=True)
    assert all(after > before and leaves == 1 and fit == (False,) and
               (not carrier or all(len(ids) > 1 for ids in extents))
               for carrier, before, after, leaves, extents, fit in records), records
    assert records[0][0] and sum(record[0] for record in records) >= 2
    assert all(mark in texts[1] for mark in ("UMARK", "CMARK", "RMARK"))


def test_d6_system_row_records_unchanged(lineage, monkeypatch):
    remainder = _d6_passes(lineage, monkeypatch, "new", system=True, plain=False)
    full = _d6_passes(lineage, monkeypatch, "old", system=True, plain=True)
    assert full == remainder
    assert all(not record[0] and record[3] == 1 for record in full[0])


@pytest.mark.parametrize("near_threshold", [False, True], ids=["equal", "shorter"])
def test_d9_guard_restore_controls_hidden_only_hold(lineage, monkeypatch, summaries, near_threshold):
    with _closing(_arm_engine(lineage, "store")) as engine, monkeypatch.context() as fault:
        view = _adoptable_plain(engine, covered_replay=True)
        _bound_backlog(engine, view)
        engine._config.leaf_chunk_tokens = 1
        if near_threshold:
            call, tool = tool_pair("c20b-guard", "synthetic tool text " * 100)
            view[-1:] = [call, tool, _missing(calls=48)]
            engine._config.large_output_externalization_enabled = True
            engine._config.large_output_active_replay_stubbing_enabled = True
            engine._config.large_output_active_replay_stub_threshold_tokens = 250
        verdicts = _observe_guard(engine, fault, deny_coverage=True)
        _subject(engine, fault, mode="on")
        with patch.object(engine, "_start_no_progress_hold", wraps=engine._start_no_progress_hold) as hold:
            output = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert engine._last_compress_leaves[1] == 1
        [(assembly, guarded)] = verdicts
        assert guarded is not assembly and CALL not in _pairs(guarded)[1]
        stubs = [engine._missing_tool_result_stub(call["id"]) for call in view[-1]["tool_calls"]]
        stubbed = guarded + stubs
        host_tokens = count_messages_tokens(view)
        assert len(guarded) == len(view) and count_messages_tokens(stubbed) >= host_tokens
        if near_threshold:
            assert count_messages_tokens(guarded) < host_tokens
        else:
            assert guarded == view
        assert output == stubbed
        assert any(call.args == ("hidden_only",) for call in hold.call_args_list) == (not near_threshold)
        assert engine._ingest_cursor == len(output)
