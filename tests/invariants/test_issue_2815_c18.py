"""#2815: real replay on hosts whose first row is a plain summary or a carrier."""

import logging
import shutil
import sqlite3
from copy import deepcopy
from unittest.mock import patch

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.config import LCMConfig
from hermes_lcm.compaction import SweepBudgetExhausted
from hermes_lcm.engine import LCMEngine, _summary_part_text
from hermes_lcm.store_complete import _UID_DROP_INPUT
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_c8 import _guard_counts, _output_sources
from tests.invariants.test_issue_2815_uid_restart import _view
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _covered, summaries as summaries
from tests.test_issue_671_stub_first_exit import STUB, tool_pair

pytestmark = pytest.mark.issue(2815)
SHAPES = ("plain", "carrier")
CALL = "c18-missing"


@pytest.fixture(autouse=True)
def uid_on(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)


def _open(path, *, hermes_home=None):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(path / "lcm.db"), fresh_tail_count=8, leaf_chunk_tokens=400, dynamic_leaf_chunk_enabled=False,
        threshold_full_sweep_enabled=False, temporal_rollups_enabled=False,
        survival_fit=False, embeddings_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
    ), hermes_home=str(hermes_home or path))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    engine.threshold_tokens = 8_000
    return engine


def _seed(path):
    path.mkdir()
    _state_db(path, [("S", None, None), ("B", None, None)])
    return _open(path)


def _publish(engine, shape="carrier"):
    raw = _view(6)
    if shape == "plain":
        raw = raw[:21]
        engine._config.fresh_tail_count = 1
    engine.ingest(raw)
    output = engine.compress(raw, force=True, current_tokens=count_messages_tokens(raw))
    assert engine._last_compress_leaves[1] == 1
    assert _covered(engine)
    assert output[0]["role"] == "user"
    assert bool(engine._generated_context_carrier_remainder(output[0])) == (shape == "carrier")
    assert output[0]["message_uid"]
    engine._config.leaf_chunk_tokens = 50_000
    engine._config.fresh_tail_count = 2
    return raw, output


def _split(engine, output):
    carrier = output[0]
    remainder = engine._generated_context_carrier_remainder(carrier)
    absorbed, = carrier["_absorbed_message_uids"]
    summary = carrier["content"][:-(len(remainder) + 2)]
    return {"role": "user", "content": summary, "message_uid": carrier["message_uid"]}, absorbed


def _shape(engine, raw, output, shape):
    if engine._generated_context_carrier_remainder(output[0]) is None:
        assert shape == "plain"
        return deepcopy(output)
    head, absorbed = _split(engine, output)
    opening = next(i for i, row in enumerate(raw) if row["message_uid"] == absorbed)
    if shape == "carrier":
        head = {
            **head,
            "content": head["content"] + "\n\n" + raw[opening]["content"],
            "_absorbed_message_uids": [absorbed],
        }
        opening += 1
    return [head, *deepcopy(raw[opening:])]


def _missing(calls=1):
    call, _ = tool_pair(CALL, "")
    call["content"] = ""
    call["message_uid"] = "c18-last-call"
    for i in range(1, calls):
        extra, _ = tool_pair(f"{CALL}-{i}", "")
        call["tool_calls"].extend(extra["tool_calls"])
    return call


def _new_user():
    return {"role": "user", "content": "Run the synthetic lookup.", "message_uid": "c18-final-user"}


def _assert_bound(engine, view):
    assert engine._host_uid_lineage_key()[0] is not None
    assert engine._host_uid_resolve(view), view


def _run(engine, view, *, tokens=None):
    result = engine.compress(view, current_tokens=count_messages_tokens(view) if tokens is None else tokens)
    assert not engine._last_compress_leaves or engine._last_compress_leaves[1] == 0
    return result


def _spy_finish(engine, monkeypatch):
    finish = engine._finish_without_leaf
    entries = []

    def capture(state, *args):
        scope = _UID_DROP_INPUT.get()
        entries.append((scope, state.force_overflow, state.pre_leaf_condensation_passes))
        return finish(state, *args)

    monkeypatch.setattr(engine, "_finish_without_leaf", capture)
    return entries


def _spy_decisions(engine, monkeypatch):
    replay = engine._uid_trimmed_base_replay
    decisions = []

    def capture(*args):
        decision = replay(*args)
        decisions.append((args[1], decision))
        return decision

    monkeypatch.setattr(engine, "_uid_trimmed_base_replay", capture)
    return decisions


def _assert_restored(engine, output, before):
    assert engine._missing_tool_result_stub(CALL) in output
    assert [row for row in output if row in before] == before
    assert engine._ingest_cursor == len(output)


def _pairs(output):
    return (
        [call["id"] for row in output for call in row.get("tool_calls", ())],
        [row["tool_call_id"] for row in output if row["role"] == "tool"],
    )


@pytest.mark.parametrize(("shape", "exit_case"), [
    ("plain", "fresh"), ("plain", "refused"), ("plain", "hidden"),
    ("carrier", "refused"), ("carrier", "two-absorbed"),
])
def test_restore_stubs(tmp_path, monkeypatch, summaries, shape, exit_case):
    _compare_no_leaf(tmp_path, monkeypatch, shape, exit_case)


def _compare_no_leaf(tmp_path, monkeypatch, shape, exit_case):
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    baseline = _no_leaf_arm(tmp_path / "unset", monkeypatch, shape, exit_case, on=False)
    current = _no_leaf_arm(tmp_path / "on", monkeypatch, shape, exit_case, on=True)
    base_view, base, base_status, base_reason, base_cursor, _, _ = baseline
    view, output, status, reason, cursor, scopes, drop_sets = current
    assert view == base_view
    unchanged = exit_case in {"refused", "two-absorbed"}
    assert (base is base_view) == unchanged
    if unchanged:
        assert output is view
        assert (status, reason) == (base_status, base_reason)
        assert CALL not in _pairs(output)[1]
    elif exit_case == "carrier-conflict":
        # The error exit returns a copy even when its rows stay equal.
        assert output == view == base
        assert (status, reason) == (base_status, base_reason)
        assert CALL not in _pairs(output)[1]
    else:
        assert _pairs(output) == _pairs(base)
        assert [row for row in output if row in view] == view
        assert CALL in _pairs(output)[1]
    assert cursor == len(output) and base_cursor == len(base)
    [scope] = scopes
    assert scope.uid_trimmed
    assert (scope.legacy_replay is not None) == (exit_case in {"plain-conflict", "missing-uid"})
    if drop_sets is not None:  # The HEAD receipts precede the extracted helper.
        assert drop_sets == ([set(), set()] if unchanged or exit_case == "carrier-conflict" else [{0}])
    print(f"{shape} {exit_case}: unset={'own' if base is base_view else 'equal copy' if base == base_view else 'changed'}; "
          f"mode={'own' if output is view else 'equal copy' if output == view else 'changed'}; "
          f"status/reason={base_status, base_reason, status, reason}; drops={drop_sets}")


def _no_leaf_arm(path, monkeypatch, shape, exit_case, *, on):
    path.mkdir()
    with monkeypatch.context() as fault, _closing(_open(path, hermes_home=path.parent)) as engine:
        hidden_calls = []
        conflicts = []
        if exit_case in {"plain-conflict", "missing-uid"}:
            view = _adoptable_plain(engine)
        else:
            view = _seed_no_leaf_view(engine, fault, shape, exit_case, hidden_calls)
        _assert_bound(engine, view)
        resolved = engine._host_uid_resolve(view)
        assert any(view[index].get("message_uid") and row["store_id"] > engine._last_compacted_store_id
                   for index, row in resolved.items())
        if exit_case.endswith("conflict"):
            engine._config.leaf_chunk_tokens = 1

            def refuse_publication(*args, **kwargs):
                conflicts.append(_UID_DROP_INPUT.get())
                raise sqlite3.OperationalError("database is locked")

            fault.setattr(engine, "_publish_leaf_node", refuse_publication)
        if exit_case == "missing-uid" and on:
            uid = view[1]["message_uid"]
            assembly = engine._assemble_context

            def lose_uid(*args, **kwargs):
                rows = [dict(row) for row in assembly(*args, **kwargs)]
                for row in rows:
                    if row.get("message_uid") == uid:
                        row.pop("message_uid")
                    elif uid in row.get("_absorbed_message_uids", ()):
                        row["_absorbed_message_uids"] = [item for item in row["_absorbed_message_uids"] if item != uid]
                return rows

            fault.setattr(engine, "_assemble_context", lose_uid)
        entries = _spy_finish(engine, fault)
        drops = None
        compute_drops = getattr(engine, "_uid_pretrim_replay_drops", None)
        if on and compute_drops is not None:
            drops = []

            def observe_drops(*args):
                decision = compute_drops(*args)
                drops.append(decision[0])
                return decision

            fault.setattr(engine, "_uid_pretrim_replay_drops", observe_drops)
        if not on:
            fault.delenv("LCM_HOST_MESSAGE_UID")
        fault.setattr(engine, "_c20_subject_pass", on, raising=False)
        output = _run(engine, view)
        if exit_case == "hidden" and on:
            assert len(hidden_calls) >= 1
        if exit_case.endswith("conflict"):
            assert conflicts and not entries
        else:
            assert len(entries) == 1 and not entries[0][1]
        scopes = conflicts or [scope for scope, _, _ in entries]
        return (view, output, engine._last_compression_status, engine._last_compression_noop_reason,
                engine._ingest_cursor, scopes, drops)


def _seed_no_leaf_view(engine, fault, shape, exit_case, hidden_calls):
    raw, published = _publish(engine, shape)
    view = _shape(engine, raw, published, shape)
    if shape == "carrier":
        # The split retains the opening user. A separate covered occurrence
        # lets UID trim run while the carrier itself remains in the view.
        view.insert(1, raw[0])
    if exit_case == "fresh":
        view = view[:2]  # The first durable row after the frontier is fresh.
    if exit_case == "hidden":
        later = _view(8)[21:]
        engine.ingest([*view, *later])
        visible = next(row for row in reversed(later) if row["role"] == "user")
        view = [view[0], visible]  # Omit the uncovered backlog before this bound row.

        def refuse_hidden(*args, **kwargs):
            assert engine._last_hidden_backlog
            hidden_calls.append(engine._last_hidden_backlog)
            raise SweepBudgetExhausted("foreground_soft_target")

        fault.setattr(engine, "_summarize_leaf_chunk_with_rescue", refuse_hidden)
    if exit_case == "refused":
        engine._config.survival_fit = True
        fault.setattr(engine, "_summary_route_available", lambda: False)
        if shape == "plain":
            view.extend(_view(8)[21:])
    if exit_case == "two-absorbed":
        view[0]["_absorbed_message_uids"].append("c18-unbound")
    if exit_case not in {"fresh", "hidden"}:
        view.append(_new_user())
    view.append(_missing())
    if exit_case in {"fresh", "hidden"}:
        engine._config.fresh_tail_count = 2
    return view


@pytest.mark.parametrize("victim", ["uidless-tail", "summary"])
def test_cap_preserves_uncovered_context(tmp_path, monkeypatch, summaries, victim):
    with _closing(_seed(tmp_path / "store")) as engine:
        raw, published = _publish(engine, "plain")
        view = _shape(engine, raw, published, "plain")
        # A small unbound tail between the head and the newest unanswered tool
        # group exposes the selector's reserve for missing results.
        older = view[1]
        newest = {"role": "user", "content": "Newest instruction.", "message_uid": "c18-newest"}
        view.extend([newest, _missing(calls=3 if victim == "uidless-tail" else 1)])
        engine.ingest(view)
        if victim == "uidless-tail":
            older.pop("message_uid")
        engine._config.fresh_tail_count = 1
        _assert_bound(engine, view)
        observed = count_messages_tokens(view)
        engine._config.max_assembly_tokens = observed + 1
        assert observed < engine._effective_assembly_token_cap()
        entries = _spy_finish(engine, monkeypatch)
        output = _run(engine, view, tokens=observed)
        [(scope, forced, _)] = entries
        assert scope.uid_trimmed and scope.legacy_replay is not None and not forced
        if victim == "uidless-tail":
            assert older in output
        else:
            assert {row["message_uid"] for row in view[1:]} <= {row.get("message_uid") for row in output}
            assert _output_sources(engine, output)


def _carrier_replay(engine, origin, *, re_glue=True, two_leaves=False, coalesce_replies=False):
    # The first pass has a sole real opening user. Its retained anchor is
    # below the frontier but absent from the new leaf's source IDs.
    raw = _view(6)
    opening = raw[0]
    replies = [row for row in raw if row["role"] != "user"]
    if coalesce_replies:
        replies = engine._sanitize_active_context_messages(replies, insert_missing_tool_stubs=False)
    retained_pass = [{"role": "system", "content": "Synthetic test system."}, opening, *replies]
    engine._config.fresh_tail_count = 3
    if two_leaves:
        engine._config.dynamic_leaf_chunk_max = 480
    engine.ingest(retained_pass)
    engine.compress(retained_pass, force=True, current_tokens=count_messages_tokens(retained_pass))
    assert engine._last_compress_leaves[1] == 1
    if two_leaves:
        engine.compress(retained_pass, force=True, current_tokens=count_messages_tokens(retained_pass))
        assert engine._last_compress_leaves[1] == 1
        assert len(engine._dag.get_uncondensed_at_depth("S", 0)) == 2
    opening_id = engine._host_uid_resolve([opening])[0]["store_id"]
    assert opening_id < engine._last_compacted_store_id and opening_id not in _covered(engine)
    engine._config.leaf_chunk_tokens = 50_000
    plain = {"role": "user", "content": _summary_part_text(engine._dag.get_node(1))}
    later = [raw[4], raw[8]] if re_glue else []
    if origin == "engine":
        previous = _run(engine, [plain, opening, *replies, raw[4], raw[8], _missing()])
        head = previous[0]
        assert head.get("_absorbed_message_uids") == [opening["message_uid"]]
    else:
        previous = _run(engine, [plain, opening, *replies])
        summary, remainder = previous[:2]
        assert _output_sources(engine, [summary]) and remainder["content"] == opening["content"]
        head = {**summary, "content": summary["content"] + "\n\n" + opening["content"],
                "_absorbed_message_uids": [opening["message_uid"]]}
        engine.ingest([*previous, *later])
    return [head, *replies, *later, _missing()]


def _binding_records(engine, uid):
    return engine._store.connection.execute(
        "SELECT kind, proof_kind, binding_version FROM host_uid_bindings WHERE lineage_key=? AND uid=? "
        "ORDER BY kind, proof_kind, binding_version", (engine._host_uid_lineage_key()[0], uid),
    ).fetchall()


@pytest.mark.parametrize("origin", ["host", "engine"])
@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_carrier_adopts_with_proven_engine_uid(tmp_path, monkeypatch, summaries, origin, cold):
    path = tmp_path / "store"
    engine = _seed(path)
    try:
        view = _carrier_replay(engine, origin)
        if cold:
            if origin == "host":
                # The host has already used this merged row while the original
                # binding was live. Exercise restart with that durable proof.
                _assert_bound(engine, view)
                _run(engine, view)
            engine.shutdown()
            engine = _open(path)
            engine._config.leaf_chunk_tokens = 50_000
            engine._config.fresh_tail_count = 3
        _assert_bound(engine, view)
        users = [row for row in view[2:] if row["role"] == "user"]
        for row in users[-2:]:
            row["api_content"] = row["content"] + "\nHost sidecar."
        own = view[0]["message_uid"]
        absorbed, = view[0]["_absorbed_message_uids"]
        decisions = _spy_decisions(engine, monkeypatch)
        entries = _spy_finish(engine, monkeypatch)
        observations = []
        lookup = engine._host_uid_engine_uids

        def observe(uids, **kwargs):
            known = lookup(uids, **kwargs)
            if _UID_DROP_INPUT.get() and set(uids) == {own, absorbed}:
                observations.append(known)
            return known

        monkeypatch.setattr(engine, "_host_uid_engine_uids", observe)
        with (
            patch.object(engine, "_assemble_context", wraps=engine._assemble_context) as assembly,
            patch.object(engine, "_reconcile_ingest_cursor_from_store",
                         wraps=engine._reconcile_ingest_cursor_from_store) as reconcile,
        ):
            output = _run(engine, view)
        known = lookup({own, absorbed})
        records = [tuple(record) for record in _binding_records(engine, own)]
        print(f"{origin} cold={cold} known={known!r} own={own} bindings={records} lookups={observations}")
        assert known == {own}
        if cold:
            assert reconcile.call_count >= 1
        [(scope, _, _)] = entries
        assert scope.uid_trimmed and scope.removed
        assert len(decisions) == 1 and decisions[0][1] is not None
        assert observations == [{own}]
        assert assembly.call_count == 1
        remainder = {"role": "user", "content": engine._generated_context_carrier_remainder(view[0]),
                     "message_uid": absorbed}
        assert assembly.call_args.args[1][0] == remainder
        assert output[0]["_absorbed_message_uids"] == [absorbed]
        assert output[0].keys() == {"role", "content", "message_uid", "_absorbed_message_uids"}
        by_uid = {row["message_uid"]: row for row in view[1:] if "message_uid" in row}
        kept = [row for row in output[1:] if row.get("message_uid") in by_uid]
        assert all(row == by_uid[row["message_uid"]] for row in kept)
        assert all(row in kept for row in users[-2:])
        assert engine._missing_tool_result_stub(CALL) in output
        assert engine._ingest_cursor == len(output)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("uid_kind", ["host-own", "engine-absorbed"])
def test_carrier_declines_ambiguous_uid(tmp_path, monkeypatch, summaries, uid_kind):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine)
        head = dict(view[1], content=view[0]["content"] + "\n\nA quoted summary and a new question.")
        if uid_kind == "host-own":
            head["message_uid"] = "c18-host-quotation"
        else:
            own = view[0]["message_uid"]
            other = {"role": "user", "content": "Another synthetic engine emission."}
            engine._mint_engine_uids([(other, "summary", None, "summary")])
            engine._host_uid_record_engine([other])
            absorbed = other["message_uid"]
            assert engine._host_uid_engine_uids({own, absorbed}) == {own, absorbed}
            head.update(message_uid=own, _absorbed_message_uids=[absorbed])
        view.insert(1, head)
        _assert_bound(engine, view)
        decisions = _spy_decisions(engine, monkeypatch)
        output = _run(engine, view)
        assert decisions == [(False, None)]
        assert head in output


def _adoptable_plain(engine, *, covered_replay=False):
    raw, published = _publish(engine, "plain")
    # A complete covered last turn follows the head when the guard must
    # verify the coverage it is about to drop. Keep its tool pair whole.
    older = raw[16:20] if covered_replay else []
    later = _view(7)[21:]
    view = [published[0], *older, *published[1:], *later, _new_user(), _missing()]
    _assert_bound(engine, view)
    return view


def test_wrapper_stubs_after_publication_conflict(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine)
        engine._config.leaf_chunk_tokens = 1
        failures = []

        def refuse_publication(*args, **kwargs):
            failures.append(_UID_DROP_INPUT.get())
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, "_publish_leaf_node", refuse_publication)
        with patch.object(engine, "_finish_without_leaf", wraps=engine._finish_without_leaf) as finish:
            output = _run(engine, view)
        [scope] = failures
        assert finish.call_count == 0 and scope.uid_trimmed
        _assert_restored(engine, output, scope.before)
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID")
        assert _pairs(output) == _pairs(_run(engine, view))


def test_leaf_guard_wrapper_stubs_after_refusal(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine, covered_replay=True)
        engine._config.leaf_chunk_tokens = 1
        verdicts = _observe_guard(engine, monkeypatch, deny_coverage=True)
        trimmed = []
        guard = engine._guard_uid_covered_drop

        def record(rows):
            trimmed.append(_UID_DROP_INPUT.get())
            return guard(rows)

        monkeypatch.setattr(engine, "_guard_uid_covered_drop", record)
        with patch.object(engine, "_sanitize_active_context_messages",
                          wraps=engine._sanitize_active_context_messages) as sanitize:
            output = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert engine._last_compress_leaves[1] == 1
        [scope] = trimmed
        assert scope.uid_trimmed
        [(argument, verdict)] = verdicts
        assert verdict is not argument
        assert any(call.args[0] is scope.before and call.kwargs.get("insert_missing_tool_stubs") is True
                   for call in sanitize.call_args_list)
        _assert_restored(engine, output, scope.before)


def _observe_guard(engine, monkeypatch, *, deny_coverage=False):
    guard = engine._guard_uid_covered_drop
    verdicts = []

    def capture(rows):
        with monkeypatch.context() as fault:
            if deny_coverage:
                fault.setattr(engine._dag, "source_message_ids", lambda *args, **kwargs: [])
            result = guard(rows)
        verdicts.append((rows, result))
        return result

    monkeypatch.setattr(engine, "_guard_uid_covered_drop", capture)
    return verdicts


def test_real_guard_refuses_uncovered_drop(tmp_path, monkeypatch, summaries, caplog):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine, covered_replay=True)
        entries = _spy_finish(engine, monkeypatch)
        verdicts = _observe_guard(engine, monkeypatch, deny_coverage=True)
        with caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
            output = _run(engine, view)
        [(scope, forced, _)] = entries
        assert scope.legacy_replay and not forced
        [(argument, verdict)] = verdicts
        assert verdict is not argument
        assert _guard_counts(caplog)[1] > 0
        _assert_restored(engine, output, scope.before)


def test_exception_after_adoption_resets_scope(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine)
        scopes = []

        def broken_assembly(*args, **kwargs):
            scopes.append(_UID_DROP_INPUT.get())
            raise ValueError("c18 synthetic assembly failure")

        monkeypatch.setattr(engine, "_assemble_context", broken_assembly)
        with pytest.raises(ValueError, match="c18 synthetic assembly failure"):
            _run(engine, view)
        [scope] = scopes
        assert scope.uid_trimmed and scope.legacy_replay is not None
        assert engine._last_compression_status == "error"
        assert _UID_DROP_INPUT.get() is None


def test_binding_change_before_adoption(tmp_path, monkeypatch, summaries):
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
        with patch.object(engine, "_assemble_context", wraps=engine._assemble_context) as assembly:
            assert _run(engine, view) is view
        [(action, output)] = exits
        assert action == "return" and output is view
        assert engine._session_id == "B"
        assert assembly.call_count == 0


def _refuse_leaf(*args, **kwargs):
    raise SweepBudgetExhausted("foreground_soft_target")


def test_forced_overflow_starts_from_pretrim_view(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine, covered_replay=True)
        engine._config.max_assembly_tokens = count_messages_tokens(view)
        entries = _spy_finish(engine, monkeypatch)
        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _refuse_leaf)
        with patch.object(engine, "_assemble_overflow_recovery_context",
                          wraps=engine._assemble_overflow_recovery_context) as overflow:
            _run(engine, view)
        [(scope, forced, _)] = entries
        assert scope.uid_trimmed and scope.legacy_replay and forced
        assert overflow.call_count >= 1
        assert overflow.call_args_list[0].args[1] == scope.before


@pytest.mark.parametrize("shape", SHAPES)
def test_cold_bind_matches_unset(tmp_path, monkeypatch, summaries, shape):
    path = tmp_path / "store"
    with _closing(_seed(path)) as warm:
        _, host = _publish(warm, shape)
        proof = warm._active_emission_proof()
        assert proof
        host[0]["display_metadata"] = {"host_display_label": "Synthetic earlier context"}
    snapshot = tmp_path / "original"
    shutil.copytree(path, snapshot)
    inputs = []
    for suffix in ("on", "on-proof", "unset"):
        # Keep the state.db path (part of lineage) identical across binds.
        shutil.copytree(snapshot, path, dirs_exist_ok=True)
        if suffix == "unset":
            monkeypatch.delenv("LCM_HOST_MESSAGE_UID")
        with _closing(_open(path)) as cold:
            cold._config.leaf_chunk_tokens = 50_000
            cold._config.fresh_tail_count = 1 if shape == "plain" else 2
            if suffix != "unset":
                _assert_bound(cold, host)
            if suffix == "on-proof":
                cold._last_emission_descriptors = proof
            view = deepcopy(host)
            output = _run(cold, view)
            inputs.append((output, output is view))
    print(f"cold {shape} caller-list identity: {[same for _, same in inputs]}")
    assert inputs[0] == inputs[1] == inputs[2]


def test_adopted_assembly_preserves_no_cap_argument(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _adoptable_plain(engine)
        assert engine._effective_assembly_token_cap() is None
        with patch.object(engine, "_assemble_context", wraps=engine._assemble_context) as assembly:
            _run(engine, view)
        assert assembly.call_count == 1
        assert assembly.call_args.kwargs["assembly_cap_override"] is None


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fault", ["aged-result", "missing-uid"])
def test_adopted_assembly_on_production_shape(tmp_path, monkeypatch, summaries, shape, fault):
    with _closing(_seed(tmp_path / "store")) as engine:
        if shape == "carrier":
            view = _carrier_replay(engine, "engine")
            first_uid, = view[0]["_absorbed_message_uids"]
        else:
            view = _adoptable_plain(engine)
            first_uid = view[1]["message_uid"]
        _assert_bound(engine, view)
        resolved = engine._host_uid_resolve(view)
        first_tool = next(
            row for index, row in enumerate(view) if row["role"] == "tool"
            and (index not in resolved or resolved[index]["store_id"] > engine._last_compacted_store_id)
        )
        if fault == "aged-result":
            engine._config.large_output_active_replay_stubbing_enabled = True
            engine._config.large_output_externalization_enabled = True
            engine._config.large_output_active_replay_stub_aged_threshold_tokens = 10
            engine._config.large_output_active_replay_stub_threshold_tokens = 10_000
        else:
            assembly = engine._assemble_context

            def lose_uid(*args, **kwargs):
                rows = [dict(row) for row in assembly(*args, **kwargs)]
                for row in rows:
                    if row.get("message_uid") == first_uid:
                        row.pop("message_uid")
                    elif first_uid in row.get("_absorbed_message_uids", ()):
                        row["_absorbed_message_uids"] = [uid for uid in row["_absorbed_message_uids"] if uid != first_uid]
                return rows

            monkeypatch.setattr(engine, "_assemble_context", lose_uid)
        verdicts = _observe_guard(engine, monkeypatch)
        entries = _spy_finish(engine, monkeypatch)
        output = _run(engine, view)
        [(scope, _, _)] = entries
        assert scope.legacy_replay
        [(argument, verdict)] = verdicts
        assert verdict is argument
        if fault == "missing-uid":
            assert output == engine._sanitize_active_context_messages(scope.before, insert_missing_tool_stubs=True)
            assert engine._missing_tool_result_stub(CALL) in output
            assert engine._ingest_cursor == len(output)
        else:
            stub = next(row for row in output if row.get("tool_call_id") == first_tool["tool_call_id"])
            assert stub["content"].startswith(STUB)
            assert stub["message_uid"] == first_tool["message_uid"]


def test_carrier_emits_standalone_remainder_keys(tmp_path, monkeypatch, summaries):
    with _closing(_seed(tmp_path / "store")) as engine:
        view = _carrier_replay(engine, "engine", re_glue=False)
        _assert_bound(engine, view)
        absorbed, = view[0]["_absorbed_message_uids"]
        remainder = engine._generated_context_carrier_remainder(view[0])
        decisions = _spy_decisions(engine, monkeypatch)
        output = _run(engine, view)
        assert len(decisions) == 1 and decisions[0][1] is not None
        assert _output_sources(engine, [output[0]])
        assert output[1] == {"role": "user", "content": remainder, "message_uid": absorbed}


@pytest.mark.parametrize("shape", SHAPES)
def test_pre_leaf_condensation_is_adopted(tmp_path, monkeypatch, summaries, shape):
    with _closing(_seed(tmp_path / "store")) as engine:
        if shape == "carrier":
            view = _carrier_replay(engine, "engine", two_leaves=True)
        else:
            _, published = _publish(engine, "plain")
            history = [*published, *_view(9)[21:33]]
            engine._config.leaf_chunk_tokens = 1
            engine._config.fresh_tail_count = 1
            published = engine.compress(history, force=True, current_tokens=count_messages_tokens(history))
            assert engine._last_compress_leaves[1] == 1
            assert engine._generated_context_carrier_remainder(published[0]) is None
            view = [*published, *_view(9)[33:], _new_user(), _missing()]
        assert len(engine._dag.get_uncondensed_at_depth("S", 0)) == 2
        _assert_bound(engine, view)
        engine._config.threshold_full_sweep_enabled = True
        engine._config.condensation_fanin = 2
        engine._config.summary_prefix_target_tokens = 1
        engine.threshold_tokens = 100
        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _refuse_leaf)
        entries = _spy_finish(engine, monkeypatch)
        with patch.object(engine, "_assemble_context", wraps=engine._assemble_context) as assembly:
            output = _run(engine, view)
        [(scope, forced, condensations)] = entries
        assert scope.legacy_replay and not forced and condensations == 1
        assert assembly.call_count == 1
        assert "[Session Arc Summary (d1," in output[0]["content"]
        assert engine._missing_tool_result_stub(CALL) in output
