"""UID-bound replay and store-complete closure regressions from the c16b review."""

from copy import deepcopy

import pytest

from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.invariants.test_issue_2781_placeholder_walk import (
    LOG,
    PAD,
    PRUNED,
    _bindings,
    _canonical,
    _copy,
    _covered,
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
from tests.invariants.test_issue_2815_c16b import _resolved_ids, _words

pytestmark = pytest.mark.issue(2815)


def _bind_and_resolve(engine, stored):
    ids = [_canonical(engine, row) for row in stored]
    assert ids == sorted(ids)
    assert _resolved_ids(engine, stored) == ids
    return ids


def _shortened_shape(engine, *, retained, uid_form):
    prefix, summary = _opening(engine, [])
    prior, blocker, later = [_msg(name, "user") for name in ("prior", "blocker", "later")]
    caller, result = _group("shortened-call", 1)
    caller["content"] = "Full text before the call." + PAD * 4
    short = dict(caller, content="short")
    if uid_form == "no-uid":
        short = _no_uid(short)
    elif uid_form == "fresh-uid":
        short["message_uid"] = "fresh-never-ingested"
    stored = [prior, *([blocker] if retained else []), caller, result, later]
    engine.ingest([*prefix, *stored])
    _bind_and_resolve(engine, stored)
    return summary, prior, short, result, later, blocker


@pytest.mark.parametrize("uid_form", ["no-uid", "same-uid", "fresh-uid"])
@pytest.mark.parametrize("retained", [True, False], ids=["blocker", "budget"])
def test_e1_shortened_caller_walk(tmp_path, monkeypatch, summaries, retained, uid_form):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        _, prior, short, result, later, blocker = _shortened_shape(engine, retained=retained, uid_form=uid_form)
        chunk = [prior, short, result, later]
        view = [*chunk, blocker] if retained else chunk
        mapping = engine._get_store_id_map_for_messages(view)
        assert id(short) not in mapping
        assert mapping[id(result)] == _canonical(engine, result)
        budget = count_messages_tokens(chunk if retained else [prior, short]) + (0 if retained else 5)
        leaf = engine._store_complete_input(chunk, {}, mapping, view, chunk, _frontier(engine), (), budget)
        assert [row for row, _ in leaf] == [prior]


def test_e1_shortened_caller_return(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        summary, prior, short, result, later, blocker = _shortened_shape(engine, retained=True, uid_form="no-uid")
        tail = _msg("tail")
        engine.ingest([tail])
        frontier = _frontier(engine)
        output = _pass(engine, deepcopy([summary, prior, short, result, later, blocker, tail]))
        sources = _present_node_sources(engine, output) - set(range(1, frontier + 1))
        assert (sources, short in output, result in output) == ({_canonical(engine, prior)}, True, True)


@pytest.mark.parametrize("size", ["one", "group", "all"])
def test_e2_live_answer_after_hidden_caller_copy(tmp_path, monkeypatch, summaries, size):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [])
        prior, blocker, later, tail = [_msg(name, "user") for name in ("prior", "blocker", "later", "tail")]
        caller, result = _group("hidden-copy", 1)
        stored = [prior, caller, blocker, _no_uid(caller), result, later, tail]
        engine.ingest([*prefix, *stored])
        ids = _bind_and_resolve(engine, [prior, caller, blocker, result, later, tail])
        rows = _rows(engine)[len(prefix):]
        assert [row["store_id"] for row in rows] == [*ids[:3], ids[2] + 1, *ids[3:]]
        assert engine._stored_row_forms(rows[3]) == engine._stored_row_forms(rows[1])
        engine._config.leaf_chunk_tokens = {
            "one": 1,
            "group": count_messages_tokens([prior, caller, result]) + 5,
            "all": count_messages_tokens([prior, caller, result, later]) - 5,
        }[size]
        frontier = _frontier(engine)
        output = _pass(engine, [summary, prior, caller, result, later, blocker, tail])
        sources = _present_node_sources(engine, output) - set(range(1, frontier + 1))
        assert (sources, caller in output, result in output) == ({ids[0]}, True, True)


@pytest.mark.parametrize("pruned", [False, True], ids=["full-result", "pruned-placeholder"])
def test_e3_live_answer_after_copy_exclusion(tmp_path, monkeypatch, summaries, pruned):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [_msg("initial")])
        caller, result = _group("excluded-copy", 1)
        prior = _msg("prior", "user")
        retained, later, tail = [_msg(name) for name in ("retained", "later", "tail")]
        engine.ingest([*prefix, prior, caller, _no_uid(caller), retained, result, later, tail])
        ids = _bind_and_resolve(engine, [prior, caller, retained, result, later, tail])
        rows = _rows(engine)[len(prefix):]
        assert [row["store_id"] for row in rows] == [*ids[:2], ids[1] + 1, *ids[2:]]
        assert engine._stored_row_forms(rows[2]) == engine._stored_row_forms(rows[1])
        live = dict(result, content=PRUNED) if pruned else result
        assert _resolved_ids(engine, [caller, live]) == [ids[1], ids[3]]
        view = [prior, caller, live, later, retained, tail]
        # Keep the result outside the selected chunk to isolate the stored bound.
        chunk = [prior, caller, later]
        mapping = engine._get_store_id_map_for_messages(view)
        leaf = engine._store_complete_input(
            chunk, {}, mapping, view, chunk, _frontier(engine), (), count_messages_tokens(chunk))
        output = _pass(engine, [summary, *view])
        assert caller in output and live in output, (
            _present_node_sources(engine, output), engine._last_compression_noop_reason)
        assert [row for row, _ in leaf] == [prior]


@pytest.mark.parametrize("bind_version", [False, True], ids=["no-version-binding", "version-bound"])
def test_e4_warm_ingest_keeps_the_resolved_caller(tmp_path, monkeypatch, summaries, bind_version):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        caller, result = _group("call", 1)
        caller["timestamp"] = 1_790_000_000.0
        result["timestamp"] = 1_790_000_001.0
        earlier = _group("earlier", ids=[result["tool_call_id"]])
        prefix, summary = _opening(engine, earlier)
        assert len(earlier[1]["content"]) > 200 and earlier[1]["content"] != result["content"]
        assert _canonical(engine, earlier[1]) in _covered(engine)
        retained, after, tail = _msg("retained", "user"), _msg("after"), _msg("tail")
        later = [_msg("later", "user"), _msg("latest")]
        copy_caller, version_row = _copy([caller, result], 1)
        copy_caller.pop("timestamp")
        version_row.pop("timestamp")
        replay_caller, _ = _group("replay-caller", ids=[result["tool_call_id"]])
        engine.ingest([*prefix, caller, result, retained, copy_caller, version_row, replay_caller, after, tail, *later])
        canonical = _canonical(engine, result)
        version = _rows(engine)[len(prefix) + 4]["store_id"]
        assert engine._store.get(version)["content"] == PRUNED
        expected_bindings = [(canonical, "canonical")]
        if bind_version:
            engine._store.add_host_uid_bindings(_lineage(engine), [
                (version, result["message_uid"], "version", "version_new"),
            ])
            expected_bindings.append((version, "version"))
        assert _bindings(engine, [result])[result["message_uid"]] == expected_bindings
        placeholder = dict(result, content=PRUNED)
        placeholder.pop("timestamp")
        assert _resolved_ids(engine, [replay_caller, placeholder]) == [_canonical(engine, replay_caller), canonical]
        first = _pass(engine, [summary, caller, placeholder, after, retained, tail])
        assert canonical in _present_node_sources(engine, first)
        assert engine._ingest_cursor == len(first)
        live = [*first, _no_uid(caller), _no_uid(result), replay_caller, placeholder, *later]
        before = {row["store_id"] for row in _rows(engine)}
        engine.ingest(live)
        stored = [(row["role"], row["content"], row.get("tool_call_id"))
                  for row in _rows(engine) if row["store_id"] not in before]
        assert stored == ([] if bind_version else [("tool", PRUNED, result["tool_call_id"])])


@pytest.mark.parametrize(("mult", "reordered", "expected"), [
    (0.5, True, "closed"), (0.8, True, "closed"), (1.1, True, "closed"),
    (6, True, "stall"), (0.8, False, "contiguous"),
], ids=["lift-low", "lift-mid", "lift-high", "stall", "contiguous"])
def test_e5_restore_yields_to_budget_lift(tmp_path, monkeypatch, summaries, caplog, mult, reordered, expected):
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
        big = _msg("big", content=_words("bigrow", int(budget * mult)))
        t = _msg("t", "user", _words("tango", 120))
        tail = [_msg("tail1"), _msg("tail2", "user")]
        stored = [x, p, caller, q, big, answer, t, *tail]
        engine.ingest([*prefix, *stored])
        ids = _bind_and_resolve(engine, stored)
        assert engine._context_aware_leaf_cap() is None
        ratio = count_message_tokens(big) / budget
        assert (17 < ratio < 18) if expected == "stall" else (1.3 < ratio < 3.01)
        engine._config.leaf_chunk_tokens = budget
        view = [summary, *([p, q, x] if reordered else [x, p, q]), t, *tail]
        output = _pass(engine, deepcopy(view))
        sources = _present_node_sources(engine, output) - set(range(1, frontier + 1))
        assert (engine._last_compress_leaves[1], sources) == (
            1, set(ids[:{"closed": 7, "stall": 4, "contiguous": 2}[expected]]))
        assert ("reason=contiguity" in caplog.text) == (expected == "stall")
