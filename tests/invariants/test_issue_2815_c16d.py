"""Regressions for skipped answers and publishable prefixes in the UID walk."""

from copy import deepcopy

import pytest

from hermes_lcm.engine import _summary_part_text
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.invariants.test_issue_2781_placeholder_walk import (
    LOG,
    PAD,
    PRUNED,
    _frontier,
    _group,
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
from tests.invariants.test_issue_2815_c16c import _bind_and_resolve

pytestmark = pytest.mark.issue(2815)


def _copy_shape(engine, *, copies, placeholder_copy=False):
    prefix, summary = _opening(engine, [])
    prior, blocker, later, tail = [_msg(name, "user") for name in ("prior", "blocker", "later", "tail")]
    caller, result = _group("call", 1)
    caller["content"] = "Full text before the call." + PAD * 4
    skipped = [_no_uid(caller), _no_uid(dict(result, content=PRUNED) if placeholder_copy else result)] if copies else []
    engine.ingest([*prefix, prior, caller, blocker, result, *skipped, later, tail])
    ids = _bind_and_resolve(engine, [prior, caller, blocker, result, later, tail])
    stored = _rows(engine)[len(prefix):]
    skipped_ids = [ids[3] + 1, ids[3] + 2] if copies else []
    assert [row["store_id"] for row in stored] == [*ids[:4], *skipped_ids, *ids[4:]]
    if copies:
        engine._dag.add_node(SummaryNode(session_id="OTHER", summary="Previously covered copy.",
                                         token_count=4, source_token_count=100, source_ids=skipped_ids))
        assert engine._store_complete_node_covered(set(ids) | set(skipped_ids)) == set(skipped_ids)
    return summary, prior, caller, blocker, result, later, tail, ids


@pytest.mark.parametrize("copies", [True, False], ids=["covered-copy", "no-copy"])
def test_h1_skipped_group_copy_walk(tmp_path, monkeypatch, summaries, copies):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        _, prior, caller, blocker, result, later, _, ids = _copy_shape(engine, copies=copies)
        chunk = [prior, caller, result, later]
        view = [*chunk, blocker]
        mapping = engine._get_store_id_map_for_messages(view)
        assert [mapping[id(row)] for row in chunk] == [ids[0], ids[1], ids[3], ids[4]]
        leaf = engine._store_complete_input(chunk, {}, mapping, view, chunk, _frontier(engine), (),
                                            count_messages_tokens(chunk))
        assert [row for row, _ in leaf] == [prior]


@pytest.mark.parametrize("copies", [True, False], ids=["covered-copy", "no-copy"])
def test_h1_skipped_group_copy_return(tmp_path, monkeypatch, summaries, copies):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        summary, prior, caller, blocker, result, later, tail, ids = _copy_shape(engine, copies=copies)
        engine._config.leaf_chunk_tokens = count_messages_tokens([prior, caller, result, later])
        frontier = _frontier(engine)
        output = _pass(engine, [summary, prior, caller, result, later, blocker, tail])
        sources = {sid for sid in _present_node_sources(engine, output) if sid > frontier}
        assert (sources, caller in output, result in output) == ({ids[0]}, True, True)


@pytest.mark.parametrize("pruned", [False, True], ids=["full-result", "pruned-placeholder"])
def test_h2_skipped_placeholder_copy_return(tmp_path, monkeypatch, summaries, pruned):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        summary, prior, caller, blocker, result, later, tail, ids = _copy_shape(
            engine, copies=True, placeholder_copy=True)
        live = dict(result, content=PRUNED) if pruned else result
        assert _resolved_ids(engine, [caller, live]) == [ids[1], ids[3]]
        engine._config.leaf_chunk_tokens = count_messages_tokens([prior, caller, live, later])
        output = _pass(engine, [summary, prior, caller, live, later, blocker, tail])
        sources = _present_node_sources(engine, output)
        assert (caller in output and live in output) or {ids[1], ids[3]} <= sources, (
            caller in output, live in output, sorted(sources))


def _restore_shape(engine, form, mult):
    prefix, summary = _opening(engine, [])
    x = _msg("x", "user", _words("xray", 120))
    p = _msg("p", "assistant", _words("papa", 120))
    caller, answer = _group("hidden", 1)
    q = _msg("q", "user", _words("quebec", 120))
    u = _msg("u", "user", "u-short note")
    base = sum(count_message_tokens(row) for row in (x, p, caller, q)) + 40
    big = _msg("big", content=_words("bigrow", int(base * mult)))
    t = _msg("t", "user", _words("tango", 120))
    tail = [_msg("tail1"), _msg("tail2", "user")]
    through_q = [*([u] if form == "lower" else []), x, *([u] if form == "carrier" else []), p, caller, q]
    stored = [*through_q, big, answer, t, *tail]
    engine.ingest([*prefix, *stored])
    ids = _bind_and_resolve(engine, stored)
    budget = base + (count_message_tokens(u) if form != "none" else 0)
    engine._config.leaf_chunk_tokens = budget
    if form == "carrier":
        seed, = engine._dag.get_session_nodes("S")
        carrier = dict(u, content=_summary_part_text(seed) + "\n\n" + u["content"])
        assert engine._generated_context_carrier_remainder(carrier) == u["content"]
        head = [carrier]
    else:
        head = [summary, *([u] if form != "none" else [])]
    return [*head, p, q, x, t, *tail], ids[:len(through_q)], ids[:-2], [t, *tail], u


def _two_passes(engine, view, caplog):
    start = _frontier(engine)
    passes = []
    for _ in range(2):
        caplog.clear()
        view = _pass(engine, deepcopy(view))
        sources = {sid for sid in _present_node_sources(engine, view) if sid > start}
        open_group = any("first tool group remains open" in rec.message and "reason=contiguity" in rec.message
                         for rec in caplog.records)
        passes.append((_frontier(engine), sources, open_group, view))
    print("passes:", [(end, sorted(sources), open_group,
                      [row.get("message_uid") for row in output])
                     for end, sources, open_group, output in passes])
    return passes


@pytest.mark.parametrize(("form", "mult"), [
    ("unmapped", 0.8), ("unmapped", 6), ("carrier", 0.8), ("carrier", 6), ("none", 0.8), ("lower", 0.8),
], ids=["unmapped-low", "unmapped-high", "carrier-low", "carrier-high", "no-insertion", "lower-source-observation"])
def test_h5_restore_across_two_passes(tmp_path, monkeypatch, summaries, caplog, form, mult):
    _mode(monkeypatch, "on")
    caplog.set_level("INFO", logger=LOG)
    with _open(tmp_path / "store") as engine:
        view, through_q, closed, retained, u = _restore_shape(engine, form, mult)
        previous_summaries = len(summaries)
        passes = _two_passes(engine, view, caplog)
        end, sources, open_group, _ = passes[0]
        if form == "none":
            assert (end, sources, open_group) == (closed[-1], set(closed), False)
        elif form != "lower":
            assert (end, sources, open_group) == (through_q[-1], set(through_q), True)
            # The surviving user turn can be glued to the returned summary.
            assert all(any(row["content"] in message["content"] for message in output)
                       for _, _, _, output in passes for row in retained)
            assert u["content"] in summaries[previous_summaries]
            print("inserted content in returned rows:",
                  [any(u["content"] in message["content"] for message in output) for _, _, _, output in passes])


def test_h6_restore_prefix_with_sources(tmp_path, monkeypatch, summaries, caplog):
    _mode(monkeypatch, "on")
    caplog.set_level("INFO", logger=LOG)
    with _open(tmp_path / "store") as engine:
        view, through_q, _, _, _ = _restore_shape(engine, "lower", 6)
        assert engine._retained_escape_enabled and engine._context_aware_leaf_cap() is None
        walks = []
        original = engine._store_complete_input

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            walks.append((kwargs.get("allow_retained_escape"), kwargs.get("lift_allowed", True),
                          engine._store_complete_blocked))
            return result

        monkeypatch.setattr(engine, "_store_complete_input", observe)
        start = _frontier(engine)
        output = _pass(engine, view)
        sources = {sid for sid in _present_node_sources(engine, output) if sid > start}
        assert (True, True, None) in walks
        assert (_frontier(engine), sources, "reason=contiguity" in caplog.text) == (
            through_q[-1], set(through_q), True)
