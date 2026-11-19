"""#2896 (v0.30.2): main's cut first; the notice block only on that same cut, never a deeper one (C2 to C8).

"Main's list" here is this tree with the block step turned off (``_survival_place_block`` returning its input): the
rest of the fit is main's code, so its rows are main's. The two-tree comparison against a real copy of main is
tests/test_issue_2896_kept_rows_equal_main.py (C1, C11); the main values named below are the review probes' runs on
main. Synthetic data only.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import types

import pytest

import tests.conftest  # noqa: F401  (registers hermes_lcm)
import hermes_lcm.engine as lcm_engine
import hermes_lcm.store_complete as lcm_store_complete
from hermes_lcm.emission_projection import survival_notice_block
from tests import notice_fit_cases as cases
from tests.test_issue_1124_copy_exclusion import _append, _frontier, _publish, _tagged_segment, _text
from tests.test_issue_1124_copy_exclusion import engine as engine_1124  # noqa: F401 (the 1124 file's fixture)

NOT_FIT = "the notice does not fit beside the kept turns"
NO_CARRIER = "no system row and no glued carrier for the notice"


@pytest.fixture
def host(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = cases.rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)


@pytest.fixture
def as_main(monkeypatch):
    """A switch: ``as_main(engine)`` turns that engine's block step off, so it returns main's list."""
    def switch(e):
        monkeypatch.setattr(e, "_survival_place_block", lambda fitted, *args, **kwargs: fitted)
        return e
    return switch


def _fit_at(tmp_path, shape, seed, cap, after_exception=True, main=None, name="e"):
    e = cases.engine(tmp_path / name)
    if main is not None:
        main(e)
    try:
        result, record = cases.fit(e, cases.build_case(e, random.Random(seed), shape), cap, after_exception)
        record["head_glued"] = e._generated_context_carrier_remainder(result[0]) is not None
        return record
    finally:
        e.shutdown()


@pytest.mark.parametrize("calls,expected", [(1, [2, 6, 10, 14, 18, 22]), (3, [6, 10, 14, 18, 22, 26])],
                         ids=["short-control", "full-page"])
def test_c2_1124_forced_compress_frontiers_equal_main(engine_1124, monkeypatch, calls, expected):  # noqa: F811
    """C2: #1124 r3's forced-compress frontiers are exactly main's (the existing file asserts a lower bound)."""
    engine = engine_1124
    monkeypatch.setattr(lcm_store_complete, "_SCAN_LIMIT", 4)
    originals = [m for i in range(calls) for m in _tagged_segment(i)]
    ids = _append(engine, originals)
    _publish(engine, [(m, [sid]) for m, sid in zip(originals, ids)])
    _append(engine, originals)
    _append(engine, [_text("real new turn " * 20, "user"), _text("real new reply " * 20)])
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("Earlier turns", 1))
    view, frontiers = [], []
    for turn in range(6):
        rows = [_text(f"host turn {turn}-{i} " * 20, "user") for i in range(4)]
        _append(engine, rows + [_text("fresh", "user"), _text("fresh reply")])
        view += rows + [_text("fresh", "user"), _text("fresh reply")]
        engine._ingest_cursor = len(view)
        view = engine.compress(view, force=True, bypass_cooldown=True)
        frontiers.append(_frontier(engine))
        engine.note_turn_complete()
    assert frontiers == expected


def test_c3_the_block_band_keeps_mains_cut_and_carrier(tmp_path, host, caplog, as_main):
    """C3: carrier case 0 at caps 1,033 to 1,107 (the block's own width): main drops 10 rows and keeps
    [C:U5, A5, U6, A6] with the carrier registered; so does this tree, with the notice not placed and one true
    warning. From 1,108 the block fits on that same cut; below 1,033 main's deeper cut has no carrier."""
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    seen = {}
    for cap in [1000, 1032, *range(1033, 1108, 4), 1107, 1108, 1200]:
        caplog.clear()
        record = _fit_at(tmp_path, "carrier", 0, cap, name=f"c{cap}")
        main = _fit_at(tmp_path, "carrier", 0, cap, main=as_main, name=f"m{cap}")
        assert (record["rows"], record["dropped"], record["emission_row"]) == (
            main["rows"], main["dropped"], main["emission_row"]), cap
        seen[cap] = (record["dropped"], record["head_glued"], sum(record["blocks"]),
                     [w for w in (NOT_FIT, NO_CARRIER) if w in caplog.text])
    assert all(seen[cap] == (10, True, 0, [NOT_FIT]) for cap in range(1033, 1108, 4)), seen
    assert seen[1107] == (10, True, 0, [NOT_FIT]) and seen[1108][:3] == seen[1200][:3] == (10, True, 1)
    assert seen[1000] == seen[1032] == (12, False, 0, [NO_CARRIER])


def _objective_case(tmp_path, seed, name, main=None):
    """A carrier an earlier fit marked, ahead of a preserved objective (main has the same carrier unmarked), then a
    stored old turn and a newest turn, fitted at a cap that holds the unmarked head plus the newest turn with no room
    for a notice."""
    rng = random.Random(seed)
    e = cases.engine(tmp_path / name)
    if main is not None:
        main(e)
    try:
        raw = []
        for i in range(6):
            raw += [{"role": "user", "content": f"[U{i}] " + cases.words(rng, rng.randint(20, 200)), "timestamp": 10.0 * i},
                    {"role": "assistant", "content": f"[A{i}] " + cases.words(rng, rng.randint(20, 400))}]
        raw += [{"role": "user", "content": "[OLD] " + cases.words(rng, 120), "timestamp": 500.0},
                {"role": "assistant", "content": "[OLDA] " + cases.words(rng, 200)},
                {"role": "user", "content": "[NEWEST] " + cases.words(rng, 40), "timestamp": 600.0}]
        e.ingest(raw)
        summary = cases.part(e, "summary " + cases.words(rng, 30))
        nodes = e._verified_lcm_summary_prefix(summary)[1]
        block = "" if main is not None else survival_notice_block(summary, nodes, N_OLD)
        head = [{"role": "user", "content": f"{summary}{block}\n\n{raw[0]['content']}"},
                {"role": "user", "content": f"{cases.OBJ}\n" + cases.words(rng, 8)}]
        active = [*head, *[dict(m) for m in raw[1:]]]
        assert e._generated_context_carrier_remainder(active[0]) == raw[0]["content"]
        plain = [{"role": "user", "content": f"{summary}\n\n{raw[0]['content']}"}, head[1], raw[-1]]
        _, record = cases.fit(e, active, e._survival_measure(plain) + 2, False)
        return record
    finally:
        e.shutdown()


@pytest.mark.parametrize("seed", range(4))
def test_c4_a_marked_carrier_ahead_of_an_objective_keeps_mains_summary(tmp_path, host, as_main, seed):
    """C4: [marked carrier, objective] plus an old and a newest turn at the tight cap: main's rows, summary kept,
    and the earlier notice gone."""
    record = _objective_case(tmp_path, seed, "c")
    main = _objective_case(tmp_path, seed, "m", main=as_main)
    assert record["fitted"] and record["rows"] == main["rows"] and record["nodes"] == main["nodes"] and main["nodes"]
    assert sum(record["blocks"]) <= 1 and record["dropped"] == main["dropped"]


def test_c5_two_carriers_one_block_in_a_carrier_that_stays_glued(tmp_path, host, caplog, as_main):
    """C5: two carriers (case 0) at cap 1,308: main's rows, exactly one block, in a carrier that stays glued; at a
    cap with no room, the warning says the notice does not fit."""
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    seed = cases.SHAPES.index("two_carriers") * 1000
    record = _fit_at(tmp_path, "two_carriers", seed, 1308, name="c")
    main = _fit_at(tmp_path, "two_carriers", seed, 1308, main=as_main, name="m")
    assert record["rows"] == main["rows"] and record["dropped"] == main["dropped"]
    assert sum(record["blocks"]) == 1
    e = cases.engine(tmp_path / "glued")
    try:
        result, _ = cases.fit(e, cases.build_case(e, random.Random(seed), "two_carriers"), 1308, True)
        holder = next(m for m in result if cases.OPEN in str(m.get("content")))
        assert e._generated_context_carrier_remainder(holder) is not None
    finally:
        e.shutdown()
    for cap in range(1000, 1308, 7):  # every fit that places no block says why, once
        caplog.clear()
        swept = _fit_at(tmp_path, "two_carriers", seed, cap, name=f"s{cap}")
        if swept["fitted"] and not sum(swept["blocks"]):
            assert caplog.text.count("it is not placed") == 1, cap
            assert (NOT_FIT in caplog.text) is (swept["head_glued"] or any(
                r == "user" and c.startswith("[Recent") and "\n\n[U" in c for r, c in swept["rows"])), cap


N_OLD = ("[LCM survival fit: 2 earlier messages (store ids 1..2) are stored verbatim but not in live context; "
         "lcm_grep / lcm_load_session reach them.]")


@pytest.mark.parametrize("case", range(4))
def test_c6_an_earlier_block_never_stays_beside_a_new_one(tmp_path, host, case):
    """C6: two glued carriers, the first holding an earlier fit's block, over 31 caps: at most one block, and the
    earlier notice never stays."""
    seed = cases.SHAPES.index("two_carriers") * 1000 + case
    probe = cases.engine(tmp_path / "p")
    try:
        full = probe._survival_measure(cases.build_case(probe, random.Random(seed), "two_carriers"))
    finally:
        probe.shutdown()
    for k, cap in enumerate(range(max(50, full // 4), full, max(1, full // 40))):
        e = cases.engine(tmp_path / f"c{k}")
        try:
            active = cases.build_case(e, random.Random(seed), "two_carriers")
            content = active[0]["content"]
            end = e._verified_lcm_summary_prefix_end(content)
            nodes = e._verified_lcm_summary_prefix(content)[1]
            active[0] = {**active[0], "content": content[:end] + survival_notice_block(content[:end], nodes, N_OLD)
                         + content[end:]}
            assert e._generated_context_carrier_remainder(active[0]) is not None
            result = e._survival_fit(active, list(active), 0, "t", after_exception=True, request_cap=cap)
            if result is not active:
                assert sum(str(m.get("content")).count(cases.OPEN) for m in result) <= 1, cap
                assert not any(N_OLD in str(m.get("content")) for m in result), cap
        finally:
            e.shutdown()


@pytest.mark.parametrize("pattern,text,stored", [
    (r"\Aheartbeat\Z", f"heartbeat{cases.PASTED}", True),
    (r"^ok thanks$", f"ok thanks{cases.PASTED}", True),
    (r"hello\n\nworld", f"hello{cases.PASTED}\n\nworld", True),
    (r"^ok thanks$", "ok thanks", False),  # control: no block, main's match
    (r"LCM survival fit", f"my question{cases.PASTED}", True),  # a block only keeps a row
], ids=["anchored-heartbeat", "anchored-ok", "across-the-block", "control", "notice-text"])
def test_c7_a_pasted_block_never_makes_a_row_ignored(tmp_path, pattern, text, stored):
    """C7: a row is ignored only when its text matches both with and without its notice blocks."""
    e = cases.engine(tmp_path, [pattern])
    try:
        rows = [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": text}, {"role": "assistant", "content": "an answer"}]
        e.ingest(rows)
        kept = [r["content"] for r in e._store.get_session_messages(e._session_id, limit=1000)]
        assert (text in kept) is stored and len(kept) == 3 + stored
    finally:
        e.shutdown()


def test_c7_a_fresh_turn_merged_behind_a_marked_carrier_is_kept(tmp_path, host):
    """C7: the host merges a fresh turn behind the fit's marked carrier; a pattern on the notice's words keeps it."""
    e = cases.engine(tmp_path, [r"LCM survival fit"])
    try:
        result, record = cases.fit(e, cases.build_case(e, random.Random(0), "carrier"), 1200, True)
        assert record["blocks"][0] == 1
        merged = {"role": "user", "content": result[0]["content"] + "\n\n[FRESH] a fresh turn", "timestamp": 9e3}
        assert not e._matches_ignore_message_patterns(merged)
        e.ingest([merged, {"role": "assistant", "content": "[FRESHA] ok"}])
        assert any("[FRESH] a fresh turn" in json.dumps(r["content"])
                   for r in e._store.get_session_messages(e._session_id, limit=10_000))
    finally:
        e.shutdown()


def _gap_store(tmp_path, cap, name, main=None):
    e = cases.engine(tmp_path / name)
    if main is not None:
        main(e)
    try:
        active = cases.build_case(e, random.Random(0), "carrier")
        e._ingest_cursor, e._ingest_cursor_needs_reconcile = len(active), False
        result = e._survival_fit(active, list(active), 0, "t", request_cap=cap)
        e._bind_survival_fit_emission(active, result)
        merged = cases.merge(result)
        e.ingest(result)
        e.ingest(merged)
        new = [{"role": "user", "content": "[NEW] synthetic newest", "timestamp": 9e3},
               {"role": "assistant", "content": "[NEWA] ok"}]
        e.ingest([*merged, *new])
        e.on_session_end("S", [*merged, *new])
        return cases.stored(e), any("[NEW] synthetic" in str(r[1]) for r in cases.stored(e))
    finally:
        e.shutdown()


@pytest.mark.parametrize("cap", [900, 1000, 1033, 1050, 1080, 1107, 1108, 1200])
def test_c8_the_warm_store_keeps_the_next_user_turn_where_main_does(tmp_path, host, as_main, cap):
    """C8: replay as returned, merged, a new turn and session end store main's rows; the next user turn is stored
    wherever main stores it (from cap 1,033 on, main's carrier cut)."""
    rows, new_stored = _gap_store(tmp_path, cap, "c")
    main_rows, main_new = _gap_store(tmp_path, cap, "m", main=as_main)
    assert rows == main_rows and new_stored is main_new is (cap >= 1033)
