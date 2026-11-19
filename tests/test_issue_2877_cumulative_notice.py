"""#2877 (N1): the survival-fit notice names every range of this session still out of view, each with its call.

- (a) The catch-up grid's steady 600k prefix cell, walked by tests/behaviour/catchup_grid_runner.py (its content-aware
  A0/A0b loss oracle, in its own isolated process, as the grid runs it). With today's notice (the builder patched
  out) a later fit no longer names 37 rows of an earlier one; with N1 no row is left without a pointer (strict).
- (b) The budget: 12 ranges out of view name the newest 8 and count the rest in one clause, within the constant.
- (c) The calls work: each call a notice names, parsed from its text and run as written through the real
  lcm_load_session handler (paging with next_cursor), returns exactly the stored rows of its range.
- (d) Copies, both directions, and naming recomputed from the current list on every call.
- The builder's text is pinned by literals, called directly (outside any fit): one range, several ranges, and the
  list the emergency cut returns after dropping the summary prefix; the fit's own notice equals that call.
- The builder's optional store-id map and read only save work: without them it returns byte-identical text.
- Never fewer than the base: the rows a fit drops are named in any case, a rotation's parent rows under the parent's
  session id.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import (
    _NOTICE_MAX_RANGES, _NOTICE_MAX_TOKENS, _OUT_OF_VIEW_SQL, OutOfViewRead, read_out_of_view, survival_fit_notice)
from hermes_lcm.tokens import count_tokens
from hermes_lcm.tools import lcm_load_session
from tests.behaviour.catchup_grid_runner import loss_check

TREE = Path(__file__).resolve().parents[1]
RUNNER = TREE / "tests" / "behaviour" / "catchup_grid_runner.py"
# The grid's s-600k-l40-prefix cell (tests/test_catchup_budget_grid.py _steady_cells), as data.
CELL = {"group": "S", "id": "s-600k-l40-prefix", "passes": 60, "raw_tokens": 150_000, "over_tokens": 20_000,
        "growth_tokens": 26_000, "turn_gap": 60, "finalize": 32, "chunk": 24_000, "summary_tokens": 1_500,
        "window": 600_000, "pre": 37, "estimate": 44, "call": 40, "delta": 0, "prefix": "1-2x"}
TODAY = "survival_fit.survival_fit_notice = lambda *args, **kwargs: None  # today's single-range notice\n"
NOTICE = re.compile(r"^\[LCM survival fit: (\d+) earlier messages \(store ids (\d+)\.\.(\d+)\) are stored verbatim but not "
                    r"in live context; read each range with lcm_load_session and limit=(\d+), paging with next_cursor: "
                    r"(.*)\.\]$")
GROUP = re.compile(r"session_id=([^:;\s]+): ((?:after_store_id=\d+ for store ids \d+\.\.\d+(?:, )?)+)")
ITEM = re.compile(r"after_store_id=(\d+) for store ids (\d+)\.\.(\d+)")


def _calls(notice: str) -> list[tuple[int, int, dict]]:
    """(first, last, arguments) of every range the notice names, its arguments parsed from the text."""
    head = NOTICE.match(notice)
    assert head, notice
    return [(int(first), int(last), {"session_id": session, "after_store_id": int(after), "limit": int(head.group(4))})
            for session, items in GROUP.findall(head.group(5)) for after, first, last in ITEM.findall(items)]


def _syspaths() -> list[str]:
    """The host stub and dependency directories of this process, never this tree."""
    found = []
    for entry in sys.path:
        path = Path(entry).resolve() if entry else None
        if path is None or not path.is_dir() or path == TREE or TREE in path.parents:
            continue
        if entry.endswith("site-packages") or (path / "agent" / "context_engine.py").is_file():
            found.append(str(path))
    return found


def _walk(tmp_path: Path, name: str, patch: str = "") -> list[dict]:
    cells, out, home = tmp_path / f"{name}-cells.json", tmp_path / f"{name}.json", tmp_path / f"{name}-home"
    cells.write_text(json.dumps([CELL]))
    home.mkdir()
    command = [sys.executable, "-I", "-B", str(RUNNER), "--tree", str(TREE), "--cells", str(cells), "--out", str(out),
               "--workdir", str(tmp_path / f"{name}-work")]
    for entry in _syspaths():
        command += ["--syspath", entry]
    if patch:
        (tmp_path / f"{name}-patch.py").write_text(patch)
        command += ["--patch", str(tmp_path / f"{name}-patch.py")]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0"}
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(out.read_text())["records"]


def test_a_two_fits_in_a_row_leave_no_row_without_a_pointer(tmp_path):
    today, n1 = _walk(tmp_path, "today", TODAY), _walk(tmp_path, "n1")
    lost = {r["pass"]: r["loss"]["missing_count"] for r in today if r["loss"] and r["loss"]["missing_count"]}
    assert sum(lost.values()) == 37, lost  # the premise: today's notice names only the newest fit's range
    fits = [r["pass"] for r in n1 if r["loss"] and r["loss"]["survival_fit_notice"]]
    assert any(b == a + 1 for a, b in zip(fits, fits[1:])), fits  # fits in consecutive passes
    for t, r in zip(today, n1):
        if r["loss"] is None:
            continue
        loss = r["loss"]
        assert loss["missing_count"] == 0 and not loss["unpaired"], (r["pass"], loss["missing"][:2])
        assert not loss["noop_not_identical"] and not loss["noop_changed"], r["pass"]
        assert loss["missing_count_by_id"] == 0, r["pass"]
        base = t["loss"] or {"missing_count": 0, "behind_notice_count": 0}
        assert loss["behind_notice_count"] <= base["missing_count"] + base["behind_notice_count"], r["pass"]


# -- direct constructions ------------------------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_reserve=0.0))
    obj._hermes_home = str(tmp_path)
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
    yield obj
    obj.shutdown()


def _store(engine, contents, session="S") -> list[tuple[int, dict]]:
    """Append rows (user and assistant alternate) and return (store id, list object) for each."""
    out = []
    for index, content in enumerate(contents):
        row = {"role": "user" if index % 2 == 0 else "assistant", "content": content}
        out.append((engine._store.append(session, dict(row), conversation_id="conv"), row))
    return out


def _listed(rows, keep) -> tuple[list[dict], dict]:
    """The list objects of the rows whose store id is in ``keep``, and their ``{id(row): store id}`` map."""
    kept = [(store_id, row) for store_id, row in rows if store_id in keep]
    return [row for _, row in kept], {id(row): store_id for store_id, row in kept}




def test_builder_text_one_range_and_several_ranges_are_pinned_by_literals(engine):
    rows = _store(engine, [f"row {i}" for i in range(1, 11)])
    messages, ids = _listed(rows, {4, 5, 6, 7, 8, 9, 10})
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) == (
        "[LCM survival fit: 3 earlier messages (store ids 1..3) are stored verbatim but not in live context; read each "
        "range with lcm_load_session and limit=200, paging with next_cursor: session_id=S: after_store_id=0 for store "
        "ids 1..3.]")
    messages, ids = _listed(rows, {3, 4, 7})
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) == (
        "[LCM survival fit: 7 earlier messages (store ids 1..10) are stored verbatim but not in live context; read each "
        "range with lcm_load_session and limit=200, paging with next_cursor: session_id=S: after_store_id=0 for store "
        "ids 1..2, after_store_id=4 for store ids 5..6, after_store_id=7 for store ids 8..10.]")
    messages, ids = _listed(rows, set(range(1, 11)))
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) is None  # nothing out of view


def _rough(monkeypatch):
    """Deterministic chars / 4 counts (#916's harness), without the host estimator."""
    def count(message):
        return len(str(message.get("content") or "")) // 4

    def measure(messages):
        return sum(count(message) for message in messages)

    monkeypatch.setattr(survival_fit, "count_message_tokens", count)
    monkeypatch.setattr(survival_fit, "count_messages_tokens", measure)
    monkeypatch.setattr(survival_fit, "_host_estimate", measure)


def _emergency_list(engine):
    """A summary row over store ids 1..4 too large to keep with the newest turn, three stored turns (5..10) and the
    newest turn (11..12): the emergency cut drops the summary prefix with the turns."""
    pad = " older words" * 100
    rows = _store(engine, [f"[R{i}]{pad}" for i in range(1, 11)] + ["[N] the newest question", "[N] answer " * 150])
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="saved history " * 1000, token_count=1,
                                            source_token_count=1, source_ids=[1, 2, 3, 4], expand_hint="turns"))
    summary = {"role": "user", "content": f"[Recent Summary (d0, node {node})]\n{'saved history ' * 1000}\n"
                                          "[Expand for details: turns]"}
    active = [{"role": "system", "content": "system prompt"}, summary, *(row for _, row in rows[4:])]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    return active


def test_builder_text_on_the_emergency_path_is_pinned_by_a_literal(engine, monkeypatch, caplog):
    _rough(monkeypatch)
    active = _emergency_list(engine)
    assert engine._verified_lcm_summary_prefix(active[1]["content"])[0] is not None
    statements = []
    engine._store.connection.set_trace_callback(statements.append)
    try:
        result = engine._survival_fit(active, list(active), 0, "test", request_cap=900)
    finally:
        engine._store.connection.set_trace_callback(None)
    assert [row["content"] for row in result[1:]] == [active[-2]["content"], active[-1]["content"]]
    assert "dropped the summary prefix" in caplog.text
    notice = engine._last_survival_fit["notice"]
    assert notice == ("[LCM survival fit: 10 earlier messages (store ids 1..10) are stored verbatim but not in live "
                      "context; read each range with lcm_load_session and limit=200, paging with next_cursor: "
                      "session_id=S: after_store_id=0 for store ids 1..10.]")
    assert result[0]["content"] == f"system prompt\n\n{notice}"
    # the same text from a direct call on the returned list, outside the fit (no node is shown: the prefix left)
    assert survival_fit_notice(engine, result, set()) == notice
    # the out-of-view read: the emergency path reads twice (the prefix cut, then the cut that may drop the prefix),
    # each one recursive statement and one bounded content read, for every cut it tries
    assert sum("WITH RECURSIVE walk(root, node_id)" in s for s in statements) == 2, statements
    assert sum("SELECT store_id, session_id, role, content FROM messages WHERE store_id IN" in s for s in statements) == 2
    plan = " ".join(str(row) for row in engine._store.connection.execute(
        "EXPLAIN QUERY PLAN " + _OUT_OF_VIEW_SQL, ("[1]", "S")).fetchall())
    assert "idx_msg_session" in plan, plan


def test_a_normal_fit_reads_the_session_once(engine, monkeypatch):
    """No summary prefix: one recursive statement and one bounded content read for every cut the fit tries."""
    _rough(monkeypatch)
    active = [row for row in _emergency_list(engine) if not row["content"].startswith("[Recent Summary")]
    statements = []
    engine._store.connection.set_trace_callback(statements.append)
    try:
        result = engine._survival_fit(active, list(active), 0, "test", request_cap=900)
    finally:
        engine._store.connection.set_trace_callback(None)
    assert len(result) == 3 and engine._last_survival_fit["notice"] == survival_fit_notice(engine, result, set())
    assert sum("WITH RECURSIVE walk(root, node_id)" in s for s in statements) == 1, statements
    assert sum("SELECT store_id, session_id, role, content FROM messages WHERE store_id IN" in s for s in statements) == 1


def test_b_twelve_ranges_name_the_newest_eight_and_count_the_rest(engine):
    rows = _store(engine, [f"row {i}" for i in range(1, 37)])
    messages, ids = _listed(rows, {i for i in range(1, 37) if i % 3 == 0})  # 12 ranges of two rows
    notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
    named = [(first, last) for first, last, _ in _calls(notice)]
    assert named == [(i, i + 1) for i in range(13, 37, 3)] and len(named) == _NOTICE_MAX_RANGES
    assert notice.startswith("[LCM survival fit: 24 earlier messages (store ids 1..35) are stored verbatim")
    assert notice.endswith("; and 4 earlier ranges (store ids 1..11); lcm_grep reaches them.]")
    assert count_tokens(notice) <= _NOTICE_MAX_TOKENS


def test_b_the_notice_bound_holds_for_long_ids():
    """The constant's case: 12-digit store ids, two sessions (a rotation's parent) with 128-character ids, far more
    ranges than are named. The single-range notice stays within today's size plus 40 tokens."""
    order = [10 ** 12 - 2 * 60 + i for i in range(120)]
    foreign = [10 ** 12 + 1000 + 2 * i for i in range(4)]
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         dict.fromkeys(foreign, "fedcba9876543210" * 8))
    listed = [{"role": "user", "content": "kept"} for _ in order[1::2]]
    stub = SimpleNamespace(_session_id="0123456789abcdef" * 8, _store=None)
    notice = survival_fit_notice(stub, listed, set(), left=foreign, read=read,
                                 store_ids={id(m): s for m, s in zip(listed, order[1::2])})
    sessions = {args["session_id"] for _, _, args in _calls(notice)}
    assert len(_calls(notice)) == _NOTICE_MAX_RANGES and len(sessions) == 2 and "; and 56 earlier ranges" in notice
    assert count_tokens(notice) <= _NOTICE_MAX_TOKENS, count_tokens(notice)
    today = survival_fit._NOTICE.format(n=3, first=1, last=3)
    single = survival_fit.survival_notice_text(3, [("20261119_120000_abcdef12", 1, 3)])
    assert count_tokens(single) <= count_tokens(today) + 40, (count_tokens(single), count_tokens(today))
    assert json.dumps(notice)[1:-1] == notice and json.dumps(single)[1:-1] == single  # nothing JSON escapes


def _load(engine, args: dict) -> list[dict]:
    """Run the call as written (its arguments parsed from the notice text), then follow next_cursor."""
    pages = [json.loads(lcm_load_session(dict(args), engine=engine))]
    while pages[-1]["next_cursor"] is not None:
        pages.append(json.loads(lcm_load_session({**args, "after_store_id": pages[-1]["next_cursor"]}, engine=engine)))
    return [message for page in pages for message in page["messages"]]


def test_c_each_named_call_loads_its_range_byte_equal(engine):
    """Rows stay under lcm_load_session's 4,000-character content default, so a page carries them whole. Another
    session's rows are interleaved: a range follows this session's rows, and its call reads only them."""
    rows = []
    for i in range(1, 601):
        row = {"role": "user", "content": f"[S{i}] " + "w" * (i % 7 * 300)} if i % 2 else {
            "role": "assistant", "content": f"[S{i}] reply " + "v" * 900}
        rows.append((engine._store.append("S", dict(row), conversation_id="conv"), row))
        if i % 50 == 0:
            _store(engine, [f"[other {i}]"], session="other")
    keep = {store_id for n, (store_id, _) in enumerate(rows, 1) if 450 < n <= 460 or 500 < n <= 510}
    messages, ids = _listed(rows, keep)
    notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
    named = _calls(notice)
    assert len(named) == 3, notice  # the other session's ids do not split a range
    stored = engine._store.get_session_messages("S", limit=10_000)
    for first, last, arguments in named:
        loaded = [m for m in _load(engine, arguments) if m["store_id"] <= last]
        expected = [m for m in stored if first <= m["store_id"] <= last]
        assert len(expected) > 1 and not any(m["content_truncated"] for m in loaded)
        assert [(m["store_id"], m["role"], m["content"]) for m in loaded] == [
            (m["store_id"], m["role"], m["content"]) for m in expected]
    assert sum(1 for m in stored if m["store_id"] in keep) == 20 and len(stored) == 600


def _summary_row(engine, node_id: int) -> dict:
    node = engine._dag.get_node(node_id)
    return {"role": "user", "content": f"[Recent Summary (d0, node {node_id})]\n{node.summary}\n[Expand for details: turns]"}


def _returned(engine, notice, rows) -> list[dict]:
    return [{"role": "system", "content": f"system prompt\n\n{notice}"}, *rows]


def test_d_copies_both_directions_and_nothing_carries_over(engine):
    rows = _store(engine, ["same words", "r2", "u3", "r4", "same words", "r6", "c words", "r8", "c words", "r10"])
    # an earlier notice names store ids 1..2 and 5 (row 5 is a byte-equal copy of row 1; no copy is in view)
    messages, ids = _listed(rows, {3, 4, 6, 7, 8, 9, 10})
    notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
    assert [(a, b) for a, b, _ in _calls(notice)] == [(1, 2), (5, 5)], notice
    # that range is summarised since: a leaf over 1..2, shown in the list
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="the same words, summarised", token_count=1,
                                            source_token_count=1, source_ids=[1, 2], expand_hint="turns"))
    messages, ids = _listed(rows, {3, 4, 9, 10})
    notice = survival_fit_notice(engine, messages, {node}, store_ids=ids)
    # row 5's copy is under the shown node and row 7's copy (9) is in the list: neither is named
    assert [(a, b) for a, b, _ in _calls(notice)] == [(6, 6), (8, 8)], notice
    result = _returned(engine, notice, [_summary_row(engine, node), *messages])
    check = loss_check(engine, result, result, did_work=True)
    # The grid oracle counts a copy in the list as a multiset: "c words" is stored twice (7, 9) and returned once, so
    # it flags the later stored copy, 9, which is in the list itself. The copies rule counts the returned copy as
    # covering row 7; nothing else is flagged.
    assert check["heads"] == 1 and {m["store_id"] for m in check["missing"]} == {9}, check["missing"]
    # the node's summary is not in this list: row 1 and its copy, row 5, have no copy in view, so both are named
    notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
    assert [(a, b) for a, b, _ in _calls(notice)] == [(1, 2), (5, 6), (8, 8)], notice
    check = loss_check(engine, _returned(engine, notice, messages), _returned(engine, notice, messages), did_work=True)
    assert {m["store_id"] for m in check["missing"]} == {9}, check["missing"]
    # a row back in view is not named
    messages, ids = _listed(rows, set(range(1, 11)))
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) is None


def test_the_optional_map_and_read_only_save_work(engine):
    """Byte-identical text with and without ``store_ids`` and ``read``, including a read made for more nodes than the
    list shows (the fit reads once for every cut it tries, and an emergency cut can drop a summary)."""
    rows = _store(engine, ["same words", "r2", "u3", "r4", "same words", "r6", "c words", "r8", "c words", "r10",
                           "u11", "r12"])
    first = engine._dag.add_node(SummaryNode(session_id="S", summary="one", token_count=1, source_token_count=1,
                                             source_ids=[1, 2], expand_hint="turns"))
    second = engine._dag.add_node(SummaryNode(session_id="S", summary="two", token_count=1, source_token_count=1,
                                              source_ids=[3, 4], expand_hint="turns"))
    wide = read_out_of_view(engine._store, "S", {first, second}, kept={first})
    cases = 0
    parent = [engine._store.append("P", {"role": "user", "content": f"parent {i}"}, conversation_id="conv")
              for i in range(3)]
    for keep in ({9, 10}, {5, 9, 11, 12}, {11, 12}, set(), set(range(1, 13))):
        messages, ids = _listed(rows, keep)
        for shown in ({first}, {first, second}, set(), {second}):
            for left in ((), (1, 5), (*parent, 7)):
                plain = survival_fit_notice(engine, messages, shown, left=left)
                args = (keep, shown, left)
                assert survival_fit_notice(engine, messages, shown, left=left, store_ids=ids) == plain, args
                assert survival_fit_notice(engine, messages, shown, left=left, store_ids=ids, read=wide) == plain, args
                cases += plain is not None
    assert cases >= 36


def test_rows_that_left_are_named_in_any_case(engine):
    """``left`` (the rows a fit dropped, as the base's notice names them) is named even when a copy is in view."""
    rows = _store(engine, ["same words", "r2", "u3", "r4", "same words", "r6"])
    messages, ids = _listed(rows, {1, 2, 3, 4, 6})
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) is None  # row 5's copy (1) is in the list
    assert survival_fit_notice(engine, messages, set(), left=[5], store_ids=ids) == (
        "[LCM survival fit: 1 earlier messages (store ids 5..5) are stored verbatim but not in live context; read each "
        "range with lcm_load_session and limit=200, paging with next_cursor: session_id=S: after_store_id=4 for store "
        "ids 5..5.]")


def test_after_a_rotation_a_dropped_parent_row_is_named_with_its_session(engine, monkeypatch):
    """The list maps two rows into the parent session, as a rotation's carry range does (the map is stubbed to add
    them); the fit drops them, and the notice names them under the parent's session id with a call that loads them."""
    _rough(monkeypatch)
    pad = " older words" * 100
    parent = [engine._store.append("P", {"role": role, "content": f"[P{i}]{pad}"}, conversation_id="conv")
              for i, role in enumerate(("user", "assistant"))]
    rows = _store(engine, [f"[R{i}]{pad}" for i in range(1, 5)] + ["[N] the newest question", "[N] answer " * 150])
    carried = [{"role": "user", "content": f"[P0]{pad}"}, {"role": "assistant", "content": f"[P1]{pad}"}]
    active = [{"role": "system", "content": "system prompt"}, *carried, *(row for _, row in rows)]
    real = engine._get_store_id_map_for_messages
    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda messages, *a, **k: {
        **real(messages, *a, **k), **{id(m): s for m, s in zip(carried, parent) if any(m is x for x in messages)}})
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    result = engine._survival_fit(active, list(active), 0, "test", request_cap=900)
    assert [row["content"] for row in result[1:]] == [rows[-2][1]["content"], rows[-1][1]["content"]]
    notice = engine._last_survival_fit["notice"]
    named = {(first, last): args for first, last, args in _calls(notice)}
    assert named[(parent[0], parent[1])]["session_id"] == "P", notice
    loaded = [m for m in _load(engine, named[(parent[0], parent[1])]) if m["store_id"] <= parent[1]]
    assert [m["content"] for m in loaded] == [f"[P0]{pad}", f"[P1]{pad}"]
    assert named[(rows[0][0], rows[3][0])]["session_id"] == "S", notice


def test_head_bytes_are_stable_while_the_out_of_view_set_does_not_change(tmp_path, monkeypatch):
    """Two fits from the same stored state return byte-identical system rows, and the builder recomputed on the returned
    list (a later pass with the same rows out of view) gives the same notice."""
    _rough(monkeypatch)
    heads = []
    for name in ("one", "two"):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"{name}.db"), survival_reserve=0.0))
        engine._hermes_home = str(tmp_path)
        engine.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
        try:
            active = _emergency_list(engine)
            result = engine._survival_fit(active, list(active), 0, "test", request_cap=900)
            heads.append(result[0]["content"])
            assert heads[-1].endswith(survival_fit_notice(engine, result, set()))
        finally:
            engine.shutdown()
    assert heads[0] == heads[1]
