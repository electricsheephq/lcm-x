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
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import (
    _NOTICE_MAX_RANGES, _NOTICE_MAX_TOKENS, _OUT_OF_VIEW_SQL, OutOfViewRead, read_out_of_view, survival_fit_notice)
from hermes_lcm.tokens import count_tokens
from hermes_lcm.tools import lcm_expand, lcm_load_session
from tests.behaviour.catchup_grid_runner import loss_check

TREE = Path(__file__).resolve().parents[1]
RUNNER = TREE / "tests" / "behaviour" / "catchup_grid_runner.py"
# The grid's s-600k-l40-prefix cell (tests/test_catchup_budget_grid.py _steady_cells), as data.
CELL = {"group": "S", "id": "s-600k-l40-prefix", "passes": 60, "raw_tokens": 150_000, "over_tokens": 20_000,
        "growth_tokens": 26_000, "turn_gap": 60, "finalize": 32, "chunk": 24_000, "summary_tokens": 1_500,
        "window": 600_000, "pre": 37, "estimate": 44, "call": 40, "delta": 0, "prefix": "1-2x"}
TODAY = "survival_fit.survival_fit_notice = lambda *args, **kwargs: None  # today's single-range notice\n"
# the parsers also read the reviewed head's form (one limit for all ranges), so a cell fails there on its own claim
NOTICE = re.compile(r"^\[LCM survival fit: (\d+) earlier messages \(store ids (\d+)\.\.(\d+)\) are stored verbatim but not "
                    r"in live context; read with lcm_load_session(?: limit=(\d+))?, next_cursor, and "
                    r"lcm_expand\(store_id\) if truncated: (.*)\.\]$", re.S)
GROUP = re.compile(r"session_id=([^;\s]+) ((?:after_store_id=\d+(?: limit=\d+)? to \d+(?:, )?)+)")
ITEM = re.compile(r"after_store_id=(\d+)(?: limit=(\d+))? to (\d+)")


def _calls(notice: str) -> list[tuple[int, int, dict]]:
    """(first, last, arguments) of every range the notice names, its arguments parsed from the text."""
    head = NOTICE.match(notice)
    assert head, notice
    calls = [(int(after) + 1, int(stop), {"session_id": session, "after_store_id": int(after),
                                          "limit": int(limit or head.group(4))})
             for session, items in GROUP.findall(head.group(5)) for after, limit, stop in ITEM.findall(items)]
    assert len(calls) == len(ITEM.findall(head.group(5))), notice  # every range says where it starts and stops
    return calls


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
    if (TREE / "agent" / "context_engine.py").is_file():  # CI writes its host stub into the repo root, which is skipped
        shutil.copytree(TREE / "agent", tmp_path / f"{name}-stub" / "agent")
        command += ["--syspath", str(tmp_path / f"{name}-stub")]
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
        "[LCM survival fit: 3 earlier messages (store ids 1..3) are stored verbatim but not in live context; read with "
        "lcm_load_session, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=0 limit=3 "
        "to 3.]")
    messages, ids = _listed(rows, {3, 4, 7})
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) == (
        "[LCM survival fit: 7 earlier messages (store ids 1..10) are stored verbatim but not in live context; read with "
        "lcm_load_session, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=0 limit=2 "
        "to 2, after_store_id=4 limit=2 to 6, after_store_id=7 limit=3 to 10.]")
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
    return measure


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
                      "context; read with lcm_load_session, next_cursor, and lcm_expand(store_id) if truncated: "
                      "session_id=S after_store_id=0 limit=10 to 10.]")
    assert result[0]["content"] == f"system prompt\n\n{notice}"
    # the same text from a direct call on the returned list, outside the fit (no node is shown: the prefix left)
    assert survival_fit_notice(engine, result, set()) == notice
    # the out-of-view read: the emergency path reads twice (the prefix cut, then the cut that may drop the prefix),
    # each one recursive statement and one bounded content read, for every cut it tries
    assert sum("WITH RECURSIVE walk(root, node_id)" in s for s in statements) == 2, statements
    assert sum("SELECT store_id, session_id, role, content, tool_calls, tool_call_id FROM messages" in s for s in statements) == 2
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
    assert sum("SELECT store_id, session_id, role, content, tool_calls, tool_call_id FROM messages" in s for s in statements) == 1


def test_b_twelve_ranges_name_the_newest_eight_and_count_the_rest(engine):
    rows = _store(engine, [f"row {i}" for i in range(1, 37)])
    messages, ids = _listed(rows, {i for i in range(1, 37) if i % 3 == 0})  # 12 ranges of two rows
    notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
    named = [(first, last) for first, last, _ in _calls(notice)]
    assert named == [(i, i + 1) for i in range(13, 37, 3)] and len(named) == _NOTICE_MAX_RANGES
    assert notice.startswith("[LCM survival fit: 24 earlier messages (store ids 1..35) are stored verbatim")
    assert notice.endswith("; and 4 earlier ranges (store ids 1..11); lcm_grep reaches them.]")
    assert count_tokens(notice) <= _NOTICE_MAX_TOKENS


# An 81-character session id with colons, the reference store's longest shape (884 of its 1,378 ids hold ':').
COLON_ID = ("agent:main:telegram:dm:" + "0123456789:" * 6)[:81]


@pytest.mark.parametrize("ids", [("0123456789abcdef" * 8, "fedcba9876543210" * 8), (COLON_ID, COLON_ID[::-1])],
                         ids=["128-hex", "81-colons"])
def test_b_the_notice_bound_holds_for_long_ids(ids):
    """The constant's case: 12-digit store ids, two sessions (a rotation's parent) with long ids, far more ranges than
    are named; each id parses back exactly. The single-range notice stays within today's size plus 40 tokens."""
    assert len(COLON_ID) == 81 and COLON_ID.count(":") > 4
    order = [10 ** 12 - 2 * 60 + i for i in range(120)]
    foreign = [10 ** 12 + 1000 + 2 * i for i in range(4)]
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         dict.fromkeys(foreign, ids[1]))
    listed = [{"role": "user", "content": "kept"} for _ in order[1::2]]
    stub = SimpleNamespace(_session_id=ids[0], _store=None)
    notice = survival_fit_notice(stub, listed, set(), left=foreign, read=read,
                                 store_ids={id(m): s for m, s in zip(listed, order[1::2])})
    calls = _calls(notice)
    sessions = {args["session_id"] for _, _, args in calls}
    # the cap folds the oldest named ranges into the counted clause: 64 ranges in all
    assert 4 <= len(calls) <= _NOTICE_MAX_RANGES and sessions == set(ids), notice
    assert f"; and {64 - len(calls)} earlier ranges" in notice
    assert count_tokens(notice) <= _NOTICE_MAX_TOKENS, count_tokens(notice)
    today = survival_fit._NOTICE.format(n=3, first=1, last=3)
    single = survival_fit.survival_notice_text(3, [("20261119_120000_abcdef12", 1, 3)])
    # the single-range, current-session notice for a 1-character id and 1-digit store ids stays within b488389d's 228
    # characters; the ceiling for every input (b488389d's formula) is the cell below
    assert len(survival_fit.survival_notice_text(3, [("S", 1, 3)])) <= 228
    assert count_tokens(single) <= count_tokens(today) + 40, (count_tokens(single), count_tokens(today))
    assert json.dumps(notice)[1:-1] == notice and json.dumps(single)[1:-1] == single  # nothing JSON escapes


def _load(engine, args: dict, stop: int = 10 ** 15) -> list[dict]:
    """Follow the notice exactly: the call as written (its arguments parsed from the text), paging with next_cursor,
    stopping after the range's last store id, and a row marked content_truncated read whole with lcm_expand (paging
    with its next_content_offset while has_more is set)."""
    rows, page = [], json.loads(lcm_load_session(dict(args), engine=engine))
    while True:
        for message in page["messages"]:
            if message["store_id"] > stop:
                return rows
            if message["content_truncated"]:
                part = json.loads(lcm_expand({"store_id": message["store_id"]}, engine=engine))
                message = {**message, **part, "content_truncated": False}
                while part["has_more"]:
                    part = json.loads(lcm_expand({"store_id": message["store_id"],
                                                  "content_offset": part["next_content_offset"]}, engine=engine))
                    message["content"] += part["content"]
            rows.append(message)
        if page["next_cursor"] is None:
            return rows
        page = json.loads(lcm_load_session({**args, "after_store_id": page["next_cursor"]}, engine=engine))


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
        loaded = _load(engine, arguments, last)
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
        "[LCM survival fit: 1 earlier messages (store ids 5..5) are stored verbatim but not in live context; read with "
        "lcm_load_session, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=4 limit=1 "
        "to 5.]")


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
    loaded = _load(engine, named[(parent[0], parent[1])], parent[1])
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


def test_a_session_id_with_colons_parses_back_and_its_call_loads_the_rows(tmp_path):
    """An 81-character session id holding colons: the notice gives exactly that id, and the call loads the range."""
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "colon.db"), survival_reserve=0.0))
    engine._hermes_home = str(tmp_path)
    engine.on_session_start(COLON_ID, platform="telegram", conversation_id="conv", context_length=20_000)
    try:
        rows = _store(engine, [f"row {i}" for i in range(1, 9)], session=COLON_ID)
        messages, ids = _listed(rows, {6, 7, 8})
        notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
        (first, last, args), = _calls(notice)
        assert (first, last) == (1, 5) and args == {"session_id": COLON_ID, "after_store_id": 0, "limit": 5}, notice
        loaded = _load(engine, args, last)
        assert [m["content"] for m in loaded] == [f"row {i}" for i in range(1, 6)]
    finally:
        engine.shutdown()



# -- review fix round 1 (each cell red on b488389d, green after) ----------------------------------------------------

def test_fix1_following_the_notice_reads_a_long_row_whole_and_nothing_outside_the_range(engine):
    """A range holding a 4,001-character row, with rows after it: following the notice exactly (the call, next_cursor,
    stop after the range's last id, lcm_expand for a truncated row) returns every row of the range byte-equal and no
    row outside it."""
    rows = _store(engine, [f"row {i}" for i in range(1, 10)] + ["X" * 4000 + "Z"] + [f"row {i}" for i in range(11, 14)])
    messages, ids = _listed(rows, {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13})
    (first, last, args), = _calls(survival_fit_notice(engine, messages, set(), store_ids=ids))
    loaded = _load(engine, args, last)
    assert (first, last) == (10, 10) and [(m["store_id"], m["content"]) for m in loaded] == [(10, "X" * 4000 + "Z")]


def test_fix2_the_token_cap_holds_by_the_products_own_counter():
    """Long ids in three sessions: the notice folds its oldest named ranges into the counted clause until the
    product's counter (tokens.count_tokens: tiktoken cl100k when present, else its character estimate) is within
    the cap. The cl100k case runs only where tiktoken and its encoding load (not in CI or on the lane's machine)."""
    order = [10 ** 12 - 2 * 60 + i for i in range(120)]
    foreign = [10 ** 12 + 1000 + 2 * i for i in range(6)]
    ids = ("a1" * 64, "b2" * 64, "c3" * 64)
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         {x: ids[1 + (i % 2)] for i, x in enumerate(foreign)})
    listed = [{"role": "user", "content": "kept"} for _ in order[1::2]]
    stub = SimpleNamespace(_session_id=ids[0], _store=None)
    notice = survival_fit_notice(stub, listed, set(), left=foreign, read=read,
                                 store_ids={id(m): s for m, s in zip(listed, order[1::2])})
    assert count_tokens(notice) <= _NOTICE_MAX_TOKENS, count_tokens(notice)
    assert 1 <= len(_calls(notice)) < _NOTICE_MAX_RANGES and "earlier ranges" in notice


def test_fix2_the_cap_under_cl100k_where_tiktoken_is_present(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # the encoding file is fetched over the network when not cached
        pytest.skip(f"cl100k_base not loadable offline: {exc}")
    monkeypatch.setattr(survival_fit, "count_tokens", lambda text: len(encoder.encode(text or "")))
    order = [10 ** 12 - 2 * 60 + i for i in range(120)]
    foreign = [10 ** 12 + 1000 + 2 * i for i in range(4)]
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         dict.fromkeys(foreign, "b2" * 64))
    listed = [{"role": "user", "content": "kept"} for _ in order[1::2]]
    notice = survival_fit_notice(SimpleNamespace(_session_id="a1" * 64, _store=None), listed, set(), left=foreign,
                                 read=read, store_ids={id(m): s for m, s in zip(listed, order[1::2])})
    assert len(encoder.encode(notice)) <= _NOTICE_MAX_TOKENS


def _two_fits(engine, active, more_contents, cap=900):
    """Fit, then fit again on the returned list plus new rows; both lists' rows are stored first."""
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    first = engine._survival_fit(active, list(active), 0, "test", request_cap=cap)
    more = _store(engine, more_contents, session=engine._session_id)
    second_in = [*first, *(row for _, row in more)]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(second_in), False
    return first, engine._survival_fit(second_in, list(second_in), 0, "test", request_cap=cap)


def test_fix3_a_session_id_holding_a_bracket_keeps_one_notice(tmp_path, monkeypatch):
    _rough(monkeypatch)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "b.db"), survival_reserve=0.0))
    engine._hermes_home = str(tmp_path)
    engine.on_session_start("chat]1", platform="telegram", conversation_id="conv", context_length=20_000)
    try:
        pad = " older words" * 100
        rows = _store(engine, [f"[R{i}]{pad}" for i in range(1, 5)] + ["[N] the newest question", "[N] answer " * 150],
                      session="chat]1")
        active = [{"role": "system", "content": "system prompt"}, *(row for _, row in rows)]
        first, second = _two_fits(engine, active, [f"[M{i}]{pad}" for i in range(1, 3)] + ["[N2] next", "[N2] " * 300])
        assert first[0]["content"].count("[LCM survival fit: ") == 1
        assert second[0]["content"].count("[LCM survival fit: ") == 1, second[0]["content"]
        assert second[0]["content"].endswith(engine._last_survival_fit["notice"])
    finally:
        engine.shutdown()


def _tool_turn(i):
    call = {"id": f"call_{i}", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"cmd": f"ls d{i}"})}}
    return [{"role": "user", "content": f"[u{i}] please list d{i}" + " w" * 60},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": f"[t{i}] " + "file " * 80, "tool_call_id": f"call_{i}", "name": "terminal"},
            {"role": "assistant", "content": f"[a{i}] d{i} holds files" + " v" * 60}]


def test_fix4_tool_call_rows_are_not_copies_of_each_other(engine, monkeypatch):
    """Empty-content tool-call rows differ by their calls: two fits on a tool-heavy session name one unbroken range of
    the rows that left, not one split at every tool-call row still in the list."""
    measure = _rough(monkeypatch)
    msgs = [m for i in range(12) for m in _tool_turn(i)]
    for m in msgs:
        engine._store.append("S", dict(m), conversation_id="conv")
    active = [{"role": "system", "content": "system prompt"}, *(dict(m) for m in msgs)]
    for share in (0.7, 0.45):
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
        active = engine._survival_fit(active, list(active), 0, "tc", request_cap=int(measure(active) * share))
    kept = sorted(engine._get_store_id_map_for_messages([m for m in active if m["role"] != "system"]).values())
    assert [(first, last) for first, last, _ in _calls(engine._last_survival_fit["notice"])] == [(1, kept[0] - 1)]


def test_fix4_an_unmapped_listed_row_shows_its_stored_copy(engine):
    rows = _store(engine, ["u1", "r2", "the same words", "r4"])
    messages, ids = _listed(rows, {1, 2, 4})
    copy = {"role": "user", "content": "the same words"}  # in the list, but no store id maps to it
    assert survival_fit_notice(engine, [*messages, copy], set(), store_ids=ids) is None


def test_fix5_a_rotations_parent_rows_stay_named_on_the_next_fit(engine, monkeypatch):
    """The session carries parent rows P:1..2 (its carry range, as a rotation records it). Fit one drops them; fit two,
    on fit one's list plus new turns, drops only child rows and still names P:1..2 under the parent's session id."""
    _rough(monkeypatch)
    pad = " older words" * 100
    parent = [engine._store.append("P", {"role": role, "content": f"[P{i}]{pad}"}, conversation_id="conv")
              for i, role in enumerate(("user", "assistant"))]
    rows = _store(engine, [f"[R{i}]{pad}" for i in range(1, 5)] + ["[N] the newest question", "[N] answer " * 150])
    carried = [{"role": "user", "content": f"[P0]{pad}"}, {"role": "assistant", "content": f"[P1]{pad}"}]
    monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, **k: [("P", parent[0] - 1, parent[1])])
    real = engine._get_store_id_map_for_messages
    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda messages, *a, **k: {
        **real(messages, *a, **k), **{id(m): s for m, s in zip(carried, parent) if any(m is x for x in messages)}})
    active = [{"role": "system", "content": "system prompt"}, *carried, *(row for _, row in rows)]
    first, second = _two_fits(engine, active, [f"[M{i}]{pad}" for i in range(1, 3)] + ["[N2] next", "[N2] " * 300])
    assert not any(str(m["content"]).startswith("[P") for m in second)
    named = {(a, b): args["session_id"] for a, b, args in _calls(engine._last_survival_fit["notice"])}
    assert named.get((parent[0], parent[1])) == "P", engine._last_survival_fit["notice"]


# -- review fix round 2 (each cell red on fcd5a1ee, green after) ----------------------------------------------------

def test_r2_following_the_notice_reads_a_row_past_lcm_expands_budget_whole(engine):
    """A row over lcm_expand's token budget comes back in parts (has_more, next_content_offset): following the notice
    reads it whole."""
    long = " ".join(f"w{i}" for i in range(12_000))
    rows = _store(engine, ["row 1", long, "row 3"])
    messages, ids = _listed(rows, {1, 3})
    (first, last, args), = _calls(survival_fit_notice(engine, messages, set(), store_ids=ids))
    assert json.loads(lcm_expand({"store_id": 2}, engine=engine))["has_more"]  # the premise: one call is not enough
    assert [(m["store_id"], m["content"]) for m in _load(engine, args, last)] == [(2, long)]


def test_r2_a_one_row_ranges_call_reads_exactly_that_row(engine):
    """Each range's call carries its own limit (at most 200): a one-row range with many later rows reads that row."""
    rows = _store(engine, [f"row {i}" for i in range(1, 400)])
    messages, ids = _listed(rows, set(range(1, 400)) - {10})
    (first, last, args), = _calls(survival_fit_notice(engine, messages, set(), store_ids=ids))
    page = json.loads(lcm_load_session(dict(args), engine=engine))
    assert (first, last) == (10, 10) and [m["store_id"] for m in page["messages"]] == [10], args


def _rotation(engine, monkeypatch, end=None):
    """Parent rows P:1..4 and child rows 5..10; the carry range is half-open, ("P", 2, end): P:3 and P:4 are carried
    (end 4), and the map gives the carried list rows their parent ids, as a rotation does."""
    _rough(monkeypatch)
    pad = " older words" * 100
    parent = [engine._store.append("P", {"role": role, "content": f"[P{i}]{pad}"}, conversation_id="conv")
              for i, role in zip(range(1, 5), ("user", "assistant", "user", "assistant"))]
    rows = _store(engine, [f"[R{i}]{pad}" for i in range(1, 5)] + ["[N] the newest question", "[N] answer " * 150])
    carried = [{"role": "user", "content": f"[P3]{pad}"}, {"role": "assistant", "content": f"[P4]{pad}"}]
    monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, **k: [("P", parent[1], end or parent[3])])
    real = engine._get_store_id_map_for_messages
    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda messages, *a, **k: {
        **real(messages, *a, **k), **{id(m): s for m, s in zip(carried, parent[2:]) if any(m is x for x in messages)}})
    return pad, parent, rows, carried


def _named(engine, session="P"):
    return [(a, b) for a, b, args in _calls(engine._last_survival_fit["notice"]) if args["session_id"] == session]


def test_r2_the_carry_range_is_half_open_and_a_carried_row_in_view_is_not_named(engine, monkeypatch):
    """Fit one drops the carried rows: P:3..4 are named, not P:2 (the range's exclusive start). Fit two keeps a child
    row byte-equal to P:3: P:3 is in view and only P:4 is named."""
    pad, parent, rows, carried = _rotation(engine, monkeypatch)
    active = [{"role": "system", "content": "system prompt"}, *carried, *(row for _, row in rows)]
    first, second = _two_fits(engine, active, [f"[P3]{pad}", f"[M2]{pad}", "[N2] next", "[N2] " * 300], cap=1400)
    assert any(m.get("content") == f"[P3]{pad}" for m in second)  # the premise: P:3's copy is in the list
    assert _named(engine) == [(4, 4)], engine._last_survival_fit["notice"]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    engine._survival_fit(active, list(active), 0, "test", request_cap=900)
    assert _named(engine) == [(3, 4)], engine._last_survival_fit["notice"]


def test_r2_a_carried_row_under_a_shown_summary_is_not_named(engine, monkeypatch):
    """The list holds no carried row but keeps a summary whose node covers P:3..4: they are in view, not named."""
    pad, parent, rows, carried = _rotation(engine, monkeypatch)
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="parent turns three and four", token_count=1,
                                            source_token_count=1, source_ids=parent[2:], expand_hint="turns"))
    summary = {"role": "user", "content": f"[Recent Summary (d0, node {node})]\nparent turns three and four\n"
                                          "[Expand for details: turns]"}
    active = [{"role": "system", "content": "system prompt"}, summary, *(row for _, row in rows)]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    result = engine._survival_fit(active, list(active), 0, "test", request_cap=1500)
    assert result[1]["content"] == summary["content"] and engine._last_survival_fit["notice"], result  # the premise
    assert _named(engine) == [], engine._last_survival_fit["notice"]


def test_r2_the_carry_read_does_not_grow_with_the_ranges_span(engine, monkeypatch):
    """The same carried rows inside a 100,000-id span and a tight one: the fit's own reads are the same statements (one
    indexed read of the source session's own rows per carry range, whose bound alone differs) and the notice is the
    same."""
    pad, parent, rows, carried = _rotation(engine, monkeypatch)
    active = [{"role": "system", "content": "system prompt"}, *carried, *(row for _, row in rows)]
    own = ("SELECT store_id FROM messages WHERE session_id", "WITH RECURSIVE walk(root, node_id)",
           "SELECT store_id, session_id, role, content, tool_calls, tool_call_id FROM messages")
    seen = []
    for end in (parent[3], 100_000):
        monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, end=end, **k: [("P", parent[1], end)])
        statements = []
        engine._store.connection.set_trace_callback(statements.append)
        try:
            engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
            engine._survival_fit(active, list(active), 0, "test", request_cap=900)
        finally:
            engine._store.connection.set_trace_callback(None)
        reads = [x.replace("store_id <= 100000", f"store_id <= {parent[3]}") for x in statements if x.startswith(own)]
        seen.append((reads, engine._last_survival_fit["notice"]))
    assert seen[1] == seen[0] and sum(x.startswith(own[0]) for x in seen[0][0]) == 1, [len(x[0][-1]) for x in seen]
    plan = " ".join(str(row) for row in engine._store.connection.execute(
        "EXPLAIN QUERY PLAN " + survival_fit._CARRIED_SQL, ("P", 2, 100_000)).fetchall())
    assert "idx_msg_session" in plan, plan


def test_r2_tool_calls_keys_follow_replay_identity(engine):
    """An unmapped listed row whose calls are a duplicate-key JSON string is not a copy of a stored row holding the
    last key's value (replay identity keeps them apart); a host's string calls, stored and listed again, are a copy."""
    engine._store.append("S", {"role": "assistant", "content": "same", "tool_calls": [{"id": "new"}]}, conversation_id="conv")
    engine._store.append("S", {"role": "assistant", "content": "other", "tool_calls": '[{"id":"x"}]'}, conversation_id="conv")
    kept = _store(engine, ["u3", "r4"])
    messages, ids = _listed(kept, {3, 4})
    listed = [*messages, {"role": "assistant", "content": "same", "tool_calls": '[{"id":"old","id":"new"}]'},
              {"role": "assistant", "content": "other", "tool_calls": '[{"id":"x"}]'}]
    (first, last, _), = _calls(survival_fit_notice(engine, listed, set(), store_ids=ids))
    assert (first, last) == (1, 1)


# -- the notice shape (each session id kept; the ceiling; the budget edge) ------------------------------------------

def _b488389d(count, ranges, rest=None):
    """The reviewed head b488389d's notice for the same input, as its formula wrote it (the ceiling's reference)."""
    groups = []
    for session, first, last in ranges:
        if not groups or groups[-1][0] != session:
            groups.append((session, []))
        groups[-1][1].append(f"after_store_id={first - 1} for store ids {first}..{last}")
    calls = "; ".join(f"session_id={session} " + ", ".join(items) for session, items in groups)
    calls += survival_fit._NOTICE_REST.format(count=rest[0], first=rest[1], last=rest[2]) if rest else ""
    bounds = [x for _, first, last in ranges for x in (first, last)] + (list(rest[1:]) if rest else [])
    return ("[LCM survival fit: {n} earlier messages (store ids {first}..{last}) are stored verbatim but not in live "
            "context; read each range with lcm_load_session and limit=200, paging with next_cursor: {calls}.]").format(
        n=count, first=min(bounds), last=max(bounds), calls=calls)


EDGE_IDS = {1: "S", 24: COLON_ID[:24], 81: COLON_ID}
EDGE_TESTS = {"916": ("test_issue_916_projection_keeps_head.py",
                      ("test_newest_user_keeps_opening_under_survival_budget",
                       "test_head_plus_mark_re_ingests_without_storing_again")),
              "608": ("test_issue_608_sweep_budget_stop.py",
                      ("test_a_second_recovery_attempt_shrinks_again_and_stops_where_nothing_can_leave",))}


def _edge_run(work: Path, lengths, mutate=lambda key, source: source) -> dict:
    """The #916 and #608 budget-edge cells as written (``mutate`` may edit their source), with the session id "S"
    replaced engine-wide by an id of each length (every "S" literal in their source), in one isolated pytest process:
    {(file, length, test): (outcome, the fit's logged tokens and budget, the failure text, its terminal message)}."""
    targets = []
    for key, (name, tests) in EDGE_TESTS.items():
        source = mutate(key, (TREE / "tests" / name).read_text())
        for length in lengths:
            sid = EDGE_IDS[length]
            assert len(sid) == length and set(sid) <= set("abcdefghijklmnopqrstuvwxyz0123456789:_-S")
            (work / f"test_edge_{key}_{length}.py").write_text(source.replace('"S"', json.dumps(sid)))
            targets += [f"test_edge_{key}_{length}.py::{test}" for test in tests]
    (work / "pytest.ini").write_text((TREE / "pytest.ini").read_text())
    paths = [str(TREE), *_syspaths()]
    if (TREE / "agent" / "context_engine.py").is_file():  # CI's host stub, as _walk takes it
        shutil.copytree(TREE / "agent", work / "stub" / "agent")
        paths.append(str(work / "stub"))
    report = work / "report.xml"
    script = (f"import sys; sys.path[:0] = {paths!r}; import pytest; sys.exit(pytest.main(['-p', 'tests.conftest', "
              f"'-p', 'no:cacheprovider', '-o', 'addopts=', '-o', 'junit_logging=all', '--junitxml', {str(report)!r}, "
              f"'--basetemp', {str(work / 'bt')!r}, *{targets!r}]))")
    home = work / "home"
    home.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0"}
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=work, env=env, capture_output=True,
                          text=True, timeout=600)
    assert done.returncode in (0, 1) and report.is_file(), done.stdout[-2000:] + done.stderr[-2000:]
    runs = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        logged = "".join(node.text or "" for node in case.iter() if node.tag in ("system-out", "system-err"))
        fits = [tuple(map(int, found)) for found in re.findall(r"tokens=(\d+)->(\d+), budget=(\d+)\)", logged)]
        failed = [node for node in case if node.tag in ("failure", "error")]
        failure = "".join(node.text or "" for node in failed)
        runs[(case.get("classname"), case.get("name"))] = (not failed, fits, failure,
                                                           failed[0].get("message", "") if failed else "")
    return runs


@pytest.fixture(scope="module")
def edge_runs(tmp_path_factory):
    return _edge_run(tmp_path_factory.mktemp("edge"), sorted(EDGE_IDS))


class EdgeMoved(Exception):
    """A budget-edge cell as written failed with a longer session id: the accepted trade-off below."""


def _edge_reason(what: str) -> str:
    return ("accepted trade-off (#2877): every named range keeps its explicit session id, so a longer id moves the "
            f"budget edge against main by the id's length (view only, the same cliff main has at its own edge): {what}; "
            "no stored row is lost")


# the cause each trade-off names: the line that fails and the message it raises, in the terminal exception (the last
# section of a chained traceback), so a later failure (in a finally, at shutdown) is never taken for it
EDGE_MESSAGE = {"916": "AssertionError: assert False", "608": "AssertionError: assert (1 == 1 and 2 == 1)"}
EDGE_CAUSE = {
    "916": re.compile(r">\s+assert (?:newest|projected)\[\"content\"\]\.startswith\(view\[-1\]\[\"content\"\]\[:1200\]\)\n"),
    "608": re.compile(r">\s+assert _count\(caplog, \"could not reach budget\"\) == 1 and _fit_records\(engine\) == 1\n"
                      r"E\s+AssertionError: assert \(1 == 1 and 2 == 1\)\n(?:E .*\n)*?E\s+\+\s+and\s+2 = _fit_records\(")}
_916_NEWEST = ("the 1-character id's fit uses 397 of 399 tokens; the longer id's notice does not fit those 2, so the "
               "newest user row goes mark-only")
_608_FLOOR = ("the floor attempt (budget 140) writes a second fit record where the 1-character id's returns the same "
              "list once")
EDGE = [
    *(pytest.param(key, test, length, fits, marks=() if length == 1 else pytest.mark.xfail(
        strict=True, raises=EdgeMoved, reason=_edge_reason(f"{what}, measured {fits}")), id=f"{key}-{short}-id{length}")
      for key, test, short, what, by_length in (
          ("916", "test_newest_user_keeps_opening_under_survival_budget", "newest", _916_NEWEST,
           {1: [(594, 397, 399)], 24: [(594, 102, 399)], 81: [(594, 116, 399)]}),
          ("916", "test_head_plus_mark_re_ingests_without_storing_again", "reingest", _916_NEWEST,
           {1: [(594, 397, 399)], 24: [(594, 102, 399)], 81: [(594, 116, 399)]}),
          ("608", "test_a_second_recovery_attempt_shrinks_again_and_stops_where_nothing_can_leave", "D7", _608_FLOOR,
           {1: [(2896, 1508, 1850), (1508, 787, 1131), (2896, 426, 140)],
            24: [(2896, 1514, 1850), (1514, 793, 1136), (2896, 432, 140)],
            81: [(2896, 1528, 1850), (1528, 807, 1148), (2896, 446, 140)]}))
      for length, fits in by_length.items())]


@pytest.mark.parametrize("key,test,length,fits", EDGE)
def test_the_budget_edge_with_a_longer_session_id_is_pinned(edge_runs, key, test, length, fits):
    """Strict: the #916 and #608 budget-edge cells as written pass with a 1-character session id and fail with 24 and
    81 characters (colon keys), each at its measured fits (tokens before, after, budget), and only at the line its
    trade-off names: any other failure is a real one."""
    _edge_verdict(edge_runs, key, test, length, fits)


def _edge_verdict(runs, key, test, length, fits):
    """EdgeMoved when every case of the cell failed with the named cause as its terminal exception; passes when every
    case passed; any other outcome is an AssertionError."""
    cases = {name: run for (module, name), run in runs.items()
             if module == f"test_edge_{key}_{length}" and name.split("[")[0] == test}
    assert cases and all(run[1] == fits for run in cases.values()), cases
    if all(run[0] for run in cases.values()):
        return
    chain = re.compile(r"\n(?:During handling of the above exception, another exception occurred|The above exception "
                       r"was the direct cause of the following exception):\n")
    assert all(not passed and message.startswith(EDGE_MESSAGE[key]) and EDGE_CAUSE[key].search(chain.split(text)[-1])
               for passed, _, text, message in cases.values()), [(m, t[-600:]) for _, _, t, m in cases.values()]
    raise EdgeMoved(f"#{key} {test} with a {length}-character session id")


@pytest.mark.parametrize("length", sorted(EDGE_IDS))
def test_the_notice_is_no_longer_than_b488389ds_for_the_same_input(length):
    """The ceiling: for one session id length, single ranges from 1- to 12-digit store ids and of several widths, several
    ranges, a second session (a rotation's parent) and the counted clause, the notice is never longer than b488389d's."""
    sid, other = EDGE_IDS[length], COLON_ID[::-1][:length]
    inputs = [(last - first + 1, [(sid, first, last)], None) for digits in range(1, 13) for width in (1, 3, 200, 201, 5000)
              for first in [10 ** (digits - 1)] for last in [first + width - 1]]
    inputs += [(9, [(sid, 1, 3), (sid, 12, 14), (sid, 1_200, 1_202)], None),
               (8, [(other, 7, 9), (sid, 10 ** 6, 10 ** 6 + 4)], (4, 1, 5)),
               (40, [(sid, 10 ** 11 + 3 * i, 10 ** 11 + 3 * i + 1) for i in range(8)], (12, 1, 10 ** 10))]
    for count, ranges, rest in inputs:
        text = survival_fit.survival_notice_text(count, ranges, rest)
        assert len(text) <= len(_b488389d(count, ranges, rest)), (text, _b488389d(count, ranges, rest))


def test_r2_one_range_whose_id_passes_the_cap_falls_back_to_the_counted_notice(tmp_path):
    """NB3: a single range whose session id alone passes the token cap: the counted form (no call), within the cap."""
    sid = "a1" * 600
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "long.db"), survival_reserve=0.0))
    engine._hermes_home = str(tmp_path)
    engine.on_session_start(sid, platform="telegram", conversation_id="conv", context_length=20_000)
    try:
        rows = _store(engine, [f"row {i}" for i in range(1, 7)], session=sid)
        messages, ids = _listed(rows, {4, 5, 6})
        notice = survival_fit_notice(engine, messages, set(), store_ids=ids)
        assert notice == survival_fit._NOTICE.format(n=3, first=1, last=3), notice[:200]
        assert count_tokens(notice) <= _NOTICE_MAX_TOKENS
    finally:
        engine.shutdown()


def test_r2_the_counted_fallback_spans_interleaved_sessions_ranges():
    """NB3 with two sessions: this session's rows 1..9 (one range in its row order) and a parent's rows 4..5 between
    them, whose long id keeps the newest named range over the cap. The counted notice's span is 1..9, the true bounds."""
    order = [1, 2, 3, 8, 9]
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         {4: "p1" * 600, 5: "p1" * 600})
    notice = survival_fit_notice(SimpleNamespace(_session_id="S", _store=None), [], set(), left=[4, 5], read=read,
                                 store_ids={})
    assert notice == survival_fit._NOTICE.format(n=7, first=1, last=9), notice[:200]


# -- review fix round 3 (each cell red on 49931b41, green after) ----------------------------------------------------

def test_r3_a_carried_row_whose_copy_is_under_a_kept_summary_is_not_named(engine):
    """P:3's byte-equal child copy S:8 sits under the kept node, and no uncovered child row has P:3's role and length:
    the read still loads S:8's key, so P:3 is in view and not named."""
    add = lambda s, role, content: engine._store.append(s, {"role": role, "content": content}, conversation_id="conv")
    for role, content in (("user", "p one"), ("assistant", "p two"), ("user", "same"), ("assistant", "p four")):
        add("P", role, content)
    for role, content in (("user", "child five"), ("assistant", "child six"), ("assistant", "child seven"),
                          ("user", "same"), ("assistant", "child nine"), ("user", "child ten q")):
        add("S", role, content)
    listed = [{"role": "user", "content": "[N] newest q"}, {"role": "assistant", "content": "[N] newest answer"}]
    ids = {id(m): add("S", m["role"], m["content"]) for m in listed}
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="covers eight", token_count=1, source_token_count=1,
                                            source_ids=[8], expand_hint="t"))
    read = read_out_of_view(engine._store, "S", {node}, kept={node}, also=[3, *ids.values()])  # the fit's form
    notice = survival_fit_notice(engine, listed, {node}, carried=[3], store_ids=ids, read=read)
    assert not [args for _, _, args in _calls(notice) if args["session_id"] == "P"], notice


def test_r3_a_carried_copy_under_a_kept_summary_does_not_push_a_real_range_out_of_its_call(engine):
    """Out-of-view carried P:1, a carried P:3 whose copy is under the kept node, and seven child ranges: eight real
    ranges, so P:1 keeps its call."""
    add = lambda s, role, content: engine._store.append(s, {"role": role, "content": content}, conversation_id="conv")
    p1 = add("P", "user", "p one")
    add("P", "assistant", "p two")
    p3 = add("P", "user", "same")
    rows = [add("S", "assistant", f"{kind} {i}") for i in range(7) for kind in ("gone", "shown")]
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="n", token_count=1, source_token_count=1,
                                            source_ids=[add("S", "user", "same")], expand_hint="t"))
    listed = [{"role": "assistant", "content": f"shown {i}"} for i in range(7)]
    ids = dict(zip(map(id, listed), rows[1::2]))
    read = read_out_of_view(engine._store, "S", {node}, kept={node}, also=[p1, p3, *ids.values()])
    notice = survival_fit_notice(engine, listed, {node}, carried=[p1, p3], store_ids=ids, read=read)
    assert ("P", p1, p1) in {(args["session_id"], a, b) for a, b, args in _calls(notice)}, notice
    assert "earlier ranges" not in notice, notice


def test_r3_the_copy_kept_twin_through_a_fit(engine, monkeypatch):
    """A parent's carried row whose byte-equal child copy is under the summary the fit keeps: through _survival_fit,
    with or without an uncovered child row of the same role and length, the row is not named."""
    _rough(monkeypatch)
    for twin in (False, True):
        session = f"C{int(twin)}"
        engine.on_session_start(session, platform="telegram", conversation_id="cv", context_length=200_000)
        store = engine._store
        texts = [f"parent turn {i} " + "alpha beta " * (20 + 3 * i) for i in range(5)]
        roles = ("user", "assistant", "user", "assistant", "user")
        parent = [store.append(f"PAR{int(twin)}", {"role": r, "content": t}, conversation_id="cv")
                  for r, t in zip(roles, texts)]
        copy = store.append(session, {"role": "user", "content": texts[2]}, conversation_id="cv")
        node = engine._dag.add_node(SummaryNode(session_id=session, summary="child summary of the copied parent turn",
                                                token_count=1, source_token_count=1, source_ids=[copy], expand_hint="x"))
        summary = {"role": "user", "content": f"[Recent Summary (d0, node {node})]\nchild summary of the copied parent "
                                              "turn\n[Expand for details: x]"}
        tail = ([{"role": "user", "content": "Z" * len(texts[2])}] if twin else []) + [
            {"role": "user", "content": "newest question"}, {"role": "assistant", "content": "newest answer"}]
        bulk = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old {i} " + "gamma " * 120} for i in range(4)]
        for m in [*bulk, *tail]:
            store.append(session, dict(m), conversation_id="cv")
        monkeypatch.setattr(engine, "_load_compression_carry_ranges",
                            lambda *a, p=parent, s=f"PAR{int(twin)}", **k: [(s, p[0] - 1, p[-1])])
        active = [{"role": "system", "content": "sys"}, summary, *bulk, *tail]
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
        result = engine._survival_fit(active, list(active), 0, "test", request_cap=_rough(monkeypatch)(active) // 2)
        assert any(m.get("content") == summary["content"] for m in result), twin  # the premise: the summary is kept
        named = {x for a, b, args in _calls(engine._last_survival_fit["notice"]) if args["session_id"] == f"PAR{int(twin)}"
                 for x in parent if a <= x <= b}
        assert parent[2] not in named and set(parent) - {parent[2]} <= named, (twin, engine._last_survival_fit["notice"])


def test_r3_the_counted_clause_spans_every_range_it_folds():
    """This session's rows 1 and 100 (one range in its row order) and nine one-row parent ranges between them: the cap
    folds the two oldest by start, and the counted clause and the head both span 1..100."""
    order, parent = [1, 100], list(range(2, 19, 2))
    read = OutOfViewRead(frozenset(), frozenset(), order, {s: i for i, s in enumerate(order)}, {}, order, {}, {}, {},
                         dict.fromkeys(parent, "P"))
    notice = survival_fit_notice(SimpleNamespace(_session_id="S", _store=None), [], set(), carried=parent, read=read,
                                 store_ids={})
    assert notice.startswith("[LCM survival fit: 11 earlier messages (store ids 1..100)"), notice
    assert "; and 2 earlier ranges (store ids 1..100); lcm_grep reaches them" in notice, notice


def _interleaved(engine, monkeypatch):
    """Twenty turns where this session's rows and its parent's carried rows alternate in the store (one each: S at odd
    ids 1..39, P at even ids 2..40), then newer rows; one fit. Returns the notice, the rows its calls name, the spans it
    states (head and counted clause) and the out-of-view rows."""
    _rough(monkeypatch)
    pad = " older words" * 60
    parent = []
    for i in range(20):
        engine._store.append("S", {"role": "user" if i % 2 == 0 else "assistant", "content": f"[S{i}]{pad}"},
                             conversation_id="conv")
        parent.append(engine._store.append("P", {"role": "user" if i % 2 == 0 else "assistant",
                                                 "content": f"[P{i}]{pad}"}, conversation_id="conv"))
    for i in range(20, 26):
        engine._store.append("S", {"role": "user" if i % 2 == 0 else "assistant", "content": f"[S{i}]{pad}"},
                             conversation_id="conv")
    monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, **k: [("P", parent[0] - 1, parent[-1])])
    recent = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"[K{i}]{pad}"} for i in range(26, 34)]
    for m in recent:
        engine._store.append("S", dict(m), conversation_id="conv")
    active = [{"role": "system", "content": "system prompt"}, *recent]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    result = engine._survival_fit(active, list(active), 0, "test", request_cap=_rough(monkeypatch)(active) // 2)
    notice = engine._last_survival_fit["notice"]
    conn = engine._store.connection
    named = set()
    for first, last, args in _calls(notice):
        named |= {x for (x,) in conn.execute("SELECT store_id FROM messages WHERE session_id = ? AND store_id >= ? AND "
                                             "store_id <= ?", (args["session_id"], first, last))}
    spans = [tuple(map(int, found)) for found in re.findall(r"store ids (\d+)\.\.(\d+)\)", notice)]
    listed = {m["content"] for m in result}
    out = [x for x, content in conn.execute("SELECT store_id, content FROM messages ORDER BY store_id")
           if content not in listed]
    assert parent == list(range(2, 41, 2)) and set(range(41, 53)) <= set(out), out  # the premise
    return notice, named, spans, out


def _interleaved_parent_calls(notice) -> list[tuple[int, int]]:
    return [(a, b) for a, b, args in _calls(notice) if args["session_id"] == "P"]


# the newest 8 ranges by start id are named: the parent's one-row ranges at 26, 28, ..., 40; the rest are counted
INTERLEAVED_PARENT_CALLS = [(x, x) for x in range(26, 41, 2)]


def test_r3_interleaved_parent_and_child_rows_leave_no_row_outside_every_pointer(engine, monkeypatch):
    """Every out-of-view row is named or inside a counted span (rows 41..52 included), and, separately from the
    spans, each expected parent call is there."""
    notice, named, spans, out = _interleaved(engine, monkeypatch)
    assert not [x for x in out if x not in named and not any(a <= x <= b for a, b in spans)], notice
    assert _interleaved_parent_calls(notice) == INTERLEAVED_PARENT_CALLS, notice


def test_r4_the_interleaved_cell_fails_when_a_parent_call_is_omitted(engine, monkeypatch):
    """A mutant that names one range fewer drops P:26's call into the counted clause: the spans still hold every row,
    and only the separate call check catches it."""
    monkeypatch.setattr(survival_fit, "_NOTICE_MAX_RANGES", _NOTICE_MAX_RANGES - 1)
    notice, named, spans, out = _interleaved(engine, monkeypatch)
    assert not [x for x in out if x not in named and not any(a <= x <= b for a, b in spans)], notice
    assert _interleaved_parent_calls(notice) == INTERLEAVED_PARENT_CALLS[1:] != INTERLEAVED_PARENT_CALLS, notice


SHUTDOWN_RAISES = {  # a mutant whose engine shutdown raises in the cell's finally, after the accepted assertion failed
    "608": ('        assert _count(caplog, "could not reach budget") == 1 and _fit_records(engine) == 1\n'
            '    finally:\n        engine.shutdown()\n'),
    "916": ('        assert engine._store.get_session_messages("S")[-1]["content"] == view[-1]["content"]\n'
            '    finally:\n        engine.shutdown()\n')}


def test_r4_a_failure_at_shutdown_is_never_taken_for_the_accepted_cause(tmp_path):
    """With a 24-character id the accepted assertion fails, then shutdown raises in the finally: the accepted line is
    in the traceback, but the terminal exception is the shutdown's, so the wrapper raises AssertionError, never
    EdgeMoved (its strict xfail would go red)."""
    def mutate(key, source):
        assert source.count(SHUTDOWN_RAISES[key]) == 1
        return source.replace(SHUTDOWN_RAISES[key], SHUTDOWN_RAISES[key] + '        raise RuntimeError("shutdown failed")\n')

    runs = _edge_run(tmp_path, [24], mutate)
    for param in EDGE:
        key, test, length, fits = param.values
        if length != 24 or "reingest" in param.id:
            continue
        cases = [run for (module, name), run in runs.items() if module == f"test_edge_{key}_24" and name == test]
        assert cases and all(EDGE_CAUSE[key].search(text) and "shutdown failed" in message for _, _, text, message in cases)
        with pytest.raises(AssertionError):
            _edge_verdict(runs, key, test, length, fits)


def test_r3_an_emergency_fit_runs_each_carry_query_once(engine, monkeypatch, caplog):
    """The emergency cut re-reads the out-of-view rows for its own shown set; the carried ids of the same list are
    kept, so each carry range's statement runs once per fit."""
    _rough(monkeypatch)
    active = _emergency_list(engine)
    parent = [engine._store.append("P", {"role": "user", "content": f"[P{i}] parent"}, conversation_id="conv")
              for i in range(2)]
    monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, **k: [("P", parent[0] - 1, parent[-1])])
    statements = []
    engine._store.connection.set_trace_callback(statements.append)
    try:
        engine._survival_fit(active, list(active), 0, "test", request_cap=900)
    finally:
        engine._store.connection.set_trace_callback(None)
    assert "dropped the summary prefix" in caplog.text  # the premise: the emergency path ran
    assert sum(x.startswith("SELECT store_id FROM messages WHERE session_id = 'P'") for x in statements) == 1, statements
