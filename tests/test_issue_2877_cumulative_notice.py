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
NOTICE = re.compile(r"^\[LCM survival fit: (\d+) earlier messages \(store ids (\d+)\.\.(\d+)\) are stored verbatim but not "
                    r"in live context; read with lcm_load_session limit=(\d+), next_cursor, and lcm_expand\(store_id\) "
                    r"if truncated: (.*)\.\]$", re.S)
GROUP = re.compile(r"session_id=([^;\s]+) ((?:after_store_id=\d+ to \d+(?:, )?)+)")
ITEM = re.compile(r"after_store_id=(\d+) to (\d+)")


def _calls(notice: str) -> list[tuple[int, int, dict]]:
    """(first, last, arguments) of every range the notice names, its arguments parsed from the text."""
    head = NOTICE.match(notice)
    assert head, notice
    calls = [(int(after) + 1, int(stop), {"session_id": session, "after_store_id": int(after), "limit": int(head.group(4))})
             for session, items in GROUP.findall(head.group(5)) for after, stop in ITEM.findall(items)]
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
        "lcm_load_session limit=200, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=0 "
        "to 3.]")
    messages, ids = _listed(rows, {3, 4, 7})
    assert survival_fit_notice(engine, messages, set(), store_ids=ids) == (
        "[LCM survival fit: 7 earlier messages (store ids 1..10) are stored verbatim but not in live context; read with "
        "lcm_load_session limit=200, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=0 "
        "to 2, after_store_id=4 to 6, after_store_id=7 to 10.]")
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
                      "context; read with lcm_load_session limit=200, next_cursor, and lcm_expand(store_id) if truncated: "
                      "session_id=S after_store_id=0 to 10.]")
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
    # no longer than the single-range notice of the reviewed head b488389d (228 characters): budget-edge tests depend on it
    assert len(survival_fit.survival_notice_text(3, [("S", 1, 3)])) <= 228
    assert count_tokens(single) <= count_tokens(today) + 40, (count_tokens(single), count_tokens(today))
    assert json.dumps(notice)[1:-1] == notice and json.dumps(single)[1:-1] == single  # nothing JSON escapes


def _load(engine, args: dict, stop: int = 10 ** 15) -> list[dict]:
    """Follow the notice exactly: the call as written (its arguments parsed from the text), paging with next_cursor,
    stopping after the range's last store id, and a row marked content_truncated read whole with lcm_expand."""
    rows, page = [], json.loads(lcm_load_session(dict(args), engine=engine))
    while True:
        for message in page["messages"]:
            if message["store_id"] > stop:
                return rows
            if message["content_truncated"]:
                message = {**message, **json.loads(lcm_expand({"store_id": message["store_id"]}, engine=engine)),
                           "content_truncated": False}
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
        "lcm_load_session limit=200, next_cursor, and lcm_expand(store_id) if truncated: session_id=S after_store_id=4 "
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
        assert (first, last) == (1, 5) and args == {"session_id": COLON_ID, "after_store_id": 0, "limit": 200}, notice
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
    monkeypatch.setattr(engine, "_load_compression_carry_ranges", lambda *a, **k: [("P", parent[0], parent[1])])
    real = engine._get_store_id_map_for_messages
    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", lambda messages, *a, **k: {
        **real(messages, *a, **k), **{id(m): s for m, s in zip(carried, parent) if any(m is x for x in messages)}})
    active = [{"role": "system", "content": "system prompt"}, *carried, *(row for _, row in rows)]
    first, second = _two_fits(engine, active, [f"[M{i}]{pad}" for i in range(1, 3)] + ["[N2] next", "[N2] " * 300])
    assert not any(str(m["content"]).startswith("[P") for m in second)
    named = {(a, b): args["session_id"] for a, b, args in _calls(engine._last_survival_fit["notice"])}
    assert named.get((parent[0], parent[1])) == "P", engine._last_survival_fit["notice"]
