"""#2896: the survival notice on a list with no leading system row, as one marked block in the summary head.

The summary verifier (_verified_lcm_summary_prefix) admits exactly one marked block, right after the last verified
part and before a separator or the end of the text. Everything that reads the head through it sees the same remainder,
parts and coverage as without the block. Anything else that looks like the block (malformed, before the first part,
between parts) is content, exactly as on main; so are notices a user pasted and notice-like text inside a summary.
"""

from __future__ import annotations

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.emission_projection import survival_notice_block
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value



@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_reserve=0.0))
    obj._hermes_home = str(tmp_path)
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
    yield obj
    obj.shutdown()


def _part(engine, summary="earlier turns, condensed", hint="turns") -> str:
    node = engine._dag.add_node(SummaryNode(session_id="S", summary=summary, token_count=1, source_token_count=1,
                                            source_ids=[1, 2], expand_hint=hint))
    return f"[Recent Summary (d0, node {node})]\n{summary}\n[Expand for details: {hint}]"


def _block(engine, parts: str, count=3, first=1, last=3) -> str:
    """The block the fit writes after ``parts``: its digest binds it to those parts, their node ids and the notice."""
    notice = survival_fit.survival_notice_text(count, [("S", first, last)])
    return survival_notice_block(parts, engine._verified_lcm_summary_prefix(parts)[1], notice)


def _read(engine, content: str) -> dict:
    end, nodes = engine._verified_lcm_summary_prefix(content)
    message = {"role": "user", "content": content}
    return {"end": end, "nodes": nodes, "remainder": engine._generated_context_carrier_remainder(message),
            "scaffold": engine._is_verified_replay_scaffold_message(message),
            "replayed": engine._is_replayed_context_scaffold_message(message)}


def _as_on_main(engine, monkeypatch, content: str) -> dict:
    with monkeypatch.context() as patch:
        patch.setattr(lcm_engine, "admitted_notice_block", lambda *args: None)  # main: no block is admitted
        return _read(engine, content)


@pytest.mark.parametrize("parts", [1, 2])
def test_a_a_carrier_with_the_block_reads_as_without_it(engine, parts):
    """(a) The block widens the prefix by exactly its own bytes; the remainder, the parts and the coverage are those
    of the same carrier without it, and the block's start is the parts' end."""
    summary = "\n\n---\n\n".join(_part(engine, f"turns {i}") for i in range(parts))
    block = _block(engine, summary)
    plain, carrier = f"{summary}\n\nthe user's row", f"{summary}{block}\n\nthe user's row"
    without, with_block = _read(engine, plain), _read(engine, carrier)
    assert without["end"] == len(summary) and with_block["end"] == len(summary) + len(block)
    assert {**with_block, "end": None} == {**without, "end": None} and with_block["remainder"] == "the user's row"
    assert survival_fit.survival_notice_block_start(carrier, with_block["end"]) == len(summary)
    assert survival_fit.survival_notice_block_start(plain, without["end"]) is None


def test_b_a_pure_summary_row_with_the_block_is_verified_scaffold(engine):
    """(b) The block may end the row: a summary row holding only its parts and the block is verified scaffold."""
    summary = _part(engine)
    row = _read(engine, summary + _block(engine, summary))
    assert row == {**_read(engine, summary), "end": len(summary) + len(_block(engine, summary))}
    assert row["scaffold"] and row["replayed"] and row["remainder"] is None


def test_c_hermes_merged_form_gives_the_same_remainder(engine):
    """(c) A host that merges the summary row (with its block) and the next user row as prev + blank line + next."""
    summary_row = _part(engine)
    summary_row += _block(engine, summary_row)
    merged = summary_row + "\n\n" + "next user row"  # Hermes _merge_consecutive_users
    assert _read(engine, merged)["remainder"] == "next user row"
    assert _read(engine, merged)["nodes"] == _read(engine, summary_row)["nodes"]


@pytest.mark.parametrize("shape", ["unclosed", "inner text", "no notice", "before the first part",
                                   "between parts", "not a separator after", "no digest", "another summary's digest",
                                   "another notice's digest"])
def test_d_anything_else_is_not_a_block(engine, monkeypatch, shape):
    """(d) A malformed, misplaced or unbound block is content: the verifier reads it exactly as main does. A block's
    digest binds it to the parts before it and its notice, so a hand-written or copied block is not one."""
    one, two = _part(engine, "turns one"), _part(engine, "turns two")
    block = _block(engine, one)
    notice = block.split("\n")[3]
    content = {
        "unclosed": f"{one}{block[:-len('</lcm-survival-notice>')]}\n\nuser row",
        "inner text": f"{one}{block.replace('[LCM survival fit:', 'note [LCM survival fit:')}\n\nuser row",
        "no notice": f"{one}\n\n<lcm-survival-notice>\nplain words.]\n</lcm-survival-notice>\n\nuser row",
        "before the first part": f"{block.lstrip()}\n\n{one}\n\nuser row",
        "between parts": f"{one}{block}\n\n---\n\n{two}\n\nuser row",
        "not a separator after": f"{one}{block}user row",
        "no digest": f"{one}\n\n<lcm-survival-notice>\n{notice}\n</lcm-survival-notice>\n\nuser row",
        "another summary's digest": f"{one}{_block(engine, two)}\n\nuser row",
        "another notice's digest": f"{one}{block.replace('store ids 1..3', 'store ids 1..4')}\n\nuser row",
    }[shape]
    assert _read(engine, content) == _as_on_main(engine, monkeypatch, content)
    if shape != "before the first part":
        assert _read(engine, content)["end"] == len(one)


def test_e_notices_a_user_pasted_and_notice_text_inside_a_summary_are_untouched(engine, monkeypatch):
    """(e) A notice inside the user's own row, and notice-like text inside a node's summary, are read as on main."""
    part = _part(engine)
    pasted = f"{part}\n\nI saw this earlier: {_block(engine, part).strip()} what does it mean?"
    quoted_part = _part(engine, f"the summary quotes{_block(engine, part)}\nand goes on")
    quoted = f"{quoted_part}\n\nuser row"
    for content in (pasted, quoted):
        assert _read(engine, content) == _as_on_main(engine, monkeypatch, content)
    assert _read(engine, pasted)["remainder"].startswith("I saw this earlier: <lcm-survival-notice digest=")
    assert _read(engine, quoted) == {**_read(engine, quoted), "end": len(quoted_part), "remainder": "user row"}


# -- the fit on a host list with no system row (the 650 harness: Hermes' shape, every turn stored) -----------------

from hermes_lcm.emission_projection import (  # noqa: E402
    _carrier_emission_key, _emission_identity, _finalize_emission_descriptors)
from tests.test_issue_650_survival_fit_keeps_summary import (  # noqa: E402,F401
    _engine as _hermes_engine, _hidden_backlog, _rows, _turn, host)
from tests.test_issue_2877_cumulative_notice import _calls, _load  # noqa: E402

OPEN = "\n\n<lcm-survival-notice digest="


def _notice(content: str) -> str:
    """The notice inside the head's one block."""
    assert content.count(OPEN) == 1, content[-400:]
    return content.split(OPEN, 1)[1].split("\n")[1]


def _without_block(engine, message: dict) -> dict:
    content = message["content"]
    end = engine._verified_lcm_summary_prefix_end(content)
    start = survival_fit.survival_notice_block_start(content, end)
    return {**message, "content": content[:start] + content[end:]}


def test_2_the_notice_reaches_a_list_with_no_system_row_and_its_calls_work(tmp_path, host):  # noqa: F811 (the imported pytest fixture)
    """Cell 2: the fit's notice is the re-formed carrier's one block, and each named call, run through the real
    handlers, returns its stored rows; the carrier is the fit's emission."""
    engine = _hermes_engine(tmp_path)
    try:
        view = _hidden_backlog(engine)
        result = engine.compress(view, current_tokens=host(view))
        assert result[0]["role"] == "user" and engine._last_survival_fit is not None
        notice = _notice(result[0]["content"])
        assert notice == engine._last_survival_fit["notice"]
        stored = {r["store_id"]: r["content"] for r in _rows(engine)}
        for first, last, args in _calls(notice):
            loaded = _load(engine, args, last)
            assert [(m["store_id"], m["content"]) for m in loaded] == [
                (x, stored[x]) for x in sorted(stored) if first <= x <= last], (first, last)
        emission = engine._survival_fit_emission
        assert emission["kind"] == "carrier" and emission["row"] is result[0]
        assert emission["span"].startswith(result[0]["content"][:result[0]["content"].index(OPEN)])
        assert len(_finalize_emission_descriptors(result, [emission], engine._emission_binding())) == 1
    finally:
        engine.shutdown()


def test_3_reopen_and_restart_store_nothing_twice(tmp_path, host):  # noqa: F811 (the imported pytest fixture)
    """Cell 3: the host sends the fitted list back (same process, then a cold one with a new turn): nothing is stored
    twice, and every row maps to its own stored row (the head to the user row behind its summary and block)."""
    engine = _hermes_engine(tmp_path)
    try:
        view = _hidden_backlog(engine)
        result = engine.compress(view, current_tokens=host(view))
        assert OPEN in result[0]["content"]
        ids = engine._get_store_id_map_for_messages(result)
        by_id = {r["store_id"]: r["content"] for r in _rows(engine)}
        own = [engine._generated_context_carrier_remainder(m) or normalize_content_value(m["content"]) for m in result]
        assert [by_id.get(ids.get(id(m))) for m in result] == own
        stored = len(_rows(engine))
        engine.ingest(result)
        assert len(_rows(engine)) == stored
        new = _turn("NEW", 9000.0)
        engine.ingest([*result, *new])
        engine.on_session_end("S", [*result, *new])
        assert len(_rows(engine)) == stored + 2
    finally:
        engine.shutdown()
    cold = _hermes_engine(tmp_path)
    try:
        before = len(_rows(cold))
        cold.ingest([*result, *new])
        assert len(_rows(cold)) == before
    finally:
        cold.shutdown()


def test_4_two_fits_in_a_row_leave_one_block(tmp_path, host):  # noqa: F811 (the imported pytest fixture)
    """Cell 4: a second fit on the first's list plus new turns replaces the block: one block, the newest notice."""
    engine = _hermes_engine(tmp_path)
    try:
        view = _hidden_backlog(engine)
        first = engine.compress(view, current_tokens=host(view))
        grown = [*first, *[r for i in range(10) for r in _turn(f"G{i:02d}", 20_000.0 + 10 * i)]]
        engine.ingest(grown)
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(grown), False
        second = engine._survival_fit(grown, list(grown), host(grown), "test", request_cap=host(first))
        assert second is not grown and second[0]["content"].count(OPEN) == 1
        assert _notice(second[0]["content"]) == engine._last_survival_fit["notice"] != _notice(first[0]["content"])
    finally:
        engine.shutdown()


def test_5_the_cap_covers_the_wrapped_block(engine):
    """Cell 5: on the block path the cap counts the wrapper; a notice whose block passes it is the counted one."""
    parts = _part(engine)
    nodes = engine._verified_lcm_summary_prefix(parts)[1]
    order = [10 ** 12 - 2 * 60 + i for i in range(120)]
    ranges = [("a1" * 64, x, x) for x in order[::2]]
    worst = survival_fit.survival_notice_text(60, ranges[-8:], (52, order[0], order[-17]))
    block = survival_fit._survival_block(parts, nodes, worst)
    assert survival_fit.count_tokens(block) <= survival_fit._NOTICE_MAX_TOKENS
    over = survival_fit.survival_notice_text(3, [("x" * 1500, 1, 3)])
    block = survival_fit._survival_block(parts, nodes, over)
    assert survival_fit.count_tokens(block) <= survival_fit._NOTICE_MAX_TOKENS
    assert block.split("\n")[3] == survival_fit._NOTICE.format(n=3, first=1, last=3)
    assert _read(engine, parts + block)["scaffold"]  # the counted notice is a block too


def test_7_no_system_row_and_no_summary_head_places_nothing_and_warns_once(tmp_path, host, caplog):  # noqa: F811 (the imported pytest fixture)
    """Cell 7: with no system row and no glued carrier the notice is not placed; one warning."""
    engine = _hermes_engine(tmp_path, window=24_000)
    try:
        turns = [r for i in range(60) for r in _turn(f"T{i:03d}", 10.0 * i)]
        engine.ingest(turns)
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(turns), False
        with caplog.at_level("WARNING", logger="hermes_lcm"):
            result = engine._survival_fit(turns, list(turns), host(turns), "test")
        assert result is not turns and len(result) < len(turns)
        assert not any(OPEN in str(m.get("content")) for m in result)
        assert caplog.text.count("no system row and no glued carrier for the notice") == 1
    finally:
        engine.shutdown()


def test_8_head_bytes_are_stable_while_nothing_changes(tmp_path, host):  # noqa: F811 (the imported pytest fixture)
    """Cell 8: two engines from the same stored state return byte-identical heads."""
    heads = []
    for name in ("one", "two"):
        engine = _hermes_engine(tmp_path / name)
        try:
            view = _hidden_backlog(engine)
            heads.append(engine.compress(view, current_tokens=host(view))[0]["content"])
        finally:
            engine.shutdown()
    assert OPEN in heads[0] and heads[0] == heads[1]


def test_ignore_patterns_never_drop_a_host_row_glued_behind_the_block(tmp_path, monkeypatch):
    """An ignore pattern on the notice's words never matches a carrier through its block: the host's row behind it is
    judged on its own text."""
    class Pattern:  # the optional regex package's search(text, timeout=...), over stdlib re
        def __init__(self, text):
            self.pattern = __import__("re").compile(text)

        def search(self, text, timeout=None):
            return self.pattern.search(text)

    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "ig.db"), survival_reserve=0.0))
    engine.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
    engine._compiled_ignore_message_patterns = [Pattern("LCM survival fit")]
    try:
        parts = _part(engine)
        carrier = {"role": "user", "content": f"{parts}{_block(engine, parts)}\n\na fresh user turn"}
        assert not engine._matches_ignore_message_patterns(carrier)
        assert engine._matches_ignore_message_patterns({"role": "user", "content": "about the LCM survival fit"})
        assert engine._matches_ignore_message_patterns(
            {**carrier, "content": carrier["content"] + " about the LCM survival fit"})
    finally:
        engine.shutdown()


@pytest.mark.parametrize("separator", ["\n\n", "\n\n---\n\n"], ids=["blank-line", "rule"])
def test_the_carrier_key_leaves_the_block_out(engine, separator):
    """The carry-forward and retirement key of a carrier is the same with and without its block, for either carrier
    separator (a head carrier glued with a rule keeps it, A4)."""
    parts = _part(engine)
    rows = [{"role": "user", "content": f"{parts}{_block(engine, parts)}{separator}the user's row"},
            {"role": "user", "content": f"{parts}{separator}the user's row"}]
    keys = []
    for row, span in zip(rows, (f"{parts}{_block(engine, parts)}{separator}", f"{parts}{separator}")):
        (descriptor,) = _finalize_emission_descriptors([row], [{
            "kind": "carrier", "span": span, "row": row, "retained_source": {"store_id": 7},
            "full_identity": _emission_identity(row)}], {"s": 1})
        keys.append(_carrier_emission_key(descriptor))
    assert keys[0] == keys[1]


# -- the grid's no-system-row twin, walked as the grid walks it ----------------------------------------------------

import json as _json  # noqa: E402
import shutil as _shutil  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402

from tests.test_issue_2877_cumulative_notice import CELL, RUNNER, TREE, _syspaths  # noqa: E402

TWIN = {**CELL, "group": "N", "id": "s-600k-l40-prefix-nosys", "no_system_row": True, "twin_of": "s-600k-l40-prefix"}


def _walk_twin(tmp_path) -> list[dict]:
    cells, out, home = tmp_path / "cells.json", tmp_path / "walk.json", tmp_path / "home"
    cells.write_text(_json.dumps([TWIN]))
    home.mkdir()
    command = [_sys.executable, "-I", "-B", str(RUNNER), "--tree", str(TREE), "--cells", str(cells), "--out", str(out),
               "--workdir", str(tmp_path / "work")]
    for entry in _syspaths():
        command += ["--syspath", entry]
    if (TREE / "agent" / "context_engine.py").is_file():  # CI's host stub, as the 2877 cell takes it
        _shutil.copytree(TREE / "agent", tmp_path / "stub" / "agent")
        command += ["--syspath", str(tmp_path / "stub")]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0"}
    done = _subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return _json.loads(out.read_text())["records"]


def test_1_the_no_system_twin_leaves_no_cut_row_without_a_pointer(tmp_path, record_property):
    """Cells 1, 2 and 9 on the grid's s-600k-l40-prefix-nosys twin (its runner and content-aware A0 oracle, in its
    own process): the fits' rows are behind the notice (A0b), never without a pointer (A0), on every pass. Cell 9:
    A0 stays 0 on the passes after a fit too; those that return no notice are counted (recorded: no assembly caller
    is added)."""
    records = _walk_twin(tmp_path)
    fits = [r["pass"] for r in records if r["loss"] and r["loss"]["survival_fit_notice"]]
    lost = {r["pass"]: r["loss"]["missing_count"] for r in records if r["loss"] and r["loss"]["missing_count"]}
    assert not lost and not any(r["loss"]["missing_count_by_id"] for r in records if r["loss"]), (fits, lost)
    assert fits, "the premise: the twin reaches the survival fit, and its notice is in the list"
    assert sum(r["loss"]["behind_notice_count"] for r in records if r["loss"]) > 0  # the cut rows are behind it
    later = [r["pass"] for r in records if r["loss"] and r["pass"] > fits[0] and not r["loss"]["survival_fit_notice"]]
    record_property("passes_after_a_fit_without_a_notice", len(later))
    print(f"\n#2896 twin: fits at passes {fits}; passes after the first fit with no notice: {later}")


# -- review round 1 (Astra): each cell red on 9e69e970 ------------------------------------------------------------

from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX  # noqa: E402
from tests.test_issue_650_survival_fit_keeps_summary import _BOUNDARY  # noqa: E402


class _Pattern:  # the optional regex package's search(text, timeout=...), over stdlib re
    def __init__(self, text):
        self.pattern = __import__("re").compile(text)

    def search(self, text, timeout=None):
        return self.pattern.search(text)


def test_a1_a_header_shaped_user_turn_behind_the_block_keeps_its_own_text(engine):
    """A1: the host row glued behind the block starts with a summary-shaped header that verifies as no part (a quoted
    header): the block is still the summary's, the remainder is the host row, and an ignore pattern on the notice's
    words does not match the carrier. Main (no block): the remainder is the same host row."""
    parts = _part(engine)
    turn = "[Recent Summary (d0, node 999)]\nmy actual question"
    carrier = {"role": "user", "content": f"{parts}{_block(engine, parts)}\n\n{turn}"}
    assert engine._generated_context_carrier_remainder(carrier) == turn
    engine._compiled_ignore_message_patterns = [_Pattern("LCM survival fit")]
    assert not engine._matches_ignore_message_patterns(carrier)


def test_a2_a_late_callback_for_another_binding_never_matches_the_notice(engine):
    """A2: a session-end suffix of session S's list arrives while the engine is bound to B: the carrier's block is the
    fit's, so an ignore pattern on the notice's words never drops the host row behind it. Main: no block, no match."""
    parts = _part(engine)
    carrier = {"role": "user", "content": f"{parts}{_block(engine, parts)}\n\na fresh user turn"}
    engine._compiled_ignore_message_patterns = [_Pattern("LCM survival fit")]
    engine._session_id = "B"
    assert not engine._matches_ignore_message_patterns(carrier)


def test_a3_a_standalone_summary_head_takes_no_block(tmp_path, host, monkeypatch):  # noqa: F811 (the imported pytest fixture)
    """A3 (T1-344): a fit whose head is a standalone summary row (list user rows: no carrier), here on the exception
    path, places no block and registers no emission for it: main's shape. The host glues the summary into its next user
    row, and a later fit reaches those rows through that carrier."""
    engine = _hermes_engine(tmp_path)
    try:
        view = _hidden_backlog(engine, list_users=True, stamped_replies=True)
        first = engine.compress(view, current_tokens=host(view))
        engine.on_session_start("S", **_BOUNDARY)
        grown = [*first, *[r for i in range(12) for r in _turn(f"G{i:02d}", 20_000.0 + 10 * i, True, True)]]
        engine.ingest(grown)

        def fail(*args, **kwargs):
            raise RuntimeError("summariser down")

        monkeypatch.setattr(engine, "_compress_impl", fail)
        fitted = engine.compress(grown, current_tokens=host(grown))
        assert engine._last_survival_fit is not None and fitted[0]["role"] == "user"
        assert engine._verified_lcm_summary_prefix_end(fitted[0]["content"]) == len(fitted[0]["content"])
        assert not any(OPEN in str(m.get("content")) for m in fitted)
        assert getattr(engine, "_survival_fit_emission", None) is None
    finally:
        engine.shutdown()


def test_a4_a_carrier_ahead_of_a_preserved_objective_takes_the_block(engine, monkeypatch, caplog):
    """A4: the kept prefix is [verified carrier, preserved objective]; the fit cuts old stored turns. On each cut the
    carrier takes the notice, which names the cut rows, when it fits beside the kept turns; otherwise the warning
    says so. Main: no system row, no notice (the rows have no pointer)."""
    measure = _rough(monkeypatch)
    pad = " older words" * 100
    rows = _store_turns(engine, [f"[U{i}]{pad}" for i in range(6)])
    parts = _part(engine, "the first turns")
    carrier = {"role": "user", "content": f"{parts}\n\n{rows[0][1]['content']}"}
    assert engine._generated_context_carrier_remainder(carrier) == rows[0][1]["content"]
    objective = {"role": "user", "content": f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nfinish the report"}
    active = [carrier, objective, *(row for _, row in rows[1:])]
    placed = 0
    for cap in range(measure(active) - 50, measure(active) - 900, -37):
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
        caplog.clear()
        with caplog.at_level("WARNING", logger="hermes_lcm"):
            result = engine._survival_fit(active, list(active), 0, "test", request_cap=cap)
        assert result is not active and len(result) < len(active)
        assert "no system row and no glued carrier for the notice" not in caplog.text
        if OPEN in result[0]["content"]:
            placed += 1
            assert engine._generated_context_carrier_remainder(result[0]) == rows[0][1]["content"]
            assert engine._survival_fit_emission["row"] is result[0]
        else:
            assert "the notice does not fit beside the kept turns" in caplog.text, cap
    assert placed


def _rough(monkeypatch):
    def count(message):
        return len(str(message.get("content") or "")) // 4

    def measure(messages):
        return sum(count(message) for message in messages)

    monkeypatch.setattr(survival_fit, "count_message_tokens", count)
    monkeypatch.setattr(survival_fit, "count_messages_tokens", measure)
    monkeypatch.setattr(survival_fit, "_host_estimate", measure)
    return measure


def _store_turns(engine, users):
    out = []
    for i, text in enumerate(users):
        for row in ({"role": "user", "content": text}, {"role": "assistant", "content": f"reply {i}" + " v" * 200}):
            out.append((engine._store.append("S", dict(row), conversation_id="conv"), row))
    return out
