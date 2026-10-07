"""#581 store-complete leaves and #582 survival fit (v0.24.5 SPEC tests (a)-(h)).

Engine-level: a host list in; stored rows, summary coverage, the lifecycle frontier and the returned
list out. Summaries are stubbed; they record every summarizer input."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import types
from collections import Counter

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
import hermes_lcm.store_complete as lcm_store_complete
import hermes_lcm.survival_fit as lcm_survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError

PAD = " alpha beta gamma delta" * 30


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def _engine(tmp_path, session="S", context_length=200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start(session, platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, *, tool: bool = False, stamp_user: bool = True) -> list[dict]:
    """A user row (host-stamped unless ``stamp_user`` is False); unstamped assistant/tool rows as eva's view."""
    user = {"role": "user", "content": f"[{tag}] user turn{PAD}"}
    if stamp_user:
        user["timestamp"] = ts
    rows = [user]
    if tool:
        call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        rows += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag}{PAD}"}]
    return rows + [{"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages(engine._session_id, limit=100_000)


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id or 0)


def _covered(engine) -> list[int]:
    return [store_id for node in engine._dag.get_session_nodes(engine._session_id)
            if node.source_type == "messages" for store_id in node.source_ids]


def _assert_contiguous(engine) -> None:
    """Every owned row at or below the frontier is covered by a summary exactly once, or excluded by an
    existing rule (none are here): the publication proof, re-checked from the outside."""
    covered = Counter(_covered(engine))
    frontier = _frontier(engine)
    below = [int(row["store_id"]) for row in _rows(engine) if int(row["store_id"]) <= frontier]
    assert below and all(covered[store_id] == 1 for store_id in below), (frontier, below, covered)


def _eva_store(tmp_path):
    """The eva shape: owned rows the host view does not show, below its first mapped row.
    - natively compacted history: stamped users, unstamped assistant and tool rows, one unstamped user;
    - older duplicate copies (from a past ingest bug) of assistant/tool rows the view shows.
    The view is the host's native summary followed by the live rows."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0, tool=True), *_turn("H2", 110.0), *_turn("H3", 120.0, tool=True),
           *_turn("H4", 0.0, stamp_user=False)]
    live = [row for i in range(10, 16) for row in _turn(f"T{i}", 100.0 * i, tool=i % 2 == 0)]
    native = {"role": "user", "content": "[CONTEXT COMPACTION] earlier turns were summarized by the host" + PAD,
              "timestamp": 999.0}
    try:
        engine.ingest(old)
        for message in live[1:4]:  # older copies of rows the view shows (unstamped assistant/tool rows)
            engine._store.append("S", dict(message), conversation_id="conv")
        engine.ingest([*old, native, *live])
    finally:
        engine.shutdown()
    return [native, *live]


# -- (a) the eva shape --------------------------------------------------------------------------------

def test_a_eva_shape_publishes_and_the_frontier_advances_over_passes(tmp_path, summaries):
    view = _eva_store(tmp_path)
    # restart: a fresh bind on the stored session, as the gateway does; bounded leaves of ~3 rows
    engine = _engine(tmp_path, dynamic_leaf_chunk_enabled=True, dynamic_leaf_chunk_max=400)
    try:
        before = [(int(r["store_id"]), r["role"], r["content"]) for r in _rows(engine)]
        frontiers, statuses, live = [_frontier(engine)], [], view
        for _ in range(4):
            live = engine.compress(live)
            statuses.append(engine._last_compression_status)
            frontiers.append(_frontier(engine))
            engine.note_turn_complete()  # #904: each hidden drain pass belongs to a new foreground turn
            engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                    platform="telegram", conversation_id="conv")
        after = [(int(r["store_id"]), r["role"], r["content"]) for r in _rows(engine)]
        assert statuses[:3] == ["compacted"] * 3, (statuses, engine._last_compression_noop_reason)
        assert frontiers[1] > frontiers[0] and frontiers[2] > frontiers[1] and frontiers[3] > frontiers[2], frontiers
        assert after == before  # 0 row changes: nothing lost, nothing re-stored
        _assert_contiguous(engine)
        hidden_tool = next(r for r in _rows(engine) if r["role"] == "tool" and "result of H1" in r["content"])
        assert int(hidden_tool["store_id"]) in _covered(engine)
        assert any("result of H1" in text for text in summaries)  # read from the store into the leaf
    finally:
        engine.shutdown()


# -- (b) a hidden-only leaf ---------------------------------------------------------------------------

def test_b_hidden_only_leaf_consumes_no_host_row(tmp_path, summaries):
    """No host raw chunk (the view is all fresh tail), owned hidden backlog below it: scheduled, not a no-op."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0, tool=True), *_turn("H2", 0.0, stamp_user=False)]
    tail = _turn("T9", 900.0)
    try:
        engine.ingest([*old, *tail])
        hidden = [int(r["store_id"]) for r in _rows(engine)][: len(old)]
        result = engine.compress(list(tail))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert [m for m in result if m.get("content") in {t["content"] for t in tail}] == tail  # 0 host rows consumed
        assert set(hidden) & set(_covered(engine)) and _frontier(engine) < min(
            int(r["store_id"]) for r in _rows(engine) if r["content"] == tail[0]["content"])
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


# -- (c) the endpoint stops before an unresolved retained occurrence -------------------------------------

def test_c_leaf_ends_before_a_retained_occurrence(tmp_path, summaries):
    """R (an unstamped user row) is stored early; the host re-orders it into the fresh tail. The leaf
    never covers R while the view retains it, and publication never conflicts."""
    engine = _engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    r = {"role": "user", "content": "[R] retained occurrence" + PAD}
    stored = [*_turn("T1", 100.0), r, *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    try:
        engine.ingest(stored)
        r_id = next(int(row["store_id"]) for row in _rows(engine) if row["content"] == r["content"])
        view = [m for m in stored if m is not r] + [dict(r)]
        engine._ingest_cursor = len(view)  # the host holds the same rows, R moved to the tail
        statuses = []
        for _ in range(3):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses, engine._last_compression_noop_reason
        assert r_id not in _covered(engine) and _frontier(engine) < r_id
        assert _frontier(engine) > 0
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


def test_b_native_on_off_host_summary_chunk_leaves_cover_the_stored_rows(tmp_path, summaries):
    """The native-on-off shape: a native-ON ref left a host summary (a row the store does not map) as the
    only raw chunk row, with the rows it summarised stored above frontier 0. The leaf covers stored rows
    (the host summary never takes the whole budget) instead of publishing no coverage."""
    engine = _engine(tmp_path, fresh_tail_count=6, leaf_chunk_tokens=300)
    old = [row for i in range(1, 9) for row in _turn(f"T{i}", 10.0 * i)]
    host_summary = {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
                                               "into the summary below." + PAD * 3, "timestamp": 95.0}
    tail = [row for i in range(9, 12) for row in _turn(f"T{i}", 10.0 * i)]
    try:
        engine.ingest([*old, *tail])
        view = [host_summary, *tail]
        engine._ingest_cursor = len(view)
        statuses = []
        for _ in range(3):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert statuses[0] == "compacted" and "error" not in statuses, (statuses, engine._last_compression_noop_reason)
        assert _frontier(engine) > 0
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


def test_c_bound_leaf_passes_rows_of_another_conversation_under_the_session(tmp_path, summaries):
    """Eva thread 1: the bound session also holds rows of ANOTHER conversation, interleaved in store
    order. The bound conversation's obligation is its own rows (and blank ones) only: its leaves pass
    the other conversation's rows without covering them, and publication never conflicts."""
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=300)
    first = [row for i in range(1, 4) for row in _turn(f"T{i}", 10.0 * i)]
    second = [row for i in range(4, 7) for row in _turn(f"T{i}", 10.0 * i)]
    tail = [row for i in range(8, 10) for row in _turn(f"T{i}", 10.0 * i)]
    try:
        engine.ingest(first)
        foreign = [engine._store.append("S", {"role": "assistant", "content": f"[X{i}] other conversation" + PAD},
                                        conversation_id="other") for i in range(3)]
        view = [*first, *second, *tail]
        engine.ingest(view)
        statuses = []
        for _ in range(6):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses, (statuses, engine._last_compression_noop_reason)
        assert _frontier(engine) > max(foreign), (_frontier(engine), foreign, statuses)
        assert set(foreign).isdisjoint(_covered(engine))
        assert all("other conversation" not in text for text in summaries)
        _assert_contiguous_for_conversation(engine)
    finally:
        engine.shutdown()


def _assert_contiguous_for_conversation(engine) -> None:
    """#5 per conversation: every row of the bound (or blank) conversation at or below the frontier
    is covered exactly once; rows of other conversations are neither required nor covered."""
    covered = Counter(_covered(engine))
    frontier = _frontier(engine)
    own = [int(r["store_id"]) for r in _rows(engine)
           if int(r["store_id"]) <= frontier and str(r.get("conversation_id") or "").strip() in ("", "conv")]
    assert own and all(covered[store_id] == 1 for store_id in own), (frontier, own, covered)


def _stage(engine, expected, covered, node_source=None):
    node = SummaryNode(session_id="S", summary="s", token_count=1, source_token_count=1,
                       source_ids=list(node_source or covered))

    def stage(conn, node_id) -> None:
        engine._lifecycle.stage_compaction_publication(conn, "conv", "S", node_id, expected, list(covered))

    engine._dag.add_node(node, before_commit=stage)


def _mixed_rows(engine):
    """Own A, other X, own B, other Y, own C: two conversations interleaved under session S."""
    store, ids = engine._store, {}
    for tag, conversation in (("A", "conv"), ("X", "other"), ("B", "conv"), ("Y", "other"), ("C", "conv")):
        ids[tag] = store.append("S", {"role": "assistant", "content": f"[{tag}] row" + PAD}, conversation_id=conversation)
    return ids


def test_5_bound_frontier_passes_other_conversation_rows_but_not_an_unproven_bound_row(tmp_path):
    """#5 per conversation: A..C (own rows A, B, C) publishes past the other conversation's X and Y;
    skipping the unproven own row B, which sits between X and Y, is refused."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        with pytest.raises(LifecyclePublicationConflictError, match="not contiguous"):
            _stage(engine, 0, [ids["A"], ids["C"]])
        assert _frontier(engine) == 0
        _stage(engine, 0, [ids["A"], ids["B"], ids["C"]])
        assert _frontier(engine) == ids["C"]
    finally:
        engine.shutdown()


def _plans_of(conn, run) -> list[str]:
    """EXPLAIN QUERY PLAN of every SELECT on messages that ``run()`` executes on ``conn``."""
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        run()
    finally:
        conn.set_trace_callback(None)
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT") and " messages" in sql]
    return [" ".join(str(row[-1]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql)) for sql in selects]


def test_r3_owned_reads_seek_the_conversation_index(tmp_path):
    """R3 F3: the store-complete range read and the proof's obligation read are index seeks on
    (conversation_id, session_id, store_id), never a walk over another conversation's rows."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        plans = _plans_of(engine._store.connection, lambda: engine._store_complete_owned_rows(0, None, []))
        assert plans and all("idx_msg_conversation_session" in plan for plan in plans), plans
        plans = _plans_of(engine._dag.connection, lambda: _stage(engine, 0, [ids["A"], ids["B"], ids["C"]]))
        owned = [plan for plan in plans if "store_id" in plan and "json_each" not in plan and "summary" not in plan]
        assert owned and all("idx_msg_conversation_session" in plan for plan in owned), plans
        assert _frontier(engine) == ids["C"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("legacy", ["   ", "\t", None, " conv "], ids=["spaces", "tab", "null", "padded"])
def test_r4_a_legacy_unnormalized_conversation_row_stays_owned(tmp_path, legacy):
    """R4-1: a legacy row whose conversation id is whitespace-only, NULL or padded (writes strip; older
    stores may not) is in the leaf's owned read and in the proof's obligation: the frontier cannot pass it."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        conn = engine._store.connection
        try:
            conn.execute("UPDATE messages SET conversation_id = ? WHERE store_id = ?", (legacy, ids["B"]))
            conn.commit()
        except sqlite3.IntegrityError:
            pytest.skip("this schema refuses a NULL conversation id")
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)  # a fresh bind probes the store once
    try:
        rows, _truncated = engine._store_complete_owned_rows(0, None, [])
        assert [int(r["store_id"]) for r in rows] == [ids["A"], ids["B"], ids["C"]]
        with pytest.raises(LifecyclePublicationConflictError, match="not contiguous"):
            _stage(engine, 0, [ids["A"], ids["C"]])
        assert _frontier(engine) == 0
        _stage(engine, 0, [ids["A"], ids["B"], ids["C"]])
        assert _frontier(engine) == ids["C"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("indexed", [True, False])
def test_r4_a_the_bind_probe_finds_legacy_ids_with_or_without_the_index(tmp_path, indexed):
    """The bind probe (index seeks per distinct id, or one scan without the index) finds every legacy
    value and maps it to the normalized id it stands for."""
    from hermes_lcm.db_bootstrap import owned_conversation_values, refresh_legacy_conversation_ids

    conn = sqlite3.connect(str(tmp_path / "probe.db"))
    conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, conversation_id TEXT)")
    if indexed:
        conn.execute("CREATE INDEX idx_msg_conversation_session ON messages (conversation_id, session_id, store_id)")
    conn.executemany("INSERT INTO messages (session_id, conversation_id) VALUES ('S', ?)",
                     [("conv",), ("",), ("  ",), (None,), (" conv",), ("other",), ("other ",)])
    legacy = refresh_legacy_conversation_ids(conn)
    assert {key: sorted(map(repr, values)) for key, values in legacy.items()} == {
        "": sorted(map(repr, [None, "  "])), "conv": ["' conv'"], "other": ["'other '"]}
    assert sorted(map(repr, owned_conversation_values(conn, "conv"))) == sorted(
        map(repr, ["conv", "", " conv", None, "  "]))
    conn.close()


def test_r4i_a_stored_row_outside_the_view_never_releases_a_view_reservation(tmp_path):
    """Integration guard (#590 form i x #581): a hidden stored composite row in the summarizer input whose
    (stamp, form) equals an unmapped view occurrence is not that occurrence, so it never subtracts the
    occurrence's key from the view's reservations: the stored row the view occurrence holds stays reserved."""
    u_text, r_text = "[U] please go on" + PAD, "[R] failed turn" + PAD
    c_text = r_text + "\n\n" + u_text
    engine = _engine(tmp_path)
    try:
        engine.ingest([{"role": "user", "content": u_text, "timestamp": 10.0}])
        engine.ingest([{"role": "user", "content": u_text, "timestamp": 10.0},
                       {"role": "user", "content": c_text, "timestamp": 30.0}])
        engine.ingest([{"role": "user", "content": u_text, "timestamp": 10.0},
                       {"role": "user", "content": c_text, "timestamp": 30.0},
                       {"role": "user", "content": r_text, "timestamp": 30.0}])
        stored = {row["content"]: int(row["store_id"]) for row in _rows(engine)}
        composite_row = stored[c_text]
        shown_u = {"role": "user", "content": u_text, "timestamp": 10.0}
        shown_c = {"role": "user", "content": c_text, "timestamp": 30.0}
        hidden = {"role": "user", "content": c_text, "timestamp": 30.0}  # the stored composite, not a view row
        seen = []
        reserve = engine._identity_anchor_reserved
        engine._identity_anchor_reserved = lambda pool, shown: seen.append(reserve(pool, shown)) or seen[-1]
        engine._identity_anchor_summary_input([shown_c, hidden], {}, view=[shown_u, shown_c])
        assert seen and all(composite_row in reserved for reserved in seen), (composite_row, seen)
    finally:
        engine.shutdown()


def test_proof_rejects_covering_another_conversations_row(tmp_path):
    """A leaf that claims another conversation's row as coverage is still refused."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        with pytest.raises(LifecyclePublicationConflictError, match="ownership"):
            _stage(engine, 0, [ids["A"], ids["X"], ids["B"]])
        assert _frontier(engine) == 0 and _covered(engine) == []
    finally:
        engine.shutdown()


# -- (d) the carry set ----------------------------------------------------------------------------------

def _state_db(tmp_path, sessions):
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", sessions)
    conn.commit()
    conn.close()


def test_d_owned_rows_are_the_session_plus_the_publication_carry_set(tmp_path, summaries):
    """A compression child C carries parent P's rows (P, 1, 4]; P's hidden carried row 3 is read into the
    leaf. P's row 6, outside the carry, and a sibling session's rows are never read or covered."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None), ("X", None, None)])
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path))
    try:
        engine.on_session_start("C", platform="telegram", context_length=200_000, conversation_id="conv")
        store = engine._store
        parent = [*_turn("P1", 10.0), *_turn("P2", 20.0)]
        ids = [store.append("P", dict(m), conversation_id="conv") for m in parent]  # 1..4
        sibling = store.append("X", {"role": "user", "content": "[X] sibling row" + PAD}, conversation_id="conv")
        outside = store.append("P", {"role": "assistant", "content": "[P] outside the carry" + PAD}, conversation_id="conv")
        store.write_metadata_json([engine._identity_anchor_carry_key()], json.dumps([["P", 0, ids[-1]]]))
        live = [row for i in range(3, 7) for row in _turn(f"C{i}", 100.0 * i)]
        view = [parent[0], parent[1], parent[3], *live]  # P's row 3 is hidden (the host dropped it)
        engine.ingest(view)
        engine.compress(view)
        covered = set(_covered(engine))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert ids[2] in covered and {sibling, outside}.isdisjoint(covered)
        assert all("sibling row" not in text and "outside the carry" not in text for text in summaries)
    finally:
        engine.shutdown()


# -- (e)-(g) the survival fit ------------------------------------------------------------------------------

WINDOW = 6000
TARGET = int(WINDOW * 0.85)
NOTICE = "[LCM survival fit:"


def _host_rough(messages) -> int:
    """A stand-in for the host's rough request estimator (chars / 4 over content and tool calls)."""
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture
def host_estimator(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _host_rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    return _host_rough


def _long_view(turns=24) -> list[dict]:
    """A system prompt and ``turns`` stored turns (every third with a tool call): ~2x the window."""
    rows = [row for i in range(turns) for row in _turn(f"L{i}", 10.0 * i, tool=i % 3 == 0)]
    return [{"role": "system", "content": "system prompt"}, *rows]


def _assert_fitted(engine, view, result, host) -> None:
    """(e): the fitted list is under target by the host estimator, starts at a user turn after the
    system slot, carries the notice there (never as a row) and raised exactly one warning."""
    assert host(view) > TARGET and host(result) <= TARGET, (host(view), host(result))
    assert result[0]["role"] == "system" and NOTICE in result[0]["content"]
    assert result[1]["role"] == "user" and not any(NOTICE in str(m.get("content")) for m in result[1:])
    assert [m for m in result[1:]] == view[-len(result) + 1:]  # the newest whole turns, unchanged
    first = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
    second = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
    assert first and "LCM" in first and second is None and engine.emit_automatic_compaction_status is False


def test_e_survival_fit_after_an_injected_publication_conflict(tmp_path, summaries, host_estimator, caplog):
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)

        def conflict(*args, **kwargs):
            raise LifecyclePublicationConflictError("injected conflict (ids only)")

        engine._lifecycle.stage_compaction_publication = conflict
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=host_estimator(view))
        assert engine._last_compression_status == "error"
        _assert_fitted(engine, view, result, host_estimator)
        assert sum("LCM survival fit applied" in r.getMessage() for r in caplog.records) == 1
        assert "publication_invariant_conflict" in caplog.text
        counter = engine._store.read_metadata_json("survival_fit:counter")
        assert counter["count"] == 1 and counter["last_reason"] == "publication_invariant_conflict"
    finally:
        engine.shutdown()


def test_e_survival_fit_after_a_sweep_deadline(tmp_path, summaries, host_estimator, monkeypatch):
    monkeypatch.setattr(lcm_compaction, "_THRESHOLD_FULL_SWEEP_MAX_SECONDS", 0.0)
    engine = _engine(tmp_path, context_length=WINDOW, context_threshold=0.5, threshold_full_sweep_enabled=True)
    view = _long_view()
    try:
        engine.ingest(view)
        result = engine.compress(view, current_tokens=host_estimator(view))
        _assert_fitted(engine, view, result, host_estimator)
    finally:
        engine.shutdown()


def test_e_survival_fit_after_a_lock_after_commit(tmp_path, summaries, host_estimator, monkeypatch):
    """One leaf publishes, condensation then hits a SQLite lock: the committed list is fitted."""
    engine = _engine(tmp_path, context_length=WINDOW, fresh_tail_count=60)  # a long retained tail
    view = _long_view()
    try:
        engine.ingest(view)

        def locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, "_maybe_condense", locked)
        result = engine.compress(view, current_tokens=100)  # below the list: one bounded leaf, no overflow
        assert _frontier(engine) > 0 and engine._last_compression_noop_reason == \
            "summary publication blocked by SQLite lock"
        assert host_estimator(result) <= TARGET and NOTICE in result[0]["content"]
        assert result[1]["role"] == "user" and not any(NOTICE in str(m.get("content")) for m in result[1:])
        assert engine.get_automatic_compaction_status_message(phase="compress", default_message="x")
    finally:
        engine.shutdown()


def test_f_newest_turn_over_budget_is_projected(tmp_path, summaries, host_estimator):
    """The newest user turn alone is over the window: a bounded projection, never empty, raw rows intact."""
    engine = _engine(tmp_path, context_length=WINDOW)
    call = {"id": "call_big", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    big = {"role": "tool", "tool_call_id": "call_big", "content": "row " * 12_000}
    view = [*_long_view(4), {"role": "user", "content": "[N] newest", "timestamp": 99.0},
            {"role": "assistant", "content": "", "tool_calls": [call]}, big]
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        result = engine.compress(view, current_tokens=host_estimator(view))
        assert result and host_estimator(result) <= TARGET, host_estimator(result)
        assert result[1]["content"] == "[N] newest" and result[-1]["role"] == "tool"
        assert result[-1]["content"] != big["content"] and result[-1]["tool_call_id"] == "call_big"
        stored = [r for r in _rows(engine) if r["role"] == "tool" and r["content"] == big["content"]]
        assert len(stored) == 1  # the raw row stays stored verbatim
    finally:
        engine.shutdown()


def test_f_projection_bounds_huge_tool_call_arguments_in_the_view(tmp_path, summaries, host_estimator):
    """R3 F1: the newest turn is over budget because of an assistant row's tool-call ARGUMENTS. The
    projection bounds them in the view copy (valid JSON with the provenance notice); ids and the
    call/result pairing stay intact; the stored row keeps its arguments verbatim."""
    engine = _engine(tmp_path, context_length=1000)  # budget 850
    args = json.dumps({"path": "notes.txt", "text": "arg " * 10_000})
    call = {"id": "call_huge", "type": "function", "function": {"name": "write_file", "arguments": args}}
    view = [{"role": "user", "content": "[N] newest", "timestamp": 99.0},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_huge", "content": "ok"}]
    try:
        engine.ingest(view)
        assert host_estimator(view) > 10_000
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        result = engine.compress(view, current_tokens=host_estimator(view))
        assert host_estimator(result) <= 850, host_estimator(result)
        assert [m["role"] for m in result] == ["user", "assistant", "tool"]
        (projected,) = result[1]["tool_calls"]
        assert projected["id"] == "call_huge" and projected["function"]["name"] == "write_file"
        assert "LCM survival fit" in json.loads(projected["function"]["arguments"])["lcm_survival_fit"]
        assert result[2]["tool_call_id"] == "call_huge"
        stored = [r for r in _rows(engine) if r["role"] == "assistant" and args in str(r.get("tool_calls"))]
        assert len(stored) == 1  # the raw row keeps its arguments
    finally:
        engine.shutdown()


def test_g_fitted_list_re_ingests_without_new_rows(tmp_path, summaries, host_estimator):
    """The host adopts the fitted list, archives the session, and a cold process resumes it: only a
    genuinely new turn adds rows."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        assert len(fitted) < len(view)
        before = len(_rows(engine))
        engine.on_session_end("S", fitted)
        assert len(_rows(engine)) == before
    finally:
        engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        new = _turn("NEW", 500.0)
        cold.ingest([*fitted, *new])
        added = _rows(cold)[before:]
        assert len(added) == len(new), [(r["role"], r["store_id"]) for r in added]
    finally:
        cold.shutdown()


def _conflicted(engine) -> None:
    engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
        LifecyclePublicationConflictError("injected"))


def test_r3a_an_unstored_summary_looking_row_is_never_dropped(tmp_path, summaries, host_estimator, monkeypatch):
    """R3-A LOSSLESS: the store write for the newest turn fails (a lock); a reply that merely mentions
    "CONTEXT SUMMARY" is not durable, so the fit never cuts past it, and it is stored once the lock clears."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)
        reply = {"role": "assistant", "content": "Here is the CONTEXT SUMMARY you asked for:" + " word" * 6000}
        view2 = [*view, reply, {"role": "user", "content": "[N] newest question", "timestamp": 999.0}]
        real = engine._store._append_protected_batch

        def locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine._store, "_append_protected_batch", locked)
        engine.ingest(view2)  # the per-turn ingest swallows the failure
        try:
            kept = engine.compress(view2, current_tokens=host_estimator(view2))
        except sqlite3.OperationalError:
            kept = view2  # compress() re-raised: the host keeps its list
        assert any(m is reply for m in kept)
        monkeypatch.setattr(engine._store, "_append_protected_batch", real)  # the lock clears
        host = [*kept, {"role": "assistant", "content": "answer to newest"}]
        engine.ingest(host)
        engine.on_session_end("S", host)
        assert any(r["content"] == reply["content"] for r in _rows(engine)), "reply never reached the store"
    finally:
        engine.shutdown()


def _big_user_view() -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[r for i in range(4) for r in _turn(f"L{i}", 10.0 * i, tool=i % 3 == 0)],
            {"role": "user", "content": "[N] newest " + "word " * 12_000, "timestamp": 99.0},
            {"role": "assistant", "content": "ok reply"}]


def _big_tool_view() -> list[dict]:
    call = {"id": "call_big", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    return [{"role": "system", "content": "system prompt"},
            *[r for i in range(4) for r in _turn(f"L{i}", 10.0 * i, tool=i % 3 == 0)],
            {"role": "user", "content": "[N] newest", "timestamp": 99.0},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_big", "content": "row " * 12_000}]


def _list_system_view() -> list[dict]:
    view = _long_view()
    return [{"role": "system", "content": [{"type": "text", "text": "system prompt"}]}, *view[1:]]


@pytest.mark.parametrize("conflict", [True, False])
def test_r3b_a_projected_row_re_ingests_as_its_source_row(tmp_path, summaries, host_estimator, conflict):
    """R3-B (Q1): the newest user turn alone is over the window; its projected view copy keeps the host
    stamp, and the next ingest of the fitted list plus a new turn stores only the new turn."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _big_user_view()
    try:
        engine.ingest(view)
        if conflict:
            _conflicted(engine)
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        assert engine._last_survival_fit and any(NOTICE in str(m.get("content")) for m in fitted[1:])
        before = len(_rows(engine))
        new = _turn("NEW", 500.0)
        engine.ingest([*fitted, *new])
        added = _rows(engine)[before:]
        assert [r["content"] for r in added] == [m["content"] for m in new]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("view_of", [_big_user_view, _big_tool_view, _list_system_view],
                         ids=["projected-user", "projected-tool", "list-system"])
def test_r3b_a_fitted_list_resumes_cold_without_new_rows(tmp_path, summaries, host_estimator, view_of):
    """R3-B (Q3, Q2): the host adopts the fitted list (a projected newest user row and the reply after it;
    a projected tool row; a list-content system slot carrying the notice) and a cold process resumes it:
    only the new turn is stored, never the notice, never the earlier reply again."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = view_of()
    try:
        engine.ingest(view)
        _conflicted(engine)
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        assert engine._last_survival_fit
        before = len(_rows(engine))
        engine.on_session_end("S", fitted)
    finally:
        engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        new = _turn("NEW", 500.0)
        cold.ingest([*fitted, *new])
        added = _rows(cold)[before:]
        assert [r["content"] for r in added] == [m["content"] for m in new]
        assert not any(NOTICE in str(r["content"]) for r in _rows(cold))
    finally:
        cold.shutdown()


@pytest.mark.parametrize("case", ["dropped-only", "projected", "old-record"])
def test_rc2_survival_counter_records_projections(tmp_path, summaries, host_estimator, case):
    """The doctor counter counts fits that projected a row (a persisted projection limits rollback, #601);
    a record written before the key existed stays unknown (no key), never a false zero."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _big_user_view() if case == "projected" else _long_view()
    try:
        if case == "old-record":
            engine._store.write_metadata_json(["survival_fit:counter"], json.dumps({"count": 1, "last_reason": "x"}))
        engine.ingest(view)
        _conflicted(engine)
        engine.compress(view, current_tokens=host_estimator(view))
        record = engine._store.read_metadata_json("survival_fit:counter")
        expected = {"dropped-only": 0, "projected": 1, "old-record": None}[case]
        assert record["count"] == (2 if case == "old-record" else 1)
        assert record.get("projected_count") == expected and (expected is not None or "projected_count" not in record)
    finally:
        engine.shutdown()


def test_rc3_survival_counter_update_is_atomic_across_engines(tmp_path):
    """R3-2: two engines on one lcm.db. B reads the counter, A records a projecting fit, then B records a
    drop-only fit: B's stale read never overwrites A's projection (projected_count never decreases)."""
    import threading
    import time as _time

    a, b = _engine(tmp_path), _engine(tmp_path)
    fit_a = threading.Thread(target=a._survival_record, args=("a", 1, [1], 10, 5, 10, True, "n"))
    real = b._store.read_metadata_json

    def stale_read(key):
        value = real(key)
        if key == "survival_fit:counter" and not fit_a.is_alive() and fit_a.ident is None:
            fit_a.start()  # A records while B holds what it read
            _time.sleep(0.3)
        return value

    b._store.read_metadata_json = stale_read
    try:
        b._survival_record("b", 1, [2], 10, 5, 10, False, "n")
        fit_a.join(10)
        record = a._store.read_metadata_json("survival_fit:counter")
        assert record["count"] == 2 and record["projected_count"] >= 1, record
    finally:
        a.shutdown()
        b.shutdown()


@pytest.mark.parametrize("eligible", [False, True], ids=["nothing-eligible", "one-eligible-row"])
def test_rc4_projected_count_counts_only_fits_that_replaced_a_row(tmp_path, summaries, host_estimator, caplog,
                                                                  eligible):
    """R4-1: the newest turn alone is over budget and older turns are dropped. When no row of that turn can be
    projected (each under 256 tokens), the fit projected nothing: projected_count 0 and projected=False."""
    rows = [{"role": "assistant", "content": f"[P{i}] " + "alpha " * 150} for i in range(30)]  # ~230 tokens each
    if eligible:
        rows[0] = {"role": "assistant", "content": "[P0] " + "alpha " * 3000}
    view = [{"role": "system", "content": "system prompt"},
            *[r for i in range(3) for r in _turn(f"L{i}", 10.0 * i)],
            {"role": "user", "content": "[N] newest", "timestamp": 99.0}, *rows]
    engine = _engine(tmp_path, context_length=WINDOW)
    try:
        engine.ingest(view)
        _conflicted(engine)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            engine.compress(view, current_tokens=host_estimator(view))
        record = engine._store.read_metadata_json("survival_fit:counter")
        assert record["count"] == 1 and record["projected_count"] == int(eligible), record
        line = next(r.getMessage() for r in caplog.records if "LCM survival fit applied" in r.getMessage())
        assert f"projected={eligible}" in line, line
    finally:
        engine.shutdown()


def _projection_of(engine, store_id) -> dict:
    """The view copy the survival fit makes of stored row ``store_id`` (its host stamp kept)."""
    row = engine._store.get(store_id)
    fields = engine._survival_projected_fields(row, 3000, lcm_survival_fit._HEAD, lcm_survival_fit._TAIL)
    return {"role": row["role"], "timestamp": row["observed_at"], **{k: v for k, v in fields.items() if v is not None}}


@pytest.mark.parametrize("foreign", ["ok reply", "another conversation's text"], ids=["identical", "different"])
def test_rc2_projection_followers_read_only_the_sources_conversation(tmp_path, foreign):
    """A row of another conversation under the same session, stored between the projected source and its
    reply, is never taken (byte-identical) and never ends the walk early (different)."""
    engine = _engine(tmp_path)
    try:
        source = engine._store.append("S", {"role": "user", "content": "[N] newest " + "word " * 600,
                                            "timestamp": 99.0}, conversation_id="conv")
        engine._store.append("S", {"role": "assistant", "content": foreign}, conversation_id="other")
        reply = engine._store.append("S", {"role": "assistant", "content": "ok reply"}, conversation_id="conv")
        view = [_projection_of(engine, source), {"role": "assistant", "content": "ok reply"}]
        taken = engine._survival_projection_followers(view, 0, engine._store.get(source), {0: 99.0})
        assert [(k, int(row["store_id"])) for k, row in taken] == [(1, reply)]
    finally:
        engine.shutdown()


def test_rc3_projection_followers_of_a_blank_conversation_source_stay_in_the_active_conversation(tmp_path):
    """R3-1: a legacy source row with a blank conversation never widens the read to every conversation of
    the session: a byte-identical reply of another conversation is not taken, so the active reply is stored."""
    engine = _engine(tmp_path)
    try:
        source = engine._store.append("S", {"role": "user", "content": "[N] newest " + "word " * 600,
                                            "timestamp": 99.0}, conversation_id="")
        engine._store.append("S", {"role": "assistant", "content": "ok reply"}, conversation_id="foreign")
        view = [_projection_of(engine, source), {"role": "assistant", "content": "ok reply"}]
        assert engine._survival_projection_followers(view, 0, engine._store.get(source), {0: 99.0}) == []
    finally:
        engine.shutdown()


def test_rc2_projection_followers_bind_to_the_named_source(tmp_path):
    """Two identical user rows under one stamp; the projection names the later one. Matched to the earlier
    row, the walk would read the earlier reply as this one: it takes nothing unless the row is the source."""
    engine = _engine(tmp_path)
    try:
        user = {"role": "user", "content": "[N] newest " + "word " * 600, "timestamp": 99.0}
        earlier = engine._store.append("S", dict(user), conversation_id="conv")
        engine._store.append("S", {"role": "assistant", "content": "ok reply"}, conversation_id="conv")
        later = engine._store.append("S", dict(user), conversation_id="conv")
        view = [_projection_of(engine, later), {"role": "assistant", "content": "ok reply"}]
        assert engine._survival_projection_followers(view, 0, engine._store.get(earlier), {0: 99.0}) == []
    finally:
        engine.shutdown()


def test_rc2_a_new_reply_after_a_projected_user_row_is_stored(tmp_path, summaries, host_estimator):
    """B-ROLL-1 (rc2) control: only the rows stored right after the projection's source are its replies; a
    different reply in that place is new and is stored."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _big_user_view()
    try:
        engine.ingest(view)
        _conflicted(engine)
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        before = len(_rows(engine))
        engine.on_session_end("S", fitted)
    finally:
        engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        changed = [*fitted[:-1], {"role": "assistant", "content": "a different reply"}]
        cold.ingest(changed)
        assert [r["content"] for r in _rows(cold)[before:]] == ["a different reply"]
    finally:
        cold.shutdown()


def _projected_tool_fit(tmp_path, host_estimator, **patches):
    """The big-tool view fitted after an injected conflict: (fitted list, its projected tool message)."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _big_tool_view()
    engine.ingest(view)
    _conflicted(engine)
    for name, value in patches.items():
        setattr(engine._store, name, value)
    fitted = engine.compress(view, current_tokens=host_estimator(view))
    projected = next(m for m in fitted if m.get("role") == "tool")
    assert NOTICE in projected["content"] and projected["content"] != view[-1]["content"]
    return engine, fitted, projected


@pytest.mark.parametrize("change", ["tool_name", "one_byte"])
def test_r4_b_a_new_message_never_takes_a_projected_rows_identity(tmp_path, summaries, host_estimator, change):
    """R4-2: the host's list holds, where the projection was, a genuinely new message -- the projection
    with another tool name (same content and tool_call_id), or with one byte changed. A cold resume
    stores it as a new row: it never takes the projected row's identity."""
    engine, fitted, projected = _projected_tool_fit(tmp_path, host_estimator)
    if change == "tool_name":
        new = {**projected, "tool_name": "another_tool"}
    else:
        text = projected["content"]
        new = {**projected, "content": text[:5] + ("X" if text[5] != "X" else "Y") + text[6:]}
    before = len(_rows(engine))
    engine.on_session_end("S", fitted)
    engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        cold.ingest([new if m is projected else m for m in fitted])
        added = _rows(cold)[before:]
        assert [(r["content"], r["tool_name"]) for r in added if r["role"] == "tool"] == [
            (new["content"], new.get("tool_name"))], [(r["role"], r["tool_name"]) for r in added]
        # (a changed tool result re-stores its call group on a cold resume: reconcile's #259 policy, as at base)
    finally:
        cold.shutdown()


def test_r4_b_a_failed_metadata_write_never_stores_a_projection(tmp_path, summaries, host_estimator):
    """R4-3: the projection needs no metadata: with every metadata write failing, a cold resume of the
    fitted list stores only the new turn."""
    def refuse(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    engine, fitted, _projected = _projected_tool_fit(tmp_path, host_estimator, write_metadata_json=refuse)
    before = len(_rows(engine))
    engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        new = _turn("NEW", 500.0)
        cold.ingest([*fitted, *new])
        assert [r["content"] for r in _rows(cold)[before:]] == [m["content"] for m in new]
    finally:
        cold.shutdown()


def test_r4_b_seventy_projections_then_a_cold_resume_store_no_stub(tmp_path, summaries, host_estimator):
    """R4-3: 70 fits, each projecting its newest over-budget user turn (no table to evict), then a cold
    process resumes the list of all 70 projected turns: only the new turn is stored."""
    engine = _engine(tmp_path, context_length=WINDOW)
    system = {"role": "system", "content": "system prompt"}
    turns = [[{"role": "user", "content": f"[B{i}] big " + "word " * 6000, "timestamp": 10.0 + i},
              {"role": "assistant", "content": f"ok {i}"}] for i in range(70)]
    try:
        engine.ingest([system, *[m for turn in turns for m in turn]])
        projected = []
        for turn in turns:
            fitted = engine._survival_fit([system, *turn], [system, *turn], 0, "test")
            assert fitted[1]["content"] != turn[0]["content"] and NOTICE in fitted[1]["content"]
            projected += fitted[1:]
        before = len(_rows(engine))
    finally:
        engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        new = _turn("NEW", 900.0)
        cold.ingest([system, *projected, *new])
        added = _rows(cold)[before:]
        assert [r["content"] for r in added] == [m["content"] for m in new], len(added)
    finally:
        cold.shutdown()


def test_r4_add_a_exception_path_never_trusts_a_stale_cursor(tmp_path, summaries, host_estimator, monkeypatch):
    """R4 addendum (a) LOSSLESS: the host rewrites an older user row (same list length, so the cursor
    still equals the length) and compress()'s ingest raises: the rewritten row is not stored, so the fit
    never cuts past it, and it is stored once the store recovers."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)
        index = next(i for i, m in enumerate(view) if m["role"] == "user" and i > 3)  # a stamped host row
        rewritten = {**view[index], "content": view[index]["content"] + " (edited by the host)"}
        view2 = [rewritten if i == index else m for i, m in enumerate(view)]
        assert engine._ingest_cursor == len(view2) and not engine._ingest_cursor_needs_reconcile

        def failing(*args, **kwargs):  # compress()'s ingest raises before it can move the cursor
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, "_ingest_messages", failing)
        try:
            kept = engine.compress(view2, current_tokens=host_estimator(view2))
        except sqlite3.OperationalError:
            kept = view2  # compress() re-raised: the host keeps its list
        assert any(m is rewritten for m in kept), "an unstored row left the view"
        monkeypatch.undo()
        engine.ingest([*kept, {"role": "assistant", "content": "answer"}])
        assert any(r["content"] == rewritten["content"] for r in _rows(engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("view_of", [_long_view, _list_system_view], ids=["str-system", "list-system"])
def test_r4_add_b_consecutive_fits_keep_one_notice(tmp_path, summaries, host_estimator, view_of):
    """R4 addendum (b): a second fit replaces the first fit's notice in the system slot (one notice,
    carrying the newest counts), for string and list content."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = view_of()
    try:
        engine.ingest(view)
        _conflicted(engine)
        first = engine.compress(view, current_tokens=host_estimator(view))
        notice1 = engine._last_survival_fit["notice"]
        view2 = [*first, *[r for i in range(24) for r in _turn(f"M{i}", 1000.0 + i, tool=i % 3 == 0)]]
        engine.ingest(view2)
        second = engine.compress(view2, current_tokens=host_estimator(view2))
        notice2 = engine._last_survival_fit["notice"]
        system = json.dumps(second[0]["content"]) if isinstance(second[0]["content"], list) else second[0]["content"]
        assert notice2 != notice1 and system.count(NOTICE) == 1 and notice2 in system, system[-400:]
        assert "system prompt" in system
    finally:
        engine.shutdown()


def test_r4_add_c_a_fit_short_of_budget_says_so(tmp_path, summaries, host_estimator, caplog):
    """R4 addendum (c): the newest turn is many small rows (none projectable) over the budget: the smaller
    list is still returned, with one WARNING and reached_budget=false in the fit record and the counter."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = [*_long_view(), {"role": "user", "content": "[N] newest", "timestamp": 999.0},
            *[{"role": "assistant", "content": f"part {i}" + PAD} for i in range(40)]]
    try:
        engine.ingest(view)
        _conflicted(engine)
        with caplog.at_level(logging.WARNING):
            result = engine.compress(view, current_tokens=host_estimator(view))
        assert len(result) < len(view) and host_estimator(result) > TARGET
        assert engine._last_survival_fit["reached_budget"] is False
        assert sum("could not reach budget" in r.getMessage() for r in caplog.records) == 1
        counter = engine._store.read_metadata_json("survival_fit:counter")
        assert counter["last_reached_budget"] is False and counter["unreached_budget_count"] == 1
    finally:
        engine.shutdown()


def test_r5_a_fresh_stamped_copy_of_a_projection_is_a_new_occurrence(tmp_path, summaries, host_estimator):
    """R5-1: a fitted user projection P stays live; after an assistant turn a message byte-equal to P
    arrives with a FRESH host timestamp. It is not a copy of the projection (a copy keeps the source
    row's stamp): it is stored as a new occurrence with its own stamp and maps to that row, while the
    carried P still maps to its source row."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _big_user_view()
    try:
        engine.ingest(view)
        _conflicted(engine)
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        projected = next(m for m in fitted if m.get("role") == "user" and NOTICE in str(m.get("content")))
        source = next(int(r["store_id"]) for r in _rows(engine) if r["content"] == view[-2]["content"])
        again = {**projected, "timestamp": 9999.0}
        host = [*fitted, {"role": "assistant", "content": "an answer" + PAD}, again]
        before = len(_rows(engine))
        engine.ingest(host)
        added = _rows(engine)[before:]
        assert [(r["role"], r["content"], r["observed_at"]) for r in added if r["role"] == "user"] == [
            ("user", again["content"], 9999.0)], [(r["role"], r["observed_at"]) for r in added]
        mapping = engine._get_store_id_map_for_messages(host)
        assert mapping.get(id(projected)) == source and mapping.get(id(again)) == added[-1]["store_id"], (
            mapping.get(id(projected)), mapping.get(id(again)), source)
    finally:
        engine.shutdown()


def test_r6_1_a_restamped_copy_of_an_unstamped_sources_projection_is_that_row(tmp_path, summaries, host_estimator):
    """R6-1 (Hermes 0.21.2): the projected tool row's source was stored unstamped; the host re-inserts the
    projection copy with a FRESH stamp. A cold resume still recognises it (byte equality holds): 0 new rows.
    (A stamped source with another stamp stays a new occurrence: R5-1's test.)"""
    engine, fitted, projected = _projected_tool_fit(tmp_path, host_estimator)
    source = [r for r in _rows(engine) if r["role"] == "tool"][-1]  # call_big's result: stored unstamped
    assert source["observed_at"] is None
    before = len(_rows(engine))
    engine.on_session_end("S", fitted)
    engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        restamped = [{**m, "timestamp": 500.0} if m is projected else m for m in fitted]
        full = [*restamped, *_turn("NEW", 900.0)]
        cold.ingest(full)
        assert [r["content"] for r in _rows(cold)[before:]] == [m["content"] for m in full[len(restamped):]]
        assert cold._get_store_id_map_for_messages(full).get(id(full[fitted.index(projected)])) == \
            int(source["store_id"])
    finally:
        cold.shutdown()


def test_r6_3_a_failed_legacy_probe_is_retried_not_cached_empty(tmp_path, monkeypatch):
    """R6-3: the bind probe fails once (a transient lock); it is not recorded as "no legacy ids": the next
    read probes again and the legacy whitespace-only row stays owned."""
    import hermes_lcm.db_bootstrap as db_bootstrap

    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        engine._store.connection.execute("UPDATE messages SET conversation_id = '  ' WHERE store_id = ?", (ids["B"],))
        engine._store.connection.commit()
    finally:
        engine.shutdown()
    real, calls = db_bootstrap._stored_conversation_ids, []

    def flaky(conn):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(conn)

    monkeypatch.setattr(db_bootstrap, "_stored_conversation_ids", flaky)
    db_bootstrap._LEGACY_CONVERSATION_IDS.clear()
    engine = _engine(tmp_path)  # the bind probe fails
    try:
        rows, _truncated = engine._store_complete_owned_rows(0, None, [])
        assert len(calls) >= 2 and ids["B"] in [int(r["store_id"]) for r in rows]
    finally:
        engine.shutdown()


def test_r6_4_a_pending_fit_warning_never_crosses_to_another_conversation(tmp_path, summaries, host_estimator):
    """R6-4: a fit in conversation A leaves its warning pending; the engine re-binds to B: A's warning is
    not delivered in B, and B's first fit warns."""
    engine = _engine(tmp_path, context_length=WINDOW)
    try:
        view = _long_view()
        engine.ingest(view)
        _conflicted(engine)
        engine.compress(view, current_tokens=host_estimator(view))
        assert engine._survival_fit_pending_warning
        engine.on_session_start("S2", platform="telegram", context_length=WINDOW, conversation_id="convB")
        assert engine.get_automatic_compaction_status_message(phase="compress", default_message="x") is None
        view_b = [{"role": "system", "content": "system prompt"},
                  *[r for i in range(24) for r in _turn(f"B{i}", 2000.0 + i, tool=i % 3 == 0)]]
        engine.ingest(view_b)
        _conflicted(engine)
        engine.compress(view_b, current_tokens=host_estimator(view_b))
        assert engine.get_automatic_compaction_status_message(phase="compress", default_message="x")
    finally:
        engine.shutdown()


def _group(tag, result_words):
    call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag} " + "row " * result_words}]


@pytest.mark.parametrize("first", [False, True], ids=["second-group-crosses", "first-group-oversized"])
def test_r6_5_store_complete_admits_a_tool_group_whole_or_not_at_all(tmp_path, summaries, first):
    """R6-5: a hidden tool group whose results cross the leaf budget ends the leaf before its call once
    the leaf covers a stored row; a first oversized group is admitted whole. Never a call/result split."""
    engine = _engine(tmp_path, leaf_chunk_tokens=1200)
    big = _group("BIG", 2500)
    old = ([*big, {"role": "assistant", "content": "after big" + PAD}] if first else
           [*_turn("H1", 100.0, tool=True), {"role": "user", "content": "[H2] turn" + PAD, "timestamp": 110.0},
            *big, {"role": "assistant", "content": "after big" + PAD}])
    tail = _turn("T9", 900.0)
    try:
        engine.ingest([*old, *tail])
        rows = _rows(engine)
        call_id = next(int(r["store_id"]) for r in rows if r["role"] == "assistant" and r.get("tool_calls")
                       and "call_BIG" in json.dumps(r["tool_calls"]))
        engine.compress(list(tail))
        covered = set(_covered(engine))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert (call_id in covered) == (call_id + 1 in covered)  # no call/result split
        assert (call_id in covered) == first
    finally:
        engine.shutdown()


def _scan_cap_store(engine, call_pos, parallel=False, calls=None):
    """``call_pos - 1`` small hidden rows, an assistant tool-call row at owned position ``call_pos`` (two
    calls when ``parallel``, ``calls`` when given), its result rows, a reply, then the fresh tail (the only
    host view)."""
    hidden = [{"role": "user", "content": f"h{i:05d} q", "timestamp": float(i)} if i % 2 else
              {"role": "assistant", "content": f"h{i:05d} a"} for i in range(1, call_pos)]
    ids = ["call_edge", "call_side"] if parallel else ["call_edge"]
    ids = ids if calls is None else ["call_edge", *[f"call_{k}" for k in range(1, calls)]]
    calls = [{"id": cid, "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for cid in ids]
    group = [{"role": "assistant", "content": "", "tool_calls": calls},
             *[{"role": "tool", "tool_call_id": cid, "content": f"RESULT_OF_{cid}"} for cid in ids],
             {"role": "assistant", "content": "reply after the edge tool call"}]
    tail = [{"role": "user", "content": "[T9] fresh tail user", "timestamp": 1e6},
            {"role": "assistant", "content": "[T9] fresh tail reply"}]
    engine.ingest([*hidden, *group, *tail])
    call_id = next(int(r["store_id"]) for r in _rows(engine) if r.get("tool_calls"))
    return tail, call_id


def _unmatched_calls(messages) -> list:
    answered = {str(m.get("tool_call_id") or "") for m in messages if m.get("role") == "tool"}
    return [call["id"] for m in messages for call in (m.get("tool_calls") or []) if call["id"] not in answered]


def _run_scan_cap_leaves(tmp_path, monkeypatch, caplog, cap, call_pos, parallel=False, calls=None,
                         covered_before=False):
    """Two passes over the scan-cap store; returns (call id, covered after pass 1, log lines, serializer inputs)."""
    if cap is not None:
        monkeypatch.setattr(lcm_store_complete, "_SCAN_LIMIT", cap)
    engine = _engine(tmp_path, leaf_chunk_tokens=500_000)
    serialized: list[list] = []
    real = engine._serialize_messages_with_clip
    monkeypatch.setattr(engine, "_serialize_messages_with_clip",
                        lambda messages, session_id=None: serialized.append(list(messages)) or real(messages, session_id))
    try:
        tail, call_id = _scan_cap_store(engine, call_pos, parallel, calls)
        if covered_before:  # every row before the call is already covered by a summary (excluded, not in the leaf)
            engine._dag.add_node(SummaryNode(session_id="OTHER", summary="s", token_count=1, source_token_count=1,
                                             source_ids=list(range(1, call_id))))
        with caplog.at_level(logging.INFO, logger="hermes_lcm.store_complete"):
            view = engine.compress(list(tail))
            assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
            first = set(_covered(engine))
            engine.note_turn_complete()  # #904: the following user turns release the hidden-only hold
            engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                    platform="telegram", conversation_id="conv")
            view = [*view, *[m for j in range(6) for m in (  # the host continues past the retained tail
                {"role": "user", "content": f"[G{j}] new user turn " + "pad " * 20, "timestamp": 2e6 + j},
                {"role": "assistant", "content": f"[G{j}] new reply " + "pad " * 20})]]
            engine.ingest(view)
            engine._config.leaf_chunk_tokens = 200  # the next leaf: the rest of the backlog and host rows
            engine.compress(view)
            assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        every = Counter(int(r[0]) for r in engine._store.connection.execute(
            "SELECT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
            "WHERE node.source_type = 'messages'").fetchall())  # any session's summaries
        below = [int(r["store_id"]) for r in _rows(engine) if int(r["store_id"]) <= _frontier(engine)]
        assert below and all(every[store_id] == 1 for store_id in below), (_frontier(engine), every)
        lines = [r.getMessage() for r in caplog.records if "store-complete leaf" in r.getMessage()]
        return call_id, first, set(_covered(engine)), lines, serialized
    finally:
        engine.shutdown()


@pytest.mark.parametrize("parallel", [False, True], ids=["single-call", "parallel-calls"])
def test_rc2_b_group_1_scan_cap_ends_the_leaf_before_an_unread_tool_group(tmp_path, summaries, monkeypatch, caplog,
                                                                         parallel):
    """B-GROUP-1: the owned-row scan stops at ``_SCAN_LIMIT`` inside a tool group (the call row read, a
    result not): the leaf ends before the call and says cut=True; the next leaf starts at the call."""
    cap = 10
    call_id, first, covered, lines, serialized = _run_scan_cap_leaves(
        tmp_path, monkeypatch, caplog, cap, cap - 1 if parallel else cap, parallel)
    assert max(first) == call_id - 1 and call_id not in first, (sorted(first), call_id)
    assert "cut=True" in lines[0], lines
    assert {call_id, call_id + 1} <= covered, (sorted(covered), call_id)
    assert not summaries[1].startswith("[TOOL RESULT"), summaries[1][:200]  # never the orphan result
    assert -1 < summaries[1].find("read_file") < summaries[1].find("[TOOL RESULT"), summaries[1][:200]
    assert serialized and not any(_unmatched_calls(messages) for messages in serialized)


def test_rc2_b_group_1_group_read_whole_before_the_cap_stays_whole(tmp_path, summaries, monkeypatch, caplog):
    """Control: the call at cap-1 and its result at the cap: the group is read whole and stays in the leaf."""
    cap = 10
    call_id, first, _covered_after, lines, serialized = _run_scan_cap_leaves(
        tmp_path, monkeypatch, caplog, cap, cap - 1)
    assert {call_id, call_id + 1} <= first and max(first) == cap, (sorted(first), call_id)
    assert "cut=False" in lines[0], lines
    assert "read_file" in summaries[0] and "RESULT_OF_call_edge" in summaries[0]
    assert not any(_unmatched_calls(messages) for messages in serialized)


def test_rc2_b_group_1_product_scan_limit(tmp_path, summaries, monkeypatch, caplog):
    """The default-cap shape at the real constant: 1,999 hidden rows, the call at 2000, its result at 2001."""
    cap = lcm_store_complete._SCAN_LIMIT
    call_id, first, covered, lines, _serialized = _run_scan_cap_leaves(tmp_path, monkeypatch, caplog, None, cap)
    assert call_id == cap and max(first) == cap - 1 and "cut=True" in lines[0], (call_id, max(first), lines)
    assert {cap, cap + 1} <= covered


@pytest.mark.parametrize("shape", ["covered-before", "first-owned-row"])
def test_rc2_b_group_1_a_first_group_at_the_scan_cap_is_read_whole(tmp_path, summaries, monkeypatch, caplog, shape):
    """A first tool group (no covered row before it in the leaf) open at the scan cap: one bounded larger
    scan reads its results and the group is admitted whole, never a call without its result.
    covered-before: 1,999 rows an existing summary covers, the call at 2000, its result at 2001.
    first-owned-row: the call is the first owned row, twelve parallel calls, their results past cap 10."""
    covered_before = shape == "covered-before"
    call_id, first, covered, lines, serialized = _run_scan_cap_leaves(
        tmp_path, monkeypatch, caplog, None if covered_before else 10, lcm_store_complete._SCAN_LIMIT
        if covered_before else 1, calls=None if covered_before else 12, covered_before=covered_before)
    results = 1 if covered_before else 12
    assert set(range(call_id, call_id + results + 1)) <= first, (sorted(first)[-5:], call_id)
    assert serialized and not any(_unmatched_calls(messages) for messages in serialized)
    assert "[TOOL RESULT" not in summaries[1].split("\n\n")[0], summaries[1][:200]  # next leaf: no orphan result


class _Heartbeat:
    """An ignore pattern without the optional ``regex`` engine (CI does not install it)."""
    pattern = "HEARTBEAT_PING"

    def search(self, text, timeout=None):
        return object() if self.pattern in str(text) else None


@pytest.mark.parametrize("ignored", [True, False])
def test_r3c_a_skipped_prompt_ends_an_ignored_dependent_run(tmp_path, summaries, ignored):
    """R3-C (#5): after a resume re-binds the frontier at 0, a stored ignore-matched prompt makes the
    next replies dependent; a node-covered (skipped) prompt ends that run, so the first uncovered reply
    to a real prompt is summarized, never passed uncovered."""
    engine = _engine(tmp_path)
    view = []
    for i in range(1, 9):
        text = "HEARTBEAT_PING check" + PAD if (i == 2 and ignored) else f"[U{i}] turn" + PAD
        view += [{"role": "user", "content": text, "timestamp": 10.0 * i},
                 {"role": "assistant", "content": f"reply U{i}" + PAD}]
    view.append({"role": "user", "content": "[U9] turn" + PAD, "timestamp": 90.0})
    try:
        engine.ingest(view)
        engine.compress(view)
        reply = next(int(r["store_id"]) for r in _rows(engine) if r["content"].startswith("reply U8"))
        engine.on_session_start("C", boundary_reason="compression", old_session_id="S", platform="telegram",
                                conversation_id="conv")
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:
        assert _frontier(engine) == 0
        engine._compiled_ignore_message_patterns = [_Heartbeat()]  # the operator adds an ignore pattern
        tail = [{"role": "user", "content": "[U10] new" + PAD, "timestamp": 100.0},
                {"role": "assistant", "content": "reply U10" + PAD}]
        engine.ingest(tail)
        engine.compress(list(tail))
        covered = {int(r[0]) for r in engine._store.connection.execute(
            "SELECT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
            "WHERE node.source_type = 'messages'").fetchall()}  # any session's summaries
        assert not (_frontier(engine) >= reply and reply not in covered), (_frontier(engine), reply)
    finally:
        engine.shutdown()


def test_h_survival_fit_off_leaves_the_result_unchanged(tmp_path, summaries, host_estimator):
    engine = _engine(tmp_path, context_length=WINDOW, survival_fit=False)
    view = _long_view()
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        result = engine.compress(view, current_tokens=host_estimator(view))
        assert result == view and engine.emit_automatic_compaction_status is False
        assert engine._store.read_metadata_json("survival_fit:counter") is None
    finally:
        engine.shutdown()


# -- (h) flag off: unchanged --------------------------------------------------------------------------

def test_h_identity_anchor_off_leaves_the_eva_shape_unchanged(tmp_path, summaries, monkeypatch):
    """With LCM_IDENTITY_ANCHOR=false no row is read from the store: the eva shape fails open as before."""
    view = _eva_store(tmp_path)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    engine = _engine(tmp_path, survival_fit=False)
    try:
        result = engine.compress(view)
        assert engine._last_compression_status == "error"
        assert engine._last_compression_noop_reason == "summary publication could not prove contiguous source coverage"
        assert len(result) == len(view) and _covered(engine) == []
    finally:
        engine.shutdown()
