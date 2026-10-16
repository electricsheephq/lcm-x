"""#436: host-timestamp-anchored, occurrence-bound message identity (REVISION 1).

(a)-(e) are the REVISION 1 acceptance counterexamples; the rest pin one rule each. Engine-level only:
a host list in, stored rows / relations / summary coverage out."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections import Counter

import pytest

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SYSTEM = {"role": "system", "content": "stable system prompt"}
PAD = " alpha beta gamma delta" * 40


def _u(text: str, ts: float | None) -> dict:
    return {"role": "user", "content": text} if ts is None else {"role": "user", "content": text, "timestamp": ts}


def _a(text: str, ts: float | None) -> dict:
    return {"role": "assistant", "content": text} if ts is None else {"role": "assistant", "content": text, "timestamp": ts}


def _turns(start: int, count: int, base_ts: float) -> list[dict]:
    rows = []
    for index in range(start, start + count):
        rows += [_u(f"[T{index}] user turn {index}:{PAD}", base_ts + index * 10),
                 _a(f"reply to T{index}", base_ts + index * 10 + 1)]
    return rows


@pytest.fixture
def summaries(monkeypatch):
    """Every summarizer input, in call order."""
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def _engine(tmp_path, session: str = "S", conversation: str = "conv") -> LCMEngine:
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
    return engine


def _rows(engine: LCMEngine, session: str | None = None) -> list[dict]:
    return [row for row in engine._store.get_session_messages(session or engine._session_id) if row.get("role") != "system"]


def _relations(engine: LCMEngine) -> list[tuple]:
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'").fetchone():
        return []
    return conn.execute("SELECT store_id, kind, related_store_id, ordinal FROM message_relations ORDER BY relation_id").fetchall()


def _state_db(tmp_path, sessions: list[tuple[str, str | None, str | None]]) -> None:
    """The host's state.db next to lcm.db: (id, parent_session_id, end_reason)."""
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", sessions)
    conn.commit()
    conn.close()


def _assert_claims_are_in_the_input(engine: LCMEngine, captured: list[str]) -> list[int]:
    """Every leaf source a summary claims had its own copy of its full text in a summarizer input:
    per text, the claims never outnumber the occurrences across the inputs (T5)."""
    text = "\n".join(captured)
    claimed = [sid for node in engine._dag.get_session_nodes(engine._session_id)
               if node.source_type == "messages" for sid in node.source_ids]
    rows = engine._store.get_batch(claimed)
    wanted = Counter(str(rows[store_id].get("content") or "").strip() for store_id in claimed)
    for content, claims in wanted.items():
        assert not content or text.count(content) >= claims, (
            f"{claims} claim(s) of {content[:60]!r} but {text.count(content)} cop(ies) in the input"
        )
    return claimed


# -- (a)-(e) REVISION 1 acceptance -------------------------------------------------------------------

def test_a_unrelated_same_timestamp_row_is_not_absorbed(tmp_path, summaries):
    """Astra's counterexample: R and U share one host stamp and nothing else. U never absorbs R:
    no relation joins them, and R is claimed only with its own bytes in the summarizer input."""
    engine = _engine(tmp_path)
    r = _u("unrelated retained text" + PAD, 500.0)
    u = _u("new user text" + PAD, 500.0)
    head = _turns(1, 3, 0.0)  # Hermes hands the engine no system row
    try:
        engine.ingest([*head, r])
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)  # restart; the host no longer shows R, U carries the same (batch) stamp
    try:
        live = [*head, u, _a("reply to U", 502.0), *_turns(10, 4, 600.0)]
        engine.compress(live)
        by_text = {str(row["content"]): int(row["store_id"]) for row in _rows(engine)}
        r_id, u_id = by_text[r["content"]], by_text[u["content"]]
        assert not [rel for rel in _relations(engine) if {r_id, u_id} <= {rel[0], rel[2]}]
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert u_id in claimed
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
def test_b_summary_never_claims_a_row_whose_text_is_not_in_its_input(tmp_path, summaries, restart):
    """H1 persist override: R's survivor is rewritten to U under R's stamp, R leaves the host list.
    U is stored once, and a summary may claim R only with R's own bytes in the summarizer input."""
    engine = _engine(tmp_path)
    r = _u("interrupted prompt R" + PAD, 500.0)
    u = _u("follow-up U" + PAD, 500.0)
    head = _turns(1, 3, 0.0)  # Hermes hands the engine no system row
    try:
        engine.ingest([*head, r])
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        live = [*head, u, _a("reply to U", 502.0), *_turns(10, 4, 600.0)]
        engine.ingest(live)
        engine.compress(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert texts.count(r["content"]) == 1 and texts.count(u["content"]) == 1
        assert engine._last_compression_status == "compacted"  # R pending forever would stall publication
        u_id = next(int(row["store_id"]) for row in _rows(engine) if row["content"] == u["content"])
        assert u_id in _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()


def test_r4_one_live_occurrence_never_hides_a_second_stored_one(tmp_path, summaries):
    """Two identical occurrences at one host stamp, the host view keeps one of them plus later rows:
    the owed occurrence is rehydrated with its bytes, or the frontier stays below it; publication
    never trips on a coverage hole."""
    engine = _engine(tmp_path)
    x = _u("same payload twice" + PAD, 500.0)
    head = _turns(1, 3, 0.0)
    try:
        engine.ingest([*head, dict(x), dict(x)])
        live = [*head, dict(x), _a("reply to x", 501.0), *_turns(10, 4, 600.0)]
        engine.ingest(live)
        engine.compress(live)
        x_ids = [int(row["store_id"]) for row in _rows(engine) if row["content"] == x["content"]]
        assert len(x_ids) == 2
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        frontier = int(engine._last_compacted_store_id or 0)
        assert all(store_id in claimed or store_id > frontier for store_id in x_ids)
        assert summaries[-1].count("same payload twice") == 2 or x_ids[1] > frontier
    finally:
        engine.shutdown()


def test_c_same_conversation_sibling_session_gets_no_carry(tmp_path, summaries):
    """A session that merely shares the conversation id is not a compression child: its replay of
    the other session's rows gets no carry, and its publication claims only its own rows."""
    _state_db(tmp_path, [("A", None, "user_exit"), ("B", None, None)])
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path, "A")
    try:
        engine.ingest(history)
        a_ids = {int(row["store_id"]) for row in _rows(engine, "A")}
        engine.on_session_start("B", platform="cli", context_length=200_000, conversation_id="conv")
        live = [*history, *_turns(10, 4, 600.0)]
        engine.ingest(live)
        assert engine._load_compression_carry_ranges() == []
        engine.compress(live)
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert not a_ids & set(claimed)
        assert len(_rows(engine, "B")) == len(live) - 1  # B keeps its own copy of every host row
    finally:
        engine.shutdown()


def test_c_positive_control_a_verified_compression_child_inherits_its_parents_rows(tmp_path):
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path, "P")
    try:
        engine.ingest(history)
        engine.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
        engine.ingest([*history, *_turns(10, 1, 600.0)])
        assert [str(row["content"]) for row in _rows(engine, "C")] == [
            m["content"] for m in _turns(10, 1, 600.0)
        ]
        assert {source for source, _a, _b in engine._load_compression_carry_ranges()} == {"P"}
    finally:
        engine.shutdown()


def test_d_two_identical_gateway_messages_at_one_timestamp_are_both_stored(tmp_path):
    engine = _engine(tmp_path)
    ok = _u("ok", 700.0)
    try:
        head = [SYSTEM, *_turns(1, 2, 0.0)]
        engine.ingest([*head, dict(ok), _a("reply one", 701.0)])
        live = [*head, dict(ok), _a("reply one", 701.0), dict(ok), _a("reply two", 702.0)]
        engine.ingest(live)
        assert [str(row["content"]) for row in _rows(engine)].count("ok") == 2
    finally:
        engine.shutdown()
    restarted = _engine(tmp_path)  # a restart replays the same host list: still exactly two
    try:
        restarted.ingest(live)
        assert [str(row["content"]) for row in _rows(restarted)].count("ok") == 2
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("flag", ["true", "false"])
def test_e_legacy_null_observed_at_store_behaves_unchanged(tmp_path, monkeypatch, summaries, flag):
    """Rows without a host stamp (a legacy store, a host that sends none) take today's path."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", flag)
    history = [SYSTEM, *[{k: v for k, v in m.items() if k != "timestamp"} for m in _turns(1, 4, 0.0)]]
    engine = _engine(tmp_path)
    try:
        engine.ingest(history)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:
        live = [*history, _u("[T9] later" + PAD, None), _a("reply to T9", None), *_turns(10, 3, 600.0)]
        out = engine.compress(live)
        rows = _rows(engine)
        assert all(row.get("observed_at") is None for row in rows[: len(history) - 1])
        assert [str(row["content"]) for row in rows] == [m["content"] for m in live[1:]]
        assert engine._last_compression_status == "compacted"
        assert _relations(engine) == []
        _assert_claims_are_in_the_input(engine, summaries)
        assert out
    finally:
        engine.shutdown()


# -- per rule ----------------------------------------------------------------------------------------

def test_r1_replay_after_restart_is_matched_per_occurrence(tmp_path):
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path)
    try:
        engine.ingest(history)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:  # the host re-issues its list with a row it dropped: the rest are the same occurrences
        live = [SYSTEM, *_turns(1, 1, 0.0), *_turns(3, 2, 0.0), *_turns(10, 1, 600.0)]
        engine.ingest(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert len(texts) == len(history) - 1 + 2 and len(set(texts)) == len(texts)
    finally:
        engine.shutdown()


def test_r2_a_live_composite_of_stored_rows_is_recognised_with_a_witness(tmp_path, summaries):
    """H3 merge: C = R + "\\n\\n" + U, both stored, carries R's stamp. Nothing new is stored; the
    decomposition is recorded; a summary of C claims R and U with their bytes in its input."""
    engine = _engine(tmp_path)
    r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
    try:
        head = [SYSTEM, *_turns(1, 3, 0.0)]
        engine.ingest([*head, r, u])  # a failed turn: two consecutive user rows
        composite = _u(r["content"] + "\n\n" + u["content"], 500.0)
        live = [*head, composite, _a("reply to U", 511.0), *_turns(10, 4, 600.0)]
        engine.ingest(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert composite["content"] not in texts and texts.count(r["content"]) == 1
        ids = {str(row["content"]): int(row["store_id"]) for row in _rows(engine)}
        kinds = {(rel[1], rel[2]) for rel in _relations(engine)}
        assert {("composite", ids[r["content"]]), ("composite", ids[u["content"]])} <= kinds
        engine.compress(live)
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert {ids[r["content"]], ids[u["content"]]} <= set(claimed)
    finally:
        engine.shutdown()


def test_r3_remainder_is_stored_once_byte_exact(tmp_path):
    """A held head plus a new remainder: U is stored once, with its exact bytes and separators, and
    the recorded constituents rebuild the host composite byte for byte."""
    engine = _engine(tmp_path)
    r = _u("head R\n\n  indented line \n\n\nthree newlines\t", 500.0)
    remainder = "  remainder U \n\nsecond  paragraph\n\n\n  tail \t"
    try:
        head = [SYSTEM, *_turns(1, 2, 0.0)]
        engine.ingest([*head, r])
        composite = _u(r["content"] + "\n\n" + remainder, 500.0)
        engine.ingest([*head, composite, _a("reply", 501.0)])
        rows = _rows(engine)
        texts = [str(row["content"]) for row in rows]
        assert texts.count(r["content"]) == 1 and texts.count(remainder) == 1
        assert composite["content"] not in texts
        stored_u = next(row for row in rows if row["content"] == remainder)
        assert stored_u.get("observed_at") is None  # its own host stamp is unknown
        group = sorted((rel for rel in _relations(engine) if rel[1] == "composite"), key=lambda rel: rel[3])
        by_id = {int(row["store_id"]): str(row["content"]) for row in rows}
        assert "\n\n".join(by_id[rel[2]] for rel in group) == composite["content"]
        engine.ingest([*head, composite, _a("reply", 501.0)])  # re-reading it stores nothing
        assert len(_rows(engine)) == len(rows)
    finally:
        engine.shutdown()


def test_r5_null_stamp_is_backfilled_only_for_the_proven_occurrence(tmp_path):
    engine = _engine(tmp_path)
    try:
        head = [SYSTEM, *[{k: v for k, v in m.items() if k != "timestamp"} for m in _turns(1, 2, 0.0)]]
        engine.ingest(head)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:  # a restart: the host now stamps the same rows
        engine.ingest([SYSTEM, *_turns(1, 2, 0.0), *_turns(10, 1, 600.0)])
        rows = _rows(engine)
        assert len(rows) == 6
        assert [row.get("observed_at") for row in rows[:4]] == [m["timestamp"] for m in _turns(1, 2, 0.0)]
    finally:
        engine.shutdown()


def test_r6_an_in_place_rewrite_is_a_new_version_that_supersedes(tmp_path):
    """The same host object observed before and after with other content: both versions kept, the
    new one records ``supersedes``. Without that observation a mismatch is a distinct occurrence."""
    engine = _engine(tmp_path)
    try:
        head = [SYSTEM, *_turns(1, 2, 0.0)]
        prompt = _u("draft prompt" + PAD, 500.0)
        engine.ingest([*head, prompt])
        prompt["content"] = "edited prompt" + PAD
        engine.ingest([*head, prompt, _a("reply", 501.0)])
        rows = _rows(engine)
        ids = {str(row["content"]): int(row["store_id"]) for row in rows}
        assert "draft prompt" + PAD in ids and "edited prompt" + PAD in ids
        assert (ids["edited prompt" + PAD], "supersedes", ids["draft prompt" + PAD], None) in _relations(engine)
    finally:
        engine.shutdown()


def test_flag_off_writes_no_identity_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    engine = _engine(tmp_path)
    try:
        r = _u("R" + PAD, 500.0)
        engine.ingest([SYSTEM, r])
        engine.ingest([SYSTEM, _u(r["content"] + "\n\nU", 500.0)])
        assert _relations(engine) == []
        assert not engine._store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='message_relations'"
        ).fetchone()
    finally:
        engine.shutdown()


def test_r6_an_unstamped_same_object_rewrite_keeps_both_versions(tmp_path):
    """No host timestamps anywhere: the same host dict rewritten draft -> edited after LCM stored it.
    The new bytes are stored as a version that supersedes the old; a re-read adds nothing."""
    engine = _engine(tmp_path)
    try:
        first = _u("the assistant said hi", None)
        prompt = _u("draft", None)
        engine.ingest([first, prompt])
        prompt["content"] = "edited"
        engine.ingest([first, prompt])
        engine.ingest([first, prompt])
        rows = _rows(engine)
        assert [row["content"] for row in rows] == ["the assistant said hi", "draft", "edited"]
        ids = {str(row["content"]): int(row["store_id"]) for row in rows}
        assert (ids["edited"], "supersedes", ids["draft"], None) in _relations(engine)
    finally:
        engine.shutdown()


# -- G-REL-1 residuals ---------------------------------------------------------------------------------

def _h2_merge_turn_history(engine) -> tuple[list[dict], dict]:
    """H2 (0.21.5) after a crash turn: R has no reply, the merge turn's live list holds C = R + "\\n\\n"
    + U under R's stamp (the witness). The durable list then re-flushes the survivor (U under R's
    stamp) and C as rows of their own, and the rotation turn's prompt X twice (the child's copy, then
    the turn flush) -- after LCM already stored X's reply."""
    head = _turns(1, 3, 0.0)
    r, u = _u("[R] crash-turn prompt" + PAD, 500.0), _u("[U] merge-turn prompt" + PAD, 510.0)
    c, reply = _u(r["content"] + "\n\n" + u["content"], 500.0), _a("reply to U", 511.0)
    x, reply_x = _u("[X] rotation-turn prompt" + PAD, 600.0), _a("reply to X", 601.0)
    engine.ingest([*head, r, u])
    engine.ingest([*head, c, reply])
    live = [*head, r, u, _u(u["content"], 500.0), reply, *_turns(10, 1, 520.0), dict(c), *_turns(11, 2, 540.0), x]
    engine.ingest(live)
    engine.ingest([*live, reply_x])
    live = [*live, dict(x), reply_x, *_turns(20, 3, 700.0)]
    engine.ingest(live)
    return live, {"r": r, "u": u, "c": c, "x": x}


def _sweep_engine(tmp_path) -> LCMEngine:
    """Over threshold with one-row leaf chunks: every pass publishes the oldest raw row (or tool group)."""
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, threshold_full_sweep_enabled=True,
                       database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start("S", platform="cli", context_length=4_000, conversation_id="conv")
    return engine


def test_563_h2_merge_turn_views_are_replays_and_never_brick_publication(tmp_path, summaries):
    """#563: the witnessed composite and its survivor are views of stored R and U (never new rows);
    a leaf chunk of views alone, or one the host re-ordered around X's re-flush, still publishes."""
    engine = _sweep_engine(tmp_path)
    try:
        live, m = _h2_merge_turn_history(engine)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert texts.count(m["r"]["content"]) == 1 and texts.count(m["u"]["content"]) == 1
        assert m["c"]["content"] not in texts
        statuses = []
        for turn in range(30, 34):  # leaf chunks of one row: each pass meets the views and X's re-flush
            live = engine.compress(live)
            statuses.append(engine._last_compression_status)
            live = [*live, *_turns(turn, 1, 1000.0 + turn * 10)]
            engine.ingest(live)
        assert "error" not in statuses and "compacted" in statuses, engine._last_compression_noop_reason
        stored = [int(row["store_id"]) for row in _rows(engine)]
        assert int(engine._last_compacted_store_id or 0) >= stored[-6]
        _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()


def test_t6_a_rescue_prefix_of_rehydrated_rows_alone_consumes_no_unread_raw_row(tmp_path, monkeypatch):
    """#572 T6: R is rehydrated ahead of the raw rows. When the summarizer rejects the whole input and
    the adaptive rescue keeps only the oldest prefix [R], no raw row is consumed or claimed unread."""
    read: list[str] = []

    def summarize(**kwargs):
        if "[U]" in kwargs["text"]:
            raise RuntimeError("maximum context length exceeded")
        read.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    engine = _engine(tmp_path)
    r = _u("[R] interrupted prompt" + PAD * 12, 500.0)
    try:
        engine.ingest([r])
        engine.shutdown()
        engine = _engine(tmp_path)  # restart: the persist override left U under R's stamp; R left the list
        with contextlib.suppress(RuntimeError):  # a failed rescue may surface as the pass's error
            engine.compress([_u("[U] follow-up", 500.0), _a("reply to U", 502.0), *_turns(10, 3, 600.0)])
        frontier = int(engine._last_compacted_store_id or 0)
        u_id = next(int(row["store_id"]) for row in _rows(engine) if row["content"] == "[U] follow-up")
        claimed = [sid for node in engine._dag.get_session_nodes(engine._session_id)
                   if node.source_type == "messages" for sid in node.source_ids]
        assert u_id not in claimed and u_id > frontier, (claimed, frontier)
        assert all("[U]" not in text for text in read)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["1000-paragraphs", "overlapping-dead-ends"])
def test_t3_decomposition_search_is_bounded_and_ambiguous_on_exhaustion(shape):
    """#572 T3: a composite of ~1000 short held paragraphs (recursion depth) or of overlapping held
    texts that all dead-end (exponential paths) returns fast as "ambiguous" (None): stored whole."""
    from hermes_lcm.identity_anchor import _decompositions

    if shape == "1000-paragraphs":
        content, texts = "\n\n".join(["ok"] * 1000) + "\n\nnew tail", {"ok"}
    else:
        content, texts = "\n\n".join(["a"] * 40) + "\n\nX", {"a", "a\n\na", "a\n\na\n\na"}
    started = time.monotonic()
    assert _decompositions(content, texts, partial=False) is None
    assert len(_decompositions(content, texts, partial=True) or []) <= 3  # stops at its cap, never recurses
    assert time.monotonic() - started < 1.0
    assert _decompositions("R one\n\nU two", {"R one", "U two"}, partial=False) == [(["R one", "U two"], "")]


def test_t4_ancestry_is_cached_only_after_a_completed_read(tmp_path):
    """#572 T4: no state.db yet (or a failed read) is not cached as "no ancestors" for the session."""
    engine = _engine(tmp_path, "C")
    try:
        assert engine._identity_anchor_chain() == []  # the host has not created state.db yet
        _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
        assert engine._identity_anchor_chain() == ["P"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("path", ["delete_session_messages", "doctor_clean"])
def test_t7_relations_are_deleted_with_their_rows(tmp_path, path):
    """#572 T7: a deleted row's relations (as head or member) go in the same transaction; a reused
    store_id never inherits one."""
    import hermes_lcm.command as command_mod

    engine = _engine(tmp_path)
    try:
        r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
        head = [SYSTEM, *_turns(1, 3, 0.0)]
        engine.ingest([*head, r, u])
        engine.ingest([*head, _u(r["content"] + "\n\n" + u["content"], 500.0), _a("reply to U", 511.0)])
        assert _relations(engine)
        if path == "delete_session_messages":
            engine._store.delete_session_messages("S")
        else:
            engine.on_session_start("other", platform="cli", context_length=200_000, conversation_id="conv2")
            command_mod._delete_clean_candidates_atomically(engine, {"S"})
            assert not engine._store.get_session_messages("S")
        assert _relations(engine) == []
    finally:
        engine.shutdown()



# -- D-D' R1-ws + plan (ii): H1 persists a prompt's stripped bytes; H1 merges LCM's carrier -------------------

T13 = f"[T13] user turn 13:{PAD}"


def _crash_child(tmp_path, stored: list[dict]) -> LCMEngine:
    """Parent P stored the in-process bytes a crash left (before H1's persist strip); the host resumes in C."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "P")
    engine.ingest([SYSTEM, *_turns(1, 3, 0.0), *stored])
    engine.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def _t13(engine, session):
    return [row for row in _rows(engine, session) if str(row["content"]).startswith("[T13]")]


@pytest.mark.parametrize("path", ["fresh-child", "host-reload"])
def test_r1ws_a_stripped_view_at_the_same_stamp_is_the_stored_occurrence(tmp_path, summaries, path):
    """D-D' target: ``T13\\n`` stored, the child's ``T13`` at the SAME stamp is that occurrence; it publishes."""
    engine = _crash_child(tmp_path, [_u(T13 + "\n", 130.0)])
    head, t14 = [SYSTEM, *_turns(1, 3, 0.0)], _u("[T14] user turn 14:" + PAD, 140.0)
    live = [*head, _u(T13, 130.0), t14, _a("reply to T14", 141.0), *_turns(20, 4, 600.0)]
    try:
        if path == "host-reload":  # the first child list lacks T13; the host reloads it before the cursor
            engine.ingest([*head, t14, _a("reply to T14", 141.0)])
        engine.ingest(live)
        [parent] = _t13(engine, "P")
        assert parent["content"] == T13 + "\n" and engine._host_rewrite_override_content(parent) == T13
        assert not _t13(engine, "C")
        for _ in range(6):
            live = engine.compress(live)
            assert engine._last_compression_status != "error", engine._last_compression_noop_reason
        assert int(parent["store_id"]) in _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stored, view", [
    ([T13 + "\n"], (T13, 135.0)),  # another stamp
    ([T13 + "\n"], (T13.replace("alpha beta", "alpha  beta", 1), 130.0)),  # an internal byte
    ([T13 + "\n", T13 + "\n"], (T13, 130.0)),  # two stored occurrences, one view: consumes one
])
def test_r1ws_negatives_nothing_else_is_absorbed(tmp_path, stored, view):
    engine = _crash_child(tmp_path, [_u(text, 130.0) for text in stored])
    try:
        engine.ingest([SYSTEM, *_turns(1, 3, 0.0), _u(*view), _a("reply to T13", 136.0)])
        overrides = [engine._host_rewrite_override_content(row) for row in _t13(engine, "P")]
        if len(stored) == 2:
            assert overrides == [T13, None] and not _t13(engine, "C")
        else:
            assert overrides == [None] and [row["content"] for row in _t13(engine, "C")] == [view[0]]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("carrier", ["verified", "unverifiable"])
def test_dd_plan_ii_a_carrier_headed_composite_is_its_stored_rows(tmp_path, summaries, carrier):
    """H1 merges LCM's carrier with the user rows after it (no stamp): verified, it is those rows; else bytes."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "P")
    r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
    live = [SYSTEM, *_turns(1, 6, 0.0), r, u]
    try:
        engine.ingest(live)
        pre, live = list(live), engine.compress(live)
        head = next(m["content"] for m in live if str(m.get("content")).startswith("[Recent Summary"))
        # #659: the forged head drops the carry packet too; with it the stored bytes outgrow the #899 summariser
        # floor under leaf_chunk_tokens=1, and this test is about claims, not about clipping.
        head = head if carrier == "verified" else head.partition(
            "\n\n---\n\n[Earlier user messages")[0].replace("Earlier", "Edited", 1)
        engine.on_session_end("P", pre)
        engine.on_session_start("C", boundary_reason="compression", old_session_id="P", platform="cli",
                                context_length=200_000, conversation_id="conv")
        merged = _u(head + "\n\n" + r["content"] + "\n\n" + u["content"], None)
        live = [live[0], merged, _a("reply to U", 511.0), *_turns(20, 4, 600.0)]
        engine.ingest(live)
        stored = [row["content"] for row in _rows(engine, "C")]
        assert (merged["content"] in stored) is (carrier == "unverifiable")
        assert len(stored) == len(live) - 2 + (carrier == "unverifiable")
        for _ in range(6):
            live = engine.compress(live)
            assert engine._last_compression_status != "error", engine._last_compression_noop_reason
        _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()
