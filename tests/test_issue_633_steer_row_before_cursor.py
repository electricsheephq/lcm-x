"""#633: a /steer row the host inserts before the steady-state ingest cursor is stored.

Hermes 0.21.2+ drains a pending /steer at a turn's first iteration and inserts it as a standalone,
unstamped user row right after the newest tool result -- a tool result of a turn LCM already ingested,
so the row lands before LCM's cursor. The #436 audit re-examines that prefix; it must audit the
unstamped user row too, so the instruction is stored (a duplicate, never a loss: #259)."""

from __future__ import annotations

import json

import pytest

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s633"
STEER_TEXT = "use the staging database, not prod"
# hermes-agent agent/prompt_builder.py STEER_MARKER_OPEN / format_steer_marker
MARKER_OPEN = ("[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this position; "
               "not tool output and not a new delivery when replayed from conversation history]")
STEER_BLOCK = f"{MARKER_OPEN}\n{STEER_TEXT}\n[/OUT-OF-BAND USER MESSAGE]"
REPLY = "Migration done."
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, **overrides) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False, **overrides)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _stamp(row: dict, ts: float | None) -> dict:
    return row if ts is None else {**row, "timestamp": ts}


def _turn_one(stamped: bool) -> list[dict]:
    ts = (lambda value: value) if stamped else (lambda value: None)
    return [
        _stamp({"role": "user", "content": "Run the migration"}, ts(100.0)),
        _stamp({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}, ts(101.0)),
        _stamp({"role": "tool", "tool_call_id": "c1", "content": '{"output": "migrated", "exit_code": 0}'}, ts(102.0)),
        _stamp({"role": "assistant", "content": REPLY}, ts(103.0)),
    ]


def _turn_two(host: list[dict], stamps: str, *, after: int = 2) -> list[dict]:
    """The pre-API drain inserts the steer row after the newest tool result (``after``), then the turn runs."""
    ts = (lambda value: None) if stamps == "none" else (lambda value: value)
    host = list(host)
    host.append(_stamp({"role": "user", "content": "now deploy it"}, ts(200.0)))
    host.insert(after + 1, _stamp({"role": "user", "content": STEER_BLOCK, "display_kind": "steer"},
                                  150.0 if stamps == "all" else None))
    host.append(_stamp({"role": "assistant", "content": "Deploying to staging."}, ts(201.0)))
    return host


def _stored(engine: LCMEngine) -> list[dict]:
    return [row for row in engine._store.get_session_messages(SESSION) if row.get("role") != "system"]


def _steer_rows(engine: LCMEngine) -> list[dict]:
    return [row for row in _stored(engine) if STEER_TEXT in str(row.get("content") or "")]


def _duplicates(engine: LCMEngine) -> int:
    return sum(1 for row in _stored(engine) if row.get("content") == REPLY) - 1


def test_t1_insert_shape_real_host_stamps_stores_the_steer_in_order(tmp_path):
    """Host rows stamped with increasing timestamps, the steer row unstamped (the tested hosts)."""
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)  # post_llm_call, turn 1
        before = len(_stored(engine))
        engine.ingest(_turn_two(host, "host"))  # post_llm_call, turn 2
        rows = _stored(engine)
        contents = [str(row.get("content") or "") for row in rows]
        steer = [i for i, text in enumerate(contents) if STEER_TEXT in text]
        assert len(steer) == 1, "the /steer instruction never reached the store"
        assert rows[steer[0]]["role"] == "user"
        tool = next(i for i, row in enumerate(rows) if row.get("role") == "tool")
        new_user = contents.index("now deploy it")
        assert tool < steer[0] < new_user
        assert len(rows) == before + 3  # the steer + the two new rows; the displaced reply is not stored again
        assert _duplicates(engine) == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stamps", ["none", "host", "all"])
def test_t2_variant_matrix_anchor_on(tmp_path, stamps, record_property):
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=stamps != "none")
        engine.ingest(host)
        engine.ingest(_turn_two(host, stamps))
        assert len(_steer_rows(engine)) == 1
        duplicates = _duplicates(engine)
        record_property("duplicates", duplicates)  # none: the displaced reply past the cursor is a new row (#259)
        assert duplicates <= 1 if stamps == "none" else duplicates == 0
    finally:
        engine.shutdown()


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def test_t3_steer_survives_a_later_forced_compaction(tmp_path, summaries):
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=1)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)
        host = _turn_two(host, "host")
        engine.ingest(host)
        for turn in range(3, 7):
            host += [{"role": "user", "content": f"turn {turn}{PAD}", "timestamp": 100.0 * turn},
                     {"role": "assistant", "content": f"reply {turn}", "timestamp": 100.0 * turn + 1}]
            engine.ingest(host)
        live = engine.compress(host)
        assert engine._last_compression_status == "compacted"
        # #659: the steer is a real user row, so it may come back only as verbatim history in the carry packet.
        assert not any(STEER_TEXT in str(message.get("content") or "").partition("[Earlier user messages")[0]
                       for message in live)
        assert sum(str(message.get("content") or "").count(STEER_TEXT) for message in live) <= 1
        assert len(_steer_rows(engine)) == 1
        grep = json.loads(engine.handle_tool_call("lcm_grep", {"query": "staging database"}))
        hits = [hit for hit in grep.get("results", []) if hit.get("role") == "user"]
        assert hits, grep
    finally:
        engine.shutdown()


def test_t4_steady_state_turn_stores_exactly_the_new_rows(tmp_path):
    """No host change before the cursor: the audit is not triggered."""
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)
        before = len(_stored(engine))
        host = host + [{"role": "user", "content": "now deploy it", "timestamp": 200.0},
                       {"role": "user", "content": STEER_BLOCK},  # a steer drained after a tool batch: appended
                       {"role": "assistant", "content": "Deploying to staging.", "timestamp": 201.0}]
        engine.ingest(host)
        assert len(_stored(engine)) == before + 3
        assert _duplicates(engine) == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stamps", ["host", "none"])
def test_t4_an_unstamped_user_row_already_stored_is_not_stored_again(tmp_path, stamps):
    """Turn 1 carried a steer appended after its tool batch (stored). A turn-2 steer inserted before the
    cursor puts that stored row in the audited range: its stored copy explains it."""
    stamped = stamps != "none"
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=stamped)
        earlier = {"role": "user", "content": "[OUT-OF-BAND USER MESSAGE]\nearlier steer\n[/OUT-OF-BAND USER MESSAGE]"}
        host.insert(3, earlier)
        engine.ingest(host)
        before = len(_stored(engine))
        engine.ingest(_turn_two(host, stamps))
        rows = _stored(engine)
        assert sum(1 for row in rows if row.get("content") == earlier["content"]) == 1
        assert len(_steer_rows(engine)) == 1
        assert len(rows) == before + 3 + (0 if stamped else 1)  # none: the displaced reply (T2)
    finally:
        engine.shutdown()


def test_t4_an_unstamped_user_row_the_last_view_lacked_is_explained_by_its_stored_copy(tmp_path):
    """The stored-copy lookup, not the last view: the host drops a stored unstamped user row for one turn
    and shows it again before the cursor. Its stored copy in the tail explains it."""
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        earlier = {"role": "user", "content": "[OUT-OF-BAND USER MESSAGE]\nearlier steer\n[/OUT-OF-BAND USER MESSAGE]"}
        engine.ingest(host[:3] + [earlier] + host[3:])
        host = host + [{"role": "user", "content": "now deploy it", "timestamp": 200.0},
                       {"role": "assistant", "content": "Deploying.", "timestamp": 201.0}]
        engine.ingest(host)
        before = len(_stored(engine))
        engine.ingest(host[:3] + [earlier] + host[3:])
        assert sum(1 for row in _stored(engine) if row.get("content") == earlier["content"]) == 1
        assert len(_stored(engine)) == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("new_steer", [STEER_BLOCK, STEER_BLOCK + "\n"], ids=["identical", "edge-whitespace"])
def test_t6_a_repeated_steer_is_stored_again(tmp_path, new_steer):
    """The user repeats an earlier /steer verbatim (or up to edge whitespace). The earlier row's stored copy
    belongs to the earlier occurrence the view still shows: it never explains the new insertion."""
    engine = _engine(tmp_path)
    try:
        old = {"role": "user", "content": STEER_BLOCK, "display_kind": "steer"}
        host = [{"role": "user", "content": "Plan the migration", "timestamp": 50.0}, old,
                {"role": "assistant", "content": "Planned.", "timestamp": 51.0}]
        engine.ingest(host)
        host += _turn_one(stamped=True)
        engine.ingest(host)  # [U, old_steer, A, U, A_call, T, reply]
        before = len(_stored(engine))
        host = host + [{"role": "user", "content": "now deploy it", "timestamp": 200.0},
                       {"role": "assistant", "content": "Deploying to staging.", "timestamp": 201.0}]
        host.insert(6, {"role": "user", "content": new_steer, "display_kind": "steer"})
        engine.ingest(host)
        rows = _stored(engine)
        assert len(_steer_rows(engine)) == 2
        assert sum(1 for row in rows if row.get("content") == STEER_BLOCK) == (2 if new_steer == STEER_BLOCK else 1)
        assert len(rows) == before + 3 and _duplicates(engine) == 0
    finally:
        engine.shutdown()


def test_t6_a_repeated_steer_inserted_ahead_of_the_first_is_stored(tmp_path):
    """No new tool result since the first /steer: Hermes inserts the identical second steer after the same
    tool row, ahead of the first. The prefix then first differs at the displaced first steer; the row the
    host shows unchanged at the old position is the new occurrence, and its copy is reserved for it."""
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)
        host = _turn_two(host, "host")  # [U, A_call, T, steer, reply, U2, A2]
        engine.ingest(host)
        assert len(_steer_rows(engine)) == 1
        before = len(_stored(engine))
        host = host + [{"role": "user", "content": "and verify it", "timestamp": 300.0},
                       {"role": "assistant", "content": "Verified.", "timestamp": 301.0}]
        host.insert(3, {"role": "user", "content": STEER_BLOCK, "display_kind": "steer"})
        engine.ingest(host)
        assert len(_steer_rows(engine)) == 2
        assert len(_stored(engine)) == before + 3 and _duplicates(engine) == 0
    finally:
        engine.shutdown()


def test_t7_the_insertion_diff_is_bounded_on_a_long_repetitive_list(tmp_path):
    """A long list of equal identities: the diff is skipped (nothing counts as inserted) instead of
    running quadratic work on the ingest path."""
    import time

    engine = _engine(tmp_path)
    try:
        messages = [{"role": "user", "content": "ok"} for _ in range(8000)]
        engine._last_active_replay_source_identities = [
            engine._message_replay_identity(message, strip_carrier=False) for message in messages[1:]]
        started = time.monotonic()
        assert engine._identity_anchor_inserted(messages, 0, len(messages)) == set()
        assert time.monotonic() - started < 1.0
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#633 out of scope: Hermes <= 0.21.1 appends the steer block in place "
                                       "to an ingested tool row; the audit skips tool rows")
def test_t5_in_place_shape_is_out_of_scope(tmp_path):
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)
        host = list(host) + [{"role": "user", "content": "now deploy it", "timestamp": 200.0}]
        host[2] = {**host[2], "content": host[2]["content"] + "\n\n" + STEER_BLOCK}
        host.append({"role": "assistant", "content": "Deploying to staging.", "timestamp": 201.0})
        engine.ingest(host)
        assert _steer_rows(engine)
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#633 out of scope: LCM_IDENTITY_ANCHOR=0 has no prefix audit")
def test_t5_anchor_off_is_out_of_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0")
    engine = _engine(tmp_path)
    try:
        host = _turn_one(stamped=True)
        engine.ingest(host)
        engine.ingest(_turn_two(host, "host"))
        assert _steer_rows(engine)
    finally:
        engine.shutdown()
