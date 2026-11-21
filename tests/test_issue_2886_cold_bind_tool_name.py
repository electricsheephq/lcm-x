"""Cold-bind regressions for host-built tool results (synthetic gateway and CLI)."""

from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as lcm_engine, store, tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2886)

SESSION = "tool-name-session"
START = 1_795_000_000
EXCHANGES = 120
TOOL = "read_file"
OTHER_TOOL = "search_files"
FILL = " lorem ipsum dolor sit amet"
ZERO_BINDS = [0] * 5


class Transcript:
    """Keep host history independent of the LCM store across engine restarts."""

    def __init__(self, name_only, replay_name="tool_name", *, name_only_exchange=60):
        self.rows = []
        self.name_only = name_only
        self.replay_name = replay_name
        self.name_only_exchange = name_only_exchange

    def history(self, gateway, *, drop_head=0):
        history = deepcopy(self.rows[drop_head:])
        for row in history:
            row.pop("name", None)
            if row.get("tool_call_id") == f"call_{self.name_only_exchange:012x}" and self.replay_name != "tool_name":
                name = row.pop("tool_name", None)
                if self.replay_name == "name":
                    row["name"] = name
            if gateway and row["role"] != "user":
                row.pop("timestamp", None)
            if gateway and row["role"] == "assistant" and not row.get("tool_calls"):
                row.pop("_db_row_snapshot", None)
        return history

    def turn(self, engine, history, exchange):
        stamp = START + len(self.rows)
        call_id = f"call_{exchange:012x}"
        user = {"role": "user", "content": f"[u{exchange}] q" + FILL * 20}
        assistant = {
            "role": "assistant", "content": "", "finish_reason": "tool_calls",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": TOOL, "arguments": "{}"}}],
        }
        result = {
            "role": "tool", "tool_call_id": call_id, "name": TOOL,
            "content": f"[t{exchange}] r" + FILL * 60,
        }
        if self.name_only != "all" and not (self.name_only == "one" and exchange == self.name_only_exchange):
            result["tool_name"] = TOOL
        reply = {"role": "assistant", "content": f"[r{exchange}] a" + FILL * 10, "finish_reason": "stop"}
        live = list(history) + [user]
        for offset, row in enumerate((user, assistant, result, reply), start=1):
            row["timestamp"] = stamp + offset
        prompt_tokens = tokens.count_messages_tokens(live)
        engine.should_compress(prompt_tokens)
        engine.should_compress_preflight(live)
        for row in (user, assistant, result, reply):
            row_id = len(self.rows) + 1
            row.update(_db_persisted=True, message_uid=f"message-{row_id}")
            if row is not user:
                live.append(row)
            host_row = deepcopy(row)
            host_row["_db_row_snapshot"] = f"snapshot-{row_id}"
            if row["role"] == "tool":
                host_row["tool_name"] = row["name"]
            self.rows.append(host_row)
            if row is user or row is result:
                prompt_tokens = tokens.count_messages_tokens(live)
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 5, "total_tokens": prompt_tokens + 5}
                engine.update_from_response(usage)
                engine.should_compress(prompt_tokens)
        engine.ingest(list(live))
        engine.note_turn_complete()
        engine.on_turn_complete(deepcopy(live), usage=usage, turn_id=f"turn-{exchange}", api_call_count=2)
        return live


@pytest.fixture
def bind(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    config = LCMConfig.from_env()

    @contextmanager
    def open_engine():
        engine = LCMEngine(config=config, hermes_home=str(tmp_path))
        try:
            engine.on_session_start(
                SESSION, hermes_home=str(tmp_path), platform="telegram", model="synthetic-model",
                context_length=1_000_000, conversation_id="tool-name-conversation",
            )
            yield engine
        finally:
            engine.shutdown()

    return open_engine


def _seed(bind, transcript, *, gateway):
    with bind() as engine:
        live = []
        for exchange in range(EXCHANGES + 1):
            history = transcript.history(gateway=True) if gateway else live
            live = transcript.turn(engine, history, exchange)
        assert engine._store.get_session_count(SESSION) == len(transcript.rows)
        engine.on_session_end(SESSION, list(live))


def _bind_counts(bind, transcript, *, gateway=True, count=5):
    counts = []
    for _ in range(count):
        with bind() as engine:
            before = engine._store.get_session_count(SESSION)
            exchange = len(transcript.rows) // 4
            live = transcript.turn(engine, transcript.history(gateway), exchange)
            counts.append(engine._store.get_session_count(SESSION) - before - 4)
            transcript.turn(engine, transcript.history(True) if gateway else live, exchange + 1)
            engine.on_session_end(SESSION, transcript.history(gateway))
    return counts


def _turn_row_key(row):
    calls = row.get("tool_calls") or ()
    return row["role"], row["content"], row.get("tool_call_id"), tuple(call["id"] for call in calls)


@pytest.mark.parametrize(("drop_head", "preamble"), [
    pytest.param(40, False, id="drop-40"),
    pytest.param(1, False, id="drop-1"),
    pytest.param(4, False, id="drop-4-complete-exchange"),
    pytest.param(4, True, id="drop-4-cut-call"),
])
def test_gateway_name_only_truncated_cold_binds(bind, drop_head, preamble):
    transcript = Transcript("one")
    if preamble:
        transcript.rows = [
            {"role": "user", "content": "preamble user", "timestamp": START - 2},
            {"role": "assistant", "content": "preamble reply", "timestamp": START - 1,
             "_db_row_snapshot": "preamble"},
        ]
    _seed(bind, transcript, gateway=True)
    assert transcript.rows[drop_head]["role"] == ("tool" if preamble else (
        "assistant" if drop_head == 1 else "user"
    ))

    counts = []
    with bind() as engine:
        original = engine._store.get_session_messages(SESSION)
    all_writes = []
    for exchange in range(EXCHANGES + 1, EXCHANGES + 7, 2):
        with bind() as engine:
            for current in (exchange, exchange + 1):
                before = engine._store.get_session_count(SESSION)
                transcript.turn(engine, transcript.history(True, drop_head=drop_head), current)
                rows = engine._store.get_session_messages(SESSION)
                writes = rows[before:]
                if current == exchange:
                    counts.append(len(writes) - 4)
                all_writes.append((transcript.rows[-4:], writes))
            engine.on_session_end(SESSION, transcript.history(True, drop_head=drop_head))
    print(f"drop {drop_head}, cut call {preamble}: {counts}")
    assert counts == [0, 0, 0]
    assert rows[:len(original)] == original
    all_counts = Counter(map(_turn_row_key, rows))
    for expected, writes in all_writes:
        assert list(map(_turn_row_key, expected)) == list(map(_turn_row_key, writes))
        assert all(all_counts[_turn_row_key(row)] == 1 for row in expected)


def test_post_compaction_name_only_cold_binds_are_part_3(bind, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                        lambda **_: ("Earlier turns.\nExpand for details about: turns", 1))
    transcript = Transcript("one", name_only_exchange=20)
    with bind() as engine:
        for exchange in range(20):
            live = transcript.turn(engine, transcript.history(True), exchange)
        with monkeypatch.context() as compact:
            compact.setattr(engine._config, "fresh_tail_count", 8)
            compact.setattr(engine._config, "leaf_chunk_tokens", 400)
            compact.setattr(engine._config, "context_threshold", 0.001)
            view = engine.compress(list(live), current_tokens=900_000, force=True)
        engine.note_turn_complete()
        assert len(view) < len(live)
        transcript.rows = deepcopy(view)
        for index, row in enumerate(transcript.rows):
            row.setdefault("_db_row_snapshot", f"compacted-{index}")
            if row["role"] == "tool" and row.get("name") and not row.get("tool_name"):
                row["tool_name"] = row["name"]
        for exchange in range(20, 24):
            transcript.turn(engine, transcript.history(True), exchange)
        engine.on_session_end(SESSION, transcript.history(True))

    counts = []
    for exchange in range(24, 32, 2):
        with bind() as engine:
            before = engine._store.get_session_count(SESSION)
            transcript.turn(engine, transcript.history(True), exchange)
            counts.append(engine._store.get_session_count(SESSION) - before - 4)
            transcript.turn(engine, transcript.history(True), exchange + 1)
            engine.on_session_end(SESSION, transcript.history(True))
    print(f"post-compaction copies (part 3): {counts}")
    assert counts == [6, 0, 0, 0]


def _clear_stored_name(tmp_path, exchange=60, name=None):
    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn, conn:
        changed = conn.execute(
            "UPDATE messages SET tool_name = ? WHERE session_id = ? AND role = 'tool' AND tool_call_id = ?",
            (name, SESSION, f"call_{exchange:012x}"),
        )
        assert changed.rowcount == 1


@pytest.mark.parametrize("name_only", ["one", "all"])
def test_gateway_name_only_cold_binds(bind, tmp_path, name_only):
    transcript = Transcript(name_only)
    _seed(bind, transcript, gateway=True)
    counts = _bind_counts(bind, transcript)
    print(f"{name_only} name-only result(s): {counts}")
    assert counts == ZERO_BINDS
    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
        names = dict(conn.execute("SELECT tool_call_id, tool_name FROM messages WHERE role = 'tool'"))
    assert names == {
        f"call_{i:012x}": None if name_only == "all" or i == 60 else TOOL
        for i in range(len(transcript.rows) // 4)
    }


@pytest.mark.parametrize("missing_name", [None, ""], ids=["null", "empty"])
def test_gateway_heals_existing_nameless_rows(bind, tmp_path, missing_name):
    transcript = Transcript("none")
    _seed(bind, transcript, gateway=True)
    _clear_stored_name(tmp_path, name=missing_name)
    counts = _bind_counts(bind, transcript)
    print(f"legacy {missing_name!r} tool name: {counts}")
    assert counts == ZERO_BINDS
    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
        assert conn.execute(
            "SELECT tool_name FROM messages WHERE role = 'tool' AND tool_call_id = ?",
            (f"call_{60:012x}",),
        ).fetchall() == [(missing_name,)]


@pytest.mark.parametrize("legacy", [False, True], ids=["named", "legacy-null"])
@pytest.mark.parametrize("replay_name", ["tool_name", "name", None], ids=["tool_name", "name", "neither"])
def test_gateway_tool_name_projection_is_symmetric(bind, tmp_path, legacy, replay_name):
    transcript = Transcript("none", replay_name=replay_name)
    _seed(bind, transcript, gateway=True)
    if legacy:
        _clear_stored_name(tmp_path)

    counts = _bind_counts(bind, transcript, count=3)
    print(f"stored {'legacy NULL' if legacy else 'named'}, incoming {replay_name or 'neither'}: {counts}")
    assert counts == [0] * 3
    with bind() as engine:
        rows = engine._store.get_session_messages(SESSION)
        assert [row["tool_name"] for row in rows if row["tool_call_id"] == f"call_{60:012x}"] == [
            None if legacy else TOOL,
        ]


class KnownColdBindCopies(AssertionError):
    """Cold-bind counts that match the measured envelope for this case."""


@pytest.mark.parametrize(("original_name", "known_counts"), [
    pytest.param(
        TOOL, [2, 4, 6], id="1674-unstamped-replies",
        marks=pytest.mark.xfail(
            strict=True, raises=KnownColdBindCopies,
            reason="#1674: unstamped replies with both tool names present (2, 4, 6 extra rows)",
        ),
    ),
    pytest.param(
        None, [63, 65, 67], id="2886-copy-set-heal",
        marks=pytest.mark.xfail(
            strict=True, raises=KnownColdBindCopies,
            reason="#2886: copy-set heal with an original NULL tool name (63–67 extra rows per bind)",
        ),
    ),
])
def test_gateway_with_legacy_copy_set(bind, tmp_path, monkeypatch, original_name, known_counts):
    transcript = Transcript("one")
    _seed(bind, transcript, gateway=True)
    _clear_stored_name(tmp_path)
    # Give one bind the legacy comparison. Its replay already carries the
    # name from the host, so the original row stays NULL and its copy is named.
    with monkeypatch.context() as legacy:
        legacy.setattr(LCMEngine, "_missing_tool_name_matches", staticmethod(lambda *_: False), raising=False)
        assert _bind_counts(bind, transcript, count=1) == [63]

    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn, conn:
        tool_rows = conn.execute(
            "SELECT store_id, tool_name FROM messages WHERE session_id = ? AND role = 'tool' "
            "AND tool_call_id = ? ORDER BY store_id",
            (SESSION, f"call_{60:012x}"),
        ).fetchall()
        assert [name for _, name in tool_rows] == [None, TOOL]
        if original_name is not None:
            conn.execute("UPDATE messages SET tool_name = ? WHERE store_id = ?", (original_name, tool_rows[0][0]))
        before_store_id = conn.execute("SELECT max(store_id) FROM messages").fetchone()[0]

    counts = _bind_counts(bind, transcript, count=3)
    print(f"legacy copy set with original name {original_name!r}: {counts}")
    if counts == known_counts:
        if original_name is not None:
            with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
                copies = conn.execute(
                    "SELECT copy.role, copy.tool_calls, EXISTS ("
                    " SELECT 1 FROM messages AS original WHERE original.session_id = copy.session_id"
                    " AND original.store_id < copy.store_id AND original.role = copy.role"
                    " AND original.content = copy.content AND original.observed_at IS NOT NULL"
                    ") FROM messages AS copy "
                    "WHERE copy.session_id = ? AND copy.store_id > ? AND copy.observed_at IS NULL",
                    (SESSION, before_store_id),
                ).fetchall()
            assert copies == [("assistant", None, 1)] * sum(known_counts)
        raise KnownColdBindCopies(f"legacy copy set: {counts}")
    assert counts == [0] * 3


@pytest.mark.parametrize("stored_name", [TOOL, None], ids=["explicit", "inferred"])
def test_reused_call_id_with_different_tool_is_not_replay(bind, stored_name):
    call_id = "reused-call"
    with bind() as engine:
        engine._ingest_messages([
            {"role": "user", "content": "find the files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": TOOL, "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": call_id, "tool_name": stored_name, "content": "ok"},
            {"role": "assistant", "content": "found the files"},
            {"role": "user", "content": "intervening user"},
            {"role": "assistant", "content": "intervening reply"},
        ])
    with bind() as engine:
        engine._ingest_messages([
            {"role": "tool", "tool_call_id": call_id, "tool_name": OTHER_TOOL, "content": "ok"},
            {"role": "user", "content": "new user after different tool"},
        ])
        rows = engine._store.get_session_messages(SESSION)
        assert len(rows) == 8
        assert [row["tool_name"] for row in rows if row["tool_call_id"] == call_id] == [stored_name, OTHER_TOOL]


def test_explicit_result_names_override_the_owning_call(bind):
    # Keep the call itself identical so the result's explicit name distinguishes the invocation.
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "reused", "type": "function", "function": {"name": "inspect_file", "arguments": "{}"}},
    ]}
    result = {"role": "tool", "tool_call_id": "reused", "content": "ok", "tool_name": TOOL}
    with bind() as engine:
        engine._ingest_messages([call, result])
    with bind() as engine:
        replay = [call, dict(result, tool_name=OTHER_TOOL)]
        engine._ingest_messages(replay)
        rows = engine._store.get_session_messages(SESSION)
        assert [row["tool_name"] for row in rows if row["role"] == "tool"] == [TOOL, OTHER_TOOL]


def test_missing_name_fallback_requires_literal_explicit_name(bind):
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call", "type": "function", "function": {"name": TOOL, "arguments": "{}"}},
    ]}
    nameless = {"role": "tool", "tool_call_id": "call", "content": "ok"}
    named = dict(nameless, name=TOOL)
    stored = [call, nameless]
    with bind() as engine:
        stored_identities = [engine._message_replay_identity(row, stored_row=True) for row in stored]
        incoming_identities = [engine._message_replay_identity(row) for row in (call, named)]
        assert engine._matches_session_with_missing_tool_names(
            [call, named], incoming_identities, stored, stored_identities,
        )
        # The prepared identity still names TOOL, but the original result explicitly names OTHER_TOOL.
        assert not engine._matches_session_with_missing_tool_names(
            [call, dict(named, name=OTHER_TOOL)], incoming_identities, stored, stored_identities,
        )


@pytest.mark.parametrize(("stored_name", "incoming_name", "owners", "restamped", "replayed"), [
    (None, None, (), False, True),
    (TOOL, TOOL, (OTHER_TOOL,), False, True),
    (None, TOOL, (), False, False),
    (None, TOOL, (TOOL, TOOL), False, False),
    (None, TOOL, (OTHER_TOOL,), False, False),
    (None, TOOL, (TOOL,), True, False),
], ids=["both-missing", "both-explicit", "no-owner", "duplicate-owner", "wrong-owner", "new-user-stamp"])
def test_full_history_tool_name_fallback_keeps_existing_guards(
    bind, stored_name, incoming_name, owners, restamped, replayed,
):
    rows = [
        {"role": "user", "content": "find the files", "timestamp": START},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call", "type": "function", "function": {"name": name, "arguments": "{}"}} for name in owners
        ]},
        {"role": "tool", "tool_call_id": "call", "content": "ok", "tool_name": stored_name},
        {"role": "assistant", "content": "found the files"},
    ]
    with bind() as engine:
        engine._store.append_batch(SESSION, rows)
    with bind() as engine:
        incoming = deepcopy(rows)
        incoming[2]["tool_name"] = incoming_name
        if restamped:
            incoming[0]["timestamp"] += 1
        cursor = engine._reconcile_ingest_cursor_from_store(incoming)
    assert (cursor == len(rows)) is replayed


def _repeated_exchange():
    return [
        {"role": "user", "content": "same question", "timestamp": START},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "reused", "type": "function", "function": {"name": TOOL, "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "reused", "tool_name": TOOL, "content": "same result"},
        {"role": "assistant", "content": "same reply"},
    ]


@pytest.mark.parametrize("change", [None, "different-tool", "owner-cut", "restamp", "content", "call-id"])
def test_truncated_tool_name_fallback_keeps_existing_guards(bind, change):
    stored = [{"role": "assistant", "content": "older reply"}, *_repeated_exchange()]
    incoming = deepcopy(stored[1:])
    stored[3].pop("tool_name")
    if change == "different-tool":
        incoming[1]["tool_calls"][0]["function"]["name"] = OTHER_TOOL
        incoming[2]["tool_name"] = OTHER_TOOL
    elif change == "owner-cut":
        incoming = incoming[2:]
    elif change == "restamp":
        incoming[0]["timestamp"] += 1
    elif change == "content":
        incoming[2]["content"] += " changed"
    elif change == "call-id":
        incoming[2]["tool_call_id"] = "different-id"

    with bind() as engine:
        engine._store.append_batch(SESSION, stored)
    with bind() as engine:
        cursor = engine._reconcile_ingest_cursor_from_store(incoming)
    print(f"suffix guard {change}: cursor {cursor} of {len(incoming)}")
    assert (cursor == len(incoming)) is (change is None)


@pytest.mark.parametrize("repeats", [3, 5])
@pytest.mark.parametrize("leading_nameless", [False, True], ids=["exact-leading-result", "owner-cut"])
def test_truncated_tool_name_fallback_uses_only_the_aligned_suffix(bind, repeats, leading_nameless):
    stored = [row for _ in range(repeats) for row in _repeated_exchange()]
    incoming = deepcopy(stored[2:])
    stored[-2].pop("tool_name")
    if leading_nameless:
        stored[2].pop("tool_name")

    with bind() as engine:
        engine._store.append_batch(SESSION, stored)
    with bind() as engine:
        cursor = engine._reconcile_ingest_cursor_from_store(incoming)
    print(f"repeated groups {repeats}, leading nameless {leading_nameless}: cursor {cursor} of {len(incoming)}")
    assert (cursor == len(incoming)) is (not leading_nameless)


@pytest.mark.parametrize(("name_only", "gateway"), [("none", True), ("one", False)], ids=["ordinary", "cli"])
def test_cold_bind_controls(bind, name_only, gateway):
    transcript = Transcript(name_only)
    _seed(bind, transcript, gateway=gateway)
    counts = _bind_counts(bind, transcript, gateway=gateway)
    print(f"{'gateway' if gateway else 'CLI'} control: {counts}")
    assert counts == ZERO_BINDS


@pytest.mark.parametrize("write", ["append", "append_batch"])
def test_stored_row_bytes(tmp_path, monkeypatch, write):
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: START))
    with closing(store.MessageStore(tmp_path / "lcm.db", hermes_home=str(tmp_path))) as messages:
        rows = [
            {"role": "user", "name": "participant", "content": "question"},
            {"role": "tool", "tool_call_id": "call", "tool_name": TOOL, "name": OTHER_TOOL, "content": "café"},
        ]
        if write == "append":
            for row in rows:
                messages.append(SESSION, row, token_estimate=10)
        else:
            messages.append_batch(SESSION, rows, token_estimates=[10, 10])
    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
        conn.text_factory = bytes
        query = conn.execute("SELECT * FROM messages ORDER BY store_id")
        name_column = [column[0] for column in query.description].index("tool_name")
        result = query.fetchall()
        assert [row[name_column] for row in result] == [None, TOOL.encode()]
        digest = hashlib.sha256(repr(result).encode()).hexdigest()
        print(f"{write} stored-row SHA-256: {digest}")
