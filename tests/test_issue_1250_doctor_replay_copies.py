"""Doctor coverage for the cold-replay cells, controls, and scan budget."""

from contextlib import closing, contextmanager
from copy import deepcopy
import json
import sqlite3
from time import perf_counter

import pytest

from hermes_lcm import command, command_doctor, engine as engine_module, message_patterns, tools
from hermes_lcm.command_doctor import two_window_replay_copies_check
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.engine_helpers import _FakeTimeoutRegexEngine
from tests.harness import engine_context


S = "S-1250"
G = f"agent:main:local:dm:{S}"
C = "S2-1250"
CHECK = "two_window_replay_copies"
EARLY = [
    {"role": "user", "content": "Early GUI turn: project cobalt uses SQLite."},
    {"role": "assistant", "content": "Cobalt keeps authoritative history in SQLite."},
    {"role": "user", "content": "Early GUI turn: the deployment checkpoint is approved."},
    {"role": "assistant", "content": "Checkpoint approval recorded for cobalt."},
]
LATER = [
    {"role": "user", "content": "Desktop turn: add a nightly backup job."},
    {"role": "assistant", "content": "Nightly backup job added to the cobalt plan."},
]


def tool_group(*ids):
    return [
        {"role": "assistant", "content": "", "timestamp": 101, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for call_id in ids
        ]},
        *[
            {"role": "tool", "tool_call_id": call_id, "tool_name": "read",
             "content": f"Result for {call_id}.", "timestamp": 102 + i}
            for i, call_id in enumerate(ids)
        ],
    ]


@pytest.fixture
def cold_engine(tmp_path, monkeypatch):
    if message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Cobalt summary.", 1))
    with closing(sqlite3.connect(tmp_path / "state.db")) as conn, conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)", [(S, None, "compression"), (C, S, None)])

    @contextmanager
    def open_engine(conversation, *, fresh=0):
        engine = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"), fresh_tail_count=fresh, leaf_chunk_tokens=1,
            incremental_max_depth=0, temporal_rollups_enabled=False, embeddings_enabled=False,
        ))
        try:
            engine.on_session_start(S, platform="desktop" if conversation == G else "gui",
                                    conversation_id=conversation, context_length=200_000)
            yield engine
        finally:
            engine.shutdown()

    return open_engine


def stored(engine):
    return engine._store.get_session_messages(S)


def publish(engine, view):
    emitted = engine.compress(deepcopy(view), force=True)
    assert engine.last_compression_status == "compacted"
    assert engine._dag.get_session_nodes(S)
    return emitted


@pytest.mark.parametrize("mode", ("off", "shadow", "on"))
@pytest.mark.parametrize(("shape", "per_pass", "stored_per_pass"), [
    ("later", 3, [4, 4, 3, 3, 3, 3]),
    ("mixed", 1, [3, 2, 1, 1, 1, 1]),
])
def test_cold_replay_count(cold_engine, monkeypatch, mode, shape, per_pass, stored_per_pass):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    group = tool_group("target", "parallel") if shape == "mixed" else tool_group("target")
    group[-1].pop("timestamp")
    prompt = {"role": "user", "content": "Run the target tool.", "timestamp": 100}
    # Round 21/22 continuation setup, with G's leaf suppressed as in round 22.
    with cold_engine(S, fresh=1) as writer:
        writer.ingest(deepcopy(EARLY))
        emitted = publish(writer, EARLY)
        writer._append_off_current_session_end_suffix(
            S, [{"role": "user", "content": "An unrelated delegated occurrence."}],
            source="subagent", conversation_id=S,
        )
        writer.ingest(deepcopy([*emitted, prompt, *group[:-1]]))

    view = [*emitted, prompt, *group]
    with cold_engine(G, fresh=1) as gateway:
        gateway.ingest(deepcopy(view))
        if shape == "later":
            gateway._assemble_context({"role": "system", "content": "Continue the session."},
                                      deepcopy(view), persist=False)
            view.extend(LATER)
            gateway.ingest(deepcopy(view))
        assert two_window_replay_copies_check(gateway)["status"] == "pass"

    counts, extras, scan_ms = [], [], []
    for cold_pass, expected_stored in enumerate(stored_per_pass, start=1):
        with cold_engine(G, fresh=1) as gateway:
            before = stored(gateway)
            gateway.ingest(deepcopy(view))
            after = stored(gateway)
            assert after[:len(before)] == before
            added = after[len(before):]
            unstamped_ids = [row["store_id"] for row in added if row["observed_at"] is None]
            start = perf_counter()
            check = two_window_replay_copies_check(gateway)
            scan_ms.append((perf_counter() - start) * 1_000)
            assert check["status"] == "warn"
            detail = check["detail"]
            counts.append(detail["row_count"])
            extras.append(len(added))
            # The first two passes also copy stamped rows outside the doctor's count.
            assert len(added) == expected_stored
            assert len(unstamped_ids) == per_pass
            assert detail["row_count"] == cold_pass * per_pass
            assert "partial" not in detail
            assert detail["session_count"] == 1
            sample, = detail["sessions"]
            assert sample["session_id"] == S
            assert sample["store_ids"][-per_pass:] == unstamped_ids
    print(f"{shape}/{mode}: count={counts}; extra={extras}")
    print(f"{shape}/{mode}: scan_ms={[round(elapsed, 3) for elapsed in scan_ms]}")


def test_single_window_after_publication(cold_engine):
    with cold_engine(G) as engine:
        engine.ingest(deepcopy(EARLY))
        emitted = publish(engine, EARLY)
        group = tool_group("target")
        group[-1].pop("timestamp")
        view = [*emitted, {"role": "user", "content": "Run the target tool.", "timestamp": 100}, *group, *LATER]
        engine.ingest(deepcopy(view))
    for _ in range(6):
        with cold_engine(G) as engine:
            engine.ingest(deepcopy(view))
            assert two_window_replay_copies_check(engine) == empty_check()


def test_receipt_orphan(cold_engine):
    # Round 19 receipt/unstamped cell: one orphan result belongs to S, while G owns its call.
    prompt = {"role": "user", "content": "Run the requested tool."}
    call, result = tool_group("first")
    result["content"] = "r19reachmarker exact tool response."
    del call["timestamp"], result["timestamp"]
    with cold_engine(S) as original:
        original.ingest(deepcopy([prompt, call]))
    with cold_engine(G) as gateway:
        publish(gateway, [prompt, call])
        assert gateway._store.read_metadata_json(f"reowned_session_rows:{S}") == [
            row["store_id"] for row in stored(gateway)
        ]
    for _ in range(7):
        with cold_engine(S) as returning:
            returning.ingest(deepcopy([prompt, call, result]))
            rows = stored(returning)
            assert [(row["role"], row["conversation_id"]) for row in rows] == [
                ("user", G), ("assistant", G), ("tool", S),
            ]
            assert two_window_replay_copies_check(returning) == empty_check()


def empty_check():
    return {"check": CHECK, "status": "pass", "detail": {"row_count": 0, "session_count": 0, "sessions": []}}


def insert_rows(engine, rows):
    with engine._store.connection as conn:
        conn.executemany(
            "INSERT INTO messages (session_id, conversation_id, role, content, timestamp, observed_at) "
            "VALUES (?, ?, 'user', ?, 1700000000, ?)", rows,
        )


def test_genuine_unstamped_repeat(cold_engine):
    message = {"role": "user", "content": "Send another status update."}
    with cold_engine(S) as first:
        first.ingest([dict(message)])
    with cold_engine(G) as second:
        new_turn = [
            {"role": "user", "content": "A request made from the second window.", "timestamp": 100},
            {"role": "assistant", "content": "Request completed."},
            dict(message),
        ]
        second.ingest(deepcopy(new_turn))
        repeated = [row for row in stored(second) if row["content"] == message["content"]]
        assert [row["conversation_id"] for row in repeated] == [S, G]
        check = two_window_replay_copies_check(second)
        assert check["status"] == "warn"
        assert check["detail"]["row_count"] == 1
        assert check["detail"]["sessions"] == [{"session_id": S, "store_ids": [repeated[-1]["store_id"]]}]


@pytest.mark.parametrize("evidence", ("distinct", "stamp", "uid"))
def test_two_window_pass(tmp_path, evidence):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        insert_rows(engine, [
            (S, S, "Repeated", None),
            (S, G, "Distinct" if evidence == "distinct" else "Repeated", 100 if evidence == "stamp" else None),
        ])
        if evidence == "uid":
            engine._store.add_host_uid_bindings(S, [(2, "bound-uid", "canonical", "fixture")])
        assert two_window_replay_copies_check(engine) == empty_check()


@pytest.mark.parametrize("legacy_id", (S, f"  {S}  ", "   ", " \t\n "))
def test_no_scope_does_not_read_content(tmp_path, legacy_id):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        insert_rows(engine, [(S, S, "Repeated", None), (S, legacy_id, "Repeated", None)])
        conn = engine._store.connection
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            assert two_window_replay_copies_check(engine) == empty_check()
        finally:
            conn.set_trace_callback(None)
        assert len(statements) == 1
        assert "content" not in statements[0]


def budget_rows(total_rows):
    yield S, S, "Older pair", 100
    yield S, S, "Older pair", None
    for row in range(total_rows - 4):
        yield S, S, f"Distinct {row}", None
    yield S, G, "Newest pair", 100
    yield S, G, "Newest pair", None


@pytest.mark.parametrize("total_rows", (20_004, pytest.param(200_000, marks=pytest.mark.latency)))
def test_newest_rows_budget(tmp_path, total_rows):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        insert_rows(engine, budget_rows(total_rows))
        start = perf_counter()
        check = two_window_replay_copies_check(engine)
        elapsed = perf_counter() - start
        print(f"{total_rows} in-scope rows: check_s={elapsed:.6f}")

        assert check["status"] == "warn"
        detail = check["detail"]
        assert detail["partial"] is True
        assert detail["rows_scanned"] == 20_000
        assert detail["rows_in_scope"] == total_rows
        assert detail["row_count"] == 1
        assert detail["sessions"] == [{"session_id": S, "store_ids": [total_rows]}]
        assert "newest rows only and is a lower bound" in detail["message"]
        if total_rows == 200_000:
            assert elapsed < 2


@pytest.mark.parametrize("budget", (2, 3, 4))
def test_budget_shared_across_sessions(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(command_doctor, "TWO_WINDOW_REPLAY_SCAN_LIMIT", budget)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        insert_rows(engine, [
            ("first", S, "First", None), ("first", G, "Second", None),
            ("second", S, "Repeat", None), ("second", G, "Repeat", None),
        ])
        check = two_window_replay_copies_check(engine)
        detail = check["detail"]
        if budget < 4:
            assert check["status"] == "pass"
            assert detail["row_count"] == 0
            assert detail["partial"] is True
            assert detail["rows_scanned"] == budget
            assert detail["rows_in_scope"] == 4
            assert "newest rows only and is a lower bound" in command._doctor_text(engine)
        else:
            assert check["status"] == "warn"
            assert detail["row_count"] == 1
            assert "partial" not in detail


def test_doctor_output_and_caps(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        def json_check():
            return next(item for item in json.loads(tools.lcm_doctor({}, engine=engine))["checks"]
                        if item["check"] == CHECK)

        assert json_check() == empty_check()
        # These represent separate, genuine messages repeating text in each session.
        # A stamped predecessor still supplies the identity against which they match.
        for session in range(21):
            insert_rows(engine, [(f"s-{session:02d}", "first", "Repeated", 100), *[
                (f"s-{session:02d}", "second", "Repeated", None) for _ in range(51)
            ]])
        check = json_check()
        assert check["status"] == "warn"
        detail = check["detail"]
        assert (detail["row_count"], detail["session_count"]) == (21 * 51, 21)
        assert detail["sessions"] == [
            {"session_id": f"s-{i:02d}", "store_ids": list(range(i * 52 + 2, i * 52 + 52))}
            for i in range(20)
        ]
        assert "genuine new message" in detail["message"]
        assert "skills/hermes-lcm/references/session-lifecycle.md" in detail["message"]
        text = command._doctor_text(engine)
        assert f"{CHECK}: 1071 row(s) across 21 session(s). {detail['message']}" in text
        guidance = (
            "These rows are possible copies from two windows sharing one session. Nothing is lost. "
            "Keep one window per session until v0.31.0. The listed IDs are for a later repair."
        )
        assert f"- {CHECK}: inspect warning-only — {guidance}" in text
        assert f"- {CHECK}: inspect — inspect the reported detail" not in text
        conn = engine._store.connection
        conn.execute("PRAGMA query_only=ON")
        try:
            assert two_window_replay_copies_check(engine) == check
        finally:
            conn.execute("PRAGMA query_only=OFF")
