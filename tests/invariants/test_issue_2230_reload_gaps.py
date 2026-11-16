"""I1/I2: retained foreground callbacks remain durable across plugin reload gaps."""
import sqlite3
import time
from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.invariants.test_issue_2152_lazy_storage import preflight as preflight

from tests.harness import ScriptedSummarizer, pin_char_counter
from tests.test_issue_1385_r5_host_merge_collapse import _identity
from hermes_lcm.store import MessageStore


def unload(ctx):
    for callback in ctx.unload_callbacks:
        callback()
    ctx._manager._context_engine = None


def durable(home, session):
    with closing(sqlite3.connect(home / "lcm.db")) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY store_id", (session,))]


@pytest.mark.parametrize("timing", ["gap", "draining", "ready", "second-gap", "second-draining"])
def test_empty_reset_persists_all_turns(preflight, timing):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("empty", platform="cli")
    host_path = preflight.home / "state.db"
    with closing(sqlite3.connect(host_path)) as db, db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, end_reason TEXT)")
        db.execute("CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, timestamp REAL, message_uid TEXT)")
    if timing.startswith("second-"):
        unload(old)
        prototype, old = preflight.register(supports_unload=True, manager=old._manager)
        retained.should_compress_preflight([])
        assert retained._reload_successor is not None
    if timing.endswith("draining"):
        prototype._unload_group.closing.set()
        old._manager._context_engine = None
    else:
        unload(old)
    current = None
    if timing == "ready":
        _, current = preflight.register(supports_unload=True, manager=old._manager)
    # Actual CLI ordering: engine reset and bind precede native create_session and hook.
    retained.on_session_reset()
    retained.bind_session_state(SimpleNamespace(db_path=host_path), "fresh")
    with closing(sqlite3.connect(host_path)) as db, db:
        db.execute("INSERT INTO sessions VALUES ('fresh', 'cli', NULL, ?, NULL, NULL)", (time.time(),))
    ctx = current or old
    if not timing.endswith("gap") and "on_session_reset" in ctx.hooks:
        ctx.hooks["on_session_reset"](session_id="fresh", platform="cli", reason="new_session")
    view = [{"role": "user", "content": "first question", "timestamp": 1001.0},
            {"role": "assistant", "content": "partial reply", "timestamp": 1002.0}]
    retained.should_compress_preflight(deepcopy(view[:1]))
    retained.on_turn_complete(deepcopy(view), interrupted=True)
    snapshots = [durable(preflight.home, "fresh")]
    if current is None:
        if timing.endswith("draining"):
            unload(old)
        _, current = preflight.register(supports_unload=True, manager=old._manager)
    for n in range(2):
        view += [{"role": "user", "content": f"next question {n}", "timestamp": 1003.0 + 2*n},
                 {"role": "assistant", "content": f"next answer {n}", "timestamp": 1004.0 + 2*n}]
        with closing(sqlite3.connect(host_path)) as db, db:
            db.execute("INSERT INTO messages VALUES ('fresh', 'assistant', ?, ?, NULL)", (view[-1]["content"], view[-1]["timestamp"]))
        current.hooks["post_llm_call"](session_id="fresh", platform="cli", conversation_history=deepcopy(view))
        snapshots.append(durable(preflight.home, "fresh"))
    retained.on_session_end("fresh", deepcopy(view))
    snapshots.append(durable(preflight.home, "fresh"))
    unload(current)
    assert [len(x) for x in snapshots] == [2, 4, 6, 6]
    for rows, end in zip(snapshots, (2, 4, 6, 6)):
        assert list(map(_identity, rows)) == list(map(_identity, view[:end]))


@pytest.mark.parametrize("case", [
    "no-reset", "wrong-native-path", "preexisting", "child", "ended",
    "superseded", "old-nonempty", "advanced-lane", "stateless", "auxiliary",
    "late-create", "busy-at-bind",
])
def test_direct_reset_confirmation_still_requires_a_fresh_foreground(preflight, monkeypatch, case):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("empty", platform="cli")
    if case == "old-nonempty":
        retained.ingest([{"role": "user", "content": "must survive"}])
    before = durable(preflight.home, "empty")
    host_path = preflight.home / "state.db"
    with closing(sqlite3.connect(host_path)) as native:
        native.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, "
                       "started_at REAL, ended_at REAL, end_reason TEXT)")

        def create():
            with native:
                native.execute("INSERT INTO sessions VALUES ('fresh', 'cli', ?, ?, ?, NULL)",
                               ("parent" if case == "child" else None, time.time(),
                                time.time() if case == "ended" else None))

        if case == "preexisting":
            create()
        unload(old)
        with native:
            if case == "stateless":
                monkeypatch.setattr(retained, "_thread_context_stateless", lambda: True)
            if case == "auxiliary":
                monkeypatch.setattr(retained, "_thread_context_has_auxiliary_session", lambda _: True)
            if case != "no-reset":
                retained.on_session_reset()
            if case == "busy-at-bind":
                native.execute("BEGIN EXCLUSIVE")
            retained.bind_session_state(SimpleNamespace(
                db_path=host_path.parent / "other.db" if case == "wrong-native-path" else host_path), "fresh")
        if case == "superseded":
            retained.bind_session_state(SimpleNamespace(db_path=host_path), "newer")
        if case == "advanced-lane":
            with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
                db.execute("UPDATE lcm_lifecycle_state SET current_session_id='other' "
                           "WHERE current_session_id='empty'")
        if case not in ("preexisting", "late-create"):
            create()
        view = [{"role": "user", "content": "fresh question"}, {"role": "assistant", "content": "partial reply"}]
        if case == "late-create":
            retained.on_turn_complete(deepcopy(view), interrupted=True)
            assert durable(preflight.home, "fresh") == []
            create()
        for _ in range(2):
            retained.should_compress_preflight(deepcopy(view[:1]))
            retained.on_turn_complete(deepcopy(view), interrupted=True)
        expected = view if case in ("late-create", "busy-at-bind") else []
        assert list(map(_identity, durable(preflight.home, "fresh"))) == list(map(_identity, expected))
        assert durable(preflight.home, "empty") == before


@pytest.mark.parametrize("shape", ["retry", "orphan-prune", "unchanged"])
@pytest.mark.parametrize("operation", ["preflight", "compress"])
@pytest.mark.parametrize("form", ["floor", "r34", "upstream-uid"])
@pytest.mark.parametrize("warm_prefix", [False, True], ids=["first-turn", "later-turn"])
@pytest.mark.parametrize("gap", [1, 2], ids=["first-unload", "second-unload"])
def test_repaired_view_survives_unloaded(preflight, monkeypatch, shape, operation, form, warm_prefix, gap):
    pin_char_counter(monkeypatch)
    ScriptedSummarizer("healthy").install(monkeypatch)
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    sequence = 0

    def stamp(role, content, **extra):
        nonlocal sequence
        sequence += 1
        row = dict(role=role, content=content, timestamp=float(1000 + sequence), **extra)
        if form == "upstream-uid":
            row["message_uid"] = f"uid-{sequence}"
        return row

    warm = [stamp("user", "warm question"), stamp("assistant", "warm answer")] if warm_prefix else []
    stopped = [stamp("user", "stopped request"), stamp("assistant", "", tool_calls=[{
        "id": "call-stopped", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"fixture.txt"}'}}])]
    if shape in ("retry", "unchanged"):
        stopped += [stamp("tool", "fixture result", tool_call_id="call-stopped"),
                    stamp("assistant", "Operation interrupted.")]
    history = warm + stopped
    retained.ingest(deepcopy(warm or stopped[:1]))
    retained.on_turn_complete(deepcopy(history), interrupted=True)
    assert list(map(_identity, durable(preflight.home, "S"))) == list(map(_identity, history))
    unload(old)
    if gap == 2:
        _, old = preflight.register(supports_unload=True, manager=old._manager)
        retained.should_compress_preflight(deepcopy(history))
        assert retained._reload_successor is not None
        unload(old)
    next_user = stamp("user", "stopped request" if shape == "retry" else "next request")
    answer = stamp("assistant", "next stopped answer")
    if shape == "orphan-prune":
        merged = deepcopy(stopped[0])
        merged["content"] += "\n\n" + next_user["content"]
        if form == "r34":
            merged["_merged_turn_prefix"] = stopped[0]["content"] + "\n\n"
        elif form == "upstream-uid":
            merged["_merged_turn_prefix"] = stopped[0]["content"]
            merged["_absorbed_message_uids"] = [next_user["message_uid"]]
        current = warm + [merged]
    elif shape == "retry":
        current = warm + [next_user]
    else:
        current = history + [next_user]
    if operation == "preflight":
        retained.should_compress_preflight(deepcopy(current))
    else:
        retained.compress(deepcopy(current), force=True)
    snapshots = [durable(preflight.home, "S")]
    current += [answer]
    retained.on_turn_complete(deepcopy(current), interrupted=True)
    snapshots.append(durable(preflight.home, "S"))
    retained.on_turn_complete(deepcopy(current), interrupted=True)
    retained.on_session_end("S", deepcopy(current))
    snapshots.append(durable(preflight.home, "S"))
    expected = [history + [next_user], history + [next_user, answer], history + [next_user, answer]]
    for rows, want in zip(snapshots, expected):
        assert list(map(_identity, rows)) == list(map(_identity, want))
    # A live successor must reconcile the detached writer's receipt, including
    # after repeated callbacks; count, content and order all remain exact.
    _, current_ctx = preflight.register(supports_unload=True, manager=old._manager)
    after = [stamp("user", "after reload"), stamp("assistant", "completed reply")]
    current += after
    try:
        for _ in range(2):
            current_ctx.hooks["post_llm_call"](
                session_id="S", platform="cli", conversation_history=deepcopy(current),
            )
        retained.on_session_end("S", deepcopy(current))
        assert list(map(_identity, durable(preflight.home, "S"))) == list(map(_identity, expected[-1] + after))
    finally:
        unload(current_ctx)


@pytest.mark.parametrize("fault", ["receipt", "advanced-lane", "mixed-lane", "unrelated", "unstamped"])
def test_retired_repair_is_atomic_and_lane_scoped(preflight, monkeypatch, fault):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    history = [dict(role=role, content=content, timestamp=float(i + 1)) for i, (role, content) in enumerate([
        ("user", "opening"), ("assistant", "reply"), ("user", "retry me"), ("assistant", "stopped"),
    ])]
    retained.ingest(deepcopy(history))
    unload(old)
    current = history[:2] + [dict(role="user", content="retry me", timestamp=5.0)]
    if fault == "advanced-lane":
        with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
            db.execute("UPDATE lcm_lifecycle_state SET current_session_id='newer' WHERE current_session_id='S'")
    if fault == "mixed-lane":
        with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
            db.execute("UPDATE messages SET conversation_id='foreign' WHERE role='assistant'")
    if fault == "unrelated":
        current[0] = dict(current[0], content="someone else's opening")
    if fault == "unstamped":
        current = [{k: v for k, v in row.items() if k != "timestamp"} for row in current]
    before = durable(preflight.home, "S")
    fired = []
    write = MessageStore.write_metadata_json

    def fail_receipt(store, keys, value):
        result = write(store, keys, value)
        if any(key.startswith("retired_tail:") for key in keys):
            fired.append(True)
            raise sqlite3.OperationalError("injected receipt failure")
        return result

    with monkeypatch.context() as patch:
        if fault == "receipt":
            patch.setattr(MessageStore, "write_metadata_json", fail_receipt)
        retained.should_compress_preflight(deepcopy(current))
    assert durable(preflight.home, "S") == before
    if fault == "receipt":
        assert fired
        with closing(sqlite3.connect(preflight.home / "lcm.db")) as db:
            assert db.execute("SELECT value FROM metadata WHERE key LIKE 'retired_tail:%'").fetchall() == []
        for _ in range(2):
            retained.should_compress_preflight(deepcopy(current))
        assert list(map(_identity, durable(preflight.home, "S"))) == list(map(_identity, history + current[-1:]))
