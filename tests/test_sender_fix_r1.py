"""FIX-R1 regressions ported from the binding, registry and write-path probes."""
from copy import deepcopy
import sqlite3
import threading
import time

import pytest

from tests.harness import engine_context
from tests.test_sender_capture import (
    begin, capture_module, clean_capture_registry, expected, hook_context,
    message, payload, senders,
)

__all__ = ["clean_capture_registry", "hook_context"]
A_TEXT, B_TEXT = "synthetic tail of A", "synthetic request of B"


def merge_v0215(view):
    """Uid-less Hermes pass 3 folds into the earlier dict without a witness."""
    merged = []
    for row in view:
        if merged and merged[-1]["role"] == row["role"] == "user":
            left, right = merged[-1]["content"], row["content"]
            merged[-1]["content"] = left + "\n\n" + right if left and right else left or right
        else:
            merged.append(row)
    view[:] = merged


def assert_owners(e):
    for content, sender, source in e._store.connection.execute(
            "SELECT content, sender_id, sender_source FROM messages WHERE role='user'"):
        if content == A_TEXT:
            assert (sender, source) in {(None, None), ("telegram:synthetic-x", "gateway_principal")}
        elif content == B_TEXT:
            assert (sender, source) in {(None, None), ("telegram:synthetic-y", "gateway_principal")}


@pytest.mark.parametrize("captured", [False, True], ids=["no-capture", "captured-x"])
@pytest.mark.parametrize("path", ["compress-copy", "live-inplace"])
def test_pre_hook_merge_preserves_stored_owner(tmp_path, captured, path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        prefix = [message(), dict(role="assistant", content="synthetic reply", timestamp=100.5)]
        a = dict(role="user", content=A_TEXT, timestamp=101.0)
        e.ingest([*prefix, a])
        assert e.drain_rollup_maintenance(timeout=10)
        if captured:
            begin(e, [*prefix, a], sender_id="synthetic-x")
            assert senders(e)[-1] == ("telegram:synthetic-x", "gateway_principal")
        e.on_turn_complete([*prefix, a], failed=True, turn_id="turn-a")
        view = [*prefix, a, dict(role="user", content=B_TEXT, timestamp=102.0)]
        e.ingest(view)
        if path == "compress-copy":
            out = deepcopy(view)
            e._rekey_host_rewrite_watch(view, out)
            view = out
        merge_v0215(view)
        assert view[-1]["content"] == A_TEXT + "\n\n" + B_TEXT
        begin(e, view, turn="turn-b", sender_id="synthetic-y")
        assert_owners(e)
        e.ingest(view)
        assert_owners(e)


@pytest.mark.parametrize("variant", ["same_obj", "gw_fresh_stored_at_a", "gw_fresh_unstored_at_a"])
def test_pre_hook_merge_gateway_shapes(tmp_path, variant):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        prefix = [message(), dict(role="assistant", content="synthetic reply", timestamp=100.5)]
        a = dict(role="user", content=A_TEXT, timestamp=101.0)
        e.ingest(prefix if variant == "gw_fresh_unstored_at_a" else [*prefix, a])
        e.on_turn_complete([*prefix, a], failed=True, turn_id="turn-a")
        view = [*prefix, a] if variant == "same_obj" else deepcopy([*prefix, a])
        view.append(dict(role="user", content=B_TEXT, timestamp=102.0))
        e.ingest(view)
        assert e.drain_rollup_maintenance(timeout=10)
        out = deepcopy(view)
        e._rekey_host_rewrite_watch(view, out)
        merge_v0215(out)
        begin(e, out, turn="turn-b", sender_id="synthetic-y")
        assert_owners(e)
        e.ingest(out)
        assert_owners(e)


def test_pre_hook_merge_real_compress(tmp_path, monkeypatch):
    import hermes_lcm.engine as engine_module
    from hermes_lcm.engine import LCMEngine
    from tests.test_compression_boundary import _config, _stub_summarizer

    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    e = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        e.on_session_start("S", platform="telegram", context_length=200000, conversation_id="conv")
        view = []
        for i in range(12):
            view.extend([dict(role="user", content=f"[T{i:02d}] synthetic " + "alpha beta " * 40,
                              timestamp=10.0 + i),
                         dict(role="assistant", content=f"synthetic reply {i}", timestamp=10.5 + i)])
        view.append(dict(role="user", content=A_TEXT, timestamp=101.0))
        e.should_compress_preflight(view)
        e.on_turn_complete(list(view), failed=True, turn_id="turn-a")
        view.append(dict(role="user", content=B_TEXT, timestamp=102.0))
        e.should_compress_preflight(view)
        out = e.compress(list(view), force=True)
        a = next(row for row in out if row.get("content") == A_TEXT)
        assert any(any(obj is a for obj in entry[2]) for _, entry in capture_module()._watch(e))
        assert e.drain_rollup_maintenance(timeout=10)
        merge_v0215(out)
        begin(e, out, turn="turn-b", sender_id="synthetic-y")
        assert_owners(e)
    finally:
        e.shutdown()


def test_registered_todo_fold_still_updates(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row = message()
        e.ingest([row])
        assert e.drain_rollup_maintenance(timeout=10)
        copy = deepcopy(row)
        e._rekey_host_rewrite_watch([row], [copy])
        copy["content"] += ("\n\n[Your active task list was preserved across context compression]"
                            "\n- [ ] synthetic task")
        begin(e, [copy])
        assert senders(e) == [expected()]


def make_state(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                         [("S", "ancestor", None), ("ancestor", None, "compression")])


@pytest.mark.parametrize("anchor", ["0", "1"], ids=["anchor-off", "failed-preflight-read"])
def test_hook_cold_cache_never_opens_locked_host_db(hook_context, monkeypatch, anchor, record_property):
    ctx, view = hook_context, []
    e = ctx.engine
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    for i in range(14):
        view.extend([message(i), dict(role="assistant", content="synthetic reply", timestamp=100.5 + i)])
    view.append(message(20))
    state = e._state_db_path()
    make_state(state)
    e._identity_anchor_chain_cache = None
    holder = sqlite3.connect(state)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        e.should_compress_preflight(view)
        assert getattr(e, "_identity_anchor_chain_cache", None) is None
        real, calls = sqlite3.connect, []

        def counted(path, *args, **kwargs):
            if str(state) in str(path):
                calls.append("host-state-connect")
            return real(path, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", counted)
        start = time.monotonic()
        ctx.hooks["pre_llm_call"](**payload(view))
        elapsed = time.monotonic() - start
        record_property("fix_r1_hook", {"anchor": anchor, "elapsed_ms": elapsed * 1000,
                                         "host_connects": len(calls)})
        assert elapsed < .05
        assert not calls
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.parametrize("cache", ["warm", "cold", "stale"])
def test_update_ancestor_scope_uses_only_current_cache(tmp_path, cache):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row = message()
        e.ingest([row])
        assert e.drain_rollup_maintenance(timeout=10)
        state = e._state_db_path()
        make_state(state)
        e._identity_anchor_chain_cache = None
        if cache == "warm":
            assert e._identity_anchor_chain() == ["ancestor"]
        elif cache == "stale":
            e._identity_anchor_chain_cache = ("other-session", ["ancestor"])
        e._store.connection.execute("UPDATE messages SET session_id='ancestor'")
        e._store.connection.commit()
        begin(e, [row])
        assert senders(e) == [expected() if cache == "warm" else (None, None)]


def test_resolved_update_does_not_stamp_second_same_uid_insert(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row = message(message_uid="e" * 32)
        e.ingest([row])
        assert e.drain_rollup_maintenance(timeout=10)
        pending = begin(e, [row])
        assert pending.store_ids and senders(e) == [expected()]
        copy = deepcopy(row)
        e.ingest([row, dict(role="assistant", content="synthetic reply"), copy])
        assert len(senders(e)) == 3
        assert senders(e) == [expected(), (None, None), (None, None)]


@pytest.mark.parametrize("exists", [False, True], ids=["no-row", "updated-row"])
def test_restore_error_keeps_result_and_releases_lock(tmp_path, caplog, exists):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        e.ingest([message()])
        assert e.drain_rollup_maintenance(timeout=10)
        store, acquired = e._store, []
        conn = store.connection
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout != 0

        class RestoreError:
            def __getattr__(self, name):
                return getattr(conn, name)

            def execute(self, sql, *args):
                if sql == f"PRAGMA busy_timeout={timeout}":
                    raise sqlite3.OperationalError("synthetic restore failure")
                return conn.execute(sql, *args)

        store._conn = RestoreError()
        try:
            with caplog.at_level("DEBUG"):
                result = store.set_sender_if_null(1 if exists else 99999, *expected(), ["S"])
            assert result == int(exists)
            assert "sender_capture:restore_error" in caplog.messages
        finally:
            store._conn = conn
            conn.execute(f"PRAGMA busy_timeout={timeout}")

            def acquire():
                ok = store._write_lock.acquire(blocking=False)
                acquired.append(ok)
                if ok:
                    store._write_lock.release()

            thread = threading.Thread(target=acquire)
            thread.start()
            thread.join(2)
            if acquired == [False]:  # Restore baseline's leaked RLock for deterministic teardown.
                store._write_lock.release()
            assert acquired == [True]


@pytest.mark.parametrize("retired_during_identity", [False, True], ids=["late-worker", "retired-during-create"])
def test_retired_refusal_has_debug_reason(tmp_path, monkeypatch, caplog, retired_during_identity):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row, cap = message(), capture_module()
        if retired_during_identity:
            original = e._message_replay_identity

            def retire_then_identity(*args, **kwargs):
                cap.retire(e, "turn-a")
                return original(*args, **kwargs)

            monkeypatch.setattr(e, "_message_replay_identity", retire_then_identity)
        else:
            begin(e, [row])
            e.on_turn_complete([row], turn_id="turn-a")
        with caplog.at_level("DEBUG"):
            assert begin(e, [row]) is None
        assert "sender_capture:refused_retired" in caplog.messages


@pytest.mark.parametrize("anchor", ["0", "1"])
@pytest.mark.parametrize("path", ["live-inplace", "compress-copy",
                                 "gw_fresh_stored_at_a", "gw_fresh_unstored_at_a"])
def test_w6_empty_b_own_probe(tmp_path, monkeypatch, anchor, path):
    """Port the claimant's uid-less and gateway empty-B probe families."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        prefix = [message(), dict(role="assistant", content="synthetic reply", timestamp=100.5)]
        a = dict(role="user", content=A_TEXT, timestamp=101.0)
        e.ingest(prefix if path == "gw_fresh_unstored_at_a" else [*prefix, a])
        e.on_turn_complete([*prefix, a], failed=True, turn_id="turn-a")
        view = deepcopy([*prefix, a]) if path.startswith("gw_") else [*prefix, a]
        view.append(dict(role="user", content="", timestamp=102.0))
        e.ingest(view)
        assert e.drain_rollup_maintenance(timeout=10)
        if path != "live-inplace":
            out = deepcopy(view)
            e._rekey_host_rewrite_watch(view, out)
            view = out
        merge_v0215(view)
        assert view[-1]["content"] == A_TEXT
        begin(e, view, turn="turn-b", sender_id="synthetic-y")
        assert_owners(e)
        e.ingest(view)
        assert e.drain_rollup_maintenance(timeout=10)
        assert_owners(e)


@pytest.mark.parametrize("anchor", ["0", "1"])
@pytest.mark.parametrize("path", ["live-inplace", "compress-copy"])
@pytest.mark.parametrize("case", ["empty-b", "nonempty-b-control", "empty-b-a-prestamped"])
def test_w6_empty_b_verifier_probe(tmp_path, monkeypatch, anchor, path, case):
    """Port the independent verifier, retaining its nonempty/prestamped controls."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        prefix = [message(), dict(role="assistant", content="synthetic reply", timestamp=100.5)]
        a = dict(role="user", content=A_TEXT, timestamp=101.0)
        if case.endswith("prestamped"):
            begin(e, [*prefix, a], turn="turn-a", sender_id="synthetic-x")
        e.ingest([*prefix, a])
        assert e.drain_rollup_maintenance(timeout=10)
        e.on_turn_complete([*prefix, a], failed=True, turn_id="turn-a")
        b_text = B_TEXT if case == "nonempty-b-control" else ""
        view = [*prefix, a, dict(role="user", content=b_text, timestamp=102.0)]
        e.ingest(view)
        assert e.drain_rollup_maintenance(timeout=10)
        if path == "compress-copy":
            out = deepcopy(view)
            e._rekey_host_rewrite_watch(view, out)
            view = out
        merge_v0215(view)
        begin(e, view, turn="turn-b", sender_id="synthetic-y")
        assert_owners(e)
        e.ingest(view)
        assert e.drain_rollup_maintenance(timeout=10)
        assert_owners(e)


@pytest.mark.parametrize("anchor", ["0", "1"])
@pytest.mark.parametrize("candidate_session", ["S", "ancestor"])
@pytest.mark.parametrize("later", ["none", "user", "assistant", "tool", "other-session-user"])
def test_w6_update_latest_user_controls(tmp_path, monkeypatch, anchor, candidate_session, later):
    """Only a later user in the candidate row's own session vetoes UPDATE."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row = message()
        e.ingest([row])
        assert e.drain_rollup_maintenance(timeout=10)
        store = e._store
        sid = store.connection.execute("SELECT store_id FROM messages").fetchone()[0]
        if candidate_session == "ancestor":
            store.connection.execute("UPDATE messages SET session_id='ancestor' WHERE store_id=?", (sid,))
            store.connection.commit()
            e._identity_anchor_chain_cache = ("S", ["ancestor"])
        if later != "none":
            role = "user" if later == "other-session-user" else later
            session = "other" if later == "other-session-user" else candidate_session
            later_id = store.append(session, dict(role=role, content="synthetic later row"))
            assert later_id > sid
        pending = begin(e, [row])
        assert store.get_message_senders([sid])[sid] == ((None, None) if later == "user" else expected())
        assert bool(pending.store_ids) == (later != "user")
