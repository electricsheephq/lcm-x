"""#2306: a failed pre-compaction store must not let the delegated child's compaction drop unstored rows.

``_compress_delegate_child_bypassed`` stores the child's view before the host compaction rewrites it. When that
store fails (for example ``database is locked`` after the busy timeout), compacting anyway drops rows from the view
that were never stored, and no later store sees them again. The fix skips that compaction pass; the next pass or
turn stores the full view first.
"""

import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, host  # noqa: F401  (fixture)

CHILD_SESSIONS = ("child", "continued")


def stored_child_contents(engine):
    marks = ",".join("?" for _ in CHILD_SESSIONS)
    return [row[0] for row in engine._store.connection.execute(
        f"SELECT content FROM messages WHERE session_id IN ({marks}) ORDER BY store_id", CHILD_SESSIONS)]


def payload_view(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "payload " * 400}
            for i in range(n)]


@pytest.mark.parametrize("failure", ["real-lock", "synthetic-locked"])
@pytest.mark.parametrize("rotate", [False, True], ids=["same-id", "rotated-id"])
def test_failed_pre_compaction_store_skips_compaction_and_loses_nothing(
        host, monkeypatch, caplog, failure, rotate):  # noqa: F811
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    view = payload_view(12)
    sent = [row["content"] for row in view]
    child.turn(ctx, view[:5], engine)
    assert stored_child_contents(engine) == sent[:5]

    calls = []
    locker = None
    if failure == "real-lock":
        # A writer on another connection holds the write lock during the first pre-store only; the child's
        # busy_timeout is lowered from production's 30 s so the test waits 0.2 s.
        engine._store.connection.execute("PRAGMA busy_timeout=200")
        locker = sqlite3.connect(str(engine._store.db_path), timeout=0, isolation_level=None)
        real_store = engine._store_delegate_child_turn

        def pre_store_under_lock(*args, **kwargs):
            calls.append(args)
            if len(calls) != 1:
                return real_store(*args, **kwargs)
            locker.execute("BEGIN IMMEDIATE")
            try:
                return real_store(*args, **kwargs)
            finally:
                locker.execute("ROLLBACK")

        monkeypatch.setattr(engine, "_store_delegate_child_turn", pre_store_under_lock)
    else:
        batch = engine._store._append_protected_batch

        def locked_once(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return batch(*args, **kwargs)

        monkeypatch.setattr(engine._store, "_append_protected_batch", locked_once)

    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    monkeypatch.setattr(engine, "_bypass_compaction_target_tokens", lambda **kwargs: 64)
    aux = module("aux_session")
    fallbacks = aux.delegate_child_store_counts().get("fallbacks", 0)
    with caplog.at_level("WARNING"):
        first = child.compress(engine, view)
    assert calls, "the injected failure never ran"
    assert first == view, "compaction ran although its input was not stored"
    assert aux.delegate_child_store_counts()["fallbacks"] > fallbacks
    assert any("compaction skipped: input not stored" in record.getMessage() for record in caplog.records)
    assert stored_child_contents(engine) == sent[:5], "the failed pre-store wrote rows after all"
    if locker is not None:
        locker.close()

    # The next compaction pass stores the full view first, then compacts.
    compacted = child.compress(engine, first)
    assert len(compacted) < len(view)
    assert stored_child_contents(engine) == sent

    if rotate:
        child.rotate(engine, "continued")
    for i in range(2):
        compacted = [*compacted, {"role": "assistant", "content": f"after compression {i}"}]
        child.turn(ctx, compacted, engine)
        child.turn(ctx, compacted, engine)
    child.end(engine, compacted)

    contents = stored_child_contents(engine)
    missing = [i for i, text in enumerate(sent) if text not in contents]
    assert not missing, f"child rows never stored: turn indices {missing} of {len(sent)}"
    assert all(contents.count(text) == 1 for text in sent)
    assert contents.count("after compression 0") == 1 and contents.count("after compression 1") == 1
