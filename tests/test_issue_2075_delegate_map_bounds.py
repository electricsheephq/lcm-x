"""#2075: digest boundaries and confirmed-link cleanup preserve delegated rows."""

import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import (
    DELEGATE,
    Agent,
    child_rows,
    child_view,
    host as host,
    lineage_rows,
)


def _parent(host):
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    return ctx, module("aux_session"), engine


def _assert_rows(engine, session_id, expected):
    def identities(rows):
        return [engine._session_end_prefix_compare_identity(row, session_id=session_id) for row in rows]

    assert identities(child_rows(engine, session_id)) == identities(expected)


def test_rows_unchanged_at_scale_and_boundaries_hold_only_digests(host, monkeypatch):
    ctx, aux, engine = _parent(host)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda messages, *args: [
        {"role": "user", "content": "compressed head"}, messages[-1]])
    expected = {}
    auxiliary_ids = []
    for index in range(21):
        session_id = f"delegate-{index}"
        child = Agent(session_id, **DELEGATE)
        child.start(engine)
        original = child_view(2)
        compacted = child.compress(engine, original)
        later = {"role": "assistant", "content": f"later answer {index}"}
        for _ in range(2):
            child.turn(ctx, [*compacted, later], engine)
        expected[session_id] = [*original, later]

        auxiliary_id = f"auxiliary-{index}"
        auxiliary_ids.append(auxiliary_id)
        ctx.hooks["subagent_start"][0](child_session_id=auxiliary_id, parent_session_id="parent")
        markers = ({"_memory_write_origin": "background_review"} if index % 2 else
                   {"enabled_toolsets": ["memory", "skills"]})
        auxiliary = Agent(auxiliary_id, **{**DELEGATE, **markers})
        auxiliary.start(engine)
        auxiliary.turn(ctx, child_view(1), engine)

    for session_id, rows in expected.items():
        _assert_rows(engine, session_id, rows)
    for session_id in auxiliary_ids:
        engine.on_session_end(session_id, child_view(2))  # no Agent frame; markers must survive
        assert child_rows(engine, session_id) == []
    with aux._DELEGATE_STORE_LOCK:
        assert all(engine._delegate_child_key(sid) in aux._DELEGATE_AUXILIARY_IDS for sid in auxiliary_ids)
        boundaries = list(aux._DELEGATE_RESUME_BOUNDARIES.values())
    assert len(boundaries) == 21
    assert all(boundary for boundary in boundaries)
    assert all(isinstance(digest, bytes) and len(digest) == 32 for boundary in boundaries for digest in boundary)
    contents = {row["content"].encode() for rows in expected.values() for row in rows}
    assert all(digest not in contents for boundary in boundaries for digest in boundary)


@pytest.mark.parametrize("repetitions", [2, 4], ids=["short-boundary", "repeated-tail"])
def test_repeated_row_resume_appends_each_new_occurrence_once(host, monkeypatch, repetitions):
    ctx, _, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    user = {"role": "user", "content": "U"}
    answer = {"role": "assistant", "content": "A"}
    original = [user, *([answer] * repetitions)]
    kept = 1 if repetitions == 2 else repetitions
    compacted = [{"role": "user", "content": "C"}, *([answer] * kept)]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    assert child.compress(engine, original) == compacted
    resumed = [*compacted, answer]
    for _ in range(2):
        child.turn(ctx, resumed, engine)
    _assert_rows(engine, "child", [*original, answer])
    later = {"role": "user", "content": "later request"}
    for _ in range(2):
        child.turn(ctx, [*resumed, later], engine)
    _assert_rows(engine, "child", [*original, answer, later])


def test_counted_fallback_preserves_rows_and_occurrence_counts(host):
    ctx, aux, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    original = [{"role": "user", "content": "old head"},
                {"role": "assistant", "content": "repeat"},
                {"role": "user", "content": "repeat"},
                {"role": "assistant", "content": "repeat"}]
    child.turn(ctx, original, engine)
    view = [*(original[-3:] * 2), {"role": "user", "content": "new turn"}]
    before = aux.delegate_child_store_counts().get("fallbacks", 0)
    child.turn(ctx, view, engine)
    assert aux.delegate_child_store_counts()["fallbacks"] == before + 1
    _assert_rows(engine, "child", [*original, *original[-3:], view[-1]])
    child.turn(ctx, view, engine)
    _assert_rows(engine, "child", [*original, *original[-3:], view[-1]])


def test_compression_parent_write_failure_keeps_empty_continuation_chain(host, monkeypatch):
    ctx, aux, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(2)
    child.turn(ctx, original, engine)
    record = engine._store.record_session_lineage
    failures = []

    def fail_compression(session_id, *args, **kwargs):
        if session_id == "continued" and kwargs.get("source") == "compression":
            failures.append(session_id)
            raise sqlite3.OperationalError("synthetic compression link failure")
        return record(session_id, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "record_session_lineage", fail_compression)
        child.rotate(engine, "continued")
    assert failures == ["continued"]
    with aux._DELEGATE_STORE_LOCK:
        assert aux._DELEGATE_COMPRESSION_PARENTS[engine._delegate_child_key("continued")] == "child"
    assert engine._stored_delegate_child_link("continued") == {}
    assert child_rows(engine, "continued") == []
    later = {"role": "assistant", "content": "continued answer"}
    for _ in range(2):
        child.turn(ctx, [*original, later], engine)
    _assert_rows(engine, "child", original)
    _assert_rows(engine, "continued", [later])
    assert ("continued", "child", "compression") in lineage_rows(engine)


def test_confirmed_compression_parent_removed_and_database_chain_resumes(host):
    ctx, aux, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(2)
    child.turn(ctx, original, engine)
    child.rotate(engine, "continued")
    assert ("continued", "child", "compression") in lineage_rows(engine)
    with aux._DELEGATE_STORE_LOCK:
        assert engine._delegate_child_key("continued") not in aux._DELEGATE_COMPRESSION_PARENTS
        assert engine._delegate_child_store_sessions("continued") == ["continued", "child"]
    later = {"role": "assistant", "content": "continued answer"}
    for _ in range(2):
        child.turn(ctx, [*original, later], engine)
    _assert_rows(engine, "child", original)
    _assert_rows(engine, "continued", [later])


def test_conflicting_stored_lineage_keeps_compression_parent(host):
    ctx, aux, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(2)
    child.turn(ctx, original, engine)
    engine._store.record_session_lineage("continued", "parent", source="subagent_start")
    child.rotate(engine, "continued")
    with aux._DELEGATE_STORE_LOCK:
        assert aux._DELEGATE_COMPRESSION_PARENTS[engine._delegate_child_key("continued")] == "child"
        assert engine._delegate_child_store_sessions("continued") == ["continued", "child"]
    assert ("continued", "parent", "subagent_start") in lineage_rows(engine)
    later = {"role": "assistant", "content": "continued answer"}
    for _ in range(2):
        child.turn(ctx, [*original, later], engine)
    _assert_rows(engine, "child", original)
    _assert_rows(engine, "continued", [later])


def test_compression_cleanup_read_failure_keeps_entry_without_counting_fallback(host, monkeypatch):
    ctx, aux, engine = _parent(host)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(2)
    child.turn(ctx, original, engine)
    read = engine._stored_delegate_child_link
    cleanup_reads = []

    def fail_new_session_cleanup(session_id):
        if session_id == "continued":
            cleanup_reads.append(session_id)
            raise sqlite3.OperationalError("synthetic cleanup read failure")
        return read(session_id)

    before = aux.delegate_child_store_counts().get("fallbacks", 0)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_stored_delegate_child_link", fail_new_session_cleanup)
        child.rotate(engine, "continued")
    # The base has no cleanup read: its pass is a preservation lock-in only.
    if hasattr(engine, "_delegate_child_boundary_digest"):
        assert cleanup_reads == ["continued"]
    assert aux.delegate_child_store_counts().get("fallbacks", 0) == before
    with aux._DELEGATE_STORE_LOCK:
        assert aux._DELEGATE_COMPRESSION_PARENTS[engine._delegate_child_key("continued")] == "child"
    assert ("continued", "child", "compression") in lineage_rows(engine)
    later = {"role": "assistant", "content": "continued answer"}
    for _ in range(2):
        child.turn(ctx, [*original, later], engine)
    _assert_rows(engine, "child", original)
    _assert_rows(engine, "continued", [later])
    assert aux.delegate_child_store_counts().get("fallbacks", 0) == before


class BindAgent(Agent):
    def bind(self, engine):
        engine.on_session_start(self.session_id, platform="subagent", parent_session_id="bind-parent",
                                context_length=200_000)


@pytest.mark.parametrize("write_fails", [False, True], ids=["confirmed-removal", "write-failure-retained"])
def test_bind_parent_cleanup_preserves_later_turn_parent(host, monkeypatch, write_fails):
    ctx, _, engine = _parent(host)
    child = BindAgent("child", **DELEGATE)
    child.bind(engine)
    with engine._auxiliary_session_lock:
        assert engine._delegate_child_bind_parent_ids["child"] == "bind-parent"
    failures = []

    def fail_write(*args, **kwargs):
        failures.append(args)
        raise sqlite3.OperationalError("synthetic bind link failure")

    with monkeypatch.context() as patch:
        if write_fails:
            patch.setattr(engine._store, "record_session_lineage", fail_write)
        child.turn(ctx, child_view(), engine)
    _assert_rows(engine, "child", child_view())
    with engine._auxiliary_session_lock:
        if write_fails:
            assert failures
            assert engine._delegate_child_bind_parent_ids["child"] == "bind-parent"
            assert engine._stored_delegate_child_link("child") == {}
        else:
            assert ("child", "bind-parent", "bind_kwargs") in lineage_rows(engine)
            assert "child" not in engine._delegate_child_bind_parent_ids
    for _ in range(2):
        child.turn(ctx, child_view(1), engine)
    _assert_rows(engine, "child", child_view(1))
    assert ("child", "bind-parent", "bind_kwargs") in lineage_rows(engine)
