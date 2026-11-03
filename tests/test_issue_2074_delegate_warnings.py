# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""#2074: successful delegated-child storage is not a warning or a fallback."""

import logging
import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, child_rows, child_view, host  # noqa: F401


def test_compression_resume_logs_debug_without_counting_fallback(host, monkeypatch, caplog):
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(2)
    compacted = [{"role": "user", "content": "compressed head"}, original[-1]]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    view = child.compress(engine, original)
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        for i in range(2):
            view = [*view, {"role": "assistant", "content": f"after compression {i}"}]
            child.turn(ctx, view, engine)
            child.turn(ctx, view, engine)
    expected = [*original, *view[-2:]]
    assert [(r["role"], r["content"]) for r in child_rows(engine)] == [
        (r["role"], r["content"]) for r in expected]
    assert module("aux_session").delegate_child_store_counts().get("fallbacks", 0) == before
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.DEBUG and "child: resuming after compression boundary" in r.message
               for r in caplog.records)


def test_rewritten_head_resume_logs_debug_without_counting_fallback(host, monkeypatch, caplog):
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    original = child_view(3)
    compacted = [{"role": "user", "content": "compressed head"}, original[-1]]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    child.compress(engine, original)
    view = [{"role": "user", "content": "rewritten head"}, *original[-2:]]
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        for i in range(2):
            view = [*view, {"role": "assistant", "content": f"after rewrite {i}"}]
            child.turn(ctx, view, engine)
            child.turn(ctx, view, engine)
    expected = [*original, *view[-2:]]
    assert [(r["role"], r["content"]) for r in child_rows(engine)] == [
        (r["role"], r["content"]) for r in expected]
    assert module("aux_session").delegate_child_store_counts().get("fallbacks", 0) == before
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.DEBUG and "child: resuming after stored tail anchor" in r.message
               for r in caplog.records)


def test_conflict_child_store_succeeds_without_skipped_ingest_warning(host, caplog):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    # No host engine: a resident parent forces the store-only conflict route.
    for _ in range(2):  # the second store returns 0 (nothing new), also a success
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            child.turn(ctx, child_view())
        assert not any(r.levelno >= logging.WARNING and "skipped ingest" in r.message for r in caplog.records)
        assert any(r.levelno == logging.DEBUG and "child: rows stored without binding an engine" in r.message
                   for r in caplog.records)
        assert [(r["role"], r["content"]) for r in child_rows(parent)] == [
            (r["role"], r["content"]) for r in child_view()]
        assert ctx.offered._session_id == ""


def test_conflict_child_store_failure_keeps_skipped_ingest_warning(host, monkeypatch, caplog):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    monkeypatch.setattr(ctx.offered, "_store_delegate_child_turn", lambda *args, **kwargs: None)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        Agent("child", **DELEGATE).turn(ctx, child_view())
    assert child_rows(parent) == []
    assert any(r.levelno == logging.WARNING and "skipped ingest" in r.message
               and "resident_engine_conflict" in r.message for r in caplog.records)
    assert not any("rows stored without binding an engine" in r.message for r in caplog.records)


def test_conflict_child_link_failure_keeps_fallback_and_skipped_ingest_warning(host, monkeypatch, caplog):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)

    def fail_link(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ctx.offered, "_delegate_child_link", fail_link)
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        Agent("child", **DELEGATE).turn(ctx, child_view())
    assert child_rows(parent) == []
    assert any(r.levelno == logging.WARNING
               and "rows not stored: OperationalError: database is locked" in r.message for r in caplog.records)
    assert module("aux_session").delegate_child_store_counts().get("fallbacks", 0) == before + 1
    assert any(r.levelno == logging.WARNING and "skipped ingest" in r.message
               and "resident_engine_conflict" in r.message for r in caplog.records)
    assert not any("rows stored without binding an engine" in r.message for r in caplog.records)


@pytest.mark.parametrize("case", ["no-link", "missing-store", "empty-session", "callback-not-run"])
def test_conflict_without_child_storage_keeps_skipped_ingest_warning(host, monkeypatch, caplog, case):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("plain", parent="parent")
    if case == "missing-store":
        monkeypatch.setattr(ctx.offered, "_store_delegate_child_turn", None)
        child = Agent("child", **DELEGATE)
    elif case == "empty-session":
        child = Agent("", **DELEGATE)
        # Without an id the resolver otherwise selects the foreground parent.
        monkeypatch.setattr(module("engine_registry"), "resolve_active_lcm_engine", lambda **kwargs: None)
    elif case == "callback-not-run":
        child = Agent("child", **DELEGATE)
        registry = module("engine_registry")
        run_stably = ctx.offered._run_stably
        attempts = []

        def skip_store(operation, **kwargs):
            attempts.append(operation)
            if len(attempts) == 1:
                return run_stably(operation, **kwargs)  # real cold-use conflict
            return registry.ActiveEngineUseResult(registry.ActiveEngineUseStatus.BUSY)

        monkeypatch.setattr(ctx.offered, "_run_stably", skip_store)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        child.turn(ctx, child_view())
    assert child_rows(parent, child.session_id) == []
    assert any(r.levelno == logging.WARNING and "skipped ingest" in r.message
               and "resident_engine_conflict" in r.message for r in caplog.records)
    assert not any("rows stored without binding an engine" in r.message for r in caplog.records)
