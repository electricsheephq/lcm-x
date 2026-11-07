"""#2070: a delegated child of an excluded session that rotated by compression is refused on the clone and prototype
routes too, by asking the engine bound to the parent right now. Every doubt must fall back to the pattern check, so a
normal parent's child is always stored: id reuse in either order, a rebind, another store, a binding caught mid-rebind,
a helper failure, and an owner whose store was never opened.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, child_rows, child_view
from tests.test_issue_1931_delegate_children import host  # noqa: F401 (the fixture)


def _registry(engine):
    return sys.modules[type(engine).__module__.rsplit(".", 1)[0] + ".engine_registry"]


def _excluded_parent(host, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "old-parent")
    load, clone = host
    ctx, module = load()
    owner = clone(ctx)
    owner.on_session_start("old-parent", platform="cli", context_length=200_000)
    owner.on_session_start("parent", platform="subagent", boundary_reason="compression", old_session_id="old-parent")
    if not (owner._session_id == "parent" and owner._session_stateless and _registry(owner)
            ._ACTIVE_ENGINES_BY_SESSION_ID.get("parent") is owner):
        pytest.fail("setup: the rotated parent is not an excluded live binding")
    return ctx, module, clone, owner


def _store_child_on_a_clone(ctx, clone):
    engine = clone(ctx)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    child.turn(ctx, child_view(), engine)
    return engine


def _stored(engine):
    return [row["content"] for row in child_rows(engine)] == [row["content"] for row in child_view()]


def test_the_excluded_live_parent_refuses_its_child(host, monkeypatch):  # noqa: F811
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    _store_child_on_a_clone(ctx, clone)
    assert child_rows(owner) == []


def test_a_normal_reuse_of_the_id_on_another_engine_stores_the_child(host, monkeypatch):  # noqa: F811
    """The excluded binding first, then a normal binding of the same id on another platform: the live owner is normal."""
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    reuse = clone(ctx)
    reuse.on_session_start("parent", platform="telegram", context_length=200_000)
    if reuse._session_stateless or reuse._session_ignored:
        pytest.fail("setup: the reused id is not a normal binding")
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)


def test_a_normal_binding_made_first_still_stores_the_child(host, monkeypatch):  # noqa: F811
    """Reverse order: a normal binding of the id exists, then another engine rotates into it excluded."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "old-parent")
    load, clone = host
    ctx, _ = load()
    normal = clone(ctx)
    normal.on_session_start("parent", platform="telegram", context_length=200_000)
    owner = clone(ctx)
    owner.on_session_start("old-parent", platform="cli", context_length=200_000)
    owner.on_session_start("parent", platform="subagent", boundary_reason="compression", old_session_id="old-parent")
    if _registry(owner)._ACTIVE_ENGINES_BY_SESSION_ID.get("parent") is not owner or normal._session_stateless:
        pytest.fail("setup: the excluded binding is not the registered one over a normal binding")
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)


def test_an_owner_that_rebound_elsewhere_refuses_nothing(host, monkeypatch):  # noqa: F811
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    owner.on_session_start("elsewhere", platform="cli", context_length=200_000)
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)


def test_an_owner_on_another_store_refuses_nothing(host, monkeypatch):  # noqa: F811
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    engine = clone(ctx)
    monkeypatch.setitem(owner.__dict__, "_store", SimpleNamespace(
        db_path=Path("/another/lcm.db"), _conn=object(), close=lambda: None))
    child = Agent("child", **DELEGATE)
    child.start(engine)
    child.turn(ctx, child_view(), engine)
    assert _stored(engine)


def test_an_owner_not_registered_for_the_parent_refuses_nothing(host, monkeypatch):  # noqa: F811
    """Direct registry hits only: an excluded engine whose ``_session_id`` reads as the parent but that is not
    registered for it (a foreground scan would find it) refuses nothing."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "old-parent")
    load, clone = host
    ctx, _ = load()
    owner = clone(ctx)
    owner.on_session_start("old-parent", platform="cli", context_length=200_000)
    if not owner._session_ignored:
        pytest.fail("setup: the old binding is not excluded")
    monkeypatch.setitem(owner.__dict__, "_session_id", "parent")  # stale flags, id written, not registered
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)


def test_a_failing_check_falls_back_to_the_pattern_route(host, monkeypatch):  # noqa: F811
    ctx, module, clone, owner = _excluded_parent(host, monkeypatch)

    class Broken:
        def get(self, *args):
            raise RuntimeError("synthetic registry failure")

    monkeypatch.setattr(module("aux_session"), "_ACTIVE_ENGINES_BY_SESSION_ID", Broken())
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)  # "parent" matches no pattern: stored, never "rows not stored"


def test_an_owner_whose_store_was_never_opened_is_not_opened(host, monkeypatch):  # noqa: F811
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    bare = clone(ctx)
    if "_store" in bare.__dict__:
        pytest.fail("setup: a fresh clone already opened its store")
    bare.__dict__.update(_session_id="parent", _session_ignored=True)
    monkeypatch.setitem(_registry(owner)._ACTIVE_ENGINES_BY_SESSION_ID, "parent", bare)
    _store_child_on_a_clone(ctx, clone)
    assert "_store" not in bare.__dict__
    assert _stored(owner)


def test_a_same_id_rebind_to_a_normal_binding_during_the_check_refuses_nothing(host, monkeypatch):  # noqa: F811
    """The owner rebinds the same id normally while the slow known-child query runs: the flags are read only after
    that query (the reorder this pins), so they show the normal binding and the child is stored."""
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    engine = clone(ctx)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    known = engine._is_known_delegate_child
    rebinds = []

    def rebind_while_checking(session_id):
        if session_id == "parent" and not rebinds:
            rebinds.append(session_id)
            owner.__dict__.update(_session_stateless=False, _session_ignored=False)  # the same id, now normal
        return known(session_id)

    monkeypatch.setattr(engine, "_is_known_delegate_child", rebind_while_checking)
    child.turn(ctx, child_view(), engine)
    if not rebinds:
        pytest.fail("setup: the check did not run")
    assert _stored(engine)


def test_a_registered_owner_rebinding_to_an_excluded_id_refuses_nothing(host, monkeypatch):  # noqa: F811
    """The owner is registered for a NORMAL parent and starts rebinding to an ignored id: the new id and flags are
    written, the registry still maps the parent to it. The id re-check sees it no longer holds the parent."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "cron-x")
    load, clone = host
    ctx, _ = load()
    owner = clone(ctx)
    owner.on_session_start("parent", platform="cli", context_length=200_000)
    if owner._session_stateless or owner._session_ignored or _registry(owner)._ACTIVE_ENGINES_BY_SESSION_ID.get(
            "parent") is not owner:
        pytest.fail("setup: the owner is not a registered normal binding of the parent")
    monkeypatch.setitem(owner.__dict__, "_session_id", "cron-x")  # mid-rebind: id and flags written, not registered
    monkeypatch.setitem(owner.__dict__, "_session_ignored", True)
    _store_child_on_a_clone(ctx, clone)
    assert _stored(owner)
