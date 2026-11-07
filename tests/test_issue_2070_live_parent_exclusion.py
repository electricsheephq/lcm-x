"""#2070 (deferred to v0.30.x): a delegated child of an excluded session that rotated by compression is still stored on
the clone and prototype routes; only the shared route refuses it. The refusal is pinned here as a strict xfail.

The other tests are guards for the v0.30.x fix. Each asserts that a NORMAL parent's child IS stored, so a later refusal
mechanism cannot drop its rows: id reuse in either order, a rebind, another store, an unregistered engine, a binding
caught mid-rebind, a start that raises, two aborted starts, and an auxiliary start after them. Main stores every child on
these routes, so they pass today. A v0.30.0 attempt at the refusal (#2420) failed three of these shapes in review, one
per round.
"""
import sys
import threading
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2070 deferred to v0.30.x")
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


def _rebind_normally_with_a_hook(owner, hook):
    """Run ``hook`` inside the owner's same-id rebind to a normal binding, after the id is written and before the
    flags are refreshed: the window in which the old, excluded flags still read as the parent's binding."""
    refresh = owner._refresh_session_filters
    calls = []

    def refresh_after_hook():
        if not calls:
            calls.append(True)
            hook()
        return refresh()

    owner.__dict__["_refresh_session_filters"] = refresh_after_hook
    try:
        owner.on_session_start("parent", platform="telegram", context_length=200_000)
    finally:
        owner.__dict__.pop("_refresh_session_filters", None)
    if not calls:
        pytest.fail("setup: the hook did not run inside the rebind")


def test_a_check_inside_a_same_id_rebind_to_a_normal_binding_refuses_nothing(host, monkeypatch):  # noqa: F811
    """The interleaving the second review found: the owner rebinds the parent's id normally, and the child's check runs
    between the id write and the flag refresh. Both flag reads see the old excluded flags; the binding is being
    rewritten, so nothing is refused."""
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    engines = []
    _rebind_normally_with_a_hook(owner, lambda: engines.append(_store_child_on_a_clone(ctx, clone)))
    if owner._session_stateless or owner._session_ignored or owner._session_id != "parent":
        pytest.fail("setup: the rebind did not leave a normal binding of the parent")
    assert _stored(engines[0])


def test_a_start_that_raises_inside_the_bind_refuses_nothing(host, monkeypatch):  # noqa: F811
    """A start that raises part way through, with the child's store inside that window: the child is stored."""
    ctx, _module, clone, owner = _excluded_parent(host, monkeypatch)
    engines = []

    def store_then_raise():
        engines.append(_store_child_on_a_clone(ctx, clone))
        raise RuntimeError("synthetic failure inside the bind")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        _rebind_normally_with_a_hook(owner, store_then_raise)
    assert _stored(engines[0])


def test_two_aborted_starts_refuse_nothing(host, monkeypatch):  # noqa: F811
    """A normal owner of the parent starts rebinding to the ignored "cron-x" and raises before
    registration (id and excluded flags written); a normal rebind of the parent then raises before its flag refresh. The
    binding reads as the parent with the cron flags, still registered: mixed fields an aborted write left. The child is
    stored."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "cron-x")
    load, clone = host
    ctx, _ = load()
    owner = clone(ctx)
    owner.on_session_start("parent", platform="cli", context_length=200_000)
    if owner._session_stateless or owner._session_ignored:
        pytest.fail("setup: the owner is not a normal binding of the parent")

    def raises(*_args, **_kwargs):
        raise RuntimeError("synthetic failure inside the bind")

    for name, session_id, kwargs in [("_register_active_engine_binding", "cron-x", {"platform": "cli"}),
                                     ("_refresh_session_filters", "parent", {"platform": "telegram"})]:
        owner.__dict__[name] = raises
        try:
            with pytest.raises(RuntimeError, match="synthetic failure"):
                owner.on_session_start(session_id, **kwargs)
        finally:
            owner.__dict__.pop(name, None)
    if not (owner._session_id == "parent" and owner._session_ignored
            and _registry(owner)._ACTIVE_ENGINES_BY_SESSION_ID.get("parent") is owner):
        pytest.fail("setup: the aborted starts did not leave the parent id with the cron flags, registered")
    assert _stored(_store_child_on_a_clone(ctx, clone))


def test_an_auxiliary_start_after_two_aborted_starts_refuses_nothing(host, monkeypatch):  # noqa: F811
    """The same mixed binding, then a same-engine auxiliary child start, which returns without rebinding: the binding
    still reads as the parent with the cron flags. The child is stored."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "cron-x")
    load, clone = host
    ctx, _ = load()
    owner = clone(ctx)
    owner.on_session_start("parent", platform="cli", context_length=200_000)

    def raises(*_args, **_kwargs):
        raise RuntimeError("synthetic failure inside the bind")

    for name, session_id, kwargs in [("_register_active_engine_binding", "cron-x", {"platform": "cli"}),
                                     ("_refresh_session_filters", "parent", {"platform": "telegram"})]:
        owner.__dict__[name] = raises
        try:
            with pytest.raises(RuntimeError, match="synthetic failure"):
                owner.on_session_start(session_id, **kwargs)
        finally:
            owner.__dict__.pop(name, None)
    owner.__dict__["_is_live_auxiliary_child_session"] = lambda session_id, *_args: session_id == "aux-child"
    try:
        owner.on_session_start("aux-child", platform="cli")
    finally:
        owner.__dict__.pop("_is_live_auxiliary_child_session", None)
    if not (owner._session_id == "parent" and owner._session_ignored
            and _registry(owner)._ACTIVE_ENGINES_BY_SESSION_ID.get("parent") is owner):
        pytest.fail("setup: the auxiliary start did not leave the parent id with the cron flags, registered")
    assert _stored(_store_child_on_a_clone(ctx, clone))


def test_a_second_rebind_paused_before_its_flags_refuses_nothing(host, monkeypatch):  # noqa: F811
    """Two rebinds with real starts: a normal owner of the parent rebinds to the ignored "cron-x", then a normal rebind of
    the parent on another thread writes the id and pauses before its flags. The child stores during that pause, when the
    binding reads as the parent with the stale cron flags. The child is stored."""
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "cron-x")
    load, clone = host
    ctx, _ = load()
    owner = clone(ctx)
    owner.on_session_start("parent", platform="cli", context_length=200_000)
    owner.on_session_start("cron-x", platform="cli")
    if not owner._session_ignored:
        pytest.fail("setup: the cron binding is not excluded")
    paused, resume, wait = threading.Event(), threading.Event(), 10
    refresh = owner._refresh_session_filters

    def refresh_after_pause():
        if threading.current_thread().name == "normal-rebind" and not paused.is_set():
            paused.set()
            resume.wait(wait)
        return refresh()

    owner.__dict__["_refresh_session_filters"] = refresh_after_pause
    rebind = threading.Thread(target=owner.on_session_start, args=("parent",),
                              kwargs={"platform": "telegram", "context_length": 200_000}, name="normal-rebind")
    try:
        rebind.start()
        if not paused.wait(wait) or owner._session_id != "parent" or not owner._session_ignored:
            pytest.fail("setup: the rebind did not pause with the parent id and the cron flags")
        engine = _store_child_on_a_clone(ctx, clone)
    finally:
        resume.set()
        rebind.join(wait)
        owner.__dict__.pop("_refresh_session_filters", None)
    assert _stored(engine)
