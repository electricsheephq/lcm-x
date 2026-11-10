"""#2377 named limit for v0.30.0: after a restart, a new tool call/result identical to a stored one can be skipped.

These regressions reproduced the restart skip on main and pass with #2414.
The control pins the replay that main correctly skips.
"""
import pytest

from tests.invariants._ingest_shapes_2377 import U, V, X, call, ingest_shape
from tests.invariants.test_issue_2163_compacted_cursor import build as build

pytestmark = pytest.mark.issue(2377)


def expect_known_shape(run, new, known):
    """Only main's exact #2377 shape counts as the expected failure; a fix shows as a loud XPASS."""
    failures = run.violations(new)
    if not any(failures.values()):
        return
    if failures == known:
        raise AssertionError(f"#2377 known shape: {failures}")
    pytest.fail(f"not the #2377 shape: {failures}")


def test_restart_after_a_changed_prefix_stores_the_new_pair(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X, call(2)],
                       [U, call(), dict(X, content="Y"), call(1), X, V], 3)
    expect_known_shape(run, {3, 4, 5}, dict(missing=[3, 4], silent=[], invalid=[]))


def test_restart_after_a_stored_rewrite_stores_the_new_pair(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X], [U, call(), dict(X, content="Y"), call(), X, V])
    # Main also re-stores the earlier call (host index 1) beside the rewritten result, unlinked (store id 4).
    expect_known_shape(run, {3, 4, 5}, dict(missing=[3, 4], silent=[4], invalid=[]))


@pytest.mark.parametrize("cursor", [1, 3, 5])
def test_restart_replay_of_stored_pairs_is_skipped_and_new_pair_stored(build, monkeypatch, cursor):
    """Control (passes on main): a replay of stored pairs is skipped; the new pair after them is stored."""
    source = [U, call(), X, call(), X]
    run = ingest_shape(build, monkeypatch, source, [*source, call(), X, V], cursor)
    run.check({5, 6, 7})
    assert not ({i for i in range(cursor, 5)} - run.skips)
