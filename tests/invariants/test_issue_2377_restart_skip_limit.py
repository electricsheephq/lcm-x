"""#2377 named limit for v0.30.0: after a restart, a new tool call/result identical to a stored one can be skipped.

The strict xfails reproduce the restart skip on main. They turn green when the fix (#2414, planned for v0.30.x)
lands, and must then become ordinary tests. The control pins the replay that main correctly skips.
"""
import pytest

from tests.invariants._ingest_shapes_2377 import U, V, X, call, ingest_shape
from tests.invariants.test_issue_2163_compacted_cursor import build as build

pytestmark = pytest.mark.issue(2377)


def limit(why):
    return pytest.mark.xfail(strict=True, reason=f"#2377 known limit (fix planned for v0.30.x, #2414): {why}")


@limit("main skips the new call and result after a rewritten prefix")
def test_restart_after_a_changed_prefix_stores_the_new_pair(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X, call(2)],
                       [U, call(), dict(X, content="Y"), call(1), X, V], 3)
    run.check({3, 4, 5})


@limit("main skips the new call and result after storing a rewritten original")
def test_restart_after_a_stored_rewrite_stores_the_new_pair(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X], [U, call(), dict(X, content="Y"), call(), X, V])
    run.check({3, 4, 5})


@pytest.mark.parametrize("cursor", [1, 3, 5])
def test_restart_replay_of_stored_pairs_is_skipped_and_new_pair_stored(build, monkeypatch, cursor):
    """Control (passes on main): a replay of stored pairs is skipped; the new pair after them is stored."""
    source = [U, call(), X, call(), X]
    run = ingest_shape(build, monkeypatch, source, [*source, call(), X, V], cursor)
    run.check({5, 6, 7})
    assert not ({i for i in range(cursor, 5)} - run.skips)
