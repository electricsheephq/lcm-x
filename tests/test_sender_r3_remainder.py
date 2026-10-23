"""FIX-R5: an R3 exclusion applies to stored rows across UPDATE retries."""
import sys

import pytest

from tests.test_issue_436_identity_anchor import PAD, SYSTEM, _a, _engine, _turns, _u
from tests.test_sender_capture import capture_module, clean_capture_registry, expected  # noqa: F401


@pytest.fixture(autouse=True)
def anchor_on(monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "1")


def begin(engine, view, turn="turn-current"):
    return capture_module().create_pending(engine, dict(
        session_id=engine._session_id, turn_id=turn, platform="telegram",
        sender_id="synthetic-a", conversation_history=view))


def drain(engine):
    assert engine.drain_rollup_maintenance(timeout=10)


def users(engine):
    return engine._store.connection.execute(
        "SELECT store_id, sender_id, sender_source FROM messages WHERE role='user' ORDER BY store_id"
    ).fetchall()


def assert_null(engine):
    if any(row[1:] != (None, None) for row in users(engine)):
        raise AssertionError("excluded user row was stamped")


def shape(engine, parts=3, uid=False, marker=False):
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    first = _u("synthetic A" + PAD, 500.0)
    if uid:
        first["message_uid"] = "a" * 32
    engine.ingest([*head, first])
    drain(engine)
    tail = ["synthetic B" + PAD, "synthetic C" + PAD][:parts - 1]
    survivor = _u("\n\n".join([first["content"], *tail]), 500.0)
    if uid:
        survivor["message_uid"] = first["message_uid"]
    if marker:
        survivor["_merged_turn_prefix"] = first["content"]
    return [*head, survivor], "\n\n".join(tail)


def assert_remainder(engine, remainder):
    rows = engine._store.connection.execute(
        "SELECT store_id, observed_at FROM messages WHERE content=?", (remainder,)).fetchall()
    assert len(rows) == 1 and rows[0][1] is None
    return rows[0][0]


@pytest.mark.parametrize("parts", [2, 3], ids=["A+B", "A+B+C"])
@pytest.mark.parametrize("uid", [False, True], ids=["uidless", "first-uid"])
def test_r3_ingest_update_keeps_remainder_null(tmp_path, parts, uid):
    engine = _engine(tmp_path)
    try:
        view, remainder = shape(engine, parts, uid)
        assert begin(engine, view) is not None
        engine.ingest(view)
        drain(engine)
        assert_remainder(engine, remainder)
        assert_null(engine)
        capture_module().try_update(engine)
        assert_null(engine)
        engine.on_turn_complete(view, turn_id="turn-current")
        current = _u("synthetic next occurrence" + PAD, 600.0)
        next_view = [*view, _a("reply", 501.0), current]
        assert begin(engine, next_view, "turn-next") is not None
        engine.ingest(next_view)
        drain(engine)
        rows = users(engine)
        if [row[1:] for row in rows] != [(None, None)] * (len(rows) - 1) + [expected()]:
            raise AssertionError("next turn sender alignment differs")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("retry", ["direct", "next-ingest", "empty-store", "pending-replacement"])
def test_r3_deferred_update_retry_stays_null(tmp_path, monkeypatch, retry):
    engine = _engine(tmp_path)
    cap = capture_module()
    try:
        view, remainder = shape(engine)
        assert begin(engine, view) is not None
        # Contention-style deferral allows the row to reach each later retry unstamped.
        with monkeypatch.context() as deferred:
            deferred.setattr(engine._store, "set_sender_if_null", lambda *_: 0)
            engine.ingest(view)
            drain(engine)
        assert_remainder(engine, remainder)
        assert_null(engine)
        calls = []
        real = cap.try_update

        def trace(current):
            caller = sys._getframe(1)
            calls.append(caller.f_code.co_name)
            if retry != "empty-store" or len(calls) != 1:
                real(current)

        monkeypatch.setattr(cap, "try_update", trace)
        count = len(users(engine))
        total = engine._store.connection.execute("SELECT count(*) FROM messages").fetchone()[0]
        if retry == "direct":
            cap.try_update(engine)
        elif retry == "pending-replacement":
            engine.on_turn_complete(view, turn_id="turn-current")
            assert begin(engine, view, "turn-replacement") is not None
        else:
            if retry == "empty-store":
                # Revisit the survivor: the anchor filters its already-stored constituents.
                engine._ingest_cursor = len(view) - 1
            engine.ingest(view)
            drain(engine)
            # Rewound replay reaches the filtered-empty return at ~7200.
            assert calls.count("_ingest_messages") == (2 if retry == "empty-store" else 1)
            assert len(users(engine)) == count
            assert engine._store.connection.execute("SELECT count(*) FROM messages").fetchone()[0] == total
        assert calls
        assert_null(engine)
    finally:
        engine.shutdown()


def test_r3_stored_before_pending_remains_excluded(tmp_path):
    engine = _engine(tmp_path)
    try:
        view, remainder = shape(engine)
        engine.ingest(view)
        drain(engine)
        assert_remainder(engine, remainder)
        assert_null(engine)
        assert begin(engine, view) is not None  # create_pending's own UPDATE
        assert_null(engine)
        engine.ingest(view)
        drain(engine)
        assert_null(engine)
    finally:
        engine.shutdown()


def test_r3_session_rotation_keeps_database_exclusion(tmp_path):
    engine = _engine(tmp_path)
    try:
        view, remainder = shape(engine)
        engine.ingest(view)
        drain(engine)
        sid = assert_remainder(engine, remainder)
        assert_null(engine)
        watch = engine._host_rewrite_watch
        engine.on_session_start("C", boundary_reason="compression", old_session_id="S",
                                platform="cli", context_length=200_000, conversation_id="conv")
        drain(engine)
        assert engine._host_rewrite_watch is watch
        assert sid in watch and any(obj is view[-1] for obj in watch[sid][2])
        # Populate the existing binding cache, so SQL's ancestor scope is not a second veto.
        engine._identity_anchor_chain_cache = ("C", ["S"])
        assert begin(engine, view, "turn-rotated") is not None
        capture_module().try_update(engine)
        assert_null(engine)
    finally:
        engine.shutdown()


def test_r3_exclusion_isolated_by_database(tmp_path):
    left_path, right_path = tmp_path / "left", tmp_path / "right"
    left_path.mkdir()
    right_path.mkdir()
    left, right = _engine(left_path), _engine(right_path)
    try:
        view, remainder = shape(left)
        left.ingest(view)
        drain(left)
        excluded = assert_remainder(left, remainder)
        # Same session and store_id, but an ordinary current occurrence in another DB.
        right_view, _ = shape(right)
        current = _u("ordinary markerless occurrence" + PAD, 600.0)
        right.ingest([*right_view[:-1], _u("synthetic A" + PAD, 500.0), current])
        drain(right)
        sid = users(right)[-1][0]
        assert sid == excluded and left._session_id == right._session_id
        assert begin(right, [current]) is not None
        if right._store.get_message_senders([sid])[sid] != expected():
            raise AssertionError("another database exclusion blocked a valid UPDATE")
        assert_null(left)
    finally:
        left.shutdown()
        right.shutdown()


def test_r3_mixed_candidates_fail_closed(tmp_path):
    """Mutant discriminator: dropping the excluded candidate must not stamp the other."""
    engine = _engine(tmp_path)
    try:
        view, remainder = shape(engine)
        engine.ingest(view)
        drain(engine)
        excluded = assert_remainder(engine, remainder)
        survivor = view[-1]
        other = engine._store.append("S", survivor)
        engine._watch_stored_user_rows([(survivor, survivor, other)])
        assert excluded != other
        assert begin(engine, view) is not None
        assert_null(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("anchor,marker", [("0", False), ("1", True)])
def test_r3_existing_exclusion_controls(tmp_path, monkeypatch, anchor, marker):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    engine = _engine(tmp_path)
    try:
        view, _ = shape(engine, marker=marker)
        assert begin(engine, view) is not None
        engine.ingest(view)
        drain(engine)
        capture_module().try_update(engine)
        assert_null(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("path", ["insert", "update"])
def test_ordinary_markerless_current_stamps_once(tmp_path, path):
    engine = _engine(tmp_path)
    try:
        current = _u("ordinary current occurrence" + PAD, 500.0)
        if path == "update":
            engine.ingest([current])
            drain(engine)
        assert begin(engine, [current]) is not None
        if path == "insert":
            engine.ingest([current])
            drain(engine)
        if [row[1:] for row in users(engine)] != [expected()]:
            raise AssertionError("ordinary current occurrence was not stamped")
        before = engine._store.connection.total_changes
        capture_module().try_update(engine)
        assert engine._store.connection.total_changes == before
    finally:
        engine.shutdown()
