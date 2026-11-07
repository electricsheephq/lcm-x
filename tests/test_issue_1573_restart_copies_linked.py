"""#1573 check (C1): the repeated-restart copies of the issue's count table (V3 with three restarts, V5, V7) are
linked ``possible_replay_of`` at shipped defaults, so they are visible duplicates, never silent surplus.

The invariant holds with or without the #1573 fix: every row a restart stored beyond the genuine occurrences carries a
link to an earlier stored row with the same role and content. A fix that stops storing the copies keeps it (no copies, no
links); a change that stores a copy without a link fails it.
"""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s1573"
ROLES = {"B0": "user", "B1": "assistant", "B2": "user", "B3": "assistant"}
# shape: (views before the restarts, the view after each restart, restarts, genuine occurrences)
SHAPES = {
    "V3x3": (["B0 B1", "B0 B1 B0"], "B0 B1", 3, 3),
    "V5": (["B0 B1", "B0 B0"], "B0 B0", 1, 3),  # B1, then the host's second B0
    "V7": (["B0 B1 B2 B3", "B0 B1 B2 B3 B1"], "B0 B1 B2 B3", 1, 5),
}


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
                                        large_output_externalization_enabled=False),
                       hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000, conversation_id="c1573")
    return engine


def _view(labels, cls):
    rows = []
    for label in labels.split():
        row = {"role": ROLES[label], "content": f"row {label} distinct content"}
        if "u" in cls:
            row["message_uid"] = f"uid-{label}"
        if "s" in cls:
            row["timestamp"] = 1000.0 + int(label[1:])
        rows.append(row)
    return rows


@pytest.mark.parametrize("cls", ["n", "u", "s", "us"])  # neither, message_uid only, stamp only, both
@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_restart_copy_is_linked_as_a_possible_replay(tmp_path, shape, cls):
    before, after, restarts, genuine = SHAPES[shape]
    engine = _engine(tmp_path)
    try:
        for labels in before:
            engine.ingest(_view(labels, cls))
        # Without a stamp, V5's second B0 can first be stored by the restart's ingest, so it may still be missing here.
        before_ids = {row["store_id"] for row in engine._store.get_session_messages(SESSION)}
        if len(before_ids) > genuine:
            pytest.fail("setup: more rows than genuine occurrences were stored before the restarts")
        for _ in range(restarts):
            engine.shutdown()
            engine = _engine(tmp_path)
            engine.ingest(_view(after, cls))
        rows = {row["store_id"]: row for row in engine._store.get_session_messages(SESSION)}
        conn = engine._store._conn
        links = conn.execute("SELECT store_id, related_store_id FROM message_relations "
                             "WHERE kind='possible_replay_of'").fetchall() if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone() else []
    finally:
        engine.shutdown()
    copies = {store_id for store_id, _ in links}
    for store_id, target in links:
        assert target < store_id
        assert (rows[target]["role"], rows[target]["content"]) == (rows[store_id]["role"], rows[store_id]["content"])
    # Rows the restarts added: all but the genuine occurrences still missing before them are copies, and each needs its
    # own link (a link on a row stored before the restarts cannot stand in for one).
    unlinked = (set(rows) - before_ids) - copies
    assert len(unlinked) <= genuine - len(before_ids), "a restart copy is stored without a possible_replay_of link"
