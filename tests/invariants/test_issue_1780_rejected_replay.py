"""I1/I2: observing alias conflicts must preserve base cold-replay rows and bindings."""
from copy import deepcopy
import sqlite3
import time

import pytest

from tests.harness import engine_context, pin_char_counter


def message(tag, stamp, uid=None):
    row = {"role": "user", "content": f"Verifier fixture line {tag}.", "timestamp": float(stamp)}
    if uid:
        row["message_uid"] = uid
    return row


REPLY = {"role": "assistant", "content": "Verifier fixture reply."}


@pytest.fixture(autouse=True)
def isolated_host(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(time, "time", lambda: 1702000000.)
    (tmp_path / "home").mkdir()
    with sqlite3.connect(tmp_path / "home/state.db") as db:
        db.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY,parent_session_id TEXT,end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES('parent',NULL,NULL)")


def engine(path):
    return engine_context(path, window=128000, session_id="parent", conversation_id="conv",
                          embeddings_enabled=False, temporal_rollups_enabled=False)


def seed_store(path, reply=False):
    initial = [message("S", 1000, "uid-a"), *([REPLY] if reply else []), message("O", 1001, "uid-o")]
    with engine(path) as e:
        e.ingest(deepcopy(initial))
        e._store.add_message_relations([[(1, "alt_stamp", None, None, 99.)]])
        return [tuple(r) for r in e._store._conn.execute("SELECT * FROM messages ORDER BY store_id")]


@pytest.mark.parametrize("shape,expected", [
    ("R6", [3, 3, 3, 3]), ("R5", [4, 4, 4, 4]), ("R2grow", [4, 5, 6, 7]),
])
@pytest.mark.parametrize("legacy", [False, True], ids=["v2-writer", "base-writer"])
def test_conflict_is_observed_with_base_rows_and_bindings(tmp_path, monkeypatch, caplog, record_property, shape, expected, legacy):
    from tests.invariants.test_issue_1780_round3 import compare_replays

    initial = [message("S", 1000, "uid-a"), *([REPLY] if shape != "R6" else []), message("O", 1001, "uid-o")]
    view = [message("S", 99, "uid-x"), *([REPLY] if shape != "R6" else []), message("O", 1001, "uid-o")]
    view.append(message("T", 99, None if shape == "R2grow" else "uid-t"))
    counts, count = compare_replays(tmp_path, monkeypatch, caplog, 2, view, turns=4,
                                   grow=shape == "R2grow", initial=initial, legacy_seed=legacy)
    record_property("rows_and_would_reject", {"shape": shape, "counts": counts, "would_reject": count})
    assert counts == (expected if legacy else {"R6": [4] * 4, "R5": [6] * 4, "R2grow": [6, 7, 8, 9]}[shape])
    assert count > 0
    for observe in (False, True):
        with engine(tmp_path / str(observe)) as eng:
            old = eng._store.read_metadata_json("host_occurrence_row:parent:1")
            if legacy:
                assert old is None
                continue
            fresh_id = 3 if shape == "R6" else 4
            fresh = eng._store.read_metadata_json(f"host_occurrence_row:parent:{fresh_id}")
            coverage = eng._occurrence_fact_coverage(
                eng._store.read_metadata_json("host_occurrence_accepted_view:parent"), "parent")
            assert old["occurrence_version"] == fresh["occurrence_version"] == 2
            assert old["message_uid"] == "uid-a" and fresh["message_uid"] == "uid-x"
            assert {1, fresh_id} <= coverage
            if shape != "R6":
                reply = eng._store.read_metadata_json("host_occurrence_row:parent:5")
                assert reply["user_store_id"] == fresh_id and 5 in coverage


def test_observation_preserves_positional_cursor(tmp_path):
    seed_store(tmp_path)
    view = [message("S", 99, "uid-x"), message("O", 1001, "uid-o"), message("T", 99, "uid-t")]
    with engine(tmp_path) as e:
        plan = e._identity_anchor_prematch(view, view, 2)
        assert plan["cursor"] == 2
