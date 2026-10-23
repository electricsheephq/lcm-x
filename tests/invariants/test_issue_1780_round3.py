"""Count-only UID observation preserves base replay at production matching caps."""
from copy import deepcopy
import json
import logging
import random
import sqlite3

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1519_alias_uid_conflict import bindings, deterministic_host, message, rows

__all__ = ["deterministic_host"]


def context(path):
    return engine_context(path, window=128000, session_id="parent", conversation_id="conv",
                          embeddings_enabled=False, temporal_rollups_enabled=False)


def seed(path, n):
    with context(path) as eng:
        eng.ingest([message("shared", 1000 + i, message_uid=f"uid-{i}") for i in range(n)])
        eng._store.add_message_relations([[(r["store_id"], "alt_stamp", None, None, 99.)] for r in rows(eng)])
        return rows(eng)


def compare_replays(path, monkeypatch, caplog, n, view, turns=2, grow=False, initial=None, legacy_seed=False):
    """Pair identical on-mode stores with observation disabled/enabled; compare every row/binding."""
    from hermes_lcm.identity_anchor import IdentityAnchorMixin

    arms = []
    count = 0
    for observe in (False, True):
        home = path / str(observe)
        (home / "home").mkdir(parents=True)
        with sqlite3.connect(home / "home/state.db") as db:
            db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
            db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
        with monkeypatch.context() as patch:
            if not observe:
                patch.setattr(IdentityAnchorMixin, "_observe_alias_uid_conflicts", lambda *a: None)
            with monkeypatch.context() as writer:
                if legacy_seed:
                    from hermes_lcm.engine import LCMEngine

                    writer.setattr(LCMEngine, "_occurrence_batch_factory",
                                   lambda self, batch: self._real_user_scaffold_metadata_rows)
                if initial is None:
                    original = seed(home, n)
                else:
                    with context(home) as eng:
                        eng.ingest(deepcopy(initial))
                        eng._store.add_message_relations([[(1, "alt_stamp", None, None, 99.)]])
                        original = rows(eng)
                if legacy_seed:
                    with context(home) as eng:
                        assert eng._store._conn.execute(
                            "SELECT count(*) FROM metadata WHERE key LIKE 'host_occurrence_row:%' "
                            "OR key LIKE 'host_occurrence_accepted_view:%'").fetchone()[0] == 0
                        assert any(r["observed_at"] == 99. for r in eng._store.get_message_relations([1], "alt_stamp"))
            snapshots = []
            current_view = deepcopy(view)
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
                for turn in range(turns):
                    if grow and turn:
                        current_view.append({"role": "assistant", "content": f"Verifier later reply {turn}."})
                    with context(home) as eng:
                        if legacy_seed:
                            plan = eng._resolve_occurrence_view(current_view)
                            assert 0 not in plan["proven"]
                            assert "host_occurrence_row:parent:1" not in plan["updates"]
                        eng.ingest(deepcopy(current_view))
                        if legacy_seed:
                            assert eng._store.read_metadata_json("host_occurrence_row:parent:1") is None
                        current = rows(eng)
                        assert current[:len(original)] == original
                        snapshots.append((current, [bindings(eng, r["store_id"]) for r in current]))
            if observe:
                count = sum(r.args[0] for r in caplog.records if "would reject" in r.msg)
            arms.append(snapshots)
    assert arms[0] == arms[1]
    return [len(s[0]) for s in arms[1]], count


@pytest.mark.parametrize("n", [256, 1024])
def test_production_group_observation(tmp_path, monkeypatch, caplog, record_property, n):
    cut = 768 if n == 1024 else n
    order = list(reversed(range(cut))) + list(range(cut, n))
    view = [message("shared", 99., **({"message_uid": f"uid-{i}"}
                    if n == 256 or j % 7 else {})) for j, i in enumerate(order)]
    view.append(message("shared", 99., message_uid="uid-fresh"))
    counts, count = compare_replays(tmp_path, monkeypatch, caplog, n, view, turns=3)
    record_property("rows_and_would_reject", {"n": n, "counts": counts, "would_reject": count})
    assert counts == [n + 1] * 3
    assert count > 0


def test_seeded_cold_replay_properties(tmp_path, monkeypatch, caplog, record_property):
    """Retain the 300 production-cap shapes; observation never changes their rows or bindings."""
    rng = random.Random(1780032026)
    stats = []
    for trial in range(300):
        n = 1200 if trial == 299 else rng.randrange(256, 1201) if trial % 20 == 0 else rng.randrange(1, 65)
        cut = rng.randrange(1, n + 1)
        order = list(reversed(range(cut))) + list(range(cut, n))
        uids = [None if rng.random() < .2 else f"uid-{i}" for i in order]
        for j in rng.sample(range(n), rng.randrange(min(4, n) + 1)):
            uids[j] = f"fresh-replacement-{j}"
        uids += [f"fresh-tail-{i}" for i in range(rng.randrange(1, 4))]
        view = [message("shared", 99., **({"message_uid": uid} if uid else {})) for uid in uids]
        counts, count = compare_replays(tmp_path / f"case-{trial}", monkeypatch, caplog, n, view)
        assert counts[0] == counts[1]
        stats.append({"trial": trial, "n": n, "counts": counts, "would_reject": count})
    record_property("seeded_cases", json.dumps(stats, sort_keys=True))
    assert len(stats) == 300 and max(s["n"] for s in stats) == 1200
    assert all(s["would_reject"] > 0 for s in stats)
