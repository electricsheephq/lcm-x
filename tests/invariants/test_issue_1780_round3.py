"""A budget-limited UID guard falls back to base, per connected same-key group."""
from collections import Counter
from copy import deepcopy
import json
import logging
import random
import sqlite3

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1519_alias_uid_conflict import deterministic_host, message, rows

__all__ = ["deterministic_host"]


def context(path):
    return engine_context(path, window=128000, session_id="parent", conversation_id="conv",
                          embeddings_enabled=False, temporal_rollups_enabled=False)


def seed(path, n):
    with context(path) as eng:
        eng.ingest([message("shared", 1000 + i, message_uid=f"uid-{i}") for i in range(n)])
        eng._store.add_message_relations([[(r["store_id"], "alt_stamp", None, None, 99.)] for r in rows(eng)])
        return rows(eng)


@pytest.mark.parametrize("n", [256, 1024])
def test_production_group_budget(tmp_path, caplog, n):
    original = seed(tmp_path, n)
    cut = 768 if n == 1024 else n
    order = list(reversed(range(cut))) + list(range(cut, n))
    view = [message("shared", 99., **({"message_uid": f"uid-{i}"}
                    if n == 256 or j % 7 else {})) for j, i in enumerate(order)]
    view.append(message("shared", 99., message_uid="uid-fresh"))
    counts = []
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
        for turn in range(3):
            with context(tmp_path) as eng:
                if not turn:
                    plan = eng._identity_anchor_prematch(view, view, 0)
                    if n == 256:
                        # Repair completes: each known UID keeps its own row, fresh UID is rejected.
                        assert {idx: group[0]["store_id"] for idx, group in plan["matched"].items()} == {
                            j: i + 1 for j, i in enumerate(order)}
                        assert n not in plan["replayed"]
                eng.ingest(deepcopy(view))
                current = rows(eng)
                assert current[:n] == original
                counts.append(len(current))
    assert counts == [n + 1] * 3
    skipped = [r for r in caplog.records if "guard_skipped_budget" in r.getMessage()]
    assert bool(skipped) == (n == 1024)
    assert any("work budget" in r.getMessage() for r in caplog.records) == (n == 1024)


def test_fallback_is_per_group_and_preserves_base_verbatim(monkeypatch, caplog):
    from hermes_lcm import recognition_matching as matching

    monkeypatch.setattr(matching, "_MATCH_WORK_PER_ITEM", 0)
    monkeypatch.setattr(matching, "_MATCH_WORK_FLOOR", 2)
    stored = [{"store_id": i, "keys": ["hard"] if i < 3 else ["easy"]} for i in range(4)]
    occurrences = [(i, "hard" if i < 3 else "easy") for i in range(5)]
    forbidden = {(0, 0), (0, 2), (1, 0), (3, 3)}
    skipped = set()
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
        result = matching._match_occurrences(stored, lambda r: r["keys"], occurrences,
                                             forbidden=forbidden, skipped=skipped)
    assert result == {0: 0, 1: 1, 2: 2, 3: 4}
    assert skipped == {(0, 0), (0, 2), (1, 0)}
    assert any("guard_skipped_budget" in r.getMessage() for r in caplog.records)


def test_seeded_cold_replay_properties(tmp_path, monkeypatch, caplog, record_property):
    """300 real-store cases, paired against off/base, with unchanged production caps."""
    rng = random.Random(1780032026)
    stats = []
    for trial in range(300):
        n = 1200 if trial == 299 else rng.randrange(256, 1201) if trial % 20 == 0 else rng.randrange(1, 65)
        cut = rng.randrange(1, n + 1)
        order = list(reversed(range(cut))) + list(range(cut, n))
        # Mix a partial reversal, missing keys, known keys and positively conflicting fresh keys.
        uids = [None if rng.random() < .2 else f"uid-{i}" for i in order]
        for j in rng.sample(range(n), rng.randrange(min(4, n) + 1)):
            uids[j] = f"fresh-replacement-{j}"
        uids += [f"fresh-tail-{i}" for i in range(rng.randrange(1, 4))]
        view = [message("shared", 99., **({"message_uid": uid} if uid else {})) for uid in uids]
        arm_counts = {}
        rejected = 0
        caplog.clear()
        for mode in ("off", "on"):
            path = tmp_path / f"case-{trial}-{mode}"
            (path / "home").mkdir(parents=True)
            with sqlite3.connect(path / "home/state.db") as db:
                db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
                db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
            monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
            original = seed(path, n)
            monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
            counts = []
            for turn in range(2):
                with context(path) as eng:
                    if mode == "on" and turn == 0:
                        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
                            plan = eng._identity_anchor_prematch(view, view, 0)
                        # Fresh UIDs positively conflict with every stored canonical UID.
                        # Count only those occurrences actually rejected by the guard.
                        rejected = sum(bool(uid and uid.startswith("fresh-")) and idx not in plan["matched"]
                                       for idx, uid in enumerate(uids))
                        if any("guard_skipped_budget" in r.getMessage() for r in caplog.records):
                            rejected = 0  # The single same-key group used base; the guard rejected nothing.
                    eng.ingest(deepcopy(view))
                    current = rows(eng)
                    assert current[:n] == original, (trial, mode, "stored occurrence lost")
                    assert Counter(r["content"] for r in current)["shared"] >= max(n, len(view))
                    counts.append(len(current))
            assert counts[1] == counts[0], (trial, mode, counts)
            arm_counts[mode] = counts
        assert arm_counts["on"][0] <= arm_counts["off"][0] + rejected, (trial, arm_counts, rejected)
        stats.append({"trial": trial, "n": n, "cut": cut, "base": arm_counts["off"][0],
                      "on": arm_counts["on"][0], "rejected": rejected,
                      "budget_skip": any("guard_skipped_budget" in r.getMessage() for r in caplog.records)})
    record_property("seeded_cases", json.dumps(stats, sort_keys=True))
    assert len(stats) == 300 and max(s["n"] for s in stats) == 1200
    assert any(s["budget_skip"] for s in stats)
