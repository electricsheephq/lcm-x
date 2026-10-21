"""Replay-safe matching budgets and the #1519 release-manager mode contract."""
from copy import deepcopy
from functools import lru_cache
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1519_alias_uid_conflict import (
    CASES, TEXT, bindings, deterministic_host, digest, message, prepare, rows, targets,
)

__all__ = ["deterministic_host"]
BASE_BYTES = json.loads((Path(__file__).parents[1] / "fixtures/issue_1519_observational_base.json").read_text())["cases"]


def test_production_cap_replay_stores_only_one_fresh_occurrence(tmp_path):
    """Adapt Astra's 256-row probe without importing its harness or changing caps."""
    n = 256
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest([message("shared", 1000 + i, message_uid=f"uid-{i}") for i in range(n)])
        eng._store.add_message_relations([[(r["store_id"], "alt_stamp", None, None, 99.)] for r in rows(eng)])
    view = [message("shared", 99., message_uid=f"uid-{i}") for i in reversed(range(n))]
    view.append(message("shared", 99., message_uid="uid-fresh"))
    counts = []
    for _ in range(3):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            counts.append(len(rows(eng)))
    assert counts == [257, 257, 257]


def test_exhausted_repair_keeps_every_original_allowed_occurrence(monkeypatch):
    from hermes_lcm import recognition_matching as matching

    # The first edge is forbidden. Repair would need to displace two held edges;
    # stop inside that path and require the original allowed assignment verbatim.
    monkeypatch.setattr(matching, "_MATCH_WORK_PER_ITEM", 0)
    monkeypatch.setattr(matching, "_MATCH_WORK_FLOOR", 2)
    stored = [{"store_id": i} for i in range(3)]
    result = matching._match_occurrences(stored, lambda r: ["k"], [(i, "k") for i in range(3)],
                                        forbidden={(0, 0), (0, 2), (1, 0)})
    assert result == {1: 1, 2: 2}


def test_allowed_matching_is_maximum_up_to_seven(caplog):
    """Independent subset-DP oracle, adapted from Astra's matching probe."""
    from hermes_lcm.recognition_matching import _match_occurrences

    rng = random.Random(17802026)
    for trial in range(3500):
        n, q, keys = 1 + trial % 7, rng.randrange(1, 8), range(rng.randrange(1, 5))
        stored = [{"store_id": i, "keys": {k for k in keys if rng.random() < .6}} for i in range(n)]
        occurrences = [(j, rng.choice(list(keys))) for j in range(q)]
        forbidden = {(i, j) for i in range(n) for j in range(q) if rng.random() < .35}

        @lru_cache(None)
        def maximum(i, used):
            if i == n:
                return 0
            return max([maximum(i + 1, used)] + [1 + maximum(i + 1, used | 1 << j)
                       for j, key in occurrences if not used >> j & 1 and key in stored[i]["keys"]
                       and (i, j) not in forbidden])

        base = _match_occurrences(stored, lambda r: r["keys"], occurrences)
        result = _match_occurrences(stored, lambda r: r["keys"], occurrences, forbidden=forbidden)
        assert len(result) == maximum(0, 0)
        assert len(set(result.values())) == len(result)
        assert all((sid, j) not in forbidden and dict(occurrences)[j] in stored[sid]["keys"]
                   for sid, j in result.items())
        if not any(pair in forbidden for pair in base.items()):
            assert list(result.items()) == list(base.items())
    assert not [r for r in caplog.records if "work budget" in r.message]


@pytest.mark.parametrize("mode", ["shadow", "off"])
@pytest.mark.parametrize("case", CASES)
def test_observational_modes_equal_base(tmp_path, monkeypatch, record_property, mode, case):
    # Seed existing bindings with the acting mode, then exercise the opt-out/default.
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        view = prepare(eng, case)
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    for turn in range(2):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            proof = {"rows": digest(rows(eng)), "bindings": digest([bindings(eng, r["store_id"]) for r in rows(eng)])}
            record_property("base_bytes", proof)
            assert proof == BASE_BYTES[f"{case}-{mode}"][turn]
            assert len(targets(eng)) == 1


def test_off_alias_path_has_zero_binding_selects(tmp_path, monkeypatch, record_property):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        sql = []
        eng._store._conn.set_trace_callback(sql.append)
        view = [message(TEXT, 30., message_uid="host-new")]
        eng._identity_anchor_prematch(view, view, 0)
        eng._store._conn.set_trace_callback(None)
        reads = [s for s in sql if s.lstrip().upper().startswith("SELECT") and "host_uid_bindings" in s]
        record_property("binding_selects", len(reads))
        assert reads == []


def test_shadow_records_would_reject_without_changing_decisions(tmp_path, monkeypatch, caplog):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        view = prepare(eng, "different_known_key")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
            eng.ingest(view)
        assert len(targets(eng)) == 1
        assert any("would reject" in r.message for r in caplog.records)


def test_alias_keys_are_computed_once_per_row(monkeypatch, record_property):
    from hermes_lcm.identity_anchor import IdentityAnchorMixin

    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    counts = []
    for n in (1000, 4000):
        stored = [{"store_id": i, "observed_at": 99.} for i in range(n)]
        eng = SimpleNamespace(_host_uid_lineage_key=lambda: ("synthetic-lineage", None), _store=SimpleNamespace(
            host_uid_uids_of_stores=lambda *a: {i: {f"old-{i}"} for i in range(n)},
            host_uid_bindings_for=lambda *a: {}))
        calls = 0

        def keys_of(row):
            nonlocal calls
            calls += 1
            # Bound the broken implementation's work before it materialises 16M pairs.
            assert calls <= n + 1
            return {row["store_id"]}

        forbidden = IdentityAnchorMixin._identity_anchor_alias_uid_keys(
            eng, stored, dict.fromkeys(range(n), 10.), keys_of, [(i, i) for i in range(n)],
            [{"message_uid": f"new-{i}"} for i in range(n)])
        assert forbidden == {(i, i) for i in range(n)}
        assert calls == n
        counts.append(calls)
    record_property("keys_of_calls", counts)
    assert counts == [1000, 4000]
