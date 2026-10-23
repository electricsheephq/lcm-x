"""Replay-safe matching budgets and the #1519 release-manager mode contract."""
from copy import deepcopy
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1519_alias_uid_conflict import (
    CASES, TEXT, V2_UID_CONFLICTS, bindings, deterministic_host, digest, message, prepare, rows, targets,
)

__all__ = ["deterministic_host"]
BASE_BYTES = json.loads((Path(__file__).parents[1] / "fixtures/issue_1519_observational_base.json").read_text())["cases"]


def assert_v2_conflict(eng, before, view):
    current = rows(eng)
    assert current[:len(before)] == before
    assert [(r["role"], r["observed_at"]) for r in current[len(before):]] == [
        ("user", view[0]["timestamp"]), ("assistant", 51.)]
    original, fresh = targets(eng)
    coverage = eng._occurrence_fact_coverage(
        eng._store.read_metadata_json("host_occurrence_accepted_view:parent"), "parent")
    facts = [eng._store.read_metadata_json(f"host_occurrence_row:parent:{r['store_id']}")
             for r in (original, fresh)]
    assert all(f["occurrence_version"] == 2 for f in facts)
    assert {original["store_id"], fresh["store_id"]} <= coverage
    assert facts[0]["message_uid"] != facts[1]["message_uid"] == view[0]["message_uid"]


def prepare_writer(eng, case, monkeypatch, legacy):
    from hermes_lcm.engine import LCMEngine

    with monkeypatch.context() as writer:
        if legacy:
            writer.setattr(LCMEngine, "_occurrence_batch_factory",
                           lambda self, batch: self._real_user_scaffold_metadata_rows)
        view = prepare(eng, case)
    if legacy:
        assert eng._store._conn.execute(
            "SELECT count(*) FROM metadata WHERE key LIKE 'host_occurrence_row:%' "
            "OR key LIKE 'host_occurrence_accepted_view:%'").fetchone()[0] == 0
        sid = targets(eng)[0]["store_id"]
        assert any(r["observed_at"] == 30. for r in eng._store.get_message_relations([sid], "alt_stamp"))
    return view


def test_production_cap_observes_conflicts_without_acting(tmp_path, monkeypatch, caplog):
    from tests.invariants.test_issue_1780_round3 import compare_replays

    n = 256
    view = [message("shared", 99., message_uid=f"uid-{i}") for i in reversed(range(n))]
    view.append(message("shared", 99., message_uid="uid-fresh"))
    counts, count = compare_replays(tmp_path, monkeypatch, caplog, n, view, turns=3)
    assert counts == [257, 257, 257]
    assert count > 0


@pytest.mark.parametrize("mode", ["shadow", "off", "on"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("legacy", [False, True], ids=["v2-writer", "base-writer"])
def test_observational_modes_equal_base(tmp_path, monkeypatch, record_property, mode, case, legacy):
    # Seed existing bindings, then exercise every observational mode.
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        view = prepare_writer(eng, case, monkeypatch, legacy)
        before = rows(eng)
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    accepted = None
    for turn in range(2):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            proof = {"rows": digest(rows(eng)), "bindings": digest([bindings(eng, r["store_id"]) for r in rows(eng)])}
            record_property("base_bytes", proof)
            if not legacy and mode != "off" and case in V2_UID_CONFLICTS:
                assert_v2_conflict(eng, before, view)
                assert len(targets(eng)) == 2
                if turn:
                    assert rows(eng) == accepted
                accepted = rows(eng)
            else:
                assert proof == BASE_BYTES[f"{case}-{'shadow' if mode == 'on' else mode}"][turn]
                assert len(targets(eng)) == 1
                if legacy:
                    plan = eng._resolve_occurrence_view(view)
                    assert 0 not in plan["proven"]
                    assert f"host_occurrence_row:parent:{targets(eng)[0]['store_id']}" not in plan["updates"]


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


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("legacy", [False, True], ids=["v2-writer", "base-writer"])
def test_observer_records_would_reject_without_changing_decisions(tmp_path, monkeypatch, caplog, mode, legacy):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        view = prepare_writer(eng, "different_known_key", monkeypatch, legacy)
        before = rows(eng)
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
            eng.ingest(view)
        assert len(targets(eng)) == (1 if legacy else 2)
        if not legacy:
            assert_v2_conflict(eng, before, view)
            accepted = rows(eng)
        assert any("would reject" in r.message for r in caplog.records)
    if not legacy:
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            assert rows(eng) == accepted


def test_alias_keys_are_computed_once_per_row(monkeypatch, record_property, caplog):
    from hermes_lcm.identity_anchor import IdentityAnchorMixin

    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    counts = []
    for n in (1000, 4000):
        stored = [{"store_id": i, "observed_at": 99.} for i in range(n)]
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE host_uid_bindings(store_id, uid, lineage_key, kind)")
        db.execute("CREATE INDEX idx_host_uid_store ON host_uid_bindings(store_id)")
        db.executemany("INSERT INTO host_uid_bindings VALUES (?, ?, 'synthetic-lineage', 'canonical')",
                       [(i, f"old-{i}") for i in range(n)])
        eng = IdentityAnchorMixin()
        eng._host_uid_lineage_key = lambda: ("synthetic-lineage", None)
        eng._store = SimpleNamespace(_conn=db, _host_uid_table_exists=lambda: True)
        sql = []
        db.set_trace_callback(sql.append)
        calls = 0

        def keys_of(row):
            nonlocal calls
            calls += 1
            # Bound the broken implementation's work before it materialises 16M pairs.
            assert calls <= n + 1
            return {row["store_id"]}

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
            result = IdentityAnchorMixin._observe_alias_uid_conflicts(
                eng, stored, dict.fromkeys(range(n), 10.), keys_of, [(i, i) for i in range(n)],
                [{"message_uid": f"new-{i}"} for i in range(n)])
        assert result is None
        assert [r.args for r in caplog.records if "would reject" in r.msg] == [(n,)]
        assert calls == n
        assert len([s for s in sql if s.startswith("SELECT")]) == 2
        db.close()
        counts.append(calls)
    record_property("keys_of_calls", counts)
    assert counts == [1000, 4000]
