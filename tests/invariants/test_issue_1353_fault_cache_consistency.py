"""I1/I2: fault retries keep accepted prefixes and paired proven replay caches."""

import sqlite3
from collections import Counter
from copy import deepcopy

import pytest
import hermes_lcm.engine as lcm_engine
from tests.test_host_uid_shadow import _engine, _m, _state_db
from tests.invariants.test_issue_1353_prematch_fault_prefix import (
    _base,
    _boom,
    _contents,
)


@pytest.mark.parametrize("stamped_orig", [False, True])
@pytest.mark.parametrize(
    "stage", ["none", "backfill-first", "prematch", "rewind", "commit"]
)
def test_compaction_prefix_fault_stage(tmp_path, monkeypatch, stage, stamped_orig):
    from tests.test_issue_436_identity_anchor import _engine as _e436, _rows, _turns

    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1),
    )
    engine = _e436(tmp_path)
    try:
        original = _turns(1, 8, 0.0)
        if not stamped_orig:
            for row in original:
                row.pop("timestamp")
        engine.ingest(original)
        before = _rows(engine)
        output = engine.compress(
            original, current_tokens=engine.threshold_tokens + 1, force=True
        )
        engine.on_session_end("S", original)
        engine.on_session_start(
            "S",
            boundary_reason="compression",
            old_session_id="S",
            platform="cli",
            context_length=200_000,
            conversation_id="conv",
        )
        host = deepcopy(output)
        if not stamped_orig:
            for index, row in enumerate(host):
                row["timestamp"] = 1000.0 + index
        host.append({"role": "user", "content": "brand new turn", "timestamp": 3000.0})
        f0 = engine._ingest_failure_count
        cached = engine._last_active_replay_source_identities
        ids = [engine._message_replay_identity(m, strip_carrier=False) for m in host]
        audit_from = next(
            (
                i
                for i in range(min(len(cached), engine._ingest_cursor, len(ids)))
                if ids[i] != cached[i]
            ),
            None,
        )
        with monkeypatch.context() as p:
            if stage == "backfill-first":
                calls = {"n": 0}
                real = engine._store.backfill_observed_at_many

                def first(*a, **k):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise sqlite3.OperationalError("bf")
                    return real(*a, **k)

                p.setattr(engine._store, "backfill_observed_at_many", first)
            elif stage == "prematch":
                p.setattr(engine, "_identity_anchor_prematch", _boom)
            elif stage == "rewind":
                p.setattr(engine, "_identity_anchor_version_rewind", _boom)
            elif stage == "commit":
                real_c = engine._identity_anchor_commit

                def after(plan, remainder_ids=None):
                    real_c(plan, remainder_ids)
                    if remainder_ids is None:
                        raise sqlite3.OperationalError("after commit")

                p.setattr(engine, "_identity_anchor_commit", after)
            cur0 = engine._ingest_cursor
            engine.ingest(deepcopy(host))
        fd = engine._ingest_failure_count - f0
        for _ in range(3):
            engine.ingest(deepcopy(host))
        expected = Counter((r["role"], r["content"]) for r in before)
        expected[("user", "brand new turn")] += 1
        got = Counter((r["role"], r["content"]) for r in _rows(engine))
        print(
            f"\nCP stamped={stamped_orig} stage={stage} cursor0={cur0} audit_from={audit_from} fdelta={fd} extra={dict(got - expected)} missing={dict(expected - got)}"
        )
        assert got == expected
        assert fd == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stamped_orig", [False, True])
@pytest.mark.parametrize("stage", ["none", "backfill-first"])
def test_cp_timing(tmp_path, monkeypatch, stage, stamped_orig):
    from tests.test_issue_436_identity_anchor import _engine as _e436, _rows, _turns

    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1),
    )
    engine = _e436(tmp_path)
    try:
        original = _turns(1, 8, 0.0)
        if not stamped_orig:
            for row in original:
                row.pop("timestamp")
        engine.ingest(original)
        before = _rows(engine)
        output = engine.compress(
            original, current_tokens=engine.threshold_tokens + 1, force=True
        )
        engine.on_session_end("S", original)
        engine.on_session_start(
            "S",
            boundary_reason="compression",
            old_session_id="S",
            platform="cli",
            context_length=200_000,
            conversation_id="conv",
        )
        host = deepcopy(output)
        if not stamped_orig:
            for index, row in enumerate(host):
                row["timestamp"] = 1000.0 + index
        host.append({"role": "user", "content": "brand new turn", "timestamp": 3000.0})
        f0 = engine._ingest_failure_count
        cached = engine._last_active_replay_source_identities
        ids = [engine._message_replay_identity(m, strip_carrier=False) for m in host]
        audit_from = next(
            (
                i
                for i in range(min(len(cached), engine._ingest_cursor, len(ids)))
                if ids[i] != cached[i]
            ),
            None,
        )
        with monkeypatch.context() as p:
            if stage == "backfill-first":
                calls = {"n": 0}
                real = engine._store.backfill_observed_at_many

                def first(*a, **k):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise sqlite3.OperationalError("bf")
                    return real(*a, **k)

                p.setattr(engine._store, "backfill_observed_at_many", first)
            elif stage == "prematch":
                p.setattr(engine, "_identity_anchor_prematch", _boom)
            elif stage == "rewind":
                p.setattr(engine, "_identity_anchor_version_rewind", _boom)
            elif stage == "commit":
                real_c = engine._identity_anchor_commit

                def after(plan, remainder_ids=None):
                    real_c(plan, remainder_ids)
                    if remainder_ids is None:
                        raise sqlite3.OperationalError("after commit")

                p.setattr(engine, "_identity_anchor_commit", after)
            cur0 = engine._ingest_cursor
            engine.ingest(deepcopy(host))
        fd = engine._ingest_failure_count - f0
        n_after_fault = len(_rows(engine))
        n_before = len(before)
        counts = [len(_rows(engine))]
        for _ in range(3):
            engine.ingest(deepcopy(host))
            counts.append(len(_rows(engine)))
        print(
            f"\nTIMING stage={stage} stamped={stamped_orig} before={n_before} after_fault={n_after_fault} after_retries={counts} fail_total={engine._ingest_failure_count} consec={engine._consecutive_ingest_failures}"
        )
        expected = Counter((r["role"], r["content"]) for r in before)
        expected[("user", "brand new turn")] += 1
        got = Counter((r["role"], r["content"]) for r in _rows(engine))
        print(
            f"\nCP stamped={stamped_orig} stage={stage} cursor0={cur0} audit_from={audit_from} fdelta={fd} extra={dict(got - expected)} missing={dict(expected - got)}"
        )
        assert got == expected
        assert fd == 0
    finally:
        engine.shutdown()


def test_revert_mechanism(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = [
            _m("user", "u one", 10.0),
            _m("assistant", "a one", 11.0),
            _m("user", "u two", 12.0),
            _m("assistant", "a two", 13.0),
        ]
        e.ingest(deepcopy(host))
        rew = deepcopy(host)
        rew[0] = _m("user", "u one REWRITTEN", 20.0)
        with monkeypatch.context() as p:
            p.setattr(e, "_identity_anchor_prematch", _boom)
            e.ingest(deepcopy(rew))
        print(
            "\nafter fault: cursor",
            e._ingest_cursor,
            "cached[0]",
            e._last_active_replay_messages[0],
        )
        real = e._host_occurrence_changed
        seen = []

        def spy(m, prev):
            r = real(m, prev)
            seen.append(
                (
                    m.get("content"),
                    m.get("timestamp"),
                    prev.get("content"),
                    prev.get("timestamp"),
                    r,
                )
            )
            return r

        monkeypatch.setattr(e, "_host_occurrence_changed", spy)
        e.ingest(deepcopy(host))  # host reverts
        rows = [
            (r["store_id"], r["content"], r.get("observed_at"))
            for r in e._store.get_session_messages("S")
        ]
        print("REVERT spy", seen[:3])
        print("REVERT rows", rows)
        assert [r[1] for r in rows].count("u one") == 1
    finally:
        e.shutdown()


def _rel(e):
    c = e._store._conn
    if not c.execute(
        "SELECT 1 FROM sqlite_master WHERE name='message_relations'"
    ).fetchone():
        return []
    return [
        tuple(r)
        for r in c.execute(
            "SELECT store_id, kind, related_store_id, ordinal, observed_at FROM message_relations ORDER BY relation_id"
        )
    ]


def _fault_on(patch, e, stage):
    """Fault one of the 4 try-block calls. 'commit' faults after the real first relation write."""
    if stage == "prematch":
        patch.setattr(e, "_identity_anchor_prematch", _boom)
    elif stage == "rewind":
        patch.setattr(e, "_identity_anchor_version_rewind", _boom)
    elif stage == "backfill":
        real = e._store.backfill_observed_at_many
        patch.setattr(e._store, "backfill_observed_at_many", _boom)
    elif stage == "commit":
        real = e._identity_anchor_commit

        def after(plan, remainder_ids=None):
            if remainder_ids is None:
                real(plan, remainder_ids)
                raise sqlite3.OperationalError("after commit")
            return real(plan, remainder_ids)

        patch.setattr(e, "_identity_anchor_commit", after)


@pytest.mark.parametrize("stage", ["prematch", "rewind", "commit", "backfill"])
@pytest.mark.parametrize("healthy", [1, 2, 3])
@pytest.mark.parametrize("tail", [False, True])
def test_b_fault_then_healthy(tmp_path, monkeypatch, stage, healthy, tail):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        e.ingest(deepcopy(host))
        host[0] = _m("user", "u one REWRITTEN", 20.0)
        f0 = e._ingest_failure_count
        with monkeypatch.context() as p:
            _fault_on(p, e, stage)
            e.ingest(deepcopy(host))
        fdelta = e._ingest_failure_count - f0
        if tail:
            host.append(_m("user", "u four", 21.0))
        for _ in range(healthy):
            e.ingest(deepcopy(host))
        c = _contents(e)
        print(
            "\nB",
            stage,
            healthy,
            tail,
            "fdelta",
            fdelta,
            "rows",
            len(c),
            Counter(c).most_common(2),
            "rel",
            len(_rel(e)),
        )
        assert c.count("u one REWRITTEN") == 1
        assert c.count("u four") == int(tail)
        assert len(c) == 7 + int(tail)
        r = _rel(e)
        assert len(r) == len(set(r))
        # stage backfill: backfill only runs on reconciled/compaction prefix -> no fault at all here
        if stage != "backfill":
            assert fdelta == 1
    finally:
        e.shutdown()


def test_c_revert_after_fault_active_replay(tmp_path, monkeypatch):
    """Host rewrites row 0 (fault), then reverts to the original. Is the returned active replay the host's row?"""
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        e.ingest(deepcopy(host))
        rewritten = deepcopy(host)
        rewritten[0] = _m("user", "u one REWRITTEN", 20.0)
        with monkeypatch.context() as p:
            p.setattr(e, "_identity_anchor_prematch", _boom)
            e._ingest_messages(deepcopy(rewritten))
        out_same = e._ingest_messages(deepcopy(host))  # reverted, same length
        out_tail = e._ingest_messages(deepcopy(host) + [_m("user", "u four", 21.0)])
        print(
            "\nREVERT same-len active[0]:",
            out_same[0],
            "\nREVERT tail active[0]:",
            out_tail[0],
        )
        print("rows", _contents(e))
        assert Counter(_contents(e)) == Counter(
            [m["content"] for m in host] + ["u four"]
        )
        assert out_same[0]["content"] == "u one", out_same[0]
        assert out_tail[0]["content"] == "u one", out_tail[0]
    finally:
        e.shutdown()


@pytest.mark.parametrize("fault", [False, True])
def test_unstamped_simple_revert(tmp_path, monkeypatch, fault):
    """Simplest: stamped base, rewrite row 0 content+stamp (fault), revert row 0 to old content WITHOUT stamp."""
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = [
            _m("user", "u one", 10.0),
            _m("assistant", "a one", 11.0),
            _m("user", "u two", 12.0),
            _m("assistant", "a two", 13.0),
        ]
        e.ingest(deepcopy(host))
        rew = deepcopy(host)
        rew[0] = _m("user", "u one REWRITTEN", 20.0)
        with monkeypatch.context() as p:
            if fault:
                p.setattr(e, "_identity_anchor_prematch", _boom)
            e._ingest_messages(deepcopy(rew))
        nxt = deepcopy(host)
        nxt[0].pop("timestamp")
        out = e._ingest_messages(nxt)
        rows = [r["content"] for r in e._store.get_session_messages("S")]
        print(f"\nSREVERT fault={fault} out0={out[0]} rows={rows}")
        assert out[0]["content"] == "u one"
    finally:
        e.shutdown()


@pytest.mark.parametrize("tail", [False, True])
def test_later_rewrite_survives_earliest_revert(tmp_path, monkeypatch, tail):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        original = _base()
        e.ingest(deepcopy(original))
        changed = deepcopy(original)
        changed[0] = _m("user", "first rewrite", 20.0)
        changed[2] = _m("user", "later rewrite", 22.0)
        if tail:
            changed.append(_m("user", "fresh tail", 23.0))
        with monkeypatch.context() as patch:
            patch.setattr(e, "_identity_anchor_prematch", _boom)
            e.ingest(deepcopy(changed))
        changed[0] = original[0]
        e.ingest(deepcopy(changed))
        e.ingest(deepcopy(changed))
        assert _contents(e).count("later rewrite") == 1
        assert _contents(e).count("u one") == 1
        assert Counter(_contents(e)) == Counter(
            [m["content"] for m in original]
            + ["later rewrite"]
            + (["fresh tail"] if tail else [])
        )
    finally:
        e.shutdown()
