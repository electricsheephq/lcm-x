"""#1815 shapes 1-2: a faulted ingest never stores a row twice, and the next healthy ingest catches up.

The issue's acceptance: the durable rows after the next healthy ingest equal those of a fault-free run. A fault in the
identity-anchor prematch or the stamp backfill now aborts the whole ingest transaction (design §3.2), so the retry
classifies again from unchanged evidence.
"""

from copy import deepcopy
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _m(role, content, stamp):
    return {"role": role, "content": content, "timestamp": stamp}


BASE = [_m("user", "first question", 1.), _m("assistant", "first answer", 2.),
        _m("user", "second question", 3.), _m("assistant", "second answer", 4.)]


def _shape(name):
    if name == "1-prefix-insert":
        # A stamped user row inserted before the last row.
        return [*BASE[:3], _m("user", "inserted question", 3.5), BASE[3]]
    # 2: row 0's stamp changes and row 3's content changes in the same view.
    return [{**BASE[0], "timestamp": 1.5}, *BASE[1:3], {**BASE[3], "content": "second answer, edited"}]


def _engine(path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    engine = LCMEngine(config=LCMConfig(
        database_path=str(path), temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False))
    engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=200_000)
    return engine


def _rows(engine):
    return [(r["role"], r["content"], r["observed_at"]) for r in engine._store.get_session_messages("S")]


def _run(tmp_path, monkeypatch, shape, fault, *, persistent=False):
    engine = _engine(tmp_path / f"{fault}.db", monkeypatch)
    try:
        engine.ingest(deepcopy(BASE))
        view = _shape(shape)
        fired, healthy = [], []
        if fault != "none":
            target = engine if fault == "prematch" else engine._store
            name = "_identity_anchor_prematch" if fault == "prematch" else "backfill_observed_at_many"
            original = getattr(target, name)

            def once(*args, **kwargs):
                if (persistent and not healthy) or not fired:
                    fired.append(1)
                    raise sqlite3.OperationalError("disk I/O error")
                return original(*args, **kwargs)
            monkeypatch.setattr(target, name, once)
            engine.ingest(deepcopy(view))  # the faulted ingest
            healthy.append(1)
        engine.ingest(deepcopy(view))  # the next healthy ingest
        engine.ingest(deepcopy([*view, _m("user", "next question", 9.)]))
        return _rows(engine), fired
    finally:
        engine.shutdown()


@pytest.mark.parametrize("fault", ["prematch", "backfill"])
@pytest.mark.parametrize("shape", ["1-prefix-insert", "2-stamp-and-content"])
def test_faulted_ingest_matches_the_fault_free_run(tmp_path, monkeypatch, shape, fault):
    oracle, _ = _run(tmp_path, monkeypatch, shape, "none")
    faulted, fired = _run(tmp_path, monkeypatch, shape, fault)
    assert fired, f"the {fault} seam never ran for this shape"
    assert faulted == oracle, (faulted, oracle)
    assert len({(role, content) for role, content, _ in faulted if role == "assistant"}) == sum(
        role == "assistant" for role, _, _ in faulted), "an assistant row is stored twice"


@pytest.mark.parametrize("fault", ["prematch", "backfill"])
def test_a_fault_that_persists_still_falls_open_and_loses_nothing(tmp_path, monkeypatch, fault):
    """The retry is once only: a fault on both attempts takes the ordered-prefix path as before (a visible #1815
    limit), and after the next healthy ingest every row of the view is stored."""
    faulted, fired = _run(tmp_path, monkeypatch, "1-prefix-insert", fault, persistent=True)
    assert len(fired) >= 2, "the retry did not run"
    stored = [(role, content) for role, content, _ in faulted]
    assert all((row["role"], row["content"]) in stored for row in _shape("1-prefix-insert"))
