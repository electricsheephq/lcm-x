"""t5c review (compaction.py _compress_bypassed): a bypassed deterministic fallback mints an omitted-marker engine
uid into _engine_uids_pending; the pass must drop the uids IT minted (never recorded: bypass writes no LCM state)
and leave other pending entries alone."""

from __future__ import annotations

import pytest

from hermes_lcm import host_uid_emit as emit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_shadow import _m, _state_db


@pytest.fixture(autouse=True)
def capable(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)


def _rows(n=6):
    return [_m("user" if i % 2 == 0 else "assistant", f"row {i} " + "pad " * 50, 10.0 + i, f"host-{i}")
            for i in range(n)]


def _engine(tmp_path, session, **cfg):
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"), **cfg)
    engine = LCMEngine(config=config)
    engine._hermes_home = str(tmp_path)
    engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def _marker_uids(out):
    return [row["message_uid"] for row in out if str(row.get("content", "")).startswith("[Context omitted")]


def test_bypassed_child_fallback_leaves_no_pending_uid(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, "S")
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    sizes, minted = [], []
    try:
        for i in range(5):
            engine._mark_thread_context_stateless(f"child-{i}")
            try:
                out = engine.compress(_rows(), current_tokens=100_000, force=True)
            finally:
                engine._clear_thread_context_stateless(f"child-{i}")
            minted += _marker_uids(out)
            sizes.append(len(engine.__dict__.get("_engine_uids_pending") or {}))
        print(f"\nthread-context children: minted={len(minted)} distinct={len(set(minted))} pending sizes={sizes}")
        assert minted, "the deterministic fallback minted no marker uid (emission gate off?)"
        assert sizes == [0] * 5
    finally:
        engine.shutdown()


def test_bypassed_session_lineages_do_not_accumulate_pending(tmp_path, monkeypatch):
    ids = [f"eph-{i}" for i in range(5)]
    _state_db(tmp_path, [(sid, None, None) for sid in ids])
    engine = _engine(tmp_path, ids[0], stateless_session_patterns=["eph-*"])
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    sizes, minted = [], []
    try:
        for sid in ids:
            engine.on_session_start(sid, platform="cli", context_length=200_000, conversation_id="conv")
            assert engine._bypasses_lcm_context_management()
            out = engine.compress(_rows(), current_tokens=100_000, force=True)
            minted += _marker_uids(out)
            sizes.append(len(engine.__dict__.get("_engine_uids_pending") or {}))
        print(f"\nstateless bound sessions: minted={len(minted)} distinct={len(set(minted))} pending sizes={sizes}")
        assert len(set(minted)) == len(ids)
        assert sizes == [0] * len(ids)
    finally:
        engine.shutdown()


def test_bypassed_pass_keeps_other_pending_entries(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, "S")
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    try:
        engine._engine_uids_pending = {"foreground-uid": ("lineage", "carrier")}
        engine._mark_thread_context_stateless("child")
        try:
            out = engine.compress(_rows(), current_tokens=100_000, force=True)
        finally:
            engine._clear_thread_context_stateless("child")
        print(f"\nforeground entry kept: pending={dict(engine._engine_uids_pending)}")
        assert _marker_uids(out)
        assert engine._engine_uids_pending == {"foreground-uid": ("lineage", "carrier")}
    finally:
        engine.shutdown()


def test_nested_bypassed_pass_restores_the_outer_tracker(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, "S")
    inside = False

    class Native:
        def compress(self, rows, **kwargs):
            nonlocal inside
            inside = True
            try:
                assert _marker_uids(engine.compress(rows, current_tokens=100_000, force=True))
            finally:
                inside = False
            raise RuntimeError("outer native failure after the nested bypassed pass")

    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None if inside else Native())
    engine._mark_thread_context_stateless("child")
    try:
        out = engine.compress(_rows(), current_tokens=100_000, force=True)
        print(f"\nnested pass: marker={bool(_marker_uids(out))} pending={dict(engine._engine_uids_pending)}")
        assert _marker_uids(out)
        assert engine._engine_uids_pending == {}
    finally:
        engine._clear_thread_context_stateless("child")
        engine.shutdown()


class _ConcurrentDrop(dict):
    """The recorder's first pop lets a bypassed pass on another thread drop the other ``racing`` uids, after the
    recorder snapshotted the pending keys (compaction.py _compress_bypassed cleanup vs host_uid recorder)."""

    def __init__(self, *args, racing=()):
        super().__init__(*args)
        self.racing, self.raced = set(racing), False

    def pop(self, key, *default):
        if not self.raced and key in self.racing:
            self.raced = True
            for other in self.racing - {key}:
                dict.pop(self, other, None)
        return dict.pop(self, key, *default)


def test_recorder_publishes_when_a_bypassed_pass_drops_its_uid_meanwhile(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, "S")
    errors = []
    monkeypatch.setattr(engine, "_host_uid_count", lambda counts, exc=None: errors.append(type(exc).__name__))
    try:
        row = {"role": "user", "content": "synthetic foreground carrier"}
        engine._mint_engine_uids([(row, "summary", None, "carrier")])
        lineage, proof = engine._engine_uids_pending[row["message_uid"]]
        engine._engine_uids_pending = _ConcurrentDrop(
            {**engine._engine_uids_pending, "child-a": (lineage, proof), "child-b": (lineage, proof)},
            racing={"child-a", "child-b"})
        engine._host_uid_record_engine([row])
        bindings = engine._store.host_uid_bindings_for(lineage, {row["message_uid"]}, ("engine",))
        print(f"\nrecorder vs concurrent drop: raced={engine._engine_uids_pending.raced} errors={errors} "
              f"bindings={len(bindings)}")
        assert engine._engine_uids_pending.raced
        assert errors == []
        assert bindings
    finally:
        engine.shutdown()
