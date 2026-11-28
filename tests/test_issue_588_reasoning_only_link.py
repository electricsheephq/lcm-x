"""#588 (part of the v0.30.0 named limit): a byte-exact reasoning-only copy is linked, so no extra copy is silent."""

import contextlib
from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_1286_cleanup_rewrite_proof import SID
from tests.test_issue_2305_cursor0_replay_links import _counts, _ingest_copies, _pairs
from tests.test_issue_2305_cursor0_replay_links import host as host  # noqa: F401  (pytest fixture)


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False))
    engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=200_000)
    return engine


def test_unstamped_copy_set_links_every_extra_copy_including_reasoning_only(tmp_path, monkeypatch):
    """The #588 shape at the default config: every extra copy a fresh bind stores is linked."""
    monkeypatch.delenv("LCM_INGEST_LEDGER", raising=False)
    engine = _engine(tmp_path)
    view = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row-{i}"} for i in range(8)]
    view[3] = {"role": "assistant", "content": "<think>only reasoning here</think>"}
    engine.ingest(deepcopy(view))
    with getattr(engine._store, "ingest_atomic", contextlib.nullcontext)():
        for i in (1, 2):
            engine._store.append("S", deepcopy(view[i]), conversation_id="conv")
    engine.shutdown()
    engine = _engine(tmp_path)
    before = {row["store_id"] for row in engine._store.get_session_messages("S")}
    try:
        engine.ingest(deepcopy([*view, {"role": "user", "content": "new-1"}]))
        added = [row for row in engine._store.get_session_messages("S") if row["store_id"] not in before]
        linked = {sid for (sid,) in engine._store._conn.execute(
            "SELECT store_id FROM message_relations WHERE kind='possible_replay_of'")}
        assert [row["content"] for row in added if row["store_id"] not in linked] == ["new-1"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("text", ["", "   \n"])
def test_blank_rows_are_never_evidence(host, text):
    # Seed the blank row directly: ``host.turn(reply="")`` substitutes a default reply (review nit).
    host.live += [{"role": "user", "content": "say nothing", "timestamp": host.stamp()},
                  {"role": "assistant", "content": text}]
    host.engine.ingest(host.live)
    assert any(row["role"] == "assistant" and row["content"] == text
               for row in host.engine._store.get_session_messages(SID)), "the blank target must be stored"
    _ingest_copies(host, [{"role": "assistant", "content": text}])
    assert _pairs(host.engine) == []
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 1  # a candidate existed and stayed unlinked


def test_reasoning_only_copy_with_another_stamp_is_vetoed(host, monkeypatch):
    stamp = host.stamp()
    host.live += [{"role": "assistant", "content": "<think>plan</think>", "timestamp": stamp}]
    host.engine.ingest(host.live)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    _ingest_copies(host, [{"role": "assistant", "content": "<think>plan</think>", "timestamp": stamp + 10}])
    assert _pairs(host.engine) == []
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 1
