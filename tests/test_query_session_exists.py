"""#1161: boolean session probes keep empty/rebound ingest behavior."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


def test_has_session_messages(tmp_path):
    store = MessageStore(tmp_path / "synthetic.db")
    try:
        assert store.has_session_messages("s") is False
        store.append("s", {"role": "user", "content": "synthetic turn"})
        assert store.has_session_messages("s") is True
        assert store.has_session_messages("unknown") is False
    finally:
        store.close()


@pytest.mark.parametrize("populated", [False, True])
def test_exists_probes_preserve_rebind_and_ingest(tmp_path, monkeypatch, populated):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "synthetic.db"),
                                      temporal_rollups_enabled=False))
    try:
        engine.on_session_start("s", platform="cli", context_length=200_000)
        if populated:
            engine.ingest([{"role": "user", "content": "first"}])
        probes = []
        original = engine._store.has_session_messages

        def probe(session):
            result = original(session)
            probes.append((session, result))
            return result

        monkeypatch.setattr(engine._store, "has_session_messages", probe)
        engine._schedule_ingest_cursor_reconciliation()
        assert engine._ingest_cursor_needs_reconcile is populated
        assert probes == [("s", populated)]
        engine._ingest_cursor_needs_reconcile = False
        probes.clear()
        loads = []
        monkeypatch.setattr(engine, "_load_generated_ignored_placeholder_hash_counts", lambda: loads.append(1) or {})
        messages = ([{"role": "user", "content": "first"}] if populated else [])
        engine.ingest([*messages, {"role": "assistant", "content": "next"}])
        assert probes == [("s", populated)]
        assert loads == ([] if populated else [1])
        assert engine._store.get_session_count("s") == (2 if populated else 1)
    finally:
        engine.shutdown()
