"""Background rollup writers require explicit test opt-in (#1084)."""
import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _scheduled(tmp_path, monkeypatch):
    scheduled = []
    monkeypatch.setattr(engine_module._ROLLUP_MAINTENANCE_SCHEDULER, "schedule",
                        lambda *args, **kwargs: scheduled.append(args[0]))
    config = LCMConfig(database_path=str(tmp_path / "rollups.db"))
    assert config.temporal_rollups_enabled is True
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("opt-in", platform="cli", conversation_id="opt-in")
        return scheduled
    finally:
        engine.shutdown()


def test_unmarked_default_bind_schedules_no_rollup(tmp_path, monkeypatch):
    assert _scheduled(tmp_path, monkeypatch) == []


@pytest.mark.background_rollups
def test_marked_default_bind_schedules_one_rollup(tmp_path, monkeypatch):
    assert len(_scheduled(tmp_path, monkeypatch)) == 1
