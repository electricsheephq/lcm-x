"""Routine exit fits retain the log template at INFO; recovery fits remain WARNING."""
import logging

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("reason,level", [
    ("exit_fit:compressed", logging.INFO),
    ("recovery_attempt:locked", logging.WARNING),
])
def test_survival_fit_applied_log_level(tmp_path, caplog, reason, level):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine._survival_record(reason, 0, [], 900, 500, 600, False, "", warn_user=False)
        records = [r for r in caplog.records if r.getMessage().startswith("LCM survival fit applied (")]
        assert len(records) == 1
        assert records[0].levelno == level
        assert records[0].getMessage().startswith(f"LCM survival fit applied (reason={reason},")
    finally:
        engine.shutdown()
