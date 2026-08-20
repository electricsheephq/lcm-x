"""Host-writable contract for the engine's public compression-status properties.

The host does not only *read* these properties. ``agent/conversation_compression.py``
snapshots compressor attributes and restores them with a generic
``for name, value in restored.items(): setattr(compressor, name, value)`` loop.
Any public status property that is read-only therefore raises
``AttributeError: property ... has no setter`` on restore, which crashes context
compression rather than degrading it.

These tests pin the writability of the properties that participate in that
snapshot/restore round trip.
"""

from __future__ import annotations

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    return LCMEngine(config=LCMConfig(database_path=str(tmp_path / "status.db")))


def test_last_compression_status_is_writable(engine):
    engine.last_compression_status = "compressed"

    assert engine.last_compression_status == "compressed"
    assert engine._last_compression_status == "compressed"


def test_last_compression_status_accepts_the_empty_reset(engine):
    """The host clears the status before each compress() pass."""
    engine.last_compression_status = "compressed"

    engine.last_compression_status = ""

    assert engine.last_compression_status == ""


def test_last_compression_status_survives_a_generic_snapshot_restore(engine):
    """Reproduces the host's restore loop shape, not just a direct assignment.

    ``setattr`` on a read-only property is what actually raised in the host, so
    the regression has to go through ``setattr``/``getattr`` rather than the
    attribute syntax a reader might assume is equivalent.
    """
    engine.last_compression_status = "sanitized"
    snapshot = {"last_compression_status": getattr(engine, "last_compression_status")}

    engine.last_compression_status = "noop"
    for name, value in snapshot.items():
        setattr(engine, name, value)

    assert engine.last_compression_status == "sanitized"
