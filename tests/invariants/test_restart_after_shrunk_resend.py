"""Pin the stored rows after a shrunk resend and a restart.

Views: [B0 B1], [B0], [B0 B0], restart, [B0 B1], for bare rows and for rows
with message UIDs only. Main stores two rows. A deferred #1525 candidate
re-stored the whole view on restart (five rows). The full-view variant that
reproduces on main is tracked in #1573. A future occurrence-key design may
change how the resent B0 is counted, but a restart must not store the view
again.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests import harness


@pytest.mark.parametrize("metadata", ["bare", "message_uid"])
def test_restart_after_shrunk_resend(tmp_path: Path, metadata: str) -> None:
    rows: list[dict[str, Any]] = [
        {"role": "user", "content": "request B0"},
        {"role": "assistant", "content": "reply B1"},
    ]
    if metadata == "message_uid":
        for index, row in enumerate(rows):
            row["message_uid"] = f"old-{index}"
    make_engine: Any = harness.make_engine
    engine = make_engine(tmp_path, window=128000, conversation_id="conv")
    try:
        for view in (rows, rows[:1], [rows[0], rows[0]]):
            engine.ingest(deepcopy(view))
        engine.shutdown()
        engine = make_engine(tmp_path, window=128000, conversation_id="conv")
        engine.ingest(deepcopy(rows))
        stored = engine._store.get_session_messages("S")
        assert [(row["role"], row["content"]) for row in stored] == [
            (row["role"], row["content"]) for row in rows]
    finally:
        engine.shutdown()
