"""#2547 review r1 (F1): a store written at externalization off, replayed by an engine at externalization on.

The stored marker carries LCM-X's identity and generation lines. The upgrade replay's legacy keep must keep main's
compare (lines included): the replay rebuilds the host side with those lines and matches the stored row, so nothing is
stored again, whatever host stamps the replay carries. Main: 6 rows, 1 tool row in every cell.
"""

import sqlite3
import tempfile
from contextlib import closing
from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _marker

SESSION, CONV, CALL = "s-2547", "c-2547", "call_spill"
FULL = "NEEDLE:\n" + ("abcdef\n" * 400)


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _ingest(tmp_path, ext, history):
    home = tmp_path / "hermes"
    engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"),
                                        large_output_externalization_enabled=ext,
                                        large_output_externalization_path=str(home / "externalized")),
                       hermes_home=str(home))
    try:
        engine.on_session_start(SESSION, platform="cli", context_length=128000, conversation_id=CONV)
        engine.ingest(deepcopy(history))
    finally:
        engine.shutdown()


def _history(marker, extra=0):
    rows = [{"role": "user", "content": "dump it", "timestamp": 1762700001.0},
            {"role": "assistant", "content": "", "timestamp": 1762700002.0,
             "tool_calls": [{"id": CALL, "type": "function", "function": {"name": "dump", "arguments": "{}"}}]},
            {"role": "tool", "content": marker, "tool_call_id": CALL, "timestamp": 1762700003.0},
            {"role": "assistant", "content": "saved", "timestamp": 1762700004.0}]
    rows += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"next {i}", "timestamp": 1762700005.0 + i}
             for i in range(extra)]
    return rows


def _rows(tmp_path):
    with closing(sqlite3.connect((tmp_path / "hermes" / "lcm.db").as_uri() + "?mode=ro", uri=True)) as conn:
        return conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()


@pytest.mark.parametrize("stamps", ["all", "none", "user"])
@pytest.mark.parametrize("kind", ["spillover", "tempdir"])
def test_an_ext_off_store_replayed_at_ext_on_stores_nothing_twice(tmp_path, kind, stamps):
    path = (tmp_path / "hermes" / "cache" / "spillover" if kind == "spillover"
            else tmp_path / "tmp" / "hermes-results") / f"{CALL}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FULL.encode())
    marker = _marker(path, FULL)
    _ingest(tmp_path, False, _history(marker))
    first = _rows(tmp_path)
    if sum("[LCM persisted-output marker identity:" in (content or "") for _role, content in first) != 1:
        pytest.fail("setup: the ext-off store did not keep the marker with LCM-X's identity line")
    replay = _history(marker, extra=2)
    for row in replay:
        if stamps == "none" or (stamps == "user" and row["role"] != "user"):
            row.pop("timestamp")
    _ingest(tmp_path, True, replay)
    rows = _rows(tmp_path)
    assert (len(rows), sum(role == "tool" for role, _content in rows)) == (6, 1), f"first={len(first)}"
