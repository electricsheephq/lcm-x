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


# Row counts main 779e9d18 stores on the same inputs (rows, tool rows). The upgrade replay's prefix binding (#2584) and
# its legacy keep read the stored marker with LCM-X's lines kept, so the candidate must store exactly these.
MAIN_COUNTS = {
    "ext-all-full": (6, 1), "ext-all-partial": (9, 2), "ext-none-full": (6, 1), "ext-none-partial": (12, 3),
    "ext-user-full": (6, 1), "ext-user-partial": (12, 3), "noext-all-full": (6, 1), "noext-all-partial": (11, 3),
    "noext-none-full": (16, 3), "noext-none-partial": (20, 5), "noext-user-full": (13, 3), "noext-user-partial": (19, 5),
}  # v0.29.2 261d41ea9 stores as many or more in every cell; the stampless ext-off replays duplicate there too.


# What a replay of rows already stored should leave: the same cells with a plain tool result store exactly these.
PLAIN = {"full": (6, 1), "partial": (9, 2)}
_KNOWN_2597 = {"noext-all-partial", "noext-none-full", "noext-none-partial", "noext-user-full", "noext-user-partial",
               "ext-none-partial", "ext-user-partial"}
_REASON_2597 = ("#2597 (P2, v0.30.2): a replay of a stored ext-off saved-output marker stores the replay again; "
                "main and v0.29.2 alike")
CELLS = [pytest.param(ext, stamps, shape, id=f"{'ext' if ext else 'noext'}-{stamps}-{shape}",
                      marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=_REASON_2597)]
                      if f"{'ext' if ext else 'noext'}-{stamps}-{shape}" in _KNOWN_2597 else [])
         for ext in (False, True) for stamps in ("all", "none", "user") for shape in ("full", "partial")]


@pytest.mark.parametrize("replay_ext,stamps,shape", CELLS)
def test_replays_of_an_ext_off_store_match_main(tmp_path, replay_ext, stamps, shape):
    path = tmp_path / "hermes" / "cache" / "spillover" / f"{CALL}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FULL.encode())
    marker = _marker(path, FULL)
    _ingest(tmp_path, False, _history(marker))
    replay = _history(marker, extra=2)
    if shape == "partial":  # the host re-sends from the call on and reuses the call id with the identical marker
        replay = replay[1:] + [dict(replay[1]), dict(replay[2], timestamp=1762700010.0), {
            "role": "assistant", "content": "saved again", "timestamp": 1762700011.0}]
    for row in replay:
        if stamps == "none" or (stamps == "user" and row["role"] != "user"):
            row.pop("timestamp", None)
    for _ in range(2):
        _ingest(tmp_path, replay_ext, replay)
    rows = _rows(tmp_path)
    got = (len(rows), sum(role == "tool" for role, _content in rows))
    key = f"{'ext' if replay_ext else 'noext'}-{stamps}-{shape}"
    if got != MAIN_COUNTS[key]:  # outside the known limit: never other than main
        pytest.fail(f"{key}: {got}, main {MAIN_COUNTS[key]}")
    assert got == PLAIN[shape]  # the #2597 known-limit cells fail here until a fix
