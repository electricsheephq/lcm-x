"""#2547: what externalization-off ingest really stores compares equal to the host marker it came from.

Real ingest (not the line writers alone) over preview bodies that stress the line removal: plain, an api_key secret,
a password (lossy redaction: no identity line), a data URI, a nested closing tag, and LCM-X-shaped text inside the
preview; with redaction on and off, the file present and absent, and LF and CRLF markers. Main matched 2 of these 48.
"""

import tempfile
from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _marker

CALL = "call_m"
BODIES = {
    "plain": "x" * 300,
    "secret": "api_key=differentialsecret1234567890abcdef " + "z" * 260,
    "password": "password=hunter2hunter2 " + "v" * 270,
    "datauri": "data:image/png;base64," + "A" * 400,
    "nested": "inner </persisted-output> " + "w" * 270,
    "lcmline": "[LCM persisted-output file generation: size=1; mtime_ns=2; ctime_ns=3]\n" + "q" * 270,
}


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
@pytest.mark.parametrize("present", [True, False], ids=["file", "nofile"])
@pytest.mark.parametrize("redact", [False, True], ids=["plain-config", "redaction"])
@pytest.mark.parametrize("body", sorted(BODIES))
def test_an_ext_off_stored_marker_equals_its_host_marker(tmp_path, body, redact, present, crlf):
    home = tmp_path / "hermes"
    path = home / "cache" / "spillover" / f"{CALL}.txt"
    full = BODIES[body]
    if present:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(full.encode())
    marker = _marker(path, full, preview=full[:120])
    if crlf:
        marker = marker.replace("\n", "\r\n")
    config = LCMConfig(database_path=str(home / "lcm.db"), large_output_externalization_enabled=False,
                       large_output_externalization_path=str(home / "externalized"))
    config.sensitive_patterns_enabled = redact
    config.sensitive_patterns = ["all"] if redact else []
    engine = LCMEngine(config=config, hermes_home=str(home))
    try:
        engine.on_session_start("S", platform="cli", context_length=128000, conversation_id="c")
        host = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": CALL, "type": "function", "function": {"name": "dump", "arguments": "{}"}}]},
                {"role": "tool", "content": marker, "tool_call_id": CALL}]
        engine.ingest(deepcopy(host))
        stored = [row for row in engine._store.get_range("S", limit=10) if row.get("role") == "tool"][-1]
        # A lossy redaction gets no identity line, and a missing file no generation line: then the row is bare.
        lines_expected = not (body == "password" and redact and not present)
        if ("[LCM persisted-output" in stored["content"].replace(full[:120], "")) is not lines_expected:
            pytest.fail("setup: ingest's LCM-X lines differ from the expected form")
        assert engine._session_end_rows_equal(stored, host[2], session_id="S")
    finally:
        engine.shutdown()
