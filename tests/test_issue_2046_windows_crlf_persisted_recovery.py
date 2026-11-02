"""#2046: Windows persisted-output recovery accepts CRLF-translated spill files."""

import json
import tempfile
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import (
    recover_hermes_persisted_output,
    recover_hermes_persisted_output_with_file_stat,
)


FULL_OUTPUT = "WINDOWS_SPILLOVER_NEEDLE:\nline two\n" + ("é文abcdef\n" * 600)
SESSION_ID = "windows-crlf-spill-session"


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))


def _marker(path: Path, full: str = FULL_OUTPUT, *, count: int | None = None, preview: str | None = None) -> str:
    count = len(full) if count is None else count
    preview = full[:40] if preview is None else preview
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({count:,} characters, {count / 1024:.1f} KB).\n"
        f"Full output saved to: {path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )


def _write_windows_crlf(path: Path, full: str = FULL_OUTPUT) -> tuple[str, bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    windows_bytes = full.replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(windows_bytes)
    return _marker(path, full), windows_bytes


def _engine(tmp_path: Path) -> tuple[LCMEngine, Path]:
    output_dir = tmp_path / "externalized"
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(output_dir),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine._session_id = SESSION_ID
    return engine, output_dir


def _messages(marker: str, call_id: str = "call_win_spill") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "Calling",
            "tool_calls": [{"id": call_id, "function": {"name": "dump", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": marker},
    ]


@pytest.mark.parametrize("location", ["spillover", "legacy_tempdir"])
def test_recovers_windows_translated_crlf_file_against_canonical_marker_count(tmp_path, location):
    home = tmp_path / "hermes"
    target_dir = (
        home / "cache" / "spillover"
        if location == "spillover"
        else Path(tempfile.gettempdir()) / "hermes-results"
    )
    persisted_path = target_dir / "call_windows_crlf.txt"
    marker, windows_bytes = _write_windows_crlf(persisted_path)

    recovered = recover_hermes_persisted_output_with_file_stat(marker, hermes_home=home)

    assert recovered is not None
    content, file_stat = recovered
    assert content == FULL_OUTPUT
    assert file_stat["size"] == len(windows_bytes)
    assert len(windows_bytes) > len(FULL_OUTPUT.encode("utf-8"))


def test_recovery_preserves_intentional_crlf_when_exact_marker_contract_matches(tmp_path):
    home = tmp_path / "hermes"
    persisted_path = home / "cache" / "spillover" / "call_intentional_crlf.txt"
    persisted_path.parent.mkdir(parents=True, exist_ok=True)
    full_result = "INTENTIONAL_CRLF:\r\nline two\r\n" + ("row\r\n" * 50)
    persisted_path.write_bytes(full_result.encode("utf-8"))
    marker = _marker(persisted_path, full_result, preview=full_result[:32])

    assert recover_hermes_persisted_output(marker, hermes_home=home) == full_result


def test_recovers_windows_translation_of_intentional_crlf_without_weakening_preview(tmp_path):
    home = tmp_path / "hermes"
    persisted_path = home / "cache" / "spillover" / "call_translated_intentional_crlf.txt"
    full_result = "INTENTIONAL_TRANSLATED_CRLF:\r\nline two\r\n" + ("row\r\n" * 50)
    marker, _windows_bytes = _write_windows_crlf(persisted_path, full_result)

    assert recover_hermes_persisted_output(marker, hermes_home=home) == full_result


def test_fail_safe_rejects_crlf_spill_file_with_count_or_preview_mismatch(tmp_path):
    home = tmp_path / "hermes"
    persisted_path = home / "cache" / "spillover" / "call_mismatch_crlf.txt"
    _write_windows_crlf(persisted_path)

    assert recover_hermes_persisted_output(
        _marker(persisted_path, count=len(FULL_OUTPUT) + 1), hermes_home=home
    ) is None
    assert recover_hermes_persisted_output(
        _marker(persisted_path, preview="WRONG_PREVIEW:\nline two\n"), hermes_home=home
    ) is None


def test_ingest_and_replay_externalize_windows_crlf_spillover_losslessly(tmp_path):
    engine, output_dir = _engine(tmp_path)
    replay = None
    try:
        spill_file = tmp_path / "hermes" / "cache" / "spillover" / "call_win_spill.txt"
        marker, _windows_bytes = _write_windows_crlf(spill_file)
        messages = _messages(marker)

        engine._ingest_messages(messages)

        stored = engine._store.get_session_messages(SESSION_ID)
        assert len(stored) == 2
        assert stored[1]["content"].startswith("[Externalized tool output:")
        payload_files = list(output_dir.glob("*.json"))
        assert len(payload_files) == 1
        payload = json.loads(payload_files[0].read_text(encoding="utf-8"))
        assert payload["content"] == FULL_OUTPUT

        replay = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        replay._session_id = SESSION_ID
        replay._ingest_cursor_needs_reconcile = True
        replay._ingest_messages(messages)

        assert replay._store.get_session_count(SESSION_ID) == 2
        assert replay._store.get_session_messages(SESSION_ID) == stored
    finally:
        if replay is not None:
            replay.shutdown()
        engine.shutdown()
