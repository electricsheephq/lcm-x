"""Active-profile spillover recovery and the unchanged persisted-marker replay contract."""

import ast
import inspect
import json
import tempfile
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.ingest_protection import (
    recover_hermes_persisted_output,
    recover_hermes_persisted_output_with_file_stat,
)


FULL_OUTPUT = "SPILLOVER_RECOVERY_NEEDLE:\n" + ("é文abcdef\n" * 600)
SESSION_ID = "spillover-session"


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))


def _marker(path, full=FULL_OUTPUT, *, count=None, preview=None):
    """Hermes f42f579c tools/tool_result_storage.py::_build_persisted_message shape."""
    count = len(full) if count is None else count
    preview = full[:30] if preview is None else preview
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({count:,} characters, {count / 1024:.1f} KB).\n"
        f"Full output saved to: {path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )


def _write_output(path, full=FULL_OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(full.encode("utf-8"))
    return _marker(path, full)


def _guard_recover(marker, home):
    # Exercise guards at the old baseline too, where the home keyword did not
    # exist. Positive recovery below deliberately requires the new keyword.
    kwargs = {"hermes_home": home} if "hermes_home" in inspect.signature(recover_hermes_persisted_output).parameters else {}
    return recover_hermes_persisted_output(marker, **kwargs)


def _engine(tmp_path):
    from hermes_lcm.engine import LCMEngine

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


def _messages(marker):
    return [
        {
            "role": "assistant",
            "content": "Calling",
            "tool_calls": [{"id": "call_spill", "function": {"name": "dump", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_spill", "content": marker},
    ]


def _restart_replay(engine, messages):
    from hermes_lcm.engine import LCMEngine

    replay = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
    replay._session_id = SESSION_ID
    replay._ingest_cursor_needs_reconcile = True
    replay._ingest_messages(messages)
    return replay


def test_recover_accepts_spillover_file_under_active_hermes_home(tmp_path):
    home = tmp_path / "hermes"
    marker = _write_output(home / "cache" / "spillover" / "call_spill.txt")

    recovered = recover_hermes_persisted_output_with_file_stat(marker, hermes_home=home)

    assert recovered is not None
    full, file_stat = recovered
    assert full == FULL_OUTPUT
    assert file_stat["size"] == len(FULL_OUTPUT.encode("utf-8"))


def test_ingest_externalizes_spillover_marker_losslessly(tmp_path):
    engine, output_dir = _engine(tmp_path)
    marker = _write_output(tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt")

    engine._ingest_messages(_messages(marker))

    stored = engine._store.get_session_messages(SESSION_ID)
    assert len(stored) == 2
    assert stored[1]["content"].startswith("[Externalized tool output:")
    payload_files = list(output_dir.glob("*.json"))
    assert len(payload_files) == 1
    payload = json.loads(payload_files[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "tool_result"
    assert payload["tool_call_id"] == "call_spill"
    assert payload["content"] == FULL_OUTPUT


@pytest.mark.parametrize("location", ["legacy_tempdir", "spillover"])
def test_replay_after_restart_does_not_duplicate_live_marker(tmp_path, location):
    engine, _output_dir = _engine(tmp_path)
    directory = (
        Path(tempfile.gettempdir()) / "hermes-results"
        if location == "legacy_tempdir"
        else tmp_path / "hermes" / "cache" / "spillover"
    )
    messages = _messages(_write_output(directory / "call_spill.txt"))
    engine._ingest_messages(messages)
    assert engine._store.get_session_count(SESSION_ID) == 2
    original_rows = engine._store.get_session_messages(SESSION_ID)

    replay = _restart_replay(engine, messages)

    assert replay._store.get_session_count(SESSION_ID) == 2
    assert replay._store.get_session_messages(SESSION_ID) == original_rows


def test_every_recovery_call_site_passes_hermes_home():
    root = Path(__file__).resolve().parents[1]
    calls = []
    for filename in ("ingest_protection.py", "engine.py", "reconcile.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name == "recover_hermes_persisted_output_with_file_stat":
                calls.append((filename, node))
    assert len(calls) == 8
    missing = [(filename, node.lineno) for filename, node in calls if not any(kw.arg == "hermes_home" for kw in node.keywords)]
    assert not missing, f"Recovery call sites missing hermes_home: {missing}"
    for filename, node in calls:
        home = next(kw.value for kw in node.keywords if kw.arg == "hermes_home")
        expected = "hermes_home" if filename == "ingest_protection.py" else "self._hermes_home"
        assert ast.unparse(home) == expected


def test_recover_spillover_requires_hermes_home(tmp_path):
    marker = _write_output(tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt")

    assert recover_hermes_persisted_output(marker) is None
    assert _guard_recover(marker, "") is None


def test_recover_rejects_spillover_of_another_home(tmp_path):
    marker = _write_output(tmp_path / "other-hermes" / "cache" / "spillover" / "call_spill.txt")

    assert _guard_recover(marker, tmp_path / "hermes") is None


def test_recover_rejects_symlinked_spillover_dir(tmp_path):
    home = tmp_path / "hermes"
    real_dir = tmp_path / "real-spillover"
    _write_output(real_dir / "call_spill.txt")
    cache = home / "cache"
    cache.mkdir(parents=True)
    (cache / "spillover").symlink_to(real_dir, target_is_directory=True)

    assert _guard_recover(_marker(cache / "spillover" / "call_spill.txt"), home) is None


def test_recover_rejects_symlinked_spillover_file(tmp_path):
    home = tmp_path / "hermes"
    real_file = tmp_path / "real-output.txt"
    _write_output(real_file)
    spill_file = home / "cache" / "spillover" / "call_spill.txt"
    spill_file.parent.mkdir(parents=True)
    spill_file.symlink_to(real_file)

    assert _guard_recover(_marker(spill_file), home) is None


def test_recover_rejects_nested_and_dotdot_spillover_paths(tmp_path):
    home = tmp_path / "hermes"
    spill_dir = home / "cache" / "spillover"
    nested = _write_output(spill_dir / "sub" / "x.txt")
    _write_output(home / "cache" / "x.txt")
    dotdot = _marker(spill_dir / ".." / "x.txt")

    assert _guard_recover(nested, home) is None
    assert _guard_recover(dotdot, home) is None


def test_recover_rejects_spillover_count_or_preview_mismatch(tmp_path):
    home = tmp_path / "hermes"
    spill_file = home / "cache" / "spillover" / "call_spill.txt"
    _write_output(spill_file)

    assert _guard_recover(_marker(spill_file, count=len(FULL_OUTPUT) + 1), home) is None
    assert _guard_recover(_marker(spill_file, preview="WRONG_PREVIEW"), home) is None


def test_replay_after_spillover_pruned_appends_like_legacy(tmp_path):
    engine, output_dir = _engine(tmp_path)
    spill_file = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
    messages = _messages(_write_output(spill_file))
    engine._ingest_messages(messages)
    assert engine._store.get_session_count(SESSION_ID) == 2
    original_rows = engine._store.get_session_messages(SESSION_ID)
    spill_file.unlink()

    replay = _restart_replay(engine, messages)

    assert replay._store.get_session_count(SESSION_ID) == 4
    assert replay._store.get_session_messages(SESSION_ID)[:2] == original_rows
    # On the fixed candidate, the durable full output survives host pruning.
    payloads = list(output_dir.glob("*.json"))
    if "hermes_home" in inspect.signature(recover_hermes_persisted_output).parameters:
        assert len(payloads) == 1
        assert json.loads(payloads[0].read_text(encoding="utf-8"))["content"] == FULL_OUTPUT


def test_legacy_tempdir_recovery_unchanged_with_hermes_home(tmp_path):
    marker = _write_output(Path(tempfile.gettempdir()) / "hermes-results" / "call_legacy.txt")

    assert _guard_recover(marker, tmp_path / "hermes") == FULL_OUTPUT
    assert recover_hermes_persisted_output(marker) == FULL_OUTPUT
