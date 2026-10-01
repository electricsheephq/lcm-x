"""Issue #72: publish complete sidecars, leaving crash temps invisible and intact."""

import json
import os
from pathlib import Path

import pytest

from hermes_lcm import externalize
from hermes_lcm.config import LCMConfig
from hermes_lcm.ingest_protection import externalized_payload_stats


class _SimulatedKill(BaseException):
    """Bypass OSError cleanup to model abrupt process death."""


@pytest.fixture
def payload_store(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        large_output_externalization_enabled=True,
    )
    # Directory-creation fsyncs must finish before a test intercepts fsync.
    storage = externalize.resolve_large_output_storage_dir(config, hermes_home=str(tmp_path))
    return config, storage


def _first_write(entry, content, config, tmp_path, session_id="session-1"):
    kwargs = dict(config=config, hermes_home=str(tmp_path), session_id=session_id)
    if entry == "ingest":
        return externalize.externalize_ingest_payload(content, role="user", field_path="content", **kwargs)
    return externalize.maybe_externalize_payload(
        content, kind="tool_result", tool_call_id="call-1", role="tool", force=True, **kwargs
    )


def _assert_temp_only_first_write(entry, payload_store, tmp_path, monkeypatch):
    config, storage = payload_store
    created = []
    real_open = externalize.os.open

    def record_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            created.append(Path(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(externalize.os, "open", record_open)
    content = "complete synthetic sidecar payload"
    result = _first_write(entry, content, config, tmp_path)

    assert result is not None
    assert created
    assert result["path"] not in created
    assert all(path.name.endswith(".tmp") for path in created)
    assert result["path"].is_file()
    payload = json.loads(result["path"].read_text(encoding="utf-8"))
    assert payload["content"] == content
    assert payload["content_chars"] == len(content)
    assert payload["content_bytes"] == len(content.encode("utf-8"))
    assert payload["session_id"] == "session-1"
    assert payload["kind"] == ("ingest_payload" if entry == "ingest" else "tool_result")
    assert list(storage.glob("*.tmp")) == []


def test_first_ingest_write_never_creates_final_json_directly(payload_store, tmp_path, monkeypatch):
    _assert_temp_only_first_write("ingest", payload_store, tmp_path, monkeypatch)


def test_first_tool_result_write_never_creates_final_json_directly(payload_store, tmp_path, monkeypatch):
    _assert_temp_only_first_write("tool", payload_store, tmp_path, monkeypatch)


@pytest.mark.parametrize("entry", ["ingest", "tool"])
def test_kill_mid_first_write_leaves_no_final_json(entry, payload_store, tmp_path, monkeypatch):
    config, storage = payload_store
    content = "synthetic crash payload"

    def kill_fsync(_fd):
        raise _SimulatedKill()

    monkeypatch.setattr(externalize.os, "fsync", kill_fsync)
    with pytest.raises(_SimulatedKill):
        _first_write(entry, content, config, tmp_path)

    assert list(storage.glob("*.json")) == []
    assert externalized_payload_stats(config, str(tmp_path))["externalized_payload_count"] == 0
    assert externalize.find_externalized_payload_for_message(
        content,
        kind="ingest_payload" if entry == "ingest" else "tool_result",
        tool_call_id="" if entry == "ingest" else "call-1",
        role="user" if entry == "ingest" else "tool",
        session_id="session-1",
        config=config,
        hermes_home=str(tmp_path),
    ) is None
    # Crash leftovers remain in place but are invisible to published-payload readers.
    assert list(storage.glob("*.tmp"))


def test_publish_new_payload_refuses_existing_final_name(payload_store):
    _config, storage = payload_store
    path = storage / "x_payload_abc.json"
    path.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        externalize._publish_new_externalized_payload(path, {"content": "replacement"})

    assert path.read_text(encoding="utf-8") == "original"
    assert list(storage.glob("*.tmp")) == []


def test_reassign_not_blocked_by_stale_fixed_tmp_leftover(payload_store, tmp_path):
    config, storage = payload_store
    result = _first_write("ingest", "synthetic reassignment payload", config, tmp_path, session_id="old")
    assert result is not None
    legacy_tmp = storage / f"{result['path'].name}.tmp"
    legacy_tmp.write_text("garbage", encoding="utf-8")

    moved = externalize.reassign_externalized_payloads(
        "old", "new", config=config, hermes_home=str(tmp_path)
    )

    assert moved == 1
    assert json.loads(result["path"].read_text(encoding="utf-8"))["session_id"] == "new"
    assert legacy_tmp.read_text(encoding="utf-8") == "garbage"


def test_first_write_oserror_leaves_no_payload_or_tmp(payload_store, tmp_path, monkeypatch):
    config, storage = payload_store

    def fail_fsync(_fd):
        raise OSError("simulated durability failure")

    monkeypatch.setattr(externalize.os, "fsync", fail_fsync)
    result = _first_write("ingest", "synthetic inline fallback payload", config, tmp_path)

    assert result is None
    assert list(storage.glob("*.json")) == []
    assert list(storage.glob("*.tmp")) == []
