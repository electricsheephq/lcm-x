"""#1075: SQLite snapshot output names the separate payload directory."""

import pytest

from hermes_lcm import maintenance
from hermes_lcm.command import _fmt_size, handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import get_large_output_storage_dir


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    config.large_output_externalization_enabled = True
    config.fresh_tail_count = 3
    e = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    e._session_id = "live-session"
    e._session_platform = "telegram"
    e._conversation_id = "live-session"
    e._lifecycle.bind_session("live-session", conversation_id="live-session")
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    yield e
    e.shutdown()


def _payload_dir(engine):
    return get_large_output_storage_dir(engine._config, engine._hermes_home, create=False)


def test_backup_names_payload_directory(engine):
    path = _payload_dir(engine)
    path.mkdir(parents=True)
    payload = b'{"content": "large tool output"}'
    (path / "payload.json").write_bytes(payload)
    (path / "ignored.txt").write_text("ignored")
    (path / "nested.json").mkdir()
    (path / "nested.json" / "inner.json").write_text("ignored")

    output = handle_lcm_command("backup", engine)

    assert "status: ok" in output
    assert f"externalized_payload_dir: {path}" in output
    assert "externalized_payload_files: 1" in output
    assert f"externalized_payload_size: {_fmt_size(len(payload))}" in output
    assert "note: externalized payloads are not in the SQLite snapshot; copy externalized_payload_dir with the backup when moving it to another host" in output
    lines = output.splitlines()
    assert lines[lines.index(f"externalized_payload_dir: {path}") - 1].startswith("backup_size:")


def test_backup_missing_directory_stays_missing(engine):
    path = _payload_dir(engine)
    assert not path.exists()

    output = handle_lcm_command("backup", engine)

    assert f"externalized_payload_dir: absent ({path})" in output
    assert "externalized_payload_files:" not in output
    assert "externalized_payload_size:" not in output
    assert "note: externalized payloads" not in output
    assert not path.exists()


@pytest.mark.parametrize("post_backup_status", ["ok", "noop", "refused"])
def test_rotate_apply_names_payload_directory(engine, monkeypatch, post_backup_status):
    for index in range(10):
        engine._store.append(engine._session_id, {"role": "user", "content": f"message-{index}"}, source="test")
    engine._store.commit()
    if post_backup_status != "ok":
        original = engine.rotate_active_session

        def rotate(*, apply):
            result = original(apply=False)
            if apply:
                result.update(ok=post_backup_status != "refused", noop=True, reason="test")
            return result

        monkeypatch.setattr(engine, "rotate_active_session", rotate)

    output = handle_lcm_command("rotate apply", engine)

    assert f"status: {post_backup_status}" in output
    assert "rotate_backup_path:" in output
    assert f"externalized_payload_dir: absent ({_payload_dir(engine)})" in output


def test_inventory_resolution_error_is_unavailable(engine, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("outside allowed base")

    monkeypatch.setattr(maintenance, "get_large_output_storage_dir", fail)

    assert maintenance.externalized_payload_inventory(engine) == {"path": None, "error": "outside allowed base"}
    output = handle_lcm_command("backup", engine)
    assert "status: ok" in output
    assert "externalized_payload_dir: unavailable (outside allowed base)" in output
    assert "externalized_payload_files:" not in output


def test_rotate_preflight_noop_names_payload_directory(engine, monkeypatch):
    # The idempotent no-op returns before any backup; it still names the payload directory (#1077 review).
    original = engine.rotate_active_session

    def rotate(*, apply):
        result = original(apply=False)
        result.update(ok=True, noop=True, reason="test")
        return result

    monkeypatch.setattr(engine, "rotate_active_session", rotate)
    output = handle_lcm_command("rotate apply", engine)

    assert "rolling backup was not written" in output
    assert f"externalized_payload_dir: absent ({_payload_dir(engine)})" in output


def test_unreadable_payload_entries_are_reported(engine, monkeypatch):
    # An entry whose metadata cannot be read is reported, and keeps the copy note (#1077 review).
    path = get_large_output_storage_dir(engine._config, engine._hermes_home, create=True)
    (path / "unreadable_abcdef123456_x.json").write_text("{}")
    real_scandir = maintenance.os.scandir

    class Entry:
        def __init__(self, entry):
            self.name = entry.name

        def is_file(self, follow_symlinks=True):
            raise OSError("stale handle")

    class Entries:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            return [Entry(e) for e in self.inner.__enter__()]

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(maintenance.os, "scandir", lambda p: Entries(real_scandir(p)))
    output = handle_lcm_command("backup", engine)

    assert "externalized_payload_files: 0" in output
    assert "externalized_payload_unreadable: 1 (not counted above)" in output
    assert "note: externalized payloads are not in the SQLite snapshot" in output


def test_scan_error_keeps_the_resolved_path(engine, monkeypatch):
    # A directory-scan failure still names the payload directory (#1077 review).
    path = get_large_output_storage_dir(engine._config, engine._hermes_home, create=True)

    def deny(p):
        raise PermissionError("permission denied")

    monkeypatch.setattr(maintenance.os, "scandir", deny)
    output = handle_lcm_command("backup", engine)

    assert f"externalized_payload_dir: {path}" in output
    assert "externalized_payload_scan: unavailable (permission denied)" in output
    assert "note: externalized payloads are not in the SQLite snapshot" in output
