"""T1-604b carrier boundaries, ignore parity, and failure handling."""

import sqlite3
import time
from contextlib import ExitStack
from datetime import datetime, timezone

import pytest

from hermes_lcm.command import handle_lcm_command
from tests.harness.engine import engine_context
from tests.invariants.test_p0_sidecar_ingest import SESSION, Y, _carrier, _engine, _rows, _user, host_uid
from tests.invariants.test_p0_sidecar_ingest_r2 import (
    CONVERSATION,
    NEIGHBOUR,
    _block,
    _copies,
    _host_sql_rows,
    _rebind,
)
from tests.test_compression_boundary import _compacted_engine


@pytest.mark.parametrize("prefix", [
    "[Triggering message id: `9876543210` — use as `message_id` for reply/react/pin via the discord tools.]\n\n",
    "[Tue 2026-04-28 13:40:53 UTC] ",
], ids=["discord-note", "rendered-timestamp"])
def test_prefixed_carrier_does_not_store_injection_quote(tmp_path, prefix):
    block = _block()
    injection = f"<lcm-answer-brief>Exact evidence:\n{block}\n</lcm-answer-brief>"
    with _engine(tmp_path) as engine:
        engine.ingest([_user(NEIGHBOUR, api_content=prefix + NEIGHBOUR + "\n\n" + injection)])
        assert _rows(engine) == [("user", NEIGHBOUR)]


def test_only_blocks_before_hook_context_are_recovered(tmp_path):
    first, quoted = _block(), _block(message_id="synthetic-quoted-message")
    sidecar = f"{first}\n\n{NEIGHBOUR}\n\n<lcm-answer-brief>{quoted}</lcm-answer-brief>"
    with _engine(tmp_path) as engine:
        engine.ingest([_user(NEIGHBOUR, api_content=sidecar)])
        assert _copies(engine, first) == 1
        assert _copies(engine, quoted) == 0


def test_empty_carrier_does_not_hide_block_after_blank_line(tmp_path):
    block = _block()
    view = [_user("", api_content="\n\n" + block)]
    with _engine(tmp_path) as engine:
        engine.ingest(view)
        engine.ingest(view)
        assert _copies(engine, block) == 1


def test_recovered_block_obeys_host_ignore_message_patterns(tmp_path):
    ignored = _block("Synthetic DROP_STEER_BODY preference.")
    allowed = _block("Synthetic retained preference.", message_id="synthetic-allowed-message")
    for route in ("host", "sidecar"):
        with engine_context(
            tmp_path / route, session_id=SESSION, conversation_id=CONVERSATION, window=100_000,
            embeddings_enabled=False, ignore_message_patterns=["DROP_STEER_BODY"],
        ) as engine:
            view = ([_user(ignored), _user(allowed)] if route == "host" else
                    [_carrier(api_content=ignored + "\n\n" + allowed)])
            engine.ingest(view)
            assert _copies(engine, allowed) == 1
            assert _copies(engine, ignored) == 0, route


def _fail_sidecar_receipts(engine, patch, error):
    write = engine._store.write_metadata_json

    def failing_write(keys, *args, **kwargs):
        if keys[0].startswith("sidecar-message:v1:"):
            raise error
        return write(keys, *args, **kwargs)

    patch.setattr(engine._store, "write_metadata_json", failing_write)


def test_status_shows_only_count_and_time_after_failures(tmp_path, monkeypatch):
    block = _block()
    view = [_carrier(api_content=block)]
    with _engine(tmp_path) as engine:
        assert "sidecar_recovery_" not in handle_lcm_command("status", engine)
        with monkeypatch.context() as patch:
            _fail_sidecar_receipts(engine, patch, RuntimeError(Y))
            engine.ingest(view)
            assert "sidecar_recovery_failure_count: 1" in handle_lcm_command("status", engine).splitlines()
            before = int(time.time())
            engine.ingest(view)
            after = int(time.time())
        status = handle_lcm_command("status", engine)
        details = dict(line.split(": ", 1) for line in status.splitlines() if line.startswith("sidecar_recovery_"))
        assert set(details) == {"sidecar_recovery_failure_count", "sidecar_recovery_last_failure_at"}
        assert details["sidecar_recovery_failure_count"] == "2"
        stamp = datetime.fromisoformat(details["sidecar_recovery_last_failure_at"])
        assert stamp.tzinfo == timezone.utc
        assert before <= stamp.timestamp() <= after
        assert Y not in status
        engine.ingest(view)
        assert _copies(engine, block) == 1
        assert "sidecar_recovery_failure_count: 2" in handle_lcm_command("status", engine).splitlines()


@pytest.mark.parametrize("compaction_end", [True, False], ids=["compaction-commit", "off-current"])
def test_session_end_sidecar_busy_retries_once(tmp_path, monkeypatch, caplog, compaction_end):
    block = _block()
    with ExitStack() as stack:
        if compaction_end:
            engine, original, _ = _compacted_engine(tmp_path, monkeypatch)
            stack.callback(engine.shutdown)
            ended = "S0"
            late = [row.copy() for row in original]
            next(row for row in late if row["role"] == "user")["api_content"] = block
        else:
            engine = stack.enter_context(_engine(tmp_path))
            ended = SESSION
            row = _carrier()
            late = [dict(row, api_content=block), {"role": "assistant", "content": "Synthetic final answer."}]
            engine.ingest([row])
            _rebind(engine, "foreground", "new-conversation")
        before = _host_sql_rows(engine, ended)
        busy = sqlite3.OperationalError("database is locked")
        busy.sqlite_errorcode = sqlite3.SQLITE_BUSY
        with monkeypatch.context() as patch:
            _fail_sidecar_receipts(engine, patch, busy)
            engine.on_session_end(ended, late)
        durable = _host_sql_rows(engine, ended)
        if compaction_end:
            assert engine._compress_commit_proof["end_consumed"] is True
            assert durable == before
        else:
            assert len(durable) == 2 and durable[:1] == before
        assert _copies(engine, block) == 0
        assert "LCM sidecar recovery failed" in caplog.text
        _rebind(engine, "post-retry", "post-retry-conversation")
        assert _copies(engine, block) == 1
        assert _host_sql_rows(engine, ended) == durable
        _rebind(engine, "second-retry", "second-retry-conversation")
        assert _copies(engine, block) == 1
