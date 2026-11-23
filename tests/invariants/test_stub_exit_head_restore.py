"""I1, I2, I11: a same-length stub-first exit preserves the host turn and stored row order."""

import logging
from copy import deepcopy
from dataclasses import replace

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.config import host_message_uid_mode
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_1116_stub_first_exit_commit_proof import (
    Host,
    WEDGE,
    committed_session,
    grow_over_threshold,
    host_for,  # noqa: F401
    stub_first_result,
)
from tests.test_refused_compaction_rollback import _drain_session_end_maintenance


@pytest.fixture(params=["unset", "on", "off"])
def uid_mode(request, monkeypatch):
    if request.param == "unset":
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)
    assert host_message_uid_mode() == ("shadow" if request.param == "unset" else request.param)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    return request.param


@pytest.fixture
def committed_stub_exit(uid_mode, host_for, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    host_config = Host._config
    monkeypatch.setattr(Host, "_config", lambda host: replace(
        host_config(host), embeddings_enabled=False, temporal_rollups_enabled=False
    ))
    host = host_for("carrier")
    home = host.tmp_path / "home"
    home.mkdir(exist_ok=True)
    _state_db(home, [(host.sid, None, None)])
    committed_session(host, "carrier")

    flush = _host_flush()
    host.live = flush(host.live)
    grow_over_threshold(host)
    result, original = stub_first_result(host, caplog)
    engine = host.engine
    assert engine._uid_recognition_enabled() == (uid_mode == "on")
    assert len(result) == len(original)
    assert engine._last_compress_leaves[1] == engine._compress_host_rows_consumed == 0
    assert engine._verified_lcm_summary_prefix_end(result[0]["content"])
    assert engine._generated_context_carrier_remainder(result[0])
    assert result[0] == original[0], "fix (a): row 0 equals the host's carrier byte for byte, timestamp included"
    assert engine._compress_commit_publication == "local"

    _drain_session_end_maintenance(engine)
    host.commit_in_place(result, original)
    assert engine._compress_commit_proof["end_consumed"]
    assert not engine._ingest_cursor_needs_reconcile
    return host, flush


def _rows(host):
    return host.engine._store.get_session_messages(host.sid, limit=100_000)


def _contents(rows):
    return [(row["role"], row["content"]) for row in rows]


@pytest.mark.parametrize("flushed", [False, True], ids=["direct", "flushed"])
def test_stub_exit_replay_stores_only_new_turn(committed_stub_exit, flushed, caplog):
    host, flush = committed_stub_exit
    engine = host.engine
    before = _rows(host)
    turn = [{"role": "user", "content": "new request after the stub-first exit"},
            {"role": "assistant", "content": "reply to the new request"}]
    live = deepcopy(host.live) + turn
    if flushed:
        live = flush(live)

    engine.ingest(live)

    after = _rows(host)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert len(added) == len(turn)
    assert _contents(added) == _contents(turn)
    assert not any(engine._verified_lcm_summary_prefix_end(row["content"]) for row in added
                   if row["role"] == "user")
    assert not engine._store.get_message_relations([row["store_id"] for row in added], "generated_unproven")
    engine.ingest(deepcopy(live))
    assert _rows(host) == after

    caplog.clear()
    engine.note_turn_complete()
    host.live = live
    host.compress()
    assert WEDGE not in caplog.text
    assert "selected leaf chunk lacks raw store lineage" not in caplog.text
    assert (engine._last_compress_leaves[1] > 0 or engine.last_compression_status == "sanitized"
            or (engine.last_compression_status == "noop" and engine.last_compression_noop_reason))
    assert _rows(host) == after


def test_stub_exit_replay_stores_a_different_head(committed_stub_exit):
    host, flush = committed_stub_exit
    before = _rows(host)
    live = flush(deepcopy(host.live) + [{"role": "user", "content": "another fresh request"},
                                       {"role": "assistant", "content": "another fresh reply"}])
    live[0]["content"] += "\n\nA distinct user amendment at the head."

    host.engine.ingest(live)

    after = _rows(host)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert _contents(added) == _contents([live[0], *live[-2:]])
    assert [row["store_id"] for row in host.engine._store.get_message_relations(
        [row["store_id"] for row in added], "generated_unproven"
    )] == [added[0]["store_id"]]
