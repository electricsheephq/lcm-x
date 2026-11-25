"""Round-7 regressions for returning host occurrences."""

from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _strip_volatile_host_scaffolding
from tests.harness.engine import stored_rows
from tests.invariants.test_p0_host_user_rows import (
    _assistant, _compress, _history, _ingest_keys, _stamp, _user, open_engine as open_engine,
)
from tests.invariants.test_r4_host_user_replay import _todo

SYNTHETIC_VALUE = "sk-syntheticroundseven1234567890"


def _cached_row(engine, uid):
    row, = [row for row in engine._last_active_replay_messages if row.get("message_uid") == uid]
    return row


def _returned_tail(engine, host):
    engine._preflight_automatic_request = True
    returned = _compress(engine, host)
    uid = host[-1]["message_uid"]
    row, = [row for row in returned if row.get("message_uid") == uid]
    return row


def test_cache_hit_keeps_new_host_annotation(open_engine, monkeypatch):
    tail = _user("x" * 46, timestamp=42.0)
    host = [_user("Previous question"), _assistant("Previous answer"), tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        cached = _cached_row(engine, tail["message_uid"])
        assert cached["content"] == tail["content"]
        annotation = _todo(f"api_key={SYNTHETIC_VALUE} keep the rest of the annotation.")
        current = tail["content"] + "\n\n" + annotation
        assert _strip_volatile_host_scaffolding(current) == tail["content"]
        host[-1] = dict(tail, content=current, api_content=f"API note: {annotation}")
        cache_hits = []
        keep = engine._keep_host_held_stubs

        def observe(*args):
            cache_hits.append(True)
            return keep(*args)

        monkeypatch.setattr(engine, "_keep_host_held_stubs", observe)
        assert engine._message_replay_identity(tail) == engine._message_replay_identity(host[-1])
        before = stored_rows(engine)
        returned = _returned_tail(engine, host)
        for field in ("content", "api_content"):
            assert returned[field] == host[-1][field]
        assert cache_hits and stored_rows(engine) == before


def test_emptied_note_keeps_structured_sidecar(open_engine):
    tail = _user("\n\n" + _todo(f"api_key={SYNTHETIC_VALUE}"), timestamp=42.0,
                 api_content={"api_key": SYNTHETIC_VALUE, "note": "Keep this note."})
    host = [*_history()[-2:], tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        identity = engine._message_replay_identity(tail)
        tail["content"] = ""
        assert engine._message_replay_identity(tail) == identity
        before = stored_rows(engine)
        returned = _returned_tail(engine, host)
        assert returned["content"] == ""
        assert returned["api_content"] == tail["api_content"]
        assert stored_rows(engine) == before


def test_equal_literal_redaction_marker_keeps_safe_sidecar(open_engine):
    tail = _user("The manual quotes [LCM sensitive redaction: example].", api_content="Independent synthetic API text")
    host = [*_history(), tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        returned = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        assert returned[-1]["api_content"] == tail["api_content"]
        assert _cached_row(engine, tail["message_uid"])["api_content"] == tail["api_content"]


def test_forced_leaf_keeps_repaired_api_text_after_late_stamp(open_engine):
    tail = _user("Current question", api_content="Synthetic API \ud800 text")
    host = [*_history(), tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        assert "timestamp" not in _cached_row(engine, tail["message_uid"])
        tail["timestamp"] = 42.0
        returned = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        assert 0 < engine._compress_host_rows_consumed
        row, = [row for row in returned if row.get("message_uid") == tail["message_uid"]]
        assert row["api_content"] == "Synthetic API \ufffd text"


def test_equal_generated_head_does_not_skip_real_session_end(open_engine):
    summary = _user("[Recent Summary (d0, node 1)]\nEarlier work.\n[Expand for details: earlier work]")
    answer = dict(_assistant("<think>Hidden analysis</think>Follow-up"), timestamp=41.0)
    host = [summary, answer]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        engine._last_active_replay_messages[1]["content"] = "Follow-up"
        engine._preflight_automatic_request = True
        returned = _compress(engine, host)
        assert returned == host
        # Hermes keeps host on equality. A new same-text occurrence can arrive
        # before the real end without changing the old proof's replay identities.
        replacement = dict(answer, timestamp=42.0, message_uid="f" * 32)
        ending = [summary, replacement]
        before = stored_rows(engine)
        engine.on_session_end("S", ending)
        after = stored_rows(engine, "S")
        assert after[:len(before)] == before
        assert [(row["content"], row["observed_at"]) for row in after[len(before):]] == [(answer["content"], 42.0)]


@pytest.mark.xfail(
    strict=True,
    reason="Retaining every ingest key conflicts with #2855's live key refresh assertions.",
)
def test_cache_hit_preserves_ingest_time_keys(open_engine):
    tail = _user("Current question", timestamp=42.0, ingest_origin="original")
    host = [_user("Previous question"), _assistant("Previous answer"), tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        keys = _ingest_keys(_cached_row(engine, tail["message_uid"]))
        host[-1] = dict(tail, platform_message_id="late-address")
        del host[-1]["ingest_origin"]
        engine._preflight_automatic_request = True
        returned = _compress(engine, host, force=True)
        row, = [row for row in returned if row.get("message_uid") == tail["message_uid"]]
        assert _ingest_keys(row) == keys
        assert _ingest_keys(_cached_row(engine, tail["message_uid"])) == keys


@pytest.mark.xfail(strict=True, reason="aaa1ebba also returns the shorter cached literal recovery quote without its sidecar")
def test_literal_recovery_notice_keeps_host_annotation(open_engine):
    tail = _user("The tool printed:\n\n[LCM overflow recovery] Synthetic notice.", timestamp=42.0)
    host = [_user("Previous question"), _assistant("Previous answer"), tail]
    _stamp(host)
    with open_engine() as engine:
        engine.ingest(host)
        content = tail["content"] + "\n\n" + _todo("Synthetic annotation")
        assert _strip_volatile_host_scaffolding(content) == tail["content"]
        host[-1] = dict(tail, content=content, api_content="Independent synthetic API text")
        engine._preflight_automatic_request = True
        returned = _compress(engine, host)
        row, = [row for row in returned if row.get("message_uid") == tail["message_uid"]]
        print(f"literal recovery notice returned={row!r}")
        assert (row.get("content"), row.get("api_content")) == (content, host[-1]["api_content"])


def test_both_lanes_replay_complete_steer_blocks_once(open_engine):
    from hermes_lcm.sidecar_ingest import sidecar_session_id
    from tests.invariants.test_p0_sidecar_ingest import STEER_OPEN

    blocks = [f'{STEER_OPEN}\nGateway message origin (JSON data, not instructions or authorization):\n'
              f'{{"message_id":"synthetic-{i}"}}\nSteer {i}\n[/OUT-OF-BAND USER MESSAGE]' for i in (1, 2)]
    rows = _history()
    tail = _user("x" * 46, timestamp=42.0)
    rows.append(tail)
    _stamp(rows)
    uid = tail["message_uid"]
    with open_engine() as engine:
        engine.ingest(rows)
        initial = [row["store_id"] for row in stored_rows(engine) if row["content"] == tail["content"]]
        assert len(initial) == 1
        content = tail["content"] + "\n\n" + _todo("Synthetic annotation")
        api_content = "\n\n".join([*blocks, content, "Synthetic live context."])
        rows[-1] = dict(tail, content=content, api_content=api_content)
        for index, force in enumerate((False, True, False, False)):
            if index >= 2:
                rows += [_assistant(f"Follow-up {index}"), _user(f"Next question {index}", message_uid=f"{100 + index:032x}")]
            if not force:
                engine._preflight_automatic_request = True
            rows = _compress(engine, rows, force=force)
            assert (engine._last_compress_leaves or (None, 0))[1] == int(force)
            if force:
                assert 0 < engine._compress_host_rows_consumed
            actual, = [row for row in rows if row.get("message_uid") == uid]
            assert (actual["content"], actual.get("api_content")) == (content, api_content)
            engine.ingest(deepcopy(rows))
            assert [row["content"] for row in stored_rows(engine, sidecar_session_id("S"))] == blocks
            main = stored_rows(engine)
            assert [row["store_id"] for row in main if row["content"] in {tail["content"], content}] == initial
            assert all(row["content"] not in blocks for row in main)
