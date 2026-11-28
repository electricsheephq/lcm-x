"""Natural re-ownership and mixed-window replay (#1250, I1/I2/I5/I14)."""

import json

import pytest

from hermes_lcm import engine as engine_module, message_patterns
from tests.engine_helpers import _FakeTimeoutRegexEngine
from tests.invariants.test_issue_1250_publication_reown import (
    EARLY, GATEWAY, LATER, SESSION, _bind_gateway, _digests, _hook_for, _rows,
    open_engine as _open_engine,
)

open_engine = _open_engine
pytestmark = pytest.mark.issue(1250)
REOWN_KEY = f"reowned_session_rows:{SESSION}"


@pytest.fixture
def differential_base(request):
    return request.path.name.startswith("test_zz_1250_r7_")


def _start(writer):
    writer.on_session_start(SESSION, platform="gui", conversation_id=SESSION, context_length=200_000)


def _step(engine, name, view, action, new=(), *, copies=(), may_reown=False):
    before_rows = _rows(engine)
    before = _digests(before_rows)
    expected = before + _digests(new)
    result = action()
    after_rows = _rows(engine)
    after = _digests(after_rows)
    delta = after - expected
    print(json.dumps({
        "step": name, "binding": engine._conversation_id, "rows": [before.total(), after.total()],
        "view": [[*key, count] for key, count in sorted(_digests(view).items())],
        "stored": [[*key, count] for key, count in sorted(after.items())],
        "extra": [[*key, count] for key, count in sorted(delta.items())],
    }))
    assert after == expected + _digests(copies)
    after_by_id = {row["store_id"]: row for row in after_rows}
    for row in before_rows:
        current = after_by_id[row["store_id"]]
        assert row == (dict(current, conversation_id=row["conversation_id"]) if may_reown else current)
    return result


def _mixed(open_engine, *, sequential=False):
    writer, gateway = open_engine(), open_engine()
    _start(writer)
    _step(writer, "early ingest", EARLY, lambda: writer.ingest(list(EARLY)), EARLY)
    if sequential:
        writer.shutdown()
    _bind_gateway(gateway)
    full = EARLY + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    before = _rows(gateway)
    gateway_view = _step(gateway, "gateway compress", full, lambda: gateway.compress(full), may_reown=True)
    expected_ids = ([row["store_id"] for row in before if row["conversation_id"] == SESSION]
                    if gateway.last_compression_status == "compacted" else None)
    receipt = gateway._store.read_metadata_json(REOWN_KEY)
    assert (sorted(receipt) if receipt is not None else None) == expected_ids
    owners = [GATEWAY] * len(full) if expected_ids else [SESSION] * len(EARLY) + [GATEWAY] * len(LATER)
    assert [row["conversation_id"] for row in _rows(gateway)] == owners
    return writer, gateway, gateway_view


@pytest.mark.parametrize("restarts,plain", [(1, False), (2, False), (1, True)],
                         ids=["a", "b", "plain-ingest"])
def test_raw_replay_after_natural_stall(open_engine, differential_base, restarts, plain):
    writer, gateway, gateway_view = _mixed(open_engine)
    writer_view = list(EARLY)
    for i in range(3):
        writer_view = _step(writer, f"writer compress {i + 1}", writer_view,
                            lambda: writer.compress(writer_view))
    assert (gateway.last_compression_status == "error") is differential_base
    stalled, raw = (gateway, gateway_view) if differential_base else (writer, writer_view)
    binding = stalled._conversation_id
    assert raw == (EARLY + LATER if differential_base else EARLY)
    for i in range(restarts):
        _step(stalled, f"start {i + 1}", [], lambda: stalled.on_session_start(
            SESSION, conversation_id=binding, context_length=200_000))
        assert stalled._conversation_id == binding
        if plain:
            _step(stalled, "plain ingest", raw, lambda: stalled.ingest(raw))
        else:
            _step(stalled, f"raw compress {i + 1}", raw, lambda: stalled.compress(raw, force=True))


def test_restarted_window_with_self_stored_and_new_rows(open_engine, differential_base):
    writer, gateway, _ = _mixed(open_engine)
    assert (gateway.last_compression_status == "error") is differential_base
    held = [dict(role="assistant", content="Reply stored by the session window during the stall."),
            dict(role="user", content="Request stored by the session window during the stall.")]
    new = [dict(role="assistant", content="New reply after restart."),
           dict(role="user", content="New request after restart.")]
    # Keep the raw snapshot to model the restarted host's transcript. The live
    # engine adopts the compression result when it adds the turns during the stall.
    view = _step(writer, "writer compress", EARLY, lambda: writer.compress(list(EARLY)))
    view += held
    _step(writer, "self-stored turns", view, lambda: writer.ingest(view), held)
    _step(writer, "same-binding start", [], lambda: _start(writer))
    raw = EARLY + held + new
    # On GA the G publication refuses and S adopts a compacted view. Resuming
    # S with its raw transcript still copies the six previously stored rows.
    view = _step(writer, "mixed raw compress", raw, lambda: writer.compress(raw, force=True), new,
                 copies=EARLY + held if differential_base else ())
    _step(writer, "adopted view", view, lambda: writer.ingest(view))


def test_cold_post_turn_hook_after_gateway_publication(open_engine, differential_base):
    _, gateway, _ = _mixed(open_engine, sequential=True)
    assert (gateway.last_compression_status == "error") is differential_base
    gateway.shutdown()
    full = EARLY + LATER
    for i in range(3):
        cold = open_engine()
        assert not cold._session_id
        hook = _hook_for(cold)
        turn = [dict(role="assistant", content=f"Cold reply {i + 1}."),
                dict(role="user", content=f"Cold follow-up {i + 1}.")]
        full += turn
        _step(cold, f"cold hook {i + 1}", full,
              lambda: hook(session_id=SESSION, platform="gui", conversation_history=full), turn)
        assert cold._session_id == SESSION
        cold.shutdown()


def test_sequential_explicit_session_id_resume(open_engine, differential_base):
    _, gateway, _ = _mixed(open_engine, sequential=True)
    assert (gateway.last_compression_status == "error") is differential_base
    resumed = open_engine()
    _step(resumed, "explicit S start", [], lambda: _start(resumed))
    assert resumed._conversation_id == SESSION
    full = EARLY + LATER
    _step(resumed, "explicit S full ingest", full, lambda: resumed.ingest(full))
    _step(resumed, "explicit S full compress", full, lambda: resumed.compress(full, force=True))


def test_single_window_same_binding_restart(open_engine):
    writer = open_engine()
    _start(writer)
    view = EARLY + LATER
    _step(writer, "single-window ingest", view, lambda: writer.ingest(view), view)
    for i in range(3):
        view = _step(writer, f"single-window compress {i + 1}", view, lambda: writer.compress(view))
    _step(writer, "single-window start", [], lambda: _start(writer))
    _step(writer, "single-window adopted replay", view, lambda: writer.compress(view, force=True))


@pytest.mark.parametrize("rotate_before_conflict", [False, True])
@pytest.mark.parametrize("finalize_next", [False, True], ids=["one-transition", "two-transitions"])
def test_competing_gateway_rotates_after_reownership(open_engine, monkeypatch, rotate_before_conflict, finalize_next):
    writer, gateway, gateway_view = _mixed(open_engine)
    calls = []

    def summarize(**_):
        calls.append(None)
        return "Cobalt summary.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)

    def rotate():
        gateway.on_session_end(SESSION, gateway_view)
        gateway.on_session_start("S2-1250", platform="desktop", conversation_id=GATEWAY)
        binding = gateway._lifecycle.get_by_conversation(GATEWAY)
        assert (binding.current_session_id, binding.last_finalized_session_id) == ("S2-1250", SESSION)
        if finalize_next:
            gateway.on_session_end("S2-1250", [])
            binding = gateway._lifecycle.get_by_conversation(GATEWAY)
            assert (binding.current_session_id, binding.last_finalized_session_id) == (None, "S2-1250")

    if rotate_before_conflict:
        rotate()
    view = _step(writer, "first stalled compress", EARLY, lambda: writer.compress(list(EARLY)))
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert len(calls) == 1
    if not rotate_before_conflict:
        rotate()
    assert _step(writer, "rotated gateway back-off", view, lambda: writer.compress(view, force=True)) == view
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert len(calls) == 1


def test_replay_receipt_skips_ignored_host_message(open_engine, monkeypatch):
    if message_patterns._regex_engine is None:
        monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)
    config = {"ignore_message_patterns": ["^HEARTBEAT"]}
    writer, gateway = open_engine(**config), open_engine(**config)
    heartbeat = {"role": "user", "content": "HEARTBEAT ok"}
    raw = EARLY[:2] + [heartbeat] + EARLY[2:]
    _start(writer)
    _step(writer, "initial ingest with heartbeat", raw, lambda: writer.ingest(raw), EARLY)
    _bind_gateway(gateway)
    full = raw + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    _step(gateway, "gateway re-owns the rows", full, lambda: gateway.compress(full), may_reown=True)
    assert gateway.last_compression_status == "compacted"
    assert set(gateway._store.read_metadata_json(REOWN_KEY)) == {
        row["store_id"] for row in _rows(writer)[:len(EARLY)]
    }
    writer.shutdown()

    resumed = open_engine(**config)
    _start(resumed)
    new = [
        {"role": "user", "content": "A new request after the heartbeat."},
        {"role": "assistant", "content": "A new reply after the heartbeat."},
    ]
    _step(resumed, "raw replay with heartbeat", raw + new, lambda: resumed.ingest(raw + new), new)
    assert resumed._last_ingest_reconciliation["reason"] == "replayed re-owned session rows"
    assert resumed._last_ingest_reconciliation["cursor"] == len(raw)


def test_replay_receipt_stops_at_unmatched_host_message(open_engine):
    writer, gateway, _ = _mixed(open_engine)
    raw = EARLY[:2] + [{"role": "user", "content": "An unmatched request."}] + EARLY[2:]
    stored = _rows(gateway)
    assert writer._cursor_from_reowned_session_rows(raw, stored, len(stored)) == 0


def test_short_view_skips_full_receipt_scan(open_engine, monkeypatch):
    writer, gateway, _ = _mixed(open_engine)
    stored = _rows(gateway)

    def unexpected_read(*args, **kwargs):
        pytest.fail("a view shorter than the receipt should not load the session")

    monkeypatch.setattr(writer._store, "get_session_messages", unexpected_read)
    assert writer._cursor_from_reowned_session_rows(EARLY[:2], stored[-1:], len(stored)) == 0


def test_malformed_replay_receipt_does_not_prevent_reownership(open_engine):
    writer, gateway = open_engine(), open_engine()
    _start(writer)
    writer.ingest(list(EARLY))
    before = _rows(writer)
    writer._store.write_metadata_json([REOWN_KEY], "malformed")
    _bind_gateway(gateway)
    full = EARLY + LATER
    _step(gateway, "gateway ingest", full, lambda: gateway.ingest(full), LATER)
    _step(gateway, "gateway publishes with malformed receipt", full, lambda: gateway.compress(full), may_reown=True)
    assert gateway.last_compression_status == "compacted"
    assert sorted(gateway._store.read_metadata_json(REOWN_KEY)) == [row["store_id"] for row in before]
    assert all(row["conversation_id"] == GATEWAY for row in _rows(gateway))
