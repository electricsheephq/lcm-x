"""Natural re-ownership and mixed-window replay (#1250, I1/I2/I5/I14)."""

import json

import pytest

from hermes_lcm import engine as engine_module
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
    assert gateway._store.read_metadata_json(REOWN_KEY) == expected_ids
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
def test_competing_gateway_rotates_after_reownership(open_engine, monkeypatch, rotate_before_conflict):
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
