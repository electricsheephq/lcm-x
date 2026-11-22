"""The deferred stub-first exit must return the replay it persisted after UID trimming."""

from collections import Counter

import pytest

from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_628_summary_circuit import _open_circuit
from tests.test_issue_671_stub_first_exit import exit_engine, make_engine as make_engine, tool_view


@pytest.mark.parametrize("stop", ["open-circuit", "leaf-error"])
def test_uid_trim_keeps_stub_first_fallback(make_engine, monkeypatch, tmp_path, stop):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    _state_db(home, [("issue-671-0", None, None)])
    engine = exit_engine(make_engine, depth_target_ratio=120 / 200_000,
                         temporal_rollups_enabled=False, l3_truncate_tokens=2)
    view = tool_view(prefix=({"role": "user", "content": "earlier ask"},
                             {"role": "assistant", "content": "earlier answer"}))
    for index, row in enumerate(view):
        row["message_uid"] = f"message-{index}"
    engine.ingest(view)
    session = engine._session_id

    def census():
        return Counter((row["store_id"], row["role"], row["content"]) for row in
                       engine._store.get_session_messages(session))

    before = census()
    assert len(before) == len(view)
    covered = [sid for sid, _, content in before if content in {"earlier ask", "earlier answer"}]
    assert len(covered) == 2
    engine._dag.add_node(SummaryNode(session_id=session, source_ids=covered,
                                     summary="An earlier exchange about setup.", token_count=7))
    # A durable frontier ahead of the in-process marker exercises the UID coverage check.
    engine._lifecycle.advance_frontier(engine._conversation_id, session, max(covered))

    trim, fallback = engine._drop_uid_covered_live_rows, engine._use_stub_first_fallback
    trims, exits, first_leaf_calls = [], [], []

    def record_trim(rows):
        trimmed = trim(rows)
        trims.append((rows, trimmed))
        return trimmed

    def record_fallback(state):
        candidate = fallback(state)
        if candidate is not None:
            exits.append((state.leaf_passes, candidate))
        return candidate

    def fail_first_leaf(state, leaf, sweep_step_done):
        first_leaf_calls.append((state.leaf_passes, bool(leaf.summary_input_chunk)))
        raise RuntimeError("synthetic first-leaf summary failure")

    monkeypatch.setattr(engine, "_drop_uid_covered_live_rows", record_trim)
    monkeypatch.setattr(engine, "_use_stub_first_fallback", record_fallback)
    if stop == "open-circuit":
        _open_circuit(engine)
        assert not engine._summary_route_available()
    else:
        monkeypatch.setattr(engine, "_summarize_selected_leaf", fail_first_leaf)

    result = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert census() == before
    assert first_leaf_calls == ([(0, True)] if stop == "leaf-error" else [])
    [(pre_trim, trimmed)] = trims
    assert any(row["content"] == "earlier answer" for row in pre_trim)
    assert len(trimmed) < len(pre_trim)
    assert all(row["content"] != "earlier answer" for row in trimmed)
    [(leaves, candidate)] = exits
    assert leaves == 0
    assert result is candidate and result != pre_trim, "host did not get the finalized stub-first candidate"
    sweep = engine._last_threshold_full_sweep
    assert engine._last_stub_first_exit["tokens_after"] == engine._survival_measure(result) + sweep["fixed_load_tokens"]
    digest = engine._compacted_active_replay_snapshot_digest(result)
    assert digest and engine._load_compacted_active_replay_snapshot_digests()[-1:] == [digest]
