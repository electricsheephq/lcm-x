"""Under a UID trim, the adoption-only depth stop (#3049) still frees the adopted rows on this line.

Review of the port (#3088): the zero-leaf finalizer validates the trimmed, adopted list and clears the
UID scope, so the zero-leaf UID restore in _compress_owned does not return the pre-trim list. The stop
therefore stays as on main: no extra summary call, no hold, and the host gets fewer tokens.
"""

from collections import Counter

import pytest

from hermes_lcm.tokens import count_messages_tokens
from tests.test_depth_target_ratio import SUMMARY, WINDOW, _leaves, _sweep, _sweep_engine, _target_after, _view
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_457_resume_controls import _HostCancel
from tests.test_issue_671_stub_first_exit import make_engine as make_engine


@pytest.mark.parametrize("uid_mode", ["on", "off"], ids=["uid-trim", "uid-off"])
def test_depth_retry_after_uid_trim(make_engine, monkeypatch, tmp_path, uid_mode):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    _state_db(home, [("issue-671-0", None, None)])
    view = _view(7)
    # Only the first committed turn has UIDs. The second still needs replay adoption.
    for index, row in enumerate(view[1:3], start=1):
        row["message_uid"] = f"message-{index}"
    engine = _sweep_engine(make_engine, view)
    observed = count_messages_tokens(view)
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs["depth"])
        if len(calls) == 3:
            raise _HostCancel()
        return SUMMARY, 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    engine.ingest(view)

    def census():
        return Counter((row["store_id"], row["role"], row["content"]) for row in
                       engine._store.get_session_messages(engine._session_id))

    before = census()
    assert before.total() == len(view)
    with pytest.raises(_HostCancel):
        engine.compress(view, current_tokens=observed)
    committed = [leaf.node_id for leaf in _leaves(engine)]
    assert len(committed) == 2 and calls == [0, 0, 0]
    assert census() == before
    # One turn of replay savings can now reach the target.
    engine._config.depth_target_ratio = _target_after(view, 1) / WINDOW

    trim = engine._drop_uid_covered_live_rows
    admit = engine._admit_leaf_and_adopt_replay
    trims, adoptions = [], []

    def record_trim(rows):
        trimmed = trim(rows)
        trims.append((rows, trimmed))
        return trimmed

    def record_adoption(state, *args):
        rows = state.working_messages
        reached_before = engine._depth_target_reached(state)
        action = admit(state, *args)
        if state.resumed_prefix:
            dropped = [row["content"] for row in rows if row not in state.working_messages]
            adoptions.append((reached_before, engine._depth_target_reached(state), state.budget.leaves, dropped))
        return action

    monkeypatch.setattr(engine, "_drop_uid_covered_live_rows", record_trim)
    monkeypatch.setattr(engine, "_admit_leaf_and_adopt_replay", record_adoption)
    result = engine.compress(view, current_tokens=observed)

    status = _sweep(engine)
    leaves = [leaf.node_id for leaf in _leaves(engine)]
    assert leaves[:2] == committed
    assert census() == before
    engine.ingest(result)
    assert census() == before
    if uid_mode == "on":
        [(pre_trim, trimmed)] = trims
        assert trimmed == pre_trim[:1] + pre_trim[3:]
        assert len(trimmed) == len(pre_trim) - 2
        assert adoptions == [(False, True, 0, [row["content"] for row in view[3:5]])]
        assert status["stop_reason"] == "depth_target_reached" and status["depth_target_met"] is True
        assert status["leaf_passes"] == 0 and leaves == committed and calls == [0, 0, 0]
        assert count_messages_tokens(result) < count_messages_tokens(pre_trim)
        assert engine._no_progress_hold is None and engine._last_compression_status == "compacted"
    else:
        assert not trims
        assert adoptions == [(False, True, 0, [row["content"] for row in view[1:5]])]
        assert status["stop_reason"] == "depth_target_reached" and status["depth_target_met"] is True
        assert status["leaf_passes"] == 0 and leaves == committed and calls == [0, 0, 0]
        assert count_messages_tokens(result) < observed
