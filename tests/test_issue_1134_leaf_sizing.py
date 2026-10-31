"""#1134: small windows use capped sizing for non-sweep admission."""

import pytest

from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_671_stub_first_exit import (  # noqa: F401
    make_engine, payload_of, run, summaries,
)


@pytest.mark.parametrize("survival_fit", [False, True], ids=["REGRESSION-fit-off", "CONTROL-default-fit"])
def test_non_sweep_admits_capped_55k_leaf(make_engine, summaries, caplog, survival_fit):  # noqa: F811
    engine = make_engine(threshold_full_sweep_enabled=False, dynamic_leaf_chunk_enabled=False,
                         large_output_externalization_enabled=False,
                         large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
                         **({"survival_fit": False} if not survival_fit else {}))
    engine.context_length = 55_000
    engine.threshold_tokens = 41_250
    view = [{"role": "system", "content": "system"}]
    for i in range(10):
        view += [{"role": "user", "content": f"turn {i} " + payload_of(1120)},
                 {"role": "assistant", "content": payload_of(1120)}]
    view += [{"role": "user", "content": payload_of(10_000)},
             {"role": "assistant", "content": payload_of(10_000)}]
    assert 22_000 <= count_messages_tokens(view[1:-2]) < 24_000
    assert count_messages_tokens(view) > engine.threshold_tokens
    run(engine, view, caplog)
    assert engine._last_compression_noop_reason != "raw backlog outside fresh tail is below leaf chunk threshold"
    assert engine._dag.get_session_nodes(engine._session_id)
    assert summaries
    assert engine._store.get_session_count(engine._session_id) == len(view)
