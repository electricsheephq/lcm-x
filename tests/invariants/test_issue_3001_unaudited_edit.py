"""Issue #3001 / #3054. A refused pass may not spend an unaudited user edit (acceptance cell for the next attempt)."""

import pytest

from hermes_lcm.engine import LCMEngine
from tests.invariants.test_issue_1353_prematch_fault_prefix import _boom
from tests.test_compression_boundary import _turn
from tests.test_refused_compaction_rollback import (
    _defer_rewriting_pass_refusal,
    committed_host,  # noqa: F401
    refusal_uid_mode,  # noqa: F401
)


@pytest.mark.xfail(strict=True, reason="#3053 is held: the admitted-input fingerprint skips an unaudited user edit; "
                   "the next attempt (#3054, fall back to the last adopted proof) must store it once")
def test_cold_bind_stores_edit_after_refused_ingest_fault(
    refusal_uid_mode, committed_host, monkeypatch,  # noqa: F811
):
    engine, kept, binding = committed_host
    engine.ingest(kept)
    assert engine._ingest_cursor == len(kept) == 19
    assert kept[-3] == _turn(19)[0]
    edit = {**kept[-3], "content": "Updated user instruction from turn 19", "timestamp": 1_760_000_000.0}
    kept[-3] = edit
    failures = engine._ingest_failure_count
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_identity_anchor_prematch", _boom)
        _defer_rewriting_pass_refusal(engine, kept, patch)
    assert engine._ingest_failure_count == failures + 1
    before = engine._store.get_session_messages("S0")
    assert all(row["content"] != edit["content"] for row in before)

    resumed = LCMEngine(config=engine._config, hermes_home=str(engine._hermes_home))
    try:
        resumed.on_session_start("S0", **binding)
        reply = _turn(20)[1]
        resumed.ingest(kept + [reply])
        after = resumed._store.get_session_messages("S0")
        assert after[:len(before)] == before
        added = after[len(before):]
        assert added[-1]["content"] == reply["content"]
        edits = [(row["role"], row["observed_at"]) for row in added if row["content"] == edit["content"]]
        assert edits == [(edit["role"], edit["timestamp"])]
    finally:
        resumed.shutdown()
