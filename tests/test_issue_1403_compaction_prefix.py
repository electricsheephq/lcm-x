"""#1403: a real compaction commit proves its prefix for the next ingest only."""
from collections import Counter
from copy import deepcopy

import hermes_lcm.engine as lcm_engine

from tests.test_issue_436_identity_anchor import _engine, _rows, _turns


def test_accepted_compaction_prefix_is_not_a_changed_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1))
    engine = _engine(tmp_path)
    try:
        original = _turns(1, 8, 0.)
        for row in original:
            row.pop("timestamp")
        engine.ingest(original)
        before = _rows(engine)
        assert all(row["observed_at"] is None for row in before)
        output = engine.compress(original, current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted"
        assert len(output) < len(original)
        engine.on_session_end("S", original)
        engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                platform="cli", context_length=200_000, conversation_id="conv")
        assert engine._compress_commit_proof
        # The host stamps persisted compaction output after LCM stored its NULL rows.
        host = deepcopy(output)
        for index, row in enumerate(host):
            row["timestamp"] = 1000. + index
        cached = engine._last_active_replay_source_identities
        assert [engine._message_replay_identity(row, strip_carrier=False) for row in host] != cached[:len(host)]
        # A same-text suffix is outside the accepted prefix and is a fresh occurrence.
        survivor = next(row for row in reversed(host) if row["role"] == "user"
                        and any(old["content"] == row["content"] for old in before))
        fresh = dict(survivor, timestamp=2000.)
        host.append(fresh)
        engine.ingest(host)
        expected = Counter((row["role"], row["content"]) for row in before)
        expected[(fresh["role"], fresh["content"])] += 1
        assert Counter((row["role"], row["content"]) for row in _rows(engine)) == expected
        assert engine._compress_commit_proof is None
        for row in host[:-1]:
            copies = [old for old in _rows(engine) if old["content"] == row["content"]]
            if len(copies) == 1:
                assert copies[0]["observed_at"] == row["timestamp"]
        engine.ingest(deepcopy(host))
        assert Counter((row["role"], row["content"]) for row in _rows(engine)) == expected
    finally:
        engine.shutdown()
