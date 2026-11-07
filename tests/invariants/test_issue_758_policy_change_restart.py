"""#758(b): a redaction-policy change between restarts must not re-store the session's rows.

Durable redaction stores each row as redacted by the policy active when it was written. The ingest ledger's identity
digest is taken over the pre-redaction identity (design §3.1, M4), so a cold restart under a different policy still
recognises the replayed rows. The issue's shape: 20 rows with one credential-shaped user line, then restarts that
each change the policy and send the same history plus one new row. Each restart stores only its new row.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SECRET = "deploy key: api_key=sk-test-0123456789abcdef0123456789abcdef"


def _history():
    rows = []
    for i in range(10):
        text = f"user turn {i}: please continue the task " + (SECRET if i == 3 else "")
        rows += [{"role": "user", "content": text, "timestamp": 1000.0 + 2 * i},
                 {"role": "assistant", "content": f"assistant reply {i}", "timestamp": 1001.0 + 2 * i}]
    return rows


def _engine(tmp_path, enabled, patterns):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), sensitive_patterns_enabled=enabled,
        sensitive_patterns=list(patterns), temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False))
    engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=200_000)
    return engine


POLICIES = {
    "on-then-off": [(False, ["api_key"]), (True, ["api_key"]), (False, ["api_key"])],
    "pattern-added": [(True, ["bearer_token"]), (True, ["bearer_token", "api_key"]), (True, ["api_key"])],
}


@pytest.mark.parametrize("shape", sorted(POLICIES))
def test_policy_change_between_restarts_stores_only_new_rows(tmp_path, shape):
    history = _history()
    counts = []
    for step, (enabled, patterns) in enumerate(POLICIES[shape]):
        engine = _engine(tmp_path, enabled, patterns)
        try:
            if step:
                history = history + [{"role": "user", "content": f"new request after restart {step}: {SECRET}",
                                      "timestamp": 2000.0 + step}]
            engine.ingest([dict(row) for row in history])
            assert engine._ingest_failure_count == 0, engine._last_ingest_error
            rows = engine._store.get_session_messages("S")
            counts.append(len(rows))
            # Every row once, in order: no lost row hidden behind a duplicate. A stored row keeps the text its
            # write-time policy produced, so compare the text before the first colon (the secret comes after it).
            assert [(r["role"], r["content"].split(":")[0]) for r in rows] == [
                (m["role"], m["content"].split(":")[0]) for m in history], [(r["role"], r["content"][:40]) for r in rows]
            if step:  # positive control: the policy in force is the one this step set
                redacted = enabled and "api_key" in patterns
                assert (SECRET.split("=")[1] not in rows[-1]["content"]) == redacted, rows[-1]["content"]
        finally:
            engine.shutdown()
    assert counts == [20, 21, 22], counts
