"""#758 (optional second step): a compacted session keeps its commit proof across a policy change.

The compaction-commit proof binds the rows LCM returned, which the host keeps as written under
that compaction's policy. Re-redacting them under a changed policy broke the binding, so every
restart re-stored the rows after the scaffold prefix until the next compaction.

All secrets are synthetic. ``summarize_with_escalation`` is stubbed; no model is called.
"""
from __future__ import annotations

import pytest

from hermes_lcm import engine as lcm_engine

from .test_issue_758_redaction_policy_restart import (  # noqa: F401  (shared fixtures/constants)
    API, ENV, NARROW_API, OFF, ON_ALL, ON_API, PWD, SID, WIDE_API, _engine,
)

PROVEN = "replayed proven post-compaction continuation"


def _summary(**_kwargs):
    return "Earlier conversation summary. Expand for details: old turns", 1


def _compacted(tmp_path, monkeypatch, cfg1, cfg2, *, fresh_line="", later_line="", wide2=False):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    host = [{"role": "system", "content": "you are a test agent"}]
    for i in range(8):
        text = f"user turn {i}" + (f"\n{fresh_line}" if i == 7 and fresh_line else "")
        host += [{"role": "user", "content": text}, {"role": "assistant", "content": f"assistant reply {i}"}]
    engine = _engine(tmp_path, monkeypatch, {**cfg1, "fresh_tail_count": 4, "leaf_chunk_tokens": 10})
    try:
        engine.ingest(list(host))
        host = list(engine.compress(list(host), force=True))
        assert engine.last_compression_status == "compacted"
        host += [{"role": "user", "content": "later user" + (f"\n{later_line}" if later_line else "")},
                 {"role": "assistant", "content": "later reply"}]
        engine.ingest(list(host))
        counts = [engine._store.get_session_count(SID)]
    finally:
        engine.shutdown()
    reasons = []
    for turn in range(3):
        host.append({"role": "user", "content": f"new user row after restart {turn}"})
        engine = _engine(tmp_path, monkeypatch, {**cfg2, "fresh_tail_count": 4, "leaf_chunk_tokens": 10}, wide=wide2)
        try:
            engine.ingest(list(host))
            counts.append(engine._store.get_session_count(SID))
            reasons.append(engine._last_ingest_reconciliation["reason"])
        finally:
            engine.shutdown()
    return counts, reasons


@pytest.mark.parametrize(
    ("cfg1", "cfg2", "fresh_line", "later_line", "wide2"),
    [
        pytest.param(OFF, ON_ALL, API, "", False, id="turned-on-secret-in-fresh-tail"),
        pytest.param(ON_API, ON_API, ENV, "", True, id="widened-491-secret-in-fresh-tail"),
        pytest.param(ON_ALL, OFF, "", API, False, id="turned-off-secret-after-compaction", marks=pytest.mark.xfail(strict=True, reason="#758 remaining: no record of the policy a row was written under, so a restart after redaction is turned off or a pattern removed still re-stores")),
        pytest.param(OFF, ON_ALL, "", API, False, id="turned-on-secret-after-compaction"),
    ],
)
def test_compacted_session_keeps_its_proof_across_a_policy_change(
    tmp_path, monkeypatch, cfg1, cfg2, fresh_line, later_line, wide2
):
    counts, reasons = _compacted(tmp_path, monkeypatch, cfg1, cfg2, fresh_line=fresh_line,
                                 later_line=later_line, wide2=wide2)
    assert counts == [19, 20, 21, 22]  # main: 19 -> 26 -> 34 -> 43
    assert reasons == [PROVEN] * 3


def test_compacted_password_row_after_compaction_is_still_not_proof(tmp_path, monkeypatch):
    # A digest-less placeholder never proves a row, whichever side carries it (#484 item 11b).
    counts, reasons = _compacted(tmp_path, monkeypatch, ON_ALL, ON_API, later_line=PWD)
    assert counts[1] > counts[0] + 1
    assert PROVEN not in reasons[:1]
