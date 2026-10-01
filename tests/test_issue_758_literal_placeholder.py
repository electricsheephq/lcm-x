"""#758 fix round: a placeholder the HOST holds as text never stands in for a stored raw secret.

Codex review of 3f42971e (R1/R5): with redaction off in both runs, a stored raw secret and a host row
carrying the literal ``[LCM sensitive redaction: ...]`` text of that secret (right digest) compared
equal in the policy-neutral retry, so the host's row was proven replay and never stored. Main
(cursor 0) stores it. The retry now reads placeholder provenance from the host's unredacted rows.

All secrets are synthetic. ``summarize_with_escalation`` is stubbed; no model is called.
"""
from __future__ import annotations

import pytest

from hermes_lcm import engine as lcm_engine
from hermes_lcm import ingest_protection as ip

from .test_issue_758_redaction_policy_restart import OFF, ON_ALL, SID, _engine

SECRET = "sk-test1234567890abcdefXYZ"
RAW = f"use api_key={SECRET} for the build"
LITERAL = f"use api_key={ip._sensitive_placeholder('api_key', SECRET)} for the build"
OTHER = ip._sensitive_placeholder("api_key", "sk-someone-else-0987654321")
assert ip.redact_sensitive_text(RAW, type("C", (), {"sensitive_patterns_enabled": True,
                                                      "sensitive_patterns": ["api_key"]})()) == LITERAL


def _two_runs(tmp_path, monkeypatch, stored, host, cfg1, cfg2):
    engine = _engine(tmp_path, monkeypatch, cfg1)
    try:
        engine.ingest(list(stored))
    finally:
        engine.shutdown()
    engine = _engine(tmp_path, monkeypatch, cfg2)
    try:
        direct = engine._reconcile_ingest_cursor_from_store(list(host))  # default call: no provenance
        engine._ingest_cursor_needs_reconcile = True
        engine.ingest(list(host))
        rows = engine._store.get_session_messages(SID, limit=100)
        return direct, [r["content"] for r in rows], engine._last_ingest_reconciliation["reason"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("cfg2", [OFF, ON_ALL], ids=["off-both-runs", "turned-on"])
def test_host_literal_placeholder_is_not_proof_of_a_stored_raw_secret(tmp_path, monkeypatch, cfg2):
    """E1 path (codex R1): stored [context, raw secret], host [context, literal placeholder]."""
    context = {"role": "user", "content": "context row"}
    direct, contents, reason = _two_runs(
        tmp_path, monkeypatch,
        [context, {"role": "assistant", "content": RAW}],
        [context, {"role": "assistant", "content": LITERAL}],
        OFF, cfg2,
    )
    assert reason == "persisted ambiguous delta"  # main's outcome: the batch is stored whole
    assert len(contents) == 4 and contents[-1] == LITERAL
    assert direct == 0  # the default call (no host provenance) never retries


def test_placeholder_text_stored_verbatim_still_replays_across_a_policy_change(tmp_path, monkeypatch):
    """The host keeps typed placeholder text the store also holds: it compares as text, and the
    retry still proves the turned-on row next to it."""
    rows = []
    for i in range(10):
        text = f"user turn {i}" + (f"\n{RAW}" if i == 3 else "") + (f"\nnote {OTHER}" if i == 5 else "")
        rows += [{"role": "user", "content": text}, {"role": "assistant", "content": f"assistant reply {i}"}]
    host = rows + [{"role": "user", "content": "new user row after restart"}]
    _direct, contents, reason = _two_runs(tmp_path, monkeypatch, rows, host, OFF, ON_ALL)
    assert len(contents) == 21
    assert reason == "replayed durable tail across a redaction policy change"


def _summary(**_kwargs):
    return "Earlier conversation summary. Expand for details: old turns", 1


def test_host_literal_placeholder_after_a_compaction_is_not_proof(tmp_path, monkeypatch):
    """E2 path (codex R5): unchanged OFF policy, exact proof-bound prefix, then a row stored raw after
    the compaction that the host now shows as literal placeholder text."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    host = [{"role": "system", "content": "you are a test agent"}]
    for i in range(8):
        host += [{"role": "user", "content": f"user turn {i}"}, {"role": "assistant", "content": f"assistant reply {i}"}]
    cfg = {**OFF, "fresh_tail_count": 4, "leaf_chunk_tokens": 10}
    engine = _engine(tmp_path, monkeypatch, cfg)
    try:
        engine.ingest(list(host))
        result = list(engine.compress(list(host), force=True))
        assert engine.last_compression_status == "compacted"
        engine.ingest(result + [{"role": "user", "content": RAW}])
        stored = engine._store.get_session_count(SID)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path, monkeypatch, cfg)
    try:
        engine.ingest(result + [{"role": "user", "content": LITERAL}])
        contents = [r["content"] for r in engine._store.get_session_messages(SID, limit=200)]
        reason = engine._last_ingest_reconciliation["reason"]
    finally:
        engine.shutdown()
    assert reason != "replayed proven post-compaction continuation"
    assert len(contents) > stored and contents[-1] == LITERAL


def test_a_proven_restart_never_scans_for_placeholders(tmp_path, monkeypatch):
    """Codex NB: the effective-match gate runs first; an unaffected restart does no placeholder scan."""
    from hermes_lcm import reconcile as lcm_reconcile

    calls = []
    real = lcm_reconcile.ReconcileMixin._window_has_redaction_placeholder
    monkeypatch.setattr(lcm_reconcile.ReconcileMixin, "_window_has_redaction_placeholder",
                        staticmethod(lambda *args: calls.append(1) or real(*args)))
    rows = []
    for i in range(10):
        rows += [{"role": "user", "content": f"user turn {i}" + (f"\n{RAW}" if i == 3 else "")},
                 {"role": "assistant", "content": f"assistant reply {i}"}]
    host = rows + [{"role": "user", "content": "new user row after restart"}]
    _direct, contents, reason = _two_runs(tmp_path, monkeypatch, rows, host, ON_ALL, ON_ALL)
    assert (len(contents), reason) == (21, "replayed durable tail")
    assert calls == []  # neither the default call nor the ingest scans
