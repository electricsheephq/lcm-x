"""P-S2: old-writer stale cleanup proof, followed by two cold gateway starts.

Copied minimal host/cleanup helpers from test_issue_1286_cleanup_rewrite_proof;
all product APIs used here also exist at the pinned v0.29.0 writer.
Run under pytest in a subprocess, as occurrence_upgrade_probe.py does.
"""
import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

HOME, OUT = Path(os.environ["P2S_HOME"]), Path(os.environ["P2S_OUT"])
PHASE, VARIANT, CHANGE = (os.environ[key] for key in ("P2S_PHASE", "P2S_VARIANT", "P2S_CHANGE"))
# on: sensitive-pattern redaction enabled and one warm user row carries a bearer token (the reference agent's
# failing session held redacted rows in its live list), so stored rows differ from the host's raw rows.
REDACT = os.environ.get("P2S_REDACT", "off")
SECRET_USER = "[S] call the API with Authorization: Bearer p2sTestToken0123456789abcdef" + " alpha beta gamma delta" * 30
SID, CID = "cleanup-rewrite-proof", "cleanup-rewrite-proof-conv"
PAD = " alpha beta gamma delta" * 30
RAW = ("r " * 30_000) + "RAW_NEEDLE " + ("r " * 30_000)
THINK = "<think>weigh the options before replying</think>"
STEER = {"role": "user", "content": "steer: keep the answer short"}


def _no_summary(*args, **kwargs):
    raise AssertionError("only the initial published compaction may summarize")


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _no_summary)


def _stub_summarizer(*args, **kwargs):
    text = kwargs.get("text") if "text" in kwargs else (args[0] if args else "")
    tags = sorted(set(re.findall(r"\[T(\d+)\]", text or "")))
    return "Stub summary covers " + " ".join("T" + tag for tag in tags) + ".\nExpand for details about: stub", 1


def _bind():
    engine = LCMEngine(config=LCMConfig(
        database_path=str(HOME / "lcm.db"),
        large_output_externalization_path=str(HOME / "externalized"),
        fresh_tail_count=2, leaf_chunk_tokens=100, fresh_tail_pressure_yield_enabled=False,
        deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5,
        **({"sensitive_patterns_enabled": True} if REDACT == "on" else {}),
    ), hermes_home=str(HOME / "home"))
    engine.on_session_start(SID, platform="telegram", conversation_id=CID, context_length=200_000)
    return engine


class Host:
    """Minimal #1286 host: stamped users, unstamped replies, in-place commits."""

    def __init__(self):
        self.clock, self.n_turns = 1_760_000_000.0, 0
        self.live = [{"role": "system", "content": "system prompt"}]
        self.engine = _bind()

    def stamp(self):
        self.clock += 1.0
        return self.clock

    def turn(self, user=None, reply=None):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        self.live += [{"role": "user", "content": user or f"[T{i}] user turn{PAD}", "timestamp": self.stamp()},
                      {"role": "assistant", "content": reply or f"reply to T{i}{PAD}"}]
        self.engine.ingest(self.live)

    def cleanup_pass(self, preflight):
        rough = count_messages_tokens(self.live)
        self.engine.threshold_tokens = 10 * rough
        if preflight:
            self.engine.context_length = int(rough * 1.6)
            assert self.engine.should_compress_preflight(self.live) is True
        else:
            assert self.engine._mark_below_threshold_maintenance() is True
        before = deepcopy(self.live)
        result = self.engine.compress(self.live, current_tokens=rough)
        assert self.engine._last_compression_status == "sanitized"
        assert result is not self.live and len(result) == len(before) and result != before
        return result, before

    def compact(self):
        before = deepcopy(self.live)
        result = self.engine.compress(self.live, force=True)
        assert self.engine._last_compression_status == "compacted"
        self.commit_in_place(result, before)

    def commit_in_place(self, result, before):
        self.engine.on_session_end(SID, before)
        self.live = result
        self.engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="telegram",
                                     conversation_id=CID, context_length=200_000)


def _turns(host, variant, n=6):
    for i in range(n):
        if variant == "stub" and i == 2:
            host.turn(user=f"[T{host.n_turns}] {RAW}")
        elif variant == "reasoning" and i == 2:
            host.turn(reply=f"{THINK}reply to T{host.n_turns}{PAD}")
        else:
            host.turn()


def _commit_cleanup(host, variant):
    result, before = host.cleanup_pass(preflight=variant == "reasoning")
    if variant == "stub":
        assert any(str(m.get("content")).startswith("[Externalized payload: kind=raw_payload;") for m in result)
    else:
        assert THINK not in str(result) and THINK in str(before)
    host.commit_in_place(result, before)


def _change(host, change, *, insert_at, replace_at):
    if change == "insert":
        host.live.insert(insert_at, dict(STEER))
        return STEER["content"]
    content = f"[edited] the user rewrote this turn{PAD}"
    host.live[replace_at] = {"role": "user", "content": content, "timestamp": host.stamp()}
    return content


def _identity(role, content, tool_call_id, tool_calls):
    """Keep in sync with the driver's read-only SQLite identity projection."""
    calls = json.loads(tool_calls) if tool_calls else []
    call_ids = tuple((call.get("id"), call.get("function", {}).get("name"),
                      call.get("function", {}).get("arguments")) for call in calls) if role == "assistant" else ()
    return role, hashlib.sha256((content or "").encode()).hexdigest(), call_ids, tool_call_id if role == "tool" else None


def test_phase(monkeypatch):
    assert PHASE in ("write", "read")
    assert VARIANT in ("stub", "reasoning")
    assert CHANGE in ("insert", "replace")
    assert REDACT in ("off", "on")
    if PHASE == "write":
        HOME.mkdir(parents=True)
        host = Host()
        try:
            # The full field order, including the earlier published compaction.
            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
            _turns(host, None)
            host.compact()
            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _no_summary)
            _turns(host, VARIANT)
            _commit_cleanup(host, VARIANT)
            host.turn()
            host.turn(user=SECRET_USER if REDACT == "on" else None)  # kept: the change below is before it
            changed = _change(host, CHANGE, insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
            host.turn()
            assert host.engine._store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND content = ?", (SID, changed),
            ).fetchone()[0] == 1
            (HOME / "host.json").write_text(json.dumps(host.live))
        finally:
            # Gateway stop: discard the process-local proof without a second bind.
            host.engine.shutdown()
        return

    restart = int(os.environ["P2S_RESTART"])
    assert restart in (1, 2)
    live = json.loads((HOME / "host.json").read_text())
    clock = max(row.get("timestamp", 0) for row in live)
    for n in range(1, restart + 1):
        live += [{"role": "user", "content": f"P-S2 restart {n} user{PAD}", "timestamp": clock + n},
                 {"role": "assistant", "content": f"P-S2 restart {n} assistant{PAD}"}]
    proof_log = []

    class _ProofLog(logging.Handler):  # the bind's proof and reconciliation decisions (no row content)
        def emit(self, record):
            text = record.getMessage()
            if "commit proof" in text or "identity-anchor" in text or "reconcil" in text:
                proof_log.append(text[:300])

    handler = _ProofLog(level=logging.INFO)
    lcm_logger = logging.getLogger("hermes_lcm")
    lcm_logger.addHandler(handler)
    previous_level = lcm_logger.level
    lcm_logger.setLevel(logging.INFO)
    engine = _bind()
    try:
        engine.ingest(live)
        rows = [_identity(*row) for row in engine._store._conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages WHERE session_id = ? ORDER BY store_id",
            (SID,)).fetchall()]
        reconciliation = {key: value for key, value in dict(getattr(engine, "_last_ingest_reconciliation", {}) or {}).items()
                          if key in ("action", "reason", "cursor", "incoming", "session_count", "stored_tail_count")}
    finally:
        engine.shutdown()
        lcm_logger.removeHandler(handler)
        lcm_logger.setLevel(previous_level)
    OUT.write_text(json.dumps({"rows": rows, "reconciliation": reconciliation, "proof_log": proof_log}, indent=2))
