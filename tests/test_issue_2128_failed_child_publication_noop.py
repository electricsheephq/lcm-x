"""A failed rotation-child publication is still an identity no-op (#2128; the #541 cell at v0.29.1-rc3).

In rotation mode a child session's compaction publication can fail (LifecyclePublicationConflictError).
compress() then hands the host a copy of its own list: the same replay identities, a new list object. v0.29.0
called that an identity no-op whatever the object; #1286 added ``result is messages`` so that a cleanup-only
rewrite (status ``sanitized``: same identities, new public bytes) is committed. The rule also caught the failed
publication's copy, so the host's rotation into the next child left that child's first bind no proof of the summary
row LCM itself emitted, and it stored that row again as a new user row (one surplus row, the reliability matrix's
``publication-failure/rotation-child`` B2). Only a ``sanitized`` copy is a rewrite; every other copy with the same
identities is still the no-op it was in v0.29.0. The reliability cell is the end-to-end regression (the next child's
store); this test pins the predicate, and the #1286 tests keep a ``sanitized`` copy a committed rewrite.
"""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.lifecycle_state as lifecycle_state
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PARENT, CHILD, GRANDCHILD, CID = "failed-pub-parent", "failed-pub-child", "failed-pub-grandchild", "failed-pub-conv"
PAD = " alpha beta gamma delta" * 30


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """LCM's character estimate and no host estimator, as in the #651 tests."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture(autouse=True)
def _stub_summaries(monkeypatch):
    def summarize(*args, **kwargs):
        text = kwargs.get("text") if "text" in kwargs else (args[0] if args else "")
        tags = sorted(set(re.findall(r"\[T(\d+)\]", text or "")))
        return "Stub summary covers " + " ".join("T" + tag for tag in tags) + ".\nExpand for details about: stub", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)


class RotatingHost:
    """Hermes in rotation mode: stamps user rows, ingests every turn, and on a compression ends the session on the
    list compress() was given, then starts a child session that adopts compress()'s output."""

    def __init__(self, tmp_path):
        self.clock, self.n_turns, self.session = 1_760_000_000.0, 0, PARENT
        self.live: list = [{"role": "system", "content": "system prompt"}]
        self.engine = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
            fresh_tail_count=2, leaf_chunk_tokens=100, fresh_tail_pressure_yield_enabled=False,
        ), hermes_home=str(tmp_path / "home"))
        self.engine.on_session_start(PARENT, platform="telegram", conversation_id=CID, context_length=200_000)

    def turn(self):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        self.clock += 1.0
        self.live += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": self.clock},
                      {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
        self.engine.ingest(self.live)

    def compress_and_rotate(self, child):
        """Returns (compress()'s output, the list it was given, a copy of that list, the compression status)."""
        given, before, old = self.live, deepcopy(self.live), self.session
        result = self.engine.compress(given, force=True)
        status = self.engine._last_compression_status
        self.engine.on_session_end(old, before)
        self.live, self.session = result, child
        self.engine.on_session_start(child, boundary_reason="compression", old_session_id=old,
                                     platform="telegram", conversation_id=CID, context_length=200_000)
        return result, given, before, status

    def close(self):
        self.engine.shutdown()


@pytest.fixture
def host(tmp_path):
    host = RotatingHost(tmp_path)
    yield host
    host.close()


def _fail_child_publications(monkeypatch, engine, session_id):
    """The reliability probe's fault: the lifecycle refuses to stage ``session_id``'s compaction publication."""
    stage = engine._lifecycle.stage_compaction_publication

    def refuse(conn, conversation_id, sid, *args, **kwargs):
        if sid == session_id:
            raise lifecycle_state.LifecyclePublicationConflictError("injected publication failure (rotation child)")
        return stage(conn, conversation_id, sid, *args, **kwargs)

    monkeypatch.setattr(engine._lifecycle, "stage_compaction_publication", refuse)


def test_a_failed_child_publication_hands_back_a_copy_that_is_an_identity_noop(host, monkeypatch):
    for _ in range(6):
        host.turn()
    published, _given, _before, status = host.compress_and_rotate(CHILD)
    assert status == "compacted"
    summaries = [m["content"] for m in published if m.get("role") == "user" and "Stub summary" in str(m.get("content"))]
    assert summaries, "the published list carries the summary row LCM emitted"
    host.turn()
    host.turn()

    _fail_child_publications(monkeypatch, host.engine, CHILD)
    result, given, before, status = host.compress_and_rotate(GRANDCHILD)
    assert status == "error"
    # A copy of the host's list with the same rows: a new object, so ``result is messages`` alone misses it.
    assert result is not given and [m.get("content") for m in result] == [m.get("content") for m in before]
    assert host.engine._compress_commit_reason == "identity_noop"
    assert host.engine._compress_commit_proof is None
