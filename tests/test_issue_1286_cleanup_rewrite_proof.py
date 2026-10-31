"""#1286: a cleanup-only compress() the host commits leaves a durable proof, so a restart stores nothing again.

An automatic below-threshold pass ends ``sanitized`` and hands the host a rewritten list: the replay view's
externalized stub for an oversized payload, or an assistant row with its reasoning block stripped. Hermes commits
it in place (``on_session_end`` on the list compress() was given, then the compression boundary). Before the fix
the proof of that list stayed process-local, or was never built when the replay identities were equal (the stub
swap), so a gateway restart left the cold bind no proof of the committed list: reconcile fell to cursor 0 and
the unstamped rows already stored were stored again.

Each variant: turns, the automatic cleanup pass, Hermes's in-place commit, a warm turn, then a restart whose
first ingest carries the next turn. Only the two new turns' rows may be stored. The controls run the same flow
without the cleanup pass, and with a cleanup pass the host never commits (its proof must not become durable).
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "cleanup-rewrite-proof", "cleanup-rewrite-proof-conv"
PAD = " alpha beta gamma delta" * 30
# Above the 100k non-tool floor (#1016): ingest externalizes it and the replay view swaps in the stub.
RAW = ("r " * 30_000) + "RAW_NEEDLE " + ("r " * 30_000)
THINK = "<think>weigh the options before replying</think>"


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """LCM's character estimate and no host estimator, as in the #651 tests."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture(autouse=True)
def _no_summaries(monkeypatch):
    def summarize(**kwargs):
        raise AssertionError("a cleanup-only pass runs no summariser")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)


class Host:
    """Hermes around the engine: stamps user rows (the in-memory assistant rows carry no stamp), ingests every
    turn, commits a compression in place, restarts the gateway."""

    def __init__(self, tmp_path):
        self.tmp_path, self.clock, self.n_turns = tmp_path, 1_760_000_000.0, 0
        self.live: list = [{"role": "system", "content": "system prompt"}]
        self.engines: list = []
        self.engine = self.bind()

    def bind(self):
        engine = LCMEngine(config=LCMConfig(
            database_path=str(self.tmp_path / "lcm.db"),
            large_output_externalization_path=str(self.tmp_path / "externalized"),
            fresh_tail_count=2, leaf_chunk_tokens=100, fresh_tail_pressure_yield_enabled=False,
            deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5,
        ), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start(SID, platform="telegram", conversation_id=CID, context_length=200_000)
        self.engines.append(engine)
        return engine

    def turn(self, user=None, reply=None):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        self.clock += 1.0
        self.live += [{"role": "user", "content": user or f"[T{i}] user turn{PAD}", "timestamp": self.clock},
                      {"role": "assistant", "content": reply or f"reply to T{i}{PAD}"}]
        self.engine.ingest(self.live)

    def cleanup_pass(self, preflight):
        """The engine-driven automatic compress() below the threshold, which is cleanup-only. Returns (result, the
        input). ``preflight`` asks through the host's preflight (deferred maintenance under critical pressure).
        Otherwise the engine's own request marker stands in: below the threshold the preflight defers a size-only
        replay diff (the stub swap alone), so it asks for this pass only alongside another reason."""
        rough = count_messages_tokens(self.live)
        self.engine.threshold_tokens = 10 * rough
        if preflight:
            self.engine.context_length = int(rough * 1.6)  # pressure ~0.62 >= 0.5, below the threshold
            assert self.engine.should_compress_preflight(self.live) is True
        else:
            assert self.engine._mark_below_threshold_maintenance() is True
        before = deepcopy(self.live)
        result = self.engine.compress(self.live, current_tokens=rough)
        assert self.engine._last_compression_status == "sanitized"
        assert result is not self.live and len(result) == len(before) and result != before
        return result, before

    def commit_in_place(self, result, before):
        """Hermes's in-place order: end on the list compress() was given, adopt, then the compression boundary."""
        self.engine.on_session_end(SID, before)
        self.live = result
        self.engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="telegram",
                                     conversation_id=CID, context_length=200_000)

    def restart(self):
        self.engine.shutdown()
        self.engine = self.bind()

    def stored(self):
        return self.engine._store.get_session_count(SID)

    def duplicated(self):
        rows = self.engine._store._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ?", (SID,)).fetchall()
        return {key: n for key, n in Counter((row[0], row[1]) for row in rows).items() if n > 1}

    def close(self):
        for engine in self.engines:
            engine.shutdown()


@pytest.fixture
def host(tmp_path):
    host = Host(tmp_path)
    yield host
    host.close()


def _turns(host, variant):
    for i in range(6):
        if variant == "stub" and i == 2:
            host.turn(user=f"[T{i}] {RAW}")
        elif variant == "reasoning" and i == 2:
            host.turn(reply=f"{THINK}reply to T{i}{PAD}")
        else:
            host.turn()


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_committed_cleanup_rewrite_survives_a_restart(host, variant):
    _turns(host, variant)
    result, before = host.cleanup_pass(preflight=variant == "reasoning")
    if variant == "stub":
        assert any(str(m.get("content")).startswith("[Externalized payload: kind=raw_payload;") for m in result)
    else:
        assert THINK not in str(result) and THINK in str(before)
    host.commit_in_place(result, before)
    # The boundary proves the committed list: the cursor indexes it and its proof is durable.
    assert host.engine._ingest_cursor_needs_reconcile is False
    assert host.engine._ingest_cursor == len(result)
    assert host.engine._durable_commit_proof_payload() is not None
    rows = host.stored()
    host.turn()
    assert host.stored() == rows + 2
    host.restart()
    host.turn()  # the restarted engine's first ingest: the committed list, the warm turn and the next turn
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_cleanup_rewrite_the_host_never_commits_stays_local(host, variant):
    _turns(host, variant)
    result, before = host.cleanup_pass(preflight=variant == "reasoning")
    assert host.engine._compress_commit_proof and host.engine._compress_commit_proof.get("cleanup_rewrite")
    assert host.engine._durable_commit_proof_payload() is None  # durable only on the commit signal
    rows = host.stored()
    host.turn()  # the host kept its own list
    assert host.engine._durable_commit_proof_payload() is None
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_control_without_a_cleanup_pass(host, variant):
    _turns(host, variant)
    rows = host.stored()
    host.turn()
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}
