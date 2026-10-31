"""#1286: a list the host commits in place of its own leaves a durable proof of that list, so a restart stores
nothing again.

An automatic below-threshold pass ends ``sanitized`` and hands the host a rewritten list: the replay view's
externalized stub for an oversized payload, or an assistant row with its reasoning block stripped. Hermes commits
it in place (``on_session_end`` on the list compress() was given, then the compression boundary). Before the fix
the proof of that list stayed process-local, or was never built when the replay identities were equal (the stub
swap), so a gateway restart left the cold bind no proof of the committed list: reconcile fell to cursor 0 and
the unstamped rows already stored were stored again.

The field chain had one more step: a later warm turn found a row the host had changed before the cursor and
stored it at the store tail, out of host order. The durable proof's in-order walk of the rows stored after it
then failed at the cold bind, so the proof is re-keyed to the host's list after such a store.

Each test runs turns, the automatic cleanup pass, Hermes's in-place commit, warm turns, then a restart whose
first ingest carries the next turn; only the new rows may be stored. The controls run without the cleanup pass,
with a cleanup pass the host never commits, and with a crash before the re-key.
"""
from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "cleanup-rewrite-proof", "cleanup-rewrite-proof-conv"
PAD = " alpha beta gamma delta" * 30
# Above the 100k non-tool floor (#1016): ingest externalizes it and the replay view swaps in the stub.
RAW = ("r " * 30_000) + "RAW_NEEDLE " + ("r " * 30_000)
THINK = "<think>weigh the options before replying</think>"
STEER = {"role": "user", "content": "steer: keep the answer short"}  # unstamped, as the host inserts it


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


def _stub_summarizer(*args, **kwargs):
    text = kwargs.get("text") if "text" in kwargs else (args[0] if args else "")
    tags = sorted(set(re.findall(r"\[T(\d+)\]", text or "")))
    return "Stub summary covers " + " ".join("T" + tag for tag in tags) + ".\nExpand for details about: stub", 1


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

    def stamp(self):
        self.clock += 1.0
        return self.clock

    def turn(self, user=None, reply=None):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        self.live += [{"role": "user", "content": user or f"[T{i}] user turn{PAD}", "timestamp": self.stamp()},
                      {"role": "assistant", "content": reply or f"reply to T{i}{PAD}"}]
        self.engine.ingest(self.live)

    def tool_turn(self):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        call = f"call_{i}"
        self.live += [
            {"role": "user", "content": f"[T{i}] look it up{PAD}", "timestamp": self.stamp()},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call, "content": f"tool output {i}:" + " result" * 60},
            {"role": "assistant", "content": f"reply to T{i}{PAD}"},
        ]
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

    def compact(self):
        """A forced compress() that publishes a summary, committed in place (an earlier compaction's proof)."""
        before = deepcopy(self.live)
        result = self.engine.compress(self.live, force=True)
        assert self.engine._last_compression_status == "compacted"
        self.commit_in_place(result, before)

    def commit_in_place(self, result, before):
        """Hermes's in-place order: end on the list compress() was given, adopt, then the compression boundary."""
        self.engine.on_session_end(SID, before)
        self.live = result
        self.engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="telegram",
                                     conversation_id=CID, context_length=200_000)

    def restart(self):
        self.engine.shutdown()
        self.engine = self.bind()

    def rows(self):
        return [(row[0], row[1]) for row in self.engine._store._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY store_id", (SID,)).fetchall()]

    def stored(self):
        return self.engine._store.get_session_count(SID)

    def count(self, content):
        return sum(1 for _role, stored in self.rows() if stored == content)

    def duplicated(self):
        return {key: n for key, n in Counter((row[0], row[1]) for row in self.rows()).items() if n > 1}

    def close(self):
        for engine in self.engines:
            engine.shutdown()


@pytest.fixture
def host(tmp_path):
    host = Host(tmp_path)
    yield host
    host.close()


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


def _salvage(messages, keep=2):
    """Hermes's compression salvage: tool outputs over 200 characters become the placeholder, except the newest
    ``keep``; each row keeps its other keys."""
    tools = [i for i, m in enumerate(messages) if m.get("role") == "tool" and len(str(m.get("content"))) > 200]
    out = list(messages)
    for i in tools[:-keep] if keep else tools:
        out[i] = {**messages[i], "content": _HERMES_PRUNED_TOOL_PLACEHOLDER}
    return out


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_committed_cleanup_rewrite_survives_a_restart(host, variant):
    _turns(host, variant)
    _commit_cleanup(host, variant)
    rows = host.stored()
    host.turn()
    host.restart()
    host.turn()  # the restarted engine's first ingest: the committed list, the warm turn and the next turn
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_the_boundary_makes_a_committed_cleanup_proof_durable(host, variant):
    _turns(host, variant)
    result, before = host.cleanup_pass(preflight=variant == "reasoning")
    assert host.engine._compress_commit_proof.get("cleanup_rewrite") is True
    assert host.engine._durable_commit_proof_payload() is None  # durable only on the commit signal
    host.commit_in_place(result, before)
    assert host.engine._ingest_cursor_needs_reconcile is False
    assert host.engine._ingest_cursor == len(result)
    assert host.engine._durable_commit_proof_payload() is not None


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_cleanup_rewrite_the_host_never_commits_stays_local(host, variant):
    _turns(host, variant)
    host.cleanup_pass(preflight=variant == "reasoning")
    rows = host.stored()
    host.turn()  # the host kept its own list
    assert host.engine._durable_commit_proof_payload() is None
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_cleanup_rewrite_after_a_published_compaction_survives_a_restart(host, variant, monkeypatch):
    """The field order: an earlier published compaction left a durable proof that the committed cleanup list
    makes stale."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
    _turns(host, None)
    host.compact()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda *a, **k: pytest.fail("summariser called"))
    _turns(host, variant)
    _commit_cleanup(host, variant)
    rows = host.stored()
    host.turn()
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


@pytest.mark.parametrize("change", ["insert", "replace"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_row_changed_among_the_warm_turns_is_not_stored_again_after_a_restart(host, variant, change):
    """The field chain: the host changes a row between a warm turn's user and assistant rows, and the next turn
    stores it at the store tail, out of host order."""
    _turns(host, variant)
    _commit_cleanup(host, variant)
    host.turn()
    host.turn()
    rows = host.stored()
    pos = len(host.live) - 3  # the first warm turn's assistant row
    if change == "insert":
        host.live.insert(pos, dict(STEER))
    else:
        host.live[pos] = {"role": "assistant", "content": f"rewritten reply{PAD}"}
    changed = host.live[pos]["content"]
    host.turn()
    assert host.stored() == rows + 3 and host.count(changed) == 1
    host.restart()
    host.turn()
    assert host.stored() == rows + 5
    assert host.count(changed) == 1
    assert host.duplicated() == {}


@pytest.mark.parametrize("change", ["insert", "replace"])
def test_a_row_changed_inside_the_committed_list_is_not_stored_again_after_a_restart(host, change):
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    rows = host.stored()
    end = len(host.live)
    if change == "insert":
        host.live.insert(end - 2, dict(STEER))
        changed = STEER["content"]
    else:
        changed = f"[T4] the user edited this turn{PAD}"
        host.live[end - 4] = {"role": "user", "content": changed, "timestamp": host.stamp()}
    host.turn()
    assert host.count(changed) == 1
    host.restart()
    host.turn()
    assert host.stored() == rows + 5
    assert host.count(changed) == 1
    assert host.duplicated() == {}


def test_a_salvaged_commit_survives_a_restart(host):
    """Hermes salvage commits the cleanup list with old tool outputs cleared; the placeholders alias their stored
    rows (#773), warm and cold."""
    for _ in range(3):
        host.tool_turn()
    _turns(host, "reasoning", n=3)
    result, before = host.cleanup_pass(preflight=True)
    host.commit_in_place(_salvage(result), before)
    assert _HERMES_PRUNED_TOOL_PLACEHOLDER in str(host.live)
    rows = host.stored()
    host.turn()
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}


def test_a_crash_before_the_re_key_duplicates_but_loses_nothing(host, monkeypatch):
    """I1: the rows are stored before the proof is re-keyed; a crash in between (no re-key) may store rows again
    after the restart, but every host row stays stored."""
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn()
    host.turn()
    host.live.insert(len(host.live) - 3, dict(STEER))
    monkeypatch.setattr(LCMEngine, "_rekey_durable_commit_proof", lambda self, messages: None, raising=False)
    host.turn()
    host.restart()
    host.turn()
    stored = Counter(host.rows())
    assert all(stored[(m["role"], m["content"])] >= 1 for m in host.live if m["role"] in ("user", "assistant"))


def test_a_row_the_host_changes_after_the_restart_is_still_stored(host):
    """Exactly-once, the other way: the re-keyed proof never hides a row that changed after it."""
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn()
    host.turn()
    host.live.insert(len(host.live) - 3, dict(STEER))
    host.turn()  # stores the inserted row out of host order: the proof is re-keyed to this list
    host.restart()
    edited = f"[T7] the user edited this turn{PAD}"
    host.live[len(host.live) - 4] = {"role": "user", "content": edited, "timestamp": host.stamp()}
    host.turn()
    assert host.count(edited) == 1


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_control_without_a_cleanup_pass(host, variant):
    _turns(host, variant)
    rows = host.stored()
    host.turn()
    host.restart()
    host.turn()
    assert host.stored() == rows + 4
    assert host.duplicated() == {}
