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

import logging
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
SECRET = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"  # the default bearer_token pattern
# Over 64k characters of one repeated paragraph: the always-on assistant quarantine replaces it.
BROKEN = ("I am drafting the same response again. I will repeat the plan, restate the same tool loop, and "
          "continue without adding new information.\n") * 600
CHILD = "cleanup-rewrite-proof-child"


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

    def __init__(self, tmp_path, **config):
        self.tmp_path, self.clock, self.n_turns, self.config = tmp_path, 1_760_000_000.0, 0, config
        self.live: list = [{"role": "system", "content": "system prompt"}]
        self.engines: list = []
        self.engine = self.bind()

    def bind(self, session_id=SID):
        engine = LCMEngine(config=LCMConfig(
            database_path=str(self.tmp_path / "lcm.db"),
            large_output_externalization_path=str(self.tmp_path / "externalized"),
            fresh_tail_count=2, leaf_chunk_tokens=100, fresh_tail_pressure_yield_enabled=False,
            deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5, **self.config,
        ), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start(session_id, platform="telegram", conversation_id=CID, context_length=200_000)
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

    def tool_turn(self, output=None):
        i, self.n_turns = self.n_turns, self.n_turns + 1
        call = f"call_{i}"
        self.live += [
            {"role": "user", "content": f"[T{i}] look it up{PAD}", "timestamp": self.stamp()},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call, "content": output or f"tool output {i}:" + " result" * 60},
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

    def commit_in_place(self, result, before, session_id=SID):
        """Hermes's in-place order: end on the list compress() was given, adopt, then the compression boundary."""
        self.engine.on_session_end(session_id, before)
        self.live = result
        self.engine.on_session_start(session_id, boundary_reason="compression", old_session_id=session_id,
                                     platform="telegram", conversation_id=CID, context_length=200_000)

    def restart(self, session_id=SID):
        self.engine.shutdown()
        self.engine = self.bind(session_id)

    def rows(self):
        """(role, content, tool_call_id, tool_calls) per stored row, in store order."""
        return [tuple(row) for row in self.engine._store._conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages WHERE session_id = ? ORDER BY store_id",
            (SID,)).fetchall()]

    def stored(self):
        return self.engine._store.get_session_count(SID)

    def count(self, content):
        return sum(1 for row in self.rows() if row[1] == content)

    def duplicated(self):
        return {key: n for key, n in Counter(self.rows()).items() if n > 1}

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


def _change(host, change, *, insert_at, replace_at):
    """The host changes its list before the cursor: an unstamped /steer row inserted, or a stamped user row
    replaced by a new one (another stamp). Returns the changed row's content."""
    if change == "insert":
        host.live.insert(insert_at, dict(STEER))
        return STEER["content"]
    content = f"[edited] the user rewrote this turn{PAD}"
    host.live[replace_at] = {"role": "user", "content": content, "timestamp": host.stamp()}
    return content


def _restart_stores_only_the_next_turn(host):
    """A restart and one turn: the restarted engine's first ingest stores that turn's two rows and nothing again.
    (Duplicates the warm path already made stay as they were: #633's displaced unstamped reply, #259's batch.)"""
    rows, duplicated = host.stored(), host.duplicated()
    host.restart()
    host.turn()
    assert host.stored() == rows + 2
    assert host.duplicated() == duplicated


@pytest.mark.parametrize("change", ["insert", "replace"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_row_changed_among_the_warm_turns_is_not_stored_again_after_a_restart(host, variant, change):
    """The field chain: after the committed cleanup list, the host changes its list among the warm turns, and
    the next turn stores the changed row at the store tail, out of host order."""
    _turns(host, variant)
    _commit_cleanup(host, variant)
    host.turn()
    host.turn()
    changed = _change(host, change, insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
    host.turn()
    assert host.count(changed) == 1
    _restart_stores_only_the_next_turn(host)
    assert host.count(changed) == 1


@pytest.mark.parametrize("change", ["insert", "replace"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_repeated_reply_after_the_changed_row_still_re_keys(host, variant, change):
    """The row right after the one stored out of order repeats an earlier warm reply ("OK"). Mapped again on its
    own, it must take its own stored copy, not the earlier reply's (already claimed), or the re-key gives up and
    the restart stores the session again."""
    _turns(host, variant)
    _commit_cleanup(host, variant)
    host.turn(reply=f"OK{PAD}")
    host.turn(reply=f"OK{PAD}")
    host.turn()
    changed = _change(host, change, insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
    proof = host.engine._durable_commit_proof_payload()
    host.turn()
    assert host.count(changed) == 1
    assert host.engine._durable_commit_proof_payload() != proof  # re-keyed
    _restart_stores_only_the_next_turn(host)
    assert host.count(changed) == 1


@pytest.mark.parametrize("change", ["insert", "replace"])
def test_a_row_changed_after_a_published_compaction_is_not_stored_again_after_a_restart(host, change, monkeypatch):
    """The whole field order: an earlier published compaction (its summary and carried rows head the list), the
    committed cleanup list, warm turns, then a row the host changes among them. The re-key proves that list too."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
    _turns(host, None)
    host.compact()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda *a, **k: pytest.fail("summariser called"))
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn()
    host.turn()
    changed = _change(host, change, insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
    host.turn()
    assert host.count(changed) == 1
    _restart_stores_only_the_next_turn(host)
    assert host.count(changed) == 1


@pytest.mark.parametrize("change", ["insert", "replace"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_the_first_cold_bind_after_an_upgrade_re_keys_an_old_writers_stale_proof(
        host, variant, change, monkeypatch, caplog):
    """The upgrade path (P-S2): an old writer ran the whole field order, published no cleanup-only proof and had no
    re-key, so the published compaction's proof is stale at shutdown. The first cold bind after the install advances
    its cursor past the scaffold prefix and stores the rows again; it must re-key the proof whatever the
    reconciliation action, or every later restart stores them again."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
    _turns(host, None)
    host.compact()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda *a, **k: pytest.fail("summariser called"))
    with monkeypatch.context() as old_writer:
        prepare = LCMEngine._prepare_compress_commit_proof

        def without_cleanup_proof(self, proof, result):
            proof.pop("cleanup_rewrite", None)  # the old writer never marks a cleanup-only proof for the boundary
            return prepare(self, proof, result)

        old_writer.setattr(LCMEngine, "_prepare_compress_commit_proof", without_cleanup_proof)
        old_writer.setattr(LCMEngine, "_rekey_durable_commit_proof", lambda self, messages: None, raising=False)
        _turns(host, variant)
        _commit_cleanup(host, variant)
        host.turn()
        host.turn()
        changed = _change(host, change, insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
        host.turn()
        assert host.count(changed) == 1
        host.engine.shutdown()
    caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")
    caplog.clear()
    host.restart()
    host.turn()
    assert host.engine._last_ingest_reconciliation.get("action") == "advanced cursor"
    assert "LCM durable commit proof re-keyed after an out-of-order store" in caplog.text
    _restart_stores_only_the_next_turn(host)


@pytest.mark.parametrize("change", ["insert", "replace"])
def test_a_row_changed_inside_the_committed_list_is_not_stored_again_after_a_restart(host, change):
    """The first ingest after the commit no longer finds the committed list as the cursor's prefix and persists
    the batch (#259); the restart after it must not persist it again."""
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    changed = _change(host, change, insert_at=len(host.live) - 2, replace_at=len(host.live) - 4)
    host.turn()
    assert host.count(changed) == 1
    _restart_stores_only_the_next_turn(host)
    assert host.count(changed) == 1


@pytest.mark.parametrize("reply", [
    f"a new reply to the resent turn{PAD}",
    # Summary-shaped text that no DAG node verifies is real content (#486), not a scaffold.
    f"as the summary said: [Recent Summary (d0, node 7)] the plan [Expand for details: lcm_expand 7]{PAD}",
], ids=["plain", "quotes-summary"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_reply_the_warm_path_never_stored_is_not_proven_away(host, variant, reply):
    """I1: a retry or an edit-resend replaces the stamped user row and its unstamped reply between two ingests.
    The warm anchor stores the new user row out of host order but keeps the new reply as a positional replay, so
    it is never stored warm (a gap that predates the re-key). No proof may claim that reply: the restart stores it
    (with duplicates, as before the re-key), and a second restart stores only the next turn."""
    _turns(host, variant)
    _commit_cleanup(host, variant)
    host.turn()
    host.turn()
    user = f"[resent] the user resent this turn{PAD}"
    host.live[-2:] = [{"role": "user", "content": user, "timestamp": host.stamp()},
                      {"role": "assistant", "content": reply}]
    proof = host.engine._durable_commit_proof_payload()
    host.turn()
    assert (host.count(user), host.count(reply)) == (1, 0)
    assert host.engine._durable_commit_proof_payload() == proof  # not re-keyed over the unstored reply
    host.restart()
    host.turn()
    assert host.count(reply) == 1
    _restart_stores_only_the_next_turn(host)
    assert host.count(reply) == 1


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
    stored = {(row[0], row[1]) for row in host.rows()}
    assert all((m["role"], m["content"]) in stored for m in host.live if m["role"] in ("user", "assistant"))


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


def test_a_later_cleanup_pass_does_not_prove_away_a_reply_the_warm_path_never_stored(host):
    """I1: the edit-resend above leaves the new reply unstored; a later cleanup-only pass (another reasoning block
    stripped) hands the host a list that holds it. That pass's proof stays process-local, as before #1286: made
    durable at the boundary it would advance the restart's cold cursor past the reply, which is never stored."""
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn(reply=f"{THINK}reply to T{host.n_turns}{PAD}")
    host.turn()
    user, reply = f"[resent] the user resent this turn{PAD}", f"a new reply to the resent turn{PAD}"
    host.live[-2:] = [{"role": "user", "content": user, "timestamp": host.stamp()},
                      {"role": "assistant", "content": reply}]
    host.turn()
    assert (host.count(user), host.count(reply)) == (1, 0)
    _commit_cleanup(host, "reasoning")
    host.restart()
    host.turn()
    assert host.count(reply) == 1
    _restart_stores_only_the_next_turn(host)
    assert host.count(reply) == 1


def _carried(payload, session_id):
    """The store ids of ``session_id`` that a durable proof's carry ranges cover."""
    return {store_id for sid, start, end in payload["carry_ranges"] if sid == session_id
            for store_id in range(start + 1, end + 1)}


def test_a_re_key_in_a_compression_child_keeps_the_parent_rows_it_carries(host, monkeypatch):
    """A cleanup-only list rotated into a compression child carries its parent rows. A steer inserted before the
    parent tail is stored at the child's tail; the in-order map then leaves the parent rows after it unmapped and
    the re-key's second mapping finds them. The re-keyed proof carries them too, or a later compaction consumes
    their host rows without their store ids (lost provenance)."""
    proof = _rotate_into_a_child_and_insert_a_steer(host)
    host.turn()
    rekeyed = host.engine._durable_commit_proof_payload()
    assert rekeyed != proof  # re-keyed
    assert _carried(proof, SID) <= _carried(rekeyed, SID)
    _assert_a_compaction_sources_the_parent_rows_it_consumes(host, monkeypatch)


def _rotate_into_a_child_and_insert_a_steer(host):
    """A cleanup-only list rotated into a compression child, one warm turn, then a steer inserted before the carried
    parent tail (T4). Returns the child's durable proof before the steer is stored."""
    _turns(host, "reasoning")
    result, before = host.cleanup_pass(preflight=True)
    host.engine.on_session_end(SID, before)
    host.live = result
    host.engine.on_session_start(CHILD, boundary_reason="compression", old_session_id=SID, platform="telegram",
                                 conversation_id=CID, context_length=200_000)
    host.turn()
    host.live.insert(next(i for i, m in enumerate(host.live) if str(m.get("content")).startswith("[T4]")),
                     dict(STEER))
    proof = host.engine._durable_commit_proof_payload()
    assert _carried(proof, SID)  # the rotation carried the parent rows
    return proof


def _assert_a_compaction_sources_the_parent_rows_it_consumes(host, monkeypatch):
    """A forced compaction of the child: every parent row whose host row it consumes has a copy among the summaries'
    source ids (provenance)."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
    before = deepcopy(host.live)
    result = host.engine.compress(host.live, force=True)
    assert host.engine._last_compression_status == "compacted"
    kept = {m.get("content") for m in result}
    # A parent row's copies: the warm path re-stores a displaced stamped user in the child (a duplicate, I1), and
    # the summary may source either copy; a row with no copy among the sources lost its provenance.
    copies: dict = {}
    for store_id, session_id, role, content in host.engine._store._conn.execute(
            "SELECT store_id, session_id, role, content FROM messages WHERE session_id IN (?, ?)", (SID, CHILD)):
        copies.setdefault((role, content), set()).add(store_id)
        if session_id == SID:
            copies.setdefault(("parent", role, content), set()).add(store_id)
    sources = {store_id for session_id in (SID, CHILD) for node in host.engine._dag.get_session_nodes(session_id)
               if node.depth == 0 for store_id in node.source_ids}
    consumed = [(m["role"], m["content"]) for m in before  # the system prompt is never a summary source
                if m["role"] != "system" and m.get("content") not in kept
                and ("parent", m["role"], m.get("content")) in copies]
    assert consumed and [key for key in consumed if not copies[key] & sources] == []


def test_a_later_cleanup_in_a_compression_child_keeps_the_parent_rows_it_carries(host, monkeypatch):
    """Round 7: the same child, the steer stored at its tail by a turn whose reply carries reasoning, then a second
    committed cleanup (that reasoning stripped) and a restart. The cleanup proof's coverage guard maps the parent rows
    after the steer again; its carry must keep them (and the previous proof's uncovered parent ranges), as the re-key
    does, or the parent carry shrinks and a later compaction consumes their host rows without their store ids."""
    _rotate_into_a_child_and_insert_a_steer(host)
    host.turn(reply=f"{THINK}reply to T{host.n_turns}{PAD}")
    carried = _carried(host.engine._durable_commit_proof_payload(), SID)
    result, before = host.cleanup_pass(preflight=True)
    assert THINK not in str(result) and THINK in str(before)
    assert host.engine._compress_commit_proof.get("cleanup_rewrite") is True
    host.commit_in_place(result, before, session_id=CHILD)
    host.restart(session_id=CHILD)
    assert carried <= _carried(host.engine._durable_commit_proof_payload(), SID)
    _assert_a_compaction_sources_the_parent_rows_it_consumes(host, monkeypatch)


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped", "stamped"])
def test_a_one_row_cleanup_proof_does_not_swallow_a_new_identical_message(host, stamped):
    """Round 7 (I1): a committed cleanup over a session whose one effective row is a user message externalized into a
    stub. Unstamped, nothing tells that row from a new message with the same text (the stub's identity restores the
    payload), so its proof stays local: after a restart, a delta-only delivery of the new message is stored. A host
    stamp tells them apart (the proof's host-occurrence snapshot), so that proof stays durable."""
    def message():
        return {"role": "user", "content": f"[T0] {RAW}", **({"timestamp": host.stamp()} if stamped else {})}

    host.live = [message()]  # no system prompt: the one row is the whole list
    host.engine.ingest(host.live)
    assert host.stored() == 1
    result, before = host.cleanup_pass(preflight=False)
    assert any(str(m.get("content")).startswith("[Externalized payload: kind=raw_payload;") for m in result)
    host.commit_in_place(result, before)
    durable = host.engine._durable_commit_proof_payload() is not None
    host.restart()
    host.live = [message()]  # the gateway delivers only the new message
    host.engine.ingest(host.live)
    assert host.stored() == 2  # (the store holds the externalized form, so the rows are counted, not their text)
    assert durable is stamped


def _two_new_messages():
    return [{"role": "user", "content": f"[{tag}] {RAW}"} for tag in "AB"]  # distinct, unstamped, oversized


def test_a_multi_row_delta_cleanup_proof_does_not_swallow_new_identical_messages(host):
    """Round 7b (I1): a session already holds history; after a restart the gateway delivers only two new unstamped
    oversized messages (a delta, not a replay of the session), and a committed cleanup stubs them. That list does not
    replay the session from the compaction frontier, so its proof stays local: after the next restart, a new raw
    delivery of the same two texts is stored. Proven durable, it matched them and neither was stored."""
    _turns(host, None, n=3)
    host.restart()
    host.live = _two_new_messages()
    host.engine.ingest(host.live)
    rows = host.stored()
    result, before = host.cleanup_pass(preflight=False)
    assert sum(str(m.get("content")).startswith("[Externalized payload: kind=raw_payload;") for m in result) == 2
    host.commit_in_place(result, before)
    durable = host.engine._durable_commit_proof_payload()
    host.restart()
    host.live = _two_new_messages()  # genuinely new messages with the same texts, delivered alone
    host.engine.ingest(host.live)
    assert host.stored() == rows + 2  # (the store holds the externalized form, so the rows are counted)
    assert durable is None


def test_a_multi_row_cleanup_that_replays_the_session_keeps_its_proof_durable(host):
    """Round 7b control: the same two-row cleanup in a session whose host list replays its history from the frontier
    proves the first stored row after it, so the proof stays durable and a restart stores only the next turn."""
    _turns(host, None, n=3)
    host.live += _two_new_messages()
    host.engine.ingest(host.live)
    result, before = host.cleanup_pass(preflight=False)
    assert sum(str(m.get("content")).startswith("[Externalized payload: kind=raw_payload;") for m in result) == 2
    assert host.engine._compress_commit_proof.get("cleanup_rewrite") is True
    host.commit_in_place(result, before)
    assert host.engine._durable_commit_proof_payload() is not None
    _restart_stores_only_the_next_turn(host)
    assert host.duplicated() == {}


@pytest.mark.parametrize("variant", ["stub", "reasoning"])
def test_a_delta_only_bind_does_not_re_key_the_proof_to_the_delta(host, variant):
    """Round 7 (I1): a session holding rows with a durable proof, then a restart whose first delivery is only one new
    unstamped user row (a delta-only gateway). The bind replays no stored row, so it has no replayed list to bind: the
    row is stored and the proof stays. Re-keyed to the delta, the next restart's identical delta matched it and was
    not stored."""
    _turns(host, variant)
    _commit_cleanup(host, variant)
    proof = host.engine._durable_commit_proof_payload()
    delta = {"role": "user", "content": f"a new message the gateway delivers alone{PAD}"}
    host.restart()
    host.live = [dict(delta)]
    host.engine.ingest(host.live)
    assert host.count(delta["content"]) == 1
    durable = host.engine._durable_commit_proof_payload()
    host.restart()
    host.live = [dict(delta)]
    host.engine.ingest(host.live)
    assert host.count(delta["content"]) == 2
    assert durable == proof


@pytest.mark.parametrize("row", ["redacted-tool", "quarantined-reply"])
def test_a_row_stored_in_its_replay_form_still_re_keys(tmp_path, row, monkeypatch, caplog):
    """The whole field order with a warm row the store holds in its replay form (a tool output redacted by the
    sensitive patterns, an assistant loop the quarantine replaced). The re-key proves the list the cold reader
    reads, so it maps that row to its stored copy and re-keys; given the raw host row it found no stored copy,
    kept the old proof, and the restart stored the session again."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _stub_summarizer)
    host = Host(tmp_path, sensitive_patterns_enabled=True)
    try:
        _turns(host, None)
        host.compact()
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda *a, **k: pytest.fail("summariser called"))
        _turns(host, "reasoning")
        _commit_cleanup(host, "reasoning")
        if row == "redacted-tool":
            host.tool_turn(output=f"tool output: curl -H '{SECRET}' https://example.invalid ok")
        else:
            host.turn(reply=BROKEN)
        host.turn()
        host.turn()
        changed = _change(host, "insert", insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
        caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")
        host.turn()
        assert "LCM durable commit proof re-keyed after an out-of-order store" in caplog.text
        assert host.count(changed) == 1
        _restart_stores_only_the_next_turn(host)
        assert host.count(changed) == 1
    finally:
        host.close()


def test_a_failed_re_key_write_leaves_the_old_proof_and_snapshot(host, monkeypatch, caplog):
    """The re-key writes the proof, then its occurrence snapshot, inside the ingest's savepoint. A failure in the
    second write rolls both back (the old proof stays, I1); the ingest's rows stay stored once."""
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn()
    host.turn()
    host.live.insert(len(host.live) - 3, dict(STEER))
    store = host.engine._store
    key = host.engine._replay_snapshot_metadata_key("host_occurrence_snapshot")
    proof, snapshot = host.engine._durable_commit_proof_payload(), store.read_metadata_json(key)
    write, reached = store.write_metadata_json, []

    def fail_snapshot(keys, serialized, **kwargs):
        if key in keys:
            reached.append(keys)
            raise RuntimeError("snapshot write failed")
        return write(keys, serialized, **kwargs)

    monkeypatch.setattr(store, "write_metadata_json", fail_snapshot)
    caplog.set_level(logging.WARNING, logger="hermes_lcm.engine")
    host.turn()
    monkeypatch.setattr(store, "write_metadata_json", write)
    # The rollback path ran: the injected write was reached and the re-key reported its failure.
    assert reached and "LCM durable commit proof re-key failed" in caplog.text
    assert host.engine._durable_commit_proof_payload() == proof
    assert store.read_metadata_json(key) == snapshot
    assert [host.count(m["content"]) for m in [STEER, *host.live[-2:]]] == [1, 1, 1]
