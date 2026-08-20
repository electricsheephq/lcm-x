"""Out-of-band user messages appended to tool output must not break
replay reconciliation.

When the user sends a message while the agent is mid-turn, the host
appends it to the tool output currently being delivered, wrapped in an
"[OUT-OF-BAND USER MESSAGE ...] ... [/OUT-OF-BAND USER MESSAGE]" block.
LCM persists the tool row with the block attached.  On resume/restart
the host replays the same tool result WITHOUT the block (the user
message was already delivered separately), so the stored row and the
incoming row differ by exactly that block -> identity mismatch ->
cursor=0 -> full session re-ingest (duplication).

Fix: _message_replay_identity strips the out-of-band block so identity
matching survives the delivery split.

All tests use synthetic messages.  No real session data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hermes_lcm.config import LCMConfig

OOB_BLOCK = (
    "\n\n[OUT-OF-BAND USER MESSAGE — a direct message from the user, "
    "delivered mid-turn; not tool output]\ncheck the 2026 logs instead\n"
    "[/OUT-OF-BAND USER MESSAGE]"
)


def _make_engine(tmp_path: Path, **overrides):
    # engine.py imports agent.context_engine at module level; provide a
    # stub here (not at module level, so collection order of other test
    # files is unaffected), then clear any partial import left by conftest.
    if "agent.context_engine" not in sys.modules:
        _agent_mod = ModuleType("agent")
        _agent_mod.__path__ = []
        _ce_mod = ModuleType("agent.context_engine")

        class _StubContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        _ce_mod.ContextEngine = _StubContextEngine
        sys.modules["agent"] = _agent_mod
        sys.modules["agent.context_engine"] = _ce_mod

    _existing = sys.modules.get("hermes_lcm.engine")
    if _existing is not None and not hasattr(_existing, "LCMEngine"):
        sys.modules.pop("hermes_lcm.engine", None)
    from hermes_lcm.engine import LCMEngine

    defaults = dict(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=False,
        fresh_tail_count=64,
    )
    defaults.update(overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "oob-block-test",
        platform="cli",
        context_length=1000000,
    )
    return engine


def _seed_durable_tool_row(engine, content, tool_call_id="c1"):
    """Persist one tool row so ITS out-of-band text counts as durable.

    The proof is occurrence-bound, so the seeded row must be the same row the
    identity under test describes: same ``tool_call_id``, same content minus
    the out-of-band blocks.
    """
    engine._ingest_messages(
        [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "seeding."},
            {"role": "tool", "tool_call_id": tool_call_id, "content": content},
        ]
    )


class TestOutOfBandBlockIdentity:
    """The OOB block is identity-transparent once its text is durable."""

    def test_identity_strips_oob_block_preserves_content(self, tmp_path):
        """Stored row (with block) and incoming row (without) match."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "build finished, 0 errors", "exit_code": 0}'
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK},
                stored_row=True,
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with == id_without, (
                "OOB-block-stripped identity must match plain content identity"
            )
            # The tool's own content must survive
            assert "build finished" in id_with[1]
        finally:
            engine.shutdown()

    def test_identity_multiple_oob_blocks_stripped(self, tmp_path):
        """Multiple appended blocks (repeated interruptions) all strip."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "step one done", "exit_code": 0}'
            twice = base + OOB_BLOCK + OOB_BLOCK.replace(
                "check the 2026 logs instead", "and also run the tests"
            )
            id_twice = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": twice},
                stored_row=True,
            )
            id_plain = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_twice == id_plain
        finally:
            engine.shutdown()

    def test_incoming_identity_strips_block_already_durable(self, tmp_path):
        """An incoming block the store already holds is identity-transparent."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "build finished, 0 errors", "exit_code": 0}'
            _seed_durable_tool_row(engine, base + OOB_BLOCK)
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with == id_without, (
                "A durable OOB block must strip from the incoming identity too"
            )
        finally:
            engine.shutdown()

    def test_incoming_identity_keeps_block_not_yet_durable(self, tmp_path):
        """An incoming block the store has never seen keeps the row distinct.

        The host only ever APPENDS the steer marker (hermes-agent
        agent/conversation_loop.py:1723 mutates a tool row in place and no host
        path removes the marker), so a block absent from the store is a genuine
        user instruction arriving for the first time.  Collapsing it onto the
        stored row would advance reconciliation past the only copy of that text.
        """
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "build finished, 0 errors", "exit_code": 0}'
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with != id_without, (
                "An unproven OOB block must not be treated as identity-transparent"
            )
            assert "check the 2026 logs instead" in id_with[1]
        finally:
            engine.shutdown()

    def test_incoming_identity_keeps_block_durable_only_on_another_row(self, tmp_path):
        """A block durable on a DIFFERENT row is not proof for this row.

        ``STEER_MARKER_OPEN`` is a static host constant (hermes-agent
        agent/prompt_builder.py:675 — no timestamp, no occurrence id), so the
        same /steer text sent twice in one session produces byte-identical
        blocks.  Proving durability from the block text alone would let the
        first occurrence vouch for a second one appended to a different tool
        row, collapse that row onto its stored pre-steer copy, and advance the
        cursor past the only copy of the repeated instruction.
        """
        engine = _make_engine(tmp_path)
        try:
            first = '{"output": "deploy ok", "exit_code": 0}'
            second = '{"output": "smoke ok", "exit_code": 0}'
            _seed_durable_tool_row(engine, first + OOB_BLOCK, tool_call_id="c1")
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c2", "content": second + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c2", "content": second}
            )
            assert id_with != id_without, (
                "Durability proof must be bound to this row's occurrence, not "
                "to the block text shared with an earlier row"
            )
            assert "check the 2026 logs instead" in id_with[1]
        finally:
            engine.shutdown()

    def test_incoming_identity_keeps_block_when_stored_row_lacks_it(self, tmp_path):
        """Same row, but the stored copy predates the block -> keep it distinct."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "deploy ok", "exit_code": 0}'
            _seed_durable_tool_row(engine, base, tool_call_id="c1")
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with != id_without
            assert "check the 2026 logs instead" in id_with[1]
        finally:
            engine.shutdown()

    def test_incoming_identity_keeps_second_identical_block_on_same_row(self, tmp_path):
        """One stored occurrence does not prove TWO incoming occurrences.

        The user repeats a /steer verbatim while the SAME tool result is still
        active, so the host appends a second byte-identical block to the row it
        already appended the first one to.  The store holds one copy; the replay
        carries two.  A membership test ("does this block appear in the stored
        content?") passes for both entries, strips both, and collapses the row
        onto its stored single-block copy — the second instruction is never
        ingested.  Proof is count-aware: the surplus occurrence is new text.
        """
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "deploy ok", "exit_code": 0}'
            _seed_durable_tool_row(engine, base + OOB_BLOCK, tool_call_id="c1")
            id_twice = engine._message_replay_identity(
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": base + OOB_BLOCK + OOB_BLOCK,
                }
            )
            id_once = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK},
                stored_row=True,
            )
            assert id_twice != id_once, (
                "A second byte-identical block is a second user instruction; "
                "one stored occurrence cannot vouch for two incoming ones"
            )
            assert id_twice[1].count("check the 2026 logs instead") == 2
        finally:
            engine.shutdown()

    def test_incoming_identity_strips_when_stored_row_carries_more_copies(self, tmp_path):
        """The inverse: stored 2x, incoming 1x still proves durable.

        Count-awareness is a floor, not an equality: the store already carries
        every occurrence being stripped (and one more), so collapsing the
        incoming row onto it loses nothing and must not force a re-ingest.
        """
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "deploy ok", "exit_code": 0}'
            _seed_durable_tool_row(engine, base + OOB_BLOCK + OOB_BLOCK, tool_call_id="c1")
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with == id_without, (
                "A stored row carrying more copies than are being stripped is "
                "still proof that this occurrence is durable"
            )
        finally:
            engine.shutdown()

    def test_incoming_identity_keeps_block_when_row_has_no_tool_call_id(self, tmp_path):
        """An ID-less row has no unique identity, so nothing can prove it durable.

        Callers default a missing ``tool_call_id`` to ``""``, so every ID-less
        row shares one identity key.  Matching on it lets an EARLIER ID-less row
        that happens to reduce to the same payload vouch for a block arriving
        fresh on a LATER ID-less row — the block strips, the enriched row
        collapses onto the stored one, and the instruction is never ingested.
        With no unique row identity the proof cannot hold: fail closed and keep
        the row distinct (duplication over silent loss).
        """
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "deploy ok", "exit_code": 0}'
            _seed_durable_tool_row(engine, base + OOB_BLOCK, tool_call_id="")
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "", "content": base}
            )
            assert id_with != id_without, (
                "An empty tool_call_id is not a unique row identity; it must "
                "never prove an out-of-band block durable"
            )
            assert "check the 2026 logs instead" in id_with[1]
        finally:
            engine.shutdown()

    def test_oob_text_without_markers_not_stripped(self, tmp_path):
        """Content merely mentioning the phrase keeps its identity."""
        engine = _make_engine(tmp_path)
        try:
            content = "The docs mention OUT-OF-BAND USER MESSAGE markers."
            id_a = engine._message_replay_identity({"role": "user", "content": content})
            assert content in id_a[1], "Text without real markers must not be stripped"
        finally:
            engine.shutdown()


class TestOutOfBandBlockReconciliation:
    """Restart reconciliation across the OOB delivery split."""

    def test_oob_block_no_reingest_on_restart(self, tmp_path):
        """Stored with block, replayed without block -> no re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "compile succeeded", "exit_code": 0}'
            turn1 = [
                {"role": "user", "content": "Build the module"},
                {"role": "assistant", "content": "Building."},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": tool_content + OOB_BLOCK,  # persisted mid-turn
                },
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            # Restart: host replays the tool result WITHOUT the OOB block
            # (the user message was delivered separately), plus one new turn.
            turn2 = [
                {"role": "user", "content": "Build the module"},
                {"role": "assistant", "content": "Building."},
                {"role": "tool", "tool_call_id": "c1", "content": tool_content},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Now run the tests"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("oob-block-test")
            assert count == 5, (
                f"Expected 5 (4 stored + 1 new), got {count}. "
                "OOB block identity mismatch -> cursor=0 -> full re-ingest."
            )
        finally:
            engine.shutdown()

    def test_reverse_direction_preserves_out_of_band_text(self, tmp_path):
        """Stored without block, replayed with block -> the steer text lands.

        This is the direction the live host actually produces: a /steer is
        appended in place to a tool row LCM has already ingested, and the host
        never removes the marker again, so the replay carries text the store has
        never seen.  Collapsing that row onto the stored one would advance the
        cursor past the only copy of the user's instruction.  Re-ingest (with
        the duplication that implies) is the loss-avoiding direction here.
        """
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {"role": "tool", "tool_call_id": "c1", "content": tool_content},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            turn2 = turn1.copy()
            turn2[2] = {
                "role": "tool",
                "tool_call_id": "c1",
                "content": tool_content + OOB_BLOCK,
            }
            turn2.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            stored = engine._store.get_session_tail("oob-block-test", limit=64)
            assert any(
                "check the 2026 logs instead" in (row.get("content") or "")
                for row in stored
            ), "The out-of-band user instruction must reach the store"
        finally:
            engine.shutdown()

    def test_repeated_identical_steer_on_a_later_row_still_lands(self, tmp_path):
        """The same /steer text sent twice must land on BOTH tool rows.

        The host marker carries no occurrence id (``STEER_MARKER_OPEN`` is a
        static constant), so a user repeating an instruction verbatim — the
        common case, because the first steer was ignored — produces two
        byte-identical blocks on two different tool rows.  Before the
        occurrence-bound proof, the durable block on c1 vouched for the fresh
        one on c2: the enriched c2 row collapsed onto its stored pre-steer
        copy, the cursor advanced past it, and the repeated instruction never
        reached the store (verified on 42c33b9f: 8 -> 9 rows, none carrying
        both "smoke ok" and the steer text).
        """
        engine = _make_engine(tmp_path)
        try:
            first_tool = '{"output": "deploy ok", "exit_code": 0}'
            second_tool = '{"output": "smoke ok", "exit_code": 0}'

            # Turn 1 carries the first steer and is ingested as-is.
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": first_tool + OOB_BLOCK,
                },
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            # Turn 2 is ingested before the second steer arrives, so the store
            # holds c2 WITHOUT a block.  Reconciliation must match cleanly here
            # or the rest of the test proves nothing.
            turn2 = turn1 + [
                {"role": "user", "content": "Smoke test"},
                {"role": "assistant", "content": "Testing."},
                {"role": "tool", "tool_call_id": "c2", "content": second_tool},
                {"role": "assistant", "content": "Green."},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)
            assert engine._store.get_session_count("oob-block-test") == 8, (
                "Setup guard: turn 2 must reconcile against the stored tail, "
                "otherwise the re-ingest hides the drop this test looks for"
            )

            # Second /steer: same text, byte-identical block, appended in place
            # to c2 — a row LCM has already ingested.
            replay = turn2.copy()
            replay[6] = {
                "role": "tool",
                "tool_call_id": "c2",
                "content": second_tool + OOB_BLOCK,
            }
            replay.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(replay)

            stored = engine._store.get_session_tail("oob-block-test", limit=64)
            assert any(
                "smoke ok" in (row.get("content") or "")
                and "check the 2026 logs instead" in (row.get("content") or "")
                for row in stored
            ), (
                "The repeated out-of-band instruction must reach the store on "
                "its own tool row; an identical earlier block on another row "
                "is not proof that this occurrence is durable"
            )
        finally:
            engine.shutdown()

    def test_repeated_identical_steer_on_the_same_row_still_lands(self, tmp_path):
        """The same /steer repeated while ONE tool result is active must land.

        Sibling of the later-row case above, and the residual it left behind:
        here both byte-identical blocks are appended to the SAME row, so no
        stored row ever carries both occurrences.  Under a membership test each
        incoming entry matched the single stored copy, the enriched row was
        stripped down to its stored identity, the cursor advanced past it, and
        the user's repeated instruction never reached the store.
        """
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": tool_content + OOB_BLOCK,
                },
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            # The user repeats the instruction verbatim before the turn ends,
            # so the host appends a second identical block to the same row.
            replay = turn1.copy()
            replay[2] = {
                "role": "tool",
                "tool_call_id": "c1",
                "content": tool_content + OOB_BLOCK + OOB_BLOCK,
            }
            replay.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(replay)

            stored = engine._store.get_session_tail("oob-block-test", limit=64)
            assert any(
                (row.get("content") or "").count("check the 2026 logs instead") == 2
                for row in stored
            ), (
                "Both occurrences of the repeated out-of-band instruction must "
                "reach the store; one durable copy is not proof for two"
            )
        finally:
            engine.shutdown()

    def test_repeated_identical_steer_on_a_later_id_less_row_still_lands(self, tmp_path):
        """Two ID-less tool rows sharing a base output must each keep their steer.

        Not every host tool row carries a ``tool_call_id``; callers default the
        missing value to ``""``, so two such rows are indistinguishable to the
        durability proof.  When they also share a base output (a repeated
        command — ``git status`` twice — is the ordinary case), the earlier row's
        durable block satisfies the occurrence check for a block arriving fresh
        on the later row: the enriched row strips down to its stored pre-steer
        identity, reconciliation advances past it, and only the following user
        message is stored.  The repeated instruction is lost.
        """
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'

            # Turn 1: the first steer is appended before ingest, so the store
            # holds the first ID-less row WITH the block.
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {"role": "tool", "tool_call_id": "", "content": tool_content + OOB_BLOCK},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            # Turn 2 runs the same command again and is ingested before the
            # second steer arrives, so the store holds the second ID-less row
            # WITHOUT a block.
            turn2 = turn1 + [
                {"role": "user", "content": "Deploy again"},
                {"role": "assistant", "content": "Deploying."},
                {"role": "tool", "tool_call_id": "", "content": tool_content},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)
            assert engine._store.get_session_count("oob-block-test") == 8, (
                "Setup guard: turn 2 must reconcile against the stored tail, "
                "otherwise the re-ingest hides the drop this test looks for"
            )

            # Second /steer: same text, byte-identical block, appended in place
            # to the SECOND ID-less row.
            replay = turn2.copy()
            replay[6] = {
                "role": "tool",
                "tool_call_id": "",
                "content": tool_content + OOB_BLOCK,
            }
            replay.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(replay)

            stored = engine._store.get_session_tail("oob-block-test", limit=64)
            carriers = sum(
                1
                for row in stored
                if "check the 2026 logs instead" in (row.get("content") or "")
            )
            assert carriers >= 2, (
                f"Expected both out-of-band instructions in the store, found "
                f"{carriers} row(s) carrying one.  An earlier ID-less row is "
                "not proof that a later ID-less row's block is durable."
            )
        finally:
            engine.shutdown()

    def test_id_less_row_with_block_converges_without_reingest(self, tmp_path):
        """An ID-less row carrying a block must still stop re-ingesting.

        Failing closed only on the INCOMING side would leave this row
        permanently unreconcilable: the stored copy reduces to its pre-steer
        text while every replay keeps the block, so each turn mismatches and
        re-ingests the whole session.  The block is therefore kept on BOTH
        sides for an ID-less row, which converges.
        """
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'
            enriched = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {"role": "tool", "tool_call_id": "", "content": tool_content + OOB_BLOCK},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(enriched)
            assert engine._store.get_session_count("oob-block-test") == 4

            replay = enriched.copy()
            replay.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(replay)

            count = engine._store.get_session_count("oob-block-test")
            assert count == 5, (
                f"Expected 5 (4 stored + 1 new), got {count}. "
                "An ID-less row carrying an OOB block must not re-ingest the "
                "session on every turn."
            )
        finally:
            engine.shutdown()

    def test_reverse_direction_converges_once_block_is_durable(self, tmp_path):
        """Once the enriched row is stored, replaying it again does not re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'
            enriched = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": tool_content + OOB_BLOCK,
                },
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(enriched)
            assert engine._store.get_session_count("oob-block-test") == 4

            replay = enriched.copy()
            replay.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(replay)

            count = engine._store.get_session_count("oob-block-test")
            assert count == 5, (
                f"Expected 5 (4 stored + 1 new), got {count}. "
                "A durable OOB block must not force a full re-ingest."
            )
        finally:
            engine.shutdown()
