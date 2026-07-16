import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore, ReplayFloodError


def _store(tmp_path, **config_overrides):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), **config_overrides)
    return MessageStore(tmp_path / "lcm.db", ingest_protection_config=config)


class TestStoreReplayFloodBackstop:
    def test_refuses_user_replay_flood_batch(self, tmp_path):
        store = _store(tmp_path)
        stored = [{"role": "user", "content": f"durable message {i}"} for i in range(10)]
        store.append_batch("flood-session", stored)
        before = store.get_session_count("flood-session")

        flood = [dict(msg) for msg in stored[:8]]
        flood.append({"role": "user", "content": "new message riding the flood"})

        with pytest.raises(ReplayFloodError):
            store.append_batch("flood-session", flood)
        assert store.get_session_count("flood-session") == before

    def test_allows_fresh_bulk_batch_without_priors(self, tmp_path):
        store = _store(tmp_path)
        batch = [{"role": "user", "content": f"brand new message {i}"} for i in range(20)]

        ids = store.append_batch("fresh-session", batch)

        assert len(ids) == len(batch)

    def test_allows_small_duplicate_delta_below_threshold(self, tmp_path):
        store = _store(tmp_path)
        stored = [{"role": "user", "content": f"durable message {i}"} for i in range(10)]
        store.append_batch("delta-session", stored)

        delta = [dict(stored[-2]), dict(stored[-1]), {"role": "user", "content": "new turn"}]
        ids = store.append_batch("delta-session", delta)

        assert len(ids) == len(delta)

    def test_refuses_internal_identical_identity_burst(self, tmp_path):
        store = _store(tmp_path)
        store.append_batch(
            "internal-session",
            [{"role": "assistant", "content": "repeated assistant row"}],
        )

        burst = [{"role": "assistant", "content": "repeated assistant row"} for _ in range(32)]
        with pytest.raises(ReplayFloodError):
            store.append_batch("internal-session", burst)

    def test_refuses_contentless_assistant_tool_call_replay_burst(self, tmp_path):
        store = _store(tmp_path)
        tool_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-replayed",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"index": 1}},
                }
            ],
        }
        store.append_batch("contentless-tool-call-session", [tool_call])

        with pytest.raises(ReplayFloodError):
            store.append_batch(
                "contentless-tool-call-session",
                [dict(tool_call) for _ in range(32)],
            )

    def test_refuses_reordered_contentless_tool_call_replay_atomically(self, tmp_path):
        store = _store(tmp_path)
        stored = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-canonical",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"filters": {"site": "docs", "lang": "en"}, "limit": 5},
                    },
                }
            ],
        }
        reordered = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "arguments": {"limit": 5, "filters": {"lang": "en", "site": "docs"}},
                        "name": "lookup",
                    },
                    "id": "call-canonical",
                }
            ],
        }
        session_id = "canonical-tool-call-session"
        store.append_batch(session_id, [stored])
        before = store.get_session_count(session_id)

        with pytest.raises(ReplayFloodError):
            store.append_batch(session_id, [reordered for _ in range(32)])

        assert store.get_session_count(session_id) == before
        fresh = {**reordered, "tool_calls": [{**reordered["tool_calls"][0], "id": "call-fresh"}]}
        assert len(store.append_batch(session_id, [fresh])) == 1
        assert store.get_session_count(session_id) == before + 1

    def test_allows_internal_distinct_duplicate_rows(self, tmp_path):
        store = _store(tmp_path)
        stored = [{"role": "assistant", "content": f"assistant row {i}"} for i in range(40)]
        store.append_batch("distinct-session", stored)

        # Distinct internal identities never accumulate against the internal
        # threshold even when every row duplicates stored content.
        ids = store.append_batch("distinct-session", [dict(msg) for msg in stored])

        assert len(ids) == len(stored)

    def test_allows_same_content_with_distinct_assistant_tool_calls(self, tmp_path):
        store = _store(tmp_path)
        store.append_batch(
            "tool-call-session",
            [{"role": "assistant", "content": "calling a tool"}],
        )
        batch = [
            {
                "role": "assistant",
                "content": "calling a tool",
                "tool_calls": [
                    {
                        "id": f"call-{idx}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"index": idx}},
                    }
                ],
            }
            for idx in range(32)
        ]

        ids = store.append_batch("tool-call-session", batch)

        assert len(ids) == len(batch)

    @pytest.mark.parametrize("distinct_field", ["id", "function", "arguments"])
    def test_contentless_tool_call_identity_preserves_distinct_fields(self, tmp_path, distinct_field):
        store = _store(tmp_path)
        session_id = f"distinct-tool-call-{distinct_field}"
        base_call = {
            "id": "call-stable",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"index": 0}},
        }
        store.append_batch(
            session_id,
            [{"role": "assistant", "content": None, "tool_calls": [base_call]}],
        )
        batch = []
        for idx in range(32):
            call = {
                "id": f"call-{idx}" if distinct_field == "id" else "call-stable",
                "type": "function",
                "function": {
                    "name": f"lookup-{idx}" if distinct_field == "function" else "lookup",
                    "arguments": {"index": idx} if distinct_field == "arguments" else {"index": 0},
                },
            }
            batch.append({"role": "assistant", "content": None, "tool_calls": [call]})

        ids = store.append_batch(session_id, batch)

        assert len(ids) == len(batch)

    def test_zero_threshold_disables_guard(self, tmp_path):
        store = _store(
            tmp_path,
            replay_flood_threshold_external=0,
            replay_flood_threshold_internal=0,
        )
        stored = [{"role": "user", "content": f"durable message {i}"} for i in range(10)]
        store.append_batch("disabled-session", stored)

        ids = store.append_batch("disabled-session", [dict(msg) for msg in stored])

        assert len(ids) == len(stored)


class TestEngineReplayFloodBackstop:
    def test_engine_recovers_new_tail_and_subsequent_ingests_after_refusal(self, tmp_path):
        db_path = tmp_path / "flood-engine.db"
        config = LCMConfig(database_path=str(db_path))
        before_restart = LCMEngine(config=config)
        before_restart.on_session_start(
            "flood-engine-session",
            platform="cli",
            conversation_id="flood-engine-conversation",
            context_length=200000,
        )
        persisted_messages = [{"role": "system", "content": "You are concise."}]
        persisted_messages.extend(
            {"role": "user", "content": f"durable message {i}"}
            for i in range(20)
        )
        before_restart._ingest_messages(persisted_messages)
        before_restart._store.close()
        before_restart._dag.close()
        before_restart._lifecycle.close()

        after_restart = LCMEngine(config=config)
        after_restart.on_session_start(
            "flood-engine-session",
            platform="cli",
            conversation_id="flood-engine-conversation",
            context_length=200000,
        )
        # A middle chunk of the durable history is not anchored at the tail,
        # so reconciliation stays ambiguous and tries to append the whole
        # batch; the store-level backstop refuses it.
        flood = [dict(msg) for msg in persisted_messages[3:11]]
        flood.append({"role": "user", "content": "new turn riding the flood"})

        after_restart._ingest_messages(flood)

        assert after_restart._store.get_session_count("flood-engine-session") == len(persisted_messages) + 1
        assert after_restart._replay_flood_refusal_count == 1
        status = after_restart.get_status()
        assert status["replay_flood_refusals"]["count"] == 1
        assert "refused replay-like message batch" in status["replay_flood_refusals"]["last"]

        continued = flood + [
            {"role": "assistant", "content": "response after recovered turn"},
            {"role": "user", "content": "subsequent turn"},
        ]
        after_restart._ingest_messages(continued)

        rows = after_restart._store.get_session_messages("flood-engine-session")
        assert len(rows) == len(persisted_messages) + 3
        assert [row["content"] for row in rows[-3:]] == [
            "new turn riding the flood",
            "response after recovered turn",
            "subsequent turn",
        ]
        assert after_restart._replay_flood_refusal_count == 1


class TestReplayFloodConfig:
    def test_env_overrides_parse(self, monkeypatch):
        monkeypatch.setenv("LCM_REPLAY_FLOOD_THRESHOLD_EXTERNAL", "5")
        monkeypatch.setenv("LCM_REPLAY_FLOOD_THRESHOLD_INTERNAL", "64")

        config = LCMConfig.from_env()

        assert config.replay_flood_threshold_external == 5
        assert config.replay_flood_threshold_internal == 64
