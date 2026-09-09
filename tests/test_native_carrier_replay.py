"""Embedded archive references must not replace a native carrier's prose."""
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import is_externalized_placeholder, maybe_externalize_payload


def test_native_carrier_embedded_reference_preserves_cold_replay(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1000)
    home = str(tmp_path / "home")
    engine = LCMEngine(config=config, hermes_home=home)
    engine.on_session_start("carrier", conversation_id="owned", platform="cli", context_length=200000)
    archived = "different archived payload " * 1300
    inner = maybe_externalize_payload(archived, kind="raw_payload", role="user",
        session_id="carrier", config=config, hermes_home=home)
    assert inner is not None and is_externalized_placeholder(inner["placeholder"])
    whole = {"role": "user", "content": inner["placeholder"]}
    assert engine._message_replay_identity(whole)[1] == archived
    prose = "Native retained summary. " * 500 + "\nArchive reference: " + inner["placeholder"] + "\nCurrent task remains active."
    assert not is_externalized_placeholder(prose)
    messages = [{"role": "user", "content": prose}, {"role": "assistant", "content": "retained answer"}]
    try:
        engine.ingest(messages)
        stored = engine._store.get_session_messages("carrier")
        assert len(stored) == 2 and is_externalized_placeholder(stored[0]["content"])
        stored_identity = engine._message_replay_identity(stored[0], stored_row=True)
        assert stored_identity[1] == prose
        identity_matches = engine._message_replay_identity(messages[0]) == stored_identity
    finally:
        engine.shutdown()
    cold = LCMEngine(config=config, hermes_home=home)
    try:
        cold.on_session_start("carrier", conversation_id="owned", platform="cli", context_length=200000)
        cold.ingest(messages)
        assert (identity_matches, cold._store.get_session_count("carrier")) == (True, 2)
    finally:
        cold.shutdown()
