"""The explicit ratio-zero settings preserve the default path."""

import json
from dataclasses import fields

from hermes_lcm.compaction import _CompressionState
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from hermes_lcm.tools import lcm_status
from tests.behaviour.test_trace_config import AssemblyCanonical, CompressCanonical, IngestCanonical, ReconcileCanonical


def test_explicit_ratio_zero_matches_unset(tmp_path, monkeypatch):
    names = ("depth_target_ratio", "summary_prefix_target_ratio")
    for name in names:
        monkeypatch.delenv(f"LCM_{name.upper()}", raising=False)
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    unset = LCMConfig.from_env()
    for name in names:
        monkeypatch.setenv(f"LCM_{name.upper()}", "0")
    zero = LCMConfig.from_env()
    def check_sources(before, after):
        assert {k: v for k, v in before.items() if k in names} == dict.fromkeys(names, "default")
        assert after == {**before, **{name: f"env:LCM_{name.upper()}" for name in names}}

    for field in fields(LCMConfig):
        if field.name != "config_sources":
            assert getattr(unset, field.name) == getattr(zero, field.name), field.name
    check_sources(unset.config_sources, zero.config_sources)
    for canonical in (AssemblyCanonical(tmp_path), CompressCanonical(tmp_path),
                      IngestCanonical(tmp_path, []), ReconcileCanonical(tmp_path, [])):
        before, after = (canonical.value(config) for config in (unset, zero))
        before.pop("config_sources")
        after.pop("config_sources")
        assert before == after, type(canonical).__name__

    def observe(config):
        engine = LCMEngine(config=config, hermes_home=str(tmp_path))
        try:
            engine.on_session_start("S", conversation_id="conv", context_length=100_000)
            engine.ingest(view)
            engine.should_compress(count_messages_tokens(view))
            state = _CompressionState(threshold_full_sweep_active=True, observed_prompt_tokens=engine.threshold_tokens)
            target = engine._resolve_depth_target(state, force=False, recovery_attempt=False)
            return engine._no_progress_fingerprint(), target, json.loads(lcm_status({}, engine=engine))
        finally:
            engine.shutdown()

    view = [{"role": "user", "content": "Earlier question", "timestamp": 1_000.0}]
    observe(unset)  # Seed the store so both arms replay the same durable ingest ledger.
    before_hold, before_target, before_status = observe(unset)
    after_hold, after_target, after_status = observe(zero)
    assert before_hold and before_hold == after_hold
    assert before_target == after_target == 0
    assert before_status["session_id"] == after_status["session_id"] == "S"
    check_sources(before_status.pop("config_sources"), after_status.pop("config_sources"))
    assert before_status == after_status
