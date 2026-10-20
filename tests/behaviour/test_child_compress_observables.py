"""Pin the bypassed caller's output and public diagnostics before isolation."""

import logging

import pytest

from tests.harness import assistant, engine_context, pin_char_counter, user


@pytest.mark.parametrize("case,status,reason", [
    ("empty", "noop", "empty message list"),
    ("below", "noop", "LCM bypassed below threshold: auxiliary thread context"),
    ("forced", "host_fallback", "LCM bypassed: auxiliary thread context"),
])
def test_child_output_and_status(tmp_path, monkeypatch, caplog, case, status, reason):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000) as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        engine._mark_thread_context_stateless("child")
        view = [] if case == "empty" else [user("child request"), assistant("child reply")]
        try:
            with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
                result = engine.compress(view, force=case == "forced")
            assert result == view
            assert result is view
            assert engine.last_compression_status == status
            assert engine.last_compression_noop_reason == reason
            assert engine.last_compression_was_noop == (status == "noop")
            public = engine.get_status()
            assert public["last_compression_status"] == status
            assert public["last_compression_noop_reason"] == reason
            assert engine.get_automatic_compaction_status_message(phase="after", default_message="done") is None
            stops = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM compaction stop:")]
            assert len(stops) == 1
            assert stops[0].startswith(f"LCM compaction stop: reason={status} leaves=0 progress=none ")
            assert " host=" not in stops[0]
        finally:
            engine._clear_thread_context_stateless("child")
