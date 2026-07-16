from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _FakeTimeoutPattern:
    def __init__(self, pattern):
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def search(self, text, *, timeout=None):
        assert timeout is not None
        return self._compiled.search(text)


class _FakeTimeoutRegexEngine:
    """Stand-in for the optional ``regex`` package (absent in minimal CI).

    Wraps stdlib ``re`` but honors the ``timeout=`` kwarg so message-pattern
    filtering runs everywhere instead of being disabled when ``regex`` is not
    installed (which would silently skip these tests).
    """

    error = re.error

    @staticmethod
    def compile(pattern):
        return _FakeTimeoutPattern(pattern)


@pytest.fixture(autouse=True)
def _timeout_capable_regex_engine(monkeypatch):
    from hermes_lcm import message_patterns as message_patterns_mod

    monkeypatch.setattr(message_patterns_mod, "_regex_engine", _FakeTimeoutRegexEngine)


def _engine(tmp_path: Path, *, lossless: bool, session: str = "chat-1") -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        ignore_message_patterns=["^Cronjob Response:"],
        ignore_message_patterns_source="env",
        lossless_ignored_enabled=lossless,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(session, platform="telegram", conversation_id="c1", context_length=200000)
    return engine


def _protected_engine(tmp_path: Path, *, session: str = "chat-1") -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        ignore_message_patterns=["^Cronjob Response:"],
        ignore_message_patterns_source="env",
        lossless_ignored_enabled=True,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key"],
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(session, platform="telegram", conversation_id="c1", context_length=200000)
    return engine


_TURN = [
    {"role": "user", "content": "real question"},
    {"role": "assistant", "content": "real answer"},
    {"role": "user", "content": "Cronjob Response: heartbeat noise"},
    {"role": "user", "content": "another real question"},
]


class TestLosslessIgnoredCapture:
    def test_disabled_by_default_drops_and_stores_nothing(self, tmp_path):
        engine = _engine(tmp_path, lossless=False)
        engine._ingest_messages(list(_TURN))

        rows = engine._store.get_session_messages("chat-1")
        assert [r["content"] for r in rows] == [
            "real question",
            "real answer",
            "another real question",
        ]
        assert engine._store.get_ignored_session_count("chat-1") == 0

    def test_enabled_captures_dropped_row_in_sidecar(self, tmp_path):
        engine = _engine(tmp_path, lossless=True)
        engine._ingest_messages(list(_TURN))

        # Active/replay surface is unchanged — the ignored row is still absent.
        rows = engine._store.get_session_messages("chat-1")
        assert [r["content"] for r in rows] == [
            "real question",
            "real answer",
            "another real question",
        ]
        # ...but the dropped row is now durably preserved in the sidecar.
        ignored = engine._store.get_ignored_session_messages("chat-1")
        assert [r["content"] for r in ignored] == ["Cronjob Response: heartbeat noise"]
        assert engine._store.get_ignored_session_count("chat-1") == 1

    def test_active_message_sequence_is_byte_identical_with_and_without_capture(self, tmp_path):
        """Golden: enabling lossless capture must not change stored active rows."""
        off = _engine(tmp_path / "off", lossless=False)
        on = _engine(tmp_path / "on", lossless=True)
        off._ingest_messages(list(_TURN))
        on._ingest_messages(list(_TURN))

        def snapshot(engine):
            return [
                (r["role"], r["content"], r.get("tool_call_id"))
                for r in engine._store.get_session_messages("chat-1")
            ]

        assert snapshot(off) == snapshot(on)
        assert off._store.get_session_count("chat-1") == on._store.get_session_count("chat-1")

    def test_status_surfaces_sidecar_count_when_enabled(self, tmp_path):
        engine = _engine(tmp_path, lossless=True)
        engine._ingest_messages(list(_TURN))
        status = engine.get_status()
        assert status["lossless_ignored_enabled"] is True
        assert status["ignored_sidecar_count"] == 1

    def test_status_omits_sidecar_count_when_disabled(self, tmp_path):
        engine = _engine(tmp_path, lossless=False)
        engine._ingest_messages(list(_TURN))
        status = engine.get_status()
        assert status["lossless_ignored_enabled"] is False
        assert "ignored_sidecar_count" not in status

    def test_dropped_count_still_increments_with_capture(self, tmp_path):
        engine = _engine(tmp_path, lossless=True)
        engine._ingest_messages(list(_TURN))
        status = engine.get_status()
        # Lossless capture augments, it does not replace, the drop counter.
        assert status["ignore_pattern_dropped_count"] == 1

    def test_no_ignore_patterns_means_no_sidecar_rows(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            lossless_ignored_enabled=True,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("chat-1", platform="telegram", conversation_id="c1", context_length=200000)
        engine._ingest_messages(list(_TURN))
        assert engine._store.get_ignored_session_count("chat-1") == 0
        assert engine._store.get_session_count("chat-1") == len(_TURN)

    def test_ignored_only_replay_after_restart_does_not_duplicate_sidecar_rows(self, tmp_path):
        ignored_turn = [{"role": "user", "content": "Cronjob Response: heartbeat noise"}]
        first = _engine(tmp_path, lossless=True)
        first._ingest_messages(list(ignored_turn))
        assert first._store.get_session_count("chat-1") == 0
        assert first._store.get_ignored_session_count("chat-1") == 1

        restarted = _engine(tmp_path, lossless=True)
        assert restarted._ingest_cursor_needs_reconcile is True
        restarted._ingest_messages(list(ignored_turn))

        assert restarted._ingest_cursor == 1
        assert restarted._store.get_session_count("chat-1") == 0
        assert restarted._store.get_ignored_session_count("chat-1") == 1

    def test_ignored_restart_compares_protected_stored_identity(self, tmp_path):
        synthetic_value = "synthetic-" + "canary-123456"
        ignored_turn = [
            {
                "role": "user",
                "content": f"Cronjob Response: api_key={synthetic_value}",
            }
        ]
        first = _protected_engine(tmp_path)
        first._ingest_messages(list(ignored_turn))
        stored = first._store.get_ignored_session_messages("chat-1")
        assert len(stored) == 1
        assert synthetic_value not in stored[0]["content"]

        restarted = _protected_engine(tmp_path)
        restarted._ingest_messages(list(ignored_turn))

        assert restarted._ingest_cursor == 1
        assert restarted._store.get_ignored_session_count("chat-1") == 1

    def test_mixed_restart_advances_only_replayed_ignored_prefix(self, tmp_path):
        replayed = [
            {"role": "user", "content": "real question"},
            {"role": "user", "content": "Cronjob Response: old heartbeat"},
            {"role": "assistant", "content": "real answer"},
        ]
        first = _engine(tmp_path, lossless=True)
        first._ingest_messages(list(replayed))

        fresh_ignored = {"role": "user", "content": "Cronjob Response: fresh heartbeat"}
        fresh_active = {"role": "user", "content": "new question"}
        restarted = _engine(tmp_path, lossless=True)
        restarted._ingest_messages([*replayed, fresh_ignored, fresh_active])

        assert restarted._ingest_cursor == 5
        assert [
            row["content"] for row in restarted._store.get_ignored_session_messages("chat-1")
        ] == [
            "Cronjob Response: old heartbeat",
            "Cronjob Response: fresh heartbeat",
        ]
        assert [row["content"] for row in restarted._store.get_session_messages("chat-1")] == [
            "real question",
            "real answer",
            "new question",
        ]

    def test_pattern_timeout_preserves_message_without_sidecar_capture(self, tmp_path):
        class TimedOutPattern:
            pattern = "^Cronjob Response:"

            def search(self, text, *, timeout=None):
                assert timeout is not None
                raise TimeoutError("synthetic timeout")

        engine = _engine(tmp_path, lossless=True)
        engine._compiled_ignore_message_patterns = [TimedOutPattern()]
        message = {"role": "user", "content": "Cronjob Response: preserve me"}
        engine._ingest_messages([message])

        assert [row["content"] for row in engine._store.get_session_messages("chat-1")] == [
            "Cronjob Response: preserve me"
        ]
        assert engine._store.get_ignored_session_count("chat-1") == 0
