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
