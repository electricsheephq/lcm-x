"""Byte pins for the status extraction and its command-module patch seams."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_lcm import command


def status_engine(tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Supply deterministic facts without a provider, lifecycle, or worker."""
    status: dict[str, Any] = {}
    bound = case != "cold"
    if bound:
        status.update(session_ignored=True, session_stateless=True)
    if case == "error":
        status.update(
            source_lineage={"messages_total": "7", "attributed_messages": 2,
                            "normalized_unknown_messages": 3, "legacy_blank_source_messages": 2,
                            "effective_unknown_messages": 5, "error": "lineage unavailable"},
            config_source_warnings=["warning one", "warning two"],
            ignored_config_yaml_lcm_keys=["context_threshold", "embedding_model"],
            ignore_session_patterns_source="environment",
            stateless_session_patterns_source="default",
            config_sources={"context_threshold": "environment"},
        )
    if case.startswith("rotation"):
        status.update(
            last_rotate_at=1_700_000_000, rotate_backup_size=2048 if case == "rotation" else 0,
            rotate_backup_path="backup.db", raw_context_length=200_000,
            runtime_identity={"plugin_name": "lcm-x", "plugin_version": "test",
                              "plugin_git_dirty": False, "conversation_id": "foreground-conversation"},
            ingest_protection={"enabled": True, "patterns": ["alpha", "beta"], "source": "manual"},
            cache_metrics_available=True, cache_read_ratio=0.125,
        )
    db_path = tmp_path / "status.db"
    if case != "cold":
        db_path.write_bytes(b"x" * 2048)
    engine = SimpleNamespace(
        get_status=lambda: status,
        _store=SimpleNamespace(db_path=str(db_path), connection=object()),
        _config=SimpleNamespace(context_threshold=0.75, embeddings_enabled=False),
        name="lcm", current_session_id="foreground" if bound else "",
        current_session_platform="cli" if bound else "", model="model" if bound else "",
        provider="provider" if bound else "", _context_length_source="host" if bound else "",
        compression_count=2, context_length=200_000, threshold_tokens=150_000,
        current_session_ignored=False, current_session_stateless=False, side_channel_active=bound,
    )
    monkeypatch.setattr(command, "stem_index_status", lambda conn: "ready")
    monkeypatch.setattr(command, "sensitive_pattern_status", lambda config: {
        "enabled": False, "patterns": [], "source": "default",
    })
    return engine


@pytest.mark.parametrize("case", ["cold", "bound", "error", "rotation", "rotation_zero"])  # type: ignore[untyped-decorator]
def test_status_text_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    engine = status_engine(tmp_path, case, monkeypatch)
    expected = json.loads((Path(__file__).parent / "fixtures" / "command_status.json").read_text())
    assert command._status_text(engine).replace(str(tmp_path), "<tmp>") == expected[case]


@pytest.mark.parametrize("case", ["no_profile", "pending", "uncertain", "unavailable"])  # type: ignore[untyped-decorator]
def test_embedding_status_bytes(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    conn = sqlite3.connect(":memory:")
    if case == "uncertain":
        conn.execute("CREATE TABLE lcm_embedding_backfill_inflight (identity_hash TEXT, state TEXT)")
        conn.executemany("INSERT INTO lcm_embedding_backfill_inflight VALUES (?, ?)", [
            ("summary", "dispatched"), ("summary", "uncertain"), ("chunk", "uncertain"),
        ])
    calls: list[str] = []

    def read_connection(path: str) -> sqlite3.Connection:
        calls.append(path)
        if case == "unavailable":
            raise sqlite3.OperationalError("offline")
        return conn

    monkeypatch.setattr(command, "_embedding_read_connection", read_connection)
    monkeypatch.setattr(command, "_embedding_current_profile", lambda db: (
        None if case == "no_profile" else {"identity_hash": "summary"}
    ))
    monkeypatch.setattr(command, "_chunk_current_profile", lambda db: (
        None if case == "no_profile" else {"identity_hash": "chunk"}
    ))
    monkeypatch.setattr(command, "_embedding_pending_rows", lambda *args: (7, []))
    monkeypatch.setattr(command, "_chunk_backlog_estimate", lambda *args: "50+")
    monkeypatch.setattr(command, "incremental_embedding_state", lambda path: (
        ((1_700_000_000, "complete"), 3) if case == "uncertain" else (None, 0)
    ))
    engine = SimpleNamespace(_store=SimpleNamespace(db_path="status.db"), _config=SimpleNamespace())
    expected = {
        "no_profile": ["(no profile)", "(no profile)", "(never)", "0"],
        "pending": ["7", "50+", "(never)", "0"],
        "uncertain": ["7 (uncertain 2)", "50+ (uncertain 1)", "2023-11-14T22:13:20+00:00 complete", "3"],
        "unavailable": ["(unavailable: offline)", "(unavailable: offline)", "(never)", "0"],
    }[case]
    try:
        assert command._embedding_maintenance_status_lines(engine) == [
            f"{key}: {value}" for key, value in zip((
                "embedding_backlog_summaries", "embedding_backlog_chunks",
                "embedding_incremental_last_pass", "embedding_incremental_failures",
            ), expected, strict=True)
        ]
        assert calls == ["status.db"]
        if case != "unavailable":
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                conn.execute("SELECT 1")
    finally:
        conn.close()


def test_status_uses_command_embedding_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = status_engine(tmp_path, "bound", monkeypatch)
    engine._config.embeddings_enabled = True
    seen: list[Any] = []

    def embedding_lines(value: Any) -> list[str]:
        seen.append(value)
        return ["patched embedding lines"]

    monkeypatch.setattr(command, "_embedding_maintenance_status_lines", embedding_lines)
    assert command._status_text(engine).endswith("\npatched embedding lines")
    assert seen == [engine]


def test_status_router_uses_command_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = status_engine(tmp_path, "bound", monkeypatch)
    monkeypatch.setattr(command, "_status_text", lambda value: "patched status")
    assert command.handle_lcm_command("status", engine) == "patched status"


def test_status_reexports_point_to_extracted_module() -> None:
    from hermes_lcm import command_status

    assert command._status_text is command_status._status_text
    assert command._embedding_maintenance_status_lines is command_status._embedding_maintenance_status_lines


def test_render_status_needs_no_live_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_lcm.command_status import collect_status, render_status

    engine = status_engine(tmp_path, "rotation", monkeypatch)
    engine._config.embeddings_enabled = True
    monkeypatch.setattr(command, "_embedding_maintenance_status_lines", lambda value: ["embedding snapshot"])
    facts = collect_status(engine)
    expected = render_status(facts)
    engine.__dict__.clear()

    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("rendering must only read collected facts")

    with monkeypatch.context() as patch:
        for name in ("stem_index_status", "sensitive_pattern_status", "_embedding_maintenance_status_lines"):
            patch.setattr(command, name, unexpected_read)
        patch.setattr(Path, "exists", unexpected_read)
        patch.setattr(Path, "stat", unexpected_read)
        assert render_status(facts) == expected
        assert render_status(facts) == expected
    assert expected.endswith("\nembedding snapshot")
