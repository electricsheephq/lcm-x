"""Collected runtime facts and pure rendering for /lcm status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class StatusFacts:
    """A status read, detached from engine and database access for rendering."""

    status: dict[str, Any]
    db_path: Path
    db_exists: bool
    db_size: int
    session_id: str
    session_platform: str
    engine_name: str
    model: str
    provider: str
    context_length_source: str
    context_length: int
    context_threshold: float
    threshold_tokens: int
    compression_count: int
    session_ignored: bool
    session_stateless: bool
    side_channel_active: bool
    protection: dict[str, Any]
    stem_index: str
    embedding_lines: tuple[str, ...]


def collect_status(engine: Any) -> StatusFacts:
    """Read live facts through command's existing, patchable helper locations."""
    from . import command

    status = engine.get_status()
    db_path = Path(engine._store.db_path)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    protection = status.get("ingest_protection") or command.sensitive_pattern_status(engine._config)
    stem_index = command.stem_index_status(engine._store.connection)
    return StatusFacts(
        status=status, db_path=db_path, db_exists=db_exists, db_size=db_size,
        session_id=engine.current_session_id, session_platform=engine.current_session_platform,
        engine_name=engine.name, model=engine.model, provider=engine.provider,
        context_length_source=getattr(engine, "_context_length_source", ""),
        context_length=engine.context_length, context_threshold=engine._config.context_threshold,
        threshold_tokens=engine.threshold_tokens, compression_count=engine.compression_count,
        session_ignored=engine.current_session_ignored,
        session_stateless=engine.current_session_stateless, side_channel_active=engine.side_channel_active,
        protection=protection, stem_index=stem_index,
        embedding_lines=tuple(command._embedding_maintenance_status_lines(engine))
        if bool(getattr(engine._config, "embeddings_enabled", False)) else (),
    )


def render_status(facts: StatusFacts) -> str:
    """Format already-collected facts without engine, filesystem, or database reads."""
    from .command import _fmt_bool, _fmt_size

    status = facts.status
    db_path = facts.db_path
    db_exists = facts.db_exists
    db_size = facts.db_size
    session_bound = bool(facts.session_id)
    source_stats = status.get("source_lineage") or {}
    runtime_identity = status.get("runtime_identity") or {}
    source_stats = {
        "messages_total": int(source_stats.get("messages_total", 0) or 0),
        "attributed_messages": int(source_stats.get("attributed_messages", 0) or 0),
        "normalized_unknown_messages": int(source_stats.get("normalized_unknown_messages", 0) or 0),
        "legacy_blank_source_messages": int(source_stats.get("legacy_blank_source_messages", 0) or 0),
        "effective_unknown_messages": int(source_stats.get("effective_unknown_messages", 0) or 0),
        **({"error": source_stats.get("error")} if source_stats.get("error") else {}),
    }
    protection = facts.protection
    config_sources = status.get("config_sources") or {}
    config_source_warnings = status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = status.get("ignored_config_yaml_lcm_keys") or []

    uninitialized = "(uninitialized)"
    unknown = "(unknown)"
    model = (facts.model or unknown) if session_bound else uninitialized
    provider = (facts.provider or unknown) if session_bound else uninitialized
    context_length_source = (
        (facts.context_length_source or unknown)
        if session_bound
        else uninitialized
    )

    lines = [
        "LCM status",
        f"stem index: {facts.stem_index}",
        f"engine: {status.get('engine', facts.engine_name)}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"hermes_home: {runtime_identity.get('hermes_home', '') or '(unset)'}",
        f"session_id: {facts.session_id or '(unbound)'}",
        f"session_platform: {facts.session_platform or ('(unbound)' if not session_bound else '(unknown)')}",
        f"model: {model}",
        f"provider: {provider}",
        f"database_path: {db_path}",
        f"database_path_source: {runtime_identity.get('database_path_source', '(unknown)')}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {facts.compression_count}",
        f"total_compactions: {status.get('total_compactions', 0)}",
        f"total_compactions_scope: {status.get('total_compactions_scope', 'current_conversation')}",
        f"last_compression_status: {status.get('last_compression_status', 'idle')}",
        f"last_compression_noop_reason: {status.get('last_compression_noop_reason', '') or '(none)'}",
        f"context_length: {facts.context_length if session_bound else '(uninitialized)'}",
        f"raw_context_length: {status.get('raw_context_length', 0) if session_bound else '(uninitialized)'}",
        f"effective_context_length_cap: {status.get('effective_context_length_cap') or '(none)'}",
        f"effective_context_length_reason: {status.get('effective_context_length_reason') or '(none)'}",
        f"context_length_source: {context_length_source}",
        f"configured_context_threshold: {status.get('configured_context_threshold', facts.context_threshold)}",
        f"context_threshold: {status.get('context_threshold', facts.context_threshold)}",
        f"context_threshold_source: {status.get('context_threshold_source', config_sources.get('context_threshold', 'manual_or_default'))}",
        f"context_threshold_autoraised: {status.get('context_threshold_autoraised') or '(none)'}",
        f"threshold_tokens: {facts.threshold_tokens if session_bound else '(uninitialized)'}",
        f"cache_metrics_available: {_fmt_bool(status.get('cache_metrics_available'))}",
        f"last_input_tokens: {status.get('last_input_tokens', 0)}",
        f"last_output_tokens: {status.get('last_output_tokens', 0)}",
        f"last_cache_read_tokens: {status.get('last_cache_read_tokens', 0)}",
        f"last_cache_write_tokens: {status.get('last_cache_write_tokens', 0)}",
        f"last_reasoning_tokens: {status.get('last_reasoning_tokens', 0)}",
        f"cache_read_ratio: {float(status.get('cache_read_ratio', 0.0) or 0.0) * 100:.1f}%",
        f"sensitive_patterns_enabled: {_fmt_bool(protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {protection.get('source', 'default')}",
        # Filter classification for current_session_id (the foreground view).
        # When a side channel is in flight, get_status() reports the bound
        # session's flags; collect_status reads the engine properties instead
        # so this row stays consistent with the session_id row above.
        f"session_ignored: {_fmt_bool(facts.session_ignored)}",
        f"session_stateless: {_fmt_bool(facts.session_stateless)}",
        f"side_channel_active: {_fmt_bool(facts.side_channel_active)}",
        f"conversation_id: {runtime_identity.get('conversation_id', '') or '(unbound)'}",
        f"lifecycle_current_session_id: {runtime_identity.get('lifecycle_current_session_id', '') or '(none)'}",
        f"lifecycle_last_finalized_session_id: {runtime_identity.get('lifecycle_last_finalized_session_id', '') or '(none)'}",
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    last_rotate_at = status.get("last_rotate_at")
    if last_rotate_at:
        lines.append(
            f"last_rotate_at: "
            f"{datetime.fromtimestamp(float(last_rotate_at), tz=timezone.utc).isoformat(timespec='seconds')}"
        )
        rotate_backup_size = int(status.get("rotate_backup_size", 0) or 0)
        if rotate_backup_size:
            lines.append(f"rotate_backup_size: {_fmt_size(rotate_backup_size)}")
    else:
        lines.append("last_rotate_at: (never)")
    if status.get("rotate_backup_path"):
        lines.append(f"rotate_backup_path: {status['rotate_backup_path']}")

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
        )

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    if config_source_warnings:
        lines.append("config_source_warnings: " + "; ".join(config_source_warnings))
    if ignored_config_yaml_lcm_keys:
        lines.append(
            "ignored_config_yaml_lcm_keys: "
            + ", ".join(f"lcm.{key}" for key in ignored_config_yaml_lcm_keys)
        )
    if source_stats.get("error"):
        lines.append(f"source_lineage_error: {source_stats['error']}")
    lines.extend(facts.embedding_lines)
    return "\n".join(lines)


def _status_text(engine: Any) -> str:
    return render_status(collect_status(engine))


def _embedding_maintenance_status_lines(engine: Any) -> list[str]:
    """Unembedded backlog per corpus and the last incremental pass (#1014)."""
    from . import command

    db_path = engine._store.db_path
    backlog = {"summaries": "(no profile)", "chunks": "(no profile)"}

    def _count(conn: sqlite3.Connection, identity: str, pending: int | str) -> str:
        # In-flight rows are not pending; a provider failure after dispatch
        # leaves them uncertain until an operator `--retry-uncertain` run.
        try:
            uncertain = int(conn.execute(
                "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
                "WHERE identity_hash = ? AND state IN ('dispatched', 'uncertain')",
                (identity,),
            ).fetchone()[0])
        except sqlite3.OperationalError:
            uncertain = 0
        return f"{pending} (uncertain {uncertain})" if uncertain else str(pending)

    try:
        conn = command._embedding_read_connection(db_path)
        try:
            profile = command._embedding_current_profile(conn)
            if profile is not None:
                identity = str(profile["identity_hash"])
                backlog["summaries"] = _count(
                    conn, identity, command._embedding_pending_rows(conn, identity, 0)[0]
                )
            chunk_profile = command._chunk_current_profile(conn)
            if chunk_profile is not None:
                policy = command.normalize_content_policy(getattr(
                    engine._config, "embedding_content_policy", "conversational"
                ))
                identity = str(chunk_profile["identity_hash"])
                backlog["chunks"] = _count(conn, identity, command._chunk_backlog_estimate(conn, identity, policy))
        finally:
            conn.close()
    except sqlite3.Error as exc:
        backlog = {name: f"(unavailable: {exc})" for name in backlog}
    last, failures = command.incremental_embedding_state(db_path)
    last_text = "(never)" if last is None else (
        datetime.fromtimestamp(last[0], tz=timezone.utc).isoformat(timespec="seconds")
        + f" {last[1]}"
    )
    return [
        f"embedding_backlog_summaries: {backlog['summaries']}",
        f"embedding_backlog_chunks: {backlog['chunks']}",
        f"embedding_incremental_last_pass: {last_text}",
        f"embedding_incremental_failures: {failures}",
    ]
