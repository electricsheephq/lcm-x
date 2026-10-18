"""Collected facts and pure text rendering for /lcm doctor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DoctorFacts:
    """One ordered diagnostic read, detached from the engine and its connections."""

    runtime_identity: dict[str, Any]
    db_path: Path
    db_exists: bool
    db_size: int
    wal_size: int
    schema_health: dict[str, Any]
    schema_missing_tables: list[str]
    schema_existing_tables: list[str]
    schema_core_status: str
    integrity: str
    store_fts: str
    node_fts: str
    store_fts_count: int | str
    node_fts_count: int | str
    store_fts_failed_flag: dict[str, Any] | None
    node_fts_failed_flag: dict[str, Any] | None
    sqlite_mmap_bytes: int | str
    total_messages: int | str
    total_message_sessions: int | str
    total_nodes: int | str
    total_node_sessions: int | str
    journal_mode: str
    quick_check: str
    payload_storage_error: str
    payload_risks: dict[str, Any]
    externalized_stats: dict[str, Any]
    externalized_integrity: dict[str, Any]
    clean_scan: dict[str, Any]
    debt_rows: list[Any]
    source_stats: dict[str, Any]
    lifecycle_stats: dict[str, Any]
    lifecycle_fragmented: bool
    protection: dict[str, Any]
    survival_fit: dict[str, Any]
    fit_count: int | None
    projection_floor: bool
    stub_first_exit: dict[str, Any] | None
    inactive_process: str | None
    inactive_action: str
    embedding_check: dict[str, Any] | None
    embedding_action: str
    triage_guidance: list[dict[str, Any]]
    wal_reset_affected: bool
    sqlite_version: str
    stem_index: str
    host_uid_lines: tuple[str, ...]
    # Exception text remains in the corresponding result; flags preserve branch identity.
    probe_errors: dict[str, bool]


def collect_doctor(engine: Any) -> DoctorFacts:
    """Read through the original patchable helpers; keep probe and guidance order."""
    from . import command

    db_path = command.Path(engine._store.db_path)
    runtime_identity = engine.get_runtime_identity()
    store_conn = engine._store.connection
    dag_conn = engine._dag.connection

    probe_errors: dict[str, bool] = {}
    schema_health = command.inspect_lcm_schema_health(store_conn, database_path=str(db_path))
    schema_missing_raw = schema_health.get("missing_tables")
    schema_missing_tables = [str(name) for name in schema_missing_raw] if isinstance(schema_missing_raw, list) else []
    schema_existing_raw = schema_health.get("existing_tables")
    schema_existing_tables = [str(name) for name in schema_existing_raw] if isinstance(schema_existing_raw, list) else []
    schema_core_status = "error" if schema_health.get("error") else "missing" if schema_missing_tables else "ok"

    def _safe_count(conn, query: str, issue_key: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            probe_errors[issue_key] = True
            return f"error: {exc}"

    try:
        integrity_row = store_conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        integrity = f"error: {exc}"
        probe_errors["sqlite_integrity"] = True

    def _fts_text_status(result: dict[str, command.Any]) -> str:
        status = str(result.get("status") or "fail")
        return "ok" if status == "pass" else status

    try:
        store_fts_count = int(store_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        store_fts_integrity = command.check_external_content_fts_integrity(store_conn, command.build_message_fts_spec())
        store_fts = _fts_text_status(store_fts_integrity)
    except Exception as exc:  # pragma: no cover - defensive
        store_fts_count = f"error: {exc}"
        store_fts = f"error: {exc}"
        store_fts_integrity = {"status": "fail", "detail": str(exc)}
        probe_errors["messages_fts"] = True

    try:
        node_fts_count = int(dag_conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])
        node_fts_integrity = command.check_external_content_fts_integrity(dag_conn, command.build_nodes_fts_spec())
        node_fts = _fts_text_status(node_fts_integrity)
    except Exception as exc:  # pragma: no cover - defensive
        node_fts_count = f"error: {exc}"
        node_fts = f"error: {exc}"
        node_fts_integrity = {"status": "fail", "detail": str(exc)}
        probe_errors["nodes_fts"] = True

    # A prior non-blocking background integrity scan (issue #6) records a
    # persisted ``fts_integrity_failed:<table>`` flag when it finds corruption
    # without rebuilding. Surface it even when this doctor run's live deep check
    # could not confirm it (e.g. read-only access), pointing at the explicit
    # repair path.
    try:
        store_fts_failed_flag = command.load_integrity_failed(store_conn, command.build_message_fts_spec())
    except Exception:  # pragma: no cover - defensive
        probe_errors["messages_fts_flag"] = True
        store_fts_failed_flag = None
    try:
        node_fts_failed_flag = command.load_integrity_failed(dag_conn, command.build_nodes_fts_spec())
    except Exception:  # pragma: no cover - defensive
        probe_errors["nodes_fts_flag"] = True
        node_fts_failed_flag = None
    sqlite_mmap_bytes = _safe_count(store_conn, "PRAGMA mmap_size", "sqlite_mmap_size")
    total_messages = _safe_count(store_conn, "SELECT COUNT(*) FROM messages", "messages_total")
    total_message_sessions = _safe_count(
        store_conn,
        "SELECT COUNT(DISTINCT session_id) FROM messages",
        "message_sessions_total",
    )
    total_nodes = _safe_count(dag_conn, "SELECT COUNT(*) FROM summary_nodes", "summary_nodes_total")
    total_node_sessions = _safe_count(
        dag_conn,
        "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        "summary_node_sessions_total",
    )

    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    wal_path = command.Path(str(db_path) + "-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        journal_row = store_conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_row[0]) if journal_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        journal_mode = f"error: {exc}"
        probe_errors["sqlite_journal_mode"] = True
    try:
        quick_row = store_conn.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_row[0]) if quick_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        quick_check = f"error: {exc}"
        probe_errors["sqlite_quick_check"] = True
    payload_storage_error = ""
    try:
        payload_risks = command.scan_sqlite_payload_risks(store_conn)
        externalized_stats = command.externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = command.scan_externalized_payload_integrity(
            store_conn,
            engine._config,
            hermes_home=engine._hermes_home,
        )
    except Exception as exc:  # pragma: no cover - defensive
        payload_storage_error = str(exc)
        payload_risks = {
            "largest_content_rows": [],
            "largest_tool_calls_rows": [],
            "suspicious_data_uri_content_rows": [],
            "suspicious_data_uri_tool_calls_rows": [],
            "suspicious_base64_like_rows": [],
            "quarantined_assistant_rows": [],
            "suspicious_repetitive_assistant_rows": [],
            "heartbeat_noise_rows": [],
        }
        externalized_stats = {
            "externalized_payload_count": 0,
            "externalized_payload_bytes": 0,
            "externalized_payload_chars": 0,
            "externalized_payload_dir": "",
            "latest_externalized_payload_path": "",
            "latest_externalized_payload_mtime": 0,
        }
        externalized_integrity = {
            "externalized_payload_refs_total": 0,
            "externalized_payload_refs_existing": 0,
            "externalized_payload_refs_missing": 0,
            "externalized_payload_files_unreferenced": 0,
            "missing_externalized_payload_refs": [],
            "unreferenced_externalized_payload_files": [],
        }
        probe_errors["payload_storage"] = True
    clean_scan = command._scan_clean_candidates(engine)

    debt_rows = []
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "connection", None)
    if lifecycle_conn is not None:
        try:
            debt_rows = lifecycle_conn.execute(
                """
                SELECT conversation_id, debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE debt_kind IS NOT NULL AND debt_size_estimate > 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            probe_errors["lifecycle_state"] = True
            debt_rows = [(f"error: {exc}", "error", 0)]

    try:
        source_stats = engine._store.get_source_stats()
    except Exception as exc:  # pragma: no cover - defensive
        probe_errors["source_lineage"] = True
        source_stats = {
            "messages_total": 0,
            "attributed_messages": 0,
            "normalized_unknown_messages": 0,
            "legacy_blank_source_messages": 0,
            "effective_unknown_messages": 0,
            "error": str(exc),
        }
    try:
        lifecycle_stats = engine._lifecycle.get_fragmentation_stats(
            state_db_path=command._state_db_path_for_engine(engine)
        )
    except Exception as exc:  # pragma: no cover - defensive
        probe_errors["lifecycle_fragmentation"] = True
        lifecycle_stats = {"error": str(exc)}
    lifecycle_fragmented = (
        command._has_lifecycle_fragmentation(lifecycle_stats)
        if "lifecycle_fragmentation" not in probe_errors else False
    )

    protection = command.sensitive_pattern_status(engine._config)

    missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
    triage_checks: list[dict[str, command.Any]] = []
    if integrity != "ok":
        triage_checks.append({"check": "database_integrity", "status": "fail", "detail": integrity})
    if schema_health.get("error") or schema_missing_tables:
        triage_checks.append({"check": "schema_core_tables", "status": "fail", "detail": schema_health})
    if store_fts != "ok":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "warn" if store_fts == "unchecked" else "fail",
            "detail": store_fts_integrity,
        })
    if node_fts != "ok":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "warn" if node_fts == "unchecked" else "fail",
            "detail": node_fts_integrity,
        })
    if store_fts_failed_flag and store_fts != "fail":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": store_fts_failed_flag},
        })
    if node_fts_failed_flag and node_fts != "fail":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": node_fts_failed_flag},
        })
    if clean_scan["candidates"]:
        triage_checks.append({"check": "cleanup_candidates", "status": "warn", "detail": clean_scan})
    if payload_storage_error or missing_externalized_refs or any(payload_risks.get(key) for key in (
        "suspicious_data_uri_content_rows",
        "suspicious_data_uri_tool_calls_rows",
        "suspicious_base64_like_rows",
        "suspicious_repetitive_assistant_rows",
        "heartbeat_noise_rows",
    )):
        detail = {**payload_risks, **externalized_integrity}
        if payload_storage_error:
            detail["error"] = payload_storage_error
        triage_checks.append({
            "check": "payload_storage",
            "status": "fail" if payload_storage_error else "warn",
            "detail": detail,
        })
    if (protection["enabled"] and not protection["active_patterns"]) or protection["unknown_patterns"]:
        triage_checks.append({"check": "sensitive_pattern_handling", "status": "warn", "detail": protection})
    if source_stats.get("error"):
        triage_checks.append({"check": "source_lineage_hygiene", "status": "fail", "detail": source_stats})
    if lifecycle_stats.get("error") or command._has_lifecycle_fragmentation(lifecycle_stats):
        lifecycle_status = "fail" if lifecycle_stats.get("error") else "warn"
        triage_checks.append({"check": "lifecycle_fragmentation", "status": lifecycle_status, "detail": lifecycle_stats})
    try:  # #582: the survival fit kept a session alive by dropping stored turns from live context
        survival_fit = engine._store.read_metadata_json(command.SURVIVAL_FIT_COUNTER_KEY)
    except Exception:
        probe_errors["survival_fit"] = True
        survival_fit = None
    survival_fit = survival_fit if isinstance(survival_fit, dict) else {}
    fit_count = command.survival_fit_count(survival_fit)
    projection_floor = command.survival_fit_projection_floor_applies(survival_fit) if fit_count != 0 else False
    if fit_count != 0:
        triage_checks.append({"check": "survival_fit", "status": "warn", "detail": survival_fit})
    stub_first_exit = getattr(engine, "_last_stub_first_exit", None)
    inactive_process = command.inactive_record_notice(getattr(engine, "_hermes_home", ""))
    inactive_action = ""
    if inactive_process:
        inactive_check = {"check": "inactive_process", "status": "warn", "detail": inactive_process}
        triage_checks.append(inactive_check)
        inactive_action = command.doctor_guidance_for_check(inactive_check)["operator_action"]
    try:
        from .tools import _embedding_provider_health_check
        embedding_check = _embedding_provider_health_check(engine)
    except Exception as exc:  # as the lcm_doctor tool does: a check that raises reports fail, never a silent ok
        probe_errors["embedding_provider_health"] = True
        embedding_check = {"check": "embedding_provider_health", "status": "fail", "detail": str(exc)}
    embedding_action = ""
    if embedding_check and embedding_check["status"] in {"warn", "fail"}:
        triage_checks.append(embedding_check)
        embedding_action = command.doctor_guidance_for_check(embedding_check)["operator_action"]
    triage_guidance = command.doctor_guidance_for_checks(triage_checks)

    # Keep these P4 reads after guidance, in their original order.
    wal_reset_affected = command.sqlite_wal_reset_affected()
    sqlite_version = command.sqlite3.sqlite_version
    stem_index = command.stem_index_status(store_conn)
    host_uid_lines = tuple(command.host_uid_doctor_lines(engine))
    return DoctorFacts(
        runtime_identity=runtime_identity,
        db_path=db_path,
        db_exists=db_exists,
        db_size=db_size,
        wal_size=wal_size,
        schema_health=schema_health,
        schema_missing_tables=schema_missing_tables,
        schema_existing_tables=schema_existing_tables,
        schema_core_status=schema_core_status,
        integrity=integrity,
        store_fts=store_fts,
        node_fts=node_fts,
        store_fts_count=store_fts_count,
        node_fts_count=node_fts_count,
        store_fts_failed_flag=store_fts_failed_flag,
        node_fts_failed_flag=node_fts_failed_flag,
        sqlite_mmap_bytes=sqlite_mmap_bytes,
        total_messages=total_messages,
        total_message_sessions=total_message_sessions,
        total_nodes=total_nodes,
        total_node_sessions=total_node_sessions,
        journal_mode=journal_mode,
        quick_check=quick_check,
        payload_storage_error=payload_storage_error,
        payload_risks=payload_risks,
        externalized_stats=externalized_stats,
        externalized_integrity=externalized_integrity,
        clean_scan=clean_scan,
        debt_rows=debt_rows,
        source_stats=source_stats,
        lifecycle_stats=lifecycle_stats,
        lifecycle_fragmented=lifecycle_fragmented,
        protection=protection,
        survival_fit=survival_fit,
        fit_count=fit_count,
        projection_floor=projection_floor,
        stub_first_exit=stub_first_exit,
        inactive_process=inactive_process,
        inactive_action=inactive_action,
        embedding_check=embedding_check,
        embedding_action=embedding_action,
        triage_guidance=triage_guidance,
        wal_reset_affected=wal_reset_affected,
        sqlite_version=sqlite_version,
        stem_index=stem_index,
        host_uid_lines=host_uid_lines,
        probe_errors=probe_errors,
    )


def survival_fit_observation(
    survival_fit: dict[str, Any], fit_count: int | None, projection_floor: bool,
) -> str | None:
    """Describe one collected survival-fit record without reading live state."""
    if fit_count == 0:
        return None
    projected = survival_fit.get("projected_count")  # absent on a record from before the key: unknown
    if type(projected) is not int or projected < 0 or survival_fit.get("count_lost"):
        projected = None  # damaged metadata, or history lost with the count: conservatively unknown
    uncovered = survival_fit.get("last_uncovered_rows")
    uncovered_fits = survival_fit.get("uncovered_fit_count")
    unknown_fits = survival_fit.get("unknown_coverage_fit_count")
    within = ("rollback to a 0.24.x version older than v0.24.5: that plugin cannot compact stored rows that a "
              "survival fit removed from the live context; stop Hermes, move the configured database file (by "
              "default lcm.db, with its -wal and -shm companions) aside and keep it, then restore the database "
              "backup taken before the first v0.24.5 install with the plugin; a rollback of the plugin alone to "
              f"{'v0.27.0' if projection_floor else 'v0.24.5'} or later is fine")
    applied = ("applied an unknown number of times (the stored count is unreadable)" if fit_count is None
               else f"applied {fit_count} time(s)")
    if survival_fit.get("last_shortened") is False:
        applied = applied.replace("applied", "attempted", 1) + "; could not shorten the list on the last attempt"
    rollback = (f"; {within}; a rollback to v0.23.3 keeps the database file and needs native recovery ON "
                "(see triage_guidance)" if survival_fit.get("ever_shortened", True) else "")
    if projection_floor:
        rollback += ("; a survival fit projected rows (or the projection count is unknown): from v0.27.0 on, "
                     "a plugin-only rollback must target v0.27.0 or later; v0.26.x or earlier cannot recognise "
                     "a short projected row (head + mark) and stores it again on a cold resume (#601 duplicates); "
                     "to roll back further, stop Hermes and move the configured database file (by default lcm.db, "
                     "with its -wal and -shm companions) aside and keep it")
    last_cause = f"; last cause {survival_fit['last_cause']}" if survival_fit.get("last_cause") else ""
    return (f"survival_fit: {applied}; last reason "
                        f"{survival_fit.get('last_reason') or '(unknown)'}{last_cause}; projected_count "
                        f"{'unknown' if projected is None else projected}; unreached_budget_count "
                        f"{survival_fit.get('unreached_budget_count', 0)}; last_reached_budget "
                        f"{survival_fit.get('last_reached_budget', 'unknown')}; last_uncovered_rows "
                        f"{'unknown' if uncovered is None else uncovered}; uncovered_fit_count "
                        f"{'unknown' if uncovered_fits is None else uncovered_fits}; unknown_coverage_fit_count "
                        f"{'unknown' if unknown_fits is None else unknown_fits}{rollback}")


def render_doctor(facts: DoctorFacts) -> str:
    """Replay probe outcomes and format facts without live reads."""
    from .command import _fmt_bool, _fmt_size

    issues: list[str] = []
    recommended_actions: list[str] = []
    if facts.schema_missing_tables or facts.schema_health.get("error"):
        issues.append("schema_core_tables")
    if "sqlite_integrity" in facts.probe_errors:
        issues.append("sqlite_integrity")
    for status, key, table in ((facts.store_fts, "messages_fts", "messages"), (facts.node_fts, "nodes_fts", "nodes")):
        if status == "fail" or key in facts.probe_errors:
            issues.append(key)
        elif status == "unchecked":
            recommended_actions.append(f"rerun `/lcm doctor` with read-write SQLite access if a deep {table} FTS check is needed")
    if facts.store_fts_failed_flag and "messages_fts" not in issues:
        issues.append("messages_fts")
    if facts.node_fts_failed_flag and "nodes_fts" not in issues:
        issues.append("nodes_fts")
    for key in (
        "sqlite_mmap_size", "messages_total", "message_sessions_total", "summary_nodes_total",
        "summary_node_sessions_total", "sqlite_journal_mode", "sqlite_quick_check", "payload_storage", "lifecycle_state",
    ):
        if key in facts.probe_errors:
            issues.append(key)

    observations: list[str] = []
    missing_externalized_refs = int(facts.externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
    suspicious_payload_rows = sum(
        len(facts.payload_risks.get(key) or [])
        for key in (
            "suspicious_data_uri_content_rows",
            "suspicious_data_uri_tool_calls_rows",
            "suspicious_base64_like_rows",
            "suspicious_repetitive_assistant_rows",
        )
    )

    if facts.schema_health.get("error"):
        observations.append(f"schema_core_tables: error: {facts.schema_health['error']}")
        recommended_actions.append(
            "verify SQLite can read sqlite_master for the database inspected by Hermes"
        )
    elif facts.schema_missing_tables:
        observations.append(
            "schema_core_tables: missing " + ", ".join(facts.schema_missing_tables)
        )
        recommended_actions.append(
            "verify HERMES_HOME/LCM_DATABASE_PATH point at the database inspected by Hermes"
        )
    else:
        observations.append("schema_core_tables: ok")

    if facts.debt_rows:
        first = facts.debt_rows[0]
        observations.append(
            f"maintenance_debt: {len(facts.debt_rows)} conversation(s) currently carry deferred maintenance debt; first={first[0]} kind={first[1]} size={first[2]}"
        )
        recommended_actions.append(
            "let normal compaction turns reduce maintenance debt before attempting broader cleanup"
        )

    if facts.clean_scan["error"]:
        observations.append(f"cleanup_candidates: scan error: {facts.clean_scan['error']}")
    elif facts.clean_scan["candidates"]:
        observations.append(
            f"cleanup_candidates: {len(facts.clean_scan['candidates'])} pattern-matched junk/noise session candidate(s) detected"
        )
        recommended_actions.append("inspect candidate sessions with `/lcm doctor clean`")
        recommended_actions.append("create a safety snapshot first with `/lcm backup`")
    else:
        observations.append("cleanup_candidates: none")

    if missing_externalized_refs:
        issues.append("payload_storage")
        observations.append(
            f"payload_storage: {missing_externalized_refs} externalized payload ref(s) point to missing JSON files"
        )
        recommended_actions.append(
            "inspect missing externalized payload refs and restore from backups if needed"
        )
    if suspicious_payload_rows:
        observations.append(
            f"payload_storage: {suspicious_payload_rows} suspicious inline/base64 payload row(s) need review"
        )
        recommended_actions.append(
            "inspect suspicious payload rows before cleanup; restore payload files from backup before deleting or rewriting anything"
        )
    if facts.payload_storage_error:
        observations.append(f"payload_storage_error: {facts.payload_storage_error}")
        recommended_actions.append("inspect payload storage diagnostics before cleanup or deletion")

    if "source_lineage" in facts.probe_errors:
        issues.append("source_lineage")
    observations.append(
        "source_lineage: "
        f"attributed={facts.source_stats['attributed_messages']} "
        f"unknown={facts.source_stats['normalized_unknown_messages']} "
        f"legacy_blank={facts.source_stats['legacy_blank_source_messages']} "
        f"effective_unknown={facts.source_stats['effective_unknown_messages']}"
    )
    if facts.source_stats.get("error"):
        observations.append(f"source_lineage_error: {facts.source_stats['error']}")
    if facts.source_stats["legacy_blank_source_messages"]:
        observations.append(
            "legacy blank-source rows are normalized as `source=unknown` for back-compat filters"
        )

    if "lifecycle_fragmentation" in facts.probe_errors:
        issues.append("lifecycle_fragmentation")
    else:
        observations.append(
            "lifecycle_fragmentation: "
            f"lifecycle_rows={facts.lifecycle_stats['lifecycle_rows']} "
            f"empty_lifecycle_rows={facts.lifecycle_stats.get('empty_lifecycle_rows', 0)} "
            f"message_sessions={facts.lifecycle_stats['distinct_message_sessions']} "
            f"node_sessions={facts.lifecycle_stats['distinct_node_sessions']} "
            f"current_missing_in_lcm_any={facts.lifecycle_stats['lifecycle_current_missing_in_lcm_any']} "
            f"last_finalized_missing_in_lcm_any={facts.lifecycle_stats['lifecycle_last_finalized_missing_in_lcm_any']} "
            f"current_missing_in_state={facts.lifecycle_stats['lifecycle_current_missing_in_state']} "
            f"last_finalized_missing_in_state={facts.lifecycle_stats['lifecycle_last_finalized_missing_in_state']} "
            f"message_sessions_missing_in_state={facts.lifecycle_stats['lcm_message_sessions_missing_in_state']} "
            f"node_sessions_missing_in_state={facts.lifecycle_stats['lcm_node_sessions_missing_in_state']} "
            f"message_sessions_without_lifecycle_current={facts.lifecycle_stats['message_sessions_without_lifecycle_current']} "
            f"message_sessions_without_lifecycle_reference={facts.lifecycle_stats['message_sessions_without_lifecycle_reference']} "
            f"node_sessions_without_lifecycle_reference={facts.lifecycle_stats['node_sessions_without_lifecycle_reference']} "
            f"state_sessions_missing_in_lcm_any={facts.lifecycle_stats['state_sessions_missing_in_lcm_any']}"
        )
        if facts.lifecycle_stats.get("state_db_error"):
            observations.append(f"lifecycle_fragmentation_state_db_error: {facts.lifecycle_stats['state_db_error']}")
        classification = facts.lifecycle_stats.get("classification") or {}
        categories = classification.get("categories") or []
        if classification:
            observations.append(
                "lifecycle_fragmentation_classification: "
                f"{classification.get('status', 'unknown')}; {len(categories)} categories need review"
            )
            for category in categories:
                sample = ",".join(category.get("sample_session_ids") or []) or "(none)"
                observations.append(
                    "lifecycle_category "
                    f"{category.get('name')}: count={category.get('count', 0)} sample={sample}"
                )
        if facts.lifecycle_fragmented:
            recommended_actions.append(
                "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state"
            )
            recommended_actions.append(
                "treat this as read-only evidence; do not infer every mismatch is harmful"
            )

    if facts.clean_scan.get("protected_count"):
        observations.append(
            f"protected_sessions: skipped {facts.clean_scan['protected_count']} currently bound session(s) from cleanup candidates"
        )

    if facts.protection["enabled"] and facts.protection["active_patterns"]:
        observations.append(
            "sensitive_pattern_handling: enabled; matching raw secret values are replaced before SQLite, FTS, summaries, active replay, and externalized payloads"
        )
    elif facts.protection["enabled"]:
        observations.append("sensitive_pattern_handling: enabled but no active known patterns are configured")
        recommended_actions.append("set LCM_SENSITIVE_PATTERNS to one or more known names, or disable sensitive handling")
    else:
        observations.append("sensitive_pattern_handling: disabled")
    if facts.protection["unknown_patterns"]:
        issues.append("sensitive_pattern_config")
        recommended_actions.append(
            "remove unknown LCM_SENSITIVE_PATTERNS entries or replace them with supported names"
        )

    if facts.store_fts_failed_flag:
        observations.append(
            "messages_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {facts.store_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild messages_fts"
        )
    if facts.node_fts_failed_flag:
        observations.append(
            "nodes_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {facts.node_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild nodes_fts"
        )

    fit_observation = survival_fit_observation(facts.survival_fit, facts.fit_count, facts.projection_floor)
    if fit_observation is not None:
        observations.append(fit_observation)
    if facts.stub_first_exit:  # #671: a partial stop; the rows no leaf summarised stay stored as backlog
        observations.append("stub_first_exit: the last threshold compaction stopped on its free cuts without a "
                            "model call (partial); " + " ".join(f"{k}={v}" for k, v in facts.stub_first_exit.items()))
    if facts.inactive_process:
        issues.append("inactive_process")
        recommended_actions.append(facts.inactive_action)
    if facts.embedding_check and facts.embedding_check["status"] in {"warn", "fail"}:
        issues.append("embedding_provider_health")
        recommended_actions.append(facts.embedding_action)

    doctor_status = "issues-found" if facts.integrity != "ok" or issues else (
        "action-recommended" if recommended_actions else "ok"
    )
    lines = [
        "LCM doctor",
        f"status: {doctor_status}",
        f"plugin_name: {facts.runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {facts.runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {facts.runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {facts.runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {facts.runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {facts.runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {facts.runtime_identity.get('plugin_git_dirty') if facts.runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"database_path: {facts.db_path}",
        f"database_exists: {_fmt_bool(facts.db_exists)}",
        f"database_size: {_fmt_size(facts.db_size) if facts.db_exists else 'missing'}",
        f"wal_size: {_fmt_size(facts.wal_size)}",
        f"schema_core_tables: {facts.schema_core_status}",
        f"schema_missing_tables: {', '.join(facts.schema_missing_tables) or '(none)'}",
        f"schema_existing_tables: {', '.join(facts.schema_existing_tables) or '(none)'}",
        f"journal_mode: {facts.journal_mode}; sqlite_mmap_bytes={facts.sqlite_mmap_bytes}",
        f"sqlite_library: {facts.sqlite_version} (WAL-reset bug: {'affected' if facts.wal_reset_affected else 'not affected'})",
        *(["sqlite_wal_reset_advice: upgrade Python/SQLite to 3.51.3+; see https://sqlite.org/wal.html#walresetbug"]
          if facts.wal_reset_affected and facts.journal_mode.lower() == "wal" else []),
        f"quick_check: {facts.quick_check}",
        f"sqlite_integrity: {facts.integrity}",
        f"messages_total: {facts.total_messages}",
        f"message_sessions_total: {facts.total_message_sessions}",
        f"summary_nodes_total: {facts.total_nodes}",
        f"summary_node_sessions_total: {facts.total_node_sessions}",
        f"messages_fts: {facts.store_fts}",
        f"stem index: {facts.stem_index}",
        f"messages_fts_rows: {facts.store_fts_count}",
        f"nodes_fts: {facts.node_fts}",
        f"nodes_fts_rows: {facts.node_fts_count}",
        f"largest_content_rows: {facts.payload_risks['largest_content_rows']}",
        f"largest_tool_calls_rows: {facts.payload_risks['largest_tool_calls_rows']}",
        f"suspicious_data_uri_content_rows: {facts.payload_risks['suspicious_data_uri_content_rows']}",
        f"suspicious_data_uri_tool_calls_rows: {facts.payload_risks['suspicious_data_uri_tool_calls_rows']}",
        f"suspicious_base64_like_rows: {facts.payload_risks['suspicious_base64_like_rows']}",
        f"quarantined_assistant_rows: {facts.payload_risks['quarantined_assistant_rows']}",
        f"suspicious_repetitive_assistant_rows: {facts.payload_risks['suspicious_repetitive_assistant_rows']}",
        f"heartbeat_noise_rows: {facts.payload_risks['heartbeat_noise_rows']}",
        f"sensitive_patterns_enabled: {_fmt_bool(facts.protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(facts.protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {facts.protection.get('source', 'default')}",
        f"sensitive_patterns_unknown: {', '.join(facts.protection.get('unknown_patterns') or []) or '(none)'}",
        f"externalized_payload_dir: {facts.externalized_stats['externalized_payload_dir']}",
        f"externalized_payload_count: {facts.externalized_stats['externalized_payload_count']}",
        f"externalized_payload_bytes: {facts.externalized_stats['externalized_payload_bytes']}",
        f"externalized_payload_chars: {facts.externalized_stats['externalized_payload_chars']}",
        f"latest_externalized_payload_path: {facts.externalized_stats['latest_externalized_payload_path'] or '(none)'}",
        f"externalized_payload_refs_total: {facts.externalized_integrity['externalized_payload_refs_total']}",
        f"externalized_payload_refs_existing: {facts.externalized_integrity['externalized_payload_refs_existing']}",
        f"externalized_payload_refs_missing: {facts.externalized_integrity['externalized_payload_refs_missing']}",
        f"externalized_payload_files_unreferenced: {facts.externalized_integrity['externalized_payload_files_unreferenced']}",
        f"missing_externalized_payload_refs: {facts.externalized_integrity['missing_externalized_payload_refs']}",
        f"unreferenced_externalized_payload_files: {facts.externalized_integrity['unreferenced_externalized_payload_files']}",
        f"survival_fit_count: {'unknown' if facts.fit_count is None else facts.fit_count}",
        *facts.host_uid_lines,
    ]
    if facts.inactive_process:
        lines.append(f"inactive_process: {facts.inactive_process}")
    if facts.embedding_check is not None:
        detail = facts.embedding_check.get("detail")
        reason = detail.get("reason") if isinstance(detail, dict) else None
        suffix = f" ({reason})" if reason else ""
        if isinstance(detail, str):
            suffix = f" {detail}"
        lines.append(f"embedding_provider_health: {facts.embedding_check['status']}{suffix}")
    if issues:
        lines.append(f"issues: {', '.join(issues)}")
    else:
        lines.append("issues: none")
    lines.append("observations:")
    for item in observations:
        lines.append(f"- {item}")
    lines.append("recommended_actions:")
    if recommended_actions:
        for item in recommended_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("triage_guidance:")
    if facts.triage_guidance:
        for item in facts.triage_guidance:
            warning_suffix = " warning-only" if item.get("warning_only") else ""
            lines.append(
                "- "
                f"{item['check']}: {item['action']}{warning_suffix} — "
                f"{item['operator_action']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _doctor_text(engine: Any) -> str:
    return render_doctor(collect_doctor(engine))
