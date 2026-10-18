"""Doctor characterization using real temporary databases, never live profiles.

Normalized output fields: temporary paths everywhere; database_size/wal_size
(text) and database_size_bytes/wal_size_bytes (JSON), which depend on SQLite
page allocation. Runtime identity is a template: plugin_version, plugin_path,
module_path, plugin_git_commit/branch/dirty/remote and sqlite_library are substituted
from this checkout/interpreter into the expected golden, not hard-coded to the
capture host. The inactive-process PID is likewise an expected template value.
No diagnostic status, counter, timestamp, list order or SQL order is discarded.
FaultConnection only raises at a named SQL boundary; all other SQL stays real.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import pytest

from hermes_lcm import command, tools  # type: ignore[import-not-found]
from tests.harness import make_engine

CASES = (
    "healthy",
    "missing_schema",
    "schema_error",
    "integrity_error",
    "fts_fail",
    "fts_unchecked",
    "messages_fts_error",
    "nodes_fts_error",
    "flags_error",
    "count_error",
    "journal_error",
    "quick_error",
    "payload_error",
    "missing_refs",
    "suspicious_rows",
    "cleanup",
    "lifecycle",
    "debt_error",
    "source_error",
    "fragmentation_error",
    "sensitive_unknown",
    "fit_zero",
    "fit_n",
    "fit_unreadable",
    "fit_invalid_count",
    "fit_count_lost",
    "fit_projection",
    "fit_last_not_shortened",
    "fit_never_shortened",
    "fit_no_cause",
    "stub_exit",
    "inactive",
    "embedding_warn",
    "embedding_fail",
    "embedding_raise",
    "wal_affected",
)
# The JSON tool never runs these queries, so their faults exist only on the text surface.
TEXT_ONLY_FAULTS = frozenset({"messages_fts_error", "nodes_fts_error", "count_error", "debt_error"})

TOOL_ONLY = (
    "schema_raise",
    "fts_raise",
    "sensitive_raise",
    "sync_error",
    "orphan_error",
    "quality_error",
)
FIXTURES = Path(__file__).parent / "fixtures"


class FaultConnection(sqlite3.Connection):
    """A real SQLite connection with one explicit, armed fault predicate."""

    fault: Callable[[str, Any], bool] = staticmethod(lambda sql, params: False)
    error: type[Exception] = sqlite3.OperationalError
    hits: int = 0

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if self.fault(" ".join(sql.split()), parameters):
            self.hits += 1
            raise self.error("fixture SQL fault")
        return super().execute(sql, parameters)


def _fault(case: str) -> Callable[[str, Any], bool]:
    predicates: dict[str, Callable[[str, Any], bool]] = {
        "schema_error": lambda q, p: (
            q == "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ),
        "schema_raise": lambda q, p: (
            q == "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ),
        "integrity_error": lambda q, p: q == "PRAGMA integrity_check",
        "messages_fts_error": lambda q, p: q == "SELECT COUNT(*) FROM messages_fts",
        "nodes_fts_error": lambda q, p: q == "SELECT COUNT(*) FROM nodes_fts",
        "fts_raise": lambda q, p: (
            q == "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
        ),
        "flags_error": lambda q, p: (
            q == "SELECT value FROM metadata WHERE key = ?"
            and str(p[0]).startswith("fts_integrity_failed:")
        ),
        "count_error": lambda q, p: q == "PRAGMA mmap_size",
        "journal_error": lambda q, p: q == "PRAGMA journal_mode",
        "quick_error": lambda q, p: q == "PRAGMA quick_check",
        "payload_error": lambda q, p: "AS content_len" in q,
        "debt_error": lambda q, p: (
            "SELECT conversation_id, debt_kind, debt_size_estimate" in q
        ),
        "source_error": lambda q, p: "AS normalized_unknown_messages" in q,
        "fragmentation_error": lambda q, p: (
            q == "SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL"
        ),
        "sync_error": lambda q, p: (
            q == "SELECT COUNT(*) FROM messages WHERE session_id = ?"
        ),
        "orphan_error": lambda q, p: q.startswith(
            "SELECT * FROM summary_nodes WHERE session_id = ?"
        ),
        "quality_error": lambda q, p: q.startswith(
            "SELECT node_id, session_id, depth, token_count, source_token_count"
        ),
    }
    return predicates.get(case, lambda q, p: False)


def _raise(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("fixture helper fault")


@contextmanager
def doctor_engine(
    tmp_path: Path, case: str, patch: pytest.MonkeyPatch, *, trace: bool = False
) -> Iterator[Any]:
    # No session bind: no background work, host calls, or provider loads.
    engine: Any = make_engine(
        tmp_path, embeddings_enabled=False, ignore_session_patterns=["noise-*"]
    )
    try:
        engine._session_id = "noise-active" if case == "cleanup" else "session"
        conn = engine._store.connection
        if case in {"missing_refs", "suspicious_rows", "cleanup", "fts_fail"}:
            content = {
                "missing_refs": "[GC'd externalized payload: kind=raw_payload; role=assistant; chars=10; ref=missing.json]",
                "suspicious_rows": "data:image/png;base64," + "QUJD" * 2000,
            }.get(case, "fixture message")
            # Seed historical rows directly: do not exercise ingest protection.
            conn.execute(
                "INSERT INTO messages(session_id, source, role, content, timestamp, token_estimate) VALUES (?, 'unknown', 'assistant', ?, 1700000000, 7)",
                (engine._session_id, content),
            )
        if case == "cleanup":
            conn.execute(
                "INSERT INTO messages(session_id, source, role, content, timestamp, token_estimate) VALUES ('noise-old', 'unknown', 'user', 'old noise', 1700000000, 3)"
            )
        if case == "missing_schema":
            conn.execute("DROP TABLE lcm_lifecycle_state")
        if case in {"fts_fail", "flags_error"}:
            for table in ("messages_fts", "nodes_fts"):
                conn.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                    (
                        f"fts_integrity_failed:{table}",
                        json.dumps(
                            {"at": 1700000000, "detail": "fixture background failure"}
                        ),
                    ),
                )
            if case == "fts_fail":
                # Same docsize count, stale token index: only the real deep check catches it.
                conn.execute(
                    "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', 1, 'fixture message')"
                )
                conn.execute(
                    "INSERT INTO messages_fts(rowid, content) VALUES(1, 'stale message')"
                )
        if case == "lifecycle":
            conn.execute(
                "INSERT INTO lcm_lifecycle_state(conversation_id, current_session_id, debt_kind, debt_size_estimate, updated_at) VALUES('conversation', 'missing-session', 'summary', 12, 1700000000)"
            )
        if case.startswith("fit_"):
            counter: dict[str, Any] = {
                "count": 3,
                "projected_count": 0,
                "last_reason": "threshold",
                "last_cause": "overflow",
                "ever_shortened": True,
            }
            variants: dict[str, dict[str, Any]] = {
                "fit_zero": {"count": 0},
                "fit_invalid_count": {"count": "damaged"},
                "fit_count_lost": {"count_lost": True},
                "fit_projection": {"projected_count": 2},
                "fit_last_not_shortened": {"last_shortened": False},
                "fit_never_shortened": {
                    "last_shortened": False,
                    "ever_shortened": False,
                },
            }
            counter.update(variants.get(case, {}))
            if case == "fit_no_cause":
                counter.pop("last_cause")
            conn.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                (
                    "survival_fit:counter",
                    "{" if case == "fit_unreadable" else json.dumps(counter),
                ),
            )
        if case == "stub_exit":
            engine._last_stub_first_exit = {"reason": "free-cuts", "stored_rows": 2}
        if case == "inactive":
            home = Path(engine._hermes_home)
            home.mkdir(parents=True, exist_ok=True)
            (home / "lcm-x-not-active.json").write_text(
                json.dumps({"pid": os.getpid(), "reason": "fixture inactive"})
            )
        conn.commit()
        # Preserve the separate real connection/row-factory topology.
        for owner in (engine._store, engine._dag, engine._lifecycle):
            old = owner.connection
            old.commit()
            replacement = sqlite3.connect(
                str(engine._store.db_path), factory=FaultConnection
            )
            replacement.row_factory = old.row_factory
            replacement.execute("PRAGMA mmap_size=0")
            if case == "fts_unchecked":
                replacement.execute("PRAGMA query_only=ON")
            replacement.fault = _fault(case)
            if case in {"schema_raise", "fts_raise"}:
                replacement.error = RuntimeError
            owner._conn = replacement
            old.close()
        if not trace:
            patch.setattr(
                command, "sqlite_wal_reset_affected", lambda: case == "wal_affected"
            )
            patch.setattr(
                command, "stem_index_status", lambda conn: "fixture stem ready"
            )
            protection = {
                "enabled": False,
                "patterns": [],
                "active_patterns": [],
                "unknown_patterns": ["fixture-unknown"]
                if case == "sensitive_unknown"
                else [],
                "source": "default",
            }
            patch.setattr(
                command, "sensitive_pattern_status", lambda config: protection
            )
            patch.setattr(
                tools,
                "sensitive_pattern_status",
                _raise if case == "sensitive_raise" else lambda config: protection,
            )
            health = {
                "check": "embedding_provider_health",
                "status": {"embedding_warn": "warn", "embedding_fail": "fail"}.get(
                    case, "pass"
                ),
                "detail": {"reason": "fixture offline probe"},
            }
            patch.setattr(
                tools,
                "_embedding_provider_health_check",
                _raise if case == "embedding_raise" else lambda engine: health,
            )
        yield engine
        if case in {
            "schema_error",
            "schema_raise",
            "integrity_error",
            "messages_fts_error",
            "nodes_fts_error",
            "flags_error",
            "count_error",
            "journal_error",
            "quick_error",
            "payload_error",
            "debt_error",
            "source_error",
            "fragmentation_error",
            "fts_raise",
            "sync_error",
            "orphan_error",
            "quality_error",
        }:
            assert (
                sum(
                    owner.connection.hits
                    for owner in (engine._store, engine._dag, engine._lifecycle)
                )
                > 0
            )
    finally:
        for owner in (engine._store, engine._dag, engine._lifecycle):
            if isinstance(owner.connection, FaultConnection):
                owner.connection.fault = lambda q, p: False
        engine.shutdown()


# SQLite builds word a corrupted FTS5 shadow table differently; both mean the same failed check.
FTS_INTEGRITY_ERRORS = frozenset({
    'fts5: checksum mismatch for table "messages_fts"',
    "database disk image is malformed",
})


def normalize(value: Any, tmp_path: Path) -> Any:
    """Normalize scratch paths, physical SQLite allocation sizes and SQLite's FTS error wording."""
    if isinstance(value, dict):
        detail = value.get("detail")
        if value.get("check") == "messages_fts_integrity" and isinstance(detail, str) and detail in FTS_INTEGRITY_ERRORS:
            value = {**value, "detail": "<sqlite fts integrity error>"}
        return {
            k: "<size>"
            if k in {"database_size_bytes", "wal_size_bytes"}
            else normalize(v, tmp_path)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize(v, tmp_path) for v in value]
    if isinstance(value, str):
        value = value.replace(str(tmp_path.resolve()), "<tmp>").replace(
            str(tmp_path), "<tmp>"
        )
        return re.sub(r"(?m)^(database_size|wal_size): .*$", r"\1: <size>", value)
    return value


def observe_identity(engine: Any, patch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Observe the doctor's real read, avoiding a second git subprocess race."""
    observed: dict[str, Any] = {}
    original = engine.get_runtime_identity

    def read() -> Any:
        value = original()
        observed.update(value)
        return value

    patch.setattr(engine, "get_runtime_identity", read)
    return observed


def bind_template(value: Any, identity: dict[str, Any]) -> Any:
    """Keep provenance fields present and compare them to the actual runtime."""
    if isinstance(value, dict):
        return {k: bind_template(v, identity) for k, v in value.items()}
    if isinstance(value, list):
        return [bind_template(v, identity) for v in value]
    if isinstance(value, str):
        for key in (
            "plugin_version",
            "plugin_path",
            "module_path",
            "plugin_git_commit",
            "plugin_git_branch",
            "plugin_git_dirty",
            "plugin_git_remote",
        ):
            marker = "<" + key + ">"
            if value == marker:
                return identity[key]
            value = value.replace(
                marker,
                str(identity[key]) if identity[key] is not None else "(unavailable)",
            )
        value = value.replace("<sqlite_version>", sqlite3.sqlite_version).replace(
            "<pid>", str(os.getpid())
        )
    return value


def collect_trace(engine: Any, surface: str, patch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []

    def record(sql: str) -> None:
        if sql.lstrip().startswith("--") or re.search(
            r"(['\"])(main|temp)\1\s*\.", sql
        ):
            return
        events.append("sql: " + " ".join(sql.split()))

    def wrap(owner: Any, name: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            events.append("call: " + name)
            return original(*args, **kwargs)

        patch.setattr(owner, name, wrapped)

    for owner in (engine._store, engine._dag, engine._lifecycle):
        owner.connection.set_trace_callback(record)
    module = command if surface == "text" else tools
    for name in (
        "inspect_lcm_schema_health",
        "check_external_content_fts_integrity",
        "load_integrity_failed",
        "scan_sqlite_payload_risks",
        "externalized_payload_stats",
        "scan_externalized_payload_integrity",
        "sensitive_pattern_status",
        "inactive_record_notice",
        "doctor_guidance_for_checks",
    ):
        wrap(module, name)
    if surface == "text":
        for name in (
            "_scan_clean_candidates",
            "sqlite_wal_reset_affected",
            "stem_index_status",
            "host_uid_doctor_lines",
        ):
            wrap(command, name)
    wrap(tools, "_embedding_provider_health_check")
    for owner, name in (
        (engine, "get_runtime_identity"),
        (engine._store, "get_source_stats"),
        (engine._store, "read_metadata_json"),
        (engine._lifecycle, "get_fragmentation_stats"),
    ):
        wrap(owner, name)
    if surface == "text":
        command._doctor_text(engine)
    else:
        tools.lcm_doctor({}, engine=engine)
    for owner in (engine._store, engine._dag, engine._lifecycle):
        owner.connection.set_trace_callback(None)
    return events


@pytest.mark.parametrize("case", CASES)
def test_doctor_text_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    expected = json.loads((FIXTURES / "command_doctor.json").read_text())["cases"][case]
    with doctor_engine(tmp_path, case, monkeypatch) as engine:
        identity = observe_identity(engine, monkeypatch)
        assert normalize(command._doctor_text(engine), tmp_path) == normalize(
            bind_template(expected, identity), tmp_path
        )


@pytest.mark.parametrize("case", (*(c for c in CASES if c not in TEXT_ONLY_FAULTS), *TOOL_ONLY))
def test_doctor_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    expected = json.loads((FIXTURES / "lcm_doctor.json").read_text())["cases"][case]
    with doctor_engine(tmp_path, case, monkeypatch) as engine:
        identity = observe_identity(engine, monkeypatch)
        actual = tools.lcm_doctor({}, engine=engine)
        # Preserve JSON key ordering and the public serializer's exact spacing.
        assert actual == json.dumps(json.loads(actual))
        assert json.dumps(normalize(json.loads(actual), tmp_path)) == json.dumps(
            normalize(bind_template(expected, identity), tmp_path)
        )


@pytest.mark.parametrize("case", ("healthy", "fts_fail"))
@pytest.mark.parametrize("surface", ("text", "json"))
def test_doctor_read_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, surface: str
) -> None:
    file = "command_doctor.json" if surface == "text" else "lcm_doctor.json"
    expected = json.loads((FIXTURES / file).read_text())["traces"][case]
    with doctor_engine(tmp_path, case, monkeypatch, trace=True) as engine:
        events = collect_trace(engine, surface, monkeypatch)
        assert normalize(events, tmp_path) == expected
        for table in ("messages_fts", "nodes_fts"):
            save = f'sql: SAVEPOINT "lcm_fts_integrity_{table}"'
            insert = f'sql: INSERT INTO "{table}"("{table}", rank) VALUES(\'integrity-check\', 1)'
            rollback = f'sql: ROLLBACK TO "lcm_fts_integrity_{table}"'
            assert events.index(save) < events.index(insert) < events.index(rollback)
        if surface == "text":
            guidance = events.index("call: doctor_guidance_for_checks")
            assert all(
                events.index("call: " + name) > guidance
                for name in (
                    "sqlite_wal_reset_affected",
                    "stem_index_status",
                    "host_uid_doctor_lines",
                )
            )


@pytest.mark.parametrize("owner_name,helper,expected", (
    ("command", "load_integrity_failed", "issues: none"),
    ("command", "check_external_content_fts_integrity", "issues: messages_fts, nodes_fts"),
    ("store", "read_metadata_json", "survival_fit_count: 0"),
    ("store", "get_source_stats", "issues: source_lineage"),
    ("lifecycle", "get_fragmentation_stats", "issues: lifecycle_fragmentation"),
    ("command", "externalized_payload_stats", "issues: payload_storage"),
    ("command", "scan_externalized_payload_integrity", "issues: payload_storage"),
))
def test_doctor_helper_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    owner_name: str, helper: str, expected: str,
) -> None:
    with doctor_engine(tmp_path, "healthy", monkeypatch) as engine:
        owner = {"command": command, "store": engine._store,
                 "lifecycle": engine._lifecycle}[owner_name]
        calls = []

        def fail(*args: Any, **kwargs: Any) -> Any:
            calls.append(helper)
            return _raise()

        monkeypatch.setattr(owner, helper, fail)
        text = command._doctor_text(engine)
        assert expected in text.splitlines()
        assert len(calls) == (2 if helper in {
            "load_integrity_failed", "check_external_content_fts_integrity", "read_metadata_json"
        } else 1)


def test_doctor_compound_exception_issue_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with doctor_engine(tmp_path, "healthy", monkeypatch) as engine:
        engine._store.connection.fault = lambda q, p: q in {
            "PRAGMA integrity_check", "PRAGMA mmap_size", "PRAGMA journal_mode",
            "PRAGMA quick_check", "SELECT COUNT(*) FROM messages",
            "SELECT COUNT(DISTINCT session_id) FROM messages",
        }
        engine._dag.connection.fault = lambda q, p: q in {
            "SELECT COUNT(*) FROM summary_nodes",
            "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        }
        engine._lifecycle.connection.fault = lambda q, p: (
            "SELECT conversation_id, debt_kind, debt_size_estimate" in q
        )
        for helper in ("scan_sqlite_payload_risks", "check_external_content_fts_integrity"):
            monkeypatch.setattr(command, helper, _raise)
        monkeypatch.setattr(engine._store, "get_source_stats", _raise)
        monkeypatch.setattr(engine._lifecycle, "get_fragmentation_stats", _raise)
        monkeypatch.setattr(tools, "_embedding_provider_health_check", _raise)
        text = command._doctor_text(engine)
        assert next(line for line in text.splitlines() if line.startswith("issues:")) == (
            "issues: sqlite_integrity, messages_fts, nodes_fts, sqlite_mmap_size, "
            "messages_total, message_sessions_total, summary_nodes_total, "
            "summary_node_sessions_total, sqlite_journal_mode, sqlite_quick_check, "
            "payload_storage, lifecycle_state, source_lineage, lifecycle_fragmentation, "
            "embedding_provider_health"
        )
        assert text.count("payload_storage, ") == 1


def test_doctor_missing_payload_precedes_late_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with doctor_engine(tmp_path, "missing_refs", monkeypatch) as engine:
        monkeypatch.setattr(engine._store, "get_source_stats", _raise)
        monkeypatch.setattr(engine._lifecycle, "get_fragmentation_stats", _raise)
        text = command._doctor_text(engine)
        assert "issues: payload_storage, source_lineage, lifecycle_fragmentation" in text.splitlines()


@pytest.mark.parametrize("case", ("healthy", "payload_error", "fit_projection", "embedding_raise"))
def test_doctor_render_is_detached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    from hermes_lcm.command_doctor import collect_doctor, render_doctor

    with doctor_engine(tmp_path, case, monkeypatch) as engine:
        facts = collect_doctor(engine)
        expected = render_doctor(facts)
        for owner in (engine._store, engine._dag, engine._lifecycle):
            owner.connection.fault = lambda q, p: True
        for name in (
            "inspect_lcm_schema_health", "sensitive_pattern_status", "sqlite_wal_reset_affected",
            "stem_index_status", "host_uid_doctor_lines", "doctor_guidance_for_checks",
            "doctor_guidance_for_check", "survival_fit_count", "survival_fit_projection_floor_applies",
        ):
            monkeypatch.setattr(command, name, _raise)
        monkeypatch.setattr(tools, "_embedding_provider_health_check", _raise)
        monkeypatch.setattr(Path, "exists", _raise)
        monkeypatch.setattr(Path, "stat", _raise)
        assert render_doctor(facts) == expected
        assert render_doctor(facts) == expected
        # Restore pathlib before the engine's fixture teardown.
        monkeypatch.undo()


def test_doctor_render_keeps_duplicate_payload_issue_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from hermes_lcm.command_doctor import collect_doctor, render_doctor

    with doctor_engine(tmp_path, "missing_refs", monkeypatch) as engine:
        facts = collect_doctor(engine)
        # The collector's fallback cannot produce this combination today. Replay
        # both issue sites independently so rendering never deduplicates them.
        facts = replace(facts, probe_errors={"payload_storage": True, "lifecycle_state": True})
        text = render_doctor(facts)
        assert "issues: payload_storage, lifecycle_state, payload_storage" in text.splitlines()
