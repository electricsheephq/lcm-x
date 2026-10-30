"""Pinned-base return, proof, product-row and application-SQL observations."""
import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from tests.behaviour.reconcile_trace_harness import install_clock
from tests.test_issue_2055_emission_writes import PATHS, prepare_path
from tests.test_issue_2055_ingest_ledger_writes import engines as _engines

engines = _engines

GOLDEN = Path(__file__).parent / "fixtures" / "emission_shadow_base.json"
SIGNALS = ("in_place", "rotation", "rotation_drift", "refusal", "false_boundary", "rotation_skip")


def clean(value, root):
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [clean(v, root) for v in value]
    if isinstance(value, dict):
        return {k: clean(v, root) for k, v in value.items()}
    if isinstance(value, str):
        value = value.replace(root, "<fixture>")
        # Home hashes are scope salts, not outcome evidence; retain the session component.
        value = re.sub(r"[0-9a-f]{16}(:S)", r"<home>\1", value)
    return value


def snapshot(engine, before, returned, raised, trace):
    conn = engine._store._conn
    root = str(Path(engine._store.db_path).parent)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
              if row[0] not in {"ledger_meta", "occurrence_bindings", "ledger_emissions", "ingest_views", "lineage_marks",
                                "stamp_observations", "sqlite_sequence"} and "fts" not in row[0]]
    rows = {table: conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall() for table in tables}
    sql = [s for s in trace if not s.lstrip().startswith("--") and not re.search(r"(['\"])(main|temp)\1\s*\.", s)]
    return clean(dict(returned=returned, raised=raised, caller=before, rows=rows, sql=sql,
                      proof=engine._compress_commit_proof, descriptors=engine._last_emission_descriptors,
                      reason=engine._compress_commit_reason, cursor=engine._ingest_cursor,
                      reconcile=engine._ingest_cursor_needs_reconcile), root)


def observation(engines, monkeypatch, path, mode, name="case"):
    install_clock(monkeypatch)
    engine = engines(name=name, mode=mode)
    engine._hermes_home = "fixture-home"
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda *a, **kw: ("fixture-lineage", None))
    before = prepare_path(engine, monkeypatch, path if path in PATHS else "normal")
    trace, returned, raised = [], None, None
    engine._store._conn.set_trace_callback(trace.append)
    try:
        try:
            returned = engine.compress(before, current_tokens=100, force=True)
        except ValueError as exc:
            raised = [type(exc).__name__, str(exc)]
        if path == "in_place":
            engine.on_session_end("S", before)
            engine._continue_in_place_compression_boundary("S", dict(platform="cli", conversation_id="conv"))
        elif path in {"rotation", "rotation_drift"}:
            engine.on_session_end("S", before)
            if path == "rotation_drift":
                from hermes_lcm.dag import SummaryNode
                engine._dag.add_node(SummaryNode(session_id="S", summary="source summary", source_ids=[1]))
                engine._session_id = "side-channel"
            engine._continue_compression_boundary("child", "S", dict(platform="cli", conversation_id="conv"))
        elif path == "refusal":
            engine.record_rejected_compaction()
        elif path == "false_boundary":
            engine._continue_in_place_compression_boundary("wrong", {})
        elif path == "rotation_skip":
            engine._session_id = "unrelated"
            engine._continue_compression_boundary("child", "missing", dict(conversation_id="other"))
    finally:
        engine._store._conn.set_trace_callback(None)
    return snapshot(engine, before, returned, raised, trace)


def product_sql(sql):
    """Remove only the new ledger statements and their separate transaction controls."""
    product = []
    for statement in sql:
        if re.search(r"\b(ledger_meta|ledger_emissions|ingest_views|occurrence_bindings|stamp_observations|lineage_marks)\b", statement):
            continue
        if statement.upper() in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"} or statement.startswith("PRAGMA busy_timeout"):
            continue
        product.append(statement)
    return product


@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("path", [*PATHS, *SIGNALS])
def test_pinned_base_parity(engines, monkeypatch, path, mode):
    connect = sqlite3.connect

    def frozen_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.create_function("strftime", -1, lambda *args: "1700001000")
        return conn
    monkeypatch.setattr(sqlite3, "connect", frozen_connect)
    observed = observation(engines, monkeypatch, path, mode)
    target = os.environ.get("LCM_P1C1_GOLDEN_OUT")
    key = mode + ":" + path
    if target:
        repeated = observation(engines, monkeypatch, path, mode, name="repeat")
        assert repeated == observed
        output = Path(target)
        expected = json.loads(output.read_text()) if output.exists() else {}
        expected[key] = observed
        output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return
    expected = json.loads(GOLDEN.read_text())[key]
    if mode == "shadow":
        observed["sql"], expected["sql"] = product_sql(observed["sql"]), product_sql(expected["sql"])
    assert observed == expected
