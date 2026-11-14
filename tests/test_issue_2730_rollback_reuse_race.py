"""#2730 stays open: a backfill rollback can delete a payload file that a row committed during the rollback references.

The rollback reads the set of referenced payload refs once, at its start. A live ingest of the same tool result (same
content, kind, tool-call id and session) can then reuse that payload file by digest and commit a row that references
it; the rollback, still working from its first read, deletes the file, and the new row points at a missing payload.
The strict xfail pins that race by committing the identical delivery right after the reference read and before any
unlink. The sequential control commits it before the read, and the file is kept. A fix flips the xfail. The docs
already say to run the rollback with no live engine on the store. Only isolated scratch stores and synthetic tool
content are used.
"""

import copy
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm.externalize import extract_externalized_ref, get_large_output_storage_dir
from tests.test_historical_externalization_backfill import _load_script, _run_backfill
from tests.test_issue_2055_ingest_ledger_writes import engines as engines

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2730: a rollback deletes a payload file that a row committed after its reference read has reused",
)


def _require(condition, message):
    # A broken precondition is a change of the pinned shape, not the expected failure.
    if not condition:
        pytest.fail(message)


@pytest.mark.parametrize("concurrent_reuse", [
    pytest.param(False, id="sequential-control"),
    pytest.param(True, id="reuse-after-scan", marks=_PIN),
])
def test_rollback_keeps_committed_reused_payload(engines, monkeypatch, tmp_path, concurrent_reuse):
    engine = engines(sessions=[("X", None, None)])
    engine.on_session_start("X", platform="cli", context_length=200_000, conversation_id="conv-X")
    config = engine._config
    config.large_output_externalization_enabled = False
    config.large_output_externalization_threshold_chars = 100
    content = "synthetic historical tool output: rows alpha beta gamma\n" * 100

    def turn(label, stamp):
        return [
            {"role": "user", "content": label, "timestamp": stamp},
            {"role": "assistant", "content": "", "timestamp": stamp + 1, "tool_calls": [
                {"id": "C", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "C", "content": content, "timestamp": stamp + 2},
        ]

    old = turn("historical request", 1.0)
    engine.ingest(copy.deepcopy(old))
    _require(engine._ingest_failure_count == 0, "the historical turn's ingest failed")
    _require(engine._store.get_session_messages("X")[-1]["content"] == content, "the historical result is not inline")
    module = _load_script()
    home = Path(engine._hermes_home)
    database = Path(engine._store.db_path)
    manifest = tmp_path / "backfill.json"
    backfill = _run_backfill(module, home, database, config, manifest, apply=True)
    _require(backfill["counts"]["created"] == 1, f"the backfill did not create one payload: {backfill['counts']}")
    ref = backfill["items"][0]["ref"]
    payload = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
    _require(json.loads(payload.read_text())["content"] == content, "the backfilled payload does not hold the result")
    config.large_output_externalization_enabled = True
    view = old + turn("live redelivery of C", 10.0)
    committed = []

    def reuse_and_commit():
        engine.ingest(copy.deepcopy(view))
        _require(engine._ingest_failure_count == 0, f"the live redelivery's ingest failed: {engine._last_ingest_error}")
        # An independent reader proves COMMIT, not merely connection-local state.
        with sqlite3.connect(database) as reader:
            rows = reader.execute(
                "SELECT content FROM messages WHERE session_id='X' ORDER BY store_id"
            ).fetchall()
        _require(len(rows) == len(view), f"expected {len(view)} committed rows, saw {len(rows)}")
        _require(rows[2][0] == content, "backfill must leave the historical row inline")
        _require(extract_externalized_ref(rows[-1][0]) == ref, "live ingest did not reuse the payload file")
        _require(json.loads(payload.read_text())["content"] == content, "the reused payload changed")
        committed.append(rows[-1][0])

    scan = module._collect_referenced_refs
    scanned = []

    def scan_then_reuse(connection):
        refs = scan(connection)
        scanned.append(refs.copy())
        if concurrent_reuse:
            _require(ref not in refs, "the rollback's reference read already saw the reused payload")
            reuse_and_commit()
        else:
            _require(ref in refs, "the rollback's reference read missed the committed reuse")
        return refs

    if not concurrent_reuse:
        reuse_and_commit()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_collect_referenced_refs", scan_then_reuse)
        rollback = module.run_rollback(
            database_path=database, hermes_home=home, source_manifest_path=manifest,
            apply=True, config=config,
        )
    _require(len(scanned) == len(committed) == 1, f"scans={len(scanned)} commits={len(committed)}; expected one each")
    with sqlite3.connect(database) as reader:
        committed_ref = extract_externalized_ref(reader.execute(
            "SELECT content FROM messages WHERE session_id='X' ORDER BY store_id DESC LIMIT 1"
        ).fetchone()[0])
    _require(committed_ref == ref, "the committed row no longer references the payload file")
    assert payload.is_file(), (
        f"payload_exists={payload.exists()}; committed_row_references_it={committed_ref == ref}; "
        f"scan_contained_it={ref in scanned[0]}; rollback_counts={rollback['counts']}"
    )
    assert json.loads(payload.read_text())["content"] == content
    assert rollback["counts"]["deleted"] == 0
    assert rollback["counts"]["skipped_referenced"] == 1
