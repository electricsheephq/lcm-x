"""Compare rollback from base and candidate snapshots after a deferred refusal (I1, I2, I11)."""

from hashlib import sha256
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.config import LCMConfig, host_message_uid_mode
from hermes_lcm.engine import LCMEngine
from tests.compat import helpers
from tests.compat.helpers import run_old
from tests.test_compression_boundary import _stub_summarizer, _turn
from tests.test_refused_compaction_rollback import (
    _defer_rewriting_pass_refusal,
    _drain_session_end_maintenance,
    _rewriting_pass,
)


PINS = {
    "v0.30.2": "f77d6ed98c8f2e896851b31720876e4c9d10f7b4",
    "v0.30.3-rc1": "112e3035eb1716a1f847d0953514df120174f5b3",
}
SID = "S0"
BINDING = dict(platform="acp", context_length=200_000, conversation_id="refusal-test")
SETTINGS = dict(embeddings_enabled=False, temporal_rollups_enabled=False, fresh_tail_count=6, leaf_chunk_tokens=400)
LINEAGE = "test:S0"
ROW_QUERY = "SELECT store_id, role, content FROM messages ORDER BY store_id"
ROLLBACK_FIELDS = ("rows_added", "heads_added", "duplicates_added", "status", "noop_reason")

OLD_SEED = r'''
from pytest import MonkeyPatch
from tests.compat.test_3001_refused_snapshot_rollback import _seed_store, _set_uid_mode

with MonkeyPatch.context() as patch:
    _set_uid_mode(mode, patch)
    kept, before = _seed_store(Path(db).parent, data, patch, expect_refusal_mark=False)
print(json.dumps([kept, before]))
'''

OLD_RESUME = r'''
import logging
import os
import sqlite3
from contextlib import closing
from hashlib import sha256

from hermes_lcm import host_uid_emit
from hermes_lcm.config import host_message_uid_mode

if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")
inputs = json.loads(data)
host_uid_emit._host_uid_capability = True
LCMEngine._host_uid_lineage_key = lambda *_args, **_kwargs: (inputs["lineage"], None)
report = dict(rows=[], summary_heads=[], status=None, noop_reason=None, errors=[], log_lines=[])


class Capture(logging.Handler):
    def emit(self, record):
        line = record.getMessage()
        if record.exc_info or any(word in line.lower() for word in ("snapshot", "proof", "lineage", "conflict")):
            report["log_lines"].append(self.format(record))


logger = logging.getLogger("hermes_lcm")
logger.setLevel(logging.DEBUG)
logger.addHandler(Capture())
engine = None
stage = "open"
try:
    engine = LCMEngine(config=LCMConfig(**(options | inputs["settings"])), hermes_home=home)
    stage = "bind"
    engine.on_session_start(inputs["session_id"], **inputs["binding"])
    view = inputs["view"]
    stage = "ingest"
    engine.ingest(view)
    if engine._last_ingest_error:
        report["errors"].append(f"{stage}: {engine._last_ingest_error}")
    stage = "compress"
    engine.compress(view, current_tokens=tokens.count_messages_tokens(view), force=True)
    if engine._last_ingest_error and f"ingest: {engine._last_ingest_error}" not in report["errors"]:
        report["errors"].append(f"{stage}: {engine._last_ingest_error}")
except Exception as exc:
    report["errors"].append(f"{stage}: {type(exc).__name__}: {exc}")
finally:
    if engine is not None:
        report.update(status=engine.last_compression_status, noop_reason=engine.last_compression_noop_reason)
        try:
            engine.shutdown()
        except Exception as exc:
            report["errors"].append(f"shutdown: {type(exc).__name__}: {exc}")

try:
    with closing(sqlite3.connect(f"{Path(db).as_uri()}?mode=ro", uri=True)) as conn:
        for store_id, role, content in conn.execute(inputs["row_query"]):
            row = [store_id, role, sha256(content.encode()).hexdigest()]
            report["rows"].append(row)
            if role == "user" and LCMEngine._LCM_SUMMARY_PART_HEADER_RE.match(content):
                report["summary_heads"].append(row)
except Exception as exc:
    report["errors"].append(f"readback: {type(exc).__name__}: {exc}")
print(json.dumps(report))
'''


def _set_uid_mode(uid_mode, monkeypatch):
    if uid_mode == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    assert host_message_uid_mode() == ("on" if uid_mode == "on" else "shadow")


def _seed_store(directory, shape, monkeypatch, *, expect_refusal_mark):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    monkeypatch.setattr(LCMEngine, "_host_uid_lineage_key", lambda *_args, **_kwargs: (LINEAGE, None))
    db = directory / "lcm.db"
    config = LCMConfig(database_path=str(db), large_output_externalization_path=str(directory / "payloads"), **SETTINGS)
    engine = LCMEngine(config=config, hermes_home=str(directory / "home"))
    engine.on_session_start(SID, **BINDING)
    try:
        # Keep the crash fixture's two assistant rows that merge in the first adopted view.
        seed = [row for turn in range(1, 13) for row in _turn(turn)]
        seed.extend([{"role": "assistant", "content": "One more reply to turn 12."}, _turn(13)[0]])
        initial = engine.compress(seed, force=True)
        assert engine._verified_lcm_summary_prefix_end(initial[0]["content"])
        _drain_session_end_maintenance(engine)
        engine.on_session_end(SID, seed)
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, **BINDING)
        kept = initial + [_turn(13)[1]] + [row for turn in range(14, 21) for row in _turn(turn)]

        _defer_rewriting_pass_refusal(engine, kept, monkeypatch)
        refused = engine._emission_handle
        key = engine._replay_snapshot_metadata_key("host_occurrence_snapshot", SID)
        mark = engine._store.read_metadata_json(key).get("refused_emission")
        if expect_refusal_mark:
            assert (mark["lineage"], mark["view_seq"]) == (refused.lineage, refused.view_seq)
        else:
            assert mark is None
        if shape == "refused-then-adopted":
            continued = kept + [row for turn in range(21, 25) for row in _turn(turn)]
            kept = _rewriting_pass(engine, continued)
            _drain_session_end_maintenance(engine)
            engine.on_session_end(SID, continued)
            engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, **BINDING)
            adopted = engine._emission_handle
            assert refused.view_seq < adopted.view_seq and refused.lineage == adopted.lineage
            states = engine._store.connection.execute(
                "SELECT DISTINCT view_seq, state FROM ledger_emissions WHERE lineage=? AND view_seq IN (?,?)",
                (refused.lineage, refused.view_seq, adopted.view_seq),
            ).fetchall()
            assert set(states) == {(refused.view_seq, "refused"), (adopted.view_seq, "adopted")}
            assert "refused_emission" not in engine._store.read_metadata_json(key)
            assert engine._durable_commit_proof_payload()["output_metadata"]

        before = [[store_id, role, sha256(content.encode()).hexdigest()]
                  for store_id, role, content in engine._store.connection.execute(ROW_QUERY)]
        return kept, before
    finally:
        # Closing SQLite leaves the refused metadata in place. Engine shutdown restores it.
        _drain_session_end_maintenance(engine)
        engine._store.close()


def _resume(old, directory, host_root, uid_mode, data, monkeypatch):
    process = None

    def capture_process(*args, **kwargs):
        nonlocal process
        process = subprocess.run(*args, **kwargs)
        return process

    with monkeypatch.context() as patch:
        # run_old asserts success before decoding stdout. Retain the exit code on failures too.
        patch.setattr(helpers, "subprocess", SimpleNamespace(run=capture_process))
        try:
            report = run_old(old, directory, host_root, OLD_RESUME, mode=uid_mode, data=data)
        except subprocess.TimeoutExpired as exc:
            return dict(rc="timeout", errors=[f"reader stalled after {exc.timeout}s"], stderr=exc.stderr)
        except (AssertionError, json.JSONDecodeError) as exc:
            return dict(rc=process.returncode, errors=[str(exc) or "reader failed"], stderr=process.stderr)
    return dict(report, rc=process.returncode, stderr=process.stderr)


def _measure_rollback(old, original, directory, host_root, uid_mode, kept, before, monkeypatch):
    directory.mkdir()
    for suffix in ("", "-wal", "-shm"):
        source = original / f"lcm.db{suffix}"
        if source.exists():
            shutil.copyfile(source, directory / source.name)

    turn = _turn(25)
    data = json.dumps(dict(settings=SETTINGS, session_id=SID, binding=BINDING, lineage=LINEAGE,
                           view=[*kept, *turn], row_query=ROW_QUERY))
    expected = [[row["role"], sha256(row["content"].encode()).hexdigest()] for row in turn]
    report = _resume(old, directory, host_root, uid_mode, data, monkeypatch)
    problems = []
    if report["rc"] != 0 or report["errors"]:
        problems.append(f"rc={report['rc']}, errors={report['errors']}")
    if "rows" not in report:
        return report, problems

    old_ids = {row[0] for row in before}
    rows = report["rows"]
    added = [row for row in rows if row[0] not in old_ids]
    heads = {row[0] for row in report["summary_heads"] if row[0] not in old_ids}
    ordinary = [row[1:] for row in added if row[0] not in heads]
    seen = {tuple(row[1:]) for row in before}
    duplicates = 0
    for row in map(tuple, ordinary):
        duplicates += row in seen
        seen.add(row)
    report.update(rows_before=len(before), rows_added=len(added), heads_added=len(heads),
                  duplicates_added=duplicates)

    if [row for row in rows if row[0] in old_ids] != before:
        problems.append("lost, reordered, or changed pre-existing rows")
    if [row for row in ordinary if row in expected] != expected:
        problems.append("expected one new turn and reply in order")
    return report, problems


@pytest.mark.parametrize("uid_mode", ["unset", "on"])
@pytest.mark.parametrize("shape", ["refused-only", "refused-then-adopted"])
def test_refused_snapshot_rollback(tmp_path, old_tree, host_root, monkeypatch, uid_mode, shape):
    _set_uid_mode(uid_mode, monkeypatch)
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    control.mkdir()
    candidate.mkdir()
    control_seed = run_old(old_tree("v0.30.3-rc1", PINS), control, host_root, OLD_SEED, mode=uid_mode, data=shape)
    candidate_seed = _seed_store(candidate, shape, monkeypatch, expect_refusal_mark=True)

    problems = []
    for reader in PINS:
        old = old_tree(reader, PINS)
        outcomes = {}
        for original, (kept, before) in ((control, control_seed), (candidate, candidate_seed)):
            arm = original.name
            report, errors = _measure_rollback(
                old, original, tmp_path / f"{arm}-{reader}", host_root, uid_mode, kept, before, monkeypatch,
            )
            report.update(arm=arm, reader=reader, shape=shape, uid_mode=uid_mode)
            print(json.dumps(report, default=lambda value: value.decode(errors="replace")))
            problems.extend(f"{reader} {arm}: {error}" for error in errors)
            outcomes[arm] = report
        for field in ROLLBACK_FIELDS:
            base = outcomes["control"].get(field)
            changed = outcomes["candidate"].get(field)
            if base != changed:
                problems.append(f"{reader}: {field} control={base!r}, candidate={changed!r}")
    assert not problems, "\n".join(problems)
