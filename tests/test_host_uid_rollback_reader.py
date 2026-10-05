"""Persisted SHADOW emissions remain inert to pre-uid readers after rollback."""

import copy
import io
import json
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from hermes_lcm import host_uid_emit as emit
from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.host_uid_emit import IDENTITY_KEYS
from tests.test_host_uid_emit_carry import _call
from tests.test_host_uid_engine_uids import _compress, _host
from tests.test_host_uid_shadow import ROLLBACK_READERS, _engine, _m, _skip_or_fail_in_ci, _state_db


_OLD_READER = r"""
import importlib.util, json, sys
old, db, root, host_file = sys.argv[1:]
sys.path.insert(0, root)  # the git-ignored agent/ host stub
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import store as store_mod
from hermes_lcm.command import _doctor_text
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
assert store_mod.__file__.startswith(old) and importlib.util.find_spec("hermes_lcm.host_uid") is None
with open(host_file) as stream:
    host = json.load(stream)
engine = LCMEngine(config=LCMConfig(database_path=db))
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    before = len(engine._store.get_session_messages("S"))
    engine.ingest(host)
    first = engine._store.get_session_messages("S")
    engine.ingest(host)
    replay = engine._store.get_session_messages("S")
    integrity = next(line for line in _doctor_text(engine).splitlines() if line.startswith("sqlite_integrity:"))
    print(json.dumps({
        "rows": [{"role": row["role"], "content": row["content"], "order": i}
                 for i, row in enumerate(replay)],
        "before": before, "first_added": len(first) - before,
        "replay_added": len(replay) - len(first), "sqlite_integrity": integrity,
    }, sort_keys=True))
finally:
    engine.shutdown()
"""


def _without_uids(messages):
    return [{key: value for key, value in row.items() if key not in IDENTITY_KEYS} for row in messages]


@pytest.mark.parametrize("reader", sorted(ROLLBACK_READERS))
def test_persisted_shadow_uids_are_inert_to_rollback_reader(tmp_path, monkeypatch, reader):
    root = Path(__file__).resolve().parents[1]
    commit = ROLLBACK_READERS[reader]
    try:
        import agent.context_engine  # noqa: F401
    except ImportError:
        _skip_or_fail_in_ci("the agent.context_engine host stub is not importable in this checkout")
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        _skip_or_fail_in_ci(f"the {reader} commit {commit[:8]} is not in this checkout (shallow clone)")
    archive = subprocess.run(["git", "-C", str(root), "archive", commit], check=True, capture_output=True).stdout
    old = tmp_path / "old-reader"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    assert not (old / "host_uid.py").exists()

    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)
    assert host_message_uid_mode() == "shadow"
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    engine._hermes_home = str(tmp_path)
    engine._config.fresh_tail_count = 4
    host = _host()
    host[-1].update(tool_calls=[_call("rollback-call")],
                    _tool_call_uids={"rollback-call": ["host-call"]})
    host.append(_m("tool", "rollback tool result", 27.0, "host-result",
                   tool_call_id="rollback-call", _tool_call_uid="host-result"))
    try:
        emitted = _compress(engine, host)
        engine_uids = {row[0] for row in engine._store._conn.execute(
            "SELECT uid FROM host_uid_bindings WHERE kind = 'engine'")}
        minted = [row for row in emitted if row.get("message_uid") in engine_uids]
        carried = [row for row in emitted if row.get("message_uid") in {m["message_uid"] for m in host}]
        assert any(engine._verified_lcm_summary_prefix_end(row["content"]) for row in minted), emitted
        assert carried, emitted
        assert any("_tool_call_uids" in row or "_tool_call_uid" in row for row in emitted), emitted
        before_counts = Counter(row["content"] for row in engine._store.get_session_messages("S"))
    finally:
        engine.shutdown()

    new_content = "new user turn after rollback"
    persisted = copy.deepcopy(emitted) + [_m("user", new_content, 99.0, "host-after-rollback")]
    control = _without_uids(persisted)
    results = []
    for name, messages in (("uid", persisted), ("control", control)):
        db = tmp_path / f"{name}.db"
        shutil.copy2(tmp_path / "lcm.db", db)
        host_file = tmp_path / f"{name}.json"
        host_file.write_text(json.dumps(messages))
        result = subprocess.run(
            [sys.executable, "-B", "-c", _OLD_READER, str(old), str(db), str(root), str(host_file)],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
        assert result.returncode == 0, result.stderr[-3000:]
        results.append(json.loads(result.stdout))

    uid, plain = results
    assert uid["sqlite_integrity"] == plain["sqlite_integrity"] == "sqlite_integrity: ok"
    assert uid["rows"] == plain["rows"], reader
    uid_counts = Counter(row["content"] for row in uid["rows"])
    plain_counts = Counter(row["content"] for row in plain["rows"])
    assert all(count <= plain_counts[content] for content, count in uid_counts.items())
    assert all(count <= before_counts[content] + (content == new_content)
               for content, count in uid_counts.items()), uid_counts - before_counts
    assert uid_counts[new_content] == plain_counts[new_content] == 1
    assert uid["replay_added"] == plain["replay_added"] == 0
    print(f"ROLLBACK_READER_OK {reader} minted={len(minted)} carried={len(carried)} "
          f"stored={len(uid['rows'])} before={uid['before']} first_added={uid['first_added']} replay_added=0")
