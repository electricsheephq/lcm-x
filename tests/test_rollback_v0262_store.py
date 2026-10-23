"""Plugin-only rollback to v0.26.2 preserves stem-index writes, summaries, and inactive markers."""
import io
import json
import select
import subprocess
import sys
import tarfile
from pathlib import Path
from hermes_lcm import db_bootstrap as boot, inactive_record, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_engine_uids import _compress, _host
from tests.test_host_uid_shadow import ROLLBACK_READERS, _skip_or_fail_in_ci, _state_db
from tests.test_stemmed_recall_index import consistent, finish, partial
_OLD_READER = r'''
import importlib.util, json, sys
from pathlib import Path
old, db, home, root = sys.argv[1:]
sys.path.insert(0, root)  # the git-ignored agent/ host stub, as the other rollback readers do
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import inactive_record, store as store_mod, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
assert store_mod.__file__.startswith(old) and tools.__file__.startswith(old)
assert inactive_record.__file__.startswith(old)
engine = LCMEngine(config=LCMConfig(database_path=db, embeddings_enabled=False,
                                  temporal_rollups_enabled=False), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    store = engine._store
    before = store.get_session_messages("S")
    engine.ingest([{"role": "user", "content": "Rollback deployments succeeded", "timestamp": 99.0}])
    after = store.get_session_messages("S")
    inserted = [row for row in after if row["store_id"] not in {r["store_id"] for r in before}]
    assert len(inserted) == 1 and inserted[0]["content"] == "Rollback deployments succeeded"
    updated_id = store.append("S", {"role": "tool", "content": "obsolete rollback payload"})
    assert store.gc_externalized_tool_result(updated_id, "deployments updated")
    deleted_id = store.append("disposable", {"role": "tool", "content": "rollback disposable"})
    assert store.delete_session_messages("disposable") == 1
    args = {"query": "deployments", "session_scope": "all", "mode": "full_text"}
    grep = json.loads(tools.lcm_grep(args, engine=engine))
    recall = json.loads(tools.lcm_recall(args, engine=engine))
    assert inserted[0]["store_id"] in {hit["store_id"] for hit in grep["results"]}
    assert inserted[0]["store_id"] in {hit.get("store_id") for hit in recall["hits"]}
    summaries = [node.summary for node in engine._dag.get_session_nodes("S")]
    assert summaries
    assert inactive_record.inactive_record_notice(home) is None
    inactive_record.write_inactive_record(home, elapsed_s=0.1, reason="old-reader diagnostic")
    legacy = next(Path(home).glob("lcm-x-not-active.*.json"))
    print(json.dumps({"inserted": len(inserted), "updated": 1, "deleted": 1,
                      "ids": [inserted[0]["store_id"], updated_id], "deleted_id": deleted_id,
                      "summaries": summaries, "legacy": str(legacy),
                      "grep": len(grep["results"]), "recall": len(recall["hits"])}), flush=True)
    assert sys.stdin.readline().strip() == "finish"
finally:
    engine.shutdown()
'''
def test_v0262_writes_main_store_and_rolls_forward(tmp_path, monkeypatch, caplog):
    root = Path(__file__).resolve().parents[1]
    commit = ROLLBACK_READERS["v0.26.2"]
    try:
        import agent.context_engine  # noqa: F401
    except ImportError:
        _skip_or_fail_in_ci("the agent.context_engine host stub is not importable")
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                      capture_output=True).returncode:
        _skip_or_fail_in_ci(f"the v0.26.2 commit {commit} is not available")
    archive = subprocess.run(["git", "-C", str(root), "archive", commit],
                             check=True, capture_output=True).stdout
    old = tmp_path / "old-reader"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    monkeypatch.setattr(boot, "dispatch_background_stem_backfill", lambda *a: None)
    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation",
                        lambda **kwargs: ("Synthetic compacted history.", 1))
    _state_db(tmp_path, [("S", None, None)])
    db = tmp_path / "lcm.db"
    config = LCMConfig(database_path=str(db), embeddings_enabled=False, fresh_tail_count=4,
                       leaf_chunk_tokens=1, temporal_rollups_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        _compress(engine, _host())
        summaries = [node.summary for node in engine._dag.get_session_nodes("S")]
        assert summaries, "Main must compact messages into persisted summary nodes"
        conn = partial(engine._store)
        finish(engine._store)
        consistent(conn)
        before = conn.execute("SELECT store_id, content FROM messages ORDER BY store_id").fetchall()
        inactive_record.write_inactive_record(tmp_path, elapsed_s=0.1, reason="main diagnostic")
        marker = next(tmp_path.joinpath(*inactive_record.RECORD_DIR).glob("lcm-x-not-active.*.json"))
        marker_bytes = marker.read_bytes()
    finally:
        engine.shutdown()
    with subprocess.Popen([sys.executable, "-B", "-c", _OLD_READER, str(old), str(db), str(tmp_path), str(root)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, cwd=tmp_path) as process:
        try:
            assert select.select([process.stdout], [], [], 60)[0], "Old reader did not report in 60s"
            line = process.stdout.readline()
            if not line:
                stdout, stderr = process.communicate(timeout=60)
                assert process.returncode == 0, stderr
                raise AssertionError(f"Old reader produced no JSON: {stdout}")
            report = json.loads(line)
            assert "old-reader diagnostic" in inactive_record.inactive_record_notice(tmp_path)
            assert Path(report["legacy"]).exists(), "Live legacy records must be read, not cleaned"
            stdout, stderr = process.communicate("finish\n", timeout=60)
            assert process.returncode == 0, stderr
            assert not stdout, stdout
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    assert report["inserted"] == report["updated"] == report["deleted"] == 1
    assert report["grep"] >= 1 and report["recall"] >= 1
    assert report["summaries"] == summaries
    assert marker.read_bytes() == marker_bytes
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        conn = engine._store.connection
        if boot._stem_metadata(conn, "fts_stem_state") == "backfilling":
            finish(engine._store)
        consistent(conn)
        rows = dict(conn.execute("SELECT store_id, content FROM messages").fetchall())
        assert all(rows[store_id] == content for store_id, content in before)
        assert [rows[store_id] for store_id in report["ids"]] == [
            "Rollback deployments succeeded", "deployments updated"]
        assert report["deleted_id"] not in rows
        indexed = {row[0] for row in conn.execute(
            "SELECT rowid FROM messages_fts_stem WHERE messages_fts_stem MATCH 'deployment'")}
        assert set(report["ids"]) <= indexed
        recall = json.loads(tools.lcm_recall({"query": "deployment"}, engine=engine))
        assert recall["provenance"]["fts_index"] == "stem"
        assert set(report["ids"]) <= {hit.get("store_id") for hit in recall["hits"]}
        assert [node.summary for node in engine._dag.get_session_nodes("S")] == summaries
        assert "main diagnostic" in inactive_record.inactive_record_notice(tmp_path)
        assert not Path(report["legacy"]).exists(), "Main must clean the exited old reader's root record"
        assert marker.read_bytes() == marker_bytes
    finally:
        engine.shutdown()
    assert not any(text in (stderr + caplog.text).lower()
                   for text in ("no such tokenizer", "no such function", "malformed"))
    print("ROLLBACK_READER_OK v0.26.2", json.dumps(report, sort_keys=True))
