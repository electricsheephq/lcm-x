"""Regression for PR 1689 admission."""
import json
import sqlite3
import sys
from types import SimpleNamespace
from .bot1_support import S2, function


def test_admission_uses_checkpoint_version_and_source(tmp_path):
    sys.path.insert(0, str(S2))
    from s2lib import deep
    old = dict(id="F1", value="earlier", row_id="r0", row_index=0)
    final = dict(old, value="later", row_id="r2", row_index=2)
    (tmp_path / "facts.json").write_text(json.dumps([final]))
    (tmp_path / "deep_metadata.json").write_text(json.dumps(dict(checkpoint_answers={"1": [old]})))
    db = tmp_path / "store.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE messages(store_id, role, content, session_id)")
        c.execute("CREATE TABLE summary_nodes(node_id, depth, source_type, source_ids, session_id)")
    run = SimpleNamespace(rows=[dict(id="r0", role="user", content="earlier"), dict(id="r1", role="assistant", content="ok")],
        home=tmp_path, db=db, sid="test", sdir=tmp_path, man={}, ntok=len)
    ns = dict(json=json, jload=lambda p: json.loads(p.read_text()), db_ro=lambda p: sqlite3.connect(p), deep=deep, to_msg=lambda r: r)
    result = function(S2 / "run_s_lcmx.py", "receipts", ns, "Run")(run, upto=1)
    assert result["facts_planted_in_slice"] == 1
    assert result["facts_not_admitted"] == ["F1"]
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO messages VALUES (1, 'user', 'earlier', 'test')")
    assert function(S2 / "run_s_lcmx.py", "receipts", ns, "Run")(run, upto=1)["facts_admitted"] == 1
