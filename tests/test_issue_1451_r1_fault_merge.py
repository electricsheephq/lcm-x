"""#1451 r1: publication faults preserve positional proof through host merge."""
import itertools
import sqlite3
from _pytest.monkeypatch import MonkeyPatch
import pytest
import hermes_lcm.engine as E
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _turn, _host_merge_consecutive_users, _stub_summarizer
NEW = "NEW-0 typed after the compaction"
PREFIX = {"proof": "compaction_commit_proof:", "snapshot": "host_occurrence_snapshot:"}

def run(tmp, mode, tail, site, kind):
    mp = MonkeyPatch()
    mp.setattr(E, "summarize_with_escalation", _stub_summarizer())
    db = f"{tmp}/lcm.db"
    engine = LCMEngine(config=LCMConfig(database_path=db, fresh_tail_count=tail, leaf_chunk_tokens=400,
                                        large_output_externalization_path=f"{tmp}/ext"), hermes_home=f"{tmp}/home")
    real = engine._store.write_metadata_json
    arm = {"on": False, "fired": False}

    def write(keys, *a, **kw):
        out = real(keys, *a, **kw)
        if arm["on"] and not arm["fired"] and any(k.startswith(PREFIX[kind]) for k in keys):
            arm["fired"] = True
            raise sqlite3.OperationalError("injected occurrence write failure")
        return out

    engine._store.write_metadata_json = write
    host, err = [], None
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        for i in range(1, 13):
            for m in _turn(i):
                host.append(m)
                engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        arm["on"] = site == "compress"
        compressed = engine.compress(list(host), force=True)
        arm["on"] = False
        sid = "S0"
        if mode != "noboundary":
            engine.on_session_end("S0", pre)
            sid = "S0" if mode == "inplace" else "S1"
            arm["on"] = site == "boundary"
            engine.on_session_start(sid, boundary_reason="compression", old_session_id="S0", platform="acp")
            arm["on"] = False
        proof_after = engine._compress_commit_proof is not None
        host = _host_merge_consecutive_users(list(compressed) + [{"role": "user", "content": NEW}])
        engine.ingest(host)
        host.append({"role": "assistant", "content": "reply to NEW-0."})
        engine.ingest(host)
        for i in (14, 15, 16, 17, 18, 19):
            for m in _turn(i):
                host.append(m)
                engine.ingest(host)
        engine.compress(list(host), force=True)
        status2 = engine._last_compression_status
    except Exception as exc:
        err, status2, proof_after = f"{type(exc).__name__}: {exc}"[:120], None, None
    finally:
        engine.shutdown()
        mp.undo()
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    text = "\n".join(c or "" for _r, c in rows)
    turns = {f"{i:02d}" for i in list(range(1, 14)) + list(range(14, 20))}
    missing = sorted(t for t in turns if f"[T{t}] user" not in text)
    return {"mode": mode, "tail": tail, "site": site, "kind": kind, "fired": arm["fired"],
            "proof_after_boundary": proof_after, "new_rows": sum(NEW in (c or "") for _r, c in rows),
            "missing_turns": missing, "rows": len(rows), "dups": len(rows) - len(set(rows)),
            "status2": status2, "error": err}

CASES = [(mode, tail, site, kind) for mode, tail, (site, kind) in itertools.product(
    ["inplace", "rotation", "noboundary"], [6, 7],
    [("none", "proof"), ("compress", "proof"), ("compress", "snapshot"),
     ("boundary", "proof"), ("boundary", "snapshot")])
    if not (mode == "noboundary" and site == "boundary")]
@pytest.mark.parametrize("mode,tail,site,kind", CASES)
def test_fault_then_host_merge_loses_no_turn(tmp_path, mode, tail, site, kind):
    result = run(tmp_path, mode, tail, site, kind)
    assert result["error"] is None, result
    assert not result["missing_turns"], result
    assert result["new_rows"] == 1, result
