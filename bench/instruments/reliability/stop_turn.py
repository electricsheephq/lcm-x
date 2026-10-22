"""#1623: real after-tool cancellation, immutable row-shape reads, strict scoring."""
import hashlib
import json
import sqlite3
import threading


def shape(messages):
    def sha(value):
        return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode()).hexdigest()
    out = []
    for row in messages:
        calls = row.get("tool_calls") or []
        calls = json.loads(calls) if isinstance(calls, str) else calls
        out.append({"role": row["role"], "content_sha256": sha(row.get("content")),
                    "tool_call_id": row.get("tool_call_id"), "calls": [
                        {"id": c.get("id"), "name": (c.get("function") or c).get("name"),
                         "arguments_sha256": sha((c.get("function") or c).get("arguments"))} for c in calls]})
    return out


def snapshot(run, label):
    result = {"label": label}
    for name in ("lcm", "state"):
        with sqlite3.connect(f"file:{run.home / (name + '.db')}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            order = "store_id" if name == "lcm" else "id"
            rows = [dict(r) for r in con.execute(f"SELECT * FROM messages ORDER BY {order}")]
        result[name] = {"rows": shape(rows), "owners_match": all(r["session_id"] == run.sid for r in rows)}
    (run.d / f"stop-{label}.json").write_text(json.dumps(result, indent=1))


def phase(run, first):
    from . import acp_driver as AD
    from .process_cell import graceful_exit
    run.first = first
    run.proc = AD.AcpProcess(run.argv(), run.env(), run.work, run.d / f"host-{run.phase}.stderr")
    done, worker = threading.Event(), None
    try:
        run.proc.initialize(run.budget())
        if run.phase == "A":
            run.sid = run.proc.new_session(run.files, run.budget())
            run.turn(0)  # A completed prefix proves ownership before the stopped turn.
            snapshot(run, "warmup")
            def cancel():
                while not done.wait(.01):
                    if run.notes("stop_tool_ready"):
                        run.cancel()
                        return
            worker = threading.Thread(target=cancel, daemon=True)
            worker.start()
            try:
                run.turn(1, "cancel")
            except AD.JsonRpcError as exc:
                ends = run.notes("turn_end")
                if not ends or not ends[-1].get("interrupted") or ends[-1].get("reason") != "interrupted_by_user":
                    raise
                run.event(event="interrupted_response_error", error_type=type(exc).__name__)
            snapshot(run, "immediate")
        else:
            run.proc.load_session(run.sid, run.files, run.budget())
            run.turn(2)
            snapshot(run, "catchup")
        return {"exit": "clean_exit" if run.phase == "A" else "done", "next_turn": 2}
    except (AD.DriverError, OSError, ValueError, sqlite3.Error) as exc:
        return {"exit": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        done.set()
        if worker:
            worker.join(2)
        rc = run.proc.close()
        run.event(event="host_exit", returncode=rc, killed=run.proc.killed)
        if not graceful_exit(rc, run.proc.killed, run.proc.sent_term):
            raise AD.DriverError("stop cell host did not exit gracefully")
        if run.phase == "A" and (run.d / "stop-immediate.json").exists():
            snapshot(run, "end")


def score(run, last):
    from .process_cell import read_jsonl
    ends = [n for n in run.notes("turn_end") if n.get("turn_kind") == "cancel"]
    ready, observed = run.notes("stop_tool_ready"), run.notes("stop_interrupt_observed")
    dispatch = [e for e in read_jsonl(run.transcript) if e.get("event") == "tool_dispatch"]
    stopped_requests = [r for r in read_jsonl(run.d / "provider-requests.jsonl")
                        if r.get("role") == "main" and r.get("turn") == 1]
    if (last["exit"] != "done" or not ends or not ends[-1].get("interrupted")
            or ends[-1].get("reason") != "interrupted_by_user" or not ready or not observed
            or not observed[-1].get("interrupted") or len(dispatch) != 1 or not dispatch[0].get("ok")
            or len(stopped_requests) != 1 or stopped_requests[0].get("step") != 0):
        return {"verdict": "ERROR", "reason": "successful tool / real interrupted_by_user / catch-up not confirmed"}
    reads = {k: json.loads((run.d / f"stop-{k}.json").read_text()) for k in ("immediate", "end", "catchup")}
    warm_prefix = True
    if (run.d / "stop-warmup.json").exists():
        warm = json.loads((run.d / "stop-warmup.json").read_text())
        count = len(warm["state"]["rows"])
        warm_prefix = (count == 2 and warm["lcm"]["rows"] == warm["state"]["rows"]
                       and all(warm[db]["owners_match"] and v[db]["rows"][:count] == warm[db]["rows"]
                               for v in reads.values() for db in ("lcm", "state")))
        for v in reads.values():
            for db in ("lcm", "state"):
                v[db]["rows"] = v[db]["rows"][count:]
    expected = reads["immediate"]["state"]["rows"]
    once = [reads["catchup"]["lcm"]["rows"].count(r) for r in expected]
    checks = {"warm_prefix": warm_prefix, "complete_tool_group": len(expected) == 4 and [r["role"] for r in expected] == [
                  "user", "assistant", "tool", "assistant"] and bool(expected[1]["calls"])
                  and expected[1]["calls"][0]["id"] == expected[2]["tool_call_id"],
              "bound_owner": all(v[db]["owners_match"] for v in reads.values() for db in ("lcm", "state")),
              "immediate": reads["immediate"]["lcm"]["rows"] == expected,
              "end": reads["end"]["lcm"]["rows"] == expected == reads["end"]["state"]["rows"],
              "catchup_order": reads["catchup"]["lcm"]["rows"][:len(expected)] == expected,
              "once": once == [1] * len(expected)}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "stop_checks": checks,
            "stopped_rows": len(expected), "immediate_rows": len(reads["immediate"]["lcm"]["rows"]),
            "catchup_occurrences": once, "reason": "stopped-turn persistence"}
