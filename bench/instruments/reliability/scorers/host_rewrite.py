"""B9: host flush/commit identity invariants; payload-free R1 audit evidence."""
import json


def score(cell, cell_dir, phases):
    if cell.get("transport") not in (None, "acp", "gateway") or not phases or any(
            not p.get("p8", {}).get("supported") for p in phases):
        return {"verdict": "UNSUPPORTED", "reason": "in-process host flush seams unavailable or audit disabled"}
    path = cell_dir / "p8-events.jsonl"
    if not path.exists() or any(p["p8"].get("notes") for p in phases):
        return {"verdict": "UNSUPPORTED", "reason": "incomplete audit; see phase harness notes"}
    failed, reported, actions = {}, [], {}

    def fail(name, ids):
        entry = failed.setdefault(name, {"count": 0, "row_ids": []})
        entry["count"] += 1
        entry["row_ids"].extend(i for i in ids if i is not None)

    def duplicates(items):
        for item in items:
            if item["lcm"]:
                fail("I5", item["row_ids"])
            else:
                reported.append(item)

    try:
        events = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError):  # a truncated line from a killed process: the audit proves nothing
        return {"verdict": "UNSUPPORTED", "reason": "unreadable audit log"}
    if not any(isinstance(ev, dict) and ev.get("event") == "commit" for ev in events):
        return {"verdict": "UNSUPPORTED", "reason": "no committed compaction observed"}
    for ev in events:
        ids = [ev.get("target_id"), ev.get("row_id")]
        if ev["event"] == "commit":
            if (ev["active"] != 1 or ev["role"] != ev["target_role"] or ev["uid"] != ev["target_uid"]
                    or not ev["expected"] or ev["expected"] != ev["after"]):
                fail("I0", ids)
        elif ev["event"] == "flush_resolve":
            action = ev["action"]
            actions[action] = actions.get(action, 0) + 1
            if ev["active"] == 0 and (action in ("REWRITE", "ADOPT") or (action == "LEGACY" and ev.get("effect"))):
                fail("I1", ids)
            if ev["target_id"] is not None and (ev["role"] != ev["target_role"] or
                    (ev["uid"] and ev["uid"] != ev["target_uid"]) or
                    (ev["path"] == "uid_snapshot" and ev["active_count"] != 1)):
                fail("I2", ids)
            if action == "ADOPT":
                fail("I3", ids)
        elif ev["event"] == "sweep":
            duplicates(ev["duplicates"])
    for p in phases:
        duplicates(p["p8"].get("duplicates", []))
    return {"verdict": "FAIL" if failed else "PASS", "failed_invariants": failed,
            "reported_duplicates": reported, "actions": actions}
