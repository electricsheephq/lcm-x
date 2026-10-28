"""Recompute coverage verdict and zero-progress timer from immutable progress rows."""
import json
from pathlib import Path
from .catchup import state
from .common import REFERENCE_CALIBRATION_S, load, rollup


def verdict(path, max_wall):
    terminal = path.with_name("summary.json")
    if terminal.exists() and (recorded := load(terminal)).get("state") == "ABORTED" and recorded.get("reason"):
        return "ABORTED"
    if path.with_name("cross-profile.json").exists() and load(path.with_name("cross-profile.json")).get("paths"):
        return "ABORTED"
    previous, last_change, generation, final = None, 0., 0, "RUNNING"
    for row in map(json.loads, path.read_text().splitlines()):
        fingerprint = str(([v["covered"] for v in row["audit"]["classes"].values()], row["audit"]["scan"]))
        if fingerprint != previous or row.get("generation", 0) != generation:
            last_change, previous, generation = row["elapsed_s"], fingerprint, row.get("generation", 0)
        final = state(row["audit"], row["elapsed_s"], row["elapsed_s"]-last_change, row["ready"], max_wall)
    return final


def checked(recorded, computed, parent=None):
    sources = [recorded] if parent is None else [parent, recorded]
    disagreement = {k: dict(recorded=r.get(k), recomputed=v)
                    for r in sources for k, v in computed.items() if k not in r or r[k] != v}
    if disagreement:
        return dict(recorded | (parent or {}), state=rollup([r.get("state", "INVALID") for r in sources]+[computed["state"], "INVALID"]),
                    rescore_disagreement=disagreement)
    return computed


def v2(root, *, _entry=True):
    from .floor import rate
    from .recall import score, window_calibration, overlapping, guarded_result
    from .measurement import catchup_activity
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()] if path.is_file() else []
    def coverage(cell):
        raw = load(cell/"catchup/summary.json")
        seconds = raw.get("end_to_end_s")
        if seconds is None:
            seconds = max(r["elapsed_s"] for r in lines(cell/"catchup/progress.jsonl"))
        return rate(raw, seconds, load(cell/"manifest.json")["rows"], lines(cell/"catchup/events.jsonl"))
    if not (root/"summary.json").is_file() and not (root/"raw-summary.json").is_file():
        raise ValueError(f"missing root summary.json: {root}")
    recorded = load(root/"summary.json" if (root/"summary.json").exists() else root/"raw-summary.json")
    parent = recorded
    aborted = recorded if recorded.get("state") == "ABORTED" else None
    if aborted is not None:
        assert _entry, "ABORTED is only valid at the entry root"
        complete = []
        for path in root.rglob("raw-summary.json"):
            cell = path.parent
            raw = load(path)
            if "classes" not in raw and raw.get("state") in ("INVALID", "FAIL"):
                guarded = guarded_result(cell, raw)
                if guarded is not None:
                    complete.append(guarded if cell == root else v2(cell, _entry=False))
                    continue
            if all((cell/name).is_file() for name in (
                    "request.json", "recalls.jsonl", "final-audit.json")) and cell != root:
                complete.append(v2(cell, _entry=False))
        for path in root.rglob("catchup/summary.json"):
            cell = path.parent.parent
            if (cell/"manifest.json").is_file() and (cell/"catchup/events.jsonl").is_file():
                if cell == root or not (cell/"summary.json").is_file():
                    try:
                        complete.append(coverage(cell))
                    except (ValueError, KeyError):
                        continue  # Interrupted raw evidence cannot prove FAIL.
                else:
                    complete.append(v2(cell, _entry=False))
        if any(c["state"] == "FAIL" for c in complete):
            return dict(recorded, state="FAIL")
        if not all((root/name).is_file() for name in (
                "raw-summary.json", "request.json", "recalls.jsonl", "final-audit.json")):
            return recorded
        # A direct recall entry falls through to its own raw receipt scoring.
    raw_summary = load(root/"raw-summary.json") if (root/"raw-summary.json").exists() else {}
    if recorded == dict(state="PASS") and not raw_summary and not (root/"catchup").is_dir():
        return checked(recorded, dict(state="INVALID", reason="missing_receipts"))
    events = lines(root/"events.jsonl") if (root/"events.jsonl").is_file() else []
    if raw_summary.get("state", recorded.get("state")) == recorded.get("state") in ("INVALID", "FAIL"):
        reason = raw_summary.get("reason", recorded.get("reason"))
        same_reason = recorded.get("reason") == reason
        if (recorded["state"] == "INVALID" and same_reason and reason == "covered_store_incomplete" and root.name.endswith("-ON-covered")
                and not (root/"recalls.jsonl").exists() and "classes" not in raw_summary and "classes" not in recorded):
            return recorded
        if same_reason and "classes" not in raw_summary and "classes" not in recorded:
            guarded = guarded_result(root, recorded)
            if guarded is not None:
                return guarded
        if recorded["state"] == "INVALID" and same_reason and reason == "volume_cap" and any(e.get("event") == "volume_cap" for e in events):
            if not (root/"recalls.jsonl").exists() or not lines(root/"recalls.jsonl"):
                assert raw_summary.get("measurement_valid") is False
                return recorded
    if (root/"summary.json").exists() and "cells" in load(root/"summary.json"):
        recorded = load(root/"summary.json")
        present = {p.name for p in root.iterdir() if p.is_dir() and p.name in ("empty", "upgrade")}
        names = [c["baseline"] for c in recorded["cells"]]
        assert names and len(names) == len(set(names)) and set(names) == present == {"empty", "upgrade"}
        cells = [v2(root/name, _entry=False) for name in names]
        return dict(checked(dict(state=recorded["state"]), dict(state=rollup(c["state"] for c in cells))), cells=cells)
    if (root/"summary.json").exists() and "blocks" in load(root/"summary.json"):
        recorded = load(root/"summary.json")
        base = root if (root/"0-OFF").exists() else root/"recall" if (root/"recall").is_dir() else root
        present = {p.name for p in base.iterdir() if p.is_dir() and any(p.name.endswith("-"+s) for s in ("OFF", "ON-catching-up", "ON-covered"))}
        for name in present:
            if (base/name/"request.json").is_file():
                request = load(base/name/"request.json")
                assert (request["block"], request["state"]) == (int(name.split("-", 1)[0]), name.split("-", 1)[1]), "request identity mismatch"
        cells = {name: v2(base/name, _entry=False) for name in sorted(present)}
        names = [f"{c['block']}-{c['state']}" for c in recorded["blocks"]]
        assert names and len(names) == len(set(names)) and set(names) == present
        assert all({f"{b}-{s}" for s in ("OFF", "ON-catching-up", "ON-covered")} <= present
                   for b in {name.split("-", 1)[0] for name in present})
        cells = [cells[name] for name in names]
        for entry, cell in zip(recorded["blocks"], cells):
            verdict = checked(dict(state=entry["result"]["state"]), dict(state=cell["state"]))
            if "rescore_disagreement" in verdict:
                verdict["rescore_disagreement"] = cell.get("rescore_disagreement", {}) | verdict["rescore_disagreement"]
            cell.update(verdict)
        return dict(checked(dict(state=recorded["state"]), dict(state=rollup(c["state"] for c in cells))), blocks=cells)
    if (root/"catchup").is_dir():
        recorded = load(root/"summary.json")
        computed = coverage(root)
    else:
        assert (root/"recalls.jsonl").is_file(), "recall receipts or event-backed early INVALID required"
        recorded = load(root/"raw-summary.json")
        request = load(root/"request.json")
        calibrated = window_calibration(lines(root.parent/f"{request['block']}-ON-catching-up/events.jsonl"))
        rows = lines(root/"recalls.jsonl")
        pubs = lines(root/"publications.jsonl") if (root/"publications.jsonl").is_file() else []
        counted = overlapping(rows, pubs)
        activity = catchup_activity(root, min(r["start"] for r in counted), max(r["end"] for r in counted),
            min(counted, key=lambda r: r["start"])["coverage_start"],
            max(counted, key=lambda r: r["end"])["coverage_end"]) if counted else dict(catchup_activity=False)
        raw = score(rows, pubs, request["seed"], request["state"], activity)
        final = load(root/"final-audit.json")
        invalid = (request["smoke"] or request["rows"] < 250000 or len(load(root.parent/"queries.json")) < 200
            or (request["state"] == "ON-catching-up" and (all(c["missing_count"] == 0 for c in final["classes"].values())
                or not all(r["incomplete"] for r in counted))))
        capped = any(e.get("event") == "volume_cap" for e in lines(root/"events.jsonl")) if (root/"events.jsonl").exists() else False
        if invalid or capped:
            raw["state"] = rollup([raw["state"], "INVALID"])
        valid = raw["state"] != "INVALID"
        computed = score(rows, pubs, request["seed"], request["state"], activity, calibrated, REFERENCE_CALIBRATION_S)
        computed.update(measurement_valid=valid, catchup_activity=activity["catchup_activity"])
        if capped:
            computed["reason"] = "volume_cap"
        if not valid:
            computed["state"] = rollup([computed["state"], "INVALID"])
        if calibrated is None and valid:
            computed["reason"] = "calibration_window_short"
    computed = checked(recorded, computed, parent if aborted is None else None)
    if "rescore_disagreement" in computed:
        return computed
    return dict(aborted, state="FAIL") if aborted is not None and computed["state"] == "FAIL" else aborted or computed


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        root = Path(name)
        if not (root/"summary.json").is_file() and not (root/"raw-summary.json").is_file():
            print(f"missing root summary.json: {root}", file=sys.stderr)
            raise SystemExit(2)
        if (root/"catchup").is_dir() or (root/"raw-summary.json").exists() or any(
                k in load(root/"summary.json") for k in ("cells", "blocks")) or load(root/"summary.json").get("state") in ("INVALID", "ABORTED"):
            print(json.dumps(v2(root)))
            continue
        summary, audit = load(root/"summary.json"), load(root/"final-audit.json")
        final = verdict(root/"progress.jsonl", summary["max_wall"])
        coverage = sum(v["covered"] for v in audit["classes"].values())/sum(v["expected"] for v in audit["classes"].values())
        assert final == summary["state"] and coverage == summary["coverage_fraction"]
        print(json.dumps({"cell": root.name, "state": final, "coverage_fraction": coverage}))
