#!/usr/bin/env python3
"""Offline C1 deep aggregate from S2 checkpoint JSON, exact material and wall receipts.

Engine-direct synthetic replay only; wait is compress wall, not host latency.
Missing evidence remains INCOMPLETE. An observed #659 miss makes the arm FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scorer"))
from loss_class import classify_loss  # noqa: E402
from pmw_read import engine_turn_wait, mean, p90  # noqa: E402
from score_s import accounting, jlines, jload, score  # noqa: E402
from score_decision import runs  # noqa: E402

KINDS = ("current_request", "constraint_quote", "constraint_obey", "host_instruction")


def continuity_events(summary):
    events, failed, missing = [], False, False
    for event in summary.get("events", []):
        if not event.get("is_compaction"):
            continue
        probes = event.get("deep_probes", [])
        kinds, evidence = {}, {}
        for kind in KINDS:
            found = [p for p in probes if p.get("kind") == kind]
            flags = {field: found[0].get(field) if len(found) == 1 else None
                     for field in ("passed", "present_verbatim", "reader_pass")}
            value = False if False in flags.values() else True if all(v is True for v in flags.values()) else None
            evidence[kind] = flags
            kinds[kind] = value
            failed |= value is False
            missing |= value is None
        events.append({"event": event.get("event"), "row_index": event.get("row_index"),
                       "kinds": kinds, "evidence": evidence, "result": "FAIL" if False in kinds.values() else
                       "INCOMPLETE" if None in kinds.values() else "PASS"})
    return {"result": "FAIL" if failed else "INCOMPLETE" if missing or not events else "PASS", "events": events}


def total_present(records, field):
    return sum(r[field] for r in records) if records and all(r.get(field) is not None for r in records) else None


def aggregate(args):
    per_run, per_arm = {}, {}
    for seed in args.seeds:
        material = args.material / f"seed-{seed}"
        manifest = jload(material / "material.manifest.json")
        for filename, digest in manifest.get("shas", {}).items():
            if hashlib.sha256((material / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"material digest mismatch: seed-{seed}/{filename}")
        material_digest = hashlib.sha256((material / "material.manifest.json").read_bytes()).hexdigest()
        for arm, label, directory, checkpoints in runs(seed, args):
            key = f"{arm}|{label}"
            # Invalidate this run's prior outputs before any rejection or checkpoint skip.
            name = f"{arm}.seed-{seed}.{label}.json"
            for population in ("scores", "loss"):
                for checkpoint_dir in (args.out / population).glob("cp-*"):
                    (checkpoint_dir / name).unlink(missing_ok=True)
            receipt = args.logs / f"s2-{arm}-{label}.log.wall"
            if not receipt.exists() or re.findall(r"^exit (\d+) end", receipt.read_text(), re.M) != ["0"] or not (directory / "summary.json").exists():
                per_run[key] = {"arm": arm, "result": "INCOMPLETE", "reason": "missing successful wall receipt or summary"}
                continue
            summary = jload(directory / "summary.json")
            if summary.get("material_sha256") != material_digest:
                raise ValueError(f"run material digest mismatch: {key}")
            field = "host_sha" if arm == "H" else "worktree_head"
            expected = getattr(args, "expected_host_sha" if arm == "H" else "expected_product_sha", None)
            if not expected or summary.get(field) != expected:
                per_run[key] = {"arm": arm, "result": "INCOMPLETE",
                                "reason": f"missing or mismatched code pin: {field}; expected pin required"}
                continue
            result = {"arm": arm, "result": "COMPLETE" if summary.get("status") == "DONE" else "INCOMPLETE", "checkpoints": {}}
            pending = []
            for checkpoint, sub in checkpoints.items():
                if not (directory / sub / "summary.json").exists():
                    result["result"] = "INCOMPLETE"
                    continue
                cp_summary = jload(directory / sub / "summary.json")
                if cp_summary.get("material_sha256") != material_digest:
                    raise ValueError(f"checkpoint material digest mismatch: {key}/cp-{checkpoint}")
                if cp_summary.get(field) != expected:
                    result["result"] = "INCOMPLETE"
                    result.setdefault("reasons", []).append(f"cp-{checkpoint}: missing or mismatched code pin: {field}")
                    continue
                scored = score(material, directory / sub, arm)
                loss = classify_loss(scored)
                fk, corrected = scored["metrics"]["facts_kept"], scored["metrics"]["deep_corrected"]
                result["checkpoints"][str(checkpoint)] = {"facts": fk, "corrected": corrected, "loss_counts": loss["counts"]}
                for population, payload in (("scores", scored), ("loss", loss)):
                    target = args.out / population / f"cp-{checkpoint}" / f"{arm}.seed-{seed}.{label}.json"
                    pending.append((target, payload))
                if not fk["complete"] or not corrected["complete"]:
                    result["result"] = "INCOMPLETE"
            result["continuity"] = continuity_events(summary)
            result["wait"] = engine_turn_wait(summary, jlines(material / "transcript.jsonl"))
            depths = [n["depth"] for e in summary.get("events", []) for n in e.get("nodes", []) if n.get("depth") is not None]
            depth = max(depths) if depths else None
            result["dag"] = {"max_depth": depth, "generations": depth + 1 if depth is not None else None,
                             "status": "N/A" if arm == "H" else "COMPLETE" if depth is not None else "INCOMPLETE",
                             "generation_gate": "N/A" if arm == "H" else "INCOMPLETE" if depth is None else "PASS" if depth >= 2 else "FAIL"}
            usage = accounting(summary, {})
            result["summary_tokens"] = {"input": usage["summariser_input_tokens"], "output": usage["summariser_output_tokens"],
                "estimated_calls": sum(bool(c.get("estimated")) for e in summary.get("events", []) for c in e.get("summariser_calls", []))}
            if not result["wait"]["complete"] or result["continuity"]["result"] == "INCOMPLETE" or (arm != "H" and depth is None):
                result["result"] = "INCOMPLETE"
            if any(v is None for k, v in result["summary_tokens"].items() if k != "estimated_calls"):
                result["result"] = "INCOMPLETE"
            if result["continuity"]["result"] == "FAIL" or result["dag"]["generation_gate"] == "FAIL":
                result["result"] = "FAIL"
            # Publish only after every checkpoint and final run gate accepts the run.
            if result["result"] == "COMPLETE":
                for target, payload in pending:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(payload, indent=1) + "\n")
            per_run[key] = result
    for arm in args.arms:
        records = [r for r in per_run.values() if r["arm"] == arm]
        complete = len(records) == len(args.seeds) and all(r["result"] == "COMPLETE" for r in records)
        pooled = {"result": "FAIL" if any(r["result"] == "FAIL" for r in records) else "COMPLETE" if complete else "INCOMPLETE", "checkpoints": {}}
        checkpoint_ids = sorted({cp for r in records for cp in r.get("checkpoints", {})}, key=int)
        for cp in checkpoint_ids:
            cells = [r.get("checkpoints", {}).get(cp) for r in records]
            valid = len(cells) == len(args.seeds) and all(c and c["facts"]["complete"] for c in cells)
            present = [c for c in cells if c]
            classes = {}
            for cell in present:
                for name, counts in cell["facts"]["by_class_placement"].items():
                    value = classes.setdefault(name.split("|", 1)[0], [0, 0])
                    value[0] += counts[0]
                    value[1] += counts[1]
            numerator = sum(c["facts"]["correct"] for c in present)
            denominator = sum(c["facts"]["denominator"] for c in present)
            corrected = sum(c["corrected"]["correct"] for c in present)
            corrected_n = sum(c["corrected"]["denominator"] for c in present)
            pooled["checkpoints"][cp] = {"complete": valid, "runs_scored": len(present), "runs_expected": len(args.seeds),
                "correct": numerator if valid else None, "denominator": denominator if valid else None,
                "facts_kept": numerator / denominator if valid and denominator else None, "by_class": classes,
                "corrected": corrected if valid else None, "corrected_denominator": corrected_n if valid else None,
                "corrected_kept": corrected / corrected_n if valid and corrected_n else None,
                "stale_assertions": sum(len(c["corrected"]["stale_ids"]) for c in present)}
        waits = [r.get("wait", {}) for r in records]
        if len(waits) == len(args.seeds) and all(w.get("complete") for w in waits):
            values = [v for w in waits for v in w["turn_waits_s"]]
            pooled["wait"] = {"mean_s": mean(values), "p90_s": p90(values), "max_s": max(values),
                "waited_share_ge_1s": sum(v >= 1 for v in values) / len(values),
                "compactions_per_100_messages": 100 * sum(w["compactions"] for w in waits) / sum(w["messages"] for w in waits)}
        else:
            pooled["wait"] = None
        tokens = [r.get("summary_tokens", {}) for r in records]
        pooled["summary_tokens"] = {k: total_present(tokens, k) for k in ("input", "output", "estimated_calls")}
        pooled["dag_per_run"] = [r.get("dag") for r in records]
        pooled["continuity"] = "FAIL" if any(r.get("continuity", {}).get("result") == "FAIL" for r in records) else "PASS" if len(records) == len(args.seeds) and all(
            r.get("continuity", {}).get("result") == "PASS" for r in records) else "INCOMPLETE"
        per_arm[arm] = pooled
    return {"schema": "c1-deep-score-v1", "proof_boundary": "engine-direct synthetic replay; one reader; not host latency or customer proof",
            "per_run": per_run, "per_arm": per_arm}


def write_deep(args):
    args.out.mkdir(parents=True, exist_ok=True)
    result = aggregate(args)
    (args.out / "score-deep.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["per_arm"], indent=1))
    return result


def exit_status(result):
    return 1 if any(a["result"] == "FAIL" for a in result["per_arm"].values()) else 2 if any(
        a["result"] != "COMPLETE" for a in result["per_arm"].values()) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "material", "logs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["LCMX-fleet-rc3", "H"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--checkpoints", nargs="+", help="exact numeric rows or last; default: all deep answer keys")
    parser.add_argument("--expected-product-sha", help="required product pin for LCM-X runs")
    parser.add_argument("--expected-host-sha", help="required host pin for Hermes runs")
    result = write_deep(parser.parse_args(argv))
    return exit_status(result)


if __name__ == "__main__":
    raise SystemExit(main())
