#!/usr/bin/env python3
"""C1b offline split rescore. Snapshot retained receipts first; never calls models.

First: --source C1_ARTIFACTS --rca RCA_ARTIFACTS --out NEW_OUTPUT
Repeat: --inputs NEW_OUTPUT/inputs --out OTHER_OUTPUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from split_scoring import RULE_VERSION, RULES, fact_verdicts, request_ground_truth, score_event


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(source, rca, target):
    if target.exists():
        raise ValueError("snapshot already exists; reuse with --inputs (no overwrite)")
    entries = []
    def copy(root, rel, prefix="c1"):
        src, dst = root / rel, target / prefix / rel
        before = digest(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if digest(src) != before or digest(dst) != before:
            raise ValueError(f"input changed while snapshotting: {rel}")
        entries.append(dict(path=str(dst.relative_to(target)), source=str(src), sha256=before, bytes=src.stat().st_size))
    for seed in (1, 2):
        for name in ("material.manifest.json", "deep_metadata.json", "transcript.jsonl"):
            copy(source, Path(f"material/seed-{seed}/{name}"))
        run = Path(f"runs/LCMX-fleet-rc3/seed-{seed}/d{seed}-r1")
        name = "summary.json" if seed == 1 else "partial-summary.json"
        copy(source, run / name)
        for event in load(source / run / name)["events"]:
            if event.get("is_compaction"):
                for folder in ("deep-probes", "continuity"):
                    copy(source, run / folder / f"event-{event['event']}.json")
        if seed == 1:
            for cp in load(source / f"material/seed-{seed}/deep_metadata.json")["checkpoint_answers"]:
                for name in ("summary.json", "results.jsonl", "assembled_context.json"):
                    copy(source, run / f"cp-{cp}" / name)
                copy(source, Path(f"decision/scores/cp-{cp}/LCMX-fleet-rc3.seed-1.d1-r1.json"))
    for name in ("per-fact.jsonl", "events.jsonl", "RCA.md"):
        copy(rca, Path(name), "rca")
    write(target / "INPUT-MANIFEST.json", dict(entries=entries))


def count(records, field):
    values = [r[field] for r in records]
    return dict(pass_count=sum(v is True for v in values), fail_count=sum(v is False for v in values),
                unavailable=sum(v is None for v in values), total=len(values))


def render(result, output):
    text = ["# C1b retained-receipt rescore", "", "Claim class: advisory bench instrument. Rule: " + result["rule_version"],
            "Strict (historical) is unchanged. Columns are independent diagnostics, not a replacement quality score.",
            "", "## Completed seed 1 checkpoints", "",
            "| Checkpoint | strict (historical) | verbatim | semantic | scalar (numeric only) | association |",
            "|---|---|---|---|---|---|"]
    def cell(c):
        return f"{c['pass_count']}/{c['pass_count'] + c['fail_count']}" + (f" ({c['unavailable']} N/A or missing)" if c["unavailable"] else "")
    fields = ("strict (historical)", "verbatim", "semantic", "scalar", "association")
    for cp, counts in result["checkpoints"].items():
        text.append("| " + cp + " | " + " | ".join(cell(counts[k]) for k in fields) + " |")
    text += ["", "## Observed compaction events", "",
             "R-initial retained = operative objective verbatim in context OR semantic in the SAME historical R answer.",
             "R-latest = latest user instruction, including corrections. No fresh question was asked.",
             "Q/O reader compliance is reported independently from context presence; O-compliance = reader_pass.",
             "", "| Seed/status | events | strict R | R-initial verbatim | R-initial semantic | R-initial retained | R-latest compliance | Q verbatim | Q compliance | O verbatim | O compliance |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed, counts in result["events"].items():
        text.append("| " + seed + " | " + str(counts["n"]) + " | " + " | ".join(cell(counts[k]) for k in counts if k != "n") + " |")
    text += ["", "Per-fact/checkpoint verdicts: `facts-split.jsonl`; per-event source IDs, supersession and verdicts: `events-split.jsonl`.",
             "Request ground truth: `request-ground-truth.json`; snapshotted inputs and digests: `inputs/INPUT-MANIFEST.json`.",
             "Seed 2 remains partial: event evidence only, no checkpoint retention admitted. Missing cells and Hermes remain unrun.",
             "Not proven: product improvement, semantic understanding, autonomous retrieval, live-host/customer behavior, or release readiness.",
             "T1/T2 incomplete: no push/GitHub publication authorized; orchestrator owns publication. Results remain local and retained."]
    (output / "RESCORE.md").write_text("\n".join(text) + "\n")


def rescore(inputs, output, rule=RULE_VERSION):
    for name in ("facts-split.jsonl", "events-split.jsonl", "request-ground-truth.json", "rescore-summary.json", "RESCORE.md"):
        if (output / name).exists():
            raise ValueError("existing result; choose a new output directory: " + name)
    for entry in load(inputs / "INPUT-MANIFEST.json")["entries"]:
        if digest(inputs / entry["path"]) != entry["sha256"]:
            raise ValueError("snapshot digest mismatch: " + entry["path"])
    source = inputs / "c1"
    rca_facts = {(r["checkpoint"], r["id"]): r for r in lines(inputs / "rca/per-fact.jsonl")}
    rca_events = {(r["seed"], r["event"]): r for r in lines(inputs / "rca/events.jsonl")}
    facts_out, events_out, ground_truth, checkpoints, event_counts = [], [], {}, {}, {}
    for seed in (1, 2):
        metadata = load(source / f"material/seed-{seed}/deep_metadata.json")
        rows = lines(source / f"material/seed-{seed}/transcript.jsonl")
        manifest = load(source / f"material/seed-{seed}/material.manifest.json")
        for name in ("deep_metadata.json", "transcript.jsonl"):
            if digest(source / f"material/seed-{seed}" / name) != manifest["shas"][name]:
                raise ValueError("material manifest digest mismatch")
        truth = request_ground_truth(rows, metadata["continuity_probes"], metadata.get("request_policy"))
        # Material pre-dates this additive ground-truth field; never rewrite it.
        truth["provenance"] = "derived from pinned deep-v1 generator policy; no objective cancellation in material"
        ground_truth[str(seed)] = truth
        run = source / f"runs/LCMX-fleet-rc3/seed-{seed}/d{seed}-r1"
        summary = load(run / ("summary.json" if seed == 1 else "partial-summary.json"))
        material_sha = digest(source / f"material/seed-{seed}/material.manifest.json")
        if summary.get("material_sha256") != material_sha:
            raise ValueError(f"run material digest mismatch: seed-{seed}")
        if seed == 1:
            for cp, facts in sorted(metadata["checkpoint_answers"].items(), key=lambda item: int(item[0])):
                directory = run / f"cp-{cp}"
                if load(directory / "summary.json").get("material_sha256") != material_sha:
                    raise ValueError(f"checkpoint material digest mismatch: seed-{seed}/cp-{cp}")
                answers = {r["probe_id"]: r for r in lines(directory / "results.jsonl")}
                assembled = load(directory / "assembled_context.json")
                context = assembled["system"] + "\n" + "\n".join(str(r.get("content") or "") for r in assembled["view"])
                historical = load(source / f"decision/scores/cp-{cp}/LCMX-fleet-rc3.seed-1.d1-r1.json")
                records = []
                for f in facts:
                    row = answers.get(f["id"], {})
                    strict = historical["probes"][f["id"]]["success"]
                    verdicts = fact_verdicts(f, row.get("answer"), context, facts, historical=strict, rule=rule)
                    trace = rca_facts[int(cp), f["id"]]
                    if strict != trace["reader_success"] or verdicts["verbatim"] != trace["present_exact"]:
                        raise ValueError(f"historical/RCA mismatch: {cp}/{f['id']}")
                    records.append(dict(seed=seed, checkpoint=int(cp), id=f["id"], fixture=f["fixture"],
                                        fact_class=f["class"], answer=row.get("answer"), expected=f["value"], **verdicts))
                if sum(r["strict (historical)"] for r in records) != historical["metrics"]["facts_kept"]["correct"]:
                    raise ValueError("historical aggregate mismatch")
                facts_out.extend(records)
                checkpoints[cp] = {k: count(records, k) for k in verdicts}
        records = []
        for event in summary["events"]:
            if not event.get("is_compaction"):
                continue
            eid, row_index = event["event"], event["row_index"]
            receipt = load(run / f"deep-probes/event-{eid}.json")
            context = load(run / f"continuity/event-{eid}.json")["text"]
            historical = {c["kind"]: c for c in receipt["checks"]}
            if receipt["checks"] != rca_events[seed, eid]["checks"]:
                raise ValueError("historical event/RCA mismatch")
            verdicts = score_event(receipt, context, row_index, truth)
            records.append(dict(seed=seed, event=eid, row_index=row_index,
                                historical=historical, **verdicts))
        events_out.extend(records)
        columns = {
            "strict R": [r["historical"]["current_request"]["passed"] for r in records],
            **{f"R-initial {k}": [r["R-initial"][k] for r in records] for k in ("verbatim", "semantic", "retained")},
            "R-latest compliance": [r["R-latest"]["reader_compliance"] for r in records],
            **{f"{k} {label}": [r[k][field] for r in records] for k in ("Q", "O")
               for label, field in (("verbatim", "verbatim"), ("compliance", "reader_compliance"))},
        }
        event_counts[f"{seed} / {'completed' if seed == 1 else 'partial'}"] = dict(n=len(records), **{
            k: count([dict(value=v) for v in values], "value") for k, values in columns.items()})
    result = dict(schema="c1b-split-score-v1", rule_version=rule, checkpoints=checkpoints, events=event_counts)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("facts-split", facts_out), ("events-split", events_out)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    write(output / "request-ground-truth.json", ground_truth)
    write(output / "rescore-summary.json", result)
    render(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rca", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rule", choices=RULES, default=RULE_VERSION)
    args = parser.parse_args()
    if args.inputs is None:
        if args.source is None or args.rca is None:
            parser.error("provide --inputs or both --source and --rca")
        args.inputs = args.out / "inputs"
        snapshot(args.source, args.rca, args.inputs)
    result = rescore(args.inputs, args.out, rule=args.rule)
    print(json.dumps(dict(checkpoints=len(result["checkpoints"]), events=sum(c["n"] for c in result["events"].values()),
                         historical="unchanged; checked against retained scorer and RCA", model_calls=0)))


if __name__ == "__main__":
    main()
