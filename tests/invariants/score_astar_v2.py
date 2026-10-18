"""Input-derived offline occurrence scoring; historical DUP keys stay unchanged.

Usage: python score_astar_v2.py LANE --output NEW_DIRECTORY
Capture host inputs with scoring/capture_inputs.py before scoring. No network,
product imports, or historical scorer execution is needed. Missing evidence fails
closed. Output directories are new immutable snapshots, never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

Key = tuple[Any, ...]
LABELS = ("cand", "main", "f05", "iter3", "iter2", "ga", "v028", "head")
FLAGS = ("DUP", "DISTINCT_LOSS", "FRESH_NOT_ONCE", "MULTIPLICITY_LOSS")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def counts(rows: list[list[Any]], session: bool = True) -> Counter[Key]:
    return Counter(tuple(r[:3]) if session else ("S", *r[:2]) for r in rows)


def receipt(path: Path, shape: str, label: str, probes: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    sequence = None
    exact: Counter[Key] | None = None
    if shape.startswith("duplicate/"):
        rows = d["rows"]
        c = counts(rows)
        exact = Counter(tuple(r) for r in rows)
        trace = path.with_name("trace-" + path.stem + ".steps")
        sequence = json.loads(trace.read_text()) if trace.exists() else None
    elif shape.startswith("loss/"):
        rows = [[sid, *row] for sid, values in d.get("ledger", {}).items() for row in values]
        c, exact = counts(rows), Counter(tuple(r) for r in rows)
    elif shape.startswith("state/"):
        c = Counter()
        for sid in ("S", "T", "child"):
            c.update((sid, *r[:2]) for r in d.get("S_again" if sid == "S" and "S_again" in d else sid, []))
    elif shape.startswith("T1/"):
        c = counts(d["final_ledger"], False)
        sequence = [step["stored"] for step in d["steps"]]
    elif shape.startswith("T2/"):
        c = counts(d["ledger"])
        sequence = [v for v in d["steps"].values() if type(v) is int]
    elif shape.startswith("T3/"):
        c = Counter({("S", "content", k): v for k, v in d["5_append"].items()})
        sequence = [sum(d[k].values()) for k in ("1_full", "2_short", "4_restored", "5_append")]
    else:
        c = counts(d["steps"][-1]["ledger"], False)
        sequence = d.get("counts", [step["stored"] for step in d["steps"]])
    return {"counter": c, "exact": c if exact is None else exact,
            "steps": sequence or [sum(c.values())], "error": d.get("error"),
            "receipt": path.relative_to(probes).as_posix(), "label": label}


def load_rows(lane: Path) -> dict[str, dict[str, dict[str, Any]]]:
    probes = lane / "probes"
    data: dict[str, dict[str, dict[str, Any]]] = {}
    for label in LABELS:
        paths: list[tuple[Path, str]] = []
        for family in ("duplicate", "loss", "state"):
            paths.extend((p, family + "/" + p.stem) for p in
                         sorted((probes / family / "results" / label).glob("*.json")))
        for directory, prefix in (("T1-nested-shrink", "T1"), ("T2-absorbed-uid", "T2")):
            for p in sorted((probes / directory).glob("result-*.json")):
                for marker, name_prefix in (("result-" + label + "-", ""),
                                            ("result-compress-" + label + "-", "compress-")):
                    if p.stem.startswith(marker):
                        paths.append((p, prefix + "/" + name_prefix + p.stem.removeprefix(marker)))
        for directory, prefix in (("T3-aux-session-start/results", "T3"),
                                  ("T4-partial-restore/results", "T4")):
            paths.extend((p, prefix + "/" + p.stem.removeprefix(label + "-"))
                         for p in sorted((probes / directory).glob(label + "-*.json")))
        paths.extend((p, "steer/" + p.stem) for p in sorted((probes / "steer-results" / label).glob("*.json")))
        for path, name in paths:
            raw = json.loads(path.read_text())
            if name.startswith("duplicate/") and "rows" not in raw:
                continue
            if name.startswith(("T4/", "steer/")) and "steps" not in raw:
                continue
            data.setdefault(name, {})[label] = receipt(path, name, label, probes)
    return data


def load_inputs(lane: Path) -> dict[tuple[str, str], dict[str, Any]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for label in LABELS:
        for path in sorted((lane / "scoring/inputs" / label).rglob("*.json")):
            trace = json.loads(path.read_text())
            trace["source"] = str(path.relative_to(lane))
            trace["preferred_run"] = path.parent.name.endswith(("-v2", "-v3"))
            trace["capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            for result in trace["receipts"]:
                candidates[label, result].append(trace)
    linked = {}
    for key, traces in candidates.items():
        valid = [t for t in traces if t.get("capture_version", 1) >= 2]
        if not valid:
            continue
        newest = max(t["capture_version"] for t in valid)
        valid = [t for t in valid if t["capture_version"] == newest]
        preferred = [t for t in valid if t["preferred_run"]]
        selected = preferred or valid
        if len(selected) != 1:
            raise ValueError(f"Ambiguous canonical v2 captures for {key}")
        chosen = selected[0].copy()
        superseded = []
        for prior in traces:
            if prior is selected[0]:
                continue
            same_inputs = digest([host_view(e) for e in prior["events"]]) == digest([host_view(e) for e in chosen["events"]])
            same_node = prior["node"] == chosen["node"]
            superseded.append({"source": prior["source"], "sha256": prior["capture_sha256"],
                               "version": prior.get("capture_version", 1),
                               "same_node": same_node, "same_inputs": same_inputs})
        chosen["superseded"] = superseded
        linked[key] = chosen
    return linked


def host_view(event: dict[str, Any]) -> dict[str, Any]:
    return {k: event[k] for k in ("operation", "session", "messages", "kwargs", "lifecycle_args", "group")}


def text_key(sid: str, role: str, text: str, shape: str) -> Key:
    return ("S", "content", text) if shape.startswith("T3/") else (sid, role, text)


def stored(event: dict[str, Any], field: str, shape: str) -> Counter[Key]:
    return Counter(text_key(str(r[1]), str(r[2]), str(r[3]), shape) for r in event[field])


def oracle(traces: dict[str, dict[str, Any]], shape: str, reference: str) -> dict[str, Any]:
    """Bound replay discounts by host identities AND prior all-tree absorption.

    A host token is metadata plus normalized text and its within-view ordinal.
    Missing-to-known metadata adoption keeps its token; conflicting known keys
    and a new unkeyed occurrence following keyed-only history create tokens.
    A GA append is report-only only if the same token was present in an earlier
    host view, and every required tree had already stored its ordinal then.
    """
    events = traces[reference]["events"]
    universe: set[Key] = set()
    in_place: set[Key] = set()
    previous_lengths: dict[tuple[int, str], int] = {}
    tokens: dict[tuple[int, Key], dict[Key, int]] = defaultdict(dict)
    absorbed: set[tuple[int, Key, int]] = set()
    replay: Counter[Key] = Counter()
    proof: list[dict[str, Any]] = []
    final: dict[int, Counter[Key]] = {}
    sent: set[tuple[int, str, str]] = set()
    emissions: dict[tuple[int, str, str], int] = {}
    exclusions: list[dict[str, Any]] = []
    linkage: list[dict[str, Any]] = []
    parents: dict[tuple[int, str], str] = {}
    ownership: dict[tuple[int, str, str, Key], str] = {}
    for index, event in enumerate(events):
        group = int(event["group"])
        target = event.get("after_session") if event["operation"] == "compress" else None
        if event["operation"] == "on_session_start" and event["kwargs"].get("boundary_reason") == "compression":
            target = event["lifecycle_args"][0]
        if target and target != event["session"] and event["session"]:
            parents[group, target] = event["session"]
        used: Counter[Key] = Counter()
        current: dict[Key, list[int]] = defaultdict(list)
        before = stored(event, "before", shape)
        after = stored(event, "after", shape)
        final[group] = after
        view_key = (group, event["session"])
        view_length = len(event["messages"])
        shrinking = view_length < previous_lengths.get(view_key, 0)
        if event["operation"] in {"ingest", "compress"}:
            previous_lengths[view_key] = view_length
        for position, (message, normalized) in enumerate(zip(event["messages"], event["normalized"], strict=True)):
            role, text = normalized
            if not str(text).strip():
                continue
            raw = message.get("content")
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True) if raw else ""
            raw_key = (group, role, raw_text)
            for (source_group, source_role, rendered), source_index in emissions.items():
                if source_group == group and source_role == role and raw_text.startswith(rendered + "\n\n"):
                    remainder = raw_text[len(rendered) + 2:]
                    if (group, role, remainder) in sent:
                        text = remainder
                        exclusions.append({"event": index, "reason": "compress-prefix-plus-prior-input-source",
                                           "source_event": source_index, "prefix_sha256": digest(rendered),
                                           "source_text_sha256": digest(remainder)})
                        break
            if raw_key in emissions:
                exclusions.append({"event": index, "reason": "exact-compress-output", "source_event": emissions[raw_key],
                                   "role": role, "text_sha256": digest(raw_text)})
                continue
            sent.add(raw_key)
            metadata = (message.get("message_uid"), message.get("timestamp"))
            owner_key = (group, role, text, metadata)
            sid = event["session"]
            ancestor = sid
            visited: set[str] = set()
            while (group, ancestor) in parents and ownership.get(owner_key) != ancestor and ancestor not in visited:
                visited.add(ancestor)
                ancestor = parents[group, ancestor]
            if ancestor != sid and ownership.get(owner_key) == ancestor:
                linkage.append({"event": index, "from_session": sid, "owner_session": ancestor,
                                "role": role, "text_sha256": digest(text), "metadata": list(metadata)})
                sid = ancestor
            ownership.setdefault(owner_key, sid)
            key = text_key(sid, role, text, shape)
            if key not in universe and key not in in_place:
                # First appearances inside a shortened prefix are edits (#1563).
                (in_place if shrinking and position < view_length else universe).add(key)
            used[(key, *metadata)] += 1
            identity = (*metadata, used[(key, *metadata)])
            bucket = tokens[group, key]
            if identity not in bucket:
                compatible = [prior for prior in bucket if prior[-1] == identity[-1]
                              and all(old is None or old == new for old, new in zip(prior[:2], metadata, strict=True))]
                if len(compatible) == 1:
                    bucket[identity] = bucket.pop(compatible[0])
                else:
                    bucket[identity] = max(bucket.values(), default=0) + 1
            current[key].append(bucket[identity])
        peers = [trace["events"][index] for trace in traces.values()
                 if len(trace["events"]) > index and host_view(trace["events"][index]) == host_view(event)]
        complete = len(peers) == len(traces)
        for key, ordinals in current.items():
            already = sum((group, key, ordinal) in absorbed for ordinal in ordinals)
            added = max(0, after[key] - before[key])
            discount = min(added, already, max(0, after[key] - max(tokens[group, key].values())))
            if discount:
                replay[key] += discount
                proof.append({"event": index, "key": list(key), "copies": discount,
                              "prior_absorbed_ordinals": [v for v in ordinals if (group, key, v) in absorbed]})
            if complete:
                floor = min(stored(peer, "after", shape)[key] for peer in peers)
                for ordinal in ordinals:
                    if floor >= ordinal:
                        absorbed.add((group, key, ordinal))
        for message in event.get("returned_messages", []):
            content = message.get("content")
            rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True) if content else ""
            key_out = (group, str(message.get("role") or "unknown"), rendered)
            if key_out not in sent:
                emissions.setdefault(key_out, index)
    for key in list(replay):
        distinct = max((max(bucket.values(), default=0) for (_, text), bucket in tokens.items() if text == key), default=0)
        observed = max((counter[key] for counter in final.values()), default=0)
        replay[key] = min(replay[key], max(0, observed - distinct))
    return {"universe": universe, "in_place": in_place, "replay": replay, "proof": proof,
            "emission_exclusions": exclusions, "ownership_linkage": linkage,
            "input_sha256": digest([host_view(event) for event in events])}


def compare(c: Counter[Key], g: Counter[Key], ec: Counter[Key], eg: Counter[Key],
            universe: set[Key], replay: Counter[Key], metadata: str, shape: str) -> dict[str, Any]:
    floor = Counter({key: max(1, value - replay[key]) for key, value in g.items()})
    dup = {str(k): [ec[k], eg[k]] for k in ec if ec[k] > max(1, eg[k])}
    if shape.startswith("duplicate/ac-ts_adopt_uid_conflict-"):
        dup = {}  # Explicit ruling: both shrink and noshrink; preserve historical DUP keys.
    return {"DUP": dup, "DISTINCT_LOSS": [str(k) for k in g if not c[k]],
            "FRESH_NOT_ONCE": {str(k): c[k] for k in universe - set(g) if c[k] != 1},
            "MULTIPLICITY_LOSS": {str(k): [c[k], floor[k]] for k in floor if c[k] < floor[k]},
            "REPORT_ONLY_REPLAY_LOSS": {str(k): [c[k], g[k]] for k in g if c[k] < g[k] and c[k] >= floor[k]},
            "multiplicity_gating": metadata != "unstamped"}


def failed(flags: dict[str, Any]) -> bool:
    return any(flags[k] for k in FLAGS[:3]) or bool(flags["MULTIPLICITY_LOSS"] and flags["multiplicity_gating"])


def self_test() -> dict[str, bool]:
    original: Counter[Key] = Counter({("S", "user", "retained"): 1})
    duplicate = original + original
    empty: Counter[Key] = Counter()
    args = (set(original), empty, "uid", "synthetic")
    checks = {"ga_self_passes": not failed(compare(original, original, original, original, *args)),
              "planted_duplicate_flagged": bool(compare(duplicate, original, duplicate, original, *args)["DUP"]),
              "planted_distinct_loss_flagged": bool(compare(empty, original, empty, original, *args)["DISTINCT_LOSS"]),
              "fresh_missing_from_every_tree_flagged": bool(compare(empty, empty, empty, empty, *args)["FRESH_NOT_ONCE"])}
    row = {"role": "user", "content": "retained", "message_uid": "a"}
    ledger = [[1, "S", "user", "retained", None]]

    def event(messages: list[dict[str, Any]], before: list[list[Any]], after: list[list[Any]],
              **extra: Any) -> dict[str, Any]:
        return {"operation": "ingest", "session": "S", "messages": messages,
                "normalized": [[m["role"], m["content"]] for m in messages],
                "kwargs": {}, "lifecycle_args": [], "group": 0, "before": before, "after": after, **extra}

    base = [event([row], [], ledger), event([row], ledger, ledger)]
    traces = {label: {"events": json.loads(canonical(base))} for label in LABELS}
    traces["ga"]["events"][1]["after"] += [[2, "S", "user", "retained", None]]
    facts = oracle(traces, "synthetic/arbitrary", "ga")
    checks["mechanical_replay_discount"] = facts["replay"][("S", "user", "retained")] == 1
    for trace in traces.values():
        trace["events"][1]["messages"][0]["message_uid"] = "b"
    checks["distinct_uid_not_discounted"] = not oracle(traces, "synthetic/arbitrary", "ga")["replay"]
    generated = {"role": "user", "content": "generated synthetic context"}
    generated_events = [event([row], [], ledger),
                        event([row], ledger, ledger, operation="compress", returned_messages=[generated]),
                        event([generated], ledger, ledger)]
    traces = {label: {"events": generated_events} for label in LABELS}
    facts = oracle(traces, "synthetic/arbitrary", "ga")
    checks["exact_emission_excluded"] = facts["universe"] == set(original)
    generated_events.append(event([{"role": "user", "content": generated["content"] + "\n\nnew text"}], ledger, ledger))
    facts = oracle(traces, "synthetic/arbitrary", "ga")
    checks["novel_text_behind_carrier_kept"] = len(facts["universe"]) == 2
    edited = {"role": "user", "content": "edited prefix"}
    appended = {"role": "user", "content": "appended text"}
    other = {"role": "assistant", "content": "original tail"}
    scoped_events = [event([row, other], [], ledger),
                     event([edited], ledger, ledger),
                     event([edited, appended], ledger, ledger)]
    scoped = oracle({"ga": {"events": scoped_events}}, "synthetic/scope", "ga")
    missing = compare(original, original, original, original, scoped["universe"], empty, "uid", "synthetic")
    checks["appended_loss_flagged"] = str(("S", "user", "appended text")) in missing["FRESH_NOT_ONCE"]
    checks["in_place_loss_report_only_1563"] = (
        ("S", "user", "edited prefix") in scoped["in_place"]
        and str(("S", "user", "edited prefix")) not in missing["FRESH_NOT_ONCE"])
    checks["scoped_ga_self_passes"] = not failed(compare(original, original, original, original, set(original), empty, "uid", "synthetic"))
    assert all(checks.values()), checks
    return checks


def score(lane: Path) -> dict[str, Any]:
    rows, inputs = load_rows(lane), load_inputs(lane)
    known = json.loads((lane / "metadata-reference.json").read_text())
    records: list[dict[str, Any]] = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    identities: dict[str, str] = {}
    for shape, trees in sorted(rows.items()):
        metadata = known.get(shape, "uid" if shape.startswith("duplicate/uid-") else
                             "unstamped" if "unstamped" in shape or "/none-" in shape else
                             "stamped" if "timestamp" in shape or "/ts-" in shape else "mixed")
        record: dict[str, Any] = {"shape": shape, "metadata": metadata,
                                 "counts": {label: row["steps"] for label, row in trees.items()}}
        gaps = [label for label in LABELS if label not in trees or (label, trees[label]["receipt"]) not in inputs]
        if set(trees) == {"head"}:
            record["coverage_control"] = "Historical head-only attribution control; report-only"
            totals[metadata]["head_only_controls"] += 1
            records.append(record)
            continue
        if gaps:
            record["incomplete"] = gaps
            totals[metadata]["incomplete"] += 1
            records.append(record)
            continue
        traces = {label: inputs[label, trees[label]["receipt"]] for label in LABELS}
        for label, trace in traces.items():
            final_rows = trace["events"][-1]["after"]
            trees[label]["counter"] = Counter(text_key(str(r[1]), str(r[2]), str(r[3]), shape) for r in final_rows)
        reference = "main" if trees["ga"]["error"] else "ga"
        valid_traces = {label: trace for label, trace in traces.items() if not trees[label]["error"]}
        facts = oracle(valid_traces, shape, reference)
        input_facts = {label: oracle(traces, shape, label) for label in LABELS}
        universe = set().union(*(f["universe"] for f in input_facts.values()))
        record["input_sha256"] = {label: digest([host_view(e) for e in traces[label]["events"]]) for label in LABELS}
        identity = digest(record["input_sha256"])
        record["duplicate_of"] = identities.get(identity)
        identities.setdefault(identity, shape)
        record["input_sources"] = {label: {"source": trace["source"], "sha256": trace["capture_sha256"],
                                                   "superseded": trace["superseded"]} for label, trace in traces.items()}
        record["normalization"] = {label: dict(Counter(source for event in trace["events"]
                                                     for source in event.get("normalization_sources", [])))
                                   for label, trace in traces.items()}
        record["report_only_in_place_edits_1563"] = {
            label: {str(key): trees[label]["counter"][key] for key in sorted(f["in_place"], key=str)}
            for label, f in input_facts.items()}
        record["host_input_universe"] = [list(key) for key in sorted(universe, key=str)]
        record["fresh_exclusions"] = {label: {key: facts[key] for key in ("emission_exclusions", "ownership_linkage")}
                                      for label, facts in input_facts.items()}
        record["content_ledger"] = "Full captured SQL; historical receipt counters retained unchanged for DUP"
        record["replay_proof"] = facts["proof"]
        record["flags"] = {}
        for label in LABELS:
            c, g = trees[label], trees[reference]
            record["flags"][label] = compare(c["counter"], g["counter"], c["exact"], g["exact"],
                                                  universe if label != reference else set(g["counter"]),
                                                  facts["replay"], metadata, shape)
        flags = record["flags"]["cand"]
        differential = shape == "T1/unstamped-front-cold" or bool(trees["ga"]["error"])
        if differential:
            differentials = {}
            for baseline in ("main", "v028"):
                base = trees[baseline]
                floor_facts = oracle(valid_traces, shape, baseline)
                f = compare(trees["cand"]["counter"], base["counter"], trees["cand"]["exact"], base["exact"],
                            universe, floor_facts["replay"], metadata, shape)
                if shape == "T1/unstamped-front-cold":
                    a, b = trees["cand"]["steps"], base["steps"]
                    f["STEP_REGRESSION"] = len(a) != len(b) or any(x > y for x, y in zip(a, b, strict=False))
                differentials[baseline] = f
            record["differential"] = differentials
            record["gating"] = any(failed(f) or f.get("STEP_REGRESSION", False) for f in differentials.values())
        else:
            record["gating"] = failed(flags)
        record["iter2_regression"] = not failed(record["flags"]["iter2"]) and record["gating"]
        if not record["duplicate_of"]:
            total = totals[metadata]
            total["compared"] += 1
            total["differential"] += differential
            total["gating_rows"] += record["gating"]
            total["reference_self_failures"] += failed(record["flags"][reference])
            total["iter2_regressions"] += record["iter2_regression"]
            for flag in FLAGS:
                total[flag] += bool(flags[flag])
        records.append(record)
    repairs = {}
    for name in json.loads((lane / "stamped-repair-reference.json").read_text()):
        tree = rows["T4/timestamp-6+fresh-then-8+fresh" if name == "steer/repair-stamped" else name]
        repairs[name] = (tree["cand"]["steps"][2] == tree["head"]["steps"][2] == 9
                         if name == "steer/repair-stamped" else tree["cand"]["counter"] == tree["head"]["counter"])
    return {"self_test": self_test(), "totals": totals, "shapes": records, "repairs": repairs,
            "fresh_universe": "Normalized built host inputs, never the union of stored outputs",
            "product_provenance": json.loads((lane / "trees/provenance.json").read_text()),
            "scope": "Historical receipt families only; A*/C1 additions and P1-P8 are separately asserted by owner",
            "claim_class": "advisory"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", nargs="?", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(canonical(self_test()))
        return
    result = score(args.lane)
    output = args.output or args.lane / "scoring/v2-result"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "astar-v2-comparison.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"totals": result["totals"], "self_test": result["self_test"]}, indent=2))


if __name__ == "__main__":
    main()
