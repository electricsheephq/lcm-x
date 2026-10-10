#!/usr/bin/env python3
"""Validate Track S material offline; runtime receipts remain a separate gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s3_generator", Path(__file__).with_name("gen_material.py"))
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def verify(directory: Path, min_tokens: int | None = None, min_events: int | None = None) -> dict:
    def load(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    def lines(name):
        return [json.loads(line) for line in (directory / name).read_text().splitlines()]

    def check(ok, message):
        if not ok:
            raise ValueError(message)

    rows = [json.loads(line) for line in (directory / "transcript.jsonl").read_text().splitlines()]
    facts, manifest, state = load("facts.json"), load("material.manifest.json"), load("continuation.json")
    v4 = manifest.get("material_version") == "track-s-v4"
    check(v4 or manifest.get("material_version") == _gen.MATERIAL_VERSION, "material version mismatch")
    admissions = load("admission.manifest.json")
    count = _gen.token_counter()
    counts = [count(r["content"]) for r in rows]
    prefix, total = [], 0
    for n in counts:
        total += n
        prefix.append(total)
    trigger = min_tokens if min_tokens is not None else manifest["params"]["min_tokens"]
    events = min_events if min_events is not None else manifest["params"]["min_events"]
    smoke = manifest["mode"] == "smoke"
    check(trigger > 0 and events >= 2, "invalid trigger/event plan")
    check(len({r["id"] for r in rows}) == len(rows), "duplicate row ids")
    check(rows[0]["role"] == "system", "missing head system row")
    check([r["turn"] for r in rows] == sorted(r["turn"] for r in rows), "turn chronology")
    for i, r in enumerate(rows):
        check(r["role"] in ("system", "user", "assistant", "tool"), "invalid role")
        if r["role"] == "tool":
            prev = rows[i - 1]
            check(prev["role"] == "assistant" and prev["turn"] == r["turn"] and
                  prev["tool_call_id"] == r["tool_call_id"] and r["tool_call_id"], "tool pairing")
            calls = prev.get("tool_calls", [])
            check(len(calls) == 1 and calls[0].get("id") == r["tool_call_id"] and
                  calls[0].get("type") == "function" and
                  calls[0].get("function", {}).get("name") == "read_material", "tool call metadata mismatch")
    check(Counter(f["class"] for f in facts) == Counter({c: 5 for c in _gen.CLASSES}), "12 classes x 5 required")
    for cls in (() if v4 else _gen.CLASSES):
        check(Counter(f["placement"] for f in facts if f["class"] == cls) ==
              Counter(head=2, middle=1, tail=2), f"placement coverage mismatch: {cls}")
    check(len({f["id"] for f in facts}) == 60, "60 unique fact ids required")

    def source(item):
        index = item["row_index"]
        check(isinstance(index, int) and 0 <= index < len(rows), "invalid source row")
        row = rows[index]
        check(item["row_id"] == row["id"] and item["id"] in row["content"], f"unmatchable id {item['id']}")
        return row

    for f in facts:
        row = source(f)
        text, start = row["content"], f["char_offset"]
        end = start + len(f["value"])
        check(text[start:end] == f["value"] and f["answer"] == f["value"], f"missing value {f['id']}")
        check(row["role"] == f["row_role"], "fact role mismatch")
        placement = f["placement"]
        check(placement in ("head", "middle", "tail"), "invalid placement")
        if placement == "middle":
            check((v4 and row["role"] == "user" and count(text) >= 2000 and
                   .3 <= start / len(text) <= .7) or
                  (row["role"] == "tool" and len(text) > 2800 and start >= 2000 and end <= len(text) - 800),
                  f"outside clip zone {f['id']}")
        else:
            check(end <= 2000 if placement == "head" else start >= len(text) - 800,
                  f"incorrect {placement} placement {f['id']}")
        if f["stale"]:
            old = source(dict(f["stale_source"], value=f["stale"]))
            check(f["stale"] in old["content"] and f["stale_source"]["row_index"] < f["row_index"], "stale ordering")
            check(not any(f["value"] in r["content"] for r in rows[:f["row_index"]]), "current value precedes supersession")
    check(len(admissions) == 60 and {a["id"] for a in admissions} == {f["id"] for f in facts}, "admission coverage")
    by_id = {f["id"]: f for f in facts}
    for a in admissions:
        row, f = source(a), by_id[a["id"]]
        check(a["row_index"] == f["row_index"] and a["value"] == f["value"] and
              a["role"] == row["role"] and a["tool_call_id"] == row["tool_call_id"], "admission mismatch")
    items = facts + manifest["continuity"] + [state]
    check({c["id"].split("CONT-")[-1] for c in manifest["continuity"]} ==
          {"current_request", "active_constraint", "host_instruction"}, "continuity coverage")
    for item in manifest["continuity"]:
        check(item["value"] in source(item)["content"], "continuity text")
    check(all(str(state[k]) in source(state)["content"] for k in ("next_action", "path", "status", "decision_id")), "continuation state")
    last_item = max(i["row_index"] for i in items)
    check(prefix[last_item] == manifest["presented_tokens"], "presentation checkpoint mismatch")
    supersession = max(prefix[f["row_index"]] for f in facts if f["stale"])
    spans = (total - supersession - 20000) // trigger
    if v4:
        check(total >= trigger and spans >= events, "v4 token/event floor")
        check(total - prefix[last_item] >= 20000, "tail missing")
        check(max(r["turn"] for r in rows) == manifest["params"]["turns"], "v4 turn horizon")
        if smoke:
            suffix = manifest["smoke_suffix"]
            source(suffix)
            check(suffix["row_index"] > last_item and suffix["action"] == "force_compaction_after_row" and
                  suffix["timing_population"] == "WIRING-ONLY", "event-producing smoke suffix")
            check(manifest["decision_checkpoint"]["row_index"] >= suffix["row_index"], "smoke replay excludes the suffix")
    elif not smoke:
        check(prefix[last_item] < trigger, "scored item presented after slowest trigger")
        check(total >= max(400000, trigger + 20000), "total/tail token floor")
        check(spans >= events, "insufficient post-supersession trigger spans")
        trigger_row = next(i for i, n in enumerate(prefix) if n >= trigger)
        decision = next(c for c in manifest["checkpoints"] if prefix[c["row_index"]] >= prefix[trigger_row] + 20000)
        check(manifest["decision_checkpoint"]["row_index"] == decision["row_index"] and
              prefix[decision["row_index"]] >= prefix[last_item], "first common decision checkpoint")
        check(total - prefix[last_item] >= 20000, "tail missing")
    else:
        check({r["turn"] for r in rows} == set(range(1, 11)), "smoke must have ten turns")
        suffix = manifest["smoke_suffix"]
        source(suffix)
        check(suffix["row_index"] > last_item and suffix["action"] == "force_compaction_after_row" and
              suffix["timing_population"] == "WIRING-ONLY", "event-producing smoke suffix")
    targets = manifest["receipt_targets"]
    check({t["id"] for t in targets if t["kind"] == "ancestry"} == {f["id"] for f in facts if f["stale"]}, "ancestry coverage")
    external = [t for t in targets if t["kind"] == "externalization"]
    check(len(external) == 1, "externalization target required")
    for t in targets:
        row = source(t)
        check(t["row_index"] == by_id[t["id"]]["row_index"], "target source mismatch")
        if t["kind"] == "externalization":
            check(row["role"] == "tool" and len(row["content"]) > 12000 and count(row["content"]) > 25000,
                  "externalization thresholds")
        elif t["kind"] == "ancestry":
            check(t["required_events"] == events, "ancestry event count mismatch")
    check(load("canaries.json") == facts, "canary answer key mismatch")
    traps = load("traps.json")
    check(len(traps) == 5 and all(t["answer"] == "ABSTAIN" for t in traps), "trap answer mismatch")
    for trap in traps:
        name = trap["probe"].split("for fixture ", 1)[1].removesuffix("?")
        check(not any(name in r["content"] or trap["probe"] in r["content"] for r in rows),
              "trap leaked into transcript")
    expected = {f["id"]: dict(id=f["id"], kind="canary", text=f["probe"], expect="value") for f in facts}
    expected.update({t["id"]: dict(id=t["id"], kind="trap", text=t["probe"], expect="ABSTAIN") for t in traps})
    probes = lines("probes.jsonl")
    check(len(probes) == len(expected) == 65 and
          {p["id"]: p for p in probes} == expected, "probe facts/traps mismatch")
    batches = [dict(id=f"S{manifest['seed']}-B{k // 10}", probes=probes[k:k + 10],
                    text=_gen.BATCH_INSTRUCTION) for k in range(0, len(probes), 10)]
    check(lines("probe_batches.jsonl") == batches, "probe batch mismatch")
    check(lines("turns.jsonl") == [dict(turn=r["turn"], text=r["content"], role=r["role"], id=r["id"],
                                       tool_call_id=r["tool_call_id"], tool_calls=r.get("tool_calls", []))
                                   for r in rows], "turn projection mismatch")
    for name, digest in manifest["shas"].items():
        check(_gen._sha256(directory / name) == digest, f"digest mismatch: {name}")
    biases = _verify_v4(rows, facts, traps, manifest, lines("lifecycle_probes.jsonl"), prefix, count, check) if v4 else {}
    return dict(bias_receipts=biases, status="PASS", mode=manifest["mode"], rows=len(rows), tokens=total,
                decision_checkpoint_tokens=prefix[manifest["decision_checkpoint"]["row_index"]],
                decision_checkpoint_rows=manifest["decision_checkpoint"]["row_index"] + 1,
                tool_result_token_share=round(sum(n for n, r in zip(counts, rows) if r["role"] == "tool") / total, 6),
                results_over_12000_chars=sum(r["role"] == "tool" and len(r["content"]) > 12000 for r in rows),
                class_placement_role=dict(sorted(Counter(f"{f['class']}|{f['placement']}|{f['row_role']}" for f in facts).items())),
                planned_post_supersession_trigger_spans=spans if v4 or not smoke else None,
                tokenizer=manifest["tokenizer"], runtime_receipts="UNTESTED: events, ancestry, externalization, admission",
                smoke_exemptions="full horizon, tail and natural event spans" if smoke and not v4 else None)



def _verify_v4(rows, facts, traps, manifest, probes, prefix, count, check):
    guesses, shared = {}, {}
    for cls in _gen.CLASSES:
        values = [f["value"] for f in facts if f["class"] == cls]
        fragments = Counter(g for v in values for g in {v.casefold()[i:i + 6] for i in range(len(v) - 5)})
        shared[cls] = max(fragments.values(), default=0) / len(values)
        check(shared[cls] <= .2, f"shared value substring: {cls}")
        hits = 0
        for i, value in enumerate(values):
            others = values[:i] + values[i + 1:]
            prediction = os.path.commonprefix(others) + os.path.commonprefix([v[::-1] for v in others])[::-1]
            hits += prediction == value
        guesses[cls] = dict(exact=hits, total=len(values), rate=hits / len(values))
        check(guesses[cls]["rate"] <= .02, f"guessable values: {cls}")
    user_indices = [i for i, r in enumerate(rows) if r["role"] == "user"]
    users = [f for f in facts if f["row_role"] == "user"]
    histogram = Counter(min(2, user_indices.index(f["row_index"]) * 3 // len(user_indices)) for f in users)
    row_histogram = Counter(min(2, f["row_index"] * 3 // len(rows)) for f in users)
    token_histogram = Counter(min(2, prefix[f["row_index"]] * 3 // prefix[-1]) for f in users)
    check(all(.25 <= histogram[i] / len(users) <= .4 for i in range(3)), "user-third placement bias")
    check(all(.25 <= h[i] / len(users) <= .4 for h in (row_histogram, token_histogram) for i in range(3)),
          "transcript-third placement bias")
    middle = sum(f["placement"] == "middle" and count(rows[f["row_index"]]["content"]) >= 2000 and
                 .3 <= f["char_offset"] / len(rows[f["row_index"]]["content"]) <= .7 for f in users)
    check(middle / len(users) >= .2, "long user middle coverage")
    check(Counter(f["row_role"] for f in facts) == Counter(user=22, assistant=11, tool=27), "v3 role mix")
    requests = manifest["lifecycle"]
    statuses = Counter(r["status"] for r in requests)
    check(len(requests) >= 6 and all(statuses[s] for s in ("completed", "cancelled", "superseded", "pending")),
          "request lifecycle coverage")
    check(len({r["task"] for r in requests}) == len(requests) and
          all(r["replacement"] != r["task"] for r in requests), "request lifecycle identity")
    check(not any(f["stale"] and "correction_source" in f for f in facts), "corrected fact also superseded")
    for r in requests:
        check(r["task"] in rows[r["row_index"]]["content"] and r["row_id"] == rows[r["row_index"]]["id"], "request source")
        if r["status"] != "pending":
            resolution = r["resolution"]
            text = rows[resolution["row_index"]]["content"]
            check(resolution["row_index"] > r["row_index"] and r["task"] in text, "request resolution")
            check((r["status"] == "completed" and "Completed " in text) or
                  (r["status"] == "cancelled" and "don't do it" in text and r["replacement"] in text) or
                  (r["status"] == "superseded" and "Instead of " in text and r["replacement"] in text), "request status text")
    kinds = Counter(p["kind"] for p in probes)
    check(set(kinds) == {"stale_task", "corrected_value", "current_request"}, "lifecycle probe kinds")
    check({p["compaction_horizon"] for p in probes} == {1, 3, 5} and manifest["default_leaf_tokens"] == 8000,
          "lifecycle horizon coverage")
    for p in probes:
        index = p["row_index"]
        check(p["row_id"] == rows[index]["id"] and p["source_token_position"] == prefix[index] and
              p["probe_token_position"] - prefix[index] == p["compaction_horizon"] * manifest["default_leaf_tokens"] and
              p["probe_token_position"] <= prefix[-1], "lifecycle token horizon")
        if p["kind"] == "corrected_value":
            f = next(f for f in facts if p["id"] == f["id"] + "-CORRECTION")
            old = f["correction_source"]
            check(old["row_index"] < index and old["value"] != f["value"] and
                  old["value"] in rows[old["row_index"]]["content"] and
                  count(rows[index]["content"]) <= 20 and p["answer"] == f["value"], "short correction")
        elif p["kind"] == "stale_task":
            r = next(r for r in requests if r.get("resolution", {}).get("row_id") == p["row_id"])
            check(p["answer"] == f"No; do {r['replacement']}.", "stale-task answer")
        else:
            current = next(c for c in manifest["continuity"] if c["id"].endswith("current_request"))
            check(p["answer"] == current["value"] and p["row_id"] == current["row_id"], "current request answer")
    for c in manifest["continuity"]:
        if c["id"].endswith("host_instruction"):
            continue
        positions = [i for i, r in enumerate(rows) if c["id"] in r["content"] and c["value"] in r["content"]]
        check(len(positions) >= 2 and .3 <= positions[0] / len(rows) <= .7 and positions[-1] > positions[0],
              "mid-session continuity/restatement")
    by_id = {f["id"]: f for f in facts}
    for trap in traps:
        sibling = by_id[trap["sibling_id"]]
        name = trap["probe"].split("for fixture ", 1)[1][:-1]
        check(trap["class"] == sibling["class"] and
              trap["probe"].replace(name, "{entity}") == sibling["probe"].replace(sibling["fixture"], "{entity}"),
              "trap template mismatch")
    return dict(guessability=guesses, max_shared_sixgram_fraction=shared,
                placement_histogram=dict(histogram), transcript_row_histogram=dict(row_histogram),
                token_placement_histogram=dict(token_histogram),
                user_facts=len(users), long_user_middle=middle, lifecycle_counts=dict(statuses),
                lifecycle_probe_counts=dict(kinds), trap_template_check="PASS", traps=len(traps))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", type=Path)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--min-events", type=int)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.material, args.min_tokens, args.min_events), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps(dict(status="FAIL", error=str(exc))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
