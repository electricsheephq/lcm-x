#!/usr/bin/env python3
"""Validate Track S material offline; runtime receipts remain a separate gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s3_generator", Path(__file__).with_name("gen_material.py"))
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def verify(directory: Path, min_tokens: int | None = None, min_events: int | None = None) -> dict:
    def load(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    def check(ok, message):
        if not ok:
            raise ValueError(message)

    rows = [json.loads(line) for line in (directory / "transcript.jsonl").read_text().splitlines()]
    facts, manifest, state = load("facts.json"), load("material.manifest.json"), load("continuation.json")
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
    check(Counter(f["class"] for f in facts) == Counter({c: 5 for c in _gen.CLASSES}), "12 classes x 5 required")
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
            check(row["role"] == "tool" and len(text) > 2800 and start >= 2000 and end <= len(text) - 800,
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
    if not smoke:
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
    for name, digest in manifest["shas"].items():
        check(_gen._sha256(directory / name) == digest, f"digest mismatch: {name}")
    return dict(status="PASS", mode=manifest["mode"], rows=len(rows), tokens=total,
                decision_checkpoint_tokens=prefix[manifest["decision_checkpoint"]["row_index"]],
                decision_checkpoint_rows=manifest["decision_checkpoint"]["row_index"] + 1,
                tool_result_token_share=round(sum(n for n, r in zip(counts, rows) if r["role"] == "tool") / total, 6),
                results_over_12000_chars=sum(r["role"] == "tool" and len(r["content"]) > 12000 for r in rows),
                class_placement_role=dict(sorted(Counter(f"{f['class']}|{f['placement']}|{f['row_role']}" for f in facts).items())),
                planned_post_supersession_trigger_spans=spans if not smoke else None,
                tokenizer=manifest["tokenizer"], runtime_receipts="UNTESTED: events, ancestry, externalization, admission",
                smoke_exemptions="full horizon, tail and natural event spans" if smoke else None)


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
