#!/usr/bin/env python3
"""RUN-SHEET-V1M-REBANK §4.4(b)/§8 step 9 — per-shard embed-cache pair parity vs F53, recomputed from rows.

For each re-bank shard k: sum the per-question `embed_cache` {hits, misses} rows in per_question_checkpoint.jsonl
(the record; the run aggregate is derived), check the run's own `ingest.embed_cache` aggregate equals that sum, and
compare with F53's `lme-runs/m-full2-shard-k/longmemeval_metrics.json` `ingest.embed_cache`. Also exports the forward
baseline (per-question corpus_counts + embed_cache) so future rows have a per-question parity field.

Usage: cache_pair_check.py [--rebank-root /Users/m1/hermes-work/lme-runs/m-rebank-shard-] [--f53-root .../m-full2-shard-]
                           [--shards 6] [--out <dir>]
Exit 0 = every shard matches (hits AND misses, and aggregate == row sum). Exit 1 = any mismatch (table printed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKPOINT = "per_question_checkpoint.jsonl"
METRICS = "longmemeval_metrics.json"


def row_sums(checkpoint: Path) -> tuple[int, int, int, list[dict]]:
    hits = misses = rows = 0
    baseline: list[dict] = []
    with checkpoint.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "question_id" not in record:
                continue  # header line
            rows += 1
            pair = record.get("embed_cache") or {}
            h, m = int(pair.get("hits", 0)), int(pair.get("misses", 0))
            hits += h
            misses += m
            baseline.append(
                {
                    "question_id": record["question_id"],
                    "corpus_counts": record.get("corpus_counts"),
                    "embed_cache": {"hits": h, "misses": m},
                    "privacy": record.get("privacy"),
                }
            )
    return hits, misses, rows, baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebank-root", default="/Users/m1/hermes-work/lme-runs/m-rebank-shard-")
    ap.add_argument("--f53-root", default="/Users/m1/hermes-work/lme-runs/m-full2-shard-")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--out", default="/Users/m1/Codex/session-notes/2026-09-05/v1m-rebank/artifacts/cache-pair")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok = True
    table = []
    total_new = total_f53 = total_misses = 0
    for k in range(args.shards):
        rb = Path(f"{args.rebank_root}{k}")
        f53 = Path(f"{args.f53_root}{k}")
        try:
            hits, misses, rows, baseline = row_sums(rb / CHECKPOINT)
            agg = json.loads((rb / METRICS).read_text())["ingest"]["embed_cache"]
            ref = json.loads((f53 / METRICS).read_text())["ingest"]["embed_cache"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            table.append((k, "ERROR", str(exc)))
            ok = False
            continue
        agg_ok = int(agg.get("hits", -1)) == hits and int(agg.get("misses", -1)) == misses
        ref_ok = int(ref["hits"]) == hits and int(ref["misses"]) == misses
        # §4.4(b): misses must be ZERO on both sides — equal non-zero miss counts are not parity (PR #416 review): a miss is a
        # provider request the pre-spend gate never accounted for.
        miss_ok = misses == 0 and int(ref["misses"]) == 0
        ok = ok and agg_ok and ref_ok and miss_ok
        total_new += hits
        total_f53 += int(ref["hits"])
        total_misses += misses + int(ref["misses"])
        (out / f"forward-baseline-shard-{k}.json").write_text(json.dumps(baseline, indent=1, sort_keys=True))
        table.append(
            (
                k,
                "PASS" if (agg_ok and ref_ok and miss_ok) else "FAIL",
                f"rows={rows} row_sum={hits}/{misses} aggregate={agg.get('hits')}/{agg.get('misses')} "
                f"F53={ref['hits']}/{ref['misses']} aggregate==rows:{agg_ok} ==F53:{ref_ok} zero-misses:{miss_ok}",
            )
        )

    for k, status, detail in table:
        print(f"shard-{k} {status} {detail}")
    print(f"TOTAL hits new={total_new} F53={total_f53} misses(new+F53)={total_misses} (F53 registered total 2,361,553; misses must be 0 everywhere)")
    verdict = "PASS" if ok else "FAIL"
    print(f"CACHE-PAIR PARITY: {verdict}")
    (out / "verdict.txt").write_text(verdict + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
