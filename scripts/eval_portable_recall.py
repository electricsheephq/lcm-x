#!/usr/bin/env python3
"""Explicit public/synthetic-only frozen ranking evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.portable_recall_eval import (
    capture_public, capture_synthetic, evaluate, read_frozen, synthetic_cases, write_frozen,
)
from optional_scorer import BGE_MODEL, BGE_REVISION, JEV_MODEL, JevBudget, LocalBGEScorer, PublicJevScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-synthetic")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--posture", choices=("OFF", "LOCAL"), required=True)
    freeze.add_argument("--local-model", default="")
    freeze.add_argument("--source-identity", required=True, help="immutable source commit and embedding identity")
    public = commands.add_parser("freeze-public")
    public.add_argument("--dataset", type=Path, required=True, help="pinned downloaded longmemeval_s")
    public.add_argument("--dataset-sha256", required=True, help="checksum from the pinned download receipt")
    public.add_argument("--baseline-output", type=Path, required=True)
    public.add_argument("--output", "--frozen-output", dest="output", type=Path, required=True,
                        help="frozen public pool output")
    public.add_argument("--dev-corpus", type=Path, required=True, help="previously frozen synthetic registration")
    public.add_argument("--posture", choices=("OFF", "LOCAL"), required=True)
    public.add_argument("--local-model", default="")
    public.add_argument("--source-identity", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    run.add_argument("--scorer", action="append", choices=("bge", "jev"), default=[])
    run.add_argument("--timeout", type=float, default=1.0)
    run.add_argument("--allow-public-jev", action="store_true")
    run.add_argument("--jev-ledger", type=Path)
    run.add_argument("--jev-min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve frozen evidence")
    if args.command == "freeze-public":
        if args.baseline_output.exists() or args.baseline_output.resolve() == args.output.resolve():
            parser.error("baseline and frozen outputs must be distinct, new files")
        if {args.baseline_output.resolve(), args.output.resolve()} & {args.dataset.resolve(), args.dev_corpus.resolve()}:
            parser.error("outputs must not alias input files")
        dev_header, dev_cases = read_frozen(args.dev_corpus)
        expected = {f["case_id"]: f["split"] for f in synthetic_cases()}
        if (dev_header["input_kind"] != "synthetic" or dev_header["posture"] != args.posture
                or {c.case_id: c.split for c in dev_cases} != expected):
            parser.error("dev registration must be the same-posture frozen 120-case synthetic corpus")
        with tempfile.TemporaryDirectory(prefix="lcm-public-holdout-") as directory:
            cases, baseline = capture_public(
                args.dataset, expected_sha256=args.dataset_sha256, tmp_dir=Path(directory),
                posture=args.posture, local_model=args.local_model, dev_header=dev_header,
            )
        baseline["source_identity"] = args.source_identity
        header = write_frozen(args.output, cases, input_kind="public", posture=args.posture,
                              source_identity=args.source_identity, provenance=baseline["provenance"])
        baseline["frozen_corpus_sha256"] = header["corpus_sha256"]
        with args.baseline_output.open("x", encoding="utf-8") as stream:
            json.dump(baseline, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"corpus_sha256": header["corpus_sha256"], **baseline["provenance"]}))
        return
    if args.command == "freeze-synthetic":
        with tempfile.TemporaryDirectory(prefix="lcm-public-synthetic-") as directory:
            cases = capture_synthetic(tmp_dir=Path(directory), posture=args.posture,
                                      local_model=args.local_model)
        header = write_frozen(args.output, cases, input_kind="synthetic",
                              posture=args.posture, source_identity=args.source_identity)
        print(json.dumps(header, sort_keys=True))
        return
    header, cases = read_frozen(args.input)
    scorers, unavailable, identities = {}, {}, {}
    budget = None
    for name in dict.fromkeys(args.scorer):
        if name == "bge":
            identities[name] = {"model": BGE_MODEL, "revision": BGE_REVISION}
            try:
                scorers[name] = LocalBGEScorer()
            except Exception:
                unavailable[name] = "cached model or optional dependency unavailable; no model row measured"
        else:
            if not args.allow_public_jev or args.jev_ledger is None:
                parser.error("Jev requires --allow-public-jev and a dedicated --jev-ledger")
            budget = JevBudget(args.jev_ledger)
            scorers[name] = PublicJevScorer(
                input_kind=header["input_kind"], allow_public_egress=True,
                budget=budget, min_confidence=args.jev_min_confidence,
            )
            identities[name] = {"model": JEV_MODEL, "min_confidence": args.jev_min_confidence}
    report = evaluate(cases, scorers, split=args.split, timeout_s=args.timeout)
    report.update({"corpus": header, "model_identities": identities,
                   "unavailable_arms": unavailable})
    if budget:
        report["jev_budget"] = budget.summary()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "corpus_sha256": header["corpus_sha256"]}))


if __name__ == "__main__":
    main()
