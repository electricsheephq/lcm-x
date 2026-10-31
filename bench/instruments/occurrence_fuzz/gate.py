"""Compare occurrence ledgers against a reference, by base-row metadata class."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

Record = dict[str, Any]
Totals = defaultdict[str, defaultdict[str, int]]


def load(path: str) -> dict[int, Record]:
    with open(path) as fh:
        return {row["seed"]: row for row in map(json.loads, fh)}


def cls(masks: str) -> str:
    return (
        {"n": "none", "s": "stamp", "u": "uid"}[masks[0]]
        if len(set(masks)) == 1
        else "mixed"
    )


def key_of(row: Record) -> str:
    return f"{row['role']}|{row['content']}"


def score(
    seqs: dict[int, Record],
    reference: dict[int, Record],
    ledgers: Sequence[tuple[str, dict[int, Record]]],
) -> tuple[Totals, int]:
    totals: Totals = defaultdict(lambda: defaultdict(int))
    errors = 0
    for seed, seq in seqs.items():
        category = cls(seq["masks"])
        resend_keys = {
            key_of(seq["rows"][op.split(":", 1)[1]])
            for op in seq["ops"]
            if op.startswith("resend:")
        }
        ref_record = reference.get(seed)
        if ref_record is None or ref_record.get("err"):
            # A short or missing reference ledger would under-count LOSS.
            errors += 1
            continue
        ref = ref_record["ledger"]
        for name, tree in ledgers:
            result = tree.get(seed)
            if result is None or result.get("err"):
                errors += 1
                continue
            ledger = result["ledger"]
            dup = [
                key for key, count in ledger.items() if count > max(1, ref.get(key, 0))
            ]
            totals[category][f"{name}_dup"] += len(dup)
            totals[category][f"{name}_dup_nonresend"] += sum(
                key not in resend_keys for key in dup
            )
            totals[category][f"{name}_loss"] += sum(key not in ledger for key in ref)
        totals[category]["seqs"] += 1
    return totals, errors


def report(totals: Totals, errors: int, names: Sequence[str]) -> bool:
    # An empty or fully errored input exercised nothing, so it cannot pass.
    ok = any(counts["seqs"] for counts in totals.values())
    for category in sorted(totals):
        counts = totals[category]
        name = names[-1]
        passed = (counts[f"{name}_loss"] == 0 and
                  counts[f"{name}_dup_nonresend"] <= counts["base_dup_nonresend"])
        ok &= passed
        fields = [
            " ".join(f"{name} {counts[f'{name}_{metric}']}" for name in reversed(names))
            for metric in ("loss", "dup", "dup_nonresend")
        ]
        print(
            f"{category:6s} seqs {counts['seqs']:5d} | LOSS {fields[0]} | "
            f"DUP literal {fields[1]} | DUP non-resend {fields[2]} | {'PASS' if passed else 'FAIL'}"
        )
    print("errors", errors, "GATE", "PASS" if ok and errors == 0 else "FAIL")
    return ok and errors == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seqs", metavar="SEQS")
    parser.add_argument("reference", metavar="REFERENCE")
    parser.add_argument("base", metavar="BASE")
    parser.add_argument("candidates", metavar="CANDIDATE", nargs="+")
    args = parser.parse_args()
    names = ["base"] + [f"candidate{i}" for i in range(1, len(args.candidates) + 1)]
    totals, errors = score(
        load(args.seqs),
        load(args.reference),
        [
            (name, load(path))
            for name, path in zip(names, [args.base, *args.candidates])
        ],
    )
    report(totals, errors, names)


if __name__ == "__main__":
    main()
