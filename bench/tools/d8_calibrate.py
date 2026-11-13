"""Offline D8 placebo recount; counts only, never a release-gate certificate."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.tools.d8_guard import PINNED_SIZES, RULE_ID, _category, bootstrap_bounds, compare_runs, count_margin


CATEGORIES = {"longmemeval": ("abstention", "knowledge-update"), "locomo": ("adversarial",)}


def unread(reason):
    return {"status": "UNREAD", "reason": reason, "n": None, "b": None, "c": None,
            "net": None, "margin": None, "count_pass": None, "interval_pass": None,
            "pass_count": False, "pass_interval": False}


def read_receipt(path, dataset):
    """Recount verdicts, never manufacture question types or recall-health rows.

    These receipts have no per-id question type. Only abstention is identifiable
    without the source dataset: the guard defines it by the question-id suffix.
    Receipt completeness is disclosed evidence, not a reconstructed D8 certificate.
    """
    receipt = json.loads(Path(path).read_text())
    if receipt["bench"] != dataset:
        raise ValueError("dataset mismatch")
    a, aprime = receipt["arm"], receipt["aprime"]
    base, candidate = a["outcomes"], aprime["outcomes"]
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        raise ValueError("outcome maps required")
    rows = {}
    for name in CATEGORIES[dataset]:
        if name != "abstention":
            filename = "longmemeval_s_cleaned.json" if dataset == "longmemeval" else "locomo10.json"
            rows[name] = unread(f"missing question-id/category mapping: pinned {filename} would settle it")
            continue
        # This measurement-only row.json path pairs answers and applies the placebo test itself;
        # pairing/count logic shares only _category and count_margin with the guard.
        ids = sorted(qid for qid in base if _category({"questionId": qid}, name))
        values = [int(candidate.get(qid) == "correct") - int(base[qid] == "correct") for qid in ids]
        n, b, c = len(ids), values.count(1), values.count(-1)
        lower, upper = bootstrap_bounds([[value] for value in values])
        complete = (n == PINNED_SIZES[name] and base.keys() == candidate.keys()
                    and len(base) == {"longmemeval": 500, "locomo": 1986}[dataset]
                    and receipt.get("status") == "COMPLETE"
                    and all(side.get("status") == side.get("health_status") == "COMPLETE"
                            and isinstance(side.get("accounting"), dict)
                            and side["accounting"].get("failed_including_missing") == 0
                            for side in (a, aprime)))
        margin = count_margin(n)
        count_pass = abs(b - c) < margin
        interval_pass = lower is not None and lower > -2 and upper < 2
        rows[name] = {"status": "READ" if complete else "INCOMPLETE", "n": n, "b": b, "c": c,
                      "net": b - c, "margin": margin, "count_pass": count_pass,
                      "interval_pass": interval_pass, "pass_count": complete and n >= margin and count_pass,
                      "pass_interval": complete and n >= margin and interval_pass}
    return rows


def calibrate(pairs):
    """Use the scorer's pairing, categories, completeness and interval unchanged.

    A-prime is candidate; A is baseline. Raw predicates are disclosed separately
    from passes: incomplete and non-gating categories never count as passes.
    Error details are deliberately not echoed: malformed inputs can contain text.
    """
    results, summary = [], {}
    for pair in pairs:
        pair_id, dataset, config, a, aprime = pair
        if dataset not in CATEGORIES or config not in {"OFF", "LOCAL"}:
            raise ValueError("expected dataset longmemeval/locomo and configuration OFF/LOCAL")
        item = {"id": pair_id, "dataset": dataset, "configuration": config, "categories": {},
                "layout": "row.json" if Path(a).name == "row.json" else "scorer run reports"}
        try:
            if item["layout"] == "row.json":
                if Path(a).resolve() != Path(aprime).resolve():
                    raise ValueError("one paired receipt required")
                item["categories"] = read_receipt(a, dataset)
                scored = None
            else:
                scored = compare_runs(aprime, a, placebo=True)
            if scored is not None and set(scored["categories"]) != set(CATEGORIES[dataset]):
                raise ValueError("dataset mismatch")
            if scored is not None and any(value != config for value in scored["recall_configs"].values()):
                raise ValueError("configuration mismatch")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            item.update(status="UNREAD", reason=f"scorer input unreadable ({type(error).__name__})")
            scored = None
        else:
            item["status"] = "INCOMPLETE" if scored and scored["INCOMPLETE"] else "READ"
        for name in CATEGORIES[dataset]:
            key = (config, name)
            totals = summary.setdefault(key, {"configuration": config, "category": name,
                "pairs": 0, "unread": 0, "incomplete": 0, "count_passes": 0, "interval_passes": 0})
            totals["pairs"] += 1
            if name in item["categories"]:
                row = item["categories"][name]
                totals["unread"] += row["status"] == "UNREAD"
                totals["incomplete"] += row["status"] == "INCOMPLETE"
                totals["count_passes"] += row["pass_count"]
                totals["interval_passes"] += row["pass_interval"]
                continue
            if scored is None:
                totals["unread"] += 1
                item["categories"][name] = unread(item["reason"])
                continue
            category = scored["categories"][name]
            complete = not scored["INCOMPLETE"] and not category["INCOMPLETE"]
            eligible = complete and category["gating"]
            row = {"status": "READ" if complete else "INCOMPLETE", "n": category["paired_n"],
                "b": category["b"], "c": category["c"], "net": category["net"],
                "margin": category["margin_questions"], "count_pass": category["count_pass"],
                "interval_pass": category["interval_inside_band"],
                "pass_count": eligible and category["count_pass"],
                "pass_interval": eligible and category["interval_inside_band"]}
            item["categories"][name] = row
            totals["incomplete"] += not complete
            totals["count_passes"] += row["pass_count"]
            totals["interval_passes"] += row["pass_interval"]
        if any(row["status"] == "INCOMPLETE" for row in item["categories"].values()):
            item["status"] = "INCOMPLETE"
        elif item["status"] == "READ" and any(row["status"] == "UNREAD" for row in item["categories"].values()):
            item["status"] = "PARTIAL" if any(
                row["status"] != "UNREAD" for row in item["categories"].values()) else "UNREAD"
            item["reason"] = "question-id/category metadata unavailable"
        results.append(item)
    return {"rule_id": RULE_ID, "claim": "offline recorded-run recount; not a gate verdict",
            "orientation": "b: A-prime only correct; c: A only correct",
            "pairs": results, "summary": [summary[key] for key in sorted(summary)]}


def markdown(result):
    lines = ["# D8 offline calibration", "", result["claim"], "",
             "b = A-prime only correct; c = A only correct. Raw predicates are distinct from eligible passes.",
             "INCOMPLETE and UNREAD pairs are never counted as passing either rule.", "",
             "| Pair id | Layout | Config | Category | Status | n | b | c | b-c | T(n) | Count predicate | Interval predicate |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for pair in result["pairs"]:
        for name, row in pair["categories"].items():
            values = [pair["id"], pair["layout"], pair["configuration"], name, row["status"],
                      *(row[key] for key in ("n", "b", "c", "net", "margin", "count_pass", "interval_pass"))]
            lines.append("| " + " | ".join("—" if v is None else str(v) for v in values) + " |")
    lines += ["", "| Config | Category | Pairs | Unread | Incomplete | Count passes | Interval passes |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "configuration", "category", "pairs", "unread", "incomplete", "count_passes", "interval_passes")) + " |")
    for pair in result["pairs"]:
        for name, row in pair["categories"].items():
            if row["status"] == "UNREAD":
                lines.append(f"\n{pair['id']} / {name}: {row['reason']}.")
    return "\n".join(lines) + "\n"


def trinomial(n, gain, loss):
    """Exact (up to float rounding) distribution of b,c; all n questions retained."""
    if n < 0 or gain < 0 or loss < 0 or gain + loss > 1:
        raise ValueError("invalid trinomial probabilities")
    for b in range(n + 1):
        for c in range(n - b + 1):
            probability = math.comb(n, b) * math.comb(n - b, c)
            probability *= gain ** b * loss ** c * (1 - gain - loss) ** (n - b - c)
            yield b, c, probability


@lru_cache(maxsize=None)
def interval_rejects(n, b, c):
    """Original small-category method: exact single-question bootstrap distribution.

    The lower 2.5% quantile is <= -2 points iff that tail has mass >= .025.
    This is the population bootstrap, not a new finite-seed bootstrap path.
    Power is one-sided candidate detection, distinct from two-sided placebo rejects.
    """
    return math.fsum(p for x, y, p in trinomial(n, b / n, c / n)
                     if 100 * (x - y) <= -2 * n) >= .025


def loss_power(n, discordance, shift, margin):
    """Original redesign's trinomial model with a fraction forced to the loss side.

    Other questions have independent symmetric discordance. Thus E[(c-b)/n]
    is exactly shift; 17 points is .17, not five fixed losses rounded from 5.1.
    """
    gain = (1 - shift) * discordance / 2
    loss = shift + gain
    outcomes = list(trinomial(n, gain, loss))
    return {"count": math.fsum(p for b, c, p in outcomes if c - b >= margin),
            "interval": math.fsum(p for b, c, p in outcomes if interval_rejects(n, b, c))}


def margin_sweep(result):
    """Substitute margins only here; the guard and its eligibility rules stay untouched."""
    rows = []
    for total in result["summary"]:
        config, name = total["configuration"], total["category"]
        available = [(pair["id"], pair["categories"][name]) for pair in result["pairs"]
                     if pair["configuration"] == config and name in pair["categories"]]
        readable = [(pid, row) for pid, row in available if row["status"] == "READ"]
        interval_eligible = [(pid, row) for pid, row in readable if row["n"] >= count_margin(row["n"])]
        interval_failures = [pid for pid, row in interval_eligible if not row["interval_pass"]]
        for m in (3, 4, 5, 6):
            eligible = [(pid, row) for pid, row in readable if row["n"] >= max(m, (2 * row["n"] + 99) // 100)]
            failures = [pid for pid, row in eligible
                        if abs(row["net"]) >= max(m, (2 * row["n"] + 99) // 100)]
            row = {"configuration": config, "category": name, "margin_floor": m,
                   "pairs": len(available), "eligible": len(eligible), "unread": total["unread"],
                   "incomplete": total["incomplete"], "false_fails": len(failures), "failing_pairs": failures,
                   "interval_eligible": len(interval_eligible),
                   "interval_false_fails": len(interval_failures), "interval_failing_pairs": interval_failures,
                   "power": None}
            # The requested effects are abstention effects; other categories have no such power claim.
            if name == "abstention" and eligible:
                n = PINNED_SIZES[name]
                discordance = sum(r["b"] + r["c"] for _, r in eligible) / sum(r["n"] for _, r in eligible)
                margin = max(m, (2 * n + 99) // 100)
                row["power"] = {"n": n, "pooled_discordance": discordance, "margin": margin,
                                **{str(points): loss_power(n, discordance, points / 100, margin)
                                   for points in (10, 17)}}
            rows.append(row)
    return {"claim": "offline same-build recount and modelled power; not a margin choice or gate verdict",
            "method": "exact trinomial; per-configuration pooled readable complete abstention discordance p; "
                      "force fraction s to c, remainder symmetric: P(b)=(1-s)*p/2, P(c)=s+P(b); "
                      "one-sided candidate FAIL power at s=.10/.17; interval uses exact population-bootstrap "
                      "lower 2.5% quantile <= -2 points, as the original small-category analysis; "
                      "recorded false-fails use the guard's two-sided placebo predicates; "
                      "UNREAD/INCOMPLETE excluded, never treated as passes or failures",
            "rows": rows}


def sweep_markdown(result):
    lines = ["# D8 margin sweep", "", result["claim"], "", result["method"], "",
             "Power uses the original redesign's independent-question model; it is not empirical detection.",
             "Other categories' power is N/A: the requested effects are abstention losses.", "",
             "| Config | Category | m | Eligible | Unread | Incomplete | Count false-fails | Interval false-fails | Count power 10 / 17 pt | Interval power 10 / 17 pt |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in result["rows"]:
        power = row["power"]
        powers = [" / ".join(f"{power[str(p)][rule]:.4%}" for p in (10, 17)) if power else "N/A"
                  for rule in ("count", "interval")]
        values = [*(row[k] for k in ("configuration", "category", "margin_floor", "eligible", "unread",
                                    "incomplete", "false_fails", "interval_false_fails")), *powers]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Failing recorded pairs (two-sided placebo)", ""]
    for row in result["rows"]:
        lines.append(f"{row['configuration']} / {row['category']} / m={row['margin_floor']}: "
                     f"count: {', '.join(row['failing_pairs']) or 'none'}; "
                     f"interval: {', '.join(row['interval_failing_pairs']) or 'none'}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", nargs=5, required=True,
                        metavar=("ID", "DATASET", "CONFIG", "A", "A_PRIME"),
                        help="repeat for each same-build pair; CONFIG is OFF or LOCAL")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--sweep-json-output", type=Path)
    parser.add_argument("--sweep-markdown-output", type=Path)
    args = parser.parse_args(argv)
    if bool(args.sweep_json_output) != bool(args.sweep_markdown_output):
        parser.error("both sweep outputs required together")
    try:
        result = calibrate(args.pair)
    except ValueError:
        parser.error("invalid dataset or configuration")
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(markdown(result))
    if args.sweep_json_output:
        sweep = margin_sweep(result)
        args.sweep_json_output.write_text(json.dumps(sweep, indent=2) + "\n")
        args.sweep_markdown_output.write_text(sweep_markdown(sweep))
    print(markdown(result), end="")
    # Rule rejects describe the data; unread inputs make the measurement partial.
    return 2 if any(row["status"] == "UNREAD" for pair in result["pairs"]
                    for row in pair["categories"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
