"""Paired statistics; all Markdown output is tables, without corpus text."""
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from .common import write_json
from .closeout import result_card, snapshot
from .models import JUDGES, parse_spec
from .judges import parse


def mean(values):
    return sum(values) / len(values) if values else None


def mcnemar(a, b):
    n01 = sum(not x and bool(y) for x, y in zip(a, b, strict=True))
    n10 = sum(bool(x) and not y for x, y in zip(a, b, strict=True))
    n = n01 + n10
    p = min(1, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"n01": n01, "n10": n10, "p": p, "n": len(a)}


def bootstrap(a, b, resamples=10_000):
    if not a:
        return None
    delta = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(1189)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(resamples))
    return [samples[int(.025 * resamples)], samples[min(resamples - 1, int(.975 * resamples))]]


def agreement(a, b):
    if not a:
        return {"n": 0, "percent": None, "kappa": None}
    observed = mean([x == y for x, y in zip(a, b, strict=True)])
    expected = mean(a) * mean(b) + (1 - mean(a)) * (1 - mean(b))
    return {"n": len(a), "percent": observed * 100,
            "kappa": (observed - expected) / (1 - expected) if expected != 1 else None}


def percentile(values, q):
    return sorted(values)[min(len(values) - 1, math.ceil(q * len(values)) - 1)] if values else None


def correct(row, model):
    return None if invalid(row, [model]) else row["judges"][model]["correct"]


def invalid(row, models):
    requested = set(models) | {parse_spec(m)[1] for m in models if parse_spec(m)[0] == "openrouter"}
    # Judge-call failures count against their own judge; a call without attribution (older records) stays row-wide.
    calls = row.get("calls", []) + [c for c in row.get("judge_calls", [])
                                    if not c.get("requested_model") or c.get("requested_model") in requested]
    reasons = {c.get("error") for c in calls if not c.get("retried")} & {"timeout", "model_mismatch"}
    terminal = (row.get("calls") or [{}])[-1]
    if terminal.get("error") == "mcp_unavailable" or ("mcp_ready" in terminal and (
            not terminal["mcp_ready"] or terminal.get("offered_tools", 0) == 0)):
        reasons.add("mcp_unavailable")  # A recovered first attempt does not invalidate the answer.
    if row["status"] == "tool_violation" or any(c.get("violations") or c.get("status") == "tool_violation" for c in calls):
        reasons.add("tool_violation")
    if row["status"] != "ok" and not reasons:
        reasons.add("reader_error" if row["status"] == "error" else row["status"])
    for model in models if row["status"] == "ok" else ():
        judge = row.get("judges", {}).get(model, {})
        if judge.get("status") != "ok":
            reasons.add("judge_error")
        elif type(judge.get("correct")) is not bool or judge.get("verdict") == "invalid":
            reasons.add("invalid_verdict")
    return reasons


EXCLUDED = ("skipped", "not_run")  # skips: explicit denominators; not_run: spend stops (D7)
IDENTITY = ("lcmx_commit", "harness_commit", "split_label", "dataset_sha256", "id_list_sha256", "judges",
            "agreement_judge", "summaries", "agreement_sample", "agreement_seed", "agreement_fraction",
            "harness_tree_sha256", "toolset", "store_text", "hit_label", "include_adversarial")
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "wrapper_overhead_tokens")


def delivery_identity(manifest):
    # Historical manifests predate P and are exactly the default hits arm.
    return manifest.get("delivery_unit", "hits"), manifest.get("delivery_budget")


def ok_verdict(row, model):
    return not invalid(row, [model])


def usage(calls, n):
    calls = [c for c in calls if not c.get("retried")]
    return {k: sum(c[k] for c in calls) / max(1, n) if calls and all(c.get(k) is not None for c in calls)
            else None for k in TOKEN_KEYS}


def quota(calls):
    weekly = [c["weekly_used_percent"] for c in calls if c.get("weekly_used_percent") is not None]
    return {"codex_weekly_used_percent": [min(weekly), max(weekly)] if weekly else None,
            "weekly_unknown": sum(c.get("backend") == "codex" and c.get("weekly_used_percent") is None for c in calls),
            "glm_usage_limit_errors": sum(c.get("backend") == "glm" and c.get("error") == "usage_limit"
                                          for c in calls),
            "claude_usage_limit_errors": sum(c.get("backend") == "claude" and c.get("error") == "usage_limit"
                                             for c in calls)}


def score(runs, output, allow_partial=False, pair_across_commits=False):
    output = Path(output)
    arms, manifests, disagreements = {}, {}, []
    snapshots = snapshot(runs, output)
    for directory in snapshots:
        manifest = json.loads((directory / "run-manifest.json").read_text())
        rows = [json.loads(line) for line in (directory / "judged.jsonl").read_text().split("\n") if line.strip()]
        if manifest.get("status") not in ("complete", "incomplete"):
            raise ValueError("run is not fully judged")
        if {r["qid"] for r in rows} != set(manifest.get("question_ids") or []):
            raise ValueError("judged rows do not cover the run's question ids")
        for row in rows:
            for judge in row.get("judges", {}).values():
                if judge.get("raw"):
                    judge["correct"] = parse(judge["raw"][-1], "locomo" if row["qtype"].isdigit() and row["qtype"] != "5" else "lme-s")
        if len({r["qid"] for r in rows}) != len(rows):
            raise ValueError("duplicate result ids")
        arm = "/".join([manifest["config"], manifest["delivery"], manifest["reader"]])
        if "store_text" in manifest or "hit_label" in manifest:
            arm += "/" + manifest["store_text"] + "/" + manifest["hit_label"]
        unit, budget = delivery_identity(manifest)
        if (unit, budget) != ("hits", None):
            arm += f"/{unit}-{budget}"
        if pair_across_commits:
            identity = manifest["lcmx_commit"]
            arm += "@" + (identity if identity.endswith("-dirty") else identity[:12])
        if arm in arms:
            raise ValueError("duplicate arm")
        arms[arm], manifests[arm] = {r["qid"]: r for r in rows}, manifest
    sets = [set(rows) for rows in arms.values()]
    if not sets or any(ids != sets[0] for ids in sets):
        raise ValueError("paired runs require identical question ids")
    for key in IDENTITY:
        if key in ("store_text", "hit_label"):  # Experimental arm axes, compared within cross-commit pairs.
            continue
        if pair_across_commits and key in ("lcmx_commit", "toolset"):
            continue
        if len({json.dumps(m.get(key)) for m in manifests.values()}) != 1:
            raise ValueError("paired runs require identical " + key)
    if pair_across_commits:
        commits = sorted({m["lcmx_commit"] for m in manifests.values()})
        if len(commits) < 2:
            raise ValueError("pairing across commits requires at least 2 distinct lcmx_commit values")
    ran = [{q for q, r in rows.items() if r["status"] != "not_run"} for rows in arms.values()]
    partial = any(ids != ran[0] for ids in ran) or any(m.get("status") == "incomplete" for m in manifests.values())
    if partial and not allow_partial:
        raise ValueError("arms ran unequal question sets (budget stop); pass --allow-partial to score them")
    summaries, comparisons = {}, {}
    # Judges on every answer (the first is primary); an agreement judge sees a seeded sample only.
    first = next(iter(manifests.values()))
    full, extra = list(first.get("judges") or JUDGES), first.get("agreement_judge")
    pair = full[:2] if len(full) > 1 else [full[0], extra] if extra else None
    for arm, indexed in arms.items():
        rows = list(indexed.values())
        eligible = [r for r in rows if not invalid(r, full)]
        valid = [r for r in eligible if pair and all(ok_verdict(r, m) for m in pair)]
        disagreements += [{k: r[k] for k in ("qid", "question", "gold", "hypothesis")} for r in valid
                          if r["judges"][pair[0]]["correct"] != r["judges"][pair[1]]["correct"]]
        calls = [c for r in rows for c in r.get("calls", [])]
        mcp = [c for c in calls if "mcp_ready" in c]
        mcp_summary = {"ready_rate": mean([c["mcp_ready"] for c in mcp]),
            "offered_tools_distribution": dict(Counter(c["offered_tools"] for c in mcp)),
            "zero_lcm_calls_ready": sum(c["mcp_ready"] and c["lcm_tool_calls"] == 0 for c in mcp),
            "mcp_unavailable_retried": sum(c.get("retries", 0) == 1 or
                (c.get("error") == "mcp_unavailable" and bool(c.get("retried"))) for c in mcp),
            "mcp_unavailable_excluded": sum("mcp_unavailable" in invalid(r, full) for r in rows)}
        finishes = [c.get("finish_reason") for c in calls if c.get("status") == "ok"]
        judge_calls = [c for r in rows for c in r.get("judge_calls", [])]
        attempted = [r for r in rows if r["status"] not in EXCLUDED]
        accuracy = {m: mean([bool(correct(r, m)) for r in attempted]) for m in full}
        sampled = [r for r in eligible if extra and extra in r.get("judges", {}) and ok_verdict(r, extra)]
        summaries[arm] = {"n": len(rows), "scored_n": len(attempted), "accuracy": accuracy, "mcp": mcp_summary,
            "valid_n": len(eligible), "invalid_by_reason": dict(Counter(k for r in rows for k in invalid(r, full))),
            "reader_tokens_unknown": {k: sum(c.get(k) is None for c in calls) for k in TOKEN_KEYS},
            "judge_tokens_unknown": {k: sum(c.get(k) is None for c in judge_calls) for k in TOKEN_KEYS},
            "input_tokens_label": "total incl. retrieved evidence",
            "primary": mean(list(accuracy.values())) if attempted else None,
            "agreement_sample": {"judge": extra, "n": len(sampled),
                                 "accuracy": {m: mean([correct(r, m) for r in sampled if ok_verdict(r, m)]) for m in (full[0], extra)}}
                                if extra else None,
            "by_type": {t: {m: mean([bool(correct(r, m)) for r in attempted if r["qtype"] == t]) for m in full}
                        for t in sorted({r["qtype"] for r in rows})},
            "agreement": agreement(*[[r["judges"][m]["correct"] for r in valid] for m in pair]) if pair else None,
            "tool_calls_per_question": mean([r.get("tool_calls", 0) for r in eligible]),
            "tokens_per_question_population": "all attempted rows (excluding skipped and not_run), including failures",
            "reader_tokens_per_question": usage([c for r in attempted for c in r.get("calls", [])], len(attempted)),
            "judge_tokens_per_question": usage([c for r in attempted for c in r.get("judge_calls", [])], len(attempted)),
            "quota": quota(calls + judge_calls),
            "latency_p50_s": percentile([r.get("latency_s", 0) - r.get("retry_backoff_s", 0) for r in eligible], .5),
            "latency_p90_s": percentile([r.get("latency_s", 0) - r.get("retry_backoff_s", 0) for r in eligible], .9),
            "reader_cost": sum(c.get("cost", 0) for c in calls),
            "judge_cost": sum(c.get("cost", 0) for c in judge_calls),
            "cost": sum(c.get("cost", 0) for c in calls + judge_calls),
            "reader_errors": sum(r["status"] == "error" for r in rows),
            "failures": {k: sum(r["status"] == k for r in rows)
                         for k in sorted({r["status"] for r in rows} - {"ok", *EXCLUDED})},
            "judge_errors": sum(any(r.get("judges", {}).get(m, {}).get("status") == "judge_error" for m in full)
                                for r in rows),
            "judge_errors_by_judge": {m: sum(r.get("judges", {}).get(m, {}).get("status") == "judge_error"
                                             for r in rows) for m in full + ([extra] if extra else [])},
            "forced_finals": {k: sum(r.get("forced_final") == k for r in rows) for k in ("tool_calls", "input_tokens")},
            "finish_reasons": {str(k): finishes.count(k) for k in sorted(set(finishes), key=str)},
            "length_finishes": finishes.count("length"),
            "not_run": sum(r["status"] == "not_run" for r in rows),
            "skipped": sum(r["status"] == "skipped" for r in rows),
            "wall_s": manifests[arm].get("wall_s", 0)}
        if first.get("include_adversarial") and any(r["qtype"].isdigit() for r in rows):
            summaries[arm]["category_totals"] = {
                label: {"scored_n": len(group), "accuracy": {
                    m: mean([bool(correct(r, m)) for r in group]) for m in full}}
                for label, group in (("all categories", attempted),
                                     ("cats 1-4", [r for r in attempted if r["qtype"] in ("1", "2", "3", "4")]))}
    for summary in summaries.values():
        for accuracies in summary["by_type"].values():
            accuracies["primary"] = mean(list(accuracies.values())) if all(v is not None for v in accuracies.values()) else None
    for left, right in itertools.combinations(arms, 2):
        if pair_across_commits and (manifests[left]["lcmx_commit"] == manifests[right]["lcmx_commit"] or
                delivery_identity(manifests[left]) != delivery_identity(manifests[right]) or any(
                manifests[left].get(key) != manifests[right].get(key) for key in ("config", "delivery", "reader", "store_text", "hit_label"))):
            continue
        # An error in either arm (reader error, budget or tool violation) removes the question
        # from both arms: the paired tests compare answers, not failure rates.
        dropped = {q: invalid(arms[left][q], full) | invalid(arms[right][q], full) for q in sets[0]}
        ids = sorted(q for q in sets[0] if not dropped[q])
        vectors = {arm: [[correct(arms[arm][q], m) for q in ids] for m in full] for arm in (left, right)}
        agree_ids = [i for i, q in enumerate(ids) if len(full) > 1 and all(
            all(ok_verdict(arms[arm][q], m) for m in full[:2])
            and arms[arm][q]["judges"][full[0]]["correct"] == arms[arm][q]["judges"][full[1]]["correct"]
            for arm in (left, right))]
        comparisons[left + " vs " + right] = {
            "paired_n": len(ids),
            "n_pairs": len(ids), "dropped_by_reason": dict(Counter(k for reasons in dropped.values() for k in reasons)),
            "dropped_for_errors": sum(bool(reasons) for reasons in dropped.values()),
            "per_judge": {m: mcnemar(vectors[left][i], vectors[right][i]) for i, m in enumerate(full)},
            "both_agree": mcnemar([vectors[left][0][i] for i in agree_ids], [vectors[right][0][i] for i in agree_ids])
                          if len(full) > 1 else None,
            "primary_difference_ci95": bootstrap([sum(x) / len(full) for x in zip(*vectors[left])],
                                                  [sum(x) / len(full) for x in zip(*vectors[right])])}
        if first.get("include_adversarial"):
            populations = {}
            for label, types in (("all categories", None), ("cats 1-4", {"1", "2", "3", "4"}), ("category 5", {"5"})):
                selected = [q for q in ids if types is None or arms[left][q]["qtype"] in types]
                stats = {}
                for m in full:
                    a, b = [[correct(arms[arm][q], m) for q in selected] for arm in (left, right)]
                    test = mcnemar(a, b)
                    stats[m] = {**test, "b": test["n01"], "c": test["n10"],
                                "accuracy_delta": mean([int(x) - int(y) for x, y in zip(a, b, strict=True)])}
                populations[label] = {"paired_n": len(selected), "per_judge": stats}
            comparisons[left + " vs " + right]["populations"] = populations
    report = {"arms": summaries, "paired": comparisons, "bootstrap_resamples": 10_000,
              "claim": "attempted errors count incorrect; skips and not_run excluded; paired tests use valid rows only"}
    if pair_across_commits:
        report.update(paired_across="lcmx_commit", lcmx_commits=commits,
                      toolsets={arm: m.get("toolset") for arm, m in manifests.items()})
    report["tokens_500_per_arm"] = {arm: {k: 500 * (s["reader_tokens_per_question"][k] + s["judge_tokens_per_question"][k])
                                          if all(s[t][k] is not None for t in ("reader_tokens_per_question", "judge_tokens_per_question")) else None
                                          for k in ("input_tokens", "output_tokens")} for arm, s in summaries.items()}
    if all(m["split_label"] == "smoke" for m in manifests.values() if "split_label" in m) and \
            any(s["cost"] for s in summaries.values()):
        estimates = {}
        for arm, summary in summaries.items():
            factor = 500 / max(1, summary["n"])
            estimates[arm] = {"usd_500": factor * summary["cost"],
                              "wall_500_s_at_observed_workers": factor * summary["wall_s"]}
        agreement_judgments = sum(math.ceil(m.get("agreement_fraction", 0) * 500)
                                  for m in manifests.values() if m.get("agreement_judge"))
        report["full_run_estimate"] = {"arms": estimates,
            "matrix": f"500 questions x {len(estimates)} supplied arms = {500 * len(estimates)} answers, "
                      f"{500 * len(estimates) * len(full)} required judgments, "
                      f"{agreement_judgments} agreement judgments; arms: " + "; ".join(estimates),
            "total_usd": sum(e["usd_500"] for e in estimates.values()),
            "wall_s_at_observed_workers": sum(e["wall_500_s_at_observed_workers"] for e in estimates.values()),
            "assumptions": "Linear smoke extrapolation for supplied arms only, identical token counts and latency; "
                           "excludes local build and rerank. No unmeasured reader or worker scaling."}
    sample = random.Random(1189).sample(disagreements, min(50, len(disagreements)))
    report["adjudication"] = {"disagreements": len(disagreements), "sampled": len(sample), "seed": 1189}
    write_json(output / "report.json", report)
    (output / "adjudicate.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in sample))
    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"
    lines = ["| Arm | N/scored/skipped/not run | " + " | ".join(full) + " | Mean | Agreement | κ | Tools/q "
             "| Forced finals calls/tokens | Finish reasons | Reader in/cached/out/reason/overhead tokens/q "
             "| Judge in/out tokens/q | Codex weekly % seen | Cost $ | Failures | Judge errors | Wall s | p50/p90 s |",
             "|---|---:|" + "---:|" * len(full) + "---:|---:|---:|---:|---|---|---|---|---|---:|---|---|---:|---|"]
    for arm, s in summaries.items():
        agree = s["agreement"] or {}
        judge_tokens = s["judge_tokens_per_question"]
        values = [arm, f"{s['n']}/{s['scored_n']}/{s['skipped']}/{s['not_run']}", *[fmt(s["accuracy"][m]) for m in full],
                  fmt(s["primary"]), fmt(agree.get("percent")), fmt(agree.get("kappa")),
                  fmt(s["tool_calls_per_question"]), "/".join(map(str, s["forced_finals"].values())),
                  ", ".join(f"{k} {v}" for k, v in s["finish_reasons"].items()),
                  "/".join(fmt(x) for x in s["reader_tokens_per_question"].values()),
                  f"{fmt(judge_tokens['input_tokens'])}/{fmt(judge_tokens['output_tokens'])}",
                  str(s["quota"]["codex_weekly_used_percent"]), fmt(s["cost"]),
                  ", ".join(f"{k} {v}" for k, v in s["failures"].items()) or "0",
                  "/".join(map(str, s["judge_errors_by_judge"].values())), fmt(s["wall_s"]),
                  fmt(s["latency_p50_s"]) + "/" + fmt(s["latency_p90_s"])]
        lines.append("| " + " | ".join(values) + " |")
    if extra:
        lines += ["", f"| Arm | Agreement sample n | {full[0]} on sample | {extra} on sample |", "|---|---:|---:|---:|"]
        for arm, s in summaries.items():
            sample_acc = s["agreement_sample"]["accuracy"]
            lines.append(f"| {arm} | {s['agreement_sample']['n']} | {fmt(sample_acc[full[0]])} | {fmt(sample_acc[extra])} |")
    lines += ["", "| Arm | MCP ready rate | Offered tools distribution | Zero LCM calls ready | "
              "mcp_unavailable retried/excluded |", "|---|---:|---|---:|---|"]
    for arm, s in summaries.items():
        m = s["mcp"]
        lines.append(f"| {arm} | {fmt(m['ready_rate'])} | {m['offered_tools_distribution']} | "
                     f"{m['zero_lcm_calls_ready']} | {m['mcp_unavailable_retried']}/{m['mcp_unavailable_excluded']} |")
    lines += ["", "| Arm | Question type | " + " | ".join(f"{m} accuracy" for m in full) + " |",
              "|---|---|" + "---:|" * len(full)]
    for arm, summary in summaries.items():
        for qtype, accuracies in summary["by_type"].items():
            lines.append("| " + " | ".join([arm, qtype, *[str(accuracies[m]) for m in full]]) + " |")
        for label, total in summary.get("category_totals", {}).items():
            lines.append("| " + " | ".join([arm, f"{label} (n={total['scored_n']})",
                                          *[fmt(total["accuracy"][m]) for m in full]]) + " |")
    lines += ["", "| Comparison | Paired n | Dropped for errors | " + " | ".join(f"{m} exact p" for m in full) +
              " | Both agree exact p | Primary Δ 95% CI |", "|---|---:|---:|" + "---:|" * len(full) + "---:|---|"]
    for pair_name, s in comparisons.items():
        lines.append("| " + " | ".join([pair_name, str(s["paired_n"]), str(s["dropped_for_errors"]),
                                       *[str(s["per_judge"][m]["p"]) for m in full],
                                       str((s["both_agree"] or {}).get("p", "n/a")),
                                       str(s["primary_difference_ci95"])]) + " |")
    if first.get("include_adversarial"):
        lines += ["", "| Comparison | Population | Judge | Paired n | b (left wrong) | c (right wrong) | Exact p | Accuracy Δ (left-right) |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for pair_name, comparison in comparisons.items():
            for population, group in comparison["populations"].items():
                for model, stat in group["per_judge"].items():
                    lines.append("| " + " | ".join([pair_name, population, model, str(group["paired_n"]),
                        *[str(stat[k]) for k in ("b", "c", "p", "accuracy_delta")]]) + " |")
    lines += ["", "| Comparison | Exclusions |", "|---|---|"]
    for pair_name, s in comparisons.items():
        lines.append("| " + pair_name + " | " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(s["dropped_by_reason"].items())) + " |")
    if "full_run_estimate" in report:
        e = report["full_run_estimate"]
        lines += ["", "| Estimate (assumptions in report.json) | Value |", "|---|---|",
                  "| Matrix | " + e["matrix"] + " |",
                  f"| Total USD | {e['total_usd']:.2f} |",
                  f"| Wall hours, observed workers | {e['wall_s_at_observed_workers'] / 3600:.2f} |",
                  "| Supplied arms | " + "; ".join(e["arms"]) + " |"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    result_card(output, report, allow_partial, pair_across_commits, snapshots)
    return report
