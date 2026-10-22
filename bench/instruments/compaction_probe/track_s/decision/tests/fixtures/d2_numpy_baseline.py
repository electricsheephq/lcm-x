"""Calibrated D2 only; analyze_paired.py retains the independent D7 contract."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def t_cdf(t, df):
    """Even integer df: integrate the Student density with t=sqrt(df)*tan(theta).

    NIST density: https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm
    The transformed density is proportional to cos(theta)**(df-1). Integration
    by parts gives I_m=(sin(theta)*cos(theta)**(m-1)+(m-1)*I_(m-2))/m.
    Normalize by I_m(pi/2); symmetry supplies the negative half. No z approximation.
    """
    if df < 2 or df % 2:
        raise ValueError("D2 requires positive even integer degrees of freedom")
    theta = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(theta), math.cos(theta)
    integral, normalizer = s, 1.0
    for m in range(3, df, 2):
        integral = (s*c**(m-1) + (m-1)*integral) / m
        normalizer *= (m-1) / m
    mass = min(.5, .5 * integral / normalizer)
    return .5 + math.copysign(mass, t)


def t_quantile(p, df):
    if not .5 < p < 1:
        raise ValueError("upper-half quantile required")
    lo, hi = 0., 1.
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pooled(a, b):
    a, b = np.asarray(a), np.asarray(b)
    seeds, runs = a.shape
    df = 2 * seeds * (runs-1)
    sd = math.sqrt(float(((a-a.mean(axis=1, keepdims=True))**2).sum() +
                         ((b-b.mean(axis=1, keepdims=True))**2).sum()) / df)
    se = sd * math.sqrt(2 / (seeds*runs))
    d = float((b.mean(axis=1) - a.mean(axis=1)).mean())
    p = 2*(1-t_cdf(abs(d)/se, df)) if se else float(abs(d) < 1e-12)
    radius = t_quantile(.9, df) * se
    return dict(estimate=d, pooled_sd=sd, se=se, df=df, lower90=d-radius,
                upper90=d+radius, upper_below_zero=d+radius < 0, two_sided_p=p)


def permutation(a, b, count=20000, seed=1824):
    """Two-sided mean over seeds, all within-seed allocations (4+4 is 70**3)."""
    seeds, runs = len(a), len(a[0])
    observed = abs(sum(sum(y)-sum(x) for x, y in zip(a, b)) / (seeds*runs))
    allocations = math.comb(2*runs, runs)
    exact = allocations**seeds <= 1000000
    rng, dist = np.random.default_rng(seed), np.zeros(1 if exact else count)
    for x, y in zip(a, b):
        v = np.asarray(list(x) + list(y))
        if exact:
            choices = np.asarray(list(itertools.combinations(range(2*runs), runs)))
        else:
            choices = rng.random((count, 2*runs)).argsort(axis=1)[:, :runs]
        diffs = (2*v[choices].sum(axis=1)-v.sum()) / (runs*seeds)
        dist = (dist[:, None] + diffs).ravel() if exact else dist + diffs
    extreme = int(np.count_nonzero(np.abs(dist) >= observed-1e-12))
    return dict(p=extreme/len(dist) if exact else (extreme+1)/(len(dist)+1),
                method="exact" if exact else "seeded Monte Carlo (+1 correction)",
                count=len(dist), seed=None if exact else seed)


def score(data):
    runs = 9 if data["option"] == "B" else 4
    result = dict(raw_verdict="INCONCLUSIVE", disposition="", reason="incomplete comparison",
                  claim="coarse regression screen, never non-inferiority" if runs == 4 else
                  "fixed-material non-inferiority by a 5-point margin at one-sided 90% confidence",
                  cp304=None, cp176=None, permutation=None, mcnemar=None, spreads=None)
    rows = data["rows"]
    expected = {(a, s, k) for a in ("comparison", "candidate") for s in (1, 2, 3) for k in range(1, runs+1)}
    index = {(r["arm"], r["seed"], r["k"]): r for r in rows}
    replacements = sum(max(0, len(r["attempts"])-1) for r in rows)
    if (len(rows) != len(expected) or set(index) != expected or data.get("stop_reason") or
            not 0 <= data.get("elapsed_s", math.inf) < 21600 or replacements > 2 or
            any(r["state"] != "valid" or not 1 <= len(r["attempts"]) <= 2 or
                [a["state"] for a in r["attempts"]] != ["replaced"]*(len(r["attempts"])-1)+["valid"] or
                [a["id"] for a in r["attempts"]] != list(range(1, len(r["attempts"])+1)) for r in rows)):
        return result
    try:
        arrays = {cp: [[[index[a, s, k]["metrics"][cp]["share"] for k in range(1, runs+1)]
                        for s in (1, 2, 3)] for a in ("comparison", "candidate")] for cp in ("176", "304")}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for ab in arrays.values() for a in ab for s in a for v in s):
            return result
        b = c = 0
        for s, k in itertools.product((1, 2, 3), range(1, runs+1)):
            x, y = [index[a, s, k]["metrics"]["304"]["facts"] for a in ("comparison", "candidate")]
            if not x or x.keys() != y.keys() or any(type(v) is not bool for f in (x, y) for v in f.values()):
                return result
            b += sum(x[f] and not y[f] for f in x)
            c += sum(y[f] and not x[f] for f in x)
    except (KeyError, TypeError, ValueError):
        return result
    primary, placebo = pooled(*arrays["304"]), pooled(*arrays["176"])
    perm = permutation(*arrays["304"])
    p_mc = min(1., 2*sum(math.comb(b+c, i) for i in range(min(b, c)+1)) / 2**(b+c))
    result.update(cp304=primary, cp176=placebo, permutation=perm,
                  mcnemar=dict(comparison_only=b, candidate_only=c, p=p_mc, diagnostic_only=True),
                  spreads={a: {str(s): max(v)-min(v) for s, v in enumerate(array, 1)}
                           for a, array in zip(("comparison", "candidate"), arrays["304"])})
    if placebo["two_sided_p"] < .05 and abs(placebo["estimate"]) >= .05-1e-12:
        result["reason"] = "cp176 placebo"
        return result
    if runs == 4:
        passed = not (perm["p"] < .05 and primary["estimate"] <= -.05+1e-12)
    else:
        passed = primary["lower90"] > -.05+1e-12 and (not data.get("aa") or primary["estimate"] < .05-1e-12)
    result.update(raw_verdict="PASS" if passed else "FAIL", reason="calibrated rule" if runs == 9 else "coarse regression screen")
    return result


def write_result(out, result):
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "result.md").write_text(f"# D2 raw verdict: {result['raw_verdict']}\n\ndisposition: \n\n"
                                   + result["claim"] + "\n\nDiagnostics (never independent gates):\n\n```json\n"
                                   + json.dumps(result, indent=2) + "\n```\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_result(args.out, score(json.loads(args.ledger.read_text())))
