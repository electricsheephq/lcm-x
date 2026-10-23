"""#1824: offline D2 contract, independently constructed numeric fixtures."""
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scorer():
    return load("d2_score")


def ledger(runs=9, delta=0, placebo=0, aa=False):
    rows = []
    for seed in (1, 2, 3):
        for k in range(1, runs + 1):
            for arm in ("comparison", "candidate"):
                share = 0.6 + (k - (runs + 1) / 2) * 0.002
                metrics = {str(cp): {"share": share + (delta if cp == 304 else placebo)
                                    * (arm == "candidate"), "facts": {"f1": True, "f2": False}}
                           for cp in (176, 304)}
                rows.append(dict(arm=arm, seed=seed, k=k, slot=(1 if k % 2 else 2)
                                 if arm == "candidate" else (2 if k % 2 else 1),
                                 state="valid", validity_reason="complete", started=0, finished=1,
                                 artifact="synthetic", attempts=[dict(id=1, state="valid")], metrics=metrics))
    return dict(option="B" if runs == 9 else "A", aa=aa, elapsed_s=1,
                stop_reason=None, pins={"synthetic": True}, rows=rows)


@pytest.mark.parametrize("delta,placebo,expected", [(0, 0, "PASS"), (-0.1, 0, "FAIL"),
                                                   (-0.1, 0.1, "INCONCLUSIVE")])
def test_verdict_order_and_hand_computed_interval(scorer, delta, placebo, expected):
    data = ledger(delta=delta, placebo=placebo)
    result = scorer.score(data)
    assert result["raw_verdict"] == expected and result["disposition"] == ""
    d = result["cp304"]
    # sum(k-5)^2 = 60; six groups with step .002, residual df=48.
    sd = math.sqrt(6 * 60 * 0.002**2 / 48)
    assert d["df"] == 48
    assert d["pooled_sd"] == pytest.approx(sd)
    assert d["estimate"] == pytest.approx(delta, abs=1e-14)
    assert d["se"] == pytest.approx(sd * math.sqrt(2 / 27))
    assert d["lower90"] == pytest.approx(delta - scorer.t_quantile(.9, 48) * d["se"])
    assert d["upper90"] == pytest.approx(delta + scorer.t_quantile(.9, 48) * d["se"])
    assert d["upper_below_zero"] == (delta < 0)
    assert result["permutation"]["count"] >= 20000
    assert result["mcnemar"]["p"] == 1
    assert result["spreads"]["candidate"]["1"] == pytest.approx(.016)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "budget", "per_slot_budget", "elapsed", "rate"])
def test_incomplete_first_even_if_placebo_trips(scorer, kind):
    data = ledger(placebo=.1)
    if kind == "missing":
        data["rows"].pop()
    elif kind == "duplicate":
        data["rows"][-1] = data["rows"][0]
    elif kind == "budget":
        for row in data["rows"][:3]:
            row["attempts"].insert(0, dict(id=1, state="replaced"))
            row["attempts"][-1]["id"] = 2
    elif kind == "per_slot_budget":
        data["rows"][0]["attempts"] = [dict(id=i, state="replaced") for i in (1, 2, 3)]
    elif kind == "elapsed":
        data["elapsed_s"] = 21600
    else:
        data["stop_reason"] = "rate_limited"
    result = scorer.score(data)
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert result["reason"] != "cp176 placebo"


def test_aa_cap_and_strict_boundaries(scorer):
    assert scorer.score(ledger(delta=.05, aa=True))["raw_verdict"] == "FAIL"
    assert scorer.score(ledger(delta=.049, aa=True))["raw_verdict"] == "PASS"
    data = ledger(delta=-.05)
    for row in data["rows"]:
        row["metrics"]["304"]["share"] = .5 if row["arm"] == "comparison" else .45
    assert scorer.score(data)["raw_verdict"] == "FAIL"


@pytest.mark.parametrize("delta,expected", [(-.1, "FAIL"), (0, "PASS"), (.1, "PASS")])
def test_option_a_full_4_plus_4_enumeration(scorer, delta, expected):
    result = scorer.score(ledger(runs=4, delta=delta))
    assert result["raw_verdict"] == expected
    assert result["permutation"]["method"] == "exact"
    assert result["permutation"]["count"] == 70**3
    assert result["claim"] == "coarse regression screen, never non-inferiority"


def test_permutation_matches_brute_force(scorer):
    a = [[0, .2], [.3, .8]]
    b = [[.4, .6], [.9, 1]]
    observed = abs(sum(sum(y) / 2 - sum(x) / 2 for x, y in zip(a, b)) / 2)
    distribution = []
    for allocations in itertools.product(list(itertools.combinations(range(4), 2)), repeat=2):
        d = 0
        for values, chosen in zip([x + y for x, y in zip(a, b)], allocations):
            d += (sum(values[i] for i in chosen) - sum(values[i] for i in range(4) if i not in chosen)) / 4
        distribution.append(d)
    expected = sum(abs(d) >= observed - 1e-12 for d in distribution) / 36
    result = scorer.permutation(a, b)
    assert result["p"] == pytest.approx(expected)
    assert result["count"] == 36


@pytest.mark.parametrize("df,critical", [(2, 1.886), (18, 1.330), (48, 1.299)])
def test_t_quantile_against_nist_table(scorer, df, critical):
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm, column .90.
    assert scorer.t_quantile(.9, df) == pytest.approx(critical, abs=.0005)
    # Independent Simpson quadrature of the Student density, no CDF implementation reuse.
    t = scorer.t_quantile(.975, df)
    x = np.linspace(0, t, 10001)
    pdf = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    y = pdf * (1 + x*x/df)**(-(df+1)/2)
    integral = (x[1] - x[0]) / 3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())
    assert integral + .5 == pytest.approx(.975, abs=1e-10)


def test_operating_points(scorer, capsys):
    # Matches analysis.py's gaussian: complete valid cp304 NI rule only;
    # excludes cp176, A/A point cap, retries and elapsed/provider failures.
    count = 30000
    rng = np.random.default_rng(1538)
    rows = []
    for sd, loss, target in [(.06772880744575896, 0, .9204),
                             (.0937705280102746, 0, .7456),
                             (.06772880744575896, .05, .9),
                             (.0937705280102746, .05, .9)]:
        a = rng.normal(0, sd, (count, 3, 9))
        b = rng.normal(-loss, sd, (count, 3, 9))
        d = (b.mean(axis=2) - a.mean(axis=2)).mean(axis=1)
        ss = ((a-a.mean(axis=2, keepdims=True))**2).sum(axis=(1, 2))
        ss += ((b-b.mean(axis=2, keepdims=True))**2).sum(axis=(1, 2))
        lower = d - scorer.t_quantile(.9, 48) * np.sqrt(ss / 48) * math.sqrt(2 / 27)
        observed = float(np.mean(lower <= -.05 if loss else lower > -.05))
        se = math.sqrt(target * (1-target) / count)
        assert abs(observed-target) <= 3*se
        rows.append(dict(sd=sd, loss=loss, n=count, target=target, observed=observed, simulation_se=se))
    print("OPERATING_POINTS=" + json.dumps(rows, sort_keys=True))
