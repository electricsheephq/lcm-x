"""Exact frozen NumPy oracle; production scorer must retain all numeric outputs."""
import importlib.util
import math
from pathlib import Path
import random

import pytest

from test_d2_calibrated import ledger, load


def oracle():
    spec = importlib.util.spec_from_file_location("numpy_baseline", Path(__file__).parent / "fixtures/d2_numpy_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            # Item 6 deliberately corrects the A/A claim, without changing its verdict.
            if key != "claim":
                equal(actual[key], value)
    elif isinstance(expected, float):
        assert math.isfinite(actual) and abs(actual - expected) <= 1e-12
    else:
        assert actual == expected


def test_existing_fixtures_unchanged():
    scorer, old = load("d2_score"), oracle()
    for runs in (4, 9):
        for delta in (-.1, -.05, 0, .049, .05, .1):
            for placebo in (0, .1):
                for aa in (False, True):
                    data = ledger(runs=runs, delta=delta, placebo=placebo, aa=aa)
                    equal(scorer.score(data), old.score(data))


def test_3200_random_ledgers_unchanged():
    scorer, old, rng = load("d2_score"), oracle(), random.Random(1834)
    for i in range(3200):
        data = ledger(runs=4 if i % 2 else 9, aa=i % 3 == 0)
        loss = rng.uniform(-.12, .12)
        placebo = rng.choice((0, .06, -.06))
        for row in data["rows"]:
            for cp in ("176", "304"):
                row["metrics"][cp]["share"] = min(1., max(0., rng.gauss(.6, .07) +
                    ((loss if cp == "304" else placebo) if row["arm"] == "candidate" else 0)))
        if i % 13 == 0:
            data["rows"][0]["state"] = "missing"
        equal(scorer.score(data), old.score(data))


@pytest.mark.parametrize("seed", [0, 1, 1824, 2**64 + 99])
def test_seeded_monte_carlo_preserves_numpy_draws(seed):
    scorer, old = load("d2_score"), oracle()
    a = [[.11 + k * .02 for k in range(9)] for _ in range(3)]
    b = [[.13 + k * .017 for k in range(9)] for _ in range(3)]
    equal(scorer.permutation(a, b, count=71, seed=seed), old.permutation(a, b, count=71, seed=seed))
