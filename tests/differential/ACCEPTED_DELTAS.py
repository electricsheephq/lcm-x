"""Narrow, issue-backed exceptions; never suppress candidate absolute checks."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AcceptedDelta:
    id: str
    issue: str
    layer: tuple[str, ...]
    oracle: str
    reason: str
    predicate: Callable
    baselines: tuple[str, ...]


def preserved_user_growth(ga, cand):
    return sum(max(0, b["tokens"] - a["tokens"])
               for index, a in ga.get("user_rows", {}).items()
               if a["marked"] and (b := cand.get("user_rows", {}).get(index))
               and b["preserved"] and not b["marked"])


def ga_marked_user(case, ga, cand, failure=None):
    def explained(a, b, metric, limit):
        return a[metric] <= limit < b[metric] and b[metric] - limit <= preserved_user_growth(a, b)

    if case["layer"] == "projection":
        return explained(ga, cand, "out_tokens", case["limit"])
    if failure is not None:
        index = failure["detail"]["call"]
        return explained(ga["calls"][index], cand["calls"][index], "tokens", case["window"])
    failures = [(a, b) for a, b in zip(ga["calls"], cand["calls"])
                if a["tokens"] <= case["window"] < b["tokens"]]
    return bool(failures) and all(explained(a, b, "tokens", case["window"]) for a, b in failures)


ACCEPTED_DELTAS = [AcceptedDelta(
    "D916_GA_MARKED_USER", "https://github.com/electricsheephq/lcm-x/issues/916",
    ("projection", "public"), "R1", "GA fit with a mark-only user row; #916 deliberately preserves its text. Residual room: #1225.",
    ga_marked_user,
    baselines=("v0.26.2",),
)]


def last_resort_user_mark(case, ga, cand, failure=None):
    """#1233: nonempty subset of GA user marks, and no candidate overflow where GA fit."""
    return (bool(cand["user_marks"])
            and set(cand["user_marks"]) <= set(ga["user_marks"])
            and not (ga["out_tokens"] <= case["limit"] < cand["out_tokens"]))


ACCEPTED_DELTAS.append(AcceptedDelta(
    "D1233_LAST_RESORT_USER_MARK", "https://github.com/electricsheephq/lcm-x/issues/1233",
    ("projection",), "P_USER", "#1233: nonempty subset of GA user marks, and no candidate overflow where GA fit.",
    last_resort_user_mark,
    baselines=("v0.26.2",),
))


def i1123_leaf_trade(case, ga, cand, failure):
    """#1123 / PR #1129: only the observed 64K managed provider-call cell."""
    return (case.get("id") == "i1139_small_window_64k"
            and failure["detail"] == dict(metric="summarizer_calls", ga=46, candidate=65))


ACCEPTED_DELTAS.append(AcceptedDelta(
    "D1123_64K_LEAF_TRADE", "https://github.com/electricsheephq/lcm-x/issues/1123",
    ("public",), "R3", "#1123 / PR #1129: 24K leaves trade 46 for 65 provider calls at 64K; compress runs fall.",
    i1123_leaf_trade,
    baselines=("v0.26.2",),
))


def i1123_40k_compress_trade(case, ga, cand, failure):
    """Separate from the 64K delta: the triaged 40K sweep/hold cadence."""
    return (case.get("id") == "i1139_small_window_40k"
            and failure["detail"] == dict(metric="compress_runs", ga=93, candidate=113))


ACCEPTED_DELTAS.append(AcceptedDelta(
    "D1123_40K_COMPRESS_TRADE", "https://github.com/electricsheephq/lcm-x/issues/1123",
    ("public",), "R3", "#1123 / PR #1129: 40K sweep/hold cadence costs 113 vs 93 runs; provider calls fall 166 to 165.",
    i1123_40k_compress_trade,
    baselines=("v0.26.2",),
))
