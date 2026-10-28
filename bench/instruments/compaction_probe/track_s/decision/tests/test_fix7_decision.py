"""Fix 7: synthetic admitted pairs, horizon eligibility and complete-run decisions."""
import pytest

from test_eval2 import e
from test_fix6_decision import comparison, fixture, modify


@pytest.mark.parametrize("arm", ["L0", "L1"])
def test_k2_asymmetric_incomplete_pair_blocks_kill(tmp_path, arm):
    scores, material = fixture(tmp_path)
    def change(sc):
        for probe in sc["probes"].values():
            probe["class"] = "MISS"
        if sc["arm"] == arm and sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
            sc["metrics"]["facts_kept"]["complete"] = False
    modify(scores, change)
    result = comparison(scores, material)
    assert result["intervals"]["facts_user"]["high"] == 0
    assert result["incomplete_pairs"] == [dict(seed=1, checkpoint="S1-CP20")]
    assert result["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("cost,high", [(1, 4), (1.11, 12)])
def test_k2_complete_kill_is_not_blocked_by_low_axis_coverage(cost, high):
    intervals = dict(facts_user=dict(low=6 if high == 12 else 0, high=high, n_seeds=8),
                     current_request=dict(low=None, high=None, n_seeds=3))
    assert e.verdict(intervals, cost, complete=True) == "KILL"
    assert e.verdict(intervals, cost, complete=False) == "INCONCLUSIVE"


def test_k1_continuity_only_extra_rows_exclude_pair(tmp_path):
    scores, material = fixture(tmp_path, arms=("L1", "codex-native"), reader="gpt-6-astra")
    def change(sc):
        if sc["arm"] == "codex-native" and sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
            sc["beyond_declared"] = dict(count=1, facts=[], corrections=[], compactions=[], continuity=["current-request"])
    modify(scores, change)
    result = comparison(scores, material, "C")
    assert result["compared_window"]["1"] == ["S1-CP10", "S1-CP30"]
    assert result["horizon_exclusions"] == [dict(seed=1, checkpoint="S1-CP20",
        beyond_declared=dict(count=1, facts=[], corrections=[], compactions=[], continuity=["current-request"]))]


@pytest.mark.parametrize("kind", ["facts", "corrections", "compactions", "continuity"])
@pytest.mark.parametrize("unknown", [False, True])
def test_k3_excluded_pair_cost_and_estimates_are_not_charged(tmp_path, kind, unknown):
    scores, material = fixture(tmp_path, arms=("L1", "codex-native"), reader="gpt-6-astra")
    def change(sc):
        if sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
            sc["accounting"].update(reader_input_tokens=None if unknown else 1000,
                                     reader_output_tokens=100, successful_probes=5, reader_estimated_attempts=2)
            if sc["arm"] == "codex-native":
                sc["beyond_declared"] = {kind: ["synthetic-id"]}
    modify(scores, change)
    result = comparison(scores, material, "C")
    assert result["compared_window"]["1"] == ["S1-CP10", "S1-CP30"]
    assert result["cost_per_success"] == {"L1": (8*76-22)/23, "codex-native": (8*76-22)/23}
    assert result["cost_ratio"] == 1
    assert result["reader_estimated_attempts"] == {"L1": 0, "codex-native": 0}


def test_k3_fully_excluded_run_has_no_reader_or_summary_cost(tmp_path):
    scores, material = fixture(tmp_path, arms=("L1", "codex-native"), reader="gpt-6-astra")
    modify(scores, lambda sc: sc.update(beyond_declared=dict(facts=["synthetic-id"]))
           if sc["arm"] == "codex-native" else None)
    result = comparison(scores, material, "C")
    assert all(not cps for cps in result["compared_window"].values())
    assert result["cost_per_success"] == {"L1": None, "codex-native": None}


def test_k1_plain_extra_rows_remain_in_cost_and_compared_window(tmp_path):
    scores, material = fixture(tmp_path, arms=("L1", "codex-native"), reader="gpt-6-astra")
    modify(scores, lambda sc: sc.update(beyond_declared=dict(count=2, facts=[], corrections=[],
           compactions=[], continuity=[])) if sc["arm"] == "codex-native" else None)
    result = comparison(scores, material, "C")
    assert not result["horizon_exclusions"]
    assert result["compared_window"]["1"] == ["S1-CP10", "S1-CP20", "S1-CP30"]
    assert result["cost_per_success"] == {"L1": 76/3, "codex-native": 76/3}
