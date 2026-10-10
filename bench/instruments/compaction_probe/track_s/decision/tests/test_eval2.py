"""Frozen bars and session-clustered statistics, with entirely synthetic observations."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("eval2", Path(__file__).resolve().parents[1] / "eval2.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


@pytest.mark.parametrize("intervals,cost,expected", [
    ({"facts_user": dict(low=6, high=12), "facts_tool": dict(low=-1, high=1)}, 1.10, "KEEP"),
    ({"facts_user": dict(low=-1, high=4)}, 1, "KILL"),
    ({"facts_user": dict(low=3, high=15)}, 1, "INCONCLUSIVE"),
    ({"facts_user": dict(low=6, high=12), "facts_tool": dict(low=-5, high=-3)}, 1, "KILL"),
    ({"facts_user": dict(low=6, high=12), "facts_tool": dict(low=-3, high=1)}, 1, "INCONCLUSIVE"),
    ({"facts_user": dict(low=5, high=6)}, 1, "INCONCLUSIVE"),
    ({"facts_user": dict(low=6, high=12)}, 1.11, "KILL"),
    ({"facts_user": dict(low=6, high=12)}, None, "INCONCLUSIVE"),
    ({"facts_tool": dict(low=6, high=12)}, 1, "INCONCLUSIVE"),
])
def test_verdict(intervals, cost, expected):
    assert e.verdict(intervals, cost) == expected
    assert e.verdict(intervals, cost, complete=False) == "INCONCLUSIVE"


def test_bootstrap_resamples_sessions_with_fixed_rng():
    result = e.bootstrap([0, 20])
    assert result == e.bootstrap([0, 20])
    assert result == dict(effect_pts=10, low=0, high=20, n_seeds=2)
    assert e.bootstrap([10])["low"] is None
    assert e.bootstrap([8] * 8)["low"] == e.bootstrap([8] * 8)["high"] == 8


def test_axes_include_source_roles_identifiers_and_lifecycle():
    facts = [dict(id="a", row_role="user", placement="middle", **{"class": "path"}),
             dict(id="b", row_role="tool", placement="middle", **{"class": "limit"}),
             dict(id="future", row_role="assistant", placement="tail", **{"class": "name"})]
    score = dict(probes={"a": {"class": "CORRECT"}, "b": {"class": "MISS"},
                        "c": dict(kind="stale_task", compaction_horizon=3, **{"class": "CORRECT"})},
                 metrics={"facts_kept": {}})
    assert e.axes(score, facts) == dict(facts_all=[True, False], facts_user=[True], facts_tool=[False],
                                       facts_tool_middle=[False], identifier=[True], stale_task=[True])


def test_eight_seed_pairs_require_every_manifest_checkpoint_and_admitted_score(tmp_path):
    scores, material = tmp_path / "decision/scores", tmp_path / "material"
    admission = dict(schema=e.score_manifest.SCHEMA, entries={})
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for seed in range(1, 9):
        mat = material / f"seed-{seed}"
        cps = [dict(id=f"S{seed}-CP{n}", tokens=n, row_index=i) for n, i in ((20000, 4), (40000, 8))]
        facts = [dict(id=str(i), row_role="user", placement="head", **{"class": "limit"}) for i in range(6)]
        write(mat / "facts.json", facts)
        write(mat / "material.manifest.json", dict(seed=seed, checkpoints=cps, decision_checkpoint=cps[-1]))
        (mat / "lifecycle_probes.jsonl").write_text(json.dumps(dict(id="life", probe_token_position=20000)) + "\n")
        for arm in ("L0", "L1"):
            run = tmp_path / "runs" / arm / f"seed-{seed}" / "d1-r1"
            write(run / "summary.json", dict(events=[]))
            for cp in cps:
                ps = {f["id"]: {"class": "CORRECT" if arm == "L1" else "MISS"} for f in facts}
                if cp == cps[0]:
                    ps["life"] = dict(kind="stale_task", compaction_horizon=1, **{"class": "CORRECT"})
                sc = dict(arm=arm, seed=f"seed-{seed}", reader="glm-5.3", checkpoint_id=cp["id"], probes=ps,
                          behaviour=dict(compactions=1), accounting=dict(reader_input_tokens=1, reader_output_tokens=1, successful_probes=1),
                          run_dir=str(run / f"cp-{cp['id']}"), metrics=dict(facts_kept=dict(complete=True, denominator=6),
                          lifecycle=dict(complete=True, by_kind_horizon={})))
                path = scores / f"cp-{cp['id']}" / f"{arm}.seed-{seed}.d1-r1.json"
                write(path, sc)
                admission["entries"][str(path.relative_to(scores.parent))] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), receipt_sha256="a" * 64)
    write(scores.parent / "manifest.json", admission)
    paired = e.analyze(scores, material, list(range(1, 9)))
    assert paired["comparisons"]["L1−L0"]["verdict"] == "KEEP"
    assert paired["comparisons"]["L1−L0"]["intervals"]["facts_user"]["low"] == 100
    # Removing the same checkpoint from BOTH arms must never admit a subset as complete.
    for path in (scores / "cp-S1-CP20000").glob("*.json"):
        path.unlink()
    paired = e.analyze(scores, material, list(range(1, 9)))
    assert paired["comparisons"]["L1−L0"]["verdict"] == "INCONCLUSIVE"
    assert paired["comparisons"]["L1−L0"]["missing_seeds"] == [1]
    # A changed score without a matching admission receipt is excluded.
    path = scores / "cp-S2-CP20000/L1.seed-2.d1-r1.json"
    path.write_text("{}")
    paired = e.analyze(scores, material, list(range(1, 9)))
    assert paired["comparisons"]["L1−L0"]["missing_seeds"] == [1, 2]


def test_v4_loss_classification_excludes_future_unscored_sources(tmp_path):
    spec = importlib.util.spec_from_file_location("loss_class", Path(__file__).resolve().parents[1] / "loss_class.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "facts.json").write_text(json.dumps([dict(id=f, value=f, placement="head", **{"class": "limit"})
                                                   for f in ("past", "future")]))
    (tmp_path / "lifecycle_probes.jsonl").write_text("")
    scored = dict(run_dir=str(tmp_path / "run"), material=str(tmp_path), arm="L0", seed="seed-1", run="r1",
                  checkpoint_row=4, probes={"past": {"class": "MISS"}}, metrics={"facts_kept": {}})
    assert [f["id"] for f in module.classify_loss(scored)["facts"]] == ["past"]


def test_cost_counts_trap_abstention_but_not_fact_abstention():
    from score_s import accounting
    summary = dict(events=[], reader_calls=[dict(calls=[dict(prompt_tokens=10, completion_tokens=2)])])
    probes = dict(fact={"class": "ABSTAIN", "answer": "unknown", "success": False},
                  trap={"class": "ABSTAIN", "answer": "unknown", "success": True})
    result = accounting(summary, probes)
    assert result["successful_probes"] == 1 and result["cost_tokens_per_successful_task"] == 12
