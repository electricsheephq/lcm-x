"""Synthetic reader receipts: no model calls or external material."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hang_verdict", Path(__file__).resolve().parents[1] / "hang_cell_verdict.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def receipt(mode, counts=(1, 1)):
    return dict(verdict=dict(result="FAIL", complete=True, bars={str(i): False for i in range(6)}),
                pooled=dict(synthetic=dict(runs=2, faulted_calls=sum(counts), fault_mode=mode, long_waits=2,
                    max_long_waits_in_one_turn=1, max_turn_wait_s=115.0, dropped_rows=0, survival_fits=0,
                    c_per_100=10.0, waited_turn_share=0.1, mean_turn_wait_s=12.0, facts_304_mean=0.5)),
                per_run={f"synthetic|d{n}-r1": dict(status="DONE", fault_mode=mode, faulted_calls=c,
                                                long_waits=1, max_long_waits_in_one_turn=1)
                         for n, c in enumerate(counts, 1)})


def invoke(tmp_path, mode, base, candidate, bad_cell=None, bad_summary=None):
    for name, data in (("base", base), ("candidate", candidate)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "pmw-read.json").write_text(json.dumps(data))
        cell = dict(product_sha=f"{name}-sha", fault=mode, seeds=[1, 2], kit_sha="kit-sha", arm="synthetic")
        if bad_cell and bad_cell[0] == name:
            key, value = bad_cell[1:]
            if key == "missing":
                continue
            if key == "unreadable":
                (directory / "cell.json").write_text("{")
                continue
            cell[key] = value
        (directory / "cell.json").write_text(json.dumps(cell))
        for seed in cell["seeds"]:
            summary = dict(material_sha256=str(seed) * 64)
            if bad_summary and bad_summary[:2] == (name, seed):
                bad = bad_summary[2]
                if bad == "missing":
                    continue
                if bad == "no_digest":
                    summary = {}
                elif bad == "empty_digest":
                    summary["material_sha256"] = ""
                elif bad == "different":
                    summary["material_sha256"] = "f" * 64
            path = directory / "lcmx-runs" / cell["arm"] / f"seed-{seed}" / f"d{seed}-r1" / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{" if bad_summary == (name, seed, "unreadable") else json.dumps(summary))
    output = tmp_path / "verdict.json"
    rc = v.main(["--base", str(tmp_path / "base/pmw-read.json"),
                 "--candidate", str(tmp_path / "candidate/pmw-read.json"),
                 "--base-sha", "base-sha", "--candidate-sha", "candidate-sha",
                 "--variant", mode, "--json", str(output)])
    return rc, json.loads(output.read_text())


@pytest.mark.parametrize("mode", ["T", "P"])
def test_pass_and_d7_never_gates(tmp_path, mode):
    base = receipt(mode)
    candidate = receipt(mode, (2, 1)) if mode == "P" else copy.deepcopy(base)
    if mode == "P":
        candidate["pooled"]["synthetic"].update(long_waits=3, dropped_rows=42)
        candidate["per_run"]["synthetic|d1-r1"]["long_waits"] = 2
    rc, out = invoke(tmp_path, mode, base, candidate)
    assert rc == 0 and out["result"] == "PASS"
    assert out["base"]["d7"]["result"] == "FAIL"
    assert out["candidate"]["facts_304_mean"] == 0.5
    assert out["material_sha256"] == {str(seed): str(seed) * 64 for seed in (1, 2)}


@pytest.mark.parametrize("mode,change", [("T", dict(dropped_rows=1)), ("T", dict(long_waits=3)),
                                        ("P", dict(long_waits=3)), ("P", dict(max_long_waits_in_one_turn=2))])
def test_fail(tmp_path, mode, change):
    base, candidate = receipt(mode), receipt(mode)
    if mode == "P":
        candidate["per_run"]["synthetic|d1-r1"].update(change)
        candidate["pooled"]["synthetic"].update(
            long_waits=sum(run["long_waits"] for run in candidate["per_run"].values()),
            max_long_waits_in_one_turn=max(run["max_long_waits_in_one_turn"] for run in candidate["per_run"].values()))
    else:
        candidate["pooled"]["synthetic"].update(change)
    assert invoke(tmp_path, mode, base, candidate)[0] == 1


@pytest.mark.parametrize("mode", ["T", "P"])
@pytest.mark.parametrize("bad", ["invalid", "incomplete", "missing", "no_fault", "mode", "mixed", "run_zero", "sum", "run_missing"])
@pytest.mark.parametrize("side", ["base", "candidate"])
def test_invalid(tmp_path, mode, bad, side):
    base, candidate = receipt(mode), receipt(mode)
    data = base if side == "base" else candidate
    x = data["pooled"]["synthetic"]
    if bad == "invalid":
        data["verdict"]["result"] = "INVALID"
    elif bad == "incomplete":
        data["verdict"]["complete"] = False
    elif bad == "missing":
        del x["max_turn_wait_s"]
    elif bad == "no_fault":
        x["faulted_calls"] = 0
    elif bad in ("mode", "mixed"):
        x["fault_mode"] = ("P" if mode == "T" else "T") if bad == "mode" else "mixed"
    elif bad == "run_zero":
        data["per_run"]["synthetic|d1-r1"]["faulted_calls"] = 0
    elif bad == "sum":
        x["faulted_calls"] += 1
    else:
        del data["per_run"]["synthetic|d1-r1"]
    assert invoke(tmp_path, mode, base, candidate)[0] == 2


def test_t_rejects_uneven_counts_even_with_equal_pool(tmp_path):
    assert invoke(tmp_path, "T", receipt("T"), receipt("T", (0, 2)))[0] == 2


def test_p_allows_one_extra_wait_per_run(tmp_path):
    base, candidate = receipt("P"), receipt("P", (2, 2))
    candidate["pooled"]["synthetic"]["long_waits"] = 4
    for data, waits in ((base, 1), (candidate, 2)):
        for run in data["per_run"].values():
            run.update(long_waits=waits, max_long_waits_in_one_turn=1)
    rc, out = invoke(tmp_path, "P", base, candidate)
    assert rc == 0
    assert out["per_run"] == {run_id: dict(base=dict(long_waits=1, max_long_waits_in_one_turn=1),
                                         candidate=dict(long_waits=2, max_long_waits_in_one_turn=1))
                             for run_id in base["per_run"]}


@pytest.mark.parametrize("mode", ["T", "P"])
def test_invalid_different_run_ids(tmp_path, mode):
    base, candidate = receipt(mode), receipt(mode)
    candidate["per_run"]["synthetic|d3-r1"] = candidate["per_run"].pop("synthetic|d1-r1")
    assert invoke(tmp_path, mode, base, candidate)[0] == 2


@pytest.mark.parametrize("mode", ["T", "P"])
@pytest.mark.parametrize("bad", ["one_run", "arm", "run_ids"])
def test_invalid_receipt_cell_binding(tmp_path, mode, bad):
    base, candidate = receipt(mode), receipt(mode)
    # Corrupt both sides alike so cross-cell equality cannot mask a stale reader.
    for data in (base, candidate):
        if bad == "one_run":
            del data["per_run"]["synthetic|d2-r1"]
            data["pooled"]["synthetic"].update(runs=1, faulted_calls=1, long_waits=1)
        elif bad == "arm":
            data["pooled"]["other"] = data["pooled"].pop("synthetic")
        else:
            data["per_run"]["synthetic|d3-r1"] = data["per_run"].pop("synthetic|d1-r1")
    rc, out = invoke(tmp_path, mode, base, candidate)
    assert rc == 2 and out["result"] == "INVALID"


@pytest.mark.parametrize("mode", ["T", "P"])
@pytest.mark.parametrize("side", ["base", "candidate"])
@pytest.mark.parametrize("bad", ["missing", "unreadable", "no_digest", "empty_digest"])
def test_invalid_summary_material(tmp_path, mode, side, bad):
    rc, out = invoke(tmp_path, mode, receipt(mode), receipt(mode), bad_summary=(side, 2, bad))
    assert rc == 2 and out["result"] == "INVALID"


@pytest.mark.parametrize("mode", ["T", "P"])
def test_invalid_different_material(tmp_path, mode):
    rc, out = invoke(tmp_path, mode, receipt(mode), receipt(mode), bad_summary=("candidate", 2, "different"))
    assert rc == 2 and out["result"] == "INVALID"


@pytest.mark.parametrize("mode", ["T", "P"])
@pytest.mark.parametrize("side", ["base", "candidate"])
@pytest.mark.parametrize("key,value", [("missing", None), ("unreadable", None), ("product_sha", "wrong"),
                                       ("fault", "wrong"), ("seeds", [2, 3]), ("kit_sha", "wrong"), ("arm", "wrong")])
def test_invalid_cell_identity(tmp_path, mode, side, key, value):
    assert invoke(tmp_path, mode, receipt(mode), receipt(mode), (side, key, value))[0] == 2


def test_invalid_json(tmp_path):
    (tmp_path / "bad.json").write_text("{")
    assert v.main(["--base", str(tmp_path / "bad.json"), "--candidate", str(tmp_path / "missing.json"),
                   "--base-sha", "base-sha", "--candidate-sha", "candidate-sha",
                   "--variant", "T", "--json", str(tmp_path / "out.json")]) == 2
