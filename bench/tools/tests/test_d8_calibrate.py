import hashlib
import json

import pytest

from bench.tools.d8_calibrate import calibrate, loss_power, main, margin_sweep, markdown, trinomial


def write_pair(root, dataset, counts, missing=False):
    rows = []
    for category, n in (("abstention", 30), ("knowledge-update", 78), ("other", 392)) if dataset == "longmemeval" else (
        ("adversarial", 446), ("other", 1540)):
        for i in range(n):
            qid = f"{category}-{i}_abs" if category == "abstention" else f"{category}-q{i}"
            rows.append({"questionId": qid, "questionType": category, "score": 1,
                         "recallHealth": {"coverage": {"fts": "ok"}, "timeout": False, "degraded": False}})
    a, aprime = [], []
    seen = {}
    for row in rows:
        category = row["questionType"]
        index = seen.get(category, 0)
        seen[category] = index + 1
        b, c = counts.get(category, (0, 0))
        a.append({**row, "score": int(index >= b)})
        aprime.append({**row, "score": int(not b <= index < b + c)})
    if missing:
        for row in (*a, *aprime):
            if row["questionType"] == "abstention":
                row["questionId"] = row["questionId"].replace("_abs", "_other")
    paths = []
    for name, values in (("a", a), ("aprime", aprime)):
        path = root / name
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps({
            "benchmark": dataset, "recallConfig": "OFF", "evaluations": values}))
        paths.append(path)
    return ["synthetic", dataset, "OFF", *paths]


@pytest.mark.parametrize("dataset,counts", [
    ("longmemeval", {"abstention": (2, 0), "knowledge-update": (1, 4)}),
    ("locomo", {"adversarial": (12, 21)}),
])
def test_known_counts_and_read_only_inputs(tmp_path, dataset, counts):
    pair = write_pair(tmp_path, dataset, counts)
    paths = sorted(tmp_path.glob("*/report.json"))
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    result = calibrate([pair])
    assert result["pairs"][0]["status"] == "READ"
    for category, (b, c) in counts.items():
        row = result["pairs"][0]["categories"][category]
        assert (row["b"], row["c"], row["net"]) == (b, c, b-c)
        assert row["n"] == {"abstention": 30, "knowledge-update": 78, "adversarial": 446}[category]
        assert row["margin"] == (9 if dataset == "locomo" else 4)
        assert row["count_pass"] is (abs(b-c) < row["margin"])
        assert row["pass_count"] is row["count_pass"]
        assert not row["interval_pass"]
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert "Count passes" in markdown(result)


def test_missing_category_incomplete_never_counted_as_pass(tmp_path):
    result = calibrate([write_pair(tmp_path, "longmemeval", {}, missing=True)])
    assert result["pairs"][0]["status"] == "INCOMPLETE"
    assert result["pairs"][0]["categories"]["abstention"]["n"] == 0
    assert all(row["count_passes"] == row["interval_passes"] == 0 for row in result["summary"])
    assert all(row["incomplete"] == 1 for row in result["summary"])


def test_cli_unread_redacts_input_text_and_continues(tmp_path, capsys):
    pair = write_pair(tmp_path / "good", "longmemeval", {})
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "report.json").write_text('{"private benchmark text')
    json_output, md_output = tmp_path / "counts.json", tmp_path / "counts.md"
    args = ["--pair", "unread", "locomo", "OFF", str(broken), str(broken),
            "--pair", *(str(v) for v in pair), "--json-output", str(json_output),
            "--markdown-output", str(md_output)]
    assert main(args) == 2
    assert "private benchmark text" not in capsys.readouterr().out + json_output.read_text() + md_output.read_text()
    result = json.loads(json_output.read_text())
    assert result["pairs"][0]["status"] == "UNREAD"
    assert result["pairs"][1]["status"] == "READ"
    assert sum(row["unread"] for row in result["summary"]) == 1
    assert sum(row["count_passes"] for row in result["summary"]) == 2


def receipt_pair(root, b, c, dataset="longmemeval"):
    total = 500 if dataset == "longmemeval" else 1986
    ids = [f"q{i}_abs" if i < 30 else f"q{i}" for i in range(total)]
    a = {qid: "incorrect" if i < b else "correct" for i, qid in enumerate(ids)}
    aprime = {qid: "error" if b <= i < b + c else "correct" for i, qid in enumerate(ids)}
    # Counts cannot be substituted for paired discordances, even with a suggestive id.
    path = root / "row.json"
    path.write_text(json.dumps({"bench": dataset, "status": "COMPLETE", **{
        name: {"outcomes": values, "status": "COMPLETE", "health_status": "COMPLETE",
               "accounting": {"failed_including_missing": 0},
               "per_category": {"knowledge-update": {"correct": 73, "expected": 78},
                                "adversarial": {"correct": 211, "expected": 446}}}
        for name, values in (("arm", a), ("aprime", aprime))}}))
    return ["receipt", dataset, "LOCAL", path, path]


@pytest.mark.parametrize("b,c,floor3_pass", [(1, 1, True), (4, 1, False), (1, 4, False)])
def test_receipt_pins_release_manager_hand_figures(tmp_path, b, c, floor3_pass):
    pair = receipt_pair(tmp_path, b, c)
    before = hashlib.sha256(pair[3].read_bytes()).hexdigest()
    result = calibrate([pair])
    item = result["pairs"][0]
    assert item["layout"] == "row.json" and item["status"] == "PARTIAL"
    row = item["categories"]["abstention"]
    assert (row["n"], row["b"], row["c"]) == (30, b, c)
    # The hand measurement pins magnitude; swapping A/A-prime cannot change it.
    assert sorted((row["b"], row["c"])) == ([1, 1] if floor3_pass else [1, 4])
    assert abs(row["net"]) == (0 if floor3_pass else 3)
    # T1-66: v0.29.2 base passes at both floors; rc2 fails at 3, passes at 4.
    assert (abs(row["net"]) < 3) is floor3_pass
    assert row["margin"] == 4
    assert row["count_pass"] is row["pass_count"] is True
    assert item["categories"]["knowledge-update"]["status"] == "UNREAD"
    assert "longmemeval_s_cleaned.json" in item["categories"]["knowledge-update"]["reason"]
    assert hashlib.sha256(pair[3].read_bytes()).hexdigest() == before


def test_receipt_locomo_never_infers_category_from_totals(tmp_path):
    result = calibrate([receipt_pair(tmp_path, 1, 4, "locomo")])
    row = result["pairs"][0]["categories"]["adversarial"]
    assert row["status"] == "UNREAD"
    assert row["b"] is row["c"] is None
    assert "locomo10.json" in row["reason"]


@pytest.mark.parametrize("defect", ["missing candidate", "wrong total", "failed receipt", "missing health"])
def test_receipt_completeness_is_not_fabricated(tmp_path, defect):
    pair = receipt_pair(tmp_path, 1, 1)
    data = json.loads(pair[3].read_text())
    if defect == "missing candidate":
        del data["aprime"]["outcomes"]["q0_abs"]
    elif defect == "wrong total":
        for side in ("arm", "aprime"):
            del data[side]["outcomes"]["q499"]
    elif defect == "failed receipt":
        data["status"] = "INCOMPLETE"
    else:
        del data["arm"]["health_status"]
    pair[3].write_text(json.dumps(data))
    row = calibrate([pair])["pairs"][0]["categories"]["abstention"]
    assert row["status"] == "INCOMPLETE"
    assert not row["pass_count"] and not row["pass_interval"]


def test_margin_sweep_small_pair_arithmetic_and_exclusions():
    good = {"status": "READ", "n": 4, "b": 0, "c": 3, "net": -3, "interval_pass": False}
    result = {"summary": [{"configuration": "OFF", "category": "knowledge-update",
                           "unread": 1, "incomplete": 1}], "pairs": [
        {"id": name, "configuration": "OFF", "categories": {"knowledge-update": row}}
        for name, row in (("small", good), ("unread", {"status": "UNREAD"}),
                          ("incomplete", {**good, "status": "INCOMPLETE"}))]}
    rows = margin_sweep(result)["rows"]
    assert [r["false_fails"] for r in rows] == [1, 0, 0, 0]
    assert [r["failing_pairs"] for r in rows] == [["small"], [], [], []]
    assert [r["eligible"] for r in rows] == [1, 1, 0, 0]
    assert [r["interval_false_fails"] for r in rows] == [1, 1, 1, 1]
    assert all(r["power"] is None and r["unread"] == r["incomplete"] == 1 for r in rows)


def test_power_arithmetic_small_synthetic_population():
    # No noise, two trials, loss probability .1: P(c>=1)=.19, P(c>=2)=.01.
    one = loss_power(2, 0, .1, 1)
    two = loss_power(2, 0, .1, 2)
    assert one["count"] == pytest.approx(.19)
    assert two["count"] == pytest.approx(.01)
    # A single loss already puts the exact lower quantile below -2 points.
    assert one["interval"] == two["interval"] == pytest.approx(.19)
    assert sum(p for _, _, p in trinomial(4, .2, .3)) == pytest.approx(1)


def test_sweep_retains_two_percent_ceiling_above_margin_floor():
    result = {"summary": [{"configuration": "LOCAL", "category": "knowledge-update",
                           "unread": 0, "incomplete": 0}], "pairs": [
        {"id": "ceil", "configuration": "LOCAL", "categories": {"knowledge-update": {
            "status": "READ", "n": 151, "b": 0, "c": 3, "net": -3, "interval_pass": False}}}]}
    rows = margin_sweep(result)["rows"]
    assert [r["eligible"] for r in rows] == [1, 1, 1, 1]
    assert [r["false_fails"] for r in rows] == [0, 0, 0, 0]


def test_sweep_abstention_pools_complete_rows_and_keeps_guard_margin(tmp_path):
    from bench.tools.d8_guard import count_margin

    result = calibrate([receipt_pair(tmp_path, 1, 4)])
    rows = [row for row in margin_sweep(result)["rows"] if row["category"] == "abstention"]
    assert [r["false_fails"] for r in rows] == [1, 0, 0, 0]
    assert all(r["interval_false_fails"] == 1 for r in rows)
    assert all(r["power"]["pooled_discordance"] == 5 / 30 for r in rows)
    assert all(r["power"]["10"]["count"] > rows[i+1]["power"]["10"]["count"]
               for i, r in enumerate(rows[:-1]))
    assert count_margin(30) == 4


def test_cli_receipt_writes_sweep_and_discloses_partial(tmp_path, capsys):
    pair = receipt_pair(tmp_path, 1, 4)
    outputs = {name: tmp_path / name for name in ("cal.json", "cal.md", "sweep.json", "sweep.md")}
    args = ["--pair", *(str(v) for v in pair), "--json-output", str(outputs["cal.json"]),
            "--markdown-output", str(outputs["cal.md"]), "--sweep-json-output", str(outputs["sweep.json"]),
            "--sweep-markdown-output", str(outputs["sweep.md"])]
    assert main(args) == 2  # Unknown categories are not a successful full calibration.
    assert all(path.is_file() for path in outputs.values())
    assert "row.json" in capsys.readouterr().out
    assert json.loads(outputs["sweep.json"].read_text())["rows"]
