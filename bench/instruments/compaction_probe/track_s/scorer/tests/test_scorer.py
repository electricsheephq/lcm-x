"""Formula, rule and propagation tests for score_s.py / report_s.py on synthetic fixtures."""
import json

import fixtures as fx
import pytest
import report_s as rp
import score_s as sc


@pytest.fixture
def mat(tmp_path):
    return fx.material(tmp_path)


def run_score(mat, run, arm):
    return sc.score(mat, run, arm)["metrics"]


@pytest.mark.parametrize("explicit_error", [False, True])
def test_reader_truncation_excluded_from_probe_metrics(mat, tmp_path, explicit_error):
    excluded = {"X-F1", "X-F2", "X-T0", "X-STATE.next_action"}
    ans = dict(fx.GOOD | fx.GOOD_CONT)
    ans.update(dict.fromkeys(excluded, ""))
    run = fx.s2_run(tmp_path, answers=ans)
    if explicit_error:
        rows = sc.jlines(run / "results.jsonl")
        for row in rows:
            if row["probe_id"] in excluded:
                row.update(status="ERROR", error="READER_TRUNCATED: completion cap 8192 reached")
        fx.wl(run / "results.jsonl", rows)
    else:
        for batch in ("X-B0", "X-BCONT"):
            fx.wj(run / "answers" / f"{batch}.json", {"batch": batch,
                "usage": {"completion_tokens": 8192}, "reader_calls": [{"completion_tokens": 8192}]})
    score = sc.score(mat, run, "LCMX-a")
    assert {pid for pid, p in score["probes"].items() if p["class"] == "READER_TRUNCATED"} == excluded
    m = score["metrics"]
    assert m["facts_kept"]["denominator"] == 4 and m["facts_kept"]["value"] == 1.0
    assert m["facts_kept"]["reader_truncated"] == 2
    assert m["stale_rate"]["denominator"] == 0 and m["stale_rate"]["value"] is None
    assert m["trap_abstention"]["denominator"] == 1 and m["trap_abstention"]["value"] == 1.0
    assert m["trap_failure_rate"]["value"] == 0.0
    assert m["continuation"]["denominator"] == 1 and m["continuation"]["value"] == 1.0
    assert m["recall"]["denominator"] == 1 and m["recall"]["value"] == 1.0
    for name in ("facts_kept", "stale_rate", "trap_abstention", "trap_failure_rate", "continuation", "recall"):
        assert m[name]["status"].startswith("INCOMPLETE") and not m[name]["complete"]


def test_cap_in_earlier_tool_call_is_not_final_truncation(mat, tmp_path):
    answers = dict(fx.GOOD | fx.GOOD_CONT)
    answers["X-F0"] = ""
    run = fx.s2_run(tmp_path, answers=answers)
    fx.wj(run / "answers/X-B0.json", {"batch": "X-B0", "usage": {"completion_tokens": 10},
        "reader_calls": [{"completion_tokens": 8192}, {"completion_tokens": 10}]})
    score = sc.score(mat, run, "LCMX-a")
    assert score["probes"]["X-F0"]["class"] == "HALLUCINATE"
    assert score["metrics"]["facts_kept"]["denominator"] == 6


def test_all_facts_truncated_has_no_rate(mat, tmp_path):
    run = fx.s2_run(tmp_path, answers=dict.fromkeys(fx.GOOD | fx.GOOD_CONT, ""))
    for file in (run / "answers").glob("*.json"):
        fx.wj(file, {"batch": file.stem, "usage": {"completion_tokens": 8192}})
    score = sc.score(mat, run, "LCMX-a")
    for name in ("facts_kept", "stale_rate", "trap_abstention", "trap_failure_rate", "continuation", "recall"):
        assert score["metrics"][name]["value"] is None
    assert score["metrics"]["facts_kept"]["denominator"] == 0
    assert len([p for p in score["probes"].values() if p["class"] == "READER_TRUNCATED"]) == 10


def test_admission_loss_outranks_reader_truncation(mat, tmp_path):
    answers = dict(fx.GOOD | fx.GOOD_CONT)
    answers["X-F1"] = ""
    run = fx.s2_run(tmp_path, answers=answers, missing=["X-F1"])
    fx.wj(run / "answers/X-B0.json", {"batch": "X-B0", "usage": {"completion_tokens": 8192}})
    score = sc.score(mat, run, "LCMX-a")
    assert score["probes"]["X-F1"]["class"] != "READER_TRUNCATED"
    m = score["metrics"]["facts_kept"]
    assert m["denominator"] == 6 and m["correct"] == 5 and m["reader_truncated"] == 0
    assert m["lost_before_compaction"]["ids"] == ["X-F1"]


def test_truncated_error_batch_excludes_only_unanswered(mat, tmp_path):
    answers = dict(fx.GOOD | fx.GOOD_CONT)
    answers["X-F1"] = ""
    run = fx.s2_run(tmp_path, answers=answers)
    rows = sc.jlines(run / "results.jsonl")
    for row in rows:
        if row["batch_id"] == "X-B0":
            row.update(status="ERROR", error="READER_TRUNCATED: completion cap 8192 reached")
    fx.wl(run / "results.jsonl", rows)
    score = sc.score(mat, run, "LCMX-a")
    assert {pid for pid, p in score["probes"].items() if p["class"] == "READER_TRUNCATED"} == {"X-F1"}
    assert score["probes"]["X-F0"]["class"] == "CORRECT" and score["probes"]["X-T0"]["class"] == "ABSTAIN"
    m = score["metrics"]["facts_kept"]
    assert m["denominator"] == 5 and m["correct"] == 5 and m["reader_truncated"] == 1


def test_facts_kept_formula_missing_rows_are_misses_and_incomplete(mat, tmp_path):
    ans = dict(fx.GOOD | fx.GOOD_CONT)
    del ans["X-F4"]                      # no row: miss + INCOMPLETE
    ans["X-F5"] = "zeta"                 # the kit rule needs the whole answer string: miss
    ans["X-F0"] = "ALPHA workspace."     # casefold + punctuation/whitespace removal: correct
    m = run_score(mat, fx.s2_run(tmp_path, answers=ans), "LCMX-a")["facts_kept"]
    assert m["value"] == pytest.approx(4 / 6) and m["denominator"] == 6
    assert m["status"].startswith("INCOMPLETE(1 of 6 scheduled facts")
    assert m["by_class_placement"]["name|tail"] == [0, 1] and m["by_class_placement"]["name|head"] == [1, 1]


def test_lost_before_compaction_is_a_miss_and_listed(mat, tmp_path):
    m = run_score(mat, fx.s2_run(tmp_path, missing=["X-F0"]), "LCMX-a")["facts_kept"]
    assert m["value"] == pytest.approx(5 / 6) and m["lost_before_compaction"]["ids"] == ["X-F0"]


def test_stale_rule_current_plus_stale_counts_as_stale(mat, tmp_path):
    ans = dict(fx.GOOD | fx.GOOD_CONT)
    ans["X-F2"] = "snapshot-gamma because cursors mixed (was: mutable-gamma because it avoids writes)"
    m = run_score(mat, fx.s2_run(tmp_path, answers=ans), "LCMX-a")
    assert m["stale_rate"]["value"] == 1.0 and m["stale_rate"]["stale_ids"] == ["X-F2"]
    assert m["stale_rate"]["lower_is_better"] and m["facts_kept"]["value"] == 1.0


def test_traps_and_continuation(mat, tmp_path):
    ans = dict(fx.GOOD | fx.GOOD_CONT)
    ans["X-T1"] = "alpha-workspace"      # a concrete registered value is never an abstention
    ans["X-STATE.status"] = "done"
    m = run_score(mat, fx.s2_run(tmp_path, answers=ans), "LCMX-a")
    assert m["trap_abstention"]["value"] == 0.5 and m["trap_failure_rate"]["value"] == 0.5
    assert m["continuation"]["value"] == 0.5 and m["continuation"]["status"] == "OK"


def test_continuity_strict_miss_is_fail_diagnostic_hit_on_dropped_period(mat, tmp_path):
    ev = [fx.s2_event(20.0, cont=(True, True, False), row_index=10)]
    text = "Audit the replay. Constraint: Never  modify the fixtures"   # period dropped, double space
    m = run_score(mat, fx.s2_run(tmp_path, events=ev, assembled=text), "LCMX-a")["continuity"]
    cell = next(g for g in m["grid"] if g["item"] == "X-CONT-constraint")
    assert cell["strict"] is False and cell["diagnostic"] is True
    assert m["status"].startswith("FAIL(strict miss: X-CONT-constraint")
    assert "Never modify the fixtures." not in text and sc.diag_present("Never modify the fixtures.", text)
    assert not sc.diag_present("Do not touch the fixtures.", text)  # a paraphrase fails both columns


def test_continuity_without_events_is_incomplete(mat, tmp_path):
    m = run_score(mat, fx.s2_run(tmp_path, events=[]), "LCMX-a")["continuity"]
    assert m["status"] == "INCOMPLETE(no recorded events)"


def test_three_recall_labels_s1_and_s2(mat, tmp_path):
    s1 = fx.s1_run(tmp_path, open_rows=[("X-F1", "src/beta/settings.toml", 0, "grep/describe only (DIAGNOSTIC)"),
                                        ("X-F3", "src/delta/loader.py: keep ids", 2, "grep/describe only (DIAGNOSTIC)"),
                                        ("X-F2", "snapshot-gamma because cursors mixed", 3, "delegated")])
    m = run_score(mat, s1, "lossless-claw-open")["recall"]
    labels = {a["id"]: a["label"] for a in m["per_answer"]}
    assert labels == {"X-F1": "context", "X-F3": "search only", "X-F2": "expand"}
    assert m["value"] == 1.0 and m["by_label"]["expand"] == [1, 1]
    s2 = fx.s2_run(tmp_path, arm="LCMX-a-open", kind="open", tools=["lcm_grep", "lcm_expand_query"])
    assert {a["label"] for a in run_score(mat, s2, "LCMX-a-open")["recall"]["per_answer"]} == {"expand"}
    s2g = fx.s2_run(tmp_path, arm="LCMX-b-open", kind="open", tools=["lcm_grep", "lcm_describe"])
    assert {a["label"] for a in run_score(mat, s2g, "LCMX-b-open")["recall"]["per_answer"]} == {"search only"}


def test_recall_absolutes_and_receipt_untested(mat, tmp_path):
    ans = dict(fx.GOOD | fx.GOOD_CONT)
    ans["X-F1"] = "I don't know"         # clipped-middle fact not reached
    m = run_score(mat, fx.s2_run(tmp_path, answers=ans), "LCMX-a")["recall"]
    assert m["status"] == "FAIL(clipped-middle fact unreachable: X-F1)" and m["denominator"] == 3
    m2 = run_score(mat, fx.s2_run(tmp_path, arm="LCMX-u", receipts={"X-F2": "UNTESTED", "X-F3": "PRESENT"}), "LCMX-u")
    assert m2["recall"]["status"].startswith("UNTESTED(X-F2") and not m2["recall"]["complete"]


def test_codex_tool_use_breaks_context_only_rule(mat, tmp_path):
    m = run_score(mat, fx.s4_run(tmp_path, tool_calls=1), "codex-native")["recall"]
    assert m["status"].startswith("INCOMPLETE(") and "context-only rule violated" in m["status"]


def test_latency_over_120_is_fail_and_under_5_events_incomplete(mat, tmp_path):
    ev = [fx.s2_event(w, n=i) for i, w in enumerate((10.0, 20.0, 30.0, 40.0, 121.0))]
    m = run_score(mat, fx.s2_run(tmp_path, events=ev), "LCMX-a")["latency"]
    assert m["status"].startswith("FAIL(event > 120 s") and m["gate_events"]["max"] == 121.0
    ev4 = [fx.s2_event(w, n=i) for i, w in enumerate((10.0, 20.0, 30.0, 40.0))]
    m4 = run_score(mat, fx.s2_run(tmp_path, arm="LCMX-b", events=ev4), "LCMX-b")["latency"]
    assert m4["status"].startswith("INCOMPLETE(4 gate events < 5") and m4["gate_events"]["p90"] == 40.0
    wired = ev4 + [fx.s2_event(200.0, timing="WIRING-ONLY", n=4)]
    m5 = run_score(mat, fx.s2_run(tmp_path, arm="LCMX-c", events=wired), "LCMX-c")["latency"]
    assert not m5["status"].startswith("FAIL") and m5["all_events"]["max"] == 200.0   # WIRING-ONLY never gates
    ok = run_score(mat, fx.s2_run(tmp_path, arm="LCMX-d"), "LCMX-d")["latency"]
    assert ok["status"] == "OK" and ok["p90_le_60"] is True


def test_level3_rules(mat, tmp_path):
    ev = [fx.s2_event(10.0, level=3, n=0), fx.s2_event(10.0, level=1, n=1)]
    m = run_score(mat, fx.s2_run(tmp_path, events=ev), "LCMX-a")["level3"]
    assert m["value"] == 50.0 and m["status"].startswith("FAIL(level-3 leaf with an answering route")
    ev2 = [fx.s2_event(60.0, level=3, error=True, n=0), fx.s2_event(10.0, n=1)]
    assert run_score(mat, fx.s2_run(tmp_path, arm="LCMX-b", events=ev2), "LCMX-b")["level3"]["status"] == "OK"
    none = run_score(mat, fx.s2_run(tmp_path, arm="LCMX-c", events=[]), "LCMX-c")["level3"]
    assert none["status"] == "INCOMPLETE(zero leaves written)"
    assert run_score(mat, fx.s4_run(tmp_path), "codex-native")["level3"]["status"].startswith("N/A(")


def test_s1_and_s4_adapters_read_events(mat, tmp_path):
    s1 = run_score(mat, fx.s1_run(tmp_path), "lossless-claw")
    assert s1["latency"]["gate_events"]["p90"] == 10.0 and s1["continuity"]["status"] == "OK"
    assert s1["continuation"]["status"].startswith("INCOMPLETE(no continuation_field row")
    s4 = run_score(mat, fx.s4_run(tmp_path, stall_ms=(50000, None)), "codex-native")["latency"]
    assert "event wall unmeasured" in s4["status"]


def seedv(a, b):
    return (a + b) / 2, abs(a - b), None


def test_f2_parity_and_superiority_including_ties():
    tie = rp.seed_verdict(seedv(0.70, 0.74), seedv(0.72, 0.76), "parity", False)      # d=-0.02, band=0.04
    assert tie["verdict"] == "PASS" and tie["band"] == pytest.approx(0.04)
    assert rp.seed_verdict(seedv(0.70, 0.74), seedv(0.72, 0.76), "superiority", False)["verdict"] == "FAIL"
    assert rp.seed_verdict(seedv(0.80, 0.80), seedv(0.80, 0.80), "superiority", False)["verdict"] == "FAIL"
    assert rp.seed_verdict(seedv(0.90, 0.92), seedv(0.70, 0.72), "superiority", False)["verdict"] == "PASS"
    far = rp.seed_verdict(seedv(0.50, 0.50), seedv(0.80, 0.82), "parity", False)       # d=-0.31 < -2 x 0.02
    assert far["verdict"] == "FAIL" and far["beyond_2band"]
    near = rp.seed_verdict(seedv(0.78, 0.78), seedv(0.80, 0.82), "parity", False)      # d=-0.03, band 0.02
    assert near["verdict"] == "FAIL" and not near["beyond_2band"]                      # -0.03 >= -2 x 0.02
    p, f = {"verdict": "PASS", "beyond_2band": False}, {"verdict": "FAIL", "beyond_2band": False}
    assert rp.overall([p, p, f], "parity", 3) == "PASS"
    assert rp.overall([p, p, dict(f, beyond_2band=True)], "parity", 3) == "FAIL"
    assert rp.overall([p, f, f], "superiority", 3) == "FAIL" and rp.overall([p, p, f], "superiority", 3) == "PASS"


def test_f5_direction_on_a_lower_is_better_metric():
    v = rp.seed_verdict(seedv(0.30, 0.30), seedv(0.10, 0.10), "parity", True)   # candidate 0.30, comparator 0.10
    assert v["d"] == pytest.approx(-0.20) and v["verdict"] == "FAIL"
    assert rp.seed_verdict(seedv(0.10, 0.10), seedv(0.30, 0.30), "parity", True)["verdict"] == "PASS"


def test_incomplete_propagation_one_missing_run(mat, tmp_path):
    out = tmp_path / "scores"
    for arm, runs in (("LCMX-a", (1, 2)), ("lossless-claw", (1,))):
        for r in runs:
            d = fx.s2_run(tmp_path / arm, arm=arm, run=r)
            s = sc.score(mat, d, arm)
            (out / f"{arm}.{r}.json").parent.mkdir(exist_ok=True)
            (out / f"{arm}.{r}.json").write_text(json.dumps(s))
    runs = rp.load(out)
    res = rp.compare(runs, "LCMX-a", "lossless-claw", "facts_kept", "parity", ["seed-x"], 1)
    assert res["seeds"]["seed-x"] == {"verdict": "INCOMPLETE", "reason": "comparator: 1 of 2 runs"}
    assert res["overall"].startswith("INCOMPLETE")
    md, js = rp.render(runs, 3)
    assert "| Visible wait (Phase 3) | NOT IN TRACK S |" in md and "INCOMPLETE" in js["gates"]["quality_vs_lc/LCMX-a/facts_kept"]["overall"]
    missing = rp.compare(runs, "LCMX-a", "lossless-claw-open", "recall", "parity", ["seed-x"], 1)
    assert missing["seeds"]["seed-x"]["reason"] == "comparator: lossless-claw-open missing"


def test_full_two_runs_three_arms_pass_path(mat, tmp_path):
    out = tmp_path / "scores"
    out.mkdir()
    for r in (1, 2):
        for arm, d in (("LCMX-a", fx.s2_run(tmp_path / f"a{r}", arm="LCMX-a", run=r)),
                       ("lossless-claw", fx.s1_run(tmp_path / f"l{r}", run=r)), ("codex-native", fx.s4_run(tmp_path / f"c{r}", run=r))):
            (out / f"{arm}.{r}.json").write_text(json.dumps(sc.score(mat, d, arm)))
    runs = rp.load(out)
    res = rp.compare(runs, "LCMX-a", "lossless-claw", "facts_kept", "parity", ["seed-x"], 1)
    assert res["seeds"]["seed-x"]["verdict"] == "PASS" and res["overall"] == "PASS"
    cont = rp.compare(runs, "LCMX-a", "lossless-claw", "continuation", "parity", ["seed-x"], 1)
    assert cont["overall"].startswith("INCOMPLETE(1 of 1 seeds; comparator: INCOMPLETE: no continuation_field row")
    stale = rp.compare(runs, "LCMX-a", "lossless-claw", "stale_rate", "parity", ["seed-x"], 1)
    assert stale["seeds"]["seed-x"]["verdict"] == "PASS" and stale["overall"] == "PASS"


def test_non_string_answer_in_a_capped_batch_is_scored_not_raised(mat, tmp_path):
    answers = dict(fx.GOOD | fx.GOOD_CONT)
    answers["X-F0"] = 42  # an external arm can store a JSON number
    run = fx.s2_run(tmp_path, answers=answers)
    for file in (run / "answers").glob("*.json"):
        fx.wj(file, {"batch": file.stem, "usage": {"completion_tokens": 8192}})
    score = sc.score(mat, run, "LCMX-a")
    assert score["probes"]["X-F0"]["class"] not in ("READER_TRUNCATED", "CORRECT")
