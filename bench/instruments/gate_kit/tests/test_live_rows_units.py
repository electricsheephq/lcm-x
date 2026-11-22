"""Unit tests for the live-rows producer (#2868): no tree is imported here."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from bench.instruments.gate_kit.live_rows import build, lockwatch, rt, shape, verdicts  # noqa: E402
from bench.instruments.gate_kit.live_rows.host import auxiliary_client as aux  # noqa: E402
from bench.instruments.gate_kit.live_rows.l1_planner import WindowBudgetExceeded, counting_gap, reporter_labels  # noqa: E402

BARS = rt.load_kit()["bars"]


@pytest.fixture
def kit():
    return build.scaled(rt.load_kit(), 0.05)


@pytest.fixture(autouse=True)
def fresh_stand_in():
    saved = dict(aux.SETTINGS)
    aux.CALLS.clear()
    aux.HOOKS.clear()
    yield
    aux.SETTINGS.clear()
    aux.SETTINGS.update(saved)
    aux.CALLS.clear()
    aux.HOOKS.clear()


def test_shape_is_deterministic_and_hashed(kit):
    first, second = shape.build(kit), shape.build(copy.deepcopy(kit))
    assert first["input_hash"] == second["input_hash"]
    assert first["target_list"] == second["target_list"]
    changed = copy.deepcopy(kit)
    changed["seed"] += 1
    assert shape.build(changed)["input_hash"] != first["input_hash"]


def test_host_list_hides_whole_turns_and_keeps_tool_pairs(kit):
    built = shape.build(kit)
    history, live = built["target_history"], built["target_list"]
    assert 0 < len(live) < len(history)
    assert built["counts"]["target_hidden_rows"] == len(history) - len(live)
    positions = {id(row): index for index, row in enumerate(history)}
    order = [positions[id(row)] for row in live]
    assert order == sorted(order)  # the host list is a subsequence of the stored history
    calls = {c["id"] for row in live for c in row.get("tool_calls", [])}
    results = {row["tool_call_id"] for row in live if row["role"] == "tool"}
    assert calls == results  # no orphan call or result in the host list
    assert live[-1] is not history[-1]  # the stored tail turn is not in the host list


def test_full_kit_counts_are_the_draft_stand_in():
    counts = shape.build(rt.load_kit())["counts"]
    assert counts["rows"] >= 40_000 and counts["conversations"] >= 2
    assert counts["target_hidden_rows"] * 4 * 100 > 5 * 24_000  # hidden runs well past five chunks of tokens
    assert 1_700 <= counts["target_list_rows"] <= 1_900


def nonce_messages(tokens: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": 'policy. first line is exactly:\n<lcm-summary nonce="abc123">\n'},
            {"role": "user", "content": "x" * (tokens * 4)}]


def test_stand_in_latency_follows_size_with_a_floor():
    aux.configure(floor_s=44.4, rate_tps=2970.0)
    assert aux.latency_s(6_707) == pytest.approx(44.4)
    assert aux.latency_s(112_867) == pytest.approx(44.4)  # 38.0 s on record is below the floor
    assert aux.latency_s(300_000) == pytest.approx(300_000 / 2970.0)


def test_stand_in_answers_in_the_envelope_and_shorter_than_its_source():
    aux.configure(floor_s=0.01, rate_tps=1e9)
    reply = aux.call_llm(task="compression", messages=nonce_messages(4_000), max_tokens=2_000, timeout=5)
    content = reply.choices[0].message.content
    assert content.startswith('<lcm-summary nonce="abc123">') and content.endswith("</lcm-summary>")
    assert "Expand for details about:" in content.splitlines()[-2]
    assert len(content) < 4_000 * 4
    assert aux.CALLS[-1]["outcome"] == "ok"


def test_stand_in_is_cut_at_the_time_it_is_given():
    aux.configure(floor_s=5.0, rate_tps=1e9)
    started = time.monotonic()
    with aux.aux_stream_deadline(time.monotonic() + 0.2):
        with pytest.raises(TimeoutError, match="Request timed out"):
            aux.call_llm(messages=nonce_messages(100), timeout=3.0)
    assert time.monotonic() - started < 1.0
    record = aux.CALLS[-1]
    assert record["outcome"] == "cut" and record["granted_s"] <= 0.21 and record["slept_s"] <= 0.5


def test_stand_in_largest_input_on_record_needs_its_full_latency():
    aux.configure(floor_s=44.4, rate_tps=2970.0)
    assert aux.latency_s(112_867) > 26  # the first live summary call had 26 s and timed out (#2869)


def test_lock_holds_merge_short_gaps_and_filter_by_pid():
    ms = 1_000_000
    transitions = [[0, 0], [10 * ms, 111], [20 * ms, 0], [20 * ms + 1000, 111], [50 * ms, 0], [60 * ms, 222],
                   [90 * ms, 0], [100 * ms, 0]]
    found = lockwatch.holds(transitions, 0, 100 * ms)
    assert found == [(10 * ms, 50 * ms), (60 * ms, 90 * ms)]
    assert lockwatch.holds(transitions, 0, 100 * ms, pid=222) == [(60 * ms, 90 * ms)]
    assert lockwatch.longest_s(transitions, 0, 30 * ms) == pytest.approx(0.02)


def test_planner_counter_counts_the_reflected_comparison_and_aborts():
    counter = {"windows": 0, "active": True, "aborted": False}
    gap = counting_gap(1024, counter, limit=3)
    assert (2000 >= gap) and not (10 >= gap)
    assert counter["windows"] == 2
    counter["active"] = False
    assert 5000 >= gap and counter["windows"] == 2  # counted only inside the planner
    counter["active"] = True
    with pytest.raises(WindowBudgetExceeded):
        for _ in range(5):
            _ = 1 >= gap
    assert counter["aborted"]


def test_reporter_labels_are_the_2840_shape():
    labels, middle = reporter_labels(1 << 20, 1 << 10)
    assert len(labels) == 20_014 and labels == sorted(set(labels))
    assert labels[middle] - labels[middle - 1] == 1


# -- verdicts, both ways ------------------------------------------------------------------------------------------

def d9_receipt(hold=10.0, **over):
    return {"longest_hold_s": hold, "pass_completed": True, "exception": None, "rows_examined": 100, **over}


@pytest.mark.parametrize("judged,expected", [
    (d9_receipt(10.0), "PASS"), (d9_receipt(31.0), "FAIL"),
    (d9_receipt(5.0, exception="RuntimeError"), "INCONCLUSIVE"),
    (d9_receipt(5.0, pass_completed=False), "INCONCLUSIVE"),
    (d9_receipt(5.0, rows_examined=99), "INCONCLUSIVE"), (None, "UNRUN")])
def test_d9(judged, expected):
    result = verdicts.d9({"uid_on": judged, "default": d9_receipt(500.0)}, BARS, 100)
    assert result["outcome"] == expected  # the default arm's 500 s hold is recorded, never decides


def d10_receipt(hold=10.0, **over):
    return {"longest_hold_in_pass_s": hold, "pass_completed": True, "rows_examined": 100, "exceptions": [],
            "started_during_hold": {"turn": 3, "compaction": 3},
            "compaction_calls": [{"during_hold": True, "return_s": 0.2}], "turn_calls": [{"wait_s": 2.0}], **over}


def test_d10_bar1_only_in_the_opt_in_arm():
    assert verdicts.d10({"uid_on": d10_receipt(), "default": d10_receipt(900.0)}, BARS, 100)["outcome"] == "PASS"
    assert verdicts.d10({"uid_on": d10_receipt(301.0), "default": d10_receipt()}, BARS, 100)["outcome"] == "FAIL"


@pytest.mark.parametrize("over,expected", [
    ({"exceptions": ["OperationalError"]}, "FAIL"),
    ({"compaction_calls": [{"during_hold": True, "return_s": 1.0}]}, "FAIL"),
    ({"turn_calls": [{"wait_s": 30.5}]}, "FAIL"),
    ({"pass_completed": False}, "INCONCLUSIVE"),
    ({"started_during_hold": {"turn": 0, "compaction": 2}}, "INCONCLUSIVE"),
    ({"rows_examined": 10}, "INCONCLUSIVE")])
def test_d10_bars_2_to_4_in_both_arms(over, expected):
    for arm in ("uid_on", "default"):
        receipts = {"uid_on": d10_receipt(), "default": d10_receipt()}
        receipts[arm] = d10_receipt(**over)
        assert verdicts.d10(receipts, BARS, 100)["outcome"] == expected


def d11_passes(published=(True,) * 8, armed=None, lost=0):
    rows, passes = 1000, []
    for number, flag in enumerate(published):
        out = rows - 100 if flag else rows
        passes.append({"rows_in": rows, "rows_out": out, "published": flag, "hold_armed": number == armed,
                       "stored_check": {"lost": lost if number == 2 else 0, "duplicated": 0, "reordered": False},
                       "returned_equals_input": not flag})
        rows = out
    return {"rows_before": 1000, "passes": passes}


def test_d11_judged_arm_decides():
    good = {b: d11_passes() for b in ("default", "60s", "120s")}
    assert verdicts.d11({"uid_on": good, "default": {}})["outcome"] == "PASS"
    assert verdicts.d11({"uid_on": {**good, "60s": d11_passes(armed=3)}})["outcome"] == "FAIL"
    assert verdicts.d11({"uid_on": {**good, "60s": d11_passes(lost=1)}})["outcome"] == "FAIL"
    assert verdicts.d11({"uid_on": {**good, "60s": d11_passes((False,) * 8)}})["outcome"] == "INCONCLUSIVE"
    assert verdicts.d11({"uid_on": {"default": good["default"]}})["outcome"] == "UNRUN"
    assert verdicts.d11({"uid_on": good, "default": {"60s": d11_passes(armed=1)}})["outcome"] == "PASS"


def test_d12_needs_one_unchanged_no_publish_pass_per_arm_and_budget():
    mixed = d11_passes((True,) * 7 + (False,))
    cells = {arm: {b: copy.deepcopy(mixed) for b in ("default", "60s", "120s")} for arm in ("uid_on", "default")}
    assert verdicts.d12(cells)["outcome"] == "PASS"
    cells["default"]["60s"]["passes"][-1]["returned_equals_input"] = False
    assert verdicts.d12(cells)["outcome"] == "FAIL"  # judged in both arms
    cells["default"]["60s"] = d11_passes()
    assert verdicts.d12(cells)["outcome"] == "INCONCLUSIVE"
    cells["default"]["60s"]["extra_passes"] = [{"published": False, "returned_equals_input": True}]
    assert verdicts.d12(cells)["outcome"] == "PASS"
