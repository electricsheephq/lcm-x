"""#1588 exact-cell gate promotions; no host process or message payloads needed."""
import json

import pytest

from bench.instruments.reliability import cells, ci, report


def rows_for(cell, host, transport, failed):
    lane = None if transport == "in-process" else transport
    rows = [{"cell": cid, "host": host, "plugin_sha": "fixture", "verdict": "PASS", "targets": [],
             **({"transport": lane} if lane else {})} for cid in ci.expected_cells(lane)]
    targets = {c["id"]: c["targets"] for c in cells.select("all", extra=cells.continuity_cells())}
    for row in rows:
        if row["cell"].startswith("p8-control/") and row["cell"] != "p8-control/none":
            row.update(verdict="FAIL", failed_bars={"B9": {}})
        if row["cell"] == cell:
            row.update(verdict="FAIL", targets=targets[cell], failed_bars={bar: {} for bar in failed})
    return rows


@pytest.mark.parametrize("cell", list(ci.CELL_GATE))
def test_each_promoted_bar_gates(cell):
    transport, bars = ci.CELL_GATE[cell]
    assert ci.in_gate_set(cell) and cell not in ci.NON_GATE
    for bar in bars:
        problems = ci.gate(rows_for(cell, "upstream-uid", transport, [bar]), {534, 1508, 1501, 1506})
        assert len(problems) == 1 and f"uncovered bars ['{bar}']" in problems[0]


@pytest.mark.parametrize("host", ["ref-0.21.5", "floor-0.21.3", "upstream-main"])
@pytest.mark.parametrize("cell", ["overflow-recovery-restart/in-place", "overflow-recovery-restart/rotation",
                                  "upgrade-crash-after-rotation/rotation"])
def test_stamp_only_failures_need_an_open_target(cell, host):
    rows = rows_for(cell, host, "in-process", ["B1", "B2"])
    assert ci.gate(rows, {534, 1508, 1501, 1506}) == []
    assert len(ci.gate(rows, set())) == 1


@pytest.mark.parametrize("shape", ["sole-user-tool-loop", "survival-fit", "survival-fit-older-turns"])
@pytest.mark.parametrize("mode", ["in-place", "rotation"])
def test_other_continuity_cells_never_gate(shape, mode):
    cell = f"continuity/{shape}/{mode}"
    assert not ci.in_gate_set(cell) and cell in ci.NON_GATE
    assert ci.gate(rows_for(cell, "upstream-uid", "acp-process", ["B1", "B2", "F4", "B9"]), set()) == []


def test_existing_gate_and_open_target_behavior_is_unchanged():
    cell = "baseline/in-place/acp"
    assert ci.in_gate_set(cell)
    rows = rows_for(cell, "ref-0.21.5", "in-process", ["B1"])
    assert len(ci.gate(rows, set())) == 1
    next(r for r in rows if r["cell"] == cell)["targets"] = [7]
    assert ci.gate(rows, {7}) == []


def test_error_reason_keeps_both_edges_without_changing_problem_counts():
    cell = "baseline/in-place/acp"
    rows = rows_for(cell, "ref-0.21.5", "in-process", ["B1"])
    row = next(r for r in rows if r["cell"] == cell)
    row["targets"] = [7]
    assert len(ci.gate(rows, set())) == 1
    assert ci.gate(rows, {7}) == []

    start, end = "START-MARKER", "END-MARKER"
    reason = start + "x" * (1000 - len(start) - len(end)) + end
    row.update(verdict="ERROR", reason=reason)
    problems = ci.gate(rows, {7})
    assert len(problems) == 1
    assert start in problems[0] and end in problems[0] and " … " in problems[0]
    prefix = f"ERROR ref-0.21.5 {cell} (in-process): "
    assert problems[0] == prefix + reason[:200] + " … " + reason[-400:]

    row["reason"] = "s" * 150
    assert ci.gate(rows, {7}) == [prefix + row["reason"]]


@pytest.mark.parametrize("host,transport,failed,open_issues,gates", [
    ("upstream-main", "acp-process", ["B1", "B2"], {2624}, False),
    ("upstream-main", "acp-process", ["B1", "B2", "B5"], {2624}, True),
    ("r34.4-0.21.5", "acp-process", ["B1", "B2"], {2624}, True),
    ("floor-0.21.3", "acp-process", ["B1", "B2"], {2624}, True),
    ("upstream-main", "in-process", ["B1", "B2"], {2624}, True),
    ("upstream-main", "acp-process", ["B1", "B2"], set(), True),
    ("upstream-uid", "acp-process", ["B1", "B2"], {2624}, False),
    ("upstream-uid", "acp-process", ["B1", "B2", "B5"], {2624}, True),
    ("upstream-uid", "in-process", ["B1", "B2"], {2624}, True),
    ("upstream-uid", "acp-process", ["B1", "B2"], set(), True),
])
def test_abandoned_rotation_child_2624_gate_scope(host, transport, failed, open_issues, gates):
    problems = ci.gate(rows_for("crash-after-rotation/rotation", host, transport, failed), open_issues)
    assert len(problems) == int(gates)


HELD_TARGETS = ["reply-held-crash/" + name for name in ("rotation-turn", "plain-turn", "multi-hop")]
HELD_CONTROLS = ["reply-held-crash/pre-rotation", "reply-held-crash/in-place"]
CI_HOST_NAMES = list(json.loads(ci.CI_HOSTS.read_text())["hosts"])


@pytest.mark.parametrize("cell", HELD_TARGETS)
@pytest.mark.parametrize("host", ["upstream-main", "upstream-uid"])
def test_held_reply_known_only_while_2624_open(cell, host):
    rows = rows_for(cell, host, "acp-process", ["B1", "B2"])
    assert ci.gate(rows, {2624}) == []
    assert len(ci.gate(rows, set())) == 1


@pytest.mark.parametrize("cell", HELD_TARGETS)
@pytest.mark.parametrize("host", ["r34.10-0.21.5", "ref-0.21.5", "r34.15-0.21.5",
                                  "floor-0.21.3", "r34.4-0.21.5"])
def test_held_reply_known_entry_excludes_other_hosts(cell, host):
    assert len(ci.gate(rows_for(cell, host, "acp-process", ["B1", "B2"]), {2624})) == 1


@pytest.mark.parametrize("cell", HELD_TARGETS)
@pytest.mark.parametrize("host", ["upstream-main", "upstream-uid"])
@pytest.mark.parametrize("bar", ["B3", "B4", "B5"])
def test_held_reply_uncovered_bar_gates(cell, host, bar):
    problems = ci.gate(rows_for(cell, host, "acp-process", ["B1", "B2", bar]), {2624})
    assert len(problems) == 1 and f"uncovered bars ['{bar}']" in problems[0]


@pytest.mark.parametrize("cell", HELD_CONTROLS)
@pytest.mark.parametrize("host", CI_HOST_NAMES)
def test_held_reply_controls_gate_every_host(cell, host):
    assert len(ci.gate(rows_for(cell, host, "acp-process", ["B1", "B2"]), {2624})) == 1


@pytest.mark.parametrize("cell", HELD_TARGETS + HELD_CONTROLS)
@pytest.mark.parametrize("host", CI_HOST_NAMES)
def test_held_reply_errors_gate_every_host(cell, host):
    rows = rows_for(cell, host, "acp-process", ["B1", "B2"])
    next(row for row in rows if row["cell"] == cell)["verdict"] = "ERROR"
    assert len(ci.gate(rows, {2624})) == 1


def test_2624_targets_are_exact():
    assert sorted(c["id"] for c in cells.registry() if 2624 in c["targets"]) == sorted(
        ["crash-after-rotation/rotation", *HELD_TARGETS])


@pytest.mark.parametrize("host,gates_b9", [("upstream-uid", True), ("ref-0.21.5", False)])
def test_promoted_continuity_bars_are_exact(host, gates_b9):
    cell = "continuity/forced-compaction-then-turn/rotation"
    assert ci.gate(rows_for(cell, host, "acp-process", ["B5"]), set()) == []
    assert bool(ci.gate(rows_for(cell, host, "acp-process", ["B9"]), set())) is gates_b9


def test_native_rotation_and_unsupported_transport_are_not_promoted():
    assert not ci.in_gate_set("native-on-off/v0.23.3/rotation")
    cell = "native-on-off/v0.23.3/in-place"
    assert ci.gate(rows_for(cell, "upstream-uid", "acp-process", ["B3"]), set()) == []


@pytest.mark.parametrize("failed", [[], ["B7"]])
def test_promoted_failure_attribution_and_bar_scope(failed):
    cell = "overflow-recovery-restart/in-place"
    problems = ci.gate(rows_for(cell, "upstream-uid", "in-process", failed), set())
    assert bool(problems) is (not failed)


PERSISTENT = "publication-failure/rotation-child-persistent"


@pytest.mark.parametrize("open_issues", [{1797}, set()])
@pytest.mark.parametrize("numbers", [None, {}, {"B2": {}}, {"B2": {"deficit_rows": 1, "surplus_rows": 28}}])
def test_persistent_b2_deficit_or_missing_receipt_gates(numbers, open_issues):
    rows = rows_for(PERSISTENT, "fixture", "in-process", ["B2"])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    if numbers is not None:
        row["numbers"] = numbers
    problems = ci.gate(rows, open_issues)
    assert len(problems) == 1 and "uncovered bars ['B2D']" in problems[0]


@pytest.mark.parametrize("failed", [["B2"], ["B1", "B2"], ["B1"]])
@pytest.mark.parametrize("open_issues", [{1797}, set()])
def test_persistent_surplus_and_b1_never_gate(failed, open_issues):
    rows = rows_for(PERSISTENT, "fixture", "in-process", failed)
    next(r for r in rows if r["cell"] == PERSISTENT)["numbers"] = {
        "B2": {"deficit_rows": 0, "surplus_rows": 28}}
    assert ci.gate(rows, open_issues) == []


def test_persistent_b1_alone_needs_no_b2_receipt():
    assert ci.gate(rows_for(PERSISTENT, "fixture", "in-process", ["B1"]), set()) == []


def test_persistent_deficit_gate_is_in_process_only():
    assert ci.gate(rows_for(PERSISTENT, "fixture", "acp-process", ["B2"]), set()) == []


def test_existing_b2_declaration_still_excuses_a_deficit():
    cell = "baseline/in-place/acp"
    rows = rows_for(cell, "fixture", "in-process", ["B2"])
    next(r for r in rows if r["cell"] == cell).update(
        targets=[7], numbers={"B2": {"deficit_rows": 1}})
    assert "B2" in cells.ISSUES[7][0]
    assert len(ci.gate(rows, set())) == 1
    assert ci.gate(rows, {7}) == []


@pytest.mark.parametrize("deficit,view", [(0, "PASS"), (1, "FAIL")])
def test_b2_signature_reports_deficit_view_with_surplus(deficit, view):
    signature = report.signature({"verdict": "FAIL", "failed_bars": {"B2": {}},
                                  "numbers": {"B2": {"deficit_rows": deficit, "surplus_rows": 28}}})
    assert f"surplus/deficit=28/{deficit}" in signature
    assert f"deficit-only {view}" in signature
