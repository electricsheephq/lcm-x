"""#1588 exact-cell gate promotions; no host process or message payloads needed."""
import pytest

from bench.instruments.reliability import cells, ci


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
