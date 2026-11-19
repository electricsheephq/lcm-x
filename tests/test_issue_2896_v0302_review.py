"""#2896 (v0.30.2) full-diff review: one cell per merge-blocking finding, each checked against a real copy of main.

Two trees, like the grid: LCMX_GRID_BASE is a copy of the base commit (main), this tree (or LCMX_GRID_CAND) the
candidate; without a base the cells skip. tests/behaviour/notice_fit_sweep.py --sweep review runs the scenarios on
each tree in its own process. R1 to R5 are red on 020e71a7; R6 checks the sweep harness itself (raw rows, only the
fit's own blocks counted, a no-op seen as one). Synthetic data only.
"""
from __future__ import annotations

import json

import pytest

from tests.test_catchup_budget_grid import BASE_TREE, CAND_TREE, NO_BASE
from tests.test_issue_2896_kept_rows_equal_main import _sweep, out_dir  # noqa: F401 (the shared output folder)

pytestmark = pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)


@pytest.fixture(scope="module")
def runs(out_dir):  # noqa: F811
    return _sweep(BASE_TREE, "review", out_dir), _sweep(CAND_TREE, "review", out_dir)


def _keys(runs, prefix):
    base, cand = runs
    keys = sorted(k for k in base if k.startswith(prefix))
    assert keys and set(keys) == {k for k in cand if k.startswith(prefix)}
    return [(k, base[k], cand[k]) for k in keys]


def _without(rows, blocks):
    out = []
    for row in rows:
        parsed = json.loads(row)
        if isinstance(parsed.get("content"), str):
            for block in blocks:
                parsed["content"] = parsed["content"].replace(block, "")
        out.append(json.dumps(parsed, sort_keys=True))
    return out


def test_r1_a_second_fit_never_cuts_where_mains_list_already_fits(runs):
    """R1: the first fit's rows persisted, a second fit one token under its cap: main's unmarked list fits and is
    returned as it is; the candidate measures the old block, strips it and cuts a whole stored turn."""
    for key, base, cand in _keys(runs, "r1/"):
        first_block = cand["first"]["generated"]
        assert cand["second"]["fitted"] == base["second"]["fitted"], key
        assert _without(cand["second"]["rows"], first_block) == base["second"]["rows"], key


def test_r2_a_complete_stored_carrier_keeps_its_store_mapping(runs):
    """R2: the store holds the complete carrier summary + U and the distinct row U; with the carrier ahead of an
    objective, the bound fit must map the returned carrier to the row main maps it to."""
    for key, base, cand in _keys(runs, "r2/"):
        assert base["map_after"] == base["map_before"], key
        assert cand["map_after"] == base["map_after"], (key, cand["map_after"], cand["retained"])


def test_r3_marking_one_carrier_never_unmaps_its_sibling(runs):
    """R3: [carrier A, carrier B, objective] on the exception path: after the bound fit, A maps as on main."""
    for key, base, cand in _keys(runs, "r3/"):
        assert cand["map_after"] == base["map_after"], (key, cand["map_after"], base["map_after"])


def test_r4_a_generated_and_a_pasted_notice_never_ignore_a_fresh_turn(runs):
    """R4: patterns "session_id=S " and "confirm\\Z"; a fresh turn ending in confirm plus a pasted counted notice,
    merged behind the fitted carrier, is stored as on main."""
    for key, base, cand in _keys(runs, "r4/"):
        assert base["fresh_stored"] and not base["ignored"], key
        assert (cand["fresh_stored"], cand["ignored"]) == (True, False), key


def test_r5_with_a_system_row_a_marked_carrier_is_mains_bytes(runs):
    """R5: a leading system row and [marked carrier, objective, ...]: the returned head and cut equal main's."""
    for key, base, cand in _keys(runs, "r5/"):
        assert cand["fit"]["dropped"] == base["fit"]["dropped"], key
        assert cand["head"] == base["head"], key


def test_r6_the_harness_reads_pasted_markers_as_row_bytes_and_sees_a_no_op(runs):
    """R6: a user-pasted marker kept in view is the row's own bytes (no generated block, compared raw), and a fit
    that returns the list it was given is not "fitted"."""
    for key, base, cand in _keys(runs, "r6/"):
        assert base["raw_markers"] >= 1 and not base["fit"]["generated"] and not cand["fit"]["generated"], key
        assert cand["fit"]["rows"] == base["fit"]["rows"], key
        if key == "r6/100000":
            assert not base["fit"]["fitted"] and not cand["fit"]["fitted"]
