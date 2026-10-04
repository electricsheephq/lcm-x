"""Positive controls as data: the refs/hosts/cells each control runs and the red/green pattern it must show.

``run_matrix.py --control PC-1 --out <dir>`` runs a control and checks its pattern (CONTROL.json); a control
whose pattern does not hold is a harness finding, not a product one.
"""
from __future__ import annotations

P8_CONTROLS = {"p8-control/archived": "FAIL", "p8-control/other-active": "FAIL",
               "p8-control/random-snapshot": "FAIL", "p8-control/none": "PASS"}
# Measured on the four local CI hosts at 2f4b7d54: no pair has scorable B9 (missing flush seams).
P8_MUST_SUPPORT = {None: (), "acp-process": ()}

CONTROLS = {
    # Differential: the host's post-commit-proof persist strip (#494), fixed by #498 (ae1fb16d; 47bd28e7 is its
    # parent). The baseline must stay green at both refs, the trailing-whitespace shape red only before the fix.
    # customer-0.21.2 does not strip the persisted prompt, so it is not part of the pattern.
    "PC-1": {"refs": ["47bd28e7", "ae1fb16d"], "hosts": ["eva-0.21.5", "rs34-0.21.5", "upstream-main"],
             "cells": ["baseline/in-place/acp", "acp-trailing/in-place"],
             "expect": {("47bd28e7", "baseline/in-place/acp"): "PASS", ("47bd28e7", "acp-trailing/in-place"): "FAIL",
                        ("ae1fb16d", "baseline/in-place/acp"): "PASS", ("ae1fb16d", "acp-trailing/in-place"): "PASS"},
             # the re-stored batch (#494) is a surplus: R1.3/R1.4 pc1 47bd28e7 acp-trailing failed B1-B5 on all 3 hosts
             "bars": ["B1", "B2"]},
    "PC-2": {"refs": ["v0.24.2"], "hosts": "all", "cells": ["parallel-tool-group/in-place", "parallel-tool-group/rotation"],
             "expect": {("v0.24.2", "*"): "FAIL"}, "bars": ["B6"]},
    # Pinned known-bad main before the #553 fix (a moving origin/main is not a control).
    "PC-3": {"refs": ["508f893517f52a400c2bfe0b37f914e864ff806c"], "hosts": "all",
             "cells": ["crash-then-lcm-tool-in-merge-turn/in-place"],
             "expect": {("508f893517f52a400c2bfe0b37f914e864ff806c", "*"): "FAIL"}, "bars": ["B3"]},
}


def check(name: str, rows: list[dict], run_hosts: list[str] | None = None) -> list[str]:
    """Every (ref, cell, host) the control expects is present with the expected verdict (and failed bar).
    ``"all"`` means the hosts the run requested (``run_hosts``), never the hosts that happened to produce rows."""
    ctl, problems = CONTROLS[name], []
    hosts = ctl["hosts"] if ctl["hosts"] != "all" else sorted(run_hosts or [])
    if not hosts:
        return [f"{name}: no host requested"]
    if not rows:
        problems.append(f"{name}: no result rows")
    for ref in ctl["refs"]:
        for cell in ctl["cells"]:
            want = ctl["expect"].get((ref, cell)) or ctl["expect"].get((ref, "*"))
            for host in hosts:
                got = [r for r in rows if r.get("plugin_ref") == ref and r["cell"] == cell and r["host"] == host]
                if not got:
                    problems.append(f"{ref} {cell} {host}: not run")
                elif got[0]["verdict"] != want:
                    problems.append(f"{ref} {cell} {host}: {got[0]['verdict']}, expected {want}")
                elif want == "FAIL" and not set(ctl.get("bars", [])) <= set(got[0].get("failed_bars") or {}):
                    problems.append(f"{ref} {cell} {host}: FAIL without {ctl['bars']}")
    return problems
