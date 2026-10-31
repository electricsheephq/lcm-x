"""Dispatch numbers reach reusable workflows as strings; number inputs must be converted."""
import re

import yaml

from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

WORKFLOWS = REPO / ".github" / "workflows"


def test_number_inputs_to_reusable_workflows_are_converted():
    caller = yaml.safe_load((WORKFLOWS / "lcmx-1586-floor.yml").read_text())
    dispatch = caller[True]["workflow_dispatch"]["inputs"]
    checked = 0
    for name, job in caller["jobs"].items():
        uses = job.get("uses", "")
        if not uses.startswith("./.github/workflows/"):
            continue
        called = yaml.safe_load((REPO / uses).read_text())[True]["workflow_call"]["inputs"]
        for key, value in job.get("with", {}).items():
            source = re.fullmatch(r"\$\{\{\s*(?:fromJSON\()?inputs\.(\w+)\)?\s*\}\}", str(value))
            if called[key]["type"] == "number" and source and dispatch[source[1]]["type"] == "number":
                assert value == f"${{{{ fromJSON(inputs.{source[1]}) }}}}", (name, key, value)
                checked += 1
    assert checked == 4
