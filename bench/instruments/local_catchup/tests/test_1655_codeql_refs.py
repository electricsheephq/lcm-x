"""The floor workflow executes only dispatch-owned and reviewed pinned code."""

from pathlib import Path
import re

import yaml


WORKFLOW = Path(__file__).resolve().parents[4] / ".github/workflows/lcmx-1586-floor.yml"
HOST_PIN = "71d34bad1045610252eda02ee79549436896e784"


def test_checkout_inputs_are_removed():
    text = WORKFLOW.read_text()
    assert "inputs.lcmx_ref" not in text
    assert "inputs.host_ref" not in text
    # BaseLoader preserves the YAML `on` key as text.
    inputs = yaml.load(text, Loader=yaml.BaseLoader)["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"cell", "shape", "baseline", "rows", "max_minutes"}


def test_checkout_refs_are_absent_or_literal_commit_shas():
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["floor"]
    checkouts = [s["with"] for s in job["steps"]
                 if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 3
    for checkout in checkouts:
        assert "ref" not in checkout or re.fullmatch(r"[0-9a-fA-F]{40}", checkout["ref"])
    product = next(s for s in checkouts if s.get("path") == "product")
    host = next(s for s in checkouts if s.get("path") == "host")
    assert "ref" not in product
    assert host["ref"] == HOST_PIN
    assert job["env"]["PRODUCT_REF"] == "${{ github.sha }}"


def test_checked_out_heads_are_verified_before_execution():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["floor"]["steps"]
    verification = next(i for i, s in enumerate(steps)
                        if s.get("name") == "Verify checked-out commit SHAs")
    assert verification > max(i for i, s in enumerate(steps)
                              if s.get("uses", "").startswith("actions/checkout@"))
    assert not any("run" in s for s in steps[:verification])
    script = steps[verification]["run"]
    for checkout, expected in (("product", "$GITHUB_SHA"), ("host", HOST_PIN)):
        assert f'if [[ "$(git -C {checkout} rev-parse HEAD)" != "{expected}" ]]; then' in script
    assert script.count("::error::") == 2
    assert script.count("exit 1") == 2
