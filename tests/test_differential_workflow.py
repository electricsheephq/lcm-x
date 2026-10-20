"""Exercise the required check's actual shell gate without running the differential."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/differential.yml").read_text(encoding="utf-8")


def _relevance_parts() -> tuple[str, str]:
    step = WORKFLOW.split("      - name: Determine differential relevance\n", 1)[1]
    step = step.split("      - uses:", 1)[0]
    environment, script = step.split("        run: |\n", 1)
    paths = environment.split("          TURN_PATH_FILES: |-\n", 1)[1]
    return dedent(paths).strip(), dedent(script)


def test_workflow_turn_paths_match_maintainer_gate() -> None:
    paths, _ = _relevance_parts()
    tree = ast.parse((ROOT / "scripts/maintainer_gate.py").read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TURN_PATH_FILES"
                for target in node.targets)
    ]
    assert len(assignments) == 1
    assert len(paths.splitlines()) == len(set(paths.splitlines()))  # no duplicate entries
    assert set(paths.splitlines()) == ast.literal_eval(assignments[0].value)


def test_required_check_always_reports_and_propagates_failures() -> None:
    assert "  pull_request:\npermissions:" in WORKFLOW
    assert "continue-on-error:" not in WORKFLOW
    assert "  differential:\n" in WORKFLOW
    assert "    timeout-minutes: 30\n" in WORKFLOW
    assert "fetch-depth: 0" in WORKFLOW
    later = WORKFLOW.split("      - uses: actions/setup-python@", 1)[1]
    for step in re.split(r"^      - ", later, flags=re.MULTILINE):
        assert "steps.relevance.outputs.relevant == 'true'" in step
    assert "if: always() && steps.relevance.outputs.relevant == 'true'" in later
    assert 'exit "$status"' in later


@pytest.mark.parametrize(("path", "event", "expected"), [
    ("engine.py", "pull_request", "true"),
    ("docs/guide.md", "pull_request", "false"),
    ("", "pull_request", "false"),
    ("tests/differential/x", "pull_request", "true"),
    (".github/workflows/differential.yml", "pull_request", "true"),
    ("docs/guide.md", "schedule", "true"),
    ("docs/guide.md", "workflow_dispatch", "true"),
])
def test_relevance_shell(tmp_path: Path, path: str, event: str, expected: str) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "--allow-empty", "-qm", "base")
    base = git("rev-parse", "HEAD")
    if path:
        changed = tmp_path / path
        changed.parent.mkdir(parents=True, exist_ok=True)
        changed.write_text("fixture\n", encoding="utf-8")
        git("add", "--", path)
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "change")
    paths, script = _relevance_parts()
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path, check=True, capture_output=True, text=True,
        env={**os.environ, "EVENT_NAME": event, "PR_BASE_SHA": base,
             "TURN_PATH_FILES": paths, "GITHUB_OUTPUT": str(output)},
    )
    assert output.read_text(encoding="utf-8") == f"relevant={expected}\n"
    assert result.stdout == (
        "No differential-relevant changes; skipping.\n" if expected == "false" else ""
    )
