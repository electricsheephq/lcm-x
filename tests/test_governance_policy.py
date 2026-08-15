from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUE_TEMPLATE_ROOT = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


def parse_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_contributor_guide_uses_the_lcm_x_governance_path():
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert guide.startswith("# Contributing to LCM-X")
    assert "https://github.com/electricsheephq/lcm-x.git" in guide
    assert "accepted issue" in guide.lower()
    assert ".agents/skills/land-pr/SKILL.md" in guide
    assert ".agents/skills/triage-backlog/SKILL.md" in guide
    assert "curate user-facing release notes" in guide


def test_issue_forms_have_unique_ids_and_stay_within_the_platform_limit():
    issue_forms = list(ISSUE_TEMPLATE_ROOT.glob("*.yml"))

    assert issue_forms
    for path in issue_forms:
        form = parse_yaml(path)
        body = form.get("body")
        if body is None:
            continue
        ids = [element["id"] for element in body if "id" in element]
        assert len(ids) == len(set(ids)), path
        assert len(body) <= 10, path


def test_bug_form_captures_impact_regression_and_lossless_symptoms():
    bug_form = (ISSUE_TEMPLATE_ROOT / "bug_report.yml").read_text(encoding="utf-8")

    assert "id: impact_regression" in bug_form
    assert "Last known good commit (or N/A):" in bug_form
    assert "First known bad commit (or N/A):" in bug_form
    assert "Current-main reproduction or named mandatory invariant:" in bug_form
    assert "(or N/A)" in bug_form
    assert "missing, duplicated, reordered, misattributed" in bug_form
    assert "one independently reproducible problem" in bug_form


def test_pull_request_template_separates_behavior_and_release_evidence():
    template = (
        REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
    ).read_text(encoding="utf-8")

    assert "## Root cause and best-fix assessment" in template
    assert "best fix / acceptable mitigation / wrong layer" in template
    assert "## Real-behavior proof" in template
    assert "Artifact or redacted trace:" in template
    assert "Proof boundary—what this does not prove:" in template
    assert "## Release-note impact" in template
    assert "accepted issue, PR head, checks, reviews, threads, and authorization" in template
    assert "immediately before any authorized GitHub write" in template


def test_triage_skill_is_bounded_and_read_only_by_default():
    skill = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "SKILL.md"
    ).read_text(encoding="utf-8")
    write_boundary = skill.split("## Write Boundary", maxsplit=1)[1]

    assert "TODO" not in skill
    assert "exact current `main` SHA" in skill
    assert "`P0`" in skill and "`P4`" in skill
    assert "`needs-repro`" in skill
    assert "Remain read-only by default" in skill
    assert "Never process the whole backlog" in skill
    assert "Deterministic live-state checks must guard every authorized" in skill
    assert "## Minimum Capability" in skill
    assert "repository, current refs, issues, pull requests" in skill
    assert "corresponding GitHub" in skill
    assert "current non-author human code-owner" in skill

    authorized_write_steps = [
        "If a maintainer explicitly authorizes one of those mutations",
        "name the exact item and requested change",
        "re-fetch current item, repository, and authorization state",
        "require and record current non-author human code-owner",
        "stop on any other drift or ambiguity",
        "perform only the named write",
        "read back the result",
    ]
    step_positions = [write_boundary.index(step) for step in authorized_write_steps]
    assert step_positions == sorted(step_positions)


def test_repository_policy_states_the_automation_boundary():
    policy = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "## Automation Boundary" in policy
    assert "Model output alone cannot close, label, assign, push, approve, or merge." in policy
    assert "Automated repair is opt-in" in policy
    assert "Security and data-integrity work retains non-author human code-owner approval." in policy


def test_triage_prompt_and_contributor_automation_scope_are_bounded():
    prompt = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "issue, pull request, or duplicate cluster" in prompt
    assert "without writing to GitHub" in prompt
    assert "limited to the exact accepted issue and current gate" in guide
