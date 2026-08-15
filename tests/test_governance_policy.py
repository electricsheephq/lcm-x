import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUE_TEMPLATE_ROOT = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


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
        body = path.read_text(encoding="utf-8")
        if "\nbody:\n" not in body:
            continue
        ids = re.findall(r"^    id: ([a-zA-Z0-9_-]+)$", body, flags=re.MULTILINE)
        assert len(ids) == len(set(ids)), path
        assert body.count("\n  - type: ") <= 10, path


def test_bug_form_captures_impact_regression_and_lossless_symptoms():
    bug_form = (ISSUE_TEMPLATE_ROOT / "bug_report.yml").read_text(encoding="utf-8")

    assert "id: impact_regression" in bug_form
    assert "Last known good commit:" in bug_form
    assert "First known bad commit:" in bug_form
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
    assert "deterministically re-fetched before any write" in template


def test_triage_skill_is_bounded_and_read_only_by_default():
    skill = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "TODO" not in skill
    assert "exact current `main` SHA" in skill
    assert "`P0`" in skill and "`P4`" in skill
    assert "`needs-repro`" in skill
    assert "Remain read-only by default" in skill
    assert "Never process the whole backlog" in skill
    assert "Deterministic live-state checks must guard every authorized" in skill


def test_repository_policy_states_the_automation_boundary():
    policy = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "## Automation Boundary" in policy
    assert "Model output alone cannot close, label, assign, push, approve, or merge." in policy
    assert "Automated repair is opt-in" in policy
    assert "Security and data-integrity work retains non-author human code-owner approval." in policy
