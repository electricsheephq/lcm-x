import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUE_TEMPLATE_ROOT = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


def issue_form_body_shape(path: Path) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    marker = "\nbody:\n"
    if marker not in text:
        return 0, []

    body = text.split(marker, maxsplit=1)[1]
    element_count = len(re.findall(r"(?m)^  -\s+", body))
    ids = [
        value.strip().strip("'\"")
        for value in re.findall(r"(?m)^    id:\s*(\S.*?)\s*$", body)
    ]
    return element_count, ids


def test_contributor_guide_uses_the_lcm_x_governance_path():
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert guide.startswith("# Contributing to LCM-X")
    assert "https://github.com/electricsheephq/lcm-x.git" in guide
    assert "accepted issue" in guide.lower()
    assert ".agents/skills/review-pr/SKILL.md" in guide
    assert ".agents/skills/land-pr/SKILL.md" in guide
    assert ".agents/skills/triage-backlog/SKILL.md" in guide
    assert "curate user-facing release notes" in guide


def test_issue_forms_have_unique_ids_and_stay_within_the_platform_limit():
    issue_forms = list(ISSUE_TEMPLATE_ROOT.glob("*.yml"))

    assert issue_forms
    for path in issue_forms:
        element_count, ids = issue_form_body_shape(path)
        if element_count == 0:
            continue
        assert len(ids) == len(set(ids)), path
        assert element_count <= 10, path


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
    assert "Approval and source pushes are outside this" in skill
    assert "Use `land-pr` only for a separately requested" in skill
    assert "## Minimum Capability" in skill
    assert "repository, current refs, issues, pull requests" in skill
    assert "corresponding GitHub" in skill
    assert "current non-author human code-owner" in skill

    authorized_write_steps = [
        "If a maintainer explicitly authorizes one of the remaining GitHub metadata mutations",
        "name the exact item and requested change",
        "re-fetch current item, repository, and authorization state",
        "apply the routine-metadata or sensitive/terminal gate above",
        "stop on any other drift or ambiguity",
        "perform only the named write",
        "read back the result",
    ]
    step_positions = [write_boundary.index(step) for step in authorized_write_steps]
    assert step_positions == sorted(step_positions)


def test_triage_invocation_never_grants_mutation_authority():
    skill = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "Invoking this skill never authorizes a mutation." in skill
    assert "Approval and source pushes are outside this" in skill
    assert "neither invocation nor handoff creates" in skill
    assert "approval, source-push, or merge authority" in skill


def test_triage_preserves_active_upstream_continuations():
    skill = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "source lifecycle: `active-continuation`" in skill
    assert "continuation target:" in skill
    assert "Do not silently turn it into an archive-only record" in skill
    assert "return `OWNER_GATE` and preserve the item unchanged" in skill


def test_triage_separates_routine_metadata_from_sensitive_terminal_actions():
    skill = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "Routine reversible metadata means labels, assignees, and milestones." in skill
    assert "does not require a second code-owner approval solely because" in skill
    assert "Public security or data-integrity disclosure and every close or reopen" in skill
    assert "private vulnerability handling outside this public triage workflow." in skill


def test_repository_policy_states_the_automation_boundary():
    policy = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    normalized_policy = " ".join(policy.split())

    assert "## Automation Boundary" in policy
    assert "Model output alone cannot close, label, assign, push, approve, or merge." in policy
    assert "Automated repair is opt-in" in policy
    assert (
        "Security and data-integrity code changes retain non-author human code-owner approval "
        "on the normal path." in normalized_policy
    )
    assert "Public disclosure retains the stronger `triage-backlog` owner gate." in policy
    assert "it never authorizes disclosure" in policy
    assert (
        "Classification alone does not elevate routine reversible issue metadata"
        in normalized_policy
    )


def test_readiness_and_landing_have_distinct_authority_contracts():
    review_skill = (
        REPO_ROOT / ".agents" / "skills" / "review-pr" / "SKILL.md"
    ).read_text(encoding="utf-8")
    land_skill = (
        REPO_ROOT / ".agents" / "skills" / "land-pr" / "SKILL.md"
    ).read_text(encoding="utf-8")
    review_prompt = (
        REPO_ROOT / ".agents" / "skills" / "review-pr" / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")
    land_prompt = (
        REPO_ROOT / ".agents" / "skills" / "land-pr" / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")

    assert "Remain read-only" in review_skill
    assert "never approve, resolve, comment, label, assign, push, close, or merge" in review_skill
    for decision in (
        "READY_FOR_AUTHORIZED_LANDING",
        "NOT_READY",
        "NOT_DIRECTLY_LANDABLE",
        "OWNER_GATE",
        "STATE_DRIFT",
    ):
        assert decision in review_skill
    assert "without writing to GitHub" in review_prompt

    assert "sole trigger is explicit current authority to merge PR N at exact head H" in land_skill
    assert "The only permitted GitHub mutation" in " ".join(land_skill.split())
    assert "--merge --match-head-commit <HEAD_SHA>" in land_skill
    assert "approval,\nsource push, comment, label, assignment" in land_skill
    assert "merge LCM-X PR N at exact head H" in land_prompt
    assert "ready to merge" not in land_prompt


def test_pr_only_admin_exception_is_exact_head_and_high_confidence():
    policy = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    land_skill = (
        REPO_ROOT / ".agents" / "skills" / "land-pr" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "user-specific PR-only bypass" in policy
    assert "acceptance and adversarial `PASS` receipts scoring at least 95" in policy
    assert "direct push, broad/role/always bypass" in policy
    assert "bypass_mode: pull_request" in land_skill
    assert "distinct reviewer and receipt IDs" in land_skill
    assert "score at least 95" in land_skill
    assert "explicitly report zero" in land_skill
    assert "This exception replaces only the missing non-author approval" in land_skill
    assert "tied or latest `CHANGES_REQUESTED`" in land_skill


def test_review_pr_binds_live_issue_scope_and_git_object_ids():
    review_skill = (
        REPO_ROOT / ".agents" / "skills" / "review-pr" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "40-character lowercase" in review_skill
    assert "explicitly linked from the PR" in review_skill
    assert "accept the actual behavior and file scope" in review_skill
    assert "Reject an unrelated accepted issue" in review_skill
    assert "cannot prove the provenance of caller-supplied facts" in review_skill
    assert "live ruleset's required pairs" in review_skill


def test_repository_routes_readiness_and_explicit_merge_separately():
    policy = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "A readiness decision is advisory and never creates merge authority." in policy
    assert "read-only `.agents/skills/review-pr/SKILL.md`" in policy
    assert "explicit current authority names the PR number" in policy
    assert "Readiness is advisory" in guide
    assert "explicit instruction to merge PR N at exact head H" in guide


def test_triage_prompt_and_contributor_automation_scope_are_bounded():
    prompt = (
        REPO_ROOT / ".agents" / "skills" / "triage-backlog" / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "issue, pull request, or duplicate cluster" in prompt
    assert "without writing to GitHub" in prompt
    assert "limited to the exact accepted issue and current gate" in guide
