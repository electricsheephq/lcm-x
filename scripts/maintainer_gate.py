#!/usr/bin/env python3
"""Evaluate LCM-X maintainer policy receipts without network access."""

from __future__ import annotations

import json
import sys
from typing import Any


SCHEMA_VERSION = "1"
REPOSITORY = "electricsheephq/lcm-x"
RULESET_ID = 20888757
REQUIRED_CI_CHECK_PAIRS = (
    ("workflow-lint", 15368),
    ("lint", 15368),
    ("test (3.11)", 15368),
    ("test (3.12)", 15368),
    ("test (3.13)", 15368),
    ("test (3.14)", 15368),
)
# Required on the PR head only. The Differential workflow runs on pull requests (plus nightly and on dispatch),
# not on pushes to main. Strict mode makes the PR's merge-ref run cover the merged tree, so post-merge
# verification checks the CI pairs above on the merge commit.
REQUIRED_PR_ONLY_CHECK_PAIRS = (("differential", 15368),)
REQUIRED_CHECK_PAIRS = REQUIRED_CI_CHECK_PAIRS + REQUIRED_PR_ONLY_CHECK_PAIRS
# Changed-path risk map for the land-pr review obligation. It is a readiness hint
# only: it names the review lanes land-pr requires and never fails readiness.
REVIEW_POLICY_FILES = {
    ".agents/skills/review-pr/SKILL.md", ".github/PULL_REQUEST_TEMPLATE.md",
    "AGENTS.md", "CONTRIBUTING.md", "docs/review-evidence-provenance.md",
    "scripts/maintainer_gate.py",
}
MEMORY_MIGRATION_FILES = {
    "scripts/backfill_externalized_tool_outputs.py",
    "scripts/import_lossless_claw.py",
}
# Modules that directly shape main-model context, summary requests, stored turn
# content/identity/ownership, compaction selection/commit/progress, or host
# compress/rebind (docs/process/turn-path-probe-packet.md). Reachability alone
# does not qualify: explicit tools/commands and default-off features stay out.
# Default model-facing tool schemas count as context; persistence plumbing alone
# does not qualify, but row-content policy and compaction commit ownership do.
TURN_PATH_FILES = {
    "__init__.py", "assembly_plan.py", "aux_session.py", "bypass.py", "codex_routing.py", "compaction.py",
    "compaction_policy.py", "compaction_request.py", "compress_run.py",
    "dag.py", "engine.py", "escalation.py", "externalize.py", "extraction.py", "fresh_tail.py",
    "host_uid.py", "host_uid_emit.py", "identity_anchor.py", "ingest_protection.py",
    "leaf_prep.py", "lifecycle_state.py", "message_analysis.py", "message_content.py", "placeholder_ledger.py",
    "prefix_matching.py", "reconcile.py", "sanitize.py", "store_complete.py",
    "summary_attempt.py", "summary_input_clip.py", "summary_outcome.py", "survival_codec.py",
    "survival_fit.py", "tokens.py", "tool_pairs.py",
    "config.py",  # Defaults and loaders set fresh-tail cuts, filters and summary requests.
    "db_bootstrap.py",  # Conversation-range selection shapes compaction inputs and ownership.
    "diagnostics.py",  # State-db path resolution decides foreground branch restoration/rebind.
    "engine_registry.py",  # Post-turn hooks select the clone that ingests each session turn.
    "message_patterns.py",  # Pattern matches omit rows from ingest, replay and compaction.
    "model_routing.py",  # Sets provider, model and effort on the summary request.
    "plugin_identity.py",  # Engine-name matching gates post-turn ingest and clone resolution.
    "rollup_builder.py",  # Default-on maintenance builds and sends rollup summary requests.
    "rollup_periods.py",  # Default-on rollups select period boundaries and summary source rows.
    "rollup_store.py",  # Default-on rollups select sources and publish derived summaries.
    "schemas.py",  # Default model-facing tool descriptions are part of the model context.
    "session_patterns.py",  # Session matches choose LCM management versus host fallback.
    "sqlite_util.py",  # Owns the transaction/retry that commits a leaf and its frontier.
    "store.py",  # Ingest protection and row normalization decide what a turn stores.
    "tool_registry.py",  # Selects model-facing schemas exposed on the default turn.
    "reset_state.py",  # resets the ingest cursor and compaction progress on rotation
    "emission_projection.py", "recognition_matching.py",  # Split out of engine paths by #1294.
    # Split out of ingest_protection.py: what one ingested turn stores and how it is checked.
    "ingest_content.py", "ingest_integrity.py", "ingest_pem_model.py",
    "ingest_pem_scan.py", "ingest_persisted.py", "ingest_policy.py",
}
# Explicit exclusions: classify every top-level Python module, not just reachable dependencies.
NON_TURN_PATH_FILES = {
    "adaptive_retrieval.py",  # Default-off adaptive retrieval controller.
    "answer_contract.py",  # Default-off pre-answer requirements.
    "assertion_extraction.py",  # Default-off structured extraction.
    "assertion_rebuild.py",  # Explicit opt-in assertion rebuild.
    "assertion_state.py",  # Default-off assertion query state.
    "assertion_store.py",  # Default-off assertion persistence.
    "chunking.py",  # Explicit chunk backfill or default-off embeddings.
    "command.py",  # Explicit commands or default-off embedding backfill.
    "command_auth.py",  # Explicit slash-command authorization only.
    "command_router.py",  # Explicit slash-command grammar only.
    "command_status.py",  # Operator status rendering only.
    "embedding_maintenance.py",  # Default-off embedding maintenance.
    "embedding_provider.py",  # Default-off embedding provider path.
    "evidence_compiler.py",  # Default-off hook or explicit evidence tool.
    "evidence_pack.py",  # Explicit evidence tool or default-off compiler.
    "evidence_tools.py",  # Explicit evidence tool handlers only.
    "expand_query_tools.py",  # Explicit lcm_expand_query tool handler only.
    "expansion_evidence.py",  # Explicit expansion context/provenance only.
    "guidance.py",  # Exported policy loader unused by default prompt.
    "host_evidence.py",  # Unwired opt-in host evidence helper.
    "inactive_record.py",  # Inactive-plugin diagnostic receipts only.
    "inspect_tools.py",  # Explicit lcm_inspect tool handler; read-only metadata.
    "level3_repair.py",  # Explicit repair or read-only doctor scan.
    "maintenance.py",  # Explicit backup, repair or rotation only.
    "occurrence_time.py",  # Retrieval metadata; never durable turn occurrence.
    "preanswer_evidence.py",  # Unwired opt-in pre-answer helper.
    "presets.py",  # Status/dry-run suggestions; no live config mutation.
    "query_view_store.py",  # Default-off query views; optional-schema verification.
    "reasoning.py",  # Explicit computation or default-off pre-answer hook.
    "recall_delivery.py",  # Explicit recall; proactive assembly defaults off.
    "recent.py",  # Explicit recent-history tool response only.
    "requirements_compiler.py",  # Default-off pre-answer compilation.
    "retrieval_context.py",  # Retrieval-reader connection bindings only.
    "retrieval_core.py",  # Explicit retrieval; proactive assembly defaults off.
    "runtime_identity.py",  # Status/doctor provenance fields only.
    "schema_families.py",  # Verifier finding types/compatibility metadata only.
    "scope_storage.py",  # Explicit Teams scope setup/backfill; not wired to default turn.
    "search_query.py",  # Retrieval query parsing/ranking; no ingest caller.
    "selective_compiler.py",  # Two default-off pre-answer compiler gates.
    "selective_recall.py",  # Default-off pre-answer session bundle.
    "sqlite_session.py",  # Connection/transaction plumbing; caller owns rows and compaction policy.
    "tools.py",  # Explicit tool handlers; proactive assembly defaults off.
    "trajectory_store.py",  # Independent trajectory corpus; optional-schema verification only.
    "vector_store.py",  # Default-off vector persistence/retrieval.
}
FINDING_GATE_CLASSES = {"MERGE_BLOCKING", "RELEASE_BLOCKING", "NON_BLOCKING"}
TERMINAL_FINDING_DISPOSITIONS = {
    "FIXED_NOW",
    "FALSE_OR_NOT_APPLICABLE",
    "ACCEPTED_TRADEOFF_OR_WONT_FIX",
    "ACCEPTED_FOLLOW_UP",
    "ESCALATED_OWNER_GATE",
}
DECISIONS = {
    "READY_FOR_AUTHORIZED_LANDING",
    "NOT_READY",
    "NOT_DIRECTLY_LANDABLE",
    "OWNER_GATE",
    "STATE_DRIFT",
    "POST_MERGE_VERIFIED",
}


def _is_object_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _pair(value: dict[str, Any]) -> tuple[str, int]:
    return str(value.get("context", "")), int(value.get("integration_id", 0))


def _trusted_checks(
    checks: list[dict[str, Any]],
    target_sha: str,
    required_pairs: tuple[tuple[str, int], ...] = REQUIRED_CHECK_PAIRS,
) -> tuple[list[dict[str, Any]], list[str]]:
    matched: list[dict[str, Any]] = []
    blockers: list[str] = []

    for context, integration_id in required_pairs:
        exact = [
            check
            for check in checks
            if _pair(check) == (context, integration_id)
            and check.get("head_sha") == target_sha
        ]
        passing = [
            check
            for check in exact
            if check.get("status") == "completed"
            and check.get("conclusion") == "success"
        ]
        if len(exact) > 1:
            blockers.append(f"TRUSTED_CHECK_DUPLICATE:{context}:{integration_id}")
        elif len(passing) != 1:
            blockers.append(f"TRUSTED_CHECK_UNSATISFIED:{context}:{integration_id}")
        else:
            matched.append({"context": context, "integration_id": integration_id})
    return matched, blockers


def _base_blockers(data: dict[str, Any]) -> tuple[list[str], str, list[dict[str, Any]]]:
    blockers: list[str] = []
    policy = data.get("protected_policy", {})
    pr = data.get("pr", {})
    head_sha = str(pr.get("head_sha", ""))

    if data.get("schema_version") != SCHEMA_VERSION:
        blockers.append("SCHEMA_VERSION_UNSUPPORTED")
    if data.get("repository") != REPOSITORY:
        blockers.append("REPOSITORY_MISMATCH")
    if (
        policy.get("source") != "protected-main"
        or policy.get("base_ref") != "main"
        or not policy.get("base_sha")
        or policy.get("ruleset_id") != RULESET_ID
        or policy.get("strict_required_status_checks_policy") is not True
        or [_pair(item) for item in policy.get("required_checks", [])]
        != list(REQUIRED_CHECK_PAIRS)
    ):
        blockers.append("PROTECTED_POLICY_UNTRUSTED")
    if type(pr.get("number")) is not int or pr["number"] <= 0:
        blockers.append("PR_IDENTITY_INVALID")
    if not head_sha or pr.get("state") != "OPEN" or pr.get("draft") is not False:
        blockers.append("PR_STATE_DRIFT")
    if pr.get("base_ref") != "main":
        blockers.append("NON_MAIN_BASE")
    if pr.get("base_sha") != policy.get("base_sha"):
        blockers.append("BASE_POLICY_SHA_MISMATCH")

    matched, check_blockers = _trusted_checks(data.get("checks", []), head_sha)
    blockers.extend(check_blockers)

    if any(thread.get("is_resolved") is not True for thread in data.get("threads", [])):
        blockers.append("UNRESOLVED_REVIEW_THREAD")
    for finding in data.get("findings", []):
        if finding.get("verified") is not True:
            continue
        gate_class = finding.get("gate_class")
        disposition = finding.get("disposition")
        if gate_class not in FINDING_GATE_CLASSES:
            blockers.append("VERIFIED_FINDING_GATE_CLASS_INVALID")
        if disposition not in TERMINAL_FINDING_DISPOSITIONS:
            blockers.append("VERIFIED_FINDING_UNDISPOSITIONED")
        elif (
            gate_class == "MERGE_BLOCKING"
            and disposition not in {"FIXED_NOW", "FALSE_OR_NOT_APPLICABLE"}
        ):
            blockers.append("ACTIVE_MERGE_BLOCKING_FINDING")
    issue = data.get("accepted_issue", {})
    if issue.get("accepted") is not True or not issue.get("number"):
        blockers.append("ACCEPTED_ISSUE_MISSING")
    return blockers, head_sha, matched


def _changed_paths(changed_files: Any) -> list[str] | None:
    if not isinstance(changed_files, list):
        return None
    paths: list[str] = []
    for item in changed_files:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict) and isinstance(item.get("filename"), str):
            paths.append(item["filename"])
            # A rename counts both its old and its new path.
            if isinstance(item.get("previous_filename"), str):
                paths.append(item["previous_filename"])
        else:
            return None
    # A real PR changes at least one file; an empty list or a blank path means
    # the retrieval failed, so the risk is unknown.
    if not paths or any(not path.strip() for path in paths):
        return None
    return paths


def _named_risks(paths: list[str]) -> list[str]:
    risks: list[str] = []
    if any(
        path in REVIEW_POLICY_FILES or path.startswith(".agents/skills/land-pr/")
        for path in paths
    ):
        risks.append("review-provenance-policy")
    if any(
        ("/" not in path and path.endswith(".py"))
        or path in MEMORY_MIGRATION_FILES
        or path.startswith(("access_context/", "access_policy/", "teams/"))
        for path in paths
    ):
        risks.append("lcm-memory-preservation")
    if any(path in TURN_PATH_FILES for path in paths):
        risks.append("turn-path")
    return risks


def review_lanes_hint(changed_files: Any) -> dict[str, Any]:
    """Name the review lanes land-pr requires; advisory, never a blocker."""
    paths = _changed_paths(changed_files)
    if paths is None:
        return {
            "changed_files": "unknown",
            "named_risks": None,
            "required_review_lanes": ["acceptance", "adversarial"],
        }
    risks = _named_risks(paths)
    hint = {
        "changed_files": "supplied",
        "named_risks": risks,
        "required_review_lanes": (
            ["acceptance", "adversarial"] if risks else ["acceptance"]
        ),
    }
    if "turn-path" in risks:
        hint["turn_path_probe_packet"] = [
            "Consumer enumeration: rg call sites for every changed helper and effects on each consumer.",
            "Previous-GA differential row: tests/differential (opt-in).",
            "Sibling shapes: neighbouring windows, row roles, and recovery paths.",
        ]
    return hint


def _landing_authorization_gate(data: dict[str, Any], head_sha: str) -> list[str]:
    auth = data.get("merge_authorization", {})
    pr = data.get("pr", {})
    expected = {
        "repository": data.get("repository"),
        "pr_number": pr.get("number"),
        "head_sha": head_sha,
        "action": "merge",
    }
    blockers: list[str] = []
    if not auth:
        return ["EXACT_MERGE_AUTHORIZATION_MISSING"]
    for key, value in expected.items():
        if auth.get(key) != value:
            blockers.append(f"MERGE_AUTHORIZATION_{key.upper()}_MISMATCH")
    if not auth.get("actor"):
        blockers.append("MERGE_AUTHORIZATION_ACTOR_MISSING")
    return blockers


def _decision_for(blockers: list[str], mode: str) -> str:
    if "NON_MAIN_BASE" in blockers:
        return "NOT_DIRECTLY_LANDABLE"
    if any(
        blocker in {
            "SCHEMA_VERSION_UNSUPPORTED",
            "REPOSITORY_MISMATCH",
            "PR_STATE_DRIFT",
            "PR_IDENTITY_INVALID",
            "OBJECT_ID_INVALID",
            "BASE_POLICY_SHA_MISMATCH",
            "PR_NOT_MERGED",
            "MERGE_COMMIT_PR_MISMATCH",
            "PR_HEAD_NOT_IN_MERGE_PARENTS",
            "MERGE_COMMIT_NOT_ANCESTOR_OF_LIVE_MAIN",
            "POST_MERGE_BASE_MISMATCH",
            "POST_MERGE_PARENTS_INVALID",
            "POST_MERGE_ANCESTRY_INVALID",
        }
        or blocker.startswith("MERGE_AUTHORIZATION_")
        for blocker in blockers
    ):
        return "STATE_DRIFT"
    if any(
        blocker in {
            "PROTECTED_POLICY_UNTRUSTED",
            "EXACT_MERGE_AUTHORIZATION_MISSING",
            "ACCEPTED_ISSUE_MISSING",
        }
        for blocker in blockers
    ):
        return "OWNER_GATE"
    if blockers:
        return "NOT_READY"
    if mode in {"readiness", "landing"}:
        return "READY_FOR_AUTHORIZED_LANDING"
    return "POST_MERGE_VERIFIED"


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return _evaluate(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return _invalid_receipt(exc)


def _evaluate(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("input must be a JSON object")
    mode = data.get("mode")
    if mode not in {"readiness", "landing", "post_merge"}:
        return _receipt(data, "OWNER_GATE", ["MODE_UNSUPPORTED"], [], "unknown")

    if mode == "post_merge":
        return _evaluate_post_merge(data)

    blockers, head_sha, matched = _base_blockers(data)
    authorization = data.get("merge_authorization", {})
    if mode == "landing":
        blockers.extend(_landing_authorization_gate(data, head_sha))
        if authorization.get("use_admin_bypass") not in (None, False):
            blockers.append("ADMIN_BYPASS_FORBIDDEN")
    elif authorization:
        blockers.append("READINESS_INPUT_CONTAINS_MERGE_AUTHORIZATION")
    if data.get("admin_bypass_qualification"):
        blockers.append("ADMIN_BYPASS_FORBIDDEN")

    decision = _decision_for(blockers, mode)
    receipt = _receipt(data, decision, blockers, matched, "protected-normal")
    receipt["review_lanes_hint"] = review_lanes_hint(data.get("changed_files"))
    return receipt


def _evaluate_post_merge(data: dict[str, Any]) -> dict[str, Any]:
    policy = data.get("protected_policy", {})
    pr = data.get("pr", {})
    facts = data.get("post_merge", {})
    merge_commit_value = facts.get("merge_commit")
    merge_commit = merge_commit_value if isinstance(merge_commit_value, str) else ""
    merge_parents = facts.get("merge_parents")
    live_main_ancestors = facts.get("live_main_ancestors")
    blockers: list[str] = []

    if (
        type(merge_parents) is not list
        or len(merge_parents) != 2
        or merge_parents[0] == merge_parents[1]
    ):
        blockers.append("POST_MERGE_PARENTS_INVALID")
        merge_parents = []
    if type(live_main_ancestors) is not list:
        blockers.append("POST_MERGE_ANCESTRY_INVALID")
        live_main_ancestors = []

    if data.get("schema_version") != SCHEMA_VERSION:
        blockers.append("SCHEMA_VERSION_UNSUPPORTED")
    if data.get("repository") != REPOSITORY:
        blockers.append("REPOSITORY_MISMATCH")
    if (
        policy.get("source") != "protected-main"
        or policy.get("base_ref") != "main"
        or not policy.get("base_sha")
        or policy.get("ruleset_id") != RULESET_ID
        or policy.get("strict_required_status_checks_policy") is not True
        or [_pair(item) for item in policy.get("required_checks", [])]
        != list(REQUIRED_CHECK_PAIRS)
    ):
        blockers.append("PROTECTED_POLICY_UNTRUSTED")
    if type(pr.get("number")) is not int or pr["number"] <= 0:
        blockers.append("PR_IDENTITY_INVALID")
    if pr.get("base_ref") != "main" or pr.get("base_sha") != policy.get("base_sha"):
        blockers.append("POST_MERGE_BASE_MISMATCH")
    if pr.get("state") != "MERGED":
        blockers.append("PR_NOT_MERGED")
    object_ids = [
        policy.get("base_sha"),
        pr.get("base_sha"),
        pr.get("head_sha"),
        merge_commit_value,
        facts.get("live_main_sha"),
        *merge_parents,
        *live_main_ancestors,
    ]
    if not all(_is_object_id(value) for value in object_ids):
        blockers.append("OBJECT_ID_INVALID")
    if not merge_commit or pr.get("merge_commit_sha") != merge_commit:
        blockers.append("MERGE_COMMIT_PR_MISMATCH")
    if not merge_commit or pr.get("head_sha") not in merge_parents:
        blockers.append("PR_HEAD_NOT_IN_MERGE_PARENTS")
    live_main = facts.get("live_main_sha")
    if merge_commit != live_main and merge_commit not in live_main_ancestors:
        blockers.append("MERGE_COMMIT_NOT_ANCESTOR_OF_LIVE_MAIN")
    issue = data.get("accepted_issue", {})
    if issue.get("state") != "CLOSED" or issue.get("state_reason") != "COMPLETED":
        blockers.append("ISSUE_DISPOSITION_UNVERIFIED")

    matched, check_blockers = _trusted_checks(
        data.get("checks", []), merge_commit, REQUIRED_CI_CHECK_PAIRS
    )
    blockers.extend(check_blockers)
    decision = _decision_for(blockers, "post_merge")
    return _receipt(data, decision, blockers, matched, "post-merge")


def _receipt(
    data: dict[str, Any],
    decision: str,
    blockers: list[str],
    matched: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    assert decision in DECISIONS
    policy = data.get("protected_policy", {})
    pr = data.get("pr", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": data.get("mode"),
        "decision": decision,
        "authority_granted": False,
        "evaluated": {
            "repository": data.get("repository"),
            "pr_number": pr.get("number"),
            "base_ref": pr.get("base_ref"),
            "base_sha": pr.get("base_sha"),
            "head_sha": pr.get("head_sha"),
            "ruleset_id": policy.get("ruleset_id"),
            "path": path,
        },
        "matched_trusted_checks": sorted(
            matched, key=lambda item: (item["context"], item["integration_id"])
        ),
        "blocker_codes": sorted(set(blockers)),
        "proof_boundary": (
            "Advisory, network-free evaluation only. This receipt never authorizes "
            "approval, push, comment, label, assignment, close, merge, release, or deploy."
        ),
    }


def _invalid_receipt(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": None,
        "decision": "OWNER_GATE",
        "authority_granted": False,
        "evaluated": {},
        "matched_trusted_checks": [],
        "blocker_codes": ["INPUT_INVALID"],
        "proof_boundary": "Advisory evaluation only; invalid input grants no authority.",
        "error": str(exc),
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = evaluate(payload)
    except json.JSONDecodeError as exc:
        result = _invalid_receipt(exc)
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
