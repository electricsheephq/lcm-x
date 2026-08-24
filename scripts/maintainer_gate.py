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
REQUIRED_CHECK_PAIRS = REQUIRED_CI_CHECK_PAIRS + (("AI review exact-head", 15368),)
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
    base_sha: str,
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
            and (
                context != "AI review exact-head"
                or check.get("base_sha") == base_sha
            )
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

    matched, check_blockers = _trusted_checks(
        data.get("checks", []), head_sha, str(policy.get("base_sha", ""))
    )
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
    return _receipt(data, decision, blockers, matched, "protected-normal")


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
        data.get("checks", []),
        merge_commit,
        str(policy.get("base_sha", "")),
        REQUIRED_CI_CHECK_PAIRS,
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
