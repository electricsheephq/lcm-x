#!/usr/bin/env python3
"""Evaluate LCM-X maintainer policy receipts without network access."""

from __future__ import annotations

import json
import sys
from typing import Any


SCHEMA_VERSION = "1"
REPOSITORY = "electricsheephq/lcm-x"
RULESET_ID = 20888757
REQUIRED_CHECK_PAIRS = (
    ("workflow-lint", 15368),
    ("lint", 15368),
    ("test (3.11)", 15368),
    ("test (3.12)", 15368),
    ("test (3.13)", 15368),
    ("test (3.14)", 15368),
    ("Analyze (actions)", 15368),
    ("Analyze (javascript-typescript)", 15368),
    ("Analyze (python)", 15368),
)
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


def _latest_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        author = str(review.get("author", ""))
        if not author:
            continue
        current = latest.get(author, [])
        submitted_at = str(review.get("submitted_at", ""))
        current_time = str(current[0].get("submitted_at", "")) if current else ""
        if not current or submitted_at > current_time:
            latest[author] = [review]
        elif submitted_at == current_time:
            current.append(review)
    return [review for reviews_at_latest_time in latest.values() for review in reviews_at_latest_time]


def _trusted_checks(
    checks: list[dict[str, Any]],
    target_sha: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    matched: list[dict[str, Any]] = []
    blockers: list[str] = []

    for context, integration_id in REQUIRED_CHECK_PAIRS:
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


def _review_gate(data: dict[str, Any], head_sha: str) -> list[str]:
    pr = data.get("pr", {})
    author = str(pr.get("author", ""))
    latest = _latest_reviews(data.get("latest_reviews", []))
    blockers: list[str] = []

    if any(review.get("state") == "CHANGES_REQUESTED" for review in latest):
        blockers.append("LATEST_CHANGES_REQUESTED")

    approvals = [
        review
        for review in latest
        if review.get("state") == "APPROVED"
        and review.get("commit_sha") == head_sha
        and review.get("author") != author
        and review.get("codeowner") is True
    ]
    if not approvals:
        blockers.append("NON_AUTHOR_CODEOWNER_APPROVAL_MISSING")
    return blockers


def _semantic_review_gate(data: dict[str, Any], head_sha: str) -> list[str]:
    if data.get("pr", {}).get("requires_semantic_review") is not True:
        return []
    receipt = data.get("semantic_review_receipt", {})
    if (
        receipt.get("status") != "PASS"
        or receipt.get("head_sha") != head_sha
        or receipt.get("independent") is not True
    ):
        return ["SEMANTIC_REVIEW_MISSING_OR_STALE"]
    return []


def _blind_review_gate(data: dict[str, Any], head_sha: str) -> list[str]:
    receipts = data.get("blind_review_receipts", [])
    blockers: list[str] = []
    selected: list[dict[str, Any]] = []
    for lane in ("acceptance", "adversarial"):
        valid = [
            receipt
            for receipt in receipts
            if receipt.get("lane") == lane
            and receipt.get("status") == "PASS"
            and isinstance(receipt.get("score"), int)
            and receipt["score"] >= 95
            and receipt.get("head_sha") == head_sha
            and receipt.get("independent") is True
            and type(receipt.get("unresolved_findings")) is int
            and receipt["unresolved_findings"] == 0
            and isinstance(receipt.get("reviewer_id"), str)
            and bool(receipt["reviewer_id"])
            and isinstance(receipt.get("receipt_id"), str)
            and bool(receipt["receipt_id"])
        ]
        if not valid:
            blockers.append(f"BLIND_{lane.upper()}_REVIEW_MISSING")
        elif len(valid) > 1:
            blockers.append(f"BLIND_{lane.upper()}_REVIEW_AMBIGUOUS")
        else:
            selected.append(valid[0])

    if len(selected) == 2:
        if len({receipt["reviewer_id"] for receipt in selected}) != 2:
            blockers.append("BLIND_REVIEWER_IDENTITY_DUPLICATE")
        if len({receipt["receipt_id"] for receipt in selected}) != 2:
            blockers.append("BLIND_RECEIPT_IDENTITY_DUPLICATE")
    return blockers


def _bypass_gate(
    data: dict[str, Any], head_sha: str, actor_receipt: dict[str, Any]
) -> list[str]:
    policy = data.get("protected_policy", {})
    actor_id = actor_receipt.get("actor_id")
    actor_login = actor_receipt.get("actor")
    bypass_actors = policy.get("bypass_actors", [])
    bypass_actor = [
        actor
        for actor in bypass_actors
        if actor.get("actor_type") == "User"
        and actor.get("actor_id") == actor_id
        and actor.get("bypass_mode") == "pull_request"
    ]
    blockers: list[str] = []
    if not actor_login or not bypass_actor:
        blockers.append("PR_ONLY_ADMIN_BYPASS_NOT_CONFIGURED_FOR_ACTOR")
    if len(bypass_actor) != 1 or len(bypass_actors) != 1:
        blockers.append("BROAD_OR_UNSAFE_BYPASS_ACTOR_PRESENT")
    blockers.extend(_blind_review_gate(data, head_sha))
    return blockers


def _bypass_qualification_gate(
    data: dict[str, Any], head_sha: str, qualification: dict[str, Any]
) -> list[str]:
    expected = {
        "repository": data.get("repository"),
        "pr_number": data.get("pr", {}).get("number"),
        "head_sha": head_sha,
        "action": "qualify_admin_pr_only",
    }
    blockers = [
        f"ADMIN_BYPASS_QUALIFICATION_{key.upper()}_MISMATCH"
        for key, value in expected.items()
        if qualification.get(key) != value
    ]
    blockers.extend(_bypass_gate(data, head_sha, qualification))
    return blockers


def _accepted_issue_gate(data: dict[str, Any], require_closed: bool = False) -> list[str]:
    issue = data.get("accepted_issue", {})
    pr_number = data.get("pr", {}).get("number")
    valid = (
        issue.get("accepted") is True
        and type(issue.get("number")) is int
        and issue["number"] > 0
        and issue.get("repository") == REPOSITORY
        and issue.get("pr_number") == pr_number
        and issue.get("scope_matches") is True
    )
    if not valid:
        return ["ACCEPTED_ISSUE_MISSING"]
    if require_closed and (
        issue.get("state") != "CLOSED" or issue.get("state_reason") != "COMPLETED"
    ):
        return ["ISSUE_DISPOSITION_UNVERIFIED"]
    return []


def _base_blockers(data: dict[str, Any]) -> tuple[list[str], str, list[dict[str, Any]]]:
    blockers: list[str] = []
    policy = data.get("protected_policy", {})
    pr = data.get("pr", {})
    head_value = pr.get("head_sha")
    head_sha = head_value if isinstance(head_value, str) else ""

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
    if not all(
        _is_object_id(value)
        for value in (policy.get("base_sha"), pr.get("base_sha"), head_sha)
    ):
        blockers.append("OBJECT_ID_INVALID")
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
    blockers.extend(_accepted_issue_gate(data))
    blockers.extend(_semantic_review_gate(data, head_sha))
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
        }
        or blocker.startswith("MERGE_AUTHORIZATION_")
        or blocker.startswith("ADMIN_BYPASS_QUALIFICATION_")
        for blocker in blockers
    ):
        return "STATE_DRIFT"
    if any(
        blocker in {
            "PROTECTED_POLICY_UNTRUSTED",
            "EXACT_MERGE_AUTHORIZATION_MISSING",
            "PR_ONLY_ADMIN_BYPASS_NOT_CONFIGURED_FOR_ACTOR",
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
    review_blockers = _review_gate(data, head_sha)
    authorization = data.get("merge_authorization", {})
    qualification = data.get("admin_bypass_qualification", {})
    use_bypass = authorization.get("use_admin_bypass") is True
    qualify_bypass = mode == "readiness" and bool(qualification)
    if use_bypass or qualify_bypass:
        blockers.extend(
            blocker
            for blocker in review_blockers
            if blocker != "NON_AUTHOR_CODEOWNER_APPROVAL_MISSING"
        )
    else:
        blockers.extend(review_blockers)
    if mode == "landing":
        blockers.extend(_landing_authorization_gate(data, head_sha))
        if use_bypass:
            blockers.extend(_bypass_gate(data, head_sha, authorization))
        elif authorization.get("use_admin_bypass") not in (None, False):
            blockers.append("ADMIN_BYPASS_FLAG_INVALID")
    elif authorization:
        blockers.append("READINESS_INPUT_CONTAINS_MERGE_AUTHORIZATION")
    elif qualify_bypass:
        blockers.extend(_bypass_qualification_gate(data, head_sha, qualification))

    decision = _decision_for(blockers, mode)
    if use_bypass:
        path = "admin-pr-only-bypass"
    elif qualify_bypass:
        path = "admin-pr-only-qualified"
    else:
        path = "protected-normal"
    return _receipt(data, decision, blockers, matched, path)


def _evaluate_post_merge(data: dict[str, Any]) -> dict[str, Any]:
    policy = data.get("protected_policy", {})
    pr = data.get("pr", {})
    facts = data.get("post_merge", {})
    merge_commit = str(facts.get("merge_commit", ""))
    blockers: list[str] = []

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
        merge_commit,
        facts.get("live_main_sha"),
        *facts.get("merge_parents", []),
        *facts.get("live_main_ancestors", []),
    ]
    if not object_ids or not all(_is_object_id(value) for value in object_ids):
        blockers.append("OBJECT_ID_INVALID")
    if not merge_commit or pr.get("merge_commit_sha") != merge_commit:
        blockers.append("MERGE_COMMIT_PR_MISMATCH")
    if not merge_commit or pr.get("head_sha") not in facts.get("merge_parents", []):
        blockers.append("PR_HEAD_NOT_IN_MERGE_PARENTS")
    live_main = facts.get("live_main_sha")
    if merge_commit != live_main and merge_commit not in facts.get("live_main_ancestors", []):
        blockers.append("MERGE_COMMIT_NOT_ANCESTOR_OF_LIVE_MAIN")
    blockers.extend(_accepted_issue_gate(data, require_closed=True))

    matched, check_blockers = _trusted_checks(data.get("checks", []), merge_commit)
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
