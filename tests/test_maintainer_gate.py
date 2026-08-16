from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from scripts.maintainer_gate import evaluate


HEAD = "1" * 40
BASE = "2" * 40
MERGE = "3" * 40
LATER_MAIN = "4" * 40
REQUIRED = [
    {"context": "workflow-lint", "integration_id": 15368},
    {"context": "lint", "integration_id": 15368},
    {"context": "test (3.11)", "integration_id": 15368},
    {"context": "test (3.12)", "integration_id": 15368},
    {"context": "test (3.13)", "integration_id": 15368},
    {"context": "test (3.14)", "integration_id": 15368},
    {"context": "Analyze (actions)", "integration_id": 15368},
    {"context": "Analyze (javascript-typescript)", "integration_id": 15368},
    {"context": "Analyze (python)", "integration_id": 15368},
]
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "maintainer_gate.py"


def passing_checks(head_sha: str = HEAD) -> list[dict[str, object]]:
    return [
        {
            **required,
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "success",
        }
        for required in REQUIRED
    ]


def ready_payload(mode: str = "readiness") -> dict[str, object]:
    return {
        "schema_version": "1",
        "mode": mode,
        "repository": "electricsheephq/lcm-x",
        "protected_policy": {
            "source": "protected-main",
            "base_ref": "main",
            "base_sha": BASE,
            "ruleset_id": 20888757,
            "required_checks": deepcopy(REQUIRED),
            "bypass_actors": [],
        },
        "pr": {
            "number": 218,
            "base_ref": "main",
            "base_sha": BASE,
            "head_sha": HEAD,
            "author": "100yenadmin",
            "state": "OPEN",
            "draft": False,
            "requires_semantic_review": True,
        },
        "checks": passing_checks(),
        "latest_reviews": [
            {
                "author": "Tosko4",
                "state": "APPROVED",
                "commit_sha": HEAD,
                "submitted_at": "2026-08-16T10:00:00Z",
                "codeowner": True,
            }
        ],
        "threads": [{"is_resolved": True}],
        "findings": [],
        "accepted_issue": {"number": 218, "accepted": True, "state": "OPEN"},
        "semantic_review_receipt": {
            "status": "PASS",
            "head_sha": HEAD,
            "independent": True,
        },
    }


def exact_authorization(use_admin_bypass: bool = False) -> dict[str, object]:
    return {
        "repository": "electricsheephq/lcm-x",
        "pr_number": 218,
        "head_sha": HEAD,
        "action": "merge",
        "actor": "100yenadmin",
        "actor_id": 239388517,
        "use_admin_bypass": use_admin_bypass,
    }


def blind_receipts() -> list[dict[str, object]]:
    return [
        {
            "lane": "acceptance",
            "status": "PASS",
            "score": 97,
            "head_sha": HEAD,
            "independent": True,
            "reviewer_id": "checker-acceptance",
            "receipt_id": "receipt-acceptance-001",
        },
        {
            "lane": "adversarial",
            "status": "PASS",
            "score": 96,
            "head_sha": HEAD,
            "independent": True,
            "reviewer_id": "checker-adversarial",
            "receipt_id": "receipt-adversarial-001",
        },
    ]


def admin_payload() -> dict[str, object]:
    payload = ready_payload("landing")
    payload["latest_reviews"] = []
    payload["merge_authorization"] = exact_authorization(True)
    payload["protected_policy"]["bypass_actors"] = [
        {
            "actor_id": 239388517,
            "actor_type": "User",
            "bypass_mode": "pull_request",
        }
    ]
    payload["blind_review_receipts"] = blind_receipts()
    return payload


def admin_qualification() -> dict[str, object]:
    return {
        "repository": "electricsheephq/lcm-x",
        "pr_number": 218,
        "head_sha": HEAD,
        "action": "qualify_admin_pr_only",
        "actor": "100yenadmin",
        "actor_id": 239388517,
    }


def post_merge_payload(live_main: str = MERGE) -> dict[str, object]:
    payload = ready_payload("post_merge")
    payload["pr"]["state"] = "MERGED"
    payload["pr"]["merge_commit_sha"] = MERGE
    payload["checks"] = passing_checks(MERGE)
    payload["accepted_issue"] = {
        "number": 218,
        "accepted": True,
        "state": "CLOSED",
        "state_reason": "COMPLETED",
    }
    payload["post_merge"] = {
        "merge_commit": MERGE,
        "merge_parents": [BASE, HEAD],
        "live_main_sha": live_main,
        "live_main_ancestors": [MERGE] if live_main != MERGE else [],
    }
    return payload


def test_readiness_is_advisory_and_does_not_require_merge_authorization():
    receipt = evaluate(ready_payload())

    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert receipt["authority_granted"] is False
    assert "never authorizes" in receipt["proof_boundary"]


def test_json_stdin_stdout_cli_round_trip_is_advisory():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(ready_payload()),
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(result.stdout)
    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert receipt["authority_granted"] is False
    assert result.stderr == ""


def test_invocation_does_not_authorize_a_landing_write():
    receipt = evaluate(ready_payload("landing"))

    assert receipt["decision"] == "OWNER_GATE"
    assert "EXACT_MERGE_AUTHORIZATION_MISSING" in receipt["blocker_codes"]


def test_readiness_rejects_a_merge_authorization_envelope():
    payload = ready_payload()
    payload["merge_authorization"] = exact_authorization()

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "READINESS_INPUT_CONTAINS_MERGE_AUTHORIZATION" in receipt["blocker_codes"]


def test_pr_authored_policy_cannot_weaken_protected_main_policy():
    payload = ready_payload()
    payload["pr_policy"] = {"required_checks": []}
    payload["checks"] = payload["checks"][:-1]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:Analyze (python):15368" in receipt[
        "blocker_codes"
    ]


def test_non_main_base_is_not_directly_landable():
    payload = ready_payload()
    payload["pr"]["base_ref"] = "release-candidate"

    assert evaluate(payload)["decision"] == "NOT_DIRECTLY_LANDABLE"


def test_author_approval_cannot_satisfy_non_author_gate():
    payload = ready_payload()
    payload["latest_reviews"] = [
        {
            "author": "100yenadmin",
            "state": "APPROVED",
            "commit_sha": HEAD,
            "submitted_at": "2026-08-16T10:00:00Z",
            "codeowner": True,
        }
    ]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "NON_AUTHOR_CODEOWNER_APPROVAL_MISSING" in receipt["blocker_codes"]


def test_same_named_wrong_app_check_is_rejected():
    payload = ready_payload()
    payload["checks"][0]["integration_id"] = 999

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:workflow-lint:15368" in receipt[
        "blocker_codes"
    ]


def test_duplicate_trusted_check_is_rejected_even_when_one_passes():
    payload = ready_payload()
    payload["checks"].append(
        {
            **REQUIRED[0],
            "head_sha": HEAD,
            "status": "completed",
            "conclusion": "failure",
        }
    )

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_DUPLICATE:workflow-lint:15368" in receipt["blocker_codes"]


def test_verified_merge_blocker_requires_a_closing_disposition():
    payload = ready_payload()
    payload["findings"] = [
        {
            "verified": True,
            "gate_class": "MERGE_BLOCKING",
            "disposition": "ESCALATED_OWNER_GATE",
        }
    ]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "ACTIVE_MERGE_BLOCKING_FINDING" in receipt["blocker_codes"]


def test_missing_accepted_work_is_an_owner_gate():
    payload = ready_payload()
    payload["accepted_issue"] = {"number": 218, "accepted": False}

    receipt = evaluate(payload)

    assert receipt["decision"] == "OWNER_GATE"
    assert "ACCEPTED_ISSUE_MISSING" in receipt["blocker_codes"]


def test_wrong_ruleset_id_is_an_owner_gate():
    payload = ready_payload()
    payload["protected_policy"]["ruleset_id"] = 1

    receipt = evaluate(payload)

    assert receipt["decision"] == "OWNER_GATE"
    assert "PROTECTED_POLICY_UNTRUSTED" in receipt["blocker_codes"]


def test_normal_exact_authorization_is_ready_for_landing():
    payload = ready_payload("landing")
    payload["merge_authorization"] = exact_authorization()

    receipt = evaluate(payload)

    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert receipt["evaluated"]["path"] == "protected-normal"
    assert receipt["authority_granted"] is False


def test_pr_only_admin_bypass_requires_both_blind_scores_at_95():
    payload = admin_payload()
    payload["blind_review_receipts"][1]["score"] = 94

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "BLIND_ADVERSARIAL_REVIEW_MISSING" in receipt["blocker_codes"]


def test_pr_only_admin_bypass_requires_distinct_reviewer_identities():
    payload = admin_payload()
    payload["blind_review_receipts"][1]["reviewer_id"] = "checker-acceptance"

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "BLIND_REVIEWER_IDENTITY_DUPLICATE" in receipt["blocker_codes"]


def test_pr_only_admin_bypass_requires_distinct_receipt_identities():
    payload = admin_payload()
    payload["blind_review_receipts"][1]["receipt_id"] = "receipt-acceptance-001"

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "BLIND_RECEIPT_IDENTITY_DUPLICATE" in receipt["blocker_codes"]


def test_pr_only_admin_bypass_can_qualify_without_author_approval():
    receipt = evaluate(admin_payload())

    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert receipt["evaluated"]["path"] == "admin-pr-only-bypass"
    assert receipt["authority_granted"] is False


def test_pr_only_admin_bypass_does_not_waive_changes_requested():
    payload = admin_payload()
    payload["latest_reviews"] = [
        {
            "author": "Tosko4",
            "state": "CHANGES_REQUESTED",
            "commit_sha": HEAD,
            "submitted_at": "2026-08-16T11:00:00Z",
            "codeowner": True,
        }
    ]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "LATEST_CHANGES_REQUESTED" in receipt["blocker_codes"]


def test_pr_only_admin_bypass_rejects_any_extra_or_broad_actor():
    payload = admin_payload()
    payload["protected_policy"]["bypass_actors"].append(
        {"actor_id": 1, "actor_type": "Team", "bypass_mode": "always"}
    )

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "BROAD_OR_UNSAFE_BYPASS_ACTOR_PRESENT" in receipt["blocker_codes"]


def test_readiness_can_report_a_non_authoritative_admin_qualification():
    payload = admin_payload()
    payload["mode"] = "readiness"
    payload["admin_bypass_qualification"] = admin_qualification()
    del payload["merge_authorization"]

    receipt = evaluate(payload)

    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert receipt["evaluated"]["path"] == "admin-pr-only-qualified"
    assert receipt["authority_granted"] is False


def test_admin_readiness_qualification_is_bound_to_exact_head():
    payload = admin_payload()
    payload["mode"] = "readiness"
    payload["admin_bypass_qualification"] = admin_qualification()
    payload["admin_bypass_qualification"]["head_sha"] = "9" * 40
    del payload["merge_authorization"]

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert (
        "ADMIN_BYPASS_QUALIFICATION_HEAD_SHA_MISMATCH" in receipt["blocker_codes"]
    )


def test_post_merge_verifies_exact_merge_commit():
    receipt = evaluate(post_merge_payload())

    assert receipt["decision"] == "POST_MERGE_VERIFIED"
    assert len(receipt["matched_trusted_checks"]) == len(REQUIRED)


def test_post_merge_rejects_an_open_pr():
    payload = post_merge_payload()
    payload["pr"]["state"] = "OPEN"

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "PR_NOT_MERGED" in receipt["blocker_codes"]


def test_post_merge_binds_the_live_pr_merge_commit():
    payload = post_merge_payload()
    payload["pr"]["merge_commit_sha"] = "9" * 40

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "MERGE_COMMIT_PR_MISMATCH" in receipt["blocker_codes"]


def test_post_merge_rejects_the_wrong_ruleset_id():
    payload = post_merge_payload()
    payload["protected_policy"]["ruleset_id"] = 1

    receipt = evaluate(payload)

    assert receipt["decision"] == "OWNER_GATE"
    assert "PROTECTED_POLICY_UNTRUSTED" in receipt["blocker_codes"]


def test_concurrent_later_main_is_valid_when_merge_commit_is_ancestor():
    receipt = evaluate(post_merge_payload(LATER_MAIN))

    assert receipt["decision"] == "POST_MERGE_VERIFIED"
    assert "MERGE_COMMIT_NOT_ANCESTOR_OF_LIVE_MAIN" not in receipt["blocker_codes"]
