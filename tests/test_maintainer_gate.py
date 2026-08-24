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
    {"context": "AI review exact-head", "integration_id": 15368},
]
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "maintainer_gate.py"


def passing_checks(head_sha: str = HEAD) -> list[dict[str, object]]:
    return [
        {
            **required,
            "head_sha": head_sha,
            "base_sha": BASE,
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
        },
        "checks": passing_checks(),
        "threads": [{"is_resolved": True}],
        "findings": [],
        "accepted_issue": {"number": 218, "accepted": True, "state": "OPEN"},
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
    assert "TRUSTED_CHECK_UNSATISFIED:AI review exact-head:15368" in receipt[
        "blocker_codes"
    ]


def test_caller_cannot_weaken_the_exact_protected_required_check_set():
    payload = ready_payload()
    payload["protected_policy"]["required_checks"] = [
        {"context": item["context"], "integration_id": 999} for item in REQUIRED
    ]
    payload["checks"] = [
        {**check, "integration_id": 999} for check in payload["checks"]
    ]

    receipt = evaluate(payload)

    assert receipt["decision"] == "OWNER_GATE"
    assert "PROTECTED_POLICY_UNTRUSTED" in receipt["blocker_codes"]
    assert "TRUSTED_CHECK_UNSATISFIED:workflow-lint:15368" in receipt[
        "blocker_codes"
    ]


def test_non_main_base_is_not_directly_landable():
    payload = ready_payload()
    payload["pr"]["base_ref"] = "release-candidate"

    assert evaluate(payload)["decision"] == "NOT_DIRECTLY_LANDABLE"


def test_human_approval_cannot_substitute_for_the_ai_check():
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
    payload["checks"] = payload["checks"][:-1]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:AI review exact-head:15368" in receipt[
        "blocker_codes"
    ]


def test_same_named_wrong_app_check_is_rejected():
    payload = ready_payload()
    payload["checks"][0]["integration_id"] = 999

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:workflow-lint:15368" in receipt[
        "blocker_codes"
    ]


def test_ai_check_bound_to_a_prior_base_is_rejected():
    payload = ready_payload()
    payload["checks"][-1]["base_sha"] = "9" * 40

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:AI review exact-head:15368" in receipt[
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


def test_verified_finding_with_unknown_gate_class_fails_closed():
    payload = ready_payload()
    payload["findings"] = [
        {
            "verified": True,
            "gate_class": "MERGE_BLOCKNG",
            "disposition": "ACCEPTED_FOLLOW_UP",
        }
    ]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "VERIFIED_FINDING_GATE_CLASS_INVALID" in receipt["blocker_codes"]


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


def test_landing_requires_a_positive_exact_pr_number():
    payload = ready_payload("landing")
    payload["pr"].pop("number")
    payload["merge_authorization"] = exact_authorization()
    payload["merge_authorization"].pop("pr_number")

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "PR_IDENTITY_INVALID" in receipt["blocker_codes"]


def test_admin_bypass_is_forbidden_after_bootstrap():
    payload = ready_payload("landing")
    payload["merge_authorization"] = exact_authorization(True)

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "ADMIN_BYPASS_FORBIDDEN" in receipt["blocker_codes"]


def test_post_merge_verifies_exact_merge_commit():
    receipt = evaluate(post_merge_payload())

    assert receipt["decision"] == "POST_MERGE_VERIFIED"
    assert len(receipt["matched_trusted_checks"]) == len(REQUIRED) - 1


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


def test_post_merge_revalidates_the_pr_base():
    payload = post_merge_payload()
    payload["pr"]["base_ref"] = "release-candidate"

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "POST_MERGE_BASE_MISMATCH" in receipt["blocker_codes"]


def test_concurrent_later_main_is_valid_when_merge_commit_is_ancestor():
    receipt = evaluate(post_merge_payload(LATER_MAIN))

    assert receipt["decision"] == "POST_MERGE_VERIFIED"
    assert "MERGE_COMMIT_NOT_ANCESTOR_OF_LIVE_MAIN" not in receipt["blocker_codes"]


def test_post_merge_rejects_mapping_shaped_merge_parents():
    payload = post_merge_payload()
    payload["post_merge"]["merge_parents"] = {BASE: True, HEAD: True}

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "POST_MERGE_PARENTS_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_mapping_shaped_live_main_ancestors():
    payload = post_merge_payload(LATER_MAIN)
    payload["post_merge"]["live_main_ancestors"] = {MERGE: True}

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "POST_MERGE_ANCESTRY_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_invalid_merge_parent_object_ids():
    payload = post_merge_payload()
    payload["post_merge"]["merge_parents"] = [HEAD, {}]

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "OBJECT_ID_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_invalid_live_main_ancestor_object_ids():
    payload = post_merge_payload(LATER_MAIN)
    payload["post_merge"]["live_main_ancestors"] = [MERGE, {}]

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "OBJECT_ID_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_non_string_merge_commit_object_id():
    payload = post_merge_payload()
    payload["post_merge"]["merge_commit"] = int("3" * 40)
    payload["pr"]["merge_commit_sha"] = int("3" * 40)

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "OBJECT_ID_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_a_single_parent_commit():
    payload = post_merge_payload()
    payload["post_merge"]["merge_parents"] = [HEAD]

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "POST_MERGE_PARENTS_INVALID" in receipt["blocker_codes"]


def test_post_merge_rejects_duplicate_merge_parents():
    payload = post_merge_payload()
    payload["post_merge"]["merge_parents"] = [HEAD, HEAD]

    receipt = evaluate(payload)

    assert receipt["decision"] == "STATE_DRIFT"
    assert "POST_MERGE_PARENTS_INVALID" in receipt["blocker_codes"]


def test_malformed_nested_json_fails_closed_with_a_receipt():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"schema_version": "1", "mode": "readiness", "pr": []}),
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(result.stdout)
    assert receipt["decision"] == "OWNER_GATE"
    assert receipt["blocker_codes"] == ["INPUT_INVALID"]
    assert receipt["authority_granted"] is False
