from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.maintainer_gate import NON_TURN_PATH_FILES, TURN_PATH_FILES, evaluate, review_lanes_hint


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
    {"context": "differential", "integration_id": 15368},
]
# Checked on the merge commit after landing; differential runs on PRs only.
CI_REQUIRED = REQUIRED[:6]
BOTH_LANES = ["acceptance", "adversarial"]
TURN_PATH_PACKET = [
    "Consumer enumeration: rg call sites for every changed helper and effects on each consumer.",
    "Previous-GA differential row: tests/differential (opt-in).",
    "Sibling shapes: neighbouring windows, row roles, and recovery paths.",
]
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "maintainer_gate.py"


def test_top_level_modules_are_classified() -> None:
    modules = {path.name for path in SCRIPT.parents[1].glob("*.py") if path.is_file()}
    guidance = (
        "Classify every new top-level module in scripts/maintainer_gate.py's TURN_PATH_FILES "
        "or NON_TURN_PATH_FILES; for turn-path modules, update TURN_PATH_FILES in "
        ".github/workflows/differential.yml too."
    )
    assert TURN_PATH_FILES.isdisjoint(NON_TURN_PATH_FILES), guidance
    assert modules == TURN_PATH_FILES | NON_TURN_PATH_FILES, guidance


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
            "strict_required_status_checks_policy": True,
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
    payload["checks"] = [c for c in payload["checks"] if c["context"] != "test (3.14)"]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:test (3.14):15368" in receipt[
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


def test_human_approval_cannot_substitute_for_a_required_check():
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
    payload["checks"] = [c for c in payload["checks"] if c["context"] != "test (3.14)"]

    receipt = evaluate(payload)

    assert receipt["decision"] == "NOT_READY"
    assert "TRUSTED_CHECK_UNSATISFIED:test (3.14):15368" in receipt[
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


def test_retired_ai_review_check_is_not_required():
    payload = ready_payload()

    receipt = evaluate(payload)

    assert receipt["decision"] == "READY_FOR_AUTHORIZED_LANDING"
    assert "PROTECTED_POLICY_UNTRUSTED" not in receipt["blocker_codes"]
    assert [item["context"] for item in receipt["matched_trusted_checks"]] == sorted(
        item["context"] for item in REQUIRED
    )


@pytest.mark.parametrize("checks_present", [True, False], ids=["ready", "blocked"])
def test_review_lanes_hint_never_blocks_readiness(checks_present):
    payload = ready_payload()
    if not checks_present:
        payload["checks"] = []
    missing = evaluate(payload)
    payload["changed_files"] = [{"filename": "engine.py"}]
    supplied = evaluate(payload)

    assert missing["decision"] == supplied["decision"] == (
        "READY_FOR_AUTHORIZED_LANDING" if checks_present else "NOT_READY"
    )
    assert missing["blocker_codes"] == supplied["blocker_codes"]
    assert supplied["review_lanes_hint"]["turn_path_probe_packet"] == TURN_PATH_PACKET
    assert "turn_path_probe_packet" not in missing["review_lanes_hint"]
    assert missing["review_lanes_hint"]["changed_files"] == "unknown"
    assert missing["review_lanes_hint"]["required_review_lanes"] == BOTH_LANES
    assert supplied["review_lanes_hint"]["required_review_lanes"] == BOTH_LANES


@pytest.mark.parametrize(
    ("changed_files", "risks", "lanes"),
    [
        *[
            ([{"filename": path}], ["lcm-memory-preservation", "turn-path"], BOTH_LANES)
            for path in (
                "engine.py", "survival_fit.py", "__init__.py", "compaction.py", "identity_anchor.py",
                "assembly_plan.py", "compaction_policy.py", "compaction_request.py", "compress_run.py",
                "leaf_prep.py", "summary_attempt.py", "summary_outcome.py", "survival_codec.py", "tool_pairs.py",
                "config.py",
                "db_bootstrap.py",
                "diagnostics.py",
                "engine_registry.py",
                "message_patterns.py",
                "model_routing.py",
                "plugin_identity.py",
                "rollup_builder.py",
                "rollup_periods.py",
                "rollup_store.py",
                "schemas.py",
                "session_patterns.py",
                "sqlite_util.py",
                "store.py",
                "tool_registry.py",
                "reset_state.py",
                "recognition_matching.py",
                "emission_projection.py",
                "ingest_content.py",
                "ingest_integrity.py",
                "ingest_pem_model.py",
                "ingest_pem_scan.py",
                "ingest_persisted.py",
                "ingest_policy.py",
            )
        ],
        *[
            ([{"filename": path}], ["lcm-memory-preservation"], BOTH_LANES)
            for path in ("presets.py", "tools.py")
        ],
        (
            [{"filename": "engine.py", "previous_filename": "presets.py"}],
            ["lcm-memory-preservation", "turn-path"],
            BOTH_LANES,
        ),
        (
            [{"filename": "presets.py", "previous_filename": "engine.py"}],
            ["lcm-memory-preservation", "turn-path"],
            BOTH_LANES,
        ),
        ([{"filename": "docs/operator-guide.md"}, "README.md"], [], ["acceptance"]),
        (
            [{"filename": "docs/moved.md", "previous_filename": "AGENTS.md"}],
            ["review-provenance-policy"],
            BOTH_LANES,
        ),
    ],
    ids=["engine", "survival-fit", "pre-llm-hook", "compaction", "ingest-identity",
         "assembly-plan", "compaction-policy", "compaction-request", "compress-run", "leaf-prep",
         "summary-attempt", "summary-outcome", "survival-codec", "tool-pairs",
         "config",
         "db-bootstrap",
         "diagnostics",
         "engine-registry",
         "message-patterns",
         "model-routing",
         "plugin-identity",
         "rollup-builder",
         "rollup-periods",
         "rollup-store",
         "schemas",
         "session-patterns",
         "sqlite-util",
         "store",
         "tool-registry",
         "reset-state",
         "recognition-matching",
         "emission-projection",
         "ingest-content",
         "ingest-integrity",
         "ingest-pem-model",
         "ingest-pem-scan",
         "ingest-persisted",
         "ingest-policy",
         "preset-only", "tools-only",
         "rename-into-turn-path", "rename-out-of-turn-path",
         "docs-only", "rename-across-policy-boundary"],
)
def test_review_lanes_hint_maps_changed_paths(changed_files, risks, lanes):
    hint = review_lanes_hint(changed_files)

    assert hint["named_risks"] == risks
    assert hint["required_review_lanes"] == lanes
    assert set(hint) == {"changed_files", "named_risks", "required_review_lanes"} | (
        {"turn_path_probe_packet"} if "turn-path" in risks else set()
    )
    if "turn-path" in risks:
        assert hint["turn_path_probe_packet"] == TURN_PATH_PACKET


@pytest.mark.parametrize(
    "changed_files",
    [None, [], [""], [{"filename": "  "}],
     [{"filename": "docs/a.md", "previous_filename": ""}]],
    ids=["unknown", "empty-list", "blank-string", "blank-filename", "blank-previous-filename"],
)
def test_unretrieved_changed_files_keep_both_review_lanes(changed_files):
    hint = review_lanes_hint(changed_files)

    assert hint["changed_files"] == "unknown"
    assert hint["named_risks"] is None
    assert "turn_path_probe_packet" not in hint
    assert hint["required_review_lanes"] == BOTH_LANES


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


def test_non_strict_required_checks_policy_is_an_owner_gate():
    payload = ready_payload()
    payload["protected_policy"]["strict_required_status_checks_policy"] = False

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
    assert len(receipt["matched_trusted_checks"]) == len(CI_REQUIRED)


def test_post_merge_does_not_require_differential_on_the_merge_commit():
    payload = post_merge_payload()
    payload["checks"] = [c for c in payload["checks"] if c["context"] != "differential"]

    receipt = evaluate(payload)

    assert receipt["decision"] == "POST_MERGE_VERIFIED"


def test_readiness_requires_differential_on_the_pr_head():
    payload = ready_payload()
    payload["checks"] = [c for c in payload["checks"] if c["context"] != "differential"]

    receipt = evaluate(payload)

    assert receipt["decision"] != "READY_FOR_AUTHORIZED_LANDING"
    assert "TRUSTED_CHECK_UNSATISFIED:differential:15368" in receipt["blocker_codes"]


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
