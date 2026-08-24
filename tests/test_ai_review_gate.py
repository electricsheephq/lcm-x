from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from scripts.ai_review_gate import evaluate


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
HEAD = "1" * 40
BASE = "2" * 40
DIGEST = "3" * 64
REPO_ROOT = Path(__file__).resolve().parent.parent


def receipt(lane: str, *, risk: str = "routine", reviewer: str | None = None):
    return {
        "schema_version": "1",
        "repository": "electricsheephq/lcm-x",
        "pr_number": 350,
        "base_sha": BASE,
        "head_sha": HEAD,
        "risk_class": risk,
        "lane": lane,
        "reviewer_id": reviewer or f"reviewer-{lane}",
        "task_id": f"task-{lane}",
        "receipt_id": f"receipt-{lane}",
        "verdict": "PASS",
        "score": 97,
        "findings": 0,
        "evidence_digest": DIGEST,
        "issued_at": "2026-08-24T11:00:00Z",
        "expires_at": "2026-08-25T11:00:00Z",
        "policy_version": "1",
        "integration_id": 15368,
    }


def payload(paths: list[str] | None = None):
    return {
        "repository": "electricsheephq/lcm-x",
        "pr_number": 350,
        "base_sha": BASE,
        "head_sha": HEAD,
        "changed_paths": paths or ["docs/operator-guide.md"],
        "receipts": [receipt("acceptance")],
        "unresolved_threads": 0,
        "api_complete": True,
        "pagination_complete": True,
    }


def test_routine_acceptance_receipt_passes():
    result = evaluate(payload(), NOW)

    assert result["decision"] == "PASS"
    assert result["risk_class"] == "routine"
    assert result["base_sha"] == BASE
    assert result["head_sha"] == HEAD
    assert result["blockers"] == []
    assert result["receipts"][0]["evidence_digest"] == DIGEST


def test_governance_requires_distinct_acceptance_and_adversarial_receipts():
    data = payload([".github/workflows/ai-review-gate.yml"])
    data["receipts"] = [
        receipt("acceptance", risk="governance"),
        receipt("adversarial", risk="governance"),
    ]

    assert evaluate(data, NOW)["decision"] == "PASS"

    data["receipts"][1]["reviewer_id"] = "reviewer-acceptance"
    result = evaluate(data, NOW)
    assert result["decision"] == "FAIL"
    assert "DUPLICATE_REVIEWER_ID" in result["blockers"]


def test_unknown_risk_fails_closed_to_two_lanes():
    data = payload(["access_context/tools.py"])
    result = evaluate(data, NOW)

    assert result["risk_class"] == "unknown"
    assert "ADVERSARIAL_RECEIPT_COUNT_INVALID" in result["blockers"]


def test_missing_or_extra_receipts_fail():
    data = payload()
    data["receipts"] = []
    assert "RECEIPT_SET_INVALID" in evaluate(data, NOW)["blockers"]

    data = payload()
    data["receipts"].append(receipt("adversarial"))
    assert "RECEIPT_SET_INVALID" in evaluate(data, NOW)["blockers"]


def test_stale_sub95_findings_and_malformed_digest_fail():
    cases = {
        "expires_at": ("2026-08-24T11:59:59Z", "ACCEPTANCE_RECEIPT_STALE"),
        "score": (94, "ACCEPTANCE_VERDICT_INVALID"),
        "findings": (1, "ACCEPTANCE_FINDINGS_UNRESOLVED"),
        "evidence_digest": ("not-a-digest", "ACCEPTANCE_DIGEST_INVALID"),
    }
    for field, (value, blocker) in cases.items():
        data = payload()
        data["receipts"][0][field] = value
        assert blocker in evaluate(data, NOW)["blockers"]


def test_wrong_head_base_integration_policy_and_repository_fail():
    cases = {
        "head_sha": "4" * 40,
        "base_sha": "5" * 40,
        "integration_id": 999,
        "policy_version": "stale",
        "repository": "someone/fork",
    }
    for field, value in cases.items():
        data = payload()
        data["receipts"][0][field] = value
        assert "ACCEPTANCE_BINDING_MISMATCH" in evaluate(data, NOW)["blockers"]


def test_receipt_rejects_raw_or_unregistered_fields_and_unsafe_ids():
    data = payload()
    data["receipts"][0]["raw_text"] = "must never enter a receipt"
    assert "ACCEPTANCE_SCHEMA_INVALID" in evaluate(data, NOW)["blockers"]

    data = payload()
    data["receipts"][0]["reviewer_id"] = "unsafe reviewer text"
    assert "ACCEPTANCE_IDENTITY_INVALID" in evaluate(data, NOW)["blockers"]


def test_live_state_and_thread_failures_fail_closed():
    for field, value in (
        ("api_complete", False),
        ("pagination_complete", False),
        ("unresolved_threads", 1),
    ):
        data = payload()
        data[field] = value
        assert evaluate(data, NOW)["decision"] == "FAIL"


def test_duplicate_task_and_receipt_ids_fail():
    data = payload(["AGENTS.md"])
    data["receipts"] = [
        receipt("acceptance", risk="governance"),
        receipt("adversarial", risk="governance"),
    ]
    for key in ("task_id", "receipt_id"):
        candidate = deepcopy(data)
        candidate["receipts"][1][key] = candidate["receipts"][0][key]
        assert f"DUPLICATE_{key.upper()}" in evaluate(candidate, NOW)["blockers"]


def test_workflow_is_base_trusted_and_resets_each_head():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    assert "pull_request_target:" in workflow
    assert "push:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "checks: write" in workflow
    assert "AI review exact-head" in workflow
    assert "head_sha: head" in workflow
    assert "${prNumber}:${base}:${head}" in workflow
    assert "Protected base changed" in workflow
    assert "context.ref !== `refs/heads/${defaultBranch}`" in workflow
    assert "ref: protectedSha" in workflow
    assert "actions/checkout" not in workflow
    assert "pull_request.head" not in workflow
    assert "eval(" not in workflow
