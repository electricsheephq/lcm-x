#!/usr/bin/env python3
"""Validate content-free, exact-head AI review receipts."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from typing import Any

REPOSITORY = "electricsheephq/lcm-x"
INTEGRATION_ID = 15368
POLICY_VERSION = "1"
HIGH_RISK = {
    "governance", "security", "data-integrity", "migration", "persistence",
    "lifecycle", "runtime", "hermes-contract", "workflow-policy", "unknown",
}
ROUTINE_PREFIXES = ("bench/", "benchmarking/", "benchmarks/", "docs/", "tests/")
ROUTINE_FILES = {
    "README.md",
    "ROADMAP.md",
    "scripts/lcm_longmemeval.py",
}
RECEIPT_FIELDS = {
    "schema_version",
    "repository",
    "pr_number",
    "base_sha",
    "head_sha",
    "risk_class",
    "lane",
    "reviewer_id",
    "task_id",
    "receipt_id",
    "verdict",
    "score",
    "findings",
    "evidence_digest",
    "issued_at",
    "expires_at",
    "policy_version",
    "integration_id",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and SAFE_ID.fullmatch(value) is not None


def _risk(paths: Any, labels: Any) -> str:
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        return "unknown"
    if any(
        path.startswith((".github/", ".agents/"))
        or path in {"AGENTS.md", "CONTRIBUTING.md"}
        or path.startswith("scripts/maintainer_gate.py")
        or path.startswith("scripts/ai_review_gate.py")
        for path in paths
    ):
        path_risk = "governance"
    elif all(path in ROUTINE_FILES or path.startswith(ROUTINE_PREFIXES) for path in paths):
        path_risk = "routine"
    else:
        path_risk = "unknown"

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return "unknown"
    label_risks = {
        label.strip().casefold()
        for label in labels
        if label.strip().casefold() in HIGH_RISK
    }
    if path_risk in HIGH_RISK:
        return path_risk
    if len(label_risks) == 1:
        return next(iter(label_risks))
    if label_risks:
        return "unknown"
    return path_risk


def evaluate(data: Any, now: datetime | None = None) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(data, dict):
        return {"decision": "FAIL", "blockers": ["INPUT_INVALID"]}
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    labels = data.get("labels")
    if data.get("labels_complete") is not True:
        blockers.append("LABEL_STATE_INCOMPLETE")
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) for label in labels)
        or any(not label.strip() for label in labels)
    ):
        blockers.append("LABELS_INVALID")
    expected = {
        "schema_version": "1",
        "repository": REPOSITORY,
        "pr_number": data.get("pr_number"),
        "base_sha": data.get("base_sha"),
        "head_sha": data.get("head_sha"),
        "risk_class": _risk(data.get("changed_paths"), labels),
        "policy_version": POLICY_VERSION,
        "integration_id": INTEGRATION_ID,
    }
    if data.get("repository") != REPOSITORY:
        blockers.append("REPOSITORY_MISMATCH")
    if type(data.get("pr_number")) is not int or data["pr_number"] <= 0:
        blockers.append("PR_INVALID")
    for name in ("base_sha", "head_sha"):
        value = data.get(name)
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            blockers.append(f"{name.upper()}_INVALID")
    if data.get("api_complete") is not True or data.get("pagination_complete") is not True:
        blockers.append("LIVE_STATE_INCOMPLETE")
    if type(data.get("unresolved_threads")) is not int or data.get("unresolved_threads") != 0:
        blockers.append("UNRESOLVED_REVIEW_THREAD")

    receipts = data.get("receipts")
    required_lanes = {"acceptance", "adversarial"} if expected["risk_class"] in HIGH_RISK else {"acceptance"}
    selected: list[dict[str, Any]] = []
    if not isinstance(receipts, list):
        blockers.append("RECEIPTS_INVALID")
        receipts = []
    actual_lanes = {
        receipt.get("lane") for receipt in receipts if isinstance(receipt, dict)
    }
    if actual_lanes != required_lanes or len(receipts) != len(required_lanes):
        blockers.append("RECEIPT_SET_INVALID")
    for lane in required_lanes:
        matches = [r for r in receipts if isinstance(r, dict) and r.get("lane") == lane]
        if len(matches) != 1:
            blockers.append(f"{lane.upper()}_RECEIPT_COUNT_INVALID")
            continue
        receipt = matches[0]
        selected.append(receipt)
        if set(receipt) != RECEIPT_FIELDS:
            blockers.append(f"{lane.upper()}_SCHEMA_INVALID")
        if any(receipt.get(k) != v for k, v in expected.items()):
            blockers.append(f"{lane.upper()}_BINDING_MISMATCH")
        ids = (receipt.get("reviewer_id"), receipt.get("task_id"), receipt.get("receipt_id"))
        if not all(_identifier(value) for value in ids):
            blockers.append(f"{lane.upper()}_IDENTITY_INVALID")
        digest = receipt.get("evidence_digest")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            blockers.append(f"{lane.upper()}_DIGEST_INVALID")
        if receipt.get("verdict") != "PASS" or type(receipt.get("score")) is not int or receipt.get("score", 0) < 95:
            blockers.append(f"{lane.upper()}_VERDICT_INVALID")
        if type(receipt.get("findings")) is not int or receipt.get("findings") != 0:
            blockers.append(f"{lane.upper()}_FINDINGS_UNRESOLVED")
        try:
            if _time(receipt.get("issued_at")) > current or _time(receipt.get("expires_at")) <= current:
                blockers.append(f"{lane.upper()}_RECEIPT_STALE")
        except (TypeError, ValueError):
            blockers.append(f"{lane.upper()}_TIMESTAMP_INVALID")
    for key in ("reviewer_id", "task_id", "receipt_id"):
        values = [r.get(key) for r in selected]
        if len(values) != len(set(values)):
            blockers.append(f"DUPLICATE_{key.upper()}")
    evidence = [
        {
            "lane": receipt.get("lane"),
            "reviewer_id": receipt.get("reviewer_id"),
            "task_id": receipt.get("task_id"),
            "receipt_id": receipt.get("receipt_id"),
            "evidence_digest": receipt.get("evidence_digest"),
            "score": receipt.get("score"),
        }
        for receipt in selected
    ]
    return {
        "decision": "PASS" if not blockers else "FAIL",
        "repository": data.get("repository"),
        "pr_number": data.get("pr_number"),
        "base_sha": data.get("base_sha"),
        "head_sha": data.get("head_sha"),
        "risk_class": expected["risk_class"],
        "policy_version": POLICY_VERSION,
        "blockers": sorted(set(blockers)),
        "receipts": sorted(evidence, key=lambda item: str(item.get("lane"))),
    }


def main() -> int:
    try:
        result = evaluate(json.load(sys.stdin))
    except (json.JSONDecodeError, TypeError, ValueError):
        result = {"decision": "FAIL", "blockers": ["INPUT_INVALID"]}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
