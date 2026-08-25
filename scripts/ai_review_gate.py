#!/usr/bin/env python3
"""Validate content-free, exact-head AI review receipts."""

from __future__ import annotations

import json
import hashlib
import re
import sys
from datetime import datetime, timezone
from typing import Any

REPOSITORY = "electricsheephq/lcm-x"
INTEGRATION_ID = 15368
POLICY_VERSION = "1"
PACKET_SCHEMA_VERSION = "2"
PACKET_KIND = "ai-review-exact-head"
PRODUCER_LOGIN = "100yenadmin"
PRODUCER_ID = 239388517
MAX_PACKET_BYTES = 65000
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
PACKET_FIELDS = {
    "schema_version", "kind", "repository", "pr_number", "base_sha",
    "head_sha", "state_fingerprint", "receipts", "producer", "run",
    "dispatch_id",
}
SNAPSHOT_FAILURE_BLOCKERS = {
    "LIVE_STATE_INVALID", "REPOSITORY_MISMATCH", "PR_INVALID",
    "BASE_SHA_INVALID", "HEAD_SHA_INVALID", "API_INCOMPLETE",
    "PAGINATION_INCOMPLETE", "API_ERROR", "CHECKS_INCOMPLETE",
}


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and SAFE_ID.fullmatch(value) is not None


def _risk(paths: Any) -> str:
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        return "unknown"
    if any(
        path.startswith((".github/", ".agents/"))
        or path in {"AGENTS.md", "CONTRIBUTING.md"}
        or path.startswith("scripts/maintainer_gate.py")
        or path.startswith("scripts/ai_review_gate.py")
        for path in paths
    ):
        return "governance"
    if all(path in ROUTINE_FILES or path.startswith(ROUTINE_PREFIXES) for path in paths):
        return "routine"
    return "unknown"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def state_fingerprint(live: dict[str, Any]) -> str:
    """Hash only the exact-head state that receipt reuse is allowed to bind."""
    paths = live.get("changed_paths")
    events = live.get("timeline_events", [])
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("changed paths incomplete")
    if len(paths) != len(set(paths)):
        raise ValueError("changed paths duplicate")
    if not isinstance(events, list) or not all(isinstance(e, (dict, str)) for e in events):
        raise ValueError("timeline incomplete")
    return _sha({
        "base_sha": live.get("base_sha"),
        "head_sha": live.get("head_sha"),
        "changed_paths": sorted(paths),
        "lifecycle_events": sorted((_canonical(e) for e in events)),
    })


def build_packet(
    live: dict[str, Any],
    receipts: list[dict[str, Any]],
    *,
    producer: dict[str, Any],
    run: dict[str, Any],
    dispatch_id: str,
) -> dict[str, Any]:
    """Create the bounded, canonical, content-free schema-v2 receipt packet."""
    if not isinstance(live, dict) or type(live.get("pr_number")) is not int or live["pr_number"] <= 0:
        raise ValueError("pr number invalid")
    if producer != {"login": PRODUCER_LOGIN, "id": PRODUCER_ID, "type": "User"}:
        raise ValueError("producer invalid")
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "kind": PACKET_KIND,
        "repository": REPOSITORY,
        "pr_number": live.get("pr_number"),
        "base_sha": live.get("base_sha"),
        "head_sha": live.get("head_sha"),
        "state_fingerprint": state_fingerprint(live),
        "receipts": receipts,
        "producer": producer,
        "run": run,
        "dispatch_id": dispatch_id,
    }
    encoded = _canonical(packet).encode("utf-8")
    if len(encoded) > MAX_PACKET_BYTES:
        raise ValueError("packet oversized")
    return packet


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def _live_blockers(live: Any) -> list[str]:
    if not isinstance(live, dict):
        return ["LIVE_STATE_INVALID"]
    blockers: list[str] = []
    if live.get("repository") != REPOSITORY:
        blockers.append("REPOSITORY_MISMATCH")
    if type(live.get("pr_number")) is not int or live["pr_number"] <= 0:
        blockers.append("PR_INVALID")
    if live.get("base_ref") != "main":
        blockers.append("BASE_REF_INVALID")
    if live.get("state", "open") != "open":
        blockers.append("PR_NOT_OPEN")
    if live.get("draft", False) is True:
        blockers.append("PR_DRAFT")
    for field in ("base_sha", "head_sha"):
        if not _sha40(live.get(field)):
            blockers.append(f"{field.upper()}_INVALID")
    if live.get("api_complete") is not True:
        blockers.append("API_INCOMPLETE")
    if live.get("pagination_complete") is not True:
        blockers.append("PAGINATION_INCOMPLETE")
    if live.get("api_error"):
        blockers.append("API_ERROR")
    if type(live.get("unresolved_threads")) is not int or live["unresolved_threads"] != 0:
        blockers.append("UNRESOLVED_REVIEW_THREAD")
    try:
        state_fingerprint(live)
    except ValueError as exc:
        blockers.append("STATE_INCOMPLETE:" + str(exc).replace(" ", "_"))
    if not isinstance(live.get("check_runs"), list):
        blockers.append("CHECKS_INCOMPLETE")
    return blockers


def _receipt_blockers(receipts: Any, live: dict[str, Any], now: datetime) -> list[str]:
    blockers: list[str] = []
    if not isinstance(receipts, list) or len(receipts) != 2:
        return ["RECEIPT_SET_INVALID"]
    selected: list[dict[str, Any]] = []
    expected = {
        "schema_version": "1", "repository": REPOSITORY,
        "pr_number": live.get("pr_number"), "base_sha": live.get("base_sha"),
        "head_sha": live.get("head_sha"), "risk_class": _risk(live.get("changed_paths")),
        "policy_version": POLICY_VERSION, "integration_id": INTEGRATION_ID,
    }
    for lane in ("acceptance", "adversarial"):
        matches = [item for item in receipts if isinstance(item, dict) and item.get("lane") == lane]
        if len(matches) != 1:
            blockers.append(f"{lane.upper()}_RECEIPT_COUNT_INVALID")
            continue
        receipt = matches[0]
        selected.append(receipt)
        if set(receipt) != RECEIPT_FIELDS:
            blockers.append(f"{lane.upper()}_SCHEMA_INVALID")
        if type(receipt.get("pr_number")) is not int or receipt["pr_number"] <= 0:
            blockers.append(f"{lane.upper()}_PR_NUMBER_INVALID")
        if any(receipt.get(k) != v for k, v in expected.items()):
            blockers.append(f"{lane.upper()}_BINDING_MISMATCH")
        if not all(_identifier(receipt.get(k)) for k in ("reviewer_id", "task_id", "receipt_id")):
            blockers.append(f"{lane.upper()}_IDENTITY_INVALID")
        digest = receipt.get("evidence_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            blockers.append(f"{lane.upper()}_DIGEST_INVALID")
        if receipt.get("verdict") != "PASS" or type(receipt.get("score")) is not int or receipt.get("score", 0) < 95:
            blockers.append(f"{lane.upper()}_VERDICT_INVALID")
        if type(receipt.get("findings")) is not int or receipt.get("findings") != 0:
            blockers.append(f"{lane.upper()}_FINDINGS_UNRESOLVED")
        try:
            if _time(receipt.get("issued_at")) > now or _time(receipt.get("expires_at")) <= now:
                blockers.append(f"{lane.upper()}_RECEIPT_STALE")
        except (TypeError, ValueError):
            blockers.append(f"{lane.upper()}_TIMESTAMP_INVALID")
    for key in ("reviewer_id", "task_id", "receipt_id"):
        values = [item.get(key) for item in selected]
        if len(values) != len(set(values)):
            blockers.append(f"DUPLICATE_{key.upper()}")
    return blockers


def _packet_blockers(packet: Any, live: dict[str, Any], now: datetime) -> list[str]:
    if not isinstance(packet, dict) or set(packet) != PACKET_FIELDS:
        return ["PACKET_SCHEMA_INVALID"]
    blockers: list[str] = []
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION or packet.get("kind") != PACKET_KIND:
        blockers.append("PACKET_SCHEMA_INVALID")
    for key in ("repository", "pr_number", "base_sha", "head_sha"):
        if packet.get(key) != (REPOSITORY if key == "repository" else live.get(key)):
            blockers.append(f"PACKET_{key.upper()}_MISMATCH")
    try:
        if packet.get("state_fingerprint") != state_fingerprint(live):
            blockers.append("STATE_FINGERPRINT_DRIFT")
    except ValueError:
        blockers.append("STATE_FINGERPRINT_INVALID")
    producer = packet.get("producer")
    if producer != {"login": PRODUCER_LOGIN, "id": PRODUCER_ID, "type": "User"}:
        blockers.append("PACKET_PRODUCER_INVALID")
    run = packet.get("run")
    if not isinstance(run, dict) or set(run) != {"id", "attempt"} or not _identifier(run.get("id")):
        blockers.append("PACKET_RUN_INVALID")
    elif run.get("attempt") != 1:
        blockers.append("PACKET_RUN_ATTEMPT_INVALID")
    if not _identifier(packet.get("dispatch_id")):
        blockers.append("PACKET_DISPATCH_INVALID")
    encoded = _canonical(packet).encode("utf-8")
    if len(encoded) > MAX_PACKET_BYTES:
        blockers.append("PACKET_OVERSIZED")
    blockers.extend(_receipt_blockers(packet.get("receipts"), live, now))
    return blockers


def _check_blockers(live: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    runs = live.get("check_runs")
    if not isinstance(runs, list):
        return ["CHECKS_INCOMPLETE"], None
    expected_id = f"ai-review-gate:{live.get('pr_number')}:{live.get('base_sha')}:{live.get('head_sha')}"
    trusted = [run for run in runs if isinstance(run, dict) and run.get("name") == "AI review exact-head" and (run.get("app_id") == INTEGRATION_ID or run.get("app", {}).get("id") == INTEGRATION_ID)]
    exact = [run for run in trusted if run.get("external_id") == expected_id and run.get("head_sha") == live.get("head_sha")]
    if len(exact) > 1:
        return ["DUPLICATE_TRUSTED_CHECK"], None
    if not exact:
        return ["TRUSTED_CHECK_EXACT_MATCH_MISSING" if trusted else "TRUSTED_CHECK_MISSING"], None
    run = exact[0]
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return ["FAILED_CHECK_NOT_PROMOTED"], None
    summary = run.get("output_summary")
    if not isinstance(summary, str):
        summary = run.get("output", {}).get("summary") if isinstance(run.get("output"), dict) else None
    if not isinstance(summary, str) or len(summary.encode("utf-8")) > MAX_PACKET_BYTES:
        return ["PACKET_MISSING_OR_OVERSIZED"], None
    try:
        packet = json.loads(summary)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ["PACKET_MALFORMED"], None
    return [], packet


def _peer_result(live: Any, now: datetime) -> dict[str, Any]:
    blockers = _live_blockers(live)
    snapshot_error = any(
        blocker in SNAPSHOT_FAILURE_BLOCKERS or blocker.startswith("STATE_INCOMPLETE:")
        for blocker in blockers
    )
    if not blockers:
        check_errors, packet = _check_blockers(live)
        blockers.extend(check_errors)
        if packet is not None:
            blockers.extend(_packet_blockers(packet, live, now))
    return {
        "pr_number": live.get("pr_number") if isinstance(live, dict) else None,
        "decision": "PASS" if not blockers else "FAIL",
        "preserve": not blockers,
        "snapshot_error": snapshot_error,
        "blockers": sorted(set(blockers)),
    }


def evaluate_reconciliation(data: Any, now: datetime | None = None) -> dict[str, Any]:
    """Validate a fresh dispatch and reconstruct all independently valid peers."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not isinstance(data, dict) or data.get("schema_version") != PACKET_SCHEMA_VERSION:
        return {"decision": "FAIL", "blockers": ["PACKET_SCHEMA_INVALID"], "peers": []}
    peers = data.get("peers")
    if not isinstance(peers, list):
        return {"decision": "FAIL", "blockers": ["PEER_SNAPSHOT_INVALID"], "peers": []}
    if data.get("mode") == "peer_only":
        peer_results = [_peer_result(peer, current) for peer in peers]
        blockers = ["PEER_SNAPSHOT_INCOMPLETE"] if any(item["snapshot_error"] for item in peer_results) else []
        return {
            "decision": "PASS" if not blockers else "FAIL",
            "blockers": blockers,
            "peers": sorted(peer_results, key=lambda item: (item["pr_number"] is None, item["pr_number"])),
        }
    target = data.get("target") if isinstance(data.get("target"), dict) else data
    blockers = _live_blockers(target)
    producer = data.get("producer")
    if producer != {"login": PRODUCER_LOGIN, "id": PRODUCER_ID, "type": "User"}:
        blockers.append("PRODUCER_UNAUTHENTICATED")
    run = data.get("run")
    if not isinstance(run, dict) or not _identifier(run.get("id")):
        blockers.append("RUN_ID_INVALID")
    if not isinstance(run, dict) or run.get("attempt") != 1:
        blockers.append("RUN_ATTEMPT_INVALID")
    dispatch_id = data.get("dispatch_id")
    if not _identifier(dispatch_id):
        blockers.append("DISPATCH_ID_INVALID")
    known = data.get("prior_dispatch_ids", [])
    if not isinstance(known, list) or dispatch_id in known:
        blockers.append("DISPATCH_ID_NOT_FRESH")
    receipts = data.get("receipts")
    blockers.extend(_receipt_blockers(receipts, target, current))
    try:
        packet = build_packet(target, receipts, producer=producer, run=run, dispatch_id=dispatch_id)
        blockers.extend(_packet_blockers(packet, target, current))
    except (TypeError, ValueError):
        blockers.append("PACKET_INVALID")
    peer_results = [_peer_result(peer, current) for peer in peers]
    if any(item["snapshot_error"] for item in peer_results):
        blockers.append("PEER_SNAPSHOT_INCOMPLETE")
    seen_dispatches: set[str] = set()
    for item, peer in zip(peer_results, peers):
        if item["preserve"]:
            _, peer_packet = _check_blockers(peer)
            if peer_packet is not None:
                seen_dispatches.add(peer_packet.get("dispatch_id"))
    if dispatch_id in seen_dispatches:
        blockers.append("DISPATCH_ID_NOT_FRESH")
    return {
        "decision": "PASS" if not blockers else "FAIL",
        "repository": target.get("repository") if isinstance(target, dict) else None,
        "pr_number": target.get("pr_number") if isinstance(target, dict) else None,
        "base_sha": target.get("base_sha") if isinstance(target, dict) else None,
        "head_sha": target.get("head_sha") if isinstance(target, dict) else None,
        "packet": packet if not blockers else None,
        "blockers": sorted(set(blockers)),
        "peers": sorted(peer_results, key=lambda item: (item["pr_number"] is None, item["pr_number"])),
    }


def evaluate(data: Any, now: datetime | None = None) -> dict[str, Any]:
    if isinstance(data, dict) and (
        data.get("schema_version") == PACKET_SCHEMA_VERSION
        or "peers" in data
    ):
        return evaluate_reconciliation(data, now)
    blockers: list[str] = []
    if not isinstance(data, dict):
        return {"decision": "FAIL", "blockers": ["INPUT_INVALID"]}
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected = {
        "schema_version": "1",
        "repository": REPOSITORY,
        "pr_number": data.get("pr_number"),
        "base_sha": data.get("base_sha"),
        "head_sha": data.get("head_sha"),
        "risk_class": _risk(data.get("changed_paths")),
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
    # Review lanes are a fixed policy requirement.  Mutable PR labels and other
    # event metadata are organizational inputs only and cannot lower the gate.
    required_lanes = {"acceptance", "adversarial"}
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
