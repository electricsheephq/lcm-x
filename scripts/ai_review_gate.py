#!/usr/bin/env python3
"""Validate original, exact-head AI review assessments."""

from __future__ import annotations

import json
import hashlib
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

REPOSITORY = "electricsheephq/lcm-x"
INTEGRATION_ID = 15368
POLICY_VERSION = "2"
ASSESSMENT_SCHEMA_VERSION = "2"
PACKET_SCHEMA_VERSION = "2"
PACKET_KIND = "ai-review-exact-head"
PRODUCER_LOGIN = "100yenadmin"
PRODUCER_ID = 239388517
MAX_PACKET_BYTES = 65000
MAX_REVIEW_BODY_BYTES = 8192
# Public withdrawal: PR462 issuecomment-5771571369. These producer-assigned
# scores were not issued by the reviewers. Never accept or preserve them.
WITHDRAWN_RECEIPT_IDS = frozenset({
    "52eca5d8-1628-4f03-9128-34a5e1d5746f",
    "5a6b05ee-c573-4cd9-b6f4-9a944fba61ee",
})
REVIEW_POLICY_FILES = {
    ".agents/skills/review-pr/SKILL.md", ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/ai-review-gate.yml", "AGENTS.md", "CONTRIBUTING.md",
    "docs/review-evidence-provenance.md", "scripts/ai_review_gate.py",
    "scripts/maintainer_gate.py",
}
MEMORY_PRESERVATION_FILES = {
    "__init__.py", "assertion_store.py", "aux_session.py", "command.py", "compaction.py",
    "config.py",
    "dag.py", "db_bootstrap.py", "engine.py", "engine_registry.py",
    "externalize.py", "fresh_tail.py", "ingest_protection.py",
    "lifecycle_state.py", "maintenance.py", "placeholder_ledger.py",
    "query_view_store.py", "reconcile.py", "reset_state.py", "rollup_builder.py",
    "rollup_store.py",
    "schemas.py", "scope_storage.py", "sqlite_util.py", "store.py",
    "trajectory_store.py", "vector_store.py",
    "scripts/backfill_externalized_tool_outputs.py",
    "scripts/import_lossless_claw.py",
}
ASSESSMENT_FIELDS = {
    "schema_version",
    "repository",
    "pr_number",
    "base_sha",
    "head_sha",
    "named_risks",
    "lane",
    "reviewer_id",
    "review_id",
    "verdict",
    "findings",
    "scope",
    "limitations",
    "acceptance_evidence",
    "issued_at",
    "original_body",
    "producer_tracking_digest",
    "producer_tracking_expires_at",
    "policy_version",
    "integration_id",
}
REVIEW_ARTIFACT_REF_FIELDS = {"lane", "review_id"}
REVIEW_ARTIFACT_FIELDS = {"review_id", "state", "commit_id", "submitted_at", "user", "body"}
REVIEW_ARTIFACT_BODY_FIELDS = {"schema_version", "repository", "pr_number", "base_sha", "head_sha", "lane",
    "verdict", "scope", "named_risks", "findings", "limitations",
    "acceptance_evidence", "policy_version"}
REVIEW_ARTIFACT_MARKER = "<!-- lcm-x-ai-review:v2\n"
REVIEW_ARTIFACT_SUFFIX = "\n-->"
# Protected identities supply no verdict or score without a valid review body.
REVIEW_PUBLISHERS = {"acceptance": {"id": 298367747, "login": "evaos-code-review-bot[bot]"}, "adversarial": {"id": 199175422, "login": "chatgpt-codex-connector[bot]"}}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
PACKET_FIELDS = {
    "schema_version", "kind", "repository", "pr_number", "base_sha",
    "head_sha", "state_fingerprint", "assessments", "producer", "run",
    "dispatch_id", "review_artifact_refs",
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _named_risks(paths: Any) -> list[str]:
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        return []
    risks: list[str] = []
    if any(path in REVIEW_POLICY_FILES or path.startswith(".agents/skills/land-pr/") for path in paths):
        risks.append("review-provenance-policy")
    if any(
        path in MEMORY_PRESERVATION_FILES
        or path.startswith(("access_context/", "access_policy/", "teams/"))
        for path in paths
    ):
        risks.append("lcm-memory-preservation")
    return risks


def _required_lanes(live: dict[str, Any]) -> tuple[str, ...]:
    return ("acceptance", "adversarial") if _named_risks(live.get("changed_paths")) else ("acceptance",)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def state_fingerprint(live: dict[str, Any]) -> str:
    """Hash only the exact-head state that assessment reuse is allowed to bind."""
    paths = live.get("changed_paths")
    events = live.get("timeline_events", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
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
    assessments: list[dict[str, Any]],
    review_artifact_refs: list[dict[str, Any]],
    *,
    producer: dict[str, Any],
    run: dict[str, Any],
    dispatch_id: str,
) -> dict[str, Any]:
    """Create the bounded, exact-head packet from fetched original reviews."""
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
        "assessments": assessments,
        "review_artifact_refs": review_artifact_refs,
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


def _review_artifact_assessments(
    refs: Any,
    artifacts: Any,
    live: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Derive assessments only from review objects fetched by the workflow."""
    blockers: list[str] = []
    required_lanes = _required_lanes(live)
    expected_named_risks = _named_risks(live.get("changed_paths"))
    if not isinstance(refs, list) or len(refs) != len(required_lanes):
        return [], ["REVIEW_ARTIFACT_REF_SET_INVALID"]
    if not isinstance(artifacts, list):
        return [], ["REVIEW_ARTIFACTS_INVALID"]
    artifact_ids = [
        artifact.get("review_id") for artifact in artifacts
        if isinstance(artifact, dict)
    ]
    if len(artifact_ids) != len(set(artifact_ids)):
        blockers.append("DUPLICATE_REVIEW_ARTIFACT")
    assessments: list[dict[str, Any]] = []
    for lane in required_lanes:
        lane_refs = [
            ref for ref in refs
            if isinstance(ref, dict) and ref.get("lane") == lane
        ]
        if len(lane_refs) != 1:
            blockers.append(f"{lane.upper()}_REVIEW_ARTIFACT_REF_COUNT_INVALID")
            continue
        ref = lane_refs[0]
        if set(ref) != REVIEW_ARTIFACT_REF_FIELDS:
            blockers.append(f"{lane.upper()}_REVIEW_ARTIFACT_REF_SCHEMA_INVALID")
        review_id = ref.get("review_id")
        if type(review_id) is not int or review_id <= 0:
            blockers.append(f"{lane.upper()}_REVIEW_ARTIFACT_ID_INVALID")
            continue
        matches = [
            artifact for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("review_id") == review_id
        ]
        if len(matches) != 1:
            blockers.append(f"{lane.upper()}_REVIEW_ARTIFACT_COUNT_INVALID")
            continue
        artifact = matches[0]
        if set(artifact) != REVIEW_ARTIFACT_FIELDS:
            blockers.append(f"{lane.upper()}_REVIEW_ARTIFACT_SCHEMA_INVALID")
        publisher = REVIEW_PUBLISHERS[lane]
        user = artifact.get("user")
        if (
            not isinstance(user, dict)
            or set(user) != {"id", "login", "type"}
            or type(user.get("id")) is not int
            or user.get("id") != publisher["id"]
            or user.get("login") != publisher["login"]
            or user.get("type") != "Bot"
        ):
            blockers.append(f"{lane.upper()}_REVIEW_PUBLISHER_INVALID")
            continue
        if artifact.get("state") not in {"APPROVED", "COMMENTED"}:
            blockers.append(f"{lane.upper()}_REVIEW_STATE_INVALID")
        if artifact.get("commit_id") != live.get("head_sha"):
            blockers.append(f"{lane.upper()}_REVIEW_COMMIT_MISMATCH")
        body = artifact.get("body")
        if (
            not isinstance(body, str)
            or len(body.encode("utf-8")) > MAX_REVIEW_BODY_BYTES
            or body.count(REVIEW_ARTIFACT_MARKER) != 1
            or not body.endswith(REVIEW_ARTIFACT_SUFFIX)
        ):
            blockers.append(f"{lane.upper()}_REVIEW_BODY_INVALID")
            continue
        marker_at = body.rfind(REVIEW_ARTIFACT_MARKER)
        encoded = body[marker_at + len(REVIEW_ARTIFACT_MARKER):-len(REVIEW_ARTIFACT_SUFFIX)]
        try:
            assessment = json.loads(encoded, object_pairs_hook=_unique_object)
        except (TypeError, ValueError, json.JSONDecodeError):
            blockers.append(f"{lane.upper()}_REVIEW_BODY_INVALID")
            continue
        if not isinstance(assessment, dict) or set(assessment) != REVIEW_ARTIFACT_BODY_FIELDS:
            blockers.append(f"{lane.upper()}_REVIEW_BODY_SCHEMA_INVALID")
            continue
        expected = {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "repository": REPOSITORY,
            "pr_number": live.get("pr_number"),
            "base_sha": live.get("base_sha"),
            "head_sha": live.get("head_sha"),
            "lane": lane,
            "policy_version": POLICY_VERSION,
        }
        if (type(assessment.get("pr_number")) is not int
                or any(assessment.get(key) != value for key, value in expected.items())):
            blockers.append(f"{lane.upper()}_REVIEW_BINDING_MISMATCH")
        verdict = assessment.get("verdict")
        if not isinstance(verdict, str) or verdict not in {"PASS", "BLOCKED", "ABSTAIN"}:
            blockers.append(f"{lane.upper()}_REVIEW_VERDICT_INVALID")
        scope = assessment.get("scope")
        if not isinstance(scope, str) or not scope.strip():
            blockers.append(f"{lane.upper()}_REVIEW_SCOPE_INVALID")
        if assessment.get("named_risks") != expected_named_risks:
            blockers.append(f"{lane.upper()}_REVIEW_RISK_BINDING_MISMATCH")
        findings = assessment.get("findings")
        if not isinstance(findings, list) or not all(
            isinstance(item, str) and item.strip() for item in findings
        ):
            blockers.append(f"{lane.upper()}_REVIEW_FINDINGS_INVALID")
        limitations = assessment.get("limitations")
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) for item in limitations
        ):
            blockers.append(f"{lane.upper()}_REVIEW_LIMITATIONS_INVALID")
        evidence = assessment.get("acceptance_evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            blockers.append(f"{lane.upper()}_REVIEW_EVIDENCE_INVALID")
        submitted_at = artifact.get("submitted_at")
        try:
            issued_at = _time(submitted_at)
        except (TypeError, ValueError):
            blockers.append(f"{lane.upper()}_REVIEW_SUBMISSION_TIME_INVALID")
            continue
        assessments.append({
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "repository": REPOSITORY,
            "pr_number": live.get("pr_number"),
            "base_sha": live.get("base_sha"),
            "head_sha": live.get("head_sha"),
            "named_risks": expected_named_risks,
            "lane": lane,
            "reviewer_id": f"github-user:{user['id']}",
            "review_id": review_id,
            "verdict": verdict,
            "scope": scope,
            "findings": findings,
            "limitations": limitations,
            "acceptance_evidence": evidence,
            "issued_at": submitted_at,
            "original_body": body,
            "producer_tracking_digest": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "producer_tracking_expires_at": (issued_at + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
            "policy_version": POLICY_VERSION,
            "integration_id": INTEGRATION_ID,
        })
    return assessments, blockers


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
    if not isinstance(live.get("review_artifacts"), list):
        blockers.append("REVIEW_ARTIFACTS_INCOMPLETE")
    errors = live.get("review_artifact_errors")
    if not isinstance(errors, list):
        blockers.append("REVIEW_ARTIFACT_ERRORS_INCOMPLETE")
    return blockers


def _assessment_blockers(
    assessments: Any,
    live: dict[str, Any],
    now: datetime,
    *,
    check_freshness: bool = True,
) -> list[str]:
    # Tracking freshness is workflow metadata, derived from GitHub's submitted
    # time. It is not part of the reviewer's assessment body.
    blockers: list[str] = []
    required_lanes = _required_lanes(live)
    if not isinstance(assessments, list) or len(assessments) != len(required_lanes):
        return ["ASSESSMENT_SET_INVALID"]
    selected: list[dict[str, Any]] = []
    expected = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION, "repository": REPOSITORY,
        "pr_number": live.get("pr_number"), "base_sha": live.get("base_sha"),
        "head_sha": live.get("head_sha"),
        "named_risks": _named_risks(live.get("changed_paths")),
        "policy_version": POLICY_VERSION, "integration_id": INTEGRATION_ID,
    }
    for lane in required_lanes:
        matches = [item for item in assessments if isinstance(item, dict) and item.get("lane") == lane]
        if len(matches) != 1:
            blockers.append(f"{lane.upper()}_ASSESSMENT_COUNT_INVALID")
            continue
        assessment = matches[0]
        selected.append(assessment)
        if set(assessment) != ASSESSMENT_FIELDS:
            blockers.append(f"{lane.upper()}_SCHEMA_INVALID")
        if type(assessment.get("pr_number")) is not int or assessment["pr_number"] <= 0:
            blockers.append(f"{lane.upper()}_PR_NUMBER_INVALID")
        if type(assessment.get("integration_id")) is not int:
            blockers.append(f"{lane.upper()}_INTEGRATION_ID_INVALID")
        if any(assessment.get(k) != v for k, v in expected.items()):
            blockers.append(f"{lane.upper()}_BINDING_MISMATCH")
        if not _identifier(assessment.get("reviewer_id")) or type(assessment.get("review_id")) is not int:
            blockers.append(f"{lane.upper()}_IDENTITY_INVALID")
        digest = assessment.get("producer_tracking_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            blockers.append(f"{lane.upper()}_DIGEST_INVALID")
        body = assessment.get("original_body")
        if not isinstance(body, str) or hashlib.sha256(body.encode("utf-8")).hexdigest() != digest:
            blockers.append(f"{lane.upper()}_ORIGINAL_BODY_MISMATCH")
        if assessment.get("verdict") != "PASS":
            blockers.append(f"{lane.upper()}_VERDICT_NOT_PASS")
        if assessment.get("findings") != []:
            blockers.append(f"{lane.upper()}_FINDINGS_UNRESOLVED")
        if not isinstance(assessment.get("scope"), str) or not assessment["scope"].strip():
            blockers.append(f"{lane.upper()}_SCOPE_INVALID")
        if not isinstance(assessment.get("limitations"), list) or not all(
            isinstance(item, str) for item in assessment["limitations"]
        ):
            blockers.append(f"{lane.upper()}_LIMITATIONS_INVALID")
        evidence = assessment.get("acceptance_evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            blockers.append(f"{lane.upper()}_EVIDENCE_INVALID")
        try:
            issued = _time(assessment.get("issued_at"))
            expires = _time(assessment.get("producer_tracking_expires_at"))
            if check_freshness and (issued > now or expires <= now):
                blockers.append(f"{lane.upper()}_ASSESSMENT_STALE")
        except (TypeError, ValueError):
            blockers.append(f"{lane.upper()}_TIMESTAMP_INVALID")
    for key in ("reviewer_id", "review_id"):
        values = [item.get(key) for item in selected]
        if len(values) != len(set(values)):
            blockers.append(f"DUPLICATE_{key.upper()}")
    return blockers


def _dispatch_envelope_result(data: dict[str, Any], now: datetime) -> dict[str, Any]:
    blockers: list[str] = []
    if set(data) != {
        "schema_version", "mode", "target", "review_artifact_refs",
        "dispatch_id", "producer", "run",
    }:
        blockers.append("DISPATCH_ENVELOPE_SCHEMA_INVALID")
    target = data.get("target")
    if not isinstance(target, dict):
        return {
            "decision": "FAIL",
            "blockers": sorted(set(blockers + ["DISPATCH_TARGET_INVALID"])),
            "packet": None,
        }
    blockers.extend(_live_blockers(target))
    producer = data.get("producer")
    if producer != {"login": PRODUCER_LOGIN, "id": PRODUCER_ID, "type": "User"}:
        blockers.append("PRODUCER_UNAUTHENTICATED")
    run = data.get("run")
    if not isinstance(run, dict) or set(run) != {"id", "attempt"} or not _identifier(run.get("id")):
        blockers.append("RUN_ID_INVALID")
    elif run.get("attempt") != 1:
        blockers.append("RUN_ATTEMPT_INVALID")
    if not _identifier(data.get("dispatch_id")):
        blockers.append("DISPATCH_ID_INVALID")
    assessments, artifact_blockers = _review_artifact_assessments(
        data.get("review_artifact_refs"), target.get("review_artifacts"), target,
    )
    blockers.extend(artifact_blockers)
    blockers.extend(_assessment_blockers(assessments, target, now))
    packet = None
    try:
        packet = build_packet(
            target, assessments, data.get("review_artifact_refs"),
            producer=producer, run=run, dispatch_id=data.get("dispatch_id"),
        )
        blockers.extend(_packet_blockers(packet, target, now))
    except (TypeError, ValueError):
        blockers.append("PACKET_INVALID")
    return {
        "decision": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "packet": packet if not blockers else None,
    }


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
    blockers.extend(
        _assessment_blockers(packet.get("assessments"), live, now, check_freshness=False)
    )
    assessments, artifact_blockers = _review_artifact_assessments(
        packet.get("review_artifact_refs"), live.get("review_artifacts"), live,
    )
    blockers.extend(artifact_blockers)
    if _canonical(packet.get("assessments")) != _canonical(assessments):
        blockers.append("PACKET_REVIEW_ARTIFACT_MISMATCH")
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
        packet = json.loads(summary, object_pairs_hook=_unique_object)
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
    if data.get("mode") == "dispatch_envelope":
        return _dispatch_envelope_result(data, current)
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
    review_artifact_refs = data.get("review_artifact_refs")
    assessments, artifact_blockers = _review_artifact_assessments(
        review_artifact_refs, target.get("review_artifacts"), target,
    )
    blockers.extend(artifact_blockers)
    blockers.extend(_assessment_blockers(assessments, target, current))
    packet = None
    try:
        packet = build_packet(
            target, assessments, review_artifact_refs,
            producer=producer, run=run, dispatch_id=dispatch_id,
        )
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
    if not isinstance(data, dict):
        return {"decision": "FAIL", "blockers": ["INPUT_INVALID"]}
    blockers = ["LEGACY_REVIEW_FORMAT_REJECTED"]
    receipts = data.get("receipts")
    if isinstance(receipts, list):
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            receipt_id = receipt.get("receipt_id")
            if isinstance(receipt_id, str) and receipt_id in WITHDRAWN_RECEIPT_IDS:
                lane = receipt.get("lane")
                prefix = lane.upper() if lane in {"acceptance", "adversarial"} else "LEGACY"
                blockers.append(f"{prefix}_RECEIPT_WITHDRAWN")
    return {
        "decision": "FAIL",
        "repository": data.get("repository"),
        "pr_number": data.get("pr_number"),
        "base_sha": data.get("base_sha"),
        "head_sha": data.get("head_sha"),
        "policy_version": POLICY_VERSION,
        "blockers": sorted(set(blockers)),
    }


def main() -> int:
    try:
        result = evaluate(json.load(sys.stdin, object_pairs_hook=_unique_object))
    except (json.JSONDecodeError, TypeError, ValueError):
        result = {"decision": "FAIL", "blockers": ["INPUT_INVALID"]}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
