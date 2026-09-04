from __future__ import annotations

import json
import os
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.ai_review_gate import (
    build_packet,
    evaluate,
    evaluate_reconciliation,
    state_fingerprint,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
HEAD = "1" * 40
BASE = "2" * 40
DIGEST = "3" * 64
REPO_ROOT = Path(__file__).resolve().parent.parent


def _workflow_under_test() -> Path:
    return Path(
        os.environ.get(
            "AI_REVIEW_GATE_WORKFLOW",
            REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml",
        )
    )


def _extract_reconcile_script(workflow_path: Path) -> str:
    lines = workflow_path.read_text(encoding="utf-8").splitlines()
    anchors = [index for index, line in enumerate(lines) if "const runValidator" in line]
    assert len(anchors) == 1
    anchor = anchors[0]
    script_line = max(
        index for index in range(anchor) if lines[index].strip() == "script: |"
    )
    script_indent = len(lines[script_line]) - len(lines[script_line].lstrip())
    content_indent = next(
        len(line) - len(line.lstrip())
        for line in lines[script_line + 1 :]
        if line.strip()
    )
    block = []
    for line in lines[script_line + 1 :]:
        indent = len(line) - len(line.lstrip())
        if line.strip() and indent <= script_indent:
            break
        block.append(line[content_indent:] if line.strip() else "")
    script = "\n".join(block)
    assert "const runValidator" in script
    return script


def _run_workflow_scenario(
    name: str, workflow_path: Path | None = None
) -> dict[str, object]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute ai-review-gate behavioral tests")

    scenarios = {
        "s1": {"dispatch": True, "prs": [1, 2, 3], "target": 1},
        "s2": {"dispatch": True, "prs": [1, 2, 3], "target": 1},
        "s3": {"dispatch": False, "prs": [1, 2, 3], "invalidPr": 2},
        "s4": {"dispatch": False, "prs": [1, 2, 3], "throwPr": 2},
        "s5": {"dispatch": True, "prs": [1], "target": 1},
        "s6": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s7": {"dispatch": True, "prs": [1, 2, 3], "target": 1},
        "s8": {"dispatch": True, "prs": [1, 2, 3], "target": 1},
        "s8b": {"dispatch": False, "prs": [1, 2, 3]},
        "s9": {"dispatch": True, "prs": [1, 2, 3], "target": 1},
        "s10": {"dispatch": True, "prs": [1, 2], "target": 1},
    }
    config = {"name": name, **scenarios[name]}
    script = _extract_reconcile_script(workflow_path or _workflow_under_test())
    validator_start = script.index("const runValidator = async")
    validator_end = script.index("const snapshot = async", validator_start)
    script = (
        script[:validator_start]
        + "const runValidator = async input => fakeRunValidator(input);\n"
        + script[validator_end:]
    )
    driver = f"""
const cfg = {json.dumps(config)};
const emissions = [], failures = [], checks = [];
const pullCalls = new Map();
let branchCalls = 0;
let nextCheckId = 1;
const original = number => ({{number, base: {{ref: 'main', sha: `base-${{number}}`}},
  head: {{sha: `head-${{number}}`}}, state: 'open', draft: false}});
const record = (run, conclusion, summary, operation) => {{
  const match = run.external_id.match(/^ai-review-gate:(\\d+):([^:]+):(.+)$/);
  emissions.push({{pr: Number(match[1]), base: match[2], head: match[3], conclusion,
    summary, operation}});
}};
const checksApi = {{
  listForRef: async ({{ref}}) => checks.filter(run => run.head_sha === ref),
  create: async payload => {{
    const run = {{...payload, id: nextCheckId++, app: {{id: 15368}}}};
    checks.push(run); record(run, payload.conclusion, payload.output.summary, 'create');
    return {{data: run}};
  }},
  update: async payload => {{
    if (cfg.name === 's10' && payload.conclusion === 'success')
      throw Error('synthetic success write failure');
    const run = checks.find(item => item.id === payload.check_run_id);
    record(run, payload.conclusion, payload.output.summary, 'update');
    Object.assign(run, payload); return {{data: run}};
  }},
}};
const pullsApi = {{
  list: async () => cfg.prs.map(original),
  get: async ({{pull_number}}) => {{
    const count = (pullCalls.get(pull_number) || 0) + 1;
    pullCalls.set(pull_number, count);
    if (cfg.throwPr === pull_number && count === 2) throw Error('synthetic final-read failure');
    if (cfg.name === 's9' && pull_number === cfg.target && count === 3)
      throw Error('synthetic target re-snapshot failure');
    const data = original(pull_number);
    if ((cfg.name === 's5' || cfg.name === 's7') && pull_number === cfg.target && count >= 4)
      data.head.sha = `head-${{pull_number}}-live`;
    return {{data}};
  }},
  listFiles: async ({{pull_number}}) => [{{filename: `file-${{pull_number}}`}}],
}};
const github = {{
  rest: {{
    checks: checksApi,
    pulls: pullsApi,
    issues: {{listEventsForTimeline: async () => []}},
    repos: {{
      get: async () => ({{data: {{default_branch: 'main'}}}}),
      getBranch: async () => {{
        branchCalls += 1;
        if (cfg.name === 's7' && branchCalls === 2)
          throw Error('synthetic live follow-up failure');
        return {{data: {{commit: {{sha: 'protected'}}}}}};
      }},
      getContent: async () => {{ throw Error('real validator must not run'); }},
    }},
  }},
  paginate: async (fn, args) => fn(args),
  graphql: async () => ({{repository: {{pullRequest: {{reviewThreads: {{nodes: [],
    pageInfo: {{hasNextPage: false, endCursor: null}}}}}}}}}}),
}};
const context = {{repo: {{owner: 'electricsheephq', repo: 'lcm-x'}},
  eventName: cfg.dispatch ? 'repository_dispatch' : 'push',
  payload: cfg.dispatch ? {{client_payload: {{pr_number: cfg.target, receipts: [],
    dispatch_id: 'dispatch-fresh'}}, sender: {{login: '100yenadmin', id: 239388517,
    type: 'User'}}}} : {{}}, actor: '100yenadmin', ref: 'refs/heads/main', sha: 'protected'}};
const core = {{setFailed: message => failures.push(message)}};
Object.assign(process.env, {{GITHUB_RUN_ATTEMPT: '1', GITHUB_ACTOR_ID: '239388517',
  GITHUB_RUN_ID: 'run-1'}});
const peerResult = (peer, preserve) => ({{pr_number: peer.pr_number, preserve}});
async function fakeRunValidator(input) {{
  if (input.mode === 'dispatch_envelope')
    return {{run: {{status: 0}}, result: {{decision: 'PASS'}}}};
  if (cfg.name === 's1' && input.target)
    return {{run: {{status: 1}}, result: {{decision: 'FAIL',
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (cfg.name === 's2' && input.target)
    return {{run: {{status: 1}}, result: {{decision: 'FAIL', peers: []}}}};
  if (cfg.name === 's3' && input.peers.length === cfg.prs.length)
    return {{run: {{status: 1}}, result: {{decision: 'FAIL',
      peers: input.peers.map(peer => peerResult(peer, peer.pr_number !== cfg.invalidPr))}}}};
  if (cfg.name === 's4' && input.peers.length === cfg.prs.length)
    return {{run: {{status: 0}}, result: {{decision: 'PASS',
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (cfg.name === 's8' && input.target)
    return {{run: {{status: 1}}, result: {{decision: 'PASS', packet: {{scenario: cfg.name}},
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (cfg.name === 's8b' && !input.target && input.peers.length === cfg.prs.length)
    return {{run: {{status: 1}}, result: {{decision: 'PASS',
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (input.target)
    return {{run: {{status: 0}}, result: {{decision: 'PASS', packet: {{scenario: cfg.name}},
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  return {{run: {{status: 0}}, result: {{decision: 'PASS',
    peers: input.peers.map(peer => peerResult(peer, peer.pr_number !== cfg.invalidPr))}}}};
}}
async function main() {{
{script}
}}
main().then(() => console.log(JSON.stringify({{emissions, failures}}))).catch(error => {{
  console.error(error.stack); process.exitCode = 1;
}});
"""
    run = subprocess.run([node, "-e", driver], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _failure_tuples(result: dict[str, object]) -> set[tuple[int, str, str]]:
    return {
        (emit["pr"], emit["base"], emit["head"])
        for emit in result["emissions"]
        if emit["conclusion"] == "failure"
    }


def test_behavioral_s1_clean_dispatch_fail_resets_only_target():
    result = _run_workflow_scenario("s1")
    assert _failure_tuples(result) == {(1, "base-1", "head-1")}


def test_behavioral_s2_incomplete_dispatch_validation_resets_every_snapshot():
    result = _run_workflow_scenario("s2")
    assert _failure_tuples(result) == {
        (number, f"base-{number}", f"head-{number}") for number in (1, 2, 3)
    }


def test_behavioral_s3_clean_peer_only_fail_resets_only_invalid_peer():
    result = _run_workflow_scenario("s3")
    assert _failure_tuples(result) == {(2, "base-2", "head-2")}


def test_behavioral_s4_final_read_failure_resets_every_snapshot():
    result = _run_workflow_scenario("s4")
    assert _failure_tuples(result) == {
        (number, f"base-{number}", f"head-{number}") for number in (1, 2, 3)
    }


def test_behavioral_s5_drift_fails_original_and_live_tuples():
    result = _run_workflow_scenario("s5")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (1, "base-1", "head-1-live"),
    }


def test_behavioral_s6_consistent_dispatch_succeeds_only_target():
    result = _run_workflow_scenario("s6")
    successes = [
        emit for emit in result["emissions"] if emit["conclusion"] == "success"
    ]
    assert [(emit["pr"], emit["base"], emit["head"]) for emit in successes] == [
        (1, "base-1", "head-1")
    ]
    assert all(emit["pr"] == 1 for emit in result["emissions"])
    assert result["failures"] == []


def test_behavioral_s7_live_tuple_is_failed_when_follow_up_read_throws():
    result = _run_workflow_scenario("s7")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (1, "base-1", "head-1-live"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s8_pass_payload_with_nonzero_validator_fails_closed():
    result = _run_workflow_scenario("s8")
    assert _failure_tuples(result) == {
        (number, f"base-{number}", f"head-{number}") for number in (1, 2, 3)
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s8b_peer_only_pass_payload_with_nonzero_validator_fails_closed():
    result = _run_workflow_scenario("s8b")
    assert _failure_tuples(result) == {
        (number, f"base-{number}", f"head-{number}") for number in (1, 2, 3)
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s9_target_resnapshot_failure_resets_every_snapshot():
    # The target re-snapshot is an API call between reconciliation and the
    # validator; if it rejects, no peer was reconciled, so all known
    # snapshots reset (not just the dispatch target).
    result = _run_workflow_scenario("s9")
    assert _failure_tuples(result) == {
        (number, f"base-{number}", f"head-{number}") for number in (1, 2, 3)
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s10_success_write_failure_fails_the_job():
    # A success verdict whose check write is rejected must not leave the job
    # green: the trusted check stays at the pre-validation failure, and the
    # step reports the swallowed write failure.
    result = _run_workflow_scenario("s10")
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])
    assert len(result["failures"]) == 1
    assert "failure writes: synthetic success write failure" in result["failures"][0]


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
        "receipts": [receipt("acceptance"), receipt("adversarial")],
        "unresolved_threads": 0,
        "api_complete": True,
        "pagination_complete": True,
    }


def test_routine_distinct_acceptance_and_adversarial_receipts_pass():
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
    data["receipts"] = [receipt("acceptance", risk="unknown")]
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
    for field, value in (("api_complete", False), ("pagination_complete", False),
                         ("unresolved_threads", 1), ("pr_number", True)):
        data = payload()
        data[field] = value
        assert evaluate(data, NOW)["decision"] == "FAIL"


def test_nested_receipt_pr_number_requires_exact_integer_type():
    for invalid_pr_number in (True, 1.0):
        receipts = [receipt("acceptance"), receipt("adversarial")]
        for item in receipts:
            item["pr_number"] = 1
        receipts[0]["pr_number"] = invalid_pr_number

        result = evaluate_reconciliation(
            _v2_dispatch(_v2_snapshot(1), receipts_override=receipts), NOW
        )

        assert result["decision"] == "FAIL"
        assert "ACCEPTANCE_PR_NUMBER_INVALID" in result["blockers"]


def test_dispatch_envelope_preflight_rejects_malformed_receipts():
    envelope = {
        "schema_version": "2",
        "mode": "dispatch_envelope",
        "target": {
            "repository": "electricsheephq/lcm-x",
            "pr_number": 350,
            "base_sha": BASE,
            "head_sha": HEAD,
        },
        "receipts": [
            receipt("acceptance", risk="governance"),
            receipt("adversarial", risk="governance"),
        ],
        "dispatch_id": "dispatch-envelope-fresh",
    }

    assert evaluate_reconciliation(envelope, NOW)["decision"] == "PASS"

    for malformed in ([], [{}]):
        candidate = deepcopy(envelope)
        candidate["receipts"] = malformed
        result = evaluate_reconciliation(candidate, NOW)
        assert result["decision"] == "FAIL"
        assert "RECEIPT_SET_INVALID" in result["blockers"]


def test_receipt_integration_id_requires_exact_integer_type():
    for invalid_integration_id in (True, 15368.0):
        legacy = payload()
        legacy["receipts"][0]["integration_id"] = invalid_integration_id
        assert "ACCEPTANCE_INTEGRATION_ID_INVALID" in evaluate(legacy, NOW)[
            "blockers"
        ]

        envelope = {
            "schema_version": "2",
            "mode": "dispatch_envelope",
            "target": {
                "repository": "electricsheephq/lcm-x",
                "pr_number": 350,
                "base_sha": BASE,
                "head_sha": HEAD,
            },
            "receipts": [receipt("acceptance"), receipt("adversarial")],
            "dispatch_id": "dispatch-envelope-fresh",
        }
        envelope["receipts"][0]["integration_id"] = invalid_integration_id
        assert "ACCEPTANCE_INTEGRATION_ID_INVALID" in evaluate_reconciliation(
            envelope, NOW
        )["blockers"]


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


def test_every_risk_class_requires_both_distinct_lanes():
    for path, risk in (
        (["docs/operator-guide.md"], "routine"),
        ([".github/workflows/ai-review-gate.yml"], "governance"),
        (["access_context/tools.py"], "unknown"),
    ):
        data = payload(path)
        data["receipts"] = [
            receipt("acceptance", risk=risk),
            receipt("adversarial", risk=risk),
        ]
        assert evaluate(data, NOW)["decision"] == "PASS"


def test_labels_are_organizational_metadata_only():
    data = payload()
    baseline = evaluate(data, NOW)
    data["labels"] = ["routine", "security"]
    assert evaluate(data, NOW) == baseline


def test_duplicate_or_conflicting_receipts_fail_closed_for_each_identity():
    data = payload(["AGENTS.md"])
    for key in ("reviewer_id", "task_id", "receipt_id"):
        candidate = deepcopy(data)
        candidate["receipts"][1][key] = candidate["receipts"][0][key]
        assert f"DUPLICATE_{key.upper()}" in evaluate(candidate, NOW)["blockers"]


def test_workflow_is_base_trusted_and_resets_each_head():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    assert "pull_request_target:" in workflow
    assert "push:" in workflow
    assert "repository_dispatch:" in workflow
    assert "types: [ai-review-receipts]" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "checks: write" in workflow
    assert "AI review exact-head" in workflow
    assert "head_sha: head" in workflow
    assert "${prNumber}:${base}:${head}" in workflow
    assert "filter: 'all'" in workflow
    assert "Protected base changed" in workflow or "stored packet invalid" in workflow
    assert "const snapshots = [], failures = []" in workflow
    assert "core.setFailed(`Failed to reset PRs:" in workflow
    assert "context.ref !== `refs/heads/${defaultBranch}`" in workflow
    assert "ref: protectedSha" in workflow
    assert "context.sha !== protectedSha" in workflow
    assert "pull_request_target:" in workflow
    assert "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, edited]" in workflow
    assert "labeled" not in workflow
    assert "unlabeled" not in workflow
    assert "concurrency:" in workflow
    assert "group: ai-review-gate-global" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "github.run_id" not in workflow
    assert "queue:" not in workflow
    assert "actions/checkout" not in workflow
    assert "pull_request.head" not in workflow
    assert "eval(" not in workflow


def test_workflow_reconciles_all_open_prs_before_dispatch_evaluation():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    reconcile = workflow.index("const reconcileOpenPullRequests")
    dispatch_guard = workflow.index("const dispatchEvent = context.eventName === 'repository_dispatch';")
    assert reconcile < dispatch_guard
    assert "github.paginate(github.rest.pulls.list" in workflow
    assert "github.rest.pulls.get" in workflow
    assert "base: defaultBranch" in workflow
    assert "const finalReadInvalidPeers = async" in workflow
    assert "await emit(original.pr_number, original.base_sha, original.head_sha, 'failure'" in workflow
    assert "await emit(currentTuple.pr_number, currentTuple.base_sha, currentTuple.head_sha, 'failure'" in workflow
    assert "if (failures.length) core.setFailed" in workflow
    assert "return;" in workflow[dispatch_guard:]


def test_incomplete_snapshot_preserves_the_listed_candidate_tuple():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    reconcile = workflow.index("const reconcileOpenPullRequests")
    failure_snapshot = workflow.index(
        "snapshots.push({repository: `${owner}/${repo}`, pr_number: candidate.number,",
        reconcile,
    )
    helper = workflow.index("const failKnownSnapshots = async", failure_snapshot)
    failure_section = workflow[failure_snapshot:helper]

    assert "base_ref: candidate.base?.ref" in failure_section
    assert "base_sha: candidate.base?.sha" in failure_section
    assert "head_sha: candidate.head?.sha" in failure_section
    assert "state: candidate.state" in failure_section
    assert "draft: candidate.draft" in failure_section


def test_dispatch_reconciliation_failure_resets_every_known_snapshot():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    dispatch = workflow.index("if (dispatchEvent) {")
    reconciliation_guard = workflow.index("if (reconciled.failures.length)", dispatch)
    peer_resets = workflow.index(
        "await failKnownSnapshots(snapshots", reconciliation_guard
    )
    terminal = workflow.index(
        "throw Error(`open PR reconciliation incomplete", reconciliation_guard
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch);", reconciliation_guard
    )

    assert reconciliation_guard < peer_resets < terminal < target_snapshot
    assert "const writeFailures =" in workflow[reconciliation_guard:peer_resets]
    assert "failure writes:" in workflow[peer_resets:terminal]


def test_incomplete_dispatch_target_still_resets_known_peers():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    dispatch = workflow.index("if (dispatchEvent) {")
    authenticated = workflow.index("if (sender?.login !== '100yenadmin'", dispatch)
    packet_guard = workflow.index("if (!Array.isArray(dispatch.receipts)", dispatch)
    protected_ref = workflow.index("if (context.ref !== `refs/heads/${defaultBranch}`", dispatch)
    reconciliation_guard = workflow.index("if (reconciled.failures.length)", dispatch)
    peer_resets = workflow.index(
        "await failKnownSnapshots(snapshots", reconciliation_guard
    )
    terminal = workflow.index(
        "throw Error(`open PR reconciliation incomplete", peer_resets
    )
    target_guard = workflow.index("if (!priorTarget || priorTarget.api_complete !== true", dispatch)

    assert (
        authenticated
        < packet_guard
        < protected_ref
        < reconciliation_guard
        < peer_resets
        < terminal
        < target_guard
    )


def test_dispatch_envelope_preflight_precedes_peer_resets():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    dispatch = workflow.index("if (dispatchEvent) {")
    authenticated = workflow.index("if (sender?.login !== '100yenadmin'", dispatch)
    preflight = workflow.index("mode: 'dispatch_envelope'", authenticated)
    preflight_stop = workflow.index("throw Error('dispatch packet invalid')", preflight)
    reconciliation_guard = workflow.index("if (reconciled.failures.length)", preflight_stop)
    peer_resets = workflow.index("await failKnownSnapshots(snapshots", reconciliation_guard)

    assert authenticated < preflight < preflight_stop < reconciliation_guard < peer_resets
    assert "envelopeValidated.run.status !== 0" in workflow[preflight:preflight_stop]
    assert "envelopeValidated.result.decision !== 'PASS'" in workflow[
        preflight:preflight_stop
    ]


def test_workflow_rechecks_complete_target_state_before_success():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("github.paginate(github.rest.pulls.listFiles") == 1
    assert workflow.count("listFiles(prNumber)") >= 2
    assert workflow.count("reviewThreads(first:100") == 1
    assert workflow.count("unresolvedThreads(prNumber)") >= 2
    assert "live.data.head.sha !== head" in workflow
    assert "live.data.base.sha !== base" in workflow
    assert "liveBranch.data.commit.sha !== protectedSha" in workflow
    assert "context.ref !== `refs/heads/${defaultBranch}`" in workflow
    assert "JSON.stringify(liveFiles) !== JSON.stringify(target.changed_paths)" in workflow
    assert "target.unresolved_threads !== 0" in workflow
    assert "matches.length > 1" in workflow
    assert "DUPLICATE_TRUSTED_CHECK" in workflow


def test_workflow_reconciliation_failure_is_terminal_before_target_promotion():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    peer_resets = workflow.index("await failKnownSnapshots(snapshots")
    reconciliation_stop = workflow.index(
        "throw Error(`open PR reconciliation incomplete"
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch);"
    )
    target_success = workflow.index("conclusion = 'success';")

    assert peer_resets < reconciliation_stop < target_snapshot < target_success


def test_workflow_uses_supported_actor_id_environment():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    assert "process.env.GITHUB_ACTOR_ID" in workflow
    assert "context.actor_id" not in workflow


def test_workflow_captures_exact_target_dispatch_id_before_reset():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    prior_target = workflow.index(
        "const priorTarget = snapshots.find(item => item.pr_number === prNumber);"
    )
    target_guard = workflow.index(
        "if (!priorTarget || priorTarget.api_complete !== true ||"
    )
    expected_id = workflow.index(
        "const expectedExternalId = `ai-review-gate:${prNumber}:${base}:${head}`;"
    )
    exact_checks = workflow.index("check.app_id === 15368", expected_id)
    prior_validation = workflow.index(
        "const priorValidated = await runValidator({schema_version: '2', mode: 'peer_only', peers: [priorTarget]}, protectedSha);"
    )
    target_reset = workflow.index(
        "await emit(prNumber, base, head, 'failure', 'Protected base changed"
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch);"
    )

    assert (
        prior_target
        < target_guard
        < expected_id
        < exact_checks
        < prior_validation
        < target_reset
        < target_snapshot
    )
    assert "priorTarget.base_sha !== base || priorTarget.head_sha !== head" in workflow[
        target_guard:expected_id
    ]
    assert "check.external_id === expectedExternalId" in workflow[exact_checks:target_reset]
    assert "check.head_sha === head" in workflow[exact_checks:target_reset]
    assert "check.status === 'completed'" in workflow[exact_checks:target_reset]
    assert "check.conclusion === 'success'" in workflow[exact_checks:target_reset]
    assert "priorValidated.result.peers[0].preserve" in workflow[
        prior_validation:target_reset
    ]
    assert "prior_dispatch_ids: priorDispatchIds" in workflow[target_reset:]


def test_workflow_invalid_prior_target_does_not_abort_fresh_renewal():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    prior_ids = workflow.index("const priorDispatchIds = [];")
    prior_validation = workflow.index(
        "const priorValidated = await runValidator({schema_version: '2', "
        "mode: 'peer_only', peers: [priorTarget]}, protectedSha);"
    )
    optional_history = workflow.index(
        "if (priorValidated.run.status === 0 && priorValidated.result.decision === 'PASS'",
        prior_validation,
    )
    target_reset = workflow.index(
        "await emit(prNumber, base, head, 'failure', 'Protected base changed",
        optional_history,
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch);",
        target_reset,
    )

    assert prior_ids < prior_validation < optional_history < target_reset < target_snapshot
    assert "throw Error('prior target packet invalid');" not in workflow[
        prior_validation:target_reset
    ]
    assert "priorValidated.result.peers?.length === 1" in workflow[
        optional_history:target_reset
    ]
    assert "priorValidated.result.peers[0].preserve" in workflow[
        optional_history:target_reset
    ]
    assert "priorDispatchIds.push(packet.dispatch_id);" in workflow[
        optional_history:target_reset
    ]


def test_workflow_final_reads_every_preserved_peer_before_exit():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    helper = workflow.index("const finalReadPreservedPeers = async")
    dispatch = workflow.index("if (dispatchEvent) {")
    target_final_read = workflow.index(
        "const live = await github.rest.pulls.get({owner, repo, pull_number: prNumber});",
        dispatch,
    )
    peer_only = workflow.index("} else {", target_final_read)
    peer_only_return = workflow.index("return;", peer_only)

    assert helper < dispatch
    assert "if (!peer.preserve) continue;" in workflow[helper:dispatch]
    assert "const current = await snapshot(peer.pr_number, defaultBranch);" in workflow[
        helper:dispatch
    ]
    assert (
        "const finalValidated = await runValidator({schema_version: '2', "
        "mode: 'peer_only', peers: [current]}, protectedSha);"
    ) in workflow[helper:dispatch]
    assert "await emit(original.pr_number, original.base_sha, original.head_sha, 'failure'" in workflow[
        helper:dispatch
    ]
    assert "await emit(currentTuple.pr_number, currentTuple.base_sha, currentTuple.head_sha, 'failure'" in workflow[
        helper:dispatch
    ]
    dispatch_final_read = workflow.index(
        "await finalReadPreservedPeers(result.peers || [], snapshots, defaultBranch, protectedSha);",
        dispatch,
    )
    peer_only_final_read = workflow.index(
        "await finalReadPreservedPeers(result.peers || [], snapshots, defaultBranch, protectedSha);",
        peer_only,
    )
    assert dispatch_final_read < target_final_read
    assert peer_only_final_read < peer_only_return


def test_workflow_rechecks_invalid_peers_before_failure_write():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    helper = workflow.index("const finalReadInvalidPeers = async")
    dispatch = workflow.index("if (dispatchEvent) {")
    assert helper < dispatch
    assert "if (peer.preserve) continue;" in workflow[helper:dispatch]
    assert "const current = await snapshot(peer.pr_number, defaultBranch);" in workflow[
        helper:dispatch
    ]
    assert (
        "const currentValidated = await runValidator({schema_version: '2', "
        "mode: 'peer_only', peers: [current]}, protectedSha);"
    ) in workflow[helper:dispatch]
    assert "if (currentPeer.preserve) continue;" in workflow[helper:dispatch]
    assert "await emit(original.pr_number, original.base_sha, original.head_sha, 'failure'" in workflow[
        helper:dispatch
    ]
    assert "await emit(currentTuple.pr_number, currentTuple.base_sha, currentTuple.head_sha, 'failure'" in workflow[
        helper:dispatch
    ]

    invalid_call = (
        "await finalReadInvalidPeers(result.peers || [], snapshots, defaultBranch, "
        "protectedSha);"
    )
    preserve_call = (
        "await finalReadPreservedPeers(result.peers || [], snapshots, defaultBranch, "
        "protectedSha);"
    )
    dispatch_invalid = workflow.index(invalid_call, dispatch)
    dispatch_preserve = workflow.index(preserve_call, dispatch)
    peer_only = workflow.index("} else {", dispatch_preserve)
    peer_invalid = workflow.index(invalid_call, peer_only)
    peer_preserve = workflow.index(preserve_call, peer_only)
    assert dispatch_invalid < dispatch_preserve
    assert peer_invalid < peer_preserve


def test_workflow_dispatch_validation_failure_is_terminal_before_target_promotion():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    input_packet = workflow.index("const input = {schema_version: '2', target")
    # The fail-closed try opens BEFORE the pre-validation reset and the target
    # re-snapshot: an API rejection of either leaves every peer un-reconciled
    # and must reach the reset-all catch, not the target-only outer catch.
    wrapping = workflow.rindex("try {", 0, input_packet)
    target_reset = workflow.index(
        "await emit(prNumber, base, head, 'failure', 'Protected base changed", wrapping
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch);", target_reset
    )
    validator = workflow.index(
        "const validated = await runValidator(input, protectedSha);", input_packet
    )
    invalid_read = workflow.index("await finalReadInvalidPeers(result.peers", validator)
    preserved_read = workflow.index(
        "await finalReadPreservedPeers(result.peers", invalid_read
    )
    failure_catch = workflow.index("} catch (error) {", preserved_read)
    peer_resets = workflow.index("await failKnownSnapshots(snapshots", failure_catch)
    terminal = workflow.index(
        "throw Error(`peer reconciliation validation failed:", peer_resets
    )
    target_promotion = workflow.index("conclusion = 'success';", terminal)

    assert (
        wrapping
        < target_reset
        < target_snapshot
        < input_packet
        < validator
        < invalid_read
        < preserved_read
        < failure_catch
        < peer_resets
        < terminal
        < target_promotion
    )
    assert "Peer reconciliation validation failed closed:" in workflow[
        failure_catch:terminal
    ]
    assert "failure writes:" in workflow[peer_resets:terminal]


def _v2_snapshot(
    pr_number: int,
    *,
    complete: bool = True,
    changed_paths: list[str] | None = None,
):
    base, head = BASE, HEAD
    live = {
        "repository": "electricsheephq/lcm-x",
        "pr_number": pr_number,
        "base_ref": "main",
        "base_sha": base,
        "head_sha": head,
        "changed_paths": (
            ["docs/operator-guide.md"] if changed_paths is None else changed_paths
        ),
        "timeline_events": [],
        "unresolved_threads": 0,
        "api_complete": complete,
        "pagination_complete": complete,
        "state": "open",
        "draft": False,
    }
    risk = "unknown" if changed_paths == [] else "routine"
    receipts = [receipt("acceptance", risk=risk), receipt("adversarial", risk=risk)]
    for item in receipts:
        item["pr_number"] = pr_number
    packet = build_packet(live, receipts, producer={"login": "100yenadmin", "id": 239388517, "type": "User"},
                          run={"id": f"run-{pr_number}", "attempt": 1}, dispatch_id=f"dispatch-{pr_number}")
    check = {"name": "AI review exact-head", "app_id": 15368,
             "external_id": f"ai-review-gate:{pr_number}:{base}:{head}", "head_sha": head,
             "status": "completed", "conclusion": "success",
             "output_summary": json.dumps(packet, sort_keys=True, separators=(",", ":"))}
    return {**live, "check_runs": [check]}


def _v2_dispatch(target: dict[str, object], *, receipts_override=None, **extra):
    data = {key: target[key] for key in (
        "repository", "pr_number", "base_sha", "head_sha", "changed_paths",
        "timeline_events", "unresolved_threads", "api_complete", "pagination_complete",
    )}
    data.update({
        "schema_version": "2", "target": target,
        "receipts": receipts_override
        or [receipt("acceptance"), receipt("adversarial")],
        "producer": {"login": "100yenadmin", "id": 239388517, "type": "User"},
        "run": {"id": "run-target", "attempt": 1},
        "dispatch_id": "dispatch-target-fresh",
        "peers": [],
        **extra,
    })
    return data


def test_red_base_cannot_reconstruct_a_stored_peer_packet():
    peer = _v2_snapshot(351)
    result = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[peer]), NOW
    )
    assert result["decision"] == "PASS"
    assert result["peers"][0]["preserve"] is True
    packet_for_peer = json.loads(peer["check_runs"][0]["output_summary"])
    assert packet_for_peer["state_fingerprint"] == state_fingerprint(peer)


def test_dispatch_id_reuse_reads_only_validated_exact_peer_check():
    peer = _v2_snapshot(351)
    stale = deepcopy(peer["check_runs"][0])
    stale["external_id"] = f"ai-review-gate:351:{'4' * 40}:{HEAD}"
    stale_packet = json.loads(stale["output_summary"])
    stale_packet["dispatch_id"] = "dispatch-stale"
    stale["output_summary"] = json.dumps(stale_packet)
    peer["check_runs"].insert(0, stale)

    result = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[peer], dispatch_id="dispatch-351"), NOW
    )

    assert result["decision"] == "FAIL"
    assert "DISPATCH_ID_NOT_FRESH" in result["blockers"]
    assert result["peers"][0]["preserve"] is True


def test_invalid_peer_fails_only_that_peer_but_incomplete_snapshot_blocks_target():
    for field, value in (("draft", True), ("unresolved_threads", 1)):
        peer = _v2_snapshot(351)
        peer[field] = value

        result = evaluate_reconciliation(
            _v2_dispatch(_v2_snapshot(350), peers=[peer]), NOW
        )

        assert result["decision"] == "PASS"
        assert result["peers"][0]["preserve"] is False
        assert "PEER_SNAPSHOT_INCOMPLETE" not in result["blockers"]

    incomplete = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[_v2_snapshot(351, complete=False)]),
        NOW,
    )
    assert incomplete["decision"] == "FAIL"
    assert "PEER_SNAPSHOT_INCOMPLETE" in incomplete["blockers"]


def test_complete_zero_file_peer_is_not_a_repository_snapshot_failure():
    peer = _v2_snapshot(351, changed_paths=[])

    result = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[peer]), NOW
    )

    assert result["decision"] == "PASS"
    assert result["peers"][0]["preserve"] is True
    assert result["peers"][0]["snapshot_error"] is False
    assert "PEER_SNAPSHOT_INCOMPLETE" not in result["blockers"]


def test_preserved_peer_does_not_overwrite_fresh_target_packet():
    result = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[_v2_snapshot(351)]), NOW
    )

    assert result["decision"] == "PASS"
    assert result["packet"]["pr_number"] == 350
    assert result["packet"]["dispatch_id"] == "dispatch-target-fresh"
    assert result["peers"][0]["preserve"] is True


def test_red_base_accepts_dispatch_without_authenticated_producer_or_attempt_guard():
    target = _v2_snapshot(350)
    data = _v2_dispatch(target, producer={"login": "100yenadmin", "id": 239388517})
    data["run"] = {"id": "rerun", "attempt": 2}
    result = evaluate_reconciliation(data)
    assert result["decision"] == "FAIL"
    assert "PRODUCER_UNAUTHENTICATED" in result["blockers"]
    assert "RUN_ATTEMPT_INVALID" in result["blockers"]


def test_workflow_identifies_dispatch_target_before_peer_enumeration():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    target_number = workflow.index("prNumber = dispatch.pr_number;")
    target_tuple = workflow.index("base = tuple.base.sha; head = tuple.head.sha;")
    enumeration = workflow.index(
        "const reconciled = await reconcileOpenPullRequests(defaultBranch);"
    )

    # A rejection of the top-level open-PR enumeration must still reach the
    # fail-closed emit for an event-identified dispatch target: the target's
    # number and tuple are captured before reconciliation can abort.
    assert target_number < target_tuple < enumeration
    assert "if (prNumber && base && head) {" in workflow
    assert "if (prNumber && liveTuple &&" in workflow
