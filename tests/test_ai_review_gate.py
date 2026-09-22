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
    _review_artifact_assessments,
    build_packet,
    evaluate,
    evaluate_reconciliation,
    state_fingerprint,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
HEAD = "1" * 40
BASE = "2" * 40
REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEWERS = {"acceptance": (298367747, "evaos-code-review-bot[bot]", 9001),
             "adversarial": (199175422, "chatgpt-codex-connector[bot]", 9002)}


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
        "s11": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s12": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s13": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s14": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s14b": {"dispatch": False, "prs": [1, 2]},
        "s15": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s16": {"dispatch": True, "prs": [1, 2], "target": 1},
        "s16b": {"dispatch": True, "prs": [1, 2], "target": 1, "invalidPr": 2},
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
if (cfg.name === 's12')
  // A trusted exact success check whose packet consumed 'dispatch-fresh' but
  // whose live-state fingerprint no longer validates (preserve: false below).
  checks.push({{id: 0, name: 'AI review exact-head', external_id: 'ai-review-gate:1:base-1:head-1',
    head_sha: 'head-1', status: 'completed', conclusion: 'success', app: {{id: 15368}},
    output: {{summary: JSON.stringify({{dispatch_id: 'dispatch-fresh'}})}}}});
const pullCalls = new Map();
const filesCalls = new Map();
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
  getReview: async ({{review_id}}) => ({{data: {{id: review_id, state: 'COMMENTED',
    commit_id: `head-${{cfg.target || 1}}`, submitted_at: '2026-08-24T11:00:00Z',
    user: {{id: review_id, login: `reviewer-${{review_id}}`, type: 'Bot'}},
    body: 'synthetic'}}}}),
  get: async ({{pull_number}}) => {{
    const count = (pullCalls.get(pull_number) || 0) + 1;
    pullCalls.set(pull_number, count);
    if (cfg.throwPr === pull_number && count === 2) throw Error('synthetic final-read failure');
    if (cfg.name === 's9' && pull_number === cfg.target && count === 3)
      throw Error('synthetic target re-snapshot failure');
    const data = original(pull_number);
    if (['s11', 's15'].includes(cfg.name) && pull_number === cfg.target && count === 3)
      data.head.sha = `head-${{pull_number}}-live`;
    if ((cfg.name === 's5' || cfg.name === 's7') && pull_number === cfg.target && count >= 4)
      data.head.sha = `head-${{pull_number}}-live`;
    if (['s13', 's14', 's14b'].includes(cfg.name) && pull_number === 2) data.head.sha = 'head-2-live';
    // A peer's SECOND read is its final-read re-snapshot.
    if ((cfg.name === 's16' || cfg.name === 's16b') && pull_number === 2 && count === 2) data.head.sha = 'head-2-live';
    return {{data}};
  }},
  listFiles: async ({{pull_number}}) => {{
    const fcount = (filesCalls.get(pull_number) || 0) + 1;
    filesCalls.set(pull_number, fcount);
    if (cfg.name === 's13' && pull_number === 2) throw Error('synthetic peer files failure');
    // The target's SECOND files read is the one inside the re-snapshot.
    if (cfg.name === 's15' && pull_number === cfg.target && fcount === 2)
      throw Error('synthetic target re-snapshot files failure');
    if ((cfg.name === 's16' || cfg.name === 's16b') && pull_number === 2 && fcount === 2)
      throw Error('synthetic peer final-read files failure');
    if ((cfg.name === 's14' || cfg.name === 's14b') && pull_number === 2) throw null;
    return [{{filename: `file-${{pull_number}}`}}];
  }},
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
  payload: cfg.dispatch ? {{client_payload: {{pr_number: cfg.target,
    review_artifact_refs: [{{lane: 'acceptance', review_id: 9001}},
      {{lane: 'adversarial', review_id: 9002}}],
    dispatch_id: 'dispatch-fresh'}}, sender: {{login: '100yenadmin', id: 239388517,
    type: 'User'}}}} : {{}}, actor: '100yenadmin', ref: 'refs/heads/main', sha: 'protected'}};
const core = {{setFailed: message => failures.push(message)}};
Object.assign(process.env, {{GITHUB_RUN_ATTEMPT: '1', GITHUB_ACTOR_ID: '239388517',
  GITHUB_RUN_ID: 'run-1'}});
const peerResult = (peer, preserve) => ({{pr_number: peer.pr_number, preserve}});
async function fakeRunValidator(input) {{
  if (input.mode === 'dispatch_envelope')
    return {{run: {{status: 0}}, result: {{decision: 'PASS', packet: {{scenario: cfg.name}}}}}};
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
  if (cfg.name === 's11' && input.target)
    return {{run: {{status: 1}}, result: {{decision: 'FAIL',
      peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (cfg.name === 's12' && !input.target && input.peers.length === 1 &&
      input.peers[0].pr_number === cfg.target)
    return {{run: {{status: 0}}, result: {{decision: 'PASS', peers: [peerResult(input.peers[0], false)]}}}};
  if (cfg.name === 's12' && input.target)
    return (input.prior_dispatch_ids || []).includes('dispatch-fresh')
      ? {{run: {{status: 1}}, result: {{decision: 'FAIL',
          peers: input.peers.map(peer => peerResult(peer, true))}}}}
      : {{run: {{status: 0}}, result: {{decision: 'PASS', packet: {{scenario: cfg.name}},
          peers: input.peers.map(peer => peerResult(peer, true))}}}};
  if (cfg.name === 's16b' && input.target)
    return {{run: {{status: 0}}, result: {{decision: 'PASS', packet: {{scenario: cfg.name}},
      peers: input.peers.map(peer => peerResult(peer, peer.pr_number !== cfg.invalidPr))}}}};
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


def test_behavioral_s11_resnapshot_drifted_tuple_is_failed_on_rejection():
    # The re-snapshot returns a force-pushed head; the validator rejects the
    # old-tuple receipts before the live follow-up read runs. The discovered
    # drifted tuple must fail closed alongside the dispatch tuple.
    result = _run_workflow_scenario("s11")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (1, "base-1", "head-1-live"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s15_partial_target_resnapshot_fails_the_observed_live_tuple():
    # The target re-snapshot's pulls.get observes a force-pushed head, then a
    # later read inside snapshot() fails. Infra class: every known snapshot
    # resets, AND the drifted tuple the re-snapshot observed must fail too,
    # rather than keeping whatever exact-head check that head already holds.
    result = _run_workflow_scenario("s15")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (1, "base-1", "head-1-live"),
        (2, "base-2", "head-2"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


_PARTIAL_PEER_FINAL_READ_FAILURES = {
    (1, "base-1", "head-1"),
    (2, "base-2", "head-2"),
    (2, "base-2", "head-2-live"),
}


def test_behavioral_s16_partial_preserved_peer_final_read_fails_the_observed_live_tuple():
    # A preserved peer's final-read snapshot observes a force-pushed head, then a
    # later read inside snapshot() fails: the original AND the observed tuple
    # fail, and the helper's throw still resets every known snapshot.
    result = _run_workflow_scenario("s16")
    assert _failure_tuples(result) == _PARTIAL_PEER_FINAL_READ_FAILURES
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s16b_partial_invalid_peer_final_read_fails_the_observed_live_tuple():
    # Same partial failure inside the invalid-peer final read.
    result = _run_workflow_scenario("s16b")
    assert _failure_tuples(result) == _PARTIAL_PEER_FINAL_READ_FAILURES
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_final_read_helpers_recover_the_carried_live_tuple():
    workflow = _workflow_under_test().read_text(encoding="utf-8")
    needle = "if (!currentTuple && error?.liveTuple)"
    invalid = workflow.index("const finalReadInvalidPeers")
    preserved = workflow.index("const finalReadPreservedPeers")
    after = workflow.index("const repository = await github.rest.repos.get")
    assert workflow.count(needle) == 2
    assert invalid < workflow.index(needle) < preserved < workflow.rindex(needle) < after


def test_behavioral_s12_state_invalidated_prior_packet_still_blocks_replay():
    # A trusted exact success check whose packet is no longer preservable
    # (state fingerprint drift) still yields its consumed dispatch_id, so the
    # replayed dispatch is rejected as a clean FAIL (target-only reset).
    result = _run_workflow_scenario("s12")
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])
    assert _failure_tuples(result) == {(1, "base-1", "head-1")}


def test_behavioral_s13_partial_peer_snapshot_fails_live_and_listed_tuples():
    # A peer's pulls.get observes a force-pushed head, then a later read in the
    # same snapshot rejects. The incomplete reconciliation must fail the head
    # that is actually live (not only the stale list entry) and the listed one.
    result = _run_workflow_scenario("s13")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (2, "base-2", "head-2-live"),
        (2, "base-2", "head-2"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s14_falsy_throw_in_peer_snapshot_still_fails_live_and_listed_tuples():
    # A non-object rejection (throw null) must not escape the reconciliation
    # catch through an `error.message` read; the live tuple is still carried.
    result = _run_workflow_scenario("s14")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (2, "base-2", "head-2-live"),
        (2, "base-2", "head-2"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def test_behavioral_s14b_falsy_throw_in_peer_only_mode_resets_known_tuples():
    result = _run_workflow_scenario("s14b")
    assert _failure_tuples(result) == {
        (1, "base-1", "head-1"),
        (2, "base-2", "head-2-live"),
        (2, "base-2", "head-2"),
    }
    assert not any(emit["conclusion"] == "success" for emit in result["emissions"])


def assessment_body(
    lane: str,
    *,
    pr_number: int = 350,
    base_sha: str = BASE,
    head_sha: str = HEAD,
    verdict: str = "PASS",
    findings: list[str] | None = None,
):
    return {
        "schema_version": "2",
        "repository": "electricsheephq/lcm-x",
        "pr_number": pr_number,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "lane": lane,
        "verdict": verdict,
        "scope": f"{lane} review of the exact PR head",
        "findings": [] if findings is None else findings,
        "limitations": [],
        "acceptance_evidence": ["independent reviewer execution completed"],
        "policy_version": "2",
    }


def review_bundle(assessments: list[dict[str, object]]):
    refs, artifacts = [], []
    for item in assessments:
        reviewer_id, login, review_id = REVIEWERS[item["lane"]]
        refs.append({"lane": item["lane"], "review_id": review_id})
        artifacts.append({
            "review_id": review_id,
            "state": "COMMENTED",
            "commit_id": item["head_sha"],
            "submitted_at": "2026-08-24T11:00:00Z",
            "user": {"id": reviewer_id, "login": login, "type": "Bot"},
            "body": f"Independent {item['lane']} review completed.\n\n<!-- lcm-x-ai-review:v2\n"
            + json.dumps(item, sort_keys=True, separators=(",", ":"))
            + "\n-->",
        })
    return refs, artifacts


def set_review_body_field(artifact: dict[str, object], field: str, value: object):
    marker, suffix = "<!-- lcm-x-ai-review:v2\n", "\n-->"
    body = artifact["body"]
    marker_at = body.rfind(marker)
    prefix = body[:marker_at + len(marker)]
    assessment = json.loads(body[marker_at + len(marker):-len(suffix)])
    assessment[field] = value
    artifact["body"] = prefix + json.dumps(assessment, sort_keys=True,
        separators=(",", ":")) + suffix


def legacy_payload(withdrawn_id: str | None = None):
    receipt_id = withdrawn_id or "legacy-producer-receipt"
    return {
        "schema_version": "1",
        "repository": "electricsheephq/lcm-x",
        "pr_number": 350,
        "base_sha": BASE,
        "head_sha": HEAD,
        "receipts": [{"lane": "acceptance", "receipt_id": receipt_id,
                      "score": 100, "verdict": "PASS"}],
    }


def test_routine_requires_one_original_acceptance_assessment():
    result = evaluate_reconciliation(_v2_dispatch(_v2_snapshot(350)), NOW)

    assert result["decision"] == "PASS"
    assert result["base_sha"] == BASE
    assert result["head_sha"] == HEAD
    assert result["blockers"] == []
    assert [item["lane"] for item in result["packet"]["assessments"]] == ["acceptance"]
    assert "Independent acceptance review completed." in result["packet"]["assessments"][0]["original_body"]


@pytest.mark.parametrize("path", ["AGENTS.md", "store.py"])
def test_named_risks_require_distinct_adversarial_assessment(path):
    dispatch = _v2_dispatch(_v2_snapshot(350, changed_paths=[path]))
    assert evaluate_reconciliation(dispatch, NOW)["decision"] == "PASS"

    dispatch["review_artifact_refs"] = dispatch["review_artifact_refs"][:1]
    dispatch["target"]["review_artifacts"] = dispatch["target"]["review_artifacts"][:1]
    result = evaluate_reconciliation(dispatch, NOW)
    assert result["decision"] == "FAIL"
    assert "REVIEW_ARTIFACT_REF_SET_INVALID" in result["blockers"]


def test_producer_assessment_claim_is_ignored_in_favor_of_fetched_review():
    data = _v2_dispatch(_v2_snapshot(350))
    data["assessments"] = [{"lane": "acceptance", "verdict": "PASS"}]

    result = evaluate_reconciliation(data, NOW)

    assert result["decision"] == "PASS"
    assert result["packet"]["assessments"] != data["assessments"]
    assert result["packet"]["assessments"][0]["verdict"] == "PASS"
    assert result["packet"]["assessments"][0]["original_body"] == data[
        "target"
    ]["review_artifacts"][0]["body"]


@pytest.mark.parametrize("verdict", ["BLOCKED", "ABSTAIN"])
def test_explicit_original_non_pass_verdict_fails(verdict):
    dispatch = _v2_dispatch(_v2_snapshot(350), verdict=verdict)

    result = evaluate_reconciliation(dispatch, NOW)

    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_VERDICT_NOT_PASS" in result["blockers"]


def test_pass_with_unresolved_findings_fails():
    dispatch = _v2_dispatch(_v2_snapshot(350), findings=["unresolved defect"])
    result = evaluate_reconciliation(dispatch, NOW)
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_FINDINGS_UNRESOLVED" in result["blockers"]


def test_ordinary_github_review_state_never_becomes_pass_without_v2_body():
    dispatch = _v2_dispatch(_v2_snapshot(350))
    dispatch["target"]["review_artifacts"][0]["body"] = "Looks good"
    result = evaluate_reconciliation(dispatch, NOW)
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_REVIEW_BODY_INVALID" in result["blockers"]


def test_duplicate_v2_markers_are_rejected():
    dispatch = _v2_dispatch(_v2_snapshot(350))
    artifact = dispatch["target"]["review_artifacts"][0]
    artifact["body"] = "<!-- lcm-x-ai-review:v2\n{}\n-->\n\n" + artifact["body"]
    result = evaluate_reconciliation(dispatch, NOW)
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_REVIEW_BODY_INVALID" in result["blockers"]


def test_legacy_numeric_review_body_is_rejected():
    dispatch = _v2_dispatch(_v2_snapshot(350))
    dispatch["target"]["review_artifacts"][0]["body"] = (
        '<!-- lcm-x-ai-review:v1\n{"verdict":"PASS","score":100,"findings":0}\n-->'
    )
    result = evaluate_reconciliation(dispatch, NOW)
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_REVIEW_BODY_INVALID" in result["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "someone/fork"),
        ("pr_number", 999),
        ("base_sha", "4" * 40),
        ("head_sha", "5" * 40),
    ],
)
def test_fetched_review_artifact_requires_exact_target_binding(field, value):
    dispatch = _v2_dispatch(_v2_snapshot(350))
    set_review_body_field(dispatch["target"]["review_artifacts"][0], field, value)

    result = evaluate_reconciliation(dispatch, NOW)

    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_REVIEW_BINDING_MISMATCH" in result["blockers"]


def test_preserved_packet_rechecks_fetched_review_publisher_identity():
    peer = _v2_snapshot(350)
    peer["review_artifacts"][0]["user"]["id"] = 12345

    result = evaluate_reconciliation(
        {"schema_version": "2", "mode": "peer_only", "peers": [peer]}, NOW,
    )["peers"][0]

    assert result["decision"] == "FAIL"
    assert result["preserve"] is False
    assert "ACCEPTANCE_REVIEW_PUBLISHER_INVALID" in result["blockers"]


@pytest.mark.parametrize("withdrawn_id", [
    "52eca5d8-1628-4f03-9128-34a5e1d5746f",
    "5a6b05ee-c573-4cd9-b6f4-9a944fba61ee",
])
@pytest.mark.parametrize("mode", ["legacy", "dispatch", "preserved"])
def test_publicly_withdrawn_receipts_cannot_approve_or_remain_preserved(withdrawn_id, mode):
    # These IDs were publicly withdrawn in PR462 comment5771571369.
    # All other fields are synthetic valid fixtures, never a real dispatch.
    data = legacy_payload(withdrawn_id)
    if mode == "legacy":
        result = evaluate(data, NOW)
    elif mode == "dispatch":
        dispatch = _v2_dispatch(_v2_snapshot(350))
        dispatch["review_artifact_refs"][0]["review_id"] = withdrawn_id
        result = evaluate_reconciliation(dispatch, NOW)
    else:
        peer = _v2_snapshot(350)
        packet = json.loads(peer["check_runs"][0]["output_summary"])
        packet["receipts"] = data["receipts"]
        peer["check_runs"][0]["output_summary"] = json.dumps(packet)
        result = evaluate_reconciliation(
            {"schema_version": "2", "mode": "peer_only", "peers": [peer]}, NOW,
        )["peers"][0]
        assert result["preserve"] is False
    assert result["decision"] == "FAIL"
    expected = (
        "ACCEPTANCE_REVIEW_ARTIFACT_ID_INVALID"
        if mode == "dispatch"
        else ("PACKET_SCHEMA_INVALID" if mode == "preserved" else "ACCEPTANCE_RECEIPT_WITHDRAWN")
    )
    assert expected in result["blockers"]


def test_live_state_and_thread_failures_fail_closed():
    for field, value in (("api_complete", False), ("pagination_complete", False),
                         ("unresolved_threads", 1), ("pr_number", True)):
        data = _v2_dispatch(_v2_snapshot(350))
        data["target"][field] = value
        assert evaluate_reconciliation(data, NOW)["decision"] == "FAIL"


def test_review_body_pr_number_requires_exact_integer_binding():
    for invalid_pr_number in (True, 1.0):
        dispatch = _v2_dispatch(_v2_snapshot(1))
        set_review_body_field(
            dispatch["target"]["review_artifacts"][0], "pr_number", invalid_pr_number
        )
        result = evaluate_reconciliation(dispatch, NOW)

        assert result["decision"] == "FAIL"
        assert "ACCEPTANCE_REVIEW_BINDING_MISMATCH" in result["blockers"]


def test_dispatch_envelope_preflight_returns_the_bound_packet():
    target = _v2_snapshot(350, changed_paths=["AGENTS.md"])
    envelope = {
        "schema_version": "2",
        "mode": "dispatch_envelope",
        "target": target,
        "review_artifact_refs": json.loads(target["check_runs"][0]["output_summary"])["review_artifact_refs"],
        "producer": {"login": "100yenadmin", "id": 239388517, "type": "User"},
        "run": {"id": "run-envelope", "attempt": 1},
        "dispatch_id": "dispatch-envelope-fresh",
    }

    valid = evaluate_reconciliation(envelope, NOW)
    assert valid["decision"] == "PASS"
    assert valid["packet"]["assessments"][0]["original_body"] == target["review_artifacts"][0]["body"]

    for malformed in ([], [{}]):
        candidate = deepcopy(envelope)
        candidate["review_artifact_refs"] = malformed
        result = evaluate_reconciliation(candidate, NOW)
        assert result["decision"] == "FAIL"
        assert "REVIEW_ARTIFACT_REF_SET_INVALID" in result["blockers"]


def test_publisher_id_requires_exact_integer_type():
    for invalid_integration_id in (True, 298367747.0):
        dispatch = _v2_dispatch(_v2_snapshot(350))
        dispatch["target"]["review_artifacts"][0]["user"]["id"] = (
            invalid_integration_id
        )
        assert "ACCEPTANCE_REVIEW_PUBLISHER_INVALID" in evaluate_reconciliation(
            dispatch, NOW
        )["blockers"]


def test_original_body_digest_is_rechecked_for_preserved_packet():
    peer = _v2_snapshot(350)
    packet = json.loads(peer["check_runs"][0]["output_summary"])
    packet["assessments"][0]["original_body"] += "tampered"
    peer["check_runs"][0]["output_summary"] = json.dumps(packet)
    result = evaluate_reconciliation(
        {"schema_version": "2", "mode": "peer_only", "peers": [peer]}, NOW
    )["peers"][0]
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_ORIGINAL_BODY_MISMATCH" in result["blockers"]


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
    assert "github.rest.pulls.getReview" in workflow
    assert "review_artifact_refs" in workflow
    assert "'receipts' in dispatch" in workflow
    assert "dispatch.receipts" not in workflow


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


def test_incomplete_snapshot_prefers_the_observed_live_tuple_over_the_listed_one():
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

    # The tuple pulls.get observed wins; the list entry is only the fallback,
    # and when the two differ the listed tuple is queued to fail as well.
    assert "base_ref: live?.base_ref ?? candidate.base?.ref" in failure_section
    assert "base_sha: live?.base_sha ?? candidate.base?.sha" in failure_section
    assert "head_sha: live?.head_sha ?? candidate.head?.sha" in failure_section
    assert "state: live?.state ?? candidate.state" in failure_section
    assert "draft: live?.draft ?? candidate.draft" in failure_section
    assert "listedTuples.push({pr_number: candidate.number" in failure_section
    assert "carrier.liveTuple = {base_ref: pr.base.ref" in workflow[:reconcile]
    assert "${error?.message ?? String(error)}" in workflow[reconcile:helper]
    assert workflow.count("failKnownSnapshots(snapshots.concat(reconciled.listedTuples)") == 3


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
        "target = await snapshot(prNumber, defaultBranch,", reconciliation_guard
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
    packet_guard = workflow.index(
        "if (!Array.isArray(dispatch.review_artifact_refs)", dispatch
    )
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


def test_dispatch_artifact_validation_follows_complete_reconciliation():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ai-review-gate.yml"
    ).read_text(encoding="utf-8")

    dispatch = workflow.index("if (dispatchEvent) {")
    authenticated = workflow.index("if (sender?.login !== '100yenadmin'", dispatch)
    reconciliation_guard = workflow.index("if (reconciled.failures.length)", authenticated)
    peer_resets = workflow.index("await failKnownSnapshots(snapshots", reconciliation_guard)
    preflight = workflow.index("mode: 'dispatch_envelope'", peer_resets)
    preflight_stop = workflow.index("throw Error('dispatch packet invalid')", preflight)

    assert authenticated < reconciliation_guard < peer_resets < preflight < preflight_stop
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
    assert "const finalReviewArtifacts = await fetchReviewArtifacts" in workflow
    assert "const finalArtifactValidation = await runValidator" in workflow
    assert "review artifacts changed or became unavailable" in workflow
    assert "JSON.stringify(finalArtifactValidation.result.packet) !== JSON.stringify(result.packet)" in workflow
    assert "errors.push({review_id, status:" in workflow
    assert "review_artifact_errors: reviewArtifacts.errors" in workflow
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
        "target = await snapshot(prNumber, defaultBranch,"
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
        "target = await snapshot(prNumber, defaultBranch,"
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
        "const priorPreserved = priorValidated.run.status === 0 && priorValidated.result.decision === 'PASS'",
        prior_validation,
    )
    target_reset = workflow.index(
        "await emit(prNumber, base, head, 'failure', 'Protected base changed",
        optional_history,
    )
    target_snapshot = workflow.index(
        "target = await snapshot(prNumber, defaultBranch,",
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
        "target = await snapshot(prNumber, defaultBranch,", target_reset
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


def _lanes_for_paths(paths: list[str]) -> list[str]:
    protected = {
        ".github/workflows/ai-review-gate.yml", "AGENTS.md", "CONTRIBUTING.md",
        "docs/review-evidence-provenance.md", "scripts/ai_review_gate.py",
        "scripts/maintainer_gate.py", "store.py",
    }
    lanes = ["acceptance"]
    if any(path in protected or path.startswith(".agents/skills/land-pr/") for path in paths):
        lanes.append("adversarial")
    return lanes


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
    lanes = _lanes_for_paths(live["changed_paths"])
    bodies = [assessment_body(lane, pr_number=pr_number) for lane in lanes]
    refs, artifacts = review_bundle(bodies)
    live["review_artifacts"] = artifacts
    live["review_artifact_errors"] = []
    assessments, blockers = _review_artifact_assessments(refs, artifacts, live)
    assert blockers == []
    packet = build_packet(live, assessments, refs, producer={"login": "100yenadmin", "id": 239388517, "type": "User"},
                          run={"id": f"run-{pr_number}", "attempt": 1}, dispatch_id=f"dispatch-{pr_number}")
    check = {"name": "AI review exact-head", "app_id": 15368,
             "external_id": f"ai-review-gate:{pr_number}:{base}:{head}", "head_sha": head,
             "status": "completed", "conclusion": "success",
             "output_summary": json.dumps(packet, sort_keys=True, separators=(",", ":"))}
    return {**live, "check_runs": [check]}


def _v2_dispatch(
    target: dict[str, object], *, verdict: str = "PASS",
    findings: list[str] | None = None, **extra
):
    target = deepcopy(target)
    lanes = _lanes_for_paths(target["changed_paths"])
    bodies = [assessment_body(
        lane, pr_number=target["pr_number"], base_sha=target["base_sha"],
        head_sha=target["head_sha"], verdict=verdict,
        findings=findings,
    ) for lane in lanes]
    refs, artifacts = review_bundle(bodies)
    target["review_artifacts"] = artifacts
    target["review_artifact_errors"] = []
    data = {key: target[key] for key in (
        "repository", "pr_number", "base_sha", "head_sha", "changed_paths",
        "timeline_events", "unresolved_threads", "api_complete", "pagination_complete",
    )}
    data.update({
        "schema_version": "2", "target": target,
        "review_artifact_refs": refs,
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


def test_missing_target_review_artifact_fails_target_without_revoking_peer():
    peer = _v2_snapshot(351)
    dispatch = _v2_dispatch(_v2_snapshot(350), peers=[peer])
    dispatch["target"]["review_artifacts"] = []
    dispatch["target"]["review_artifact_errors"] = [
        {"review_id": 9001, "status": 404}
    ]

    result = evaluate_reconciliation(dispatch, NOW)

    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_REVIEW_ARTIFACT_COUNT_INVALID" in result["blockers"]
    assert result["peers"][0]["preserve"] is True
    assert result["peers"][0]["decision"] == "PASS"


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


def test_expired_tracking_metadata_does_not_revoke_a_promoted_packet_but_blocks_fresh_dispatch():
    # Freshness is an acceptance-time property: a stored packet stays
    # preservable after tracking metadata ages out (its lifetime is governed by
    # the state fingerprint), while the same assessment cannot promote a new dispatch.
    from scripts.ai_review_gate import evaluate as _evaluate

    later = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    peer_only = _evaluate(
        {"schema_version": "2", "mode": "peer_only", "peers": [_v2_snapshot(351)]},
        later,
    )
    assert peer_only["decision"] == "PASS"
    assert peer_only["peers"][0]["preserve"] is True

    result = evaluate_reconciliation(
        _v2_dispatch(_v2_snapshot(350), peers=[_v2_snapshot(351)]), later
    )
    assert result["decision"] == "FAIL"
    assert "ACCEPTANCE_ASSESSMENT_STALE" in result["blockers"]
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
        "const reconciled = await reconcileOpenPullRequests(defaultBranch, prNumber, dispatchRefs);"
    )

    # A rejection of the top-level open-PR enumeration must still reach the
    # fail-closed emit for an event-identified dispatch target: the target's
    # number and tuple are captured before reconciliation can abort.
    assert target_number < target_tuple < enumeration
    assert "if (prNumber && base && head) {" in workflow
    assert "for (const tuple of [resnapshotTuple, liveTuple])" in workflow
