---
name: land-pr
description: Land one electricsheephq/lcm-x pull request only after explicit authority names the PR number and exact head SHA. Use solely for a current instruction to merge that exact PR/head, never for readiness assessment, review, approval, source repair, or general closeout.
---

# Land An Authorized LCM-X Pull Request

The sole trigger is explicit current authority to merge PR N at exact head H. Invoking this skill,
a readiness request, or a `review-pr` receipt never creates that authority. The only permitted
GitHub mutation is the exact-head merge command in Section 5.

## 1. Bind Authority And Protected Policy

Record the authorizing actor, repository `electricsheephq/lcm-x`, positive PR number, 40-character
lowercase hexadecimal head SHA, and action `merge`. Stop with `OWNER_GATE` if any field is absent
or differs from the live PR.

Run from protected `main` or an immutable trusted installation. Re-fetch `AGENTS.md`, this skill,
`.github/CODEOWNERS`, and live ruleset `20888757`. Protected policy outranks PR-authored copies.
Stop if the ruleset, protected base, required check pairs, merge method, or authority is ambiguous.

## 2. Require A Ready Exact Head

Re-fetch the live linked issue and require the repository, PR link or accepted implementation
path, behavior, and file scope to match this PR. Require an open non-draft PR targeting `main`,
terminal finding dispositions, zero unresolved threads, and successful required checks on the
pinned head. Match every required check by exact `(context, integration_id)` from the live
ruleset and reject missing, pending, failed, skipped, stale-head, duplicate, or same-name
wrong-app results. Stop with `OWNER_GATE` if live required pairs differ from the evaluator's
versioned set.

For the normal path, require the protected ruleset's current non-author CODEOWNER approval. Use
each reviewer's latest opinionated review, require its commit to equal the pinned head, and reject
a tied or latest `CHANGES_REQUESTED` review.

Require independent exact-head semantic review for governance, security, data integrity,
migration, compaction, persistence, lifecycle/session identity, or Hermes host-contract changes.
CI, the PR author, this skill, and flat bot comments do not satisfy semantic review.

## 3. Constrain The Exceptional Administrator Path

Normal protected landing is the default. The specifically configured administrator may use the
ruleset's PR-only bypass only when all of the following are true. GitHub's `--admin` flag is a
broad, non-atomic administrative bypass, not an approval-only server primitive:

- explicit current authority requests the admin path for PR N at exact head H;
- live ruleset `20888757` lists that exact user actor with `bypass_mode: pull_request`;
- direct-push, role-wide, `always`, and `exempt` bypass are absent;
- every non-review gate in Section 2 passes;
- independent blind acceptance and adversarial receipts have distinct reviewer and receipt IDs,
  each report `PASS`, score at least 95, cover the exact head, and explicitly report zero
  unresolved findings;
- no tied or latest `CHANGES_REQUESTED` review exists; and
- the authorizing maintainer explicitly accepts the residual risk that a rule or check could
  change between the final live read and the administrative merge; and
- the bypass use, accepted risk, and evidence are recorded in an auditable receipt.

The policy intent is to replace only the missing non-author approval for that one merge. The
GitHub flag cannot technically enforce that narrow limit. Do not intentionally waive trusted
checks, accepted work, exact-head binding, semantic review, findings, threads, or owner decisions,
and do not describe those gates as atomically server-enforced on this path.

## 4. Evaluate, Then Re-fetch Live State

Build the documented JSON envelope for `scripts/maintainer_gate.py` in `landing` mode, including
the exact merge authorization. For the exceptional path, also include the live user-specific
PR-only bypass actor and both blind review receipts. Treat evaluator output as advisory evidence,
never authority.

Immediately before merging, independently re-fetch the repository, PR/base/head, linked issue
and scope, ruleset/check pairs, exact-head check runs and integration IDs, latest reviews,
CODEOWNER coverage, threads, semantic-review receipt, blind receipts when used, and authorization.
Stop on malformed GitHub object IDs, drift, a pending/failing gate, an unresolved finding/thread,
or any evaluator decision other than `READY_FOR_AUTHORIZED_LANDING`.

## 5. Perform The Only Mutation

For the normal protected path, run only:

```bash
gh pr merge <PR> --repo electricsheephq/lcm-x --merge --match-head-commit <HEAD_SHA>
```

For a fully qualified Section 3 exception, add only `--admin` to that same command. Never use
auto-merge, squash, rebase, a direct `main` push, force push, branch rewrite/deletion, approval,
source push, comment, label, assignment, manual issue closure, release, or deployment.

## 6. Verify Read-only Post-merge Facts

After the command, re-fetch the PR state, merge commit, live `main`, linked issue disposition,
ruleset, and required checks on the merge commit. Verify the final PR head is a merge parent and
the merge commit is equal to or an ancestor of live `main`; a concurrent later `main` is valid
when that ancestry holds. Match post-merge checks by exact `(context, integration_id)` and merge
commit SHA.

Run `scripts/maintainer_gate.py` in `post_merge` mode and record its receipt. Report authority,
path used, pinned head, merge command, merge commit, ancestry, checks, issue disposition,
provenance, and proof boundary.

## Failure Behavior

Fail closed to `OWNER_GATE`, `STATE_DRIFT`, `NOT_READY`, or `NOT_DIRECTLY_LANDABLE`. Do not repair
the branch, seek or create approval, mutate metadata, weaken policy, retry with a broader bypass,
or convert missing evidence into authority.

Claim only the named source merge. Do not claim release, deployment, runtime, customer behavior,
or universal absence of defects.
