---
name: land-pr
description: Verify and land electricsheephq/lcm-x pull requests safely. Use when asked whether an LCM-X PR is ready, to review merge readiness, to merge or land a PR, to close out a contributor change, or to check that a bug or feature PR satisfies the Hermes compatibility, lossless-data, exact-head CI, review, issue-linkage, and authorship requirements.
---

# Land LCM-X Pull Requests

Follow this workflow for `electricsheephq/lcm-x`. Read the repository-root `AGENTS.md` first;
it is authoritative when this skill and repository policy differ.

## 1. Resolve Identity And Pin The Head

Verify the repository and inspect the real PR:

```bash
git remote get-url origin
gh pr view <PR> --repo electricsheephq/lcm-x \
  --json number,title,state,isDraft,baseRefName,headRefName,headRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,author,files,closingIssuesReferences,url
```

Record `headRefOid` as the merge pin. Stop on the wrong repository, a non-`main` base, a closed
or draft PR, an unknown/conflicting merge state, or head drift.

## 2. Verify The Accepted Work

- Read every linked issue and the complete PR body/diff.
- For a bug, require a current-main reproduction or a named mandatory-invariant violation and
  a regression that fails on the base and passes on the PR.
- For a feature/design change, require a maintainer-accepted issue with problem, impact,
  alternatives, acceptance criteria, non-goals, backward-compatible defaults, Hermes
  compatibility, and documentation/config consequences.
- For an upstream-derived change, retain the upstream issue/PR links and original authorship.
- Do not infer acceptance from labels, a contributor claim, or an unmerged upstream PR.

## 3. Verify Exact-Head Checks

Read required/current checks:

```bash
gh pr checks <PR> --repo electricsheephq/lcm-x \
  --json name,state,bucket,workflow,link,startedAt,completedAt
```

Require success on the pinned head for:

- `workflow-lint`
- `lint`
- `test (3.11)`
- `test (3.12)`
- `test (3.13)`
- `test (3.14)`
- `Analyze (actions)`
- `Analyze (javascript-typescript)`
- `Analyze (python)`

Treat pending, skipped, missing, stale-head, or failing required checks as blocking. Socket or
other advisory checks may add evidence but do not replace the required GitHub checks.

## 4. Verify Review Coverage

- Require one non-author code-owner approval on the current head.
- Require every review thread to be resolved.
- For data-integrity, security, migration, compaction, persistence, lifecycle/session identity,
  or Hermes host-contract changes, require one independent semantic review covering the named
  risk lane.
- After a review-driven head change, require a delta review for the changed risk surface and
  re-read the head SHA and checks.
- Give every verified finding one terminal disposition. Do not turn unverified possibilities
  or nits into merge blockers.

Do not count CI, author self-review, a flat bot status comment, or this skill as independent
semantic review.

## 5. Check Hermes And Lossless Boundaries

- Confirm the change preserves durable context, identity, chronology, ownership, provenance,
  source coverage, tool grouping, user turns, and fresh-tail guarantees on every changed path.
- Confirm config changes update the loader, docs, bundled Hermes skill reference, examples, and
  tests together.
- Confirm tool/lifecycle changes document and test the Hermes contract and fallback path.
- Stop on an unapproved dependency, schema, service, persistence surface, breaking default, or
  unreleased/unverified Hermes host assumption.

## 6. Search Related Issues

Inspect explicit closing references, then search open issues by the PR title, feature name,
error text, and upstream IDs. Read each candidate before deciding it is covered. If coverage is
uncertain, comment or report the relationship; do not close the issue.

## 7. Merge Deterministically

Immediately before merging:

```bash
head="$(gh pr view <PR> --repo electricsheephq/lcm-x --json headRefOid --jq .headRefOid)"
gh pr view <PR> --repo electricsheephq/lcm-x \
  --json state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision
gh pr checks <PR> --repo electricsheephq/lcm-x
gh pr merge <PR> --repo electricsheephq/lcm-x --merge --match-head-commit "$head"
```

Never use auto-merge, squash, rebase merge, direct `main` pushes, force pushes, branch deletion,
or a ruleset bypass.

## 8. Verify And Close Out

Verify the result:

```bash
gh pr view <PR> --repo electricsheephq/lcm-x \
  --json state,mergeCommit,mergedAt,mergedBy,closingIssuesReferences,url
gh api repos/electricsheephq/lcm-x/commits/main --jq .sha
```

- Confirm the PR is merged and capture the merge commit.
- Confirm verified closing issues are closed as completed.
- Thank external contributors and link the landed PR.
- Leave ambiguous issues open with a precise relationship note.
- Report the pinned head, check/review summary, merge command, merge commit, issue disposition,
  authorship/provenance outcome, and proof boundary.

Claim only source/merge readiness for the named SHA. Do not claim release, deployment, runtime,
customer behavior, or absence of other defects.
