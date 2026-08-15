---
name: triage-backlog
description: Read-only LCM-X issue and pull-request triage for one report or duplicate cluster at a time. Use when assessing current-main applicability, P0-P4 priority, reproduction quality, duplicate relationships, linked PRs, ownership, Electric Sheep impact, or the next evidence gate without changing GitHub.
---

# Triage Backlog

Triage one issue, PR, or duplicate cluster into a bounded evidence packet. Treat GitHub as the
live source and make no writes unless a maintainer separately authorizes an exact mutation.

## 1. Establish Identity

Record:

- repository `electricsheephq/lcm-x`;
- exact current `main` SHA;
- item number, URL, state, author, timestamps, labels, and linked work;
- whether the item came from LCM-X, upstream `stephenschoettler/hermes-lcm`, or another source.

Stop on the wrong repository, unavailable live state, credentials in evidence, or a request that
would require customer/runtime mutation.

## 2. Search Before Classifying

Search open and closed LCM-X issues and PRs using the title, error text, component, invariant,
upstream number, and linked commits. Read plausible matches before naming a duplicate.

For a duplicate cluster:

- select the earliest or best-evidenced active item as canonical;
- preserve every original URL, author, and upstream link;
- distinguish the same root cause from merely similar symptoms;
- do not close or comment on any item.

## 3. Verify Current Applicability

For a reported bug, require either:

- reproduction on the exact current `main` SHA; or
- evidence that current code violates a named mandatory invariant in `AGENTS.md`.

Record the supported path, expected result, observed result, exact command or artifact, affected
SHA, frequency/impact, workaround, and whether data is missing, duplicated, reordered,
misattributed, or visible across sessions/profiles.

Use `needs-repro` when evidence is insufficient. Labels, CI, reviewer severity, an upstream claim,
or an unmerged PR are leads—not proof.

For a linked PR, inspect its complete diff, exact head/base, tests, checks, review history, issue
coverage, dependencies, overlap with other PRs, and any behavior it introduces. Do not call it
merge-ready; use `land-pr` for a separately authorized landing decision.

## 4. Classify And Prioritize

Use one primary classification:

- `bug`, `security`, `data-integrity`, `performance`, `feature`, `docs`, `test`,
  `best-practice`, `nit`, `needs-repro`, `duplicate`, `superseded`, or `out-of-scope`.

Assign:

- `P0`: active catastrophic data loss, disclosure, or unusable core path;
- `P1`: verified severe integrity, security, crash, lockout, or release-blocking failure;
- `P2`: significant bounded correctness or resource failure;
- `P3`: limited bug, feature gap, or moderate maintainability/performance issue;
- `P4`: documentation, tooling, portability, testing, or cosmetic work.

Also record:

- confidence: `verified-current`, `likely`, `needs-repro`, `not-a-bug`, or `superseded`;
- LCM-X/Electric Sheep impact: `direct`, `possible`, or `none`;
- disposition: `migrate`, `reference-only`, `needs-repro`, `duplicate`, `superseded`,
  `defer`, or `do-not-migrate`;
- accountable owner and the next evidence or decision gate.

Nits default to P4. Best-practice work remains P3/P4 unless current evidence proves a violated
mandatory invariant.

## 5. Return One Bounded Packet

Return:

```text
Item:
Exact main SHA:
Classification / priority:
Confidence:
Current-main applicability:
Supported-path impact:
Evidence:
Canonical duplicate / related work:
Linked PR assessment:
Electric Sheep impact:
Owner:
Recommended disposition:
Exact next gate:
Proof boundary:
```

Deduplicate findings while retaining every source URL and contributor attribution. State clearly
whether a failure is reproduced, inferred, or still hypothetical.

## Write Boundary

Remain read-only by default. Do not comment, label, assign, close, reopen, create issues, edit
milestones, approve, push, or merge.

If a maintainer explicitly authorizes one of those mutations:

1. name the exact item and requested change;
2. re-fetch current item, repository, and authorization state;
3. stop on drift or ambiguity;
4. perform only the named write;
5. read back the result.

AI output is proposal and evidence. Deterministic live-state checks must guard every authorized
write. Never process the whole backlog, manufacture issues from hypotheses, or post one comment
per item when a bounded summary will do.
