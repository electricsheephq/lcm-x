# Label Policy

This page says what each repository label means, who applies it, and when it comes off. It does
not change who may write labels: a label change needs one exact maintainer authorization and a
live read-back, as set out in [AGENTS.md: Automation Boundary](../../AGENTS.md#automation-boundary)
and the [triage-backlog Write Boundary](../../.agents/skills/triage-backlog/SKILL.md#write-boundary).
In the tables, "Maintainer" means a maintainer, or an agent acting on that maintainer's exact
authorization.

## Rules after triage

An open issue that has been triaged carries:

- exactly one priority label, `P0` to `P4`. The definitions are the label descriptions and
  [triage-backlog §4](../../.agents/skills/triage-backlog/SKILL.md#4-classify-and-prioritize);
- exactly one type label: `bug`, `enhancement`, `documentation`, `test`, `best-practice`, or
  `question`;
- at most one domain label: `data-integrity`, `performance`, or `security`;
- status labels only with the meaning given below.

A public `security` label is a disclosure decision. Vulnerability reports go through the private
path in [SECURITY.md](../../SECURITY.md), not a labeled public issue.

## Labels

### Priority

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `P0` | Verified catastrophic failure requiring immediate action | Maintainer | Re-prioritized with evidence, or the issue closes |
| `P1` | Verified severe integrity, security, crash, lockout, or release-blocking failure | Maintainer | Re-prioritized with evidence, or the issue closes |
| `P2` | Significant supported-path regression or bounded correctness failure | Maintainer | Re-prioritized with evidence, or the issue closes |
| `P3` | Limited bug, edge case, non-blocking feature, or moderate maintainability issue | Maintainer | Re-prioritized with evidence, or the issue closes |
| `P4` | Low-impact portability, documentation, testing, tooling, or cosmetic issue | Maintainer | Re-prioritized with evidence, or the issue closes |

Changing priority means removing the old P-label in the same change; an issue never holds two.

### Type

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `bug` | Broken, incorrect, or regressed behavior | Bug form applies it; maintainer confirms | Triage shows it is another type |
| `enhancement` | New feature, design change, or feature gap | Feature form applies it; maintainer confirms | Triage shows it is another type |
| `documentation` | Docs are wrong, missing, or unclear | Maintainer | Triage shows it is another type |
| `test` | Test coverage, determinism, or portability | Maintainer | Triage shows it is another type |
| `best-practice` | Maintainability or engineering-practice work without a current invariant failure | Maintainer | Evidence shows a violated invariant (relabel as `bug`) |
| `question` | The reporter asks for information, not a change | Maintainer | Answered (then close) or turned into a concrete type |

### Domain

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `data-integrity` | Durable or active-context correctness and ownership defect | Maintainer | Triage shows no integrity impact |
| `performance` | Performance, latency, contention, or resource-exhaustion issue | Maintainer | Triage shows no performance impact |
| `security` | Security or trust-boundary defect that is safe to track in public | Maintainer | Triage shows no trust-boundary impact |

### Status

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `needs-repro` | No reproduction yet on the latest release or current `main` | Maintainer | Reproduced on current `main` or a named invariant violation is shown; aging rule in [issue-triage.md](issue-triage.md#needs-repro-aging) |
| `blocked` | Waits on a named external or owner gate; a comment names the gate | Maintainer | The named gate clears |
| `accepted-defer` | Real accepted work parked; a comment names the revisit trigger | Maintainer | The trigger fires and the work is picked up, or the issue closes |
| `good first issue` | The issue has a clear acceptance test and a small, contained change | Maintainer | Someone takes it, or the acceptance test is no longer clear |
| `help wanted` | The issue has a clear acceptance test and outside help is welcome | Maintainer | Someone takes it, or the issue closes |

### Impact on supported deployments

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `eva-direct` | Direct impact on supported deployments | Maintainer | Impact disproved, or the issue closes |
| `eva-possible` | Possible impact on supported deployments; needs path-specific confirmation | Maintainer | Confirmed (switch to `eva-direct`) or disproved |

### Closing labels

These accompany a close; they are not applied to an issue that stays open.

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `duplicate` | Closing against one canonical issue | Maintainer, at close | The issue is reopened as distinct |
| `invalid` | Closing a report that is not a real issue (spam, wrong repository, empty) | Maintainer, at close | The issue is reopened |
| `wontfix` | Closing real work the project decided not to do | Maintainer, at close | The decision is reversed |

### Imported upstream records only

Apply these only to records imported from upstream projects, never to native LCM-X issues.

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `upstream-evidence` | Preserves links and attribution to the upstream report or pull request | Maintainer, at import | Never; it is provenance |
| `upstream-intake` | Attributed upstream record; not active work by itself | Maintainer, at import | Record becomes active work or closes |
| `upstream-issue` | Imported upstream issue evidence | Maintainer, at import | Never; it is provenance |
| `upstream-pr` | Imported upstream pull-request evidence | Maintainer, at import | Never; it is provenance |
| `triage-pending` | New upstream item awaiting LCM-X classification | Maintainer, at import | The item is classified |
| `active-continuation` | Active continuation of an attributed upstream item | Maintainer | Continuation ends; see [triage-backlog §4](../../.agents/skills/triage-backlog/SKILL.md#4-classify-and-prioritize) |
| `reference-only` | Preserved for provenance; not executable work | Maintainer | Never while the record is kept |
| `superseded` | Replaced by a newer issue, pull request, or decision | Maintainer | Never; it records history |
| `superseded-by-active-pr` | Preservation record superseded by an executable LCM-X PR | Maintainer | Never; it records history |

### Historical marker

| Label | When to apply | Who applies | When to remove |
|---|---|---|---|
| `parity-audit-2026-09` | Historical marker for the 2026-09 parity audit; do not apply to new issues | Nobody | Never |

## Mapping from triage-backlog terms

[triage-backlog](../../.agents/skills/triage-backlog/SKILL.md) uses its own vocabulary. Map it to
labels as follows.

| triage-backlog term | Label action |
|---|---|
| `bug`, `test`, `best-practice` | The type label of the same name |
| `feature` | `enhancement` |
| `docs` | `documentation` |
| `security`, `data-integrity`, `performance` | The domain label of the same name, plus the matching type label |
| `nit` | `P4` plus the matching type label |
| `needs-repro` | `needs-repro` |
| `duplicate` | `duplicate`, at close, per [issue-triage.md](issue-triage.md#duplicates) |
| `superseded` | `superseded` for imported upstream records; otherwise close with a pointer to the replacement |
| `out-of-scope`, `not-a-bug`, `archive-record` | Close as not planned with a reason comment, per [issue-triage.md](issue-triage.md#closing) |
| `direct` impact | `eva-direct` |
| `possible` impact | `eva-possible` |
| `none` impact | No impact label |
