# Issue Triage Runbook

How maintainers, and agents acting for them, take an issue from intake to close. Classification
uses [triage-backlog](../../.agents/skills/triage-backlog/SKILL.md); labels follow
[label-policy.md](label-policy.md). Every GitHub write still needs the authorization and live
read-back in [AGENTS.md: Automation Boundary](../../AGENTS.md#automation-boundary). Closes and
reopens also need the close gate in the
[triage-backlog Write Boundary](../../.agents/skills/triage-backlog/SKILL.md#write-boundary).

## Intake

First triage of a new issue happens within 3 business days; a `P0` the same day.

1. Search open and closed issues and PRs for duplicates first
   ([triage-backlog §2](../../.agents/skills/triage-backlog/SKILL.md#2-search-before-classifying)).
2. Check the report on the latest release and on current `main`
   ([triage-backlog §3](../../.agents/skills/triage-backlog/SKILL.md#3-verify-current-applicability)).
3. Apply labels per [label-policy.md](label-policy.md#rules-after-triage).
4. Comment, following the rules below.

## Comment rules

- Start with the type and priority, for example "Bug, P2."
- Give one concrete pointer: a PR, a commit, or an issue.
- State the next step.
- For an outside reporter: thank them and say what happens next. When the report is not yet
  confirmed, ask for a reproduction on the latest release: the LCM-X version, the Hermes host
  version, the steps, and the log line.
- Never promise dates.
- Never say fixed before the fix is merged.

## needs-repro aging

An issue labeled `needs-repro` with no reply from the reporter in 30 days is closed as not
planned. The close comment invites a reopen with a reproduction on the latest release.

## Duplicates

Close the duplicate against one canonical issue. The close comment carries over what the
duplicate adds and names the shape the canonical fix must test.

## Closing

- Always close with evidence: the merged PR or commit, and the release that contains it when
  known.
- Never close an issue that an open PR is meant to fix.
- Close as not planned with a reason comment for out-of-scope, not-a-bug, and archive records
  ([label-policy.md mapping](label-policy.md#mapping-from-triage-backlog-terms)).
- Bulk sweeps:
  1. save a plan file listing every item, label change, and comment first;
  2. have every close comment checked by a reviewer that did not draft it (a human, or a
     different model);
  3. post in batches of at most 20.

## Post-merge

After every merge, run:

```bash
python3 scripts/maintainer/postmerge_issue_check.py <PR>
```

It is read-only. For each still-open issue it flags, close the issue with a comment that links
the merge, or comment what remains. GitHub closes only the issue directly after each closing
keyword, so a PR body uses one closing keyword per issue: `Fixes #1, fixes #2`, not
`Fixes #1, #2`. This complements
[land-pr §8](../../.agents/skills/land-pr/SKILL.md#8-verify-and-close-out).

## Milestones

Only the maintainer running a release sets milestones. Issues in an active release milestone are
triaged by that release's owner.

## Monthly stale sweep

Once a month, re-check every open issue with no activity for 60 days on current `main`. Either
re-confirm it with a comment, or close it with evidence. Larger sweeps use
[issue-sweep](../../.agents/skills/issue-sweep/SKILL.md).

## Public text rules

These apply to every comment, PR body, and document:

- no customer, deployment, or person names (existing label names are the exception);
- no internal agent names or aliases;
- no local machine paths (repository-relative paths are fine);
- no secrets or tokens.
