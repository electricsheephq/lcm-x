---
name: issue-sweep
description: Bounded, maintainer-authorized LCM-X issue sweep that drafts triage read-only, gets every drafted comment checked, and applies only the authorized labels and comments in batches of at most 20.
---

# Issue Sweep

Put triage onto GitHub for a bounded set of open issues: the labels from
`docs/maintainers/label-policy.md` and one comment per issue that follows the comment rules in
`docs/maintainers/issue-triage.md`. A maintainer starts each sweep, names its scope, and
authorizes its saved plan file; nothing here runs as an automatic backlog sweep. Classification of
each issue uses `.agents/skills/triage-backlog/SKILL.md`, which stays read-only.

## Minimum Capability

Drafting requires live read access to the repository, its issues, labels, milestones, comments,
and linked pull requests. Applying additionally requires issue write access. Stop if any required
read is unavailable.

## 1. Count

Record the exact current `main` SHA and count open issues that:

- lack a P-label (`P0`-`P4`);
- lack a type label;
- have no comment;
- come from an outside reporter who is waiting for a reply.

Save the counts; they are the "before" numbers.

## 2. Scope

From the counted issues, exclude:

- issues in an active release milestone (that release's owner triages them);
- issues another maintainer owns (assigned, or claimed in a comment).

Name the remaining scope and keep it bounded; it does not grow during the sweep.

## 3. Draft Read-Only

For each issue in scope, run `triage-backlog` without writes and map its packet to:

- labels per `docs/maintainers/label-policy.md` (exactly one P-label, one type label, at most one
  domain label, status labels only with their meaning);
- one comment per the comment rules in `docs/maintainers/issue-triage.md`, including the public
  text rules.

An issue that `triage-backlog` returns as `OWNER_GATE`, or whose right action is a close, is not
labeled or commented by this sweep; list it in the plan for a maintainer.

## 4. Second Check

Every drafted comment is checked by a reviewer that did not draft it: a human, or a different
model. Fix or drop each comment the reviewer rejects.

## 5. Save The Plan

Save the plan file before any write: issue number, current labels, labels to add, comment text,
reviewer verdict, and the items left for a maintainer. The maintainer authorizes this exact plan;
the plan is the authorization record for the writes below.

## 6. Apply In Batches

Apply in batches of at most 20 issues. For each issue:

1. re-read its live labels, state, and milestone;
2. stop on that issue if it closed, changed scope, or already received a P-label or comment that
   the plan did not expect;
3. add the planned labels only; never remove a label, and never add a second P-label;
4. post the planned comment;
5. read back the labels and comment.

## 7. Recount And Report

Repeat the Section 1 counts and report before/after numbers, the issues changed, the issues
skipped with reasons, and the items left for a maintainer.

## Write Boundary

This skill writes labels and comments only, and only those in the authorized plan file. Closes and
reopens follow `docs/maintainers/issue-triage.md` and the close gate in the `triage-backlog`
Write Boundary. Never set milestones, assignees, or issue state. Invoking this skill never
authorizes a write; the maintainer's authorization of the saved plan does.
