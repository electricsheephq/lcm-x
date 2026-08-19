# LCM-X Repository Instructions

These instructions govern maintainers, contributor agents, and review bots working in
`electricsheephq/lcm-x`. LCM-X is the independent Lossless Context Memory eXtension for
Hermes. Upstream `stephenschoettler/hermes-lcm` is an evidence and attribution source, not
an authority for writes or automatic merges.

## Non-Negotiable Invariants

- Lossless means lossless. Do not delete, truncate, overwrite, or make durable user context
  unreachable during automatic maintenance, migration, reconciliation, compaction, cleanup,
  or recovery.
- Prefer degraded coverage, a safe no-op, explicit warnings, or a tracked repair over
  destructive recovery when continuity is uncertain.
- Preserve message identity, chronology, conversation/session ownership, summary provenance,
  source coverage, tool-call/result grouping, real user turns, and the active fresh tail.
- Preserve contributor authorship, upstream links, license attribution, and original commits.
- Never commit credentials, tokens, cookies, customer data, live Hermes configuration, or
  local authentication state.
- Do not change customer runtimes, publish a release, write upstream, or announce LCM-X as
  canonical unless the user explicitly authorizes that separate gate.

## Canonical Work Graph

- Use a GitHub issue for accepted bugs, features, design changes, and concrete follow-ups.
- Keep one observable acceptance gate per issue and one issue per PR unless the issue explicitly
  approves a combined change.
- Link upstream reports and PRs as evidence while retaining contributor credit.
- Treat labels, CI, reviewer severity, and unmerged PR claims as evidence inputs—not proof by
  themselves.
- Give every verified review finding one terminal disposition: fixed now, accepted follow-up
  linked to an issue, accepted tradeoff/won't-fix, false/not-applicable, or escalated.

## Bugs And Regressions

- Reproduce a bug on the exact current `main` SHA or identify the mandatory invariant it
  violates before implementation.
- Record the supported path, impact, failing proof, expected behavior, and affected SHA.
- Add the smallest regression that fails on the base and passes on the candidate.
- Do not implement a `needs-repro` report merely because an upstream PR or issue claims a fix.
- Prioritize verified supported-path impact:
  - `P0`: active catastrophic data loss, disclosure, or unusable core path.
  - `P1`: severe integrity, security, crash, lockout, or release-blocking failure.
  - `P2`: significant bounded correctness or resource failure.
  - `P3`: limited bug, feature gap, or moderate maintainability/performance issue.
  - `P4`: documentation, tooling, portability, testing, or cosmetic work.

## Features And Design Changes

- Require an accepted feature/design issue before its PR becomes ready for review or can merge.
- The issue must state the problem, user/operator impact, proposed behavior, alternatives,
  acceptance criteria, non-goals, backward-compatible defaults, and documentation/config
  consequences.
- Identify the Hermes host contract involved: plugin API, `ContextEngine` behavior, lifecycle,
  profile/session identity, tool schema, config, model routing, or no host-contract change.
- Keep existing behavior as the default unless the accepted issue explicitly approves a
  breaking change and its migration path.
- Treat best-practice work as `P3`/`P4` unless current evidence proves a violated invariant.
- Do not add a dependency, schema, background service, persistence surface, or public contract
  when the existing path can satisfy the accepted behavior.

## Hermes Compatibility

- Test against the repository's supported Python versions and the Hermes `ContextEngine`
  import/contract exercised by CI.
- Keep profile and session state isolated; never infer one Hermes profile from another.
- When changing configuration, update the runtime loader, user documentation, bundled
  `skills/hermes-lcm/references/configuration.md`, examples, and regression tests together.
- When changing plugin tools or lifecycle behavior, update the bundled Hermes skill/reference
  material and test host-facing failure and fallback paths.
- State any minimum Hermes capability explicitly. Do not silently depend on an unreleased or
  unverified Hermes host behavior.

## Validation

Before publication, run the narrowest focused regression and the checks relevant to the changed
surface. GitHub Actions is authoritative for the full supported matrix.

Default repository checks:

```bash
pytest tests/test_lcm_core.py tests/test_lcm_engine.py tests/test_packaging_install.py -q
pytest -q
bash -lc 'ulimit -n 1024 && python -m pytest tests/ -q'
ruff check .
python -m compileall -q .
python -m py_compile scripts/import_lossless_claw.py
bash -n scripts/install.sh scripts/update.sh
git diff --check
```

Run `actionlint` when workflows change. Record exact commands and results in the PR.

## Automation Boundary

- Treat AI and bot output as proposal and evidence by default.
- Deterministic tooling must re-fetch the live issue, PR head, checks, reviews, threads, and
  authorization immediately before an authorized write.
- Model output alone cannot close, label, assign, push, approve, or merge.
- Automated repair is opt-in and limited to the exact accepted issue and current gate.
- Security and data-integrity code changes retain non-author human code-owner approval on the
  normal path. Public disclosure retains the stronger `triage-backlog` owner gate. Only the
  exceptional exact-head administrator path in `land-pr` may replace code-change approval for a
  merge; it never authorizes disclosure. Classification alone does not elevate routine reversible
  issue metadata to that stronger gate.
- Use `.agents/skills/triage-backlog/SKILL.md` read-only unless a maintainer explicitly
  authorizes one exact mutation; never use it for an automatic backlog sweep.
- Invoking a skill never creates write authority. Routine reversible issue metadata needs one
  exact maintainer authorization and live read-back; public sensitive disclosure and terminal
  lifecycle actions retain the stronger owner and lifecycle gates in `triage-backlog`.
- A readiness decision is advisory and never creates merge authority.
- Preserve open executable upstream work as an active continuation. Do not silently turn it into
  an archive-only record, supersede it, or close it while its continuation state is ambiguous.

## Review And Merge

- Use the read-only `.agents/skills/review-pr/SKILL.md` for readiness questions. Use
  `.agents/skills/land-pr/SKILL.md` only after explicit current authority names the PR number
  and exact head SHA to merge.
- Pin the PR `headRefOid`; checks and semantic review must cover that head or an explicitly
  bounded delta.
- Require one independent semantic review for changes involving data integrity, security,
  migrations, compaction, lifecycle/session identity, persistence, or a Hermes host contract.
- Do not merge with failing/pending required checks, unresolved actionable threads, a changed
  head, missing issue acceptance, or unowned product/security decisions.
- Never push directly to `main`, use auto-merge, or force-push/delete `main`.
- Normal landing requires the protected ruleset's non-author CODEOWNER approval. The specifically
  configured administrator may use its user-specific PR-only bypass for one exact-head merge only
  with explicit current admin-path authority, all other gates satisfied, and independent blind
  acceptance and adversarial `PASS` receipts scoring at least 95 on that head. GitHub's `--admin`
  flag is a broad, non-atomic bypass rather than an approval-only server primitive, so authority
  must explicitly accept that residual risk and all gates must be re-fetched immediately before
  and verified after merge. This exception never permits a direct push or a configured
  broad/role/always bypass, and maintainers must not intentionally waive any non-approval gate.
- Use merge commits for PRs so contributor and upstream commits remain intact. Do not squash
  or rebase-merge into `main`.
- Merge deterministically with the pinned head:

```bash
gh pr merge <PR> --merge --match-head-commit <HEAD_SHA>
```

- After merging, verify the merge commit, linked issue state, final `main` SHA, and required
  post-merge checks. Thank external contributors and preserve uncertain issue reports rather
  than closing them speculatively.

## Maintainer And Bot Roles

- `@100yenadmin` and `@Tosko4` are code owners. The PR author cannot satisfy their own required
  approval on the normal protected path.
- Bots and agents may triage, reproduce, implement, test, review, and prepare merge evidence.
- Bots and agents may not bypass checks, review-thread resolution, issue acceptance, semantic
  review, or a product/security owner decision. The only approved exception is the explicit,
  receipt-backed, user-specific PR-only administrator path defined above and in `land-pr`, with
  its broader technical bypass behavior and residual race disclosed to the authorizing owner.
- Maintainers own feature acceptance, priority, compatibility decisions, terminal dispositions,
  and releases.
