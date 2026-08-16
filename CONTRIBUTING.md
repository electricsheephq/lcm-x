# Contributing to LCM-X

Thanks for contributing to LCM-X, Electric Sheep's independent Lossless Context Memory extension
for Hermes. The project preserves its MIT license, original commit authorship, and links to
`stephenschoettler/hermes-lcm` as upstream evidence and attribution.

Read [AGENTS.md](AGENTS.md) before opening work. It defines the lossless-data, Hermes
compatibility, validation, review, and merge invariants for contributors, maintainers, and bots.

## Start With An Accepted Issue

Use the GitHub issue forms in `electricsheephq/lcm-x`:

- bugs and regressions need current-`main` reproduction or a named mandatory-invariant
  violation, supported-path impact, and an affected commit SHA;
- features and design changes need maintainer acceptance before a PR becomes ready for review;
- concrete follow-ups need an issue instead of being buried in review comments.

Keep one independently reproducible problem per issue. Search open and closed issues and PRs
before filing, and retain upstream links and contributor credit. A label, bot assessment,
unmerged upstream PR, or reviewer severity is evidence—not proof.

## Branch And Commit

Create a focused branch from current `main` in your clone of
`https://github.com/electricsheephq/lcm-x.git`. Common branch prefixes are `fix/`, `feat/`,
`docs/`, `test/`, and `chore/`.

Use focused commits with clear subjects such as `fix: ...`, `docs: ...`, or `test: ...`. Do not
mix unrelated cleanup, generated secrets, customer data, live Hermes configuration, or local
authentication state into a change.

## Pull Requests

Use `.github/PULL_REQUEST_TEMPLATE.md` and link the accepted issue. Record:

- exact base and head SHAs;
- scope, non-goals, root cause, invariant owner, and why the change is at the correct layer;
- Hermes host-contract and lossless-data impact;
- real-behavior evidence separately from tests and CI;
- focused and default validation with exact commands and results;
- release-note impact and preserved upstream attribution.

Bug regressions should fail on the recorded base and pass on the candidate. Features must retain
backward-compatible defaults unless their accepted issue approves a breaking change and migration.

## Validation

Run the narrowest focused regression first. GitHub Actions is authoritative for the full Python
matrix. The repository defaults are:

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

Run `actionlint` when workflows change. If a check does not apply, leave it unchecked and explain
why in the PR.

## Review And Landing

Maintainers use the protected-main `.agents/skills/review-pr/SKILL.md` for read-only readiness.
Only an explicit instruction to merge PR N at exact head H triggers
`.agents/skills/land-pr/SKILL.md`:

- required checks must pass on the pinned `headRefOid` with trusted workflow identities;
- the normal path requires one non-author code owner to approve the current head;
- every actionable review thread needs a terminal disposition and resolution;
- data-integrity, security, migration, compaction, persistence, profile/session, lifecycle, and
  Hermes-contract changes need independent semantic review;
- accepted issues, exact-head state, and product/security decisions are re-fetched before merge;
- merges use merge commits only—never squash, rebase, direct-main push, or auto-merge;
- the configured administrator's exceptional PR-only path requires explicit exact-head admin
  authority, all other gates, and independent blind acceptance and adversarial `PASS` receipts
  scoring at least 95. GitHub's `--admin` flag is technically broad and non-atomic, so the
  authorizing maintainer must accept that residual risk after the final live barrier. The path
  cannot enable direct pushes or a configured broad/role/always bypass.

After merging, maintainers verify the merge commit is current `main`, linked issue disposition,
and the required checks on the exact merge commit. Maintainers curate user-facing release notes;
commit lists or model output do not replace those notes.

## Automation Boundary

AI and bot output is proposal and evidence by default. Models may triage, reproduce, implement,
test, and review, but deterministic tooling must re-fetch live state before any authorized write.
Model output alone cannot close, label, assign, push, approve, or merge. Readiness is advisory
and never supplies merge authority. Automated repair is
opt-in and limited to the exact accepted issue and current gate. Security or data-integrity code
changes retain the repository's human approval gate on the normal path, and public disclosure
retains the stronger triage owner gate. Only the exceptional exact-head administrator landing
contract may replace code-change approval for a merge; it never authorizes disclosure.
Classification alone does not elevate routine reversible issue metadata to that stronger gate.

Use the read-only `.agents/skills/triage-backlog/SKILL.md` for one issue, pull request, or duplicate
cluster at a time. Invoking it does not authorize backlog sweeps or GitHub writes. Routine labels,
assignees, and milestones still need one exact maintainer authorization and live read-back;
public sensitive disclosure and close/reopen actions retain stronger owner and lifecycle gates.
Open executable upstream work remains an active continuation instead of silently becoming an
archive-only or closed record.
