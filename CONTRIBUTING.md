# Contributing to LCM-X

Thanks for contributing to LCM-X, Electric Sheep's independent Lossless Context Memory extension
for Hermes. The project preserves its MIT license, original commit authorship, and links to
`stephenschoettler/hermes-lcm` as upstream evidence and attribution.

Read [AGENTS.md](AGENTS.md) before opening work. It defines the lossless-data, Hermes
compatibility, validation, review, and merge invariants for contributors, maintainers, and bots.

LCM-X is the project name. The installed plugin, skill, and directory remain
`hermes-lcm`, and the runtime context engine remains `lcm`; preserve those
compatibility identifiers unless a separately approved migration changes them.

## Workflow
### Start With An Accepted Issue

Use the GitHub issue forms in `electricsheephq/lcm-x`:

- bugs and regressions need current-`main` reproduction or a named mandatory-invariant
  violation, supported-path impact, and an affected commit SHA;
- features and design changes need maintainer acceptance before a PR becomes ready for review;
- concrete follow-ups need an issue instead of being buried in review comments.

Keep one independently reproducible problem per issue. Search open and closed issues and PRs
before filing, and retain upstream links and contributor credit. A label, bot assessment,
unmerged upstream PR, or reviewer severity is evidence—not proof.

### Branch And Commit

Create a focused branch from current `main` in your clone of
`https://github.com/electricsheephq/lcm-x.git`. Common branch prefixes are `fix/`, `feat/`,
`docs/`, `test/`, and `chore/`.

Use focused commits with clear subjects such as `fix: ...`, `docs: ...`, or `test: ...`. Do not
mix unrelated cleanup, generated secrets, customer data, live Hermes configuration, or local
authentication state into a change.

### Pull Requests

Use `.github/PULL_REQUEST_TEMPLATE.md` and link the accepted issue. Record:

- exact base and head SHAs;
- scope, non-goals, root cause, invariant owner, and why the change is at the correct layer;
- Hermes host-contract and lossless-data impact;
- real-behavior evidence separately from tests and CI;
- focused and default validation with exact commands and results;
- release-note impact and preserved upstream attribution.

Bug regressions should fail on the recorded base and pass on the candidate. Features must retain
backward-compatible defaults unless their accepted issue approves a breaking change and migration.

### Validation

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
git diff --cached --check
```

Workflow changes should also run:

```bash
actionlint
```

If your PR only touches a narrow surface area, include the focused command too. Example:

```bash
pytest tests/test_lcm_command.py -q
```

Packaging or install-flow changes should also verify the standalone user-plugin path:

```bash
export HERMES_HOME=/tmp/hermes-lcm-smoke
mkdir -p "$HERMES_HOME/plugins"
git clone https://github.com/electricsheephq/lcm-x "$HERMES_HOME/plugins/hermes-lcm"
# then enable `hermes-lcm` in plugins.enabled and set context.engine: lcm
hermes plugins
```

Run `actionlint` when workflows change. If a check does not apply, leave it unchecked and explain
why in the PR.

### Review And Landing

Maintainers use the protected-main `.agents/skills/land-pr/SKILL.md`:

- required checks must pass on the pinned `headRefOid` with trusted workflow identities;
- the protected `AI review exact-head` check must pass on the current head;
- routine/docs/benchmark changes need one acceptance receipt at 95 or above;
- governance, data-integrity, security, migration, persistence, profile/session, lifecycle,
  runtime, workflow-policy, unknown-risk, and Hermes-contract changes need distinct acceptance
  and adversarial receipts, each at 95 or above;
- every actionable review thread needs a terminal disposition and resolution;
- accepted issues, exact-head state, and product/security decisions are re-fetched before merge;
- merges use merge commits only—never squash, rebase, direct-main push, auto-merge, or bypass.
  The recorded #218 bootstrap transition is the sole non-repeatable exception.

After merging, maintainers verify the merge commit is current `main`, linked issue disposition,
and the required checks on the exact merge commit. Maintainers curate user-facing release notes;
commit lists or model output do not replace those notes.

## Automation Boundary

AI and bot output is proposal and evidence by default. Models may triage, reproduce, implement,
test, and review, but deterministic tooling must re-fetch live state before any authorized write.
Model output alone cannot close, label, assign, push, approve, or merge. Automated repair is
opt-in and limited to the exact accepted issue and current gate. Exact-head content-free AI
review receipts are evidence consumed by the protected deterministic gate; they do not create
write or merge authority. Public sensitive disclosure remains an explicit maintainer decision.

Use the read-only `.agents/skills/triage-backlog/SKILL.md` for one issue, pull request, or duplicate
cluster at a time. Invoking it does not authorize backlog sweeps or GitHub writes. Routine labels,
assignees, and milestones still need one exact maintainer authorization and live read-back;
public sensitive disclosure and close/reopen actions retain stronger owner and lifecycle gates.
Open executable upstream work remains an active continuation instead of silently becoming an
archive-only or closed record.
