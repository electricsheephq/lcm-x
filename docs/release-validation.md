# Release validation

Use `scripts/validate_release.sh` as the local, offline release-confidence lane. It is a
pre-tag smoke lane, **not** the release gate: any release containing product code is
rc-first and must pass the live gauntlet in `bench/specs/RELEASE-READINESS-V1.md`
(Phase A all-tools matrix on a fresh isolated clone across privacy postures, Phase B
multi-agent P0/P1 sweep of the full release diff, Phase C 30+ turn `hermes acp` soak)
against a published `vX.Y.Z-rcN` prerelease before the GA tag. See AGENTS.md,
"Releases are rc-first". The script is offline by default: it does not call
model providers, does not mutate live Hermes config, forces `HERMES_HOME`,
`LCM_DATABASE_PATH`, externalized payloads, temporary files, and caches below a
fresh validation output directory, and writes all validation artifacts there.
The validator creates that packet with private owner-only permissions (`0700`
directories and `0600` regular files, subject to executable bits).

## Command

Prerequisites:

- Run from the repository checkout.
- Use a Python environment with `pytest` installed. If `python` on `PATH` is not the intended interpreter, set `PYTHON=/path/to/python`.
- The benchmark and stress gates are standalone-checkout safe: they provide the minimal Hermes Agent `ContextEngine` base class needed for deterministic local validation when Hermes Agent is not importable.
- On a PR branch with `origin/main` available, the whitespace/conflict-marker gate checks `origin/main...HEAD` instead of only uncommitted working-tree changes, then also checks the local working tree and staged diff. Override with `LCM_RELEASE_DIFF_BASE=<rev-or-range>` when validating against another base. If no changed `origin/main...HEAD` range is available but `HEAD` has a parent, the gate checks `HEAD^...HEAD` so a detached release checkout still validates the committed release diff.
- Validation clears inherited `HERMES_PROFILE`, `LCM_HERMES_BASE_DIR`, and `LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH`, then replaces inherited `HERMES_HOME` and `LCM_DATABASE_PATH` storage selection. It binds a fresh `<output>/hermes-home/lcm.db`, contains default and per-test externalized payloads plus temporary/cache files under the output directory, and never needs a developer's real Hermes database. Clearing global payload/base overrides lets tests create their own nested sandbox homes without escaping the validation output.
- Python validation runs with `PYTHONPYCACHEPREFIX` and pytest's `--basetemp` under the output directory. It discards inherited `PYTEST_ADDOPTS` so caller-selected pytest paths or plugins cannot escape the validation packet, disables pytest's cache provider, and records git status before and after validation so release runs do not silently dirty the checkout.
- The low-file-descriptor full gate lowers the limit to 1024 only when the current shell allows it; locked-down hosts keep their existing lower limit instead of failing before pytest starts.

```bash
scripts/validate_release.sh
```

Default smoke mode runs the local gates that should be cheap enough for routine operator use:

- adaptive `git diff --check` over `origin/main...HEAD` on PR branches or `HEAD^...HEAD` in detached/no-base release checkouts, plus local working-tree and staged diff checks
- Python compile checks for the plugin and release scripts
- shell syntax checks for maintained shell scripts
- focused pytest coverage for core, command, packaging, benchmark, and stress surfaces
- deterministic benchmark smoke with a synthetic fixture
- deterministic stress smoke

For pre-tag confidence, run:

```bash
scripts/validate_release.sh --full
```

Full mode adds the whole test suite, the low-file-descriptor pytest pass, and the release stress tier. It is intentionally heavier and still avoids provider/network side effects.

## Artifact shape

Each run creates a fresh directory, by default:

```text
/tmp/hermes-lcm-release-validation-YYYYMMDD-HHMMSS/
```

Important files:

- `validation-checklist.md` — scrubbed operator checklist, command summary, and before/after git status
- `logs/*.log` — stdout/stderr for each validation command
- `pycache/` — validation-time Python bytecode cache redirected away from the source tree
- `hermes-home/lcm.db` — isolated validation-only LCM storage; never a live profile database
- `hermes-home/lcm-large-outputs/`, `pytest-basetemp/`, `tmp/`, `cache/`, and `ruff-cache/` — other generated validation state
- `benchmark-smoke/summary.json` and `benchmark-smoke/metrics.jsonl` — deterministic benchmark artifacts
- `stress-smoke/stress-summary.md` and `stress-smoke/results/stress-results.json` — deterministic stress artifacts

The checklist is safe to paste into a release note or PR validation section after reviewing any local path values. Do not paste raw logs unless they have been scrubbed for local paths, secrets, and unrelated environment details.

## Checklist template

```md
## Release validation

- Command: `scripts/validate_release.sh [--full]`
- Mode: `smoke` or `full`
- Repo: `<branch>@<commit>`
- Output dir: `<validation artifact dir>`

### Gates
- [ ] git diff/whitespace check passed
- [ ] Python compile checks passed
- [ ] shell syntax checks passed
- [ ] focused or full pytest passed
- [ ] deterministic benchmark smoke passed
- [ ] deterministic stress smoke/release passed
- [ ] git status before/after validation reviewed
- [ ] rc tag published as a prerelease (`vX.Y.Z-rcN`) — or release is docs/bench-only
- [ ] Phase A / B / C gauntlet receipts linked for this exact rc tree

### Doctor triage
- [ ] `lcm_doctor` warnings were classified as `safe/ignore`, `inspect`, or `backup-first cleanup`
- [ ] no warning-only class was auto-cleaned without operator review

### Notes
- Skipped gates:
- Warnings reviewed:
- Recommended next release-confidence step:
```

## Warning-only boundaries

Doctor warnings should remain warning-only when the runtime cannot prove that mutation is safe:

- summary quality warnings: inspect retrieval/summary behavior; do not rewrite DAG state automatically
- lifecycle fragmentation: inspect first; only use explicit backup-first lifecycle cleanup for empty lifecycle rows
- payload-storage suspicion: inspect/restore missing payload files before deleting or rewriting anything
- context pressure: usually safe to ignore unless compaction is stuck or repeatedly firing
- legacy blank-source rows: normalized as `unknown` for compatibility; only run source normalization after a backup-first review
