# Turn-path probe packet

Every spec and review packet for a change on the turn path carries these sections. The turn path covers assembly, compaction,
survival fit, ingest/re-ingest, the summary route, and the host-facing `compress`/`should_compress` contract. The packet changes
*what is compared*, not how many reviewers look. The culprit PRs behind the v0.27.0 respins had 16–21 reviews each.

## 1. Consumer enumeration
For every changed function, attribute or config key, list its call sites (`rg -n "<name>"` output, production and tests) and
how each one is affected. A reviewer walks this list; a consumer missing from it is a finding.

## 2. Previous-GA differential row
Paste the `tests/differential` summary for the change: the exact shapes and generated families run, oracle counts, and any
accepted delta used, with its predicate and reason. A new accepted delta needs its own justification. Blanket deltas are not
accepted.

## 3. Sibling shapes
List the shape family the change touches (see [fix-scope.md](fix-scope.md)) and the cells actually exercised.

## 4. Invariants
List the [invariant](../invariants.md) IDs the change could affect. For each one, give the test that would fail if the change
broke it.

## 5. Risk probes for the reviewer
Name concrete probes, each with an expected result: inputs, windows, and the state to compare. A cross-model reviewer is given
the probes. "Review this diff" is not a probe.

## Rules
- No behaviour change mixed with code moves in one PR.
- No hard PR-size cap.

## Golden traces

These deterministic traces pin ordered events and observed state before turn-path refactors:

| Family | What it pins | Regenerate from the repository root |
| --- | --- | --- |
| Compress | Compaction branch decisions, store activity, logs and engine state | `python -m tests.behaviour.compress_trace_harness --regenerate` |
| Reconcile | Ingest cursor decisions, store reads/writes, input preservation and engine state | `python -m tests.behaviour.reconcile_trace_harness --repo . --regenerate` |
| Assembly | Prompt/tail assembly, SQL and helper order, lineage and emission state | `python -m tests.behaviour.assembly_trace_harness --repo . --out tests/fixtures/assembly_trace.json` |

Use the test environment and dependencies for the repository. Review the first differing scenario and event before changing
a fixture. Inventory failures name the observed-attribute or scenario set to update deliberately; preserve completeness.
Regenerate once, attribute every changed event in the PR, never regenerate to hide a behaviour change. Name the commit or
behaviour change that explains each changed event in the PR description.
