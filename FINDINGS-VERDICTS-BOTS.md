# PR #182 bot-finding verdicts

Reviewed PR: `100yenadmin/hermes-lcm#182`

Reviewed head: `95d20d0384db203a3a673b6890bce6939b4f829c`

Base: `cb92bf40c1d4c862cb56090792113b69d88660d4`

Mode: validate-then-fix, address existing feedback only. The fail-closed
migration posture is binding. No commit, push, PR reply, or thread resolution
was performed.

## 1. Mixed legacy digests with an active profile

- Source: CodeRabbit discussion `3673283174`
- Reported priority: Major / digest summary High
- Verdict: verified; fixed in the local working tree.
- Validation: a legacy table with active profile `A` and stored digests
  `{A, B}` migrated without raising and rewrote every row to `A`.
  `validate-before-fix.xml` records the failing regression.
- Fix: migration now adopts the active digest only when every stored legacy
  digest matches it. Any foreign digest fails closed before table replacement.
- Proof: `test_legacy_state_embedding_migration_refuses_mixed_active_rows`
  passes and verifies the legacy primary-key table remains intact after the
  rejected migration.

## 2. Ambiguous migration recovery text

- Source: Codex discussion `3673282194`
- Reported priority: P2 / digest summary Medium
- Verdict: verified; fixed in the local working tree.
- Validation: schema migration blocks `build_state_semantic_index()` before a
  rebuild can start, while the old error said only to rebuild.
  `validate-before-fix.xml` records the message mismatch.
- Fix: the fail-closed error now documents the recovery explicitly: discard
  the ambiguous legacy state embeddings and run a fresh backfill.
- Proof:
  `test_legacy_state_embedding_migration_refuses_ambiguous_inactive_rows`
  requires the fresh-backfill recovery text and verifies rollback preserves the
  legacy primary-key table.

## 3. Rebuild-guard TOCTOU

- Source: CodeRabbit discussion `3673283186`
- Reported priority: Minor / digest summary Medium
- Verdict: verified; fixed in the local working tree.
- Validation: the guard probe observed `Connection.in_transaction == False`
  on the reviewed head. `validate-before-fix.xml` records the failure.
- Fix: the active-profile read and same-profile rebuild guard now run under
  `self._lock`, after `BEGIN IMMEDIATE`. A rejection follows the existing
  exception path and rolls back the transaction.
- Proof:
  `test_forced_same_profile_rebuild_refuses_to_mutate_serving_rows` observes
  the guard inside the transaction, confirms the transaction is closed after
  rejection, and confirms serving rows and rankings remain unchanged.

## 4. Test helper staging-table name reuse

- Source: CodeRabbit discussion `3673283168`
- Reported priority: Trivial / digest summary Low
- Verdict: verified; fixed in the local working tree.
- Validation: `_downgrade_state_embeddings_to_legacy_primary_key` used the
  production migration table name
  `lcm_trajectory_state_embeddings_profile_scoped`.
- Fix: the test helper now uses
  `lcm_trajectory_state_embeddings_legacy_stage` for rename, copy, and drop.
- Proof: the complete state-semantic migration test file passes.

## Validation

- Failing-first evidence: `validate-before-fix.xml` — 3 expected failures.
- Bot regression subset: `focused-bot-regressions.xml` — 3 passed.
- Migration regressions: `migration-regressions.xml` — 30 passed.
- Full CI replica: `full-suite.xml` — 2711 passed, 35 failed, 1 skipped,
  12 xfailed. The 35 failure names exactly match the prior lane baseline:
  zero new and zero missing.
- Ruff on both changed source files: passed.
- `git diff --check`: passed.

Evidence directory:
`/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/lane182bots-logs/`

Proof boundary: this proves the local working-tree fixes and exact baseline
parity in the named CI-replica environment. It does not prove a pushed head,
current-head remote CI, merge readiness, merge, release, deployment, or runtime
adoption.
