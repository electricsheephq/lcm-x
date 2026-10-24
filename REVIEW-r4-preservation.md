# Distinct lcm-memory-preservation adversarial review
Verdict PASS. No verified blocker or preservation counterexample in inspected scope.
Reviewer model:gpt-6-astra (native astra_adversarial); original product author:gpt-6.1-sol; current harness delta by resumed Codex root. Distinct lane/reviewer from acceptance.
Exact HEAD4536f171bb4ffaf95d3ff186eb6f11e4706b54e5; base849da257b52fc9833955928816dd1585184988b1. Clean worktree/branch verified; Rehydration PASS.

Evidence:
- __init__.py:479, diagnostics.py:214, embedding_provider.py:2105 changes affect message strings only. No changed ingestion, compaction, session ownership, defaults or storage operations.
- scripts/install.sh:105 enables core before companion install. No companion-decline branch disables core.
- scripts/install.sh:114-123 preserves existing companion symlinks/directories; prints replacement commands without forcing. Existing core/skill preflight intact.
- Companion plugin.yaml:4 carries dependency; __init__.py:4 registers nothing. No runtime pip/model preparation.
- Legacy counterexample checked at __init__.py:547-569: preserves already-loaded engine and prevents second-copy write hooks; migration instructions remain explicit.
- Reviewer independently read GitHub commit0f86de4520abb672295882e430022ecfe4b769d9 parents and verified exact base/head. It is current PR merge tree. Initial reviewer characterization as prior candidate was corrected after readback. Supplied decline receipts show enabled core, disabled companion and successful full-text recall.

Verified findings:none. Source-review-only verdict supported by supplied synthetic receipts. No tests, edits, independent harness acceptance, skills, review CLI, descendants or external writes. Does not establish merge/release/runtime/customer readiness; full CI/readiness reconciliation root-owned.
