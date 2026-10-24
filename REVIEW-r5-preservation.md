# Independent preservation delta review — PASS
Head: c8de1a45ec8bdea027a50c8713eee85528de7165; clean feat/1186-s1-dependency. Rehydration PASS.
Lane: targeted profile-normalization and legacy-copy preservation delta.
Author model: gpt-6.1-sol / high. Reviewer model: gpt-6-astra / high, native astra_adversarial r5_preservation_delta. Reviewer had no authorship or design-advice role.
Prior inspected coverage:4536f171..cd5939e0 plus pinned host profile/activation source; followup inspected only cd5939e0..c8de1a45. Original MERGE_BLOCKING finding fixed now.
Evidence: scripts/install.sh11 rejects noncanonical profiles with exit2 before target selection, links/directories or host calls. Explicit ASCII envelope excludes uppercase/padding/Unicode aliases/path separators. Accepted spellings are unchanged by host strip().casefold(), so original Default/padded counterexamples cannot bypass legacy guard. Tests model actual host normalization and prove rejected input creates no links/calls; enabled legacy copy plus installed new copy is preserved. Changed docs state canonical installer contract with actionable examples. Prior canonical profile/default/sticky/custom-root/legacy/optional-failure clearances preserved.
Verified findings: none remaining in inspected preservation delta.
Gaps: source-only; reviewer ran no tests/mutations and did not independently verify test/hosted reports. PASS covers named source delta only, not merge/release/runtime/customer readiness.
