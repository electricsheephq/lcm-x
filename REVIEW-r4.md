# S1-C.4 independent targeted semantic acceptance review
Verdict: PASS. No verified blocker.
Reviewer model: gpt-6-astra, native astra_reviewer. Original product author: gpt-6.1-sol; delta author: resumed Codex root. No authorship or advisory conflict.
Exact head4536f171bb4ffaf95d3ff186eb6f11e4706b54e5; prior reviewed head4be7e931d4e1a3450114fda4c57ad0ff39e9e761. One harness file35 additions/1 deletion. Branch feat/1186-s1-dependency, clean worktree verified. Rehydration PASS.

Evidence:
- SPEC.md:117-127 authorizes the synthetic hosted PM update exception.
- dependency_qualification.py:305-321 restricts the exception to PM, requires hosted CI and a scratch-contained home, exercises scanning-ON refusal, and verifies unchanged installed version and PM state.
- dependency_qualification.py:297-300,326-335 uses the actual interactive=False update, restores scanning in finally, then requires the companion scanner enabled and safe.
- dependency_qualification.py:338-344 requires new installed version, successful import probe, unchanged PM state before b and e-core-version-no-rebuild records.
- dependency_probe.py:14-25 equality includes generation environment, stamp, selected Python and committed presence. Dependency-consent paths unchanged.

Verified findings: none. Scope boundary: read-only source review only; no tests/host execution, skills, review CLI, descendants, edits or external writes. Prior review output not inspected and unchanged product scope not re-reviewed. Root retains REVIEW-r2.md and REVIEW-r3.md for unchanged prior coverage.
Recommended next steps: root obtains exact-head hosted CI and PM(b)/(e) receipts under existing authority. Source PASS alone does not establish CI/release/runtime/customer readiness or permit ready transition.
