# Independent semantic reviews

Author model named by the approved spec: Codex gpt-6.1-sol high.
Reviewers dispatched using GPT-6-Astra high, fresh independent contexts; no authorship/advice involvement. Read-only: no test execution, edits, descendants or GitHub writes.

| Lane | Candidate | Delta | Verdict | Inspected evidence |
|---|---|---|---|---|
| Acceptance | 869a87500a70588aa0e30e03066f62fb3b6f191f | BASE two-query join-order change | PASS | Identity/archive predicates, float32 byte/row alignment, deterministic rank key, coverage calculation, fixed-ID VM steps and exactness regression |
| Adversarial isolation/fallback | 869a87500a70588aa0e30e03066f62fb3b6f191f | BASE two-query join-order change | PASS | Identity/hash isolation, archived and malformed storage rejection, int8 decoder, no-NumPy path, binary-survivor rescoring |
| Acceptance delta | 59b2289558521289a71a0c959ce7a7b9854d0639 | 869a8750 → bounded multi-value staging | PASS | Same string normalization, sequence, PRIMARY KEY deduplication, INSERT OR IGNORE, unique table names, empty input, finally cleanup; helper consumers do not use staging cursor metadata or staging rowids |
| Adversarial staging | 59b2289558521289a71a0c959ce7a7b9854d0639 | 869a8750 → bounded multi-value staging | PASS | Existing 500-ID chunks fit historical999-bind cap; exact set before table is yielded; unchanged exception cleanup and caller transaction boundaries |

No verified findings. Existing cleanup-test gap: its sqlite_master query does not inspect the temp schema. Both reviewers found unchanged finally-DROP control flow; this is a pre-existing evidence limit, not a candidate defect or reason to widen this fix. Dedicated end-to-end no-NumPy/binary-survivor execution is not established by these source-only reviews; affected existing two-stage tests pass in the focused suite.

These PASS verdicts cover the named semantic surfaces only, not performance thresholds, CI, merge, release, runtime or customer readiness.

Targeted serialization delta: acceptance PASS at `cdd4c2629ebca159de3f3093227206935d4ee06b` against `59b2289558521289a71a0c959ce7a7b9854d0639`. Complete diff removes one space at vector_store.py:1165. Prior semantic coverage stands;95affectedtrace checks pass. Performance stays attached to measured59b22895; final full CI is separate. No additional finding.
