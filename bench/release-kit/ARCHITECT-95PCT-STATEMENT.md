# Architect's release statement — R2 (what was validated, what was not, and why confidence clears 95%)

*Posted with the upstream push per the consolidation checklist gate 9. Evidence: F20–F37 in `bench/`,
verdicts and review logs in `bench/release-kit/` and the fork PR #175 thread.*

## What this release's numbers mean, exactly

**LongMemEval-V1: 455/500 (91.0%), zero instrument fail-closes (F37).** Measured on commit `2edb8fc` under
the frozen F32 pin set (harness, transport binary sha, reader/judge models+efforts, dataset sha, store
snapshots verified unchanged pre/post). Paired against our previous base (444/500): net +11 on 29
discordant rows, exact McNemar p=0.061 — **we do not claim a statistically confirmed accuracy gain**; the
direction is consistent across three independent baselines (+11/+13/+12) with a flat same-code placebo
(−2, p=0.81), and the failure-enriched slice measured the same effect at p=4.0×10⁻⁵ (F36). The +11 is
genuine answer flips (the comparison arm had zero fail-closes to recover).

**Latency (−56.3 s/question, 22%, p<0.0001)** and **scaling (recall cliff eliminated at 389×, full-coverage
scan cost published, ANN successor filed as #171)** are unchanged from F34/M12 and carry their conditions
in the notes.

## Released == validated: the commit delta, stated plainly

The benchmark number was measured on `2edb8fc`. The released HEAD adds the fork-review-round fixes
(rounds 1–4 on PR #175: 35 → 11 → 6 → confirmation, every finding fixed, refuted in writing, or deferred
to a filed issue). Why the number still stands for the released HEAD:

- No round-fix alters default V1 delivered-hit selection. The deadline stops (scan, prescreen, lineage)
  fire only after an operation has already exceeded its budget — on the benchmark corpora scans complete
  in milliseconds. The delta-ref rebuild fires only on 64K response-cap eviction, which the single-shot
  bench path never reaches. The trajectory/state-semantic fixes live in a default-off arm
  (`state_semantic_quota=0`). The schema-downgrade gate, stress-checker, and bench-tooling fixes are not
  on the delivery path at all.
- This reasoning was checked independently twice per round: the implementing lane flags any
  delivery-affecting change as part of its acceptance contract, and the orchestrator re-derived the flag
  from the diffs before merging. Had any fix failed that test, the pre-declared rule was a paired
  sanity-slice re-run (the F36 precedent) before the round counted.

## What was NOT validated (published limits)

- **Scaling recall is shape-validated on a non-production embedder** (fastembed/384-dim); levels do not
  transfer, causes do (F31/F34).
- **The medium tier (7.4× store) has never been run** — it needs its own store build and is the next
  program phase, not this release.
- **Harness-side fail-open** (#165, upstream): retrieval transport failures score as capability zeros;
  our reports upstream (LongMemEval-V2 #6, #7) document the same class in the public harness.
- **The FTS arm is currently silent in delivered evidence on V1** (hit-mix 5,855 → 1 of 12,500; #172):
  net accuracy went UP because the vector arm covers, and we publish the mechanism rather than patch
  query semantics mid-release.
- **Summary lineage is not recorded at ingest** (#177): the summary arm contributes leads, not citable
  evidence, until that lands.
- Latent, CI-immune test-ordering leak in the stress smoke (#176).

## Why ≥95%

The three claims the release makes (latency established; V1 number with its p-value and zero fail-closes;
scaling shape with published cost curve) each rest on paired, pinned, twice-measured evidence with the
noise floor published alongside. The remaining 5% is concentrated where we say it is: environment
transfer (embedder, corpus regime) and the harness-side classes we've reported upstream — not in the
measurement pipeline, which caught and corrected its own errors eight separate times this cycle (the
correction trail ships in `bench/`).
