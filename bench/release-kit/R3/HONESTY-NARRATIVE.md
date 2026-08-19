# What our gates killed, and why that's the product (R3 §3 draft 1)

A measurement program earns trust by what it refuses itself. Three episodes from this cycle:

## 1. A pre-registered gate killed our own feature (F45)
Session expansion — a capability we built, reviewed through five bot rounds, and merged dormant —
went to a pre-registered scale gate: net ≥ +8 at the top rungs with a ≤1.6× token cap. It came back
net +3/+6 at the top rungs at a 5.7–6.1× token multiplier. The registered verdict band said GRAY;
the disposition was KILL-as-configured, in writing, with the mechanism analysis attached (the
expansion is untargeted: it expands ~99% of sessions to recover single-digit completions at scale).
The feature stays dormant behind a flag; the targeted variant is an unfunded backlog issue with its
own registration required. We spent the tokens, got a negative answer, and published it.

## 2. Six executions, five root causes, one number (F39→F44)
The fast-scan latency gate ran SIX times before its number was banked: two KILLs on real mechanism
deaths, three integrity fail-closes (each one correct — a trigger migration, a timestamp-writing
integrity check, a broken residency key), and a mandatory confirmation run that caught a fix batch
making the measured surface 3.17× WORSE after the gate had passed. The banked 263.6 ms @ 19,829
sessions carries that whole history. Rule extracted and now binding: any fix batch touching a
measured surface re-runs the gate before merge.

## 3. Mid-run crash, append-only recovery, blind verdict (F47)
The V2 re-baseline's treatment arm died mid-run to a transient OS-level fault. The recovery never
touched the original run tree: a continuation loaded completed units read-only, re-ran the dead
unit fresh (discarding its unscored work rather than salvaging it), and REQUIRED the resume-time
dataset freeze to hash byte-identical to the original before proceeding. The verdict was rendered
by a blind adjudicator from instrument outputs only — which first returned CANNOT-ADJUDICATE,
naming two missing pieces of evidence we then had to produce as instruments, and which twice
detected and discarded verdict-shaped text that leaked into its context. The PASS that merged the
batch is auditable end to end, and the fail-close event is disclosed on the resulting number.

## 4. We published a wrong table, and our own instrument caught it in hours (F46→F48)
F48's per-category table shipped with a label scramble inherited from a summary tuple written days
earlier in F46: three category axes were misassigned, so the finding claimed a "single-hop −12
regression, suspected quota fault" — with a remediation workstream planned — when single-hop had
actually GAINED +8.1. The error was caught the same day by a decomposition agent whose first step
was to premise-check the claim against ground truth (0/1,986 category mismatches on re-derivation)
instead of trusting the published table. The correction was append-only on every surface that
carried the error — finding docs, scoreboard rows, the tracker — with the wrong table left visible
under its correction. The planned workstream built on the phantom regression was withdrawn in the
same commit, and what replaced it was the REAL finding the recomputation surfaced: an entire
retrieval arm near-inert in delivery (3/49,650 hits), which no one had been looking for. Rule
extracted and now binding: at every claim boundary, headline numbers are recomputed from the
rawest source on disk — especially when the summary being trusted is one we wrote ourselves.

The pattern across all four: pre-registration before spend, fail-closed instruments, author≠judge,
negative results published at the same resolution as positive ones — and corrections treated as
first-class findings, same-day and append-only, because a scoreboard you can trust is one whose
errors are visible. This is what we mean when we say the market's numbers are unanchored — not
that they are dishonest, but that nothing in how they were produced would have CAUGHT any of the
failures above.
