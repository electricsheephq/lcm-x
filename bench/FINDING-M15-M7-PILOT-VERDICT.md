# M15 — M7 pilot: NO-GO on the frozen gate, with the failure mechanism precisely located

**Date:** 2026-07-25 · **Issue:** #157 · **Gate:** SPEC-M7-NEGATIVE-EVIDENCE.md §5 (frozen pre-run)
**Control:** L3 (paired, same 60q manifest, M9 parity verified: effort=low both sides, `require_evidence_gate`
the only intended diff, questions + haystack byte-identical)

---

## 1. VERDICT: NO-GO

| axis | L3 | M7 | delta | bar | result |
|---|---|---|---|---|---|
| **PRIMARY** abstention subset | 5/17 = 29.4% | **9/17 = 52.9%** | **+23.5 pts (+4q)** | strictly up | **met** |
| **FLOOR** answerable subset | 29/43 = 67.4% | 24/43 = 55.8% | −11.6 pts (−5q) | not down >2.0 pts | **BREACHED** |
| FLOOR, artifact-adjusted | 29/43 = 67.4% | 26/43 = 60.5% | **−7.0 pts** | not down >2.0 pts | **BREACHED by 5.0** |
| SECONDARY overall | 56.67% | 55.00% | −1.67 pts | reported only | net negative |
| latency (official) | 51.6s | 54.0s | +2.4s | flat-or-down | slightly worse |
| LAFS | 0.0000 | 0.0000 | — | — | — |

Neither subset move is individually significant (abstention p=0.219, answerable p=0.180) — but **the floor
was written as an absolute-magnitude bar, not a significance bar**, and −7.0 breaches it. Per the frozen
gate and the never-relax rule: **NO-GO.** This is the exact outcome spec §5 predeclared as
"primary up, floor breached."

**Mechanically the change worked perfectly:** gate-field emission **60/60 (100%)**.

## 2. Instrument note — provider errors changed the verdict's MAGNITUDE

M7 took **2** reader-side provider errors (`31f146ba`, `4329b535`) vs L3's 1. **Both were answerable
questions that L3 answered correctly**, so both landed inside the floor breach as false losses. Counting
them (M11 §5) moved the breach from −11.6 to **−7.0 points**. The verdict is unchanged; the reported
magnitude was wrong by 4.6 points until adjusted. **The rule earned its place twice in one day.**

## 3. The failure mechanism is NOT the one spec §4 predicted

§4 predicted we would import the static lane's spurious-unknown disease — the reader abstaining when it
should answer. **That is not what happened.** Of the 7 raw answerable losses:

- **all 7 carried status `directly_supported`** — not an absence status;
- **the reader went UNKNOWN on 0 of 7** (and on 0 of 7 in L3 either).

The reader remained willing to answer and simply answered *worse*. So the harm is **degradation of pack
quality on questions the mechanism was not even targeting** — not over-abstention.

**The targeting itself is good.** Cross-tab of `evidence_status` × gold class × correctness:

| status | on abstention | on answerable |
|---|---|---|
| `near_match_only` | **6/8 = 75%** | 1/4 = 25% |
| `contradicts_premise` | **2/2 = 100%** | 1/2 = 50% |
| `directly_supported` | 1/7 = 14% | 22/37 = 59.5% |

Absence statuses score **8/10 on the abstention class**, and **4 of the 5 abstention gains came via
`near_match_only`**. The channel works where it fires. Residual M7 failure is concentrated in
`directly_supported`-on-abstention (1/7 = 14%) — the agent still fails to notice a false premise 7 times.

**Why pack quality degraded:** two plausible contributors, both consistent with the data — (a) the
`## Evidence Assessment` section is prepended FIRST, occupying the position a weak 9B reader weights most
heavily (M1/M4), displacing real evidence; (b) the contract's mandatory targeted absence search consumes
agent effort and the ≤20-state span budget. Latency rising +2.4s supports (b).

## 4. The motivated next variant — and an honest bound on it

**Variant M7b: render `## Evidence Assessment` ONLY on absence statuses** (`near_match_only`,
`contradicts_premise`, `insufficient`), never on `directly_supported`. That leaves the 44
`directly_supported` packs unperturbed by rendering while keeping the signal exactly where it earns 8/10.

**Counterfactual upper bound** (directly_supported → L3 outcome, absence-status → M7 outcome):

| | value |
|---|---|
| overall | **63.33%** (vs L3 56.67%, M7 55.00%) |
| abstention | 52.9% (keeps M7's full gain) |
| answerable | 67.4% (keeps L3's level) |
| **LAFS @51.6s** | **0.6623** |
| distance to window A floor | **+4.7 pts above** |

**This is an UPPER BOUND, not an estimate, and the reason is important.** The counterfactual assumes
`directly_supported` packs would be unchanged. They would not be: the contract change makes the agent run
absence searches **before** any status is known, so its curation shifts everywhere. Measured on those 37
questions: spans +0.19, context +779 tokens (+3.9%), span counts differing on **11 of 37**. The
perturbation is real but modest (26/37 identical), so the rendering is plausibly the dominant term — but
**the true M7b result lies somewhere in [56.67%, 63.33%]** and must be measured, not assumed.

If M7b is run, the gate must be frozen with the same two-sided structure, and 60q remains a **screen**.

## 4b. `searches_per_question` — M10's metric, finally instrumented, and it confirms the diagnosis

| | L3 (gate off) | M7 (gate on) | change |
|---|---|---|---|
| overall | 5.63 | **6.90** | **+23%** |
| abstention gold | 5.18 | **7.94** | **+53%** |
| answerable gold | 5.81 | **6.49** | **+12%** |

The +53% on abstention is the mechanism working as designed — targeted absence searches cost searches, and
they bought +4 questions. **The +12% on ANSWERABLE is the important number:** the agent burns extra searches
proving absence on questions where nothing is absent. That is a **second, independent confirmation** that
`directly_supported` packs are perturbed regardless of what we render (the first was spans +0.19 / +779
tokens / 11-of-37 span counts differing), and it is why §4's 63.33% is a ceiling and not a forecast.

It also explains the +2.4s latency directly, and retires the M10-derived hope that this mechanism would cut
latency: **negative-evidence disclosure ADDS searches. It does not remove flailing.** (Consistent with M14:
low effort had already removed the flailing.)

**Design consequence for M7b — make the CONTRACT conditional, not just the rendering.** The current contract
demands a targeted absence search unconditionally, before any status is known, which is exactly why
answerable questions pay. A cheaper contract: *do the normal search first; only if it fails to surface the
presumed entity, spend one targeted confirmation search and report it.* That removes most of the +12% while
keeping the absence signal where it earns 8/10. M7b should carry both changes (conditional render +
conditional search), and the upper bound should then be closer to attainable.

## 4c. Instrument near-miss: the dispatch packet named the wrong repo

My packet named `lme-v2-official` as the harness repo. **L3 did not run from there** — it ran from a
worktree of a *different clone* (`/Volumes/LEXAR/repos/LongMemEval-V2` @ d8f1d90). `lme-v2-official` lacks
`READER_PROVIDER_JSON` provider/quantization pinning (L3 pins **fp8 / SiliconFlow / no-fallbacks**), lacks
the reader malformed-body retry and no-choices error class, and has *removed* the `--evaluator-base-url`
argument L3 passes. Running where I specified would have broken parity **in the serving-precision
dimension** — an invisible confound that no accuracy number would have revealed.

Caught by the executing agent's own parity discipline, which branched from L3's actual commit instead.
**Same class as L4's effort mismatch (M11 §7): the third config-parity near-miss in one day, and the second
caught only because parity is checked mechanically rather than assumed.** Lesson: a dispatch packet must
name the repo/commit the CONTROL actually ran from — derive it from the control's launch script, never from
memory of which clone is canonical.

**Verified, not assumed:** the declarative-only constraint held — `## Evidence Assessment` present and
rendered first in 60/60 rows, **zero `answer_policy` strings and zero reader directives** in rendered
context (2 audit flags were false positives quoting store UI copy). Tests 46 passed (baseline 35).

## 5. What this establishes for the program
- **A real, working read-time absence channel exists** and lifts the abstention class (+4q, 8/10 where it
  fires) using the harness's own dormant vocabulary. That is a genuine result about read-time absence
  signalling and is publishable as such — including the negative half.
- **The binding constraint is now pack-quality collateral, not the absence signal.** That is a materially
  more tractable problem than "the reader over-asserts", which is where M7 started.
- M14 established M7 was the only live path to a non-zero score. It failed as specified. **M7b is a
  diagnosed, targeted follow-on — not another knob** — but if it also fails, the honest conclusion is that
  the program needs a genuinely new mechanism, and that should be stated rather than absorbed by variants.

---

## 7. ★ CORRECTION — the −7.0 answerable floor was an ASYMMETRIC adjustment (superseded)

**Caught by the executing M7b agent, not by me.** §1 and §2 report an artifact-adjusted answerable floor of
**−7.0 points**. That figure **credited M7's 2 provider-error rows while leaving L3's 1 row uncredited** —
an asymmetric adjustment, which is exactly the error class that
`feedback_rescore_both_sides_before_delta_claims` exists to prevent. (That rule was written after an
asymmetric comparator fix manufactured a false +86 that was truly +24. I reaffirmed the rule earlier the
same day and then broke it.)

| treatment | answerable floor delta |
|---|---|
| **−7.0** (credit M7's 2, ignore L3's 1) | **SUPERSEDED — asymmetric, do not cite** |
| credit both arms' provider errors | **−9.3** |
| drop both arms' provider-error rows | **−10.0** |

**The NO-GO verdict is unchanged** — the error ran *against* M7, so the real breach is larger than reported,
not smaller. But the number was wrong by 2.3–3.0 points and **M7b's floor must not be benchmarked against
−7.0.** M7b's correct paired baseline is its own CONTROL arm (42/64 = 65.6%), which required no adjustment
at all: it recorded **zero** provider errors.

**Rule, sharpened:** adjust **both arms or neither**, and **state which** in the same sentence as the number.
A one-sided artifact credit is indistinguishable from cherry-picking, even when it is an honest slip — and
here it happened to flatter the mechanism under test.

## 8. ★ CORRECTION — my control ruling was right for the wrong reason

Also caught by the executing agent. I ruled that the M7b control should hold everything constant except the
gate, and justified it by claiming the control's `INSTRUCTION.md` carries "M7-era instruction improvements",
citing the bullet *"Prefer the CLI's top-ranked hits when selecting `trajectory_spans`…"*.

**That bullet is verbatim in d8f1d90 and in L3.** The control's `INSTRUCTION.md` hashes to `450668d0…` —
**byte-identical to L3's.** There were no M7-era instruction changes in it.

- The ruling's **outcome stands**: hold everything constant but the gate; carry no L3 numbers across.
- The ruling's **stated reason was false**, and non-comparability with L3 is **purely slice-based** (L3 ran a
  different 60q manifest).
- **A published inference is void:** I reported that "the control's 30.5% abstention vs L3's 29.4% confirms
  the instruction changes barely move abstention." There were no instruction changes to move it. Those two
  numbers are simply two different slices landing close — a weak consistency signal, nothing more.

**Lesson:** I verified the *conclusion* and asserted the *cause* from memory of what I had specified rather
than from the artifact. Same failure family as M17/M18 (asserting a property of the data instead of
enumerating it) — and the cause is what a later reader would build on.
