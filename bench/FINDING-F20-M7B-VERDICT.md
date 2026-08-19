# F20 — M7b: the diagnosed fix WORKED, and the channel still is not the binding constraint

**Date:** 2026-07-25 · **Issue:** #157 · **Gate:** SPEC-M7B-CONDITIONAL.md §5 (frozen pre-run)
**First finding under the F-series boundary (arch §6f).**
**Control:** paired, enriched 192q slice (128 abstention + 64 answerable), M9 parity verified while running —
zero measurement-bearing diffs, effort=low both sides, questions byte-identical by sha256, only the gate differs.

---

## 1. VERDICT: NO-GO on the primary. The floor passed.

| axis | control | M7b | delta | bar | result |
|---|---|---|---|---|---|
| **PRIMARY** abstention (n=128) | 39/128 = 30.5% | 43/128 = 33.6% | +3.1 pts (+4q) | up **and p<0.05** | ⛔ **p=0.585** |
| **FLOOR** answerable (n=64) | 42/64 = 65.6% | **45/64 = 70.3%** | **+4.7 pts (+3q)** | not down >2.0 pts | ✅ **improved** |
| SECONDARY overall | 81/192 = 42.19% | 88/192 = 45.83% | +3.64 pts | reported | up |
| latency (official) | 55.5s | 62.0s | **+6.5s** | flat-or-down | ⛔ worse |
| searches/q | 6.14 | 6.99 | +14% | flat-or-down | ⛔ worse |
| LAFS | 0.0000 | 0.0000 | — | — | — |

Discordants on the primary: **b=17, c=13**. Symmetric artifact adjustment (2 arm rows, 0 control rows) leaves
it at +3.9 / +4.0 pts either way. **Required for significance was roughly net +10 questions** (published in
advance). **NO-GO per the gate as frozen.**

## 2. The M15 diagnosis was correct and the fix worked

M15 located the M7 failure precisely: all 7 answerable losses carried `directly_supported` and the reader went
UNKNOWN on none of them, so the harm was **pack-quality collateral on untargeted questions**, not
over-abstention. The prescribed fix — render only on absence statuses, and make the absence search conditional
— **eliminated the breach entirely**: answerable went from **−7/−9.3 points (M7) to +4.7 points (M7b)**.

**That is a real, validated diagnostic win.** The mechanism no longer damages what it was not aiming at.

## 3. What the §5c criterion caught — and it is the most valuable line in this run

All **17** abstention gains were **specific-negative assertions; ZERO were bare UNKNOWNs.** The mechanism is
shaped correctly and the declarative-evidence design (never imperative) did what it was chosen to do.

**But:** M7b's UNKNOWN rate on the abstention subset **more than doubled — 13/128 (10.2%) → 28/128 (21.9%) —
and all 28 scored 0.** So the mechanism is simultaneously producing **15 additional worthless refusals** while
netting +4 real gains. Without the predeclared §5c check this would have read as "+4 gains, mechanism working";
in fact it is *both* helping and hurting inside the same subset. Bare UNKNOWN now stands at **0/47** across all
samples.

## 4. Implementation was clean — this is not an execution failure
- gate-field emission **192/192 (100%)**
- **zero** conditional-render violations (no section on `directly_supported`, no `answer_policy` leak)
- status distribution: `directly_supported` 117 · `near_match_only` 50 · `contradicts_premise` 22 ·
  `insufficient` 3
- provider errors: control **0**, arm **2** (both adjusted symmetrically per M15 §7)
- tests 50 pass (M7 baseline 46)

## 5. Latency: the M10-derived hope is now doubly retired
M7 added **+12%** searches on answerable questions; M7b still adds **+14% overall** (6.14 → 6.99) and **+6.5s**
latency even with the *conditional* search. **Negative-evidence disclosure ADDS work under both variants.** It
does not remove flailing — consistent with M14 (low effort had already removed the tail). Any future proposal
that assumes this family reduces latency should be rejected on these two measurements.

## 5b. ★ THREE REFINEMENTS from the executing agent's final report (one corrects §5)

**(a) The conditional search FAILED at its specific purpose — worse than §5 stated.** I wrote "+14% overall".
The paired breakdown is sharper and less flattering: **answerable searches +33% (5.34 → 7.08)** vs abstention
only +6% (6.53 → 6.95). **M7b now spends MORE searches on answerable questions than on abstention ones,
inverting the intended pattern.** SPEC §2b existed precisely to remove M7's +12% answerable tax; measured, the
tax got *worse*, and the +6.5s latency tracks it. The conditional-search design did not merely underdeliver —
it regressed the thing it targeted.

**(b) The render and the contract are SEPARABLE — and the contract alone carries part of the win.**
**5 of the 17** abstention gains carry `directly_supported` and **had no section rendered at all**. Those gains
come from the **contract change alone** (requiring the agent to search for and report absence), not from
rendering anything to the reader. This is a decomposition, not a new lever — it does not change the +4 net
effect size and so does not reopen the stop decision — but it is the single most useful lead for anyone who
revisits this area: *the contract is doing work the rendering is not.*

**(c) ★ Where the real ceiling is: `contradicts_premise` scores 19/19 = 100%.** Status × class on the abstention
subset: **`contradicts_premise` 19/19**, `near_match_only` 9/40, `directly_supported` 15/67, `insufficient` 0/2.

**When the agent correctly identifies a false premise, the reader gets it right every single time.** The channel
is not lossy at all — it is *rarely invoked correctly*: 19 correct premise-contradictions out of 128 abstention
questions. **The binding constraint is the AGENT's ability to detect a false premise, not the reader's handling
of the signal.** That relocates the problem one stage upstream and is the most actionable thing this run
produced. Read-time disclosure was the wrong stage to attack.

Also cleaner than my §1 note: **both M7b provider-error rows are abstention questions**, so the answerable floor
is **identical under all three adjustment conventions** — no adjustment choice can move it. The floor pass is
robust.

## 6. DECISION — stop the M7 family; the channel is not the binding constraint

Per the frozen outcome table ("primary fails → the channel is not the constraint") and the commitment recorded
in M15 §5 that a second failure would be stated plainly rather than absorbed into a third variant:

**The read-time absence channel is real, correctly shaped, and too small.** Two properly-run variants, the
second with its predecessor's failure precisely diagnosed and fixed, moved the abstention class by +4 questions
against a requirement of ~+10. **We are not one more variant away.**

- **Do NOT build M7c.** A third variant would be knob-turning against a measured ceiling.
- **Retain the code** — it is default-off, clean, tested, and the frontier product check (pending answer
  normalisation, §7) suggests it may help the consumer we actually ship. That is a product question, not a
  leaderboard one.
- **The leaderboard path via abstention is closed.** With M14 (capping closed), M13 (static cannot reach its
  cliff) and M11 (effort settled), the program's remaining leverage is **scale** — F21/#159, the 389× run —
  exactly as the strategy doc concluded independently.

## 7. Caveat on the product check
The frontier product-check numbers from this run are **provisional**: the frontier reader wraps 56–62% of
answers in LaTeX `\text{…}` and the scorer does not strip it, depressing **both** arms (control ≥+14,
arm ≥+11 exact-match recoveries). Direction may survive; magnitude is not yet usable. **Answer normalisation is
required before any frontier number is cited.** Filed as instrument bug #162.

**Second frontier caveat:** the transport records `requested_model=gpt-5.6-sol` but **`actual_model=None`**, so
Sol's serving identity is *requested, not confirmed*. Two independent reasons no frontier number from this run is
citable. Both must be fixed before the dual-consumer component of #160/#161 can be trusted.
