# Phase 1A reading protocol — written BEFORE the data lands

**Authored 2026-07-25, while the final ingest rung was still running and zero query results existed.** The point
is to fix the interpretation before seeing the curve, because Phase 1A is the experiment most likely to be read
the way we want it read. Consolidates F21's caveats into decision rules.

---

## 1. What Phase 1A is and is not

**Is:** a retrieval-only probe of one store scaled 389× (51 → 19,829 sessions; 514 → 199,641 messages), same 500
questions, 50-question fixed primary set, 3 reps, `LIMIT=25` matching the banked run's cap. Zero LLM spend.
Arms: `B` file-scan · `A1` lexical-only · `A2` semantic · `A3` semantic with FTS5-safe query · `A2u`/`A3u` the
same with a 3600s no-deadline budget (20 questions, fewer reps at the top rungs).

**Is not:** an accuracy result. No reader runs. Nothing here can move 444/500, and any sentence pairing a Phase 1A
number with a benchmark score is a category error.

**Metric integrity, verified before the run finished:** sessions are ingested under their **original** corpus ids
(`session_payload` sets `sessionId = rec["sid"]`) and `qeval.json` gold holds those same ids
(`answer_39900a0a_1`). Gold matching is therefore a direct set intersection — Phase 1A does **not** inherit the
positional-mapping trap that F27 §4b hit on the banked run. This was checked in the script, not assumed.

## 2. Pre-registered readings

Let recall@25 be the fraction of questions with **all** gold sessions returned, and latency be reported per arm
with its rung.

| observed | reading | action |
|---|---|---|
| `A2`/`A3` recall roughly flat 500 → 19,829, latency sub-linear | **The product thesis survives its first real test.** Our index holds at 389× where the corpus is no longer grep-exhaustible. | Publish as the scaling result; proceed to Phase 3. This is the outcome the program was built to test. |
| recall **degrades** with scale | **The most valuable outcome and the one we currently cannot see.** Retrieval has a scale ceiling we have never measured. | Do NOT soften it. It re-opens retrieval as a work item — which F27 closed *only for V1-small* — and it is the finding that justifies the whole scaling programme. |
| latency degrades steeply (super-linear) while recall holds | Index behaviour is fine, cost behaviour is not. | A LAFS problem, not a capability problem. Report as an engineering target, not a capability loss. |
| `B` (file-scan) matches or beats `A*` on both axes at 19,829 | **The honest negative: at this scale an agent grepping a folder is still competitive.** | Report it. This is the competitor we named in the strategy; if it wins we say so. |

## 3. Distrust conditions — check these BEFORE citing any number

**(a) ★ Arm B too fast at large N.** If `B`'s latency does not grow roughly linearly in N, it is probably
short-circuiting (early-exit on first match, `rg` cap, or a truncated file set). **A suspiciously flat arm B
invalidates the comparison in our favour** — i.e. it makes our arms look better by making the baseline look
broken. Verify B's latency scales before using it as the reference, and check `rows_returned` is not pinned at
the limit for every question.

**(b) The contention window.** The 100q cross-test held a search window 08:43:46Z–08:44:50Z that overlapped Phase
1A. **Before citing any slope, check whether any query-latency sample falls in that window** and exclude or flag
it. Latency is reported with its concurrency condition (standing rule); the honest label for anything overlapping
is CONTENDED.

**(c) Provider mismatch — this is not the production stack.** Phase 1A runs `fastembed` /
`BAAI/bge-small-en-v1.5` / 384-dim. Production and the banked 444 use **Voyage** `voyage-context-3` / 1024-dim.
So a recall number here is **not** a production recall number. State the provider in the same sentence as any
recall figure. A recall *degradation* is still informative (it is a scaling shape, and the shape is what we are
testing), but a recall *level* is not transferable.

**(d) Degraded/empty rows.** The instrumented search records `degraded`, `degraded_reason`, and `empty`.
Non-trivial degraded counts mean the arm fell back and the latency is not measuring what the label says. Report
degraded and empty counts alongside every arm, never just the mean.

**(e) The FTS5 artifact is the point of A3, not a bug to average away.** `A1`/`A2` send raw questions, which FTS5
rejects (`?`, apostrophe, comma, `&`, `$`) so hermes falls back to a LIKE scan; `A3`/`A3u` send the term-only
form. **`A3` vs `A2` is a finding about our own integration**, not a tuning knob — if A3 is much faster, we have
been shipping a query form that defeats our own index.

**(f) Uncensored arms are a different question.** `A2u`/`A3u` remove the query deadline (3600s) on 20 questions.
They answer "what would recall be if we never timed out", which is a *ceiling*, not a product number. Never mix a
`u`-arm recall with a deadline-arm latency in the same claim.

## 4. Reporting rules

- **Latency with its concurrency condition**, every time. Everything measured while another lane ran is CONTENDED.
- **Report the curve, not the endpoint.** Four rungs exist so we can see shape; a single 19,829 number discards
  the only thing that distinguishes a ceiling from a slope.
- **Degraded/empty/rows_returned alongside every mean.** A mean latency over degraded rows is not a latency.
- **No accuracy language.** No reader ran.
- **Per-arm provider stamp** (fastembed 384-dim) on every recall figure.
- If any pre-registered distrust condition fires, the affected number ships **with the caveat in the same
  sentence**, or not at all.

## 5. What Phase 1A cannot settle

It cannot tell us whether the answer layer improves, cannot move 444/500, and cannot validate Voyage-based
production recall. It also cannot speak to V1-small retrieval, which F27 already closed at 100% any-gold /
97.2% all-golds. Its whole job is the scaling shape of retrieval, on a non-production embedder, with no reader in
the loop. Keep the claim that size.

---

## 6. Pre-registered SEQUENCING option (decided before the schedule pressure, not during it)

The query chain is 28 sequential runs (~3,320 queries), ordered `B, A3, A2, A1` then the no-deadline
`A3u, A2u`. The uncensored tail carries a 3600s-per-query budget, so its duration is open-ended — plausibly
hours. The full-500 wave-1 release run is gated behind the *whole* chain, so the tail can delay the release
number by hours.

**The temptation, named in advance:** when the censored arms finish and the primary curve exists, it will be
tempting to start the release run immediately and let the uncensored arms run alongside it, because by then the
interesting curve is already in hand and the release number is what people are waiting for.

**The rule.** Releasing the gate early is permitted **only** on this reasoning, and it must be stated in the
report if used: the `u`-arms' claim is *"recall reaches X given unlimited time"* — a **recall ceiling**, not a
latency figure (§3f). Contention corrupts latency, not reachability. So if the `u`-arms are read for recall only:

1. the release run may start once `B/A3/A2/A1` have completed **all five rungs**, and
2. every `u`-arm number is labelled **CONTENDED** and **no latency figure from a `u`-arm is reported at all** —
   not as a mean, not as a p95, not "roughly".

**If either condition cannot be met, wait for the full chain.** In particular, if we later want a `u`-arm latency
number, this option is void retroactively and the arms must be re-run clean.

**What is NOT permitted:** starting the release run while any of `B/A3/A2/A1` is still running, at any rung, for
any reason. Those four arms are the primary curve and the entire latency claim; contaminating them to save
wall-clock would spend the experiment to save the schedule. The 100q cross-test already put a 64-second window of
contention into this run (§3b) and that single window is a permanent caveat on it — the cost of contention here is
not hypothetical.

**Decision procedure when the censored chain completes:** measure actual elapsed time for the censored arms, use
it to project the `u`-tail, and only then choose. If the projected tail is under ~1 hour, just wait — the option
buys little and costs a caveat.

---

## §3g–§3i — instrument amendments from the 2026-07-29 audit (accepted BEFORE any adjudication; F29 §5)

**(g) B×S0 is NON-FUNCTIONAL — excluded.** The per-question S0 filescan scopes were never materialised; all 150
B×S0 queries return 0 hits (`rg` exit 2). Any B×S0 row in the results is garbage; the B curve is its four ladder
rungs. Do not interpolate an S0 point for B.

**(h) Cross-arm cap unfairness — carry the bias direction on every A-vs-B recall comparison.** A-arms apply
LIMIT=25 to raw sub-session hits BEFORE session dedup (≈9 distinct sessions scored per query); B is scored on
session-level results. Session-recall comparisons are biased in B's FAVOUR. Report the bias with the number, or
recompute on a matched distinct-session basis. Latency comparisons are unaffected.

**(i) Persona-collapse confound — recall degradation is over-determined for persona-keyed questions.** One store
holding 500 personas makes first-person questions (single-session-user, knowledge-update, preference) partially
ill-posed for ANY memory system: rival personas' semantically-equivalent sessions legitimately crowd the top-25,
and gold-only scoring counts that as failure. Before attributing ANY recall drop to the index, quantify
rival-persona collisions per question type. Latency claims are unaffected. A recall drop concentrated in
persona-keyed types with high collision counts indicts the corpus design, not the index; a drop in
persona-neutral types is real signal.
