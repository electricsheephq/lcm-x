# R2 consolidation checklist — the 95% bar, operationalized (owner granted publish authority 2026-07-29)

**Authority:** owner pre-authorized upstream push + maintainer tagging + PR closeouts once every gate below is
green and the architect's confidence is ≥95%. Fork-first review cycles, then upstream via PR #436.

## The consolidated base (supersedes F32 §2's "ship e99f342 exactly" — intent preserved, see F32 note)
`bench/w3b-on-wave1` @ e99f342 **+ PR #169** (the two Phase 1A ceilings) **+ #434 provenance markers** (its fields were ALREADY IN the wave-1 base at e99f342 — my earlier absence check used wrong grep terms, caught by the codex handoff; the F32 run therefore already validated the fix) **+ #164(a)** (store_id on summary hits — prevents
the measured 1.6% fail-close loss). Released artifact == validated artifact: validation is Phase 1B + the
sanity slice below, run ON the consolidated base.

## Gates (ALL must be green before upstream push)
1. ☑ PR #169 MERGED after full review cycle (61d5b14) (sol·max, running); mandatory fixes applied + re-reviewed.
2. ☑ #170: #434 provenance + #164(a) store_id fix landed (validator: 8/8 F32 fail-closed qids render clean); targeted tests green; store_id commit reviewed by the architect (cross-model: codex-authored → Claude review).
3. ☑ **Phase 1B** (F34): F31 instrument re-run on the consolidated base — A3 recall does NOT collapse across the
   ladder (shape, not level, per #167 acceptance); A2 returns non-empty at 8k+ with p50 within 2× of A3
   (#168 acceptance); latency curve still sub-linear and ≤ file-scan at top rung. → F34 verdict doc.
4. ☑ **V1 sanity slice** (F35 RED → citable fix, 8-round cycle, PR #174 → F36 GREEN: fail-close 16→0, gain real +18/p=4e-05, stable across the rebuild; full-500 confirm triggered): paired ~100q (the F26 slice, known baseline 44/100) on the consolidated base under
   F32 pins — flips within the measured noise reference (≤18 discordant, |net| ≤ 6). Full 500 re-run ONLY if
   the slice moves beyond that. → It did; **full-500 confirm GREEN (F37): 455/500, 0 fail-closes, +11 vs
   banked (p=0.061, reported as measured)** — release V1 number banked, kit slots filled.
5. ☑ **Fork mono-PR review cycles** (7 rounds, 73 findings: 35→11→6→8→3→4→6, all fixed/refuted/deferred-with-issue #172 #176 #177 #178 #179; CI green every round head; stopping rule declared+posted; fork #175 MERGED 2026-07-29): consolidated branch → fork PR → Codex + CodeRabbit + evaOS review bot;
   iterate until a CLEAN ROUND (no new HIGH+ findings), minimum 2 rounds, cap 5 (owner suggested 4–5;
   diminishing-returns stop allowed after 2 clean).
6. ☑ Release docs final: RELEASE-NOTES-R2 (candour lead, locked), README section, #436 body updated,
   evidence links resolve on the fork.
7. ☑ #436 branch (`upstream-wave-1`) updated (ccc76ee, merge not force-push; docs conflicts resolved — tool count 15 from merged code; CHANGELOG: upstream v0.20.0 stands, train under Unreleased) to the consolidated content (merge, no force-push if possible);
   CI green including 3.13.
8. ☑ Closeouts (#434 closed w/ prepared comment; #423 already closed; fork #169/#174/#175 merged): #434 closed w/ prepared comment (only AFTER its fix is verifiably in #436); #423 already
   closed (verified 2026-07-29); fork-side #169 + consolidation PR merged.
9. ☑ Architect's 95% statement (posted: stephenschoettler/hermes-lcm#436 comment 5115182904) written into the release notes PR comment: what was validated, what wasn't,
   known issues (#165 harness-side; anything open).
10. ☑ Upstream: pushed + maintainer tagged w/ review guide (2026-07-29); shepherding active, comment tagging the maintainer with the review guide, monitor per
    pr-shepherding standard (never push-and-abandon).

## Explicitly NOT gated on
Stage-2 session expansion (parallel capability lane; ships in a later train regardless of outcome);
upstream #436 merge itself (maintainer's timeline); the LongMemEval-V2 upstream reports (#6/#7 — informational).
