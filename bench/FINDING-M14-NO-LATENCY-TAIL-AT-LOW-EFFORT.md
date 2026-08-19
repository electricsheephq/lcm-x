# M14 — Window B is unreachable by capping: at low effort there is no latency tail left to trim

**Date:** 2026-07-25 · **Status:** decisive negative — rules out a strategy · **Cost:** zero (existing L3 raw data)

## 1. The idea being tested

Window B of the LAFS reference frontier is `latency < 26.9s AND accuracy > 51.0%`. At the low-effort
operating point we are **already above the accuracy floor** — 56.67% vs 51.0%, **5.7 points of slack** —
and blocked only on latency (51.6s vs 26.9s). Unlike window A, window B could in principle be reached by
*spending* accuracy we already hold: cap per-question memory time, lose some correctness, buy the speed.
M10 (search-flailing: the 8+ search bucket = 45% of questions at 257s) suggested a fat tail to cut.

## 2. The distribution says no

L3 (low effort), n=60, `memory_query_duration_seconds`:

| p10 | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|
| 34.3s | 40.7s | 49.4s | 58.3s | 73.4s | 80.3s | 160.3s | 51.6s |

**This is compact, not heavy-tailed.** The top 25% of questions consume only **36%** of total time and the
top 10% only **18%** — a genuinely fat tail would concentrate far more. Exactly one question exceeds 90s.

Capping per-question memory time:

| cap | resulting mean | questions truncated |
|---|---|---|
| 90s | 50.4s | 1/60 (2%) |
| 70s | 49.6s | 8/60 (13%) |
| 60s | 48.0s | 12/60 (20%) |
| 50s | 44.8s | 26/60 (43%) |
| 40s | 38.7s | 48/60 (80%) |
| 30s | 29.9s | 59/60 (98%) |
| **25s** | **25.0s** | **59/60 (98%)** |

To reach the 26.9s ceiling you must cap at ~25s, which truncates **98% of questions**. That is not
trimming a tail — it is cutting every question off mid-work, and the 59 affected questions currently
score 56%. Accuracy would not survive it.

**Verdict: window B is unreachable by capping. Strategy closed.**

## 3. Why — and this is the useful part

**Low effort has already captured the search-flailing win.** M10 measured the 8+ search bucket at 257s,
but that was at **xhigh**. Dropping to low effort collapsed the mean from 196.9s to 51.6s (M11) precisely
by removing the flailing M10 identified. There is no second helping: the tail a cap would cut is the same
tail the effort setting already cut.

This also sharpens M10's standing. Search-flailing is real, but it is an **artifact of high reasoning
effort**, not an independent pathology to attack separately at the operating point.

## 4. Consequence for the roadmap

The only live path to a non-zero score is **window A: +1.94 accuracy points at 51.6s** — i.e. M7 (#157).
Confirmed by elimination rather than assumption: capping is closed (this finding), effort is a settled
dial (M11), the contention correction is not established and would not rescue a score anyway (M11 §7),
and the static lane cannot reach its cliff (M13).

**M7 is not merely the best remaining option; after M14 it is the only one.** If M7 fails its gate, the
program needs a genuinely new mechanism, not another knob — and that should be said plainly rather than
absorbed by re-running variants.
