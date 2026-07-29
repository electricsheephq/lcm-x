# PR #183 bot findings — verdicts

- Reviewed head: `418db28bcc85f24581cec6de75d6e8e91073f3b0`
- Base: `cb92bf40c1d4c862cb56090792113b69d88660d4`
- Mode: validate-then-fix; two Highs first; no push
- Raw source: `/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/prbots-raw/pr183_{comments,reviews}.json`
- Logs: `/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/lane183bots-logs/`

## 1. High — injected `OR` glue becomes a scored term

Verdict: **Not applicable as stated; invariant proof strengthened.**

At the reviewed head, `extract_search_terms()` already rejects every
case-insensitive member of `_BOOLEAN_OPERATORS`, so the injected uppercase
`OR` tokens do not reach fetch-limit or directness scoring. The precheck
produced terms `["dog", "vet", "appointment"]` and a directness score of `0.0`
for `"or hall or"`. Added
`test_search_prose_mode_operator_glue_contributes_zero_score` to make that
ranking invariant explicit. CodeRabbit comment `3673449204` is a duplicate
test-coverage suggestion and is included in this disposition.

Evidence: `high1-precheck.log`, `highs-fixed.log`.

## 2. High — Unicode-symbol LIKE fallback escapes the flag

Verdict: **Fixed.**

Both `MessageStore.search()` and `SummaryDAG.search()` routed Unicode symbols
to LIKE while `fts_prose_mode=False`. Failing-first tests reproduced both
routes. Symbol preservation is now passed to `requires_like_fallback()` only
when prose mode is enabled (and, for messages, the caller did not compose FTS
operators). Added flag-off route regressions for both backends and moved the
symbol-preservation assertions onto the flag-on path.

The original byte probe covered only an ASCII prose question. Its documented
blind spot was any Unicode-symbol query that distinguishes the historical FTS
route from the new LIKE route. The corpus now includes `licensed ©` plus a
symbol-bearing and symbol-free row.

Evidence: `highs-failing-first.log` (2 expected failures), `highs-fixed.log`.

## 3. Medium — trailing punctuation breaks symbol LIKE terms

Verdict: **Fixed.**

`sanitize_like_query("Find ©?")` previously yielded the term `©?`. Edge
question/exclamation punctuation is now removed when the token retains other
content, while punctuation-only fallback queries such as `???` remain intact.
The parameterized symbol search now exercises `©?`, `€?`, and `™?`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`punctuation-delta.log`.

## 4. Medium — lowercase `and`/`or` disappear before prose classification

Verdict: **Fixed.**

Classification now uses word tokens that retain ordinary lowercase
conjunctions; scoring extraction still excludes boolean-looking operator
tokens. `cats and dogs are common pets today` is now classified as prose and
builds `cats OR dogs OR common OR pets OR today`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## 5. Medium — conversational lead words leak into the disjunction

Verdict: **Fixed.**

Added `find`, `please`, `recall`, and `remember` to the prose stoplist while
retaining them as classification lead words. Parameterized regressions prove
each form reduces to the requested signal; for example,
`Can you remember my PIN?` now builds `PIN`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## 6. Medium — term-cap assertion proves formatting, not the bound

Verdict: **Fixed.**

The test now extracts actual search terms and asserts their count is bounded by
the module's `_PROSE_TERM_LIMIT`; it no longer hard-codes `12` via
`split(" OR ")`.

Evidence: `medium-low-fixed.log`.

## 7. Low — ranking test contains a tautological assertion

Verdict: **Fixed.**

The no-op truthy-set assertion was replaced with explicit ordered evidence:
the target must rank first, and the remaining ranked IDs must be exactly the
two seeded distractors.

Evidence: `medium-low-fixed.log`, `focused-parity-rerun.log`.

## 8. Low — balanced precision guard ignores curly quotes

Verdict: **Fixed.**

The prose classifier now treats a retained ASCII quoted phrase or a balanced
smart-quoted phrase (`“…”`) as a precision signal. The regression proves
smart-quoted prose remains conjunctive instead of becoming an OR query.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## Final gates

- Focused parity: **411 passed** (`focused-parity-rerun.log`).
- Full isolated CI replica, Python 3.11 and `ulimit -n 1024`:
  **2755 passed, 1 skipped, 12 xfailed** (`full-suite-final.log`,
  `full-suite-final.xml`).
- Ruff: **passed** (`ruff.log`).
- Compileall, script py-compile, shell syntax, and `git diff --check`:
  **passed** (`compileall.log`, `py-compile.log`, `bash-syntax.log`,
  `git-diff-check.log`).
- Final flag-off base/candidate outputs: **byte-identical**, 2,104 bytes each,
  SHA-256
  `d9cf3621ed51669eeaff13642b8805d393e45838e351fd1bf236defd0b9e3219`
  (`flag-off-byte-identity-final.log`).
- No commit, push, PR reply, thread resolution, merge, deploy, or release was
  performed.
