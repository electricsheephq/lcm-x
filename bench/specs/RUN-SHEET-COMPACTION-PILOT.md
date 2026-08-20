# RUN SHEET — Compaction-Quality Pilot (P0): native Codex vs LCM vs compressor vs handoff

Status: DRAFT-FOR-REGISTRATION — pins finalized at launch from smoke outputs (S0–S3);
this sheet lands on main BEFORE the six-run spend. Owner mandate 2026-08-20 (ultracode):
the gpt-5.4→5.6 migration recommendation must rest on measurement; native codex
compaction is a black box; every run doubles as reverse-engineering material.

## 1. Question + regimes
Which regime should carry LONG coding-agent runs on gpt-5.6 after the 08-31 gpt-5.4
codex-auth retirement?
- **R1** Codex CLI native compaction (server-side remote_compaction_v2; black box —
  behavioral probes only; structural observables: window_number counter,
  replacement_history composition, ~99s measured stall per compaction, token edges).
- **R2** Hermes host + LCM context engine (`context.engine: lcm`), provider openai-codex.
- **R2b** Hermes host + built-in compressor (`context.engine: compressor`) — the
  intra-host control; ONE config flip from R2.
- **R3** R2 material phase → session exit → FRESH hermes session on the same LCM home →
  probes only (the cross-session handoff native compaction cannot do). **Declared
  mechanism**: a fresh session inherits no active context; R3 measures whether the
  bundled recall policy + `lcm_recall` (whole-database by design) surface prior-session
  facts. The driver records per-probe whether any `lcm_*` retrieval tool fired
  (diagnostic, not a score), so "policy never triggered" is distinguishable from
  "retrieval failed".
Model: gpt-5.6-sol only in P0 (luna → P1). R4 (long-context >272K) deferred to P1.

## 2. Material + probes (D1/D2 of the approved design)
- One FIXED synthetic coding-session script: 35 turns × ~17K tokens of varied,
  non-repetitive filler + a one-line bounded task per turn; identical across all arms.
  Sized to cross the native trigger band (86–93% of 258,400) ≥2 times.
- 30 canaries (5 info classes: decisions / file-paths / constraints / tool-output facts /
  user preferences; ×2 per class per epoch; 3 epochs by token depth) + 5 hallucination
  traps. One registered salt for all P0 runs (see §3); pre-material zero-hit assertions
  replace salt rotation as the contamination guard. Probes never contain the value.
- Post-hoc epoch re-binning against MEASURED compaction boundaries; canaries within ±1
  turn of a boundary excluded + disclosed. **Cross-arm contrasts are computed on the
  INTERSECTION of each compared pair's surviving canary sets** (per-arm exclusion lists
  published; asymmetric exclusions must never let arms be compared on different canaries).
- Scoring: mechanical 3-way per probe — CORRECT (normalized substring) / ABSTAIN
  (declared regex list) / HALLUCINATE (any other concrete answer; unparseable lands here,
  disclosed). Negative probes score abstention as correct. No LLM judge in P0.

## 3. Runs + order (D4)
S0 cost/auth smoke → S1 resume-continuity smoke → **material freeze** (gen_material with
the registered seed; shas pinned HERE, before any behavioral smoke touches the measured
surface) → S2 pty smoke (frozen material) → S3 denominator gate → then the five 5.6 runs
in an order drawn by the registered seed (R2-A and R2-A′ adjacent by design) + the
perishable gpt-5.4 run last (see §3b). **ALL P0 runs share ONE registered salt** — full
script identity across every arm (the same value-identity argument that binds the A/A′
pair binds every cross-arm contrast). Contamination is guarded directly instead of by
salt rotation: each run starts from a fresh HERMES_HOME/LCM store (or fresh codex
session), and the driver asserts PRE-MATERIAL that zero canary values exist in the
store/session — a nonzero hit aborts the run fail-closed.
Fresh HERMES_HOME/LCM store per run; per-turn append-only results JSONL (the V1M
Amendment-2 lesson); ~2h wall per run, staggered. Single-run-per-regime remains a declared
P0 limitation: seeded order randomization reduces but cannot eliminate time-of-day/serving
drift confounds — P1 replicates.

### 3b. The gpt-5.4 run is NOT an R1 datapoint (re-scoped)
gpt-5.4's Codex OAuth window is ~1,050,000 — the 35-turn (~600K) material cannot cross its
compaction band, so it would never compact. It is banked as an **R4-class long-context
reference datapoint** (same material, same probes, no compaction expected — assert zero
`compacted` events), explicitly excluded from every compaction contrast. Its value:
retention WITHOUT compaction at 600K depth, the natural ceiling reference for P1.

## 4. Denominator gate (fail-closed, pre-spend) — PARITY REQUIRED for contrasted arms
R1 asserts model_context_window == 258,400 from token_count telemetry. R2/R2b capture the
effective window via `lcm_status` mid-session. **Any G3-registered contrast requires EQUAL
effective windows between its arms** — "declared inequality" is NOT sufficient for a
contrast (a 372K-vs-258K comparison would confound window size with regime). Because of
**lcm-x#263** (the 372K substring-match bug), the LCM arms pin their effective budget to
parity explicitly (mechanism chosen and recorded at pin time: `LCM_ABSOLUTE_THRESHOLD_TOKENS`
or an explicit context_length override — whichever `lcm_status` proves effective); S3
verifies the pinned value before spend. The #263 fix itself is a separate product PR,
never bundled into the pilot.

## 5. Pre-declared success criteria (the pilot proves MEASUREMENT, not a winner)
- **G1 integrity**: all three R1 markers parse on ≥2 boundaries; **≤10% of probes carry
  the `unparseable` flag** (every probe classifies by construction — the flag is the real
  integrity signal, so the gate binds on it, not on classifiability); ≤20% of canaries
  epoch-ambiguous.
- **G2 noise**: R2 A/A′ retention discordance published as THE band; >25pt overall →
  instrument too noisy, redesign before P1. **Retention statistic (the G2/G3 quantity),
  defined**: retention = CORRECT ÷ (30 canaries − excluded-per-§2), computed per arm on
  the contrast intersection; traps are excluded from retention and reported separately as
  the hallucination-resistance rate (trap-ABSTAIN ÷ 5). A/A′ discordance = per-probe
  label disagreement count over the shared surviving set.
- **G3 sensitivity**: ≥1 pre-registered contrast (R2 vs R2b on overall retention, or
  E0-epoch retention R1 vs R2) separates beyond the G2 band; else verdict "underpowered"
  (valid P0 outcome; blocks P1).
- **G4 denominators**: verified or declared per arm before main-run spend.
P0 SCOPE CUTS (explicit): m2 continuation-task correctness not measured (trivial per-turn
task); R1's own noise floor unmeasured — R1 contrasts read against the R2 band + caveat.

## 6. Spend + accounting
Expected marginal $0 (ChatGPT OAuth; S0 verifies, not assumes). Real budget = OAuth quota
+ wall-clock. OpenRouter untouched. Abort criteria: any arm failing its config assertion
(engine/model/sha) aborts THAT run, reported with partial data, never silently rerun.

## 7. Landing + ownership (D5)
Code: bench/instruments/compaction_probe/{gen_material,drive_codex,drive_hermes,
parse_rollout,score_probes,report_pilot}.py — codex-dispatched against this sheet;
architect writes canary/probe content, scoring rules, bands. Zero product code. Verdict:
next-free FINDING number at write time. Publication of any per-model comparison = P1,
owner content sign-off required.

## 8. Pins (every value from a command at launch; placeholders until smokes complete)
- codex binary: release path + sha256 (S0 output) · cli_version asserted from session_meta
- hermes version + config.yaml sha256 per arm · lcm-x sha (engine under test)
- material/probe script shas + run seed · effective windows per arm (S3 output)

## 9. AMENDMENT (2026-08-20, append-only, pre-scoring) — measured mechanism of the hermes one-shot arms; regime relabels

**Measured facts (R2-A, first valid run, receipts in the run's pilot-home
state.db + lcm.db):** the one-shot driver (`hermes -z` per turn) produced **70
distinct host sessions and 70 distinct conversations** for 70 turns —
`--continue` (and `--resume latest`, micro-verified separately) **do not thread
a conversation in `-z` mode**; every turn is a fresh session. Consequently the
LCM engine's in-session compaction **never fired** (0 summary_nodes; the 136K
`threshold_tokens` was never approached per-invocation), and probe retention is
delivered by LCM's **store-level cross-session context assembly** (57 tool
messages recorded in lcm.db during the run).

**What this means (relabels, not silent re-scopes):**
- The registered "R2 = single-session + LCM compaction ≤272K" DID NOT RUN and
  is headless-unreachable via `-z` (pty input path already measured dead —
  #268/#269 chain). It is renamed **R2s** and PARKED pending an ACP-transport
  spike (`hermes acp` holds one session over stdio JSON-RPC; spike dispatched;
  if it fails within budget, R2s is declared unmeasurable-headless in P0 and
  the interactive-user regime is disclosed as an unmeasured cell).
- The arms actually run are relabeled **R2r ("restart-bridge") = per-turn
  session restart + LCM store bridge** — architecturally the R3 mechanism at
  maximum granularity, and LCM's differentiated capability (native codex
  compaction cannot bridge sessions at all).
- **R3 is CONDITIONAL on the ACP spike, not subsumed** (review correction —
  the registered R3 topology is genuinely distinct: all material in ONE
  session, where the 136K compaction actually fires and answer-ready's
  five-hit per-session density cap binds differently than across 35 sessions;
  only its one-shot IMPLEMENTATION collapses into R2r). Disposition: R3 is
  transport-blocked for the same reason as R2s. If the ACP spike lands, R3
  runs via ACP (single-session material → fresh-session probes) and is the
  fullest LCM regime (in-session compaction + cross-session bridge); if the
  spike fails, R3 is declared transport-blocked alongside R2s — not silently
  absorbed.
- **Pre-registered continuation order (fixed now, regardless of any observed
  outcome):** R2r-A → R2r-A′ → R2b → R1 → R4-ref → [if the spike lands:
  R2s-A → R2s-A′ → R3] — R2s/R3 positions are declared here precisely so a
  successful spike cannot insert them discretionarily after seeing completed
  arms. (Spike outcome, recorded at PR time: **PASS** — one threaded session,
  SESSIONS=1, codeword recalled, `initialize → session/new → session/prompt`
  newline-delimited JSON-RPC; receipt in the pilot kit's `acp-spike/REPORT.md`.
  The bracketed branch is therefore ACTIVE; the ACP arm driver goes through
  the same spec→review→PR path as the one-shot driver before any R2s spend.)
- **R2b (compressor engine, same one-shot transport) is PROMOTED to the key
  control**, with the inference rule stated one-sidedly (review correction):
  R2b has NO lcm.db, so its retention can only come from host-level channels
  (state.db context, workspace memories) — R2b therefore measures the
  host-level channel's CAPACITY. **R2b ≈ 0 cleanly establishes that R2r's
  retention required LCM's store** (one-sided inference, valid). R2b ≈ R2r
  does NOT establish that R2r's answers came from the host channel — equal
  aggregates can hide different per-probe subsets and independent mechanisms;
  in that branch, attribution for R2r uses per-probe evidence instead
  (lcm.db tool messages, per-probe discordance between the arms' miss sets).
- Epoch re-binning against measured compaction boundaries is **vacuous for R2r
  arms** (no boundaries exist); the pre-planted E0/E1/E2 epoch labels remain
  reportable as material-position strata only. Boundary re-binning still
  applies to R1/R4 (rollout markers) and to R2s if the ACP spike lands.
- Cross-arm contrast R1-vs-R2r compares different session topologies
  (single-session native compaction vs per-turn restart + store bridge). This
  IS the honest product-level comparison for headless/automation users, and the
  asymmetry is a required disclosure line in the pilot report.

**Scoring amendments already public in PR #275** (scorer binds to frozen
material schema; abstain-regex expanded to real observed phrasings; concrete-
answer guard: any registered canary value in an answer classifies HALLUCINATE
over an abstention hedge). Two review corrections on top:
- **Abstain-regex FREEZE (review correction):** the expansion was made after
  observing the first valid arm's answers. Retention (CORRECT/canaries) is
  unaffected by construction (CORRECT is evaluated before the abstain regex),
  but the trap-abstention rate IS regex-sensitive, so: (a) the PR #275 scorer
  is FROZEN as the single scorer for every arm in this pilot — **no further
  abstention-pattern changes for any reason, including future arms' phrasings
  missing the set** (a missed abstention scores HALLUCINATE and is disclosed
  as a regex miss — the fail-closed direction); (b) the trap metric for every
  arm is reported with the flag "instrument amended mid-pilot, pre-official-
  scoring" and both R2r-A trap readings (1/5 original set, 5/5 expanded set)
  stay on the record; (c) trap-abstention is a secondary diagnostic in this
  pilot, never a contrast metric.
- **Crashed-arm rule (review correction — row exclusion forbidden):**
  `coverage.fts: 'none'` (or any disclosed crashed-arm marker, cf. product
  PR #273) in a run's logs invalidates the AFFECTED ARM, which is repeated.
  Row-level exclusion is forbidden: dropping exactly the observations where
  retrieval degraded is asymmetric-exclusion bias and would corrupt the
  registered denominator (§2 boundary exclusions + pairwise intersections are
  the only permitted exclusions).

## 10. AMENDMENT (2026-08-20, append-only, pre-official-report) — agentic tool access is a contamination vector; audit results; tool policy for all arms

**Threat found (measured, not hypothetical):** the hermes host is a coding
agent with filesystem tools. During the R2r/R2b runs the model spontaneously
fired generic tools (search_files / read_file / terminal / execute_code) —
including an in-run repo grep that surfaced a REAL canary value which had
leaked into repo test fixtures via the scorer PR (#275; fixed in this PR — no
frozen-material value or phrasing may appear in any repo file), a terminal/
execute_code dump of the arm's own lcm.db (all 30 values), and reads of the
instrument source that let TRAP answers identify the bench design.

**Mechanical audit (scripted, per probe-turn session; receipts in the pilot
kit):** per-turn session isolation bounded the blast radius — a tool result
only reaches the turn (=session) that fired it. Findings:
- R2r-A: 2/35 probe turns fired generic tools; 1 value-bearing (turn 36 =
  TRAP-04, terminal dump). R2r-A′: 3/35; value-bearing turns 36 (TRAP-04,
  execute_code dump + read_file) and 56 (TRAP-03, search_files → 2 unrelated
  values). R2b: 13/35 attempted; value-bearing turns 36/39/54 (1 leaked value
  each, never that turn's expected value).
- **Zero TAINTED-CORRECT rows in any arm** (no CORRECT canary answer's value
  appeared in any generic-tool result of its own session) → the banked
  retention numbers (R2r-A 86.7%, R2r-A′ 96.7%, R2b 86.7%) STAND.
- **Trap metric is design-read tainted** in the affected turns (models read
  the instrument/bench text and inferred "abstention trap"): R2r-A TRAP-04,
  R2r-A′ TRAP-04+TRAP-03, R2b TRAP-04. Already demoted to secondary (Am. 9);
  now additionally flagged per-arm as partially informed abstention.

**Policy from here (applies to every remaining and future arm; reruns of the
banked arms are NOT required for retention, and their trap rows carry the
flag):**
- Hermes arms: during home setup (BEFORE the config-sha pin) disable the
  generic toolsets and non-memory MCP servers: `web browser terminal file
  code_execution skills delegation vision image_gen tts bfl homeassistant
  cronjob` + all MCP servers. KEEP the memory surface: `memory`,
  `context_engine`, `session_search`, plus `todo`/`clarify`. The bench
  measures memory assembly, not filesystem retrieval.
- Drivers run with cwd = a fresh EMPTY sandbox directory, never a repo or the
  kit.
- R1/R4 (codex; tools cannot be disabled): DETECTION rule, pre-declared —
  parse the rollout for shell/tool items; the arm is INVALID if any tool call
  reads material/kit paths, greps a canary value, or runs the generator;
  every tool call is disclosed in the run record either way. Ground-truth
  affordance is reduced (empty cwd), not eliminated; detection is the gate.
- Prior-arm outputs (results.jsonl, consoles) also contain values → covered
  by the same deny/detection rules.

## 11. AMENDMENT (2026-08-21, append-only) — R1 measured as long-context/no-compaction; window-parity consequences; quota event + forced order deviation

**R1 result (banked; detection gate PASS — zero tool/shell items):** 30/30
retention, traps 5/5, every epoch 10/10 including E0 (falsifies silent
truncation), C1 6/6. **Native codex compaction NEVER ENGAGED**: every
turn_context advertises model_context_window=258,400, yet the final request
carried 553,914 input tokens with zero `compacted`/`context_compacted`
markers — codex exec+resume (0.148.0, OAuth) held the entire ~600K-token
thread verbatim (long-context serving silently engaged). The smokes' 86-93%
triggers came from interactive/live-session rollouts; **compaction engagement
is transport-dependent**, and measuring it for gpt-5.6 moves to interactive
transport or larger material (P1).
- **Relabel: R1-as-run = R1-lc (long-context, no-compaction).** Under §4's
  window-parity gate (554K-observed vs 272K), R1-lc vs R2r/R2s is NOT a valid
  G3 contrast; it stands as a reference row like the R4 class.
- **m3 economics (rollout-derived):** R1-lc moved 29,536,531 input tokens
  (98.0% cache-reads; 592K uncached; output 2,352) vs ~3.0-3.6M total for the
  R2r arms — the retention-vs-tokens-moved trade-off is the pilot's central
  product finding: LCM restart-bridge ≈ 10x fewer tokens moved at 86.7-96.7%
  retention vs R1-lc's 100%.
- **Third wire-contract catch (disclosed):** drive_codex result rows carry no
  `kind` field; the scorer's `kind=="probe"` filter dropped every row and
  scored the arm 0/30-unparseable-35 before the fix (PR #293). The two
  toolchain dispatches were acceptance-tested against their own fixtures,
  never against each other's real emissions — the pins PR must include one
  cross-driver scoring test per driver output shape.
- **Driver defect fixed before any valid R1 turn (disclosed):** the first R1
  attempt stalled 17 min at zero CPU (subprocess stdin inheritance +
  no per-turn timeout; PR #291); zero turns consumed, clean relaunch.

**Quota event + forced order deviation (disclosed):** the R1-lc run exhausted
the ChatGPT-OAuth codex-CLI bucket ("try again Aug 26 15:01"), killing the
R4-ref attempt before any turn. The hermes openai-codex route is measured
ALIVE (separate bucket), so the ACP arms proceed now and **R4-ref runs in the
Aug 26–31 window** (still before the gpt-5.4 retirement). Deviation from the
§9 pre-registered order (R4-ref before R2s) is quota-forced, not
discretionary. C1-lineage answer/judge phases sharing the CLI bucket are
paused until reset (recorded in that run's own record).

## 12. AMENDMENT (2026-08-20, append-only) — Amendment 10's hermes tool policy was never mechanically effective for ACP arms; R3 attempt 1 invalid; attempt-2 protocol

**Measured (receipts in lcm-x#301):** `hermes tools disable` writes per-platform state for the
cli/messaging platform list only; ACP sessions hardcode the `hermes-acp` toolset
(terminal/read_file/search_files/execute_code/browser/delegate baked in) and never read the
disable list — verified empirically (terminal executed in a home with cli+api_server disables).
Consequences:
- **R2s-A / R2s-A′ results STAND, relabeled "clean by AUDIT" not "by construction"**: their
  per-session tool audits show ZERO generic-tool attempts (memory/session_search only) — the
  restriction was behavioral reality, not an enforced mechanism.
- **R3 attempt 1 is INVALID (unscored)**: its threaded PROBE session fired search_files/terminal
  (its own lcm.db sqlite dump included) exposing 9 canary values; in a threaded session an
  exposure persists in-context for every later probe, so per-turn containment (the Am.10 basis
  for the one-shot arms) does not apply. Arm-level invalidation per the Am.9 exclusion rule.
- **R3 attempt-2 protocol (pre-declared):** (i) the fresh home receives a pilot SOUL.md
  behavioral instruction — answer only from memory/conversation; generic tools declared
  off-limits (sha echoed in the run console); (ii) the post-run per-session tool audit is THE
  GATE: any generic-tool result carrying a canary value ⇒ arm invalid; (iii) if attempt 2 also
  trips the gate, R3 is declared **unmeasurable under the current ACP tool surface** (product
  gap lcm-x#301) — a valid pilot outcome, not a retry loop.

## 13. AMENDMENT (2026-08-20 UTC, append-only, owner-directed; revised in-PR per 11 review corrections) — luna arms; multi-compaction stress; effort pin; exec retirement

**Owner review of F59 raised four gaps; review hardening added five controls. Positions
pre-registered at the end of this section. R3 is already BANKED (100%, Amendment-12 gate)
before this amendment — the order below lists only remaining arms.**

- **Reasoning-effort pin (retro-recorded):** every P0 arm served reasoning effort HIGH — hermes
  homes carry `reasoning_effort: high` (pinned by each arm's config sha) and the R1/R4 rollouts
  record `effort: high` per turn. Future arms pin effort explicitly in the run record.
- **R2s-S-ctrl (contemporaneous sol control, review-added):** a same-day R2s(sol) single-session
  rerun, identical protocol. The luna contrast reads PRIMARILY against this control (isolates
  model choice from run-date/serving drift); the banked 0.0pt sol band remains the within-period
  noise estimate only.
- **R2s-L-A / R2s-L-A′ (gpt-5.6-luna pair):** identical R2s protocol, model flipped to
  gpt-5.6-luna (`--expect-model` accordingly). Post-run audit ADDITIONALLY asserts the
  SERVED model from the host record (state.db sessions.model == gpt-5.6-luna) — the hermes-side
  twin of the R4 served-model lesson. Read: extends or refutes F59's recommendation for the
  cost tier; C1 class watched first.
- **R2s-MC (multi-compaction stress, sol):** `compression.threshold` 0.5 → 0.12 (~32,640-token
  trigger on the 272,000 window). REVISED FRAMING (review correction): the threshold change
  moves BOTH compaction frequency AND how early answers depend on retrieval, so this arm
  answers "does the R2s regime hold up under ~8× compaction pressure and a much smaller live
  tail" — NOT the isolated effect of compaction count (isolating count needs same-threshold,
  longer material: P1). Controls, all pre-spend or fail-closed:
  (a) env guard: `LCM_ABSOLUTE_THRESHOLD_TOKENS` and `LCM_CONTEXT_THRESHOLD` explicitly unset
      for every arm invocation;
  (b) pre-spend threshold gate: a warm+status one-shot on the built home must report
      context_threshold 0.12 / threshold_tokens 32,640 / source config compression.threshold,
      with compression.enabled true — else abort before any material turn;
  (c) validity gate at scoring: the run-scoped count of SUCCESSFUL compaction events (from the
      engine's own compaction telemetry — NOT summary_nodes, which one compaction can publish
      several of, rolling) must be ≥8, else the arm is marked UNDEREXPOSED and excluded from
      the count question (still reportable as a run record).
- **Host capability pin (review-added, bounded):** each hermes arm's console records
  `hermes --version` at launch; the ACP driver already hard-fails on a failed handshake.
  Full capability-contract negotiation/validation is DECLINED as beyond a pilot instrument
  (recorded trade-off, not an oversight).
- **Codex exec retirement for long arms (owner):** no future arm runs >10 turns on codex exec
  (the R1-lc quota burn). R4-ref is the standing exception: attempts 1-2 were VOID (resume
  model-fallback; parser payload-shape — both fixed with fail-loud guards, PRs #308/#311);
  attempt 3 is in flight on the fixed driver at this revision.
**Continuation order (fixed):** R4-ref(att.3) → R2s-S-ctrl → R2s-L-A → R2s-L-A′ → R2s-MC.
