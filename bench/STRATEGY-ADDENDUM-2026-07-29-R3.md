# Strategy addendum — R3 (2026-07-29, post-R2-ship; amends STRATEGY-2026-07-25.md)

*R2 shipped today (455/500 with its p-value, the latency claim, the published scaling curve — upstream
#436, all ten gates). This addendum records the R3 thesis, the F38 re-aim, and the owner-ratified
benchmark portfolio. The plan of record with full sequencing lives in the program state file; binding
gates stay in bench/specs/.*

## R3 thesis
**The metric-standard release: prove the thesis at scale, and define how the market must measure.**
The leaderboard is contested and non-comparable (different judges/readers, no reproducibility discipline,
no scaling curves anywhere); our moats are the three things R2 built — reproducibility discipline, the
only published latency+recall scaling curve, and answer-turn evidence-completeness measurement. R3's
headline is the metric-standard play backed by the extended scale curve (ANN → ms-class at 200k messages
with recall-parity evidence). Chasing vanity parity on a non-comparable leaderboard is explicitly not
the play.

## F38 re-aim (supersedes the "budget-fill on V1-small first" sequencing)
V1-small is saturated as a delivery-mechanism proving ground: realistic remaining headroom ≈1.6 pts
(hard cap 3.6), and the pre-registered Stage-2 gate was infeasible-as-registered (≈3% pass; adjudicated
on zero-spend evidence, experiment not run). Session expansion merged as dormant capability (#173,
flag-OFF byte-identical); its next gate is retrieval-completeness on the 389× instrument, pre-registered
before any run. F33's attribution narrowed: interventionally supported, salience confound documented —
public claims use the observational gradient.

## Benchmark portfolio (owner-ratified 2026-07-29 evening)
- **Primary tier move: LongMemEval V1-MEDIUM (`longmemeval_m`)** — 10× haystack (grep stops being free),
  frontier readers allowed, live leaderboard: the first official surface for the scale thesis. V2-medium
  only as a shared-store-cost secondary (weak fixed reader, empty leaderboard = visibility, not proof).
- **LoCoMo sidecar (R3.1, cred play):** competitors publish 83–94% there; adapter built under bench/tools
  discipline — pins, fail-close accounting, and a same-code noise floor BEFORE any published number.
  *(Amended 2026-07-30, owner decision: the CC BY-NC concern applies only to REPUBLISHING the dataset
  itself, not to publishing results measured on it — LoCoMo is the field's de-facto minimum and every
  memory vendor publishes on it. The publishing hold is LIFTED; the A/A′ noise floor executes first,
  unchanged.)*
- **Agentic eval: scoped, not committed.** Selection survey first (BEAM is vendor-harness/vendor-judge —
  non-comparable); adopt only an eval instrumentable to our standards. Decision at R3.1.
- **Our own scaling benchmark** (unchanged from Phase 3): publish with negative results included.
- **Sol production telemetry** (new lane): the dual-consumer principle's second consumer — benchmark
  scores argue; instrumented production evals prove. Measurement energy moves here as V1-small saturates.
- Standing rule: **one new instrument at a time, each trusted before the next** — a new benchmark is a
  new machine to debug, and we debug it before we believe it.

## Babysit lane (upstream #436)
Live with a declared stopping rule: mechanical batch 1 pushed (a53276c); three headline architecture
decisions in DECISIONS-R3-UPSTREAM-ARCH.md; six default-off/edge decisions tracked in fork #180; only
delivery-path/P1 findings get in-PR fixes from here.

## ★ Owner amendment 2026-07-30 — the TWO-TIER testing doctrine (frontier-first)
The owner reframed benchmark purpose: **"whether it helps us find faults in our system... whether it
improves real agent, real-time, real-experience results"** — and explicitly de-weighted the
fixed-older-reader theory in favor of frontier models ("we're building for how people actually use
these systems with real frontier models"). Binding consequences:
- **Tier F (fault-finding):** frontier readers/judges, relaxed instrument bar, closed judges
  acceptable, speed over purity. Purpose = surface defects our own harness cannot see and measure
  real-experience improvement. LoCoMo lives here (it already found four instrument bugs + the
  config-class retrieval gap + the adversarial-attribution weakness — that IS the win). LoCoMo-Plus
  REOPENED under this tier (the closed-judge disqualifier applied a Tier-P bar to a Tier-F tool;
  adapter delta near-zero). Newer agentic benchmarks (BEAM-class and successors) re-surveyed under
  Tier-F criteria.
- **Tier P (published claims):** unchanged — full pins, disclosure standard (F46 §5's seven points),
  A/A′ noise floors, pre-registered gates. The metric-standard thesis lives at this tier and is
  STRENGTHENED by the split: we publish under discipline while testing at frontier speed.
- Official LongMemEval baselines keep their banked meaning; the FRONTIER-reader V1-medium move
  (already owner-ratified) becomes the flagship Tier-P surface going forward.

### Tier-F adoption decision (2026-07-30, survey: TIER-F-AGENTIC-SURVEY.md in the 07-30 deepdive artifacts)
Ranked by fault-finding value per integration effort: **(1) LongMemEval-V2** — the UCLA sequel to our own
primary instrument, agent-TRAJECTORY-based (dynamic state, workflow knowledge, environment gotchas; 451
curated questions confirmed unanswerable without working memory), Apache-2.0, adapter isomorphic to our
Provider, and THE benchmark the owner personally flagged. **ADOPTED as the next Tier-F build** (after the
LoCoMo config-fix replay resolves). **(2) AMA-Bench** (MIT, real tool-using agent harness, GPT-5.2 at only
72% = headroom) — queued second. **(3) MemoryAgentBench** (test-time learning + selective forgetting — the
only real-time-write instrument found) — watchlist. BEAM re-scored MEDIUM under Tier-F (judge unblocked,
but dialogue-only scale-stress; no agentic axis); PersonaMem MEDIUM-LOW; GAIA LOW (no memory axis).
τ²-Bench keeps R4 Tier-P; its memory-specific Tier-F use needs episode-chaining (noted, not funded).

### ★ CORRECTION (2026-07-30, owner caught it immediately): "LongMemEval-V2" IS our V2
The Tier-F survey's #1 recommendation was our OWN primary instrument rediscovered under its public name —
LongMemEval-V2 (UCLA, 451 questions, web/enterprise agent trajectories) is exactly the benchmark behind our
banked 125/451 and 298/451, the lme-v2-official harness, our upstream harness reports #6/#7, and the paired
re-baseline run in flight. The survey agent searched by public name; the program's shorthand ("V2") hid the
identity; the synthesis failed to catch it. The "adopted as next Tier-F build" line is VOID.
What survives from the survey: **AMA-Bench is the genuinely-new #1 Tier-F adoption** (trajectory memory,
real tool-using agent harness, MIT, frontier headroom); MemoryAgentBench watchlist (test-time learning /
selective forgetting); BEAM/PersonaMem/GAIA re-assessments stand. And the "higher-size datasets" are
likewise already ours: V1-medium (longmemeval_m, the ratified primary move) and the V2 MEDIUM tier
(M17 — the 7.4× store nobody has ever run, parked as Phase-3/R3.2 with its go/no-go at R3.1 close).
Process note: name-alias drift between program shorthand and public benchmark names can make research
lanes rediscover owned instruments — surveys must carry the program's instrument inventory with BOTH names.

### Track C1 decision (2026-07-30 evening): AMB ADOPTED for Tier-F, with declared conditions
Three-lane assessment (wf_f4e92ea1; full reports in session-notes 2026-07-30/hermes-amb-assess/
artifacts/: AMB-HARNESS.md, AMB-COVERAGE.md, AMB-FIT.md). **ADOPT** — the owner-flagged aggregator
(vectorize-io/agent-memory-benchmark, the Hindsight board) is worth running: one clean Python
`MemoryProvider` ABC covers LoCoMo10/LongMemEvalS/PersonaMem/BEAM(4 tiers)/LifeBench in a single
adapter (~200 LOC of glue; our `hermes_lcm_bridge.py` JSONL protocol reused UNMODIFIED; their own
mastra.py precedent de-risks the subprocess shape). Conditions, all measured findings:
1. **Comparability is NOT by-construction — it must be manufactured by us.** AMB pins nothing
   (datasets download from `main`; its "LongMemEval" is the community-CLEANED re-release, not
   vanilla), the README's "a Gemini model answers" contradicts the code default (Groq — their
   issue #15), results carry no config fingerprint, there is no CI, and the board is
   vendor-submitted with the harness owner also owning the top rows (Hindsight). Our runs pin
   dataset sha256s + set OMB_ANSWER_LLM/judge explicitly + snapshot outputs + disclose per the
   seven-point standard. Board rows are cited only with run-config caveats.
2. **Legal:** the harness repo has NO code license (GitHub license=null) and redistributes ~306MB
   of third-party datasets (incl. CC BY-NC LoCoMo) without terms. We run from a LOCAL clone with
   our adapter file overlaid (our file, our repo, our license) — no public fork, no redistribution.
3. **Adapter guards (from AMB-FIT):** raw_response = results-ONLY (their LoCoMo prompt-builder
   injects raw_response into the GRADED prompt — provenance would leak into scored context);
   our fail-closed bridge aborts a batch where reference providers silently return empty — kept,
   as a documented conscious choice.
4. **Tier-F instrument findings banked for the metric-standard release:** the pin/fingerprint
   absence, the answer-LLM bug, the BEAM scoring prose/code mismatch (docs say Kendall-tau, code
   does rubric-nugget — re-verify independently before citing), and the silent-empty retrieval
   pattern. Primary sources only (their issues #13/#15/#26 + code paths).
Sequencing: adapter build dispatches to the codex lane (well-spec'd, non-urgent); NO AMB runs
while the paired V2 gate occupies the machine (5 concurrent bridge subprocesses + embedder loads
would contend). C2 (AMA-Bench) queues behind C1's adapter; C3 (MemoryArena) assessment separate.

### Track C2/C3 decisions (2026-07-30 late; assessments wf_bf8be389 → hermes-c2c3-assess artifacts)
- **C2 AMA-Bench: BUILD** (AMA-BENCH-FIT.md). High fit: memory_construction/memory_retrieve maps
  1:1 onto the bridge JSONL contract; ~300-380 LOC glue, zero bridge changes, registration via
  their register_method() = zero upstream diff. Declared mitigations, all measured findings:
  (1) `--subset openend` ONLY — mcq_set.jsonl does not exist in the public dataset despite
  README/CLI documenting it; (2) pin by SHA (repo ddfd319e, HF dataset a5777378 — no tags exist
  anywhere); (3) their judge FAILS OPEN (unparseable yes/no → silent token-F1 substitute) and
  their model client FAILS OPEN on context overflow (truncate+retry) — both patched to fail
  closed at run prep; (4) trajectory re-split uses a STRICT parser of their deterministic
  step format (raise on any non-matching line — upstream drift fails loud, no fork taken).
  Cost profile: construction near-zero LLM (deterministic summaries + local embed);
  208 episodes / 2,496 QA; the gaia tail (max 1.03M raw tokens) needs a timing pilot first.
- **C3 MemoryArena (arXiv 2602.16313 — disambiguated from WorldMemArena and an unaffiliated
  repo): DEFER, written reasons.** It fills a REAL cell nothing else covers (causally-necessary
  cross-session memory inside tool-executing envs, graded on task success; models saturated on
  LoCoMo/LME drop to 40-60%) — but: ONE commit ever (created 2026-06-01), no code license,
  HIGH multi-service self-host burden (per-domain env servers, external data pulls, memory
  microservice, up to 3 extra paid keys, two LLM cost centers), and its overlap with queued
  AMA-Bench is unresolved. REVISIT after AMA-Bench runs: the marginal value then is task-success
  grading of tool EXECUTION — adopt only if that cell still matters and the harness matures.
  Dataset itself is CC-BY-4.0 (clean). It is the MemoryAgentBench authors' successor project.
- **MemoryAgentBench: KEEP-WATCHLIST** (refresh in MEMORYAGENTBENCH-REFRESH.md). Uniqueness
  holds (only standalone Test-Time-Learning + Selective-Forgetting/Conflict-Resolution splits —
  the real-time-write fault class), and a STANDALONE TTL+CR pilot is genuinely cheap (<1000
  judge-free deterministic QA; exclude the 1.48M-token Recsys outlier). But upstream is dead
  (zero commits ~10 weeks, maintainer graduated, a community adapter PR closed unmerged in
  2 minutes) — any adoption is a private-fork port (~100-150 LOC), on demand, not a dependency.

### Owner steer 2026-07-31 — AMB judge/answerer policy (supersedes the C1 stock-judge condition)
Owner declined Gemini-anchored config ("nobody uses Gemini" — dead-model experience isn't the
program's story; consistent with the frontier-first two-tier doctrine). REVISED CONFIG OF RECORD:
- **Answerer = frontier stack we actually ship against:** sol (GPT-5.6) first; a Claude-answerer
  row queued later (strategic: the Claude Memory north star). Gemini-matched answerer DROPPED.
- **Judge = Claude (cross-family), not Gemini, not GPT:** the hygiene requirement was always
  cross-FAMILY judging (no family grades its own homework — same principle as Codex↔Claude code
  review), never Gemini per se. AMB's model client supports Anthropic natively (C1 assessment);
  judge identity + full prompts published per the seven-point standard. No GEMINI_API_KEY needed.
- **Board comparability:** already caveat-only per the C1 decision (their pins/judge/COI issues
  are documented); a different-but-disclosed judge adds one line to an existing caveat. OPTIONAL
  parked item: a one-time dual-judged slice (stock vs ours on identical outputs) to measure the
  judge delta as a bridge coefficient, if a head-to-head number is ever wanted.
- Luna (GPT-5.6 cheap tier): exploratory Tier-F sweep ANSWERER on volume lanes; never a judge.

### Owner steer 2026-08-02 — frontier embeddings + run-log discipline + parallelization unlocked
1. **Voyage/frontier embeddings across the board for FUTURE declared configs** (V1-medium, AMA,
   AMB, any new LoCoMo variant): quality frontier stack end-to-end — "we're testing and making
   for frontier models and quality frontier embeddings." The IN-FLIGHT LoCoMo A/A′ stays
   fastembed (arm A already ran; changing an instrument mid-pair voids the noise floor).
   Side benefit: API embeddings eliminate the local-embedder contention class entirely
   (the 07-31 hang was concurrent LOCAL onnx loads).
2. **Run log:** bench/RUN-LOG.md is the append-only registry of record — every run (id, date,
   config summary, artifacts path, status/result pointer) ties back to the runbook. The fork
   docs branch is the program home; a dedicated public repo rides on the scoreboard-naming call.
3. **Parallelization/memory: no longer a constraint** (owner). Lanes overlap freely; the one
   surviving rule is instrument-integrity (no config changes mid-pair; pins per lane).

## 2026-08-02 — SOTA-MODELS POLICY (owner directive; config of record for all NEW registrations)
Owner: "moving forward we're using SOTA models — I don't care about the harnesses or adapters."
Applied with verified OpenRouter pricing (2026-08-02): luna $0.10/$0.60 per M (1.05M ctx) ·
sol $5/$30 · claude-sonnet-5 $2/$10 (1M ctx) · gpt-5.2 $1.75/$14 (retired as answerer — dominated).
- **Reader/answerer default: `gpt-5.6-luna` @ medium effort** — SOTA family, largest context on
  the board, and the cheapest serious model (undercuts even qwen3.5-9b effectively). Effort
  low for high-volume simple extraction; high only on observed long-context misses.
- **Judge default: `claude-sonnet-5`** — preserves the ratified cross-family rule (no family
  judges its own answers on scored runs; luna+sol are both GPT-5.6) AND is 2.5× cheaper than
  sol on input. `sol @ low/medium` judges only where the answerer is NOT GPT-family (e.g. a
  future Claude-answerer row) — that is where the owner's sol-judge suggestion slots.
- **Already-registered instruments run AS REGISTERED** (LoCoMo C1 keeps sol/sol from F48 for
  comparability — same-family caveat stays disclosed; judge migration is a C2+ re-registration
  question). The in-flight #436 slice finishes on its frozen paired config.
- **Per-benchmark:** V2 SOTA row = NEW declared config (luna reader + sonnet-5 judge; official
  qwen rows stay visible; at luna pricing the 451q re-run ≈ single-digit dollars). V1-M
  retrieval row unchanged (no LLM). V1-M reader row + AMB + AMA = luna + sonnet-5. AMA pilot
  spec updated pre-run (never executed under gpt-5.2).
- Every switch = new declared config with own registration + A/A′; no silent retunes; old rows
  remain visible (append-only scoreboard discipline unchanged).
