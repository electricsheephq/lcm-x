# RELEASE READINESS V1 — the rc-first GA gate (owner-ratified 2026-08-26)

Every release that touches product code (anything outside bench/, docs/, tests/, and
.github/release-notes/) is **rc-first**:
no GA tag without a release candidate that has passed this gauntlet live. Point releases are not
exempt — a "small" diff on a hot path (ingest, recall, privacy) is exactly the profile that
needs a live soak. Docs/bench-only releases may skip to GA with a note in the release notes.

## Pipeline

1. **Merge** the release PRs through the two-lane exact-head receipt gate (unchanged).
2. **Tag `vX.Y.Z-rc1`** (hyphen ⇒ release.yml publishes it as a PRERELEASE) with curated notes
   at `.github/release-notes/vX.Y.Z-rc1.md`.
3. **Run the gauntlet** (phases A-C below) against the rc tag. Every phase produces a receipt
   file under the session-notes artifacts dir; the GA cut references all three.
4. **Fix-and-respin**: any P0/P1 finding → fix PR (through the gate) → `-rc2` → re-run.
   Carry-forward rule (every receipt binds an exact tree, so carrying needs proof): **Phase B
   re-runs on every respin** (it is diff-scoped by definition). Phases A and C may carry a prior
   rc's receipt ONLY when the rcN→rcN+1 diff touches nothing outside bench/, docs/, tests/, and
   .github/release-notes/ — AND nothing under bench/instruments/release_gauntlet/ or
   bench/instruments/compaction_probe/ (a changed gauntlet or soak driver invalidates its own
   receipts) — AND, for Phase C, its same-soak tuple (§Phase C) is unchanged between the receipt and
   the candidate; otherwise the phase re-runs. The carried receipt is referenced WITH the diff-scope
   proof recorded beside it: `git diff --no-renames --name-status rcN..rcN+1`, every listed path —
   rename detection off, so both sides of any move appear — inside the eligible set. A respin whose rcN→rcN+1 delta is
   confined to those same paths re-runs Phase B over that delta only and composes the result with
   the carried Phase B receipt (the composed receipt records both). **This specification is never
   carry-safe:** when the delta changes an acceptance criterion in this file, every carried receipt
   for the affected phase is re-evaluated against the new criterion from the evidence it recorded,
   and the carry record states that re-evaluation; if the recorded evidence cannot be evaluated under
   the new criterion, the phase re-runs. Any product-code delta re-runs all affected phases
   (ingest/recall/privacy deltas re-run everything).
5. **GA tag `vX.Y.Z`** only when A+B+C receipts are green for the passing rc tree. Because
   release.yml reads curated notes from the tagged tree, the GA commit may differ from the rc
   tree by EXACTLY the release-notes addition and nothing else — verified mechanically:
   `git diff --no-renames --name-status rcN..GA` must show EXACTLY ONE entry, `A .github/release-notes/vX.Y.Z.md` — the notes file for the GA tag and nothing else; a second added file, or modifying or deleting existing notes, is not the exception. GA notes =
   rc notes + gauntlet summary + receipt links.

## Phase A — Live all-tools matrix (the "clone" test)

A fresh, isolated hermes+LCM clone (fresh HOME-style env, fresh DB, HERMES_LCM_REPO = worktree
at the rc tag) — never a developer working tree. Two configurations, both driven LIVE:

- **cloud-default posture**: real cloud embedding provider (small corpus — spend is cents),
  durable sensitive patterns off (lossless store) and embedding privacy auto-on.
- **local posture**: fastembed/local provider, patterns off.

Matrix: EVERY registered `lcm_*` tool (enumerate from the tool registry at run time — currently
lcm_recall, lcm_status, lcm_doctor, lcm_expand, lcm_expand_query, lcm_describe, lcm_grep,
lcm_inspect, lcm_recent, lcm_retrieve, lcm_load_session, lcm_query_state, lcm_compute,
lcm_compile_evidence, lcm_evidence_pack — the runner FAILS if it finds a registered tool with no
matrix row, so new tools cannot ship untested) × both postures. Each row asserts a real
post-condition (hits returned, status fields present, doctor clean), never just "no exception".

Privacy batteries (behind the cloud key gate): the planted-secret battery proves the shipped
default keeps every planted secret raw in durable rows and recall while provider dispatches are
transformed, canonical, residual-free, and revision-validated. A fail-closed refusal is a valid
no-leak outcome for the chunk corpus: the chunk splitter can cut a dense planted fixture
mid-key, and the residual backstops then refuse that dispatch (report `stop_reason:
privacy_refused`, `privacy_blocked >= 1`) rather than ship it — the battery accepts exactly
that refusal shape (any other error still fails) and the raw-secret sweep covers everything
that did dispatch. Opt-out proves `privacy:off`
preserves byte-identical provider input. Durable-redaction preserves those same redaction and
placeholder checks as an opt-in posture. Misconfiguration uses an invalid pattern catalog to
prove lcm_recall raises, the proactive counter increments, status exposes privacy_policy_errors,
and assembly never breaks; its negative control proves the shipped default recall succeeds.

## Phase B — P0/P1 adversarial sweep (ultracode)

Multi-agent workflow over the FULL release diff (previous GA tag → rc tag), not per-PR deltas
(the limited-respin composition below is the only exception):
dimensions ≥ {correctness, security/privacy, performance/DoS, API contract, upgrade/migration,
concurrency} → independent finders → 2-vote adversarial verification → verdicts. The
upgrade/migration dimension is mandatory: open a previous-GA-created DB (old vectors,
placeholders, revisions) under the rc tree and exercise re-embed/migration paths. Cross-model
rule applies: finders and verifiers must not all share the author's model family.
Findings: P0/P1 verified ⇒ respin. P2 ⇒ tracked issue with disposition before GA.
**Limited respin (step 4):** when an rcN→rcN+1 delta is confined to bench/ (outside
bench/instruments/release_gauntlet/ and bench/instruments/compaction_probe/), docs/, tests/ and
.github/release-notes/, Phase B runs over
`rcN..rcN+1` only, and the composed receipt is the carried full-range receipt (previous GA → rcN)
PLUS the delta receipt (rcN → rcN+1): together they cover previous GA → rcN+1 with no gap, and the GA
notes cite both. This composed receipt IS the full-release-diff sweep the repository requires
(AGENTS.md): its two parts partition previous GA → rcN+1, and because the eligible paths exclude
product code, the product tree of rcN+1 is byte-identical to the one swept as a whole at rcN — the
respin adds no product interaction to review. Any other delta re-runs Phase B over the full range.

## Phase C — Live-session soak

A scripted multi-turn session battery over `hermes acp` (the measured headless single-session
transport) against the clone: ingest-heavy turns, recall probes, compaction crossing at least
one threshold, doctor at close. Green = zero unexpected errors in engine logs, zero NEW
publication-invariant conflicts (#247-class), recall probes hit, doctor clean. Minimum 30 turns.
**Differential rule for #247-class conflicts** (applied since the v0.23.2 train; codified
2026-09-05, #427): a non-zero conflict count passes ONLY when the same soak against the previous GA
tree reproduces the same conflicts under the mechanical comparison below. "The same soak" is a
recorded tuple that must be identical across every run the comparison uses: the host build pin, the
soak fixture files (material, probes, canaries) by sha256, the soak configuration by sha256, the
host's system prompt and every other prompt-bearing file the fresh home starts with (`SOUL.md` and
its kin) by sha256 or an explicit absence marker, the
assistant provider and model identifier (the live-model side of the soak; a fixed seed where the
provider supports one), the transport, the driver and turn-script files by sha256, the driver's normalized invocation — every
option and value that can change session continuity or turn completion (turn timeout, boot timeout,
quiet period, restart-before-probes, probes-only), as the driver manifest records them — the execution
runtime of the host and of the driver (interpreter version, SQLite library version, platform, and the
installed-distribution inventory of each virtualenv the runs use, as a sorted name+version list by
sha256; two runs on different runtimes are not the same soak), the starting state — a fresh isolated home
and an empty database for every run, or, where a seeded start is part of the fixture, the same seed
snapshot by digest (publication conflicts depend on persisted lifecycle state, so runs that start
from different data are not comparable) — and the effective engine environment: the driver inherits
the launching process's environment and the host then loads its own files (the fresh home's `.env`),
so the environment is captured AFTER every loader has run — every `LCM_*` variable and every
behaviour-affecting `HERMES_*` variable (model, generation length, thresholds) present in the
session's effective environment is recorded — name and value, or, where the value is a secret
(a credential, token or key), the name, a fixed presence marker and the credential's NON-SECRET
identity: the account, project, organisation or key identifier the provider exposes, or the
secret-manager reference and version the value was injected from; where the provider exposes none,
the evaluator compares the values of the runs under comparison privately, in memory, and publishes
only the attestation `identical` or `differs`. The same applies to any credential the assistant or
host reads from its own store rather than the environment (an auth file in the fresh home): the
store's non-secret account identity is a tuple field. Never any digest of a secret value, salted,
keyed or not — a published hash of a secret is a cross-release correlation and offline-guessing
oracle — and never the presence marker alone: two runs under different credentials can run under
different accounts, tenants, entitlements or provider rollouts, so a tuple that does not bind the
credential identity does not bind the soak. The canonical digest is the sha256 of the sorted
`name=value` list with each secret value replaced by its non-secret identity (or by the attestation
token), and the sets, identities and attestations must be identical across the runs compared. The
identity is the proof of record whenever the provider or store exposes one; the attestation is the
proof only where none is exposed — alternatives, never both required. A `differs`, or the absence of
whichever proof applies, makes the runs non-comparable and the differential unevaluable (FAIL); thresholds such as
`LCM_CONTEXT_THRESHOLD` change when compaction runs and therefore the conflict profile. The capture
has a named mechanical source stated in the receipt — a dump of the session process's environment
after launch, or, until the driver records one, the launching shell's `LCM_*`/`HERMES_*` listing
plus proof that the fresh home contributes none (its `.env` absent or its variable names listed);
the driver manifest's `env_names` list names only the variables the driver itself sets and is NOT
the effective environment. Launch the soak from a scrubbed environment that carries only the
recorded variables. Provider-side defaults cannot be pinned, so the
previous-GA runs must be contemporaneous with the candidate pair — run within the same 24 hours on
the same provider model identifier; an older previous-GA run is re-run, not reused.
1. **Record and profile.** Where the host exposes a conflict's publication point and message, a
   conflict record is `(kind, publication point, message)`, a run's profile is the multiset of its
   records, and the candidate passes iff its multiset is contained in the previous GA's multiset —
   for every identity the candidate's multiplicity is at most the previous GA's, and any excess
   occurrence is NEW — which also bounds its count; occurrences present only in the previous GA are
   recorded as REMOVED (identities do not jitter, so no band applies). Under this rule the two
   previous-GA runs must have identical identity multisets — otherwise the baseline's conflict set is
   unstable and the phase FAILS — and containment is tested against that shared multiset; the two
   candidate runs must likewise agree with each other. Where it does not (today the engine collapses
   every publication conflict to the `publication_invariant_conflict` label without its source —
   logging follow-up #430), a record is `(soak turn index, logged error code, logged error name)`
   and the profile is the multiset of records sorted by `(code, name, turn index)` — the **position
   profile**. The soak turn index is the driver's turn ordinal whose window contains the conflict's
   log timestamp. The driver records, per turn, the timestamp `ts` at which the turn ENDED and the
   turn's elapsed `wall_ms`; the start is derived, not read: `start_i = ts_i − wall_ms_i` (the two
   clocks differ by sub-second drift at most, and the derivation is a fixed computation over the
   results file). The driver's turn timestamps and the host log's conflict timestamps must come from
   the same clock — driver and host on one machine, stated in the receipt; where they do not, the
   measured offset bound is recorded and a conflict within that bound of a window boundary makes the
   run unevaluable (FAIL). Windows are half-open, turn i spans [start_i, start_{i+1}) — from its derived start
   to the next turn's derived start — and the final turn spans [start_n, ts_n], its recorded end; a
   conflict logged after ts_n belongs to no turn. Every conflict line maps to exactly one turn; an
   unmapped or doubly mapped conflict makes the profile unevaluable and the phase FAILS. The
   derivation rule is part of the receipt, so two operators derive the same profile from the same
   log and results file.
2. **Band.** Because the assistant side is a live model, position profiles jitter between runs of
   the same tree, and a very stable candidate pair must not turn ordinary baseline jitter into a
   false NEW. The band is therefore measured from TWO same-tree pairs — the candidate pair A/A′ and
   the previous-GA pair B/B′, all four runs under the same tuple — BEFORE any cross comparison is
   computed, and no later run widens a recorded band. Within each pair, sort both profiles; if the
   pair's `(code, name)` multisets differ — in count or in identity — that tree's own conflict set is
   unstable and the phase FAILS (investigate, then re-run); otherwise, within each `(code, name)`
   group, pair records index-wise and take `max_i |turn(X_i) − turn(X′_i)|` over all groups; the band
   is the larger of the two pairs' values, a whole number of soak turns, inclusive (identical
   profiles give 0). The first two completed runs of each tree under the tuple are BINDING: they are
   the pair, an unstable pair fails the phase for that candidate, and a record observed in any
   completed run that pairs with no baseline record is NEW even if a later run does not show it. A
   further run is admissible only after a recorded causal change — a new candidate tree or a new
   tuple — never on the unchanged head. Every run attempted under the tuple is recorded in the
   receipt, aborted soaks included, each with its cause, and a conflict record observed in ANY
   attempted candidate run — completed or aborted, under this tuple or a superseded one — must be
   paired against the baseline within the band or explicitly dispositioned in the receipt with its
   cause and evidence before the phase can pass; a tuple change or an abort never discards an
   observed conflict. **The maximum admissible band is 2 soak turns** (the largest same-tree
   jitter measured on record, and small against a ≥30-turn soak). A measured band above it
   means the candidate's own run-to-run behaviour is too unstable for positions to discriminate; the
   phase is then unevaluable under the positional fallback and FAILS (investigate the soak; identity
   evidence is required to pass).
3. **Match.** Two records match iff their codes and names are identical and their turn indices
   differ by at most the band. Pair the candidate profile against the previous-GA profile greedily
   in sorted order: for each candidate record, take the first unused previous-GA record that
   matches it; a candidate record with no match is NEW. Run the pairing for each candidate run
   against each baseline run separately — A against B, then A against B′, and likewise A′ — with the
   count condition of step 4 applied against that baseline run; a candidate run passes when it pairs
   completely against at least one baseline run (an unchanged candidate must lie within the measured
   jitter of at least one baseline sample), and its NEW records are those left unpaired against the
   baseline run it came closest to.
4. **Differential verdict.** The differential passes iff both A and A′ pair every record (zero
   NEW) AND neither has more records than the previous GA. It is one conjunct of Green, not Green
   itself: A and A′ must each also pass every non-conflict Phase C assertion (zero unexpected
   engine errors, recall probes hit, doctor clean) — a candidate run failing any of them establishes
   no profile and no band, and the phase fails. B and B′ must complete every turn with an intact log
   and results file so their conflict profiles can be extracted, and must pass the non-conflict
   assertions that can change publication behaviour — zero unexpected engine errors and a clean
   doctor — because an unhealthy baseline can carry spurious conflicts that absorb candidate records
   which would otherwise be NEW; a recall-probe miss by a baseline run is recorded beside its profile
   and does not block the candidate (a release that repairs a previous-GA recall failure must remain
   releasable). If a baseline run fails one of the health assertions, the differential is unevaluable
   and the phase FAILS closed; releasing a candidate that repairs that very failure is an owner
   decision recorded in the receipt (ESCALATED/OWNER GATE), with the candidate's own two clean runs
   and the disclosed baseline failure as the evidence — the gate never turns green on its own in that
   case. A candidate with more records than the
   previous GA fails regardless of positions. Publication-attempt parity is a condition of EVERY
   comparison, not only of fewer-record candidates: each candidate run's publication attempts —
   successful compactions per host telemetry (`total_compactions`, which counts successes only: a
   conflicted attempt returns before the telemetry is written) PLUS its conflict records — must be
   at least each baseline run's; both counts are receipt fields. A candidate that attempted fewer
   publications exercised fewer publication points, so a zero-NEW conclusion over it is incomplete
   and the phase fails. A candidate with fewer records passes the differential only when every one
   of its records pairs and attempt parity holds — a repaired conflict becomes a successful
   compaction, so the attempts stay equal; a skipped attempt does not — the receipt listing the
   unpaired previous-GA records as REMOVED with the delta commit believed responsible. Kind and
   aggregate count alone never suffice.
5. **Known residual of the positional fallback.** A NEW conflict that fires within the band of a
   REMOVED one, with the same code and name, is indistinguishable from it without a publication
   point: the fallback bounds the blind spot to same-band, same-kind substitution and cannot close
   it. It is accepted only while the host exposes no identity (#430); the receipt states the
   residual whenever the fallback is used, and a train whose host exposes publication point and
   message must use the identity rule in step 1 — the fallback is not available to it.
The receipt records every field of the tuple (host pin, fixture and configuration hashes, assistant
identifier, transport identifier, the sha256 of the driver, turn-script and prompt-bearing files, the starting-state
statement or seed digest, the effective `LCM_*` and behaviour-affecting `HERMES_*` set with its
canonical digest and its capture source), the previous GA's exact tag with its resolved commit and tree SHA beside
its profile, the candidate rc tag and tree SHA, the sorted profiles of A, A′, B and B′, both pair
bands and the band used, every pairing with its deviation, every unpaired record, which identity
fields the host exposed, the verdict, and — so the derived profiles can be re-derived — the sha256 and
a retained reference (path or URL) of every run's raw engine/host log, driver results file and
manifest. While #247 is
open the count is expected to be non-zero, so the differential run is mandatory whenever it is.

## Receipts

Each phase writes `PHASE-{A,B,C}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The GA release
notes link all three. Receipts are published verbatim (local paths redacted; operational identifiers — account, project or
key identifiers, secret-manager references, environment values that name infrastructure — may appear
in an abbreviated or hashed form provided the receipt states the form and equality across runs stays
readable from it, the full values living in the retained reference) as comments on the
release train tracker issue; that comment URL is the link of record and is what the GA notes link,
and the GA notes file itself — an immutable object in the GA tree — prints the sha256 of every
published receipt body it relies on, carried receipts, addenda and the carry record included, so an
edited or deleted comment is detectable from the tree alone; a digest that lives only in another
comment proves nothing (codified 2026-09-05, #427).
