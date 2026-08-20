# LCM-X Product Roadmap

Status: living document (owner-ratified structure, 2026-08-19). Detail per track lives in the
linked trackers and `bench/` docs; this page is the map, not the territory.

## What LCM-X is
Lossless Context Memory eXtension: agent memory that preserves and retrieves full-fidelity
context — no lossy one-shot compression — proven by disclosed, reproducible benchmark numbers
(`bench/scoreboard/`), and deployable as the memory layer for Hermes-class agents.

## Track A — Core runtime hardening
Continuation of the correctness line (issues #1–#6 class, persistence/retrieval roadmap **#74**,
cache-parity roadmap **#41**). Definition of done per item: fail-closed tests + no regression on
the benchmark instrument suite.

## Track B — Teams mode (multi-principal isolated memory) · tracker **#75**
State: **candidate** — draft PR **#200** (AccessContextV1 + Teams catalog + access policy,
opt-in, default-OFF, fail-closed). Path to shipped:
1. Test debt first: the draft carries fixtures but zero teams test modules — a dedicated test
   suite (isolation invariants, fail-closed denials, catalog lease semantics) is the merge gate.
2. Review + land in bounded slices (the 174-file draft splits along package seams:
   access_context → access_policy → teams catalog → engine wiring).
3. Enablement is a SEPARATE decision from landing (flag stays OFF; a release may carry the code
   dormant). Host-side identity mapping (ElectricSheep/evaOS) deliberately lives outside this
   repo — see Track C.

## Track C — Host integrations (evaOS, dashboards)
The repo exposes the seam (`teams/connector.py`, `lcm_status` for dashboards); host-side wiring
(evaOS boards, per-employee identity, dashboard widgets) belongs to the host repositories.
Coordination board: `electricsheephq/evaos-support-control#544`. Non-goal here: credentials or
provider-specific control planes in-repo.

## Track D — Benchmark program (the evidence engine)
Docs of record migrated into `bench/` (2026-08-19; see `bench/MIGRATION-NOTE.md`). Goal G0 and
per-instrument targets: `bench/GOALS-AND-ROADMAP.md`. Banked rows: `bench/scoreboard/`
(9 rows incl. LongMemEval-V1 91%, V2-agentic 66.1%, LoCoMo 54.6%, AMA 47.3%). Revival order:
1. **V1-M flagship** (500q medium tier): restart with the checkpoint/resume + embed-cache
   tooling this repo now carries (port PRs); determinism probe → prewarm → 6 shards (~9–11h).
   Target ≥90% retrieval row.
2. **LoCoMo C1** (FTS-ON registered config; pre-declared bands) → **C2** (+ attribution-
   preserving ingestion, spec + validated numbers in `bench/specs/SPEC-B3-ATTRIBUTION.md`).
3. **AMA levers** L1 (step-anchored retrieval for templated trajectories) + L2 (step-index
   provenance) from `bench/FINDING-F52-*` — benchmark-agnostic product improvements.
Discipline unchanged: registration before spend, A/A′ noise floors, fail-closed accounting,
append-only corrections (`bench/release-kit/R3/HONESTY-NARRATIVE.md`).

## Track E — Releases (cadence + discipline)
First LCM-X release: **v0.22.0** (rebrand + the post-mirror hardening + benchmark tooling;
Teams labeled candidate, not enabled). Release mechanics: notes file gate + tag →
`.github/workflows/release.yml`; local `scripts/validate_release.sh --full` before every tag.
Cadence after v0.22.0: release when a track lands a coherent user-visible unit — never batch
more than ~6 weeks of change into one cut; every release notes file states what is NOT included
(candidates, known gaps) with the same care as what is.

## Sequencing (as of 2026-08-19)
Now: PR #217 (brand) → port PRs → docs/roadmap PR → **v0.22.0** → V1-M restart.
Next: Teams test suite (B.1) ∥ C1/C2 ∥ AMA levers. Then: Teams landing slices; V2-M first-mover
row; enablement decisions with the host program.
