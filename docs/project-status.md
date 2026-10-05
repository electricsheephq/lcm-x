# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05; the naming table and latest-stable identity were updated on 2026-09-24 for the #471 rename and on 2026-09-25 for the v0.24.1 GA, and on 2026-09-26 for the v0.24.2 GA, and on 2026-09-28 for the v0.24.3 GA; and on 2026-09-28 for the v0.24.4 GA; and on 2026-09-29 for the v0.24.5 GA; and on 2026-09-30 for the v0.24.6 GA; and on 2026-09-30 for the v0.24.7 GA; and on 2026-09-30 for the v0.24.8 GA; the main identity was updated on 2026-10-01 for the v0.24.9 release candidate; and on 2026-10-01 for the v0.24.9 GA and the roadmap reset (#658); the main identity was updated on 2026-10-03 for the v0.25.0 release candidate.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.25.1` (a recall-only patch on `v0.25.0@67de214f399f47d019535bf30baccf79cacc6ec4`; ships `hermes-lcm-x` / `lcm-x`) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is `v0.25.1`, cut from the `release/v0.25.x` branch at `v0.25.0` (`67de214f399f47d019535bf30baccf79cacc6ec4`). GitHub publishes it as a non-prerelease release. Because GitHub reports the tag as mutable, operators and evidence packets must verify the exact SHA rather than trust the tag name alone.

The source snapshot used for this reconciliation is `main@7e1050407f1ed149273d312153d9103e003c7bdc` (the #822 merge). Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

Main carries the forward identity `0.25.0` (`hermes-lcm-x`) (rc-first under `bench/specs/RELEASE-READINESS-V1.md`); the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

v0.23.2 (2026-08-27) shipped those contracts: durable redaction and cloud-embedding privacy are
independent flags (`LCM_SENSITIVE_PATTERNS_ENABLED` vs `LCM_EMBEDDING_PRIVACY_ENABLED`, #374),
privacy-policy errors on the recall path fail loud (#370), and releases with product code are
rc-first under `bench/specs/RELEASE-READINESS-V1.md` (#373). Since v0.23.2, main has extracted
session-end prefix matching into a mixin with no behaviour change (#155), preferred the cached
FastEmbed model during warmup (#404), made the Teams scope backfill linear (#408), deflaked a
telemetry test (#407), fixed the exact-head gate's peer-receipt reset (#362), and added benchmark
records (#412, #413, #416). None of this changes Eva's accepted identity (exact stable v0.23.1).

## Eva acceptance state

Eva is accepted on exact stable v0.23.1 with hosted `voyage-4-large`, 1024-dimensional float32 summary vectors under the privacy-bound vector identity.

The accepted Eva packet covers:

- complete eligible summary vectors (4,559/4,559 at acceptance);
- fail-closed cloud summary/query privacy handling;
- SQLite and FTS parity;
- exact and semantic recall;
- all 15 LCM tools with supported or explicit disabled/degraded outcomes;
- provider accounting, continuity, controlled restart without replay amplification, and copied-state rollback;
- two blind final reviews at 98 and 97.

That result establishes `runtime_safe` for Eva's hosted privacy-safe configuration only. It does not prove fleet, customer, Teams, local-model production, or universal benchmark readiness.

Reranking, binary prescreen/int8, proactive recall, V4 assertion/adaptive/pre-answer features, raw-chunk embeddings, and local-model switching remain off for Eva.

## Current program

Issue #658 is the tracker, and [ROADMAP.md](../ROADMAP.md) lays the work out by horizon and milestone: H1 compaction in the v0.25.0 to v0.28.0 milestones, and the H2 recall re-baseline on the v0.24.9 GA (the default configuration, then embeddings on) in its own milestone.

## History

The v0.23.1 Retrieval Provenance Audit (#341, closed) and the post-v0.23.1 roadmap (#323, closed) are history. #252 still holds the score-sensitive dossier and the conditional `LAND` verdict; historical F53-F58 rows used older product and provider identities and remain context only. Any retrieval behavior change still requires a separate accepted issue and a fresh baseline.

Teams remains a separate dormant/pilot program with its own milestones and host acceptance. No current LCM-X evidence should be presented as Teams enablement.

## Proof boundaries

Keep these claims separate:

- source and tag identity;
- PR/CI/review/merge state;
- benchmark candidate and delivery metrics;
- answer accuracy;
- released product;
- one-profile runtime safety;
- fleet/customer readiness.

A passing clone, benchmark, or Eva acceptance packet proves only its named plane. Current details and ownership live in #658 and the GitHub milestones; #252 holds the score-sensitive eval queue.
