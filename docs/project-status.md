# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. Compatibility identifiers remain stable:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest, install directory, and skill | `hermes-lcm` |
| Runtime context engine | `lcm` |
| Latest stable | `v0.23.2@676bb48dde29f06899ff829a7cb8b1d9d2e262f6` |

Changing `hermes-lcm` or `lcm` requires a separately designed migration. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is `v0.23.2` at `676bb48dde29f06899ff829a7cb8b1d9d2e262f6`. GitHub publishes it as a non-prerelease release. Because GitHub reports the tag as mutable, operators and evidence packets must verify the exact SHA rather than trust the tag name alone.

The source snapshot used for this reconciliation is `main@7b894b65d01c70046194c7b0fd10aae1448b2a07`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

Main carries the forward identity `0.23.3` for the next point release (rc-first under `bench/specs/RELEASE-READINESS-V1.md`); the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

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

## Current evaluation program

Milestone **v0.23.1 Retrieval Provenance Audit** and issue #341 own the next finite program. The audit keeps product bytes pinned to exact stable and adds a separately identified, default-off benchmark instrument.

The registered flow is:

```text
public/scrubbed LongMemEval corpus
  -> exact-v0.23.1 product harness
  -> FTS / summary / chunk candidates
  -> shipped lcm_recall fusion
  -> content-free reference validation
  -> answer-blind metrics
```

Issue `#252` contains the score-sensitive dossier and conditional `LAND` verdict. No reader or judge is used. Historical F53-F58 rows used older product/provider identities and remain context only.

The audit can decide `KEEP CURRENT` or record `FUSION DESIGN EARNED` as oracle headroom. It cannot change retrieval behavior or claim achievable uplift. Any behavior change requires a separate accepted issue and fresh baseline.

## Later accepted work

Roadmap #323 retains the later quality dependencies:

- provider trust and state authority: #317, #298, #89;
- summarizer provenance and contracts: #324, #318, #319, #240;
- bounded assembly: PR #206, #320, and #90/PR #297;
- retrieval/diagnostics: #265 and #321;
- compaction publication/semantics: #247 and #314.

Deferred privacy/scale follow-ups #334-#337 and the P4 telemetry test #328 remain outside v0.23.1 and outside the retrieval audit.

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

A passing clone, benchmark, or Eva acceptance packet proves only its named plane. Current details and ownership live in #323, #341, and #252.
