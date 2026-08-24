# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-08-24.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. Compatibility identifiers remain stable:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest, install directory, and skill | `hermes-lcm` |
| Runtime context engine | `lcm` |
| Latest stable | `v0.23.1@81d8d41197dddc4c09b57097f4955ebae32366a9` |

Changing `hermes-lcm` or `lcm` requires a separately designed migration. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is `v0.23.1` at `81d8d41197dddc4c09b57097f4955ebae32366a9`. GitHub publishes it as a non-prerelease release. Because GitHub reports the tag as mutable, operators and evidence packets must verify the exact SHA rather than trust the tag name alone.

The source snapshot used for this reconciliation is `main@3d4fbb4c979dc09aef0b831bb50d928e0e18d68f`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

Main currently retains stale `plugin.yaml` prerelease metadata. #342 owns the P4 version-policy decision for the next release preparation; it does not invalidate the stable tag or installed stable identity.

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
