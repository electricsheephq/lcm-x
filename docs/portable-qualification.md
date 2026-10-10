# Portable recall v1 qualification

Tracker: [#1010](https://github.com/electricsheephq/lcm-x/issues/1010).
Implementation: [draft #1011](https://github.com/electricsheephq/lcm-x/pull/1011).
Claim class: advisory local preview. No merge, release, customer or store claim.

## Installed host observations

| Host | Observed capability | Qualification boundary |
| --- | --- | --- |
| Claude Code 2.1.292 / Haiku 4.5 | Incremental hooks, native manual compaction, actual SessionStart continuation attachments, MCP recall and exact expansion | Six paired synthetic scenarios; supported text/tool projections. Automatic pressure compaction and complete envelope coverage are unqualified. |
| Codex CLI 0.162.0 / GPT-6.1-Sol medium | Explicit capture before native auto-compaction, manual source-linked continuation, actual MCP recall and expansion after process restart | Six paired scenarios; MCP/manual tier. Global hooks were disabled only for isolated invocations; native hook integration is unqualified. |
| Other MCP hosts | Explicit saved-memory ingestion and scoped recall | Hook capture, automatic ingestion and continuation require separate host qualification. |
| Hermes | Existing native context engine | Portable assistance does not activate or replace that engine. |

Twelve isolated sessions were used per host. The scenario set covers exact
identifiers, a corrected decision, date/negation, native tool pairing, actual
forks and another session's source-reference denial. Both native and adapted
arms returned the required values. Claude case 1's first clean control query
was ambiguous; an explicit same-session probe returned both values. Its original
receipt remains preserved. The initial first-path fixture inherited driver stdin;
the clean qualification driver uses DEVNULL and discloses that earlier artifact.

Every adapted scenario exercised actual host MCP recall and expansion and exact
stored-span round-tripping. The tool scenario preserved matched calls/results;
scope tests rejected a different corpus's handle. Codex repeated explicit capture
after restart and again without new content, appending zero rows on the final
scan. Native controls had no adapter and still compacted and answered normally.
Claude continuation attachments contained source handles and uncertainty labels;
observed payloads were 674–1,968 UTF-8 bytes, within the conservative 2,000-token cap.

Native host tests used actual extracted packages at
`de3165b9814398903cb585ea110b1f2bdcede0e0`. The only later change in their runtime
capture/facade/package surface normalizes a manual ingest session label; host
capture already supplied normalized UUIDs. Independent isolation/preservation
review passed at `725a1e216a3543775660e13790c82a1f37d9f3f3` and the source remains
unchanged on that surface. Later packages additionally require the targeted
manual-ingestion installation smoke; this is delta coverage, not twelve new
host sessions or a universal final-artifact compaction claim.

Internal status stays conservative: configured/observed fields are distinct
from qualified fields, and native_host_qualified remains false. External receipts
support the observations above; package installation does not rewrite status to
invent qualification. Unsupported events are preserved as envelope receipts and
reported as coverage gaps. Exact quotes refer to stored projections, not an
assertion that every possible host event can be replayed or reconstructed.

## Frozen scorer decision

Use the existing retrieval policy. No classifier was promoted.
The registered synthetic set has 40 development and 80 held-out cases per
posture, ten categories and disjoint conversations. Production answer_ready /
verbatim delivery supplied at most five scoped candidates, within the 30-entry
scorer ceiling. A permutation of those five cannot improve supporting-span
Recall@5; this qualification cannot establish a benefit from broader candidate
generation or a new reader. It does not reject BGE or Jev on other workloads.

| Held-out posture | Current / lexical / BGE / Jev supporting-span Recall@5 | BGE p95 added latency | Jev p95 added latency | Decision |
| --- | --- | --- | --- | --- |
| OFF | 88.89% / 88.89% / 88.89% / 88.89% | 556 ms | 295 ms | NO_ADOPTION |
| LOCAL, BGE-small embedding floor | 88.89% / 88.89% / 88.89% / 88.89% | 998 ms | 287 ms | NO_ADOPTION |

Candidate Recall@30 and nDCG@10 also remain 88.89% on the 72 answer-bearing
held-out cases. Each model genuinely scored all 80 held-out cases per posture;
paired span-recall gain is zero with 95% interval [0,0], every category delta is
zero, and each arm delivered hits on the same eight no-answer cases as baseline.
These are retrieval false positives, not measured reader answers. The lexical
arm preserves FTS witnesses within the identical frozen pool; it does not measure
candidate changes from activating the production lexical option.

BGE is pinned to `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Its actual scoring pass is
[run 37895563615](https://github.com/electricsheephq/lcm-x/actions/runs/37895563615),
using the unchanged frozen corpora registered at `b0ca525b84216260637a784645a8613656c76071`.
A first activation-function API mismatch produced fallback-only rows; those are
retained as failed receipts and are not counted as model results.

Jev is opt-in `jev-1.13.0`, synthetic-only. A first decoder assumed exact agreement
between separately rounded score/probability fields and rejected valid wire
rounding. The corrected decoder validates bounded rounding, then calculates its
normalized signal in code. Rubric, confidence threshold, cases, source offsets
and retrieval pools stayed frozen. Including both attempts and one diagnostic,
the aggregate journal records 481 requests / 430,084 input tokens, with a published
price estimate of $0.018064. No private memory was sent and no credential entered
source or artifacts. The five-million-token / $1 authorization was not exhausted.

## Public baseline and unmeasured gates

The pinned LongMemEval-S revision is
`2ec2a557f339b6c0369619b1ed5793734cc87533`, data SHA-256
`08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`.
[Registration run 37894293829](https://github.com/electricsheephq/lcm-x/actions/runs/37894293829)
completed OFF 500. Its serial LOCAL job hit the three-hour runner limit before
exporting a public report or checkpoint, so its processed count is unmeasured.
The [LOCAL-only successor](https://github.com/electricsheephq/lcm-x/actions/runs/37914546868)
completed all 500 pinned questions in ten disjoint shards with unchanged retrieval
settings, original embedding dependency constraints and one shared model cache.
Strict aggregation validated complete question membership, source/model/config
pins and frozen-pool lineage. Evaluation source is
`a75bd7837c63aba57daa126ac505ab922622b09d`; final documentation/package metadata
changes preserve the measured runtime. No model or host campaign was repeated.

| Posture | Questions | Session-label Recall@5 | nDCG@10 | Strict source capture |
| --- | --- | --- | --- | --- |
| OFF | 500 | 84.20% | 0.83345 | 0 pools; 500 source-projection rejections |
| LOCAL, BGE-small floor | 500 | 90.63% | 0.87767 | 445 pools; 55 source-projection rejections |

These are separate embedding configurations, not a classifier gain or release
score. Public session/turn-label metrics do not measure exact supporting spans
or reader answers. Strict export rejects highlighted, truncated or otherwise
nonmatching text; accepted pools have validated unchanged stored-source slices.
Public exact-span gold is unavailable even for the 445 captured LOCAL pools.
Public fixed-pool classifier evaluation, exact-span recall, reader correctness
and confidence calibration remain UNMEASURED or unqualified. No fabricated
references, substitute source prefixes or inferred span labels are exported.

The successful successor used 39m58s elapsed across hosted jobs; it repeated
computation lost to the serial timeout. Shard-local caches and execution topology
change timing, so this does not establish an end-to-end speedup over a completed
serial baseline. OFF, frozen scoring and host receipts are retained separately.

## Distribution and later gates

See [the distribution dossier](portable-distribution.md). Three installable
preview packages cover Claude, manual Codex and hook-free MCP. Each records its
source SHA and file hashes. No host global settings, customer runtime, cloud
server, hosted transcript service, release or store submission is changed.
Publisher eligibility, listing/privacy/support assets, the OpenAI local-support
partner route and vendor approval remain later owner gates.
