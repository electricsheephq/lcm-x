# Embeddings setup — free and local options

Local semantic recall is **on by default**, using in-process `fastembed` and its
effective default model `BAAI/bge-small-en-v1.5` (384 dimensions). The configured
model stays empty: only FastEmbed supplies this built-in default. An explicit
`LCM_EMBEDDING_MODEL` wins; cloud and OpenAI-compatible providers still require
their own explicit model and retain their privacy and consent requirements.
Set `LCM_EMBEDDINGS_ENABLED=false` to opt out: no provider load or model download,
and existing rows and vectors remain intact. Other providers are explicit options.

When local embeddings are enabled by the product default, hosts with at most
2 effective CPUs use full-text recall immediately, including summaries-only
requests. Recall reports `degraded=true`, reason `cpu_limited`, and the arms'
actual coverage; doctor explains the limit. Effective capacity respects affinity
and cgroup quotas (including ancestor limits and fractional CPUs), cached per
process. Background catch-up continues. An explicit `LCM_EMBEDDINGS_ENABLED`
value or programmatic config keeps control; explicit semantic `lcm_grep` is
unchanged. Proactive recall stays inert on these hosts unless a nonblank provider
override is set. Doctor still checks provider availability and readiness; a
healthy CPU-limited host needs only an explicit opt-in if semantic recall is wanted.

## TL;DR

| Option | Cost | Signup | Install | Best for |
|---|---|---|---|---|
| Voyage AI | free tier (200M tokens for the voyage-4 group), then ~$0.02–0.12 per million tokens | yes (no credit card) | none | best quality, zero local footprint |
| fastembed | $0 | no | Host-managed dependency companion (ONNX, no torch) | local default — no accounts, no daemon |
| Ollama | $0 | no | Ollama app/daemon | you already run Ollama |
| OpenAI-compatible | provider pricing | yes (per provider) | none | reuse an existing `/v1/embeddings` service (SiliconFlow, llama.cpp, vLLM) |

All providers feed the same store. Switching provider/model is one config change plus a backfill;
each full identity keeps its own vectors, so switching **back** to a
previously-registered provider reactivates its vectors with no re-backfill (see *Switching or
removing providers*).

## Option 1 — Voyage AI (free tier)

The Voyage-4 generation (`voyage-4`, `voyage-4-large`, `voyage-4-lite`) carries **200 million free
tokens for that model group**, and signup does not require a credit card. For a personal LCM corpus
(thousands of summaries), that free allotment covers initial backfill and years of queries; past it,
embedding costs $0.02/M (`voyage-4-lite`), $0.06/M (`voyage-4`), or $0.12/M (`voyage-4-large`). A
query embeds exactly one short vector, so a semantic or hybrid query costs a fraction of a cent.
Voyage documents its allotments and rate tiers on its
[pricing](https://docs.voyageai.com/docs/pricing) and
[rate-limits](https://docs.voyageai.com/docs/rate-limits) pages — treat those as the source of
truth; the numbers above were verified 2026-07.

```bash
# Set VOYAGE_API_KEY from dash.voyageai.com
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-4-lite   # or voyage-4 / voyage-4-large
# Cloud provider-bound copies are protected automatically. Durable storage stays raw.
/lcm embed warmup                   # probes API; registers model, dimensions, privacy revision
/lcm embed backfill                 # dry run: shows counts + estimated tokens, writes nothing
/lcm embed backfill --apply         # embeds your history in bounded batches
```

Known cloud providers protect every provider-bound copy — embedding documents
and queries, recall/grep/proactive query payloads, and rerank inputs — using
`LCM_SENSITIVE_PATTERNS`, while the durable corpus remains lossless by default.
Set `LCM_SENSITIVE_PATTERNS_ENABLED=true` only if you also want irreversible
redaction on future durable ingest. To deliberately send raw embedding input,
set `LCM_EMBEDDING_PRIVACY_ENABLED=false`; this opt-out is recorded as the
`privacy:off` vector revision, so changing the posture requires re-embedding.

Notes: requests are batched under Voyage's caps — both the token budget and the 1000-item
per-request cap; over-length documents are skipped and reported, never silently truncated;
rate-limit responses are honored with bounded waits under one absolute per-operation deadline.
That deadline starts before document conversion/token counting and covers every split, retry, and
backoff, plus response decoding and validation. Automatic HTTP resend is deliberately narrower:
Voyage `429` is an authoritative rejection and may be retried within the remaining deadline, but a
timeout/network failure or `5xx` after transport starts may follow remote acceptance and is never
automatically resent. A deadline that expires after durable dispatch marking but before transport
is a typed `not started` outcome, so backfill can safely clear those exact rows. Individual
token-count calls run in bounded workers; an overrun returns at the deadline without dispatching a
later request (the timed-out worker may finish in the background while holding one of the fixed
worker slots).

## Option 2 — fastembed (local, no signup, recommended local default)

[fastembed](https://github.com/qdrant/fastembed) runs ONNX models on CPU with no PyTorch and no
external service. The host installs its Python package through the stable companion plugin.

```bash
# Use the same full commit SHA as your installed LCM-X core.
LCM_X_REF=<LCM-X-40-character-SHA>
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --ref "$LCM_X_REF"
# Accept the host's dependency consent, enable the companion if needed, then restart Hermes.
# No LCM_* settings or manual warmup are required for the local default.
/lcm doctor
```

Model preparation runs the warmup path automatically in the background; it may
download an uncached model there, never during a query or on the turn path.
Catch-up then embeds existing history in bounded background batches. An offline
cold start uses full text, with bounded preparation retries; preparation can
complete after connectivity returns. A missing/disabled companion, an uncached
or loading model, and an incomplete index all keep `lcm_recall` on full text,
including summaries-only requests, with `degraded=true`, a precise reason and
actual coverage. Full-text coverage of `none` is not a successful fallback.
`/lcm doctor` reports preparing, unavailable (with the remedy), or model ready
with separate summary coverage and chunk coverage/backlog. Model ready does not
mean index ready. Manual `/lcm embed warmup` and `/lcm embed backfill --apply`
remain operator tools, rather than prerequisites for the default local path.
Queries use the model's query-specific encoding (distinct from document encoding) so query/passage asymmetry is
preserved. When `LCM_EMBEDDINGS_ENABLED=false`, `warmup` is inert: it does not resolve a provider,
download a model, create embedding tables, or create the configured database.

FastEmbed uses one ONNX Runtime intra-op thread for `BAAI/bge-small-en-v1.5`
and two for other models; this is not a process CPU quota.
To load it from another cache, for example one shared, read-only
cache for several profiles, set fastembed's own `FASTEMBED_CACHE_PATH` (default: the user cache directory).
Pre-seed a read-only cache: `warmup` cannot download into it.

### Install consent, updates and recovery

`scripts/install.sh` links LCM-X core and its skill as before, enables the core
through the host's enable verb, then runs the companion install command above.
It passes the selected profile through `-p` to every host command with `HERMES_HOME`
set to the install root: a named profile targets `root/profiles/<name>`, and
with a root-shaped home, unset or `default` targets the root even with a sticky active profile. The installer
requires canonical lowercase, unpadded ASCII profile names and
refuses other spellings before links or host calls (use `default` for `Default`
and `work` for padded `Work`). Printed
retry/enable commands retain that root and profile. A separate older copy prevents
automatic core enablement until the printed migration is completed. Optional
companion failure preserves successful core installation and full-text recall;
for a non-Git archive, replace the retry's SHA placeholder with the core's published
full commit SHA. Context-engine selection remains manual.
A profile-shaped `HERMES_HOME=<root>/profiles/<name>` selects that profile; a conflicting `HERMES_PROFILE` is refused before links.
The installer never links the companion or runs pip. Both qualified host
pins require a repository subdirectory identifier rather than a local directory;
`--ref` pins it to the installer's checkout commit. Native-route core users run
the same companion command with their installed core's full SHA.

On hosts with the shared plugin manager, the native install owns the questions.
Noninteractive installation or a decline leaves the companion installed but
**disabled**. LCM-X stays enabled and full-text recall stays available. To consent
later, run the host's printed command:

```bash
HERMES_HOME=<root> hermes -p <profile> plugins enable hermes-lcm-x-local-embeddings
```

The reference Hermes 0.21.5 pin uses its legacy dependency installer, without a
separate dependency-consent question on enable. Its first session may need a
restart before fastembed becomes available. Restart Hermes after enable on either
host, then run `/lcm doctor` to check provider health. Use the root and profile
printed by the installer (`default` for the root profile); a profile-shaped home
is normalized to its root and profile. Doctor names the same enable command when
the companion is disabled. No model download is part of companion installation;
local model preparation and catch-up start automatically in the running plugin.

The companion is a normal host-managed plugin: use host verbs to enable, disable
or uninstall it. A pinned install does not move with `hermes plugins update`.
To deliberately replace an existing host-managed companion, use:

```bash
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --force --ref "$LCM_X_REF"
```

The installer prints this command instead of forcing replacement. These host
pins reject external symlinks before processing `--force`. If an earlier install
linked the companion, move only that link aside by hand, preserve its target,
then run the original install command without `--force`. The installer preserves
the link and prints this prerequisite and the pinned host command. Keep the
companion selected across host updates: the host records its dependency, while
ordinary LCM-X version updates leave the stable carrier manifest unchanged.
Disabling or uninstalling the carrier may remove fastembed on the next sync.

If fastembed is missing after an update, install or enable the companion, restart
Hermes and run `/lcm doctor`; background preparation and catch-up fill the gap. The
store and existing vectors are preserved while recall degrades to full text.
As a last resort on an unmanaged host, `pip install -r requirements-semantic.txt`
installs into its active environment; manual installs can be dropped on update.

## Option 3 — Ollama (trusted local daemon)

If you already run [Ollama](https://ollama.com), use its embeddings endpoint:

```bash
ollama pull nomic-embed-text        # 768-dim; mxbai-embed-large and bge-m3 also work
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=ollama
export LCM_EMBEDDING_MODEL=nomic-embed-text
export LCM_EMBEDDING_QUERY_PREFIX="search_query: "
export LCM_EMBEDDING_DOCUMENT_PREFIX="search_document: "
# LCM_OLLAMA_BASE_URL defaults to http://localhost:11434
/lcm embed warmup && /lcm embed backfill --apply
```

Prefixes are raw strings; quote them to preserve the trailing space. Empty defaults leave
requests and vector identities unchanged. Only Ollama and OpenAI-compatible providers use them.
[Contributor dhaern's measurements](https://github.com/electricsheephq/lcm-x/issues/1118#issuecomment-6358033279) on two public BEIR sets
found model-card prefixes raised EmbeddingGemma 2 nDCG@10 by about 6% and pixie-rune-8k by under 3%; these are not LCM-X history numbers.

The contributor reported that `embeddinggemma-2:270m` with Ollama's default batch size rejected inputs over about 2,000 tokens
with HTTP 413 even with `num_ctx` 8192; raising `num_batch` to 8192 as well worked. LCM-X does not send Ollama `options`.
Try setting both on the model with Modelfile `PARAMETER` entries; that method was not verified by the contributor.

Ollama requests set `truncate: false`, so an input that exceeds the model's context fails loudly
rather than being silently truncated to a misleading embedding. As with Voyage, an Ollama
timeout/network failure after transport starts is acceptance-ambiguous and is not automatically
resent.

The shipped Ollama provider assumes the configured endpoint is trusted and does
not apply the cloud privacy gates for sensitive patterns or raw-text consent. Keep
`LCM_OLLAMA_BASE_URL` on a verified local/loopback service. A forwarded,
container-network, private-network, or remote Ollama endpoint may send text off
the machine; endpoint-aware locality remains #337.

Bulk document embedding uses `LCM_EMBEDDING_BACKFILL_TIMEOUT_S` as its
per-provider-operation deadline (120 seconds by default) for Voyage, Ollama,
and fastembed. This is intentionally separate from the latency-sensitive
`LCM_EMBEDDING_QUERY_TIMEOUT_S` (3 seconds by default), so a normal document
batch or local model load is not aborted by the interactive query policy. The
optional `LCM_EMBEDDING_BACKFILL_BUDGET_S` still caps the whole apply run
between batches (`0`, the default, means no whole-run cap); the lease and
post-call ownership CAS remain authoritative independently of both timeouts.

## Option 4 — OpenAI-compatible API (reuse an existing embeddings service)

If you already run any OpenAI-format `/v1/embeddings` endpoint — SiliconFlow,
llama.cpp server, vLLM, or a local embedding model on a workstation — point
LCM at it instead of running a second embedding stack:

```bash
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=openai-compatible
export LCM_EMBEDDING_MODEL=BAAI/bge-m3     # any model id your endpoint serves
export LCM_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
export LCM_EMBEDDING_API_KEY=sk-...        # falls back to SILICONFLOW_API_KEY
/lcm embed warmup && /lcm embed backfill --apply
```

`LCM_EMBEDDING_BASE_URL` is the API root (a trailing slash is stripped) and
`/embeddings` is appended. The request body is the standard OpenAI shape —
`{"model": ..., "input": [...]}` with a `Bearer` authorization header — so any
compatible server works, including local llama.cpp/vLLM instances that cost
nothing per call. For an asymmetric model such as `nomic-embed-text`, set `LCM_EMBEDDING_QUERY_PREFIX="search_query: "`
and `LCM_EMBEDDING_DOCUMENT_PREFIX="search_document: "` to its model-card values.
Both HTTP providers send the complete prefixed input after any cloud privacy transform;
server size errors follow the existing error path, without silent truncation or a vector write
for the rejected item. There is no new client-side input limit.

The same deadline, spend-guard, and identity rules apply as for the other
remote providers: `LCM_EMBEDDING_QUERY_TIMEOUT_S` bounds interactive queries,
`LCM_EMBEDDING_BACKFILL_TIMEOUT_S` bounds each backfill batch, and vectors are
stored under the full `(provider, model, ...)` identity so switching back to
this provider reactivates existing vectors without re-backfilling.

## What you get

`lcm_grep` gains two modes on top of the existing ones:

- `semantic` — paraphrase-tolerant vector search; local providers make this $0
- `hybrid` — keyword ∪ semantic, fused with reciprocal-rank fusion (RRF); the best
  "have we discussed X?" mode. (Fusion is RRF only — there is no external reranker.)

Semantic and hybrid requests use one absolute deadline beginning at `lcm_grep` entry. It includes
provider resolution, query embedding, optional NumPy import, bounded KNN, result hydration, any FTS
fallback, both hybrid arms, and fusion. A semantic failure can degrade to full-text with
`degraded_to_fts`; the fallback uses separate read-only SQLite connections with progress
interruption. If hybrid already computed FTS results before its semantic arm times out, it may
return that existing payload without starting new I/O. If no usable result exists when time runs
out, the request returns an explicit `timeout` error and starts no later fallback/arm. Provider
authentication errors also remain operator-visible instead of degrading. Explicit `full_text`
mode itself is unchanged and byte-for-byte identical to prior behavior.

Role, time, conversation, and broader-session filters degrade to raw FTS before provider work,
because summaries cannot prove those raw-message dimensions. Source is different: SQL first selects
a bounded candidate window, then verifies descendant source lineage inside that window. The lineage
walk is itself capped; a missing legacy `source` column or an over-budget lineage graph fails closed
with `unverifiable_provenance`, so it never becomes an allow-all. A source-filtered semantic result
therefore reports bounded coverage rather than claiming universal pre-bound source coverage.

## New content is embedded automatically

With embeddings enabled (the default) and a registered profile (automatic local
preparation, or explicit `/lcm embed warmup` for other providers), LCM-X
embeds new content in the background: after each leaf summary is published (condensed summaries
are not embedding targets, as with `/lcm embed backfill`), and
once when a session binds. Each pass handles at most one backfill batch per corpus, runs off the
turn path, and uses the same lease, privacy policy, operation budget and provider timeout as
`/lcm embed backfill`. If a manual backfill or another process holds the lease, the pass skips;
the next pass picks up whatever is still unembedded. A manual backfill started while a pass runs
is refused until the pass ends, as with two manual runs. There is no new setting.

- **Summaries** are embedded with any provider. With a cloud provider this means new summaries
  are sent to it automatically, one bounded batch per pass, without a manual backfill.
- **Message chunks** are embedded automatically only with fastembed, or with Ollama whose
  `LCM_OLLAMA_BASE_URL` host is loopback (`localhost`, `127.0.0.0/8`, `::1`). Each pass takes new
  messages first, then older history newest first, one bounded batch at a time, so its cost follows
  new messages, not the store, and an existing store's recent messages are covered before its old
  history; `/lcm embed backfill` embeds the whole backlog in one go. On a cloud provider or a
  remote Ollama the chunk corpus stays operator-initiated: `--confirm-raw-text` authorizes one invocation and is never
  stored, so run `/lcm embed backfill --corpus chunks --apply --confirm-raw-text` yourself when you
  want chunk recall.
- If the provider is unavailable (for example fastembed is not installed), the pass is skipped and
  counted; the first skip in a process logs a `WARNING`, later ones log at `DEBUG`.

`/lcm status` shows `embedding_backlog_summaries`, `embedding_backlog_chunks` (what the automatic passes
still have ahead of them, counted over at most 2,000 messages and shown as `N+` beyond that; a chunk the
passes skipped inside the embedded range is not counted, and a dry run of
`/lcm embed backfill --corpus chunks` gives the exact total; with any rows left
`uncertain` by a failure after dispatch, which only `--retry-uncertain` resends), the time and outcome
of the last incremental pass (`embedding_incremental_last_pass`) and the process's skipped-pass
count (`embedding_incremental_failures`). `/lcm embed backfill --apply` is still the way to embed
an existing history in one go instead of waiting for bounded background catch-up.

## Performance & footprint

- NumPy remains optional. When available, it enables vectorized search and the
  float32 chunk-loader fast path; the import guard and pure-Python fallback remain.
  Install it in the Python environment that actually runs Hermes: for a user-managed
  environment, `python -m pip install numpy`; for a managed build, use that host's
  supported dependency-installation mechanism rather than modifying a generated venv.
  Restart long-lived hosts after changing their dependency environment.
- Float32 chunk matrices are loaded directly from their little-endian BLOBs,
  avoiding a round trip through Python float objects. Joining the BLOBs still
  copies bytes; this is not an end-to-end zero-copy pipeline. Int8 decoding,
  stored vector identities, and search coverage are unchanged.
- Reproduce the loader comparison without a profile or provider call with
  `python benchmarks/benchmark_float32_chunk_loader.py --count 2000 --dim 384`.
  The synthetic benchmark checks exact IDs, matrix values and dot-product scores;
  timings are host-dependent and are not asserted by tests.
- Metadata/id resolution uses a temp-table join rather than a giant `IN (...)` list, so it scales
  past the SQLite host-parameter limit that previously failed near ~32k ids (validated to 40k).
- Without numpy, search scans the most recent `LCM_EMBEDDING_BOUNDED_SCAN_ROWS` vectors (default
  2,000) and reports `coverage: bounded`. The candidate enumeration is bounded at the SQL layer
  (`ORDER BY` recency `+ LIMIT`), so a large corpus never materializes every id in host memory.
- With numpy, the cache is still only for that bounded candidate set and is keyed by canonical
  identity, transactional `data_version`, and candidate ids; it is not a corpus-sized matrix.

## Switching or removing providers

A prefix change (including removal) registers a new identity: restart Hermes, then run `/lcm embed warmup` and `/lcm embed backfill --apply` (add corpus/consent options for chunks). Provider caches keep their existing provider/model keys.

Prefixed profiles use distinct `summary+prefix` / `chunk+prefix` tasks as well as the
revision digest. v0.28.1 cannot select these profiles or write into them on rollback;
it uses an existing unprefixed corpus or falls back to full text. Empty prefixes keep
the original requests, tasks and identity hashes.

If you tested prefixes on v0.29.0-rc1, restart on the updated build and run
`/lcm embed warmup`, then `/lcm embed backfill --apply`. Warmup retires rc1's
revision-only profiles, even if you have since removed the prefixes. Their derived
vectors are retained but never reused: an earlier rollback may have mixed them with
unprefixed vectors. Backfill rebuilds the selected corpus from the unchanged source
rows. For chunks, also run `/lcm embed backfill --apply --corpus chunks --confirm-raw-text`
with your existing chunk policy. No raw messages or summary DAG nodes are migrated.

Change provider/model → run `/lcm embed warmup` (registers the new profile as the current identity)
→ `/lcm embed backfill --apply` (embeds under the new identity; the previous model's vectors are
kept separate and never mixed). Every vector is published under the exact identity that produced it
— the identity is captured at provider-resolution time and carried through the write, so switching
the active provider A→B mid-backfill can never rebind an A-vector onto B. If A becomes inactive
after its request was accepted but before publication, the exact A request is atomically moved to
`uncertain`, the still-owned backfill lease is released, and the run stops before another dispatch.
Because each identity
`(provider, model, revision, dim, dtype, byteorder, task)` owns its own vectors, switching **back**
to a previously-registered provider reactivates it with its existing vectors — no re-backfill
needed. The stored representation is currently restricted to `float32` / `little` / `summary`;
unsupported identity variants are rejected rather than normalized onto another profile.

Backfill records each actual remote dispatch durably. Every accepted sub-batch is published
immediately under the captured identity and current lease CAS. If remote acceptance is ambiguous or
local publication fails after acceptance, those rows become `uncertain` and normal discovery will
not bill them again. Recovery is deliberately operator-authorized:

```bash
/lcm embed backfill --apply --retry-uncertain --limit 32
```

The authorization is bound to the exact oldest uncertain rows selected by that invocation, up to
`--limit`; the risky recovery run does not mix in ordinary pending rows. Their durable uncertainty
markers are not cleared before discovery or dispatch, and any row not successfully published
(including a skipped row, definitive rejection, budget stop, or lease loss) remains `uncertain` for
another explicit decision. The command reports the uncertain count and warning because retrying may
rebill. Disable everything with `LCM_EMBEDDINGS_ENABLED=false` — data stays, behavior reverts to
FTS-only instantly.

## The chunk corpus — source-derived text, privacy transform, and consent gate

`/lcm embed backfill` has two corpora, selected with `--corpus`:

- `summary` (default) — embeds the generated **summaries** of your history.
- `chunks` — embeds provider input derived from **raw, verbatim message text**, chunked by `--policy`
  (`conversational` | `heads` | `full`), for verbatim/chunk-KNN recall.
  `conversational` (the default) embeds every non-empty user and assistant turn, however short; `heads` adds the
  head and error chunks of tool results; `full` embeds every role.
- `both` — runs the summary backfill, then the chunk backfill, in one command.

The distinction matters for privacy. The summary corpus starts from model-generated summaries; the
chunk corpus starts from raw message bytes, including tool-result output and error/traceback content.
For known cloud providers the configured provider-input privacy transform applies by default
(`LCM_EMBEDDING_PRIVACY_ENABLED` unset) and rejects residual detector matches before
transport; `LCM_EMBEDDING_PRIVACY_ENABLED=false` sends chunk text raw under the `privacy:off`
revision. Neither posture redacts your durable history — the chunk corpus is read from raw
message bytes either way. This protects configured patterns but cannot
classify every possible sensitive fact in source-derived chunk text.

Because of this, `--corpus chunks --apply` and `--corpus both --apply` **refuse** on a cloud provider
unless you pass an explicit acknowledgment:

```bash
/lcm embed backfill --corpus chunks --apply --confirm-raw-text
```

FastEmbed is in-process, so the gate is waived. Ollama is also exempt in the
current implementation; that is safe only when its configured endpoint is a
verified trusted local/loopback service (#337). Dry runs (no `--apply`) never
send anything and never require the flag.

> **Provider-input boundary.** Durable history is never retro-redacted. The cloud embedding transform
> applies only to outbound provider input, uses pattern-only placeholders, and binds its policy to the
> vector identity. Prefer a local provider for the chunk corpus when broader source content should not
> leave the machine even after configured detector matches are removed.
