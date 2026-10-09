# Portable recall frozen evaluation v1

Issue #1010; fusion #950; scorecard #898. This is an optional ranking seam and
an evaluation instrument. It changes no default, persistence, schema, capture,
tenant/session authority, retention, or answer reader. Corpus labels assert
operator-approved public/synthetic input; they do not classify private data.

## Interface and fallback

`score_candidates(query, candidates, scorer=None, timeout_s=1.0)` accepts at most
30 immutable `Candidate` objects from one already-authorized scope. IDs,
references, text, offsets and metadata survive unchanged. Returned model IDs
must equal the pool exactly; unknown/missing/duplicate IDs, nonfinite/out-of-range
scores and model exceptions fall back. Stable ties preserve baseline order.
Unavailable, timeout, abstention and invalid output preserve baseline objects and
order. Malformed caller pools raise before dispatch. Deadlines bound delivery;
an in-flight worker can finish after fallback. Four global worker slots and one
busy operation per bundled scorer bound accumulation. No automatic retries.

`LocalBGEScorer` loads cached weights only, on CPU, at immutable Hugging Face
revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` of
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
The optional direct dependency is `sentence-transformers==5.1.1`; its
[pinned API source](https://github.com/UKPLab/sentence-transformers/blob/v5.1.1/sentence_transformers/cross_encoder/CrossEncoder.py)
supports revision/local-files-only and prediction activation. CI must record its
resolved transitive environment. Weights/download/prewarm are separate from
timed scoring. Sigmoid logits are ranking scores, without calibration claims.

`PublicJevScorer` requires explicit public/synthetic permission, a dedicated
single-process `JevBudget` journal, and the existing `TYPESAFE_API_KEY` environment
reference. No key value is logged. Each candidate has an independent three-level
Score question in one request; query and candidate text are the only state.
Gold, IDs and provenance never enter it. Pin `jev-1.13.0`, validate the response
identity, full answer membership, legend, probability sums, expectation and
confidence. Optional minimum confidence must be selected on dev before holdout;
abstention falls back, without claiming answer abstention or access permission.
See [API](https://docs.typesafe.ai/api), [Score](https://docs.typesafe.ai/primitives/score)
and [model limits/pricing](https://docs.typesafe.ai/models).

Before every uncertain send the metadata-only journal fsyncs a conservative
64,000-input-token reservation. Confirmed valid usage replaces the reservation;
failure retains it. Replay preserves reservations and refuses duplicate refunds.
Aggregate ceiling is 5,000,000 input tokens and $1, at the recorded $0.042/M input
price (output free). Request JSON is capped at 32,000 bytes. Root must verify
current price/context limits before its isolated campaign; this cap is not
permission for future private egress. Do not share a journal across processes,
erase it, retry uncertain sends, or provision a new credential destination.

## Frozen corpus and production capture

`write_frozen` creates an immutable JSONL corpus with SHA-256 of canonical case
records, source/embedding identity, input kind and OFF/LOCAL posture.
`read_frozen` checks digest and identities. Every arm ranks the exact same pool;
lexical orders existing FTS matches first, then preserves remaining baseline
hits. No model expands the candidate pool. Requested candidate cap is 30;
production `lcm_recall` currently caps returned hits at **25**.

Root integrates an evaluation-only `candidate_sink` in existing
`benchmarking.longmemeval.evaluate_question`, before stores close. The sink's
keyword contract is question/config/store/dag/provider, tmp_dir,
embeddings_enabled/provider_name/chunk_provider, hits, store_id_to_turn and
capture_ms. `freeze_production_pool` accepts those production hits and explicit
`(stored_message_id, start, end)` span annotations. Offsets are stored character
projections, verified against unchanged text. Ambiguous/transformed snippets
fail capture. Summary hits remain candidates but receive no inferred raw gold.

LongMemEval revision `2ec2a557f339b6c0369619b1ed5793734cc87533` has session/turn
labels; it does not supply exact character-span gold. Root must add independently
verified annotations to measure this gate. Missing gold stays UNMEASURED.
Public dataset candidate export uses this sink, not the existing ID-only dump.
Preserve the existing pinned instrument/provider configuration and official
session/turn metrics separately; no new corpus/source identity is inferred.

The synthetic generator defines exactly 120 unique conversations: 40 dev and
80 held-out, ten categories with 12 cases each (four dev/eight holdout): exact ID,
paraphrase, correction, negation, date, contradiction, no answer, fork, scope and
literal tool argument. Gold is the precise fixture answer message; no-answer
gold is empty. Parent/other-project decoys are separate fixture inputs, excluded
from this selected corpus; the portable-host lane must independently prove
fork/scope isolation. This benchmark alone cannot prove an access boundary.
`capture_synthetic` reuses the real store/DAG and existing production instrument.
OFF disables embeddings; LOCAL uses an explicitly named real FastEmbed model.
Synthetic fixtures and embeddings can be generated only in an authorized run.

## Measurements and gate

Report frozen candidate exact-span Recall@30 (actual 25 cap), delivered exact-span
Recall@5, nDCG@10 (one support credit per distinct annotated source), no-answer
false positives, scorer status, p95 added latency and total delivery latency.
A no-answer false positive is any nonempty delivered ranking; this seam's
fallback does not hide or remove baseline false positives. Candidate coverage is
identical across arms and exposes ceiling loss before reranking. Full-span
delivery, answer correctness, calibrated probabilities and live harness behavior
are separate proof; absent a reader, answer correctness is explicitly UNMEASURED.

Only tune rubric/confidence on dev. Freeze it before holdout; do not select among
repeated holdout trials. Gate per posture: gain at least three percentage points
in delivered span Recall@5; deterministic paired conversation bootstrap 95%
interval strictly positive; each category loss no worse than two points; no
extra no-answer false positives; p95 added latency at most 1,000ms. At most one
case per conversation is accepted for the paired gate. All scorer cases must
complete without fallback, and all exact labels must exist. Otherwise report
NO_ADOPTION or INSUFFICIENT_EVIDENCE. PROMOTE covers this frozen corpus only.
Unavailable local weights generate an explicit unmeasured arm, never fake rows.

## Root-owned remote commands

After integrating the sink and source, run the focused test file remotely:
`python -m pytest -q tests/test_optional_scorer.py`.
Install the optional dependency in the isolated remote environment and cache the
exact BGE revision there; the evaluator itself refuses a model download.

`python scripts/eval_portable_recall.py freeze-synthetic --posture OFF --source-identity COMMIT_SHA:OFF --output synthetic-off.jsonl`

`python scripts/eval_portable_recall.py freeze-synthetic --posture LOCAL --local-model PINNED_EMBEDDING_MODEL --source-identity COMMIT_SHA:EMBEDDING_IDENTITY --output synthetic-local.jsonl`

`python scripts/eval_portable_recall.py run --input synthetic-off.jsonl --output off-bge.json --scorer bge`

Root-only approved synthetic campaign, reusing its aggregate journal:
`python scripts/eval_portable_recall.py run --input synthetic-off.jsonl --output off-jev.json --scorer jev --allow-public-jev --jev-ledger synthetic-jev-budget.jsonl`

Repeat LOCAL as a separate configuration. Dev and holdout reports are separate,
and evidence paths are create-only. The author ran no tests, model downloads,
inference, installation or production retrieval; remote execution and independent
semantic review remain integration gates.

## Public full-S holdout capture

`freeze-public` consumes the previously downloaded file named `longmemeval_s`.
The root's existing `lcm_longmemeval.py fetch --dataset-label s` pins revision
`2ec2a557f339b6c0369619b1ed5793734cc87533`; record its download receipt and byte
SHA-256 before this command. The loader hashes the same bytes it parses and
requires the expected checksum, 500 questions, unique question IDs and matching
instrument revision. The supplied checksum proves byte identity against that
receipt; this command does not independently contact Hugging Face or certify
the operator's download origin. It never downloads the dataset.

Freeze the synthetic registration first: the command requires a same-posture
120-case corpus with the exact registered 40-dev/80-holdout identities/splits.
Tune only on those 40 dev cases, freeze model/rubric/confidence, then treat all
500 public S questions as holdout. There is no public limit or dev split option.
The report records the synthetic corpus digest, dataset coordinates/checksum,
source identity and resolved summary/chunk embedding identities. CI must also
pin and record its cached local embedding model weights/environment.

Each question calls the existing instrument exactly once, with its default
`top_k=10`, reranking disabled, and the fresh per-question store. Its callback
freezes the same already-returned production hits before closure, without a
second baseline or re-expansion. Baseline output preserves all current
per-question session/turn metrics, mode/degradation labels and latency. Summary
turn markers remain coarse. OFF has embeddings disabled; LOCAL uses the named
real FastEmbed model. They are separate measurements under the existing
[methodology](../../benchmarks/METHODOLOGY.md).

No public exact-span gold is inferred from `has_answer` or summary labels.
Every public frozen case has `gold=null`, so exact-span metrics and promotion
remain unmeasured/insufficient. Ambiguous/transformed source snippets produce
`UNMEASURED_CAPTURE: source_projection` in that question's baseline row and are
excluded from frozen scoring; report full/captured/missing counts explicitly.
Both outputs are create-only. Do not compare that subset as full-S span recall.

`python scripts/eval_portable_recall.py freeze-public --dataset DATASET_DIR/longmemeval_s --dataset-sha256 PINNED_DOWNLOAD_SHA256 --posture OFF --dev-corpus synthetic-off.jsonl --source-identity COMMIT_SHA:OFF --baseline-output public-off-baseline.json --output public-off-pool.jsonl`

`python scripts/eval_portable_recall.py freeze-public --dataset DATASET_DIR/longmemeval_s --dataset-sha256 PINNED_DOWNLOAD_SHA256 --posture LOCAL --local-model BAAI/bge-small-en-v1.5 --dev-corpus synthetic-local.jsonl --source-identity COMMIT_SHA:PINNED_EMBEDDING_IDENTITY --baseline-output public-local-baseline.json --output public-local-pool.jsonl`

Optional cached local scoring of the exported pool uses the existing `run`
command. Public hosted inference is outside the approved synthetic-only Jev
campaign and requires its own explicit data/recipient/budget authorization.

## Synthetic citation-bearing delivery correction

The synthetic frozen pool opts into the existing production
`detail=answer_ready, include=verbatim` delivery profile, matching the portable
facade. The helper/instrument defaults remain `snippets/all`; the public full-S
baseline uses those unchanged defaults. Answer-ready applies its real production
selection, bounded hydration, reference validation and response limits. It is a
distinct delivery configuration and must not be compared as the same snippet
baseline. Scorers still receive exactly the same frozen pool within each run.

Citation-bearing capture requires published integer `content_offset` and
`content_returned_chars`; the length and stored source slice must match the
unchanged delivered text exactly. Missing/wrong offsets fail closed. Highlighted
FTS snippets are not stripped, guessed or assigned offset zero. The corpus digest
includes every case's detail/include profile; the header and report disclose
profiles and observed maximum candidate count.

**These 120 scenarios each have one session.** Production answer-ready caps
delivery at five hits per session, so these frozen pools contain at most five.
The synthetic header records that cap, and header/report explicitly report
`recall_at_5_gain_possible_by_permutation=false`. Pure reranking cannot improve
delivered Recall@5 in this configuration. nDCG may change, but it cannot satisfy
the approved three-point span-recall promotion gate. A measured NO_ADOPTION is
valid; unavailable/incomplete runs remain INSUFFICIENT_EVIDENCE. Retain the
baseline unless the approved gain gate is demonstrated. This correction does
not redesign fixtures, raise production limits or change the adoption policy.
