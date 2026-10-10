# Portable recall v1

Decision: opt-in local recall and transcript capture around host-owned compaction.
Tracker: [#1010](https://github.com/electricsheephq/lcm-x/issues/1010).
Source baseline: `003c3ae5f71f1bf307423ae7f9c13d3bf2a8dcff`.

## Ownership and boundaries

Hermes keeps its native context engine. Portable hosts retain native compaction.
A local stdio MCP facade calls production `tools.lcm_recall`, `lcm_describe`
and `lcm_expand` through a small retrieval context backed by real MessageStore
and SummaryDAG objects. It does not instantiate LCMEngine or install a fake
Hermes host. Normal package import is safe: host registration is explicit.
Portable status reports its own observed capabilities rather than invented
Hermes compression or lifecycle counters.

Each server is bound at launch to an explicit local storage root and project
scope, never a tool-selected filesystem path. Portable sessions use separate
corpora keyed by host kind, host instance/project scope and session/branch ID.
A tool selects an explicit session in that fixed namespace; its retrieval
context contains only that session's dedicated corpus. There is no implicit
all-project or cross-session search. Forks use distinct corpora and retain
explicit parent/event provenance without granting access to a parent corpus.
No existing Hermes/live database is opened or migrated by portable capture.

## Ingestion and exactness

Adapters consume supported finalized JSONL events, incrementally. Identity
includes namespace, host, session, source generation, event ID and part index.
Repeated identity and digest is a no-op. Changed content under an existing
identity is preserved as a distinct version/conflict; it never replaces the
old row. Content equality alone never deduplicates separate events. Truncated,
rotated, incomplete or unsupported input is reported; capture does not delete
previous evidence or claim complete coverage.

Use the existing metadata table for identity mappings, original source-envelope
metadata, event digests and capture checkpoints. A narrow store operation must
apply normal ingest protection and commit source rows plus identity/checkpoint
metadata atomically. Do not separately append then advance a cursor. Do not
use Hermes shadow UID bindings as authoritative deduplication. No new schema,
store/service or semantic deduplication is required.

Preserve chronology, content, observed source timestamps, assistant tool calls
and tool result call IDs. Pairing is session-local; unmatched results remain
unmatched. Preserve raw host envelope/source references where available without
changing host transcripts. Exact references point into captured stored content;
full-envelope preservation and unsupported host events are separately disclosed.
No source event is silently rewritten or discarded to improve a score.

Portable recall uses reference-strict `answer_ready` delivery. Return the local
`lcm:<store_id>:<start>-<end>` reference together with a stable corpus ID,
canonical host/session/event identity and projection version. Portable handles
are corpus-qualified and checked before expansion. Offsets are character offsets
in stored content, not host JSONL byte positions. Pagination must round-trip
without changing the quoted text. Retrieved content and summaries are untrusted
evidence, never executable instructions. Summaries are leads, not exact quotes.

## Host assistance

Claude Code is the first qualification target, then Codex manual installation.
Capture only from an explicitly configured transcript root and the session in
the hook payload. Synchronize finalized content before compaction. Hook failures
are reported and fail open for host compaction; never emit a compaction-blocking
decision to manufacture parity. Resume/after-compaction assistance supplies a
source-linked continuation capsule bounded to 2,000 tokens. The capsule covers
explicit goal, constraints, decisions and unresolved work; inferred/uncertain
facets are marked. Details remain in exact recall. Native compaction stays on.

Status distinguishes configured, observed and qualified capture, pre/postcompact
and injection capabilities; context replacement is false. An unqualified host
is MCP-only, with explicit ingestion. Hook-emulation tests are not native-host
qualification. Session-bound capture and recall work when assistance is disabled.

## Optional scoring and evaluation

A scorer consumes a query and at most 30 scoped immutable candidate IDs/texts,
returning finite scores or abstention. Code validates membership, completeness,
deterministic ties, deadline and unchanged source offsets. Default policy is
baseline, including on unavailable models, timeout or invalid output. Local BGE
is a pinned optional arm; Jev `jev-1.13.0` requires explicit public/synthetic input
opt-in and an aggregate input-token/cost ledger. Private memories never leave
this layer by default. Numerical/date/identity/evidence rules remain code-owned.

Freeze OFF and LOCAL baselines separately and keep the same candidate pools,
model, corpus and token allowance across ranking arms. Candidate recall,
delivered supporting-span recall, ranking quality and reader correctness are
separate evidence. Existing #898/#950 gates remain authoritative; this work
changes no default policy. Promotion requires the approved held-out +3pp span
recall, positive paired 95% interval, no category loss over 2pp, no added
no-answer false positives and p95 added latency <=1s. Otherwise record no-adoption.

Qualification requires six paired native-only/adapted synthetic scenarios per
host, source reconstruction, scope/fork/correction/tool integrity, restart-safe
ingestion and disabled-adapter fallback. Full evaluations run on GitHub-hosted
CI; focused local checks follow the machine budget. One independent semantic
review covers the stable preservation/isolation candidate.

## Distribution

Prepare self-contained Claude plugin and manual Codex packages. Observe actual
installed-version capabilities and disclose lower tiers. The OpenAI public
package cannot contain lifecycle hooks and local MCP needs its documented
partner route. Prepare a dossier; do not create HTTPS hosting or submit/publish.
Claude plugin and connector routes are distinct. No package receipt establishes
release, customer, universal lossless compaction or store approval.
