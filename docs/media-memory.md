# Media memory: design proposal (images, audio, video and stream capture)

**Status:** proposed design, docs only. This changes no code and no schema.

**Decisions so far:**
- Images v1 is opt-in per profile first.
- Opt-in stream capture (screen, audio, camera) is designed now, so people can use an early preview and the community can grow it
  (section 10).
- Face and speaker recognition waits (#1716).

**Slotting:** the roadmap owner slots each build behind the v0.29.0 screens and the local-embeddings default gate (#1586). Each
build ships only when its own gate passes.

**Related issues:**
- #1708: image recall research and spike results;
- #1702: words next to an image stay searchable;
- #1714: evidence before schema;
- #1715: local embedder evaluation;
- #1171: forget;
- #1191: derived memory;
- #1589: timeline;
- #448: closed when nobody needed it; this design revisits it.

## 1. Why

Agents increasingly see screenshots, photos, diagrams and, later, audio and video.

**What exists today.** LCM-X keeps an image inside its stored turn, inline or as an externalized payload. Once #1702 lands, the
words around it are searchable.

**What is missing:**
- search that finds an image by what it shows;
- a way to return an image to a vision-capable model as an image;
- a way for an agent to remember what it saw and heard outside the conversation.

**The direction this serves:**
- **one memory, any harness (H3):** the same store serves Hermes, Claude Code, Codex and others through adapters and MCP;
- **memory for agents in the world (H4):** visual and auditory memory for agents on laptops today, and for on-device agents and
  robots later.

## 2. What we measured (#1708)

The spike used public COCO photos inside synthetic chat turns, with text-only distractors and local models only.

| Measurement | Local CLIP image vectors | Comparison |
|---|---|---|
| 1,000 photos, R@1 / R@10 | 0.44 / 0.85 | text embedding over turn text plus a perfect human caption: 0.29 / 0.66 |
| Screenshots, R@10 | 0.69 (terminal screens 0.22) | text search over the exact on-screen text: 1.00 |
| CLIP + caption text, equal-weight RRF, R@1 | 0.43 (CLIP alone 0.44) | CLIP + turn text only, equal-weight RRF: 0.07 |

Cost on one Apple M4 Pro CPU (4 threads): 9–57 ms to embed an image and 2–8 ms to embed a query. The CLIP image and text models
take about 0.58 GB on disk. The protocol and full results are in #1708.

**What this supports:**
- Local CLIP is good enough to build on for photos.
- Screenshots need OCR text.
- Never fuse an arm that carries no signal: turn text such as "here's the photo" says nothing about the image.

**What it does not prove:**
- real questions;
- real screenshots read by a real OCR engine;
- galleries beyond 1,000 images (recall fell as the gallery grew);
- the smallest supported host;
- any fusion weight: the 2:1 weight was chosen and reported on the same data.

## 3. Prior art: gbrain

Garry Tan's MIT-licensed [gbrain](https://github.com/garrytan/gbrain) imports supported files, including opt-in image files, into
pages and chunks. Its design is a useful reference for:
- a modality on each chunk;
- separate image vectors next to text vectors;
- optional OCR text stored with image chunks;
- image queries refined by a text query over the image-vector index.

Its text-query routing already supports image-only, mixed text and image, and opt-in unified search. We draw on these ideas; no
gbrain code is proposed for inclusion.

gbrain is TypeScript on Bun. It runs on embedded PGlite or server Postgres, with vector search. Its default text embedder is a
hosted model. Image embedding is opt-in and needs a compatible multimodal model. It also supports local text-embedding providers.

**Where LCM-X differs:**
- **Unit of memory.** The LCM-X unit is a media occurrence inside a stored conversation turn, with that turn's lineage. gbrain's
  image importer creates file-based image pages; gbrain also ingests conversations as pages.
- **Local processors.** LCM-X would run its media processors locally and in process, in Python, with their indexes in the same
  SQLite store as the conversation record.
- **Result lists.** Media results start as their own list (section 8). gbrain's mixed routes fuse image and text hits into one
  ranking; it also has image-only routes.

We considered packaging gbrain inside LCM-X and decided against it: the stack, storage and unit of memory differ. Using both side
by side is possible; we have not built or tested that integration.

## 4. What "lossless" means for media

The store is lossless by design, and media keeps that promise.

**Conversation media is record.** This covers:
- images and audio a user or tool put into a turn, including tool-result and computer-use screenshots;
- images and audio the agent produced.

LCM-X keeps the original. When a host gave the model a smaller rendition, LCM-X keeps that too, with its dimensions, byte size and
token cost. Everything the model saw is kept, like text. The only exception is forget on request (#1171).

**Captured streams are perception, not record.** A screen recorder, microphone or robot camera produces far more than any model
sees. Captured bytes are kept under the declared capture policy of their capture session (section 10), and never described as
lossless. When bytes expire, the catalog still records that they existed and that the original is no longer available.

**Promotion.** When a captured excerpt is delivered to a model, that excerpt becomes conversation media. It is then kept like any
other record, whatever the capture policy says.

**Derived data never replaces its source.** OCR text, transcripts and vectors that LCM-X computes are disposable and rebuildable
while their originals exist. They are not a substitute for deleted originals.

**Recorder-supplied text is record.** Text a recorder supplies with a capture segment is different: LCM-X cannot regenerate it.
It is part of the capture record until forgotten; only its full-text index and vectors are disposable.

## 5. Storage

**SQLite stays the catalog at every scale we can name. Bytes live in files. Vector search is the part that grows.**

| Tier | What | Where |
|---|---|---|
| T1 lossless record | the message row as today, plus an ordered occurrence map: (message, part index or field path) → asset hash, MIME type, byte size, dimensions or duration, the rendition given to the model, whether received or produced, and optional `captured_at`, `source` and `place` | `lcm.db` |
| T2 conversation media bytes | each asset once, content-addressed (`ab/cd/<sha256>.<ext>`), immutable | the store's payload folder |
| Capture catalog | capture sessions, segments and recorder-supplied text, all authoritative (section 10) | new tables in `lcm.db` |
| Capture bytes | closed, immutable segment files owned by one occurrence; time-ordered folders | the store's payload folder |
| T3 derived index | OCR text, captions, transcripts, keyframe records as `segment(t0, t1, kind, text, vector)`, image and audio vectors, full-text search over derived text, processor id and version | new tables in `lcm.db` first |

**Why the optional fields matter.** They cost nothing at ingest and cannot be recovered later. Capture time differs from message
time for media that arrives late or from a device. Part-level occurrence identity also extends host event identity to the part
within a message.

**Why conversation media is content-addressed.** It is never deleted (except by forget), so repeated images can safely share one
file.

**Why capture bytes are owned per occurrence.** They expire. Deduplicating them would require reference counting before every
deletion, so the hash stays integrity metadata only.

### One database first

`VISION.md` promises one SQLite database with a folder beside it. Text vectors already live in `lcm.db`, and media vectors reuse
the numerical helpers in `vector_store.py`. Their schema stays separate: the existing summary-embedding metadata accepts summaries only, and the chunk-vector tables are keyed
to message text spans.

**Writes stay small.** Model work (OCR, embeddings, transcription) runs outside any transaction. Only small result batches are
written, through the store's existing writer, as vector publishing does today.

**Why not a second file now.** A second database file would isolate write locks, but not CPU, memory or disk contention. New
tables are additive, which keeps the rollback invariant (I5) simple.

**When to split.** Measure ingest wait, WAL growth, and backup and rebuild time during representative indexing. Move T3 into its
own SQLite file only if one of those breaks a declared bar. T3 is derived, so the move migrates no authoritative data. The split
file is then bound to its store's identity. A backup is a consistent T1 snapshot plus its files; T3 can be rebuilt.

**Short read transactions.** Long-running readers must not hold the catalog open; on large SQLite catalogs they starve the WAL
checkpoint ([screenpipe #6981](https://github.com/screenpipe/screenpipe/issues/6981)).

### Why bytes never go into a database

Bytes are larger than anything else here, by tens to thousands of times, and no database choice changes that. The table is a
calculated example of continuous capture, excluding overhead:

| One camera, one frame every 5 s | per day | per year |
|---|---|---|
| keyframes | 17,280 | 6.3M |
| 512-d vectors, float32 / int8 / 1-bit | 35 MB / 9 MB / 1.1 MB | 12.9 GB / 3.2 GB / 0.4 GB |
| keyframe JPEGs (~60 KB) | ~1 GB | ~380 GB |
| 720p video (~1.5 Mbit/s) | ~16 GB | ~5.9 TB |
| audio, Opus 24 kbit/s | ~260 MB | ~95 GB |

**Files are the better container.** They stream, they can be checked independently, and they back up incrementally.

**Precedent.** ROS 2 moved its default recording format from SQLite to MCAP, partly to get crash-resilient writes at high
throughput ([announcement](https://discourse.openrobotics.org/t/psa-default-ros-2-bag-storage-format-is-changing-to-mcap-in-iron/28489)).
Frigate keeps camera recordings as short MP4 segments, with one SQLite catalog row per segment file
([docs](https://docs.frigate.video/configuration/record)).

**Never one file per frame for continuous capture.** Unbounded per-frame files have exhausted the macOS vnode cache and caused kernel panics
([screenpipe #5612](https://github.com/screenpipe/screenpipe/issues/5612)). Media folders also carry a `.metadata_never_index`
marker, which asks Spotlight to skip them.

**Backup and export.** Today's backup reports the payload folder and asks for a separate copy (#1075). Capture files must appear in
that inventory too, and portable export grows with T2.

### Vector search at scale

**Conversation images.** Start by measuring an exact batched scan. Raw vector payloads are small (100k images × 512 dimensions ×
float32 ≈ 205 MB), but payload size is not memory use or query latency.
- The trigger for more is measured cold or warm p90, RAM or catch-up time breaking the smallest-host bars, not a fixed count.
- Before adopting an approximation, compare exact batched search with the existing binary prescreen on image recall.

**Long captures.** Embedding every 5-second frame of one camera would reach a million vectors in about 58 days. The preview
embeds coarser segments of supplied text and requires a time window, so the same measured trigger applies: latency, RAM or
catch-up breaking the bars, not a count. When it fires, quantized vectors and an approximate index are the candidates. Queries
there are time-bounded first, which fits #1589.
- Such an index is a rebuildable file beside SQLite, sharded by time, and tied to its model and corpus generation.
- Its recall is measured against exact search.

**Candidates to measure, none chosen:**
- on device: usearch (single file, memory-mapped, int8 and binary);
- in SQLite: sqlite-vec, pending measurement;
- embedded: the Lance format (LanceDB);
- server: pgvector.

### Why not Postgres

On a laptop or a robot, Postgres means another database engine to embed or run. It does not fix the byte problem above.

Its real strengths are concurrent writers, a mature HNSW index and multi-tenant operation:
- **Concurrent writers.** The H3 rule of one writer per store removes this advantage.
- **HNSW.** Building it over millions of float vectors needs as much RAM in Postgres as anywhere else.
- **Multi-tenant operation.** This is the case where Postgres fits: a separately justified multi-user server tier,
  behind the T3 interface. The H3 memory service ships over SQLite first.

## 6. Write order and recovery

The same order holds for conversation media and capture segments.

1. Write the complete bytes to a temporary file in the same folder, then fsync it.
2. Rename the file to its final name, then fsync the folder.
3. Commit the catalog rows (the message and its occurrence refs, or the segment) in one transaction.
4. If writing the bytes fails:
   - conversation media keeps today's lossless representation, or reports a persistence failure;
   - a capture segment is refused with an error.

   Never commit a ref to missing bytes.

**Crash safety.** A crash before step 3 leaves at most an orphan file. It never leaves a committed ref without its bytes.

**Indexing.**
- Indexing work is discovered from committed rows that lack the current processor version. A queue, if any, is only a wake-up hint.
- On restart, the indexer rescans that difference and upserts idempotently.
- Reusable results (an image vector, OCR text) are keyed by asset hash **and** processor version. Results that depend on the
  surrounding turn are keyed by occurrence.
- A missing derived row means "not indexed yet". Recall reports coverage the way it reports `preparing` today.

No cross-file transaction or queue service is needed.

## 7. Migration of existing media (additive)

Existing inline media and externalized payloads stay where they are.

**What migration does:**
- It extracts supported inline media into T2 and adds occurrence-map rows.
- It does not rewrite or delete existing rows or payload files. Old refs, `lcm_expand` pagination and replay identity stay
  unchanged.
- Repeated images share bytes and image-level results, never occurrence identity.
- Unknown parts and URL-only images keep their original form. A URL is not captured media.
- A malformed item stays on the old readable path, and the migration is not marked complete.

**Removing the old representation** is a separate, later decision. It needs:
- exact reconstruction;
- a legacy-ref resolver;
- unchanged pagination and replay identity;
- rollback compatibility.

**#1702** is independent and comes first. Its text-extraction rules carry over when refs arrive. Test text before, between and
after several images in one message.

## 8. Processing and retrieval

### Processing

A processor per modality writes derived rows:
- **image:** an image vector and OCR text, plus an optional caption from a local or host model, off by default. The OCR engine and
  the embedder are chosen by measurement. The ViT-B/32 used in the spike is a 2021 model; candidates include newer on-device
  multimodal embedders (#1715).
- **audio:** transcript segments with time ranges, plus an optional audio embedding.
- **video:** keyframes as images, plus the audio track as audio.

**Lazy, bounded models.** Models load only when media arrives and unload when idle. They run inside a thread budget beside the
summariser.

**Declared cost.** Each processor declares its model id, version and measured RAM, so `doctor` can show coverage, backlog, cost
and disk use.

**Cloud is optional.** Cloud processors are optional upgrades, off by default, under the same per-profile privacy setting as cloud
text embeddings.

### Retrieval

- **A separate list.** Recall returns a bounded `media_results` list next to the existing text results, sharing the response
  budget. Text ranking stays unchanged.
- **What a hit carries.** Each media hit is keyed by occurrence. It carries the asset hash, source turn, matched part or time range,
  processor version, and the derived text with its provenance.
- **The model asks for media.** On the agentic path the model sets an explicit `modality` parameter on `lcm_recall` and `lcm_grep`.
  LCM-X recall is a tool the model calls, so the model can say what it wants. `lcm_grep` stays lexical by default, so a grep never
  loads an image model.
- **Fusion comes later.** Fusing image-vector and OCR evidence, or weighting media against text on the pre-answer path, waits for
  measurement.
  - Weights come from a dev split and are confirmed on a held-out set, with a noise floor.
  - Weighted RRF already exists in `retrieval_core.py`. Its identity handling must first support parts: two images in one turn,
    or one image in many turns.
- **Detail questions** ("what colour was the bus?") are expand-then-ask, not retrieval, and get their own measurement.

## 9. Returning media to the model

A core media result holds:
- an occurrence ref;
- the original MIME type and bytes;
- the derived text.

**Host adapters.**
- `lcm_expand` returns a string today. Returning an image needs structured tool results.
- Each host adapter maps the result to that host's content parts in the next model request. The first proof is one Hermes round
  trip.
- A host that cannot take image parts gets a clear text-and-ref fallback. A server-local path never stands in for content the
  client can reach.

**MCP.** MCP tool results support image, audio, resource-link and embedded-resource content; there is no video type.
- The remote memory service returns a resource link that the host fetches.
- Video goes out as keyframes or as a link.

**Size limits.** Every host caps part size, and Base64 adds a third. A declared cap and a downscale step produce a delivery
rendition, stored as its own T2 object; the original stays.

**Historical media** is resolved through the existing message-lookup scope plus part identity, never by opening a bare hash.

## 10. Stream capture (opt-in)

### Principle: LCM-X is the memory, not the recorder

Recorders capture: a desktop screen recorder, a phone app, a microphone bridge, a ROS 2 node. Each one closes a segment and hands
it to LCM-X through one API. The core ships no camera, microphone or screen drivers.

This keeps the core small and portable, and leaves room for community recorders. Capture-scope controls (excluded apps, private
windows, pause rules) belong in the recorder, as plain configuration; screenpipe, Rewind and Omi all work this way. See the
guidance for recorders below.

### Preview slice (the first build)

| Part | Preview behaviour |
|---|---|
| Switch | Off by default, per profile. Turning it on changes nothing else: chat ingest, ranking, compaction and context assembly behave exactly as before. |
| Intake | One call, `ingest_segment`. Its arguments are capture session, source, incarnation id and sequence number, UTC `t0` and `t1`, MIME type, a closed segment file, and optional supplied text with its time ranges and producer name and version. It runs inside the process that owns the store's writer. A later service endpoint calls the same function and never opens a second writer. Intervals must be finite with `t0 < t1`, and text ranges must fall inside them; point events wait for their own rule. |
| Receipt | A durable receipt, or an explicit failure, including a capacity refusal. Retry identity is the store-scoped key (capture session, source, incarnation id, sequence); recorders keep an incarnation id because sequence counters reset after restarts. A retry is identical only when interval, MIME type, hash and supplied text all match; anything else is rejected. A retry after expiry returns the existing segment and its current availability, without restoring bytes. |
| Catalog | New tables for capture sessions, segments (interval, MIME type, size, hash, file, availability) and supplied text (with full-text search and an optional vector). A session's policy is frozen; changing the policy starts a new session, and new defaults never apply to old segments. Capture never creates chat messages or summary nodes. |
| Files | Closed, immutable segment files in open containers (MP4, Ogg), in date and hour folders under the payload folder. They play without LCM-X. Bytes are written before the catalog row (section 6). A small manifest next to each segment (source, span, codec, policy, retry key) lets the segment inventory be salvaged if the catalog is lost; a crash that loses recording metadata otherwise needs a reindex ([rosbag2 #1079](https://github.com/ros2/rosbag2/issues/1079)). Full recovery, including supplied text and lifecycle state, needs the SQLite backup. |
| Supplied text | Transcripts, OCR text and captions supplied by the recorder; part of the capture record (section 4). Optional local text vectors are computed by a capture-specific catch-up pass on the existing bounded maintenance scheduler, triggered on segment arrival and at startup. Full-text recall works before vectors exist. Each vector records the source-text hash, model, configuration, dimension and type, so a change finds new work rather than relabelling old vectors. |
| Recall | An explicit capture scope on `lcm_recall` with a required UTC window. A segment matches when its interval overlaps the window: `t0 < window_end` and `t1 > window_start`. Scope and interval filters apply before full-text or vector limits. Each hit returns the matched subrange of text, not the whole segment. Capture-time recency stays separate from relevance, and arrival time never counts as capture time. Hits come back in a separate `capture_results` list showing source, interval, ingest time and whether the original is still available. |
| Expand | Returns the segment's supplied text and metadata. No original bytes reach a model in the preview. |
| Policy | Recorded on every capture session. The preview has three settings: `off` (the default), `keep_days` with `max_bytes` (whichever is reached first), and `pinned` segments that never expire automatically. Eviction removes whole segments, oldest first; it never removes the open segment, a pinned segment, or a segment still waiting for a required processor. Pinned and pending bytes count against the cap. If an incoming segment cannot fit after eligible eviction, intake returns a capacity refusal and keeps existing data; an oversized segment is refused, and headroom for the temporary copy is reserved. Intake also pauses when free disk falls below a floor. Each pause or refusal leaves a catalog marker and a log line, never a silent loss. Preview text vectors are optional and never a reason to hold bytes. |
| Expiry | Deletes bytes only. The catalog row and supplied text stay searchable, marked `original_available: false`. Expiry survives interruption (pending, deleted, recorded) and never races backup or reads. |
| `doctor` | Shows the policy, retained and expired bytes, segment and derived-row counts, indexing backlog and processor identities. Capture health is the age of the last committed segment, not process liveness: a health check that reports OK while capture has stopped misleads ([screenpipe #7679](https://github.com/screenpipe/screenpipe/issues/7679)). |
| Backup | A quiesced SQLite snapshot, which holds the supplied text and lifecycle state, plus the retained capture files, named in the backup instructions. |

**Default policy when someone turns capture on:** keep 7 days of bytes, capped at 10 GB per profile. This is our starting choice,
informed by prior art:
- Omi's local rewind keeps 7 days;
- screenpipe keeps 14 days when retention is enabled, and is off by default
  ([retention.rs](https://github.com/screenpipe/screenpipe/blob/main/crates/screenpipe-engine/src/retention.rs));
- Frigate keeps alert and detection clips for 10 days ([record config](https://docs.frigate.video/configuration/record));
- Dayflow caps recordings at 10 GB, deleting whole segments oldest first and never the active one
  ([source](https://github.com/JerryZLiu/Dayflow/blob/main/Dayflow/Dayflow/Core/Recording/StoragePreferences.swift)).

Supplied text is small and stays until forgotten.

**Why off by default.** Microsoft Recall's first preview was on by default and drew wide criticism. It shipped opt-in
([update](https://blogs.windows.com/windowsexperience/2024/09/27/update-on-recall-security-and-privacy-architecture/)).

**Guidance for recorders** (recommendations, not limits):
- **Scope controls in the recorder:**
  - excluded apps, windows and URLs, pre-seeded with password managers;
  - private browsing windows skipped;
  - pause on lock, sleep or idle;
  - a visible recording indicator.
- **Audio:** 60 s segments, voice-activity-gated or with Opus DTX. Silence otherwise costs full bitrate (about 14 MB/h at
  32 kbit/s).
- **Screen:** a still every few seconds, only when the screen changed, encoded into 60–900 s video segments (never one file per
  frame). Windrecorder publishes about 2 to 100 MB
  per hour.
- **Camera:** 10–60 s segments, stream-copied without re-encoding.

**What reads stored capture data in the preview:**

| Reader | Behaviour |
|---|---|
| Recall | capture scope only |
| Expand | supplied text and metadata only |
| Embedding maintenance | optional text vectors |
| Expiry job | segment lifecycle |
| `doctor` | coverage and size |
| Backup inventory | capture files |
| Schema bootstrap | capture tables are authoritative records, never rebuildable caches |
| Context assembly and compaction | never: capture data does not enter the context window except through a recall result the model asked for |
| `lcm_grep` | unchanged in the preview |
| Portable export | waits for its own design (v0.32.0) |

**Acceptance for the preview** (bars declared before the run; the capture schema lands on main only after the first measured use
in section 13, Track B):
1. **Path tests:**
   - ingest, identical retry (same receipt), conflicting retry (rejected) and restart;
   - time-window recall: overlap rule, capture-time ranking, late arrival;
   - expiry, then recall showing `original_available: false`;
   - interrupted expiry, resumed;
   - a crash between bytes and commit leaves only an orphan file.
2. **Compatibility:**
   - `docs/process/schema-changes.md` applies: new DDL lives in `db_bootstrap.py`, through store initialization, and the PR records
     the format marker, the oldest supported reader and the removal release;
   - stores written by the previous release, and by a fleet-shaped writer, upgrade with no lost rows (I6), including retained and
     expired capture state;
   - the previous release reads and writes the upgraded store (chat included) and replays emitted rows without duplicating them,
     and the candidate reopens it (I5);
   - an unknown newer format is refused;
   - capture records never enter disposable-cache remediation.
3. **No default change:**
   - with capture off, the full suite passes;
   - the release fast-tier recall gate shows zero per-question flips against the previous release.
4. **Cost, on the smallest supported host:** a synthetic 8-hour capture (one 60 s audio segment and one screen segment per minute,
   with transcripts and OCR text) must stay within declared bars for:
   - capture ingest latency;
   - chat-turn ingest latency, unchanged within noise;
   - WAL growth;
   - time-window recall p90;
   - disk use within the budget.

### After the preview (grown with the community)

| Feature | What it adds |
|---|---|
| Processor plugins | OCR, transcription, captions and image or audio vectors computed by LCM-X from segment bytes. Each declares model, version and cost. Segments are not evicted before required processors have run. |
| Snapshot mode and keep-what-was-used | A rolling buffer (`ring`) that keeps bytes only when an event, a processor or the model asks, padded a few seconds on each side. This is ROS 2's `--snapshot-mode` and Frigate's event clips. It is the intended default for cameras, which are the heaviest stream and the one that records bystanders. It needs verified tracking of what the model actually read. |
| `derived_only` policy | Drop bytes once the required processors commit. The simplest privacy story, but nothing can be re-transcribed later. |
| Hooks and read API | On segment commit, on derived row, live transcript; time-ranged search and capped clip fetch over MCP (H3). |
| Importers | MCAP and rosbag2, plus other open recorders' catalogs and exports. |
| A published on-disk format | Versioned and documented, so the data outlives any one tool. |
| Event-based retention | Keep segments with motion, speech or detected objects longer, as Frigate does. |
| Delivering captured media to a model | Uses promotion (section 4). |
| A remote ingest endpoint | Through the H3 memory service. |
| Quantized, time-sharded vector search | For long captures (section 5). |

## 11. Harness boundary

**The shape.** One internal content-part shape (text, image, audio or video, with a MIME type and bytes or a ref) sits around the
core as a narrow media codec. It does not replace the context-engine contract.

**What the enclosing record keeps:**
- role and host event identity;
- source and session;
- part order;
- unknown host fields.

**The split of work.** Adapters translate native content and tool results. Recorders call `ingest_segment`. Persistence, processing
and ranking stay in the core. Data URIs and local paths are host-specific encodings.

**Timing.** This contract belongs in the H3 adapter specification while the adapters are designed (Hermes and Claude Code in
v0.30.0), as documentation and types. Otherwise a text-only cross-host message shape would need a second migration later.

## 12. Seams reserved, not built

- **Entities.** A `media_mention(entity, occurrence, span)` record lets faces, speakers and objects join the derived-memory entity
  graph (#1191). Without it, media becomes a silo.
- **Time.** One `segment(t0, t1, ...)` record shape across transcript, scene, caption and capture feeds the timeline (#1589).
- **Biometrics.** Face and speaker recognition waits (#1716). The default is no identification. Identification would need an
  opt-in per profile, and forget (#1171) must exist first.

## 13. Sequencing

The two tracks share one storage model and are slotted independently.

### Track A: conversation media (images, then audio and video)

**Cheap evidence first (#1714): no schema, no product code.**
1. #1702 (v0.29.0).
2. A read-only census of our own stores: media parts by source and size, counts only.
3. Spike 2:
   - real local OCR on real-style screenshots;
   - a stronger local image embedder (#1715);
   - 10,000 or more images;
   - weights chosen on dev and confirmed held-out;
   - harm on the text boards;
   - reader-judged answer quality.
4. The content-part contract in the H3 adapter specification.

**Then a first measured use** (disposable fixtures, no schema on main): save, restart, retrieve the right occurrence, and deliver
the original image through Hermes. This includes one queued image after a restart and one historical image.

**Then the build, each step behind its own gate:**
1. The schema, with additive migration and forward and rollback tests (`docs/process/schema-changes.md`).
2. A quality and host gate:
   - paired held-out retrieval and reader-answer quality;
   - no regression on the text boards or their guards;
   - cold and warm p90, RAM and disk on the smallest supported host, offline, including ARM hosts with SD-card storage.
3. Opt-in release, then a default decision.

Audio and video follow the same pattern.

### Track B: stream capture

**First measured use, before any capture schema on main.** Import closed segments and supplied text from one existing open recorder.
The import runs through a small named adapter in the writer's process; a local import command is enough. On a disposable store:
1. restart;
2. answer one bounded time-window question with the right source;
3. expire the bytes;
4. retrieve the retained text with an unavailable-original marker.

All of this must stay within bars declared before the run.

**Then the preview** (section 10), with its own acceptance bar. It can take the next free slot without waiting for Track A, because
it stores recorder-supplied text and delivers no media to models.

Processor plugins, snapshot mode and media delivery follow, each behind its own gate.

## 14. Open questions

- **Which recorder first.** The first measured use needs one existing open recorder and a small adapter: for example, a local
  speech transcriber or an open screen recorder's export. Licences differ (screenpipe moved to a commercial licence in 2026), so
  the choice is part of that step.
- **Capture defaults.** Are 7 days and 10 GB right once real users try the preview? Revisit with their numbers.

## 15. Alternatives considered

| Option | Why not now |
|---|---|
| Package gbrain inside LCM-X | Different stack, storage and unit of memory (section 3). |
| A separate media database from the start | Isolates write locks only, with no measured contention yet; revisit by the split rule (section 5). |
| Bytes as BLOBs in SQLite | Large media must stream; slower backup and VACUUM. ROS 2 moved recordings off SQLite. |
| One file per frame | Exhausts filesystem caches; segmented containers fit time-range reads and capture policies (section 5). |
| Capture rows as chat messages | They would enter compaction, summaries and chat ranking with the wrong clock; capture gets its own catalog (section 10). |
| LCM-X owning device drivers | Recorders differ per platform and device; one intake API lets the community build them (section 10). |
| Postgres by default | Another engine on every device, and it does not fix byte volume. |
| Equal-weight fusion with turn text | Measured to bury image hits when the text arm carries no signal (section 2). |
| Cloud-only multimodal embeddings | Contradicts no cloud by default; kept as an optional upgrade. |
| An approximate vector index for conversation images now | No measured latency or RAM failure yet (section 5). |

## Review record

**First draft.** Two independent reviews:
- a cross-model architecture review against the code (agree with changes: four blocking findings, four should-fix);
- a strategy red-team (three blocking findings).

**Their findings, folded in:**
- write order;
- what lossless means for media;
- additive migration;
- host delivery;
- one database first;
- evidence before schema;
- a model-set modality instead of keyword routing.

**This revision adds:**
- stream capture, designed from a code-seam analysis of LCM-X and a survey of capture systems (screenpipe, Microsoft Recall,
  Rewind, Dayflow, Frigate, ROS 2 and MCAP, Omi and others);
- gbrain statements checked against its source.
