# Media memory: design proposal (images first, then audio, video and streams)

**Status:** proposed direction, not scheduled. Multimodal memory is outside the current cycle (`ROADMAP.md` H2, item 8) and belongs
to H4, where one measured use comes before any schema commitment. This document changes no code and no schema. It answers #1708's
open questions (storage, and how hits reach the reader) and revisits #448, which was closed when nobody needed it.
**Related:** #1708 (image recall research and spike results), #1702 (words next to an image stay searchable), #1171 (forget),
#1191 (derived memory), #1589 (timeline).

## 1. Why

Agents increasingly see screenshots, photos, diagrams and, later, audio and video. Today LCM-X keeps an image inside its stored
turn, inline or as an externalized payload. Once #1702 lands, the words around it are searchable. But no search finds the image
by what it shows, and nothing returns it to a vision-capable model as an image.

The direction this serves:
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

Cost on one CPU host: 9–57 ms per image and about 2 ms per query. The CLIP image and text models take about 0.58 GB on disk.

**What this supports:**
- Local CLIP is good enough to build on for photos.
- Screenshots need OCR text.
- Never fuse an arm that carries no signal. Turn text such as "here's the photo" carries none.

**What it does not prove:**
- real questions;
- real screenshots read by a real OCR engine;
- galleries beyond 1,000 images: recall fell as the gallery grew;
- the smallest supported host;
- any fusion weight: the 2:1 weight was chosen and reported on the same data.

## 3. Prior art: gbrain

[gbrain](https://github.com/garrytan/gbrain) (MIT, by Garry Tan) already does multimodal memory over files and folders. We reuse
its design ideas, not its code:
- a modality on each indexed chunk;
- separate image vectors next to text vectors;
- OCR text as the searchable text of an image;
- search by image, fused with text search.

Where we differ, and why:
- **Storage.** gbrain uses Postgres with pgvector. LCM-X stays one SQLite database with folders beside it, so it runs on devices
  with no server.
- **Models.** gbrain defaults to a cloud multimodal embedder. LCM-X is local-first; cloud processors are optional upgrades.
- **Fusion.** Media results start as their own list (section 8).
- **Scope.** gbrain indexes files. LCM-X indexes media inside conversation turns, with turn lineage.

We considered packaging gbrain inside LCM-X and decided against it. It is a different stack (TypeScript/Bun and Postgres), it uses
cloud models by default, and it solves files rather than conversation media. The two compose side by side.

## 4. What "lossless" means for media

The store is lossless by design, and media keeps that promise.

**Conversation media is record.** This covers:
- images and audio a user or tool put into a turn, including tool-result and computer-use screenshots;
- images and audio the agent produced.

LCM-X keeps the original. When a host gave the model a smaller rendition, it keeps that too, with its dimensions, byte size and
token cost. Everything the model saw is kept, like text. The only exception is forget on request (#1171).

**Continuous streams the model never saw are perception, not record.** Examples are a robot's camera or an always-on microphone.
They are kept under a declared capture policy shown in `doctor`, and never described as lossless. That mode records which
originals are no longer available, and narrows what expand and rebuild promise for them.

**Derived data never replaces its source.** OCR text, transcripts and vectors are disposable and rebuildable while their
originals exist. They are not a substitute for deleted originals.

## 5. Storage

**SQLite stays the catalog at every scale we can name; bytes live in files; vector search is the part that grows.**

| Tier | What | Where |
|---|---|---|
| T1 lossless record | the message row as today, plus an ordered occurrence map: (message, part index or field path) → asset hash, MIME type, byte size, dimensions or duration, rendition given to the model, received or produced; optional `captured_at`, `source` and `place` | `lcm.db` |
| T2 original bytes, attachments | each asset once, content-addressed (`media/ab/cd/<sha256>.<ext>`), immutable | a folder beside `lcm.db` |
| T2 original bytes, streams (H4, later) | time-segmented append-only containers, content-addressed per closed segment; a ref is (segment hash, t0, t1); a capture policy drops whole segments | a folder beside `lcm.db` |
| T3 derived index | OCR text, captions, transcripts, keyframe records as `segment(t0, t1, kind, text, vector)`, image and audio vectors, full-text search over derived text, processor id and version | new tables in `lcm.db` first |

The optional fields cost nothing at ingest and cannot be recovered later. Capture time differs from message time for media that
arrives late or from a device. Part-level occurrence identity also extends host event identity to the part within a message.

### One database first

`VISION.md` promises one SQLite database with a folder beside it. Text vectors already live in `lcm.db`, and media vectors reuse
the numerical helpers in `vector_store.py`. Their schema stays separate: the existing embedding tables accept summaries only.

Model work (OCR, embeddings, transcription) runs outside any transaction. Only small result batches are written, through the
store's existing writer discipline, as vector publishing does today. A second database file would isolate write locks, but not
CPU, memory or disk contention. New tables are additive, which keeps the rollback invariant (I5) simple.

**When to split.** Measure ingest wait, WAL growth, and backup and rebuild time during representative indexing. Move T3 into its
own SQLite file only if one of those breaks a declared bar. T3 is derived, so the move migrates no authoritative data. The split
file is then bound to its store's identity. A backup is a consistent T1 snapshot plus its T2 files; T3 can be rebuilt.

### Why bytes never go into a database

Bytes are two to four orders of magnitude larger than anything else here, and no database choice changes that:

| One camera, one frame every 5 s | per day | per year |
|---|---|---|
| keyframes | 17,280 | 6.3M |
| 512-d vectors, float32 / int8 / 1-bit | 35 MB / 9 MB / 1.1 MB | 12.9 GB / 3.2 GB / 0.4 GB |
| keyframe JPEGs (~60 KB) | ~1 GB | ~380 GB |
| 720p video (~1.5 Mbit/s) | ~16 GB | ~5.9 TB |
| audio, Opus 24 kbit/s | ~260 MB | ~95 GB |

Files stream, deduplicate and back up incrementally. Streams need segmented containers rather than one file per frame. Robotics
went the same way when ROS 2 made MCAP its default recording format and kept SQLite for queries. Large payloads already live
beside `lcm.db` for the same reason. T2 makes portable export and backup much larger, and both must say so.

### Vector search at scale

For conversation images, an exact batched scan is fine until measurement says otherwise. Raw vectors are small:
100k images × 512 dimensions × float32 ≈ 205 MB. The trigger for more is measured cold or warm p90, RAM or catch-up time breaking
the smallest-host bars, not a fixed count. Before adopting an approximation, compare exact batched search with the existing binary
prescreen on image recall.

Streams are different. One camera reaches a million vectors in about 58 days, so H4 stream search needs quantized vectors and an
approximate index from its first version. Queries there are time-bounded first, which fits #1589. Such an index is a rebuildable
file beside SQLite, sharded by time and tied to its model and corpus generation. Its recall is measured against exact search.

Candidates to measure when H4 starts, none chosen:
- on-device: usearch (single file, memory-mapped, int8 and binary);
- in SQLite: sqlite-vec, whose approximate index is still in alpha;
- server: pgvector, or the Lance format.

### Why not Postgres

On a laptop or a robot, Postgres adds a server, connection pools and migrations, and does not fix the byte problem above. Its
real strengths are concurrent writers, a mature HNSW index and multi-tenant operation. The H3 rule of one writer per store removes
the first. Building HNSW over millions of float vectors needs as much RAM in Postgres as anywhere else. Postgres or Lance stays a
coherent option behind the T3 interface for a separately justified multi-user server tier. The H3 memory service ships over
SQLite first.

## 6. Write order and recovery

1. Write the complete bytes to a temporary file in the same folder and fsync it. Rename it to its content-addressed name, then
   fsync the folder.
2. Commit the message and its ordered occurrence refs in one T1 transaction.
3. If writing the bytes fails, keep today's lossless representation, or report a persistence failure. Never commit a ref to
   missing bytes.

**Crash behaviour.** A crash before step 2 leaves at most an orphan file. It never leaves a committed ref without its bytes.

**Indexing.**
- Work is discovered from T1 occurrences that lack the current processor version in T3. A queue, if any, is only a wake-up hint.
- On restart, the indexer rescans that difference and upserts idempotently.
- Reusable results, such as an image vector or OCR text, are keyed by asset hash **and** processor version. Anything that depends
  on the surrounding turn is keyed by occurrence.
- A missing T3 row means "not indexed yet". Recall reports media coverage the way it reports `preparing` today.

No cross-file transaction or queue service is needed.

## 7. Migration of existing media (additive)

Existing inline media and externalized payloads stay where they are.

**What migration does.**
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

#1702 is independent and comes first. Its text-extraction rules carry over when refs arrive. Test text before, between and after
several images in one message.

## 8. Processing and retrieval

### Processing

A processor per modality writes T3 rows:
- **image:** a CLIP-class vector and OCR text, plus an optional caption from a local or host model, off by default. The OCR engine
  and the embedder are chosen by measurement; the ViT-B/32 used in the spike is a 2021 model.
- **audio:** transcript segments with time ranges, plus an optional audio embedding.
- **video:** keyframes as images, plus the audio track as audio.

**Cost controls.** Models load lazily, only when media arrives, and unload when idle. They run inside a thread budget beside the
summariser. Each processor declares its model id, version and measured RAM, so `doctor` can show coverage, backlog, cost and disk
use.

Cloud processors are optional upgrades, off by default, under the same per-profile privacy setting as cloud text embeddings.

### Retrieval

- **A separate list.** Recall returns a bounded `media_results` list next to the existing text results, sharing the response
  budget. Text ranking stays unchanged.
- **What a media hit carries.** Each hit is keyed by occurrence. It carries the asset hash, source turn, matched part or time
  range, processor version, and the derived text with its provenance.
- **The model asks for media.** On the agentic path, the model sets an explicit `modality` parameter on `lcm_recall` and
  `lcm_grep`. Keyword routing fails on anaphora ("that one") and on non-English queries. `lcm_grep` stays lexical by default, so
  a grep never loads CLIP.
- **Fusion comes later.** Fusing CLIP and OCR evidence, and weighting media against text on the pre-answer path, waits for
  measurement. Weights come from a dev split and are confirmed on a held-out set, with a noise floor. Weighted RRF already exists
  in `retrieval_core.py`. Its identity handling must first support parts: two images in one turn, or one image in many turns.
- **Detail questions** ("what colour was the bus?") are expand-then-ask, not retrieval, and get their own measurement.

## 9. Returning media to the model

A core media result holds:
- an occurrence ref;
- the original MIME type and bytes;
- the derived text.

**Hosts.**
- `lcm_expand` returns a string today. Returning an image needs structured tool results.
- Each host adapter maps the media result to that host's content parts in the next model request. The first proof is one Hermes
  round trip.
- Over MCP, the content types are image, audio, resource link and embedded resource; there is no video type. The remote memory
  service returns a resource link the host fetches. Video goes out as keyframes or a link.
- A host that cannot take image parts gets a clear text-and-ref fallback. A server-local path never stands in for content the
  client can reach.

**Sizes.** Every host caps the size of a part, and Base64 adds a third. A declared cap and a downscale step produce a delivery
rendition, stored as its own T2 object; the original stays.

**Historical media** is resolved through the existing message-lookup scope plus part identity, never by opening a bare hash.

## 10. Harness boundary

**The shape.** One internal content-part shape (text, image, audio or video, with a MIME type and bytes or a ref) sits around the
core as a narrow media codec. It does not replace the context-engine contract.

**The enclosing record** keeps:
- role and host event identity;
- source and session;
- part order;
- unknown host fields.

**The split of work.** Adapters translate native content and tool results. Persistence, processing and ranking stay in the core.
Data URIs and local paths are host-specific encodings.

**Timing.** This contract belongs in the H3 adapter specification while the adapters are designed (Hermes and Claude Code in
v0.30.0), as documentation and types. Otherwise a text-only cross-host message shape would need a second migration later.

## 11. Seams reserved, not built

- **Entities.** A `media_mention(entity, occurrence, span)` record lets faces, speakers and objects join the derived-memory
  entity graph (#1191). Without it, media becomes a silo.
- **Time.** One `segment(t0, t1, ...)` record shape across transcript, scene and caption feeds the timeline (#1589).
- **Biometrics (flag only).** Face and speaker embeddings are biometric data. The default is no identification. Identification
  needs an opt-in per profile, and forget (#1171) must exist first.

## 12. Sequencing and triggers

**Do now. Cheap, no schema, no product code.**
1. **#1702** (v0.29.0): the words next to an image stay searchable.
2. **Media census.** A read-only census of our own stores, counts only: image and audio parts by source (upload, tool result,
   computer-use screenshot), with their sizes and share of turns.
3. **Spike 2** on public and synthetic data:
   - a real local OCR engine on real-style screenshots;
   - a stronger local image embedder;
   - a gallery of 10,000 or more;
   - weights chosen on dev and confirmed held-out;
   - harm on the text boards: media hits must not displace text hits within the delivery budget;
   - reader-judged answer quality, not only R@k.
4. **The content-part and media-ref contract** in the H3 adapter specification.

**Triggers. The first to fire starts step 5.**
- The census shows media on a declared share of turns in at least two stores, and a measured miss rate on real media questions.
- An on-device or robot deployment needs it.
- The H3 second-harness identity work needs part-level refs.

When a trigger fires, the roadmap owner picks the release. Images v1 touches the lossless ingest path, so it is a release theme of
its own.

**Then, each step gated on bars declared before its run:**

5. **First measured use** (disposable fixtures, no schema on main): save, restart, retrieve the right occurrence, and deliver the
   original image through Hermes. This includes one queued image after a restart and one historical image.
6. **Schema:** the minimal occurrence map and T3 tables. The migration is additive and has forward and rollback tests
   (`docs/process/schema-changes.md`). Interrupted ingest and reindex must recover.
7. **Quality and host gate:**
   - paired held-out retrieval and reader-answer quality;
   - no regression on the text boards or their guards (abstention, latency);
   - cold and warm p90, RAM and disk on the smallest supported host, offline, including ARM hosts with SD-card storage.
8. **Default decision** (opt-in per profile, or on by default), then a canary.

**Horizons, not decisions:**
- audio;
- video and streams, with the stream T2 shape and quantized time-sharded search;
- MCP media delivery, which needs no Postgres backend;
- a server-tier T3 backend.

## 13. Open questions

- **Default.** Opt-in per profile first, or on by default once step 7 passes? The models add about 0.6 GB of disk, and about
  0.5–1.9 GB of RAM while active.
- **Capture policy for streams.** When, and with which defaults (section 4)?
- **Biometrics.** Is opt-in identification ever on the roadmap (section 11)?

## 14. Alternatives considered

| Option | Why not now |
|---|---|
| Package gbrain inside LCM-X | Different stack and storage, cloud by default, and it indexes files rather than conversation media (section 3). |
| A separate media database from the start | Isolates write locks only, with no measured contention yet; revisit by the split rule (section 5). |
| Bytes as BLOBs in SQLite | Large media must stream; slower backup and VACUUM. |
| One file per frame for streams | Millions of files a year per camera; segmented containers fit time-range reads and capture policies (section 5). |
| Postgres by default | Needs a server; does not fix byte volume; contradicts local-first and on-device. |
| Equal-weight fusion with turn text | Measured to bury image hits when the text arm carries no signal (section 2). |
| Keyword media-intent routing | Fails on anaphora and non-English queries; the model sets `modality` instead (section 8). |
| Cloud-only multimodal embeddings | Contradicts no cloud by default; kept as an optional upgrade. |
| An approximate vector index for conversation images now | No measured latency or RAM failure yet (section 5). |

## Review record

The draft had two independent reviews:
- a cross-model architecture review against the code (agree with changes: four blocking findings, four should-fix);
- a strategy red-team (three blocking findings).

Their findings are folded in:
- write order (section 6);
- what lossless means for media (section 4);
- additive migration (section 7);
- host delivery (section 9);
- one database first (section 5);
- triggers before any schema (section 12);
- a model-set modality instead of keyword routing (section 8).
