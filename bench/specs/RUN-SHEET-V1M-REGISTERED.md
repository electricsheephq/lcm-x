# RUN SHEET (DRAFT) — LongMemEval-V1 MEDIUM registered run (flagship coverage row)

Status: DRAFT — finalize (pins + final shard assignment) at #198 merge + smoke completion.
Registered BEFORE launch; this sheet is the registration. G0 channel: coverage row (flagship).

## 1. Measured basis (spend ladder rung 2 — smoke, in progress)
- Smoke `lme-m-smoke-1`: 25q, Voyage voyage-context-3, PR #198 instrument. Per-question
  wall-clock derived mechanically from per-question DB mtimes at n=16 stores:
  **median 30.5 min, mean 32.5, p90 39.7, range 28.4–44.2** (sequential; interval spacing
  confirms no internal question-level parallelism). Final numbers re-derived from the completed
  smoke report before launch; if they differ materially (>±20% on median), shard math re-runs.
- OpenRouter dependency: **NONE** (confirmed mechanically: smoke progressed 17q on an exhausted
  OpenRouter balance; scoring is retrieval-metric, embeddings are Voyage). The credit outage
  blocking other lanes does NOT block this run.

## 2. Shard plan
- Full set: 500 questions (dataset longmemeval_m, sha fb5413e3…, revision 2ec2a557… — pinned in
  specs/PREP-V1-MEDIUM-DATASET.md; prepared-dir manifest verifies per-question shas fail-closed).
- Sequential estimate: 500 × 32.5 min ≈ 271 h. **Shard count: 6** (≈45 h ≈ 1.9 days/arm).
  Rationale: 8 shards saves ~11 h but doubles exposure to the disk-I/O-starvation class
  (concurrent SQLite writers on this machine, documented ops incidents) and approaches Voyage
  RPM limits during ingest bursts; 4 shards wastes a day. 6 = measured-risk middle. Each shard:
  own `HERMES_HOME`, own `TMPDIR` under LEXAR, own question-id file (500 split 6 ways by fixed
  interleave qid[i::6] — no adjacency clustering), own output dir; freeze-writers protocol armed.
- Shard launch stagger: 5 min apart (bridge/model-init windows never overlap — the 300s-timeout
  class fired twice under concurrent inits).
- Recovery: per-question outputs are append-only; a dead shard resumes by re-running its qid
  file (completed questions detected + skipped via prepared/output manifests). EINTR-class
  continuation pattern applies.

## 3. Pins (finalized at launch — every value from a command, none hand-typed)
- Instrument: #198 MERGED at 9f29d4b2 (squash, 2026-08-02T04:14:53Z; 11 bot passes, 45 threads resolved). Product: engine sha at
  merge. Dataset: prepared-dir manifest sha. Env: FULL `_EnvFieldSpec`-inventory diff per F49 §5
  (every flag explicit or listed default — first run under the new standing rule).
- Provider: voyage / voyage-context-3, batched embedding path (#198), key from Keychain at
  runtime, never in configs.

## 4. Pre-declared reporting bars (seven-point row)
- Headline metric: the instrument's primary retrieval metric over all 500 (fail-closed
  accounting; 0-delivered rows disclosed, never dropped).
- Noise floor: A/A′ on a fixed 100-question subset (seed 20260802, uniform) — full-500 A′ would
  double a ~2-day run for variance information a 100q pair supplies; subset policy DISCLOSED on
  the row. Discordance + aggregate spread reported like F48.
- GOALS target line: V1-M ≥90% is the FLAGSHIP target for the QA (reader) configuration — this
  registered run banks the RETRIEVAL row first; the reader row is a separate registration (needs
  OpenRouter credits restored). No conflation of the two on the scoreboard.
- Negative result ships at the same resolution as positive.

## 5. Abort/park criteria
- >2 shards dead of the same cause → park the run, root-cause first (no blind restarts).
- Voyage 429 sustained >30 min across shards → halve shard count, document, continue (same
  registration: shard topology is operational, not measured-surface).
- Disk <15 GB free on LEXAR → park (each question's store ~30–37 MB; 500q ≈ 15–18 GB + prepared).

## AMENDMENT 2 (2026-08-03 — CORRECTION + restart after host reboot killed all 6 shards)
CORRECTION: §2's recovery claim ("a dead shard resumes by re-running its qid file — completed
questions detected + skipped") was WRONG — the instrument accumulated results in memory and
wrote the report only at run end. It was an UNVERIFIED capability written into a registered
document; ~17h × 6 shards of progress (~190 questions of Voyage ingest, est. $5–15) was lost
to a host reboot with only env captures on disk. Process failure banked to memory.
REMEDY BEFORE RESTART: checkpoint/resume patch (per-question progressive jsonl + --resume,
fail-closed torn-line handling, uninterrupted-run report-equivalence test) — implemented, PR'd,
and merged BEFORE relaunch; the restarted run pins the new instrument sha. The restart is a
fresh registration of the same design (this sheet + amendments); no partial results carry over.
