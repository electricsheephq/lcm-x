# Background leaf preparation instrument v1

Two runner modes in `run.py`: `background-prep-off-replay` (G1), and
`background-prep-thinktime-on-off` (G2). Both use isolated OFF and ON processes,
fresh stores, and identical per-seed input. Only the summary provider is stubbed.
The real engine, escalation, worker, chunking, fences and publication run.
Hermes source is imported read-only; plugin registration is skipped.
Both product and host checkouts must be clean, including untracked files. The
protocol binds the host Git SHA and context-engine hash. Token estimation is
pinned before product consumers import their token helpers. Cells execute the
captured instrument copy, including its replay and scoring modules.
Each cell checks the host HEAD against the parent's protocol SHA before replay.
Output must resolve outside the product, host and instrument source trees.
Use Python 3.11 or newer (the product's CI minimum).

```sh
python3 -B bench/instruments/compaction_probe/bgprep/run.py background-prep-off-replay \
  --product-sha PRODUCT_SHA --hermes-source HERMES_SOURCE \
  --out FRESH_G1_DIR --gap-seconds 0 --seeds 1 --smoke
python3 -B bench/instruments/compaction_probe/bgprep/run.py background-prep-thinktime-on-off \
  --product-sha PRODUCT_SHA --hermes-source HERMES_SOURCE \
  --out FRESH_G2_DIR --gap-seconds 0 --seeds 1 --smoke
```

The card tool defaults to the operator's `bench-card.py`; override with `--card-tool`.
`--real-route --summary-model MODEL` opts into the product's real summary route
using inherited authentication. No live configuration or auth is copied into a run.
This route is a separate protocol and requires its own same-route reference.

## Frozen workload and measurement

64 user/reply pairs per seed (1, 2, 3); each message has a unique row tag and
400 seeded vocabulary words. The sustained stream exercises initial publication,
then repeated threshold crossings against already-published coverage. Window
60,000; threshold 0.75; chunk 2,000; fresh tail 2, matching the slice-1 probe's
scale. Full effective configs, inputs and code hashes are saved with each cell.
Foreground spend limit is disabled to prevent zero-gap replay speed from tripping
it; prep retains the slice-1 test's 64 calls/600s. Rollups, embeddings and
externalization are off. These are instrument settings, not fleet defaults.

Each user arrival ingests and compresses at the threshold before its assistant
reply. Each reply ingests, notifies turn completion, then invokes the real
post-turn prep seam. Real gaps run from that turn end to the next user arrival;
prep runs inside this gap. Zero-gap smokes explicitly drain the worker after each
turn for deterministic observability; this is recorded and never a GA measurement.
The final drain is observed separately. All stored rows and live view identities,
leaf source IDs and frontier are snapshotted before scoring at every probe.

G1 requires zero OFF constructions/starts, enqueue attempts/jobs, worker model
calls, promotion lookups/promotions at every probe, intact ordered row/coverage
identities, foreground compaction, and positive ON activity on all observers.
The foreground model calls in OFF are expected and reported separately.

G2 requires intact rows in every ON seed (OFF is checked too), and pooled ON
promoted leaves / all leaf operations at threshold crossings >= frozen reference.
The denominator observes the foreground rescue entry or prepared-leaf summary
entry once per operation, including failed foreground leaves; provider retries
are not separate leaf operations. Published-node counts are recorded separately.
The denominator excludes condensations and non-crossing passes; per-seed rates
are reported only. No-crossing runs cannot pass observability/rate checks.
Active unsummarized rows above the frontier count as covered only when their
exact raw identity is present in the live view, including verified carrier remainders.
Live coverage matches role/content because carrier projections may omit timestamps;
ordered stored-versus-expected identity independently requires the persisted timestamp.
Matching live tail identities must occur exactly once per stored tail row, in store
order; unmatched summary carriers are ignored. Leaf source IDs must increase strictly
within each leaf and across leaves in node order. Scores report `tail_order_preserved`,
`duplicate_tail`, `covered_in_active` (non-tail identities beyond their tail allowance) and `leaf_order_preserved`; violations fail integrity.
Missing storage, duplicate storage/IDs, duplicate leaf coverage, dangling sources,
changed identity (role, content, or persisted host `observed_at` timestamp) or
chronology also fail. Leaf ancestry condensations are not
counted as a second covering of their sources.

Recorded only: nearest-rank p90 visible wait and share of turns waiting >=1s,
with counts, from first crossing through the final turn. Boundary: user arrival
ingest + compression before model dispatch. This excludes provider reply time,
post-turn prep and think time. It is an engine harness measurement, not a live
Hermes end-user latency claim. Facts kept is explicitly unavailable (no Track S
material/reader); D7 remains separate and its thresholds are not applied.

## Reference and re-score

After both smokes pass, invoke G2 at the approved main SHA with
`--freeze-reference --gap-seconds 60 --seeds 1 2 3 --jobs 6`. It refuses another product
SHA and writes a read-only `REFERENCE.json` only after a passing complete run
and successful result-card validation. Recorded card commands use the repo-relative
runner path and placeholders for local path arguments.
`--jobs N` runs N cells at once, each in its own process and store; it is part of the
protocol, so a candidate must use the same value. If the projected wall time from gaps
alone (63 gaps per cell, `ceil(cells / jobs)` waves) exceeds three hours, it writes
`projection.json` and exits 3 without running cells or inventing a reference. For this
64-turn protocol, serial ON-only gaps total 11,340 seconds and the serial pair 22,680
seconds; `--jobs 6` needs one wave, about 3,780 seconds of gaps.
A later candidate uses `--reference REFERENCE.json`; mismatched host contract,
instrument, workload, route or settings invalidate the comparison. Changing this
protocol requires a new baseline. Reference files are snapshotted into candidate
runs. Reference creation is never implied by smoke success.
Loading a reference also checks its pooled ON, protocol, seeds and per-seed rates
against the sibling card-covered `summary.json` and `run.json`; edits that disagree
invalidate the comparison (exit 2).

From a completed run root, `python3 instrument/run.py --rescore .` reads only
raw snapshots and prints the verdict, every per-seed score and paired deltas.
It does not modify the settled card. Both completed runners create and check
`RESULT-MANIFEST.json` and `RESULT.md` using the card tool. Raw probes, provider
inputs/outputs, effective configs, final SQLite backups, workloads and the
instrument source are retained. Nothing is published, packed or deleted by this
instrument. Publishing evidence and deciding release readiness belong to the
release manager.
