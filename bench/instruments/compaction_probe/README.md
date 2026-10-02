# Compaction probe material

The four-argument `generate(seed, out_dir, turns, tokens_per_turn)` API and CLI
defaults retain the legacy 30-canary schema. Track S requires explicit
`--placements --classes12` (or both corresponding API arguments set to `True`).
Selecting only one is an error.

## S8 and stub2k material

The S8 decision run and the stub2k gate used the same three full seeds and the
wiring smoke, generated with these flags. Run from the repository root, with
`python` set to the validation interpreter:

```sh
scratch=$(mktemp -d "$TMPDIR/compaction-material.XXXXXX")
for seed in 1 2 3; do
  python bench/instruments/compaction_probe/gen_material.py \
    --seed "$seed" --out-dir "$scratch/seed-$seed" \
    --placements --classes12 --min-tokens 244800 --min-events 2
done
python bench/instruments/compaction_probe/gen_material.py \
  --seed 1 --out-dir "$scratch/smoke-seed-1" \
  --placements --classes12 --min-tokens 244800 --min-events 2 --smoke
for name in seed-1 seed-2 seed-3 smoke-seed-1; do
  python bench/instruments/compaction_probe/verify_material.py \
    "$scratch/$name" --min-tokens 244800 --min-events 2
done
```

The defaults `--turns 35 --tokens-per-turn 17000` apply to these commands.
`--min-events` plans opportunities and ancestry receipt requirements; it does
not prove runtime events. The smoke's forced-event suffix is wiring only.

New manifests identify `material_version: track-s-v3`. Answer payloads use a
separate seeded value stream, independent of fixture identifiers and sibling
answers; fixed class wording (such as `MiB` and `because`) retains its shape.
Traps use canary wording and nonce-shaped identifiers from their own seeded
stream, cover five distinct classes, and never appear in the transcript.
Trap ids and fixture names share their class index and slot (5–9); scoring uses
explicit trap metadata. Stale and current rows both name their fixture.
The verifier requires this material version, exact call metadata and turn/batch
projections, and two head, one middle, and two tail facts per class.
Material at `6748ed97` has the prior v2 answers and traps; do not mix its answer
keys or run receipts with regenerated v3 material. Material at `5f66cf10` remains
reproducible using that revision and the explicit flags above; do not mix its
answer keys or run receipts with regenerated v3 material.

Track S `transcript.jsonl` is the full replay source. Its `turns.jsonl` projection
now retains `id`, `role`, `tool_call_id`, and `tool_calls` alongside `turn` and
`text`. The four checked-in `drive_*.py` scripts submit user prompts only and
reject non-user material rows before starting a session. A genuine-role Track S
replay must use an adapter that supports those roles and call/result pairing.

Run both test naming styles explicitly; the older `tests_*.py` files are not
collected by pytest's default `test_*.py` pattern:

```sh
python -m pytest bench/instruments/compaction_probe/ tests/tests_compaction_probe.py \
  -q -p no:xdist -p no:cacheprovider
python -m pytest bench/instruments/compaction_probe/tests_compaction_probe_b.py \
  -q -p no:xdist -p no:cacheprovider
```
