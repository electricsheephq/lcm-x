# Occurrence fuzz instrument

This instrument keeps the seeded host-view generator and ledger comparison used
for [#1525](https://github.com/electricsheephq/lcm-x/issues/1525).
It measures ingest behavior without proposing an engine fix.

- `gen.py` produces deterministic JSONL sequences for a half-open seed range.
  Rows contain role/content plus optional timestamps or message UIDs. Views
  exercise shrink, append, front-drop, reorder, regrow and resend operations.
  Reset and restart operations have both `ops` labels and uppercase view markers.
  `ep` generates longer, episode-heavy sequences.
- `driver.py` runs those views on one checkout with isolated databases, producing
  counts, final role/content ledgers and error markers. Reset calls the session
  reset hook. Restart shuts down and drains maintenance, then opens the same
  database and conversation. Each per-sequence scratch directory must be new;
  it is removed after successful shutdown. Run each checkout in a fresh process.
  The caller supplies the supported Hermes host import path through `PYTHONPATH`.
  Importing the driver starts no engines. Input `-` streams stdin to stdout
  (the output argument is then unused).
- `gate.py` scores any number of candidate ledgers against a reference ledger
  (the previous GA) and a base ledger (main). It prints per-class counts and
  `GATE PASS` or `GATE FAIL`; as in the original scorer, the verdict is text,
  not a process exit status. Candidates are numbered in argument order.

From the repository root, with `python` bound to a supported interpreter and the
host import path configured, run:

```sh
python bench/instruments/occurrence_fuzz/gen.py 0 3000 sequences.jsonl base
python bench/instruments/occurrence_fuzz/gen.py 0 2500 episodes.jsonl ep
nice -n 19 python bench/instruments/occurrence_fuzz/driver.py REFERENCE_TREE sequences.jsonl reference.jsonl reference-work
nice -n 19 python bench/instruments/occurrence_fuzz/driver.py BASE_TREE sequences.jsonl base.jsonl base-work
nice -n 19 python bench/instruments/occurrence_fuzz/driver.py CANDIDATE_TREE sequences.jsonl candidate.jsonl candidate-work
python bench/instruments/occurrence_fuzz/gate.py sequences.jsonl reference.jsonl base.jsonl candidate.jsonl
nice -n 19 python -m pytest tests/test_occurrence_fuzz_instrument.py -q -o addopts= -p no:cacheprovider
```

Repeat the driver and gate commands with `episodes.jsonl` for the episode set.
All inputs to one comparison must have the same seeds. Keep generated inputs
and ledger outputs together so the scores can be reproduced without rerunning
engines.

Scoring is per sequence and distinct `(role, content)` key, encoded as
`role|content`, in the final ledger:

- **DUP:** count exceeds `max(1, reference count)`. This counts duplicate keys,
  rather than excess rows.
- **LOSS:** a reference key is absent from the compared ledger.
- **Resend exception:** a DUP key is excused only if `ops` contains
  `resend:<rowid>` and that row's key equals the DUP key. Other resends do not
  excuse it; LOSS is never excused.
- **Class:** base-row `masks` select `none` (all `n`), `stamp` (all `s`), `uid`
  (all `u`), or `mixed`. Fresh-row metadata does not change the class.

The final candidate passes a class when LOSS is zero and non-resend DUP keys
do not exceed the base's count. Earlier candidates are diagnostic comparisons,
as in the original gate. The overall gate also requires zero missing or errored
records in every ledger, the reference included, and at least one scored
sequence. The driver records an exception from ingest, reset or restart as that
sequence's error. Literal DUP counts remain visible even when resend-excused.
