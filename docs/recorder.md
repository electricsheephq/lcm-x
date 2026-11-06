# Event recorder (opt-in, #2243)

The recorder writes what Hermes asks of the LCM engine to local files so the run can be replayed offline against
another build. It is off by default and needs Python 3.12 or newer (it uses `sys.monitoring`). Recording never changes
what LCM stores, returns or logs to the host; with the flag off, `recorder.py` is never imported and nothing is set or
created.

## Turning it on and off

| Key | Default | Meaning |
| --- | --- | --- |
| `LCM_RECORDER_ENABLED` | `false` | Install the recorder at plugin registration. |
| `LCM_RECORDER_DIR` | `$HERMES_HOME/lcm-recorder/` | Where recordings go. |
| `LCM_RECORDER_MAX_BYTES` | 20 GiB | Cap on event segments and shared blobs across every recording under `LCM_RECORDER_DIR` (snapshots and checkpoints are not counted). |

All three are read with the profile-scoped environment at registration; restart the host to change them. On Python
3.11 the flag logs one warning and installs nothing. If `sys.monitoring` tool ids 3 and 4 are both taken, it logs one
warning and installs nothing. One recorder serves the whole process; a second registration (PluginManager) adds its
engine to the same recording when it uses the same database; foreign engines are skipped.

## What is recorded

- **Roots:** every call into the engine from outside it. These are the public methods of Hermes's `ContextEngine`
  class plus a fixed list of names Hermes calls beyond it (`bind_session_state`, `clone_for_agent`, `shutdown`,
  `record_rejected_compaction`, `_automatic_compression_blocked`, `_compression_block_reason`,
  `record_completed_compaction`, `maybe_seed_preflight_display_tokens`, `carry_over_new_session_context`, where the
  engine class has them), every hook, tool, command and system-prompt section the plugin registers, the summary call
  seam (`_invoke_summary_attempt`), and the two atexit flushes. A root called from inside another root is recorded
  as nested.
- **Arguments and results:** each root's parameters (not free variables) and its return value or exception (type and
  a hash of the text, never the text). Message lists are recorded as views: each row gets a token on first sight, and
  later views say which rows are the same object, the same object edited in place, or new.
- **Summary attempts:** prompt hash and length, model, limits, the returned text, the route and contract, timing
  offsets, and an error stand-in carrying the fields LCM's classifiers read.
- **State the engine reads from the host:** engine attributes Hermes assigns between calls, `LCM_*` environment
  changes, config file paths with hash and mtime, the `sessions` rows LCM reads from Hermes's `state.db`, the
  `/lcm` task-local context, auxiliary-agent frames LCM inspects, and per-engine seeds (the compaction telemetry
  epoch), and persisted-output read outcomes (path, generation stat, hash and content blob).
- **At start:** a snapshot of `lcm.db` pinned at one transaction, a copy of the large-output payload directory with a
  manifest, the plugin's git SHA when readable, Python and host versions.

## Where it goes

```
<dir>/proc/<pid>-<start>-<db8>/events/YYYYMMDD-NN.jsonl   one JSON event per line, seq-numbered
<dir>/proc/<pid>-<start>-<db8>/snapshot/                  lcm.db at the pin, payload copy, manifest
<dir>/proc/<pid>-<start>-<db8>/checkpoint/<hours>/        sealed lcm.db backups at +24, +72 and +144 h
<dir>/proc/<pid>-<start>-<db8>/STATUS.json                end state, counters, coverage, wall-time range
<dir>/blobs/<aa>/<sha256>                                 message and text bodies, shared across processes
```

Each Hermes process gets its own directory. Two directories on the same database whose wall-time ranges overlap are
flagged `overlap` by the reader. Payload files are pinned before monitoring starts; a separate writer
thread writes queued events while the host's turn thread copies values.

`PY_UNWIND` cannot be enabled locally: its callback runs for every unwound Python frame in the process, including
code outside LCM. In the exception-heavy lab test (`test_2263_s4_exception_unwind_budget`), 10,000 exceptions unwinding
nine frames took 4.99 ms with recording off and 9.59 ms with it on: about 0.051 microseconds extra per frame. The test
prints the measurement and applies a generous scaled bound; this number describes that workload.

## Privacy

Recordings capture host inputs before LCM ingest filters: sensitive-pattern redaction and ignore and stateless
session patterns do not reduce what the recorder stores. Recordings are local files only and hold message content.
Directories are created 0700 and files 0600. A profile backup that copies `$HERMES_HOME` copies the recording too. Secrets are not written: the values passed to
`update_model(api_key=..., base_url=...)`, `session_db`, and any `LCM_*` key or config field whose name has a whole
`_`-separated segment matching KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or AUTH are recorded only as set or unset.

## Limits, coverage and the replayable prefix

- **Cap and disk floor:** when the recording reaches `LCM_RECORDER_MAX_BYTES`, or free disk falls below 20 GiB, a
  `cap` event is written and recording stops. The cap spans every recording under `LCM_RECORDER_DIR`, including
  shared `blobs/`. Before snapshots, payload copies and checkpoints, free space minus the planned write must remain
  at least 20 GiB. A refused snapshot records `disk_floor` in STATUS and writes no events. LCM keeps running unchanged.
- **Queue admission:** the writer queue admits at most 10,000 events or 512 MiB; an event refused there writes a
  `gap` event and recording stops.
- **Coverage ends:** recording stops being complete at the first of: a gap, the cap, a value that cannot be copied
  (`unsupported`), `hermes_home_changed`, a lost observation, a root stack mismatch, or a recorder failure (`recorder.failed`; the failure
  never reaches the engine or Hermes). `STATUS.json` names the first cause and its seq.
- **Replayable prefix:** the reader returns events up to the first missing seq or the first coverage end, whichever
  comes first. A missing, torn or hash-mismatched blob cuts the prefix before its event with a `blob_*` cause.
  Everything before it is complete; a checkpoint crossing a hole or damaged blob is rejected.
- **Checkpoints:** at +24, +72 and +144 hours, between roots, the recorder backs up `lcm.db` and writes a
  `checkpoint_sealed` event with the seq it covers. Reading with a checkpoint returns the prefix through that seq, and
  a replay of it must match the backup, without needing an exit.
- **Ended without exit:** Hermes ends its gateway and CLI with `os._exit`, so atexit never runs and a recording
  normally ends as "ended without exit". That is a normal end; the sealed checkpoints are the comparison points. When
  atexit does run, the recorder writes `exit_begin` and records the exit flushes synchronously after a successful
  drain. A drain timeout stops admission with `exit_timeout` in status. Recording stops in a forked child (`fork`)
  before it can wait on inherited locks; the parent recording continues.

## Declared gaps (not failures)

- Clock-driven behaviour, cancellation (`_compression_cancelled_check`), host stream deadlines and lock interleaving
  are not proven by replay. Elapsed-time fields (`duration_ms`, `last_compaction_duration_ms`, the no-progress hold's
  wall-clock `until`) are excluded from comparison.
- LLM callers that call the host's `call_llm` directly (`extraction.py`, `assertion_extraction.py`,
  `host_evidence.py`, `selective_compiler.py`, `expand_query_tools.py`): their tables are outside the comparison and
  replay stubs them. The `lcm_expand_query` tool return depends on that synthesis and is excluded from comparison.
- A caller that reaches the engine other than through a root is not recorded.
- Changes to payload files made outside LCM after the pin are not recorded.
- A `state.db` change committed while a root runs is seen at the next root's entry.
- Tails pending when a Hermes process exits are not drained or recorded (#2253).
- Sessions marked `aux_context` (auxiliary agents) are excluded from the day-1 check.
- The compaction telemetry epoch is seeded per engine and a re-rolled epoch is recorded at root exit.
  Per-call ids (lease and request ids) are not recorded.

## Reading and replaying

`recorder_read.read(proc_dir, checkpoint=None)` returns the prefix, its end and cause, the events with nesting, and the
snapshot paths. `tests/recorder_replay.py` is a mini-replayer: it loads the plugin against a fake Hermes context,
restores the snapshot, drives the recorded roots in order on one thread per recorded thread, serves recorded summary
attempts and persisted-output inputs, restores the recorded effective configuration in the replay home, and compares every root's return and the in-scope tables.

Replaying a recording that captured a `config.yaml` needs PyYAML; the harness imports it only then.

A root that exceeds the replayer's per-root timeout is abandoned: the replay fails with "isolation kept", and the
model-call block stays installed while that worker thread is alive. The environment, `sys.path`, the clock and the
secrets are restored as usual, so an abandoned root that later reads an `LCM_*` setting sees the caller's value. Treat
the process as spent after such a failure.
