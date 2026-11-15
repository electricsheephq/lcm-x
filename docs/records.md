# Optional records container

The canonical message database keeps its existing schema and reader compatibility.
Opening, ingesting, compacting and closing it creates no records file. PR-S registers
no product table. An in-memory store has no records file.

A records writer binds to `MessageStore.db_identity`, appends `.records` to its full
resolved filename, and retains that identity across retries and profile changes.
For example, `lcm.db` and `lcm.sqlite3` use different containers. Format 1 uses SQLite
rollback journaling (`DELETE`), with oldest reader v0.30.2 and no planned removal.
Newer formats are refused before any DDL or format stamp changes.
Every records write transaction rechecks the format under its writer reservation before completing intents or changing data.

Writers register their table's DDL in `store.py` through
`register_records_table(name, ddl, owner_columns)`. The ordered mapping identifies
the record's owner-id column first (mapped to `messages.store_id`), followed by
occurrence evidence columns mapped to canonical columns. Include immutable
occurrence evidence beyond session, row id or equal text, such as both `timestamp`
and `ingested_at`. Session and conversation are read from the owner, never trusted
from a record. Each writer creates its own registered table on a checked insert.

`open_records_writer(store)` is the only file creator; its busy timeout is zero.
`insert(table, values)` returns `None` on success or a refusal reason for an
uncommitted/mismatched owner or pending purge intent. SQLite errors propagate to
the records caller; they do not enter canonical ingest failure handling.

`records_read_cut(store)` pins a read-only records snapshot before a canonical
snapshot. Use its `canonical` connection for canonical output and
`read_records(cut, table)` for attached rows and an unattached count. Unattached
rows remain stored and hidden; there is no absence-based sweep.

Purges first persist occurrence-specific intents with a separate 50 ms busy wait.
Failure refuses the canonical purge. Completion callbacks run after canonical
commit or rollback; errors are logged and counted without changing that outcome.
A committed purge deletes its records; rollback deletes only its own intent.
Maintenance never drops an intent whose matching owner remains present. A crash
can leave such an intent blocking records writes until a retried purge completes.
`RecordsWriter.maintain()` is the explicit maintenance point for intents whose
owners are gone. Only a successful records write completes pending intents; a
refused insert rolls back its attempted completion. `maintain()` has no product
caller until #2707's PR-2 wires it.

Store backups and timestamped/rolling operator backups copy records with SQLite's
backup API, naming them `<backup filename>.records`. Both copies finish before
publication; a failed copy preserves the previous pair. Restore the pair together.
A non-blocking folder lock serializes backups and refuses contention with `another LCM-X backup in this folder is running`.
If restoring the old records fails, the restart marker and previous records are kept for recovery on the next run.
Restoring canonical data alone yields no records, or hidden/countable unattached
records if a newer records file is left beside it. Recorder snapshots and
checkpoints exclude records and replay canonical messages only.

Doctor reports absence, refused format, or format, size, attached/unattached counts
and pending purge intents. Its message count remains a count of `messages`.

The v0.30.2 `queued_user_rows` writer preserves earlier queued user texts that
whole-replace Hermes hosts overwrite before ingest. Pre-LLM capture freezes text,
host timestamp and UID. A committed owner supplies its store id, timestamp,
ingested time and content digest. No canonical row or canonical writer changes.
Records are recall sources with `[owner_store_id, ordinal]` locators; their session
and conversation always come from the attached owner on the same read cut.
Format remains 1, oldest records reader v0.30.2, with no removal planned.

Listings place visible records immediately before their owner. On this host no
absorbed-UID evidence survives reload, so records are never superseded. A later
user row in the owner's session containing a record's stripped text as contiguous
raw double-newline segments withholds it as `overlap`. Empty stripped records
stay listed. Each owner receives a floor of 64 later user rows and 1 MiB; the
shared scan beyond these floors stops at 200,000 rows or 64 MiB. Undecided records
are withheld as `unscanned`. Every listing page includes counts and paginated
locators without their text; every attached record remains exactly expandable.
Search, recall and the read-only MCP reader expose typed record provenance.
Replay, ledger, compaction, reconciliation and classifier paths use messages only.

Capture conservatively refuses ambiguous ownership, surviving run objects,
prefix-preserving hosts, earlier ingests, reused heads, backward overlaps,
exhausted backward windows, inline payloads and externalized whole joins. Ignore
patterns also apply. Refusals retain main's missing-text behavior and are counted.
Drains run outside canonical write locks, bounded by 256 entries, 1 MiB and 50 ms,
with separate shutdown/exit budgets. Busy writes retain frozen entries; entries
still queued at exit are the crash-window residual and are counted. Doctor reports
attachment, supersession, withholding, capture refusals and drain/exit counters;
its message count remains canonical messages only. An `unscanned` count on a real
host is the evidence gate for a future indexed scan tier.
