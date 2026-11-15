# Optional records container

The canonical message database keeps its existing schema and reader compatibility.
Opening, ingesting, compacting and closing it creates no records file. PR-S registers
no product table. An in-memory store has no records file.

A records writer binds to `MessageStore.db_identity`, appends `.records` to its full
resolved filename, and retains that identity across retries and profile changes.
For example, `lcm.db` and `lcm.sqlite3` use different containers. Format 1 uses SQLite
rollback journaling (`DELETE`), with oldest reader v0.30.2 and no planned removal.
Newer formats are refused before any DDL or format stamp changes.

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
owners are gone; every records insert also attempts that completion.

Store backups and timestamped/rolling operator backups copy records with SQLite's
backup API, naming them `<backup filename>.records`. Both copies finish before
publication; a failed copy preserves the previous pair. Restore the pair together.
Restoring canonical data alone yields no records, or hidden/countable unattached
records if a newer records file is left beside it. Recorder snapshots and
checkpoints exclude records and replay canonical messages only.

Doctor reports absence, refused format, or format, size, attached/unattached counts
and pending purge intents. Its message count remains a count of `messages`.
