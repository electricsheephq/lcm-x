# Schema changes

The store's DDL lives in nine modules today (`store.py`, `db_bootstrap.py`, `dag.py`, `vector_store.py`, `query_view_store.py`,
`scope_storage.py`, `trajectory_store.py`, `teams/catalog.py`, `command.py`), with more than 25 `ALTER TABLE` statements between them.
One migration (the `conversation_id` columns) exists twice. These rules stop the spread while the schema is consolidated.

## Rules for a PR that changes the schema
1. **No new DDL owners.** New `CREATE`/`ALTER`/`DROP` statements go into one of the store modules above, never into a new module or
   a command, tool or engine path. `command.py`'s existing DDL is grandfathered until consolidation; it takes no new DDL.
2. **Record the migration.** The PR body states, for each migration:
   - the new on-disk format;
   - the oldest reader that can still open it;
   - the release in which the old format's code path can be removed.
3. **Test both directions.** Each migration gets a forward test (an older database opens and upgrades) and a rollback test, in
   the compat suite (`tests/compat`). The rollback test shows that the previous release still opens the upgraded database, as
   invariant I5 (`docs/invariants.md`) requires. An intentionally incompatible format is an I5 exception and needs all of: a
   negative test (the previous release refuses the database cleanly: no partial open, no writes, the file unchanged), a recovery
   path recorded in the PR and the release notes, and the format added to I5's list of unsupported formats. An explanation alone
   never replaces the test.
4. **One migration, one place.** Do not copy an existing migration; call the function that already performs it.

When transaction ownership is consolidated, a compat registry file (format → oldest reader → removal version) replaces the PR-body
record in rule 2, and DDL moves to the schema modules only.
