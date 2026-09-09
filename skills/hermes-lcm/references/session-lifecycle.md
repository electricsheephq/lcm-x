# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Do not promise that `/new` deletes historical LCM data. Earlier rows remain in `lcm.db` unless an explicitly authorized cleanup removes them, and they remain available through bounded cross-session recall.

Before binding, finalization, or rollover replaces existing lifecycle pointers,
LCM makes already-proven legacy ownership explicit on rows whose conversation ID
is blank. This requires the existing unambiguous binding predicate: another
conversation binding or any explicitly foreign row under that producer prevents
promotion. The proof read, blank-owner update, and lifecycle transition share one
SQLite write transaction. Source IDs, producer session IDs, content, and nonblank
conversation IDs remain unchanged.

This preserves attribution before it would be forgotten; it does not infer an
owner for already-orphaned or ambiguous legacy rows. Operators must not use a
neighboring row, a shared producer ID alone, or a carried summary as repair proof.

## `/lcm rotate`

`/lcm rotate` is different from `/new`:

- it keeps the current `session_id` and `conversation_id`;
- preview is read-only;
- apply creates/updates the rolling rotate backup first;
- it preserves the configured fresh tail;
- it advances the lifecycle frontier past older raw messages so bootstrap does not replay them into active context;
- it does not delete raw source rows or call a summarization model.

Run normal compaction before rotate when older material must be represented in summary nodes. Even without a summary, pre-tail raw rows remain recoverable through `lcm_load_session` and `lcm_expand`.

Rotate refuses ignored or stateless sessions. Repeating an already-satisfied rotate reports a no-op and preserves the previous known-good rolling backup.

Use a separate session when the user wants a new active conversational boundary. Use rotate when the problem is active transcript/frontier size without changing identity.
