# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Do not promise that `/new` deletes historical LCM data. Earlier rows remain in `lcm.db` unless an explicitly authorized cleanup removes them, and they remain available through bounded cross-session recall.

## Compaction commits

Hermes commits a compaction by calling the session-end hook with the list it compressed, then the compression-start hook. LCM-X treats that call as a commit, not a session end. It does not re-store those rows. The session keeps its identity, published frontier and ingest position across the boundary, both when Hermes keeps the session id (in place) and when it rotates to a new one. A genuine session end (exit, `/new`, or an end after a cancelled compaction) still finalizes the session. A session that resumes later continues from the frontier it finalized itself, never another session's.

Native recovery carries version 3 adoption proof across in-place and rotating compactions using the same replay identity as other commit proofs. User edge whitespace is tolerated only at proof-bound positions; only proven orphan tool results may be omitted, while lossy redacted identities fall back to conservative reconciliation.

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
