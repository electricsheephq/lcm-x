# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Do not promise that `/new` deletes historical LCM data. Earlier rows remain in `lcm.db` unless an explicitly authorized cleanup removes them, and they remain available through bounded cross-session recall.

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

## Retained logical-conversation compaction

Explicit message `conversation_id` ownership governs replay reconciliation, leaf publication, summary selection, and condensation across producer sessions. A shared producer session does not make another conversation's rows eligible. Blank legacy owners require a positive, unique durable lifecycle binding; unrelated ambiguous rows remain unchanged and unclaimed. Selected active occurrences without owned lineage fail safely. An existing summary can prove a leading covered prefix only when its complete owned lineage is valid and the summary remains selected in the assembled context. These rules do not relabel raw messages, deduplicate by text, or authorize a frontier reset.

Hermes may keep the same session ID during in-place compaction. LCM handles that as a compression continuation, preserving the logical conversation and committed coverage without moving raw messages. The host may first flush the original, longer history; reconciliation must distinguish that already-ingested history from the shorter active projection. Ordinary session endings retain their proven ingestion cursor. A cold replay of the compacted projection must not append its existing messages again.

Unregistered replay shortcuts require both conversation ownership and evidence from the producer that emitted the messages. Identical text or a reused tool-call ID from another producer does not prove that a new occurrence is a replay. Explicitly registered compacted snapshots retain their existing carry-over proof.

Hermes keeps the system prompt outside the persisted conversation. LCM therefore
registers its own assembled summary projection even when that projection has no
system row. Standalone generated summary messages carry Hermes's existing
`_compressed_summary` flag so host sequence repair preserves their boundary with
the next real user turn. Real retained user messages and folded source messages
do not receive this flag. The flag does not establish source ownership or replay
proof; those still require LCM's committed lineage and registered snapshot.
