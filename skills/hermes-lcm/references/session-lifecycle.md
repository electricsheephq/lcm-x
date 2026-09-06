# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Compaction source mapping and publication are scoped to the logical
conversation, not only the currently bound host session. A source row keeps
its producing `session_id` and global `store_id`; explicit `conversation_id`
rows may therefore be validated after rollover through the current and
last-finalized lifecycle bindings. Legacy rows with a blank conversation id
are admitted only when their producing session is one of those proven
bindings. Ambiguous or foreign rows fail closed before a summary node or
frontier advance is committed.

When active-context assembly folds a retained assistant run into one provider
message, LCM records the complete ordered `store_id` range behind that fold.
The range is resolved by primary key and checked against the same conversation
ownership predicate before publication, so chronology and producing session
ownership survive rollover without rewriting raw rows. A missing, duplicate, or
ambiguous lineage row rejects the publication rather than advancing the
frontier.

Rollover finalization, retained-node reassignment, and lifecycle rebinding are
staged on one existing `lcm.db` SQLite transaction. A fault rolls the whole
state back; retrying a committed rollover is idempotent and does not duplicate
or reclaim source rows.

At session admission, LCM runs a read-only ownership audit for the requested
logical conversation. It rejects a session that would make blank legacy rows
ambiguous or orphaned; unrelated conversation rows do not widen that audit.
Zero-frontier, debt-free lifecycle rows with no messages or DAG nodes for their
own current binding are recorded as non-owning aliases; a source-bearing row
that shares a producing session remains ambiguous.
Publication then checks only the indexed current/last-finalized owner-session
scope, so unrelated DAG nodes do not enter the hot transaction.

If a compression boundary callback is stale, LCM keeps the committed frontier
and active binding unless Hermes' read-only `state.db` proves one unambiguous
compression successor for that exact child. A proven duplicate callback is a
no-op; an ambiguous or missing host successor fails closed without rebinding
or moving source rows. If LCM is already on an intermediate session, that
session must itself be proven on the host chain from the callback's old session
to the requested child; a newer local session is preserved.

When active cleanup drops an orphan or late tool row, adjacent assistant turns
may become one provider message. Their lineage is recorded after tool cleanup
from the exact durable assistant occurrences, excluding the dropped tool row;
an unmapped assistant occurrence prevents lineage publication.

Summary escalation carries one absolute sweep deadline through every model
level and rescue attempt. A late result is discarded and the pending source and
maintenance debt remain available for a later bounded retry.

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
