# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Do not promise that `/new` deletes historical LCM data. Earlier rows remain in `lcm.db` unless an explicitly authorized cleanup removes them, and they remain available through bounded cross-session recall.

## CLI reset-time bind host contract

The minimum Hermes capability for CLI `/new` rebinding is a reset-time call to
`bind_session_state(session_db, session_id)` with the new, nonempty session id,
after `on_session_reset`. This is a capability requirement, not a version-only
check: the pinned 0.21.3 floor and 0.21.5 hosts implement it in
`AIAgent.reset_session_state` (`run_agent.py`).

For nonempty history, `CLISessionMixin.new_session`
(`hermes_cli/cli_session_mixin.py`) calls `on_session_end(old, history)`, changes
the host session id and performs a bare agent reset. The bare call to
`AIAgent._transition_context_engine_session` does not call `on_session_start`;
`AIAgent.reset_session_state` then offers the new id to the engine through
`bind_session_state`. Gateway starts use `on_session_start` directly.

`LCMEngine.bind_session_state` (`engine.py`) uses only the explicit id; it does
not adopt or write the host session database. After the foreground session is
finalized, it starts the new binding through `_on_session_start_unlocked`.
Empty-history `/new` skips the end hook: `_on_session_reset_unlocked` records
the exact old binding, and `bind_session_state` permits finalization only with
that reset proof and no stored or attempted ingest, summary, frontier, or prior
lifecycle activity. Live foregrounds, auxiliary sessions and duplicate binds
cannot use this path to replace the foreground or reset its ingest position.
Initially ignored or stateless sessions can pass the same empty-reset proof;
the target session filters are applied at the new bind.

An empty, missing or `None` id is a detach request and raises
`NotImplementedError` without changing the binding. Hosts with
`agent/background_review.py::_detach_fork_compression` catch this refusal and
keep review-fork compression disabled. Their warning
`background-review compressor detachment failed; keeping compression DISABLED`
and its traceback are expected fallback output, not an LCM-X crash. Returning
normally would allow the host to re-enable compression on a fork still bound
to its parent session.

## Compaction commits

Hermes commits a compaction by calling the session-end hook with the list it compressed, then the compression-start hook. LCM-X treats that call as a commit, not a session end. It does not re-store those rows. The session keeps its identity, published frontier and ingest position across the boundary, both when Hermes keeps the session id (in place) and when it rotates to a new one. A genuine session end (exit, `/new`, or an end after a cancelled compaction) still finalizes the session. It also pauses this engine's background embedding maintenance: queued passes are dropped and nothing new is scheduled until the next live bind re-arms it (a compaction commit leaves maintenance running; see `architecture.md` for the shutdown and process-exit waits). A session that resumes later continues from the frontier it finalized itself, never another session's.

`LCM_NATIVE_RECOVERY`: removed in v0.25.0; ignored if set. Native adoption proofs and replay-snapshot digests that older versions wrote are still read, so a host replay of an adopted native summary reconciles after a restart or rotation.

## `/lcm rotate`

`/lcm rotate` is a read-only preview of the tail-preserving frontier for the
active session. It keeps `session_id` and `conversation_id` and reports the
configured fresh tail, pre-tail message count, current and planned lifecycle
frontier, and rolling backup path. The preview refuses ignored or stateless sessions.

`/lcm rotate apply` is refused on Hermes (#1141): Hermes reloads its own
transcript after a restart, so advancing the frontier without summaries leaves
rows LCM-X cannot map. No backup is created and no lifecycle state is changed.
Use `/compress here N` to summarise everything except the most recent N messages.
If an earlier LCM-X already rotated this session and Hermes restarted since, compaction can stop on rows below that frontier and `/compress here N` will not repair it: start a new session with `/new` (raw rows stay recoverable via `lcm_load_session`; in-place fix tracked in #1184).
Raw source rows remain recoverable through `lcm_load_session` and `lcm_expand`.

Use a separate session when the user wants a new active conversational boundary.
