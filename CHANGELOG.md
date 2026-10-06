# Changelog

This file is the repository-root version history. Curated notes under `.github/release-notes/` exist for
every `v0.x` tag from v0.21.0-rc2 onward (plus a `v0.21.0-rc1` file with no matching tag today); older tags,
including the `v1.0.0-beta.*` prereleases, have none. GitHub Releases are published from tags (rc tags as prereleases).

## Unreleased

### v0.27.0

- Fix: after a survival fit with projected rows or an unknown projection count, the doctor advises a plugin-only rollback to v0.27.0 or later; older targets require stopping Hermes and moving the database aside to avoid cold-resume duplicates. (#919)

## v0.26.0 (host message identity)

- Feature: host message identity in shadow mode. On a host that stamps `message_uid`, LCM records which stored row each
  uid belongs to in a droppable `host_uid_bindings` side table and counts agreement with its existing identity matching;
  `/lcm doctor` reports the `host_uid_*` counts. Ingest, replay and emission decisions are unchanged, and hosts without
  the uid feature see no new key. New setting `LCM_HOST_MESSAGE_UID=off|shadow|on` (default `shadow`; `on` is reserved and behaves
  as `shadow`). No schema version change. (#643)
- Change: LCM's emit path keeps the host's uid on rows it reshapes (adjacent-assistant merges, ignored-message
  placeholders, over-cap tool stubs, the cached replay prefix), and rows LCM generates carry a deterministic engine uid
  when the host declares `message_uid` persistence-only. Provider-visible bytes are unchanged. (#643)
- Fix: deleting messages also deletes their host-uid bindings, and the lineage key is namespaced by Hermes home. (#836)
- Fix: a forced-overflow recovery note, echoed alone or merged onto the previous user turn by the host, binds back to
  the recovery emission when the proof covers it, instead of being stored as a user turn. (#534, partial)
- Fix: after a crash on rotation, a held composite user row whose parts the host also keeps is no longer stored twice;
  older re-sequenced rows are readmitted only as one whole composite LCM recorded. (#821)
- Fix: an older unstamped stored row can serve as a composite constituent only inside its donor's run of back-to-back
  user rows (within 256 store ids), so a new turn is not consumed by an older identical row. (#851)
- Fix: on a host that rotates sessions, a branch of a session that ended by compression no longer treats that
  session as its ancestor, so its copied rows are stored as its own instead of matched to the parent's. (#891)
- Fix: one privacy-refused document no longer stops a cloud embedding backfill; refused documents are withheld and never sent,
  and an apply run that withheld documents and had no failure reports `partial` with a `privacy_withheld` count. (#759)
- Fix: stored vectors with an older privacy revision no longer make `lcm_recall` raise; recall answers from full text,
  for every `include`, marked degraded with an `embedding_identity_stale:` reason, and `lcm_doctor` warns with the
  remedy (`/lcm embed warmup`, then `/lcm embed backfill --apply`). An invalid privacy policy still raises. (#387, #882)
- Fix: `lcm_doctor` also checks the chunk identity once the store holds chunk vectors that recall can use, and
  `stale_tasks` names each stale corpus; a summaries-only `lcm_recall` falls back to full text when the summary
  identity goes stale during the scan. (#888)
- Perf: the payload lookup index uses set buckets, globs once per digest per scope, and shares one listing between the
  stub-first trial and its final assembly. (#827)
- Perf: when NumPy is available, float32 chunk vectors load straight into the search matrix instead of through Python
  floats; NumPy stays optional. Both chunk loaders reject a malformed stored value instead of decoding it. (#834)
- Docs: `docs/identity-ledger.md` records, for each of the 460 identity rules, what happens once a host uid decides
  identity, which rules are dead, and the entry conditions for `on` mode. (#643)
- Bench: an upstream host with message uids joins reliability CI with the P8 controls required (#866); the flush audit
  (B9) on in-process and acp-process transports (#859); the #534 recovery restart cell; B2 composite pairing through the
  host's merge record (#823); per-question `lcm_recall` health and FTS relevance ranking in the recall harnesses (#817);
  the Phase C scorer's multiset-v2 split rule (#710); the rollback-reader tests run in CI and cover `v0.25.1` (#841).

## v0.25.1 - 2026-10-05 (recall: natural-language queries with embeddings off)

- Fix: with embeddings off (the default), `lcm_recall` on a natural-language question returned only rows containing
  every word, newest first, so it mostly surfaced large tool outputs or nothing. The full-text arm now ORs the
  question's content words and ranks by relevance. `lcm_grep` and explicit full-text queries are unchanged. (#864, #870)
- Fix: the OR query keeps quoted phrases that contain stop words, routes emoji-only questions, centres answer-ready
  windows on a phrase matched across separators, folds case duplicates, and treats NOT/NEAR as literal words. (#873, #874)

## v0.25.0 - 2026-10-04 (long sessions: drain speed + correctness)

- Fix: a host refusal after a compaction stored leaves holds automatic compaction until turn end, with a
  600 s backstop; refusals without a stored leaf keep the existing hold. (#597)
- Change: hidden-backlog checks report `hidden_rows` as a count, a bounded count (`N+`), or `unknown` in
  sweep status and compaction logs; a capped empty scan warns once per conversation. (#597)
- Bench: the drain rotation scorer excludes the session-transition call from phase 2, so D1 measures
  only post-rotation compactions; clean-exit cells are unchanged. (#597)
- Removed native recovery (`LCM_NATIVE_RECOVERY`), which was off by default and off on every managed profile; the key is ignored if set (#777, #509). Stores written while it was on keep their adoption proofs and snapshot digests, which LCM still reads.
- Fix: the objective anchor no longer picks LCM's own summary row. When the reverse scan for the latest user
  request reaches a verified LCM summary row before any real user row, it returns no objective anchor, so a long tool
  run across several compactions no longer re-renders a summary as the preserved objective. The host-carrier half
  stays open (#576). (#84)
- Fix: assembly and condensation see every summary depth in deep sessions. The depths came from a node query capped
  at 1,000 rows, depth 0 first, so from about 760 leaves the oldest condensed levels dropped out of the model's
  context. No stored row was lost. (#750)
- Perf: replay assembly lists the large-output payload directory once per assembly instead of once per aged-tier
  stub; a miss falls back to the original glob. (#808)
- Change: a routine exit fit (`reason=exit_fit:`) logs "LCM survival fit applied" at INFO; every other survival-fit
  line stays WARNING and the text is unchanged. (#735)
- Fix: a SQLite lock on the frontier read right after a committed threshold-sweep condensation reports the completed
  pass count in the fail-open telemetry instead of 0. (#24)
- Docs: `compression.target_ratio` belongs to the host's built-in compressor; LCM-X does not read it. (#605)
- Bench: drain and hidden-backlog cells (#801); gate exemptions limited to each open target's declared bars (#813)
  and, where declared, to one host (#822); the managed-profile runtime joins the nightly matrix; the B2 scorer pairs a
  held user composite with the parts the host stored apart (#804); an explicit `--embeddings on|off` LongMemEval arm
  (#811).

## v0.24.9 - 2026-10-01 (drain: one foreground time budget, stub-first exit, exit fit, scan allowance)

- Fix: with semantic embeddings enabled and the provider package missing or misconfigured, `lcm_doctor` and
  `/lcm doctor` report an `embedding_provider_health` warning (provider, model, reason, fix); it also warns when
  embeddings are off but the store still holds an active embedding profile. The probe is offline. (#672, contributor PR #673)
- Fix: with embeddings enabled and the provider unavailable, plugin load logs one WARNING with the fix;
  `requirements-semantic.txt` names the dependency and `docs/embeddings-setup.md` covers recovery after a host update.
  Load logging with embeddings off is unchanged. (#674, contributor PR #675)
- Fix: when a cancelled compaction had committed a leaf through the fresh-tail pressure yield, the host's retry adopts
  it before the summary-route stop applies, so the retry returns the shorter list with no summariser call. (#695)
- Fix: exit-fit follow-ups: a host-native result and an automatic call that turns into forced overflow recovery take
  no exit cap; a survival-fit record whose fits never shortened the list gets log guidance instead of rollback advice;
  the uncovered-row count reads unknown when dropped rows have no store id. (#715)
- Fix: on the foreground leaf and condensation paths, a source of at most 2 x `l3_truncate_tokens` (counted on the
  serialized text) whose first non-empty summary is not shorter is stored whole as a level 3 node: the route gets no
  circuit event (#628), one INFO line replaces the rejection WARNING, and no further route or level 2 call runs.
  Larger sources, empty results and callers without the small-source flag are unchanged. (#722)
- Fix: an automatic compaction whose input reached the threshold first tries its free cuts (#671): the summary
  prefix, the externalized placeholders and the aged-tier stubs, assembled without a model call. When the list is at
  or under min(threshold - leaf chunk, 0.95 x threshold) by the survival fit's host measure, it is returned with no
  leaf, no condensation and no summariser call; the rows no leaf summarised stay stored and are recorded as backlog.
  The stop reason `stub_first_exit` is a partial stop in `lcm_status` (`last_stub_first_exit`), the doctor and the
  `LCM compaction stop:` line (tokens before and after, the target, the backlog rows left). Recovery attempts, forced
  overflow and calls below the threshold never take it; when the cuts miss the target the compaction runs as before.
- Change: active-replay tool-result stubbing has two tiers (#671). A new tool result is stubbed at ingest from
  10,000 tokens (`LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS`, code default was 25,000), and at a
  compaction a tool result outside the fresh tail is stubbed from 2,000 tokens (new
  `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS`, `0` = the first-sight value). Both still need the
  two opt-in flags; the stub text is unchanged; results answering `lcm_describe` / `lcm_expand` stay whole (#587).
- Fix: the 10-minute sweep hold after a sweep spent its budget before the first leaf (#608) holds only the
  conversation it was armed for, and a session reset or a profile re-bind clears it. Both holds and the 60 s boundary
  cooldown read the monotonic clock, so a wall clock set back or forward neither extends nor ends them (`lcm_status`
  still shows a wall-clock `until`). While the #608 hold is active and the request is at or over the survival ceiling,
  an automatic compaction runs no sweep and no summariser call: it returns the list through the survival fit (status
  `noop`, reason `held`); a manual `/compress`, forced overflow and the host's recovery attempt are unchanged. A stored
  survival-fit count that is not a number no longer makes every later count write fail: the record restarts at 1 with
  `count_lost: true`. The `LCM compaction #` line prints the host's token figure (`host_tokens=`) when the host passed
  one. (#618)
- Fix: the #608 sweep hold is published through the host's transient-block interface like the #651 hold:
  `_automatic_compression_blocked` blocks while it holds this conversation (not for a recovery attempt, forced overflow,
  the survival ceiling, a pending cleanup-only pass or a bypassed session) and `_compression_block_reason` returns
  `cooldown:lcm_sweep_budget`. The host's forced preflight after a recovery attempt does not consult this gate, so that
  path is not changed by this fix. (#625)
- Fix: with the sweep off, the `LCM compaction stop:` line names a time stop (`time_budget_exhausted`,
  `soft_target_reached`) instead of `compacted`; pre-compaction extraction (`LCM_EXTRACTION_ENABLED`) gets at most the
  time left to the foreground hard bound on every foreground path, not the full summary timeout; and while every
  summary route is refused (#628), a leaf or condensation source within `LCM_L3_TRUNCATE_TOKENS`, which is stored whole
  with no call, is no longer stopped. (#605)
- Fix: level 3 repair continues after a group's SQLite error, rechecks sources during commit, records legacy-node
  provenance and schedules rollups; overlapping condensations retain reservations, and the read-only tool scan
  pages up to 50 fragments and ancestors from the foreground session. (#698)
- Bench: process-transport cells count a routine exit fit once (the in-process cells' normalization), and the
  reliability bar B8 also fails on a non-exit `LCM survival fit could not shorten the list` line. (#714)
- Fix: a store-complete leaf counts its extended scan allowance from the first open tool group's call row, so
  rows an earlier summary covers no longer use it up and the group's results reach a summary; when a full page
  holds only rows the leaf excludes (covered, ignored, passive, system), the scan reads up to four pages before
  it stops, so a session that resumes above a long covered run compacts again instead of no-oping. (#621)
- Fix: an automatic threshold compaction now fits its result below 95% of the host threshold (an exit fit), so the next
  API call does not compact again. The exit fit drops only whole older turns between the summary prefix and the fresh
  tail (never the summary prefix, a fresh-tail row or part of the newest turn); when no such cut reaches 95% the list
  stays as it is (an INFO `LCM exit fit skipped` line), and a list over the survival budget gets today's fit and its
  user warning. A routine exit fit shows the user no warning; its log line names how many dropped rows no summary covers
  yet. A list the fit cannot shorten now logs a WARNING and shows in `/lcm doctor`. (#668, #599)
- Fix: forced overflow recovery now counts the plain stub the final pass adds for every tool result it does not keep,
  before it keeps the tool call, so the assembled request stays inside the cap; when a call's externalized-output stub
  does not fit, the call keeps a plain stub instead of being dropped; a failed rotation-lineage write is logged at
  WARNING (both session ids, the error class) and rotation continues. (#697)
- Docs: the release gauntlet spec requires at least one committed compaction in the Phase C soak (else the soak is
  inconclusive), states the Phase C lossless multiset bar as the scorer checks it, and defines the differential rule's
  row identity (tool calls included; one fixture session); the AGENTS.md summary carries the "no worse than the base" guard. (#705)
- Fix: a threshold sweep has one time budget counted from `compress()` entry, not from after ingest (#605). Its
  progress call (a condensation pass when the summary frontier is over its target, #653, else the first leaf) is
  tried while at least 15 s of usable time is left: the hard bound (`LCM_FOREGROUND_HARD_SECONDS`, 120) less a
  finalize reserve (5-20 s, the p90 of recent finalize steps), so a stale estimate cannot starve a compaction. After
  a stored leaf or condensed node, a call starts only if its route's recent duration (p90 of the last 8 calls, 15 s
  floor, 30 s cold) says it ends, with the reserve, by the hard bound and by the new soft target
  (`LCM_FOREGROUND_SOFT_SECONDS`, default 60, `0` = off); the pre-leaf condensation no longer takes half the passes
  and half the time. Each call's timeout is at most the usable time left. The stop reason `soft_target_reached` is a
  partial stop in `lcm_status`. One INFO line per compaction, `LCM compaction stop:`, gives the reason,
  leaves, the progress call kind, elapsed seconds, the seconds before, during and after the calls, and the backlog
  left. A step already running (a store step, assembly, the fit, the host's stream read between
  two chunks) is not interrupted, so a compaction can still end past the hard bound; the stop line measures it.
- Fix: the same budget now covers a compaction with the sweep off, a manual `/compress`, a forced overflow recovery
  and a host recovery attempt, and the condensation passes after their leaf (#605). A manual `/compress` and a forced
  overflow recovery use the hard bound only. A time stop writes no level 3 node: with no leaf, a forced overflow
  recovery assembles to its cap with no call; the others keep their rows and hold the threshold answer as after a
  sweep budget stop. A leaf or condensation source within `LCM_L3_TRUNCATE_TOKENS` (512) is written whole as a
  level 3 node with no summariser call; rollups and the level 3 repair still call the model. (#605)
- Fix: with the full sweep off, a leaf is sized to at most max(`LCM_LEAF_CHUNK_TOKENS`, `LCM_DYNAMIC_LEAF_CHUNK_MAX`)
  of input, and to at most 40% of a known window of 50k tokens or more, so its call fits the time budget; the rest
  stays for later compactions. A single message or tool group larger than that is still kept whole, as before. A
  forced overflow recovery still takes the whole candidate. (#605)
- Fix: a threshold sweep whose raw prefix empties re-reads the owned hidden backlog before it reports the prefix
  drained, and takes the next hidden-only leaf while one remains, inside the same time and pass budget. (#597)
- Fix: on a host that offers `aux_stream_deadline`, each sweep summariser call runs under the earlier of the host's
  deadline and the budget's (#605). A timeout is a budget cut only when the budget bound the call (its timeout was the
  time left, below the configured one) or the host deadline fired: it ends the chain without counting against the
  route's circuit. A configured-timeout hit stays an ordinary route failure and the fallback route runs. A
  foreground compaction takes one spend-guard slot, at its first call; rollups count every call on their own guard,
  so maintenance cannot use up the foreground's. (#605)
- Fix: a Hermes process in which LCM-X did not become active (a slow load or another context engine in the slot) keeps its own
  record file, so overlapping processes no longer overwrite each other's notice; `lcm_doctor` and `/lcm doctor` report
  a live one as an `inactive_process` warning with the fix to apply; after a slot conflict the already-registered
  `post_llm_call` hook no longer writes `lcm.db` in that process. (#696)
- Docs: the release gauntlet spec (`bench/specs/RELEASE-READINESS-V1.md`) now states the rules the releases since
  v0.23.2 ran under: the differential rule for Phase B shapes and Phase C conflicts the previous GA also shows, the
  Phase B hands-on lane, recall-probe scoring (`exact` / `recoverable` / `LOSS`; only LOSS fails), provider failures
  the engine handles as designed, and receipts published as GA release assets. (#427)
- Docs: the embedding-dependency recovery note is linked from the README, the operator guide and the bundled skill, and two doctor messages say that a hand-installed dependency (not every optional dependency) can be dropped by a Hermes update. (#702)
- Fix (rc2): `/lcm doctor` now runs the `embedding_provider_health` check; in v0.24.9-rc1 only the `lcm_doctor` tool ran it, while the docs and the check's own fix text pointed users to `/lcm doctor`. (#734)
- Fix (rc2): with the full threshold sweep off (the default), an automatic compaction forms a leaf even when the backlog outside the fresh tail is under `leaf_chunk_tokens`, and an automatic exit fit drops only older turns a summary covers; before, the exit fit could drop uncovered turns at every compaction. (#738)
- Fix (rc3): after a stub-first exit, the next turn keeps the tool-output stubs the agent already holds, so a cleanup-only pass never returns a larger list and no tool output is stored twice. (#772)
- Fix (rc4): after the host refuses a compaction, or a pass stores nothing and shortens nothing (as after a failed
  summary publication; the 10-minute no-progress hold, #651),
  an automatic compaction at or over the survival ceiling summarises again, as in v0.24.8. In v0.24.9-rc3 it only fitted
  the list: a rotation session could stop compacting, and an in-place hidden-backlog drain waited up to 10 minutes
  (#802). Only the #608 sweep hold still makes that pass fit-only. (#541)

## v0.24.8 - 2026-09-30 (repairs: level 3 fragment repair, tool-output stubs that name the read-back call, rollup stop, no silent fallback on a slow load)

- Fix: a summary route that cannot serve its model (HTTP 400/404 whose message names the model as unknown, not
  found, not existing or not supported) opens its circuit on the first failure instead of the second, so level 2 is not
  attempted, and logs one WARNING per episode that names the provider and model the host used (`route_info`, on
  hosts that take it), says whether LCM-X sent a model, and names the fix (`auxiliary.compression.provider` +
  `auxiliary.compression.model` in the profile's `config.yaml`, or a consistent `model.provider` / `model.default`
  pair). Repeats log at DEBUG; a success ends the episode. Other failures keep the threshold and cooldown. `lcm_status`
  gains `summary_route` (`state`, `seconds_left`, `last_error_class`, `provider`, `model`). Rollups get their own
  config-error episode under their own breaker keys (#669); the live key is still the configured model, not the
  effective route. (#682)
- Fix: a host retry after a cancelled but committed compaction adopts the committed summary (#457) while every
  summary route is refused: on the first pass the adoption runs before the summary-route stop (#628), since it needs no
  summary route. With nothing to adopt the stop applies as before, and no new leaf is written while the circuit is
  open. (#640)
- Fix: a host recovery call (`compress(..., bypass_cooldown=True)`, #608) after a preflight that saw a
  compaction-boundary cooldown runs the summariser: it also clears the cooldown's cleanup-only handoff, so a list at
  or over the threshold and under the survival ceiling is summarised instead of only sanitised and fitted. An ordinary
  automatic call during the cooldown is still cleanup-only, and forced overflow is unchanged. The native-recovery
  handoff is kept (native recovery owns a below-threshold list). (#684)
- Feature: every new leaf and condensed summary node records its escalation level (1, 2 or 3) and the model that
  produced it, in a new `summary_node_provenance` table written in the node's own transaction. `lcm_describe` shows
  both; `lcm_status` counts nodes by level (`unrecorded` for older and imported nodes, which are not backfilled). No
  schema version change and no new `summary_nodes` column, so a plugin rollback still opens the store. Refs #441
- Feature: `LCM_SUMMARY_PROMPT_VERSION` (default `1`, unchanged prompts) opts in to summariser prompt v2: six fixed
  headings, focus directives in the trusted policy with only the tagged topic in the transcript message, and a 3x
  output ceiling. Refs #646
- Fix: a temporal rollup whose summary comes back as a level 3 truncation is not stored; the rollup stays pending for
  its next build and one warning is logged (a level 3 result that is the whole source is still stored). Rollup
  maintenance starts no build while the summary route is refused, and rollups record circuit results under their own
  breaker keys, so a rollup burst can neither open nor close the live compaction route's circuit. (#669)
- Docs (#685): the operator guide and the skill reference now say that since #652 a leaf whose level 1 and level 2 results are rejected is not stored while the survival fit can rescue the request; level 3 is still written when the cut is the whole source, in a forced overflow recovery, with the survival fit off or when the model window is unknown.
- Fix: a tool-output stub names the tool and says how to read the original, for example
  `[Externalized tool output: tool=read_file; tool_call_id=…; chars=N; bytes=N; read it with
  lcm_expand(externalized_ref="R"); ref=R]`. `ref=` stays the last field and the stub stays one line of at most 512
  characters, so v0.24.7 still reads it after a rollback; stored old stubs keep their text. The payload records the
  tool name; the LCM system note says how to read any tool-output stub. A ref written before a compression-boundary
  rotation resolves in `lcm_expand` and `lcm_describe` through the recorded rotation lineage (up to 32 sessions back);
  payload files are not rewritten, and a ref from an unrelated session still does not resolve. (#680)
- Fix: forced overflow recovery keeps the newest tool call when a user row precedes it and its result is over the
  recovery cap. The call is answered by the tool-pair stub, or by the result's #680 stub when it was externalized, as
  the shape without a user row already was; the recovered context stays within the cap and the stub is not stored.
  Capped assembly outside forced recovery is unchanged. (#636)
- Fix: a slow plugin load no longer leaves Hermes silently without LCM-X. When Hermes 0.21.5+ ignores
  `register_context_engine()` because the load overran `plugins.load_timeout_seconds`, `register()` logs one ERROR
  with the load time and the setting, and does not print "LCM plugin loaded — … active"; another engine in the slot
  gets a WARNING. The affected process leaves `lcm-x-not-active.json` in the Hermes home, and `lcm_status` and
  `/lcm doctor` in other processes report it while that process runs. The active line now carries the load time.
  The store open no longer waits for the write lock in steady state: a due FTS deep check whose claim cannot get
  the lock within 50 ms is skipped for that open (a later open runs it), the `ingested_at` NULL backfill scan runs
  once per store behind a migration marker, and with temporal rollups on the rollup marker and range normalization
  write only when needed. The lossless-claw importer writes `ingested_at` itself. (#622)
- Fix: when the host refuses a compaction of an LCM-bypassed session (an auxiliary side channel or a stateless
  session) as larger, the foreground session's automatic compaction is no longer held for up to 600 seconds; the
  same refusal on the foreground session still arms the no-progress hold. (#665)
- Doctor: `/lcm doctor repair level3` (and `lcm_doctor` with `action: repair_level3`) finds the level 3 truncation
  fragments that a refused summary route wrote before v0.24.6/v0.24.7, and every condensed node built on them, per
  session, with a check that each fragment's source rows are still stored. It is a read-only scan and changes
  nothing. `/lcm doctor repair level3 apply` (slash command only) takes a backup, then re-summarises each affected
  group in place from the stored raw rows, leaves first and each condensed node from its repaired children, one
  transaction per group; node ids, source links and raw rows are kept and the summary FTS index is updated in the same
  transaction. It refuses while no summary route is available, never writes a level 3 result, skips a group with any
  source row or child node missing, rolls a group back if it changed during the repair, and ends with a second scan. (#667)

## v0.24.7 - 2026-09-30 (P0 for long-running sessions: the survival fit keeps the summary, no in-turn thrash below the threshold, no level-3 fragments, a bounded summary prefix, steer rows stored)

- Config: the leaf summary target is configurable with `LCM_LEAF_TARGET_RATIO` (default 0.20),
  `LCM_LEAF_TARGET_MIN_TOKENS` (default 2000) and `LCM_LEAF_TARGET_MAX_TOKENS` (default 12000). With the defaults every
  target and `max_tokens` value is unchanged; an out-of-range value falls back to its default with a config warning.
  (#614)
- Fix: a mid-turn `/steer` that Hermes 0.21.2+ inserts as an unstamped user row before the steady-state ingest
  cursor is stored. The #436 prefix audit now also checks unstamped user rows against stored copies and moves the
  cursor back to one no stored copy explains; with host timestamps the displaced reply is not stored twice. The
  in-place shape of Hermes 0.21.1 and earlier and `LCM_IDENTITY_ANCHOR=0` are not covered. (#633)
- Fix: the summary envelope accepts three named formatting mistakes: `</summary>` in place of `</lcm-summary>`, one
  `<summary>` wrapper around the whole body, and one layer of quotes or emphasis on the closing hint (stored as the
  plain `Expand for details about:` line). The nonce opening tag, its uniqueness, the body minimum and a closing hint
  are still required. A discarded reply's warning names the failed check (`check=envelope|nonce_count|short_body|
  closing_hint`), and an accepted mistake logs one INFO line with its name. (#612)
- Fix: when a leaf or condensation gets only a level 3 truncation back and the compaction is not a forced overflow
  recovery with the survival fit able to run (on, and the model window known), no leaf or node is written; the
  compaction stops with `summary_result_rejected`, logs one warning and keeps the rows and nodes for a later compaction.
  A level 3 result that is the whole source (it fits the truncation budget) is still written. A sweep whose
  condensation before the leaves was rejected does not condense again after them in the same compaction. (#652)
- Fix: a threshold sweep whose summary prefix is over its target condenses it before the leaves, in at most half of
  the sweep's passes and time, so sweeps stopped by their budget no longer grow the prefix. Assembly renders the
  newest uncondensed summaries of each depth (oldest first), and a summary budget keeps the newest parts of a depth.
  At or below the target the sweep is unchanged. Every summariser attempt of a sweep (each route, each level) is
  bounded by the sweep's deadline, and one with less than 15 s left is not started. (#653)
- Fix: the survival fit keeps LCM's summary prefix and removes only whole oldest turns after it, choosing the cut by
  the final list's size, notice included. A carrier (the summary joined to the first user row) is split so the user
  row leaves with its reply, then re-formed as assembly forms it. The notice goes only into a real system row. When no
  whole-turn cut can hold the prefix, the v0.24.6 rule applies with one WARNING `LCM survival fit dropped the summary
  prefix (emergency: …)`, and the result is never larger than v0.24.6's. (#650)
- Fix (rc2): a Hermes update that brings in the new plugin package manager (`pm/`, after v2026.9.24) no longer disables
  `hermes-lcm-x`. The lint settings moved from `pyproject.toml` to `ruff.toml` and `pyproject.toml` is gone, so the
  checkout is not a uv workspace member; the stale root `uv.lock` is removed too. Before, the manager staged the
  lint-only `pyproject.toml` with a name and no version, `uv lock` failed and the plugin went into
  `plugins.disabled`. PR #632 diagnosed the same failure. (#631)
- Fix (rc2): below the host's compaction threshold the host count decides on every preflight branch: an automatic
  `compress()` after any preflight request is cleanup-only when the host's `current_tokens` is known and below the
  threshold, and an automatic call the #651 hold blocks is cleanup-only (the hold never blocks the survival ceiling).
  The one-shot flags clear on a session reset or rebind. Forced, `/compress` and provider-overflow calls are
  unchanged. (#677)
- Fix (rc2): the survival fit protects only DAG-verified summary rows as its prefix; a row quoting a summary header for a
  missing or foreign node is an ordinary row, so its turn leaves whole (the v0.24.6 rule). (#678)

## v0.24.6 - 2026-09-30 (#628, #627: a summary circuit that counts rejections apart from failures, and images priced per image)

- Fix: a summary result rejected for its content (empty, reasoning only, output contract violated, not shorter than
  its source) no longer counts as a failure of the summary route. Only a call that raises or times out counts toward
  the failure threshold (2); rejections open the route at their own threshold,
  `LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD` (default 6). Every rejected result logs its reason. (#628)
- Fix: while every summary route is refused, a compaction that is not a forced overflow recovery stops before the
  next leaf or condensed node instead of writing level 3 truncations, when the survival fit can run (on, and the
  model window known); the rows stay for a later compaction, and a
  stop before the first leaf holds the threshold answer until a route is allowed again (at most 600 s). The
  compaction line counts the level 3 leaves it wrote. (#628)
- Fix: `/lcm doctor` and the operator guide no longer call a plugin-only rollback within 0.24.x supported after a
  survival fit that dropped turns without a projection. Every store with a survival fit gets the backup restore. A
  stored count that cannot be read is treated as a fit. (#620)
- Fix: a failed survival-fit counter write is logged at WARNING instead of DEBUG. (#618)
- Docs: the exception of the #600 fix counts rows that an earlier summary already covers toward the 8,000 owned rows.
  (#621)
- Tests: the #608 tests no longer depend on which token counter is importable. (#615)
- Fix: a message whose structured content holds an image counts the image at the host's per-image price (default
  1,500) instead of the characters of the encoded image, so one screenshot no longer counts as about 130,000 tokens
  and no longer forces a compaction or cuts the fresh tail. Image data inside plain text still counts as text. (#627)
- Docs: the rollback advice names its target 'a version older than v0.24.5' and the configured database file. (#620)

## v0.24.5 - 2026-09-29 (#581, #582, #594: summaries sourced from the store, and a survival fit)

- Fix: a leaf summary can cover stored rows the host no longer shows (rows the host compacted in place while native
  recovery was ON, older duplicate copies); the leaf input is a bounded, contiguous run of the conversation's stored
  rows. Sessions that ended every pass in `publication_invariant_conflict` publish again. (#581, #591)
- Fix: when a compaction cannot shrink the context under the model window, the oldest whole user turns are dropped from
  that turn's context until it fits or until nothing stored can leave. A fit that shortens the list but stays over
  budget logs a WARNING; when nothing stored can leave, the list reaches the host unchanged (a warning for that case:
  #599) (`LCM_SURVIVAL_FIT`, default on; nothing is deleted). (#582, #591)
- Fix: a compaction the host refused, or another agent's session end on the shared lifecycle row, no longer leaves the
  next compaction conflicting at frontier 0. (#594, #591)
- Fix (rc2): a store-complete leaf no longer ends on an assistant tool call when the owned-row scan stops at its
  2,000-row cap inside a tool group; the leaf ends before the call and the next leaf starts with it, so no summary
  covers a call without its result. (#600, #604)
- Fix (rc2): on a cold resume after a survival fit that projected the newest user row, the replies stored right
  after the source row are recognised as replays instead of being stored again. (#602, #604)
- Docs (rc2): the operator guide states the supported rollbacks (to v0.23.3 only with native recovery ON, keeping
  lcm.db; within 0.24.x a plugin reinstall alone only while no survival-fit projection was persisted, else stop
  Hermes, move the current lcm.db aside and restore the backup taken before the first v0.24.5 install with the
  plugin), and `/lcm doctor` scopes that advice to fits that projected a row (`projected_count`); rolling back
  to v0.23.3 also reverts the v0.24.0 config migration. (#601, #603, #604)
- Fix (rc2): a threshold sweep whose 120 s budget is spent stops instead of raising `TimeoutError` to the host (which some
  Hermes versions treat as a stall and reset the session): the input list comes back unchanged with status `noop` and
  stop reason `time_budget_exhausted`, and no summariser call starts with less than 15 s left. After a stop before the
  first leaf, one WARNING names the step timings and the threshold answer is no for 10 minutes while the request is
  below the survival ceiling (the window minus the survival reserve); overflow recovery is not held. A recovery
  attempt for a request the provider rejected comes back under the compaction threshold even when no leaf could be
  stored. (#608, #617)

## v0.24.4 - 2026-09-28 (#436: message identity anchored on the host timestamp)

- Fix: a message the host shows is matched to its stored copy by the host timestamp plus its full content, counted per
  occurrence, instead of by an ordered content prefix. Host re-issues (compaction generations, rotation handoff, ACP
  persist replace, restore) no longer re-store the tail, a merged row is absorbed only with an exact decomposition
  proof (new additive `message_relations` table; schema version unchanged), and a summary claims only sources in its
  input. `LCM_IDENTITY_ANCHOR=false` restores the v0.24.3 identity path; the #589 tool-call-id fixes
  stay active either way. (#436, #553, #561, #563, #572)
- Fix (rc2): a repeated user message the host merges into a failed turn is stored as its own occurrence; a row the
  host shows on its own is never a piece of a merged row, and a storage rebind clears the identity caches. (#436, #580)
- Fix (rc3): the reservation also covers a row the host shows under a key other than the one stored: a row saved before
  the host stamped it, and a row shown through a recorded alias timestamp. (#436, #583, #590)
- Fix (rc3): a provider tool-call id reused in a later turn no longer makes the context bypass drop that later turn's
  result, and no longer exempts an unrelated result from active-replay stubbing. (#586, #587, #589)

## v0.24.3 - 2026-09-28 (#559: a leaf chunk never splits a parallel tool-call group)

- Fix: a leaf chunk ends only at a tool-group boundary, so a compaction never summarizes an assistant row with parallel
  tool calls apart from one of its results; before, the host dropped the orphaned result and every later publication
  in that session failed with `publication_invariant_conflict` (a freeze, no rows lost). (#559, #560)
- Docs: `LCM_MODEL_THRESHOLDS` for profiles that mix a 200k-token primary route with a 1M-token backup; all four
  supported `lcm:` YAML keys. (#554, #558)

## v0.24.2 - 2026-09-26 (post-v0.24.1 fix train: committed-frontier resume, replay binding of superseded outputs, forced-overflow recovery, todo-span identity, merge-append alignment, Anthropic tool schemas)

- Fix: a retried compaction after a host cancel resumes from the committed frontier instead of re-summarizing the covered prefix; non-consumable rows are preserved in place. (#457)
- Fix: a gen-2 retry after an adopted compaction binds the replay of the superseded output as a replay; only proven emissions (DAG-verified summaries, an objective re-render of a stored own-session row, the exact LCM note) are skipped, never a tool-carrying head. (#524; rotation child #526)
- Fix: forced-overflow recovery never returns an empty or system-only transcript; an over-cap newest user turn is announced instead of dropped. (#91, #529)
- Fix: the replay identity cuts only the Hermes todo-annotation span, so a row merged behind the annotation is stored. (#516)
- Fix: prior-proof consumption declines malformed projections strictly and records a fresh proof on failure. (#514)
- Fix: a host merge-append behind the retained last user row aligns in both walks; the post-adoption re-store wedge is closed. (#535)
- Fix: a rotation restart before the last carried prompt is answered no longer re-stores the restored list into the empty child; the durable walk accepts the host's replacement of that row and stale carry ranges are voided on the cursor-0 fallback. (#519)
- Fix: `lcm_compile_evidence` no longer declares a top-level `allOf`, which Anthropic's API rejects; a request carrying
  the lcm-x tool list failed with HTTP 400 on Anthropic routes since v0.21.0-rc2 (400→200 measured on one route). (#550)
- Tests: the `LCM_TEST_HERMES_AGENT_ROOT` opt-in is isolated from later test modules. (#513)
- Docs: the README and operator guide say how a `--ref`-pinned install moves to a new commit
  (`hermes plugins install … --force --ref <sha>`); `hermes plugins update` re-pins catalog installs only. (#523)

## v0.24.1 - 2026-09-25 (#488: replay skips bound to proven emitted occurrences; #517 rollback-readable proof)

- Fix: replay reconciliation binds every skip to a proven emitted occurrence (#488; #510 emission
  descriptors and the pure projection, #515 the consumer switch). The compaction commit proof's
  descriptors are version 4 while the durable record keeps wire version 3 (#517; versions 2 and 3 are
  still read for exact cursor matching): every emitted summary or
  objective row carries a descriptor (span digest, suffix witness, role, same-prefix ordinal,
  multiplicity witness, and the `output_occurrence` index and length). A live row is skipped only where
  its projection binds an emitted occurrence; an unproven row keeps its full identity and is stored; a
  merged composite maps to its own stored row, or — only when the remainder is in the proof's own
  effective output — to the row Hermes merged behind the emitted one, and otherwise stays unmapped
  until it is stored. Under a version-2/3 proof (until a session's first version-4 compaction) the
  v0.24.0 replay rules stay in force, narrowed so an objective head with a merged row is stored whole.
  Malformed descriptors are declined (full identity for their row), never raised on.

- Docs: the migration verify step names `hermes plugins list` (bare `hermes plugins` opens the interactive
  toggle); `scripts/install.sh` prints the same command. (#481)
- Fix: the durable compaction-commit proof keeps the version-3 wire format so v0.24.0 reads it
  after a rollback and that conversation compacts again; emission descriptors ride along under
  `descriptor_version: 4`, and this reader relabels such a record to version 4. (#517)

## v0.24.0 - BREAKING: plugin renamed to hermes-lcm-x, engine to lcm-x (2026-09-25)

**BREAKING.** The plugin manifest is renamed `hermes-lcm` → `hermes-lcm-x` and the context engine
`lcm` → `lcm-x`, to give LCM-X a distinct identity for Hermes plugin-catalog admission. (#471)
Hermes matches `plugins.enabled` against the manifest name, not the install directory, so a
config that enables only `hermes-lcm` stops loading LCM-X after an in-place update.

Migration (run exactly one LCM copy; change the config while Hermes is stopped):

1. Stop Hermes, then update the code: `git pull --ff-only` in the existing clone (it may stay at
   `plugins/hermes-lcm`) or install a new checkout with `scripts/install.sh`.
2. Config: in `plugins.enabled`, replace `hermes-lcm` with `hermes-lcm-x`, and set
   `context.engine: lcm-x`. Keeping `hermes-lcm` listed is harmless only when `plugins/hermes-lcm`
   is this same checkout; with a separate older copy installed, both would load. LCM-X then
   logs `Another LCM generation is already loaded` and stays inert (it registers no engine,
   tools or ingestion hooks), so only one copy writes `lcm.db`. The legacy
   `context.engine: lcm` still selects LCM-X through an alias, with a once-per-process
   `DEPRECATED LCM-X config` warning, an `identity_migration` field in `lcm_status`, and an
   `identity_migration` `warn` check in `lcm_doctor`.
3. Start Hermes and verify `hermes plugins list` lists `hermes-lcm-x` and the log shows
   `LCM plugin loaded — lossless context management active`. Remove a separate old directory by
   hand later; keep it until then for rollback.
4. If the config is not updated, Hermes logs `Context engine 'lcm' not found — falling back to
   built-in compressor` and runs without LCM-X. The existing `lcm.db` is untouched, but turns
   handled while Hermes runs without LCM-X (built-in compressor) are not in `lcm.db` and their
   compacted content may not be recoverable; update the config before restarting Hermes after
   the update.
5. Managed fleets (PCS / managed-plugin payloads) stay pinned to v0.23.x until their config
   stages `hermes-lcm-x` in `plugins.enabled`; that staging is a separate fleet-migration
   change owned outside this repository.
6. `scripts/install.sh` now installs `plugins/hermes-lcm-x` and `skills/hermes-lcm-x`, reuses an
   existing `plugins/hermes-lcm` link to the same checkout (relative links included), and
   prints these steps when it finds a legacy install or config. It never edits `config.yaml`
   or deletes the old directory.

Hermes plugin-catalog admission (#471):

- `post_llm_call` is registered through `ctx.register_hook` (direct `PluginManager._hooks` append
  kept only for hosts without `register_hook`); `plugin.yaml` declares `provides_hooks`
  (`pre_llm_call`, `post_llm_call`, `subagent_start`, `subagent_stop`).
- `scripts/update.sh` is removed (catalog rule: no self-updater). Update a checkout with
  `git pull --ff-only` + `scripts/install.sh`; catalog installs use
  `hermes plugins update hermes-lcm-x`.
- Install-scanner `dangerous` hits neutralized without behavior change (PEM markers in planted
  bench fixtures built by concatenation, bench env captures without piping the `env` listing, one code
  comment reworded); `hermes plugins validate` reports `security scan — safe`. The README lists
  the remaining informational findings.
- `FINDINGS-VERDICTS-*.md` moved from the repository root to `docs/history/`.

- The `DEPRECATED LCM-X config` notice (log warning, `lcm_status.identity_migration`,
  `lcm_doctor`) now leads with stop Hermes, edit `config.yaml`, start Hermes again. When the
  legacy engine alias is active it also names the fallback: the alias is fixed when the plugin
  registers, so editing `context.engine` while Hermes runs makes new sessions fall back to the
  built-in compressor until restart. `identity_migration` gains a `steps` list (stop, the
  config edits, start); `change` is unchanged (config edits only). The README and operator-guide step-4 check now reads "`hermes-lcm-x` enabled, `hermes-lcm` not
  enabled" instead of "no separate `hermes-lcm`", matching the kept-for-rollback directory.
  (#477)

- Governance: the required `AI review exact-head` check, its workflow and its validator are
  retired (#474). Enforcement is strict exact-head CI (the six checks), required review-thread
  resolution and merge commits; AI review is recorded evidence under the `land-pr` review
  obligation, and `scripts/maintainer_gate.py` reports the required review lanes as a hint.

- Fix: Hermes compaction boundaries no longer reset the frontier or store duplicate rows (#483).
  Hermes commits a compaction by calling `on_session_end(sid, <compress input>)` before
  `on_session_start(..., boundary_reason="compression")`. LCM-X handled that call as a real
  session end: it re-stored the fresh tail and finalized the session, so the next summary
  publication failed contiguity (in-place compaction) or the rotation child skipped its first
  turns. compress() now records a commit proof. An end call with exactly that input is a commit:
  no re-ingest, but the session is still finalized with its own frontier. An in-place compression
  start rebinds it and keeps the frontier and cursor. A session rebinding after its own finalize
  resumes the frontier it finalized, never another session's. The first
  post-compaction ingest trusts the cursor only when the host list matches the proof, and
  otherwise remaps or reconciles. Without a transferred proof (native recovery, a failed proof, a
  real end), the in-place or rotation start reconciles the cursor instead of keeping it. A
  host-merged summary carrier is identified by its glued row (each summary part is verified
  against its DAG node). Cursor reconciliation treats unverified summary-shaped text as content
  (#486), and digest-less redactions never count as proof, including native recovery replay and
  the store matcher (it stops before such a row). The proof is bound to its session, conversation
  and Hermes home. A durable proof carries the cursor across restart/resume, including an empty
  rotation child (commit or native recovery proof), and moves to the child on rotation. Stores
  that already hold #483 duplicates are not detected or repaired by this release (#485).

Unchanged: `lcm.db` (name and location), the `lcm:` config block, `LCM_*` environment
variables, all `lcm_*` tool names, `/lcm`, the bundled skill name `hermes-lcm`, and the log line
`LCM plugin loaded — lossless context management active`.

_Folded from "Unreleased" at v0.24.1: the rc4 fix train and follow-ups that shipped in v0.24.0._

- Fix: Hermes ACP turns no longer store duplicate rows or hit `publication_invariant_conflict`
  when the host trims the prompt at turn end (#498). LCM watches the user rows it stored; when
  the host rewrites that same object by edge whitespace only, it records an identity override
  under `host_rewrite_identity:<store_id>` (protected like ingest, bound to a digest of the
  stored content, refused for lossy redactions; a failed capture retries and never blocks the
  ingest; the in-process override cache is FIFO-bounded and reloads on a miss). Stored content is never modified. Only position-bound matchers read an override or
  tolerate edge whitespace: the retained user anchor, head-anchored restart
  replay (row i vs incoming i, only for a list extending past the stored session, so a rewrite
  LCM never saw before a crash is covered too), the durable proof walk, and the in-order store-id
  mapper (either form, one-to-one). Full-replay and tail classification stay exact.
  The durable commit proof moves to version 3; version-2 (rc3) proofs are still honoured.

- Native recovery now compresses only history before LCM-X's protected fresh
  tail, carries that tail forward verbatim, and records adopted-output proof so
  host commits do not duplicate durable rows. (#482, #487)

- Native rejection logs one secret-free warning with its reason class (such as
  `prefix_too_short`, `suffix_changed`, or `native_aborted`), adoption logs one
  info line, and `last_compression_noop_reason` carries the rejection class.

- Native adoption proof records summary positions in the same effective-row
  space used by replay reconciliation. The first mismatch after the summary
  remains the delta start; unsafe skip landings are guarded by leaving the
  landing unconsumed. Re-issued byte-identical call-only rows remain the known
  pre-existing #500 case.

- Fix: proof-backed Hermes rotation children now carry authority only for the exact parent rows
  mapped from the adopted compaction output, without moving raw-row ownership. Publication SQL
  reads only the current session and those carried ranges, while ignored carried rows receive
  explicit exclusion proofs. Metadata-only sanitized results remain host no-ops, but a failed
  publication keeps its distinct replay-safe result so Hermes can perform its rotation heal. When
  there is no system message, LCM now emits a verified user-role summary plus the following
  historical string user row as the same `summary\n\nrow` carrier Hermes would build, preserving
  that row's proof, store-id and carry identity before rotation. The tail's only user row is the
  current prompt and is deliberately not folded, so that residual adjacency remains; list-content
  rows and system-message contexts are unchanged. Before this carrier fix, #498 alone produced
  640 stored rows, 468 duplicates, 7 publication conflicts and 19 sessions in the affected
  80-turn default rotation-plus-trailing cell. (#495)

## v0.23.3 - maintenance point release

- Session-end prefix matching extracted from `engine.py` into `prefix_matching.py` as a mixin;
  no behaviour change (verified AST-identical at review). (#155)
- FastEmbed warmup prefers the locally cached model before enabling downloads; explicit warmup
  stays the only path that may download a missing model. (#404, addresses #235 — contributed by
  @Tosko4)
- Teams scope backfill is linear (ascending rowid cursor through session, derived and rollup
  backfills). (#408, closes #386 — contributed by @Tosko4)
- Test: the atomic compaction-telemetry contention regression no longer depends on runner
  scheduling; production's 100 ms best-effort policy is unchanged. (#407, closes #328 —
  contributed by @Tosko4)
- Governance: the exact-head receipt gate preserves valid peer receipts when one pull request's
  gate fails cleanly, fails closed on a transient error during its final read, and requires a
  `dispatch_id` in receipt dispatches. (#362)
- Records: BASELINE-LEDGER rows for the v0.23.2 security/privacy train plus the #155 refactor
  row, "ledger entry only, no re-baseline" (#412); the F53 V1-M
  re-bank registration (#413) and its park record FINDING-F62 (#416); contributor credits and
  full v0.23.2 PR coverage (#396). Documentation names v0.23.2 as the latest stable release and
  carries the forward identity `hermes-lcm v0.23.3 (15 tools)`.

## v0.23.2 - security + lossless point release

- Lossless by default: durable sensitive-pattern redaction (`LCM_SENSITIVE_PATTERNS_ENABLED`,
  default off) is now fully independent of cloud-embedding privacy. Known cloud providers
  protect only the provider-bound copy, controlled by the new `LCM_EMBEDDING_PRIVACY_ENABLED`
  (unset = auto-on for cloud); `false` is an explicit opt-out bound to the `privacy:off`
  vector revision. Cloud embedding no longer requires durable redaction, and a disabled
  durable policy no longer blocks cloud dispatch. Changing the posture changes vector
  identity and requires a new `/lcm embed warmup`. (#374)
- `lcm_recall` no longer silently degrades to full-text on an embedding-privacy policy
  error; proactive recall counts them and `lcm_status` exposes `privacy_policy_errors`. (#370)
- Rerank payloads are protected under the embedding-privacy resolution: the rerank query and
  snippets are transformed before leaving the machine when privacy is ON. (#371)
- Releases containing product code are rc-first: `bench/specs/RELEASE-READINESS-V1.md` is
  the GA gate. (#373)
- SECURITY: fixed a private-key redaction ordering bypass that could leak PEM key material to
  cloud embedding providers (#365 → #366); the rc gauntlet then caught and closed a truncated-PEM
  leak (#383 → #384) and its over-block regression (#389 → #391) through successive adversarial
  review rounds. The cloud-embedding redaction is
  best-effort by design (durable store is lossless regardless); boundary of record: #394.
- Current-schema database clones now open read-only, so a Hermes per-agent host clone no
  longer fails against a concurrent WAL writer and silently falls back to the built-in lossy
  compressor (#364 — contributed by @Tosko4).
- Stable release identity: `0.23.2` across plugin manifest, README, and operator docs, with
  rc/GA expressed only in tags and notes filenames; non-tautological downgrade guard (#385 → #388).
- Release/benchmark integrity: rc-first gauntlet is the GA gate (#373) with live-battery and
  runner hardening (#382 #390); privacy-trio/instrument boundary rows reconciled into the
  baseline ledger (#368); FINDING-F61 attribution result registered (#381); exact-head review
  governance hardening (#349 #358).
- Contributors: external code this release: @Tosko4 (#364). Review signal:
  chatgpt-codex-connector, evaos-code-review-bot, CodeRabbit, CodeQL, plus independent
  cross-model adversarial reviews recorded on the PRs.

## v0.23.1 - 2026-08-23

Stable privacy-only release for hosted summary embeddings; runtime bytes unchanged from
`v0.23.1-rc1`, no schema change.

- Cloud summary-vector privacy: every supported cloud summary-vector dispatch fails closed
  unless sensitive-pattern handling is enabled, nonempty, recognized, current, and
  residual-clean; provider input uses pattern-only placeholders that reveal no raw value,
  length, bytes, or secret-derived digest. Durable messages, summaries, FTS rows, and
  payloads are unchanged. (#330)
- Vector identity includes `privacy:v1:<active-pattern-hash>`; policy drift requires a fresh
  warmup before cloud dispatch. Complete and truncated private-key blocks are conservatively
  replaced before transport. (#332 #333 #338)
- Immutable `v0.23.1-rc1` release preparation; dry-run/apply reports add aggregate selected,
  transformed, blocked, and policy-revision fields with no content or identifiers. (#339)
- Follow-ups tracked, off in the shipped config: #334 trajectory cloud privacy, #335
  prescreen identity composition, #336 rerank payload privacy, #337 remote Ollama locality.

## v0.23.0 - 2026-08-22

Stable promotion of the isolation-only rc2 candidate; runtime and schema behavior
byte-identical to `v0.23.0-rc2`. 17 adversarially reviewed PRs with RED/GREEN receipts; every
score-sensitive change carries an architect verdict (#252) and a `bench/BASELINE-LEDGER.md`
boundary row.

- Replay-proof hardening: occurrence-bounded, tool-identified replay proofs (#177); out-of-band
  block durability binds to unique row identity with ID-less rows failing closed (#203); the
  vetoed fork-guard is removed with its drop contract fenced (#259). Ambiguity resolves to
  visible duplication, never silent loss.
- Retrieval: a slow full-text arm can no longer starve semantic recall (#173); cross-session
  summary-DAG expansion with correct provenance (#183); per-session retrieval exclusion (#184);
  a crashed search arm discloses instead of reporting 'ok' (#273); configurable Voyage reranker
  (#172); bounded tool-extracted evidence provenance for expand-query (#196).
- Integrity state publishes only with proof valid at publish time: verified-pass clearing and a
  CAS-fenced background-scan publish (#261 #179 #168 #198).
- Compaction's auto-derived focus keeps the newest user request authoritative for over-cap
  host-composed turns (#297; P1 #90 fixed; summary-steering ledger row appended).
- Teams slice 1 (AccessContextV1) lands DORMANT — pure additive, nothing consumes it,
  single-user behavior unchanged; enablement stays pilot-gated (#286, #75/#83).
- Governance: the main ruleset is satisfiable again — phantom CodeQL contexts dropped,
  last-push-approval deadlock removed, `--admin` an exception not the path (#241).
  Release-validation storage isolation (#325); regression coverage salvaged from closed PRs
  (#254 #257); gpt-5.4 retirement note (#262).
- Contributors: upstream-carried work from @stephenschoettler (#168), @Tosko4 (#173),
  @masidigital (#177), @davidrobertson (#183), @TurgutKural (#203, #205-prep).

## v0.23.0-rc1 - 2026-08-21

The correctness batch: 17 product PRs, each adversarially reviewed with RED/GREEN receipts;
score-sensitive changes carry architect verdicts (#252) and boundary rows in
`bench/BASELINE-LEDGER.md`.

- Replay-proof/data-integrity: #177 (occurrence-bounded replay proofs + tool-name identity;
  one-time snapshot-digest re-persist on upgrade), #203 (OOB proofs bound to unique row
  identity; ID-less rows fail closed), #259 (fork-guard removed per architect veto; drop
  contract fenced).
- Retrieval: #173 (FTS cannot starve semantic recall; hardened preflight), #183
  (cross-session summary DAG expansion), #184 (session exclusion filters), #273 (crashed
  search discloses, never reports 'ok'), #172 (configurable Voyage reranker).
- Reliability/ops: #261 (FTS bootstrap race; integrity state publishes only with
  proof-at-publish-time), #179 (doctor payload-ref provenance), #168 (durable compaction
  totals in status), #198 (separate summary/expansion reasoning controls; warn-and-ignore
  config posture).
- Teams candidate (NOT enabled): #286 — AccessContextV1 contract package, pure additive.
- Tests/docs: #254 #257 (salvaged regression coverage), #262 (gpt-5.4 retirement note).
- #297 — P1 #90: recovered/compacted sessions keep the newest user request authoritative
  (auto-focus middle-elision + newest-backwards budget); summary-steering ledger row.
- Governance: #241 — the main ruleset is satisfiable again (phantom CodeQL contexts dropped,
  last-push-approval deadlock removed); --admin is an exception, not the path.

Known gaps are listed in `.github/release-notes/v0.23.0.md` (notably #90 under
investigation, #244/#288/#265/#260 accepted follow-ups, Teams slices 2-5 in v0.24.0).

## v0.22.0 - 2026-08-19

- Rename the project-facing documentation to **LCM-X — Lossless Context Memory
  eXtension** while preserving the compatibility identifiers `hermes-lcm`
  (plugin/skill/install path) and `lcm` (runtime engine).
- Point current install, CI, contribution, and tag links at
  `electricsheephq/lcm-x`; retain upstream links only as labeled provenance.
- Document the RC2 memory-evaluation evidence separately from the unmerged LCM
  Teams, RC2 reconciliation, and Codex/whitepaper candidates.

## v0.21.0-rc2 - 2026-08-05

### Changed

- #492 corrects the optional `tiktoken` trajectory-state chunking path to
  preserve UTF-8 character boundaries while keeping each decoded chunk within
  its token budget. If the budget cannot contain one complete Unicode
  character, the path fails explicitly instead of emitting replacement
  characters.

### Evaluation baseline included in RC2

- RC2 contains the deterministic LongMemEval retrieval harness, the committed
  500-question FastEmbed result, and the vendored judged-QA adapter described
  in [`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md). These evaluation
  surfaces landed before RC2; RC2 includes them rather than introducing all of
  them in the RC2-only delta.
- The full judged-QA result and recommended Voyage retrieval run remain pending.
  Retrieval metrics are configuration-specific evidence, not a release,
  runtime-safety, or customer-readiness claim.

## v0.21.0-rc1 - 2026-08-03

### Highlights

- Add the trajectory/experience-memory subsystem and the opt-in assertion,
  evidence, query-view, and adaptive-retrieval surfaces delivered by the
  consolidated wave-1 merge (#436).
- Keep the core SQLite schema at version 5. New feature stores use additive,
  named migrations in the same profile database, while disabled/default-off
  installs do not create optional assertion, query-view, or embedding tables.
- Improve large-store and startup behavior with bounded vector/metadata work,
  lock-contention retry during WAL conversion, and deferred temporal-rollup
  maintenance (#361, #440, #446, #447).

### Changed

- #436 adds the consolidated trajectory/experience-memory, retrieval,
  exact-evidence, citable-delivery, privacy, scale, and release-validation wave.
  Its committed benchmark results are directional evidence for the documented
  harness and corpus, not universal provider or workload guarantees.
- #361 retries WAL conversion when connection setup meets lock contention.
- #440 moves temporal-rollup maintenance off the session-start critical path;
  bounded background work is eventual and `lcm_recent` retains its fallback.
- #446 and #447 batch large fixture setup for embedding/vector metadata release
  coverage without changing runtime behavior.

### Upgrade notes

- Back up `lcm.db`, update the plugin checkout, restart Hermes, send one normal
  message, then verify `plugin_version: 0.21.0-rc1` and the expected database
  path with `lcm_status`. The core schema remains version 5.
- No manual core migration or embedding backfill is required from v0.20.0.
- Query/evidence tool schemas are exposed after upgrade, but assertion
  extraction, assertion storage, query-view storage, pre-answer evidence, and
  adaptive retrieval remain opt-in. Review provider/privacy boundaries before
  enabling model- or embedding-backed paths.

- Added nested-default-JSON-bounded, tool-extracted `lcm_expand_query` evidence provenance so successful and degraded answers retain synthesis-context identities, occurrences, paths, and excerpts while explicitly distinguishing locator coverage from unverified replay, semantic entailment, and caller authorization.

## v0.20.0 - 2026-07-23

Release focus: Lossless-Claw parity plus the merged cross-session recall and temporal retrieval stack.

- Completed the five selected Lossless-Claw parity behaviors: recoverable active-replay stubs for large externalized tool results; token-bounded fresh tails that preserve the newest message and complete tool-call/result groups; dry-run-first historical tool-output backfill with guarded rollback; bounded active-session externalized-payload search with strict ownership and recoverability checks; and bounded atomic threshold full sweeps with one final active-context publication. (#380, #381, #382, #413)
- Shipped the merged #413 recall and temporal surface: `lcm_recall`, `lcm_recent`, and `lcm_load_session`; semantic and hybrid retrieval over summaries and message chunks; temporal rollups with bounded fallback; optional proactive recall; and the corresponding benchmark and reproduction documentation.
- Release boundary: stock installs keep large-output externalization, active-replay stubbing, embeddings, temporal rollups, proactive recall, and threshold full sweeps disabled by default. Payload search requires explicit `content_scope`; historical backfill remains an operator-invoked, dry-run-first command. Committed benchmark results are directional evidence under their documented model and harness, not a universal provider-parity claim. This release does not include the later work tracked in #423, #434, or #436.

## v0.19.0 - 2026-07-07

Release focus: data-safety hardening, operator diagnostics, import tooling, benchmarking, and the WS5 engine decomposition.

- Hardened lossless storage and replay boundaries: GC tombstones preserve surrounding text, ingest failures surface in status/doctor, ignored-message drops are counted, persisted Hermes tool outputs and redacted durable retries replay losslessly, and auxiliary bypass/session fallback edge cases are covered. (#298, #308, #310, #312, #313)
- Strengthened storage and downgrade safety with serialized lifecycle/DAG writes, monotonic frontiers, path-contained externalized payloads, ReDoS-safe redaction, wrapped-base64 handling, a summary spend guard, and a schema-too-new open guard. (#300, #301, #302)
- Added operator and migration surfaces: read-only `lcm_inspect`, JSONL session export import, compression no-op status, compaction telemetry, benchmark-backed preset validation, and steady-state hot-path benchmarks. (#295, #303, #306, #307, #309, #320)
- Added CI-backed ruff linting and release/validation-friendly tooling updates, including follow-up JSONL import hardening and metadata JSON access through `MessageStore`. (#314, #315, #316)
- Began and documented the behaviour-preserving WS5 decomposition of the ~9k-line `engine.py`: stateful method clusters became `*Mixin` classes (`compaction.py`, `reconcile.py`, `aux_session.py`, `placeholder_ledger.py`) mixed back into `LCMEngine`, and pure/helper groups became plain modules (`engine_registry.py`, `codex_routing.py`, `sqlite_util.py`, `runtime_identity.py`, `message_analysis.py`). (#323, #324, #325, #326, #327, #328, #329, #330, #331, #332, #333, #334, #335, #336, #337, #338, #339)

## v0.18.1 - 2026-06-30

Release focus: compaction privacy, clone/hook integrity, doctor signal accuracy, and model-context safety.

- Excluded ignored backlog and stripped injected context before compaction, preventing ignored or synthetic context from entering LCM summaries. (#283, #282)
- Preserved Discord lane metadata, active LCM clone resolution, and context metadata through cloned engines and post hooks. (#292, #293, #289)
- Hardened runtime identity, raw tool call integrity refs, payload integrity checks, and doctor path/lifecycle diagnostics. (#281, #278, #279, #291, #273, #280)
- Updated Codex OAuth effective context window safety defaults. (#274, #276)
- Completed focus-topic demotion behavior and preserved raw session ownership across compression rollover. (#268, #269)
- Refreshed operator docs, community-health files, and release-validation guidance. (#272)

## v0.18.0 - 2026-06-18

Release focus: retrieval depth, durability, status provenance, and long-session correctness.

- Added recursive evidence support for `lcm_expand_query`, improving synthesized answers from expanded LCM context. (#266)
- Hardened externalized payload durability. (#265)
- Avoided duplicate ingest protection work on hot paths. (#262)
- Aggregated DAG status stats for cheaper health surfaces. (#264)
- Preserved source lineage after long sessions. (#263)
- Surfaced LCM config provenance in runtime status. (#261)
- Fixed per-turn ingest for WebUI sessions and batch timestamp deduplication. (#260)

## v0.17.0 - 2026-06-14

Release focus: automatic focus-topic derivation and lifecycle hygiene.

- Added auto-derived focus topics during compression.
- Added empty lifecycle-row garbage collection to prevent unbounded accumulation. (#256)
- Improved runtime context indicators.

## v0.16.x - 2026-06

Release focus: engine isolation, WAL durability, database-path clarity, and startup cost control.

- Isolated LCM engine state per agent. (#247)
- Preferred bound sessions on sibling chains when the host has zero DAG.
- Tuned compaction defaults and clarified context-threshold ownership. (#245)
- Clarified `LCM_DATABASE_PATH` override behavior. (#249)
- Hardened WAL durability and graceful-close checkpoints. (#237)
- Throttled startup FTS integrity checks to reduce launch time. (#236)

## Links

- Version tags: https://github.com/electricsheephq/lcm-x/tags
- Release workflow: [`.github/workflows/release.yml`](.github/workflows/release.yml)
- Validation expectations: [`CONTRIBUTING.md`](CONTRIBUTING.md)
