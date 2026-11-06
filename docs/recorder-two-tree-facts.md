# Recorder (#2243) turn-path probe packet: consumers, two-tree table, differential, shapes, probes

M = main 4aca75d5a, R = rc1 7594beb7, H = Hermes bench pin 71d34bad. "held" = the spec's fact matched the source.

## 2. Two-tree table: spec facts re-checked
| Fact | M | R | H | Result |
|---|---|---|---|---|
| `engine = LCMEngine(...)`, next engine call, ctx lookups | `__init__.py:587`, :591-596, :604-881 | :587, :591-596, :604-862 | — | held |
| `_wrapped` tool handler shares one code object | `__init__.py:110-114` | :110-114 | — | held |
| `/lcm` task-local context | `__init__.py:153-210` | :153-210 | — | held |
| post_llm_call cold-prototype fallback | `__init__.py:~863` | :844-846 | — | held |
| `counted` decorator (compress only) | `compaction.py:82-95`, :731 | :82-95, :731 | — | held |
| `record_rejected_compaction` | `engine.py:2397` | :2789 | `conversation_compression.py:3248` | held |
| `_automatic_compression_blocked` | `engine.py:2448` | :2839 | :1563 via `type()` | held |
| `_compression_block_reason` | `engine.py:2478` | :2869 | :1569 | held |
| `bind_session_state` / `clone_for_agent` / `shutdown` | `engine.py:4551` / :1171 / :11774 | :4931 / :1567 / :10941 | `run_agent.py:419-422`, `agent_init.py:2004` | held |
| `record_completed_compaction`, `maybe_seed_preflight_display_tokens` | absent | absent | `codex_runtime.py:236`, `turn_context_compaction.py:271` | "where present": absent on both trees |
| `_invoke_summary_attempt`, single caller | `escalation.py:750`, :1107 | :750, :1107 | — | held |
| classifiers incl. `exc.status == 429` | `summary_attempt.py:44-95`, :78 | same | — | held |
| `engine_registry` weak dicts | `engine_registry.py:24-25` | same | — | held |
| aux predicate | `aux_session.py:496` | :493 | — | held; reads `_subagent_id`, `_parent_subagent_id`, `_delegate_depth`, `_memory_write_origin`, `_memory_write_context`, `log_prefix`, `enabled_toolsets`, `ephemeral_system_prompt` (both trees) |
| M 48-frame scan | `aux_session.py:809` (`_FRAME_DEPTH`), :981-995 | absent (32-frame scans only, :571/:612/:638/:773) | — | held, but see risk 1 |
| watch structures | `reconcile.py:767-775` (`_host_rewrite_watch` in `__dict__`, `_detached_host_rewrite_state` on the store), :804-835 | :778-809 | — | held; plus `sender_capture._PENDING[*].obj` (both trees) |
| lineage tag | `host_uid.py:~95` sha256(resolved state.db)[:16] | :93 | — | held |
| sessions columns read | `host_uid.py:41-56,118`, `identity_anchor.py:95-99`, `aux_session.py:721,783,1006`, `engine.py:2590`, `lifecycle_state.py:547` | same set | — | held: the 7-column projection covers every read |
| payload dir | `externalize.py:190-225` (`lcm-large-outputs`), in-place rewrite :1246-1254 | same | — | held (files are not write-once) |
| atexit callables | `maintenance_scheduler.py:152-154`, :409-428 | `engine.py:404-408`, :664-694 | — | held |
| host-written attributes | — | — | `conversation_compression.py:176-189`, :414, :1970-1982; `codex_runtime.py:192`, :233-243 | held; extra H writes found: `_tail_token_budget`, `model_thresholds` (`agent_init.py:1981`), `_config_context_length` (:1758), `_context_probed`/`_context_probe_persistable` (`turn_usage.py:171-172`), `last_compression_rough_tokens`, `_verify_compaction_cleared_threshold`, `_proactive_prune_rearm_tokens`, `_flush_scan_cursor_invalidated`, `_compression_attempt_*` |
| bounded-hook thread name reuse | — | — | `plugins_dispatch.py:333` | held |
| `os._exit` on every gateway/CLI exit | — | — | `gateway/run.py:6034-6078`, `hermes_cli/main.py:97` | held |
| fresh structural clones to on_turn_complete (strings shared) | — | — | `conversation_loop.py:1018-1026`, :1338 | held |
| sessions row created after on_session_start | — | — | `run_agent.py:313-343`, `agent_init.py:2144` | held |
| PluginManager per home / force reload | — | — | `plugins.py:1715`, :1294 | held |

## Risks found and their decisions
1. M's `_delegate_child_agent` (aux_session.py:981-995) matches any frame `self` whose `session_id` equals the
   child id, not only frames the aux predicate accepts; a predicate-filtered descriptor does not cover it.
2. SQLite-side clocks (`DEFAULT (strftime('%s','now'))`, db_bootstrap.py:802, :917, :1269, :2120, :2159, :2183) are
   not frozen by freezing Python's clock, so "all columns equal" in the record-only test needs a listed exclusion.
3. `_last_compress_aborted` is a property on LCM (bypass.py:140-145); H's attempt-state restore list names it.
4. The CI host stub `agent/context_engine.py` lacks several H ContextEngine methods (`select_context`,
   `has_content_to_compress`, ...), so roots come from the union of the imported class and the fixed H list.
5. H calls engine names LCM does not define (`get_active_compression_failure_cooldown`,
   `note_usage_less_response`, `snapshot_preflight_display_tokens`, ...): absent on both trees, guarded by
   `getattr` in H, not store-affecting.
6. `carry_over_new_session_context` (M `engine.py:5634`, R :5962, store-affecting) is called by H on the engine
   (`run_agent.py:375`, `_call_engine_hook`) and is not a ContextEngine method. It was missing from the spec's fixed
   list; added to `EXTRA_ENGINE_ROOTS` with a test.
7. `_compaction_telemetry_counter_epoch` is `uuid4` per engine (M `engine.py:921`, `reset_state.py:31`) and lands
   in `metadata` rows. It is recorded once per engine as a `seed` event and the replayer applies it. One regenerated
   inside a root (in-root reset) is a declared gap.

Decisions 1-6 of Delivery checkpoint 2: (1) the aux descriptor records every frame either rule accepts, with the rule;
(2) the record-only test excludes the SQLite-default time columns listed in `tests/test_recorder.py`
(`SQL_CLOCK_COLUMNS`, with DDL file:line); (3) the extra watch-list attributes above are recorded; (4) roots are the
imported class plus the fixed list; (5) payload copy at the pin; (6) env keys masked by name segment.

## 1. Consumer enumeration

Recording observes through `sys.monitoring` (PY_START/PY_RETURN on root code objects, a filtered PY_UNWIND); no
function, method or attribute is wrapped or replaced, callbacks return `None` (never `DISABLE`) and catch everything
except `KeyboardInterrupt`. So every call site keeps the same arguments, return and exceptions; the record-only test
(stores, returns, status) and the failure-isolation tests are the proof. Call-site counts from `rg -n` (production /
tests in M; production in H with an engine receiver; names collide with other receivers in the raw counts):

| Root | M prod | M tests | H production call sites (engine receiver) |
|---|---|---|---|
| compress | 34 | 1043 | `conversation_compression.py:2987` |
| should_compress / _info / _preflight | 20 / 0 / 7 | 186 / 1 / 203 | `turn_preflight.py:102`, :305; `turn_context_compaction.py:306`, :77, :438 |
| should_defer_preflight_to_real_usage | 0 | 0 | `turn_preflight_gate.py:97` (LCM: absent, inherited default) |
| on_session_start / _end / _reset | 36 / 14 / 7 | 1487 / 391 / 45 | `run_agent.py:359-373`, :187; `agent_init.py:2144`; `conversation_compression.py:1674`, :2537; `cli_session_mixin.py:479` |
| carry_over_new_session_context | 1 (engine.py:5726) | 0 | `run_agent.py:375` |
| on_turn_complete | 4 | 34 | `conversation_loop.py:1338` |
| select_context | 1 | 0 | `conversation_loop.py:1291` |
| prune_tool_results_only / has_content_to_compress | 0 / 0 | 0 / 0 | `turn_preflight.py:376`; `conversation_compression_manual.py:107` |
| update_from_response / update_model | 0 / 10 | 121 / 78 | `codex_runtime.py:164`, :189; `turn_usage.py:91`; `turn_overflow.py:335`; `chat_completion_helpers.py:2025`; `turn_recovery.py:1675` |
| get_status / get_tool_schemas / handle_tool_call | 9 / 4 / 20 | 157 / 22 / 350 | `agent_init.py:2124`; `tool_executor.py:1677`, :1680 |
| get_automatic_compaction_status_message | 0 | 10 | `context_engine.py:41` |
| clone_for_agent / bind_session_state | 2 / 0 | 33 / 19 | `agent_init.py:1925`; `run_agent.py:422`; `agent_init.py:2004`; `conversation_compression.py:1683` |
| record_rejected_compaction / _automatic_compression_blocked / _compression_block_reason | 0 | 54 / 23 / 13 | `conversation_compression.py:3248`, :1563, :1569 |
| record_completed_compaction / maybe_seed_preflight_display_tokens | absent | 0 | `conversation_compression.py:3515`; `codex_runtime.py:236`; `turn_context_compaction.py:271` |
| shutdown | engine teardown | many | plugin/engine teardown (name collides widely) |
| `_flush_pending_tails_at_exit`, `_quiesce_embedding_maintenance` | atexit only | 0 / 1 | — (atexit) |
| `_invoke_summary_attempt` | `escalation.py:1107` (single caller) | 0 | — |

Plugin surfaces in `__init__.py` and how each is captured: every `ctx.register_hook` (pre_llm_call, post_llm_call,
on_session_start/end, subagent_start/stop and the rest), `ctx.register_tool` (all `lcm_*` tools, which share the
`_wrapped` code object and are told apart by the captured name), `ctx.register_command` (`/lcm`, with its task-local
context) and `ctx.register_system_prompt_section` are captured by instance-attribute registrars on ctx that record
the callback's code object and name, then call the original registrar unchanged.

Paths into the engine that are not roots (declared gaps): background threads started by the engine itself
(maintenance scheduler, embedding maintenance, background leaf prep), whose summary calls appear as `bg` roots;
direct `call_llm` callers (`extraction.py`, `assertion_extraction.py`, `host_evidence.py`, `selective_compiler.py`,
`expand_query_tools.py`), whose tables are outside the comparison. Engine names H reads that LCM does not define
(`get_active_compression_failure_cooldown`, `note_usage_less_response`, `snapshot_preflight_display_tokens`, ...)
are guarded by `getattr` in H and are not store-affecting. `name` and `_last_compress_aborted` are properties
(attribute reads, not roots).

## 3. Previous-GA differential row

`LCM_DIFFERENTIAL=1 pytest tests/differential -rx -p no:randomly` against v0.28.0 (`709099b3`), candidate the dirty
worktree: 97 passed, 2 xfailed (#1225, #1247, both existing), 18 summaries over 1,076 cases, 0 unexplained, 0
accepted deltas used. The differential runs with the flag off, which is the shape that must not change.

## 4. Sibling shapes exercised

| Shape | Test |
|---|---|
| flag off | `test_flag_off_installs_nothing_in_a_fresh_process` (also on 3.11 guard) |
| flag on | record-only, round trip, replay smoke, views, identity, state captures |
| flag on, recording failure | `test_recorder_failures_never_reach_the_host[*]`, writer stall/gap, unwritable/disk-full, KeyboardInterrupt |
| flag on across a rebind | record-only scenario rebind; `foreign_path` counted for an engine on another db path |
| two engines on one path | `test_overlapping_process_directories_are_flagged` |
| second install | `test_second_install_shares_one_tool_id_and_process_directory` |
| Python 3.11 with the flag on | `test_python_below_312_with_the_flag_on_installs_nothing` |
| Hermes-style `os._exit` end | `test_exit_paths[os_exit]` ("ended without exit"), `test_exit_paths[atexit]` |

## 5. Invariants

I1 (exactly-once storage) and I2 (lossless) cannot change: the record-only test compares every table and column
(SQLite-default clock columns listed) with the flag on and off. I13 (trial assembly publishes nothing) also applies:
recording reads views and never assembles. No schema change (I5/I6 untouched).

## 6. Risk probes for the reviewer

(a) nested host edit after a root: `test_views_edits_aliasing_clones_and_watched_rows` expects `e` ops for rows 0-1;
(b) a compress that raises: `test_a_raising_compress_is_recorded_and_reaches_the_host_unchanged` expects the same
exception at the host and its type and text hash in the event; (c) two clones on two threads and a collected clone:
`test_identity_nesting_threads_and_engine_gone`; (d) budget numbers: `test_turn_path_budget` (run alone,
`RECORDER_BUDGET_STRICT=1` for the 1x bounds); (e) a commit between the pin and the backup:
`test_snapshot_pin_excludes_a_later_commit_and_payloads_match_the_manifest`; (f) `test_frame_neutrality` and the
flag-off import check; (g) `test_exit_paths`; (h) `SystemExit` in a callback or capture method:
`test_recorder_failures_never_reach_the_host[*-SystemExit]`; (i) a sessions row after `on_session_start`:
`test_state_captures`, `test_sessions_absence_and_read_error_are_their_own_outcomes`; (j)
`test_sealed_checkpoint_prefix_replays_without_exit`.

