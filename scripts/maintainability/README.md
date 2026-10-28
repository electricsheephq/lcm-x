# Report-only maintainability inventory

Run from the repository root, with Python 3.11+ and Ruff on PATH:

```sh
python scripts/maintainability/report.py
python scripts/maintainability/report.py --format json
python scripts/maintainability/report.py --baseline scripts/maintainability/baseline.json --write-baseline
```

Numbers, import violations, missing dependencies and operational errors are advisory.
Only argument usage errors return nonzero. Warnings mean incomplete evidence, not a
clean inventory. An incomplete scan will not replace the baseline. No extraction is
required to land a small fix. This script does not modify source or Ruff configuration.

The baseline represents the source tree containing it; it deliberately has no embedded
commit SHA (which would become stale on committing the baseline itself). JSON contains
all metrics, sites and changes; Markdown is limited to 60 lines with the largest function
LOC increases first. Baseline refresh is explicit; ordinary runs never write it.

## Definitions

- Scan Python files recursively, pruning hidden directories, `venv`, `node_modules` and
  `__pycache__`. Functions include tests and tooling. Production means root Python modules
  plus `access_context/`, `access_policy/`, and `teams/` (the audit's scope).
- Function identity is `file::qualname`, including nested functions and classes. AST LOC
  runs from `def` through `end_lineno`, including blank lines but excluding decorators.
  File LOC counts physical lines. Repeated definitions with the same qualified name use
  the last definition. Ruff provides CC, branch and statement counts. Thresholds are
  overridden to zero on the command line to obtain all positive values; these are
  measurement thresholds, never build limits. Noqa and per-file ignores are disabled
  for this measurement. `c901_over_10` permits comparison with the audit's default C901
  count. Ruff uses one thread, no cache and a 15-second subprocess budget.
- Report every positive change to a baselined function's LOC/CC/branches/statements;
  report new functions exceeding 80 LOC or CC 15. Decreases are visible in the inventory
  but do not create warnings. Missing Ruff values remain null, never guessed.
- Import contracts inspect all AST imports (including lazy and TYPE_CHECKING imports)
  in the six storage modules named in `report.py`. Relative, bare and `hermes_lcm`
  imports of the six orchestration modules are reported. Dynamic imports are not inferred.
- Attribute inventory follows LCMEngine's repository-defined base class names without
  importing product code. It records direct `self.attr` Store targets (including
  tuple/annotated/augmented assignments), with writer modules. It excludes container
  mutations, `setattr`, deletes and unrelated classes. New writer means a new module.
- `CONFIG_GETATTR` in `report.py` is the exact lexical definition: a `getattr` prefix
  with `config` or `_config` (possibly dotted), a quoted field and a following comma
  for the default argument. Production only; comments/strings can match. It does not
  recognize aliases, computed field names or optional whitespace before `(`.
- Private test references are AST attribute accesses whose immediate receiver name
  ends in `engine` or `store`, optionally underscore-delimited and numbered, plus
  private imports from `hermes_lcm.*`. This is a naming heuristic, not type inference;
  other aliases and generic private attributes are deliberately excluded.
- DDL matches CREATE TABLE, CREATE [UNIQUE] INDEX and ALTER TABLE in production string
  literals, including literal fragments of f-strings. Standalone strings/docstrings
  and comments are excluded. `db_bootstrap.py` is the schema owner; all other DDL sites
  are reported. Totals include the schema owner for audit comparison. SQL is not executed
  or parsed; dynamically assembled keywords are outside this metric.
- Site identity is `file::qualname::content#occurrence`, independent of line number.
  DDL uses a digest of its literal instead of copying SQL. The baseline retains line
  numbers for navigation. Reordering identical sites does not change their count.

## Attribute lifetime declarations

`scripts/maintainability/attr_lifetimes.json` maps attribute names to one of the
five lifetimes below. A value is either a lifetime string or an object containing
`"lifetime"` and an optional one-line string `"why"`; other keys are malformed.
For example, a new field could declare
`"_new_session_counter": {"lifetime": "session", "why": "Cleared by the session reset helper."}`.
The committed file starts as `{}`: existing baseline attributes need no declaration.
Declare new fields when adding them; this is a reviewer statement of intended
ownership and cleanup, not a proof that the implementation respects the lifetime.

Operational definitions, anchored to source at `ec07512f` (production code is
unchanged by this lane):

- **`process`**: assigned once during engine construction and never reset during
  that engine object's life. The object is the boundary, rather than the operating
  system process: `LCMEngine.__init__` creates `_stable_use_lock` and
  `_condensation_inflight_lock` at `engine.py:514` and `engine.py:517`; session and
  profile reset helpers do not replace them. Mutable contents can still change.
- **`profile`**: bound to the store/profile/config; storage rebinding replaces the
  database-backed helpers and clears state belonging to the old profile.
  Construction takes the config at `engine.py:505` and binds storage at
  `engine.py:536`; `_bind_storage` creates `_store`, `_dag`, and `_lifecycle`
  at `engine.py:927`. `_rebind_storage_for_home` closes/replaces helpers and calls
  `_reset_profile_runtime_state` at `engine.py:1108`. With a configured shared
  database path, the home changes and runtime state resets without replacing
  those helpers (`engine.py:1087`); config itself is not reloaded by this hook.
- **`conversation`**: survives session rotation within one conversation; clear or
  replace it when the conversation changes (including a profile change), rather
  than on each session callback. `_survival_host_overhead` records a peak keyed
  by conversation at `survival_fit.py:111`; `_bind_lifecycle_state` clears
  `_survival_overhead_observation` only when that key changes (`engine.py:2437`).
  Profile rebinding also clears it (`engine.py:1026`); session reset does not.
- **`session`**: cleared or rebound at session start/reset/rebind. The helpers
  `_reset_session_counters`, `_reset_compaction_progress`, and
  `_reset_session_scoped_runtime_state` start at `reset_state.py:14`,
  `reset_state.py:42`, and `reset_state.py:49`.
  `on_session_start` calls through to the rebind paths (`engine.py:3900`,
  `engine.py:4180`); `on_session_reset` clears runtime state (`engine.py:4599`,
  `engine.py:4628`). Verified compression carry-over may restore progress.
  `on_session_end` finalizes state rather than clearing every field
  (`engine.py:4201`, `engine.py:4555`); bypass compressor state is explicitly
  cleared on end/reset/replacement (`bypass.py:85`, `bypass.py:124`, `engine.py:4607`).
- **`invocation`**: valid only during one compress, assemble, or tool call, with
  cleanup on every exit, including early return and exception. The tail bound
  is initialized at `engine.py:608`; `_fresh_tail_pressure_yield_invocation`
  sets up a clean scope (`engine.py:2702`) and uses `finally` for cleanup
  (`engine.py:2740`). Nested calls restore the outer invocation's saved state;
  a reset inside the call wins over restoration. This anchor covers compaction
  entry points, not an assertion that every tool/assembly field is scoped so.

Comparison adds `new_attributes_without_lifetime`, `invalid_lifetimes`, and
`stale_lifetimes` as sorted attribute-name lists. Invalid declarations do not
satisfy coverage for new fields; stale means absent from the current AST inventory
(which excludes dynamic `setattr` writes). Missing declarations produce one warning
and act as an empty mapping. Unreadable/non-object files also warn and use an empty
mapping. All diagnostics remain advisory with exit 0. JSON retains complete lists;
Markdown reserves space for every category and truncates details with omitted counts
within 60 lines. Baseline refresh remains a separate explicit action.

A post-GA CI step can run `python scripts/maintainability/report.py >> "$GITHUB_STEP_SUMMARY"`.
No workflow is installed by this lane. Keep the step report-only; do not compare numbers
in a blocking assertion. Use the same Ruff version when comparing baselines.
