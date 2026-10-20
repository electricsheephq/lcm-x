# C1b split scoring (`c1b-terms-v1`)

This is an offline diagnostic of existing reader answers. It makes no model
calls and changes neither `score_s.py` nor `score_deep.py`'s historical metrics.
The output labels that original score **strict (historical)**. It is the original
normalized full-answer classifier plus the original admission/truncation rules,
read from the retained scorer output and checked against the RCA per-fact trace.
It is not the same as case-sensitive full-string presence in context.

Run from the repository root (C1, RCA and OUT are retained artifact directories):

```sh
python3 -B bench/instruments/compaction_probe/track_s/decision/rescore_split.py \
  --source "$C1" --rca "$RCA" --out "$OUT"
```

The command snapshots the required JSON/text receipts before scoring, including
reader answers and assembled context. It never copies or opens a store/DB.
It refuses to overwrite an existing snapshot. Recompute from that snapshot:

```sh
python3 -B bench/instruments/compaction_probe/track_s/decision/rescore_split.py \
  --inputs "$OUT/inputs" --out "$RECOMPUTED"
```

Snapshot digests and the original material-manifest digests are checked. The
historical strict results, context-presence flags and event flags are checked
against retained scorer/RCA receipts; a mismatch stops the rescore.

## Independent per-fact verdicts

Each fact at each completed seed-1 checkpoint has all five fields; the four new
fields do not change its strict verdict. True/false are pass/fail. Null is missing
evidence, or scalar N/A for a nonnumeric class; aggregate counts retain that distinction
through the class and numeric denominator. Partial seed 2 supplies event evidence
only, so it cannot inflate the completed-checkpoint retention population.

* **verbatim:** the exact, case-sensitive full expected checkpoint value occurs
  in the assembled system/view content, independently of the reader's answer.
* **semantic:** the answer includes the expected value identifier and every
  class-specific qualifier group below, and passes association. Matching is
  deterministic, case-insensitive, and uses alphanumeric boundaries around
  identifiers. This is a term rule, not a general semantic judge or entailment
  claim. No fixture-specific answer exceptions are added.
* **scalar:** for `limit` and `tool_number` only, the exact digit token occurs in
  the answer regardless of prefix or units. A larger digit token or an embedded
  alphanumeric ID does not pass. Other classes, including hex build IDs, are N/A.
* **association:** the answer to the recorded probe ID contains the correct
  identifier, with no other fixture's distinct identifier, explicit sibling
  fixture label, or stale identifier. The probe ID provides implicit fixture
  binding, so an answer need not repeat its fixture label. Correct and sibling
  values together fail conservatively; this does not interpret quoted denials
  of a sibling value. This is distinct from qualifier/rationale retention.

| Class | Identifier | Required qualifier groups (AND between groups) |
|---|---|---|
| name, path, build_id, prefix | Full current value | None |
| limit | Exact digit token | MiB |
| tool_number | Exact digit token | rows/row; verified (`rows_verified` satisfies both) |
| decision, superseded_value | Value before ` because ` | If present: every rationale word; the deep correction rationale permits `read-only`/`read only` and a `preserv…` stem, plus replay and ownership |
| error_with_fix | `E_` + hex | cursor; missing; rebuild…; index… |
| pending_task | Hex ID | verif…; rollback; before; release |
| early_user_constraint | Full `fixtures/…` path | never modify / do not modify / read-only / read only / preserve… followed by unchanged |
| file_change | Full `src/…` path | preserv…; tool_call_id; replay |

Thus bare numeric digits can pass scalar and association while failing strict
and semantic. An abbreviated fix that omits “cursor missing” still fails semantic;
an abbreviated pending task that omits “before release” also fails. These are
deliberately conservative qualifier checks, not an inferred retention uplift.

## Request and constraint probes

The deep generator records `request_policy` in `deep_metadata.json`. Its initial
objective is never superseded in this material. Each user row replaces the prior
**immediate instruction** only; a fact correction supersedes that fact's value,
not the standing audit/report objective. The latest user row includes corrections,
even when the reader instead repeats the preceding continuation instruction.
Legacy deep-v1 receipts have no such field: the rescore derives this policy from
their pinned generator/material and explicitly labels that provenance. Original
material and digests stay unchanged. Ground truth records exact source row IDs,
row indices, immediate-instruction supersession, and `initial.superseded_by`
(null, or an explicit revoking row). A revoked objective is N/A for retention.

* **R-initial** reports operative status, exact initial-objective presence in
  context, semantic term coverage in the historical R answer, and retention
  evidence = context verbatim OR answer semantic. No later continuation is
  silently treated as the initial objective.
* **R-latest** reports the latest admitted user instruction's source row and
  compliance from the same historical R answer. Semantic matching requires all
  alphanumeric expected terms, ignoring case/punctuation/order and the function
  words `the and a an of to in only`. Seed digits and all other terms remain
  mandatory. The same term rule checks R-initial's historical reader answer.
* **Q** reports exact constraint presence and the original `reader_pass` separately.
* **O** reports exact constraint presence and **O-compliance = reader_pass**.

The original R/Q/O/H event flags are retained as historical evidence. Reusing an
answer to the ambiguous “current request” question is not a fresh R-initial
read test. In particular, an R-initial diagnostic failure cannot establish loss
of the semantic objective from context. Host-instruction controls remain unchanged.

No product code, retrieval settings, model prompts/calls, release gates or
customer runtimes are changed. Publication is a separate authorized action;
unpublished result sets and every re-score dependency must be retained.
