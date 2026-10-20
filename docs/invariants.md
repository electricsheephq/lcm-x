# LCM-X invariants

These are the properties every change must keep. Specs, review packets and fix PRs cite them by ID (for example "touches I3, I4").

**What counts as enforcement.** An invariant is enforced only when a test fails with a named past defect's fix reverted and passes
with it restored. That red/green pair is the row's *seed proof*. A test that merely exists, or names the invariant, is a lead, not
coverage. Rows without a seed proof are listed as gaps. There is no zero-gap target: a gap is closed by a proof, never by relabelling.

`tests/test_invariant_catalogue.py` checks this file mechanically: IDs are unique, every referenced test file exists, and every
row marked `proven` names its receipt.

## Catalogue

| ID | Statement | Enforcing code | Tests today (leads) | Seed defect | Seed proof |
|---|---|---|---|---|---|
| I1 | **Exactly-once storage.** Every host occurrence is stored once across re-ingest, cold resume, rebind, rotation, crash and retry. Only named exceptions apply: the #259 policy (duplication over loss) and #923. | reconcile.py, identity_anchor.py, recognition_matching.py, emission_projection.py | test_issue_436_identity_anchor.py; test_issue_436_matching.py; test_issue_633_steer_row_before_cursor.py; test_issue_650_survival_fit_keeps_summary.py | #633; #1000 (fix merged; release closure pending); #1179 (open) | proven: [#633](invariants/receipts.md#i1-633), [#1000](invariants/receipts.md#i1-1000) |
| I2 | **Lossless.** Compaction and survival fit never delete or alter a stored row. A row leaves the live context only with durability evidence. | survival_fit.py; compaction.py; emission_projection.py | test_issue_581_582_store_complete_survival.py | #581, #582 | proven: [#582](invariants/receipts.md#i2-582) |
| I3 | **Fit versus previous GA.** If the previous GA fits window W for input X at defaults, the candidate fits. Otherwise the candidate sets `_last_overflow_recovery_failed` and warns; it never sends over the window silently. | survival_fit.py; engine.py (`_assemble_context`) | test_issue_1220_survival_short_rows.py; tests/differential | #1220, #1227 | proven: [#1220](invariants/receipts.md#i3-1220) |
| I4 | **Objective retention.** The newest user turn and the objective reach the model verbatim, or as head plus mark with a note, whenever the previous GA kept them. Known residual: #1225. | engine.py (`_assemble_context`) | test_issue_1221_objective_default_bound.py | #1221 | gap |
| I5 | **Supported rollback.** Supported older readers read and write candidate stores and replay emitted rows without re-storing them or wedging, and the candidate reopens the store. Unsupported formats keep their documented recovery path: v0.26.x cannot recognise v0.27.0 short projections (head + mark). | dag.py; store.py; diagnostics.py | test_rollback_v0262_store.py; test_host_uid_rollback_reader.py; test_issue_919_rollback_advice_projection.py | #601 | gap |
| I6 | **Forward migration.** Stores written by the previous GA, and fleet-shaped stores, open with no row loss and no destructive drop. Unknown newer shapes are refused. | db_bootstrap.py | test_schema_stamp_remediation.py | to choose | gap |
| I7 | **Summary outcome equivalence.** Typed results keep current behaviour: config errors are handled as today, rejected results open their own circuit, configured timeouts count as route failures, and content-filter and context-rescue handling are unchanged. Policy changes ship separately. | escalation.py | test_issue_628_summary_circuit.py; test_issue_1120_budget_cut_by_deadline.py | #628 chain; #1120 (open) | proven: [#628](invariants/receipts.md#i7-628) |
| I8 | **Tool pairs.** An emitted result follows its call occurrence; a missing result gets a stub, repeated IDs included. | engine.py (pair repair) | test_overflow_reserves_sanitizer_stubs.py | #1048 | proven: [#1048](invariants/receipts.md#i8-1048) |
| I9 | **Group boundaries.** The fresh tail and leaf cuts never split an assistant/tool group. | fresh_tail.py; compaction.py | test_fresh_tail.py; test_issue_559_tool_group_boundary.py | #559 | proven: [#559](invariants/receipts.md#i9-559) |
| I10 | **Projection codec.** A projection never grows a row. Emitted bytes identify their source on cold replay, and changed bytes never take over that identity. | survival_fit.py; reconcile.py; emission_projection.py | test_issue_916_projection_keeps_head.py; test_issue_959_projection_marks.py | #916, #959 | proven: [#916](invariants/receipts.md#i10-916), [#959](invariants/receipts.md#i10-959) |
| I11 | **Progress.** Compactions per 100 user messages stay at or below the previous GA's plus slack, and at or below 25 at 32K and at 1M windows. | compaction.py (holds, sweep) | tests/differential | #1139 | gap |
| I12 | **Child isolation.** A bypassed child compress leaves the parent's per-compress state unchanged. | compaction.py; bypass.py | none | #1157, #1222 (open) | gap |
| I13 | **Trial assembly publishes nothing:** no payloads, lineage, proof, emissions or recall. | engine.py (trial assembly) | test_issue_671_stub_first_exit.py | #671 | proven: [#671](invariants/receipts.md#i13-671) |
| I14 | **Atomic publication.** Node, provenance and frontier commit in one DAG transaction, with no stale frontier generation. Background compaction (#787) relies on it. | dag.py; lifecycle_state.py | test_issue_5_contiguous_frontier.py | #5 | proven: [#5](invariants/receipts.md#i14-5) |

## Recording a seed proof

1. Name the seed defect and its fix commit.
2. With the fix reverted, run the binding test and record that it fails, with the commit and command.
3. With the fix restored, record that it passes.
4. Set the row's *Seed proof* cell to `proven: <receipt>`, where the receipt is a PR/issue comment or a
   repository-owned receipt holding the fix SHA, commands, exact inverse patch, failing assertion, and both run counts.

The [seed receipts](invariants/receipts.md) pin their test source to rc8, not this documentation branch. A proven seed
checks sensitivity to that named past defect; it does not assert exhaustive coverage of the whole invariant.
The same document records why I4, I5, I6, I11, and I12 stay gaps, including I4's fixed #1221 seed and open #1225 residual.

A seed without a merged fix cannot supply a reversion proof. Its binding is a strict xfail that reproduces it.
A different, fixed seed may supply the row's proof, while the open defect remains explicit; issue closure alone is not
fix evidence. I4 and I12 remain gaps under the residual acceptance scope recorded in the receipts.

## Non-invariants

Some properties are deliberately best-effort and must not be asserted as invariants:
- Assembly does not always fit. Forced overflow recovery is best-effort and reports failure through
  `_last_overflow_recovery_failed` and a warning.
- Rollback to every older version with every feature on is not supported. Supported reader floors are listed in the I5 tests and
  in the doctor's rollback advice.
