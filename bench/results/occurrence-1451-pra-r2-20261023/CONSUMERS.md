# In-memory proof consumers (r2)

A failed publication retains input/output identity, carry ranges, emissions, cursor authority, and end_consumed; keeps output_metadata in memory; and sets durable=False. This is process-local positional authority. No serializer adds the local flag to a wire record. Only successful atomic publication restores durability. A retry with no snapshot writes JSON null for the snapshot key under the same transaction, so no stale occurrence evidence attaches.

| Reader / writer | Kind | Behaviour when durable=False |
|---|---|---|
| compaction.py:770 absence check and _bind_survival_fit_emission | positional | Proof remains present, so no replacement binding; existing in-memory emissions survive. |
| compaction.py:775 descriptor count | positional | Counts local emission descriptors; no durability claim. |
| compaction.py:_record_compress_commit_proof (960–1064) | positional / construction | Creates a new candidate; fresh metadata is prepared before publication. Construction failure can still clear an unconstructed proof. Publication failure preserves the constructed one. |
| compaction.py:_carry_prior_emissions / _bind_survival_fit_emission | positional | Local descriptors still bind generated spans; durable fallback uses the guarded durable reader. |
| compaction.py:_persist_compress_commit_proof | needs durability | Retries publication, never serializes durable=False; sets True only after the atomic writer returns. On failure retains the local snapshot and logs once after rollback. |
| compaction.py:_write_compress_commit_proof_payload | needs durability | Writes inside the engine-owned atomic transaction; does not report a commit independently. |
| engine.py:1207,1491–1500 / FrontierState.commit_proof | positional storage | Getter/setter retains the complete positional proof and flag; it is not persistence. |
| engine.py:4185,4196–4197 in-place boundary | positional | Republish when required, reread the same positional proof, preserve end_consumed/cursor checks even if publication fails. |
| engine.py:4506,4570 rotation | positional | Keep carry and rekey proof/scope/input to child. A failed child publication leaves the gate armed. A successful retry commits proof plus unavailable snapshot atomically. |
| engine.py:4684–4687 auxiliary copy | positional | Deep-copy process-local proof; preserve the flag, never claim publication. |
| engine.py:5282–5299 session end | positional | #483 exact-input one-shot skip stays active; mark end_consumed; do not re-ingest the whole session. |
| engine.py:_prepare_compress_commit_proof | needs durability (candidate construction) | Fresh snapshot construction; publication failure preserves stamps as process-local positional evidence. |
| engine.py:_write_compress_commit_proof_payload | needs durability | Validate only when metadata is present and durable is not False. While durable=False, atomically publish proof and JSON null snapshot on retry; no stale/invalid stamps become durable. |
| engine.py:_durable_commit_proof_payload / occurrence attach | needs durability | Return None for a same-session local durable=False proof, even if a prior durable key remains. No occurrence snapshot attaches. |
| engine.py:6823–6880 and 6896 ingest gate | positional | Compare exact output/input or remap host merge, then consume proof normally. Retained in-memory stamps witness a fresh occurrence. A malformed non-list, wrong-length or non-dict container is locally unavailable before direct/remapped ingest; the original process-local proof is preserved. |
| reconcile.py:608 _proof_output_effective_digests | positional | Use local effective identities for merged-remainder authority; durable fallback uses guarded reader. |
| frontier_state.py reset/filter/transfer/mark_end_consumed | positional | Normal lifecycle storage/reset/copy operations retain or consume the same flag; no serialization. |
| Test and fixture hits (consumer-enumeration.txt) | positional or explicit durability assertions | Test-only seeded historical fixtures remain historical. New tests assert rollback, unavailable durable reader, retry, exact carry IDs and counts. |

## Old-writer grammar

v0.28.1 77b2b4f795fb6a4f84c55abf6d0441da4469971e engine.py:6490–6527 and BASE10407e12 engine.py:6643–6674 zip the entire output into output_metadata; the non-null identity sequence is output_effective. v0.28.1 and BASE emission_projection.py:103 record output_occurrence.of=len(messages) for every final descriptor. reconcile.py:665–702 projects an emitted empty remainder to null, but an unresolved carrier can retain full identity. It also projects descriptor-less system scaffolds and preserved todo rows to null at their actual positions. Therefore null positions cannot equal descriptor positions. Legacy validation checks field types and the ordered non-null digest sequence; full scaffold digest count or all descriptor `of` counts provide output length. Without either length witness, old wire bytes cannot distinguish added null scaffold rows from real host scaffolds; the effective sequence and field grammar remain authoritative, as for the old writer. No guessed scaffold-position exception is used.

## output_metadata reader walk (r2)

- engine.py:6641 prepares the local snapshot; 6650-6660 is the guarded atomic durable writer.
- engine.py:6664-6719 validates/attaches only digest-matched durable snapshots; the same-session durable=False guard is unchanged.
- engine.py:6756-6786 remaps snapshots. Its positional call at 6893 follows the ingest container guard at 6827-6831; calls at 6982 and reconcile.py:3295 consume only the guarded durable reader.
- engine.py:6868/6873 are positional occurrence checks and may use local stamps after failed publication.
- reconcile.py:3298/3309 are durability-dependent UID membership/length checks on the guarded durable payload, never directly on the local failed proof.
- externalize.py and ingest_protection.py hits name tool-output metadata, not occurrence-proof metadata; unchanged. Test/harness hits are fixture construction/assertions.

No DDL, wire-format or digest-twin change. Snapshot JSON null and durable proof publish together on retry. A subsequent independently requested successful publication may validate the retained local snapshot afresh; this round does not broaden that existing behavior.
