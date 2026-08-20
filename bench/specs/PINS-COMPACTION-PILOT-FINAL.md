# PINS-FINAL DRAFT (mechanically derived 2026-08-20T14:15:16Z; values from commands, per feedback_mechanical_derivation_integrity_artifacts)

## Material (seed 186697847, regenerated post-#274)
f45942d9b5878fca24820b51ac20496ae9d22e6cef41a11176d381bc094745ac  /Users/m1/Codex/session-notes/2026-08-20/compaction-pilot/material/canaries.json
a508387376f8245e62e5b2091f0837d6b903095e705c695784e525b049d03029  /Users/m1/Codex/session-notes/2026-08-20/compaction-pilot/material/material.manifest.json
fbdd6a4275b24956634210ecf3825d98ae2b3884a9d27c67ddd1449626c5aaaf  /Users/m1/Codex/session-notes/2026-08-20/compaction-pilot/material/probes.jsonl
a9c910532f25c9bc18a37e9f398eebde8a7d4704993012710d5a9916e46eb235  /Users/m1/Codex/session-notes/2026-08-20/compaction-pilot/material/turns.jsonl

## Codex binary (R1-lc)
b0308517b20543012fa2171aa3d46ce455a7456c4eb2a552ab9468ba4eeb1e50  /Users/m1/.codex/packages/standalone/releases/0.148.0-aarch64-apple-darwin/bin/codex

## Engine tree (hermes arms plugin symlink target)
91d5706f8fcd02944250ea8b0e899a21d158dcbc

## Scoring/driver tree at official scoring
77b62ddec5f3b44ae3f1b790fde20180cd8f0a79

## Per-arm config shas + session identity (from run manifests)
R2-A: config_sha256=5267db99c6fcf74a engine=lcm model=gpt-5.6-sol
R2-Aprime: config_sha256=5267db99c6fcf74a engine=lcm model=gpt-5.6-sol
R2b: config_sha256=01f0909307be802a engine=compressor model=gpt-5.6-sol
R2s-A: config_sha256=650b94a28c60b0eb engine=lcm model=gpt-5.6-sol

## R1-lc rollout
/Users/m1/.codex/sessions/2026/08/20/rollout-2026-08-20T20-55-59-01a01f75-0913-7350-8291-c9e8f70e0fa8.jsonl
6fe264341dfbecb8c6334657a60e6a94b9961b8a5ed90dc37016acdc284ef8dc  /Users/m1/.cod

## S3 numbers (registered)
context_length=272000 threshold_tokens=136000 effective_cap=NONE (372K latent, #263)
R1 observed: model_context_window=258400 advertised; last-request input 553914; 0 compaction events

## Per-arm engine-tree receipts (from run console echoes — the ground truth)
R2-A / R2-Aprime / R2b / R2s-A / R2s-Aprime: tree=91d5706f (ALL six banked arms — one engine identity)
R3 att.1 (INVALID) + att.2 (killed ~10min, engine-pin drift caught): tree=ae3af5af — wt-lcmx-main had
drifted to post-train main between 14:27Z and 15:06Z (mover unidentified; low priority). Reset to
91d5706f before R3 att.3 (the counting attempt). SOUL.md sha (Am.12 protocol): echoed per run console.

## Amendment index (run-sheet §§9-12)
9: measured one-shot mechanism (R2r relabel, R2s parked→ACP spike, order pre-registration, abstain-regex
   FREEZE, arm-level crash rule) · 10: tool-access contamination audit + fixture purge · 11: R1-lc
   relabel (no compaction; paginated), m3, quota event + forced order deviation · 12: ACP tool policy
   correction (behavioral SOUL + audit gate), R3 att.1 invalidation.

## Cross-driver scoring tests (Am.11 obligation — all in tests_compaction_probe_b.py)
- gen_material/oneshot row shape: test_score_probes_gen_material_frozen_schema
- drive_codex row shape (no kind field): test_score_probes_accepts_drive_codex_result_rows
- drive_hermes_acp emits the oneshot row contract (driver tests assert row shape; smokes verified live).

## R4-ref status
VOID attempt 2026-08-20 (OAuth CLI quota exhausted by R1-lc; reset Aug 26 15:01). Runs in the
Aug 26–31 window; reference-only (excluded from contrasts by §3b), appended when banked.
