# Track D prep — V1-medium (longmemeval_m) dataset verified + pinned (2026-07-31)

| Item | Value |
|---|---|
| Source | HF dataset `xiaowu0162/longmemeval`, file `longmemeval_m` |
| Revision | `2ec2a557f339b6c0369619b1ed5793734cc87533` (same pin as the banked `longmemeval_s` lineage) |
| Local path | `/Volumes/LEXAR/hermes-work/longmemeval-data/longmemeval_m` |
| sha256 | `fb5413e3b077c62927daab794836991a2fcfa61ceacab57dc679fb02daaff2d9` |
| Size | 2,745,274,681 bytes = 9.9× the small tier (matches the 10× haystack design) |
| Shape check | 500 `question_id` / 500 `question_type` (streaming grep — no full JSON load; the file is too large for json.load on this box) |

Remaining Track D steps (in order, machine-gated where noted):
1. Store build under storefreeze discipline (M17-class build rules) — MACHINE-GATED:
   after the paired V2 gate AND the LoCoMo A/A′ (build contends: embedding CPU).
2. pins.yaml for the medium lane (dataset sha above + product sha + transport + env).
3. Power memo via pairedgate BEFORE any registered run (instrument 4 refuses
   underpowered gates by design).
4. Registered baseline run with a FRONTIER reader (Tier-P: full seven-point
   disclosure; transport pinned per 6e.13). V2-medium go/no-go only AFTER this lands.
