# RUN SHEET — LoCoMo declared-config A/A′ + scored read (Track B)
Status: PRE-STAGED 2026-07-30 (~23:35 local). LAUNCH GATE: paired V2 run PAIR-DONE
(no heavy local runs before the machine frees; Track A gate read executes first).

## Config of record (every knob measured or disclosed; F46 §4/§9 lineage)
| Knob | Value | Justification |
|---|---|---|
| Harness | memorybench feat/locomo-hermes-prep @ **19d9c0b** + quota merge **f55eba3** | PR #3 (captions caption-only, truthful truncation, judge rubric, adversarial gold, pins) + PR #5 (quota mode) |
| Product | hermes-lcm fork main @ the #197 merge commit | includes LCM_CHUNK_MIN_CONVERSATIONAL_TOKENS knob; NOT the frozen-gate arms (LoCoMo lane owns its checkout) |
| `HERMES_MB_FUSION` | `quota:fts=1,chunk=2` | FUSION-EMBEDDER-DIAGNOSIS selected policy: 15/24 buried gold @top-25, controls 4/4, ~100-param plateau |
| `LCM_CHUNK_MIN_CONVERSATIONAL_TOKENS` | `10` | the 12 excluded substantive gold turns measure 11–30 cl100k tokens (CHUNK-ELIGIBILITY-DIAGNOSIS); 10 admits all measured gold, keeps the acknowledgment filter (vs 0 = max bloat, vs `full` policy = tool/system bloat) |
| `HERMES_MB_ANSWER_READY_CONTENT_CHARS` | `2400` | pinned + exported (PR #3 round 4) |
| Answerer | `codex gpt-5.6-sol @ medium` | frontier model per two-tier doctrine; same transport class as arm A for noise-floor comparability |
| Judge | `sol @ low`, prompts @ pinned sha (defaults.ts 7662f6…) | narrowed rubric (premise-rejection only); STRICTER than stock LoCoMo judge |
| Transport | codex CLI pinned (re-verify binary sha at prep; amendment protocol if drifted) | 6e.13 |
| Stores | REBUILT from scratch under this config | ingestion changed (captions, threshold): a store built under other knobs is a different corpus |

## Sequence
1. **Prep** (machine free): fresh worktree checkouts at the pinned shas; `bun install`;
   update `data/pins-locomo.yaml`: harness re-pin (run-prep protocol per git_sha_note),
   product sha, env block gains `HERMES_MB_FUSION` + `LCM_CHUNK_MIN_CONVERSATIONAL_TOKENS`;
   `pinverify.py pre-run` must PASS.
2. **A/A′**: two identical full runs (1,986q), fresh stores each, detached (nohup),
   artifacts → session-notes/<date>/hermes-locomo-declared/artifacts/{a,a-prime}/.
   Adjudicate agreement stats (discordant count = the noise floor) BEFORE any scored claim.
3. **Scored read**: the A arm IS the scored candidate if A/A′ agreement is healthy;
   report with the seven-point disclosure (F46 §5) + this sheet as the config appendix.
   Scoreboard row appended (supersedes locomo10-1986-arm-a-2026-07-30 row via
   `superseded_by`; the 47% row STAYS visible — that is the standard).
4. **No projections** until measured (§6e.8). The F46 ledger's ~10 residual missing
   turns + adversarial attribution weakness are the expected fault-finding surfaces.

## Abort/fail-close rules
Pin mismatch at any arm start = fail-closed abort (fix, re-pin explicitly, log
amendment). Malformed HERMES_MB_FUSION raises by construction (PR #5). Mid-run
session death: resume with same run id (checkpoint protocol, 2026-07-30 precedent).
