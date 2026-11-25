# Compaction flow and token counters

LCM-X starts a threshold sweep when prompt pressure reaches the trigger. An
optional depth target can stop the leaf passes while older raw rows are still
in the view. The summary block has its own budget, and assembly and exit fit
apply their own limits.

```mermaid
flowchart TD
  subgraph WP["Whole prompt the model receives"]
    L1["1. Fixed load: system prompt, tool schemas, injected files (host-owned, re-sent on every request)"]
    L2["2. Summary block (LCM-X; summary-prefix budget: LCM_SUMMARY_PREFIX_TARGET_TOKENS when positive, else a share of the window when LCM_SUMMARY_PREFIX_TARGET_RATIO is set, else one leaf budget capped at a quarter of the trigger)"]
    L3["3. Raw rows and the protected fresh tail"]
  end
  ING["Turn end: ingest stores the new rows"] --> HG{"Host gate, counter A: host prompt figure >= trigger share of the window?"}
  ING --> PF{"Preflight, counter B: LCM-X count of the message list >= trigger, or a cleanup is due?"}
  HG -->|no| REQ["Request goes out unchanged"]
  PF -->|no| REQ
  HG -->|yes| CMP["compress()"]
  PF -->|yes| CMP
  CMP --> SF{"Stub-first exit, counter A' (list + fixed load): free cuts reach min(built-in target, depth target)?"}
  SF -->|yes| OUT
  SF -->|"no; free cuts staged when they reach the built-in target"| PC["Condense the summary block first if it is over its budget"]
  PC --> LEAF["Leaf pass: summarise the oldest chunk outside the fresh tail, or promote a prepared leaf"]
  LEAF --> STOP{"Stop? Counter E: host figure minus tokens freed so far"}
  STOP -->|"E plus reserves <= depth target"| CN["Condense the summary block to its budget"]
  STOP -->|"raw prefix drained"| CN
  STOP -->|"time or pass budget spent"| ASM
  STOP -->|otherwise| LEAF
  LEAF -->|"first leaf not stored, free cuts staged"| FB["Return the staged free stub-first cuts, as ratio 0 would; depth target unmet"]
  FB --> OUT
  CN --> ASM["Assembly: summary block, raw rows, fresh tail"]
  ASM --> FIT["Exit fit, counter A': window and exit cap only, never the depth target; drops only rows a summary covers"]
  FIT --> OUT["Return the list; the host commits; status reports target, fixed load, whole prompt after, met or not"]
```

| Counter | What it measures | Where |
| --- | --- | --- |
| A | Host-reported prompt size, including fixed load | `should_compress()` reads the host figure |
| B | Tokens in the message list | `should_compress_preflight()` calls `count_messages_tokens()` |
| A′ | The measured list plus fixed load | `_stub_first_exit()` and survival fit use `_survival_measure()` and `_survival_host_overhead()` |
| E | Pass estimate after the net savings from leaves and, with an active depth target, condensations | `_prepare_compression_sweep()` initializes it, `_admit_leaf_and_adopt_replay()` adjusts for a resumed prefix, and publication credits the savings |

The depth check adds the proactive recall budget when recall is enabled. It
also adds the observed fixed load when the host supplied no prompt figure for
this pass. Status measures the final list, including any recall block, and adds
fixed load.

A retry can stop as soon as it adopts a committed prefix that meets the depth
target. It then returns the adopted summary and skips post-leaf condensation.

While the depth target is active, the sweep scales positive token savings using
the host prompt figure minus the booked fixed load, divided by the LCM-X count
of the input list. The scale is clamped to 1–4 and defaults to 1 when the host
figure or list count is missing. Increases use the unscaled token count.

With an active depth target and the resolved foreground defaults (60 s soft,
120 s hard), a sweep that has stored a leaf or condensed node can admit
additional calls while its estimate stays above the depth target. A call the
base budget would refuse gets a timeout of 85% of the smaller of the configured
summary timeout and the time left to the catch-up bound (the start of the pass
plus 210 s, minus the greater of the finalization reserve and 20 s), and it is
admitted only when its estimate fits that timeout. Pre-leaf condensation keeps its leaf reservation and credits the frontier
saving before the first leaf check. Each stored post-leaf condensation group
refreshes the estimate and closes catch-up as soon as it reaches the target.
Forced, overflow and recovery passes, non-default foreground settings, and
passes with the ratio unset or zero use the base admission budget.
