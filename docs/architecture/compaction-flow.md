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
  STOP -->|"E <= depth target"| CN["Condense the summary block to its budget"]
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
| E | Pass estimate after the net savings from each leaf | `_prepare_compression_sweep()` initializes it, `_admit_leaf_and_adopt_replay()` adjusts for a resumed prefix, and `_advance_after_leaf()` updates it after publication |

The depth check adds the conversation's observed fixed load to E when the host
has supplied no prompt figure for the current pass. Status reports the final
list measured with `_survival_measure()` plus fixed load, while the stop decision
uses the pass estimate or the stub-first check.
