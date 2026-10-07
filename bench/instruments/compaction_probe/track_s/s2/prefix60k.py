#!/usr/bin/env python3
"""S6 A3: freeze the NAMED 60k WORKLOAD row of a seed ONCE (r3 §R1, r4 F1 timing population (i)); no model calls.

Freeze row = the first material checkpoint at which an 8k-leaf sweep on the fleet config covers >= 60,000 tokens:
covered = tokens of the non-system rows OUTSIDE the fleet fresh tail (LCM_FRESH_TAIL_COUNT 24, _MAX_TOKENS 24,000),
the tail resolved by the engine's own `fresh_tail.resolve_fresh_tail_boundary`, tokens by its own counter. Refuses to
overwrite an existing file (the row never changes after the first run).  Usage: prefix60k.py <seed dir>
"""
import json, sys  # noqa: E401
from pathlib import Path

S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from s2lib import arms as A, seam  # noqa: E402

sdir = Path(sys.argv[1]).resolve()
out = sdir / "prefix60k.json"
if out.exists():
    raise SystemExit(f"{out} exists; the freeze row never changes after the first run")
m = seam.load_engine()
from hermes_lcm import fresh_tail  # noqa: E402  (registered by load_engine)
rows = [json.loads(x) for x in (sdir / "transcript.jsonl").read_text().splitlines() if x.strip()]
man = json.loads((sdir / "material.manifest.json").read_text())
tail_n, tail_tok = int(A.FLEET["LCM_FRESH_TAIL_COUNT"]), int(A.FLEET["LCM_FRESH_TAIL_MAX_TOKENS"])
walk = []
for cp in man["checkpoints"]:
    view = [{"role": r["role"], "content": r["content"], **({"tool_call_id": r["tool_call_id"]} if r["role"] == "tool" else {})}
            for r in rows[:cp["row_index"] + 1] if r["role"] != "system"]
    b = fresh_tail.resolve_fresh_tail_boundary(view, fresh_tail_count=tail_n, fresh_tail_max_tokens=tail_tok)
    covered = m.tokens.count_messages_tokens(view[:b.start])
    walk.append({"checkpoint": cp["id"], "row_index": cp["row_index"], "material_tokens": cp["tokens"],
                 "tail_rows": b.count, "tail_tokens": b.tokens, "sweep_covered_tokens": covered})
    if covered >= 60_000:
        break
else:
    raise SystemExit("no checkpoint reaches 60k covered tokens")
hit = walk[-1]
res = {"seed": man["seed"], "freeze_row_index": hit["row_index"], "freeze_row_id": rows[hit["row_index"]]["id"],
       "freeze_turn": rows[hit["row_index"]]["turn"], "checkpoint": hit["checkpoint"], "material_tokens": hit["material_tokens"],
       "sweep_covered_tokens": hit["sweep_covered_tokens"], "leaf_chunk_tokens": int(A.FLEET["LCM_LEAF_CHUNK_TOKENS"]),
       "how": "first manifest checkpoint whose non-system rows outside the fleet fresh tail (count 24, max 24,000 tokens; "
              "hermes_lcm.fresh_tail.resolve_fresh_tail_boundary at 9edfa46) total >= 60,000 tokens by the engine's "
              "count_messages_tokens (char estimate, the material's own tokenizer family); frozen before any comparison",
       "walk": walk, "material_sha": man["shas"]["transcript.jsonl"]}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "walk"}, indent=1))
