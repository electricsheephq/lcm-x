"""S1 round 2 (decision 1): put every recorded event of a run on the KIT token axis, offline.

kit_tokens_fed = the kit's own estimate (gen_material.token_counter = the checkout's count_tokens, offline char
fallback; the counter S3 used for the material) summed over the rows fed to the engine at that moment, system row
included. Printed beside the engine's own estimate. Rows fed per event: `fed_rows` when the run recorded it (runs after
round 2); otherwise derived from turns.json (start_ms + ingest_ms): exact after a turn's ingest, a range during it.
Usage: kit_positions.py <run dir> <material dir>   (rewrites <run dir>/summary.json with a `kit_axis` block)
"""
import importlib.util
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "gen_material.py"
run, mat = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("s3_gen", KIT)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
count = gen.token_counter()
rows = [json.loads(line) for line in (mat / "transcript.jsonl").read_text().splitlines() if line.strip()]
turns = json.loads((run / "turns.json").read_text())
events = json.loads((run / "events.json").read_text())
summary = json.loads((run / "summary.json").read_text())
cum = [0]
for r in rows:
    cum.append(cum[-1] + count(r["content"]))


def position(turn, t_ms, fed=None):
    if fed is not None:
        return {"fed_rows": fed, "kit_tokens_fed": cum[fed], "fed_rows_source": "recorded"}
    i = next(k for k, t in enumerate(turns) if t["turn"] == turn)
    t, before = turns[i], (turns[i - 1]["last_row"] + 1 if i else 1)  # row 0 = system row, fed to the slot at start
    after = t["last_row"] + 1
    if t_ms >= t["start_ms"] + t["ingest_ms"]:
        return {"fed_rows": after, "kit_tokens_fed": cum[after], "fed_rows_source": "derived: after the turn's ingest"}
    if t_ms < t["start_ms"]:
        return {"fed_rows": before, "kit_tokens_fed": cum[before], "fed_rows_source": "derived: before the turn"}
    return {"fed_rows": [before, after], "kit_tokens_fed": [cum[before], cum[after]], "fed_rows_source": "derived: during ingest (range)"}


out_events = [{"type": e["type"], "to": e.get("to"), "t_ms": e["t_ms"], "turn": e["turn"],
               "engine_tokens_last_afterTurn": e.get("tokens_at_observation"), **position(e["turn"], e["t_ms"], e.get("fed_rows"))}
              for e in events["db_events"] if e["type"] != "context_items"]
calls = [{"t_start_ms": c["t_start_ms"], "turn": c["turn"], "latency_ms": c["latency_ms"], **position(c["turn"], c["t_start_ms"])}
         for c in events["summariser_calls"]]
f = events.get("forced")
forced = f and {"after_row": f["after_row"], "engine_tokens_at_trigger": f["tokens_at_trigger"],
                **position(None, 0, f.get("fed_rows", f["after_row"] + 1)),
                "fed_rows_source": "recorded" if "fed_rows" in f else "derived: after_row + 1"}
per_turn = [{"turn": t["turn"], "fed_rows": t["last_row"] + 1, "kit_tokens_fed": cum[t["last_row"] + 1],
             "engine_assembled_tokens": t["assembled_tokens"], "ratio_engine_over_kit": round(t["assembled_tokens"] / cum[t["last_row"] + 1], 3)}
            for t in turns]
summary["kit_axis"] = {
    "tokenizer": "kit gen_material.token_counter (repo count_tokens, offline char fallback) - estimate, not model tokens",
    "note": "engine tokens = lossless-claw's own assembled estimate at the last afterTurn (post-compaction context once a "
            "publication lands); kit tokens = raw rows fed so far. The two measure different things after a publication.",
    "engine_threshold_tokens": summary["config"]["threshold_tokens"], "per_turn": per_turn, "events": out_events,
    "summariser_calls": calls, "forced": forced}
(run / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary["kit_axis"].items() if k != "note"}, indent=1))
