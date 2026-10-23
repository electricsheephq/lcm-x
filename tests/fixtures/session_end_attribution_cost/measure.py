"""Adapt #1880's alternating paired harness; run on a remote CI worker only.

Scopes: full warm preflight and the post_llm_call ingest span, excluding hook
routing, note_turn_complete, leaf preparation and all host/model work.
"""
import io
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
import tests.conftest  # noqa: E402, F401 (register the plugin with CI's host stub)
from hermes_lcm.compaction import CompactionMixin  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

base = "440976dc4ca05caa8e82ccc8e7adcb167fa70bd9"
source = subprocess.check_output(["git", "show", base + ":compaction.py"], text=True)
module = types.ModuleType("hermes_lcm.compaction_baseline")
module.__package__ = "hermes_lcm"
sys.modules[module.__name__] = module
exec(compile(source, "baseline-compaction.py", "exec"), module.__dict__)
classes = {"baseline": module.CompactionMixin, "candidate": CompactionMixin}
out.mkdir(parents=True, exist_ok=False)
scratch = Path(os.environ["RUNNER_TEMP"]) / "session-end-cost"
scratch.mkdir()
config = LCMConfig(database_path=str(scratch / "cost.db"), embeddings_enabled=False,
                   temporal_rollups_enabled=False)
engine = LCMEngine(config=config, hermes_home=str(scratch / "home"))
engine.on_session_start("cost", conversation_id="cost", context_length=200_000)
view = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Synthetic row {i}",
         "timestamp": 1760000000.0 + i} for i in range(24)]
engine.ingest(view)
summary = []
log = logging.getLogger("hermes_lcm.engine")
sink = io.StringIO()
handler = logging.StreamHandler(sink)
log.handlers, log.propagate = [handler], False
with (out / "pairs.jsonl").open("w") as raw:
    for level in (logging.INFO, logging.DEBUG):
        log.setLevel(level)
        for scope in ("preflight", "post_llm_call_ingest"):
            values = {key: [] for key in classes}
            for i in range(210):
                row = dict(level=logging.getLevelName(level), scope=scope, pair=i - 10,
                           order=["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"])
                for label in row["order"]:
                    engine._should_compress_preflight_impl = classes[label]._should_compress_preflight_impl.__get__(engine)
                    started = time.perf_counter_ns()
                    if scope == "preflight":
                        result = engine.should_compress_preflight(view)
                        assert result is False
                    elif label == "candidate":
                        with engine._attribution_span("post_llm_call", len(view)):
                            engine.ingest(view)
                    else:
                        engine.ingest(view)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    row[label + "_ms"] = elapsed
                    if i >= 10:
                        values[label].append(elapsed)
                if i >= 10:
                    raw.write(json.dumps(row) + "\n")
            a, b = [statistics.median(values[key]) for key in ("baseline", "candidate")]
            delta = statistics.median([b - a for a, b in zip(values["baseline"], values["candidate"])])
            limit = max(a * .1, .05)
            result = dict(scope=scope, level=logging.getLevelName(level), n=200,
                          baseline_median_ms=a, candidate_median_ms=b, paired_median_delta_ms=delta,
                          median_delta_ms=b-a, allowed_increase_ms=limit,
                          baseline_p10_ms=sorted(values["baseline"])[19], baseline_p90_ms=sorted(values["baseline"])[179],
                          candidate_p10_ms=sorted(values["candidate"])[19], candidate_p90_ms=sorted(values["candidate"])[179],
                          passed=b-a <= limit and delta <= limit)
            summary.append(result)
            print(json.dumps(result), flush=True)
assert engine._store.get_session_count("cost") == len(view)
engine.shutdown()
(out / "summary.json").write_text(json.dumps(dict(base=base, rows=summary), indent=2) + "\n")
assert all(row["passed"] for row in summary)
