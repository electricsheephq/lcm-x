#!/usr/bin/env python3
"""H: pinned Hermes lean engine, shared S2 replay/probes, launched only via launch_h.sh."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("S2_H_SANDBOX") != "1" or os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") != "1" or not sys.dont_write_bytecode:
    raise SystemExit("use launch_h.sh (scratch home, lazy installs disabled, sandbox, -B)")
src = Path(os.environ["HERMES_SRC"]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reliability"))
import hosts  # noqa: E402
HOST = json.loads((src.parent / "HOST.json").read_text())
HOST_PROOF = hosts.verify("H", dict(src=str(src), sha=HOST["sha"]))
sys.path.insert(0, str(src))
from agent import context_compressor as CC  # noqa: E402
from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough  # noqa: E402
import run_s_lcmx as S2  # noqa: E402
from s2lib import seam, arms, reader  # noqa: E402


class Engine:
    def __init__(self, ctx, threshold=.75):
        self.real = CC.ContextCompressor(model=S2.HOST_MODEL, config_context_length=ctx, threshold_percent=threshold, quiet_mode=True)
        self.real.threshold_percent, self.real.threshold_tokens = threshold, int(ctx * threshold)
        self.threshold_tokens, self.system = self.real.threshold_tokens, None

    def ingest(self, view):
        pass

    def should_compress_preflight(self, view):
        return estimate_messages_tokens_rough(([self.system] if self.system else []) + view) >= self.threshold_tokens


class Run(S2.Run):
    def guard_state(self):
        real, now = self.engine.real, time.time()
        ineffective, fallback = getattr(real, "_ineffective_compression_count", 0), getattr(real, "_fallback_compression_streak", 0)
        deadline = getattr(real, "_anti_thrash_recovery_deadline", 0)
        until = getattr(real, "_structural_no_op_backoff_until", 0)
        tripped, backoff = ineffective >= 2 or fallback >= 2, until > time.monotonic()
        # Observe fields only: the host gate mutates/persists recovery state when called.
        blocked = tripped and (deadline <= 0 or deadline > now)
        return dict(ineffective_count=ineffective, fallback_streak=fallback, anti_thrash_tripped=tripped,
                    recovery_deadline=deadline, structural_backoff_until=until, structural_backoff_active=backoff,
                    would_block=blocked or backoff)

    def event(self, view, cur, i, row, forced):
        seam.take()
        guards = self.guard_state()
        t0, before = time.monotonic(), self.engine.real.compression_count
        new = self.engine.real.compress(copy.deepcopy([self.sysmsg] + view), current_tokens=cur, force=forced, bypass_cooldown=True)
        calls, _ = seam.take()
        text = "\n".join(str(m.get("content") or "") for m in new)
        new = [m for m in new if m.get("role") != "system"]
        def section_bytes(heading):
            return len((text.split(heading, 1)[1].split("\n## ", 1)[0] if heading in text else "").encode())
        self.events.append(dict(event=len(self.events), row_index=i, turn=row["turn"], is_compaction=self.engine.real.compression_count > before,
            threshold_tokens=self.engine.threshold_tokens, threshold_percent=self.engine.real.threshold_percent, tail_mode=self.engine.real.tail_mode,
            summariser_errors=[c.get("exception") or c.get("error") for c in calls if c.get("exception") or c.get("error")],
            skipped_compaction=self.engine.real.compression_count == before,
            guard_state_before=guards, guard_state_after=self.guard_state(),
            would_suppress_compaction=not forced and cur >= self.engine.threshold_tokens and guards["would_block"],
            tokens_at_trigger=cur, tokens_after=self.ntok([self.sysmsg] + new), compress_wall_s=time.monotonic() - t0,
            summariser_calls=calls, empty_no_room=None, timing_label="WIRING-ONLY" if forced else "full-stream",
            user_section_bytes=section_bytes(CC._LEAN_USER_MESSAGES_HEADING), identifier_index_bytes=section_bytes(CC._LEAN_ANCHOR_HEADING),
            continuity=S2.continuity(self.man, self.system, new)))
        return new

    def snapshot(self, cp, view):
        out = self.dir / f"cp-{cp['id']}"
        out.mkdir()
        self.snaps.append(dict(row=cp["row_index"], checkpoint=cp, dir=out, db=None, home=None,
                               view=copy.deepcopy(view), n_events=len(self.events)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--run", default="1")
    ap.add_argument("--checkpoints", default="lifecycle")
    ap.add_argument("--context-length", type=int, default=S2.DEFAULT_CTX)
    ap.add_argument("--threshold", type=float, default=.75)
    ap.add_argument("--lane", default="glm", choices=("glm", "codex-sol", "codex-astra", "fake"))
    ap.add_argument("--reader", default="glm", choices=("glm", "astra-low", "fake"))
    a = ap.parse_args()
    a.batches, a.slice, a.prefix60k = 0, None, False
    sdir = S2.MATERIAL / f"seed-{a.seed}"
    cps = S2.CP.select(sdir, a.checkpoints)
    man = S2.jload(sdir / "material.manifest.json")
    rows = S2.jlines(sdir / "transcript.jsonl")[:cps[-1]["row_index"] + 1]
    out = S2.RUNS / "H" / f"seed-{a.seed}" / a.run
    out.mkdir(parents=True, exist_ok=False)
    isolation = dict(host_tree=HOST_PROOF, write_denied=True, sandbox_profile=str(S2.TS / "h-isolation.sb"))
    for label, parent in (("outside_output", S2.S2), ("host_src", src), ("hermes_home", hosts.real_hermes_dir())):
        try:
            with (parent / ".eval2-write-denied-check").open("x"):
                pass
        except PermissionError:
            isolation[label + "_write_denied"] = True
        else:
            raise SystemExit(f"FAILED: sandbox write restriction absent for {label}")
    host = HOST
    arm = arms.resolve("H")
    if arm.get("worktree") and Path(arm["worktree"]).resolve() != src:
        raise SystemExit("Hermes arm worktree must match HERMES_SRC")
    if arm.get("sha", host["sha"]) != host["sha"]:
        raise SystemExit("Hermes host SHA mismatch")
    seam.set_lane(a.lane, out / "lane-scratch")
    CC.call_llm = seam.call_llm
    m = SimpleNamespace(tokens=SimpleNamespace(count_messages_tokens=estimate_messages_tokens_rough, count_tokens=estimate_tokens_rough))
    run = Run(m, arm, f"seed-{a.seed}", a, sdir, rows, man, out)
    run.engine = Engine(a.context_length, a.threshold)
    run.engine.system = run.sysmsg
    run.replay()
    if a.reader == "fake":
        from s2lib.fakes import Reader
        rd = Reader()
    else:
        rd = reader.GLMReader() if a.reader == "glm" else reader.AstraLowReader(out / "reader-scratch")
    summary = dict(context_length=a.context_length, threshold_tokens=run.engine.threshold_tokens,
                   threshold_percent=run.engine.real.threshold_percent, tail_mode=run.engine.real.tail_mode,
                   timeout_s=S2.replay_timeout(man["decision_checkpoint"]["tokens"]), arm=arm, worktree_head=host["sha"], worktree=str(src), isolation=isolation, status="DONE", seed=f"seed-{a.seed}",
                   reader_readback=rd.readback, material_sha256=S2.sha(sdir / "material.manifest.json"))
    for cp in run.snaps:
        run.reader_calls = []
        res = S2.probe(run, cp["view"], rd, False, cp)
        cp_status = "DONE" if rd.readback.get("pin_ok", True) else "FAILED"
        if cp_status == "FAILED":
            summary["status"] = "FAILED"
        (cp["dir"] / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
        (cp["dir"] / "summary.json").write_text(json.dumps(dict(summary, status=cp_status, checkpoint_id=cp["checkpoint"]["id"], checkpoint=cp["row"],
            stop_row_index=cp["row"], reader_calls=run.reader_calls, events=run.events[:cp["n_events"]],
            final_context=dict(tokens=run.ntok([run.sysmsg] + cp["view"])),
            final_continuity=S2.continuity(man, run.system, cp["view"]), receipts=[])))
    (out / "summary.json").write_text(json.dumps(dict(summary, events=run.events, checkpoints=[c["id"] for c in cps])))
    return 1 if summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
