"""Synthetic Track S fixtures: a tiny material seed and fake runs in the S1 / S2 / S4 writer formats."""
import json
from pathlib import Path

CONT = [("X-CONT-host", "system", "Keep the chronology."), ("X-CONT-request", "user", "Audit the replay."),
        ("X-CONT-constraint", "user", "Never modify the fixtures.")]
FACTS = [  # id, class, placement, value, stale
    ("X-F0", "name", "head", "alpha-workspace", None),
    ("X-F1", "path", "middle", "src/beta/settings.toml", None),
    ("X-F2", "superseded_value", "tail", "snapshot-gamma because cursors mixed", "mutable-gamma because it avoids writes"),
    ("X-F3", "file_change", "middle", "src/delta/loader.py: keep ids", None),
    ("X-F4", "name", "tail", "epsilon-workspace", None),
    ("X-F5", "decision", "head", "zeta because it is faster", None),
]
TRAPS = ["X-T0", "X-T1"]
BATCHES = [["X-F0", "X-F1", "X-F2", "X-T0"], ["X-F3", "X-F4", "X-F5", "X-T1"]]
STATE = {"id": "X-STATE", "next_action": "run the check", "status": "pending", "row_id": "R9", "row_index": 9,
         "row_role": "assistant"}
GOOD = {f[0]: f[3] for f in FACTS} | {t: "I don't know" for t in TRAPS}
GOOD_CONT = {"X-STATE.next_action": "run the check", "X-STATE.status": "pending"}


def wj(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def wl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def material(root: Path, mode="full") -> Path:
    d = root / "seed-x"
    wj(d / "facts.json", [{"id": i, "class": c, "placement": pl, "value": v, "answer": v, "stale": s, "probe": f"q {i}"}
                          for i, c, pl, v, s in FACTS])
    wj(d / "traps.json", [{"id": t, "answer": "ABSTAIN", "probe": f"q {t}"} for t in TRAPS])
    wj(d / "continuation.json", STATE)
    wl(d / "probe_batches.jsonl", [{"id": f"X-B{n}", "text": "Reply as JSON.", "probes": [
        {"id": p, "kind": "trap" if p in TRAPS else "canary", "text": f"q {p}"} for p in b]} for n, b in enumerate(BATCHES)])
    wj(d / "material.manifest.json", {"mode": mode, "continuity": [
        {"id": i, "row_role": r, "value": v} for i, r, v in CONT], "receipt_targets": [
        {"id": "X-F2", "kind": "ancestry", "required_events": 2}, {"id": "X-F3", "kind": "externalization"}]})
    return d


def batch_of(pid):
    return "X-BCONT" if pid.startswith("X-STATE") else f"X-B{next(n for n, b in enumerate(BATCHES) if pid in b)}"


def s2_event(wall, level=1, timing="full-stream", cont=(True, True, True), error=False, row_index=5, n=0):
    return {"event": n, "turn": 1 + n, "row_index": row_index, "trigger": "engine preflight gate", "status": "compacted",
            "is_compaction": True, "nodes": [{"node_id": n + 1, "depth": 0}], "tokens_at_trigger": 1000 * (n + 1),
            "timing_label": timing, "compress_wall_s": wall, "levels": [{"depth": 0, "level": level}],
            "summariser_calls": [{"latency_s": wall, "error": "timeout" if error else None}],
            "continuity": [{"id": c[0], "present": p} for c, p in zip(CONT, cont)]}


def s2_run(root: Path, arm="LCMX-a", run=1, answers=None, events=None, receipts=None, missing=(), assembled=None,
           tools=None, kind="plain") -> Path:
    d = root / f"{arm}-r{run}"
    answers = dict(GOOD | GOOD_CONT) if answers is None else answers
    rows = [{"schema": "s2-result-v1", "kind": "probe", "probe_id": pid, "arm": arm, "arm_kind": kind, "run": run,
             "probe_kind": "continuation_field" if pid.startswith("X-STATE") else "canary", "answer": a, "status": "OK",
             "timed_out": False, "batch_id": batch_of(pid), "tool_calls_batch": len(tools or []), "answer_wall_s_batch": 2.0}
            for pid, a in answers.items()]
    wl(d / "results.jsonl", rows)
    rec = {"X-F2": "PASS", "X-F3": "PRESENT"} if receipts is None else receipts
    wj(d / "summary.json", {"arm": {"name": arm, "kind": kind, "unsupported": None}, "status": "DONE", "stop_row_index": 10,
                            "events": [s2_event(30.0, n=i) for i in range(5)] if events is None else events,
                            "receipts": [{"id": k, "status": v} for k, v in rec.items()],
                            "admission": {"facts_not_admitted": list(missing)}, "final_context": {"tokens": 500},
                            "reader_readback": {"model": "glm-5.3"}})
    for b in {batch_of(p) for p in answers}:
        wj(d / "answers" / f"{b}.json", {"batch": b, "tool_log": [{"tool": t} for t in (tools or [])]})
    if assembled is not None:
        wj(d / "assembled_context.json", {"system": CONT[0][2], "view": [{"role": "user", "content": assembled}]})
    return d


def s1_run(root: Path, answers=None, open_rows=(), run=1) -> Path:
    """open_rows: [(probe_id, answer, tool_calls, recall_path)] for the lossless-claw-open arm."""
    d = root / f"lc-r{run}"
    answers = dict(GOOD) if answers is None else answers
    rows = [{"schema": "s1-result-v1", "kind": "probe", "probe_id": p, "arm": "lossless-claw", "arm_kind": "plain",
             "run": run, "answer": a, "status": "OK", "batch_id": batch_of(p), "answer_wall_s_batch": 5.0} for p, a in answers.items()]
    rows += [{"schema": "s1-result-v1", "kind": "probe", "probe_id": p, "arm": "lossless-claw-open", "arm_kind": "open",
              "run": run, "answer": a, "status": "OK", "tool_calls": n, "tokens_read_back": 10 * n, "answer_wall_s": 4.0,
              "recall_path": path, "attribution": "answer"} for p, a, n, path in open_rows]
    wl(d / "results.jsonl", rows)
    ok = {c[0]: True for c in CONT}
    wj(d / "summary.json", {"admission": {"facts_missing": []}, "receipts": [{"id": "X-F2", "status": "PASS"},
                                                                             {"id": "X-F3", "status": "PRESENT"}]})
    wj(d / "turns.json", [{"turn": 1, "start_ms": 0, "continuity": ok, "assembled_tokens": 100},
                          {"turn": 2, "start_ms": 20000, "continuity": ok, "assembled_tokens": 120}])
    wj(d / "events.json", {"forced": None, "summariser_calls": [{"t_start_ms": 1000, "latency_ms": 9000, "error": None}],
                           "levels": [], "db_events": [
        {"t_ms": 1000, "turn": 1, "tokens_at_observation": 900, "type": "pending_batch", "batch_id": "b1", "to": "planning"},
        {"t_ms": 11000, "turn": 1, "type": "pending_node", "batch_id": "b1", "to": "ready"},
        {"t_ms": 12000, "turn": 1, "type": "pending_batch", "batch_id": "b1", "to": "published"}]})
    return d


def s4_run(root: Path, answers=None, tool_calls=0, run=1, stall_ms=(50000,)) -> Path:
    d = root / f"codex-r{run}"
    answers = dict(GOOD) if answers is None else answers
    wl(d / "results.jsonl", [{"schema": "s4-result-v1", "kind": "probe", "probe_id": p, "arm": "codex-native", "run": run,
                              "arm_kind": "hosted, own runtime; pinned reader on fork", "reader": "gpt-6-astra·low",
                              "answer": a, "status": "OK", "batch_id": batch_of(p), "tool_calls_batch": tool_calls,
                              "timing_label": "full-stream"} for p, a in answers.items()])
    wj(d / "summary.json", {"status": "COMPLETED", "admission": {"facts_not_admitted": []},
                            "compactions": [{"window_number": i + 1, "stall_ms": s, "prev_token_count_input": 9000, "line_no": 1}
                                            for i, s in enumerate(stall_ms)],
                            "survival": [{"window_number": i + 1, "continuity_verbatim": {c[0]: True for c in CONT}}
                                         for i in range(len(stall_ms))]})
    wj(d / "rollout-parse.json", {"token_series": [{"last_input": 700}]})
    return d
