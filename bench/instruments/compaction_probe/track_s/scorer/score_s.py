#!/usr/bin/env python3
"""Track S scorer: one arm run -> one score JSON (r3 §R3/§R5, r4 F1/F2/F5). Stdlib only; no model or network call.

score_s.py --material <seed dir> --run <arm run dir> --arm <name> --out <json>
Adapts to the three writers as they exist (S1 replay.ts `s1-result-v1`, S2 run_s_lcmx.py `s2-result-v1`, S4
run_s_codex.py `s4-result-v1`). A field the gate needs that the arm did not write is reported with its name; the
scorer never invents a value. Status: OK | INCOMPLETE(..) | UNRUN(..) | UNTESTED(..) | FAIL(..) | N/A(..).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

KIT = ("lcm-x bench/instruments/compaction_probe/score_probes.py @ 350958c "
       "(ABSTAIN_PATTERNS L23-39, normalize L43-50, _raw_answer L167-175, classify L178-213)")
# ---- copied from KIT, unchanged in behaviour ------------------------------------------------------------------
ABSTAIN_PATTERNS = [
    r"(?i)\b(don'?t|do not|didn'?t|did not|doesn'?t|does not|can'?t|cannot|"
    r"couldn'?t|could not)\s+(know|recall|find|locate|see|have|remember|"
    r"mention|state|specify|provide|say|share|define|give|record|include)\b",
    r"(?i)\b(isn'?t|is not|wasn'?t|was not|aren'?t|are not|weren'?t|were not|"
    r"not)\s+(present|mentioned|specified|stated|provided|included|available|"
    r"recorded|defined|given|said|shared|sure|in (the|our|my))\b",
    r"(?i)\bno (record|memory|information|mention|such)\b",
    r"(?i)\bno\b.{0,60}\b(was|were|has been|have been)\s+(provided|"
    r"established|decided|set|given|specified|mentioned|recorded|defined|"
    r"chosen)\b",
    r"(?i)\bnever (mentioned|stated|specified|provided|said|defined|recorded|"
    r"shared|gave|included|decided|chose|set|established|agreed|picked)\b",
    r"(?i)\bunable to (find|locate|recall|see)\b",
    r"(?i)\banswer is:?\s*unknown\b",
    r"(?i)\bunsure\b",
]
ABSTAIN_RE = tuple(re.compile(p) for p in ABSTAIN_PATTERNS)


def normalize(value) -> str:
    """KIT L43-50: casefold and remove punctuation/whitespace for substring matching."""
    text = str(value or "").casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def classify(answer, expected, trap: bool, *, timed_out=False, known_values=()) -> str:
    """KIT L167-213 (answer string in place of the result row): CORRECT | ABSTAIN | HALLUCINATE | TIMEOUT."""
    if timed_out:
        return "TIMEOUT"
    unparseable = not isinstance(answer, str) or not answer.strip()
    answer = "" if unparseable else answer.replace("’", "'").replace("‘", "'")
    norm = normalize(answer)
    concrete = any(n and n in norm for n in (normalize(v) for v in known_values))
    abstain = not unparseable and not concrete and any(p.search(answer) for p in ABSTAIN_RE)
    if trap:
        return "ABSTAIN" if abstain else "HALLUCINATE"
    if not unparseable and normalize(expected) and normalize(expected) in norm:
        return "CORRECT"
    return "ABSTAIN" if abstain else "HALLUCINATE"
# ---- end of KIT copy -------------------------------------------------------------------------------------------


RANK = ("OK", "N/A", "UNTESTED", "INCOMPLETE", "UNRUN", "FAIL")
LABELS = ("context", "search only", "expand")
EXPAND_TOOLS, CONT_META = {"lcm_expand_query", "lcm_expand"}, ("id", "row_id", "row_index", "row_role")  # other tools = search


def status(issues) -> str:
    """Highest-ranked issue kind wins; every reason of that kind is kept. `issues` = [(kind, reason)]."""
    if not issues:
        return "OK"
    top = max((k for k, _ in issues), key=RANK.index)
    return f"{top}({'; '.join(dict.fromkeys(r for k, r in issues if k == top))})"


def diag_norm(text: str) -> str:
    """DIAGNOSTIC continuity (S2 decision 4): runs of whitespace collapsed, trailing punctuation dropped."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    return text


def diag_present(value: str, haystack: str) -> bool:
    return diag_norm(value) in re.sub(r"\s+", " ", haystack or "")


def pct(xs, q):
    """Nearest-rank percentile (q in 0..100); None on an empty population."""
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)] if xs else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    return {"count": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "max": max(xs) if xs else None}


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cont_files(run_dir: Path) -> dict:
    """S6 A2: per-event continuity text written by the writers -> {event index: record}."""
    d = run_dir / "continuity"
    return {int(p.stem.split("-")[1]): jload(p) for p in d.glob("event-*.json")} if d.exists() else {}


def jlines(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


# ---- arm adapters: each returns the normalised run dict ----------------------------------------------------------
def row_of(r, label, calls=None, tokens=None, wall=None, attribution=None, flags=()):
    return {"id": r["probe_id"], "kind": r.get("probe_kind"), "answer": r.get("answer"), "timed_out": bool(r.get("timed_out")),
            "error": r.get("error"), "row_status": r.get("status"), "batch": r.get("batch_id"), "label": label,
            "calls": calls, "tokens_read_back": tokens, "wall_s": wall, "attribution": attribution, "flags": list(flags)}


def s1_label(r):
    p = r.get("recall_path") or ""
    if r.get("arm_kind") != "open" or not r.get("tool_calls"):
        return "context", ()
    return ("expand", ()) if p == "delegated" else ("expand", ("delegation did not run",)) if p.startswith(
        "expand_query called") else ("search only", ())


def adapt_s1(run_dir: Path, rows, arm):
    summ, ev, turns = (jload(run_dir / f) for f in ("summary.json", "events.json", "turns.json"))
    out_rows = []
    for r in rows:
        lab, fl = s1_label(r)
        op = r.get("arm_kind") == "open"
        out_rows.append(row_of(r, lab, r.get("tool_calls", 0) if op else 0, r.get("tokens_read_back") if op else None,
                               r.get("answer_wall_s", r.get("answer_wall_s_batch")), r.get("attribution"), fl))
    forced = ev.get("forced")
    fstart = forced["start_ms"] if forced else math.inf
    cf = cont_files(run_dir)
    texts = {b: f["text"] for f in cf.values() for b in f.get("batch_ids", [])}
    checks = [(t["start_ms"], t["continuity"]) for t in turns] + ([(fstart, forced["continuity_after"])] if forced else []) + (
        [(ev["checkpoint_continuity"]["t_ms"], ev["checkpoint_continuity"]["continuity"])] if ev.get("checkpoint_continuity") else [])
    batches = {}
    for e in ev.get("db_events", []):
        if e.get("batch_id"):
            batches.setdefault(e["batch_id"], []).append(e)
    events = []
    for bid, es in batches.items():
        first = min(es, key=lambda e: e["t_ms"])
        if first["t_ms"] >= fstart:
            continue
        ready = [e["t_ms"] for e in es if e["type"] == "pending_node" and e.get("to") == "ready"]
        pub = next((e["t_ms"] for e in es if e["type"] == "pending_batch" and e.get("to") == "published"), None)
        after = [c for t, c in checks if pub is not None and t > pub and t != fstart] if pub is not None and pub < fstart else []
        cont = after[0] if after else (forced["continuity_after"] if forced else turns[-1]["continuity"])
        st = (ev.get("store_timing") or {}).get(bid) or {}  # S7 D5: the store's ready_at, not the 500 ms poller
        end = max(ready) if ready else math.inf
        calls = [c for c in ev.get("summariser_calls", []) if first["t_ms"] - 1000 <= c["t_start_ms"] <= end]
        events.append({"label": f"deferred preparation {bid}" + ("" if pub is not None and pub < fstart else
                                                                  " (published by the forced suffix)" if pub else " (unpublished)"),
                       "turn": first.get("turn"), "tokens": first.get("tokens_at_observation"), "compaction": True,
                       "timing": "full-stream", "wall_s": st["wall_s"] if st.get("wall_s") is not None else
                       (max(ready) - first["t_ms"]) / 1000 if ready else None,
                       "summariser_s": [c["latency_ms"] / 1000 for c in calls],
                       "route_failed": any(c.get("error") for c in calls), "levels": [], "cont": cont, "text": texts.get(bid),
                       "publication_s": (pub - max(ready)) / 1000 if pub is not None and ready else None})
    for d in ev.get("direct_compactions", []):  # no prepared batch: the engine's legacy fallback compaction (its own log line)
        after = [c for t, c in checks if t > d["t_ms"] and t != fstart]
        calls = [c for c in ev.get("summariser_calls", []) if d["t_ms"] - d["duration_ms"] - 1000 <= c["t_start_ms"] <= d["t_ms"]]
        events.append({"label": f"direct compaction {d['id']} (legacy fallback, no prepared batch)", "turn": d.get("turn"), "tokens":
                       d.get("tokens_before"), "compaction": True, "timing": "full-stream", "wall_s": d["duration_ms"] / 1000, "summariser_s":
                       [c["latency_ms"] / 1000 for c in calls], "route_failed": any(c.get("error") for c in calls), "levels": [],
                       "cont": after[0] if after else turns[-1]["continuity"], "text": texts.get(d["id"])})
    if forced:
        res = forced.get("result") or {}
        events.append({"label": forced.get("label", "forced"), "turn": turns[-1]["turn"], "tokens": forced.get("tokens_at_trigger"),
                       "compaction": bool(res.get("compacted")), "timing": "WIRING-ONLY", "wall_s": forced["wall_ms"] / 1000,
                       "summariser_s": [], "route_failed": False, "levels": [], "cont": forced["continuity_after"]})
    fallbacks = sum(lv.get("level") == "fallback" for lv in ev.get("levels", []))
    return {"runtime": "lossless-claw", "writer": "s1-result-v1 (s1/replay.ts)", "rows": out_rows, "events": events,
            "population": summ.get("population", "full-stream"),
            "run_status": "DONE", "store_backed": True, "provenance": "pinned reader (engine-direct)",
            "reader": (rows[0].get("reader_readback") or {}).get("model") if rows else None,
            "admission_missing": summ["admission"].get("facts_missing", []),
            "admission_source": "summary.json admission.facts_missing (store + large_files, read at the checkpoint)",
            "receipts": {r["id"]: r["status"] for r in summ.get("receipts", [])},
            "tokens_kept": (forced or {}).get("assembled_tokens_after", turns[-1].get("assembled_tokens")),
            "levels_note": f"{len(ev.get('levels', []))} summaries; {fallbacks} fallback; normal vs aggressive not distinguished at the store",
            "wall_label": "ENGINE WALL (deferred preparation planning->ready; store created_at (1 s) -> ready_at when recorded, "
                          "else 500 ms poll; NOT host-visible wait)",
            "label_map": {"plain arm / open row with tool_calls=0": "context", "grep/describe only (DIAGNOSTIC)": "search only",
                          "delegated": "expand", "expand_query called, delegation did not run": "expand (INCOMPLETE: delegation did not run)"},
            "gaps": ([] if cf else ["continuity DIAGNOSTIC: assembled text per check not recorded (no continuity/event-*.json)"])
                    + ([] if any(r.get("probe_kind") == "continuation_field" for r in rows) else
                       ["continuation: replay.ts asks no continuation batch (no continuation_field rows)"])
                    + ["level-3: not an LCM-X runtime (store does not distinguish normal/aggressive)",
                     "admission is read at the checkpoint, not before the first compaction"]}


def adapt_s2(run_dir: Path, rows, arm):
    summ = jload(run_dir / "summary.json")
    tool_log = {}
    for p in (run_dir / "answers").glob("*.json") if (run_dir / "answers").exists() else []:
        a = jload(p)
        tool_log[a.get("batch")] = {t.get("tool") for t in a.get("tool_log") or []}
    out_rows = []
    for r in rows:
        tools = tool_log.get(r.get("batch_id"), set()) if r.get("arm_kind") == "open" else set()
        lab = "expand" if tools & EXPAND_TOOLS else "search only" if tools else "context"
        fl = ("tool attribution is per batch",) if r.get("arm_kind") == "open" else ()
        out_rows.append(row_of(r, lab, r.get("tool_calls_batch"), r.get("tokens_read_back_batch"),
                               r.get("answer_wall_s_batch"), r.get("attribution") or "batch", fl))
    ctx_p = run_dir / "assembled_context.json"
    final_text = None
    if ctx_p.exists():
        ctx = jload(ctx_p)
        final_text = (ctx.get("system") or "") + "\n" + "\n".join(str(m.get("content") or "") for m in ctx.get("view", []))
    events, cf = [], cont_files(run_dir)
    for e in summ.get("events", []):
        calls = e.get("summariser_calls", [])
        is_final = final_text is not None and e.get("row_index") == summ.get("stop_row_index")
        events.append({"label": f"event {e['event']} ({e.get('trigger')}, {e.get('status')})", "turn": e.get("turn"),
                       "row_index": e.get("row_index"), "tokens": e.get("tokens_at_trigger"),
                       "compaction": bool(e.get("is_compaction", bool(e.get("nodes")))), "timing": e.get("timing_label"),
                       "wall_s": e.get("compress_wall_s"), "summariser_s": [c.get("latency_s") for c in calls],
                       "route_failed": any(c.get("error") or c.get("exception") for c in calls) or bool(e.get("spend_guard_skips")),
                       "levels": [(n["depth"], n["level"], n.get("l3_kind")) for n in e.get("nodes", [])] if str(e.get(
                           "level_source")).startswith("stored") else [(lv.get("depth"), lv.get("level"), None) for lv in e.get("levels", [])],
                       "cont": {c["id"]: c["present"] for c in e.get("continuity", [])}, "final_text": final_text if is_final else None,
                       "text": (cf.get(e["event"]) or {}).get("text")})
    arm_cfg = summ.get("arm") or {}
    return {"runtime": "lcmx", "writer": "s2-result-v1 (s2/run_s_lcmx.py)", "rows": out_rows, "events": events,
            "population": summ.get("population", "full-stream"),
            "run_status": "UNRUN" if arm_cfg.get("unsupported") else summ.get("status", "missing"),
            "store_backed": arm_cfg.get("kind") != "control", "provenance": "pinned reader (engine-direct)",
            "reader": (summ.get("reader_readback") or {}).get("model"),
            "admission_missing": (summ.get("admission") or {}).get("facts_not_admitted", []),
            "admission_source": "summary.json admission.facts_not_admitted (store + externalized payloads, read at the checkpoint)",
            "receipts": {r["id"]: r["status"] for r in summ.get("receipts", [])},
            "tokens_kept": (summ.get("final_context") or {}).get("tokens"), "levels_note": "per summariser call (levels[].level)",
            "wall_label": "SUMMARISER WALL (compress() duration; NOT host-visible wait)",
            "label_map": {"plain arm / open batch with empty tool_log": "context",
                          "open batch tool_log within {lcm_grep, lcm_describe, lcm_recall}": "search only",
                          "open batch tool_log has lcm_expand_query or lcm_expand": "expand (batch attribution)"},
            "gaps": ([] if cf else ["continuity DIAGNOSTIC: only the final assembled context is stored (assembled_context.json); "
                                    "earlier events carry strict booleans only"]) + [
                     "recall calls/tokens/wall: batch-attributed (tool_calls_batch), not per answer",
                     "admission is read at the checkpoint, not before the first compaction"]}


def adapt_s4(run_dir: Path, rows, arm):
    summ = jload(run_dir / "summary.json")
    rp = jload(run_dir / "rollout-parse.json") if (run_dir / "rollout-parse.json").exists() else {}
    out_rows = [row_of(r, "context" if not r.get("tool_calls_batch") else "search only", r.get("tool_calls_batch"), None,
                       r.get("answer_wall_s_batch"), r.get("attribution"),
                       () if not r.get("tool_calls_batch") else ("context-only rule violated (tool items in fork)",)) for r in rows]
    timing = "WIRING-ONLY" if any(r.get("timing_label") == "WIRING-ONLY" for r in rows) else "full-stream"
    surv, cf = {s.get("window_number"): s for s in summ.get("survival", [])}, cont_files(run_dir)
    events = [{"label": f"native compaction window {c.get('window_number')} (rollout line {c.get('line_no')})",
               "turn": (surv.get(c.get("window_number")) or {}).get("turn"),  # S7 D4 (None on runs before the writer change)
               "tokens": c.get("prev_token_count_input"), "compaction": True, "timing": timing,
               "wall_s": c["stall_ms"] / 1000 if c.get("stall_ms") is not None else None, "summariser_s": [],
               "route_failed": False, "levels": [], "cont": (surv.get(c.get("window_number")) or {}).get("continuity_verbatim", {}),
               "text": (cf.get(i) or {}).get("text")} for i, c in enumerate(summ.get("compactions", []))]
    series = rp.get("token_series") or [{}]
    return {"runtime": "codex-native", "writer": "s4-result-v1 (s4/run_s_codex.py)", "rows": out_rows, "events": events,
            "run_status": summ.get("status", "missing"), "store_backed": False,
            "provenance": (rows[0].get("arm_kind") if rows else "hosted, own runtime"),
            "reader": (rows[0].get("reader") if rows else summ.get("model")),
            "admission_missing": (summ.get("admission") or {}).get("facts_not_admitted", []),
            "admission_source": "summary.json admission.facts_not_admitted (rollout, per row and role)",
            "receipts": {}, "tokens_kept": series[-1].get("last_input"),
            "levels_note": "n/a (server-side compaction, summary encrypted)",
            "wall_label": "NATIVE EVENT WALL (rollout ContextCompaction start->end; labelled reference)",
            "label_map": {"tool_calls_batch=0": "context",
                          "tool_calls_batch>0": "search only (INCOMPLETE: r4 N2 context-only rule violated)"},
            "gaps": ["event turn index: not recorded (rollout line number only)",
                     "continuity STRICT: replacement-history plain text only (survival[].continuity_verbatim)"
                     + ("; DIAGNOSTIC = continuity/event-<i>.json (+ AGENTS.md)" if cf else "; DIAGNOSTIC text not stored"),
                     "tokens kept: last rollout input-token observation, not a post-compaction count",
                     "event timing label: run-level (results rows timing_label), not per event"]}


def detect(run_dir: Path, rows):
    schema = (rows[0].get("schema") if rows else "") or ""
    if schema.startswith("s1") or (run_dir / "events.json").exists():
        return adapt_s1
    if schema.startswith("s4") or (run_dir / "rollout-parse.json").exists():
        return adapt_s4
    return adapt_s2


# ---- metrics ---------------------------------------------------------------------------------------------------
def metric(value, issues, **extra):
    return {"value": value, "status": status(issues), "issues": [f"{k}: {r}" for k, r in dict.fromkeys(issues)],
            "complete": not any(k in ("INCOMPLETE", "UNRUN", "UNTESTED") for k, _ in issues), **extra}


def score(material: Path, run_dir: Path, arm: str) -> dict:
    facts = jload(material / "facts.json")
    traps = jload(material / "traps.json")
    cont_state = jload(material / "continuation.json")
    batches = jlines(material / "probe_batches.jsonl")
    man = jload(material / "material.manifest.json")
    all_rows = jlines(run_dir / "results.jsonl")
    rows = [r for r in all_rows if r.get("arm") == arm and r.get("kind", "probe") == "probe"]
    run = detect(run_dir, all_rows)(run_dir, rows, arm) if (run_dir / "summary.json").exists() else {
        "runtime": "unknown", "writer": None, "rows": [], "events": [], "run_status": "UNRUN", "store_backed": False,
        "admission_missing": [], "receipts": {}, "gaps": ["summary.json missing"], "label_map": {}}
    capped = {a.get("batch") for p in (run_dir / "answers").glob("*.json") if
              ((a := jload(p)).get("reader_calls") or [a.get("usage")])[-1] and
              ((a.get("reader_calls") or [a.get("usage")])[-1] or {}).get("completion_tokens", 0) >= 8192}
    # only unanswered probes of a capped batch are excluded; an admission-proven loss stays a loss
    truncated = {r["id"] for r in run["rows"] if not (r["answer"] or "").strip() and
                 ((r["row_status"] == "ERROR" and "READER_TRUNCATED" in (r["error"] or "")) or r["batch"] in capped)}
    truncated -= set(run["admission_missing"])
    base = [("INCOMPLETE", f"READER_TRUNCATED: {len(truncated)} probes excluded: {sorted(truncated)}")] if truncated else []
    if run["run_status"] not in ("DONE", "COMPLETED"):
        base.append(("UNRUN", f"run status {run['run_status']}"))
    if not rows:
        base.append(("UNRUN", f"no result rows for arm {arm}"))
    by_id, dups = {}, set()
    for r in run["rows"]:
        if r["id"] in by_id:
            dups.add(r["id"])  # KIT L325-328: a duplicated probe id scores as an unparseable miss
        else:
            by_id[r["id"]] = r
    if any(r["row_status"] == "UNAVAILABLE" for r in run["rows"]):
        base.append(("UNRUN", "UNAVAILABLE rows (context does not fit the reader)"))
    known = [f["value"] for f in facts]
    batch_of = {p["id"]: b["id"] for b in batches for p in b["probes"]}
    unasked_batches = sorted({batch_of[i] for i in batch_of if i not in by_id})
    lost = sorted(set(run["admission_missing"]) & {f["id"] for f in facts})
    probes = {pid: {"class": "READER_TRUNCATED", "batch": by_id[pid]["batch"], "answer": by_id[pid]["answer"]} for pid in truncated}
    truncated_facts = {f["id"] for f in facts} & truncated
    facts = [f for f in facts if f["id"] not in truncated]
    traps = [t for t in traps if t["id"] not in truncated]
    cont_state = {k: v for k, v in cont_state.items() if f"{cont_state['id']}.{k}" not in truncated}

    def judge(pid, expected, trap):
        r = by_id.get(pid)
        if r is None or pid in dups:
            cls = "MISSING" if r is None else "HALLUCINATE"
        else:
            cls = classify(r["answer"], expected, trap, timed_out=r["timed_out"], known_values=known)
        probes[pid] = {"batch": batch_of.get(pid), "class": cls, "answer": (r or {}).get("answer"), "label": (r or {}).get("label")}
        return cls, r

    missing = lambda ids: [i for i in ids if i not in by_id]  # noqa: E731
    # facts kept (60 scheduled facts; missing / unparsable / timed out = misses; not-admitted = misses, listed)
    grid, correct = {}, 0
    for f in facts:
        cls, _ = judge(f["id"], f["value"], False)
        ok = cls == "CORRECT" and f["id"] not in lost
        correct += ok
        g = grid.setdefault(f"{f['class']}|{f['placement']}", [0, 0])
        g[0], g[1] = g[0] + ok, g[1] + 1
    miss = missing(f["id"] for f in facts)
    iss = base + ([("INCOMPLETE", f"{len(miss)} of {len(facts)} scheduled facts have no result row (batches not run: "
                    f"{', '.join(unasked_batches)})")] if miss else [])
    m = {"facts_kept": metric(correct / len(facts) if facts else None, iss, correct=correct, reader_truncated=len(truncated_facts), denominator=len(facts), by_class_placement=grid,
                              lost_before_compaction={"ids": lost, "source": run.get("admission_source")})}
    # stale-value rate (lower is better): the superseded value asserted, even alongside the current one
    sup = [f for f in facts if f.get("stale")]
    stale_ids, head_ids = [], []
    for f in sup:
        a = normalize((by_id.get(f["id"]) or {}).get("answer"))
        if a and normalize(f["stale"]) in a:
            stale_ids.append(f["id"])
        if a and normalize(f["stale"].split(" because ")[0]) in a:
            head_ids.append(f["id"])
    ms = missing(f["id"] for f in sup)
    m["stale_rate"] = metric(len(stale_ids) / len(sup) if sup else None,
                             base + ([("INCOMPLETE", f"no row for {ms}")] if ms else []) + ([] if sup else [("INCOMPLETE", "no superseded facts")]),
                             lower_is_better=True, stale_ids=stale_ids, denominator=len(sup),
                             diagnostic_value_token_only_ids=head_ids)
    # traps
    abst = [t["id"] for t in traps if judge(t["id"], None, True)[0] == "ABSTAIN"]
    mt = missing(t["id"] for t in traps)
    ti = base + ([("INCOMPLETE", f"no row for {mt}")] if mt else [])
    m["trap_abstention"] = metric(len(abst) / len(traps) if traps else None, ti, abstained=abst, denominator=len(traps))
    m["trap_failure_rate"] = metric(1 - len(abst) / len(traps) if traps else None, ti, lower_is_better=True)
    # continuation, field by field
    fields = [k for k in cont_state if k not in CONT_META]
    ok_f, mc = [], []
    for k in fields:
        pid, r = f"{cont_state['id']}.{k}", by_id.get(f"{cont_state['id']}.{k}")
        if r is None:
            mc.append(pid)
        elif normalize(cont_state[k]) and normalize(cont_state[k]) in normalize(r["answer"]):
            ok_f.append(k)
        probes[pid] = {"batch": (r or {}).get("batch"), "class": "CORRECT" if k in ok_f else "MISSING" if r is None else "MISS",
                       "answer": (r or {}).get("answer")}
    m["continuation"] = metric(len(ok_f) / len(fields) if fields else None, base + ([("INCOMPLETE", f"no continuation_field row for {mc}")] if mc else []),
                               correct_fields=ok_f, denominator=len(fields))
    m["continuity"] = continuity(run, man)
    m["level3"] = level3(run)
    m["latency"] = latency(run)
    m["recall"] = recall(run, man, facts, by_id, probes, base, lost)
    comp = [e for e in run["events"] if e["compaction"]]
    # S7 D1/D7 checkpoint labels: no-event = 0 compactions by this checkpoint; assembly-starved = the
    # reader request carried 0 summary blocks while the store held >= 1 summary (counted by the writer in the request it built; field summary_blocks).
    blocks = sorted({r.get("summary_blocks") for r in all_rows if r.get("arm") == arm and r.get("summary_blocks") is not None})
    summ = jload(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    stored = summ.get("store_summaries", len(comp))  # summaries in the store at the checkpoint (writer field)
    labels = ["no-event"] * (not comp) + ["assembly-starved"] * bool(stored and blocks == [0])
    calls = [c for b in summ.get("reader_calls") or [] for c in b.get("calls") or [] if c]
    usage = {"calls": len(calls), "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
             "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls), "source": "summary.json reader_calls (per call)"}
    return {"schema": "score-s-v1", "checkpoint_row": summ.get("checkpoint", {}).get("row") if isinstance(
                summ.get("checkpoint"), dict) else summ.get("checkpoint"), "labels": labels, "summary_blocks": blocks,
            "reader_usage": usage if calls else None, "arm": arm, "seed": material.name, "run": (rows[0].get("run") if rows else None),
            "run_id": rows[0].get("run_id") if rows else None, "material": str(material), "run_dir": str(run_dir),
            "runtime": run["runtime"], "writer": run["writer"], "reader": run.get("reader"), "provenance": run.get("provenance"),
            "population": run.get("population", "full-stream"),
            "store_backed": run["store_backed"], "smoke": man.get("mode") == "smoke", "kit_rule": KIT, "metrics": m,
            "behaviour": {"compactions": len(comp),
                          "events": [{k: e.get(k) for k in ("label", "turn", "tokens", "wall_s", "timing", "compaction", "publication_s")}
                                     for e in run["events"]],
                          "wall": stats([e["wall_s"] for e in comp]), "wall_label": run.get("wall_label"),
                          "summariser_wall": stats([s for e in comp for s in e["summariser_s"]]),
                          "levels": run.get("levels_note"), "tokens_kept_at_checkpoint": run.get("tokens_kept")},
            "label_map": run["label_map"], "gaps": run["gaps"], "batches_not_run": unasked_batches,
            "duplicate_rows": sorted(dups), "probes": probes}


def continuity(run, man):
    items = [c["id"] for c in man.get("continuity", [])]
    values = {c["id"]: c["value"] for c in man.get("continuity", [])}
    grid, iss = [], []
    for i, e in enumerate(run["events"]):
        for it in items:
            strict, txt = e["cont"].get(it), e.get("text")
            diag = diag_present(values[it], txt) if txt is not None else True if strict else (
                diag_present(values[it], e["final_text"]) if e.get("final_text") else None)
            grid.append({"event": i, "label": e["label"], "timing": e["timing"], "item": it, "strict": strict, "diagnostic": diag})
            if strict is False:
                iss.append(("FAIL", f"strict miss: {it} after {e['label']}"))
            elif strict is None:
                iss.append(("INCOMPLETE", f"no continuity field for {it} after {e['label']}"))
    if not run["events"]:
        iss.append(("INCOMPLETE", "no recorded events"))
    out = metric(sum(bool(g["strict"]) for g in grid) / len(grid) if grid else None, iss,
                 rule="ABSOLUTE: one strict miss = FAIL", grid=grid)
    out["complete"] = bool(grid) and all(g["strict"] is not None for g in grid)
    return out


def level3(run):
    if run["runtime"] != "lcmx":
        return {"value": None, "status": "N/A(level-3 incidence is measured on LCM-X arms only)", "complete": False}
    leaves = [(e, lv, k) for e in run["events"] if e["compaction"] for d, lv, *k in e["levels"] if d == 0 and lv is not None]
    if not leaves:
        return metric(None, [("INCOMPLETE", "zero leaves written")], leaves=0, lower_is_better=True)
    l3 = [(e, lv) for e, lv, k in leaves if lv == 3 and (k or [None])[0] != "verbatim"]  # S8: verbatim level 3 exempt (#652)
    answered = [e["label"] for e, _ in l3 if not e["route_failed"]]
    return metric(100 * len(l3) / len(leaves), [("FAIL", f"level-3 leaf with an answering route: {answered}")] if answered else [],
                  leaves=len(leaves), level3_leaves=len(l3), lower_is_better=True,
                  level3_verbatim=sum(lv == 3 and (k or [None])[0] == "verbatim" for _, lv, k in leaves),
                  route_rule="route answered = no summariser call in that event raised or timed out and no spend-guard skip")


def latency(run):
    comp = [e for e in run["events"] if e["compaction"]]
    if run.get("population") == "prefix60k":  # r4 F1 (i): its own row; every call and event of the run, actual counts
        w = stats([e["wall_s"] for e in comp])
        return metric(w["p90"], [("FAIL", "prefix60k event > 120 s")] * bool((w["max"] or 0) > 120) + [("INCOMPLETE", "no event")] * (not comp),
                      population="prefix60k", lower_is_better=True, events=w, calls=stats([s for e in comp for s in e["summariser_s"]]),
                      wall_label=run.get("wall_label"), timing_label="WIRING-ONLY (smoke)")
    gate = [e for e in comp if e["timing"] != "WIRING-ONLY"]
    iss = []
    walls = [e["wall_s"] for e in gate]
    if any(w is None for w in walls):
        iss.append(("INCOMPLETE", "event wall unmeasured on " + ", ".join(e["label"] for e in gate if e["wall_s"] is None)))
    over = [e["label"] for e in gate if e["wall_s"] is not None and e["wall_s"] > 120]
    if over:
        iss.append(("FAIL", f"event > 120 s: {over}"))
    if len(gate) < 5:
        iss.append(("INCOMPLETE", f"{len(gate)} gate events < 5 ({len(comp) - len(gate)} WIRING-ONLY excluded)"))
    st = stats(walls)
    return metric(st["p90"], iss, lower_is_better=True, gate_events=st, all_events=stats([e["wall_s"] for e in comp]),
                  p90_le_60=(st["p90"] <= 60) if st["p90"] is not None else None, wall_label=run.get("wall_label"),
                  summariser_wall=stats([s for e in gate for s in e["summariser_s"]]),
                  host_visible_wait="not measured here (engine-direct arms have no host turn; Visible-wait row is NOT IN TRACK S)")


def recall(run, man, facts, by_id, probes, base, lost):
    tags = {}
    for f in facts:
        if f["placement"] == "middle":
            tags.setdefault(f["id"], []).append("clipped-middle")
    for t in man.get("receipt_targets", []):
        tags.setdefault(t["id"], []).append({"externalization": "externalized-file", "ancestry": "superseded-decision"}[t["kind"]])
    tags = {pid: tg for pid, tg in tags.items() if probes.get(pid, {}).get("class") != "READER_TRUNCATED"}
    iss, per_label, per_answer, n_ok = list(base), {k: [0, 0] for k in LABELS}, [], 0
    for pid, tg in sorted(tags.items()):
        r, p = by_id.get(pid), probes.get(pid, {})
        ok = p.get("class") == "CORRECT" and pid not in lost
        n_ok += ok
        rc = run["receipts"].get(pid)
        untested = run["store_backed"] and rc is not None and rc not in ("PASS", "PRESENT")
        if r is None:
            iss.append(("INCOMPLETE", f"no row for {pid}"))
        else:
            per_label[r["label"]][0] += ok
            per_label[r["label"]][1] += 1
            per_answer.append({"id": pid, "tags": tg, "correct": ok, "label": r["label"], "calls": r["calls"],
                               "tokens_read_back": r["tokens_read_back"], "wall_s": r["wall_s"], "attribution": r["attribution"],
                               "receipt": rc, "flags": r["flags"]})
            for fl in r["flags"]:
                if fl.startswith(("delegation did not run", "context-only")):
                    iss.append(("INCOMPLETE", f"{pid}: {fl}"))
        if untested:
            iss.append(("UNTESTED", f"{pid}: receipt {rc}"))
        elif r is not None and not ok and ("clipped-middle" in tg or "externalized-file" in tg):
            iss.append(("FAIL", f"{'externalized-file' if 'externalized-file' in tg else 'clipped-middle'} fact unreachable: {pid}"))
    return metric(n_ok / len(tags) if tags else None, iss, correct=n_ok, denominator=len(tags), by_label=per_label,
                  per_answer=per_answer, label_map=run["label_map"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--material", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    out = score(a.material, a.run, a.arm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{a.arm} {out['seed']} run {out['run']}: " + ", ".join(f"{k}={v['status']}" for k, v in out["metrics"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
