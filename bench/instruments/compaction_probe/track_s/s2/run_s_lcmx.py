#!/usr/bin/env python3
"""Track S unit S2: drive the REAL LCM-X engine (worktree @ 9edfa46) through a Track S material stream, per arm.

Feeds the transcript as the host would (ingest the visible list, then compress() only when the engine's own
preflight gate says so), records every compaction event + continuity probes, the store-backed receipts, and answers
the probe batches with a pinned reader, each batch from a fresh CLONE of the compacted store. See README.md.
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, logging, os, re, shutil, sqlite3, sys, tempfile, time  # noqa: E401
from pathlib import Path

S2 = Path(__file__).resolve().parent
TS = Path(os.environ["TRACK_S_OUT"]).resolve()
MATERIAL = Path(os.environ["TRACK_S_MATERIAL"]).resolve()
RUNS = TS / "lcmx-runs"
DEFAULT_CTX = 272_000  # the engine is told this window; threshold = the arm's LCM_CONTEXT_THRESHOLD x window
HOST_MODEL = "eval-s2-host"
sys.path.insert(0, str(S2))
from s2lib import arms as A, reader as R, seam  # noqa: E402

SUMMARY_RE = re.compile(r"\[[^\[\]\n]*Summary \(d\d+, node \d+\)\]")  # engine.py:5189/7394 summary part header (S7 D7)
CFG_SKIP = {"config_sources", "config_source_warnings", "ignored_config_yaml_lcm_keys"}

jload = lambda p: json.loads(p.read_text())  # noqa: E731
jlines = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None  # noqa: E731
db_ro = lambda db: sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # noqa: E731

class SpendSkips(logging.Handler):  # S8: spend-guard skips (no model call) per compress() = route not answered
    n = 0
    def emit(self, r): self.n += "spend guard active" in r.getMessage()  # noqa: E704
SKIPS = SpendSkips()

def apply_env(env: dict, home: Path, db: Path):
    """The arm's config reaches the engine only through its own `LCM_*` env inputs; storage stays inside the run."""
    for k in [k for k in os.environ if k.startswith("LCM_")]:
        del os.environ[k]
    os.environ.update(env)
    os.environ.update({"LCM_DATABASE_PATH": str(db), "HERMES_HOME": str(home), "LCM_HERMES_BASE_DIR": str(home.parent)})
    home.mkdir(parents=True, exist_ok=True)

def config_dict(cfg) -> dict:
    return json.loads(json.dumps({f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                                  if f.name not in CFG_SKIP}, default=str))

def new_engine(m, home: Path, sid: str, ctx: int):
    e = m.engine.LCMEngine(config=m.config.LCMConfig.from_env(), hermes_home=str(home))
    e.on_session_start(sid, platform="eval", conversation_id=sid, model=HOST_MODEL, provider="eval", context_length=ctx)
    e.update_model(model=HOST_MODEL, context_length=ctx, provider="eval")
    return e

def select_rows(sdir: Path, checkpoint: str, slice_n: int | None):
    rows, man = jlines(sdir / "transcript.jsonl"), jload(sdir / "material.manifest.json")
    if checkpoint == "auto":
        stop = man["decision_checkpoint"]["row_index"]
    else:
        cps = [c for c in man["checkpoints"] if c["tokens"] >= int(checkpoint)]
        if not cps:
            raise SystemExit(f"no checkpoint >= {checkpoint} tokens in the manifest")
        stop = cps[0]["row_index"]
    if slice_n is not None:
        stop = min(stop, max(i for i, r in enumerate(rows) if r["turn"] <= slice_n))
    return rows[:stop + 1], man, stop

def to_msg(row: dict) -> dict:
    msg = {"role": row["role"], "content": row["content"]}
    if row["role"] == "tool":
        msg["tool_call_id"] = row["tool_call_id"]
    if row["role"] == "assistant" and row.get("tool_calls"):
        msg["tool_calls"] = row["tool_calls"]
    return msg

def continuity(man: dict, system: str, view: list[dict]) -> list[dict]:
    text = R.flatten("", view)
    return [{"id": c["id"], "present": c["value"] in system or c["value"] in text, "in_view": c["value"] in text,
             "in_system_slot": c["value"] in system} for c in man["continuity"]]

class Run:
    def __init__(self, m, arm, seed, args, sdir, rows, man, run_dir):
        self.m, self.arm, self.seed, self.args, self.sdir, self.rows, self.man, self.dir = \
            m, arm, seed, args, sdir, rows, man, run_dir
        self.home, self.db, self.sid = run_dir / "hermes-home", run_dir / "lcm.db", f"s2-{arm['name']}-{seed}-r{args.run}"
        self.system = "\n".join(r["content"] for r in rows if r["role"] == "system")
        self.sysmsg = {"role": "system", "content": self.system}
        self.events, self.node_event, self.receipts_out, self.gates = [], {}, [], 0
        self.cps, self.snaps, self.reader_calls = set(args.checkpoints or ()), [], []  # S7 D1
        self.ntok = m.tokens.count_messages_tokens
        self.run_id = f"{arm['name']}/{seed}/r{args.run}/{int(time.time())}"
        self.timing_label = "WIRING-ONLY" if (man.get("mode") == "smoke" or args.slice or args.prefix60k) else "decision"
        self.population = "prefix60k" if args.prefix60k else "full-stream"

    def replay(self):
        suffix = self.man.get("smoke_suffix") or {}
        force_row = suffix.get("row_id") if suffix.get("action") == "force_compaction_after_row" else None
        view = []
        for i, row in enumerate(self.rows):
            if row["role"] == "system":
                continue  # host instructions: the assembled context's system slot, never an ingested row
            view.append(to_msg(row))
            nxt = self.rows[i + 1] if i + 1 < len(self.rows) else None
            if nxt is None or nxt["role"] == "assistant" or i in self.cps:  # a model call next (a probe at a checkpoint)
                self.engine.ingest(view)
                self.gates += 1
                if nxt is None and self.args.prefix60k:  # S6 A3: trigger lowered to this (freeze) gate's view tokens
                    self.p60 = {"freeze_row_index": i, "fleet_threshold_tokens": self.engine.threshold_tokens,
                                "lowered_threshold_tokens": self.ntok(view), "how": "engine.threshold_tokens set at the freeze gate"}
                    self.engine.threshold_tokens = self.p60["lowered_threshold_tokens"]
                if self.engine.should_compress_preflight(view):
                    view = self.event(view, self.ntok([self.sysmsg] + view), i, row, forced=False)
            if force_row and row["id"] == force_row:
                self.engine.ingest(view)
                view = self.event(view, self.ntok([self.sysmsg] + view), i, row, forced=True)
            if i in self.cps:  # S7 D1: store snapshot + receipts/admission as of this row; probes run after the replay
                self.receipts_out = []
                adm = self.receipts(upto=i)
                db, home = clone_store(self.db, self.home, self.dir / f"cp-{i}" / "store")
                self.snaps.append({"row": i, "dir": self.dir / f"cp-{i}", "db": db, "home": home, "view": list(view),
                                   "admission": adm, "receipts": list(self.receipts_out), "n_events": len(self.events)})
        return view

    def event(self, view, cur, i, row, forced):
        with db_ro(self.db) as c:
            before = c.execute("SELECT COALESCE(MAX(node_id), 0) FROM summary_nodes").fetchone()[0]
        seam.take()
        SKIPS.n = 0
        t_start, m0 = time.time(), time.monotonic()
        new = self.engine.compress(view, current_tokens=cur, force=forced)
        wall = round(time.monotonic() - m0, 3)
        calls, sums = seam.take()
        cols = ("node_id", "depth", "source_type", "source_ids", "token_count", "source_token_count")
        with db_ro(self.db) as c:
            nodes = [dict(zip(cols + ("level", "summary"), r)) for r in c.execute(  # S8: level = the stored provenance
                f"SELECT {', '.join('n.' + x for x in cols)}, p.escalation_level, n.summary FROM summary_nodes n LEFT JOIN "
                "summary_node_provenance p ON p.node_id = n.node_id WHERE n.node_id > ? AND n.session_id = ? ORDER BY n.node_id",
                (before, self.sid))]
        n, hist = len(self.events), {}
        for nd in nodes:
            self.node_event[nd["node_id"]] = n
            nd["n_sources"] = len(json.loads(nd.pop("source_ids") or "[]"))
            vh = {x["sha"] for x in sums if x.get("verbatim")}  # S8: verbatim level 3 = summary == serialized source
            nd["l3_kind"] = None if nd["level"] != 3 else "verbatim" if hashlib.sha256(
                nd.pop("summary").encode()).hexdigest() in vh else "truncated"
            nd.pop("summary", None)
        for s in sums:
            hist[f"depth{s['depth']}:L{s['level']}"] = hist.get(f"depth{s['depth']}:L{s['level']}", 0) + 1
        ev = {"event": n, "turn": row["turn"], "row_index": i, "row_id": row["id"],
              "trigger": "forced (smoke suffix force_compaction_after_row)" if forced else "engine preflight gate",
              "is_compaction": bool(nodes),  # False = the engine's cleanup-only pass (stub/externalize), no summary
              "timing_label": "WIRING-ONLY" if forced else self.population, "population": self.population,
              "tokens_at_trigger": cur,
              "threshold_tokens": self.engine.threshold_tokens, "rows_before": len(view), "rows_after": len(new),
              "tokens_after": self.ntok([self.sysmsg] + new), "status": self.engine.last_compression_status,
              "noop_reason": self.engine.last_compression_noop_reason, "nodes": nodes, "levels": sums,
              "leaves": sum(x["depth"] == 0 for x in nodes), "condensations": sum(x["depth"] > 0 for x in nodes),
              "level_histogram": hist, "level_source": "stored nodes: summary_node_provenance.escalation_level (S8); levels[] = returns",
              "spend_guard_skips": SKIPS.n,
              "compress_wall_s": wall, "compress_wall_label": "SUMMARISER WALL (compress() duration; NOT host-visible wait)",
              "t_start": t_start, "t_end": t_start + wall, "summariser_calls": calls,
              "continuity": continuity(self.man, self.system, new)}
        self.events.append(ev)
        cdir = self.dir / "continuity"  # S6 A2: the assembled context the next model call receives after this event
        cdir.mkdir(exist_ok=True)
        (cdir / f"event-{n}.json").write_text(json.dumps({"event": n, "row_index": i, "source": "system slot + view "
            "returned by compress()", "text": self.system + "\n" + "\n".join(
                c if isinstance(c := m.get("content"), str) else json.dumps(c) for m in new)}))
        print(f"  event {n}: row {i} turn {row['turn']} tokens {cur} -> {ev['tokens_after']} leaves {ev['leaves']} "
              f"cond {ev['condensations']} wall {wall}s status {ev['status']}", flush=True)
        return new

    def receipts(self, upto=None):
        """Externalization + ancestry receipts and admission, read from the live store (rows 0..upto fed)."""
        rows = self.rows[: None if upto is None else upto + 1]
        payloads = []
        for p in sorted((self.home / "lcm-large-outputs").glob("*.json")):
            try:
                payloads.append((p.name, jload(p)))
            except (OSError, json.JSONDecodeError):
                continue
        with db_ro(self.db) as c:
            msgs = c.execute("SELECT store_id, role, content FROM messages WHERE session_id = ?", (self.sid,)).fetchall()
            nodes = {r[0]: (r[1], r[2], set(json.loads(r[3] or "[]"))) for r in c.execute(
                "SELECT node_id, depth, source_type, source_ids FROM summary_nodes WHERE session_id = ?", (self.sid,))}
        fed = {r["id"]: i for i, r in enumerate(rows)}
        for t in self.man.get("receipt_targets", []):
            rec = {"id": t["id"], "kind": t["kind"], "row_id": t["row_id"]}
            if t["row_id"] not in fed:
                rec.update(status="UNTESTED", reason="target row not fed before the stop point")
            elif t["kind"] == "externalization":
                call_id = rows[fed[t["row_id"]]]["tool_call_id"]
                hit = [(n, p) for n, p in payloads if p.get("tool_call_id") == call_id]
                rec.update(tool_call_id=call_id, status="PRESENT" if hit else "UNTESTED",
                           reason=None if hit else "no externalize record for the target's tool_call_id",
                           refs=[{"ref": n, "content_chars": p.get("content_chars"),
                                  "fact_tag_in_payload": f"[{t['id']}]" in (p.get("content") or "")} for n, p in hit])
            else:
                sids = {sid for sid, _r, content in msgs if f"[{t['id']}]" in (content or "")}
                chain, front = [], {n for n, (_d, st, src) in nodes.items() if st == "messages" and src & sids}
                while front:
                    chain += sorted(front)
                    front = {n for n, (_d, st, src) in nodes.items() if st == "nodes" and src & front}
                evs = sorted({self.node_event[n] for n in chain if n in self.node_event})
                after = [x for x in evs if self.events[x]["row_index"] >= fed[t["row_id"]]]
                need = t.get("required_events", 2)
                rec.update(store_ids=sorted(sids), chain=[{"node_id": n, "depth": nodes[n][0]} for n in chain],
                           events=evs, events_after_supersession=len(after), required_events=need,
                           status="PASS" if len(after) >= need else "UNTESTED",
                           reason=None if len(after) >= need else f"{len(after)} event(s) after supersession < {need}")
            self.receipts_out.append(rec)
        blob = "\n".join(x[2] or "" for x in msgs) + "\n".join(p.get("content") or "" for _n, p in payloads)
        planted = [f for f in jload(self.sdir / "facts.json") if f["row_id"] in fed]
        fed_rows = [r for r in rows if r["role"] != "system"]
        by_role = lambda seq: {k: sum(1 for x in seq if x == k) for k in ("user", "assistant", "tool")}  # noqa: E731
        return {"rows_fed_by_role": by_role([r["role"] for r in fed_rows]),
                "rows_stored_by_role": by_role([x[1] for x in msgs]), "tokens_fed": self.ntok(
                    [to_msg(r) for r in fed_rows]), "externalized_payloads": len(payloads),
                "facts_planted_in_slice": len(planted), "facts_admitted": sum(f["value"] in blob for f in planted),
                "facts_not_admitted": [f["id"] for f in planted if f["value"] not in blob]}

def clone_store(src_db: Path, src_home: Path, dst: Path):
    (dst / "hermes-home").mkdir(parents=True)
    with db_ro(src_db) as s, sqlite3.connect(dst / "lcm.db") as d:
        s.backup(d)
    if (src_home / "lcm-large-outputs").exists():
        shutil.copytree(src_home / "lcm-large-outputs", dst / "hermes-home" / "lcm-large-outputs")
    return dst / "lcm.db", dst / "hermes-home"

def batches_for(sdir: Path):
    out, cont = jlines(sdir / "probe_batches.jsonl"), jload(sdir / "continuation.json")
    out.append({"id": out[0]["id"].rsplit("-", 1)[0] + "-BCONT", "text": out[0]["text"],
                "added_by": "S2 (continuation.json, scored field by field)", "probes": [
                    {"id": f"{cont['id']}.{k}", "kind": "continuation_field", "expect": "value", "gold": cont[k],
                     "text": f"For the pending mid-task continuation, what is its `{k}`?"}
                    for k in cont if k not in ("id", "row_id", "row_index", "row_role")]})
    return out

def probe(run: Run, view: list[dict], reader, is_store: bool, src=None) -> list[dict]:
    src_db, src_home, out = (src["db"], src["home"], src["dir"]) if src else (run.db, run.home, run.dir)
    blocks = sum(len(SUMMARY_RE.findall(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))))
                 for m in view)  # S7 D7: summary blocks in the reader request (the view it is sent)
    facts = {f["id"]: f for f in jload(run.sdir / "facts.json")}
    traps = {t["id"]: t for t in jload(run.sdir / "traps.json")}
    receipts = {r["id"]: r for r in run.receipts_out}
    is_open = run.arm["open"]
    schemas = [s for s in run.m.engine.LCMEngine.get_tool_schemas(run.engine) if s.get("name") in R.PUBLIC_TOOLS] \
        if is_open else None
    guidance = run.m.guidance.get_recall_policy() if is_open else ""
    ctx_tokens = run.ntok([run.sysmsg] + view)
    unavailable = run.arm["name"] == "C0" and ctx_tokens + R.ANSWER_RESERVE > R.READER_WINDOW[run.args.reader]
    (out / "answers").mkdir(exist_ok=True)
    results = []
    for b in batches_for(run.sdir)[: run.args.batches or None]:
        prompt = b["text"] + "\n" + "\n".join(f"{p['id']}: {p['text']}" for p in b["probes"])
        clone_id = clone_info = answers = err = tools_engine = None
        meta, attempts = {}, []
        if unavailable:
            err = f"UNAVAILABLE: C0 context {ctx_tokens} tokens + reserve > {run.args.reader} window"
        else:
            try:
                if is_store:  # a fresh clone per batch (db backup + payload dir), opened as an engine
                    clone_id = f"clones/{b['id']}"
                    cdb, chome = clone_store(src_db, src_home, out / clone_id)
                    apply_env(run.arm["env"], chome, cdb)
                    tools_engine = new_engine(run.m, chome, run.sid, run.args.context_length)
                    st = tools_engine.get_status()
                    clone_info = {"store_messages": st.get("store_messages"), "dag_nodes": st.get("dag_nodes")}
                answers, cap_hit = {}, False

                def answered(a):  # as score_s reads it: None or a blank string is unanswered; any other value is an answer
                    return a is not None and (not isinstance(a, str) or bool(a.strip()))

                for attempt in range(2):  # one retry when the final call hits the cap; the first non-empty answer wins
                    got, meta = R.answer(reader, run.system, view, prompt, tools_engine if is_open else None,
                                         schemas, guidance, run.m.tokens.count_tokens)
                    attempts.append(dict(meta))
                    for pid, a in (got or {}).items():
                        if not answered(answers.get(pid)):
                            answers[pid] = a
                    capped = ((meta.get("reader_calls") or [meta.get("usage")])[-1] or {}).get("completion_tokens", 0) >= 8192
                    complete = all(answered(answers.get(p["id"])) for p in b["probes"])
                    cap_hit = cap_hit or capped
                    if not capped or complete:
                        break
                if cap_hit and not complete:  # probes still unanswered after a cap: truncated, never lost
                    err = "READER_TRUNCATED: completion cap 8192 reached"
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                if tools_engine is not None:
                    tools_engine.shutdown()
        meta["attempts"] = attempts
        meta["reader_calls"] = [c for a in attempts for c in a.get("reader_calls") or []]
        meta.update({k: sum(a.get(k) or 0 for a in attempts) for k in ("tool_calls", "tokens_read_back", "wall_s")
                     if any(a.get(k) is not None for a in attempts)})
        run.reader_calls.append({"batch": b["id"], "calls": meta.get("reader_calls")})
        (out / "answers" / f"{b['id']}.json").write_text(json.dumps(
            {"batch": b["id"], "prompt_chars": len(prompt), "clone": clone_info, "error": err, **meta}, indent=1))
        print(f"  batch {b['id']}: {'ERROR ' + err if err else 'answered ' + str(len(answers or {}))} "
              f"wall {meta.get('wall_s')}s calls {meta.get('tool_calls', 0)}", flush=True)
        for p in b["probes"]:
            f, rc, ans = facts.get(p["id"]) or {}, receipts.get(p["id"]), (answers or {}).get(p["id"])
            results.append({
                "schema": "s2-result-v1", "kind": "probe", "probe_id": p["id"], "probe_kind": p["kind"],
                "expect": p["expect"], "gold": p.get("gold") or f.get("answer") or (traps.get(p["id"]) or {}).get("answer"),
                **{k: f.get(k) for k in ("stale", "placement", "row_role")}, "fact_class": f.get("class"),
                "answer": ans, "answered": ans is not None, "timed_out": bool(err and "timeout" in err.lower()),
                "error": err, "status": "UNAVAILABLE" if unavailable else ("ERROR" if err else "OK"),
                "arm": run.arm["name"], "arm_kind": run.arm["kind"], "seed": run.seed, "run": run.args.run,
                "run_id": run.run_id, "lane": run.args.lane, "reader": run.args.reader,
                "reader_readback": dict(reader.readback), "batch_id": b["id"], "clone_id": clone_id,
                "store_backed": is_store, "receipt": None if not rc else
                {"kind": rc["kind"], "status": rc["status"] if is_store else "NO STORE"},
                "tool_calls_batch": meta.get("tool_calls", 0), "tokens_read_back_batch": meta.get("tokens_read_back", 0),
                "answer_wall_s_batch": meta.get("wall_s"), "runaway_guard_hit": meta.get("runaway_guard_hit", False),
                "attribution": "batch" if is_open else None, "context_format": meta.get("context_format"),
                "checkpoint_row": src["row"] if src else None, "summary_blocks": blocks,
                "timing_label": run.timing_label})
    return results

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="arm name ('ALL' with --dry-run)")
    ap.add_argument("--seed", required=True, help="1|2|3|smoke-1")
    ap.add_argument("--run", default="1", help="run id (1|2, or a labelled id such as smoke-m-1)")
    ap.add_argument("--lane", choices=("glm", "codex-sol", "codex-astra"), default="glm")
    ap.add_argument("--reader", choices=("astra-low", "glm"), default="glm")
    ap.add_argument("--slice", type=int, default=None, help="stop after this turn (smoke: 10)")
    ap.add_argument("--checkpoint", default="auto", help="auto (r4 F1 decision checkpoint) | <tokens>")
    ap.add_argument("--checkpoints", type=lambda v: [int(x) for x in v.split(",")], default=None,
                    help="S7 D1: row indexes probed from store snapshots (cp-<row>/); the replay stops at the last")
    ap.add_argument("--context-length", type=int, default=DEFAULT_CTX)
    ap.add_argument("--batches", type=int, default=0, help="answer only the first N batches (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="print effective config per arm; no model calls")
    ap.add_argument("--prefix60k", action="store_true", help="S6 A3: replay to the frozen prefix60k row, the arm's "
                    "own compaction fires once there (LCM_ABSOLUTE_THRESHOLD_TOKENS = view tokens at that gate); no probes")
    args = ap.parse_args()
    m = seam.load_engine()
    sdir = MATERIAL / ("smoke-seed-1" if args.seed == "smoke-1" else f"seed-{args.seed}")
    rows, man, stop = select_rows(sdir, args.checkpoint, args.slice)
    if args.checkpoints:  # the replay runs to the last checkpoint row (row 304 = the end of the seed material)
        stop = max(args.checkpoints)
        rows = jlines(sdir / "transcript.jsonl")[:stop + 1]
    if args.prefix60k:  # the frozen row (prefix60k.py); never recomputed here
        stop = jload(sdir / "prefix60k.json")["freeze_row_index"]
        rows = jlines(sdir / "transcript.jsonl")[:stop + 1]
    if args.dry_run:
        return dry_run(m, args, rows, stop)
    arm, seed = A.resolve(args.arm), ("smoke-1" if args.seed == "smoke-1" else f"seed-{args.seed}")
    run_dir = RUNS / arm["name"] / seed / str(args.run)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to reuse {run_dir} (fresh store per run)")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"arm": arm, "seed": seed, "run": args.run, "lane": args.lane, "reader": args.reader,
               "worktree": str(seam.WORKTREE), "worktree_head": seam.PINNED, "context_length": args.context_length,
               "stop_row_index": stop, "slice": args.slice, "checkpoint": args.checkpoint,
               "fleet_keys_excluded": A.FLEET_EXCLUDED, "harness_overrides": A.HARNESS_OVERRIDES, "started": time.time(),
               "population": "prefix60k" if args.prefix60k else "full-stream"}
    if arm["unsupported"]:
        summary.update(status="UNSUPPORTED", reason=arm["unsupported"])
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return print(f"{arm['name']}: UNSUPPORTED - {arm['unsupported']}")
    handler = logging.FileHandler(run_dir / "engine.log")
    logging.getLogger("hermes_lcm").addHandler(handler)
    logging.getLogger("hermes_lcm").addHandler(SKIPS)
    logging.getLogger("hermes_lcm").setLevel(logging.INFO)
    run, is_store = Run(m, arm, seed, args, sdir, rows, man, run_dir), arm["kind"] != "control"
    apply_env(arm["env"], run.home, run.db)
    (run_dir / "config.json").write_text(json.dumps({**config_dict(m.config.LCMConfig.from_env()),
                                                     "harness_overrides": A.HARNESS_OVERRIDES}, indent=1, sort_keys=True))
    summary["tokenizer"] = "tiktoken" if m.tokens._get_encoder() is not None else "char-estimate"
    isolation = {}
    if is_store:
        seam.set_lane(args.lane, run_dir / "lane-scratch")
        run.engine = new_engine(m, run.home, run.sid, args.context_length)
        summary["threshold_tokens"] = run.engine.threshold_tokens
        t0 = time.monotonic()
        view = run.replay()
        summary["replay_wall_s"] = round(time.monotonic() - t0, 2)
        summary["admission"] = run.receipts()
        run.engine.shutdown()
        isolation["db_sha256_before_probes"] = sha(run.db)
    else:
        view = [to_msg(r) for r in rows if r["role"] != "system"]
        view = view[-int(arm["env"]["LCM_FRESH_TAIL_COUNT"]):] if arm["name"] == "C1" else view
    run.engine = m.engine.LCMEngine.__new__(m.engine.LCMEngine)  # closed; only for the tool-schema lookup
    summary.update(gates=run.gates, events=run.events, receipts=run.receipts_out,
                   final_context={"rows": len(view), "tokens": run.ntok([run.sysmsg] + view)},
                   final_continuity=continuity(man, run.system, view))
    (run_dir / "assembled_context.json").write_text(json.dumps({"system": run.system, "view": view}))
    if args.prefix60k:  # a timing population only: no probes
        summary.update(status="DONE", finished=time.time(), prefix60k=getattr(run, "p60", None))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return print(f"wrote {run_dir}/summary.json (prefix60k: {len(run.events)} event(s), no probes)")
    reader = R.GLMReader() if args.reader == "glm" else R.AstraLowReader(run_dir / "reader-scratch")
    for cp in run.snaps if is_store else ():  # S7 D1: one scorer-shaped directory per checkpoint
        run.receipts_out, run.reader_calls = cp["receipts"], []
        sha0, res = sha(cp["db"]), probe(run, cp["view"], reader, is_store, cp)
        sha1 = sha(cp["db"])
        cp_status = "DONE" if sha0 == sha1 and reader.readback.get("pin_ok", True) else "FAILED"
        if cp_status == "FAILED":
            summary["status"] = "FAILED"
        evs = run.events[: cp["n_events"]]
        (cp["dir"] / "continuity").mkdir(exist_ok=True)
        for e in evs:
            shutil.copy2(run_dir / "continuity" / f"event-{e['event']}.json", cp["dir"] / "continuity")
        (cp["dir"] / "assembled_context.json").write_text(json.dumps({"system": run.system, "view": cp["view"]}))
        (cp["dir"] / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
        (cp["dir"] / "summary.json").write_text(json.dumps({**summary, "stop_row_index": cp["row"], "checkpoint": cp["row"],
            "events": evs, "receipts": cp["receipts"], "admission": cp["admission"], "reader_calls": run.reader_calls,
            "final_context": {"rows": len(cp["view"]), "tokens": run.ntok([run.sysmsg] + cp["view"])}, "summary_blocks": res[0]["summary_blocks"] if res else None,
            "store_summaries": db_ro(cp["db"]).execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            "final_continuity": continuity(man, run.system, cp["view"]), "reader_readback": reader.readback, "status": cp_status,
            "isolation": {"snapshot_sha256_before_probes": sha0, "snapshot_sha256_after_probes": sha1}, "finished": time.time()}, indent=1, default=str))
        print(f"wrote {cp['dir']} ({len(res)} rows)", flush=True)
    if run.snaps:
        summary.setdefault("status", "DONE")
        summary.update(finished=time.time(), checkpoints=[c["row"] for c in run.snaps])
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return 1 if summary["status"] == "FAILED" else 0
    results = probe(run, view, reader, is_store)
    isolation["db_sha256_after_probes"] = sha(run.db)
    summary.update(reader_readback=reader.readback, isolation=isolation, status="DONE", finished=time.time())
    if (is_store and isolation["db_sha256_before_probes"] != isolation["db_sha256_after_probes"]
            or not reader.readback.get("pin_ok", True)):
        summary["status"] = "FAILED"
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {run_dir}/results.jsonl ({len(results)} rows), summary.json, config.json")
    return 1 if summary["status"] == "FAILED" else 0

def dry_run(m, args, rows, stop):
    out_dir = TS / "s2" / "dry-run"
    out_dir.mkdir(exist_ok=True)
    defaults = config_dict(m.config.LCMConfig())  # the printed line = every value the arm moves off the default
    sysmsg = {"role": "system", "content": "\n".join(r["content"] for r in rows if r["role"] == "system")}
    view = [to_msg(r) for r in rows if r["role"] != "system"]
    print(f"dry run: seed {args.seed}, rows 0..{stop} ({len(rows)} rows, "
          f"{m.tokens.count_messages_tokens([sysmsg] + view)} tokens incl. system), "
          f"context_length {args.context_length}; no model calls")
    for name in A.all_arms():
        arm = A.resolve(name)
        if arm["unsupported"]:
            print(f"{name:26s} UNSUPPORTED: {arm['unsupported']}")
            (out_dir / f"{name}.config.json").write_text(json.dumps({"status": "UNSUPPORTED",
                                                                     "reason": arm["unsupported"]}, indent=1))
            continue
        tmp = Path(tempfile.mkdtemp(prefix="s2-dry-"))  # $TMPDIR, deleted below
        try:
            apply_env(arm["env"], tmp / "hermes-home", tmp / "lcm.db")
            cfg = config_dict(m.config.LCMConfig.from_env())
            e = new_engine(m, tmp / "hermes-home", "dry", args.context_length)
            thr = e.threshold_tokens
            e.shutdown()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cfg.pop("database_path", None)
        eff = {"threshold_tokens": thr, **{k: v for k, v in cfg.items() if v != defaults.get(k)}}
        if arm["kind"] == "control":
            t = m.tokens.count_messages_tokens([sysmsg] + (view if name == "C0" else view[-24:]))
            eff = {"control": arm["note"], "context_tokens": t,
                   **{f"available_{r}": t + R.ANSWER_RESERVE <= w for r, w in R.READER_WINDOW.items()}}
        (out_dir / f"{name}.config.json").write_text(json.dumps(
            {"arm": arm["note"], "effective": eff, "full_config": cfg}, indent=1, sort_keys=True))
        print(f"{name:26s} " + " ".join(f"{k}={v}" for k, v in eff.items()))
    print(f"fleet keys excluded: {json.dumps(A.FLEET_EXCLUDED)}")

if __name__ == "__main__":
    raise SystemExit(main())
