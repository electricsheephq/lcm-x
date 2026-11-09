"""Synthetic recall-over-MCP gate kit; approved #2225 bars v1."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C = json.loads((HERE / "corpus.json").read_text())
Q = json.loads((HERE / "questions.json").read_text())
PROMPT = "You can search the memory of earlier Claude Code sessions in this project with the lcm-x tools. Use them to answer the question. Reply with the exact value only. If the memory holds nothing about it, reply exactly NOT FOUND.\n\nQuestion: "

def read(path):
    return json.loads(Path(path).read_text())
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

def tokens():
    return C["facts"] + [{**p[age], "id": p["id"] + "-" + age, "session": p[age + "_session"], "class": "latest"} for p in C["superseded"] for age in ("old", "new")]

def log_text(spec):
    rng = random.Random(spec["seed"])
    return "".join((spec["insert"]["text"] if i == spec["insert"]["line"] else "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=160))) + "\n" for i in range(1, spec["lines"] + 1))

def normalize(text):
    return re.sub(r"[^A-Z0-9 -]", "", " ".join(text.upper().split()))

def judge(answer, question, error=False):
    text = normalize(answer)
    found = {f["token"] for f in tokens() if f["token"] in text}
    bad = error or not text
    if question["kind"] == "negative":
        correct = not bad and "NOT FOUND" in text and not found
        return {"correct": correct, "fabricated": not bad and not correct, "verdict": "error" if bad else "correct" if correct else "fabricated"}
    if question["kind"] == "latest":
        pair = next(p for p in C["superseded"] if p["id"] == question["pair"])
        old, new = pair["old"]["token"] in found, pair["new"]["token"] in found
        verdict = "miss" if bad else "both" if old and new else "current" if new else "stale" if old else "miss"
        return {"correct": verdict == "current", "verdict": verdict}
    correct = not bad and found == {question["gold"]} and bool(re.search(r"(?<![A-Z0-9-])" + re.escape(question["gold"]) + r"(?![A-Z0-9-])", text))
    return {"correct": correct, "verdict": "correct" if correct else "miss"}

def package():
    if "hermes_lcm" not in sys.modules:
        pkg = types.ModuleType("hermes_lcm")
        pkg.__path__ = [str(ROOT)]
        sys.modules["hermes_lcm"] = pkg
    from hermes_lcm import claude_code_store
    return claude_code_store

def empty_store(home):
    cc = package()
    store = cc.open_store(cc.admit(home, hermes_home=home.parent / "unused-hermes"), home)
    cc.to_rollback_journal(store.connection)
    store.close()
    return {"created": True}

def arm_d(home, manifest):
    cc, records = package(), []
    store_home = home / ".lcm-x/claude-code"
    with sqlite3.connect((store_home / "lcm.db").as_uri() + "?mode=ro", uri=True) as db:
        for f in tokens():
            sid = manifest["sessions"][f["session"]]["session_id"]
            sessions = [sid] + ([r[0] for r in db.execute("SELECT child_session_id FROM session_lineage WHERE parent_session_id=? AND child_role='subagent'", (sid,))] if f["class"] == "subagent" and db.execute("SELECT 1 FROM sqlite_master WHERE name='session_lineage'").fetchone() else [])
            matches = []
            for session in sessions:
                for ident, role, content in db.execute("SELECT store_id, role, content FROM messages WHERE session_id=?", (session,)):
                    payloads = [(None, content or "")] + [(ref, (cc.load_externalized_payload(ref, config=None, hermes_home=str(store_home)) or {}).get("content", "")) for ref in cc.extract_externalized_refs(content or "")]
                    matches.extend({"store_id": ident, "session_id": session, "role": role, "sha256": hashlib.sha256(text.encode()).hexdigest(), "externalized_ref": ref} for ref, text in payloads if role in f["roles"] and f["token"] in text)
            records.append({"id": f["id"], "token": f["token"], "sessions": sessions, "matches": matches, "pass": bool(matches)})
    return {"tokens": records, "store_bytes": sum(p.stat().st_size for p in store_home.rglob("*") if p.is_file()), "host_load": os.getloadavg()}

def server(plugin, home):
    spec = read(plugin / ".mcp.json")["mcpServers"]["lcm-x"]
    spec = json.loads(json.dumps(spec).replace("${CLAUDE_PLUGIN_ROOT}", str(plugin.resolve())))
    spec["command"] = shutil.which(spec["command"]) or spec["command"]
    spec["env"] = {**spec.get("env", {}), "LCMX_CC_HOME": str(home.resolve())}
    return spec

def environment(home):
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"), "KIT_HOME": str(home)}

class MCP:
    def __init__(self, spec, home):
        self.proc = subprocess.Popen([spec["command"], *spec.get("args", [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env={**environment(home), **spec["env"]}, cwd=home, bufsize=0)
        self.sel, self.buf, self.ident, self.calls = selectors.DefaultSelector(), bytearray(), 0, []
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
    def rpc(self, method, params, notify=False):
        self.ident += 1
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method, "params": params, **({} if notify else {"id": self.ident})}) + "\n").encode())
        end = time.monotonic() + 60
        while not notify:
            while b"\n" not in self.buf:
                if not self.sel.select(max(0, end - time.monotonic())):
                    raise TimeoutError("MCP timeout")
                chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
                if not chunk:
                    raise RuntimeError("MCP closed")
                self.buf.extend(chunk)
            line, _, self.buf = self.buf.partition(b"\n")
            reply = json.loads(line)
            if reply.get("id") == self.ident:
                if "error" in reply or reply.get("result", {}).get("isError"):
                    raise RuntimeError("MCP error")
                return reply["result"]
    def call(self, name, args):
        start = time.monotonic()
        try:
            body = json.loads(self.rpc("tools/call", {"name": name, "arguments": args})["content"][0]["text"])
            if "error" in body or body.get("message_search_error"):
                raise RuntimeError("tool error")
            return body
        finally:
            self.calls.append({"tool": name, "ms": (time.monotonic() - start) * 1000})
    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()
        self.sel.close()

def expand(mcp, args):
    content, offset, ref = "", 0, None
    for _ in range(800):
        page = mcp.call("lcm_expand", {**{k: v for k, v in args.items() if k != "session_id" or "store_id" not in args}, "content_offset": offset, "max_tokens": 50000})
        content += page.get("content") or ""
        ref = ref or page.get("externalized_ref")
        if not page.get("has_more"):
            return expand(mcp, {"externalized_ref": ref, "session_id": args["session_id"]}) if ref and "store_id" in args else content
        next_offset = page.get("next_content_offset")
        if not isinstance(next_offset, int) or next_offset <= offset:
            raise RuntimeError("nonadvancing page")
        offset = next_offset
    raise RuntimeError("page bound")

def arm_t(plugin, home, direct, empty=False):
    direct = {"tokens": tokens()} if empty else direct
    mcp, rows = MCP(server(plugin, home / ".lcm-x/claude-code"), home), []
    try:
        mcp.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "recall-kit", "version": "1"}})
        mcp.rpc("notifications/initialized", {}, True)
        for d in direct["tokens"]:
            if empty and d["id"].endswith("-old"):
                continue
            row = {"id": d["id"], "pass": False}
            try:
                hits = mcp.call("lcm_grep", {"query": d["token"], "mode": "full_text"})["results"]
                row["pass"] = not hits if empty else False
                for hit in [] if empty else hits:
                    if hit.get("session_id") not in d["sessions"] or d["token"] not in hit.get("snippet", "").replace(">>>", "").replace("<<<", ""):
                        continue
                    text = expand(mcp, {"store_id": hit["store_id"], "session_id": hit["session_id"]})
                    ident = str(hit["store_id"]).split(":")[-1]
                    row["pass"] |= d["token"] in text and any(str(m["store_id"]) == ident and m["session_id"] == hit["session_id"] and hashlib.sha256(text.encode()).hexdigest() == m["sha256"] for m in d["matches"])
            except (RuntimeError, TimeoutError, ValueError, KeyError) as exc:
                row["error"] = type(exc).__name__
            rows.append(row)
    finally:
        mcp.close()
    return {"tokens": rows, "calls": mcp.calls, "empty": empty}

def claude(command, prompt, args, home, cwd, env=None):
    start, result, init, calls = time.monotonic(), {}, None, {}
    try:
        # The launcher isolates HOME and CLAUDE_CONFIG_DIR itself (from KIT_HOME); it needs the caller's own
        # environment to find its backend credentials, so HOME is not replaced here.
        proc = subprocess.run([command, "-p", prompt, *args, "--effort", "low", "--output-format", "stream-json", "--verbose"], env={**os.environ, "KIT_HOME": str(home), **(env or {})}, cwd=cwd, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"rc": -1, "wall_s": time.monotonic() - start, "is_error": True, "error": "timeout", "answer": ""}
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            init = event.get("mcp_servers", [])
        if event.get("type") == "result":
            result = event
        for block in (event.get("message") or {}).get("content", []) if event.get("type") == "assistant" else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls[block["name"]] = calls.get(block["name"], 0) + 1
    error = proc.returncode != 0 or result.get("is_error", True)
    return {"rc": proc.returncode, "wall_s": time.monotonic() - start, "is_error": error, "answer": result.get("result", ""), "session_id": result.get("session_id"), "tool_calls": calls, "reader_not_connected": init is None or [(s.get("name"), s.get("status")) for s in init] != [("lcm-x", "connected")], "limit": bool(error and re.search(r"rate.?limit|usage.?limit|limit.{0,25}(reached|exceeded)|hit.{0,25}limit", json.dumps(result), re.I))}

def quiesce(home):
    end, quiet = time.monotonic() + 180, 0
    while time.monotonic() < end:
        waiting = any(p.suffix in (".json", ".claimed") for p in (home / "spool").glob("*"))
        try:
            with (home / "capture.lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            waiting = True
        except FileNotFoundError:
            pass
        quiet = 0 if waiting else quiet + 1
        if quiet >= 3:
            return True
        time.sleep(.5)
    return False

def build(args):
    project, manifest = args.home / "proj/recall-fixture", {"sessions": {}, "stop": False}
    project.mkdir(parents=True, exist_ok=False)
    for name, spec in C["files"].items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec["text"] if "text" in spec else log_text(spec))
    for session in C["sessions"]:
        mode, _, parent = session["start"].partition(":")
        sid = str(uuid.uuid4()) if mode == "new" else manifest["sessions"][parent]["session_id"]
        record = {"session_id": sid, "calls": []}
        manifest["sessions"][session["label"]] = record
        for i, prompt in enumerate(session["turns"]):
            flags = ["--session-id" if mode == "new" and i == 0 else "--resume", sid] + (["--fork-session"] if mode == "fork" and i == 0 else [])
            call = claude(args.claude, prompt, ["--plugin-dir", str(args.plugin), "--dangerously-skip-permissions", *flags], args.home, project, session.get("env"))
            sid = call.get("session_id") or sid
            record["session_id"] = sid
            record["calls"].append({k: v for k, v in call.items() if k not in ("answer", "reader_not_connected")})
            if call.get("limit") or call["is_error"] or (mode == "fork" and i == 0 and (not call.get("session_id") or call["session_id"] == manifest["sessions"][parent]["session_id"])):
                manifest.update(stop=True, stop_reason="limit" if call.get("limit") else "call_error")
                break
        if not manifest["stop"]:
            quiet = quiesce(args.home / ".lcm-x/claude-code")
            script = args.plugin / "lcm-x/scripts/lcm_x_capture.py"
            audit = subprocess.run([sys.executable, str(script), "audit", "--home", str(args.home / ".lcm-x/claude-code"), "--session", sid], capture_output=True, timeout=900)
            report = json.loads(audit.stdout)
            record["audit"] = {"rc": audit.returncode, **{k: report.get(k) for k in ("ok", "verified", "failures", "unverified")}}
            shapes = {}
            for path in (args.home / ".claude/projects").glob(f"*/{sid}.jsonl"):
                for line in path.read_text().splitlines():
                    e = json.loads(line)
                    if e.get("type") == "assistant" and (e.get("message") or {}).get("id"):
                        shape = re.sub(r"\d", "9", e["message"]["id"][:12])
                        shapes[shape] = shapes.get(shape, 0) + 1
            record["backend"] = {"msg_id_shapes": shapes}
            manifest["stop"] = not (quiet and audit.returncode == 0 and report.get("ok") and report.get("verified"))
        write(args.out / "corpus-manifest.json", manifest)
        if manifest["stop"]:
            break
    return manifest

def arm_a(args):
    stopped = threading.Event()
    spec = server(args.plugin, args.store_home)
    config = args.reader_home / "mcp.json"
    write(config, {"mcpServers": {"lcm-x": spec}})
    def one(q):
        if stopped.is_set():
            return None
        cwd = args.reader_home / (q["id"] + "-" + uuid.uuid4().hex)
        cwd.mkdir(parents=True)
        call = claude(args.claude, PROMPT + q["question"], ["--strict-mcp-config", "--mcp-config", str(config), "--tools", "", "--allowedTools", "mcp__lcm-x__*", "--settings", '{"disableAllHooks":true}'], args.reader_home, cwd, {"KIT_LCMX_CC_HOME": str(args.store_home), "LCMX_CAPTURE": "off"})
        if call.get("limit"):
            stopped.set()
        return {"id": q["id"], **call}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = [r for r in pool.map(one, [q for q in Q if not args.only or q["kind"] == args.only]) if r]
    return {"answers": rows, **({"stop_reason": "limit"} if stopped.is_set() else {})}

def score(manifest, d, t, te, a, a2=None, ae=None):
    dm, tm = ({r["id"]: r["pass"] for r in arm["tokens"]} for arm in (d, t))
    runs, rows = [a] + ([a2] if a2 else []), []
    for q in Q:
        ids = [q["fact"]] if q["kind"] == "positive" else [q["pair"] + "-old", q["pair"] + "-new"] if q["kind"] == "latest" else []
        answers = [next((r for r in run["answers"] if r["id"] == q["id"]), {}) for run in runs]
        verdicts = [judge(r.get("answer", ""), q, r.get("is_error", False) or r.get("reader_not_connected", False) or r.get("rc", 0) != 0) for r in answers]
        rows.append({"id": q["id"], "kind": q["kind"], "class": next((f["class"] for f in C["facts"] if f["id"] == q.get("fact")), q["kind"]), "D": bool(ids) and all(dm.get(i, False) for i in ids), "T": bool(ids) and all(tm.get(i, False) for i in ids), "A": verdicts, "tool_calls": [r.get("tool_calls", {}) for r in answers], "wall_s": [r.get("wall_s") for r in answers]})
        if rows[-1]["T"] and not all(v["correct"] for v in verdicts):
            rows[-1]["classification"] = ""
    positives = [sum(r["A"][i]["correct"] for r in rows if r["kind"] == "positive") for i in range(len(runs))]
    fabricated = [sum(r["A"][i].get("fabricated", False) for r in rows if r["kind"] == "negative") for i in range(len(runs))]
    def valid(run, qs):
        return not run.get("stop_reason") and {r["id"] for r in run["answers"]} == {q["id"] for q in qs} and all(not r.get("is_error") and not r.get("reader_not_connected", True) and r.get("rc") == 0 and bool(r.get("answer", "").strip()) for r in run["answers"])
    empty_ok = len(te["tokens"]) == 44 and {r["id"] for r in te["tokens"]} == {f["id"] for f in tokens() if not f["id"].endswith("-old")} and all(r["pass"] for r in te["tokens"]) and ae is not None and valid(ae, Q[:40]) and not any(judge(r["answer"], next(q for q in Q if q["id"] == r["id"]))["correct"] for r in ae["answers"])
    latency = {}
    for name in {c["tool"] for c in t["calls"]}:
        values = sorted(c["ms"] for c in t["calls"] if c["tool"] == name)
        latency[name] = {"p50_ms": values[math.ceil(len(values) * .5) - 1], "p90_ms": values[math.ceil(len(values) * .9) - 1], "max_ms": max(values)}
    tripwire = any(v["p90_ms"] > 1000 or v["max_ms"] > 5000 for v in latency.values())
    bars = {"empty_control": empty_ok, "capture": not manifest.get("stop") and set(manifest["sessions"]) == {s["label"] for s in C["sessions"]} and all(s.get("audit", {}).get("rc") == 0 and s["audit"].get("ok") and s["audit"].get("verified") for s in manifest["sessions"].values()), "D": all(dm.get(f["id"], False) for f in tokens()), "T": all(tm.get(f["id"], False) for f in tokens()), "A": min(positives) >= 36, "false_recall": max(fabricated) <= 1, "answers_complete": all(valid(run, Q) for run in runs)}
    second = 34 <= positives[0] <= 38 and a2 is None
    verdict = "VOID" if not empty_ok else "STOP" if not bars["capture"] or not bars["D"] else "FAIL" if not bars["T"] else "SECOND RUN REQUIRED" if second else "FAIL" if not all(bars.values()) else "CLASSIFICATION REQUIRED" if tripwire or any("classification" in r and r["kind"] == "positive" for r in rows) else "PASS"
    return {"verdict": verdict, "bars": bars, "positives": positives, "fabricated": fabricated, "second_run_required": second, "rows": rows, "latency": latency, "tripwire": tripwire, "latency_classification": "" if tripwire else "not triggered", "store_bytes": d.get("store_bytes"), "host_load": d.get("host_load")}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    fields = {"build-corpus": "claude plugin home out", "arm-d": "home manifest out", "arm-t": "plugin home d out", "make-empty-store": "home", "arm-a": "claude plugin store-home reader-home out", "score": "corpus-manifest d t t-empty a out"}
    for name, required in fields.items():
        sub = commands.add_parser(name)
        for key in required.split():
            sub.add_argument("--" + key, required=True, type=str if key == "claude" else Path)
        if name == "arm-t":
            sub.add_argument("--empty", action="store_true")
        if name == "arm-a":
            sub.add_argument("--only", choices=["positive"])
            sub.add_argument("--parallel", type=int, choices=range(1, 4), default=1)
        if name == "score":
            sub.add_argument("--a2", type=Path)
            sub.add_argument("--a-empty", type=Path)
    args = parser.parse_args()
    if args.cmd == "score":
        inputs = args.out / "inputs"
        for key in ("corpus_manifest", "d", "t", "t_empty", "a", "a2", "a_empty"):
            source = getattr(args, key)
            if source:
                write(inputs / (key + ".json"), read(source))
        write(inputs / "corpus.json", C)
        write(inputs / "questions.json", Q)
        result = score(*[read(inputs / (k + ".json")) for k in ("corpus_manifest", "d", "t", "t_empty", "a")], read(inputs / "a2.json") if args.a2 else None, read(inputs / "a_empty.json") if args.a_empty else None)
        result.update(code={"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}, keep=["RESULT.md", "RESULT-MANIFEST.json", "inputs/"], headline={"metric": "positive_recall", "value": min(result["positives"]), "n": 40, "baseline": None}, published={"T1": None, "T2": None})
        (args.out / "RESULT.md").write_text(f'# Recall over MCP: {result["verdict"]}\n\nBars: {json.dumps(result["bars"])}\n\nPositive counts: {result["positives"]}; fabricated: {result["fabricated"]}.\n\nLatency per tool: {json.dumps(result["latency"])}; tripwire: {result["tripwire"]}.\n\n| id | kind | class | D | T | A verdict | A tool calls | wall s | classification |\n|---|---|---|---|---|---|---|---|---|\n' + '\n'.join('| ' + ' | '.join(str(r[k]) for k in ("id", "kind", "class", "D", "T")) + ' | ' + ', '.join(v["verdict"] for v in r["A"]) + f' | {r["tool_calls"]} | {r["wall_s"]} | {r.get("classification", "")} |' for r in result["rows"]) + '\n\nNot proven: synthetic facts, one model, one small store; no real owner history, large-store recall, Grok sessions or customer readiness. Shared-Mac latency only. Latest-value rows are not gated. Bars are fixed by approved BARS v1 before any run. T1/T2: unpublished.\n')
    else:
        result = build(args) if args.cmd == "build-corpus" else arm_d(args.home, read(args.manifest)) if args.cmd == "arm-d" else arm_t(args.plugin, args.home, read(args.d), args.empty) if args.cmd == "arm-t" else arm_a(args) if args.cmd == "arm-a" else empty_store(args.home)
    write((args.out if hasattr(args, "out") else args.home) / ("RESULT-MANIFEST.json" if args.cmd == "score" else "corpus-manifest.json" if args.cmd == "build-corpus" else args.cmd + ".json"), result)

if __name__ == "__main__":
    main()
