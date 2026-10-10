#!/usr/bin/env python3
"""S4 codex-native reference arm: genuine-role replay on the installed Codex CLI, native auto-compaction, forked probes.

Spec: ../../SPEC-TRACK-S.md §S4, SPEC-RECONCILIATION-r3 §R1 (hosted replay mapping, pinned reader) / §R5, r4 F1/N2.
Reuses the kit's drive_codex.py (event helpers) and parse_rollout.py (compaction/token telemetry) read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the kit worktree
KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "track_s"))
import checkpoints as CP  # noqa: E402
import drive_codex as DC  # noqa: E402  (kit; unchanged)
import parse_rollout as PR  # noqa: E402  (kit; unchanged)

TS = Path(os.environ["TRACK_S_OUT"]).resolve()           # <D>/eval/track-s
S4 = TS / "s4"
HOME = S4 / "home"                                  # isolated CODEX_HOME
RUNS = TS / "codex-runs"
USER_AUTH = None  # supplied by --auth-file; never copied into the repository
MODEL, EFFORT = "gpt-6-astra", "low"
NATIVE_TRIGGER = 244_800                            # 90 % of the 272k window (r3 §R1; re-read from the rollout)
TURN_TIMEOUT_S = 1800
DRIVER_TAG = "[replay driver · turn {n}]"
TOOL_ITEM_TYPES = {"command_execution", "mcp_tool_call", "web_search", "file_change", "patch_apply",
                   "collab_tool_call", "image_view", "todo_list"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- isolated home
def run_home(seed: str, run: str) -> Path:
    """One isolated home per (seed, run), as the artifacts are keyed RUNS/<seed>/<run>."""
    if not all(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", part) and part not in (".", "..") for part in (seed, run)):
        raise SystemExit("STOP: invalid seed or run id; no run started")
    return S4 / "home" / seed / run


def codex_bin() -> Path:
    return Path(shutil.which("codex") or "codex").resolve()


def env() -> dict:
    e = dict(os.environ)
    e["CODEX_HOME"] = str(HOME)
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY"):   # subscription lane only: never a metered key
        e.pop(k, None)
    return e


def setup_home() -> dict:
    """HOME holds only auth.json (copied, never printed) and whatever the CLI itself writes."""
    if (HOME / "config.toml").exists():
        raise RuntimeError("isolated CODEX_HOME contains config.toml; refusing to run with non-stock config")
    info = {"codex_home": str(HOME), "config_toml": "absent (stock defaults; ~/.codex/config.toml NOT loaded)"}
    if any(path.is_symlink() for path in (HOME.parent.parent, HOME.parent, HOME, HOME / "auth.json")):
        raise RuntimeError("isolated CODEX_HOME path contains a symlink; refusing to copy auth")
    HOME.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(HOME.parent, 0o700)
    HOME.mkdir(parents=True, exist_ok=True)   # the dry run builds the home too: auth isolation is checked first
    os.chmod(HOME, 0o700)
    dst = HOME / "auth.json"
    shutil.copy2(USER_AUTH, dst)
    os.chmod(dst, 0o600)
    info["auth_copied"] = dst.exists()
    if dst.exists():
        info["auth_copy_sha256_prefix"] = sha(dst)[:12]   # identity of the copy only, not its content
        info["auth_last_refresh"] = json.loads(dst.read_text()).get("last_refresh")
        st = subprocess.run([str(codex_bin()), "login", "status"], env=env(), capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=60)
        # record only a status class; the CLI's text is not stored
        info["login_status"] = "logged_in" if st.returncode == 0 and "logged in" in (st.stdout + st.stderr).lower() \
            else f"NOT logged in (exit {st.returncode})"
        info["auth_mode_chatgpt"] = "chatgpt" in (st.stdout + st.stderr).lower()
    return info


# ---------------------------------------------------------------- material -> workspace + turn prompts
def material_dir(seed: str) -> Path:
    return Path(os.environ["TRACK_S_MATERIAL"]).resolve() / ("smoke-seed-1" if seed == "smoke-1" else f"seed-{seed}")


def materialise(rows: list[dict], ws: Path, dictation: str = "user") -> dict:
    """System row -> AGENTS.md (the CLI's workspace-instructions input); each tool row -> calls/<call_id>.out;
    with --dictation file, each assistant text row -> notes/<row_id>.md (read by the agent, then reproduced)."""
    (ws / "calls").mkdir(parents=True, exist_ok=True)
    files = {}
    for r in rows:
        if r["role"] == "system":
            p = ws / "AGENTS.md"
            p.write_text(r["content"] + "\n", encoding="utf-8")
        elif r["role"] == "tool":
            p = ws / "calls" / f"{r['tool_call_id']}.out"
            p.write_text(r["content"], encoding="utf-8")
        elif dictation == "file" and r["role"] == "assistant" and not r.get("tool_call_id"):
            (ws / "notes").mkdir(exist_ok=True)
            p = ws / "notes" / f"{r['id']}.md"
            p.write_text(r["content"], encoding="utf-8")
        else:
            continue
        files[str(p.relative_to(ws))] = {"row_id": r["id"], "chars": len(r["content"]), "sha256": sha(p)}
    return files


def plan_turns(rows: list[dict], dictation: str = "user") -> list[dict]:
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["turn"]].append((i, r))
    plans = []
    for n in sorted(by):
        users = [r for _, r in by[n] if r["role"] == "user"]
        notes = [r for _, r in by[n] if r["role"] == "assistant" and not r.get("tool_call_id")]
        calls = [r for _, r in by[n] if r["role"] == "tool"]
        parts = [r["content"] for r in users]
        drv = [DRIVER_TAG.format(n=n)]
        if notes and dictation == "file":
            drv.append(f"Before anything else, run each command below with your shell tool, one call per command; then "
                       f"write the {len(notes)} note(s) they print as your own message, reproduced verbatim and in order:")
            drv += [f"    cat notes/{r['id']}.md" for r in notes]
        elif notes:
            drv.append(f"Before anything else, write the following {len(notes)} note(s) as your own message, reproduced "
                       "verbatim and in order (the text between the markers only, not the markers):")
            for r in notes:
                drv += ["<<<NOTE>>>", r["content"], "<<<END NOTE>>>"]
        if calls:
            drv.append(("Then run" if notes else "Run") + " each command below with your shell tool, one call per "
                       "command, in order, exactly as written; run no other command and do not summarise the output:")
            drv += [f"    cat calls/{r['tool_call_id']}.out" for r in calls]
        drv.append(f"Finish with the line: TURN {n} DONE")
        parts.append("\n".join(drv))
        plans.append({"turn": n, "row_ids": [r["id"] for _, r in by[n]], "prompt": "\n\n".join(parts),
                      "n_user": len(users), "n_notes": len(notes), "n_calls": len(calls),
                      "note_chars": sum(len(r["content"]) for r in notes)})
    return plans


# ---------------------------------------------------------------- CLI invocation
def base_cmd(kind: str, sid: str | None, ws: Path, extra: list[str]) -> list[str]:
    pin = ["--json", "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}", "--skip-git-repo-check"] + extra
    if kind == "start":
        return [str(codex_bin()), "exec", *pin, "-c", 'sandbox_mode="read-only"', "-C", str(ws), "-"]
    if kind == "resume":
        return [str(codex_bin()), "exec", "resume", sid, *pin, "-c", 'sandbox_mode="read-only"', "-"]
    if kind == "fork":   # r4 N2: the runtime's own fork; context-only (shell tools disabled + rejection check)
        return [str(codex_bin()), "exec", "fork", sid, *pin, "--disable", "shell_tool", "--disable", "unified_exec",
                "-c", 'sandbox_mode="read-only"', "-"]
    raise ValueError(kind)


def token_hours_left() -> float:
    """Hours until the copied access token expires; reads only the JWT `exp` claim, never prints the token."""
    import base64
    tok = json.loads((HOME / "auth.json").read_text()).get("tokens", {}).get("access_token", "")
    part = tok.split(".")[1] if tok.count(".") == 2 else ""
    try:
        exp = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("exp")
    except ValueError:
        exp = None
    return (exp - time.time()) / 3600 if isinstance(exp, (int, float)) else -1.0


def token_guard(min_hours: float, where: str) -> None:
    """DECISIONS-S4 #4: never let the run trigger a refresh inside the isolated home."""
    if token_hours_left() < min_hours:
        msg = f"STOP ({where}): copied access token has < {min_hours:g} h left; refusing (no refresh, no re-login)"
        print(msg, flush=True)
        raise SystemExit(msg)


def run_cli(cmd: list[str], prompt: str, cwd: Path, log_stem: Path) -> dict:
    token_guard(0.25, "before call")   # in-run margin; the 3 h start gate is in main()
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(f"{log_stem}.stderr", "w"), text=True)
    threading.Thread(target=lambda: (proc.stdin.write(prompt), proc.stdin.close()), daemon=True).start()
    timer = threading.Timer(TURN_TIMEOUT_S, proc.kill)
    timer.start()
    events, first = [], None
    with open(f"{log_stem}.events.jsonl", "w") as fh:
        for line in proc.stdout:
            now = time.time()
            first = first or now
            fh.write(json.dumps({"recv_ts": now, "line": line.rstrip("\n")}) + "\n")
            try:
                ev = json.loads(line)
                if isinstance(ev, dict):
                    ev["_recv_ts"] = now
                    events.append(ev)
            except json.JSONDecodeError:
                pass
    rc = proc.wait()
    timer.cancel()
    t1 = time.time()
    msgs = [e["item"].get("text", "") for e in events
            if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message"]
    tools = [e["item"] for e in events if e.get("type") in ("item.started", "item.completed")
             and (e.get("item") or {}).get("type") in TOOL_ITEM_TYPES]
    usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
    return {"rc": rc, "timed_out": (t1 - t0) >= TURN_TIMEOUT_S, "submit_ts": t0, "first_event_ts": first,
            "end_ts": t1, "wall_s": round(t1 - t0, 2), "thread_id": DC._thread_id(events), "agent_messages": msgs,
            "tool_items": len(tools), "event_types": dict(Counter(e.get("type") for e in events)), "usage": usage}


# ---------------------------------------------------------------- rollout extracts
def rollout_path(sid: str) -> Path:
    paths = PR._find_rollouts(sid, HOME / "sessions")
    if not paths:
        raise FileNotFoundError(f"no rollout for {sid} under {HOME / 'sessions'}")
    return paths[-1]


def texts(v) -> list[str]:
    out = []
    if isinstance(v, str):
        # 0.159 code-mode tool outputs carry the command result as a JSON object ({"chunk_id",...,"output"})
        if v.lstrip().startswith("{") and '"output"' in v:
            try:
                obj = json.loads(v)
                if isinstance(obj, dict) and isinstance(obj.get("output"), str):
                    v = obj["output"] + (f"\n[original_token_count={obj['original_token_count']}]"
                                         if "original_token_count" in obj else "")
            except json.JSONDecodeError:
                pass
        out.append(v)
    elif isinstance(v, list):
        for x in v:
            out += texts(x)
    elif isinstance(v, dict):
        for k in ("text", "output", "content", "arguments", "input"):
            if k in v:
                out += texts(v[k])
    return out


def rollout_items(sid: str) -> list[dict]:
    """Per response_item: role/kind, text, and the replay turn it belongs to (by the driver tag)."""
    items, turn = [], 0
    ctx = []
    for r in jl(rollout_path(sid)):
        t, p = r.get("type"), r.get("payload") if isinstance(r.get("payload"), dict) else {}
        if t == "turn_context":
            ctx.append({"turn": turn, "model": p.get("model"), "effort": p.get("effort"), "ts": r.get("timestamp")})
            continue
        if t == "world_state":
            items.append({"turn": turn, "kind": "world_state", "text": json.dumps(p, ensure_ascii=False)})
            continue
        if t != "response_item":
            continue
        kind = p.get("type")
        text = "\n".join(texts(p.get("content") if kind == "message" else p))
        if kind == "message" and p.get("role") == "user":
            for n in range(turn + 1, turn + 3):
                if DRIVER_TAG.format(n=n) in text:
                    turn = n
        items.append({"turn": turn, "kind": kind, "role": p.get("role"), "call_id": p.get("call_id"),
                      "name": p.get("name"), "text": text})
    return items, ctx


def admission(rows: list[dict], facts: list[dict], items: list[dict]) -> dict:
    by_turn = defaultdict(list)
    for it in items:
        by_turn[it["turn"]].append(it)
    all_text = {k: "\n".join(it["text"] for it in items if (it.get("role") or it["kind"]) == k)
                for k in ("user", "developer", "assistant", "world_state")}
    per_row, row_status = [], {}
    for r in rows:
        its, c = by_turn.get(r["turn"], []), r["content"].strip()
        if r["role"] == "system":
            where = [k for k, v in all_text.items() if c in v]
            st = {"status": "admitted" if where else "not_admitted", "where": where}
        elif r["role"] == "user":
            ok = any(c in it["text"] for it in its if it.get("role") == "user")
            st = {"status": "admitted" if ok else "not_admitted", "as": "user message"}
        elif r["role"] == "assistant" and r.get("tool_call_id"):
            calls = [it for it in its if it["kind"] in ("function_call", "custom_tool_call", "local_shell_call")
                     and f"{r['tool_call_id']}.out" in it["text"]]
            st = {"status": "admitted" if calls else "not_admitted", "as": "agent's own tool call",
                  "n_calls": len(calls)}
        elif r["role"] == "assistant":
            said = "\n".join(it["text"] for it in its if it.get("role") == "assistant")
            st = {"status": "admitted" if c in said else ("partial" if c[:200] in said else "not_admitted"),
                  "as": "agent utterance (dictated in the user turn)"}
        else:  # tool
            cids = {it["call_id"] for it in its if it["kind"] in ("function_call", "custom_tool_call")
                    and f"{r['tool_call_id']}.out" in it["text"]}
            outs = [it["text"] for it in its if it["kind"] in ("function_call_output", "custom_tool_call_output")
                    and it["call_id"] in cids]
            o = "\n".join(outs)
            st = {"status": "admitted" if c in o else ("truncated" if outs else "not_admitted"),
                  "as": "tool output of the agent's call", "planted_chars": len(c), "output_chars": len(o),
                  "runtime_truncated": "truncated" in o.lower()}
        row_status[r["id"]] = st
        per_row.append({"row_id": r["id"], "turn": r["turn"], "role": r["role"], **st})
    fact_rows = {r["id"]: r for r in rows}
    fstat = []
    for f in facts:
        r = fact_rows.get(f["row_id"])
        if r is None:
            continue
        rs = row_status[r["id"]]
        in_role = {"user": all_text["user"], "assistant": all_text["assistant"]}
        tool_text = "\n".join(it["text"] for it in items if it["kind"] in ("function_call_output",
                                                                            "custom_tool_call_output"))
        role_text = tool_text if f["row_role"] == "tool" else in_role.get(f["row_role"], "")
        fstat.append({"id": f["id"], "row_role": f["row_role"], "placement": f["placement"],
                      "admitted_in_role": f["value"] in role_text, "row_status": rs["status"],
                      "also_in_user_text": f["row_role"] != "user" and f["value"] in all_text["user"],
                      "also_in_tool_text": f["row_role"] == "assistant" and f["value"] in tool_text})
    return {"rows_by_role_status": dict(Counter(f"{x['role']}:{x['status']}" for x in per_row)),
            "facts_planted_in_slice": len(fstat), "facts_admitted_in_role": sum(x["admitted_in_role"] for x in fstat),
            "facts_not_admitted": [x["id"] for x in fstat if not x["admitted_in_role"]],
            "rows": per_row, "facts": fstat}


def survival(sid: str, facts: list[dict], man: dict, rdir: Path | None = None, rows: list[dict] = (), workspace: Path | None = None) -> list[dict]:
    """What the CLI kept at each `compacted` row: readable replacement-history text vs encrypted payload.
    With `rdir` (S6 A2): continuity/event-<i>.json = that text + the host-instruction text the CLI sends (the
    driver-installed workspace/AGENTS.md), i.e. the plain text the next model call receives after event i."""
    agents = (workspace or rdir / "workspace") / "AGENTS.md" if workspace or rdir else None
    host = agents.read_text(encoding="utf-8") if agents and agents.exists() else ""
    out, turn, window, last_row = [], 0, None, {r["turn"]: i for i, r in enumerate(rows)}
    for r in jl(rollout_path(sid)):  # S7 D4: the replay turn (driver tag) and the CLI-reported window at each compaction
        p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        if r.get("type") == "response_item" and p.get("type") == "message" and p.get("role") == "user":
            text = "\n".join(texts(p.get("content")))
            turn = next((n for n in range(turn + 1, turn + 3) if DRIVER_TAG.format(n=n) in text), turn)
        if p.get("type") == "token_count":
            window = (p.get("info") or {}).get("model_context_window") or window
        if r.get("type") != "compacted":
            continue
        hist = p.get("replacement_history") or []
        plain = "\n".join(t for it in hist for t in texts(it.get("content") if isinstance(it, dict) else it))
        plain += "\n" + (p.get("message") or "")
        enc = sum(1 for it in hist for c in (it.get("content") or [] if isinstance(it, dict) else [])
                  if isinstance(c, dict) and c.get("encrypted_content"))
        enc += sum(1 for it in hist if isinstance(it, dict) and it.get("encrypted_content"))
        if rdir:
            jdump(rdir / "continuity" / f"event-{len(out)}.json", {
                "event": len(out), "window_number": p.get("window_number"), "ts": r.get("timestamp"),
                "source": "replacement_history plain text + workspace/AGENTS.md (host instructions the CLI sends)",
                "encrypted_parts": enc, "text": plain + "\n" + host})
        out.append({"window_number": p.get("window_number"), "ts": r.get("timestamp"), "turn": turn,
                    "turn_last_row_index": last_row.get(turn), "cli_window": window,
                    "replacement_items": dict(Counter(f"{it.get('type')}/{it.get('role')}"
                                                      for it in hist if isinstance(it, dict))),
                    "encrypted_parts": enc, "plain_chars": len(plain),
                    "facts_verbatim_in_plain_text": sorted(f["id"] for f in facts if f["value"] in plain),
                    "continuity_host_visible": {c["id"]: c["value"] in plain + "\n" + host for c in man.get("continuity", [])},
                    "continuity_verbatim": {c["id"]: c["value"] in plain for c in man.get("continuity", [])}})
    return out


def beyond_declared(rows, facts, stop, survival=(), continuity=()):
    """Turn-end replay exposure beyond the declared fact-admission horizon; IDs/counts only."""
    indices = [i for i in range(len(rows)) if i > stop]
    extra = set(indices)
    return dict(count=len(indices), row_indices=indices,
                facts=[f["id"] for f in facts if f.get("row_index") in extra],
                corrections=[f["id"] for f in facts if f.get("correction_source") and
                             (f.get("row_index") in extra or f["correction_source"].get("row_index") in extra)],
                continuity=[c["id"] for c in continuity if c.get("first_presentation_row_index", min(
                    (i for i, r in enumerate(rows) if c["value"] in (r.get("content") or "")),
                    default=c.get("row_index", 0))) in extra],
                compactions=[e["window_number"] for e in survival if e.get("turn_last_row_index") in extra])


# ---------------------------------------------------------------- probes
def batches_for(sdir: Path, checkpoint=None) -> list[dict]:
    out, cont = jl(sdir / "probe_batches.jsonl"), json.loads((sdir / "continuation.json").read_text())
    out.append({"id": out[0]["id"].rsplit("-", 1)[0] + "-BCONT", "text": out[0]["text"],
                "probes": [{"id": f"{cont['id']}.{k}", "kind": "continuation_field", "expect": "value", "gold": cont[k],
                            "text": f"For the pending mid-task continuation, what is its `{k}`?"}
                           for k in cont if k not in ("id", "row_id", "row_index", "row_role")]})
    return CP.augment(sdir, out, checkpoint)


def parse_answers(msgs: list[str]) -> dict | None:
    for m in reversed(msgs):
        s, e = m.find("{"), m.rfind("}")
        if s >= 0 and e > s:
            try:
                obj = json.loads(m[s:e + 1], strict=False)
                return obj.get("answer", obj) if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------- run
def main(a=None, state=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--auth-file", required=True, type=Path, help="CLI auth file for the isolated external arm")
    ap.add_argument("--seed", required=True, help="smoke-1 | 1 | 2 | 3")
    ap.add_argument("--run", default="1", help="run id (1|2, or a labelled id such as smoke-m-1)")
    ap.add_argument("--slice", type=int, default=0, help="stop after this material turn (0 = decision checkpoint)")
    ap.add_argument("--force-event", action="store_true",
                    help="smoke suffix: one driver checkpoint turn with a low model_auto_compact_token_limit")
    ap.add_argument("--batches", type=int, default=0, help="probe only the first N batches (0 = all + BCONT)")
    ap.add_argument("--dictation", choices=("user", "file"), default="user",
                    help="assistant rows: dictated in the user turn (smoked) | read from notes/<row>.md (not smoked)")
    ap.add_argument("--stop-row", default=None, help="v3 row index or v4 manifest checkpoint id")
    ap.add_argument("--checkpoints", help="v4 ids, all or lifecycle: probe then continue the unchanged parent")
    ap.add_argument("--readmit", action="store_true", help="recompute admission.json from the run's rollout; no calls")
    ap.add_argument("--dry-run", action="store_true", help="commands, home, workspace layout; no model calls")
    if a is not None:
        ap.parse_args = lambda: a
    a = ap.parse_args()
    global HOME
    HOME = run_home(a.seed, a.run)
    if a.checkpoints and state is None:
        selected = CP.select(material_dir(a.seed), a.checkpoints)
        state = dict(sid=None, turn_log=[], row=-1)
        for cp in selected:
            a.stop_row = cp["id"]
            if main(a, state):
                reason = "checkpoint fork changed parent" if state.get("parent_changed") else "checkpoint fork failed"
                jdump(RUNS / f"seed-{a.seed}" / a.run / "summary.json", dict(state["summary"], status="FAILED", reason=reason))
                raise SystemExit(f"FAILED: {reason}; replay fallback forbidden")
        jdump(RUNS / f"seed-{a.seed}" / a.run / "summary.json", dict(state["summary"], replay_mode="one parent", checkpoints=[c["id"] for c in selected]))
        return 0
    global USER_AUTH
    USER_AUTH = a.auth_file.resolve()

    sdir = material_dir(a.seed)
    rows, facts = jl(sdir / "transcript.jsonl"), json.loads((sdir / "facts.json").read_text())
    man = json.loads((sdir / "material.manifest.json").read_text())
    cp = CP.select(sdir, a.stop_row or "decision")[0] if (sdir / "lifecycle_probes.jsonl").exists() else None
    stop = cp["row_index"] if cp else man["decision_checkpoint"]["row_index"] if a.stop_row is None else int(a.stop_row)
    effective = max(i for i, r in enumerate(rows) if r["turn"] == rows[stop]["turn"]) if state is not None else stop
    rows = [r for i, r in enumerate(rows) if i <= effective and (not a.slice or r["turn"] <= a.slice)]
    plans = plan_turns(rows[(state["row"] + 1) if state else 0:], a.dictation)
    seed = "smoke-seed-1" if a.seed == "smoke-1" else f"seed-{a.seed}"
    rdir = (S4 / "dry-run" / (seed + ("" if a.dictation == "user" else "-dictation-file"))) if a.dry_run \
        else RUNS / seed / str(a.run)
    root = rdir
    if state is not None:
        rdir = root / f"cp-{cp['id']}"
    if a.readmit:   # offline: re-read the rollout (after an extractor fix), no model calls
        summ = json.loads((rdir / "summary.json").read_text())
        recorded = Path((summ.get("home") or {}).get("codex_home") or HOME)  # runs before #951 used one shared home
        HOME = recorded if recorded.is_dir() else HOME
        items, _ = rollout_items(summ["thread_id"])
        adm = admission(rows, facts, items)
        jdump(rdir / "admission.json", adm)
        summ["admission"] = {k: v for k, v in adm.items() if k not in ("rows", "facts")}
        summ["survival"] = survival(summ["thread_id"], facts, man, rdir, rows, root / "workspace")
        summ["beyond_declared"] = beyond_declared(rows, facts, stop, summ["survival"], man.get("continuity", []))
        jdump(rdir / "summary.json", summ)
        print(json.dumps(summ["admission"], indent=1))
        return 0
    if not a.dry_run and rdir.exists() and any(rdir.iterdir()):
        raise SystemExit(f"refusing to reuse {rdir}")
    home = setup_home() if not state or not state["sid"] else state["home"]
    if home.get("login_status") != "logged_in":
        raise SystemExit("STOP: the isolated CODEX_HOME is not authenticated; no run started")
    token_guard(3.0, "start gate")
    print("expiry ok (>= 3h)", flush=True)
    home["token_expiry_gate"] = "expiry ok (>= 3h)"
    ws = root / "workspace"
    layout = materialise(rows, ws, a.dictation)
    (rdir / "turns").mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "run": a.run, "model": MODEL, "effort": EFFORT, "codex_version": subprocess.run(
        [str(codex_bin()), "--version"], capture_output=True, text=True).stdout.strip(), "codex_bin": str(codex_bin()),
        "home": home, "dictation": a.dictation, "rows": len(rows), "turns": len(plans), "stop_row_index": rows[-1]["id"], "checkpoint": stop,
        "material_sha": man["shas"]["transcript.jsonl"], "late_corrected_value_checkpoint": CP.late_checkpoint(sdir), "workspace": str(ws), "layout": layout,
        "material_sha256": sha(sdir / "material.manifest.json"),
        "checkpoint_row": stop, "effective_row": effective, "beyond_declared": beyond_declared(rows, facts, stop, continuity=man.get("continuity", [])),
        "kit": {"path": str(KIT), "drive_codex_sha": sha(KIT / "drive_codex.py"),
                "parse_rollout_sha": sha(KIT / "parse_rollout.py")}}
    if a.dry_run:
        for p in plans:
            (rdir / "turns" / f"turn-{p['turn']:03d}.prompt.txt").write_text(p["prompt"], encoding="utf-8")
        cmds = [base_cmd("start", None, ws, [])] + [base_cmd("resume", "<THREAD_ID>", ws, [])] * (len(plans) - 1)
        if a.force_event:
            cmds.append(base_cmd("resume", "<THREAD_ID>", ws, ["-c", "model_auto_compact_token_limit=<0.6 x last input>"]))
        summary["commands"] = [" ".join(c) + f"  < turn-{i + 1:03d}.prompt.txt" for i, c in enumerate(cmds)]
        summary["probe_command"] = " ".join(base_cmd("fork", "<CHECKPOINT_THREAD_ID>", ws, [])) + "  < batch prompt"
        summary["plans"] = [{k: v for k, v in p.items() if k != "prompt"} | {"prompt_chars": len(p["prompt"])}
                            for p in plans]
        jdump(rdir / "dry-run.json", summary)
        print(json.dumps({k: summary[k] for k in ("codex_version", "home", "rows", "turns")}, indent=1))
        print("\n".join(summary["commands"][:3] + ["..."] + summary["commands"][-1:] + [summary["probe_command"]]))
        return 0

    summary["started"] = time.time()
    sid, turn_log = (state["sid"], list(state["turn_log"])) if state else (None, [])
    for p in plans:
        stem = rdir / "turns" / f"turn-{p['turn']:03d}"
        Path(f"{stem}.prompt.txt").write_text(p["prompt"], encoding="utf-8")
        res = run_cli(base_cmd("start" if sid is None else "resume", sid, ws, []), p["prompt"], ws, stem)
        sid = sid or res["thread_id"]
        turn_log.append({"turn": p["turn"], **{k: v for k, v in res.items() if k != "agent_messages"},
                         "n_notes": p["n_notes"], "note_chars": p["note_chars"], "n_calls": p["n_calls"]})
        print(f"turn {p['turn']}: rc {res['rc']} wall {res['wall_s']}s tools {res['tool_items']} "
              f"usage {res['usage']}", flush=True)
        if res["rc"] != 0 or not sid:
            summary.update(status="FAILED", failed_turn=p["turn"])
            break
    summary["thread_id"] = sid
    if cp:
        summary.update(checkpoint_id=cp["id"], stop_row_index=stop)
    if a.force_event and summary.get("status") != "FAILED":
        last_in = (PR.parse_rollout(sid, HOME / "sessions")["token_series"] or [{"last_input": 100_000}])[-1]
        limit = max(10_000, int(last_in["last_input"] * 0.6))
        stem = rdir / "turns" / "turn-checkpoint"
        prompt = "[replay driver · checkpoint] Reply with exactly: CHECKPOINT OK"
        res = run_cli(base_cmd("resume", sid, ws, ["-c", f"model_auto_compact_token_limit={limit}"]), prompt, ws, stem)
        turn_log.append({"turn": "checkpoint", "auto_compact_limit": limit, "timing_label": "WIRING-ONLY",
                         **{k: v for k, v in res.items() if k != "agent_messages"}})
        print(f"checkpoint turn: limit {limit} rc {res['rc']} wall {res['wall_s']}s", flush=True)
        if res["rc"] != 0 or res["timed_out"]:
            summary.update(status="FAILED", failed_turn="checkpoint")
    summary["turn_log"] = turn_log

    # instrument from the rollout BEFORE any probe (r3 §R5)
    cp_rollout = rollout_path(sid)
    parsed = PR.parse_rollout(sid, HOME / "sessions")
    items, ctx = rollout_items(sid)
    summary["readback_per_turn_context"] = ctx
    summary["pin_ok"] = all(c["model"] == MODEL and c["effort"] == EFFORT for c in ctx) and bool(ctx)
    summary["compactions"] = parsed["compactions"]
    summary["model_context_window"] = parsed["model_context_window"]
    summary["history_mode"] = parsed["history_mode"]
    jdump(rdir / "rollout-parse.json", parsed)
    adm = admission(rows, facts, items)
    jdump(rdir / "admission.json", adm)
    summary["admission"] = {k: v for k, v in adm.items() if k not in ("rows", "facts")}
    cont = json.loads((sdir / "continuation.json").read_text())
    summary["survival"] = survival(sid, facts, man, rdir, rows, ws)
    summary["beyond_declared"] = beyond_declared(rows, facts, stop, summary["survival"], man.get("continuity", []))
    summary["cp_rollout_sha_before_probes"] = sha(cp_rollout)
    (rdir / "sessions").mkdir(exist_ok=True)
    shutil.copy2(cp_rollout, rdir / "sessions" / cp_rollout.name)

    # probes: each batch from its OWN fork of the checkpoint thread (never sequential in one session)
    facts_by = {f["id"]: f for f in facts}
    traps = {t["id"]: t for t in json.loads((sdir / "traps.json").read_text())}
    pdir = rdir / "probe-cwd"
    pdir.mkdir(exist_ok=True)
    if (ws / "AGENTS.md").exists():   # host instructions stay in the fork's cwd; the calls/ files do not
        shutil.copy2(ws / "AGENTS.md", pdir / "AGENTS.md")
    results, forks = [], []
    batches = batches_for(sdir, cp)[: a.batches or None]
    for b in batches:  # a failed fork is re-read once from the unchanged checkpoint parent
        suffix = ".reread" if b.get("reread") else ""
        prompt = ("[probe] Answer from this conversation's memory only; do not run any tool or read any file.\n"
                  + b["text"] + "\n" + "\n".join(f"{p['id']}: {p['text']}" for p in b["probes"]))
        stem = rdir / "turns" / f"probe-{b['id']}{suffix}"
        res = run_cli(base_cmd("fork", sid, ws, []), prompt, pdir, stem)
        fsid, err = res["thread_id"], None
        rb = host_revoked = None
        if fsid:
            fr = rollout_path(fsid)
            shutil.copy2(fr, rdir / "sessions" / fr.name)
            _, fctx = rollout_items(fsid)
            rb = fctx[-1] if fctx else None
            host_revoked = "previously provided AGENTS.md instructions no longer apply" in fr.read_text()
            if not rb or rb["model"] != MODEL or rb["effort"] != EFFORT:
                err = f"READBACK MISMATCH {rb}"
        if res["tool_items"]:
            err = f"REJECTED: {res['tool_items']} tool/file retrieval item(s) during a context-only probe"
        if res["rc"] != 0 or not fsid or res["timed_out"]:
            err = err or f"fork failed rc {res['rc']}"
        answers = None if err else parse_answers(res["agent_messages"])
        if answers is None and not err:
            err = "unparseable answer"
        forks.append({"batch": b["id"], "fork_thread_id": fsid, "readback": rb, "wall_s": res["wall_s"],
                      "tool_items": res["tool_items"], "error": err, "usage": res["usage"],
                      "host_instructions_revoked_in_fork": host_revoked})
        jdump(rdir / "answers" / f"{b['id']}{suffix}.json", {"batch": b["id"], "fork": fsid, "messages": res["agent_messages"],
                                                      "error": err})
        print(f"probe {b['id']}: fork {fsid} wall {res['wall_s']}s err {err}", flush=True)
        if err and not b.get("reread"):
            batches.append(dict(b, reread=True))
            continue
        for p in b["probes"]:
            f = facts_by.get(p["id"]) or {}
            ans = (answers or {}).get(p["id"])
            results.append({
                "schema": "s4-result-v1", "kind": "probe", "probe_id": p["id"], "probe_kind": p["kind"],
                "expect": p["expect"], "gold": p.get("gold") or p.get("answer") or f.get("answer") or (traps.get(p["id"]) or {}).get("answer"),
                **{k: p[k] for k in ("checkpoint_id", "schedule", "compaction_horizon") if k in p},
                "stale": f.get("stale"), "placement": f.get("placement"), "row_role": f.get("row_role"),
                "fact_class": f.get("class"), "answer": ans, "answered": ans is not None,
                "timed_out": res["timed_out"], "error": err, "status": "ERROR" if err else "OK", "reader_rereads": int(bool(b.get("reread"))),
                "arm": "codex-native", "arm_kind": "hosted, own runtime; pinned reader on fork", "seed": seed,
                "run": a.run, "run_id": f"codex-native/{seed}/r{a.run}/{sid}", "lane": "codex-cli-subscription",
                "reader": f"{MODEL}·{EFFORT}", "reader_readback": rb, "batch_id": b["id"], "clone_id": fsid,
                "store_backed": False, "receipt": "NO STORE", "tool_calls_batch": res["tool_items"],
                "answer_wall_s_batch": res["wall_s"], "attribution": "batch", "context_format": "native codex fork",
                "timing_label": "WIRING-ONLY" if a.force_event or a.slice else "full-stream"})
    with open(rdir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary["forks"] = forks
    summary["reader_rereads"] = sum(r["reader_rereads"] for r in results)
    summary["cp_rollout_sha_after_probes"] = sha(cp_rollout)
    summary["isolation_ok"] = summary["cp_rollout_sha_before_probes"] == summary["cp_rollout_sha_after_probes"]
    summary["continuation"] = cont
    summary["user_codex_sessions_touched"] = [str(p) for p in (USER_AUTH.parent / "sessions").rglob("*.jsonl")
                                              if sid in p.name or any(f["fork_thread_id"] and f["fork_thread_id"]
                                                                      in p.name for f in forks)]
    summary["auth_copy_sha256_prefix_after"] = sha(HOME / "auth.json")[:12]
    summary["auth_refreshed_in_run"] = summary["auth_copy_sha256_prefix_after"] != home.get("auth_copy_sha256_prefix")
    summary["reader_calls"] = [dict(batch=f["batch"], calls=[dict(f["usage"], latency_s=f["wall_s"]) if f["usage"] is not None else None]) for f in forks]
    summary["finished"] = time.time()
    if not summary["pin_ok"] or not summary["isolation_ok"] or summary.get("auth_refreshed_in_run", False):
        summary["status"] = "FAILED"
    summary.setdefault("status", "COMPLETED")
    if state is not None:
        state.update(parent_changed=not summary["isolation_ok"], summary=summary)
    if state is not None and summary["status"] == "COMPLETED":
        state.update(sid=sid, turn_log=turn_log, row=effective, home=home, summary=summary)
    jdump(rdir / "summary.json", summary)
    print(json.dumps({k: summary.get(k) for k in ("status", "thread_id", "pin_ok", "isolation_ok",
                                                  "model_context_window", "history_mode")}, indent=1))
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
