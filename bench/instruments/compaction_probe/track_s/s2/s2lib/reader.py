"""Pinned readers, separate from the summariser lane (r3 §R1): `astra-low` = codex exec gpt-6-astra at effort low
(read back from the CLI header on every call), `glm` = glm-5.3 (model read back from each response). `-open` arms get a
JSON tool protocol over the PUBLIC LCM tools, dispatched through `LCMEngine.handle_tool_call` -> tools.py on a clone."""
from __future__ import annotations

import json, re, shutil, subprocess, sys, time, urllib.error, urllib.request, uuid  # noqa: E401
from pathlib import Path

READER_TIMEOUT_S, RUNAWAY_GUARD = 600, 20
PUBLIC_TOOLS = ("lcm_grep", "lcm_expand_query", "lcm_describe", "lcm_recall")
# Windows for the C0 UNAVAILABLE rule only (plugin token counter, minus an answer reserve). glm = ASSUMPTION.
READER_WINDOW, ANSWER_RESERVE = {"astra-low": 272_000, "glm": 200_000}, 16_000

def flatten(system: str, view: list[dict]) -> str:
    """Role-labelled text of the assembled context (codex reader; GLM fallback when the API rejects a row shape)."""
    out = [f"[SYSTEM]\n{system}"] if system else []
    for m in view:
        c = m.get("content")
        c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        calls = "".join(f"\n[tool call {tc.get('id')}: {tc.get('function', {}).get('name')}"
                        f"({tc.get('function', {}).get('arguments')})]" for tc in (m.get("tool_calls") or []))
        tag = m.get("role", "?").upper() + (f" {m['tool_call_id']}" if m.get("tool_call_id") else "")
        out.append(f"[{tag}]\n{c}{calls}")
    return "\n\n".join(out)

def _convo(turns, replies):
    return [x for i, t in enumerate(turns) for x in [{"role": "user", "content": t}]
            + ([{"role": "assistant", "content": replies[i]}] if i < len(replies) else [])]

class GLMReader:
    def __init__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
        from lanes import glm  # GLM_API_KEY from the environment only
        self._glm, self._k = glm, glm._key()
        self.readback = {"model": None, "effort": "n/a (glm-5.3 lane sends no effort input)"}

    def _post(self, msgs):
        req = urllib.request.Request(self._glm.URL, method="POST", data=json.dumps(
            {"model": self._glm.MODEL, "messages": msgs, "max_tokens": 8192}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._k})
        with urllib.request.urlopen(req, timeout=READER_TIMEOUT_S) as r:
            data = json.loads(r.read().decode())
        self.readback["model"] = data.get("model")
        if self.readback["model"] != self._glm.MODEL:
            self.readback["pin_ok"] = False  # retained even if a later response matches
            raise RuntimeError(f"reader pin mismatch: requested {self._glm.MODEL}, read back {self.readback['model']}")
        return (data["choices"][0]["message"].get("content") or ""), data.get("usage")

    def ask(self, system, view, turns, replies):
        native = ([{"role": "system", "content": system}] if system else []) + [
            {**{k: m[k] for k in ("role", "tool_call_id", "tool_calls") if k in m},
             "content": m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content") or "")}
            for m in view]
        try:
            text, usage = self._post(native + _convo(turns, replies))
            fmt = "native-roles"
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            text, usage = self._post([{"role": "user", "content": flatten(system, view)}] + _convo(turns, replies))
            fmt = "flattened (native rejected: HTTP 400)"
        return text, {"context_format": fmt, "usage": usage}

class AstraLowReader:
    MODEL, EFFORT = "gpt-6-astra", "low"

    def __init__(self, scratch: Path):
        self.scratch, self.readback = scratch.resolve(), {"model": None, "effort": None}

    def ask(self, system, view, turns, replies):
        convo = "".join(f"\n\n[{m['role'].upper()}]\n{m['content']}" for m in _convo(turns, replies))
        base = self.scratch / uuid.uuid4().hex[:12]
        (work := base / "work").mkdir(parents=True)
        prompt, out = base / "prompt.txt", base / "last.txt"
        prompt.write_text("Answer from the conversation below only; use no shell or file tools.\n\n"
                          + flatten(system, view) + convo)
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-m", self.MODEL,
               "-c", f"model_reasoning_effort={self.EFFORT}", "--output-last-message", str(out), "-"]
        with prompt.open() as stdin:
            p = subprocess.run(cmd, cwd=work, stdin=stdin, capture_output=True, text=True, timeout=READER_TIMEOUT_S)
        for key, pat in (("model", r"^model:\s*(\S+)"), ("effort", r"^reasoning effort:\s*(\S+)")):
            m = re.search(pat, (p.stderr or "") + (p.stdout or ""), re.M)
            self.readback[key] = m.group(1) if m else "UNREAD"
        if (self.readback["model"], self.readback["effort"]) != (self.MODEL, self.EFFORT):
            self.readback["pin_ok"] = False
            raise RuntimeError(f"reader pin mismatch: read back {self.readback}")
        if p.returncode != 0:
            raise RuntimeError(f"codex reader exit {p.returncode}")
        text = out.read_text() if out.exists() else ""
        shutil.rmtree(base, ignore_errors=True)
        return text, {"context_format": "flattened (codex exec)"}

def parse_json_obj(text: str):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.JSONDecoder(strict=False).raw_decode(t[t.index("{"):])[0]
    except (ValueError, json.JSONDecodeError):
        return None

def _answers(obj):
    return {str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in obj.items()} if isinstance(obj, dict) else None

def answer(reader, system, view, batch_prompt, tools_engine=None, schemas=None, guidance="", count_tokens=len):
    """One batch answer -> (answers|None, meta). With `tools_engine` the reader may call the public tools; calls,
    tokens read back and wall are attributed to the BATCH (labelled, not divided)."""
    t0 = time.monotonic()
    meta = {"tool_calls": 0, "tool_log": [], "tokens_read_back": 0, "runaway_guard_hit": False}
    if tools_engine is None:
        text, info = reader.ask(system, view, [batch_prompt], [])
        if info.get("usage") is not None:
            info["usage"].setdefault("latency_s", time.monotonic() - t0)
        meta.update(info, wall_s=round(time.monotonic() - t0, 2), raw_reply=text, reader_calls=[info.get("usage")])
        return _answers(parse_json_obj(text)), meta
    turns, replies = [("You may search the compacted conversation history with these tools before answering:\n"
                       + json.dumps(schemas, ensure_ascii=False) + "\n\nRecall guidance:\n" + guidance
                       + "\n\nTo call a tool reply with ONLY {\"tool\": \"<name>\", \"args\": {...}}. When done reply "
                         "with ONLY {\"answer\": {<probe id>: <answer string>}}.\n\n" + batch_prompt)], []
    meta["reader_calls"] = []  # S7 D7: per-call usage (prompt/completion tokens as the lane reports them)
    while True:
        text, info = reader.ask(system, view, turns, replies)
        meta["reader_calls"].append(info.get("usage"))
        obj = parse_json_obj(text)
        if not (isinstance(obj, dict) and obj.get("tool")):
            break
        replies.append(text)
        if meta["tool_calls"] >= RUNAWAY_GUARD:
            meta["runaway_guard_hit"] = True
            turns.append("Tool budget exhausted (20 calls). Reply now with ONLY {\"answer\": {...}}.")
            text, info = reader.ask(system, view, turns, replies)
            meta["reader_calls"].append(info.get("usage"))
            break
        meta["tool_calls"] += 1
        c0 = time.monotonic()
        try:
            result = (tools_engine.handle_tool_call(obj["tool"], obj.get("args") or {})
                      if obj["tool"] in PUBLIC_TOOLS else json.dumps({"error": "not a public tool"}))
        except Exception as exc:
            result = json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]})
        meta["tokens_read_back"] += count_tokens(result)
        meta["tool_log"].append({"tool": obj["tool"], "args": obj.get("args"), "result_chars": len(result),
                                 "wall_s": round(time.monotonic() - c0, 2)})
        turns.append(f"Tool result for {obj['tool']}:\n{result}")
    meta.update(info, wall_s=round(time.monotonic() - t0, 2), raw_reply=text, attribution="batch")
    obj = parse_json_obj(text)
    return _answers(obj.get("answer") if isinstance(obj, dict) and isinstance(obj.get("answer"), dict) else obj), meta
