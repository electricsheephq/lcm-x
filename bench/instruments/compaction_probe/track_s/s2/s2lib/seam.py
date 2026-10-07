"""Load the pinned worktree as `hermes_lcm` and own the host seam `agent.auxiliary_client.call_llm`.

Same seam as the Track P harness (`../../harness/lcmx.py`) plus two host behaviours the S2 arms need: the plugin's
`timeout` kwarg is enforced by the lane (as the host's client does) and its `reasoning_config` effort reaches the codex
lanes. Lanes are the harness adapters (`../../harness/lanes/`). The `agent.context_engine` base class is the worktree's
own `benchmarking.standalone` fallback; no Hermes code is imported.
"""
from __future__ import annotations

import hashlib, importlib, importlib.util, os, shutil, subprocess, sys, threading, time, types, uuid  # noqa: E401
from pathlib import Path

WORKTREE = Path(os.environ["S2_PRODUCT_WORKTREE"]).resolve()
PINNED = os.environ.get("S2_PRODUCT_SHA", "7ed790c84b493395bdc1c929c1ee4d7ea0eaaceb")  # S8: v0.24.8 GA
HARNESS = Path(__file__).resolve().parents[2] / "harness"
LOCK = threading.Lock()
CALLS: list[dict] = []      # every seam call since the last take()
SUMMARIES: list[dict] = []  # every summarize_with_escalation return since the last take()
_LANE = {"lane": None}

def load_engine():
    """Register the worktree as `hermes_lcm` (package __init__ not executed, as the harness and tests do)."""
    head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"worktree HEAD {head} != pinned {PINNED}")
    dirty = subprocess.run(["git", "-C", str(WORKTREE), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("product worktree has tracked changes; refusing to import")
    if "hermes_lcm" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "hermes_lcm", str(WORKTREE / "__init__.py"), submodule_search_locations=[str(WORKTREE)])
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [str(WORKTREE)]
        sys.modules["hermes_lcm"] = mod
    importlib.import_module("hermes_lcm.benchmarking.standalone").ensure_agent_context_engine_importable()
    aux = types.ModuleType("agent.auxiliary_client")
    aux.call_llm = call_llm
    sys.modules["agent.auxiliary_client"] = sys.modules["agent"].auxiliary_client = aux
    m = types.SimpleNamespace(**{n: importlib.import_module(f"hermes_lcm.{n}")
                                 for n in ("config", "engine", "tokens", "escalation", "guidance")})
    _wrap_level_recorder(m.engine)
    return m

def take():
    with LOCK:
        out = (CALLS[:], SUMMARIES[:])
        del CALLS[:], SUMMARIES[:]
        return out

def _wrap_level_recorder(engine_mod):
    """Observe (never alter) the level the summariser returns; 9edfa46 stores no level column in the DAG."""
    orig = engine_mod.summarize_with_escalation
    if getattr(orig, "_s2_wrapped", False):
        return

    def recorded(*a, **kw):
        t0, rec = time.monotonic(), {"depth": kw.get("depth"), "source_tokens": kw.get("source_tokens"),
                                     "token_budget": kw.get("token_budget"), "timeout_s": kw.get("timeout")}
        try:
            text, level = orig(*a, **kw)
            rec.update(level=level, summary_chars=len(text or ""), verbatim=level == 3 and text == kw.get("text"),
                       sha=hashlib.sha256((text or "").encode()).hexdigest())  # S8: stored level 3 = verbatim?
            return text, level
        except Exception as exc:
            rec.update(level=None, error=type(exc).__name__)
            raise
        finally:
            rec["wall_s"] = round(time.monotonic() - t0, 3)
            with LOCK:
                SUMMARIES.append(rec)
    recorded._s2_wrapped = True
    engine_mod.summarize_with_escalation = recorded

def set_lane(name: str, scratch: Path):
    sys.path.insert(0, str(HARNESS))
    lanes = importlib.import_module("lanes")
    if name == "glm":
        lane = lanes.get_lane("glm")
    else:
        lane = CodexEffortLane(name, {"codex-sol": "gpt-6.1-sol", "codex-astra": "gpt-6-astra"}[name], scratch)
    _LANE["lane"] = lane
    return lane

def call_llm(**kwargs):
    """Host stand-in: route to the lane, enforce the plugin's timeout, pass effort, record every call."""
    lane = _LANE["lane"]
    if lane is None:
        raise RuntimeError("no summariser lane bound (dry run makes no model calls)")
    msgs = kwargs.get("messages") or []
    system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
    user = [m["content"] for m in msgs if m.get("role") != "system"]
    effort, timeout = (kwargs.get("reasoning_config") or {}).get("effort"), kwargs.get("timeout")
    rec = {"task": kwargs.get("task"), "max_tokens": kwargs.get("max_tokens"), "temperature": kwargs.get("temperature"),
           "timeout_s": timeout, "effort": effort, "model_kw": kwargs.get("model"), "t_start": time.time(),
           "prompt_chars": sum(len(m.get("content") or "") for m in msgs)}
    t0 = time.monotonic()
    try:
        if isinstance(lane, CodexEffortLane):
            reply = lane.call_with(system, user, timeout, effort)
        else:  # GLM adapter: its module-level socket timeout = the plugin's per-request timeout for this call
            glm = sys.modules["lanes.glm"]
            with LOCK:
                saved, glm.CALL_TIMEOUT_S = glm.CALL_TIMEOUT_S, float(timeout or glm.CALL_TIMEOUT_S)
                try:
                    reply = lane.call(system, user, kwargs.get("max_tokens"))
                finally:
                    glm.CALL_TIMEOUT_S = saved
        rec.update(reply.meta, reply_chars=len(reply.text or ""))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=reply.text))])
    except Exception as exc:  # recorded, then re-raised so the plugin takes its own L2/L3 path
        rec.update(exception=type(exc).__name__, error=str(exc)[:300])
        raise
    finally:
        rec["latency_s"] = round(time.monotonic() - t0, 2)
        with LOCK:
            CALLS.append(rec)

class CodexEffortLane:
    """The harness `codex exec` command shape with the plugin's effort (else the lane default medium) and timeout;
    scratch under the run dir instead of `<D>/eval/scratch`. Calls are sequential (one engine per process)."""

    def __init__(self, name, model, scratch: Path):
        self.name, self.model, self.scratch = name, model, scratch.resolve()

    def call_with(self, system, user_parts, timeout, effort):
        lanes = sys.modules["lanes"]
        base = self.scratch / uuid.uuid4().hex[:12]
        (work := base / "work").mkdir(parents=True)
        prompt, out = base / "prompt.txt", base / "last-message.txt"
        prompt.write_text(lanes.flat_prompt(system, user_parts))
        eff, limit = effort or "medium", float(timeout or lanes.CALL_TIMEOUT_S)
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-m", self.model,
               "-c", f"model_reasoning_effort={eff}", "--output-last-message", str(out), "-"]
        t0 = time.monotonic()
        try:
            with prompt.open() as stdin:
                p = subprocess.run(cmd, cwd=work, stdin=stdin, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            raise lanes.LaneError(f"timeout after {limit}s", limit)
        dt = round(time.monotonic() - t0, 2)
        if p.returncode != 0:
            raise lanes.LaneError(f"codex exit {p.returncode}", dt)
        text = out.read_text() if out.exists() else ""
        shutil.rmtree(base, ignore_errors=True)
        return lanes.Reply(text, {"lane_model": self.model, "lane_effort": eff, "latency_s": dt,
                                  "max_tokens_enforced": False})
