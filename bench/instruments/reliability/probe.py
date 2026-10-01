"""One phase of one reliability cell, run by the HOST python with cwd = the host source tree.

Imports only the stdlib and host modules; the plugin under test loads through the Hermes plugin
loader from ``HERMES_HOME/plugins/``. The provider client is a MagicMock scripted per turn, the
host aux LLM and the LCM summariser are stubbed, and sockets are blocked. Generalises ``_PROBE`` /
``_CRASH_PROBE`` in tests/test_real_turn_loop_acp_override.py. Every host shape it emulates is
cited as ``file:line`` in the current host tree (``citations`` in phase-<X>.json).

Last stdout line: ``{"exit": done|crash|clean_exit|tip_switch|unsupported, "next_turn": N, ...}``.
"""
import argparse
import hashlib
import inspect
import io
import ipaddress
import json
import logging
import os
import pwd
import re
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ANCHORS = {  # shape -> (host file, text on the cited line)
    "acp_persist": ("acp_adapter/server.py", "persist_user_message=user_text"),
    "acp_strip": ("acp_adapter/server.py", "user_text = _extract_text(prompt).strip()"),
    "acp_restore": ("acp_adapter/session.py", "get_messages_as_conversation(session_id, repair_alternation=True)"),
    "acp_cancel": ("acp_adapter/server.py", "request_hard_interrupt(state.agent)"),
    "acp_retry": ("acp_adapter/server.py", "def _attach_interrupted_prompt"),
    "acp_compress": ("acp_adapter/commands.py", "def _cmd_compress"),
    "gateway_transcript": ("gateway/session_transcript.py", "def load_transcript"),
    "gateway_user_text": ("gateway/run_turn.py", "def _hmwa_apply_message_timestamp"),
    "gateway_run": ("gateway/run_turn_runner.py", "return agent.run_conversation(api_message"),
    "user_merge": ("agent/agent_runtime_helpers.py", "def _merge_consecutive_users"),
    "orphan_drop": ("agent/agent_runtime_helpers.py", "def _drop_stray_tool_results"),
    "rotation_start": ("agent/conversation_compression.py", "def _notify_context_engine_compression_complete"),
    "rotation_end": ("agent/conversation_compression.py", "agent.commit_memory_session(messages)"),
    "session_transition": ("run_agent.py", "def _transition_context_engine_session"),
    "summary_aborted": ("agent/context_compressor.py", '"summary_generation_aborted"'),
    "cron_agent": ("cron/scheduler.py", 'platform="cron"'),
    "cron_close": ("cron/scheduler.py", "agent.close()"),
    "commit_telemetry": ("agent/conversation_compression.py", "context compression attempt telemetry: %s"),
    "failed_turn_notice": ("agent/turn_failure_copy.py", "FAILED_TURN_NOTICE = ("),
    "tool_dispatch": ("model_tools.py", "def handle_function_call("),
    "gateway_key_start": ("agent/agent_init.py", 'conversation_id=getattr(agent, "_gateway_session_key", None)'),
    "gateway_key_rotation": ("agent/conversation_compression.py", 'conversation_id=getattr(agent, "_gateway_session_key", None)'),
    "engine_tool_dispatch": ("agent/tool_executor.py", "agent.context_compressor.handle_tool_call(function_name"),
}
# Host modules whose executed file must lie in the verified host tree (import provenance).
HOST_PREFIXES = ("run_agent", "hermes_state", "hermes_cli", "hermes_constants", "agent", "model_tools", "tools",
                 "acp_adapter", "gateway", "cron")
FAULT_CITES = {  # the host code path each fault or scenario relies on; uncitable -> the cell is UNSUPPORTED
    "crash_after_compaction_before_reply": ["user_merge"],
    "crash_mid_tool_call": ["user_merge"],
    "crash_after_rotation_before_child_row": ["rotation_start"],
    "crash_between_session_end_and_start": ["rotation_end", "rotation_start"],
    "cancel_then_retry": ["acp_cancel", "acp_retry"],
}
COMMITTED = '"commit_status":"committed"'
LOG_COUNTS = {
    "publication_invariant_conflict": "publication_invariant_conflict",
    "commit_logged": "as a compaction commit",
    "summary_generation_aborted": "summary_generation_aborted",
    "native_unusable": "native recovery did not produce a usable summary",
    "orphan_log": "orphaned tool result",
    "resident_engine_conflict": "resident_engine_conflict",
    "skipped_ingest_resident_conflict": "skipped ingest: stable engine use ended with resident_engine_conflict",
    "recorded_replaced": "LCM recorded host-replaced rows",
    "survival_fit": "LCM survival fit applied",
    "exit_fit": "LCM survival fit applied (reason=exit_fit:",
    "exit_fit_skipped": "LCM exit fit skipped",
}
FILLER = "alpha beta gamma delta "


def _log_counts(log: str) -> dict:
    counts = {k: log.count(v) for k, v in LOG_COUNTS.items()}
    # #668: routine threshold headroom is not a compaction miss; an exit over the window budget logs as a plain fit
    counts["survival_fit"] -= counts["exit_fit"]
    return counts


def phase_log_fields(log: str) -> dict:
    """The log-derived phase fields, computed before the phase JSON is written: the counts, and (for
    native-on-off, the pre_publication_counts diagnostic) the counts after this phase's first publication."""
    first_commit = log.find("LCM compaction #")
    return {"compactions_logged": len(re.findall(r"LCM compaction #\d+", log)),
            "log_counts": _log_counts(log),
            "log_counts_after_commit": None if first_commit < 0 else {
                k: v for k, v in _log_counts(log[first_commit:]).items()
                if k in ("publication_invariant_conflict", "survival_fit")}}


def provenance(cell, extra=()):
    """Where every loaded host module (and each cited, not yet loaded one) was executed from; a file outside the
    verified host tree, or the plugin outside its exported tree, is a violation."""
    import importlib.util
    src, tree = Path(cell["host_src"]).resolve(), Path(cell["plugin"]["tree"]).resolve()
    mods, bad = {}, []
    names = [n for n in list(sys.modules) if n.split(".")[0] in HOST_PREFIXES or n.startswith("hermes_plugins.")]
    for name in names + [n for n in extra if n not in sys.modules]:
        try:
            origin = getattr(sys.modules.get(name), "__file__", None) or (importlib.util.find_spec(name) or SimpleNamespace(origin=None)).origin
        except (ImportError, ValueError):
            origin = None
        if not origin or origin in ("built-in", "frozen"):
            continue
        mine = name == cell["plugin"]["module"] or name.startswith(cell["plugin"]["module"] + ".")
        path, root = Path(origin).resolve(), tree if mine else src
        mods[name] = str(path)
        if root not in path.parents:
            bad.append(f"{name} -> {path}")
    exe_ok = Path(sys.executable).absolute() == Path(cell["host_python"]).absolute()
    if not exe_ok:
        bad.append(f"sys.executable {sys.executable} != {cell['host_python']}")
    return {"executable": sys.executable, "version": sys.version.split()[0], "modules": len(mods),
            "core": {k: mods.get(k) for k in ("run_agent", "hermes_state", "hermes_cli.plugins", "model_tools",
                                              "agent.context_compressor", "agent.conversation_compression")},
            "cited": {k: mods.get(k) for k in extra}, "violations": bad[:20]}


def check_result(name, args, result):
    """(ok, detail, chars) for one executed tool call: no error payload, and read_file returned the whole file."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and (data.get("error") or data.get("success") is False):
        return False, f"error: {str(data.get('error'))[:160]}", len(text)
    if name == "read_file":
        path = Path(str((args or {}).get("path", "")))
        want = len(path.read_text().splitlines()) if path.is_file() else None
        if not isinstance(data, dict) or data.get("truncated") or want is None or data.get("total_lines") != want:
            return False, f"read_file result incomplete (total_lines {(data or {}).get('total_lines')} of {want})", len(text)
    return True, "", len(text)


def cite(key):
    rel, needle = ANCHORS[key]
    try:
        lines = Path(rel).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return next((f"{rel}:{i}" for i, line in enumerate(lines, 1) if needle in line), None)


def refusal(cell_dir):
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    for name in ("HERMES_HOME", "HOME"):
        value = os.environ.get(name)
        path = Path(value).resolve() if value else None
        if path is None or path == real_home or path == real_home / ".hermes" or real_home / ".hermes" in path.parents:
            return f"{name}={value!r} is unset or resolves to the real home or under ~/.hermes"
    resolved = Path(cell_dir).resolve()
    if any(str(resolved) == p or str(resolved).startswith(p + "/") for p in ("/tmp", "/private/tmp")):
        return f"--cell-dir {resolved} is under /tmp"
    for key, value in os.environ.items():  # an LCM path override may only point inside this cell
        if key.upper().startswith("LCM_") and re.search(r"(_PATH|_DIR|_HOME|_FILE)$", key, re.I) and value:
            if resolved not in Path(value).expanduser().resolve().parents:
                return f"{key}={value!r} points outside the cell dir"
    return None


def user_text(cell, prefix, t):
    ut = cell["user_text"]
    if t in ut.get("continue_turns", []):
        return "continue"
    n = int(ut.get("identical_turns", {}).get(str(t), t))
    sep = "\n\n" if ut.get("separator_turns") == "all" or n in ut.get("separator_turns", []) else ""
    body = (FILLER * ut["repeat"]).rstrip()
    if sep:  # >=64 paragraph separators inside the prompt (#545)
        words = body.split(" ")
        step = max(1, len(words) // 70)
        body = sep.join(" ".join(words[i:i + step]) for i in range(0, len(words), step))
    text = f"[{prefix}{n:02d}] user turn {n}: {body} end."
    if t in ut.get("edge_ws_turns", []):
        text = "  " + text + " \n"
    return text + ("\n" if ut.get("trailing_ws") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--start-turn", type=int, default=1)
    ap.add_argument("--cell-dir", required=True)
    a = ap.parse_args()
    if (why := refusal(a.cell_dir)) is not None:
        print(json.dumps({"exit": "refused", "reason": why}), flush=True)
        sys.exit(3)
    cell = json.loads(Path(a.cell).read_text())
    cell_dir, phase, first = Path(a.cell_dir), a.phase, a.start_turn
    switched = cell_dir / "faults-fired.jsonl"  # native-on-off: the older ref is the plugin until the switch
    if cell.get("from_plugin") and not (switched.exists() and "plugin_switch" in switched.read_text()):
        cell = {**cell, "plugin": cell["from_plugin"]}
    out = {"phase": phase, "start_turn": first, "citations": {k: cite(k) for k in ANCHORS}}
    tfile = open(cell_dir / "transcript.jsonl", "a", encoding="utf-8")
    buf = io.StringIO()
    counters = {"compacted_turns": [], "lcm_tool_calls": 0, "orphan_drops": 0, "native_max": 0, "failed": []}

    lock = threading.Lock()

    def event(**ev):
        with lock:
            tfile.write(json.dumps({"phase": phase, **ev}) + "\n")
            tfile.flush()
            os.fsync(tfile.fileno())
    cited_modules = []

    def finish(exit_kind, **extra):
        if "host_src" in cell:  # import provenance: fail closed on any host module run from outside the host tree
            out["provenance"] = provenance(cell, cited_modules)
            if out["provenance"]["violations"] and exit_kind not in ("unsupported", "refused"):
                exit_kind, extra = "error", {"reason": "import provenance: " + "; ".join(out["provenance"]["violations"][:3])}
        log = buf.getvalue()
        out.update(exit=exit_kind, **extra, counters=counters, **phase_log_fields(log), session_count=session_count())
        (cell_dir / f"phase-{phase}.json").write_text(json.dumps(out, indent=1, default=str))
        (cell_dir / f"probe-{phase}.hermes.log").write_text("\n".join(
            line for line in log.splitlines() if "LCM" in line or "WARNING" in line or "ERROR" in line
            or "compress" in line.lower() or "orphan" in line)[-2_000_000:])
        print(json.dumps({"exit": exit_kind, **extra}), flush=True)
        tfile.close()
        if exit_kind in ("crash", "clean_exit", "tip_switch", "plugin_switch"):  # between turns, as _CRASH_PROBE
            os._exit(0)  # the host process dies here: no atexit, no flush, no engine shutdown

    faults = {f["kind"]: f for f in cell.get("faults", [])}
    fired_path = cell_dir / "faults-fired.jsonl"
    fired = {json.loads(x)["kind"] for x in fired_path.read_text().splitlines()} if fired_path.exists() else set()

    def fire(kind, turn, **extra):
        with open(fired_path, "a") as fh:
            fh.write(json.dumps({"kind": kind, "phase": phase, "turn": turn}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        fired.add(kind)
        out["fired"] = out.get("fired", []) + [kind]
        event(turn=turn, event="crash" if kind.startswith("crash") else kind, fault=kind, session_prefix=cur.get("prefix", "T"),
              host_commits=buf.getvalue().count(COMMITTED) - cur["commits0"], **extra)

    needed = {"acp": ["acp_persist", "acp_strip"] + (["acp_restore"] if phase != "A" else []),
              "gateway": ["gateway_transcript", "gateway_user_text", "gateway_run", "gateway_key_start",
                          "gateway_key_rotation"]}[cell["transport"]]
    needed += [k for f in faults for k in FAULT_CITES.get(f, [])]
    needed += ["orphan_drop"] if cell.get("tool_plan") else []
    needed += ["commit_telemetry", "summary_aborted"] if cell.get("native_recovery") else []
    needed += ["acp_compress"] if cell.get("final_compaction_check", True) else []
    needed += ["cron_agent", "cron_close"] if cell.get("cron_every") else []
    needed += ["tool_dispatch", "engine_tool_dispatch"] if cell.get("tool_plan") else []
    cited_modules += sorted({ANCHORS[k][0][:-3].replace("/", ".") for k in needed})
    if missing := [k for k in needed if not out["citations"][k]]:
        finish("unsupported", reason=f"host shape not citable at this sha: {missing}")
        return

    guard_sockets(local_ok=False)
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    from hermes_state import SessionDB
    from run_agent import AIAgent
    import agent.context_compressor as host_cc
    import model_tools
    if (early := provenance(cell, cited_modules)["violations"]) if "host_src" in cell else None:
        finish("error", reason="import provenance: " + "; ".join(early[:3]))
        return
    try:  # the host's own failed-turn boundary copy: the only assistant row the host may write on its own
        from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
        notices = [FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE] if out["citations"]["failed_turn_notice"] else []
    except ImportError:
        notices = []
    out["failed_turn_notices"] = notices
    home = Path(os.environ["HERMES_HOME"])
    files = cell_dir / "files"
    files.mkdir(exist_ok=True)
    (files / "small.txt").write_text("small deterministic file\n")
    (files / "big.txt").write_text("".join(f"line {i:05d}: " + FILLER * 8 + "\n" for i in range(cell.get("big_lines", 400))))

    def aux_llm(**kwargs):
        text = "## Goal\nstub\n## Progress\nstub" if kwargs.get("task") == "compression" else "Title"
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="aux", usage=None)
    import agent.title_generator as host_tg
    host_tg.call_llm = aux_llm
    host_cc.call_llm = aux_llm
    n_summ = {"c": 0}

    def summarise(*args, **kw):  # tag-preserving; "U05" never collides with a "[T05]" user tag
        n_summ["c"] += 1
        text = kw.get("text") if "text" in kw else (args[0] if args else "")
        tags = sorted(set(re.findall(r"\[([A-Z]\d{2,3})\] user", text or "")))
        return f"Stub summary #{n_summ['c']} covers " + " ".join("U" + x for x in tags) + ".\nExpand for details about: stub", 1

    window = int(cell["window"])
    cur = {"turn": 0, "step": 0, "native": 0, "sess": "S0", "ended": None, "final": False, "commits0": 0,
           "issued": set(), "seen": set()}

    # A real gateway builds its agent with the chat's stable session key (gateway/run_turn_runner.py
    # ``gateway_session_key=ctx.session_key``); the host forwards it to the engine as conversation_id.
    gw_key = f"rel:{re.sub(r'[^A-Za-z0-9_.-]+', '__', cell['id'])}:chat1" if cell["transport"] == "gateway" else None
    if gw_key and "gateway_session_key" not in inspect.signature(AIAgent.__init__).parameters:
        finish("unsupported", reason="AIAgent takes no gateway_session_key at this host sha")
        return
    out["gateway_session_key"] = gw_key

    def build(session_id, platform, key=None):
        with patch("agent.process_bootstrap.OpenAI"):
            ag = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, session_db=SessionDB(db_path=home / "state.db"), session_id=session_id,
                         skip_context_files=True, skip_memory=True, platform=platform,
                         enabled_toolsets=cell.get("toolsets", ["todo", "context_engine", "file"]),
                         **({"gateway_session_key": key} if key else {}))
        ag.client, ag.tool_delay, ag.save_trajectories = MagicMock(), 0, False
        ag.compression_in_place = bool(cell["in_place"])
        ag._compression_feasibility_checked = True
        before = getattr(ag.context_compressor, "context_length", None)
        ag.context_compressor.update_model("test/model", window, base_url="https://openrouter.ai/api/v1",
                                           api_key="k", provider="openrouter")
        out["context_length"] = {"host_resolved": before, "set_via_update_model": window}
        return ag

    sdb_read = SessionDB(db_path=home / "state.db")
    sid = "S0"
    if phase != "A" and cell["transport"] == "gateway":  # a restarted gateway binds the durable tip
        sid = sdb_read.get_compression_tip("S0") or "S0"
    agent = build(sid, "acp", gw_key)
    out["platform"] = "acp"
    engine = agent.context_compressor
    out["engine"] = getattr(engine, "name", None)
    if out["engine"] != cell["plugin"]["engine"]:
        finish("error", reason=f"engine {out['engine']!r} is not the plugin under test")
        return
    for name, module in list(sys.modules.items()):
        if name.startswith("hermes_plugins.") and hasattr(module, "summarize_with_escalation"):
            module.summarize_with_escalation = summarise
    lcm_db = home / "lcm.db"

    def depth0():
        try:
            con = sqlite3.connect(f"file:{lcm_db}?mode=ro", uri=True)
            try:
                return con.execute("SELECT COUNT(*) FROM summary_nodes WHERE depth = 0 AND source_type = 'messages'"
                                   " AND session_id NOT LIKE 'cron\\_job\\_%' ESCAPE '\\'").fetchone()[0]  # the chat lineage
            finally:
                con.close()
        except sqlite3.Error:
            return None

    etype = type(engine)
    orig_compress, orig_tool = etype.compress, etype.handle_tool_call
    orig_start, orig_end = etype.on_session_start, getattr(etype, "on_session_end", None)

    def traced_compress(self, messages, *args, **kwargs):
        result = orig_compress(self, messages, *args, **kwargs)
        status = getattr(self, "_last_compression_status", None)
        self._probe_calls, self._probe_status = getattr(self, "_probe_calls", 0) + 1, status  # per-call evidence
        if status in ("compacted", "host_native"):  # a committed pass: LCM's own or the host-native summary
            counters["compacted_turns"].append(cur["turn"])
        event(turn=cur["turn"], event="compaction", session=getattr(self, "_session_id", None), final=cur["final"],
              session_prefix=cur.get("prefix", "T"),
              compression_status=status, noop_reason=getattr(self, "_last_compression_noop_reason", None),
              depth0_nodes=depth0() if status == "compacted" else None,
              rejection=rejection(self), native_attempts=cur["native"])
        return result
    etype.compress = traced_compress

    depth = threading.local()

    def dispatched(run, name, args, call_id, via):
        """Record one real host tool execution (outermost hook only) and whether its result is a success."""
        depth.n = getattr(depth, "n", 0) + 1
        try:
            result = run()
        except BaseException as exc:
            if depth.n == 1:
                event(turn=cur["turn"], event="tool_dispatch", tag=f"{cur.get('prefix', 'T')}{cur['turn']:02d}", id=call_id,
                      name=name, args=args, via=via, ok=False, detail=f"raised {exc!r}"[:200], chars=0)
            raise
        finally:
            depth.n -= 1
        if depth.n == 0:
            ok, detail, chars = check_result(name, args, result)
            event(turn=cur["turn"], event="tool_dispatch", tag=f"{cur.get('prefix', 'T')}{cur['turn']:02d}", id=call_id,
                  name=name, args=args, via=via, ok=ok, detail=detail, chars=chars)
        return result

    def traced_tool(self, name, args, **kwargs):
        counters["lcm_tool_calls"] += 1
        f = faults.get("crash_mid_tool_call")
        if f and "crash_mid_tool_call" not in fired and cur["turn"] == f["turn"]:
            fire("crash_mid_tool_call", cur["turn"], tool=name)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        return dispatched(lambda: orig_tool(self, name, args, **kwargs), name, args, None, out["citations"]["engine_tool_dispatch"])
    etype.handle_tool_call = traced_tool
    orig_hfc = model_tools.handle_function_call

    def traced_hfc(*a, **kw):
        name, args = (a[0] if a else kw.get("function_name")), (a[1] if len(a) > 1 else kw.get("function_args"))
        return dispatched(lambda: orig_hfc(*a, **kw), name, args, kw.get("tool_call_id"), out["citations"]["tool_dispatch"])
    model_tools.handle_function_call = traced_hfc

    def traced_start(self, session_id, *args, **kwargs):
        rotation = kwargs.get("boundary_reason") == "compression"
        if rotation and cur["ended"] and "crash_between_session_end_and_start" in faults \
                and "crash_between_session_end_and_start" not in fired:
            fire("crash_between_session_end_and_start", cur["turn"], old=cur["ended"], new=session_id)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        cur["ended"] = None
        result = orig_start(self, session_id, *args, **kwargs)
        if rotation and "crash_after_rotation_before_child_row" in faults and \
                "crash_after_rotation_before_child_row" not in fired:
            fire("crash_after_rotation_before_child_row", cur["turn"], new=session_id)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        return result
    etype.on_session_start = traced_start
    if orig_end is not None:
        def traced_end(self, session_id, *args, **kwargs):
            cur["ended"] = session_id
            return orig_end(self, session_id, *args, **kwargs)
        etype.on_session_end = traced_end

    native_cls = getattr(host_cc, "ContextCompressor", None)
    if native_cls is not None and native_cls is not etype:
        orig_native = native_cls.compress

        def counted_native(self, *args, **kwargs):
            cur["native"] += 1
            return orig_native(self, *args, **kwargs)
        native_cls.compress = counted_native
    import agent.agent_runtime_helpers as helpers
    passes = getattr(helpers, "_SEQUENCE_REPAIR_PASSES", None)
    drop = getattr(helpers, "_drop_stray_tool_results", None)
    if passes and drop in passes:
        def counted_drop(messages):
            kept, n = drop(messages)
            counters["orphan_drops"] += n
            return kept, n
        helpers._SEQUENCE_REPAIR_PASSES = tuple(counted_drop if p is drop else p for p in passes)
    else:
        out["orphan_hook"] = "unavailable at this host sha; B6 orphan count falls back to the log"

    pf = faults.get("publication_failure")
    if pf:
        mod = sys.modules.get(cell["plugin"]["module"] + ".lifecycle_state")
        lifecycle = getattr(engine, "_lifecycle", None)
        err = getattr(mod, "LifecyclePublicationConflictError", None)
        if lifecycle is None or err is None:
            finish("unsupported", reason="plugin tree has no lifecycle publication stage to inject into")
            return
        orig_stage, stages = lifecycle.stage_compaction_publication, {"n": 0}

        def inject(conn, conversation_id, session_id, *args, **kwargs):
            stages["n"] += 1
            # rotation_child: only the first child publication (#541's bar), unless the cell asks for every one
            child = pf["where"] == "rotation_child" and session_id != "S0" and (pf.get("persistent") or pf["kind"] not in fired)
            if child or pf["where"] == f"pass_{stages['n']}":
                if pf["kind"] not in fired:
                    fire(pf["kind"], cur["turn"], where=pf["where"])
                raise err(f"injected publication failure ({pf['where']})")
            return orig_stage(conn, conversation_id, session_id, *args, **kwargs)
        lifecycle.stage_compaction_publication = inject

    asst = cell["assistant"]
    plan = {}
    for group in cell.get("tool_plan", []):  # "restart": the first turn of every phase after A (the merge turn)
        for t in ([first] if phase != "A" else []) if group["turns"] == "restart" else group["turns"]:
            plan.setdefault(t, []).append(group["calls"])

    def response(content, prompt_tokens, calls=None, t=0, key=""):
        ids = [f"call_{key}_{k}" for k in range(len(calls or []))]
        cur["issued"] |= set(ids)
        tcs = [SimpleNamespace(id=ids[k], type="function", function=SimpleNamespace(
            name=c["name"], arguments=json.dumps(c.get("args", {})).replace("{files}", str(files))))
            for k, c in enumerate(calls or [])] or None
        for k, tc in enumerate(tcs or []):  # the planned call, independent of what the host does with it
            event(turn=t, event="tool_issue", tag=key.rsplit("_", 1)[0], id=tc.id, name=tc.function.name,
                  args=json.loads(tc.function.arguments), expect=(calls[k].get("expect") or {}))
        msg = SimpleNamespace(content="" if tcs else content, tool_calls=tcs)
        r = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")],
                            model="test/model")
        r.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=20, total_tokens=prompt_tokens + 20)
        return r

    def reply_text(prefix, t):
        if asst.get("mode") == "repeat-identical" and t in asst.get("repeat_turns", []):
            return "noted, the same as before."
        return f"reply to {prefix}{t:02d}: noted item {t}."

    def scripted(ag, prefix, t, est, cancel=False):
        def provider(*_a, **kw):
            if phase == "A" and prefix == "T" and "crash_after_compaction_before_reply" in faults and \
                    "crash_after_compaction_before_reply" not in fired and t in counters["compacted_turns"]:
                fire("crash_after_compaction_before_reply", t)
                finish("crash", next_turn=t + 1, turn=t)
            step, cur["step"] = cur["step"], cur["step"] + 1
            for m in kw.get("messages") or []:  # a tool result as the model receives it (post host transform)
                cid = m.get("tool_call_id") if m.get("role") == "tool" else None
                if cid in cur["issued"] and cid not in cur["seen"]:
                    cur["seen"].add(cid)
                    body = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                    event(turn=t, event="tool_seen", id=cid, sha=hashlib.sha256(body.encode()).hexdigest(), chars=len(body))
            sent = sum(len(str(m.get("content") or "")) for m in kw.get("messages") or []) // 4 + 800
            usage = int((sent if asst.get("real_usage") else est) * float(asst.get("usage_scale", 1.0)))
            if cancel and step == 0:  # the ACP cancel lands while the provider call is in flight
                from agent.interrupt_compat import request_hard_interrupt
                request_hard_interrupt(ag)
                event(turn=t, event="cancel")
                time.sleep(float(cell.get("cancel_wait", 2.0)))
            groups = plan.get(t, []) if prefix == "T" else []
            if step < len(groups):
                for c in groups[step]:
                    event(turn=t, event="tool_call", name=c["name"], session_prefix=prefix)
                return response("", usage, groups[step], t, f"{prefix}{t:02d}_{step}")
            # the scripted reply the provider RETURNED for this attempt (tagged at call time: a cancelled call may
            # return after the host moved on), independent of what the host keeps
            event(turn=t, event="emit", tag=f"{prefix}{t:02d}", text=reply_text(prefix, t))
            return response(reply_text(prefix, t), usage)
        ag.client.chat.completions.create.side_effect = provider

    def held_after(result, text, prefix, t):
        """The user row the host holds for this turn (after its persist override / merge) and what follows it."""
        msgs = result.get("messages") if isinstance(result.get("messages"), list) else []
        tag = f"[{prefix}{int(cell['user_text'].get('identical_turns', {}).get(str(t), t)):02d}]"
        idx = [i for i, m in enumerate(msgs) if m.get("role") == "user" and isinstance(m.get("content"), str)
               and (text != "continue" and tag in m["content"])]
        if text == "continue":  # position-bound: only the turn's own row, the LAST user row, and only if it is it
            users = [i for i, m in enumerate(msgs) if m.get("role") == "user" and isinstance(m.get("content"), str)]
            idx = users[-1:] if users and msgs[users[-1]]["content"].strip().endswith("continue") else []
        tags = {}
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                for x in set(re.findall(r"\[([A-Z]\d{2,3})\] user turn", m["content"])):
                    tags[x] = tags.get(x, 0) + 1
        if not idx:
            return None, False, [], tags, []
        after = msgs[idx[-1] + 1:]
        reply = reply_text(prefix, t)
        # Only the host's own cited failed-turn boundary copy (agent/turn_failure_copy.py) may appear unscripted.
        own = [m["content"] for m in after if m.get("role") == "assistant" and isinstance(m.get("content"), str)
               and m["content"].strip() in notices]
        return (msgs[idx[-1]]["content"], any(m.get("role") == "assistant" and m.get("content") == reply for m in after),
                [m for m in after if m.get("role") == "tool"], tags, own)

    def run_turn(ag, prefix, t, history, kind="normal", persist_strip=None, task_id="S0"):
        text = user_text(cell, prefix, t)
        if kind == "retry":  # acp_adapter/server.py: plain text after a cancel re-attaches the cancelled prompt
            from acp_adapter.server import _attach_interrupted_prompt
            text = _attach_interrupted_prompt(text.strip(), text.strip())
        persist = text.strip() if (persist_strip if persist_strip is not None else cell["transport"] == "acp") else text
        est = sum(len(str(m.get("content") or "")) for m in history) // 4 + len(text) // 4 + 800
        cur.update(turn=t, prefix=prefix, step=0, native=0, commits0=buf.getvalue().count(COMMITTED), issued=set(), seen=set())
        event(turn=t, event="user_sent" if kind != "retry" else "retry", tag=f"{prefix}{t:02d}", role="user",
              session_prefix=prefix, content=text, persist=persist if persist != text else None,
              content_sha256=hashlib.sha256(text.encode()).hexdigest())
        scripted(ag, prefix, t, est, cancel=kind == "cancel")
        result = ag.run_conversation(user_message=text, conversation_history=history, task_id=task_id,
                                     persist_user_message=persist)
        held, reply_held, tools, user_tags, host_replies = held_after(result, text, prefix, t)
        failed = bool(result.get("failed")) or not result.get("completed", True)
        if failed and kind != "cancel":
            counters["failed"].append(f"{prefix}{t:02d}")
        counters["native_max"] = max(counters["native_max"], cur["native"])
        for m in tools:
            event(turn=t, event="tool_result", role="tool", size=len(str(m.get("content") or "")))
        event(turn=t, event="turn_end", tag=f"{prefix}{t:02d}", session_prefix=prefix, kind=kind,
              **({"held_same": True} if held == persist else {"held": held}),
              reply=reply_text(prefix, t) if reply_held else None, failed=failed, native_attempts=cur["native"],
              interrupted=bool(result.get("interrupted")), session=ag.session_id, user_tags=user_tags, host_notices=host_replies,
              tools_planned=len(cur["issued"]), tools_answered=len(cur["seen"]),
              host_commits=buf.getvalue().count(COMMITTED) - cur["commits0"])
        return result

    def cron_run(k):  # cron/scheduler.py: a fresh platform="cron" agent per fire, no history, closed after
        cron_sid = f"cron_job_{k:02d}"
        ag = build(cron_sid, "cron")
        box = {}

        def work():
            try:
                box["r"] = run_turn(ag, "K", k, [], persist_strip=False, task_id=cron_sid)
            except Exception as exc:  # recorded, never raised past the probe
                box["e"] = repr(exc)
        th = threading.Thread(target=work, name=f"cron-{k}")
        th.start()
        th.join()
        with_db = getattr(ag, "_session_db", None)
        if with_db is not None and hasattr(with_db, "end_session"):
            with_db.end_session(cron_sid, "cron_complete")
        ag.close()
        if "e" in box:
            counters["failed"].append(f"K{k:02d}")

    history = []
    if phase != "A":  # ACP _restore reads the stable ACP id; a gateway reads the durable tip (load_transcript)
        history = sdb_read.get_messages_as_conversation(sid, repair_alternation=True)
    cancel = faults.get("cancel_then_retry")
    backlog_log = out.setdefault("final_backlog", [])

    def low_backlog(last_turn):
        return backlog_low(cell_dir, last_turn, backlog_log)
    for t in extend_turns(cell, first, low_backlog):
        f = faults.get("plugin_switch")  # native-on-off: exit between turns; run_matrix swaps in the candidate
        if f and t == f["turn"] and t != first and "plugin_switch" not in fired:
            fire("plugin_switch", t)
            finish("plugin_switch", next_turn=t)
        f = faults.get("clean_exit_before_turn")
        if f and phase != "A" and t == f.get("turn", first + f.get("after_restart", 0)) and t != first \
                and "clean_exit_before_turn" not in fired:
            fire("clean_exit_before_turn", t)
            finish("clean_exit", next_turn=t)
        if cell["transport"] == "gateway" and not (phase == "A" and t == first):  # load_transcript, every turn
            tip = sdb_read.get_compression_tip(agent.session_id) or agent.session_id
            if tip != agent.session_id:  # a real gateway serves the next message from the tip
                finish("tip_switch", next_turn=t, tip=tip)
            history = sdb_read.get_messages_as_conversation(tip, repair_alternation=True)
        if cancel and t == cancel["turn"] and "cancel_then_retry" not in fired:
            result = run_turn(agent, "T", t, history, kind="cancel")
            history = result["messages"] if isinstance(result.get("messages"), list) else history
            if result.get("interrupted"):  # the fault counts only once the host really interrupted the turn
                fire("cancel_then_retry", t)
                result = run_turn(agent, "T", t, history, kind="retry")
        else:
            result = run_turn(agent, "T", t, history)
        if isinstance(result.get("messages"), list):
            history = result["messages"]
        if cell["transport"] == "gateway" and agent.session_id != sid:  # spec: restart the gateway at the tip, never
            finish("tip_switch", next_turn=t + 1, tip=agent.session_id)  # switch the resident engine in process
        if cell.get("cron_every") and t % int(cell["cron_every"]) == 0:
            cron_run(t // int(cell["cron_every"]))
    if cell.get("final_compaction_check", True):
        cur["final"] = True  # its compaction events are B4 evidence, not B5 passes
        out["final_check"] = {**final_check(agent, history, buf), "backlog_checks": backlog_log}
    finish("done", next_turn=None)


LOCAL_HOSTS = ("::1", "localhost")


def is_loopback(host) -> bool:
    """Only the exact names in LOCAL_HOSTS, or an IP literal that is loopback: ``127.attacker.example`` is a name
    the real resolver would look up, so it is not local."""
    host = host.decode() if isinstance(host, bytes) else host
    if str(host) in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


def guard_sockets(local_ok, on_refuse=None):
    """Refuse outbound traffic at every Python socket entry: connect, connect_ex, sendto, sendmsg (UDP and
    literal addresses included), create_connection and name resolution. ``local_ok`` lets loopback through (the
    R2 host observer); R1's in-process probe needs no network at all. ``on_refuse(what, host, port)`` records."""
    def local(host):
        return local_ok and (host is None or is_loopback(host))

    def refuse(what, host, port):
        if on_refuse:
            on_refuse(what, str(host), port)
        raise OSError(f"network blocked by probe: {what} {host}:{port}")

    def blocked(sock, name, address):
        if not local_ok and name in ("connect", "connect_ex"):  # R1: no socket connects at all, any family
            return True
        inet = sock.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple)
        return inet and not local(address[0])

    def wrap(name, pos):
        orig = getattr(socket.socket, name)

        def guarded(self, *args):
            address = args[pos] if len(args) > max(pos, 0) or (pos < 0 and args) else None
            if (address is not None or name in ("connect", "connect_ex")) and blocked(self, name, address):
                host, port = address[:2] if isinstance(address, tuple) else (address, None)
                refuse(name, host, port)
            return orig(self, *args)
        setattr(socket.socket, name, guarded)
    for name, pos in (("connect", 0), ("connect_ex", 0), ("sendto", -1), ("sendmsg", 3)):
        wrap(name, pos)
    orig_gai, orig_cc = socket.getaddrinfo, socket.create_connection

    def getaddrinfo(host, port, *a, **k):
        if not local(host):
            try:
                refuse("getaddrinfo", host, port)
            except OSError as exc:
                raise socket.gaierror(socket.EAI_NONAME, str(exc)) from None
        return orig_gai(host, port, *a, **k)

    def create_connection(address, *a, **k):
        if not local(address[0]):
            refuse("create_connection", address[0], address[1])
        return orig_cc(address, *a, **k)
    socket.getaddrinfo, socket.create_connection = getaddrinfo, create_connection


MIN_BACKLOG_TURNS, MAX_EXTRA_TURNS = 3, 3


def backlog_low(cell_dir, last_turn, log):
    """True when an automatic pass committed within the last MIN_BACKLOG_TURNS turns, so the final forced
    compaction would find no raw backlog outside the fresh tail (LCM ingests lazily, so lcm.db cannot show the
    backlog before that compaction). Read from the cell's own ``compaction`` events; every check is recorded."""
    path = Path(cell_dir) / "transcript.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    passes = [e["turn"] for e in events if e.get("event") == "compaction" and not e.get("final")
              and e.get("compression_status") in ("compacted", "host_native")]
    since = last_turn - max(passes) if passes else None
    log.append({"after_turn": last_turn, "last_pass_turn": max(passes) if passes else None, "turns_since_pass": since})
    return since is not None and since < MIN_BACKLOG_TURNS


def extend_turns(cell, first, low):
    """The cell's turns from ``first``, then up to MAX_EXTRA_TURNS extra scripted turns (tagged and scored like any
    turn) while ``low()`` says the final forced compaction has no backlog. Shared by R1 and R2."""
    turns = int(cell["turns"])
    yield from range(first, turns + 1)
    if not cell.get("final_compaction_check", True):
        return
    for t in range(max(first, turns + 1), turns + MAX_EXTRA_TURNS + 1):
        if not low(t - 1):
            return
        yield t


def session_count():
    try:
        con = sqlite3.connect(f"file:{Path(os.environ['HERMES_HOME']) / 'state.db'}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def rejection(engine):
    """LCM's native-recovery rejection reason for the call that just ended (it doubles as the noop reason)."""
    reason = getattr(engine, "_last_native_recovery_rejection", None)
    return reason if reason and reason == getattr(engine, "_last_compression_noop_reason", None) else None


def final_check(agent, history, buf):
    """Force a compaction through the host's own ACP ``/compress`` entry point (acp_adapter/commands.py
    ``_cmd_compress``: ``compress_now`` where the host has it, else ``_compress_context(force=True)``). A first
    attempt that only consumed LCM's one-shot preflight cleanup handoff (``sanitized``) is followed by one more
    invocation of the same entry point on the installed history, as a user re-running /compress would."""
    engine, attempts = agent.context_compressor, []
    system = getattr(agent, "_cached_system_prompt", "") or ""
    try:  # select the API once, BEFORE invoking it; an exception from the selected path fails the check
        from agent.conversation_compression_manual import compress_now, parse_compress_args
        from agent.conversation_compression import finalize_context_engine_compression_notification
        entry = "compress_now"
    except ImportError as exc:  # older hosts (no manual-compression module): _cmd_compress calls _compress_context
        if not (isinstance(exc, ModuleNotFoundError) and exc.name == "agent.conversation_compression_manual"):
            return {"entry": "compress_now", "exception": repr(exc)[:500], "engine_calls": 0, "engine_status": None,
                    "attempts": [], "conflicts": 0, "published": False, "outcome": "failed"}
        from acp_adapter.commands import _estimate_tokens
        entry = "_compress_context(force=True)"
    for _ in range(2):
        before, rec = buf.getvalue().count("publication_invariant_conflict"), {}
        calls0 = getattr(engine, "_probe_calls", 0)
        saved = getattr(agent, "_session_db", None)
        try:
            agent._session_db = None  # "Stable ACP session id: suppress _compress_context's SQLite session split."
            rec["entry"] = entry
            if entry == "compress_now":
                res = compress_now(agent, history, parse_compress_args(""), system_message=system, task_id="S0")
                rec["host_status"] = res.status
                if res.status == "compressed":
                    finalize_context_engine_compression_notification(agent, committed=True)
                    history = list(res.after_messages)
            else:
                approx = _estimate_tokens(history, agent, system, getattr(agent, "tools", None) or None)
                history, _ = agent._compress_context(list(history), system, approx_tokens=approx, task_id="S0", force=True)
                rec["host_status"] = "compressed"
        except Exception as exc:
            rec["exception"] = repr(exc)[:500]
        finally:
            agent._session_db = saved
        rec["engine_calls"] = getattr(engine, "_probe_calls", 0) - calls0  # a compress() inside THIS invocation
        rec["engine_status"] = getattr(engine, "_probe_status", None) if rec["engine_calls"] else None
        rec["noop_reason"] = getattr(engine, "_last_compression_noop_reason", None)
        rec["rejection"] = rejection(engine)
        rec["conflicts"] = buf.getvalue().count("publication_invariant_conflict") - before
        attempts.append(rec)
        if rec["engine_status"] != "sanitized" or "exception" in rec:
            break
    last = attempts[-1]
    ok = ("compacted", "host_native") if os.environ.get("LCM_NATIVE_RECOVERY") == "true" else ("compacted",)
    conflicts = sum(a["conflicts"] for a in attempts)
    # Published only on a fresh host "compressed" AND an engine pass from this very invocation that committed.
    published = (last.get("host_status") == "compressed" and last["engine_calls"] > 0 and last["engine_status"] in ok
                 and not conflicts and "exception" not in last)
    failed = bool(conflicts or "exception" in last or last["engine_status"] == "error")
    return {**last, "attempts": attempts, "conflicts": conflicts, "published": published,
            "outcome": "published" if published else "failed" if failed else "inconclusive"}


if __name__ == "__main__":
    main()
