"""Observe (never steer) a real Hermes host process for one R2 cell phase.

Loaded by ``sitecustomize`` in the host process the R2 runner spawns (``hermes acp``), with ``REL_OBSERVER_DIR`` =
the cell dir and ``REL_PHASE``. It wraps, without changing arguments or results, the same host seams R1's probe
traces in-process: ``AIAgent.run_conversation`` (the user row the host held, the reply, failure/interrupt), the
context engine's ``compress`` / ``handle_tool_call``, ``model_tools.handle_function_call``, the host orphan-drop
pass, the native ContextCompressor, and ``compress_now`` (the ACP ``/compress`` entry). Events go to the cell's
``transcript.jsonl`` in R1's schema; control data (compaction commits, turn ends, counters, final attempts,
network attempts) to ``observer.jsonl``; host log records to ``probe-<phase>.hermes.log`` (flushed per record, so a
SIGKILL loses nothing already logged). The turn context comes from ``turn.json``, written by the driver before
each prompt. Sockets to anything but localhost are refused and recorded (containment + evidence).
"""
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

DIR = Path(os.environ.get("REL_OBSERVER_DIR", "."))
PHASE = os.environ.get("REL_PHASE", "A")
COMMITTED = '"commit_status":"committed"'
_lock = threading.Lock()
cur = {"turn": 0, "prefix": "T", "kind": "normal", "native": 0, "final": False, "commits0": 0}
counters = {"compacted_turns": [], "lcm_tool_calls": 0, "orphan_drops": 0, "native_max": 0, "failed": [], "compactions": []}
logstate = {"commits": 0, "conflicts": 0, "survival_fits": 0}
_depth = threading.local()
_r1 = None
_tap = None
_p8_finish = {"supported": False, "notes": [], "duplicates": []}.copy


def _p8_pin(messages):
    pass


def _append(path, rec):
    data = (json.dumps(rec, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        while data:  # one write per record: O_APPEND keeps records from the runner and the host whole
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)


def event(**ev):
    _append(DIR / "transcript.jsonl", {"phase": PHASE, **ev})


def note(what, **rec):
    _append(DIR / "observer.jsonl", {"phase": PHASE, "ts": time.time(), **rec, "kind": what})


def snapshot(**extra):
    note("counters", counters=counters, orphan_hook=cur.get("orphan_hook"), p8=_p8_finish(), **extra)


def install_p8():
    global _p8_pin, _p8_finish
    if cur.get("p8_installed"):
        return
    cur["p8_installed"] = True
    try:
        cell = json.loads((DIR / "cell.json").read_text())
        path = DIR / "faults-fired.jsonl"
        fired = {json.loads(x)["kind"] for x in path.read_text().splitlines()} if path.exists() else set()

        def fire(kind, turn, **extra):
            _append(path, {"kind": kind, "phase": PHASE, "turn": turn})
            fired.add(kind)
            event(turn=turn, event=kind, fault=kind, session_prefix=cur["prefix"], **extra)

        _, _p8_pin, _p8_finish = _r1.install_p8(
            DIR, PHASE, {f["kind"]: f for f in cell["faults"]}, fired, fire, cur,
            checkpoint=lambda: note("p8", p8=_p8_finish()))
    except Exception as exc:
        state = {"supported": False, "notes": [type(exc).__name__], "duplicates": []}
        _p8_finish = state.copy


def load_turn():
    try:
        tc = json.loads((DIR / "turn.json").read_text())
    except (OSError, ValueError):
        tc = {"prefix": "X", "t": 0, "kind": "unknown", "reply": None}
    cur.update(turn=tc["t"], prefix=tc["prefix"], kind=tc["kind"], reply=tc.get("reply"), native=0,
               final=tc["kind"] == "final", commits0=logstate["commits"])
    return tc


def tag():
    return f"{cur['prefix']}{cur['turn']:02d}"


def install():
    global _r1
    spec = importlib.util.spec_from_file_location("_rel_r1_probe", Path(__file__).resolve().parents[1] / "probe.py")
    _r1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_r1)
    if (why := _r1.refusal(str(DIR))) is not None:
        note("refused", reason=why)
        os._exit(3)
    _r1.guard_sockets(local_ok=True, on_refuse=lambda what, host, port: note(
        "network_attempt", what=what, host=host, port=port, thread=threading.current_thread().name))
    sys.meta_path.insert(0, _PostImport())


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self.inner, self.patch = inner, patch

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            self.patch(module)
        except Exception as exc:  # an observer failure is evidence, never a host crash
            note("observer_error", where=module.__name__, error=repr(exc)[:300])

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _PostImport(importlib.abc.MetaPathFinder):
    """Patch a host module right after it executes, before any importer binds its names."""

    def find_spec(self, name, path, target=None):
        patch = PATCHES.get(name)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, patch)
                return spec
        return None


class _Tap(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        self.fh = open(DIR / f"probe-{PHASE}.hermes.log", "a", encoding="utf-8")

    def emit(self, record):
        try:
            line = self.format(record)
        except Exception:
            return
        with _lock:
            logstate["commits"] += line.count(COMMITTED)
            logstate["conflicts"] += line.count("publication_invariant_conflict")
            logstate["survival_fits"] += line.count("LCM survival fit applied")
            self.fh.write(line + "\n")
            self.fh.flush()


def ensure_tap():
    global _tap
    root = logging.getLogger()
    if _tap is None:
        _tap = _Tap()
    if _tap not in root.handlers:
        root.addHandler(_tap)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def dispatched(run, name, args, call_id, via):
    """Record one real host tool execution (outermost hook only) and whether its result is a success."""
    _depth.n = getattr(_depth, "n", 0) + 1
    try:
        result = run()
    except BaseException as exc:
        if _depth.n == 1:
            event(turn=cur["turn"], event="tool_dispatch", tag=tag(), id=call_id, name=name, args=args, via=via,
                  ok=False, detail=f"raised {exc!r}"[:200], chars=0)
        raise
    finally:
        _depth.n -= 1
    if _depth.n == 0:
        ok, detail, chars = _r1.check_result(name, args, result)
        event(turn=cur["turn"], event="tool_dispatch", tag=tag(), id=call_id, name=name, args=args, via=via,
              ok=ok, detail=detail, chars=chars)
    return result


def depth0():
    try:
        con = sqlite3.connect(f"file:{Path(os.environ['HERMES_HOME']) / 'lcm.db'}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM summary_nodes WHERE depth = 0 AND source_type = 'messages'").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def store_cover():
    """(message-sourced depth-0 summaries, distinct store ids they cover) in lcm.db, or (None, None)."""
    try:
        con = sqlite3.connect(f"file:{Path(os.environ['HERMES_HOME']) / 'lcm.db'}?mode=ro", uri=True)
        try:
            return con.execute(
                "SELECT (SELECT COUNT(*) FROM summary_nodes WHERE depth = 0 AND source_type = 'messages'), "
                "(SELECT COUNT(DISTINCT source.value) FROM summary_nodes AS node, json_each(node.source_ids) AS source "
                "WHERE node.depth = 0 AND node.source_type = 'messages')").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None, None


def list_counts(messages, result) -> dict:
    """``in``/``out``: the lengths of the list handed to ``compress`` and of the list it returned;
    ``host_rows_summarized``: the input rows absent from the output, i.e. the host rows a leaf replaced with a
    summary. A leaf built only from stored rows the host no longer holds removes none. An input row is retained when
    the output holds that very dict or, since LCM returns retained rows as equal copies (measured at 2e04a205), an
    equal dict (``==``: every key, timestamp included) not already matched; ``copied`` counts the latter. Taken
    from a list captured before the call, so an in-place mutation of the input list cannot hide a row."""
    if not isinstance(result, list) or not isinstance(messages, list):
        return {"in": len(messages) if isinstance(messages, list) else None,
                "out": len(result) if isinstance(result, list) else None, "host_rows_summarized": None, "copied": None}
    free = {id(m) for m in result}
    unmatched = [m for m in messages if id(m) not in free]
    free -= {id(m) for m in messages}
    pool, copied = [m for m in result if id(m) in free], 0
    for m in unmatched:
        k = next((i for i, o in enumerate(pool) if o == m), None)
        if k is not None:
            pool.pop(k)
            copied += 1
    return {"in": len(messages), "out": len(result), "host_rows_summarized": len(unmatched) - copied, "copied": copied}


def patch_engine(agent):
    engine = getattr(agent, "context_compressor", None)
    etype = type(engine)
    if engine is None or getattr(etype, "_rel_patched", False):
        return
    etype._rel_patched = True
    orig_compress, orig_tool = etype.compress, etype.handle_tool_call
    cell_path = DIR / "cell.json"
    in_place = json.loads(cell_path.read_text()).get("in_place") if cell_path.exists() else None

    def observer_failed(where, exc):
        try:  # an observer failure is evidence (the cell ERRORs on it), never a change to the engine call
            note("observer_error", where=where, error=repr(exc)[:300])
        except Exception:
            pass

    def elapsed(started):
        return None if started is None else round(time.monotonic() - started, 3)

    def traced_compress(self, messages, *args, **kwargs):
        started, given, cover0 = None, messages, (None, None)
        fits0 = logstate["survival_fits"]
        try:
            started = time.monotonic()
            given, cover0 = list(messages) if isinstance(messages, list) else messages, store_cover()
        except Exception as exc:
            observer_failed("traced_compress:before", exc)
        try:
            result = orig_compress(self, messages, *args, **kwargs)
        except BaseException as exc:
            try:
                counters["compactions"].append({"turn": cur["turn"], **list_counts(given, None), "status": "error",
                                                "final": cur["final"], "secs": elapsed(started),
                                                "error": f"{type(exc).__name__}: {exc}"[:300]})
            except Exception as obs:
                observer_failed("traced_compress:error-record", obs)
            raise
        try:
            secs = elapsed(started)
            _p8_pin(result)
            status = getattr(self, "_last_compression_status", None)
            cover = store_cover()  # leaves written by this call and the stored rows they newly cover (hidden or host)
            delta = [None if a is None or b is None else b - a for a, b in zip(cover0, cover)]
            counters["compactions"].append({"turn": cur["turn"], **list_counts(given, result), "status": status,
                                            "final": cur["final"], "secs": secs, "leaves": delta[0], "rows_covered": delta[1]})
            self._probe_calls, self._probe_status = getattr(self, "_probe_calls", 0) + 1, status
            fitted = logstate["survival_fits"] > fits0
            if status in ("compacted", "host_native"):
                counters["compacted_turns"].append(cur["turn"])
            if status in ("compacted", "host_native") or fitted:
                pool, removed = list(result) if isinstance(result, list) else [], []
                for row in given if isinstance(given, list) else []:
                    match = next((i for i, out in enumerate(pool) if row == out), None)
                    if match is None:
                        removed.append(row)
                    else:
                        pool.pop(match)
                tags = sorted({t for row in removed if row.get("role") == "assistant"
                               for t in re.findall(r"reply to ([A-Z]\d{2,3}):", str(row.get("content") or ""))})
                note("compaction_committed" if status in ("compacted", "host_native") else "survival_fit_committed",
                     turn=cur["turn"], prefix=cur["prefix"], status=status, final=cur["final"],
                     compaction_kind="forced" if cur["final"] else "survival fit" if fitted else
                     "in-place" if in_place else "rotation" if in_place is False else None,
                     survival_fit=fitted, fit_reason=(getattr(self, "_last_survival_fit", None) or {}).get("reason")
                     if fitted else None, compacted_reply_tags=tags)
            event(turn=cur["turn"], event="compaction", session=getattr(self, "_session_id", None), final=cur["final"],
                  session_prefix=cur["prefix"], compression_status=status,
                  noop_reason=getattr(self, "_last_compression_noop_reason", None),
                  depth0_nodes=depth0() if status == "compacted" else None, rejection=_r1.rejection(self),
                  native_attempts=cur["native"])
        except Exception as exc:
            observer_failed("traced_compress:after", exc)
        return result
    etype.compress = traced_compress

    def traced_tool(self, name, args, **kwargs):
        counters["lcm_tool_calls"] += 1
        return dispatched(lambda: orig_tool(self, name, args, **kwargs), name, args, None, "engine_tool_dispatch")
    etype.handle_tool_call = traced_tool
    import agent.context_compressor as host_cc
    native_cls = getattr(host_cc, "ContextCompressor", None)
    if native_cls is not None and native_cls is not etype:
        orig_native = native_cls.compress

        def counted_native(self, *args, **kwargs):
            cur["native"] += 1
            return orig_native(self, *args, **kwargs)
        native_cls.compress = counted_native


def held_after(result, text_tag, reply, notices):
    """R1 probe ``held_after``: the user row the host holds for this turn and what follows it."""
    msgs = result.get("messages") if isinstance(result.get("messages"), list) else []
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if text_tag is None:  # a "continue" prompt: position-bound, only the LAST user row and only if it is it
        idx = users[-1:] if users and msgs[users[-1]]["content"].strip().endswith("continue") else []
    else:
        idx = [i for i in users if f"[{text_tag}]" in msgs[i]["content"]]
    tags = {}
    for i in users:
        for x in set(re.findall(r"\[([A-Z]\d{2,3})\] user turn", msgs[i]["content"])):
            tags[x] = tags.get(x, 0) + 1
    if not idx:
        return None, False, [], tags, []
    after = msgs[idx[-1] + 1:]
    own = [m["content"] for m in after if m.get("role") == "assistant" and isinstance(m.get("content"), str)
           and m["content"].strip() in notices]
    return (msgs[idx[-1]]["content"], any(m.get("role") == "assistant" and m.get("content") == reply for m in after),
            [m for m in after if m.get("role") == "tool"], tags, own)


def patch_run_agent(mod):
    cls = mod.AIAgent
    orig_init, orig_run, orig_cc = cls.__init__, cls.run_conversation, getattr(cls, "_compress_context", None)

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        ensure_tap()
        install_p8()
        patch_engine(self)
        note("agent_built", session=getattr(self, "session_id", None), platform=k.get("platform"),
             engine=getattr(getattr(self, "context_compressor", None), "name", None), model=getattr(self, "model", None),
             base_url=str(getattr(self, "base_url", "")))
    cls.__init__ = init

    def run_conversation(self, *a, **k):
        ensure_tap()
        tc = load_turn()
        try:
            from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
            notices = [FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE]
        except ImportError:
            notices = []
        persist = k.get("persist_user_message")
        user = k.get("user_message", a[0] if a else None)
        event(turn=cur["turn"], event="host_prompt", tag=tag(), kind=cur["kind"], session=self.session_id,
              user_sha256=hashlib.sha256(str(user).encode()).hexdigest(), persist_differs=persist != user)
        try:
            result = orig_run(self, *a, **k)
        except BaseException as exc:
            counters["failed"].append(tag())
            event(turn=cur["turn"], event="turn_end", tag=tag(), session_prefix=cur["prefix"], kind=cur["kind"],
                  held=None, reply=None, failed=True, exception=repr(exc)[:300], session=self.session_id)
            snapshot(failed_turn_notices=notices)
            raise
        text_tag = None if tc.get("text") == "continue" else tc.get("text_tag") or tag()
        held, reply_held, tools, user_tags, own = held_after(result, text_tag, tc.get("reply"), notices)
        failed = bool(result.get("failed")) or not result.get("completed", True)
        if failed and cur["kind"] != "cancel":
            counters["failed"].append(tag())
        counters["native_max"] = max(counters["native_max"], cur["native"])
        for m in tools:
            event(turn=cur["turn"], event="tool_result", role="tool", size=len(str(m.get("content") or "")))
        interrupted = bool(result.get("interrupted"))
        event(turn=cur["turn"], event="turn_end", tag=tag(), session_prefix=cur["prefix"], kind=cur["kind"], held=held,
              reply=tc.get("reply") if reply_held else None, failed=failed, native_attempts=cur["native"],
              interrupted=interrupted, session=self.session_id, user_tags=user_tags, host_notices=own,
              host_commits=logstate["commits"] - cur["commits0"])
        note("turn_end", tag=tag(), turn_kind=cur["kind"], interrupted=interrupted, failed=failed, session=self.session_id,
             reply_tag=next(iter(re.findall(r"reply to ([A-Z]\d{2,3}):", tc.get("reply") or "")), None)
             if reply_held and not failed and not interrupted else None)
        snapshot(failed_turn_notices=notices, provenance=_r1.provenance(json.loads((DIR / "cell.json").read_text()), []))
        return result
    cls.run_conversation = run_conversation

    if orig_cc is not None:  # hosts whose /compress calls _compress_context(force=True) directly
        def compress_context(self, *a, **k):
            if cur.get("in_compress_now") or not k.get("force") or load_turn()["kind"] != "final":
                return orig_cc(self, *a, **k)
            return final_attempt(self, "_compress_context(force=True)", lambda: orig_cc(self, *a, **k), None)
        cls._compress_context = compress_context


def final_attempt(agent, entry, run, status_of):
    engine = agent.context_compressor
    rec, calls0 = {"entry": entry}, getattr(engine, "_probe_calls", 0)
    try:
        res = run()
        rec["host_status"] = status_of(res) if status_of else "compressed"
        return res
    except Exception as exc:
        rec["exception"] = repr(exc)[:500]
        raise
    finally:
        rec["engine_calls"] = getattr(engine, "_probe_calls", 0) - calls0
        rec["engine_status"] = getattr(engine, "_probe_status", None) if rec["engine_calls"] else None
        rec["noop_reason"] = getattr(engine, "_last_compression_noop_reason", None)
        rec["rejection"] = _r1.rejection(engine)
        note("final_attempt", **rec)
        snapshot()


def patch_manual(mod):
    orig = mod.compress_now

    def compress_now(agent, history, request, **kwargs):
        load_turn()
        if cur["kind"] != "final":
            return orig(agent, history, request, **kwargs)
        cur["in_compress_now"] = True
        try:
            return final_attempt(agent, "compress_now", lambda: orig(agent, history, request, **kwargs), lambda r: r.status)
        finally:
            cur["in_compress_now"] = False
    mod.compress_now = compress_now


def patch_model_tools(mod):
    orig = mod.handle_function_call

    def traced(*a, **kw):
        name, args = (a[0] if a else kw.get("function_name")), (a[1] if len(a) > 1 else kw.get("function_args"))
        return dispatched(lambda: orig(*a, **kw), name, args, kw.get("tool_call_id"), "tool_dispatch")
    mod.handle_function_call = traced


def patch_inline_tools(mod):
    """#659 P2: ``todo_list`` is an agent-level inline tool (``INLINE_TOOL_EXECUTORS``), dispatched before the registry,
    so ``handle_function_call`` never sees it. Only that entry is wrapped: no other cell plans an inline tool."""
    table = getattr(mod, "INLINE_TOOL_EXECUTORS", None)
    run = table.get("todo_list") if isinstance(table, dict) else None
    if run is None:
        return

    def traced(agent, args, ctx):
        return dispatched(lambda: run(agent, args, ctx), "todo_list", args, getattr(ctx, "tool_call_id", None),
                          "inline_tool_dispatch")
    table["todo_list"] = traced


def patch_helpers(mod):
    passes, drop = getattr(mod, "_SEQUENCE_REPAIR_PASSES", None), getattr(mod, "_drop_stray_tool_results", None)
    if not (passes and drop in passes):
        cur["orphan_hook"] = "unavailable at this host sha; B6 orphan count falls back to the log"
        return

    def counted_drop(messages):
        kept, n = drop(messages)
        counters["orphan_drops"] += n
        return kept, n
    mod._SEQUENCE_REPAIR_PASSES = tuple(counted_drop if p is drop else p for p in passes)


PATCHES = {"run_agent": patch_run_agent, "model_tools": patch_model_tools, "agent.agent_runtime_helpers": patch_helpers,
           "agent.conversation_compression_manual": patch_manual, "agent.inline_tool_executors": patch_inline_tools}
