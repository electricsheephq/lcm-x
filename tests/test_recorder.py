"""#2243: the opt-in event recorder records host calls without changing what the engine stores, returns or logs."""

import copy
import gc
import hashlib
import importlib
import itertools
import json
import logging
import os
import shutil
import sqlite3
import stat
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
from pathlib import Path

import pytest

from tests.recorder_replay import FakeCtx, HostModules, compare_stores, load_plugin, replay, scrub, store_rows, \
    stand_in_error, unload_plugin, write_state_db

REPO = Path(__file__).resolve().parents[1]
SLOT = "_lcm_recorder_process"
needs312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring needs Python 3.12 or newer")
FROZEN = 1_800_000_000.0
SECRET = "sk-recorder-secret-0123456789"
SECRET_HOST = "https://secret-host.recorder.invalid"
# Columns SQLite stamps from its own clock, which freezing Python's clock does not freeze. DEFAULT stamps:
# lcm_lifecycle_state.updated_at (db_bootstrap.py:802), lcm_rollup_invalidations.created_at (:917, :1269),
# lcm_assertion_sources.processed_at (:2120), lcm_assertions.created_at (:2159), lcm_assertion_relations.created_at
# (:2183). Statement stamps: lcm_assertion_sources.invalidated_at (:2277, :2287), lcm_migration_state.completed_at
# (:2563, :3307), lcm_query_corpus_state.changed_at (query_view_store.py:390, :408, :437), lcm_query_views.updated_at
# (:421, :450).
SQL_CLOCK_COLUMNS = frozenset({
    "lcm_lifecycle_state.updated_at", "lcm_rollup_invalidations.created_at", "lcm_assertion_sources.processed_at",
    "lcm_assertions.created_at", "lcm_assertion_relations.created_at", "lcm_assertion_sources.invalidated_at",
    "lcm_migration_state.completed_at", "lcm_query_corpus_state.changed_at", "lcm_query_views.updated_at"})


def _recorder():
    holder = sys.modules.get(SLOT)
    return getattr(holder, "recorder", None)


class Lab:
    """One plugin load per call against a Hermes-like home: state.db rows, task-local context, a fake ctx."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.modules, self.engines = [], []
        for key in [k for k in os.environ if k.startswith("LCM_")]:
            monkeypatch.delenv(key)
        for key, value in {"LCM_EMBEDDINGS_ENABLED": "false", "LCM_ENABLE_SLASH_COMMAND": "1",
                           "LCM_FRESH_TAIL_COUNT": "2", "LCM_LEAF_CHUNK_TOKENS": "64",
                           "LCM_TEMPORAL_ROLLUPS_ENABLED": "false"}.items():
            monkeypatch.setenv(key, value)
        self.host = HostModules(tmp_path / "home").__enter__()

    def home(self, name="home"):
        home = self.tmp / name
        home.mkdir(parents=True, exist_ok=True)
        write_state_db(home / "state.db", [])
        return home

    def load(self, home, *, record=True, ctx=None, summary=None):
        self.host.home = Path(home)
        self.mp.setenv("LCM_RECORDER_ENABLED", "1" if record else "0")
        self.mp.setenv("LCM_RECORDER_DIR", str(Path(home) / "lcm-recorder"))
        module = load_plugin(REPO, prefix="hermes_lcm_rec")
        self.modules.append(module)
        prepare(module, summary)
        ctx = ctx or FakeCtx()
        module.register(ctx)
        self.engines.append(ctx.offered)
        return module, ctx

    def stop(self):
        rec = _recorder()
        if rec is not None:
            rec.writer.drain(10)
            importlib.import_module(self.modules[-1].__name__ + ".recorder").reset_for_tests()
        return rec

    def close(self):
        for engine in self.engines:
            try:
                engine.shutdown()
            except Exception:
                pass
        if self.modules:
            self.stop()
        for module in self.modules:
            unload_plugin(module)
        self.host.__exit__()


def prepare(module, summary=None):
    tokens = importlib.import_module(module.__name__ + ".tokens")
    tokens._get_encoder = lambda: None
    tokens._count_tokens_cached.cache_clear()
    escalation = importlib.import_module(module.__name__ + ".escalation")
    escalation._call_llm_for_summary = summary or (
        lambda prompt, max_tokens, **kw: f"Earlier turns retained ({hashlib.sha256(prompt.encode()).hexdigest()[:8]}). "
                                         "Expand for details about: ledgers")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    lab = Lab(tmp_path, monkeypatch)
    yield lab
    lab.close()


class Agent:
    """A host agent frame: Hermes calls the engine and the hooks from inside its methods."""

    def __init__(self, session_id, parent="", **markers):
        self.session_id, self._parent_session_id = session_id, parent
        for name, value in markers.items():
            setattr(self, name, value)

    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _clone(message):
    if isinstance(message, dict):
        return {k: _clone(v) if isinstance(v, (dict, list)) else v for k, v in message.items()}
    if isinstance(message, list):
        return [_clone(v) if isinstance(v, (dict, list)) else v for v in message]
    return message


PROSE = ("I will inspect the next ledger record and compare its details with the earlier evidence. "
         "The result should tell us which step to take next and which facts still need checking. ")


def turn(history, n):
    history.extend([
        {"role": "user", "content": f"question {n} about the ledger: " + PROSE * 3},
        {"role": "assistant", "content": f"checking {n}: " + PROSE, "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": f"ledger line {n}: balance {n * 7}. " + PROSE * 8},
        {"role": "assistant", "content": f"answer {n}: balance {n * 7}. " + PROSE * 2},
    ])


def state_row(home, sid, source="cli", parent=None, end_reason=None):
    write_state_db(Path(home) / "state.db", [(sid, source, parent, FROZEN, None, end_reason, None)])


def scenario(lab, module, ctx, home):
    """Every root the record-only test drives: two sessions, growth, adopted and refused compress, end, rebind,
    a clone, hooks, a tool, /lcm and on_turn_complete. Returns what each root returned."""
    proto, out = ctx.offered, []
    post, pre = ctx.hooks["post_llm_call"][0], ctx.hooks["pre_llm_call"][0]
    a = Agent("s1")
    out.append(a.call(proto.update_model, model="m1", context_length=200_000, base_url=SECRET_HOST, api_key=SECRET))
    out.append(a.call(proto.on_session_start, "s1", platform="cli", conversation_id="c1", context_length=200_000))
    state_row(home, "s1")  # Hermes creates the row on first use, after on_session_start
    history = []
    for n in range(1, 4):
        turn(history, n)
        out.append(a.call(pre, session_id="s1", turn_id=f"t{n}", user_message=history[-4]["content"],
                          conversation_history=list(history), platform="cli", sender_id=""))
        out.append(a.call(post, session_id="s1", platform="cli", conversation_history=list(history)))
    history[1]["tool_calls"][0]["function"]["arguments"] = '{"cmd": "ls"}'  # nested host edit after a root
    history[0]["content"] = "  " + history[0]["content"] + " "  # in-place edge-whitespace rewrite
    turn(history, 4)
    out.append(a.call(post, session_id="s1", platform="cli", conversation_history=list(history)))
    out.append(a.call(proto.on_turn_complete, [_clone(m) for m in history], usage={"prompt_tokens": 900}))
    proto.threshold_tokens = 5_000  # host-written between roots
    proto.compression_count = 3
    out.append(a.call(proto.should_compress, 6_000))
    lab.mp.setenv("LCM_SENDER_CAPTURE", "1")
    out.append(a.call(ctx.tools["lcm_grep"], {"query": "ledger"}))
    lab.host.context.update({"HERMES_SESSION_ID": "s1", "HERMES_SESSION_KEY": "c1"})
    out.append(a.call(ctx.commands["lcm"], "status"))
    lab.host.context.clear()
    out.append(a.call(ctx.sections["lcm-x"], {"session_id": "s1"}))
    compressed = a.call(proto.compress, list(history), current_tokens=1_000_000, force=True)
    out.append(compressed)
    out.append(a.call(proto.on_session_start, "s1b", platform="cli", conversation_id="c1",
                      boundary_reason="compression", old_session_id="s1"))
    state_row(home, "s1", end_reason="compression")
    state_row(home, "s1b", parent="s1")
    history = list(compressed)
    turn(history, 5)
    out.append(a.call(post, session_id="s1b", platform="cli", conversation_history=list(history)))
    out.append(a.call(proto.compress, list(history), current_tokens=1_000_000, force=True))
    out.append(a.call(proto.record_rejected_compaction))  # the host refused the result
    out.append(a.call(proto._automatic_compression_blocked))
    out.append(a.call(proto._compression_block_reason))
    child = a.call(proto.clone_for_agent)
    lab.engines.append(child)
    b = Agent("s2")
    out.append(b.call(child.bind_session_state, object(), "s2"))
    out.append(b.call(child.on_session_start, "s2", platform="cli", conversation_id="c2", context_length=200_000))
    state_row(home, "s2")
    other = []
    turn(other, 1)
    out.append(b.call(post, session_id="s2", platform="cli", conversation_history=list(other),
                      context_compressor=child))
    out.append(b.call(ctx.hooks["subagent_start"][0], child_session_id="hooked", parent_session_id="s2",
                      child_subagent_id="sa-9", child_role="leaf"))
    hooked = Agent("hooked")
    view = [{"role": "user", "content": "Inspect the ledger."}, {"role": "assistant", "content": "balance 42"}]
    out.append(hooked.call(post, session_id="hooked", platform="cli", conversation_history=view))
    out.append(b.call(ctx.hooks["subagent_stop"][0], child_session_id="hooked", parent_session_id="s2"))
    out.append(a.call(proto.on_session_end, "s1b", list(history)))
    rebound = a.call(proto.clone_for_agent)
    lab.engines.append(rebound)
    other_home = lab.home("other-profile")
    out.append(Agent("p2").call(rebound.on_session_start, "p2", platform="cli", hermes_home=str(other_home)))
    out.append(Agent("p2").call(rebound.get_status))  # bound to another db: not recorded
    out.append(b.call(child.shutdown))
    return out


def normalize(value):
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if not str(k).endswith("_ms")}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def prepared_home(lab, name="prepared"):
    """A home whose lcm.db (and its ledger salt) exists before the recording starts."""
    home = lab.home(name)
    module, ctx = lab.load(home, record=False)
    ctx.offered.on_session_start("prep", platform="cli", context_length=200_000)
    history = []
    turn(history, 0)
    ctx.hooks["post_llm_call"][0](session_id="prep", platform="cli", conversation_history=history)
    ctx.offered.shutdown()
    return home


def copy_home(src, dst):
    shutil.copytree(src, dst)
    return dst


def run_scenario(lab, home, *, record):
    lab.mp.setattr(time, "time", lambda: FROZEN)
    module, ctx = lab.load(home, record=record)
    out = scenario(lab, module, ctx, home)
    status = normalize(ctx.offered.get_status())
    ctx.offered.shutdown()
    rec = lab.stop() if record else None
    lab.mp.undo_time = None
    return module, ctx, out, status, rec


@pytest.fixture
def recorded(lab):
    base = prepared_home(lab)
    home = copy_home(base, lab.tmp / "on")
    module, ctx, out, status, rec = run_scenario(lab, home, record=True)
    return {"base": base, "home": home, "module": module, "ctx": ctx, "out": out, "status": status, "rec": rec,
            "proc": rec.proc, "reader": importlib.import_module(module.__name__ + ".recorder_read")}


# -- flag off, versions, tool ids, installs --------------------------------------------------------------

_OFF_SCRIPT = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from tests.recorder_replay import FakeCtx, HostModules, load_plugin
    home = Path(sys.argv[1])
    tools = [sys.monitoring.get_tool(i) for i in (3, 4)] if hasattr(sys, "monitoring") else None
    with HostModules(home):
        module = load_plugin(Path.cwd())
        ctx = FakeCtx()
        before = set(vars(ctx))
        module.register(ctx)
        ctx.offered.shutdown()
    print(json.dumps({
        "modules": sorted(k for k in sys.modules if k.endswith(".recorder") or k.endswith(".recorder_read")
                          or k == "_lcm_recorder_process"),
        "ctx_attrs": sorted(set(vars(ctx)) - before - {"offered"}),
        "dir": (home / "lcm-recorder").exists(),
        "tools": [tools, [sys.monitoring.get_tool(i) for i in (3, 4)] if hasattr(sys, "monitoring") else None]}))
""")


def test_flag_off_installs_nothing_in_a_fresh_process(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LCM_")}
    env.update(LCM_EMBEDDINGS_ENABLED="false", LCM_RECORDER_DIR=str(tmp_path / "home" / "lcm-recorder"))
    (tmp_path / "home").mkdir()
    done = subprocess.run([sys.executable, "-c", _OFF_SCRIPT, str(tmp_path / "home")], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["modules"] == []
    assert result["ctx_attrs"] == []
    assert result["dir"] is False
    assert result["tools"][0] == result["tools"][1]


def test_python_below_312_with_the_flag_on_installs_nothing(lab, caplog, monkeypatch):
    home = lab.home()
    module = load_plugin(REPO, prefix="hermes_lcm_rec")
    lab.modules.append(module)
    recorder = importlib.import_module(module.__name__ + ".recorder")
    monkeypatch.setattr(recorder, "_MIN_PYTHON", (99, 0))
    monkeypatch.setattr(recorder, "_MIN_PYTHON", (3, 12) if sys.version_info < (3, 12) else (99, 0))
    caplog.set_level(logging.WARNING)
    lab.host.home = home
    monkeypatch.setenv("LCM_RECORDER_ENABLED", "1")
    ctx = FakeCtx()
    module.register(ctx)
    lab.engines.append(ctx.offered)
    warnings = [r for r in caplog.records if "needs Python 3.12" in r.getMessage()]
    assert len(warnings) == 1
    assert _recorder() is None
    assert "register_hook" not in vars(ctx)
    assert not (home / "lcm-recorder").exists()


@needs312
def test_tool_id_three_taken_uses_four_and_both_taken_installs_nothing(lab, caplog):
    mon = sys.monitoring
    mon.use_tool_id(3, "someone-else")
    try:
        lab.load(lab.home("a"))
        assert _recorder().tool == 4 and mon.get_tool(4) == "lcm-recorder"
        lab.stop()
        mon.use_tool_id(4, "another")
        caplog.set_level(logging.WARNING)
        module, ctx = lab.load(lab.home("b"))
        assert _recorder() is None
        assert len([r for r in caplog.records if "tool ids 3 and 4" in r.getMessage()]) == 1
        assert "register_hook" not in vars(ctx)
    finally:
        for tool in (3, 4):
            if mon.get_tool(tool) in ("someone-else", "another"):
                mon.free_tool_id(tool)


@needs312
def test_second_install_shares_one_tool_id_and_process_directory(lab):
    home = lab.home()
    _module, ctx1 = lab.load(home)
    rec = _recorder()
    _module, ctx2 = lab.load(home)
    assert _recorder() is rec
    assert [sys.monitoring.get_tool(i) for i in (3, 4)].count("lcm-recorder") == 1
    for ctx in (ctx1, ctx2):
        ctx.offered.on_session_start("s", platform="cli")
    rec.writer.drain(5)
    assert len(list((home / "lcm-recorder" / "proc").iterdir())) == 1
    reader = importlib.import_module(lab.modules[-1].__name__ + ".recorder_read")
    reading = reader.read(rec.proc)
    engines = {e["engine"] for e in reading["roots"] if e["name"] == "on_session_start"}
    assert len(engines) == 2, ([(e.get("kind"), e.get("name"), e.get("engine")) for e in reading["events"]],
                               rec.counters, rec.first_cause)


# -- record-only -----------------------------------------------------------------------------------------

@needs312
def test_record_only_flag_on_and_off_store_and_return_the_same(lab):
    """Both runs use one home path in turn (rows carry it) and the same uuid4 sequence (engine.py:921)."""
    base, run = prepared_home(lab), lab.tmp / "run"
    results = {}
    for name, record in (("off", False), ("on", True)):
        ids = itertools.count(1)
        lab.mp.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(ids)))
        results[name] = run_scenario(lab, copy_home(base, run), record=record)
        gc.collect()
        run.rename(lab.tmp / name)
    (_m, _c, out_off, status_off, _), (_m, _c, out_on, status_on, rec) = results["off"], results["on"]
    assert normalize(out_on) == normalize(out_off)
    assert json.dumps(scrub(status_on)) == json.dumps(scrub(status_off))
    off = store_rows(lab.tmp / "off" / "lcm.db", excluded_columns=SQL_CLOCK_COLUMNS, excluded_prefixes=())
    on = store_rows(lab.tmp / "on" / "lcm.db", excluded_columns=SQL_CLOCK_COLUMNS, excluded_prefixes=())
    assert set(off) == set(on)
    assert {t for t in off if off[t] != on[t]} == set()
    assert rec.counters["recorder.failed"] == 0
    assert rec.first_cause is None, rec.first_cause
    assert rec.counters["recorder.foreign_path"] >= 1


@needs312
def test_frame_neutrality(lab):
    home = lab.home()
    depths = {}
    for record in (False, True):
        module, ctx = lab.load(home, record=record)
        engine = ctx.offered
        engine_module = importlib.import_module(module.__name__ + ".engine")
        seen = []
        original = engine_module.LCMEngine.get_status

        def probe(self):
            seen.append((len(traceback.extract_stack()), self._in_process_auxiliary_session_id_from_stack(),
                         self._in_process_parent_session_id({}, "child")))
            return original(self)
        engine_module.LCMEngine.get_status = probe
        try:
            aux = Agent("child", parent="s1", _subagent_id="sa-1", _delegate_depth=1)
            engine._auxiliary_session_ids.add("child")
            aux.call(engine.get_status)
        finally:
            engine_module.LCMEngine.get_status = original
        depths[record] = seen
        lab.stop()
    assert depths[True] == depths[False]


# -- round trip, replay smoke, negative controls --------------------------------------------------------

@needs312
def test_round_trip_rebuilds_every_view_and_compress_result(recorded):
    reading = recorded["reader"].read(recorded["proc"])
    assert reading["cause"] == "end" and reading["end_state"] == "ended without exit"
    roots = [e for e in reading["roots"] if e["kind"] in ("engine", "hook")]
    compress = [e for e in roots if e["name"] == "compress"]
    assert len(compress) == 2
    out = recorded["out"]
    canon = lambda v: json.dumps(v, sort_keys=True).encode()  # noqa: E731
    assert canon(compress[0]["result"].messages) == canon(out[next(i for i, v in enumerate(out)
                                                                    if isinstance(v, list) and v)])
    posts = [e for e in roots if e["name"] == "post_llm_call"]
    assert [len(e["args"]["kwargs"]["conversation_history"].messages) for e in posts][:4] == [4, 8, 12, 16]
    assert all(c["parent_seq"] == compress[0]["seq_enter"] for c in compress[0]["children"])
    assert any(c["kind"] == "summary" for c in compress[0]["children"]), (
        [(c["kind"], c["name"]) for c in compress[0]["children"]], compress[0].get("compression"),
        [(e["kind"], e["name"], e["thread"]) for e in reading["roots"] if e["kind"] == "bg"])


def _replay(recorded, tmp_path, name, mutate=None, checkpoint=None):
    """Replay at the recorded home path (rows carry it): the recording moves aside, each replay home moves after."""
    if "at" not in recorded:
        at, moved = recorded["home"], recorded["home"].with_name(recorded["home"].name + "-recorded")
        at.rename(moved)
        recorded.update(at=at, home=moved, proc=moved / recorded["proc"].relative_to(at))
    rep = replay(recorded["proc"], plugin_root=REPO, workdir=tmp_path / name, mutate=mutate, checkpoint=checkpoint,
                 prepare=lambda module: prepare(module, lambda *a, **k: "replay must serve recorded attempts"),
                 home=recorded["at"])
    reference = recorded["home"] / "lcm.db" if checkpoint is None else (
        recorded["proc"] / "checkpoint" / str(checkpoint) / "lcm.db")
    diffs = compare_stores(reference, rep.store, excluded_columns=SQL_CLOCK_COLUMNS,
                           excluded_sessions=rep.excluded_sessions)
    gc.collect()
    recorded["at"].rename(tmp_path / f"{name}-home")
    return rep, diffs


def _drop(predicate):
    def mutate(recording, reader):
        target = next(e for e in recording["events"] if predicate(e))
        recording["events"].remove(target)
        for event in recording["events"]:
            if target in event.get("children", ()):
                event["children"].remove(target)
    return mutate


def _drop_child(predicate):
    def mutate(recording, reader):
        for event in recording["events"]:
            child = next((c for c in event.get("children", ()) if predicate(c)), None)
            if child is not None:
                event["children"].remove(child)
                recording["events"].remove(child)
                return
        raise AssertionError("no such child")
    return mutate


def _break_link(recording, reader):
    for event in recording["roots"]:
        view = (event.get("args") or {}).get("kwargs", {}).get("conversation_history")
        if isinstance(view, reader.ViewValue) and any(op[0] == "e" for op in view.ops):
            i = next(i for i, op in enumerate(view.ops) if op[0] == "e" and view.messages[i]["role"] == "user")
            view.ops[i] = ("n", 10_000_000, None)
            return
    raise AssertionError("no in-place edit recorded")


@needs312
def test_replay_smoke_matches_and_negative_controls_fail(recorded, tmp_path, lab):
    lab.mp.setattr(time, "time", lambda: FROZEN)
    rep, diffs = _replay(recorded, tmp_path, "clean")
    (tmp_path / "replay-diffs.txt").write_text(repr((rep.misses, rep.served,
        [(d[0], d[1], d[2], str(d[3]), str(d[4])) for d in rep.diffs],
        {t: (len(a or []), len(b or []), [r for r in (a or []) if r not in (b or [])],
             [r for r in (b or []) if r not in (a or [])]) for t, (a, b) in diffs.items()})))
    assert rep.diffs == [] and diffs == {} and rep.misses == 0, (
        [(d[0], d[1], d[2], str(d[3])[:300], str(d[4])[:300]) for d in rep.diffs],
        {t: (len(a or []), len(b or []), [r for r in (a or []) if r not in (b or [])][:2],
             [r for r in (b or []) if r not in (a or [])][:2]) for t, (a, b) in diffs.items()}, rep.misses)
    assert rep.compared > 20 and rep.served >= 1
    controls = {
        "summary": _drop_child(lambda c: c.get("kind") == "summary"),
        "in_place_link": _break_link,
        "threshold": _drop(lambda e: e.get("kind") == "attrs" and "threshold_tokens" in e["values"]),
        "subagent_start": _drop(lambda e: e.get("name") == "subagent_start"),
        "rejected": _drop(lambda e: e.get("name") == "record_rejected_compaction"),
    }
    for name, mutate in controls.items():
        rep, diffs = _replay(recorded, tmp_path, name, mutate)
        assert rep.diffs or diffs or rep.misses, f"negative control {name} did not fail"


# -- views ------------------------------------------------------------------------------------------------

@needs312
def test_views_edits_aliasing_clones_and_watched_rows(recorded):
    reading = recorded["reader"].read(recorded["proc"])
    posts = [e for e in reading["roots"] if e["name"] == "post_llm_call"]
    fourth = posts[3]["args"]["kwargs"]["conversation_history"]
    ops = [op[0] for op in fourth.ops]
    assert ops[0] == "e" and ops[1] == "e" and ops[2:12].count("s") == 10
    assert fourth.messages[1]["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'
    turn_complete = next(e for e in reading["roots"] if e["name"] == "on_turn_complete")
    assert {op[0] for op in turn_complete["args"]["messages"].ops} == {"c"}


@needs312
def test_aliasing_and_watched_row_keeps_its_token(lab):
    module, ctx = lab.load(lab.home())
    engine = ctx.offered
    engine.on_session_start("s", platform="cli")
    rec = _recorder()
    row = {"role": "user", "content": "watched"}
    engine.__dict__["_host_rewrite_watch"] = {1: (None, "watched", [row])}
    engine.on_turn_complete([row, row])
    for n in range(3):
        engine.on_turn_complete([{"role": "user", "content": f"other {n}"}])
    row["content"] = "watched, edited"
    engine.on_turn_complete([row])
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    views = [e["args"]["messages"] for e in reader.read(rec.proc)["roots"] if e["name"] == "on_turn_complete"]
    first, last = views[0].ops, views[-1].ops
    assert first[0][1] == first[1][1] and first[1][0] == "s"  # aliasing: one token twice
    assert last[0][0] == "e" and last[0][1] == first[0][1]


# -- identity, nesting, threads ---------------------------------------------------------------------------

@needs312
def test_identity_nesting_threads_and_engine_gone(lab):
    module, ctx = lab.load(lab.home())
    proto, rec = ctx.offered, _recorder()
    proto.on_session_start("s", platform="cli")
    proto.should_compress_info(10) if hasattr(proto, "should_compress_info") else None
    clones = [proto.clone_for_agent() for _ in range(2)]
    barrier = threading.Barrier(2)

    def work(engine, sid):
        barrier.wait()
        engine.on_session_start(sid, platform="cli")
        engine.get_status()
    threads = [threading.Thread(target=work, args=(c, f"t{i}"), name="hermes-hook-x") for i, c in enumerate(clones)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    del clones, threads
    gc.collect()
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    reading = reader.read(rec.proc)
    roots = reading["roots"]
    clone_tokens = [e["result"].value for e in roots if e["name"] == "clone_for_agent"]
    starts = [e for e in roots if e["name"] == "on_session_start" and e["engine"] in clone_tokens]
    assert {e["engine"] for e in starts} == set(clone_tokens)
    assert len({e["thread_token"] for e in starts}) == 2 and len({e["thread"] for e in starts}) == 1
    gone = {e["engine"] for e in reading["events"] if e["kind"] == "engine_gone"}
    assert set(clone_tokens) <= gone
    info = [e for e in roots if e["name"] == "should_compress_info"]
    assert all(c["name"] == "should_compress" for c in info[0]["children"]) if info else True
    seqs = sorted(s for e in reading["events"] for s in ((e["seq"],) if "seq" in e else (e["seq_enter"], e["seq_exit"])))
    assert seqs == list(range(1, len(seqs) + 1))


@needs312
def test_summary_attempt_without_an_open_root_is_a_bg_root(lab):
    module, ctx = lab.load(lab.home())
    escalation = importlib.import_module(module.__name__ + ".escalation")
    thread = threading.Thread(target=lambda: escalation._invoke_summary_attempt("prompt", 10, model="m"))
    thread.start()
    thread.join()
    rec = _recorder()
    rec.writer.drain(5)
    roots = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc)["roots"]
    bg = [e for e in roots if e["kind"] == "bg"]
    assert len(bg) == 1 and bg[0]["thread"] != threading.current_thread().name and bg[0]["parent_seq"] is None


@needs312
def test_two_tools_sharing_one_code_object_keep_their_names(recorded, lab):
    module, ctx = lab.load(lab.home("tools"))
    ctx.tools["lcm_grep"]({"query": "x"})
    ctx.tools["lcm_status"]({})
    rec = _recorder()
    assert ctx.tools["lcm_grep"].__code__ is ctx.tools["lcm_status"].__code__
    rec.writer.drain(5)
    roots = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc)["roots"]
    assert [e["name"] for e in roots if e["kind"] == "tool"] == ["lcm_grep", "lcm_status"]


# -- arguments, summaries, state captures ------------------------------------------------------------------

@needs312
def test_a_raising_compress_is_recorded_and_reaches_the_host_unchanged(lab):
    module, ctx = lab.load(lab.home())
    rec, engine = _recorder(), ctx.offered
    engine.on_session_start("s1", platform="cli", conversation_id="c1")
    engine._compaction_gate = None  # compress raises TypeError inside its own frame
    with pytest.raises(TypeError, match="context manager") as raised:
        engine.compress([{"role": "user", "content": "x"}])
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    root = next(e for e in reader.read(rec.proc)["roots"] if e["name"] == "compress")
    assert root["status"] == "raise" and root["exception"]["type"] == "TypeError"
    assert root["exception"]["sha256"] == hashlib.sha256(str(raised.value).encode()).hexdigest()
    assert rec.counters["recorder.failed"] == 0


@needs312
def test_session_carry_over_hermes_calls_is_a_root(lab):
    """Hermes calls carry_over_new_session_context on the engine at a session switch (bench pin run_agent.py:375)."""
    module, ctx = lab.load(lab.home())
    rec, engine = _recorder(), ctx.offered
    engine.on_session_start("s1", platform="cli", conversation_id="c1")
    moved = engine.carry_over_new_session_context("s1", "s2")
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    roots = [e for e in reader.read(rec.proc)["roots"] if e["name"] == "carry_over_new_session_context"]
    assert len(roots) == 1 and roots[0]["result"] == moved


@needs312
def test_arguments_are_parameters_only(recorded):
    reading = recorded["reader"].read(recorded["proc"])
    compress = next(e for e in reading["roots"] if e["name"] == "compress")
    assert set(compress["args"]) == {"self", "messages", "args", "kwargs"}
    bind = next(e for e in reading["roots"] if e["name"] == "bind_session_state")
    assert bind["args"]["session_db"] == recorded["reader"].Ref("set", True)
    assert not reading["status"]["marks"] or not any(m.get("unsupported") for m in reading["status"]["marks"].values())


@needs312
def test_summary_attempt_fields_and_classifier_parity(lab):
    class Overloaded(Exception):
        status = 429

        def __init__(self):
            super().__init__("Error code: 400 - context length exceeded")
            self.status_code = 400
            self.body = {"error": "context_length_exceeded"}

    def summary(prompt, max_tokens, **kw):
        raise Overloaded()
    module, ctx = lab.load(lab.home(), summary=summary)
    escalation = importlib.import_module(module.__name__ + ".escalation")
    attempt = escalation._invoke_summary_attempt("prompt text", 77, model="m", timeout=5.0)
    assert attempt.error is not None
    rec = _recorder()
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    event = next(e for e in reader.read(rec.proc)["roots"] if e["kind"] == "bg")
    assert event["args"]["prompt_sha256"] == hashlib.sha256(b"prompt text").hexdigest()
    assert event["args"]["max_tokens"] == 77 and event["args"]["timeout"] == 5.0
    stand_in = stand_in_error(event["result"]["error"])
    classifiers = importlib.import_module(module.__name__ + ".summary_attempt")
    for name in ("is_summary_route_config_error", "is_summary_context_length_error", "is_summary_content_filter_error"):
        assert getattr(classifiers, name)(stand_in) == getattr(classifiers, name)(attempt.error), name
    without_status = copy.deepcopy(event["result"]["error"])
    without_status["attrs"].pop("status")
    assert classifiers.is_summary_context_length_error(stand_in_error(without_status)) != \
        classifiers.is_summary_context_length_error(attempt.error)


@needs312
def test_state_captures(recorded):
    reading = recorded["reader"].read(recorded["proc"])
    attrs = [e for e in reading["events"] if e["kind"] == "attrs"]
    assert [e["values"]["threshold_tokens"] for e in attrs if "threshold_tokens" in e["values"]].count(5_000) == 1
    assert [e["values"].get("compression_count") for e in attrs].count(3) == 1
    env = [e for e in reading["events"] if e["kind"] == "env"]
    assert len([e for e in env if e["values"].get("LCM_SENDER_CAPTURE") == "1"]) == 1
    sessions = [e["rows"] for e in reading["events"] if e["kind"] == "sessions"]
    first_s1 = next(i for i, rows in enumerate(sessions) if isinstance(rows.get("s1"), list))
    assert sessions[first_s1]["s1"][0] == "s1"
    command = next(e for e in reading["roots"] if e["kind"] == "command")
    assert command["context"]["HERMES_SESSION_ID"] == "s1" and command["context"]["HERMES_SESSION_KEY"] == "c1"


@needs312
def test_sessions_absence_and_read_error_are_their_own_outcomes(lab):
    home = lab.tmp / "bare"
    home.mkdir()
    module, ctx = lab.load(home)
    engine, rec = ctx.offered, _recorder()
    engine.on_session_start("s", platform="cli")
    write_state_db(home / "state.db", [("s", "cli", None, 1.0, None, None, None)])
    engine.get_status()
    rec.state_db.conn = None
    os.replace(home / "state.db", home / "state.db.moved")
    (home / "state.db").write_bytes(b"not a database")
    engine.get_status()
    rec.writer.drain(5)
    reading = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc)
    outcomes = [e["rows"] for e in reading["events"] if e["kind"] == "sessions"]
    assert reading["process_start"]["sessions"] == "no_database"
    assert {"s": "no_database"} in outcomes
    assert any("$full" in rows for rows in outcomes)
    assert any(rows.get("s") == "read_error" or rows.get("$full") == "read_error" for rows in outcomes)


# -- snapshot, payloads, writer, coverage -----------------------------------------------------------------

@needs312
def test_snapshot_pin_excludes_a_later_commit_and_payloads_match_the_manifest(lab):
    home = prepared_home(lab, "pin")
    payloads = home / "lcm-large-outputs"
    payloads.mkdir(exist_ok=True)
    (payloads / "ref-1.json").write_text('{"content": "big"}')
    stall = threading.Event()
    module = load_plugin(REPO, prefix="hermes_lcm_rec")
    lab.modules.append(module)
    recorder_module = importlib.import_module(module.__name__ + ".recorder")
    original = recorder_module._Writer.__init__

    def init(self, rec):
        original(self, rec)
        self.stall = stall
    lab.mp.setattr(recorder_module._Writer, "__init__", init)
    prepare(module)
    lab.host.home = home
    lab.mp.setenv("LCM_RECORDER_ENABLED", "1")
    lab.mp.setenv("LCM_RECORDER_DIR", str(home / "lcm-recorder"))
    ctx = FakeCtx()
    module.register(ctx)
    lab.engines.append(ctx.offered)
    with sqlite3.connect(home / "lcm.db") as conn:
        conn.execute("CREATE TABLE after_pin (x)")
    conn.close()
    stall.set()
    rec = _recorder()
    rec.writer.drain(10)
    snap = sqlite3.connect(rec.proc / "snapshot" / "lcm.db")
    assert snap.execute("SELECT name FROM sqlite_master WHERE name='after_pin'").fetchall() == []
    assert snap.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0
    snap.close()
    manifest = json.loads((rec.proc / "snapshot" / "large-outputs.manifest.json").read_text())
    assert [m["name"] for m in manifest] == ["ref-1.json"]
    copied = rec.proc / "snapshot" / "large-outputs" / "ref-1.json"
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == manifest[0]["sha256"]
    assert rec.proc.stat().st_mode & 0o777 == 0o700


@needs312
def test_writer_stall_admits_one_oversized_view_then_gaps(lab, monkeypatch):
    module, ctx = lab.load(lab.home())
    rec, engine = _recorder(), ctx.offered
    recorder_module = importlib.import_module(module.__name__ + ".recorder")
    rec.writer.drain(5)
    stall = threading.Event()
    rec.writer.stall = stall
    monkeypatch.setattr(recorder_module, "MAX_QUEUE_BYTES", 1 << 16)
    big = [{"role": "user", "content": "x" * 50, "n": i} for i in range(2_000)]
    engine.on_turn_complete(big)
    started = time.monotonic()
    for _ in range(20):
        engine.get_status()
    assert time.monotonic() - started < 2
    rec.writer.stall = None
    stall.set()
    rec.writer.drain(10)
    reading = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc)
    assert reading["cause"] == "gap"
    assert len([e for e in reading["events"] if e["kind"] == "gap"]) == 0  # the gap itself ends the prefix
    events = recorder_module_events(rec)
    assert [e["kind"] for e in events].count("gap") == 1
    turn_complete = next(e for e in reading["roots"] if e["name"] == "on_turn_complete")
    assert [m["n"] for m in turn_complete["args"]["messages"].messages] == list(range(2_000))


def recorder_module_events(rec):
    return [json.loads(line) for seg in sorted((rec.proc / "events").glob("*.jsonl"))
            for line in seg.read_text().splitlines() if line]


@needs312
@pytest.mark.parametrize("cause", ["hole", "before_seq", "serialization", "unsupported", "mismatch"])
def test_coverage_ends_at_the_right_seq(lab, cause, monkeypatch):
    module, ctx = lab.load(lab.home())
    engine, rec = ctx.offered, _recorder()
    recorder_module = importlib.import_module(module.__name__ + ".recorder")
    engine.on_session_start("s", platform="cli")
    engine.get_status()
    rec.writer.drain(5)
    monkeypatch.setattr(rec, "_before_root", lambda *args: None)  # no state events between the two roots
    expected = rec.last_seq
    if cause == "hole":
        rec._next()
        engine.get_status()
    elif cause == "before_seq":
        rec.failed("test")
        engine.get_status()
    elif cause == "serialization":
        real = recorder_module.plain

        def failing(value, blob=None):
            if isinstance(value, dict) and value.get("name") == "get_status":
                raise TypeError("boom")
            return real(value, blob)
        monkeypatch.setattr(recorder_module, "plain", failing)
        engine.get_status()
        rec.writer.drain(5)
        monkeypatch.setattr(recorder_module, "plain", real)
    elif cause == "unsupported":
        engine.handle_tool_call("lcm_status", {"x": object()})
    else:
        stack = rec._stack()
        stack.append(recorder_module._Entry(None, None, {"seq_enter": rec._next(), "engine": 1, "kind": "engine",
                                                         "name": "orphan"}, engine, False, "orphan"))
        engine.get_status()
        stack.clear()
    engine.get_status()
    rec.writer.drain(5)
    reading = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc)
    assert reading["prefix_end"] == expected, (cause, reading["prefix_end"], expected, reading["cause"])


# -- exit -------------------------------------------------------------------------------------------------

_EXIT_SCRIPT = textwrap.dedent("""
    import atexit, os, sqlite3, sys
    from pathlib import Path
    from tests.recorder_replay import FakeCtx, HostModules, load_plugin
    home, mode = Path(sys.argv[1]), sys.argv[2]
    holder = []
    atexit.register(lambda: holder and holder[0].get_status())  # runs after every later-registered hook (LIFO)
    with HostModules(home):
        module = load_plugin(Path.cwd())
        ctx = FakeCtx()
        module.register(ctx)
        engine = ctx.offered
        holder.append(engine)
        engine.on_session_start("s", platform="cli", conversation_id="c")
        history = [{"role": "user", "content": "repeat", "timestamp": 1760000001.0},
                   {"role": "assistant", "content": "answer", "timestamp": 1760000002.0}]
        ctx.hooks["post_llm_call"][0](session_id="s", platform="cli", conversation_history=list(history))
        history += [{"role": "user", "content": "again", "timestamp": 1760000003.0},
                    {"role": "assistant", "content": "done", "timestamp": 1760000004.0}]
        if mode == "os_exit":
            engine.on_session_end("s", history)
            os._exit(0)
        locker = sqlite3.connect(str(home / "lcm.db"), isolation_level=None)
        locker.execute("BEGIN IMMEDIATE")
        engine.on_session_end("s", history)  # busy: the tail stays pending for the exit flush
        locker.rollback()
        locker.close()
        sys.exit(0)
""")


@needs312
@pytest.mark.parametrize("mode", ["os_exit", "atexit"])
def test_exit_paths(tmp_path, mode):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LCM_")}
    env.update(LCM_EMBEDDINGS_ENABLED="false", LCM_RECORDER_ENABLED="1", LCM_RECORDER_DIR=str(home / "rec"))
    done = subprocess.run([sys.executable, "-c", _EXIT_SCRIPT, str(home), mode], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    sys.path.insert(0, str(REPO))
    import recorder_read  # noqa: E402  (the reader imports nothing from the plugin)
    proc = next((home / "rec" / "proc").iterdir())
    reading = recorder_read.read(proc)
    names = [e.get("name") or e["kind"] for e in reading["events"] if e.get("parent_seq") is None]
    if mode == "os_exit":
        assert names[-1] == "on_session_end" and reading["end_state"] == "ended without exit"
        assert json.loads((proc / "STATUS.json").read_text())["ended"] == "ended without exit"
    else:
        assert "exit_begin" in names
        after = names[names.index("exit_begin") + 1:]
        assert after.index("_flush_pending_tails_at_exit") < after.index("get_status")
        rows = sqlite3.connect(home / "lcm.db").execute("SELECT content FROM messages ORDER BY store_id").fetchall()
        assert [r[0] for r in rows] == ["repeat", "answer", "again", "done"]


# -- failure isolation ------------------------------------------------------------------------------------

@needs312
@pytest.mark.parametrize("error", [RuntimeError, SystemExit])
@pytest.mark.parametrize("where", ["_enter", "_exit", "_register"])
def test_recorder_failures_never_reach_the_host(lab, monkeypatch, error, where):
    module, ctx = lab.load(lab.home())
    engine, rec = ctx.offered, _recorder()
    engine.on_session_start("s", platform="cli")
    expected = engine.get_status()

    def boom(*args, **kwargs):
        raise error("recorder bug")
    monkeypatch.setattr(rec, where, boom)
    if where == "_register":
        ctx2 = FakeCtx()
        captured = rec._capturer(ctx2.register_hook, "hook")
        captured("post_llm_call", print)
        assert ctx2.hooks["post_llm_call"] == [print]
    else:
        status = engine.get_status()
        assert normalize(status) == normalize(expected)
        engine._compaction_gate = None  # the application raises inside the monitored compress frame
        with pytest.raises(TypeError, match="context manager"):
            engine.compress([{"role": "user", "content": "x"}])
    assert rec.counters["recorder.failed"] >= 1 and rec.first_cause is not None


@needs312
def test_keyboard_interrupt_inside_a_callback_is_reraised(lab, monkeypatch):
    module, ctx = lab.load(lab.home())
    rec = _recorder()

    def interrupt(*args):
        raise KeyboardInterrupt
    with monkeypatch.context() as mp:  # undone before teardown's roots run
        mp.setattr(rec, "_enter", interrupt)
        with pytest.raises(KeyboardInterrupt):
            ctx.offered.get_status()
    assert rec.first_cause["cause"] == "interrupted"


@needs312
@pytest.mark.parametrize("fault", ["unwritable", "disk_full"])
def test_unwritable_directory_and_full_disk_change_nothing_the_host_sees(lab, monkeypatch, fault):
    module, ctx = lab.load(lab.home())
    engine, rec = ctx.offered, _recorder()
    rec.writer.drain(5)
    expected = normalize(engine.get_status())
    if fault == "unwritable":
        def refuse(*args, **kwargs):
            raise PermissionError("read-only")
        monkeypatch.setattr(rec.writer, "_append", refuse)
    else:
        def full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(rec.writer, "_append", full)
    for _ in range(3):
        assert normalize(engine.get_status()) == expected
    rec.writer.drain(5)
    assert rec.counters["recorder.failed"] >= 1 and rec.stopped == "writer_failed"


# -- profiles, processes, secrets -------------------------------------------------------------------------

@needs312
def test_overlapping_process_directories_are_flagged(lab, tmp_path):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    ctx.offered.get_status()
    rec.writer.drain(5)
    twin = rec.proc.parent / (rec.proc.name.split("-", 1)[0] + "9-x-" + rec.proc.name.rsplit("-", 1)[-1])
    shutil.copytree(rec.proc, twin)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    assert reader.overlaps(rec.root) == [tuple(sorted((rec.proc.name, twin.name)))]


@needs312
def test_secrets_never_reach_the_recording(recorded, lab):
    lab.mp.setenv("LCM_EMBEDDING_API_KEY", SECRET)
    module, ctx = lab.load(lab.home("secret"))
    ctx.offered.update_model("m", 100_000, base_url=SECRET_HOST, api_key=SECRET)
    ctx.offered.get_status()
    rec = _recorder()
    rec.writer.drain(5)
    for root in (recorded["home"] / "lcm-recorder", rec.root):
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".db"):
                data = path.read_bytes()
                assert SECRET.encode() not in data and SECRET_HOST.encode() not in data, path


# -- sealed checkpoint ------------------------------------------------------------------------------------

@needs312
def test_sealed_checkpoint_prefix_replays_without_exit(lab, tmp_path, monkeypatch):
    base = prepared_home(lab)
    home = copy_home(base, lab.tmp / "live")
    lab.mp.setattr(time, "time", lambda: FROZEN)
    module, ctx = lab.load(home)
    recorder_module = importlib.import_module(module.__name__ + ".recorder")
    rec = _recorder()
    engine = ctx.offered
    post = ctx.hooks["post_llm_call"][0]
    engine.on_session_start("s1", platform="cli", context_length=200_000)
    state_row(home, "s1")
    history = []
    for n in range(1, 3):
        turn(history, n)
        post(session_id="s1", platform="cli", conversation_history=list(history))
    monkeypatch.setattr(recorder_module, "clock", lambda: rec.started + 24 * 3600 + 1)
    engine.get_status()
    rec.writer.drain(10)
    turn(history, 3)
    post(session_id="s1", platform="cli", conversation_history=list(history))  # after the watermark
    rec.writer.drain(10)
    reading = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc, checkpoint=24)
    assert reading["sealed"]["through"] == reading["prefix_end"]
    assert reading["end_state"] == "ended without exit"
    recorded = {"proc": rec.proc, "home": home}
    lab.stop()
    rep, diffs = _replay(recorded, tmp_path, "sealed", checkpoint=24)
    assert rep.diffs == [] and diffs == {}, (rep.diffs[:3], list(diffs)[:3])
    rep, diffs = _replay(recorded, tmp_path, "sealed-drop", checkpoint=24,
                         mutate=_drop(lambda e: e.get("name") == "post_llm_call"))
    assert rep.diffs or diffs


# -- budget -----------------------------------------------------------------------------------------------

def _timed(rec, samples):
    enter, exit_ = rec._enter, rec._exit

    def timed_enter(*args):
        t = time.perf_counter()
        try:
            return enter(*args)
        finally:
            samples[-1] += time.perf_counter() - t

    def timed_exit(*args):
        t = time.perf_counter()
        try:
            return exit_(*args)
        finally:
            samples[-1] += time.perf_counter() - t
    rec._enter, rec._exit = timed_enter, timed_exit


@needs312
def test_turn_path_budget(lab):
    """Turn-thread recorder cost of one post_llm_call root and one on_turn_complete root on 2,000 rows."""
    strict = os.environ.get("RECORDER_BUDGET_STRICT") == "1"
    scale = 1 if strict else 4
    ctx = FakeCtx()
    ctx._manager._context_engine = object()  # inactive engine: the hook root records its view and returns
    module, ctx = lab.load(lab.home(), ctx=ctx)
    rec, post = _recorder(), ctx.hooks["post_llm_call"][0]
    engine = ctx.offered
    import random
    rnd = random.Random(7)
    view = []
    for n in range(500):
        turn(view, n)
        view[-2]["content"] = "o" * rnd.randint(20_000, 100_000)
    rec.writer.drain(30)
    stall = threading.Event()
    rec.writer.stall = stall  # the writer holds a backlog while the turn thread is measured
    numbers = {}
    samples = [0.0]
    _timed(rec, samples)
    post(session_id="s", platform="cli", conversation_history=list(view))  # the session's first view
    numbers["first_view_ms"] = samples[-1] * 1000
    rec._enter, rec._exit = type(rec)._enter.__get__(rec), type(rec)._exit.__get__(rec)
    state = {"view": view}

    def grow():
        state["view"] = state["view"] + [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
        return list(state["view"])
    steady = []
    for _ in range(60):
        steady.append(0.0)
        rec_samples = [0.0]
        _timed(rec, rec_samples)
        post(session_id="s", platform="cli", conversation_history=grow())
        steady[-1] = rec_samples[-1]
        rec._enter, rec._exit = type(rec)._enter.__get__(rec), type(rec)._exit.__get__(rec)
    numbers["steady_median_ms"] = statistics.median(steady) * 1000
    numbers["steady_p99_ms"] = sorted(steady)[int(len(steady) * 0.99) - 1] * 1000
    equal = []
    for _ in range(30):
        fresh = [{k: (v.encode().decode() if isinstance(v, str) else copy.deepcopy(v)) for k, v in m.items()}
                 for m in state["view"]]
        samples = [0.0]
        _timed(rec, samples)
        post(session_id="s", platform="cli", conversation_history=fresh)
        state["view"] = fresh
        equal.append(samples[-1])
        rec._enter, rec._exit = type(rec)._enter.__get__(rec), type(rec)._exit.__get__(rec)
    numbers["equal_strings_median_ms"] = statistics.median(equal) * 1000
    numbers["equal_strings_p99_ms"] = sorted(equal)[int(len(equal) * 0.99) - 1] * 1000
    edited = []
    for i in range(30):
        state["view"][1000 + i]["content"] = f"rewritten {i}"
        samples = [0.0]
        _timed(rec, samples)
        post(session_id="s", platform="cli", conversation_history=list(state["view"]))
        edited.append(samples[-1])
        rec._enter, rec._exit = type(rec)._enter.__get__(rec), type(rec)._exit.__get__(rec)
    numbers["edited_median_ms"] = statistics.median(edited) * 1000
    clones = []
    for _ in range(10):
        samples = [0.0]
        _timed(rec, samples)
        engine.on_turn_complete([_clone(m) for m in state["view"]])
        clones.append(samples[-1])
        rec._enter, rec._exit = type(rec)._enter.__get__(rec), type(rec)._exit.__get__(rec)
    numbers["turn_complete_clones_max_ms"] = max(clones) * 1000
    stall.set()
    rec.writer.stall = None
    rec.writer.drain(120)
    started = time.monotonic()
    for _ in range(5):
        engine.on_turn_complete([_clone(m) for m in state["view"]])
    rec.writer.drain(300)
    numbers["writer_turn_complete_s_per_turn"] = (time.monotonic() - started) / 5
    print("RECORDER BUDGET", json.dumps(numbers, sort_keys=True))
    assert rec.stopped is None, "budget samples must all be recorded"
    assert numbers["steady_median_ms"] <= 5 * scale and numbers["steady_p99_ms"] <= 25 * scale
    assert numbers["equal_strings_median_ms"] <= 20 * scale and numbers["equal_strings_p99_ms"] <= 60 * scale
    assert numbers["edited_median_ms"] <= 5 * scale
    assert numbers["first_view_ms"] <= 60 * scale and numbers["turn_complete_clones_max_ms"] <= 60 * scale


# FIX-PACKET-L3-1-r1 regressions: each test names its review finding.
@needs312
@pytest.mark.parametrize("interloper", ["other_session", "bg"])
def test_fix_r1_01_sessions_cross_root(lab, interloper):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    ctx.offered.on_session_start("X", platform="cli")
    state_row(lab.host.home, "X")
    if interloper == "other_session":
        ctx.offered.on_session_start("Y", platform="cli")
    else:
        rec.session = ""
        rec._before_root(None, "bg", None, {}, None)
    ctx.offered.on_session_start("X", platform="cli")
    rec.writer.drain(5)
    events = recorder_module_events(rec)
    assert any(e["kind"] == "sessions" and isinstance(e["rows"].get("X"), list) for e in events)


@needs312
def test_fix_r1_02_idle_engine_attribute_write(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    clone = ctx.offered.clone_for_agent()
    clone.threshold_tokens = 100
    ctx.offered.get_status()
    a_proto, z_clone = ctx.offered, clone
    def hook():
        if a_proto is not None:
            z_clone.threshold_tokens = 555
    ctx.register_hook("attr_write", hook)
    hook()
    clone.get_status()
    rec.writer.drain(5)
    assert any(e["kind"] == "attrs" and e["values"].get("threshold_tokens") == 555
               for e in recorder_module_events(rec))


@needs312
def test_fix_r1_02_nested_engine_is_busy(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    clone = ctx.offered.clone_for_agent()
    token = rec._token(clone)
    def nested(self):
        assert rec.busy[token] > 0
    lab.mp.setattr(type(clone), "get_status", nested)
    rec._watch(nested.__code__, "engine", "get_status")
    a_proto, z_clone = ctx.offered, clone
    def hook():
        if a_proto is not None:
            z_clone.get_status()
    ctx.register_hook("nested_busy", hook)
    hook()


@needs312
def test_fix_r1_03_ended_sessions_release_views(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    for n in range(12):
        sid = f"ended-{n}"
        ctx.offered.on_session_start(sid, platform="cli")
        ctx.offered.on_turn_complete([{"role": "user", "content": sid}])
        ctx.offered.on_session_end(sid, [])
        assert sid not in rec.views
    assert len(rec.rows) == 0


@needs312
def test_fix_r1_03_gone_engine_releases_views_and_attrs(lab):
    import weakref
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    clone = ctx.offered.clone_for_agent()
    import types
    rec.session = "gone"
    rec.cur = types.SimpleNamespace(engine=clone, views=0)
    rec._copy([{"role": "user", "content": "gone"}], [])
    rec.cur = None
    rec.pre_llm, rec.pre_llm_session = tuple(rec.rows), "gone"
    token, ref = rec._token(clone), weakref.ref(clone)
    del clone
    gc.collect()
    ctx.offered.get_status()
    assert ref() is None
    assert "gone" not in rec.views and token not in rec.attr_base
    assert not rec.rows and not rec.pre_llm


@needs312
def test_fix_r1_03_views_are_bounded(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    for n in range(300):
        rec.session = str(n)
        rec._copy([{"role": "user", "content": str(n)}], [])
    assert len(rec.views) <= 256


@needs312
def test_fix_r1_04_env_before_first_root(lab):
    module, ctx = lab.load(lab.home())
    lab.mp.setenv("LCM_CONTEXT_THRESHOLD", "0.73")
    ctx.offered.get_status()
    rec = _recorder()
    rec.writer.drain(5)
    assert any(e["kind"] == "env" and e["values"].get("LCM_CONTEXT_THRESHOLD") == "0.73"
               for e in recorder_module_events(rec))


@needs312
def test_fix_r1_05_finalizer_does_not_wait_for_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    done = threading.Event()
    def gone():
        rec._gone(999)
        done.set()
    with rec.lock:
        thread = threading.Thread(target=gone, daemon=True)
        thread.start()
        returned = done.wait(0.5)
    thread.join(2)
    assert returned
    ctx.offered.get_status()
    rec.writer.drain(5)
    assert any(e["kind"] == "engine_gone" and e["engine"] == 999 for e in recorder_module_events(rec))


@needs312
def test_fix_r1_05_logging_outside_recorder_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    recorder_module = importlib.import_module(module.__name__ + ".recorder")
    observed = []
    class Handler(logging.Handler):
        def emit(self, record):
            observed.append(rec.lock._is_owned())
    handler = Handler()
    recorder_module.logger.addHandler(handler)
    try:
        with rec.lock:
            rec.lose("probe")
        rec.writer.drain(5)
        rec.writer.sync = True
        rec.cap("probe")
        rec.writer.drain(5)
        assert observed and not any(observed)
    finally:
        recorder_module.logger.removeHandler(handler)


@needs312
@pytest.mark.parametrize("mode", ["stop", "failed_exit", "failed_copy", "unwind"])
def test_fix_r1_06_exit_releases_stack_and_engine(lab, mode):
    import weakref
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    clone = ctx.offered.clone_for_agent()
    ref = weakref.ref(clone)
    real_exit, real_copy = rec._exit, rec._copy
    def boom(*args):
        raise RuntimeError("injected exit")
    def root(self):
        if mode in ("stop", "unwind"):
            rec.stopped = "probe"
        elif mode == "failed_exit":
            rec._exit = boom
        else:
            rec._copy = boom
        if mode == "unwind":
            raise ValueError("host")
    lab.mp.setattr(type(clone), "get_status", root)
    rec._watch(root.__code__, "engine", "get_status")
    if mode == "unwind":
        with pytest.raises(ValueError):
            clone.get_status()
    else:
        clone.get_status()
    rec._exit, rec._copy = real_exit, real_copy
    assert rec._stack() == []
    del clone
    gc.collect()
    assert ref() is None
    rec.stopped = None
    ctx.offered.should_compress(1)
    rec.writer.drain(5)
    assert [e for e in recorder_module_events(rec) if e.get("name") == "should_compress"][-1]["parent_seq"] is None


@needs312
def test_fix_r1_07_atexit_timeout_stops_admission(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    entered, release = threading.Event(), threading.Event()
    def seal():
        entered.set()
        release.wait(5)
        rec._emit({"kind": "checkpoint_sealed", "hours": 24, "through": rec.last_seq}, single=True)
    rec.writer.offer(("call", seal))
    assert entered.wait(2)
    try:
        rec._atexit()
        assert not rec.writer.sync
        done = threading.Event()
        thread = threading.Thread(target=lambda: (ctx.offered.get_status(), done.set()), daemon=True)
        thread.start()
        assert done.wait(0.5)
        status = json.loads((rec.proc / "STATUS.json").read_text())
        assert status["end_state"] == "exit_timeout" and not rec.writer.sync
    finally:
        release.set()


@needs312
def test_fix_r1_08_sync_cap_is_single(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    rec.writer.sync = True
    rec.max_bytes = 0
    rec.cap("test")
    assert [e["kind"] for e in recorder_module_events(rec)].count("cap") == 1


@needs312
def test_fix_r1_09_writer_system_exit_stops_and_reports(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    def boom():
        raise SystemExit("writer probe")
    rec.writer.offer(("call", boom))
    rec.writer.join(2)
    assert rec.writer.dead is not None and rec.stopped is not None
    status = json.loads((rec.proc / "STATUS.json").read_text())
    assert status["end_state"] != "running" and status["first_cause"]["cause"] == "writer:SystemExit"


@needs312
def test_fix_r1_10_queue_charges_slots_and_strings(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    big = [None] * 1_000_000
    assert rm._cost(big) >= 8_000_000
    assert rm._cost("x" * 100_000) >= 100_000
    rec.writer.drain(5)
    stall = threading.Event()
    rec.writer.stall = stall
    lab.mp.setattr(rm, "MAX_QUEUE_BYTES", 16_000_000)
    try:
        for _ in range(5):
            rec._emit({"kind": "large", "value": list(big)}, single=True)
        assert rec.stopped == "gap"
    finally:
        stall.set()


@needs312
def test_fix_r1_11_payload_rewrite_after_pin_and_mode(lab):
    home = prepared_home(lab, "pin-rewrite")
    payloads = home / "lcm-large-outputs"
    payloads.mkdir(exist_ok=True)
    source = payloads / "payload.json"
    source.write_text('{"content":"before"}')
    source.chmod(0o644)
    module = load_plugin(REPO, prefix="hermes_lcm_rec")
    lab.modules.append(module)
    rm = importlib.import_module(module.__name__ + ".recorder")
    original = rm.Recorder._monitor_on
    def monitor(self):
        replacement = source.with_suffix(".tmp")
        replacement.write_text('{"content":"after"}')
        replacement.replace(source)
        original(self)
    lab.mp.setattr(rm.Recorder, "_monitor_on", monitor)
    prepare(module)
    lab.host.home = home
    lab.mp.setenv("LCM_RECORDER_ENABLED", "1")
    lab.mp.setenv("LCM_RECORDER_DIR", str(home / "lcm-recorder"))
    ctx = FakeCtx()
    module.register(ctx)
    lab.engines.append(ctx.offered)
    rec = _recorder()
    rec.writer.drain(5)
    captured = rec.proc / "snapshot" / "large-outputs" / source.name
    manifest = json.loads((rec.proc / "snapshot" / "large-outputs.manifest.json").read_text())
    assert captured.read_text() == '{"content":"before"}'
    assert captured.stat().st_mode & 0o777 == 0o600
    assert manifest[0]["sha256"] == hashlib.sha256(captured.read_bytes()).hexdigest()


@needs312
@pytest.mark.parametrize("damage", ["missing", "torn", "mismatch"])
def test_fix_r1_12_blob_damage_cuts_prefix(lab, damage):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    before = rec.last_seq
    rec._emit({"kind": "probe_blob", "body": rm.Blob("unique damaged body")}, single=True)
    rec.writer.drain(5)
    event = next(e for e in recorder_module_events(rec) if e["kind"] == "probe_blob")
    sha = event["body"]["$b"]
    blob = rec.root / "blobs" / sha[:2] / sha
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_bytes(b'"torn' if damage == "torn" else b'"other"')
    reading = importlib.import_module(module.__name__ + ".recorder_read").read(rec.proc, verify=damage != "torn")
    assert reading["prefix_end"] == before and reading["cause"] == "blob_" + damage


@needs312
def test_fix_r1_12_checkpoint_past_hole_rejected(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    hole = rec._next()
    rec._emit({"kind": "checkpoint_sealed", "hours": 24, "through": hole}, single=True)
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    with pytest.raises(reader.RecordingError):
        reader.read(rec.proc, checkpoint=24)


@needs312
def test_fix_r1_12_blob_fsynced_before_reference(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    rec.writer.drain(5)
    synced = []
    real = os.fsync
    lab.mp.setattr(os, "fsync", lambda fd: (synced.append(os.fstat(fd).st_size), real(fd))[-1])
    with rec.writer.io:
        rec.writer._blob(rm.Blob("durable unique blob"))
        assert len(synced) >= 1


@needs312
@pytest.mark.parametrize("backing_present", [True, False])
def test_fix_r1_13_persisted_output_input_replayed_without_file(lab, backing_present):
    lab.mp.setattr(time, "time", lambda: FROZEN)
    lab.mp.setattr(time, "time_ns", lambda: int(FROZEN * 1e9))
    gmtime = time.gmtime
    lab.mp.setattr(time, "gmtime", lambda *args: gmtime(*args) if args else gmtime(FROZEN))
    home = prepared_home(lab, "persisted-home")
    backing = home / "cache" / "spillover" / "result.txt"
    backing.parent.mkdir(parents=True)
    content = "persisted result " * 100
    if backing_present:
        backing.write_text(content)
    marker = (f"<persisted-output>\nOutput too large ({len(content)} characters).\nFull output saved to: {backing}"
              "\nPreview (first 9 chars):\npersisted\n...\n</persisted-output>")
    module, ctx = lab.load(home)
    ctx.offered.on_session_start("persisted", platform="cli")
    ctx.hooks["post_llm_call"][0](session_id="persisted", platform="cli", conversation_history=[
        {"role": "user", "content": "read result"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "persist", "type": "function",
         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "persist", "content": marker}])
    ctx.offered.shutdown()
    rec = lab.stop()
    events = recorder_module_events(rec)
    assert any(e["kind"] == "input" and e.get("path") == str(backing) for e in events)
    backing.unlink(missing_ok=True)
    recorded = {"home": home, "proc": rec.proc}
    rep, diffs = _replay(recorded, lab.tmp, "replay-persisted")
    assert not rep.diffs
    assert not rep.inputs and not diffs


@needs312
def test_fix_r1_14_replay_database_paths_are_isolated(lab):
    home = lab.home()
    lab.mp.setenv("LCM_DATABASE_PATH", str(home / "custom.db"))
    lab.mp.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH", str(home / "custom-payloads"))
    module, ctx = lab.load(home)
    ctx.offered.on_session_start("isolated", platform="cli")
    ctx.hooks["post_llm_call"][0](session_id="isolated", platform="cli", conversation_history=[
        {"role": "user", "content": "original stays intact"}])
    ctx.offered.shutdown()
    rec = lab.stop()
    original = home / "custom.db"
    before = original.read_bytes(), original.stat().st_mtime_ns
    rep = replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "replay-paths", prepare=prepare)
    assert (original.read_bytes(), original.stat().st_mtime_ns) == before
    assert Path(rep.engines[1]._db_path).resolve() == rep.store.resolve()
    assert rep.store.is_relative_to(lab.tmp / "replay-paths")


@needs312
def test_fix_r1_14_replay_rejects_escaping_paths_before_register(lab):
    module, ctx = lab.load(lab.home())
    rec = lab.stop()
    def mutate(recording, reader):
        recording["process_start"]["env"]["LCM_DATABASE_PATH"] = str(lab.tmp / "escape.db")
    with pytest.raises(ValueError, match="outside replay home"):
        replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "rejected", mutate=mutate, prepare=prepare)


@needs312
def test_fix_r1_15_replay_effective_yaml_configuration(lab):
    pytest.importorskip("yaml")  # replay writes config.yaml with PyYAML; the CI test matrix has none
    home = lab.home()
    (home / "config.yaml").write_text("lcm:\n  context_threshold: 0.73\nauxiliary:\n  compression:\n    max_concurrency: 7\n")
    module, ctx = lab.load(home)
    ctx.offered.get_status()
    rec = lab.stop()
    rep = replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "replay-config", prepare=prepare)
    assert rep.engines[1]._config.context_threshold == 0.73
    assert (rep.state.path.parent / "config.yaml").exists()
    assert not rep.diffs


@needs312
def test_fix_r1_fork_child_stops_without_inherited_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    with rec.lock:
        held = threading.Event()
        def fork_child():
            pid = os.fork()
            if pid == 0:
                if rec.stopped != "fork":
                    os._exit(1)
                ctx.offered.get_status()
                rec._atexit()
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            held.status = status
            held.set()
        thread = threading.Thread(target=fork_child, daemon=True)
        thread.start()
        assert held.wait(2)
    assert held.status == 0


@needs312
def test_fix_r1_c9_dead_writer_drain_is_immediate(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    rec.writer.dead = OSError("dead")
    start = time.monotonic()
    assert rec.writer.drain(0.5) is False
    assert time.monotonic() - start < 0.1


@needs312
def test_fix_r1_c9_start_failure_stops_writer(lab):
    module = load_plugin(REPO, prefix="hermes_lcm_rec")
    lab.modules.append(module)
    rm = importlib.import_module(module.__name__ + ".recorder")
    writers = []
    original = rm._Writer.start
    def start(self):
        writers.append(self)
        original(self)
    def boom(self, *args):
        raise RuntimeError("start probe")
    lab.mp.setattr(rm._Writer, "start", start)
    lab.mp.setattr(rm.Recorder, "_resolve", boom)
    lab.mp.setenv("LCM_RECORDER_ENABLED", "1")
    module.register(FakeCtx())
    assert writers
    writers[0].join(1)
    assert not writers[0].is_alive()


def test_fix_r1_c9_existing_directory_mode_unchanged(tmp_path):
    import recorder
    target = tmp_path / "user-dir"
    target.mkdir(mode=0o755)
    recorder._mkdir(target)
    assert target.stat().st_mode & 0o777 == 0o755
    created = recorder._mkdir(target / "created")
    assert created.stat().st_mode & 0o777 == 0o700


@needs312
def test_fix_r1_c6_epoch_reroll_recorded(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    def reroll(self):
        self._compaction_telemetry_counter_epoch = "rerolled"
    lab.mp.setattr(type(ctx.offered), "get_status", reroll)
    rec._watch(reroll.__code__, "engine", "get_status")
    ctx.offered.get_status()
    rec.writer.drain(5)
    assert any(e["kind"] == "attrs" and e["values"].get("_compaction_telemetry_counter_epoch") == "rerolled"
               for e in recorder_module_events(rec))


def test_fix_r1_c10_docs_match_secret_segments_and_epoch():
    text = (REPO / "docs" / "recorder.md").read_text()
    assert "whole\n`_`-separated segment" in text
    assert "one regenerated inside" not in text


@needs312
def test_fix_r1_n1_payload_copies_are_private(lab):
    home = lab.home()
    payload = home / "lcm-large-outputs" / "public.json"
    payload.parent.mkdir()
    payload.write_text('{}')
    payload.chmod(0o644)
    module, ctx = lab.load(home)
    rec = _recorder()
    rec.writer.drain(5)
    assert (rec.proc / "snapshot" / "large-outputs" / payload.name).stat().st_mode & 0o777 == 0o600


@needs312
def test_fix_r1_15_config_file_changes_replayed(lab):
    pytest.importorskip("yaml")  # replay writes config.yaml with PyYAML; the CI test matrix has none
    home = lab.home()
    config = home / "config.yaml"
    config.write_text("auxiliary:\n  compression:\n    max_concurrency: 7\n")
    module, ctx = lab.load(home)
    config.write_text("auxiliary:\n  compression:\n    max_concurrency: 19\n")
    ctx.offered.get_status()
    rec = lab.stop()
    rep = replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "replay-config-changes", prepare=prepare)
    config_module = importlib.import_module(rep.module.__name__ + ".config")
    assert config_module.hermes_auxiliary_compression_max_concurrency(rep.state.path.parent) == 19


# FIX-PACKET-L3-1-r2: deterministic interleavings and bounded cost probes.
@needs312
def test_fix_r2_01_checkpoint_waits_for_detached_return(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    detached, release = threading.Event(), threading.Event()
    pop = rec._pop
    def pause(code, frame):
        entries = pop(code, frame)
        if entries and threading.current_thread().name == "unfinished-return":
            detached.set()
            assert release.wait(2)
        return entries
    lab.mp.setattr(rec, "_pop", pause)
    rec.started = -1e9
    rec.checkpoints = [24]
    thread = threading.Thread(target=ctx.offered.get_status, name="unfinished-return", daemon=True)
    thread.start()
    try:
        assert detached.wait(2)
        ctx.offered.get_status()
        rec.writer.drain(5)
        assert not [e for e in recorder_module_events(rec) if e["kind"] == "checkpoint_sealed"]
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    rec.writer.drain(5)
    rec.writer.drain(5)  # The seal queues its event behind an already queued sync.
    events = recorder_module_events(rec)
    checkpoint = next(e for e in events if e["kind"] == "checkpoint_sealed")
    assert all(not (e["seq_enter"] <= checkpoint["through"] < e["seq_exit"])
               for e in events if "seq_enter" in e)


@needs312
def test_fix_r2_02_gone_precedes_next_root(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    clone = ctx.offered.clone_for_agent()
    token = rec.engines[clone]
    del clone
    gc.collect()
    assert token in rec.gone
    ctx.offered.get_status()
    rec.writer.drain(5)
    events = recorder_module_events(rec)
    gone = next(e["seq"] for e in events if e["kind"] == "engine_gone" and e["engine"] == token)
    root = next(e["seq_enter"] for e in events if e.get("name") == "get_status")
    assert gone < root


@needs312
def test_fix_r2_03_unwatched_unwind_keeps_cur(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    def unwatched():
        raise ValueError("unwatched")
    def catch():
        try:
            unwatched()
        except ValueError:
            pass
    with rec.lock:
        marker = rec.cur = object()
        thread = threading.Thread(target=catch)
        thread.start()
        thread.join(1)
        try:
            assert not thread.is_alive()
            assert rec.cur is marker
        finally:
            rec.cur = None


@needs312
def test_fix_r2_04_inputs_follow_owning_root(lab):
    from tests.recorder_replay import Replayer, _StateServer, _Threads
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    persisted = importlib.import_module(module.__name__ + ".ingest_persisted")
    markers = {}
    for name in ("A", "B", "unparented"):
        path = lab.host.home / "cache" / "spillover" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name[0])
        markers[name] = (f"<persisted-output>\nOutput too large (1 characters).\nFull output saved to: {path}"
                         f"\nPreview (first 1 chars):\n{name[0]}\n...\n</persisted-output>")
    def read(text):
        return persisted.recover_hermes_persisted_output_with_file_stat(text, hermes_home=str(lab.host.home))
    read(markers["unparented"])
    entered, b_done = threading.Event(), threading.Event()
    def A():
        entered.set()
        assert b_done.wait(2)
        return read(markers["A"])[0]
    def B():
        return read(markers["B"])[0]
    rec._register("tool", ("A", A), {})
    rec._register("tool", ("B", B), {})
    thread = threading.Thread(target=A)
    thread.start()
    assert entered.wait(2)
    try:
        B()
    finally:
        b_done.set()
        thread.join(2)
    rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    recording = reader.read(rec.proc)
    roots = [e for e in recording["roots"] if e["name"] in ("A", "B")]
    assert [e["name"] for e in roots] == ["A", "B"]
    assert not [e for e in recording["roots"] if e["kind"] == "input"]
    recording["events"] = [e for e in recording["events"] if e in roots or e.get("kind") == "input"]
    (lab.tmp / "replay-home").mkdir()
    state = _StateServer(lab.tmp / "replay-home" / "state.db")
    state.full(recording["process_start"]["sessions"])
    rep = Replayer(recording, module, ctx, reader, state)
    ctx.tools.update({name: (lambda n=name: rep.persisted_output(markers[n])[0]) for name in ("A", "B")})
    # The replay callbacks have no recorded parameters.
    workers = _Threads()
    try:
        rep.run(None, threads=workers)
    finally:
        workers.close()
    assert rep.diffs == []


@needs312
def test_fix_r2_05_view_cost_walks_only_copied_blobs(lab):
    module, ctx = lab.load(lab.home())
    rm = importlib.import_module(module.__name__ + ".recorder")
    view = rm.View([("s", i) if i % 2 else ("c", i, i - 1) for i in range(1200)])
    start = time.perf_counter()
    for _ in range(200):
        rm._cost({"view": view})
    print("VIEW COST ms/record", (time.perf_counter() - start) * 1000 / 200)
    sizes, original = [], sys.getsizeof
    lab.mp.setattr(sys, "getsizeof", lambda value: (sizes.append(type(value)), original(value))[-1])
    assert rm._cost(view) >= 1200 * 100
    assert len(sizes) < 10
    assert rm._cost(rm.View([("n", 1, rm.Blob([None] * 1_000_000))])) >= 8_000_000


@needs312
@pytest.mark.parametrize("count", [1000, 10000])
def test_fix_r2_06_sessions_scan_budget(lab, count):
    """A root reads its own session chains and the rows inserted since the last look, never the whole table."""
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    path = rec.state_db.path
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(f"s{i}", "cli", None, 1.0, None, None, "model") for i in range(count)])
        conn.execute("CREATE TABLE other (x)")
    ctx.offered.on_session_start("s0", platform="cli")
    rec.state_db.full()
    writes = {"own": ("UPDATE sessions SET model_config = ? WHERE id = 's0'", lambda i: (str(i),)),
              "other_row": ("UPDATE sessions SET model_config = ? WHERE id = 's1'", lambda i: (str(i),)),
              "insert": ("INSERT INTO sessions VALUES (?, 'cli', 's0', 1.0, NULL, NULL, NULL)", lambda i: (f"n{i}",)),
              "other_table": ("INSERT INTO other VALUES (?)", lambda i: (i,))}
    medians = {}
    for case, (sql, args) in writes.items():
        samples = []
        for i in range(25):
            with sqlite3.connect(path) as conn:
                conn.execute(sql, args(i))
            start = time.perf_counter()
            ctx.offered.get_tool_schemas()
            samples.append((time.perf_counter() - start) * 1000)
        medians[case] = statistics.median(samples)
    print("SESSIONS ROOT", count, json.dumps(medians, sort_keys=True))
    last = rec.state_db.last
    assert last["s0"][-1] == "24" and all(f"n{i}" in last for i in range(25))
    assert last["s1"][-1] == "model"  # no root read s1's chain, so its change waits for one that does
    scale = 1 if os.environ.get("RECORDER_BUDGET_STRICT") == "1" else 4  # as test_turn_path_budget
    assert max(medians.values()) <= 5 * scale


@needs312
def test_fix_r2_06_row_outside_the_chain_is_recorded_at_its_own_root(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    write_state_db(rec.state_db.path, [("X", "cli", None, FROZEN, None, None, None)])
    ctx.offered.on_session_start("X", platform="cli")
    with sqlite3.connect(rec.state_db.path) as conn:
        conn.execute("UPDATE sessions SET end_reason = 'user_exit' WHERE id = 'X'")
    rec.session = ""
    rec._before_root(None, "bg", None, {}, None)
    def ended():
        return [e for e in recorder_module_events(rec)
                if e["kind"] == "sessions" and (e["rows"].get("X") or [None] * 6)[5] == "user_exit"]
    rec.writer.drain(5)
    assert not ended()
    ctx.offered.get_status()
    rec.writer.drain(5)
    assert len(ended()) == 1


@needs312
def test_fix_r2_06_parent_changed_with_its_new_child_is_recorded(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    write_state_db(rec.state_db.path, [("P", "cli", None, FROZEN, None, None, None)])
    rec.state_db.full()
    with sqlite3.connect(rec.state_db.path) as conn:
        conn.execute("INSERT INTO sessions VALUES ('C', 'cli', 'P', 2.0, NULL, NULL, NULL)")
        conn.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = 'P'")
    rec.session = ""
    rec._before_root(None, "bg", None, {"session_id": "C"}, None)
    rec.writer.drain(5)
    rows = [e["rows"] for e in recorder_module_events(rec) if e["kind"] == "sessions"]
    assert any(isinstance(r.get("C"), list) for r in rows)
    assert any((r.get("P") or [None] * 6)[5] == "compression" for r in rows)


@needs312
def test_fix_r2_06_deleted_row_falls_back_to_the_full_diff(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    write_state_db(rec.state_db.path, [("X", "cli", None, FROZEN, None, None, None),
                                       ("Y", "cli", None, FROZEN, None, None, None)])
    rec.state_db.full()
    with sqlite3.connect(rec.state_db.path) as conn:
        conn.execute("DELETE FROM sessions WHERE id = 'X'")
    rec.session = ""
    rec._before_root(None, "bg", None, {}, None)
    rec.writer.drain(5)
    assert any(e["kind"] == "sessions" and e["rows"].get("X") == "no_row" for e in recorder_module_events(rec))
    assert "X" not in rec.state_db.last and isinstance(rec.state_db.last["Y"], tuple)


@needs312
@pytest.mark.parametrize("link_error", [None, "EXDEV", "EPERM"])
def test_fix_r2_07_payload_pin_defers_copy_and_hash(lab, link_error):
    import errno
    home = lab.home()
    source = home / "lcm-large-outputs" / "pin.json"
    source.parent.mkdir()
    source.write_text("before")
    source.chmod(0o644)
    module = load_plugin(REPO, prefix="hermes_lcm_rec")
    lab.modules.append(module)
    prepare(module)
    rm = importlib.import_module(module.__name__ + ".recorder")
    release = threading.Event()
    snapshot, monitor, sha = rm.Recorder._snapshot, rm.Recorder._monitor_on, rm._sha256_file
    hashed, copies = [], []
    copyfileobj = shutil.copyfileobj
    def copy_on_writer(*args, **kwargs):
        copies.append(threading.current_thread().name)
        return copyfileobj(*args, **kwargs)
    lab.mp.setattr(shutil, "copyfileobj", copy_on_writer)
    if link_error:
        lab.mp.setattr(os, "link", lambda *a, **kw: (_ for _ in ()).throw(OSError(getattr(errno, link_error), link_error)))
    def deferred(self, *args):
        assert release.wait(2)
        snapshot(self, *args)
    def at_monitor(self):
        pin = self.proc / "snapshot" / "large-outputs" / source.name
        assert (pin.stat().st_ino == source.stat().st_ino) is (link_error is None)
        assert source.stat().st_mode & 0o777 == 0o644
        replacement = source.with_suffix(".tmp")
        replacement.write_text("after")
        replacement.replace(source)
        release.set()
        monitor(self)
    def hash_on_writer(path):
        if path.name == source.name:
            hashed.append(threading.current_thread().name)
        return sha(path)
    lab.mp.setattr(rm.Recorder, "_snapshot", deferred)
    lab.mp.setattr(rm.Recorder, "_monitor_on", at_monitor)
    lab.mp.setattr(rm, "_sha256_file", hash_on_writer)
    try:
        lab.host.home = home
        lab.mp.setenv("LCM_RECORDER_ENABLED", "1")
        lab.mp.setenv("LCM_RECORDER_DIR", str(home / "lcm-recorder"))
        ctx = FakeCtx()
        module.register(ctx)
        lab.engines.append(ctx.offered)
        rec = _recorder()
        assert rec is not None
        assert rec.writer.drain(5)
        captured = rec.proc / "snapshot" / "large-outputs" / source.name
        assert captured.read_text() == "before"
        assert captured.stat().st_mode & 0o777 == 0o600
        assert hashed == ["lcm-recorder-writer"]
        if link_error is None:
            assert copies == ["lcm-recorder-writer"]
    finally:
        release.set()


@needs312
def test_fix_r2_c7_stopped_registration_avoids_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    with rec.lock:
        rec._fork_child()
        thread = threading.Thread(target=lambda: ctx.register_command("child", lambda: None), daemon=True)
        thread.start()
        thread.join(0.2)
        done = not thread.is_alive()
    thread.join(1)
    assert done


@needs312
def test_fix_r2_c8_cap_never_waits_for_host_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    rec._atexit()
    def tick():
        with rec.writer.io:
            rec.cap("exit disk floor")
    with rec.lock:
        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        thread.join(0.2)
        done = not thread.is_alive()
    thread.join(1)
    assert done
    rec.cap("retry")
    assert rec.stopped == "cap"


@needs312
def test_fix_r2_c9_blob_directories_and_fsync_order(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    rec.writer.drain(5)
    order = []
    fsync, replace = os.fsync, os.replace
    def synced(fd):
        info = os.fstat(fd)
        order.append("directory" if stat.S_ISDIR(info.st_mode) else "file")
        return fsync(fd)
    def renamed(src, dst):
        order.append("rename")
        return replace(src, dst)
    lab.mp.setattr(os, "fsync", synced)
    lab.mp.setattr(os, "replace", renamed)
    with rec.writer.io:
        rec.writer._blob(rm.Blob("directory durability unique"))
    assert order == ["file", "rename", "directory", "directory"]


@needs312
def test_fix_r2_c10_checkpoint_logging_outside_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    seen = []
    lab.mp.setattr(rm, "_pin", lambda path: (_ for _ in ()).throw(OSError("pin failed")))
    lab.mp.setattr(rm.logger, "warning", lambda *args: seen.append(rec.lock._is_owned()))
    with rec.lock:
        rec._checkpoint(24)
    rec._log_pending()
    assert seen == [False]


@needs312
def test_fix_r2_c10_writer_death_releases_queued_sync(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    entered, release = threading.Event(), threading.Event()
    def die():
        entered.set()
        assert release.wait(2)
        raise SystemExit("queued-sync")
    rec.writer.offer(("call", die))
    assert entered.wait(1)
    waiter = threading.Event()
    rec.writer.offer(("sync", waiter), force=True)
    release.set()
    assert waiter.wait(0.5)
    assert rec.writer.dead is not None


@needs312
def test_fix_r2_c10_replay_restores_from_env(lab):
    module, ctx = lab.load(lab.home())
    ctx.offered.get_tool_schemas()
    rec = lab.stop()
    originals = []
    from tests import recorder_replay as rp
    load = rp.load_plugin
    def loaded(*args, **kwargs):
        module = load(*args, **kwargs)
        cls = importlib.import_module(module.__name__ + ".config").LCMConfig
        originals.append((cls, cls.__dict__["from_env"]))
        return module
    lab.mp.setattr(rp, "load_plugin", loaded)
    replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "restore-config", prepare=prepare)
    cls, original = originals[0]
    assert cls.__dict__["from_env"] is original


@needs312
def test_fix_r2_c11_legacy_identity_replayed(lab):
    yaml = pytest.importorskip("yaml")  # as test_fix_r1_15_config_file_changes_replayed
    home = lab.home()
    values = {"context": {"engine": "lcm", "other": "r3-unread-setting"},
              "plugins": {"enabled": ["hermes-lcm", "hermes-lcm-x"], "other": {"setting": "r3-unread-setting"}}}
    (home / "config.yaml").write_text(yaml.safe_dump(values))
    lab.mp.setattr(sys.modules["hermes_cli.config"], "load_config", lambda: values)
    module, ctx = lab.load(home)
    ctx.offered.get_status()
    rec = lab.stop()
    def prepared(module):
        prepare(module)
        config = sys.modules["hermes_cli.config"]
        config.load_config = lambda: yaml.safe_load((config.get_hermes_home() / "config.yaml").read_text())
    rep = replay(rec.proc, plugin_root=REPO, workdir=lab.tmp / "legacy-replay", prepare=prepared)
    assert rep.engines[1].name == "lcm"
    assert rep.diffs == []
    assert not any(b"r3-unread-setting" in path.read_bytes() for path in rec.proc.rglob("*") if path.is_file())


@needs312
def test_fix_r2_cq1_blob_fsynced_before_referring_event(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    rec.writer.drain(5)
    order, real = [], os.fsync
    def synced(fd):
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            order.append(st.st_ino)
        return real(fd)
    with rec.writer.io:
        lab.mp.setattr(os, "fsync", synced)
        rec.writer._append({"kind": "fsync_probe", "body": rm.Blob("fsync reference unique")})
        rec.writer._tick(force=True)
        event_inode = os.fstat(rec.writer.segment.fileno()).st_ino
    assert len(order) >= 2
    assert order[0] != event_inode and order[-1] == event_inode


@needs312
def test_fix_r2_cq1_blob_cut_crosses_checkpoint(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    rec._emit({"kind": "damaged", "value": rm.Blob("checkpoint damaged unique")}, single=True)
    through = rec.last_seq
    rec._emit({"kind": "checkpoint_sealed", "hours": 24, "through": through}, single=True)
    rec.writer.drain(5)
    event = next(e for e in recorder_module_events(rec) if e["kind"] == "damaged")
    sha = event["value"]["$b"]
    (rec.root / "blobs" / sha[:2] / sha).unlink()
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    with pytest.raises(reader.RecordingError, match="checkpoint crosses blob_missing"):
        reader.read(rec.proc, checkpoint=24)


@needs312
@pytest.mark.parametrize("first", [{"session_id": "Y"}, {}])
def test_fix_r3_01_new_child_after_an_unrelated_root_reads_its_parent(lab, first):
    """One commit inserts C under P and ends P. A root elsewhere records the insert first; C's root still walks to P."""
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    write_state_db(rec.state_db.path, [("P", "cli", None, FROZEN, None, None, None),
                                       ("Y", "cli", None, FROZEN, None, None, None)])
    rec.state_db.full()
    with sqlite3.connect(rec.state_db.path) as conn:
        conn.execute("INSERT INTO sessions VALUES ('C', 'cli', 'P', 2.0, NULL, NULL, NULL)")
        conn.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = 'P'")
    rec.session = ""
    rec._before_root(None, "bg", None, first, None)
    rec._before_root(None, "bg", None, {"session_id": "C"}, None)
    rec.writer.drain(5)
    rows = [e["rows"] for e in recorder_module_events(rec) if e["kind"] == "sessions"]
    assert any(isinstance(r.get("C"), list) for r in rows)
    assert any((r.get("P") or [None] * 6)[5] == "compression" for r in rows)


@needs312
def test_fix_r3_02_write_error_death_releases_queued_sync(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    entered, release = threading.Event(), threading.Event()
    def fail():
        entered.set()
        assert release.wait(2)
        raise OSError("disk gone")
    rec.writer.offer(("call", fail))
    assert entered.wait(1)
    waiter = threading.Event()
    rec.writer.offer(("sync", waiter), force=True)
    release.set()
    assert waiter.wait(0.5)
    assert rec.writer.dead is not None and rec.writer.drain(5) is False


@needs312
def test_fix_r3_03_read_error_mid_look_rereads_every_row(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    db = rec.state_db
    write_state_db(db.path, [("P", "cli", None, FROZEN, None, None, None), ("C", "cli", "P", FROZEN, None, None, None)])
    db.full()
    with sqlite3.connect(db.path) as conn:
        conn.execute("UPDATE sessions SET model_config = 'm' WHERE id = 'C'")
        conn.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = 'P'")
    real, reads = db.conn, []
    class Flaky:
        def execute(self, sql, *args):
            if sql.endswith("WHERE id = ?"):
                reads.append(args)
                if len(reads) == 2:
                    raise sqlite3.OperationalError("database is locked")
            return real.execute(sql, *args)
    db.conn = Flaky()
    first = db.changes(["C"])
    assert first == {"C": "read_error"}
    second = db.changes(["C"])
    assert {row[0]: row[5] for row in second["$full"]}["P"] == "compression"
    from tests.recorder_replay import _StateServer
    (lab.tmp / "replay-state").mkdir()
    server = _StateServer(lab.tmp / "replay-state" / "state.db")
    server.full([["P", "cli", None, FROZEN, None, None, None], ["C", "cli", "P", FROZEN, None, None, None]])
    server.apply(first)
    server.apply(second)
    with sqlite3.connect(server.path) as conn:
        assert dict(conn.execute("SELECT id, end_reason FROM sessions")) == {"P": "compression", "C": None}
    conn.close()


@needs312
def test_fix_r3_05_cap_after_a_finished_engine_is_read_as_cap(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rec.writer.drain(5)
    rec.writer.sync = True
    rec.gone.append(10_000)
    rec.cap("test")
    assert rec.writer.drain(5)
    reader = importlib.import_module(module.__name__ + ".recorder_read")
    assert reader.read(rec.proc)["cause"] == "cap"


@needs312
def test_fix_r3_06_forked_child_returns_without_the_inherited_lock(lab):
    module, ctx = lab.load(lab.home())
    rec = _recorder()
    rm = importlib.import_module(module.__name__ + ".recorder")
    held, release, done = threading.Event(), threading.Event(), threading.Event()
    def hold():
        with rec.lock:
            held.set()
            release.wait(5)
    def child():
        frame = sys._getframe()
        for leave in (rec._discard, lambda c, f: rec._exit(c, f, "return", None)):
            rec._stack().append(rm._Entry(frame, frame.f_code, {"kind": "engine", "seq_enter": 1}, None, True, "x"))
            leave(frame.f_code, frame)
        done.set()
    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(1)
    rec._fork_child()
    try:
        threading.Thread(target=child, daemon=True).start()
        assert done.wait(0.5)
    finally:
        release.set()
        holder.join(2)


@needs312
@pytest.mark.parametrize("change", ["replace", "delete"])
def test_fix_r3_08_replaced_or_deleted_state_db_is_noticed(lab, change):
    module, ctx = lab.load(lab.home())
    db = _recorder().state_db
    write_state_db(db.path, [("P", "cli", None, FROZEN, None, None, None)])
    db.full()
    assert db.changes(["P"]) is None
    if change == "replace":
        fresh = db.path.with_name("state.db.next")
        write_state_db(fresh, [("P", "cli", None, FROZEN, None, "user_exit", None)])
        os.replace(fresh, db.path)
        assert {row[0]: row[5] for row in db.changes(["P"])["$full"]}["P"] == "user_exit"
    else:
        db.path.unlink()
        assert db.changes(["P"]) == {"P": "no_database"}


@needs312
def test_fix_r3_09_chain_walk_reaches_as_deep_as_the_lineage_read(lab):
    module, ctx = lab.load(lab.home())
    db = _recorder().state_db
    write_state_db(db.path, [(f"s{i}", "cli", f"s{i - 1}" if i else None, FROZEN, None, "compression", None)
                             for i in range(258)])
    db.full()
    with sqlite3.connect(db.path) as conn:
        conn.execute("UPDATE sessions SET model_config = 'm' WHERE id = 's0'")
    conn.close()
    assert db.changes(["s257"])["s0"][6] == "m"
