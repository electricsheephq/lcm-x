"""Mini-replayer for LCM recorder process directories (#2243): the replay smoke and the day-1 check.

Imports only names present on main and on the v0.29.2 rc1 tree; the plugin and Hermes are loaded from the paths it
is given. ``replay`` runs ``register(fake_ctx)`` on a copy of the recorded snapshot and payloads, drives the recorded
roots in ``seq_enter`` order (each thread token on its own thread, one root at a time), serves recorded summary
attempts by key, and compares every root's return value and status. ``compare_stores`` compares the in-scope tables.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
import queue
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import types
from collections import defaultdict, deque
from pathlib import Path

# Tables outside the comparison: rollups, assertions, entities and vectors (background or LLM-synthesis owned),
# plus FTS shadow tables that mirror compared ones.
EXCLUDED_TABLE_PREFIXES = ("lcm_rollup", "lcm_assertion", "lcm_entit", "lcm_embedding_", "lcm_chunk_", "lcm_memory_",
                           "assertion", "entity", "entities", "vector", "embedding", "chunk",
                           "messages_fts", "nodes_fts", "sqlite_")
# Tool returns that depend on LLM synthesis (expand_query_tools.py:642-665 on both trees): declared gaps.
EXCLUDED_TOOL_RETURNS = frozenset({"lcm_expand_query"})
# Elapsed-time fields (clock-driven, a declared gap), blanked in return values and JSON cells before comparing:
# compaction.py:1762/:2775/:2946 and engine.py:976 ``duration_ms``; engine.py:1945/:2044 ``last_compaction_duration_ms``.
# get_status's ``no_progress_hold.until`` is wall clock plus a monotonic remainder (engine.py:2390).
# Its lifecycle binding and update timestamps also follow the replay clock.
CLOCK_FIELDS = frozenset({"duration_ms", "last_compaction_duration_ms"})
CLOCK_PATHS = frozenset({("no_progress_hold", "until"), ("lifecycle", "current_bound_at"),
                         ("lifecycle", "updated_at")})
# SQL defaults and Python wall-clock writes; row identities remain in the comparison.
SQL_CLOCK_COLUMNS = frozenset({
    "messages.timestamp", "messages.ingested_at", "summary_nodes.created_at", "summary_nodes.earliest_at",
    "summary_nodes.latest_at", "summary_node_provenance.created_at", "host_uid_bindings.created_at",
    "session_lineage.created_at", "message_relations.created_at", "lcm_lifecycle_state.updated_at",
    "lcm_lifecycle_state.current_bound_at", "lcm_lifecycle_state.last_finalized_at",
    "lcm_lifecycle_state.debt_updated_at", "lcm_lifecycle_state.last_maintenance_attempt_at",
    "lcm_lifecycle_state.last_rollover_at", "lcm_lifecycle_state.last_reset_at",
    "lcm_rollup_invalidations.created_at", "lcm_assertion_sources.processed_at", "lcm_assertions.created_at",
    "lcm_assertion_relations.created_at", "lcm_assertion_sources.invalidated_at", "lcm_migration_state.completed_at",
    "lcm_query_corpus_state.changed_at", "lcm_query_views.updated_at", "lcm_query_views.created_at",
    "lcm_query_views.expires_at", "lcm_query_views.lease_expires_at"})
PATH_SETTINGS = ("database_path", "large_output_externalization_path", "extraction_output_path")
ROOT_TIMEOUT = 60
_LOADS = itertools.count()


class ReplayModelBlocked(TimeoutError):
    pass


class _AuxiliaryStub(types.ModuleType):
    def __init__(self):
        super().__init__("agent.auxiliary_client")
        self.replayer = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def blocked(*args, **kwargs):
            caller = traceback.extract_stack(limit=3)[-2]
            if self.replayer is not None:
                self.replayer.blocked_calls += 1
                self.replayer.diffs.append(("blocked_model", name, caller.filename, caller.lineno, caller.name))
            raise ReplayModelBlocked(f"replay blocked agent.auxiliary_client.{name}")
        return blocked


@contextmanager
def _root_sources(event, module):
    wall, ns = time.time, time.time_ns
    started = wall()  # preserve deliberately frozen lab clocks while advancing with a live clock
    time.time = lambda: event["t_enter"][0] + (wall() - started)
    time.time_ns = lambda: int(time.time() * 1_000_000_000)
    store = importlib.import_module(module.__name__ + ".store")
    source = getattr(store, "secrets", None)
    if source is not None and event.get("ledger_salt"):
        salt = bytes.fromhex(event["ledger_salt"])
        store.secrets = types.SimpleNamespace(token_bytes=lambda n: salt if n == len(salt) else source.token_bytes(n))
    try:
        yield
    finally:
        if source is not None:
            store.secrets = source
        time.time, time.time_ns = wall, ns


class FakeCtx:
    """The PluginContext surface ``register()`` uses."""
    context_engine_tool_handlers_receive_messages = True

    def __init__(self):
        self._manager = types.SimpleNamespace(_context_engine=None)
        self.offered = None
        self.hooks, self.tools, self.commands, self.sections = {}, {}, {}, {}

    def register_context_engine(self, engine):
        self.offered = engine
        self._manager._context_engine = engine

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)

    def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
        self.tools[name] = handler

    def register_command(self, name, handler, description=""):
        self.commands[name] = handler

    def register_system_prompt_section(self, name, callback):
        self.sections[name] = callback

    def register_skill(self, *args, **kwargs):
        return None


class HostModules:
    """Install the Hermes modules LCM imports (home, task-local session context, profile names)."""

    def __init__(self, home):
        self.home, self.context, self.saved = Path(home), {}, {}
        self.profiles = {}

    def __enter__(self):
        hermes_cli = types.ModuleType("hermes_cli")
        hermes_cli.__path__ = []
        config = types.ModuleType("hermes_cli.config")
        config.get_hermes_home = lambda: self.home
        def load_config():
            path = self.home / "config.yaml"
            if not path.exists():
                return {}
            import yaml  # only a recording that captured a config file needs a YAML reader

            return yaml.safe_load(path.read_text()) or {}
        config.load_config = load_config
        hermes_cli.config = config
        gateway = types.ModuleType("gateway")
        gateway.__path__ = []
        session_context = types.ModuleType("gateway.session_context")
        session_context.get_session_env = lambda name, default="": self.context.get(name, default)
        gateway.session_context = session_context
        constants = types.ModuleType("hermes_constants")
        constants.profile_name_for_home = lambda home: self.profiles.get(str(Path(home)), "")
        for name, module in (("hermes_constants", constants), ("hermes_cli", hermes_cli),
                             ("hermes_cli.config", config), ("gateway", gateway),
                             ("gateway.session_context", session_context)):
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_plugin(plugin_root, prefix="hermes_lcm_replay"):
    name = f"{prefix}_{next(_LOADS)}"
    spec = importlib.util.spec_from_file_location(name, str(Path(plugin_root) / "__init__.py"),
                                                  submodule_search_locations=[str(plugin_root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unload_plugin(module):
    name = module.__name__
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        sys.modules.pop(key, None)


def write_state_db(path, rows):
    """A state.db holding the recorded ``sessions`` projection (id, source, parent, started, ended, reason, model)."""
    path = Path(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, "
                     "started_at REAL, ended_at REAL, end_reason TEXT, model_config TEXT)")
        conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()


class _StateServer:
    """Serve the recorded sessions outcomes, including no database and read errors (files are renamed, never removed)."""

    def __init__(self, path):
        self.path, self.parked = Path(path), itertools.count()

    def _park(self):
        if self.path.exists():
            os.replace(self.path, self.path.with_name(f"state.db.parked-{next(self.parked)}"))

    def full(self, rows):
        self._park()
        if isinstance(rows, list):
            write_state_db(self.path, [tuple(r) for r in rows])
        elif rows == "read_error":
            self.path.write_bytes(b"not a database")

    def apply(self, rows):
        if "$full" in rows:
            return self.full(rows["$full"])
        for sid, row in rows.items():
            if row == "no_database":
                self._park()
            elif row == "read_error":
                self._park()
                self.path.write_bytes(b"not a database")
            else:
                if self.path.exists():
                    try:
                        sqlite3.connect(self.path).execute("SELECT 1 FROM sessions LIMIT 1").fetchall()
                    except sqlite3.DatabaseError:
                        self._park()
                if row == "no_row":
                    if self.path.exists():
                        with sqlite3.connect(self.path) as conn:
                            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                        conn.close()
                else:
                    write_state_db(self.path, [tuple(row)])


class _Threads:
    """One replay thread per recorded thread token; the driver runs one root at a time."""

    def __init__(self):
        self.workers = {}
        self.abandoned = []

    def run(self, token, fn):
        worker = self.workers.get(token)
        if worker is None:
            inbox = queue.Queue()

            def loop():
                while True:
                    job = inbox.get()
                    if job is None:
                        return
                    job()
            thread = threading.Thread(target=loop, name=f"replay-thread-{token}", daemon=True)
            thread.start()
            worker = self.workers[token] = (inbox, thread)
        box = {}
        done = threading.Event()

        def job():
            try:
                box["value"] = fn()
                box["status"] = "return"
            except BaseException as exc:  # the root's own outcome, compared below
                box["value"], box["status"] = exc, "raise"
            done.set()
        worker[0].put(job)
        if not done.wait(ROOT_TIMEOUT):
            self.abandoned.append(self.workers.pop(token))
            worker[0].put(None)
            raise TimeoutError("replay root timed out")
        return box["status"], box["value"]

    def close(self):
        for inbox, thread in self.workers.values():
            inbox.put(None)
        for inbox, thread in [*self.workers.values(), *self.abandoned]:
            thread.join(5)
        return sum(thread.is_alive() for _, thread in self.abandoned)


def _stub(*args, **kwargs):
    return False


class Replayer:
    def __init__(self, recording, module, ctx, reader, state):
        self.rec, self.module, self.ctx, self.reader, self.state = recording, module, ctx, reader, state
        self.engines = {1: ctx.offered}
        self.contexts = {1: ctx}
        self.blocked_calls = self.key_misses = 0
        self.objects = {}
        self.homes = {}
        self.seeds = {}
        self.misses = 0
        self.served = 0
        self.diffs = []
        self.compared = 0
        self.pool = defaultdict(deque)
        self.root_attempts = deque()
        self.inputs = deque()
        for event in recording["events"]:
            if event.get("kind") == "summary":
                self.pool[self._key(event["args"])].append(event)

    @staticmethod
    def _key(args):
        return args["prompt_sha256"], args["model"], args["max_tokens"]

    def summary(self, prompt, max_tokens, model="", timeout=None, reasoning_effort="", *, budget=None):
        attempt_module = importlib.import_module(self.module.__name__ + ".summary_attempt")
        key = (hashlib.sha256(str(prompt).encode()).hexdigest(), model, max_tokens)
        event = self.pool[key].popleft() if self.pool[key] else None
        if event is not None and event in self.root_attempts:
            self.root_attempts.remove(event)
        elif event is None and self.root_attempts:
            self.key_misses += 1
            self.diffs.append(("summary_key_miss", "summary", "key", self._key(self.root_attempts[0]["args"]), key))
            event = self.root_attempts.popleft()
            self.pool[self._key(event["args"])].remove(event)
        started = time.monotonic()
        if event is None:
            self.misses += 1
            return attempt_module.SummaryAttempt(text=None, error=RuntimeError("replay: no recorded summary attempt"),
                                                 route={}, contract="",
                                                 timing=attempt_module.SummaryTiming(started, started, False))
        self.served += 1
        result = event["result"]
        _start, end, installed = result["timing"]
        ended = None if end is None else started + end - (_start or 0)
        return attempt_module.SummaryAttempt(
            text=result["text"], error=stand_in_error(result["error"]), route=dict(result["route"] or {}),
            contract=result["contract"] or "", timing=attempt_module.SummaryTiming(started, ended, installed))

    def persisted_output(self, text, *, hermes_home=None):
        persisted = importlib.import_module(self.module.__name__ + ".ingest_persisted")
        if not persisted._is_hermes_persisted_output_marker(text):
            return None
        if not self.inputs:
            raise AssertionError("replay: no recorded persisted-output input")
        event = self.inputs.popleft()
        if text != event["args"]["text"]:
            raise AssertionError("replay: persisted-output input differs")
        result = event["result"]
        return (result["content"], result["stat"]) if result["outcome"] == "ok" else None

    # -- values -------------------------------------------------------------------------------------
    def value(self, raw, key=None):
        reader = self.reader
        if isinstance(raw, reader.ViewValue):
            out = []
            for (op, token, _of), content in zip(raw.ops, raw.messages):
                if op == "s":
                    out.append(self.objects[token])
                elif op == "e":
                    obj = self.objects[token]
                    obj.clear()
                    obj.update(copy.deepcopy(content))
                    out.append(obj)
                else:
                    obj = self.objects[token] = copy.deepcopy(content)
                    out.append(obj)
            return out
        if isinstance(raw, reader.Ref):
            if raw.kind == "engine":
                return self.engines.get(raw.value)
            if raw.kind == "set":
                if not raw.value:
                    return None if key != "api_key" and key != "base_url" else ""
                return object() if key == "session_db" else ("set" if key in ("api_key", "base_url") else _stub)
            return None
        if isinstance(raw, dict):
            return {k: (self.home(v) if k == "hermes_home" else self.value(v, k)) for k, v in raw.items()}
        if isinstance(raw, list):
            return [self.value(v) for v in raw]
        if isinstance(raw, tuple):
            return tuple(self.value(v) for v in raw)
        return raw

    def home(self, recorded):
        if recorded == self.rec["process_start"]["hermes_home"]:
            return str(self.state.path.parent)
        if recorded in self.rec["process_start"].get("replay_homes", {}):
            target = Path(self.rec["process_start"]["replay_homes"][recorded])
            target.mkdir(parents=True, exist_ok=True)
            return str(target)
        if recorded not in self.homes:
            target = self.state.path.parent / f"home-{len(self.homes) + 1}"
            target.mkdir(parents=True, exist_ok=True)
            self.homes[recorded] = str(target)
        return self.homes[recorded]

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace(str(self.state.path.parent), self.rec["process_start"]["hermes_home"])
        if isinstance(value, list) and value and all(isinstance(m, dict) and "role" in m for m in value):
            return [copy.deepcopy(m) for m in value]
        if isinstance(value, dict):
            return {k: self.normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.normalize(v) for v in value)
        if isinstance(value, set):
            return {self.normalize(v) for v in value}
        for token, engine in self.engines.items():
            if value is engine and engine is not None:
                return self.reader.Ref("engine", token)
        if callable(value):
            return self.reader.Ref("set", True)
        return value

    def recorded(self, value):
        if isinstance(value, self.reader.ViewValue):
            return value.messages
        if isinstance(value, dict):
            return {k: self.recorded(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.recorded(v) for v in value)
        return value

    # -- driving ------------------------------------------------------------------------------------
    def tool_name(self, event):
        if event["kind"] == "tool":
            return event["name"]
        if event["kind"] == "engine" and event["name"] == "handle_tool_call":
            return self.value((event.get("args") or {}).get("name"), "name")
        return None

    def call(self, event):
        kind, name, args = event["kind"], event["name"], event.get("args") or {}
        positional, kwonly, star, starstar = event["sig"]
        values = {k: self.value(v, k) for k, v in args.items()}
        if kind == "engine":
            target = values.pop("self")
            fn = getattr(target, name)
            positional = positional[1:]
        elif kind == "hook":
            fn = self.contexts.get(event.get("engine"), self.ctx).hooks[name][0]
        elif kind == "tool":
            fn = self.contexts.get(event.get("engine"), self.ctx).tools[name]
        elif kind == "command":
            fn = self.contexts.get(event.get("engine"), self.ctx).commands[name]
        elif kind == "prompt_section":
            fn = self.contexts.get(event.get("engine"), self.ctx).sections[name]
        else:
            module = sys.modules.get(f"{self.module.__name__}.maintenance_scheduler") or importlib.import_module(
                f"{self.module.__name__}.engine")
            fn = getattr(module, name)
        call_args = [values.pop(n) for n in positional if n in values]
        call_args += list(values.pop(star, ())) if star else []
        call_kwargs = dict(values.pop(starstar, {}) or {}) if starstar else {}
        call_kwargs.update(values)
        return fn(*call_args, **call_kwargs)

    def bind_result(self, event, value):
        result = event.get("result")
        if isinstance(result, self.reader.ViewValue) and isinstance(value, list):
            for (_op, token, _of), obj in zip(result.ops, value):
                self.objects[token] = obj
        if isinstance(result, self.reader.Ref) and result.kind == "engine":
            self.engines[result.value] = value
            for name, seeded in self.seeds.pop(result.value, {}).items():
                setattr(value, name, self.value(seeded))

    def apply_state(self, event, host):
        kind = event["kind"]
        if kind == "register":
            self.apply_state({"kind": "env", "values": event["env"]}, host)
            host.home = Path(self.home(event["hermes_home"]))
            host.home.mkdir(parents=True, exist_ok=True)
            ctx = FakeCtx()
            self.module.register(ctx)
            token = event["engine"]
            self.engines[token], self.contexts[token] = ctx.offered, ctx
            for name, value in self.seeds.pop(token, {}).items():
                setattr(ctx.offered, name, self.value(value))
        elif kind == "seed":
            engine = self.engines.get(event["engine"])
            if engine is None:
                self.seeds[event["engine"]] = dict(event["values"])
            for name, value in event["values"].items() if engine is not None else ():
                setattr(engine, name, self.value(value))
        elif kind == "attrs":
            engine = self.engines.get(event["engine"])
            for name, value in (event["values"] or {}).items():
                if engine is None:
                    continue
                if isinstance(value, self.reader.Ref) and value.kind == "absent":
                    engine.__dict__.pop(name, None)
                elif isinstance(value, self.reader.Ref) and value.kind == "set":
                    setattr(engine, name, _stub if value.value else None)
                else:
                    setattr(engine, name, self.value(value))
        elif kind == "env":
            for key in [k for k in os.environ if k.startswith("LCM_")]:
                os.environ.pop(key)
            for key, value in event["values"].items():
                if isinstance(value, str) and key != "HERMES_HOME":
                    os.environ[key] = _env_value(key, value, self.rec["process_start"], self.state.path.parent)
            os.environ["LCM_RECORDER_ENABLED"] = "0"
        elif kind == "config_files":
            _config_files(event["files"], self.rec["process_start"], self.state.path.parent, self.reader)
        elif kind == "sessions":
            self.state.apply(event["rows"])
        elif kind == "engine_gone":
            self.engines.pop(event["engine"], None)
            gc.collect()

    def run(self, host, *, threads):
        for event in self.rec["events"]:
            if event.get("parent_seq") is not None:
                continue
            kind = event["kind"]
            if "seq_enter" not in event:
                self.apply_state(event, host)
                continue
            if kind == "bg":
                continue
            if kind == "command":
                host.context.clear()
                context = event.get("context") or {}
                host.context.update({k: v for k, v in context.items() if isinstance(v, str)})
                engine = self.engines.get(event.get("engine"))
                if engine is not None:
                    host.profiles[str(Path(engine._hermes_home))] = context.get("engine_profile", "")
            self.root_attempts = deque(c for c in _walk(event) if c.get("kind") == "summary")
            self.inputs = deque(sorted((c for c in _walk(event) if c.get("kind") == "input"),
                                       key=lambda c: c["seq_enter"]))
            with _root_sources(event, self.module):
                status, value = threads.run(event["thread_token"], lambda e=event: self.call(e))
            if status == "return":
                self.bind_result(event, value)
            self.compare(event, status, value)
        return self

    def compare(self, event, status, value):
        self.compared += 1
        if status != event["status"]:
            self.diffs.append((event["seq_enter"], event["name"], "status", event.get("exception") or event["status"],
                               "".join(traceback.format_exception(value)) if status == "raise" else status))
            return
        if status == "raise":
            if type(value).__qualname__ != event["exception"]["type"]:
                self.diffs.append((event["seq_enter"], event["name"], "exception", event["exception"]["type"],
                                   type(value).__qualname__))
            return
        if self.tool_name(event) in EXCLUDED_TOOL_RETURNS:
            return
        expected, actual = scrub(self.recorded(event.get("result"))), scrub(self.normalize(value))
        if expected != actual:
            self.diffs.append((event["seq_enter"], event["name"], "result", expected, actual))


def _walk(event):
    for child in event.get("children") or ():
        yield child
        yield from _walk(child)


def stand_in_error(record):
    """Rebuild the error the summary classifiers read; ``TimeoutError`` survives through the MRO."""
    if record is None:
        return None
    base = TimeoutError if "TimeoutError" in record["mro"] else Exception
    attrs = record["attrs"]

    class StandIn(base):
        def __str__(self):
            return record["str"]
    StandIn.__name__ = StandIn.__qualname__ = record["type"]
    error = StandIn(record["str"])
    for key in ("status", "status_code", "body"):
        if key in attrs:
            setattr(error, key, attrs[key])
    if record.get("response"):
        response = types.SimpleNamespace()
        for key in ("status_code", "text"):
            if f"response.{key}" in attrs:
                setattr(response, key, attrs[f"response.{key}"])
        error.response = response
    return error


def replay(proc, *, plugin_root, workdir, host_path=None, checkpoint=None, mutate=None, prepare=None, home=None):
    """Replay one process directory; returns the Replayer (``diffs``, ``misses``, ``store``).

    ``prepare(module)`` installs the same host helpers the recording ran with (a test's token counter). ``home``
    (a path that does not exist yet) replays at the recorded Hermes home, so rows that carry the home path or a
    digest of it compare equal."""
    plugin_root, workdir = Path(plugin_root), Path(workdir)
    reader = _load_reader()
    recording = reader.read(proc, checkpoint=checkpoint)
    if mutate is not None:
        mutate(recording, reader)
    start = recording["process_start"]
    home = Path(home) if home is not None else workdir / "home"
    home.mkdir(parents=True)
    config_homes = {str(Path(path).parent) for e in recording["events"] for path in e.get("files", {})}
    config_homes |= {e["hermes_home"] for e in recording["events"] if e.get("kind") == "register"}
    start["replay_homes"] = {path: str(home / f"profile-{i}") for i, path in enumerate(sorted(config_homes))
                             if path != start["hermes_home"]}
    db = home / Path(start["db_path"]).name
    if recording["snapshot"] is not None:
        shutil.copy2(recording["snapshot"], db)
    start["payload_dir"] = (recording["status"].get("snapshot") or {}).get("payload_dir", "")
    if recording["payloads"].is_dir():
        shutil.copytree(recording["payloads"], home / "lcm-large-outputs")
    state = _StateServer(home / "state.db")
    state.full(start["sessions"])
    saved_path = list(sys.path)
    if host_path is not None and str(host_path) not in sys.path:
        sys.path.insert(0, str(host_path))
    replay_env = {key: _env_value(key, value, start, home)
        for key, value in (start.get("env") or {}).items() if isinstance(value, str) and key != "HERMES_HOME"}
    saved_env = {k: v for k, v in os.environ.items() if k.startswith("LCM_")}
    for key in saved_env:
        os.environ.pop(key)
    os.environ.update(replay_env)
    os.environ["LCM_RECORDER_ENABLED"] = "0"
    threads = _Threads()
    module = config_module = original_descriptor = None
    auxiliary = _AuxiliaryStub()
    saved_auxiliary = sys.modules.get("agent.auxiliary_client")
    missing = object()
    agent = None
    try:
        agent = importlib.import_module("agent")
        saved_attribute = vars(agent).get("auxiliary_client", missing)
        sys.modules["agent.auxiliary_client"] = agent.auxiliary_client = auxiliary
        with HostModules(home) as host:
            module = load_plugin(plugin_root)
            _config_files(start.get("config_files") or {}, start, home, reader)
            config_module = importlib.import_module(module.__name__ + ".config")
            original_descriptor = config_module.LCMConfig.__dict__["from_env"]
            original_from_env = config_module.LCMConfig.from_env
            effective = original_from_env()
            for key, value in start["config"].items():
                if not isinstance(value, reader.Ref):
                    if key in PATH_SETTINGS and value:
                        value = _replay_path(value, start, home)
                    setattr(effective, key, value)
            for path in (effective.database_path or db, effective.large_output_externalization_path or home / "lcm-large-outputs",
                         effective.extraction_output_path or home / "lcm-extractions", state.path):
                _inside_home(path, home)
            first_config = [effective]
            config_module.LCMConfig.from_env = classmethod(
                lambda cls: first_config.pop() if first_config else original_from_env())
            if prepare is not None:
                prepare(module)
            host_uid = importlib.import_module(module.__name__ + ".host_uid")
            host_uid.hashlib = _TagShim(str((home / "state.db").resolve()), start["state_db_path"])
            escalation = importlib.import_module(module.__name__ + ".escalation")
            ctx = FakeCtx()
            replayer = Replayer(recording, module, ctx, reader, state)
            auxiliary.replayer = replayer
            escalation._invoke_summary_attempt = replayer.summary
            for name in ("ingest_persisted", "ingest_protection", "engine", "reconcile"):
                persisted = importlib.import_module(module.__name__ + "." + name)
                persisted.recover_hermes_persisted_output_with_file_stat = replayer.persisted_output
            module.register(ctx)
            replayer.engines[1] = ctx.offered
            replayer.run(host, threads=threads)
            replayer.store = db
            replayer.excluded_sessions = {s for s, marks in recording["marks"].items() if marks.get("aux_context")}
            return replayer
    finally:
        abandoned = threads.close()
        sys.path[:] = saved_path
        if agent is not None and not abandoned:
            if saved_auxiliary is None:
                sys.modules.pop("agent.auxiliary_client", None)
            else:
                sys.modules["agent.auxiliary_client"] = saved_auxiliary
            if saved_attribute is missing:
                vars(agent).pop("auxiliary_client", None)
            else:
                agent.auxiliary_client = saved_attribute
        if original_descriptor is not None:
            config_module.LCMConfig.from_env = original_descriptor
        for key in [k for k in os.environ if k.startswith("LCM_")]:
            os.environ.pop(key)
        os.environ.update(saved_env)
        if abandoned:
            raise TimeoutError(f"replay root timed out; isolation kept for {abandoned} abandoned worker(s)")


def _inside_home(path, home):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(Path(home).resolve()):
        raise ValueError("path resolves outside replay home")
    return str(path)


def _replay_path(value, start, home):
    path = Path(value).expanduser().resolve()
    mappings = [(start["db_path"], home / Path(start["db_path"]).name),
                (start.get("payload_dir"), home / "lcm-large-outputs"), (start["hermes_home"], home),
                *[(old, Path(new)) for old, new in start.get("replay_homes", {}).items()]]
    mappings = sorted(((Path(src).expanduser().resolve(), dst) for src, dst in mappings if src),
                      key=lambda pair: len(pair[0].parts), reverse=True)
    for source, dest in mappings:
        if path.is_relative_to(source):
            return _inside_home(dest / path.relative_to(source), home)
    return _inside_home(path, home)


def _env_value(key, value, start, home):
    if key in {"LCM_" + name.upper() for name in PATH_SETTINGS}:
        return _replay_path(value, start, home) if value else value
    if Path(value).is_absolute():
        try:
            return _replay_path(value, start, home)
        except ValueError:
            pass
    return value


def _config_files(files, start, home, reader):
    def clean(value, key=None):
        if isinstance(value, reader.Ref):
            return None
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items() if not isinstance(v, reader.Ref)}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and key in PATH_SETTINGS and value:
            return _replay_path(value, start, home)
        if isinstance(value, str) and Path(value).is_absolute():
            try:
                return _replay_path(value, start, home)
            except ValueError:
                pass
        return value
    for source, entry in files.items():
        target = Path(_replay_path(source, start, home))
        if entry.get("state") == "absent":
            target.unlink(missing_ok=True)
        else:
            values = clean(entry.get("values") or {})
            import yaml  # only a recording that captured a config file needs a YAML writer

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(values))


class _TagShim:
    """``host_uid``'s lineage tag hashes the resolved state.db path: present the recorded path, not the replay's."""

    def __init__(self, replay_path, recorded_path):
        import hashlib as real
        self._real = real
        self._map = {replay_path.encode(): str(Path(recorded_path).resolve()).encode()}

    def sha256(self, data=b"", **kwargs):
        return self._real.sha256(self._map.get(data, data), **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _load_reader():
    spec = importlib.util.spec_from_file_location(f"_recorder_read_{next(_LOADS)}",
                                                 Path(__file__).resolve().parents[1] / "recorder_read.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scrub(value, parent=None):
    """``value`` with every CLOCK_FIELDS key and CLOCK_PATHS (parent, key) blanked, including inside JSON text."""
    if isinstance(value, dict):
        return {k: ("<clock>" if k in CLOCK_FIELDS or (parent, k) in CLOCK_PATHS else scrub(v, k))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scrub(v) for v in value)
    if isinstance(value, str) and value[:1] in "{[" and any(
            f in value for f in CLOCK_FIELDS | {key for _, key in CLOCK_PATHS}):
        try:
            return json.dumps(scrub(json.loads(value)))
        except ValueError:
            return value
    return value


def store_rows(db, *, excluded_columns=(), excluded_sessions=(), excluded_prefixes=EXCLUDED_TABLE_PREFIXES):
    """Every in-scope table's rows, sorted, without the excluded columns and the excluded sessions' rows."""
    out = {}
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        for table in tables:
            if excluded_prefixes and table.startswith(tuple(excluded_prefixes)):
                continue
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            keep = [c for c in columns if f"{table}.{c}" not in excluded_columns]
            session_cols = [i for i, c in enumerate(keep) if c in ("session_id", "conversation_id")]
            rows = conn.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in keep)} FROM "{table}"').fetchall()
            rows = [r for r in rows if not any(r[i] in excluded_sessions for i in session_cols)]
            out[table] = sorted((tuple(scrub(c) for c in r) for r in rows), key=repr)
    finally:
        conn.close()
    return out


def compare_stores(expected, actual, **kwargs) -> dict:
    kwargs.setdefault("excluded_columns", SQL_CLOCK_COLUMNS)
    a, b = store_rows(expected, **kwargs), store_rows(actual, **kwargs)
    return {t: (a.get(t), b.get(t)) for t in sorted(set(a) | set(b)) if a.get(t) != b.get(t)}
