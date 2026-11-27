"""Main-against-branch differential acceptance suite for #2230 plugin unload.

Always runs full branch-side invariants (never skipped on shallow CI shards).
When a comparison base tree is available (via ``LCMX_2230_BASE_TREE``,
``LCMX_2230_BASE_REF``, or auto-detected ``upstream/main`` / ``origin/main`` /
``main`` != ``HEAD``), runs every scenario identically against ``main`` and asserts
that the branch stores every distinct row ``main`` stores, stores no row more
times than ``main`` does, and closes all SQLite handles cleanly on unload.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from tests.differential.paired import ROOT, _load
from tests.differential.trees import PrerequisiteUnavailable, materialize, resolve

_ROW_SQL = "SELECT session_id, conversation_id, role, content FROM messages ORDER BY store_id"


def _git_sha(ref: str) -> str | None:
    res = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True, text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else None


@pytest.fixture(scope="module")
def diff_trees(tmp_path_factory):
    """Return ``(branch_root, main_tree_or_None)`` and clean up loaded modules."""
    require_base = os.environ.get("LCMX_2230_REQUIRE_BASE") == "1"
    main_tree: Path | None = None
    env_tree = os.environ.get("LCMX_2230_BASE_TREE")
    env_ref = os.environ.get("LCMX_2230_BASE_REF")

    if env_tree:
        candidate = Path(env_tree)
        if not candidate.is_dir():
            pytest.fail(f"LCMX_2230_BASE_TREE={env_tree!r} is not a directory")
        main_tree = candidate
    elif env_ref is not None:
        if env_ref in ("", "none"):
            if require_base:
                pytest.fail("LCMX_2230_REQUIRE_BASE=1 but LCMX_2230_BASE_REF disables base comparison")
        else:
            try:
                sha = resolve(env_ref)
                main_tree = materialize(sha, tmp_path_factory.mktemp("diff2230-base"))
            except PrerequisiteUnavailable as exc:
                pytest.fail(f"LCMX_2230_BASE_REF={env_ref!r} unavailable: {exc}")
    else:
        head_sha = _git_sha("HEAD")
        for ref in ("upstream/main", "origin/main", "main"):
            sha = _git_sha(ref)
            if sha and sha != head_sha:
                main_tree = materialize(sha, tmp_path_factory.mktemp("diff2230-base"))
                break
        if main_tree is None and require_base:
            pytest.fail("LCMX_2230_REQUIRE_BASE=1 but no local upstream/main, origin/main, or main != HEAD")

    loaded_prefixes: list[str] = []
    yield ROOT, main_tree, loaded_prefixes
    for prefix in loaded_prefixes:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                del sys.modules[key]


class _Env:
    """Isolated runtime harness wrapping either ``main`` or ``branch`` under one API."""

    def __init__(self, tree: Path, *, is_branch: bool, monkeypatch, loaded_prefixes: list[str], tag: str):
        self.tree = tree
        self.is_branch = is_branch
        self.monkeypatch = monkeypatch
        self.loaded_prefixes = loaded_prefixes
        self.tag = tag
        self._load_counter = 0
        self.tmpdir = tempfile.TemporaryDirectory(prefix=f"lcm2230-{tag}-")
        self.root = Path(self.tmpdir.name)
        self.home = self.root / "home"
        self.foreign_home = self.root / "foreign"
        self.state_db = self.home / "state.db"
        self.lcm_db = self.home / "lcm.db"
        self.engines: list = []
        self.ctxs: list = []
        self._install_host(self.home)
        self.pkg = self.fresh_pkg()

    def _install_host(self, home: Path) -> None:
        host = ModuleType("hermes_cli")
        cfg = ModuleType("hermes_cli.config")
        cfg.get_hermes_home = lambda: home
        cfg.load_config = lambda: {}
        host.config = cfg
        self.monkeypatch.setitem(sys.modules, "hermes_cli", host)
        self.monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg)
        self.monkeypatch.setenv("HERMES_HOME", str(self.root / "launching"))
        self.monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "1")
        self.monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")

    def fresh_pkg(self):
        self._load_counter += 1
        name = f"_diff2230_{self.tag}_{self._load_counter}"
        self.loaded_prefixes.append(name)
        return _load(self.tree, name)

    def register(self, *, pkg=None, manager=None, profile_scoped: bool = True, supports_unload: bool = True):
        pkg = pkg or self.pkg
        if manager is None:
            manager = SimpleNamespace(_context_engine=None, home_path=self.home if profile_scoped else None)
        else:
            manager._context_engine = None
            manager.home_path = self.home if profile_scoped else None
        ctx = SimpleNamespace(
            _manager=manager,
            _load_abandoned=False,
            context_engine_tool_handlers_receive_messages=True,
            hooks={}, tools={}, commands={},
        )
        registered: list = []

        def take_slot(engine):
            registered.append(engine)
            self.engines.append(engine)
            manager._context_engine = engine

        ctx.register_context_engine = take_slot
        ctx.register_skill = lambda *a, **k: None
        ctx.register_system_prompt_section = lambda *a: None
        ctx.register_hook = lambda n, h: ctx.hooks.update({n: h})
        ctx.register_tool = lambda **k: ctx.tools.update({k["name"]: k["handler"]})
        ctx.register_command = lambda n, h, **k: ctx.commands.update({n: h})
        if supports_unload:
            ctx.unload_callbacks = []
            ctx.on_unload = ctx.unload_callbacks.append
        pkg.register(ctx)
        self.ctxs.append(ctx)
        return registered[-1], ctx

    def unload(self, ctx) -> None:
        if not self.is_branch:
            return
        for cb in getattr(ctx, "unload_callbacks", ()):
            cb()
        ctx._manager._context_engine = None

    def set_draining(self, prototype, ctx) -> None:
        if not self.is_branch:
            return
        prototype._unload_group.closing.set()
        ctx._manager._context_engine = None

    def init_state_db(self, sessions=(), messages=()) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.state_db)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, "
                "parent_session_id TEXT, started_at REAL, ended_at REAL, end_reason TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, "
                "timestamp REAL, message_uid TEXT, tool_call_id TEXT, tool_calls TEXT)"
            )
            if sessions:
                db.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?)", sessions)
            if messages:
                db.executemany(
                    "INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES (?, ?, ?, ?, ?)",
                    messages,
                )

    def bind(self, engine, session_id: str) -> None:
        engine.bind_session_state(SimpleNamespace(db_path=self.state_db), session_id)

    def rows(self, db_path: Path | None = None) -> list[tuple]:
        path = db_path or self.lcm_db
        if not path.exists():
            return []
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            return db.execute(_ROW_SQL).fetchall()

    def session_rows(self, session_id: str, db_path: Path | None = None) -> list[tuple]:
        return [r for r in self.rows(db_path) if r[0] == session_id]

    def open_fds_for(self, db_path: Path | None = None) -> list[str]:
        fd_dir = Path("/proc/self/fd")
        if not fd_dir.is_dir():
            return []
        target = os.path.realpath(db_path or self.lcm_db)
        hits: list[str] = []
        for entry in fd_dir.iterdir():
            try:
                resolved = os.path.realpath(os.readlink(entry))
            except OSError:
                continue
            if resolved == target or resolved.startswith(target + "-"):
                hits.append(resolved)
        return hits

    def close(self) -> None:
        for ctx in reversed(self.ctxs):
            for cb in getattr(ctx, "unload_callbacks", ()):
                try:
                    cb()
                except Exception:
                    pass
        for engine in reversed(self.engines):
            try:
                engine.shutdown()
            except Exception:
                pass
        self.tmpdir.cleanup()


def _assert_main_parity(label: str, main_rows: list[tuple], branch_rows: list[tuple]) -> None:
    mc, bc = Counter(main_rows), Counter(branch_rows)
    missing = set(mc) - set(bc)
    assert not missing, f"[{label}] branch missing rows stored by main: {missing}"
    extra_dups = {k: (bc[k], mc[k]) for k in mc if bc[k] > mc[k]}
    assert not extra_dups, f"[{label}] branch duplicated rows beyond main: {extra_dups}"
    unseen_dups = {k: v for k, v in bc.items() if k not in mc and v > 1}
    assert not unseen_dups, f"[{label}] branch duplicated rows absent from main: {unseen_dups}"


def _assert_handles_closed_and_no_recreate(env: _Env, late_call) -> None:
    if not env.is_branch:
        return
    assert env.open_fds_for() == []
    if env.lcm_db.exists():
        env.lcm_db.unlink()
    late_call()
    assert not env.lcm_db.exists()


def _run_differential(diff_trees, monkeypatch, label: str, fn):
    branch_root, main_tree, prefixes = diff_trees
    with monkeypatch.context() as branch_patch:
        b_env = _Env(branch_root, is_branch=True, monkeypatch=branch_patch, loaded_prefixes=prefixes, tag="branch")
        try:
            b_rows = fn(b_env)
        finally:
            b_env.close()
    if main_tree is None:
        return None, b_rows
    with monkeypatch.context() as main_patch:
        m_env = _Env(main_tree, is_branch=False, monkeypatch=main_patch, loaded_prefixes=prefixes, tag="main")
        try:
            m_rows = fn(m_env)
        finally:
            m_env.close()
    _assert_main_parity(label, m_rows, b_rows)
    return m_rows, b_rows


def _msg(role: str, content: str, ts: float | None = None, **extra) -> dict:
    row = {"role": role, "content": content, **extra}
    if ts is not None:
        row["timestamp"] = ts
    return row


# ---------------------------------------------------------------------------
# Round 8 Blocking Items 1-7 & Not-blocking #2, #3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reload_mode", ["gap", "same-module", "fresh-import"])
def test_item1_transient_readonly_pending_tail_not_dropped(diff_trees, monkeypatch, reload_mode):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("s1", platform="cli")
        q0 = _msg("user", "item1 q0", 1.0)
        a0 = _msg("assistant", "item1 a0", 2.0)
        e.ingest([deepcopy(q0)])

        engine_mod = importlib.import_module(env.pkg.__name__ + ".engine")
        scheduler_mod = importlib.import_module(env.pkg.__name__ + ".maintenance_scheduler")
        if hasattr(scheduler_mod, "_SESSION_END_SHUTDOWN_TIMEOUT_MS"):
            env.monkeypatch.setattr(scheduler_mod, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 50)

        with closing(sqlite3.connect(env.lcm_db, isolation_level=None)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            try:
                e.on_session_end("s1", [deepcopy(q0), deepcopy(a0)])
            finally:
                writer.rollback()

        if env.is_branch:
            ro_active = threading.Event()
            ro_active.set()
            seen_ro = threading.Event()
            orig_append = e._append_ingest_batch

            def fail_ro(batch):
                if ro_active.is_set():
                    seen_ro.set()
                    raise sqlite3.OperationalError("attempt to write a readonly database")
                return orig_append(batch)

            env.monkeypatch.setattr(e, "_append_ingest_batch", fail_ro)
            env.monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "0.05")
            try:
                env.unload(ctx)
                assert seen_ro.wait(5)
                assert e._pending_session_tails
            finally:
                ro_active.clear()
            deadline = time.monotonic() + 5
            while not e._stable_use_closed and time.monotonic() < deadline:
                time.sleep(0.01)
            assert e._stable_use_closed
        elif reload_mode == "fresh-import":
            # In baseline main, simulate process shutdown flushing pending tails before fresh import.
            e.on_session_end("s1", [deepcopy(q0), deepcopy(a0)])

        cur_ctx = ctx
        if reload_mode == "same-module" and env.is_branch:
            _, cur_ctx = env.register(manager=ctx._manager)
        elif reload_mode == "fresh-import":
            _, cur_ctx = env.register(pkg=env.fresh_pkg(), manager=ctx._manager)

        q1 = _msg("user", "item1 q1", 3.0)
        a1 = _msg("assistant", "item1 a1", 4.0)
        if not env.is_branch and e._pending_session_tails:
            e.on_session_end("s1", [deepcopy(q0), deepcopy(a0)])
        cur_ctx.hooks["post_llm_call"](
            session_id="s1", platform="cli", conversation_history=[q0, a0, q1, a1], context_compressor=e,
        )
        e.on_session_end("s1", [q0, a0, q1, a1])
        out = env.rows()
        _ = engine_mod
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("s1", [q0, a0, q1, a1]))
        return out

    _, b_rows = _run_differential(diff_trees, monkeypatch, f"item1-{reload_mode}", scenario)
    assert len(b_rows) == 4


@pytest.mark.parametrize("window", ["gap", "draining"])
@pytest.mark.parametrize("past_gate", [False, True])
def test_item2_undo_to_oldest_turn_without_state_db_row(diff_trees, monkeypatch, window, past_gate):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("undo-s", platform="cli")
        prefix = []
        pair_count = 15 if past_gate else 2
        for i in range(pair_count):
            prefix.append(_msg("user", f"old-q-{i}", float(2 * i + 1)))
            prefix.append(_msg("assistant", f"old-a-{i}", float(2 * i + 2)))
        e.ingest(deepcopy(prefix))
        env.init_state_db([("undo-s", "cli", None, 1.0, None, None)])

        if window == "gap":
            env.unload(ctx)
        else:
            env.set_draining(proto, ctx)

        u1 = [_msg("user", "rewound q1", 100.0)]
        e.should_compress_preflight(deepcopy(u1))
        after_pre1 = env.rows()
        assert len(after_pre1) == len(prefix) + 1

        t1 = u1 + [_msg("assistant", "rewound a1", 101.0)]
        e.on_turn_complete(deepcopy(t1), interrupted=True)
        after_int1 = env.rows()
        assert len(after_int1) == len(prefix) + 2

        u2 = t1 + [_msg("user", "rewound q2", 102.0)]
        e.should_compress_preflight(deepcopy(u2))
        after_pre2 = env.rows()
        assert len(after_pre2) == len(prefix) + 3

        t2 = u2 + [_msg("assistant", "rewound a2", 103.0)]
        e.on_session_end("undo-s", deepcopy(t2))
        final_rows = env.rows()
        assert len(final_rows) == len(prefix) + 4
        if window == "draining":
            env.unload(ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.should_compress_preflight(deepcopy(t2)))
        return after_pre1 + after_int1 + after_pre2 + final_rows

    _run_differential(diff_trees, monkeypatch, f"item2-{window}-gate{past_gate}", scenario)


@pytest.mark.parametrize("reload_after_switch", [False, True])
def test_item3_switch_away_and_back_leaves_no_stale_nomination(diff_trees, monkeypatch, reload_after_switch):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("A", platform="cli")
        history = [_msg("user", f"seed-q-{i}", float(2 * i + 1)) if j == 0
                   else _msg("assistant", f"seed-a-{i}", float(2 * i + 2))
                   for i in range(15) for j in (0, 1)]
        e.ingest(deepcopy(history))
        env.init_state_db([("A", "cli", None, 1.0, None, None), ("Z", "cli", None, 2.0, None, None)])

        e.on_session_reset()
        env.bind(e, "Z")  # Refused on main because A is non-empty and unfinalized.
        # /resume A skips bind_session_state on Hermes.

        env.unload(ctx)
        cur_ctx = ctx
        if reload_after_switch and env.is_branch:
            _, cur_ctx = env.register(manager=ctx._manager)

        u1 = history + [_msg("user", "after switch back q1", 100.0)]
        e.should_compress_preflight(deepcopy(u1))
        t1 = u1 + [_msg("assistant", "interrupted a1", 101.0)]
        e.on_turn_complete(deepcopy(t1), interrupted=True)
        r1 = env.rows()
        assert len(r1) == len(history) + 1

        u2 = t1 + [_msg("user", "after switch back q2", 102.0)]
        e.should_compress_preflight(deepcopy(u2))
        t2 = u2 + [_msg("assistant", "interrupted a2", 103.0)]
        e.on_turn_complete(deepcopy(t2), interrupted=True)
        r2 = env.rows()
        assert len(r2) == len(history) + 3

        e.on_session_end("A", deepcopy(t2))
        r3 = env.rows()
        if reload_after_switch:
            env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("A", deepcopy(t2)))
        return r1 + r2 + r3

    _run_differential(diff_trees, monkeypatch, f"item3-reload{reload_after_switch}", scenario)


@pytest.mark.parametrize("case", ["child", "preexisting", "no-reset"])
def test_item4_gap_cli_reset_variants_match_main(diff_trees, monkeypatch, case):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("empty", platform="cli")
        parent = "parent" if case == "child" else None
        sessions = [("empty", "cli", None, 1.0, None, None)]
        if case == "preexisting":
            sessions.append(("fresh", "cli", parent, 0.5, None, None))
        env.init_state_db(sessions)
        env.unload(ctx)
        if case != "no-reset":
            e.on_session_reset()
        env.bind(e, "fresh")
        if case != "preexisting":
            env.init_state_db(sessions + [("fresh", "cli", parent, time.time(), None, None)])

        v1 = [_msg("user", "gap q1", 10.0), _msg("assistant", "gap a1", 11.0)]
        e.should_compress_preflight(deepcopy(v1[:1]))
        e.on_turn_complete(deepcopy(v1), interrupted=True)
        r_gap = env.rows()

        cur_ctx = ctx
        if env.is_branch:
            _, cur_ctx = env.register(manager=ctx._manager)
        v2 = v1 + [_msg("user", "hook q2", 12.0), _msg("assistant", "hook a2", 13.0)]
        cur_ctx.hooks["post_llm_call"](session_id="fresh", platform="cli", conversation_history=deepcopy(v2), context_compressor=e)
        e.on_session_end("fresh", deepcopy(v2))
        r_end = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("fresh", deepcopy(v2)))
        return r_gap + r_end

    _run_differential(diff_trees, monkeypatch, f"item4-{case}", scenario)


@pytest.mark.parametrize("op", ["preflight", "compress", "interrupted", "session-end"])
@pytest.mark.parametrize("reloaded", [False, True])
def test_item5a_mixed_lane_matches_main(diff_trees, monkeypatch, op, reloaded):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("S", platform="cli")
        h = [_msg("user", "q0", 1.0), _msg("assistant", "a0", 2.0)]
        e.ingest(deepcopy(h))
        env.unload(ctx)
        with closing(sqlite3.connect(env.lcm_db)) as db, db:
            db.execute(
                "INSERT INTO messages (session_id, conversation_id, role, content, token_estimate, timestamp) "
                "SELECT session_id, 'foreign', role, 'foreign row', token_estimate, timestamp "
                "FROM messages ORDER BY store_id LIMIT 1"
            )
        cur_ctx = ctx
        if reloaded and env.is_branch:
            _, cur_ctx = env.register(manager=ctx._manager)
        cur = h + [_msg("user", "q1", 3.0)]
        if op == "preflight":
            e.should_compress_preflight(deepcopy(cur))
        elif op == "compress":
            e.compress(deepcopy(cur), force=True)
        elif op == "interrupted":
            cur.append(_msg("assistant", "a1", 4.0))
            e.on_turn_complete(deepcopy(cur), interrupted=True)
        else:
            cur.append(_msg("assistant", "a1", 4.0))
            e.on_session_end("S", deepcopy(cur))
        out = env.rows()
        if reloaded:
            env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.should_compress_preflight(deepcopy(cur)))
        return out

    _run_differential(diff_trees, monkeypatch, f"item5a-{op}-reload{reloaded}", scenario)


@pytest.mark.parametrize("history", ["debt", "frontier", "maintenance"])
def test_item5b_empty_reset_lifecycle_history_matches_main(diff_trees, monkeypatch, history):
    sql = {
        "debt": "debt_kind='review-repro', debt_size_estimate=1",
        "frontier": "last_finalized_frontier_store_id=7",
        "maintenance": "last_maintenance_attempt_at=7.0",
    }[history]

    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("A", platform="cli")
        with closing(sqlite3.connect(env.lcm_db)) as db, db:
            db.execute(f"UPDATE lcm_lifecycle_state SET {sql} WHERE conversation_id='A'")
        env.unload(ctx)
        e.on_session_reset()
        env.bind(e, "B")
        env.init_state_db([("B", "cli", None, time.time(), None, None)])
        q = [_msg("user", "first turn", 2.0)]
        e.should_compress_preflight(deepcopy(q))
        out = env.rows()
        _assert_handles_closed_and_no_recreate(env, lambda: e.should_compress_preflight(deepcopy(q)))
        return out

    _, b_rows = _run_differential(diff_trees, monkeypatch, f"item5b-{history}", scenario)
    expected_sid = "B" if history == "debt" else "A"
    assert b_rows == [(expected_sid, expected_sid, "user", "first turn")]


@pytest.mark.parametrize("source", ["cli", "desktop"])
@pytest.mark.parametrize("used_a", [False, True])
def test_item6_tui_desktop_branch_during_unload_stores_after_fresh_import(diff_trees, monkeypatch, source, used_a):
    def scenario(env: _Env):
        proto1, ctx1 = env.register()
        seed = [_msg("user", "seed q0", 1.0), _msg("assistant", "seed a0", 2.0)]
        e_a = proto1.clone_for_agent()
        e_a.on_session_start("A", platform=source)
        if used_a:
            e_a.ingest(deepcopy(seed))
        env.unload(ctx1)
        env.init_state_db([
            ("A", source, None, 1.0, None, None),
            ("B", source, "A", 3.0, None, None),
        ])
        proto2, ctx2 = env.register(pkg=env.fresh_pkg(), manager=ctx1._manager)
        if used_a:
            e_a.should_compress_preflight(deepcopy(seed))
        view = deepcopy(seed)
        stages: list[tuple] = []
        for idx in (1, 2, 3):
            u = _msg("user", f"branch q{idx}", float(2 * idx + 1))
            a = _msg("assistant", f"branch a{idx}", float(2 * idx + 2))
            view.extend([u, a])
            with closing(sqlite3.connect(env.state_db)) as db, db:
                db.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES ('B', 'assistant', ?, ?, NULL)",
                    (a["content"], a["timestamp"]),
                )
            ctx2.hooks["post_llm_call"](session_id="B", platform=source, conversation_history=deepcopy(view))
            b_now = env.session_rows("B")
            assert len(b_now) == 2 + 2 * idx
            stages.extend(b_now)
        all_rows = env.rows()
        env.unload(ctx2)
        _assert_handles_closed_and_no_recreate(
            env,
            lambda: ctx2.hooks["post_llm_call"](session_id="B", platform=source, conversation_history=deepcopy(view)),
        )
        return stages + all_rows

    _run_differential(diff_trees, monkeypatch, f"item6-{source}-used{used_a}", scenario)


@pytest.mark.parametrize("timing", ["gap", "draining", "same-module-before", "same-module-after",
                                    "fresh-import-before", "fresh-import-after"])
@pytest.mark.parametrize("pass_compressor", [True, False])
def test_item7_cli_branch_around_reload_no_extra_duplication(diff_trees, monkeypatch, timing, pass_compressor):
    def scenario(env: _Env):
        proto1, ctx1 = env.register()
        e = proto1.clone_for_agent()
        e.on_session_start("A", platform="cli")
        seed = [
            _msg("user", "A q0", 1.0), _msg("assistant", "A a0", 2.0),
            _msg("user", "A q1", 3.0), _msg("assistant", "A a1", 4.0),
        ]
        e.ingest(deepcopy(seed))
        env.init_state_db([
            ("A", "cli", None, 1.0, 4.5, "branched"),
            ("R", "cli", "A", 4.5, None, None),
        ])
        cur_ctx = ctx1
        if timing == "draining":
            env.set_draining(proto1, ctx1)
        elif timing in ("gap", "same-module-after", "fresh-import-after"):
            env.unload(ctx1)
            if timing == "same-module-after" and env.is_branch:
                _, cur_ctx = env.register(manager=ctx1._manager)
            elif timing == "fresh-import-after":
                _, cur_ctx = env.register(pkg=env.fresh_pkg(), manager=ctx1._manager)
        elif timing in ("same-module-before", "fresh-import-before"):
            e.on_session_reset()
            env.bind(e, "R")
            env.unload(ctx1)
            if timing == "same-module-before" and env.is_branch:
                _, cur_ctx = env.register(manager=ctx1._manager)
            elif timing == "fresh-import-before":
                _, cur_ctx = env.register(pkg=env.fresh_pkg(), manager=ctx1._manager)
        if not timing.endswith("-before"):
            e.on_session_reset()
            env.bind(e, "R")

        u2 = deepcopy(seed) + [_msg("user", "R q2", 5.0)]
        e.should_compress_preflight(deepcopy(u2))
        s1 = env.rows()
        t2 = u2 + [_msg("assistant", "R a2", 6.0)]
        with closing(sqlite3.connect(env.state_db)) as db, db:
            db.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES ('R', 'assistant', ?, ?, NULL)",
                (t2[-1]["content"], t2[-1]["timestamp"]),
            )
        extra = {"context_compressor": e} if pass_compressor else {}
        cur_ctx.hooks["post_llm_call"](session_id="R", platform="cli", conversation_history=deepcopy(t2), **extra)
        s2 = env.rows()
        e.on_session_end("R", deepcopy(t2))
        s3 = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("R", deepcopy(t2)))
        return s1 + s2 + s3

    _run_differential(diff_trees, monkeypatch, f"item7-{timing}-comp{pass_compressor}", scenario)


def test_not_blocking_2_new_x_resume_a_reload_mid_turn(diff_trees, monkeypatch):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("A", platform="cli")
        h = [_msg("user", "q0", 1.0), _msg("assistant", "a0", 2.0)]
        e.ingest(deepcopy(h))
        e.on_session_reset()
        env.bind(e, "X")
        # /resume A skips bind_session_state on Hermes.
        u1 = h + [_msg("user", "q1", 3.0)]
        e.should_compress_preflight(deepcopy(u1))
        env.unload(ctx)
        cur_ctx = ctx
        if env.is_branch:
            _, cur_ctx = env.register(manager=ctx._manager)
        t1 = u1 + [_msg("assistant", "a1", 4.0)]
        cur_ctx.hooks["post_llm_call"](
            session_id="A", platform="cli", conversation_history=deepcopy(t1), context_compressor=e,
        )
        mid = env.rows()
        e.on_session_end("A", deepcopy(t1))
        end = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("A", deepcopy(t1)))
        return mid + end

    _run_differential(diff_trees, monkeypatch, "nb2-new-x-resume-a-midturn", scenario)


def test_not_blocking_3_turn_completed_inside_unload_drain(diff_trees, monkeypatch):
    def scenario(env: _Env):
        proto, ctx = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("A", platform="cli")
        h = [_msg("user", "q0", 1.0), _msg("assistant", "a0", 2.0)]
        e.ingest(deepcopy(h))
        env.set_draining(proto, ctx)
        t1 = h + [_msg("user", "q1", 3.0), _msg("assistant", "a1", 4.0)]
        ctx.hooks["post_llm_call"](
            session_id="A", platform="cli", conversation_history=deepcopy(t1), context_compressor=e,
        )
        mid = env.rows()
        assert len(mid) == 4
        e.on_session_end("A", deepcopy(t1))
        end = env.rows()
        env.unload(ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("A", deepcopy(t1)))
        return mid + end

    _run_differential(diff_trees, monkeypatch, "nb3-turn-in-drain", scenario)


# ---------------------------------------------------------------------------
# Core Lifecycle Sequences from test_issue_2152_lazy_storage & test_issue_2230_reload_gaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timing", ["gap", "draining", "ready", "second-gap"])
def test_diff_empty_and_nonempty_cli_new_sequences(diff_trees, monkeypatch, timing):
    def scenario(env: _Env):
        proto, old = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("orig", platform="cli")
        seed = [_msg("user", "orig q", 1.0), _msg("assistant", "orig a", 2.0)]
        e.ingest(deepcopy(seed))
        env.init_state_db([("orig", "cli", None, 1.0, 2.5, "new_session"), ("fresh", "cli", None, 3.0, None, None)])
        if timing == "second-gap":
            env.unload(old)
            proto, old = env.register(manager=old._manager)
            e.should_compress_preflight(deepcopy(seed))
        if timing == "draining":
            env.set_draining(proto, old)
        else:
            env.unload(old)
        cur_ctx = old
        if timing == "ready" and env.is_branch:
            _, cur_ctx = env.register(manager=old._manager)

        e.on_session_end("orig", deepcopy(seed))
        e.on_session_reset()
        env.bind(e, "fresh")
        first = [_msg("user", "fresh q1", 10.0), _msg("assistant", "fresh a1", 11.0)]
        e.should_compress_preflight(deepcopy(first[:1]))
        e.on_turn_complete(deepcopy(first), interrupted=True)
        s1 = env.rows()
        if timing in ("gap", "draining", "second-gap") and env.is_branch:
            env.unload(cur_ctx)
            _, cur_ctx = env.register(manager=old._manager)
        second = first + [_msg("user", "fresh q2", 12.0), _msg("assistant", "fresh a2", 13.0)]
        cur_ctx.hooks["post_llm_call"](
            session_id="fresh", platform="cli", conversation_history=deepcopy(second), context_compressor=e,
        )
        e.on_session_end("fresh", deepcopy(second))
        s2 = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("fresh", deepcopy(second)))
        return s1 + s2

    _run_differential(diff_trees, monkeypatch, f"cli-new-{timing}", scenario)


@pytest.mark.parametrize("shape", ["retry", "orphan-prune", "unchanged"])
@pytest.mark.parametrize("op", ["preflight", "compress"])
def test_diff_repaired_view_across_unload(diff_trees, monkeypatch, shape, op):
    def scenario(env: _Env):
        proto, old = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("rep", platform="cli")
        warm = [_msg("user", "warm q", 1.0), _msg("assistant", "warm a", 2.0)]
        stopped = [
            _msg("user", "stopped req", 3.0),
            _msg("assistant", "", 4.0, tool_calls=[{
                "id": "call-1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
            }]),
        ]
        if shape in ("retry", "unchanged"):
            stopped += [_msg("tool", "ok", 5.0, tool_call_id="call-1"), _msg("assistant", "Interrupted.", 6.0)]
        hist = warm + stopped
        e.ingest(deepcopy(warm))
        e.on_turn_complete(deepcopy(hist), interrupted=True)
        env.unload(old)

        nxt = _msg("user", "stopped req" if shape == "retry" else "next req", 7.0)
        ans = _msg("assistant", "next ans", 8.0)
        if shape == "orphan-prune":
            merged = deepcopy(stopped[0])
            merged["content"] += "\n\n" + nxt["content"]
            merged["_merged_turn_prefix"] = stopped[0]["content"] + "\n\n"
            cur = warm + [merged]
        elif shape == "retry":
            cur = warm + [nxt]
        else:
            cur = hist + [nxt]
        if op == "preflight":
            e.should_compress_preflight(deepcopy(cur))
        else:
            e.compress(deepcopy(cur), force=True)
        s1 = env.rows()
        cur += [ans]
        e.on_turn_complete(deepcopy(cur), interrupted=True)
        s2 = env.rows()
        e.on_session_end("rep", deepcopy(cur))
        s3 = env.rows()
        _assert_handles_closed_and_no_recreate(env, lambda: e.on_session_end("rep", deepcopy(cur)))
        return s1 + s2 + s3

    _run_differential(diff_trees, monkeypatch, f"repaired-{shape}-{op}", scenario)


def test_diff_two_clones_and_rollover_finalized_lanes(diff_trees, monkeypatch):
    def scenario(env: _Env):
        proto, old = env.register()
        c0, c1, finisher = proto.clone_for_agent(), proto.clone_for_agent(), proto.clone_for_agent()
        c0.on_session_start("lane-0", platform="cli", conversation_id="conv-0")
        c1.on_session_start("lane-1", platform="cli", conversation_id="conv-1")
        finisher.on_session_start("lane-1-cur", platform="cli", conversation_id="conv-1")
        v0 = [_msg("user", "c0 q0", 1.0), _msg("assistant", "c0 a0", 2.0)]
        v1 = [_msg("user", "c1 q0", 3.0), _msg("assistant", "c1 a0", 4.0)]
        c0.ingest(deepcopy(v0))
        c1.ingest(deepcopy(v1))
        c1.on_session_end("lane-1", deepcopy(v1))
        finisher.ingest([_msg("user", "cur q0", 5.0)])
        env.unload(old)

        cur_ctx = old
        if env.is_branch:
            _, cur_ctx = env.register(manager=old._manager)
        v0_next = v0 + [_msg("user", "c0 q1", 6.0), _msg("assistant", "c0 a1", 7.0)]
        v1_stale = v1 + [_msg("assistant", "stale reply", 8.0)]
        cur_ctx.hooks["post_llm_call"](
            session_id="lane-0", conversation_id="conv-0", platform="cli", conversation_history=deepcopy(v0_next),
        )
        cur_ctx.hooks["post_llm_call"](
            session_id="lane-1", conversation_id="conv-1", platform="cli", conversation_history=deepcopy(v1_stale),
        )
        assert cur_ctx._manager._context_engine._lifecycle.get_by_conversation("conv-1").current_session_id == "lane-1-cur"
        out = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: c0.on_session_end("lane-0", deepcopy(v0_next)))
        return out

    _run_differential(diff_trees, monkeypatch, "two-clones-rollover", scenario)


def test_diff_retained_compression_and_tool_call_across_reload(diff_trees, monkeypatch):
    def scenario(env: _Env):
        tokens_mod = importlib.import_module(env.pkg.__name__ + ".tokens")
        esc_mod = importlib.import_module(env.pkg.__name__ + ".escalation")
        env.monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "2")
        env.monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "16")
        env.monkeypatch.setattr(tokens_mod, "_get_encoder", lambda: None)
        env.monkeypatch.setattr(esc_mod, "_call_llm_for_summary", lambda *a, **k: "Earlier turns retained.")

        proto, old = env.register()
        e = proto.clone_for_agent()
        e.on_session_start("comp-s", platform="cli", context_length=200_000)
        hist = [
            _msg(role, f"handoff {label} {word} " * 40, float(idx + 1))
            for idx, (label, (role, word)) in enumerate(
                (lbl, pair) for lbl in ("one", "two") for pair in (("user", "question"), ("assistant", "answer"))
            )
        ]
        e.ingest(deepcopy(hist[:2]))
        env.unload(old)
        cur_ctx = old
        if env.is_branch:
            _, cur_ctx = env.register(manager=old._manager)
        e.should_compress_preflight(deepcopy(hist))
        s1 = env.rows()
        out_msgs = e.compress(deepcopy(hist), current_tokens=e.threshold_tokens + 1, force=True)
        assert len(out_msgs) < len(hist)
        tool_resp = e.handle_tool_call(
            "lcm_grep", {"query": "handoff", "mode": "full_text", "session_scope": "current"}, messages=out_msgs,
        )
        assert '"results"' in tool_resp
        s2 = env.rows()
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: e.should_compress_preflight(deepcopy(hist)))
        return s1 + s2

    _run_differential(diff_trees, monkeypatch, "retained-compress-and-tools", scenario)


def test_diff_foreign_home_profile_isolation(diff_trees, monkeypatch):
    def scenario(env: _Env):
        proto, old = env.register()
        local, foreign = proto.clone_for_agent(), proto.clone_for_agent()
        local.on_session_start("local-sid", platform="cli", conversation_id="local-lane")
        foreign.on_session_start(
            "foreign-sid", platform="cli", conversation_id="foreign-lane", hermes_home=str(env.foreign_home),
        )
        lv = [_msg("user", "local q0", 1.0), _msg("assistant", "local a0", 2.0)]
        fv = [_msg("user", "foreign q0", 3.0), _msg("assistant", "foreign a0", 4.0)]
        local.ingest(deepcopy(lv))
        foreign.ingest(deepcopy(fv))
        env.unload(old)
        cur_ctx = old
        if env.is_branch:
            _, cur_ctx = env.register(manager=old._manager)
        lv2 = lv + [_msg("user", "local q1", 5.0), _msg("assistant", "local a1", 6.0)]
        fv2 = fv + [_msg("user", "foreign q1", 7.0), _msg("assistant", "foreign a1", 8.0)]
        cur_ctx.hooks["post_llm_call"](
            session_id="local-sid", conversation_id="local-lane", platform="cli", conversation_history=deepcopy(lv2),
        )
        foreign.on_session_end("foreign-sid", deepcopy(fv2))
        local_rows = env.rows(env.lcm_db)
        foreign_rows = env.rows(env.foreign_home / "lcm.db")
        assert [(r[2], r[3]) for r in foreign_rows] == [(m["role"], m["content"]) for m in fv2]
        env.unload(cur_ctx)
        _assert_handles_closed_and_no_recreate(env, lambda: foreign.on_session_end("foreign-sid", deepcopy(fv2)))
        return [("L",) + r for r in local_rows] + [("F",) + r for r in foreign_rows]

    _run_differential(diff_trees, monkeypatch, "foreign-home-isolation", scenario)
