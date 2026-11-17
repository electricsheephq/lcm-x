"""Real Hermes PluginManager proof for the hermes-lcm-x / lcm-x rename (#471).

Runs a real Hermes checkout's ``PluginManager`` and ``_select_context_engine`` in
a subprocess with an isolated ``HERMES_HOME``. Skipped unless a Hermes Python is
available: set ``LCM_REAL_HERMES_PYTHON`` (and ``LCM_REAL_HERMES_SRC`` when the
source tree is not importable from that interpreter's cwd), or run pytest from
an interpreter that can import ``hermes_cli.plugins``.
"""

from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = textwrap.dedent(
    """
    import io, json, logging, os
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    manager = P.get_plugin_manager()
    loaded = {key: bool(p.enabled) for key, p in manager._plugins.items() if "lcm" in key}
    from agent.agent_init import _select_context_engine
    selected = _select_context_engine({"context": {"engine": os.environ["PROBE_ENGINE"]}})
    print(json.dumps({
        "loaded": loaded,
        "selected": getattr(selected, "name", None),
        "log": buf.getvalue(),
    }))
    """
)


def _hermes_python() -> tuple[str, str] | None:
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    if python:
        return python, src
    if importlib.util.find_spec("hermes_cli") is not None:
        try:
            if importlib.util.find_spec("hermes_cli.plugins") is not None:
                return sys.executable, src
        except (ImportError, ValueError):
            return None
    return None


HERMES = _hermes_python()
pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")


def _run(
    tmp_path, *, plugin_dir: str, enabled: list[str], engine: str,
    probe: str = _PROBE, probe_args: tuple[str, ...] = (),
) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    target = home / "plugins" / plugin_dir
    if os.name == "nt":
        target.mkdir()
        for source in (*REPO_ROOT.glob("*.py"), REPO_ROOT / "plugin.yaml"):
            shutil.copy2(source, target / source.name)
        for name in ("adapters", "access_context", "access_policy", "skills"):
            shutil.copytree(REPO_ROOT / name, target / name)
    else:
        target.symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [" + ", ".join(enabled) + "]\n"
        f"context:\n  engine: {engine}\n",
        encoding="utf-8",
    )
    python, src = HERMES
    env = {
        **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC") if key in os.environ},
        "HOME": str(tmp_path / "home"),
        "PATH": os.environ.get("PATH", "") if os.name == "nt" else "/usr/bin:/bin",
        "HERMES_HOME": str(home),
        "PROBE_ENGINE": engine,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "LCM_EMBEDDINGS_ENABLED": "0",
        "LCM_TEMPORAL_ROLLUPS_ENABLED": "0",
    }
    if os.name == "nt":
        env.update({key: str(home) for key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP")})
    completed = subprocess.run(
        [python, "-c", probe, *probe_args],
        cwd=src or None,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_new_install_selects_lcm_x_without_warning(tmp_path):
    result = _run(tmp_path, plugin_dir="hermes-lcm-x", enabled=["hermes-lcm-x"], engine="lcm-x")
    assert result["loaded"] == {"hermes-lcm-x": True}
    assert result["selected"] == "lcm-x"
    assert "LCM plugin loaded — lossless context management active" in result["log"]
    assert "DEPRECATED LCM-X config" not in result["log"]


def test_both_names_with_legacy_engine_selects_lcm_and_warns(tmp_path):
    result = _run(
        tmp_path, plugin_dir="hermes-lcm", enabled=["hermes-lcm", "hermes-lcm-x"], engine="lcm"
    )
    assert result["loaded"] == {"hermes-lcm-x": True}
    assert result["selected"] == "lcm"
    assert "LCM plugin loaded — lossless context management active" in result["log"]
    assert result["log"].count("DEPRECATED LCM-X config") == 1
    assert "set `context.engine: lcm-x`" in result["log"]


def test_in_place_rename_with_only_legacy_name_is_the_documented_failure(tmp_path):
    """BREAKING (#471): Hermes gates plugins.enabled on the manifest name before
    any plugin code runs, so the plugin cannot warn; Hermes' own line is the signal."""
    result = _run(tmp_path, plugin_dir="hermes-lcm", enabled=["hermes-lcm"], engine="lcm")
    assert result["loaded"] == {"hermes-lcm-x": False}
    assert result["selected"] is None
    assert "LCM plugin loaded" not in result["log"]
    assert "Context engine 'lcm' not found — falling back to built-in compressor" in result["log"]


_RELOAD_PROBE = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing
    from pathlib import Path
    from threading import Event, Thread
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import json, os, sqlite3, sys
    from agent import conversation_loop
    from hermes_cli import plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    timing = sys.argv[1]
    plugin = "hermes-lcm-x"
    home = Path(os.environ["HERMES_HOME"])
    database = home / "lcm.db"
    state_path = home / "state.db"
    state = SessionDB(db_path=state_path)
    session = "retained-across-import"
    first = [("user", "first question"), ("assistant", "first answer")]
    stopped = [("user", "interrupted question"), ("assistant", "partial answer")]
    expected = first + stopped

    plugins.discover_plugins(force=True)
    manager = plugins.get_plugin_manager()
    original = plugins.get_plugin_context_engine()
    engine_module = sys.modules[type(original).__module__]
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            model="test/model", quiet_mode=True, session_db=state, session_id=session,
            skip_context_files=True, skip_memory=True, platform="cli", enabled_toolsets=["todo"],
        )
    retained = agent.context_compressor
    assert type(retained) is type(original) and retained is not original
    agent.client, agent.tool_delay, agent.save_trajectories = MagicMock(), 0, False
    agent._compression_feasibility_checked = True
    retained.update_model("test/model", 64000, provider="openrouter")

    def rows(path=database, order="store_id"):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(f"SELECT role, content FROM messages ORDER BY {order}").fetchall()

    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=first[1][1], tool_calls=None), finish_reason="stop",
        )],
        model="test/model", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=6, total_tokens=106),
    )
    agent.client.chat.completions.create.side_effect = [response]
    with patch("agent.title_generator.call_llm", return_value=response):
        initial = agent.run_conversation(user_message=first[0][1], conversation_history=[])
    assert initial["final_response"] == first[1][1]
    assert rows() == first

    notify = conversation_loop._notify_context_engine_turn_complete
    notifications = []
    delivery_errors = []

    def reload_plugin():
        assert plugins.get_plugin_context_engine() is None
        plugins.discover_plugins(force=True)
        replacement = plugins.get_plugin_context_engine()
        assert replacement is not None
        assert sys.modules[type(replacement).__module__] is not engine_module
        return replacement

    def unload_plugin():
        assert manager.unload(plugin)
        assert plugins.get_plugin_context_engine() is None

    def stop_provider(*args, **kwargs):
        agent._current_streamed_assistant_text = stopped[1][1]
        raise InterruptedError("synthetic provider interruption")

    def deliver_at_boundary(owner, messages, **meta):
        assert owner is agent and owner.context_compressor is retained
        assert meta["interrupted"]
        assert rows(state_path, "id") == expected
        assert rows() == first

        if timing == "old-engine-draining":
            group = original._unload_group
            release = Event()
            worker = Thread(target=release.wait, args=(30,), daemon=True)
            assert group.start_worker(worker)
            with ThreadPoolExecutor(1) as pool:
                unloading = pool.submit(unload_plugin)
                try:
                    assert group.closing.wait(5)
                    assert not unloading.done() and not retained._stable_use_closed
                    notify(owner, messages, **meta)
                    assert rows() == expected
                    assert not retained._stable_use_closed
                finally:
                    release.set()
                unloading.result(timeout=15)
        else:
            unload_plugin()
            if timing == "between-registrations":
                notify(owner, messages, **meta)
                return

            replacement = reload_plugin()
            assert retained._reload_successor is None
            if timing == "after-takeover":
                retained.should_compress_preflight(messages[:-1])
                successor = retained._reload_successor
                assert successor is not None and successor._unload_group is replacement._unload_group
                assert rows() == expected[:-1]
                notify(owner, messages, **meta)
                return

            construct = replacement._clone_for_agent
            entered, release = Event(), Event()

            def pause_construction():
                entered.set()
                assert release.wait(15)
                return construct()

            with (
                patch.object(replacement, "_clone_for_agent", new=pause_construction),
                ThreadPoolExecutor(1) as pool,
            ):
                delivering = pool.submit(notify, owner, messages, **meta)
                try:
                    assert entered.wait(5), "the stop did not reach replacement clone construction"
                    assert rows() == first
                    if timing == "second-unload-during-clone":
                        unload_plugin()
                finally:
                    release.set()
                delivering.result(timeout=15)
            if timing == "during-clone":
                successor = retained._reload_successor
                assert successor is not None and successor._unload_group is replacement._unload_group

    def observe_delivery(owner, messages, **meta):
        # Hermes logs notification errors and keeps going. Check them outside its callback.
        try:
            deliver_at_boundary(owner, messages, **meta)
        except Exception as exc:
            delivery_errors.append(exc)
        else:
            notifications.append((messages, meta))

    with patch.object(agent, "_interruptible_api_call", new=stop_provider), patch.object(
        conversation_loop, "_notify_context_engine_turn_complete", new=observe_delivery,
    ):
        stopped_turn = agent.run_conversation(
            user_message=stopped[0][1], conversation_history=initial["messages"],
        )
    if delivery_errors:
        raise delivery_errors[0]
    assert stopped_turn["interrupted"] and len(notifications) == 1
    assert rows() == expected, rows()
    if plugins.get_plugin_context_engine() is None:
        reload_plugin()
    messages, meta = notifications[0]
    # Replay the notification and end the same session to expose duplicate writes on takeover.
    notify(agent, messages, **meta)
    retained.on_session_end(session, messages)
    assert rows() == expected, rows()

    unload_plugin()
    database.unlink()
    notify(agent, messages, **meta)
    assert not database.exists()
    print(json.dumps({"deleted_during_hermes": True}))
    """
)


@pytest.mark.parametrize("timing", [
    "old-engine-draining",
    "between-registrations",
    "during-clone",
    "second-unload-during-clone",
    "after-takeover",
])
def test_retained_interruption_survives_real_plugin_reimport(tmp_path, timing):
    result = _run(
        tmp_path, plugin_dir="hermes-lcm-x", enabled=["hermes-lcm-x"], engine="lcm-x",
        probe=_RELOAD_PROBE, probe_args=(timing,),
    )
    assert result["deleted_during_hermes"]


# --- Two physical LCM copies during migration (#471 fix round 1) -------------
# The v0.23.3 GA tree is the "separate legacy checkout" at plugins/hermes-lcm.
LEGACY_GA_SHA = "523161769329174514a0b995025201e3dc316df2"

_TURNS_PROBE = textwrap.dedent(
    """
    import io, json, logging, os, sqlite3
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    manager = P.get_plugin_manager()
    loaded = {key: bool(p.enabled) for key, p in manager._plugins.items() if "lcm" in key}
    engine = manager._context_engine
    hooks = len(manager._hooks.get("post_llm_call", []))
    from agent.agent_init import _select_context_engine
    selected = _select_context_engine({"context": {"engine": os.environ["PROBE_ENGINE"]}})
    history, rows = [], []
    db = os.path.join(os.environ["HERMES_HOME"], "lcm.db")
    for turn in range(3):
        history = history + [
            {"role": "user", "content": f"dual-copy probe turn {turn} question"},
            {"role": "assistant", "content": f"dual-copy probe turn {turn} answer"},
        ]
        P.invoke_hook(
            "post_llm_call", session_id="dual-copy-session", task_id="t", turn_id=f"turn-{turn}",
            user_message=history[-2]["content"], assistant_response=history[-1]["content"],
            conversation_history=list(history), model="probe-model", platform="cli",
        )
        conn = sqlite3.connect(db)
        rows.append(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        conn.close()
    print(json.dumps({
        "loaded": loaded,
        "registered_engine": getattr(engine, "name", None),
        "post_llm_call_hooks": hooks,
        "selected": getattr(selected, "name", None),
        "rows_after_each_turn": rows,
        "log": buf.getvalue(),
    }))
    """
)


def _legacy_checkout(dest: Path) -> bool:
    dest.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", LEGACY_GA_SHA],
        capture_output=True, check=False,
    )
    if archive.returncode != 0:
        return False
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    return (dest / "plugin.yaml").read_text(encoding="utf-8").startswith("name: hermes-lcm\n")


def _run_turns(tmp_path, *, enabled: list[str], engine: str) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    if not _legacy_checkout(home / "plugins" / "hermes-lcm"):
        pytest.skip(f"legacy GA tree {LEGACY_GA_SHA} not available in this clone")
    (home / "plugins" / "hermes-lcm-x").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [" + ", ".join(enabled) + "]\n"
        f"context:\n  engine: {engine}\n",
        encoding="utf-8",
    )
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _TURNS_PROBE],
        cwd=src or None,
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(home),
            "PROBE_ENGINE": engine,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_documented_migration_with_separate_legacy_checkout_runs_one_copy(tmp_path):
    """Documented path: the old checkout stays on disk, only hermes-lcm-x is enabled."""
    result = _run_turns(tmp_path, enabled=["hermes-lcm-x"], engine="lcm-x")
    assert result["loaded"] == {"hermes-lcm": False, "hermes-lcm-x": True}
    assert result["selected"] == "lcm-x"
    assert result["post_llm_call_hooks"] == 1
    assert result["rows_after_each_turn"] == [2, 4, 6]


@pytest.mark.parametrize("engine", ["lcm-x", "lcm"])
def test_both_copies_enabled_never_double_ingests(tmp_path, engine):
    """Both names enabled with a separate legacy checkout (the pre-fix documented
    state): the legacy copy loads first; LCM-X must not add a second writer."""
    result = _run_turns(tmp_path, enabled=["hermes-lcm", "hermes-lcm-x"], engine=engine)
    assert result["loaded"] == {"hermes-lcm": True, "hermes-lcm-x": True}
    assert result["registered_engine"] == "lcm"
    assert result["rows_after_each_turn"] == [2, 4, 6]
    assert result["post_llm_call_hooks"] == 1
    assert "Another LCM generation is already loaded" in result["log"]


def test_same_checkout_legacy_link_with_both_names_is_one_copy(tmp_path):
    """install.sh reuses plugins/hermes-lcm when it resolves to this checkout; keeping
    hermes-lcm enabled next to hermes-lcm-x is then harmless (one physical copy)."""
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-lcm, hermes-lcm-x]\ncontext:\n  engine: lcm-x\n",
        encoding="utf-8",
    )
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _TURNS_PROBE],
        cwd=src or None,
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(home),
            "PROBE_ENGINE": "lcm-x",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["loaded"] == {"hermes-lcm-x": True}
    assert result["selected"] == "lcm-x"
    assert result["post_llm_call_hooks"] == 1
    assert result["rows_after_each_turn"] == [2, 4, 6]
