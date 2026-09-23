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


def _run(tmp_path, *, plugin_dir: str, enabled: list[str], engine: str) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / plugin_dir).symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [" + ", ".join(enabled) + "]\n"
        f"context:\n  engine: {engine}\n",
        encoding="utf-8",
    )
    python, src = HERMES
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "HERMES_HOME": str(home),
        "PROBE_ENGINE": engine,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [python, "-c", _PROBE],
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
