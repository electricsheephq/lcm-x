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
