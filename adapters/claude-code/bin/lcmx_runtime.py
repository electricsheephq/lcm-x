"""Path resolution shared by the LCM-X Claude Code plugin's entry points.

Stdlib only and Python 3.9 compatible: hooks run on the system ``python3``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# One store per harness, so each store has one owning writer: (setting, default directory name).
HOMES = {"claude-code": ("LCMX_CC_HOME", "claude-code"), "grok-build": ("LCMX_GROK_HOME", "grok-build")}


def store_home(environ: Mapping[str, str], host: str = "claude-code") -> Optional[Path]:
    """The store home: ``lcm.db``, externalized payloads, spool, lock, logs and runtime.

    None when the setting is a relative path, which would resolve against whichever
    project directory a hook happens to run in."""
    setting, directory = HOMES[host]
    configured = environ.get(setting)
    if not configured:
        return Path.home() / ".lcm-x" / directory
    path = Path(configured).expanduser()
    return path if path.is_absolute() else None


def claude_projects_root(environ: Mapping[str, str]) -> Optional[Path]:
    """Claude Code keeps session transcripts under ``<config dir>/projects``; None if relative."""
    configured = environ.get("CLAUDE_CONFIG_DIR")
    base = Path(configured).expanduser() if configured else Path.home() / ".claude"
    return base / "projects" if base.is_absolute() else None


def runtime_python(environ: Mapping[str, str], home: Path) -> Optional[Path]:
    """The Python 3.11+ interpreter with LCM-X's dependencies, if one is set up."""
    configured = environ.get("LCMX_PYTHON")
    candidates = [Path(configured).expanduser()] if configured else []
    plugin_data = environ.get("CLAUDE_PLUGIN_DATA")
    if plugin_data:  # survives plugin updates; removed with the plugin, unlike the store
        candidates.append(Path(plugin_data).expanduser() / "venv" / "bin" / "python")
    candidates.append(home / "venv" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def package_root(environ: Mapping[str, str]) -> Optional[Path]:
    """The LCM-X package: a vendored copy in the plugin, or the checkout holding the plugin."""
    configured = environ.get("LCMX_PACKAGE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates += [PLUGIN_ROOT / "lcm-x", PLUGIN_ROOT.parents[1]]
    for candidate in candidates:
        if (candidate / "plugin.yaml").is_file() and (candidate / "store.py").is_file():
            return candidate
    return None
