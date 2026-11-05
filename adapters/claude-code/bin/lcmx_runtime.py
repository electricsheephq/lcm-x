"""Path resolution shared by the LCM-X Claude Code plugin's entry points.

Stdlib only and Python 3.9 compatible: hooks run on the system ``python3``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Every hook imports this module: typing serves the annotations only, and shutil loads only
# when a Python is looked up on PATH.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import List, Mapping, Optional

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# One store per harness, so each store has one owning writer: (setting, default directory name).
HOMES = {"claude-code": ("LCMX_CC_HOME", "claude-code"), "grok-build": ("LCMX_GROK_HOME", "grok-build")}
VERSIONED_PYTHONS = ("python3.14", "python3.13", "python3.12", "python3.11")
# Core needs only the standard library: a found interpreter runs isolated from what is installed in it.
ISOLATED = ["-I", "-S"]


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


def relative(environ: Mapping[str, str], setting: str) -> bool:
    """Whether a path setting is relative. A hook checks paths from the project folder Claude Code
    runs it in, the worker runs in the store home, and the reader in whatever folder Claude Code
    starts it in, so a relative ``LCMX_PYTHON`` or ``LCMX_PACKAGE_ROOT`` names no one file for all
    of them: it is refused."""
    configured = environ.get(setting)
    return bool(configured) and not Path(configured).expanduser().is_absolute()


def runtime_python(environ: Mapping[str, str], home: Path) -> Optional[Path]:
    """The Python 3.11+ interpreter with LCM-X's dependencies, if one is set up; None when
    ``LCMX_PYTHON`` is relative."""
    if relative(environ, "LCMX_PYTHON"):
        return None
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


def found_python(environ: Mapping[str, str]) -> Optional[Path]:
    """A Python 3.11+ on the hook's ``PATH``: ``python3`` when it is the 3.11+ interpreter running
    this hook, else a versioned name, newest first. Only that ``PATH``'s absolute entries are
    searched: a relative one would name the project folder Claude Code runs the hook in."""
    import shutil

    search = os.pathsep.join(entry for entry in environ.get("PATH", "").split(os.pathsep) if os.path.isabs(entry))
    own = shutil.which("python3", path=search)
    if own and sys.version_info >= (3, 11) and sys.executable and (
            os.path.realpath(own) == os.path.realpath(sys.executable)):
        return Path(own)
    for name in VERSIONED_PYTHONS:
        found = shutil.which(name, path=search)
        if found:
            return Path(found)
    return None


def runtime_command(environ: Mapping[str, str], home: Path) -> Optional[List[str]]:
    """How to run the worker and the reader. A set-up interpreter (``LCMX_PYTHON`` or a venv) runs
    as it is, so optional dependencies such as FastEmbed load. Otherwise a found Python 3.11+ runs
    with the standard library only (``-I -S``), so nothing installed in it, Hermes included, loads.
    A relative ``LCMX_PYTHON`` gives None: no found Python stands in for the one it names."""
    python = runtime_python(environ, home)
    if python is not None:
        return [str(python)]
    if relative(environ, "LCMX_PYTHON"):
        return None
    found = found_python(environ)
    return [str(found), *ISOLATED] if found is not None else None


def package_root(environ: Mapping[str, str]) -> Optional[Path]:
    """The LCM-X package: a vendored copy in the plugin, or the checkout holding the plugin; None
    when ``LCMX_PACKAGE_ROOT`` is relative."""
    if relative(environ, "LCMX_PACKAGE_ROOT"):
        return None
    configured = environ.get("LCMX_PACKAGE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates += [PLUGIN_ROOT / "lcm-x", PLUGIN_ROOT.parents[1]]
    for candidate in candidates:
        if (candidate / "plugin.yaml").is_file() and (candidate / "store.py").is_file():
            return candidate
    return None
