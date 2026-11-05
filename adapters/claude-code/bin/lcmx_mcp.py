"""Start LCM-X's read-only MCP server (scripts/lcm_x_mcp.py, #1216) over the Claude Code store.

Stdlib only and Python 3.9 compatible: Claude Code starts it with the system ``python3``,
and it replaces itself with the LCM-X runtime interpreter.

A valid ``reader-override.json`` in the store home names a command to start instead, which gets
the plain reader's argv in ``LCMX_LOCAL_READER_CMD``. Anything else about that file, or a command
that does not start, starts the plain reader. The plugin never writes the file.

Only a file this user owns can choose the command, and only a command this user or root owns can
start; neither may be writable by group or others, so another local user cannot run code as this one.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcmx_runtime  # noqa: E402

OVERRIDE = "reader-override.json"
READER_CMD = "LCMX_LOCAL_READER_CMD"


def untrusted(info: os.stat_result, owners: Tuple[int, ...]) -> str:
    """Why another user could have written this file ("" when only an owner in ``owners`` could)."""
    if info.st_uid not in owners:
        return "not owned by this user" if len(set(owners)) == 1 else "not owned by this user or root"
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "writable by group or others"
    return ""


def reader_override(path: Path) -> Tuple[Optional[List[str]], str]:
    """The command the override file names, else None and why it is not used ("" for no file)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)  # opening a FIFO waits for a writer otherwise
    except FileNotFoundError:
        return None, ""
    except OSError as error:
        return None, "could not be read (%s)" % (error.strerror or error)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):  # a FIFO or a device can block or never end
            return None, "not a regular file"
        reason = untrusted(info, (os.geteuid(),))
        if reason:
            return None, reason
        os.set_blocking(fd, True)
        with open(fd, "rb", closefd=False) as handle:
            text = handle.read()
    except OSError as error:
        return None, "could not be read (%s)" % (error.strerror or error)
    finally:
        os.close(fd)
    try:
        spec = json.loads(text)
    except (ValueError, RecursionError):  # bad UTF-8 is a ValueError too
        return None, "not valid JSON"
    if not isinstance(spec, dict):
        return None, "not a JSON object"
    version = spec.get("version")
    if type(version) is not int:  # true is a bool and 1.0 a float, not the integer 1
        return None, '"version" is not an integer'
    if version != 1:
        return None, '"version" is not 1'
    command = spec.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        return None, '"command" is not an array of strings'
    if not command:
        return None, '"command" is empty'
    if not os.path.isabs(command[0]):
        return None, '"command[0]" is not an absolute path'
    if not (os.path.isfile(command[0]) and os.access(command[0], os.X_OK)):
        return None, '"command[0]" is missing or not executable'
    try:
        reason = untrusted(os.stat(command[0]), (os.geteuid(), 0))  # root owns system interpreters
    except OSError:
        return None, '"command[0]" is missing or not executable'
    if reason:
        return None, '"command[0]" is ' + reason
    return command, ""


def main() -> int:
    environ = os.environ
    host = "grok-build" if environ.get("GROK_SESSION_ID") else "claude-code"
    home = lcmx_runtime.store_home(environ, host)
    if home is None:
        print("lcm-x: %s must be an absolute path" % lcmx_runtime.HOMES[host][0], file=sys.stderr)
        return 1
    runtime = lcmx_runtime.runtime_command(environ, home)
    package = lcmx_runtime.package_root(environ)
    reader = package / "scripts" / "lcm_x_mcp.py" if package else None
    relative = [name for name in ("LCMX_PYTHON", "LCMX_PACKAGE_ROOT") if lcmx_runtime.relative(environ, name)]
    if relative:
        problem = "%s must be an absolute path" % relative[0]
    elif runtime is None:
        problem = ("no Python 3.11 or newer on PATH for LCM-X: install one, or run `uv venv --python 3.11 %s`"
                   % (home / "venv"))
    elif reader is None or not reader.is_file():
        problem = "the LCM-X package with scripts/lcm_x_mcp.py was not found"
    else:
        problem = ""
    if problem:
        print("lcm-x: " + problem, file=sys.stderr)
        return 1
    logs = home / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    plain = [*runtime, str(reader), "--store", str(home / "lcm.db"), "--hermes-home", str(home),
             "--log-dir", str(logs)]
    command, reason = reader_override(home / OVERRIDE)
    if command is not None:
        try:  # once started, the override owns recall and its own fallback to the plain reader
            os.execve(command[0], command, {**environ, READER_CMD: json.dumps(plain)})
        except (OSError, ValueError) as error:  # ValueError: a NUL byte in an argument
            reason = "the command did not start (%s)" % error
    if reason:
        print("lcm-x: %s: %s; starting the plain reader" % (home / OVERRIDE, reason), file=sys.stderr, flush=True)
    os.execv(plain[0], plain)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
