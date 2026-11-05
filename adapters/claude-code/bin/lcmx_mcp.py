"""Start LCM-X's read-only MCP server (scripts/lcm_x_mcp.py, #1216) over the Claude Code store.

Stdlib only and Python 3.9 compatible: Claude Code starts it with the system ``python3``,
and it replaces itself with the LCM-X runtime interpreter.

A valid ``reader-override.json`` in the store home names a command to start instead, which gets
the plain reader's argv in ``LCMX_LOCAL_READER_CMD``. Anything else about that file, or a command
that does not start, starts the plain reader. The plugin never writes the file.

Only a file this user owns can choose the command, and only a command this user or root owns can
start; neither may be writable by group or others, and no folder on the way to the command may be
writable by another local user, so another local user cannot run code as this one.
"""
from __future__ import annotations

import grp
import json
import os
import pwd
import re
import stat
import subprocess
import sys
from functools import lru_cache
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


def other_members(gid: int, owners: Tuple[int, ...]) -> List[str]:
    """The users of group ``gid`` not in ``owners``, as the user database lists them now: its listed
    members and every user whose primary group it is. Groups nested in a directory service are not
    expanded. Raises KeyError for a group the database does not list."""
    others = set()
    for name in grp.getgrgid(gid).gr_mem:
        try:
            uid = pwd.getpwnam(name).pw_uid
        except KeyError:
            uid = None
        if uid not in owners:
            others.add(name)
    others.update(user.pw_name for user in pwd.getpwall() if user.pw_gid == gid and user.pw_uid not in owners)
    return sorted(others)


def group_name(gid: int) -> str:
    """The group's name, else its number."""
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


LOCAL_SOURCES = {"files", "systemd", "altfiles"}
NSSWITCH = Path("/etc/nsswitch.conf")
SEARCH_PATH = ("/usr/bin/dscl", "/Search", "-read", "/", "SearchPath")


def nsswitch_is_local(text: str) -> bool:
    """Whether nsswitch.conf takes users and groups only from local sources: every `passwd` and
    `group` line, and any `initgroups` line, which gives a user their other groups at login."""
    sources = {}
    for line in text.splitlines():
        database, _, rest = line.split("#", 1)[0].partition(":")
        if database.strip() in ("passwd", "group", "initgroups"):
            words = re.sub(r"\[[^]]*\]", " ", rest).split()  # [NOTFOUND=return] is an action
            sources.setdefault(database.strip(), []).append(words)
    return {"passwd", "group"} <= set(sources) and all(words and set(words) <= LOCAL_SOURCES
                                                         for lines in sources.values() for words in lines)


def search_path_is_local(output: str) -> bool:
    """Whether the directory search path macOS uses, as `dscl /Search -read / SearchPath` prints it,
    has only local nodes. The default policy's path can include an LDAP server that DHCP names."""
    key, _, rest = output.partition(":")
    nodes = rest.split()  # a node with spaces in its name splits too, and its first part is not local
    return key.strip() == "SearchPath" and bool(nodes) and all(node.startswith(("/Local/", "/BSD/"))
                                                               for node in nodes)


def libc_name() -> str:
    """glibc's name and version, such as "glibc 2.39"; "" for another C library, such as musl."""
    try:
        return os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (OSError, ValueError):  # ValueError: a C library that does not know the name
        return ""


def macos_database_is_local(command=SEARCH_PATH, timeout: float = 2.0) -> bool:
    try:
        found = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout,
                               check=True)
    except (OSError, subprocess.SubprocessError):  # a timeout too
        return False
    return search_path_is_local(found.stdout)


def linux_database_is_local(nsswitch: Path = NSSWITCH) -> bool:
    if not libc_name().startswith("glibc "):  # musl does not read nsswitch.conf
        return False
    try:
        return nsswitch_is_local(nsswitch.read_text(encoding="utf-8", errors="replace"))
    except OSError:  # without the file, glibc's defaults are not provably local
        return False


@lru_cache(maxsize=None)
def user_database_is_local() -> bool:
    """Whether every user and group comes from a local source, so listing them finds every member of
    a group. A directory service may not list its users (SSSD by default), and then nothing can
    prove that a group has no other member. Worked out once per launch, and never stored."""
    if sys.platform == "darwin":
        return macos_database_is_local()
    if sys.platform.startswith("linux"):
        return linux_database_is_local()
    return False


def folder_untrusted(folder: os.stat_result, entry: os.stat_result, owners: Tuple[int, ...]) -> str:
    """Why another local user could replace ``entry`` in ``folder`` ("" when only ``owners`` can)."""
    if folder.st_uid not in owners:
        return "not owned by this user or root"
    if not folder.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return ""
    if folder.st_mode & stat.S_ISVTX:  # sticky: only the entry's owner, the folder's owner or root can replace it
        return "" if entry.st_uid in owners else "shared, and the entry in it belongs to another user"
    if folder.st_mode & stat.S_IWOTH:
        return "writable by others"
    group = group_name(folder.st_gid)
    if not user_database_is_local():
        return "writable by its group %s (user database not provably local)" % group
    try:
        others = other_members(folder.st_gid, owners)
    except KeyError:
        return "writable by its group %s, which the user database does not list" % group
    return "writable by its group %s, whose members include %s" % (group, ", ".join(others)) if others else ""


def path_untrusted(path: str, owners: Tuple[int, ...]) -> str:
    """Why another local user could change what the absolute ``path`` names ("" when none can).

    Resolves it as the kernel does and checks every folder an entry is looked up in: ``/``, each
    folder on the way, and the folder of each symlink on it. When all pass, nobody else can redirect
    the path between this check and the start, so the path is started as given: a venv's Python
    finds its venv from the path it was started by."""
    pending = path.split("/")
    folder, links = "/", 0
    while pending:
        name = pending.pop(0)
        if name in ("", "."):
            continue
        entry = os.path.dirname(folder) if name == ".." else os.path.join(folder, name)  # `..` is looked up here too
        try:
            info = os.lstat(entry)
            reason = folder_untrusted(os.stat(folder), info, owners)
            target = os.readlink(entry) if not reason and stat.S_ISLNK(info.st_mode) else None
        except OSError as error:
            return "not resolvable (%s)" % (error.strerror or error)
        if reason:
            return "under %s, which is %s" % (folder, reason)
        if target is None:
            folder = entry
            continue
        links += 1
        if links > 40:
            return "not resolvable (too many symlinks)"
        if target.startswith("/"):
            folder = "/"
        pending[:0] = target.split("/")
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
    reason = reason or path_untrusted(command[0], (os.geteuid(), 0))
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
    if runtime is None:
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
