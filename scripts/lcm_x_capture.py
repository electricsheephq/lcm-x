#!/usr/bin/env python3
"""LCM-X capture for Claude Code: ``worker`` drains the plugin's spool into its store.

The plugin's hook entry (adapters/claude-code/bin/lcmx_hook.py) starts it detached, or, on
PreCompact, waits for it with ``--deadline``. Exit codes: 0 every record drained, 5 a bounded
run reached its deadline, 3 records left for another reason (a failed record, the store left in
WAL mode, or another worker took the lock), 4 a bounded run found the lock held and drained
nothing, 2 the store was refused, 1 an error. A worker that finds the lock held at its start exits
before it loads any store code. Design: docs/claude-code-adapter.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hermes_lcm"


def _ensure_local_package_importable() -> None:
    """Make local plugin modules importable when this file is run directly."""
    if PACKAGE_NAME in sys.modules:
        return
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(PLUGIN_DIR)]
    pkg.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = pkg


def _lock_held(home: Path) -> bool:
    """Whether another worker holds the store lock. Checked before any store code loads, so a
    worker started while another runs exits at once: the async PostToolBatch hook starts one
    on every tool batch without probing the lock itself."""
    import fcntl

    try:
        with open(home / "capture.lock", "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass  # no store home yet: the worker creates it
    return False


def _run_audit(home: Path, session: str | None) -> int:
    """Print the read-only audit report without loading the worker's reader settings."""
    _ensure_local_package_importable()
    from hermes_lcm.claude_code_audit import AuditError, audit

    try:
        report = audit(home, session=session)
    except AuditError as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker", help="drain the spool into the store")
    worker.add_argument("--home", required=True, type=Path, help="the store home (absolute)")
    worker.add_argument("--deadline", type=float, help="seconds to wait and drain (PreCompact)")
    audit = commands.add_parser("audit", help="check capture without changing the store")
    audit.add_argument("--home", required=True, type=Path, help="the store home (absolute)")
    audit.add_argument("--session", help="audit this session and its subagents")
    args = parser.parse_args(argv)
    if args.command == "audit":
        return _run_audit(args.home, args.session)
    home = args.home.expanduser()
    if not home.is_absolute():
        return 2
    if _lock_held(home):  # the holder sees every spooled record
        return 4 if args.deadline is not None else 0
    # The Hermes home is read before the store's own settings replace HERMES_HOME: the
    # writer refuses any store home inside it.
    hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    _ensure_local_package_importable()
    from hermes_lcm.claude_code_store import run_worker
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.mcp_reader import apply_profile_environment

    # Same settings as the reader (<home>/.env), so both compute the same embedding identity.
    apply_profile_environment(home)
    report = run_worker(home, deadline_s=args.deadline, config=LCMConfig.from_env(), hermes_home=hermes_home)
    if report.get("refused"):
        return 2
    if report.get("lock") == "held":
        return 4  # a bounded drain found another worker holding the store; it takes the record
    if not report["complete"]:
        if args.deadline is None:
            return 1
        return 5 if report.get("deadline") else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
