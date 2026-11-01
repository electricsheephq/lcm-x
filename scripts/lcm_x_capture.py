#!/usr/bin/env python3
"""LCM-X capture for Claude Code: ``worker`` drains the plugin's spool into its store.

The plugin's hook entry (adapters/claude-code/bin/lcmx_hook.py) starts it detached, or, on
PreCompact, waits for it with ``--deadline``. Exit codes: 0 every record drained, 3 records
left (deadline or a held lock), 2 the store was refused, 1 an error. Design:
docs/claude-code-adapter.md.
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
    if not report["complete"]:
        return 3 if args.deadline is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
