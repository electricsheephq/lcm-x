#!/usr/bin/env python3
"""Explicit local portable recall CLI; hooks always fail open with exit zero."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_portable():
    package_root = Path(__file__).resolve().parents[1]
    name = "hermes_lcm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, package_root / "__init__.py", submodule_search_locations=[str(package_root)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    from hermes_lcm import portable
    return portable


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", required=True, help="Dedicated absolute portable storage root; never a Hermes home.")
    result.add_argument("--project", required=True)
    result.add_argument("--instance", required=True)
    result.add_argument("--host", required=True, choices=("claude", "codex", "manual"))
    commands = result.add_subparsers(dest="mode", required=True)
    commands.add_parser("serve", help="Four read-only MCP tools over stdio, legacy initialize protocol family.")
    ingest = commands.add_parser("ingest", help="Explicit operator input JSON on stdin; no model ingestion tool.")
    ingest.add_argument("--session", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--session", required=True)
    capture.add_argument("--transcript-root", required=True)
    capture.add_argument("--transcript", required=True)
    capsule = commands.add_parser("capsule")
    capsule.add_argument("--session", required=True)
    hook = commands.add_parser("hook", help="SessionStart/PreCompact/PostCompact hook JSON on stdin; exit 0 on failure.")
    hook.add_argument("--transcript-root", required=True)
    hook.add_argument("--no-assistance", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    hook_mode = args.mode == "hook"
    try:
        module = load_portable()
        portable = module.PortableRecall(args.root, project=args.project, instance=args.instance, host=args.host)
        if args.mode == "serve":
            module.serve(portable, sys.stdin, sys.stdout)
            return 0
        if args.mode in {"ingest", "hook"}:
            raw = sys.stdin.read(module.MAX_WIRE + 1)
            if len(raw.encode("utf-8")) > module.MAX_WIRE:
                raise module.PortableError("input_limit")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise module.PortableError("input_object_required")
        if hook_mode:
            output, receipt = portable.hook(payload, transcript_root=args.transcript_root,
                                            assistance=not args.no_assistance)
            print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
            if output:
                print(json.dumps(output, ensure_ascii=False))
            return 0
        if args.mode == "ingest":
            result = portable.ingest(args.session, event_id=payload.get("event_id"),
                                     messages=payload.get("messages"), generation=payload.get("generation", "manual-v1"),
                                     envelope=payload.get("envelope"))
        elif args.mode == "capture":
            result = portable.capture(args.session, transcript=args.transcript, transcript_root=args.transcript_root)
        else:
            result = portable.capsule(args.session)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        # Exceptions may contain transcript/path material: expose type only.
        print(json.dumps({"status": "failed_open" if hook_mode else "error",
                          "error_type": type(exc).__name__}), file=sys.stderr)
        return 0 if hook_mode else 1


if __name__ == "__main__":
    raise SystemExit(main())
