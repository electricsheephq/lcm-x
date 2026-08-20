#!/usr/bin/env python3
"""Sequential ingest/query driver for the promoted scale389 instrument."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.tools.pinverify import verify as verify_pins  # noqa: E402
from bench.tools.storefreeze import (  # noqa: E402
    verify_manifest,
    write_manifest,
)


SCRIPT = Path(__file__).with_name("phase1a.py")
DEFAULT_RESULTS = Path(
    "/Volumes/LEXAR/Codex/session-notes/2026-07-29/"
    "hermes-r3-1/artifacts/laneINSTPREP-logs/scale389-results"
)


def _verify_store(args: argparse.Namespace, phase: str) -> None:
    if bool(args.store_dir) != bool(args.store_manifest):
        raise SystemExit("--store-dir and --store-manifest must be provided together")
    if not args.store_dir:
        return
    result = verify_manifest(args.store_dir, args.store_manifest)
    report = args.results / f"STORE-{phase.upper()}.json"
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["ok"]:
        raise SystemExit(f"store verification failed: {report}")


def _verify_pins(args: argparse.Namespace, phase: str) -> None:
    if not args.pins:
        return
    output = args.results / f"PINS-{phase.upper().replace('-', '')}.txt"
    passed, _, _ = verify_pins(args.pins, phase, output)
    if not passed:
        raise SystemExit(f"pin verification failed: {output}")


def _run(args: argparse.Namespace, command: list[str]) -> None:
    environment = dict(os.environ)
    environment["SCALE389_RESULTS"] = str(args.results)
    subprocess.run(
        [sys.executable, str(SCRIPT), *command],
        check=True,
        env=environment,
    )


def run_ingest(args: argparse.Namespace) -> None:
    for scale in args.scales:
        _run(args, ["ingest-scale", str(scale)])
    _run(args, ["ingest-s0", "--uncensored-n", str(args.uncensored_n)])
    _run(args, ["materialise-files", "--uncensored-n", str(args.uncensored_n)])


def run_queries(args: argparse.Namespace) -> None:
    for arm in args.arms:
        for scale in args.scales:
            if arm in {"A2u", "A3u"} and scale == "S0":
                continue
            _run(
                args,
                [
                    "query",
                    arm,
                    scale,
                    "--uncensored-n",
                    str(args.uncensored_n),
                ],
            )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--store-manifest", type=Path)
    parser.add_argument(
        "--freeze-results-manifest",
        type=Path,
        help="write a bench.tools.storefreeze manifest after a successful chain",
    )
    parser.add_argument(
        "--uncensored-n",
        type=int,
        default=50,
        help="u-arm population (default: full 50-question primary set)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    add_common(ingest)
    ingest.add_argument(
        "--scales", type=int, nargs="+", default=[500, 2000, 8000, 19829]
    )
    ingest.set_defaults(handler=run_ingest)

    queries = subparsers.add_parser("queries")
    add_common(queries)
    queries.add_argument(
        "--arms",
        nargs="+",
        choices=("A1", "A2", "A3", "A2u", "A3u", "B"),
        default=["B", "A3", "A2", "A1", "A3u", "A2u"],
    )
    queries.add_argument(
        "--scales",
        nargs="+",
        choices=("S0", "500", "2000", "8000", "19829"),
        default=["S0", "500", "2000", "8000", "19829"],
    )
    queries.set_defaults(handler=run_queries)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    _verify_pins(args, "pre-run")
    _verify_store(args, "pre-run")
    args.handler(args)
    _verify_store(args, "post-run")
    _verify_pins(args, "post-run")
    if args.freeze_results_manifest:
        write_manifest(args.results, args.freeze_results_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
