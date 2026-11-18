#!/usr/bin/env python3
"""Print the recorded summary share of each Track S reader prompt as JSON lines."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+", help="replay directories containing summary.json")
    args = parser.parse_args(argv)
    status = 0
    for folder in args.folders:
        try:
            record = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            summary_tokens, prompt_tokens = record["summary_tokens"], record["prompt_tokens"]
            if type(summary_tokens) is not int or summary_tokens < 0 or type(prompt_tokens) is not int or prompt_tokens <= 0:
                raise ValueError("invalid summary_tokens or prompt_tokens")
            result = {
                "arm": record["arm"]["name"], "seed": record["seed"], "run": record["run"],
                "checkpoint": record.get("checkpoint_id", record.get("checkpoint")),
                "summary_tokens": summary_tokens, "prompt_tokens": prompt_tokens,
                "share": summary_tokens / prompt_tokens,
            }
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"{folder}: {exc}", file=sys.stderr)
            status = 2
            continue
        print(json.dumps(result))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
