#!/usr/bin/env python3
"""Count which replay-identity memo class each message falls in (#2856; for a later stored-row count).

Reads a JSON file holding a list of messages and classifies each with reconcile.py's
``_replay_identity_memo_gate``. Prints ONLY counts: total, pure, per impure kind, rows with more than one kind,
and live/stored (``--stored``: the messages are stored rows). Never prints content, ids or names.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
if "hermes_lcm" not in sys.modules:
    package = types.ModuleType("hermes_lcm")
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["hermes_lcm"] = package

from hermes_lcm.message_content import normalize_content_value  # noqa: E402
from hermes_lcm.reconcile import (  # noqa: E402
    _REPLAY_IDENTITY_MEMO_GATE, _freeze_replay_identity_value, _replay_identity_memo_gate,
)


def classify(messages, stored: bool = False) -> dict:
    counts = Counter(total=0, pure=0, multiple_kinds=0, unkeyable=0)
    counts.update({f"kind_{kind}": 0 for kind, _ in _REPLAY_IDENTITY_MEMO_GATE})
    for message in messages:
        counts["total"] += 1
        try:
            frozen = _freeze_replay_identity_value(message.get("tool_calls"))
        except (TypeError, RecursionError, AttributeError):
            counts["unkeyable"] += 1  # no memo key: never remembered
            continue
        kinds = _replay_identity_memo_gate(str(message.get("role") or "unknown"),
                                           normalize_content_value(message.get("content")) or "", frozen)
        counts["pure"] += not kinds
        counts["multiple_kinds"] += len(kinds) > 1
        counts.update(f"kind_{kind}" for kind in kinds)
    counts["stored" if stored else "live"] = counts["total"]
    return dict(counts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("messages", type=Path, help="JSON file holding a list of messages")
    parser.add_argument("--stored", action="store_true", help="the messages are stored rows")
    args = parser.parse_args(argv)
    messages = json.loads(args.messages.read_text(encoding="utf-8"))
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        print("expected a JSON list of message objects", file=sys.stderr)
        return 2
    for key, value in sorted(classify(messages, args.stored).items()):
        print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
