"""End read between two captures of one agent (the rc canary): were start rows preserved, were the window's host rows stored.

Inputs are two capture folders, each with its runner pin and the pin's sha256 as the release manager sent it after a
move that recorded exit 0. Each folder is verified against its pin (the runner's snapshot.verify) and opened read-only.
Identity is the frozen kit's, loaded through the runner after P1 verifies the kit. No credit is granted (release manager,
2026-11-10): a credit needs the replay evidence an end read of one agent does not have, so a row that would need one is
reported as unmatched with a hint class, never as lost.

- Check 1, preservation: bar 3 (bars.bar3) with the start store rows as the snapshot and the end store rows as the
  final rows.
- Check 2, coverage: the end capture's host rows above the start capture's highest host id (the window) against the
  end store's rows above the start capture's highest store_id, matched one to one and in order per session by
  identity alone (bars._ordered).

Output: one JSON object with counts, ids, sha256s and the verdict rule; never row content, a path or a settings value.

Usage: python -m bench.tools.d5_end_read --config DECLARATION --instrument-dir KIT --scratch DIR
         --start DIR --start-pin FILE --start-pin-sha256 HEX --end DIR --end-pin FILE --end-pin-sha256 HEX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from bench.instruments.d5_clone import bars, capture, run as runner, snapshot

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
VERDICT_RULE = ("FAIL when check 1 fails or any window host row is LOST; LOST only for a window host row whose session "
                "holds no stored row at all in the end store; INCONCLUSIVE while any window host row is unmatched, when "
                "the window is empty or both pins are the same, or when a pin or the kit does not verify; PASS only with "
                "check 1 PASS, a non-empty window and 0 unmatched")
HINTS = ("equal after stripping edge whitespace", "payload swapped for a reference", "no hint")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha12(value):
    return hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _hint(host, candidates, end_dir):
    """The hint class for one unmatched host row against its session's unpaired stored rows (never a credit)."""
    text = host.get("content")
    for row in candidates:
        if row.get("role") != host.get("role"):
            continue
        stored = row.get("content")
        if isinstance(text, str) and isinstance(stored, str) and text != stored and text.strip() == stored.strip():
            return HINTS[0], row["store_id"]
        if (isinstance(text, str) and runner._swap_payload(end_dir, stored) == text) or \
                (isinstance(stored, str) and runner._swap_payload(end_dir, text) == stored):  # Either side the reference.
            return HINTS[1], row["store_id"]
    return HINTS[2], None


def coverage(start, end, identity, end_dir):
    """Check 2 over the runner's row dicts ({"store": [...], "host": [...]}) of both captures."""
    top_host = max((h["id"] for h in start["host"]), default=0)
    top_store = max((r["store_id"] for r in start["store"]), default=0)
    window = [h for h in end["host"] if h["id"] > top_host]
    new = [r for r in end["store"] if r["store_id"] > top_store]
    stored_sessions = {r["session_id"] for r in end["store"]}
    counts, hints, per_session, unmatched = Counter(), Counter(), {}, []
    for session in dict.fromkeys(h["session_id"] for h in window):
        hosts = [h for h in window if h["session_id"] == session]
        rows = [r for r in new if r["session_id"] == session]
        pairs = bars._ordered(rows, hosts, lambda r, h: identity(r) == identity(h))
        used_rows, used_hosts = {i for i, _ in pairs}, {j for _, j in pairs}
        spare = [r for i, r in enumerate(rows) if i not in used_rows]
        s = per_session.setdefault(_sha12(session), Counter())
        s["window_host_rows"] += len(hosts)
        s["matched"] += len(pairs)
        for j, host in enumerate(hosts):
            if j in used_hosts:
                continue
            if session not in stored_sessions:
                counts["lost"] += 1
                s["lost"] += 1
                unmatched.append({"host_id": host["id"], "session_sha256_12": _sha12(session), "class": "LOST"})
                continue
            hint, store_id = _hint(host, spare, end_dir)
            hints[hint] += 1
            s["unmatched"] += 1
            unmatched.append({"host_id": host["id"], "session_sha256_12": _sha12(session), "class": hint,
                              "candidate_store_id": store_id})
        counts["matched"] += len(pairs)
        counts["unmatched"] += len(hosts) - len(pairs) - s["lost"]
    counts["window_host_rows"], counts["new_stored_rows"] = len(window), len(new)
    counts["stored_rows_not_matched"] = len(new) - counts["matched"]
    return {"window_host_rows": counts["window_host_rows"], "new_stored_rows": counts["new_stored_rows"],
            "matched": counts["matched"], "lost": counts["lost"], "unmatched": counts["unmatched"],
            "stored_rows_not_matched": counts["stored_rows_not_matched"],
            "unmatched_by_hint": {h: hints[h] for h in HINTS}, "unmatched_rows": unmatched,
            "per_session": {k: dict(v) for k, v in sorted(per_session.items())},
            "start_top": {"host_id": top_host, "store_id": top_store}}


def verdict(check1, check2):
    if check1["verdict"] == FAIL or check2["lost"]:
        return FAIL
    if check1["verdict"] != PASS or check2["unmatched"] or not check2["window_host_rows"]:
        return INCONCLUSIVE  # An empty window proves nothing about coverage.
    return PASS


def _capture(pin, folder, pin_sha256, scratch):
    """(None, reason) unless the pin file hashes to the sha256 sent and the folder verifies against it."""
    if not _HEX64.fullmatch(pin_sha256 or ""):
        return None, "the pin sha256 is not 64 lowercase hex"
    if capture._sha256(pin) != pin_sha256:
        return None, "the pin file does not hash to the sha256 sent"
    try:
        result = snapshot.verify(pin, folder, scratch)
    except (OSError, ValueError, KeyError, TypeError):
        return None, "the capture could not be verified against its pin"
    if result.get("verdict") != PASS:
        return None, "the capture does not verify against its pin"
    return result, None


def end_read(args):
    out = {"schema": "d5-end-read/1", "verdict_rule": VERDICT_RULE, "credits": "none (release manager, 2026-11-10)",
           "pins_sha256": {"start": args.start_pin_sha256, "end": args.end_pin_sha256}}
    if args.start_pin_sha256 == args.end_pin_sha256:
        return {**out, "verdict": INCONCLUSIVE, "reason": "the start and end pins are the same capture"}
    for side in ("start", "end"):
        _, reason = _capture(getattr(args, f"{side}_pin"), getattr(args, side), getattr(args, f"{side}_pin_sha256"),
                             args.scratch)
        if reason:
            return {**out, "verdict": INCONCLUSIVE, "reason": f"{side}: {reason}"}
    config = {**json.loads(Path(args.config).read_text(encoding="utf-8")), "instrument_dir": args.instrument_dir}
    host = runner.LocalHost(config, args.scratch)
    if host.pins().get("verdict") != PASS:
        return {**out, "verdict": INCONCLUSIVE, "reason": "P1: the kit does not verify against its manifest"}
    identity = host.rules()["identity"]
    try:
        start, end = host.rows(args.start), host.rows(args.end)
    except Exception as exc:  # noqa: BLE001  # Never a message: it can carry a path.
        return {**out, "verdict": INCONCLUSIVE, "reason": f"a store could not be read ({type(exc).__name__})"}
    check1 = bars.bar3(start["store"], end["store"], identity)
    check2 = coverage(start, end, identity, args.end)
    return {**out, "verdict": verdict(check1, check2), "check1_preservation": check1, "check2_coverage": check2,
            "start_rows": {k: len(v) for k, v in start.items()}, "end_rows": {k: len(v) for k, v in end.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the run declaration (JSON): manifest, rules_module, interpreter")
    parser.add_argument("--instrument-dir", required=True, help="the frozen kit folder P1 verifies")
    parser.add_argument("--scratch", required=True, help="a folder outside both captures (snapshot.verify's scratch)")
    for side in ("start", "end"):
        parser.add_argument(f"--{side}", required=True, help=f"the {side} capture's snapshot folder")
        parser.add_argument(f"--{side}-pin", required=True, help=f"the {side} capture's runner pin")
        parser.add_argument(f"--{side}-pin-sha256", required=True, help="the pin's sha256 as the release manager sent it")
    args = parser.parse_args(argv)
    try:
        result = end_read(args)
    except Exception as exc:  # noqa: BLE001  # A fixed line, never a traceback: one can show a path typed here.
        result = {"schema": "d5-end-read/1", "verdict_rule": VERDICT_RULE, "verdict": INCONCLUSIVE,
                  "reason": f"unexpected {type(exc).__name__}"}
    print(json.dumps(result, indent=1, sort_keys=True))
    return {PASS: 0, FAIL: 1}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
