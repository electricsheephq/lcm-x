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

Usage: python -m bench.tools.d5_end_read --config DECLARATION --config-sha256 HEX --instrument-dir KIT --scratch DIR
         --start DIR --start-pin FILE --start-pin-sha256 HEX --end DIR --end-pin FILE --end-pin-sha256 HEX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import message_content
import message_patterns
from bench.instruments.d5_clone import bars, capture, run as runner, snapshot
from session_patterns import build_session_match_keys, compile_session_patterns, matches_session_pattern

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
VERDICT_RULE = ("FAIL when check 1 fails or any window host row is LOST; LOST only for a window host row whose session "
                "holds no stored row at all in the end store, matches no declared pattern and has no ancestor that does; "
                "a LOST row, and an unmatched row with no hint, is an unconfirmed loss candidate until a replay confirms it "
                "(the spec's canary rule); INCONCLUSIVE while any window host row is unmatched, when the window is empty "
                "or both pins are the same, when a pin, the declaration or the kit does not verify, when the declaration "
                "has no settings.lcm session patterns, when message patterns are declared and the regex package that bounds "
                "their matching is missing, or when any session in the end store that matches a declared "
                "pattern holds a stored row; a window row in a session matching a declared ignore or stateless pattern is "
                "counted apart as not stored by design, never LOST or matched; a window row matching a declared message "
                "pattern is unmatched, never LOST; PASS only with check 1 PASS, a non-empty "
                "window of other rows and 0 unmatched")
HINTS = ("equal after stripping edge whitespace", "payload swapped for a reference",
         "session never stored; a parent session matches a declared pattern",
         "matches a declared message pattern", "no hint")
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
    return HINTS[4], None


def session_matchers(lcm_settings, platforms, parents=None):
    """(by_design, inherits): by_design(session) is True when the session itself matches a declared ignore or stateless
    pattern (LCM-X's own matcher, with the session's platform from the end capture's host db as a key, as the engine
    does); LCM-X never stores it. inherits(session) is True when an ancestor through the host db's parent links matches.
    The engine passes bypass on only at a compression handover, while branch and delegate children are stored normally,
    so inherits() never makes a row "by design": it only turns a would-be LOST row into an unmatched one (INCONCLUSIVE)."""
    compiled = compile_session_patterns([*lcm_settings["ignore_session_patterns"],
                                         *lcm_settings["stateless_session_patterns"]])
    parents = parents or {}

    def by_design(session):
        return bool(compiled) and matches_session_pattern(
            build_session_match_keys(session, platform=platforms.get(session) or ""), compiled)

    def inherits(session):
        seen, session = {session}, parents.get(session)
        while session and session not in seen:  # The seen set ends a cycle; a finite chain is walked to its root.
            if by_design(session):
                return True
            seen.add(session)
            session = parents.get(session)
        return False
    return by_design, inherits


def coverage(start, end, identity, end_dir, by_design=lambda session: False, inherits=lambda session: False,
             message_excluded=lambda row: False):
    """Check 2 over the runner's row dicts ({"store": [...], "host": [...]}) of both captures."""
    top_host = max((h["id"] for h in start["host"]), default=0)
    top_store = max((r["store_id"] for r in start["store"]), default=0)
    new = [r for r in end["store"] if r["store_id"] > top_store]
    stored_sessions = {r["session_id"] for r in end["store"]}
    in_window = [h for h in end["host"] if h["id"] > top_host]
    designed = {h["session_id"] for h in in_window if by_design(h["session_id"])}
    window = [h for h in in_window if h["session_id"] not in designed]
    disagree = sorted(_sha12(s) for s in stored_sessions if by_design(s))  # Anywhere in the end store.
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
            if message_excluded(host):  # The engine may have dropped it by a declared message pattern: unproven.
                hints[HINTS[3]] += 1
                s["unmatched"] += 1
                unmatched.append({"host_id": host["id"], "session_sha256_12": _sha12(session), "class": HINTS[3],
                                  "candidate_store_id": None})
                continue
            if session not in stored_sessions and inherits(session):  # Maybe a bypass handover: never LOST, unproven.
                hints[HINTS[2]] += 1
                s["unmatched"] += 1
                unmatched.append({"host_id": host["id"], "session_sha256_12": _sha12(session), "class": HINTS[2],
                                  "candidate_store_id": None})
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
    return {"not_stored_by_design": {"sessions": len(designed), "rows": len(in_window) - len(window),
                                     "session_sha256_12": sorted(_sha12(s) for s in designed)},
            "declared_pattern_sessions_with_stored_rows": disagree,
            "window_host_rows": counts["window_host_rows"], "new_stored_rows": counts["new_stored_rows"],
            "matched": counts["matched"], "lost": counts["lost"], "unmatched": counts["unmatched"],
            "stored_rows_not_matched": counts["stored_rows_not_matched"],
            "unmatched_by_hint": {h: hints[h] for h in HINTS}, "unmatched_rows": unmatched,
            "per_session": {k: dict(v) for k, v in sorted(per_session.items())},
            "start_top": {"host_id": top_host, "store_id": top_store}}


def verdict(check1, check2):
    if check1["verdict"] == FAIL or check2["lost"]:
        return FAIL
    if check1["verdict"] != PASS or check2["unmatched"] or not check2["window_host_rows"] \
            or check2.get("declared_pattern_sessions_with_stored_rows"):
        return INCONCLUSIVE  # An empty window proves nothing about coverage.
    return PASS


def _may_match(content, patterns):
    """True when a declared message pattern matches the row's text as the runtime reads it, or when a search cannot be
    decided within the runtime's time bound: the runtime may have matched it, so the row is never LOST."""
    texts = [message_content.text_content_for_pattern_matching(content)]
    if isinstance(content, str) and content.lstrip()[:1] in ("[", "{"):
        try:  # A host row holds structured content as JSON text; the runtime matched the live parts' text.
            texts.append(message_content.text_content_for_pattern_matching(json.loads(content)))
        except Exception:  # noqa: BLE001  # Not JSON: the raw text above is all there is.
            pass
    for text in dict.fromkeys(t for t in texts if isinstance(t, str) and t):
        for pattern in patterns:
            try:
                if pattern.search(text, timeout=message_patterns.MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS):
                    return True
            except Exception:  # noqa: BLE001  # A timeout, or a search that cannot be bounded: undecided.
                return True
    return False


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


def _sessions(folder):
    """({session id: platform}, {session id: parent session id}) from the end capture's host db (read-only); a missing
    column gives an empty map."""
    db = next((Path(folder) / n for n in ("host-0.db", "state.db") if (Path(folder) / n).is_file()), None)
    if db is None:
        return {}, {}
    con = capture._readonly(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
        platforms = dict(con.execute("SELECT id, source FROM sessions")) if "source" in cols else {}
        parents = dict(con.execute("SELECT id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL")) \
            if "parent_session_id" in cols else {}
        return platforms, parents
    finally:
        con.close()


def end_read(args):
    out = {"schema": "d5-end-read/1", "verdict_rule": VERDICT_RULE, "credits": "none (release manager, 2026-11-10)",
           "pins_sha256": {k: v for k, v in (("start", args.start_pin_sha256), ("end", args.end_pin_sha256))
                           if isinstance(v, str) and _HEX64.fullmatch(v)}}  # Never an unvalidated argument.
    if args.start_pin_sha256 == args.end_pin_sha256:
        return {**out, "verdict": INCONCLUSIVE, "reason": "the start and end pins are the same capture"}
    for side in ("start", "end"):
        _, reason = _capture(getattr(args, f"{side}_pin"), getattr(args, side), getattr(args, f"{side}_pin_sha256"),
                             args.scratch)
        if reason:
            return {**out, "verdict": INCONCLUSIVE, "reason": f"{side}: {reason}"}
    raw = Path(args.config).read_bytes() if _HEX64.fullmatch(args.config_sha256 or "") else b""
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:  # The bytes hashed are the bytes parsed.
        return {**out, "verdict": INCONCLUSIVE, "reason": "the declaration does not hash to the sha256 sent"}
    out["config_sha256"] = args.config_sha256
    config = {**json.loads(raw.decode("utf-8")), "instrument_dir": args.instrument_dir}
    lcm = (config.get("settings") or {}).get("lcm") if isinstance(config.get("settings"), dict) else None
    if not isinstance(lcm, dict) or not all(isinstance(lcm.get(k), list) for k in
                                            ("ignore_session_patterns", "stateless_session_patterns")):
        return {**out, "verdict": INCONCLUSIVE,
                "reason": "the declaration has no settings.lcm ignore and stateless session patterns"}
    host = runner.LocalHost(config, args.scratch)
    if host.pins().get("verdict") != PASS:
        return {**out, "verdict": INCONCLUSIVE, "reason": "P1: the kit does not verify against its manifest"}
    identity = host.rules()["identity"]
    try:
        start, end = host.rows(args.start), host.rows(args.end)
    except Exception as exc:  # noqa: BLE001  # Never a message: it can carry a path.
        return {**out, "verdict": INCONCLUSIVE, "reason": f"a store could not be read ({type(exc).__name__})"}
    check1 = bars.bar3(start["store"], end["store"], identity)
    out = {**out, "check1_preservation": check1, "start_rows": {k: len(v) for k, v in start.items()},
           "end_rows": {k: len(v) for k, v in end.items()}}
    unproven = FAIL if check1["verdict"] == FAIL else INCONCLUSIVE  # A failed check 1 stands whatever follows.
    log = logging.getLogger(message_patterns.__name__)
    was_disabled, log.disabled = log.disabled, True  # Its warnings quote a pattern: a settings value.
    try:
        declared = lcm.get("ignore_message_patterns") or []
        if not isinstance(declared, list) or not all(isinstance(x, str) for x in declared):
            return {**out, "verdict": unproven,
                    "reason": "the declaration's ignore_message_patterns is not a list of strings"}
        if declared and message_patterns._regex_engine is None:
            return {**out, "verdict": unproven, "reason": "declared message patterns need the regex package for "
                                                          "bounded matching; it is not installed"}
        patterns = message_patterns.compile_message_patterns(declared)  # An invalid pattern is skipped, as at runtime.
        check2 = coverage(start, end, identity, args.end, *session_matchers(lcm, *_sessions(args.end)),
                          message_excluded=lambda row: _may_match(row.get("content"), patterns))
    except Exception as exc:  # noqa: BLE001  # Never a message: it can carry a path.
        return {**out, "verdict": unproven, "reason": f"check 2 could not be read ({type(exc).__name__})"}
    finally:
        log.disabled = was_disabled
    return {**out, "verdict": verdict(check1, check2), "check2_coverage": check2}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the run declaration (JSON): manifest, rules_module, interpreter")
    parser.add_argument("--config-sha256", required=True, help="the declaration's sha256 as the release manager sent it")
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
