"""Public-clean, read-only evidence records for a release-gauntlet cell."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3


_STAMP, _LEVEL = r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[.,]\d+)?", r"(?P<level>WARNING|ERROR|INFO|DEBUG|CRITICAL)"
# The ACP stderr format, then the host's file format and its verbose console format (hermes_logging.py
# _LOG_FORMAT and _LOG_FORMAT_VERBOSE; both may carry a " [session]" tag after the level).
LOG_FORMATS = (re.compile(rf"^{_STAMP} \[{_LEVEL}\] (?P<logger>[\w.]+): "),
               re.compile(rf"^{_STAMP} {_LEVEL}(?: \[[^\]]*\])? (?P<logger>[\w.]+): "),
               re.compile(rf"^(?:\d{{4}}-\d\d-\d\d[ T])?\d\d:\d\d:\d\d(?:[.,]\d+)? - (?P<logger>[\w.]+) - "
                          rf"{_LEVEL}(?: \[[^\]]*\])? - "))
# A traceback's final line: any dotted exception name ending in Error, Exception, ExceptionGroup, Exit or
# Interrupt, including the bare names (Exception, SystemExit, KeyboardInterrupt).
FINAL = re.compile(r"^((?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*?)?(?:Error|Exception|ExceptionGroup|Exit|Interrupt))(?::|\s*$)")
MISSING = re.compile(r"No module named '([\w.]+)'")
REFUSAL = "NotImplementedError: LCM-X cannot detach a context engine"
THREAD = re.compile(r"thread=bg-review(?::([^\s]+))?")
SECTIONS = set("model providers custom_providers compression lcm context plugins mcp_servers auth telemetry "
               "terminal auxiliary platforms gateway cron memory display agent tools toolsets".split())


KEY_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")  # Fixed doctor key names; anything else is hashed.


def _fingerprint(entry):
    """A doctor detail entry's hash after paths and numbers are masked, so one cause hashes alike across cells."""
    text = entry if isinstance(entry, str) else json.dumps(entry, sort_keys=True, default=str)
    return _sha(re.sub(r"\d+", "#", re.sub(r"(?:/[^\s'\"]+)+", "<path>", text)))[:12]


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def _db(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")  # Keep all reads of this store in one snapshot.
        yield conn
    finally:
        conn.close()


def _log_event(line):
    return next((match for match in (form.match(line) for form in LOG_FORMATS) if match), None)


def trace_events(log):
    runs, outside, imports, other, events = [], Counter(), Counter(), Counter(), Counter()
    overlapping = ambiguous = unclosed = sequence = 0
    per_event = defaultdict(Counter)  # (logger, module) -> {event ordinal: tracebacks}
    for path in log:
        active, logger, traceback, warning, counted, current = [], None, False, None, False, None
        with Path(path).open(encoding="utf-8") as stream:
            for raw in stream:
                line = raw.rstrip("\r\n")
                stripped = line.lstrip(" \t|")
                event = _log_event(line)
                if event:
                    logger, sequence = event["logger"], sequence + 1
                    current = sequence
                    warning, counted = logger if event["level"] in {"WARNING", "ERROR"} else None, False
                thread = THREAD.search(line)
                if thread and "OpenAI client created" in line:
                    overlapping += bool(active)
                    runs.append({"tid": thread[1], "refusals": 0, "detach_warnings": 0})
                    active.append(len(runs) - 1)
                eligible = [i for i in active if not thread or not thread[1] or runs[i]["tid"] == thread[1]]
                for name, seen in (("refusals", stripped.startswith(REFUSAL)),
                                   ("detach_warnings", "background-review compressor detachment failed" in line)):
                    if seen:
                        if eligible:
                            runs[eligible[-1]][name] += 1
                        else:
                            outside[name] += 1
                if "Background review complete:" in line:
                    ambiguous += bool(len(active) > 1 and (not thread or not thread[1]))
                    if eligible:
                        active.remove(eligible[0])  # A tidless completion closes the oldest open run.
                header = stripped.endswith("Traceback (most recent call last):")
                final = FINAL.match(stripped) if traceback else None
                if final:
                    if stripped.startswith(("ModuleNotFoundError", "ImportError")):
                        missing = MISSING.search(stripped)
                        imports[(logger, missing[1] if missing else None)] += 1
                        per_event[(logger, missing[1] if missing else None)][current] += 1
                        if warning is not None and not counted:
                            events[warning] += 1
                            counted = True
                    elif not stripped.startswith(REFUSAL):
                        other[final[1]] += 1
                if final or (line and not line[0].isspace() and line[0] != "|"):
                    traceback = False  # Includes non-Error terminators such as KeyboardInterrupt.
                traceback = traceback or header
        # An open run at end of file may have absorbed events from outside any run.
        unclosed += sum(1 for i in active if runs[i]["refusals"] or runs[i]["detach_warnings"])
    determined = not (overlapping or ambiguous or unclosed)
    holds = all(r["refusals"] == r["detach_warnings"] <= 1 for r in runs) and not any(outside.values())
    return {"bg_review_runs": len(runs), "refusals_total": sum(r["refusals"] for r in runs) + outside["refusals"],
            "detach_warnings_total": sum(r["detach_warnings"] for r in runs) + outside["detach_warnings"],
            "runs_by_refusal_count": dict(sorted(Counter(str(r["refusals"]) for r in runs).items())),
            "per_run": [{k: v for k, v in r.items() if k != "tid"} for r in runs],
            "refusals_outside_runs": outside["refusals"], "detach_warnings_outside_runs": outside["detach_warnings"],
            "overlapping_runs": overlapping, "ambiguous_completions": ambiguous,
            "unclosed_runs_with_events": unclosed, "rule_determined": determined,
            "per_event_rule_holds": holds if determined else None,
            "optional_import_traces_total": sum(imports.values()), "events_by_logger": dict(sorted(events.items())),
            "by_event": [{"logger": key[0], "missing_module": key[1], "count": count,
                          "per_event_counts": sorted(per_event[key].values())}
                         for key, count in sorted(imports.items(), key=lambda item: tuple(x or "" for x in item[0]))],
            "unattributed": sum(n for (logger, _), n in imports.items() if logger is None),
            "other_tracebacks": dict(sorted(other.items()))}


def lifecycle_rows(lcm_db):
    with _db(lcm_db) as conn:
        counts = [Counter({str(row[0]): row[1] for row in conn.execute(
            f"SELECT session_id, COUNT(*) FROM {table} GROUP BY session_id") if row[0]})
            for table in ("messages", "summary_nodes")]
        stored = set(counts[0]) | set(counts[1])
        rows = conn.execute("SELECT conversation_id, current_session_id, last_finalized_session_id "
                            "FROM lcm_lifecycle_state ORDER BY conversation_id").fetchall()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        conversations = Counter()  # Normalized as db_bootstrap.owned_conversation_values does: strip, NULL -> ''.
        if "conversation_id" in columns:
            for value, count in conn.execute("SELECT conversation_id, COUNT(*) FROM messages GROUP BY conversation_id"):
                conversations[str(value or "").strip()] += count

        def owned(conversation_id):  # A conversation owns its own id and the blank id.
            key = str(conversation_id or "").strip()
            return conversations[key] + (conversations[""] if key else 0)
        empty = []
        for row in rows:
            refs = {str(value) for value in (row[1], row[2]) if value}
            if not refs or refs.isdisjoint(stored):
                empty.append({"conversation_id_h12": _sha(str(row[0]))[:12] if row[0] else None,
                              "current_session_h12": _sha(str(row[1]))[:12] if row[1] else None,
                              "last_finalized_session_h12": _sha(str(row[2]))[:12] if row[2] else None,
                              "messages_in_referenced_sessions": sum(counts[0][sid] for sid in refs),
                              "nodes_in_referenced_sessions": sum(counts[1][sid] for sid in refs),
                              "affected_stored_rows": owned(row[0])})
        affected = sum(row["affected_stored_rows"] for row in empty)
        safe = all(not any(row[key] for key in ("affected_stored_rows", "messages_in_referenced_sessions",
                                               "nodes_in_referenced_sessions")) for row in empty)
        safe = safe and "conversation_id" in columns  # Without the column, affected rows cannot be counted.
        return {"lifecycle_rows": len(rows), "empty_lifecycle_rows": len(empty), "empty_rows": empty,
                "conversation_column": "conversation_id" in columns, "affected_stored_rows": affected,
                "blank_conversation_rows": conversations[""],
                "identity_order_check": "PASS" if safe else "NOT_EVALUATED"}


def _identities(path, order):
    sessions = defaultdict(list)
    with _db(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        fields = [name for name in ("session_id", "role", "content", "tool_calls", "tool_call_id") if name in columns]
        for row in conn.execute(f"SELECT {', '.join(fields)} FROM messages ORDER BY {order}"):
            row = dict(row)
            calls = json.loads(row.get("tool_calls") or "[]") if row["role"] == "assistant" else []
            calls = [(c.get("id"), c.get("function", c).get("name"), c.get("function", c).get("arguments"))
                     for c in calls]
            identity = (row["role"], _sha(row.get("content") or ""), calls,
                        row.get("tool_call_id") if row["role"] == "tool" else None)
            sessions[row["session_id"]].append(_sha(json.dumps(identity, sort_keys=True)))
    return sessions, fields


def reorder(lcm_db, state_db):
    (lcm, lc), (host, hc) = _identities(lcm_db, "store_id"), _identities(state_db, "id")
    results = []
    only = {side: sorted(_sha(str(sid))[:12] for sid in keys) for side, keys in (
        ("lcm", lcm.keys() - host.keys()), ("host", host.keys() - lcm.keys()))}
    for sid in sorted(lcm.keys() | host.keys()):
        a, b = lcm[sid], host[sid]
        shared = sum((Counter(a) & Counter(b)).values())
        where = defaultdict(list)
        for j, key in enumerate(b):
            where[key].append(j)
        tails = []  # Hunt-Szymanski: the LCS is the longest increasing run of matched host positions.
        for key in a:
            for pos in reversed(where.get(key, ())):
                i = bisect_left(tails, pos)
                if i == len(tails):
                    tails.append(pos)
                else:
                    tails[i] = pos
        results.append({"session_h12": _sha(str(sid))[:12], "lcm_rows": len(a), "host_rows": len(b),
                        "matched": shared, "lcm_only": len(a) - shared, "host_only": len(b) - shared,
                        "reordered_rows": shared - len(tails), "columns_compared": {"lcm": lc, "host": hc}})
    missing = sorted({"tool_calls", "tool_call_id"} - set(hc))
    return {"sessions": len(results), "per_session": results, "identity_degraded": bool(missing),
            "missing_columns": missing, **{key: sum(row[key] for row in results) for key in (
                "lcm_rows", "host_rows", "matched", "lcm_only", "host_only", "reordered_rows")},
            **{side + "_only_sessions": len(ids) for side, ids in only.items()},
            **{side + "_only_sessions_h12": ids for side, ids in only.items()}}


def _flatten(value, prefix=""):
    if isinstance(value, (dict, list)) and value:
        items = [("." + (str(k) if not prefix and k in SECTIONS else "h:" + _sha(str(k))[:8]), v)
                 for k, v in value.items()] if isinstance(value, dict) else [(f"[{i}]", v) for i, v in enumerate(value)]
        return {key: leaf for segment, child in items for key, leaf in _flatten(child, prefix + segment).items()}
    return {prefix: value}


def config_diff(a, root_a, b, root_b):
    import yaml  # only config-diff needs a YAML reader; the release kit environment provides PyYAML

    try:
        configs = [_flatten(yaml.safe_load(Path(path).read_text(encoding="utf-8"))) for path in (a, b)]
    except (yaml.YAMLError, RecursionError):
        raise ValueError("unreadable config") from None  # never echo config text
    missing, paths, others = object(), [], []
    for key in sorted(configs[0].keys() | configs[1].keys()):
        va, vb = (config.get(key, missing) for config in configs)
        if type(va) is type(vb) and va == vb:
            continue
        rooted = all(isinstance(value, str) and root and (value == root or value.startswith(root + "/"))
                     for value, root in ((va, root_a), (vb, root_b)))
        path_only = rooted and va[len(root_a):] == vb[len(root_b):]
        (paths if path_only else others).append(key)
    return {"differing_keys": sorted(paths + others), "path_only_keys": paths, "other_keys": others,
            "verdict": "OTHER" if others else "PATHS_ONLY"}


def doctor_match(candidate, previous):
    checks = []
    for path in (candidate, previous):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value == []:
            raise ValueError("empty doctor input")
        checks.append([{**row, "status": row["status"].lower()} for item in (value if isinstance(value, list) else [value])
                       for row in item["checks"]])

    def signature(row):
        detail = row.get("detail")
        if isinstance(detail, dict):
            shape = ("dict", tuple(sorted(detail)))
        elif isinstance(detail, (list, str)):  # Each entry's normalized text, as a hash: a new cause stays unmatched.
            shape = (type(detail).__name__, tuple(sorted(_fingerprint(entry) for entry in (
                detail if isinstance(detail, list) else [detail]))))
        else:
            shape = (type(detail).__name__, ())
        conditions = ()
        if row["check"] == "lifecycle_fragmentation" and isinstance(detail, dict):
            conditions = tuple((key, detail[key] > 0 if key == "empty_lifecycle_rows" else bool(detail[key]))
                               for key in ("empty_lifecycle_rows", "state_db_error") if key in detail)
        return row["check"], row["status"], shape, conditions

    warnings = [Counter(signature(row) for row in rows if row["status"] == "warn") for rows in checks]
    results = [{"check": sig[0], "candidate": count, "previous": warnings[1][sig], "matched": count <= warnings[1][sig]}
               for sig, count in sorted(warnings[0].items(), key=lambda item: repr(item[0]))]
    unmatched = sorted({row["check"] for row in results if not row["matched"]})
    failed = sorted({row["check"] for rows in checks for row in rows if row["status"] == "fail"})
    statuses = {"pass", "warn", "fail", "ok", "skip"}
    unknown = sorted({row["check"] for rows in checks for row in rows if row["status"] not in statuses})
    signatures = [{"check": sig[0], "status": sig[1] if sig[1] in statuses else "unknown_status",
                   "detail_keys": [key if KEY_NAME.fullmatch(str(key)) else "h:" + _sha(str(key))[:12]
                                   for key in sig[2][1]] if sig[2][0] == "dict" else [],
                   **({"detail_fingerprints": list(sig[2][1])} if sig[2][0] in ("list", "str") else {}),
                   "detail_type": sig[2][0], **dict(sig[3])}
                  for sig in sorted({signature(row) for row in checks[0]}, key=repr)]
    return {"warnings": results, "unmatched": unmatched, "failed_checks": failed, "unknown_status": unknown,
            "candidate_signatures": signatures,
            "verdict": "UNMATCHED" if unmatched or unknown or any(row["status"] == "fail" for row in checks[0]) else "MATCHED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, flags in {"trace-events": ("log",), "lifecycle-rows": ("lcm-db",),
                        "reorder": ("lcm-db", "state-db"),
                        "config-diff": ("a", "root-a", "b", "root-b"),
                        "doctor-match": ("candidate", "previous")}.items():
        command = commands.add_parser(name)
        for flag in flags:
            command.add_argument("--" + flag, required=True, **({"action": "append"} if flag == "log" else {}))
        command.add_argument("--out", required=True, type=Path)
    args = vars(parser.parse_args())
    command, out = args.pop("command"), args.pop("out")
    try:
        result = globals()[command.replace("-", "_")](**args)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        out.write_text(encoded, encoding="utf-8")
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, IndexError, AttributeError, ImportError):
        parser.exit(2, "Evidence input/output error\n")  # Never echo paths or input text.
    print(encoded, end="")


if __name__ == "__main__":
    main()
