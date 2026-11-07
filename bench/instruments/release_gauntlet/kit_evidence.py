"""Public-clean, read-only evidence records for a release-gauntlet cell."""
import argparse
import builtins
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
# A traceback's final line: a dotted or bare exception name, then ":" or the end of the line, at the traceback's edge.
# The name need not end in Error (ProviderCircuitOpen, KeyboardInterrupt).
FINAL = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)(?::|\s*$)")
# Only a builtin exception's bare name is emitted as is; any other name (a custom class, a dotted name or a word that
# merely looks like one after a cut traceback) is emitted as a hash, which still matches across cells.
BUILTIN_EXCEPTIONS = frozenset(n for n, v in vars(builtins).items() if isinstance(v, type) and issubclass(v, BaseException))
SUB_EDGE = re.compile(r"[ \t]*\| \S")  # An exception group's final line: one space after "|"; source lines have more.
MISSING = re.compile(r"No module named '([\w.]+)'")
REFUSAL = "NotImplementedError: LCM-X cannot detach a context engine"
THREAD = re.compile(r"thread=bg-review(?::([^\s]+))?")
SECTIONS = set("model providers custom_providers compression lcm context plugins mcp_servers auth telemetry "
               "terminal auxiliary platforms gateway cron memory display agent tools toolsets".split())


def _key_name(key):
    """Every doctor detail key is hashed: a key can be data (a session or table name), and a hash still matches."""
    return "h:" + _sha(str(key))[:12]


def _cause(value):
    """A detail value reduced to its cause: a bool or None as such, a number as whether it is non-zero (NaN counts),
    a string as its masked fingerprint, and a dict or list recursively. Same cause, same output; values never leak."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return "f:" + _fingerprint(value)
    if isinstance(value, dict):
        return {_key_name(key): _cause(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_cause(item) for item in value]
    return "t:" + type(value).__name__


def _fingerprint(entry):
    """A doctor detail entry's hash after absolute paths and long numbers are masked, so one cause hashes alike
    across cells. Short numbers (status codes) and URL hosts stay significant."""
    text = entry if isinstance(entry, str) else json.dumps(entry, sort_keys=True, default=str)
    text = re.sub(r"(?<![\w:/.])/[^\s'\"]+", "<path>", text)  # Absolute paths only, never a URL's path.
    return _sha(re.sub(r"\d{4,}", "#", text))[:12]


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
                # A final line starts at column 0, or one space after an exception group's "|" prefix; an indented
                # bare name is a source line inside the traceback, not its end.
                piped = line.lstrip(" \t").startswith("|")
                at_edge = bool(SUB_EDGE.match(line)) if piped else bool(line) and not line[0].isspace()
                final = FINAL.match(stripped) if traceback and at_edge else None
                if final:
                    if stripped.startswith(("ModuleNotFoundError", "ImportError")):
                        missing = MISSING.search(stripped)
                        # Any other import failure keeps a masked fingerprint, so two different failures never match.
                        module = missing[1] if missing else "f:" + _fingerprint(stripped)
                        imports[(logger, module)] += 1
                        per_event[(logger, module)][current] += 1
                        if warning is not None and not counted:
                            events[warning] += 1
                            counted = True
                    elif not stripped.startswith(REFUSAL):
                        name = final[1]  # Any other column-0 word is hashed: a cut traceback can precede raw text.
                        other[name if name in BUILTIN_EXCEPTIONS else "h:" + _sha(name)[:12]] += 1
                if final or (line and not line[0].isspace() and line[0] != "|"):
                    if traceback and not final:
                        other["(no final line)"] += 1  # Cut short: counted, never dropped.
                    traceback = False
                traceback = traceback or header
            if traceback:
                other["(no final line)"] += 1
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


def _host_rewrites(conn):
    """LCM-X's recorded host rewrites, {store_id: host form's content}: an override is valid only while its recorded
    hash of the stored content still matches the row (reconcile.py's rule). An externalized payload is not resolved."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
        return {}
    overrides = {}
    for key, value in conn.execute("SELECT key, value FROM metadata WHERE key LIKE 'host_rewrite_identity:%'"):
        if not str(key).startswith("host_rewrite_identity:"):  # LIKE ignores case and reads "_" as a wildcard.
            continue
        try:
            payload = json.loads(str(value)) if value else None
            store_id = int(key.split(":", 1)[1])
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("content"), str) and not payload.get("strip_payload"):
            overrides[store_id] = payload
    return overrides


def _flag(value, missing):
    """A 0/1 column read as a bool, whatever its storage class (0, 0.0, "0", "0.0" and "false" are false); NULL gives
    *missing*."""
    if value is None:
        return missing
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    if not isinstance(value, str):
        return value != 0
    text = value.strip().lower()
    try:
        return float(text) != 0
    except ValueError:
        return text not in ("false", "")


def _identities(path, order, host=False):
    """Per session, each row's identity hash in order. On a host with transcript generations (an ``active`` or
    ``compacted`` column), a compaction writes rows back with carried stamps. The kit does not resolve those copies (a
    kit limit): a row repeating an earlier row's identity and ``timestamp`` in its session is kept and counted in
    ``repeats``, and that session's counts are not definitive. Without a ``timestamp`` column repeats cannot be seen:
    ``resolved`` is False. A host can also re-issue a row with a fresh stamp, so an inactive row, or a compacted row
    whose identity repeats in its session, is counted in ``generation_rows`` and its session is not definitive too.
    On LCM-X's side, a user row with a valid recorded host rewrite takes the host form's identity (``credited``)."""
    sessions, repeats, generation_rows, credited = defaultdict(list), Counter(), Counter(), Counter()
    with _db(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        fields = [name for name in ("session_id", "role", "content", "tool_calls", "tool_call_id") if name in columns]
        generations = host and bool({"active", "compacted"} & columns)
        stamped = generations and "timestamp" in columns
        flags = sorted({"active", "compacted"} & columns) if generations else []
        overrides = {} if host else _host_rewrites(conn)
        seen, compacted = set(), defaultdict(list)
        select = fields + ["timestamp"] * stamped + flags + (["store_id"] if overrides else [])
        for row in conn.execute(f"SELECT {', '.join(select)} FROM messages ORDER BY {order}"):
            row = dict(row)
            calls = json.loads(row.get("tool_calls") or "[]") if row["role"] == "assistant" else []
            calls = [(c.get("id"), c.get("function", c).get("name"), c.get("function", c).get("arguments"))
                     for c in calls]
            raw = row.get("content")
            content = raw or ""
            # Only stored text with something besides edge whitespace: NULL, a number or a blob is never credited.
            override = overrides.get(row.get("store_id")) if (
                row["role"] == "user" and isinstance(raw, str) and raw.strip()) else None
            if (override and override.get("stored_sha256") == hashlib.sha256(content.encode()).hexdigest()
                    and override["content"] == content.strip()):  # An edge-trim rewrite only, as the host stores it.
                content = override["content"]  # Never stripped without a valid recorded override.
                credited[row["session_id"]] += 1
            identity = (row["role"], _sha(content), calls,
                        row.get("tool_call_id") if row["role"] == "tool" else None)
            key = _sha(json.dumps(identity, sort_keys=True))
            if stamped and row["timestamp"] is not None:
                slot = (row["session_id"], key, float(row["timestamp"]))
                repeats[row["session_id"]] += slot in seen
                seen.add(slot)
            # Each row counts once: an inactive row now, an active compacted row once its session is read.
            inactive = "active" in row and not _flag(row["active"], True)  # NULL is active, as the host reads it.
            generation_rows[row["session_id"]] += inactive
            compacted[row["session_id"]].append((key, _flag(row.get("compacted"), False) and not inactive))
            sessions[row["session_id"]].append(key)
    for sid, marks in compacted.items():
        counts = Counter(key for key, _flag in marks)
        generation_rows[sid] += sum(1 for key, flag in marks if flag and counts[key] > 1)
    return sessions, fields, repeats, stamped or not generations, generation_rows, credited


def reorder(lcm_db, state_db):
    (lcm, lc, _, _, _, credited), (host, hc, repeats, resolved, generation_rows, _) = (
        _identities(lcm_db, "store_id"), _identities(state_db, "id", True))
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
                        "reordered_rows": shared - len(tails), "host_stamp_repeat_rows": repeats[sid],
                        "host_generation_rows": generation_rows[sid], "host_rewrites_credited": credited[sid],
                        "counts_definitive": resolved and not repeats[sid] and not generation_rows[sid],
                        # Rows sharing an identity are matched by count and order only: a lost occurrence that a
                        # duplicate of another occurrence replaces is not seen. This counts the rows where it could be.
                        "repeated_identity_rows": sum(n for n in Counter(b).values() if n > 1),
                        "columns_compared": {"lcm": lc, "host": hc}})
    missing = sorted({"tool_calls", "tool_call_id"} - set(hc))
    return {"sessions": len(results), "per_session": results, "identity_degraded": bool(missing),
            "host_copies_resolved": resolved and not any(repeats.values()) and not any(generation_rows.values()),
            "sessions_not_definitive": sum(1 for row in results if not row["counts_definitive"]),
            "missing_columns": missing, **{key: sum(row[key] for row in results) for key in (
                "lcm_rows", "host_rows", "matched", "lcm_only", "host_only", "reordered_rows", "host_stamp_repeat_rows",
                "host_generation_rows", "host_rewrites_credited", "repeated_identity_rows")},
            **{side + "_only_sessions": len(ids) for side, ids in only.items()},
            **{side + "_only_sessions_h12": ids for side, ids in only.items()}}


def _flatten(value, prefix=""):
    if isinstance(value, (dict, list)) and value:
        items = [("." + (str(k) if not prefix and k in SECTIONS else "h:" + _sha(str(k))[:8]), v)
                 for k, v in value.items()] if isinstance(value, dict) else [(f"[{i}]", v) for i, v in enumerate(value)]
        return {key: leaf for segment, child in items for key, leaf in _flatten(child, prefix + segment).items()}
    return {prefix: value}


def _below(value, root):
    """A value's path below its root, or None when it is not under it. Trailing separators on the root are ignored;
    a bare "/" stays the root."""
    if not isinstance(value, str) or not value or not root:  # An empty value is not a path.
        return None
    base = root.rstrip("/")
    if value == root or value.rstrip("/") == base:
        return ""
    return value[len(base):] if value.startswith(base + "/") else None


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
        below = [_below(value, root) for value, root in ((va, root_a), (vb, root_b))]
        path_only = None not in below and below[0] == below[1]
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
    if not checks[0]:
        raise ValueError("empty doctor input")  # No candidate checks is missing evidence, never a match.

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
        if isinstance(detail, dict):  # Every dict value as a cause flag, so a new cause cannot hide in a warning.
            conditions = tuple(sorted((_key_name(key), json.dumps(_cause(value), sort_keys=True))
                                      for key, value in detail.items()))
        return row["check"], row["status"], shape, conditions

    warnings = [Counter(signature(row) for row in rows if row["status"] == "warn") for rows in checks]
    results = [{"check": sig[0], "candidate": count, "previous": warnings[1][sig], "matched": count <= warnings[1][sig]}
               for sig, count in sorted(warnings[0].items(), key=lambda item: repr(item[0]))]
    unmatched = sorted({row["check"] for row in results if not row["matched"]})
    failed = sorted({row["check"] for rows in checks for row in rows if row["status"] == "fail"})
    statuses = {"pass", "warn", "fail", "ok", "skip"}
    unknown = sorted({row["check"] for rows in checks for row in rows if row["status"] not in statuses})
    signatures = [{"check": sig[0], "status": sig[1] if sig[1] in statuses else "unknown_status",
                   "detail_keys": [_key_name(key) for key in sig[2][1]] if sig[2][0] == "dict" else [],
                   **({"detail_fingerprints": list(sig[2][1])} if sig[2][0] in ("list", "str") else {}),
                   "detail_type": sig[2][0],
                   "causes": {key: json.loads(value) for key, value in sig[3]}}
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
