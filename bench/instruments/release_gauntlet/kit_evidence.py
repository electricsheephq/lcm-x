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


LOG = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:,\d+)? \[(WARNING|ERROR|INFO)\] ([\w.]+): ")
FINAL = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))\b")
MISSING = re.compile(r"No module named '([\w.]+)'")
REFUSAL = "NotImplementedError: LCM-X cannot detach a context engine"


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


def trace_events(log):
    runs, outside, events, other = [], 0, Counter(), Counter()
    for path in log:
        active, logger, traceback = None, None, False
        with Path(path).open(encoding="utf-8") as stream:
            for raw in stream:
                line = raw.rstrip("\r\n")
                event = LOG.match(line)
                if event:
                    logger = event[2]
                if "thread=bg-review" in line and "OpenAI client created" in line:
                    runs.append(0)
                    active = len(runs) - 1
                if line.startswith(REFUSAL):
                    if active is None:
                        outside += 1
                    else:
                        runs[active] += 1
                if "Background review complete: thread=bg-review" in line:
                    active = None
                if line == "Traceback (most recent call last):":
                    traceback = True
                final = FINAL.match(line) if traceback else None
                if final:
                    if line.startswith(("ModuleNotFoundError", "ImportError")):
                        missing = MISSING.search(line)
                        events[(logger, missing[1] if missing else None)] += 1
                    elif not line.startswith(REFUSAL):
                        other[final[1]] += 1
                    traceback = False
    return {"bg_review_runs": len(runs), "refusals_total": sum(runs) + outside,
            "runs_by_refusal_count": dict(sorted(Counter(map(str, runs)).items())),
            "refusals_outside_runs": outside,
            "per_event_rule_holds": all(n <= 1 for n in runs) and outside == 0,
            "optional_import_traces_total": sum(events.values()),
            "by_event": [{"logger": key[0], "missing_module": key[1], "count": count}
                         for key, count in sorted(events.items(), key=lambda item: tuple(x or "" for x in item[0]))],
            "unattributed": sum(n for (logger, _), n in events.items() if logger is None),
            "other_tracebacks": dict(sorted(other.items()))}


def lifecycle_rows(lcm_db):
    with _db(lcm_db) as conn:
        counts = [Counter({str(row[0]): row[1] for row in conn.execute(
            f"SELECT session_id, COUNT(*) FROM {table} GROUP BY session_id") if row[0]})
            for table in ("messages", "summary_nodes")]
        stored = set(counts[0]) | set(counts[1])
        rows = conn.execute("SELECT conversation_id, current_session_id, last_finalized_session_id "
                            "FROM lcm_lifecycle_state ORDER BY conversation_id").fetchall()
        empty, affected_sessions = [], set()
        for row in rows:
            refs = {str(value) for value in (row[1], row[2]) if value}
            if not refs or refs.isdisjoint(stored):
                affected_sessions.update(refs)
                empty.append({"conversation_id_h12": _sha(str(row[0]))[:12] if row[0] else None,
                              "current_session_h12": _sha(str(row[1]))[:12] if row[1] else None,
                              "last_finalized_session_h12": _sha(str(row[2]))[:12] if row[2] else None,
                              "messages_in_referenced_sessions": sum(counts[0][sid] for sid in refs),
                              "nodes_in_referenced_sessions": sum(counts[1][sid] for sid in refs)})
        affected = sum(counts[0][sid] for sid in affected_sessions)
        return {"lifecycle_rows": len(rows), "empty_lifecycle_rows": len(empty), "empty_rows": empty,
                "affected_stored_rows": affected, "identity_order_check": "PASS" if affected == 0 else "FAIL"}


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
    return sessions


def reorder(lcm_db, state_db):
    lcm, host = _identities(lcm_db, "store_id"), _identities(state_db, "id")
    results = []
    for sid in sorted(lcm.keys() & host.keys()):
        a, b = lcm[sid], host[sid]
        ca, cb = Counter(a), Counter(b)
        unique = {key for key in ca.keys() & cb.keys() if ca[key] == cb[key] == 1}
        positions = {key: i for i, key in enumerate(key for key in b if key in unique)}
        tails = []  # Unique identities turn the LCS into an increasing subsequence of host positions.
        for key in (key for key in a if key in unique):
            pos = positions[key]
            i = bisect_left(tails, pos)
            if i == len(tails):
                tails.append(pos)
            else:
                tails[i] = pos
        results.append({"session_h12": _sha(str(sid))[:12], "matched": len(unique),
                        "reordered_rows": len(unique) - len(tails)})
    return {"sessions": len(results), "matched": sum(row["matched"] for row in results),
            "reordered_rows": sum(row["reordered_rows"] for row in results), "per_session": results}


def _flatten(value, prefix=""):
    if isinstance(value, (dict, list)) and value:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return {key: leaf for name, child in items
                for key, leaf in _flatten(child, f"{prefix}.{name}" if prefix else str(name)).items()}
    return {prefix: value}


def config_diff(a, root_a, b, root_b):
    import yaml  # only config-diff needs a YAML reader; the release kit environment provides PyYAML

    try:
        configs = [_flatten(yaml.safe_load(Path(path).read_text(encoding="utf-8"))) for path in (a, b)]
    except yaml.YAMLError:
        raise ValueError("unreadable config") from None  # never echo config text
    missing, paths, others = object(), [], []
    for key in sorted(configs[0].keys() | configs[1].keys()):
        va, vb = (config.get(key, missing) for config in configs)
        if type(va) is type(vb) and va == vb:
            continue
        normalized = ["<ROOT>" + value[len(root):] if root and value.startswith(root) else value
                      for value, root in ((va, root_a), (vb, root_b))] if isinstance(va, str) and isinstance(vb, str) else []
        (paths if normalized and normalized[0] == normalized[1] else others).append(key)
    return {"differing_keys": sorted(paths + others), "path_only_keys": paths, "other_keys": others,
            "verdict": "OTHER" if others else "PATHS_ONLY"}


def doctor_match(candidate, previous):
    checks = []
    for path in (candidate, previous):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        checks.append((value[-1] if isinstance(value, list) else value)["checks"])
    warnings = [Counter(row["check"] for row in rows if row["status"] == "warn") for rows in checks]
    results = [{"check": name, "candidate": count, "previous": warnings[1][name],
                "matched": count <= warnings[1][name] and warnings[1][name] > 0}
               for name, count in sorted(warnings[0].items())]
    unmatched = [row["check"] for row in results if not row["matched"]]
    failed = sorted({row["check"] for rows in checks for row in rows if row["status"] == "fail"})
    candidate_failed = any(row["status"] == "fail" for row in checks[0])
    return {"warnings": results, "unmatched": unmatched, "failed_checks": failed,
            "verdict": "UNMATCHED" if unmatched or candidate_failed else "MATCHED"}


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
