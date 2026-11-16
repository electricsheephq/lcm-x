#!/usr/bin/env python3
"""Check capture message preservation using explicit transcripts and read-only contract SQL.

Repeat --transcript SESSION=PATH in observation order, including historical versions.
Subagents use SESSION/agent:ID. Output contains counts and ids, never transcript text.
Raw ledger preservation and payload-file recovery are outside this row checker.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def joined(parts):
    return "\n\n".join(p["text"] for p in parts if p.get("type") == "text" and isinstance(p.get("text"), str))


def project(entry):
    """Independent projection-v1; returns (component uid, role, content, call id, calls)."""
    uid, message = entry.get("uuid"), entry.get("message")
    if entry.get("type") not in {"user", "assistant"} or not isinstance(message, dict):
        return []
    if not isinstance(uid, str) or not uid:
        raise ValueError("missing_uid")
    content = message.get("content", [])
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else [
        p for p in content or [] if isinstance(p, dict)]
    rows = []
    if entry["type"] == "assistant":
        text = joined(parts)
        calls = [{"id": p.get("id"), "type": "function", "function": {
            "name": p.get("name"), "arguments": json.dumps(p.get("input", {}), ensure_ascii=False)}}
            for p in parts if p.get("type") == "tool_use"]
        if entry.get("isApiErrorMessage"):
            rows.append(("system", text, None, []))
        elif text or calls:
            rows.append(("assistant", text, None, calls))
    else:
        for part in parts:
            if part.get("type") == "tool_result":
                result = part.get("content")
                if isinstance(result, list):
                    result = [p for p in result if isinstance(p, dict)]
                    result = joined(result) if all(p.get("type") == "text" for p in result) else result
                elif not isinstance(result, str):
                    result = ""
                rows.append(("tool", result, part.get("tool_use_id"), []))
        rest = [p for p in parts if p.get("type") in {"text", "image"}]
        if rest:
            role = "system" if any(entry.get(f) for f in (
                "isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")) else "user"
            rows.append((role, joined(rest) if all(p.get("type") == "text" for p in rest) else rest, None, []))
    return [(uid if len(rows) == 1 else f"{uid}#{i}", *row) for i, row in enumerate(rows)]


def signature(row):
    uid, role, content, call_id, calls = row
    if isinstance(content, (list, dict)):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return uid, role, canonical(content), call_id, canonical(calls)


def expected_rows(inputs):
    expected, seen, unprojected = defaultdict(list), defaultdict(set), Counter()
    for session, path in inputs:
        expected[session]  # Include a transcript with no projected rows.
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, UnicodeError):
                    unprojected[session] += 1
                    continue
                if not isinstance(entry, dict) or (entry.get("type") in {"user", "assistant"}
                        and isinstance(entry.get("message"), dict) and not (
                            isinstance(entry.get("uuid"), str) and entry["uuid"])):
                    unprojected[session] += 1
                    continue
                digest = canonical({k: v for k, v in entry.items() if k not in {"sessionId", "promptId"}})
                identity = entry.get("uuid"), digest
                if identity not in seen[session]:
                    expected[session].extend(signature(row) for row in project(entry))
                    seen[session].add(identity)
    return expected, unprojected


def check(home, inputs):
    expected, unprojected = expected_rows(inputs)
    with closing(sqlite3.connect((home / "lcm.db").resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "messages" not in tables:
            raise ValueError("missing_messages")
        messages = [dict(r) for r in conn.execute(
            "SELECT store_id, session_id, role, content, tool_call_id, tool_calls FROM messages ORDER BY store_id")]
        bindings = defaultdict(list)
        if "host_uid_bindings" in tables:
            for r in conn.execute("SELECT store_id, uid, lineage_key, kind FROM host_uid_bindings"):
                bindings[r["store_id"]].append(dict(r))
        revisions = list(conn.execute(
            "SELECT store_id, related_store_id FROM message_relations WHERE kind='supersedes'")) \
            if "message_relations" in tables else []
        parents = dict(conn.execute("SELECT child_session_id, parent_session_id FROM session_lineage")) \
            if "session_lineage" in tables else {}
    grouped, lineage_bindings = defaultdict(list), {}
    for row in messages:
        grouped[row["session_id"]].append(row)
        lineage_bindings[row["store_id"]] = sorted(
            (b for b in bindings[row["store_id"]] if b["lineage_key"] == "claude-code:" + row["session_id"]),
            key=lambda b: (b["kind"] != "canonical", b["uid"]))
    # Split components use <transcript uid>#<index> (claude_code_capture.project_entry).
    revision_uids = {row_id: re.sub(r"#[0-9]+\Z", "", bound[0]["uid"]) if bound else None
                     for row_id, bound in lineage_bindings.items()}
    predecessors, owners = defaultdict(list), {r["store_id"]: r["session_id"] for r in messages}
    for new, old in revisions:
        if (new not in owners or old not in owners or old >= new or owners[new] != owners[old]
                or revision_uids[new] is None or revision_uids[new] != revision_uids[old]):
            raise ValueError("invalid_revision_ancestry")
        predecessors[new].append(old)
    superseded = {old for _, old in revisions}
    tool_ids = {r["store_id"]: r["tool_call_id"] for r in messages}
    origins, call_origins = {}, {}
    for row in messages:  # store_id order makes every predecessor's origin available.
        row_id = row["store_id"]
        origins[row_id] = min([row_id, *(origins[old] for old in predecessors[row_id]
                                      if row["tool_call_id"] == tool_ids[old])])
        call_origins[row_id] = {
            call_id: min([row_id, *(call_origins[old][call_id] for old in predecessors[row_id]
                                   if call_id in call_origins[old])])
            for call_id in (call.get("id") for call in json.loads(row["tool_calls"] or "[]"))}
    reports = []
    unexpected = [{"session_id": s, "rows": len(grouped[s])} for s in sorted(grouped.keys() - expected.keys())]
    for session in sorted(expected):
        wanted, actual, ids = expected.get(session, []), [], []
        ambiguous, binding_excess = Counter(), 0
        calls, results = defaultdict(list), defaultdict(list)
        for row in grouped[session]:
            row_id = row["store_id"]
            tool_calls = json.loads(row["tool_calls"] or "[]")
            content = row["content"]
            bound = lineage_bindings[row_id]
            binding_excess += sum(b["lineage_key"] != "claude-code:" + session for b in bindings[row_id])
            bound = bound or [{"uid": None}]
            actual.append(signature((bound[0]["uid"], row["role"], content, row["tool_call_id"], tool_calls)))
            ids.append(row_id)
            binding_excess += len(bound) - 1
            # Extra claims are ambiguous, charged to duplication rather than also to loss.
            ambiguous.update(signature((b["uid"], row["role"], content, row["tool_call_id"], tool_calls))
                             for b in bound[1:])
            if row_id not in superseded:
                for call in tool_calls:
                    calls[call.get("id")].append(call_origins[row_id][call.get("id")])
                if row["role"] == "tool":
                    results[row["tool_call_id"]].append(origins[row_id])
        # Only transcript previews can stand in for recovered content.
        opaque = {r for r in wanted if r[1] == "tool"
                  and isinstance(text := json.loads(r[2]), str)
                  and text.lstrip().startswith("<persisted-output>")}
        def normalize(row):
            text = json.loads(row[2])
            match = re.fullmatch(
                r'\[Externalized tool output: .*; chars=(\d+); bytes=(\d+); .*; '
                r'ref=\d{8}_\d{6}_.+_([0-9a-f]{12})_[0-9a-f]+\.json\]', text) \
                if row[1] == "tool" and isinstance(text, str) else None
            if match:
                for candidate in wanted:
                    source = json.loads(candidate[2])
                    if (candidate not in opaque and candidate[:2] + candidate[3:] == row[:2] + row[3:] and isinstance(source, str)
                            and (len(source), len(source.encode("utf-8")), hashlib.sha256(source.encode("utf-8")).hexdigest()[:12])
                            == (int(match[1]), int(match[2]), match[3])):
                        return candidate
            if row[1] == "tool" and isinstance(text, str) and text.startswith("[Externalized tool output:"):
                return row[:2] + ("null",) + row[3:]
            return row
        # Literal references and markers are content when the transcript kept them verbatim.
        exact, normalized = Counter(wanted), []
        for row in actual:
            if exact[row]:
                exact[row] -= 1
                normalized.append(row)
            else:
                normalized.append(normalize(row))
        actual = normalized
        ambiguous = Counter(map(normalize, ambiguous.elements()))
        # Reserve exact content matches before spending any occurrence's preview slot.
        remaining = Counter(r for r in wanted if r not in opaque)
        previews = Counter(r[:2] + r[3:] for r in wanted if r in opaque)
        pending, unverified = [], 0
        for i, row in enumerate(actual):
            if remaining[row]:
                remaining[row] -= 1
            else:
                pending.append(i)
        for i in pending:
            identity = actual[i][:2] + actual[i][3:]
            if previews[identity]:
                actual[i] = actual[i][:2] + ("null",) + actual[i][3:]
                previews[identity] -= 1
                unverified += 1
        wanted = [r[:2] + ("null",) + r[3:] if r in opaque else r for r in wanted]
        need, have = Counter(wanted), Counter(actual)
        missing = need - have - ambiguous
        lost = sum(missing.values())
        duplicated = sum((have - need).values()) + binding_excess
        positions = defaultdict(list)
        for i, row in enumerate(wanted):
            positions[row].append(i)
        observed = []
        for row in actual:
            if positions[row]:
                observed.append(positions[row].pop(0))
        order = observed == sorted(observed)
        unpaired = sum(not uid or len(calls[uid]) != 1 or len(results[uid]) != 1
                       or calls[uid][0] >= results[uid][0] for uid in calls.keys() | results.keys())
        report = {"lineage": "claude-code:" + session, "expected": len(wanted), "actual": len(actual),
                  "lost": lost, "duplicated": duplicated, "order_kept": order, "unpaired": unpaired,
                  "store_ids": ids, "parent": parents.get(session),
                  "unverified": unverified, "unprojected": unprojected[session]}
        report["pass"] = not lost and not duplicated and order and not unpaired
        reports.append(report)
    return {"pass": bool(reports) and not unexpected and all(r["pass"] for r in reports),
            "lineages": reports, "unexpected_sessions": unexpected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--transcript", action="append", required=True, metavar="SESSION=PATH")
    args = parser.parse_args()
    try:
        inputs = [(session, Path(path)) for session, path in (item.split("=", 1) for item in args.transcript)]
        report = check(args.home, inputs)
    except Exception as exc:
        report = {"pass": False, "errors": 1, "error_type": type(exc).__name__}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
