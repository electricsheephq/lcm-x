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
from time import perf_counter


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
    previews, sources = {}, defaultdict(list)
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
                    for row in project(entry):
                        sig = signature(row)
                        expected[session].append(sig)
                        sources[session].append(json.dumps(row[2], ensure_ascii=False, sort_keys=True)
                                                if isinstance(row[2], (list, dict)) else row[2])
                        if row[1] == "tool" and isinstance(row[2], str) and row[2].lstrip().startswith("<persisted-output>"):
                            previews[session, len(expected[session]) - 1] = (path, entry.get("toolUseResult"))
                    seen[session].add(identity)
    return expected, unprojected, previews, sources


# Ref names carry a SHA-256 prefix; no payload file is opened by this checker.
REFERENCE = re.compile(
    r"\[Externalized (?:tool output|payload|LCM ingest payload): [^\]\r\n]*?"
    r"chars=(\d+); bytes=(\d+); [^\]\r\n]*?"
    r"ref=\d{8}_\d{6}_[^\]\r\n]+_([0-9a-f]{12})_[0-9a-f]+\.json\]")
INGEST = re.compile(r"\[Externalized LCM ingest payload: [^\]\r\n]*\]")
FILE_CHUNK_BYTES = 64 * 1024 * 1024


def saved_preview(text, transcript, metadata):
    """Read only a named saved output inside this transcript's session tool-results/."""
    match = re.search(r"Full output saved to:\s*([^\r\n]+)", text)
    named = match[1].strip() if match else (metadata or {}).get("persistedOutputPath")
    if not named:
        return None, "saved_output_unnamed"
    ancestors = list(transcript.parents)[:5] if transcript.name.startswith("agent-") else []
    root = next((up.parent for up in ancestors if up.name == "subagents"), transcript.with_suffix(""))
    folder = root / "tool-results"
    try:
        path = Path(named).expanduser()
        if (root.is_symlink() or folder.is_symlink() or path.is_symlink()
                or not path.resolve().is_relative_to(folder.resolve())):
            return None, "saved_output_outside_session"
        if not path.exists():
            return None, "saved_output_missing"
        if not path.is_file():
            return None, "saved_output_not_a_file"
        with path.open("rb") as handle:
            chunks = iter(lambda: handle.read(FILE_CHUNK_BYTES), b"")
            return b"".join(chunks).decode("utf-8"), None
    except UnicodeError:
        return None, "saved_output_binary"
    except (OSError, ValueError, RuntimeError):
        return None, "saved_output_unreadable"


def fingerprint(text):
    data = text.encode("utf-8")
    return len(text), len(data), hashlib.sha256(data).hexdigest()[:12]


def index_source(text, index):
    index[fingerprint(text)].add(text)
    try:
        value = json.loads(text) if text[:1] in {"[", "{"} else None
    except ValueError:
        value = None
    def visit(node):
        if isinstance(node, str):
            index[fingerprint(node)].add(node)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)
    visit(value)
    return value


def reference_matches(match, source, index):
    key = int(match[1]), int(match[2]), match[3]
    return source in index.get(key, ())


def content_matches(stored, source, structured, index):
    if stored == source:
        return True
    if not isinstance(stored, str) or not isinstance(source, str):
        return False
    match = REFERENCE.fullmatch(stored)
    if match and reference_matches(match, source, index):
        return True
    # Verify each replacement at its own position, preserving all surrounding text.
    position, start = 0, 0
    for match in INGEST.finditer(stored):
        literal = stored[start:match.start()]
        if not source.startswith(literal, position):
            break
        position += len(literal)
        ref = REFERENCE.fullmatch(match[0])
        if ref is None:
            return False
        span = source[position:position + int(ref[1])]
        index[fingerprint(span)].add(span)
        if not reference_matches(ref, span, index):
            break
        position += len(span)
        start = match.end()
    else:
        if start and stored[start:] == source[position:]:
            return True
    if structured is None or not INGEST.search(stored):
        return False
    try:
        value = json.loads(stored)
    except ValueError:
        return False
    def compare(left, right):
        if isinstance(left, str) and isinstance(right, str):
            return content_matches(left, right, None, index)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(compare(a, b) for a, b in zip(left, right))
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(compare(left[k], right[k]) for k in left)
        return type(left) is type(right) and left == right
    return compare(value, structured)


def check(home, inputs):
    started = perf_counter()
    expected, unprojected, previews, transcript_sources = expected_rows(inputs)
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
    message_ids = {r["store_id"] for r in messages}
    dangling = [b for row_id in sorted(bindings) if row_id not in message_ids for b in bindings[row_id]]
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
        # Prepare each transcript occurrence once, then pair the k-th row of each uid.
        sources, reasons, index = [], {}, defaultdict(set)
        for i, row in enumerate(wanted):
            source = transcript_sources[session][i]
            if previews and (session, i) in previews:
                recovered, reason = saved_preview(source, *previews[session, i])
                if reason:
                    reasons[i] = reason
                else:
                    source = recovered
                    wanted[i] = row[:2] + (canonical(source),) + row[3:]
            sources.append(source)
        slots = defaultdict(list)
        for i, row in enumerate(wanted):
            slots[row[0]].append(i)
        used, unverified, unverified_reasons = Counter(), 0, []
        for n, row in enumerate(actual):
            k, uid = used[row[0]], row[0]
            used[uid] += 1
            if k >= len(slots[uid]):
                continue
            i = slots[uid][k]
            target = wanted[i]
            if row[:2] + row[3:] != target[:2] + target[3:]:
                continue
            if i in reasons:
                actual[n] = target
                unverified += 1
                unverified_reasons.append({"uid": uid, "reason": reasons[i]})
            elif row != target:
                source = sources[i]
                structured = index_source(source, index) if isinstance(source, str) else None
                if content_matches(json.loads(row[2]), source, structured, index):
                    actual[n] = target
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
                  "unverified": unverified, "unprojected": unprojected[session],
                  "unverified_reasons": unverified_reasons}
        report["pass"] = not lost and not duplicated and order and not unpaired
        reports.append(report)
    return {"pass": bool(reports) and not unexpected and not dangling and all(r["pass"] for r in reports),
            "lineages": reports, "unexpected_sessions": unexpected, "dangling_bindings": dangling,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3)}


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
