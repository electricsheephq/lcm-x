#!/usr/bin/env python3
"""Check capture message preservation using explicit transcripts and read-only contract SQL.

Repeat --transcript SESSION=PATH in observation order, including historical versions.
Supply each rewritten transcript's set-aside copy before its replacement.
Subagents use SESSION/agent:ID. Output contains counts and ids, never transcript text.
For a reused agent id's later transcript, use SESSION/agent:/ plus its path below subagents/.
Externalized references and ingest-payload replacements inside tool-call arguments
verify source character and UTF-8 byte counts and SHA-256 prefixes.
A preview naming a file outside tool-results/ of every own session folder of its lineage (found as capture finds
them, both sides resolved) verifies as a preview copy, its stored row equal to the preview text, only when a row of
another lineage verified against that same file (a forked session's copy of its parent's preview); every other
preview is checked against its saved file or stays unverified. Each lineage reports its preview_copies count.
Raw ledger preservation and payload-file recovery are outside this row checker.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
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
                        structured = isinstance(row[2], (list, dict))
                        sources[session].append((json.dumps(row[2], ensure_ascii=False, sort_keys=True)
                                                 if structured else row[2], structured))
                        if row[1] == "tool" and isinstance(row[2], str) and row[2].lstrip().startswith("<persisted-output>"):
                            previews[session, len(expected[session]) - 1] = entry.get("toolUseResult")
                    seen[session].add(identity)
    return expected, unprojected, previews, sources


# Ref names carry a SHA-256 prefix; no payload file is opened by this checker.
REFERENCE = re.compile(
    r"\[Externalized (?:tool output|payload|LCM ingest payload): [^\]\r\n]*?"
    r"chars=(\d+); bytes=(\d+); [^\]\r\n]*?"
    r"ref=\d{8}_\d{6}_[^\]\r\n]+_([0-9a-f]{12})_[0-9a-f]+\.json\]")
INGEST = re.compile(r"\[Externalized LCM ingest payload: [^\]\r\n]*\]")
MAX_SAVED_OUTPUT_BYTES = 64 * 1024 * 1024


SAVED_TO = re.compile(r"Full output saved to:\s*(?P<path>[^\r\n]+)")
SET_ASIDE_TAGS = (".orphaned-", ".superseded-")  # claude_code_store.SET_ASIDE_TAGS
SUBAGENT_DEPTH = 4  # claude_code_capture.SUBAGENT_DEPTH: workflow agents sit at workflows/<run-id>/ (#2562)


def session_folder(transcript):
    """Capture's session_dir(): <session>/ beside <session>.jsonl (only the last suffix is dropped), or, for a
    subagent transcript, the folder holding its subagents/ (up to four folders below it)."""
    depth = SUBAGENT_DEPTH + 1 if transcript.name.startswith("agent-") else 1  # a main transcript is never agent-*
    folder = next((up for up in list(transcript.parents)[:depth] if up.name == "subagents"), None)
    return folder.parent if folder is not None else transcript.with_suffix("")


def own_session_folders(transcript):
    """The folders capture may have read this supplied transcript's saved outputs from: its session_dir() and, for a
    set-aside copy (a name holding .orphaned- or .superseded-), which capture drains under its replacement's
    lineage, also the replacement's: the name up to its first tag, ending in .jsonl. Both readings of such a name
    are kept, so a live transcript whose name holds a tag is covered too."""
    folders = [session_folder(transcript)]
    cut = min((transcript.name.find(tag) for tag in SET_ASIDE_TAGS if tag in transcript.name), default=-1)
    if cut > 0:
        live = transcript.name[:cut]
        folders.append(session_folder(transcript.with_name(live if live.endswith(".jsonl") else live + ".jsonl")))
    return folders


def outside_named_path(text, metadata, transcripts):
    """The resolved path this preview names when it lies outside tool-results/ of every folder capture may have read
    this lineage's saved outputs from (own_session_folders() of each supplied transcript), else None. Both sides are
    resolved, so a linked file or folder cannot make an outside file look inside. Capture keeps such a preview as
    the row's text, because its saved-output recovery reads only the folder of the lineage's live transcript
    (claude_code_capture._recover_saved_output). check() expects that preview only for a copy: a row of another
    lineage verified against the same resolved file (a forked session's copy of its parent's preview names the
    parent's file). A relative or ~ path resolves in this process's working directory and HOME."""
    match = SAVED_TO.search(text)
    saved = metadata.get("persistedOutputPath") if isinstance(metadata, Mapping) else None
    named = (match["path"].strip() if match else None) or (saved if isinstance(saved, str) and saved else None)
    if not named or not transcripts:
        return None
    try:
        path = Path(named).expanduser().resolve()
        if any(path.is_relative_to((folder / "tool-results").resolve())
               for transcript in transcripts for folder in own_session_folders(transcript)):
            return None
        return path
    except (OSError, RuntimeError, ValueError):
        return None


def saved_preview(text, transcripts, metadata):
    """Try every supplied path's session folder, without reading oversized outputs. Returns the saved file's text,
    the reason it could not be read, and the resolved file it read."""
    match = re.search(r"Full output saved to:\s*([^\r\n]+)", text)
    named = match[1].strip() if match else None
    if not named and isinstance(metadata, Mapping):
        named = metadata.get("persistedOutputPath")
    if not named:
        return None, "saved_output_unnamed", None
    if isinstance(named, str) and len(named) >= 2 and named[0] == named[-1] and named[0] in "\"'":
        named = named[1:-1]
    reason = "saved_output_outside_session"
    for transcript in transcripts:
        ancestors = list(transcript.parents)[:5] if transcript.name.startswith("agent-") else []
        root = next((up.parent for up in ancestors if up.name == "subagents"), transcript.with_suffix(""))
        folder = root / "tool-results"
        try:
            path = Path(named).expanduser()
            if (root.is_symlink() or folder.is_symlink() or path.is_symlink()
                    or not path.resolve().is_relative_to(folder.resolve())):
                continue
            if not path.exists():
                reason = "saved_output_missing"
            elif not path.is_file():
                reason = "saved_output_not_a_file"
            elif path.stat().st_size > MAX_SAVED_OUTPUT_BYTES:
                reason = "saved_output_too_large"
            else:
                with path.open("rb") as handle:
                    data = handle.read(MAX_SAVED_OUTPUT_BYTES + 1)
                if len(data) > MAX_SAVED_OUTPUT_BYTES:  # Bound the read even if the file grew.
                    reason = "saved_output_too_large"
                else:
                    return data.decode("utf-8"), None, path.resolve()
        except UnicodeError:
            reason = "saved_output_binary"
        except (OSError, ValueError, RuntimeError, TypeError):
            reason = "saved_output_unreadable"
    return None, reason, None


def fingerprint(text):
    data = text.encode("utf-8")
    return len(text), len(data), hashlib.sha256(data).hexdigest()[:12]


def index_source(text, index, structured):
    index[fingerprint(text)].add(text)
    if not structured:
        return None
    try:
        value = json.loads(text) if text[:1] in {"[", "{"} else None
    except (ValueError, RecursionError):
        value = None
    pending = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, str):
            index[fingerprint(node)].add(node)
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, dict):
            pending.extend(node.values())
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
        span = source[position:position + int(ref[1])] if ref else ""
        if ref:
            index[fingerprint(span)].add(span)
        if ref and reference_matches(ref, span, index):
            position += len(span)
        elif source.startswith(match[0], position):
            position += len(match[0])  # A quoted marker is literal source text.
        else:
            break
        start = match.end()
    else:
        if start and stored[start:] == source[position:]:
            return True
    if structured is None or not INGEST.search(stored):
        return False
    try:
        value = json.loads(stored)
    except (ValueError, RecursionError):
        return False
    pending = [(value, structured)]
    while pending:
        left, right = pending.pop()
        if isinstance(left, str) and isinstance(right, str):
            if not content_matches(left, right, None, index):
                return False
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right))
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return False
            pending.extend((left[k], right[k]) for k in left)
        elif type(left) is not type(right) or left != right:
            return False
    return True


def tool_calls_match(stored_calls, source_calls, index):
    """Verify argument replacements while preserving call identity and JSON structure."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_argument_key")
            value[key] = item
        return value

    if not isinstance(stored_calls, list) or not isinstance(source_calls, list):
        return False
    if len(stored_calls) != len(source_calls):
        return False
    for stored, source in zip(stored_calls, source_calls):
        if not isinstance(stored, dict) or not isinstance(source, dict) or stored.keys() != source.keys():
            return False
        if any(canonical(stored.get(key)) != canonical(source.get(key)) for key in ("id", "type")):
            return False
        left, right = stored.get("function"), source.get("function")
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        if canonical(left.get("name")) != canonical(right.get("name")):
            return False
        left, right = left.get("arguments"), right.get("arguments")
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        if left == right:
            continue
        try:
            stored_value = json.loads(left, object_pairs_hook=unique_object)
        except json.JSONDecodeError:
            # Valid stored JSON is matched structurally; stored text that is not JSON passes only when
            # every source character is exact text or lies inside a placeholder verified at its own
            # position. The product writes such text when a payload run takes in an escape's letter.
            if content_matches(left, right, None, index):
                continue
            return False
        except (ValueError, RecursionError):
            return False
        try:
            # The source is json.dumps output (project() above, as capture builds it): valid JSON with
            # no escaped slash, duplicate key or NaN. Anything else fails closed.
            pending = [(stored_value, json.loads(right, object_pairs_hook=unique_object))]
        except (ValueError, RecursionError):
            return False
        while pending:
            left, right = pending.pop()
            if isinstance(left, str) and isinstance(right, str):
                if not content_matches(left, right, None, index):
                    return False
            elif isinstance(left, dict) and isinstance(right, dict):
                if left.keys() != right.keys():
                    return False
                pending.extend((left[key], right[key]) for key in left)
            elif isinstance(left, list) and isinstance(right, list):
                if len(left) != len(right):
                    return False
                pending.extend(zip(left, right))
            elif type(left) is not type(right) or left != right:
                return False
            elif isinstance(left, float) and repr(left) != repr(right):
                return False
    return True


def check(home, inputs):
    started = perf_counter()
    expected, unprojected, previews, transcript_sources = expected_rows(inputs)
    transcript_paths = defaultdict(list)
    for session, path in inputs:
        transcript_paths[session].append(path)
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
    unexpected = [{"session_id": s, "rows": len(grouped[s])} for s in sorted(grouped.keys() - expected.keys())]

    def store_side(session):
        actual, ids, ambiguous, binding_excess = [], [], Counter(), 0
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
        return actual, ids, ambiguous, binding_excess, calls, results

    def transcript_side(session, owned):
        """Prepare each transcript occurrence once. A preview naming a path outside every own folder is expected as
        its own text only when `owned` holds that resolved path; every other preview takes saved_preview()."""
        wanted, sources, reasons, files, named_outside, copies = list(expected[session]), [], {}, {}, set(), set()
        for i, row in enumerate(wanted):
            source, structured = transcript_sources[session][i]
            if (session, i) in previews:
                metadata = previews[session, i]
                named = outside_named_path(source, metadata, transcript_paths[session])
                if named is not None:
                    named_outside.add(named)
                if named is not None and named in owned:
                    copies.add(i)  # capture stored the preview itself, so the stored row must equal it exactly
                else:
                    recovered, reason, file = saved_preview(source, transcript_paths[session], metadata)
                    if reason:
                        reasons[i] = reason
                    else:
                        source, files[i] = recovered, file
                        wanted[i] = row[:2] + (canonical(source),) + row[3:]
            sources.append((source, structured))
        return wanted, sources, reasons, files, named_outside, copies

    def compare(session, store, wanted, sources, reasons):
        """Pair the k-th row of each uid; returns the lineage's report and the occurrences whose rows verified."""
        actual, ids, ambiguous, binding_excess, calls, results = store
        actual, index, slots = list(actual), defaultdict(set), defaultdict(list)
        for i, row in enumerate(wanted):
            slots[row[0]].append(i)
        used, unverified, unverified_reasons, verified = Counter(), 0, [], set()
        for n, row in enumerate(actual):
            k, uid = used[row[0]], row[0]
            used[uid] += 1
            if k >= len(slots[uid]):
                continue
            i = slots[uid][k]
            target = wanted[i]
            if row[:2] + row[3:4] != target[:2] + target[3:4]:
                continue
            if row[4] != target[4] and (not INGEST.search(grouped[session][n]["tool_calls"] or "")
                    or not tool_calls_match(json.loads(row[4]), json.loads(target[4]), index)):
                continue
            if i in reasons:
                actual[n] = target
                unverified += 1
                unverified_reasons.append({"uid": uid, "reason": reasons[i]})
            elif row == target:
                verified.add(i)
            else:
                source, is_structured = sources[i]
                structured = index_source(source, index, is_structured) if isinstance(source, str) else None
                if content_matches(json.loads(row[2]), source, structured, index):
                    actual[n] = target
                    verified.add(i)
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
        return report, verified

    # Pass 1 checks every row with no preview copies, and records which lineages' verified rows read each file.
    stores, firsts, owners = {}, {}, defaultdict(set)
    for session in sorted(expected):
        stores[session] = store_side(session)
        wanted, sources, reasons, files, named_outside, _ = transcript_side(session, frozenset())
        report, verified = compare(session, stores[session], wanted, sources, reasons)
        for i in verified & files.keys():
            owners[files[i]].add(session)
        firsts[session] = report, named_outside
    # Pass 2 re-checks a lineage whose preview names a file that a verified row of ANOTHER lineage read: a preview
    # copy. A lineage never owns its own copy, even where saved_preview() reads a folder session_folder() does not.
    reports = []
    for session in sorted(expected):
        report, named_outside = firsts[session]
        owned = {path for path in named_outside if owners.get(path, set()) - {session}}
        copies = set()
        if owned:
            wanted, sources, reasons, _, _, copies = transcript_side(session, owned)
            report, verified = compare(session, stores[session], wanted, sources, reasons)
            copies &= verified
        report["preview_copies"] = len(copies)
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
