#!/usr/bin/env python3
"""Check a hosted copy of a capture store against its manifest and, when given, its origin store as of the cut.

Hosted copy format 1. Standard library only; this file shares no code with the tools that make or restore a copy.
Every database is opened mode=ro; the copy and a rebuilt file also immutable=1, so the bytes read are the bytes
hashed. --origin must be the origin store as of the copy's cut, for example a copy saved at the cut. A live store is
not a supported input: rows written after the cut fail `tables` with only_in_origin. With --objects and --rebuild-dir
it rebuilds the generation from its parts and delta objects with its own decoder, inside that folder only; otherwise
it writes no file of its own (for an origin in WAL mode, SQLite may still touch its -shm file during the mode=ro
read). The copy and a rebuilt file must be in rollback-journal mode (header bytes 18-19). One JSON report on stdout:
counts, keys, ids, names and codes, never stored text or page bytes. Exit 0 pass, 1 fail, 2 refused input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import sys
import zlib

CONTRACT_VERSION = 2  # docs/store-contract.md
CONTRACT_DIGEST = "sha256:3818b2f6eebec8b3d6e00707ebc2b46e70c970f039671e79c0bdfa94dc0abac0"
CONTRACT_DIGESTS = {
    1: "sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9",
    CONTRACT_VERSION: CONTRACT_DIGEST,
}
LAYOUT, OWNER_KEY, STAMP_KEY = 1, "capture:owner", "import:origin"  # claude_code_store.OWNER_KEY
# The contract's generated table: each table's columns in a fresh store's order; ":N" is the primary-key position.
_TABLES = {
    "capture_events": "event_id:1 session_id lineage_key source_kind generation position uid entry_type entry_class "
                      "line_digest content_digest disposition store_ids notes raw raw_from projection_version captured_at",
    "host_uid_bindings": "store_id uid lineage_key kind binding_version proof_kind created_at first_check disagree_seen",
    "ingest_views": "lineage:1 view_seq:3 kind:2 state entries prefix_chain",
    "lcm_lifecycle_state": "conversation_id:1 current_session_id last_finalized_session_id current_frontier_store_id "
                           "last_finalized_frontier_store_id debt_kind debt_size_estimate current_bound_at "
                           "last_finalized_at debt_updated_at last_maintenance_attempt_at last_rollover_at "
                           "last_reset_at updated_at",
    "lcm_migration_state": "step_name:1 completed_at",
    "ledger_emissions": "emission_id:1 lineage view_seq output_index descriptor_digest kind source_store_ids state "
                        "next_occ_id observed_stamp recovered_store_id",
    "ledger_meta": "key:1 value",
    "lineage_marks": "lineage:1 max_store_id legacy_proof_digest max_leaf_id stamp_policy",
    "message_relations": "relation_id:1 store_id kind related_store_id ordinal observed_at created_at",
    "messages": "store_id:1 session_id source conversation_id role content tool_call_id tool_calls tool_name timestamp "
                "token_estimate pinned ingested_at observed_at observed_at_source sender_id sender_source",
    "metadata": "key:1 value",
    "occurrence_bindings": "occ_id:1 lineage ordinal store_id identity_digest stamp disposition tag key_rank view_seq",
    "session_lineage": "child_session_id:1 parent_session_id child_subagent_id parent_subagent_id child_role source "
                       "created_at",
    "stamp_observations": "lineage occ_id stamp kind view_seq source",
    "summary_nodes": "node_id:1 session_id depth summary token_count source_token_count source_ids source_type "
                     "created_at earliest_at latest_at expand_hint",
}
COLUMNS = {t: [c.split(":")[0] for c in spec.split()] for t, spec in _TABLES.items()}
PRIMARY = {t: [c.split(":")[0] for c in sorted((c for c in spec.split() if ":" in c), key=lambda c: int(c.split(":")[1]))]
           for t, spec in _TABLES.items()}
# A table's row key (F6): its primary key, else all its contracted columns; capture_events by its ledger keys.
KEYS = {t: PRIMARY[t] or COLUMNS[t] for t in COLUMNS} | {"host_uid_bindings": ["store_id", "uid", "lineage_key", "kind"]}
LEDGER_KEYS = {b"line": ("lineage_key", "generation", "position"), b"file": ("lineage_key", "uid", "content_digest", "position")}
EXTERNALIZED = r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"  # externalize.py
INGEST = r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]"  # ingest_content.py
MINTED = r"\d{8}_\d{6}_.+_[0-9a-f]{12}_[0-9a-f]+\.json"  # externalize.py: the names the product mints
PATTERNS, MINTED_RE = (re.compile(EXTERNALIZED), re.compile(INGEST)), re.compile(MINTED)
SHA, TOKEN = re.compile("[0-9a-f]{64}"), re.compile("[0-9a-f]{32}")
UUID = re.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
OBJECT_KEY = {kind: re.compile(kind + r"/([0-9a-f]{64})\.[0-9a-f]{32}") for kind in ("parts", "deltas", "payloads")}
MAGIC, CHUNK, CAP, PAGE_SIZES = b"LCMXPD1\n", 1 << 20, 50, {1 << k for k in range(9, 17)}


class Refused(Exception):
    def __init__(self, code, **detail):
        super().__init__(code)
        self.code, self.detail = code, detail


def is_int(value):
    return type(value) is int and value >= 0


def is_name(name):
    """F1 payload name: 1 to 255 bytes, no slash, backslash or NUL, not . or .."""
    try:
        size = len(name.encode("utf-8"))
    except (AttributeError, UnicodeError):
        return False
    return 0 < size <= 255 and not {"/", "\\", "\0"} & set(name) and name not in {".", ".."}


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def same(a, b):
    """JSON equality that keeps types apart (1, 1.0 and true differ)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same, a, b))
    return a == b


def loads(raw):
    def refuse(constant):
        raise ValueError(constant)
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw, parse_constant=refuse)


def regular(path):
    """lstat of a regular file that is not a symlink, else None."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return info if stat.S_ISREG(info.st_mode) else None


def chunks(handle, size=CHUNK):
    while block := handle.read(size):
        yield block


def file_hash(path):
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for block in chunks(handle):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def header_page_size(head):
    value = int.from_bytes(head[16:18], "big") if len(head) >= 18 else None
    return 65536 if value == 1 else value


def page_digests(handle, page_size):
    """F7: the sha256 of each page, in page order (a short last page is hashed as it is)."""
    for page in chunks(handle, page_size):
        yield hashlib.sha256(page).digest()


def scan(path, page_size):
    """(length, page size in bytes 16-17, file sha256, page_list_sha256) of one file, read in pages."""
    whole, listing, length, head = hashlib.sha256(), hashlib.sha256(), 0, b""
    with open(path, "rb") as handle:
        for page in chunks(handle, page_size):
            head, length = head or page[:18], length + len(page)
            whole.update(page)
            listing.update(hashlib.sha256(page).digest())
    return length, header_page_size(head), whole.hexdigest(), listing.hexdigest()


def inflate(stream, data):
    """Output of data through a decompressobj, in pieces of at most CHUNK bytes: memory stays bounded."""
    out = stream.decompress(data, CHUNK)
    while True:
        yield out
        if not stream.unconsumed_tail and len(out) < CHUNK:
            return
        out = stream.decompress(stream.unconsumed_tail, CHUNK)


def records(handle, page_size, expect=None, after=-1):
    """Yield (index, page) from one F8 delta object, decoded in a stream; Refused with the F8 code (or
    delta_count when the header count is not expect, delta_order when an index is not above after)."""
    head = handle.read(16)
    if len(head) < 16 or head[:8] != MAGIC:
        raise Refused("delta_magic")
    size, count = struct.unpack(">II", head[8:])
    if size != page_size:
        raise Refused("delta_page_size")
    if expect is not None and count != expect:
        raise Refused("delta_count")
    stream, pending, seen, record = zlib.decompressobj(), b"", 0, 4 + page_size
    try:
        for block in chunks(handle):
            if stream.eof:
                raise Refused("delta_trailing")
            for out in inflate(stream, block):
                pending, at = pending + out, 0
                while len(pending) - at >= record:
                    index = int.from_bytes(pending[at:at + 4], "big")
                    if seen == count:
                        raise Refused("delta_stream")
                    if index <= after:
                        raise Refused("delta_order")
                    yield index, pending[at + 4:at + record]
                    seen, after, at = seen + 1, index, at + record
                pending = pending[at:]
            if stream.unused_data:
                raise Refused("delta_trailing")
    except zlib.error:
        raise Refused("delta_stream") from None
    if not stream.eof or pending or seen != count:
        raise Refused("delta_stream")


def checked(objects, entry):
    """F10 step 2 for one part or object: present as a regular file, of its bytes, hashing to its key."""
    key = entry["key"]
    path = Path(objects, key)  # the key matched its grammar before the run began
    info = regular(path)
    if info is None:
        raise Refused("object_missing", key=key)
    if info.st_size != entry["bytes"]:
        raise Refused("object_bytes", key=key)
    if file_hash(path)[0] != key.split("/")[1].split(".")[0]:
        raise Refused("object_sha256", key=key)
    return path


def rebuild(block, objects, folder, generation):
    """F10 steps 2 to 4 into <folder>/lcm.db.partial, renamed to lcm.db only after step 4 holds."""
    size, base, partial = block["page_size"], block["base"], Path(folder, "lcm.db.partial")
    with open(partial, "xb+") as out:
        digest = hashlib.sha256()
        for part in base["parts"]:
            with open(checked(objects, part), "rb") as handle:
                for data in chunks(handle):
                    out.write(data)
                    digest.update(data)
        if out.tell() != base["file_bytes"] or digest.hexdigest() != base["file_sha256"]:
            raise Refused("base_digest_mismatch", generation=base["generation"])
        pages = base["file_bytes"] // size
        for entry in block["deltas"]:
            last, new = -1, 0
            for obj in entry["objects"]:
                with open(checked(objects, obj), "rb") as handle:
                    try:
                        for index, page in records(handle, size, obj["records"], last):
                            if index >= entry["file_pages"]:
                                raise Refused("delta_index")
                            out.seek(index * size)
                            out.write(page)
                            last, new = index, new + (index >= pages)
                    except Refused as refused:
                        refused.detail.setdefault("key", obj["key"])
                        raise
            if entry["file_pages"] > pages and new != entry["file_pages"] - pages:
                raise Refused("delta_gap", generation=entry["generation"])
            out.truncate(entry["file_pages"] * size)
            pages = entry["file_pages"]
    found = scan(partial, size)
    if found != (block["file_bytes"], size, block["file_sha256"], block["page_list_sha256"]):
        raise Refused("rebuild_digest_mismatch", generation=generation)
    os.rename(partial, Path(folder, "lcm.db"))


# -- the manifest: refused inputs and the database block's shape --

BLOCK = {"page_size": is_int, "file_bytes": is_int, "file_sha256": lambda v: matches(SHA, v),
         "page_list_sha256": lambda v: matches(SHA, v), "base": lambda v: isinstance(v, dict),
         "deltas": lambda v: isinstance(v, list)}
BASE = {"generation": is_int, "file_bytes": is_int, "file_sha256": BLOCK["file_sha256"], "parts": BLOCK["deltas"]}
PART = {"key": lambda v: isinstance(v, str), "bytes": is_int}
ENTRY = {"generation": is_int, "file_pages": is_int, "objects": BLOCK["deltas"]}
OBJECT = {**PART, "records": is_int}


def typed(obj, spec):
    return isinstance(obj, dict) and all(k in obj and ok(obj[k]) for k, ok in spec.items())


def shape(manifest):
    """F10 step 1: bad_database_block, chain_generation, or None."""
    block = manifest.get("database")
    if not (typed(block, BLOCK) and typed(block["base"], BASE) and all(typed(p, PART) for p in block["base"]["parts"])
            and all(typed(e, ENTRY) and all(typed(o, OBJECT) for o in e["objects"]) for e in block["deltas"])):
        return "bad_database_block"
    size, base, deltas = block["page_size"], block["base"], block["deltas"]
    if (size not in PAGE_SIZES or not base["file_bytes"] or base["file_bytes"] % size
            or sum(p["bytes"] for p in base["parts"]) != base["file_bytes"] or any(e["file_pages"] < 1 for e in deltas)
            or block["file_bytes"] != (deltas[-1]["file_pages"] * size if deltas else base["file_bytes"])
            or not deltas and block["file_sha256"] != base["file_sha256"]):
        return "bad_database_block"
    generation = manifest.get("generation")
    if (any(e["generation"] != base["generation"] + 1 + k for k, e in enumerate(deltas)) or not is_int(generation)
            or (deltas[-1]["generation"] if deltas else base["generation"]) != generation):
        return "chain_generation"
    return None


def bad_name(manifest):
    """The field of the first manifest name that fails its F1 grammar, or None. Containers of the wrong type are
    left to the checks; a part or object key that is not a string is left to the shape check."""
    if not matches(UUID, manifest.get("copy_id")):
        return "copy_id"
    attempt = manifest.get("attempt")
    if isinstance(attempt, dict) and "token" in attempt and not matches(TOKEN, attempt["token"]):
        return "attempt.token"
    for i, entry in enumerate(manifest["payloads"] if isinstance(manifest.get("payloads"), list) else []):
        if not (isinstance(entry, dict) and is_name(entry.get("name")) and matches(SHA, entry.get("sha256"))
                and matches(OBJECT_KEY["payloads"], entry.get("key"))):
            return f"payloads[{i}]"
    minted = manifest.get("unresolved_minted")
    for i, entry in enumerate(minted if isinstance(minted, list) else []):
        if not (isinstance(entry, dict) and is_name(entry.get("name")) and matches(MINTED_RE, entry["name"])):
            return f"unresolved_minted[{i}]"
    block = manifest.get("database") if isinstance(manifest.get("database"), dict) else {}
    base = block.get("base") if isinstance(block.get("base"), dict) else {}
    keys = [("parts", p) for p in (base["parts"] if isinstance(base.get("parts"), list) else [])]
    for entry in block.get("deltas", []) if isinstance(block.get("deltas"), list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("objects"), list):
            keys += [("deltas", o) for o in entry["objects"]]
    for kind, entry in keys:
        if isinstance(entry, dict) and isinstance(entry.get("key"), str) and not matches(OBJECT_KEY[kind], entry["key"]):
            return f"database.{kind}"
    return None


def admit(args):
    """The refused inputs, judged before any check runs; returns the parsed manifest."""
    if (args.objects is None) != (args.rebuild_dir is None):
        raise Refused("rebuild_needs_both")
    if args.rebuild_dir is not None:
        try:
            empty = stat.S_ISDIR(os.lstat(args.rebuild_dir).st_mode) and not os.listdir(args.rebuild_dir)
        except OSError:
            empty = False
        if not empty:
            raise Refused("rebuild_dir_not_empty")
    if os.path.lexists(Path(args.copy, "capture-owner.json")):
        raise Refused("copy_is_capture_home")
    if args.origin is not None:
        database = Path(args.origin, "lcm.db")
        try:
            with open(database, "rb") as handle:
                header = handle.read(20)[18:20]
        except OSError:
            header = b""
        if len(header) == 2 and max(header) >= 2 and not (
                Path(f"{database}-wal").exists() and Path(f"{database}-shm").exists()):
            raise Refused("origin_wal_without_sidecars")
    try:
        manifest = loads(Path(args.manifest).read_bytes())
    except (OSError, ValueError, RecursionError):
        raise Refused("manifest_unreadable") from None
    if not isinstance(manifest, dict):
        raise Refused("manifest_unreadable")
    if is_int(manifest.get("layout")) and manifest["layout"] > LAYOUT:
        raise Refused("newer_layout")
    contract = manifest.get("stamp", {}).get("contract") if isinstance(manifest.get("stamp"), dict) else None
    if isinstance(contract, dict) and is_int(contract.get("version")) and contract["version"] > CONTRACT_VERSION:
        raise Refused("replica_newer_contract")
    field = bad_name(manifest)
    if field is not None:
        raise Refused("bad_name", field=field)
    return manifest


# -- reading a database --

def connect(path, frozen):
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + ("?mode=ro&immutable=1" if frozen else "?mode=ro"),
                               uri=True)
    except sqlite3.Error:
        return None
    conn.text_factory = bytes  # text is compared as stored and decoded only where F3 says
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def tables_in(conn):
    return {r[0].decode() for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def meta(conn, key):
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def sequence(conn):
    if "sqlite_sequence" not in tables_in(conn):
        return None
    row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'messages'").fetchone()
    return None if row is None else row[0]


def show(key):
    values = [v.decode("utf-8", "backslashreplace") if isinstance(v, bytes) else v for v in key]
    return values[0] if len(values) == 1 else values


def ordered(keys):
    return sorted(keys, key=lambda k: [(type(v).__name__, v) for v in k])


def row_key(table, row, keys=None):
    names = LEDGER_KEYS.get(row["source_kind"], ("source_kind", "event_id")) if table == "capture_events" else KEYS[table] if keys is None else keys
    return tuple(row[n] for n in names)


def keyed(conn, table, columns, keys):
    """{key: Counter of row digests}; each value hashed with its SQLite type, so blobs are never held."""
    rows = defaultdict(Counter)
    listed = ", ".join(f'typeof("{c}"), "{c}"' for c in columns) or "1"  # no columns: one empty digest per row
    for values in conn.execute(f'SELECT {listed} FROM "{table}"'):
        digest = hashlib.sha256()
        for kind, value in zip(values[::2], values[1::2]):
            digest.update(kind + b":" + (b"%d:" % len(value) + value if isinstance(value, bytes) else repr(value).encode()))
        rows[row_key(table, dict(zip(columns, values[1::2])), keys)][digest.digest()] += 1
    return rows


def references(conn, home):
    """F3 over one database and home: ({resolved name: (sha256, bytes)}, {minted: sessions}, {quoted: sessions},
    store_ids whose text does not decode)."""
    found, bad = defaultdict(set), []
    for store_id, *values in conn.execute(
            "SELECT store_id, session_id, CASE WHEN typeof(content) = 'text' THEN content END, "
            "CASE WHEN typeof(tool_calls) = 'text' THEN tool_calls END FROM messages"):
        try:
            session, *texts = [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]
        except UnicodeDecodeError:
            bad.append(store_id)
            continue
        for text in filter(None, texts):
            for pattern in PATTERNS:
                for match in pattern.finditer(text):
                    found[match.group(1)].add(session)
    resolved, minted, quoted = {}, {}, {}
    for name, sessions in found.items():
        if is_name(name) and regular(Path(home, "lcm-large-outputs", name)):
            resolved[name] = file_hash(Path(home, "lcm-large-outputs", name))
        else:
            (minted if is_name(name) and MINTED_RE.fullmatch(name) else quoted)[name] = sorted(sessions)
    return resolved, minted, quoted, sorted(bad)


def undecodable(refs, side):
    return [{"code": "undecodable_text", "side": side, "store_ids": refs[3][:CAP], "count": len(refs[3])}] if refs[3] else []


def diff(mine, theirs, other):
    return {"only_in_copy": sorted(mine - theirs)[:CAP], "only_in_copy_count": len(mine - theirs),
            f"only_in_{other}": sorted(theirs - mine)[:CAP], f"only_in_{other}_count": len(theirs - mine)}


# -- the checks: each returns (failures, extra report fields) --

def rollback_journal(path):
    """Header bytes 18-19 are 1 and 1 (record section 3.3): immutable=1 then reads what any reader reads."""
    with open(path, "rb") as file:
        file.seek(18)
        return [] if file.read(2) == b"\x01\x01" else [{"code": "copy_journal_mode"}]


def file_digest(copy, block):
    path = Path(copy, "lcm.db")
    if regular(path) is None:
        return [{"code": "database_missing"}], {}
    found = dict(zip(("file_bytes", "page_size", "file_sha256", "page_list_sha256"), scan(path, block["page_size"])))
    return [{"code": field} for field, value in found.items() if value != block[field]] + rollback_journal(path), {}


def rebuilt(copy, block, objects, folder, generation):
    try:
        rebuild(block, objects, folder, generation)
    except Refused as refused:
        return [{"code": refused.code, **refused.detail}], {}
    size, failures = block["page_size"], rollback_journal(Path(folder, "lcm.db"))
    with open(Path(folder, "lcm.db"), "rb") as mine, open(Path(copy, "lcm.db"), "rb") as theirs:
        for page, (a, b) in enumerate(itertools.zip_longest(chunks(mine, size), chunks(theirs, size))):
            if a != b:
                return failures + [{"code": "rebuild_differs", "page": page}], {}
    return failures, {}


def integrity(conn):
    rows = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    return ([] if rows == [b"ok"] else [{"code": "integrity", "messages": len(rows)}]), {}


def payload_files(manifest, refs, copy):
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        return [{"code": "manifest_payloads"}], {}
    failures, listed = undecodable(refs, "copy"), set()
    for entry in payloads:  # every entry passed its grammar in admit()
        name = entry["name"]
        if name in listed:
            failures.append({"code": "payload_listed_twice", "name": name})
        listed.add(name)
        path = Path(copy, "lcm-large-outputs", name)
        if regular(path) is None:
            failures.append({"code": "payload_missing", "name": name})
        elif not (is_int(entry.get("bytes")) and file_hash(path) == (entry["sha256"], entry["bytes"])):
            failures.append({"code": "payload_differs", "name": name})
        if OBJECT_KEY["payloads"].fullmatch(entry["key"]).group(1) != entry["sha256"]:  # F2: the key names the file's digest
            failures.append({"code": "payload_key_sha256", "name": name})
    failures += [{"code": "payload_unlisted", "name": n} for n in sorted(refs[0].keys() - listed)]
    failures += [{"code": "payload_unreferenced", "name": n} for n in sorted(listed - refs[0].keys())]
    return failures, {"listed": len(listed), "resolved": len(refs[0])}


def unresolved(manifest, refs, origin_refs):
    sides, failures = {}, undecodable(refs, "copy")
    lists = [manifest.get("unresolved_minted"), manifest.get("unresolved_quoted")]
    if all(isinstance(x, list) and all(isinstance(e, dict) and isinstance(e.get("name"), str) for e in x) for x in lists):
        sides["manifest"] = [{e["name"] for e in x} for x in lists]
    else:
        failures.append({"code": "manifest_unresolved"})
    if origin_refs is not None:
        sides["origin"] = [set(origin_refs[1]), set(origin_refs[2])]
        failures += undecodable(origin_refs, "origin")
    for side, names in sides.items():
        for kind, mine, theirs in zip(("minted", "quoted"), (set(refs[1]), set(refs[2])), names):
            if mine != theirs:
                failures.append({"code": f"unresolved_{kind}", "against": side, **diff(mine, theirs, side)})
    counts = {side: {"minted": len(n[0]), "quoted": len(n[1])} for side, n in sides.items()}
    return failures, {"counts": {"copy": {"minted": len(refs[1]), "quoted": len(refs[2])}, **counts}}


def stamp_check(conn, manifest, owner_file):
    failures = [{"code": "store_uuid_row"}] if conn.execute(
        "SELECT 1 FROM metadata WHERE key = 'store_uuid'").fetchone() else []
    raw = meta(conn, STAMP_KEY)
    if raw is None:
        return failures + [{"code": "stamp_missing"}], {}
    try:
        stamp = loads(raw)
    except (ValueError, RecursionError):
        return failures + [{"code": "stamp_unreadable"}], {}
    if not same(stamp, manifest.get("stamp")):
        failures.append({"code": "stamp_manifest"})
    if not isinstance(stamp, dict):
        return failures + [{"code": "stamp_shape"}], {}
    for field in ("copy_id", "generation", "created_at"):
        if not same(stamp.get(field), manifest.get(field)):
            failures.append({"code": "stamp_field", "field": field})
    known_contract = any(same(stamp.get("contract"), {"version": version, "digest": digest})
                         for version, digest in CONTRACT_DIGESTS.items())
    if not same(stamp.get("format"), 1) or not known_contract:
        failures.append({"code": "stamp_format"})
    try:
        owner = loads(meta(conn, OWNER_KEY) or b"null")
    except (ValueError, RecursionError):
        owner = None
    owner = owner if isinstance(owner, dict) else {}
    if not (isinstance(stamp.get("origin_id"), str) and same(owner.get("store_uuid"), stamp["origin_id"])
            and same(owner.get("host"), stamp.get("origin_harness"))):
        failures.append({"code": "stamp_origin"})
    if owner_file is not None:
        try:
            given = loads(Path(owner_file).read_bytes())
        except (OSError, ValueError, RecursionError):
            given = None
        if not (isinstance(given, dict) and same(given.get("store_uuid"), stamp.get("origin_id"))):
            failures.append({"code": "stamp_owner_file"})
    version = meta(conn, "schema_version")
    present = tables_in(conn)
    facts = {"max_store_id": conn.execute("SELECT coalesce(max(store_id), 0) FROM messages").fetchone()[0],
             "schema_version": 0 if version is None else int(version) if re.fullmatch(rb"[0-9]+", version) else None,
             "projection_versions": sorted(r[0].decode("utf-8", "backslashreplace") for r in conn.execute(
                 "SELECT DISTINCT projection_version FROM capture_events")) if "capture_events" in present else []}
    failures += [{"code": "stamp_database", "field": f} for f, value in facts.items() if not same(stamp.get(f), value)]
    return failures, {}


def counts(conn, manifest):
    present, listed = tables_in(conn), manifest.get("tables")
    if not isinstance(listed, dict):
        return [{"code": "manifest_tables"}], {}
    actual = {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in COLUMNS if t in present}
    failures = [{"code": "table_count", "table": t} for t in sorted(actual.keys() | listed.keys())
                if not same(actual.get(t), listed.get(t))]
    if not same(manifest.get("sequence"), {"messages": sequence(conn)}):
        failures.append({"code": "sequence"})
    return failures, {"tables": actual}


def ledger(conn):
    if "capture_events" not in tables_in(conn):
        return [], {"counts": {"rows": 0}}
    failures, tally = [], Counter()
    columns = ["event_id", "source_kind", "lineage_key", "generation", "position", "uid", "content_digest",
               "line_digest", "disposition", "store_ids", "raw_from"]
    for values in conn.execute(f"SELECT {', '.join(columns)}, raw IS NOT NULL FROM capture_events ORDER BY event_id"):
        row, tally["rows"] = dict(zip(columns, values)), tally["rows"] + 1
        key = show(row_key("capture_events", row))
        held_missing = False
        if row["disposition"] == b"held":
            try:
                ids = [i for i in loads(row["store_ids"] or b"[]") if is_int(i)]
            except (ValueError, RecursionError, TypeError):
                ids = []
            held_missing = any(conn.execute("SELECT 1 FROM messages WHERE store_id = ?", (i,)).fetchone() is None
                               for i in ids)
            tally["held_missing_rows"] += held_missing
        source = row["event_id"] if values[-1] else row["raw_from"]
        blob = None if source is None else conn.execute("SELECT raw FROM capture_events WHERE event_id = ?",
                                                         (source,)).fetchone()
        if source is not None and blob is None:
            failures.append({"code": "raw_from_missing", "key": key})
        elif blob is None or blob[0] is None:
            tally["no_bytes"] += not held_missing
        else:
            stream, digest = zlib.decompressobj(), hashlib.sha256()
            try:
                for out in inflate(stream, blob[0]):
                    digest.update(out)
            except zlib.error:
                stream = None
            if stream is None or not stream.eof or stream.unused_data:
                failures.append({"code": "raw_undecodable", "key": key})
            elif digest.hexdigest().encode() != row["line_digest"]:
                failures.append({"code": "raw_digest", "key": key})
    return failures, {"counts": {"rows": 0, **tally}}


def tables(conn, origin):
    """F6 by key and every contracted column, read by name; the one exemption is the copy's import:origin row."""
    mine, theirs, failures, detail = tables_in(conn), tables_in(origin), [], {}
    for table, columns in COLUMNS.items():
        if (table in mine) != (table in theirs):
            failures.append({"code": "table_presence", "table": table})
        if table not in mine or table not in theirs:
            continue
        have = [{r[1].decode() for r in c.execute(f"PRAGMA table_info(\"{table}\")")} & set(columns) for c in (conn, origin)]
        common = [c for c in columns if c in have[0] & have[1]]
        short = table == "capture_events" and not {"source_kind", "event_id"} <= set(common)
        if have[0] != have[1] or short:  # one columns failure per table
            failures.append({"code": "columns", "table": table})
        if short:
            continue
        named = [c for c in KEYS[table] if c in common]  # F6 allows a contracted column absent on both sides
        keys = named or common
        copy_rows, origin_rows = keyed(conn, table, common, keys), keyed(origin, table, common, keys)
        if table == "metadata" and (STAMP_KEY.encode(),) not in origin_rows:
            copy_rows.pop((STAMP_KEY.encode(),), None)
        found = {"only_in_origin": origin_rows.keys() - copy_rows.keys(), "only_in_copy": copy_rows.keys() - origin_rows.keys(),
                 "changed": {k for k in origin_rows.keys() & copy_rows.keys() if origin_rows[k] != copy_rows[k]}}
        for code, keys in found.items():
            if keys:
                failures.append({"code": code, "table": table, "count": len(keys)})
                if named:  # a whole row is no key: codes and counts only, never stored text
                    detail.setdefault(table, {}).update({code: [show(k) for k in ordered(keys)[:CAP]], f"{code}_count": len(keys)})
    if sequence(conn) != sequence(origin):
        failures.append({"code": "sequence"})
    return failures, {"tables": detail}


def payloads(refs, origin_refs):
    mine, theirs = refs[0], origin_refs[0]
    failures = [{"code": "payload_only_in_copy", "name": n} for n in sorted(mine.keys() - theirs.keys())]
    failures += [{"code": "payload_only_in_origin", "name": n} for n in sorted(theirs.keys() - mine.keys())]
    failures += [{"code": "payload_sha256", "name": n} for n in sorted(mine.keys() & theirs.keys()) if mine[n] != theirs[n]]
    return failures + undecodable(origin_refs, "origin"), {}


def run(check, *args):
    try:
        failures, extra = check(*args)
    except Exception as error:  # a check that cannot finish fails by name; no stored text reaches the report
        failures, extra = [{"code": "error", "error": type(error).__name__}], {}
    return {"ran": True, "pass": not failures, "codes": sorted({f["code"] for f in failures}),
            "failures": failures[:CAP], "failure_count": len(failures), **extra}


def check(args, manifest):
    block, generation, copy = manifest.get("database"), manifest.get("generation"), Path(args.copy)
    code, checks = shape(manifest), {}
    checks["chain"] = run(lambda: ([{"code": code}] if code else [], {}))
    skipped = {"ran": False, "reason": "chain"}
    checks["file_digest"] = run(file_digest, copy, block) if code is None else skipped
    checks["rebuild"] = {"ran": False} if args.objects is None else skipped if code is not None else run(
        rebuilt, copy, block, args.objects, args.rebuild_dir, generation)
    use_rebuilt = checks["rebuild"].get("pass") is True
    conn = connect(Path(args.rebuild_dir, "lcm.db") if use_rebuilt else copy / "lcm.db", True)
    origin = None if args.origin is None else connect(Path(args.origin, "lcm.db"), False)
    cache = {}

    def refs(side):
        if side not in cache:
            cache[side] = references(*((conn, copy) if side == "copy" else (origin, args.origin)))
        return cache[side]

    try:
        checks["integrity"] = run(integrity, conn)
        checks["payload_files"] = run(lambda: payload_files(manifest, refs("copy"), copy))
        checks["unresolved"] = run(lambda: unresolved(manifest, refs("copy"), None if args.origin is None else refs("origin")))
        checks["stamp"] = run(stamp_check, conn, manifest, args.owner_file)
        checks["counts"] = run(counts, conn, manifest)
        checks["ledger"] = run(ledger, conn)
        if args.origin is not None:
            checks["tables"] = run(tables, conn, origin)
            checks["payloads"] = run(lambda: payloads(refs("copy"), refs("origin")))
    finally:
        for opened in (conn, origin):
            if opened is not None:
                opened.close()
    passed = all(c["pass"] for c in checks.values() if c["ran"]) and checks["file_digest"]["ran"]
    return {"tool": "cloud_copy_outside_in", "format": 1, "mode": "manifest" if args.origin is None else "origin",
            "copy_id": manifest.get("copy_id"), "generation": generation if is_int(generation) else None,
            "database_read": "rebuilt" if use_rebuilt else "copy", "pass": passed, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copy", required=True, help="a copy home: lcm.db and lcm-large-outputs/")
    parser.add_argument("--manifest", required=True, help="the copy's manifest, gen/<generation>.json")
    parser.add_argument("--origin", help="the origin home as of the copy's cut")
    parser.add_argument("--owner-file", help="the origin machine's capture-owner.json")
    parser.add_argument("--objects", help="a folder holding the manifest's parts/ and deltas/ keys as files")
    parser.add_argument("--rebuild-dir", help="an existing, empty folder for the rebuilt lcm.db")
    args = parser.parse_args(argv)
    try:
        report, status = check(args, admit(args)), None
    except Refused as refused:
        report, status = {"tool": "cloud_copy_outside_in", "format": 1, "pass": False, "refused": refused.code,
                          **refused.detail}, 2
    print(json.dumps(report, sort_keys=True))
    return status if status is not None else 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
