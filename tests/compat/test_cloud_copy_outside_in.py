"""Synthetic proofs of the independent copy checker (scripts/cloud_copy_outside_in.py).

The clean copy is three generations of one origin store, made here by this file's own encoder from hosted copy
format 1 alone: a base cut into parts, then two deltas of whole pages. Every other test is one mutation of it.
"""
from __future__ import annotations

import ast
import base64
from contextlib import closing
import copy as deepcopy
import functools
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tracemalloc
from types import SimpleNamespace
import zlib

import pytest

from hermes_lcm import claude_code_store as cc
from hermes_lcm import externalize, ingest_content
from hermes_lcm.config import LCMConfig
from tests.test_claude_code_store import _entry, _transcript

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/cloud_copy_outside_in.py"
PLANT = "PLANTED-7c1d-stored-text"
COPY_ID = "0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0"
LOST = "20251101_120000_lost_0123456789ab_18e0.json"  # a minted name whose file exists nowhere
QUOTED = "[Externalized payload: example; ref=...]"
PART_SIZE = 32768
PER_OBJECT = 4  # records per delta object, so one entry holds several objects
BLOB = base64.b64encode(bytes(range(256)) * 16).decode()
RUNS = itertools.count()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=None)
def contract():
    """docs/store-contract.md: version, digest, and each table's column names and primary key, parsed here."""
    text = (ROOT / "docs/store-contract.md").read_text(encoding="utf-8")
    generated = text.split("<!-- Generated from a fresh store")[1].split("<!-- End generated table. -->")[0]
    tables = {}
    for name, columns in re.findall(r"^\| `(\w+)` \| (.+?) \| [^|]* \|$", generated, re.M):
        tokens = columns.split("; ")
        assert len(tokens) % 4 == 0, name
        names = [token.split(":")[0] for token in tokens[0::4]]
        positions = [int(token.removeprefix("pk=")) for token in tokens[3::4]]
        tables[name] = (names, [n for p, n in sorted(zip(positions, names)) if p])
    return (int(re.search(r"^Contract version: (\d+)$", text, re.M).group(1)),
            re.search(r"^Digest: (sha256:[0-9a-f]{64})$", text, re.M).group(1), tables)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("cloud_copy_outside_in", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the origin store, built as the capture writer's tests build it --

def make_origin(base, sessions, lineage=True):
    home = base / "home"
    origin = SimpleNamespace(base=base, home=home, database=cc.admit(home, hermes_home=base / "hermes"), paths={})
    feed(origin, sessions)
    if lineage:
        store = cc.open_store(origin.database, home)
        try:
            store.record_session_lineage("s2", "s1", source="test")
        finally:
            store.close()
    return origin


def feed(origin, sessions):
    config = LCMConfig(database_path=str(origin.database), embeddings_enabled=False,
                       large_output_externalization_enabled=True, large_output_externalization_threshold_chars=200)
    store = cc.open_store(origin.database, origin.home, config)
    try:
        for session, entries in sessions.items():
            if session not in origin.paths:
                origin.paths[session] = _transcript(origin.base, [], session=session)
            with origin.paths[session].open("a", encoding="utf-8") as handle:
                handle.write("".join(json.dumps(entry) + "\n" for entry in entries))
            assert cc.drain(store, cc.Lineage(session, None, origin.paths[session])) == "drained"
    finally:
        store.close()


FIRST = {"s1": [
    _entry("user", "u1", f"remember {PLANT}; quoted {QUOTED} and lost [Externalized tool output: gone; ref={LOST}]"),
    _entry("assistant", "a1", [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "notes.txt"}}]),
    _entry("user", "r1", [{"type": "tool_result", "tool_use_id": "t1", "content": f"{PLANT} " + "z" * 600}]),
], "s2": [_entry("user", "v1", "attached\n" + BLOB + "\nend", session="s2")]}


def more(gen):
    return {"s1": [_entry("assistant", f"g{gen}a{k}", f"generation {gen} reply {k} " + "w" * 1500) for k in range(3)],
            "s2": [_entry("user", f"g{gen}b", f"generation {gen} note", session="s2")]}


# -- this file's own encoder of hosted copy format 1 --

def page_size(data):
    value = int.from_bytes(data[16:18], "big")
    return 65536 if value == 1 else value


def pages(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]


def page_list(data, size):
    return sha(b"".join(hashlib.sha256(page).digest() for page in pages(data, size)))


def put(objects, kind, blob, gen):
    key = f"{kind}/{sha(blob)}.{gen:032x}"
    (objects / key).parent.mkdir(parents=True, exist_ok=True)
    (objects / key).write_bytes(blob)
    return key


def delta_object(size, records, count=None, compressed=None):
    body = b"".join(struct.pack(">I", index) + page for index, page in records)
    stream = zlib.compress(body, 6) if compressed is None else compressed
    return b"LCMXPD1\n" + struct.pack(">II", size, len(records) if count is None else count) + stream


def base_block(data, gen, objects, part_size=PART_SIZE):
    size = page_size(data)
    parts = [{"key": put(objects, "parts", data[i:i + part_size], gen), "bytes": len(data[i:i + part_size])}
             for i in range(0, len(data), part_size)]
    return {"page_size": size, "file_bytes": len(data), "file_sha256": sha(data), "page_list_sha256": page_list(data, size),
            "base": {"generation": gen, "file_bytes": len(data), "file_sha256": sha(data), "reason": "first",
                     "part_size": part_size, "parts": parts}, "deltas": []}


def delta_block(data, gen, head, objects):
    size, before = page_size(data), pages(head.data, page_size(head.data))
    changed = [(i, page) for i, page in enumerate(pages(data, size)) if i >= len(before) or before[i] != page]
    entry = {"generation": gen, "file_pages": len(data) // size, "objects": []}
    for k in range(0, len(changed), PER_OBJECT):
        blob = delta_object(size, changed[k:k + PER_OBJECT])
        entry["objects"].append({"key": put(objects, "deltas", blob, gen), "bytes": len(blob),
                                 "records": len(changed[k:k + PER_OBJECT])})
    block = deepcopy.deepcopy(head.block)
    block.update(file_bytes=len(data), file_sha256=sha(data), page_list_sha256=page_list(data, size))
    block["deltas"].append(entry)
    return block


def backup(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as live, closing(sqlite3.connect(target)) as saved:
        live.backup(saved)
        assert saved.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"


def references(database):
    found = {}
    with closing(sqlite3.connect(database)) as conn:
        for session, *texts in conn.execute("SELECT session_id, content, tool_calls FROM messages"):
            for text in texts:
                for pattern in (externalize._EXTERNALIZED_REF_RE, ingest_content._INGEST_PLACEHOLDER_RE):
                    for match in pattern.finditer(text) if isinstance(text, str) else ():
                        found.setdefault(match.group(1), set()).add(session)
    return found


def cut(origin, out, gen, head=None):
    """One generation: the origin saved as of its cut, its stamped copy home, its objects and its manifest."""
    saved, home, objects = out / f"origin-{gen}", out / f"gen-{gen}", out / "objects"
    backup(origin.database, saved / "lcm.db")
    shutil.copy2(origin.home / "capture-owner.json", saved / "capture-owner.json")
    (saved / "lcm-large-outputs").mkdir()
    if (origin.home / "lcm-large-outputs").is_dir():
        shutil.copytree(origin.home / "lcm-large-outputs", saved / "lcm-large-outputs", dirs_exist_ok=True)
    (home / "lcm-large-outputs").mkdir(parents=True)
    shutil.copyfile(saved / "lcm.db", home / "lcm.db")
    found = references(home / "lcm.db")
    resolved = sorted(n for n in found if (saved / "lcm-large-outputs" / n).is_file())
    for name in resolved:
        shutil.copyfile(saved / "lcm-large-outputs" / name, home / "lcm-large-outputs" / name)
    version, digest, tables = contract()
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        meta = dict(conn.execute("SELECT key, value FROM metadata"))
        owner = json.loads(meta["capture:owner"])
        stamp = {"format": 1, "origin_id": owner["store_uuid"], "copy_id": COPY_ID, "origin_harness": owner["host"],
                 "generation": gen, "parent": None if head is None else {
                     "generation": head.gen, "manifest_sha256": sha(head.manifest.read_bytes())},
                 "contract": {"version": version, "digest": digest}, "schema_version": int(meta.get("schema_version", 0)),
                 "projection_versions": sorted(r[0] for r in conn.execute(
                     "SELECT DISTINCT projection_version FROM capture_events")),
                 "unit": {"version": None, "core_commit": None, "core_version": None},
                 "max_store_id": conn.execute("SELECT coalesce(max(store_id), 0) FROM messages").fetchone()[0],
                 "created_at": 1800000000.5 + gen}
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("import:origin", json.dumps(
            stamp, sort_keys=True, separators=(",", ":"), ensure_ascii=False)))
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        counts = {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables if t in present}
        sequence = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'messages'").fetchone()
    data = (home / "lcm.db").read_bytes()
    block = base_block(data, gen, objects) if head is None else delta_block(data, gen, head, objects)
    unresolved = [n for n in found if n not in resolved]
    minted = sorted(n for n in unresolved if externalize._GENERATED_PAYLOAD_REF_RE.fullmatch(n))
    manifest = {
        "layout": 1, "copy_id": COPY_ID, "generation": gen, "created_at": stamp["created_at"], "stamp": stamp,
        "database": block, "payloads": [
            {"name": n, "sha256": sha((home / "lcm-large-outputs" / n).read_bytes()),
             "bytes": (home / "lcm-large-outputs" / n).stat().st_size,
             "key": put(objects, "payloads", (home / "lcm-large-outputs" / n).read_bytes(), gen)} for n in resolved],
        "unresolved_minted": [{"name": n, "sessions": sorted(found[n])} for n in minted],
        "unresolved_quoted": [{"name": n, "sessions": sorted(found[n])} for n in sorted(unresolved) if n not in minted],
        "tables": counts, "sequence": {"messages": sequence[0] if sequence else None},
        "attempt": {"generation": gen, "token": f"{gen:032x}"}, "publisher": None, "pins": [], "superseded": [],
        "forgotten": [], "guard": {"never": "read"}}
    (out / f"manifest-{gen}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return SimpleNamespace(gen=gen, copy=home, origin=saved, manifest=out / f"manifest-{gen}.json", data=data,
                           block=block)


def build_chain(base, sessions=FIRST, steps=(2, 3), between=None):
    origin = make_origin(base, sessions, lineage=sessions is FIRST)
    if between:
        between(origin, 1)
    gens = [cut(origin, base, 1)]
    for gen in steps:
        if between:
            between(origin, gen)
        else:
            feed(origin, more(gen))
        gens.append(cut(origin, base, gen, gens[-1]))
    return SimpleNamespace(base=base, gens=gens)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return build_chain(tmp_path_factory.mktemp("chain"))


def padding(origin, gen):
    """A store that shrinks at generation 2 (DELETE then VACUUM) and grows past generation 1 at 3."""
    feed(origin, more(gen)) if gen > 1 else None
    with closing(sqlite3.connect(origin.database)) as conn:
        with conn:
            conn.execute("DELETE FROM metadata WHERE key = 'pad:test'")
            if gen != 2:
                conn.execute("INSERT INTO metadata(key, value) VALUES ('pad:test', ?)", ("p" * 150000 * gen,))
        conn.execute("VACUUM")


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    return build_chain(tmp_path_factory.mktemp("wave"), between=padding)


def clone(chain, target, gen=3):
    g = chain.gens[gen - 1]
    shutil.copytree(chain.base / "objects", target / "objects")
    shutil.copytree(g.copy, target / "copy")
    shutil.copytree(g.origin, target / "origin")
    shutil.copy2(g.manifest, target / "manifest.json")
    return SimpleNamespace(root=target, copy=target / "copy", origin=target / "origin", objects=target / "objects",
                           manifest=target / "manifest.json", gen=g)


# -- running the checker --

def cli(*args):
    done = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, timeout=120)
    assert PLANT not in done.stdout and PLANT not in done.stderr  # test 18, on every run of every test
    return done.returncode, json.loads(done.stdout) if done.stdout.strip() else None


def check(c, mode="origin", objects=True, extra=()):
    args = ["--copy", c.copy, "--manifest", c.manifest, *extra]
    if mode == "origin":
        args += ["--origin", c.origin]
    if objects:
        folder = c.root / f"rebuild-{next(RUNS)}"
        folder.mkdir()
        args += ["--objects", c.objects if objects is True else objects, "--rebuild-dir", folder]
    return cli(*args)


def codes(report, name):
    return report["checks"][name]["codes"]


def failed(report, name):
    return report["checks"][name]["ran"] and not report["checks"][name]["pass"]


def sql(database, *statements):
    with closing(sqlite3.connect(database)) as conn, conn:
        for statement in statements:
            conn.execute(*statement) if isinstance(statement, tuple) else conn.execute(statement)


def query(database, statement, *args):
    with closing(sqlite3.connect(f"{Path(database).as_uri()}?mode=ro", uri=True)) as conn:
        return conn.execute(statement, args).fetchall()


def edit(c, change):
    manifest = json.loads(c.manifest.read_text(encoding="utf-8"))
    change(manifest)
    c.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def plain_row(c, gen=2):
    return query(c.copy / "lcm.db", "SELECT store_id FROM messages WHERE content LIKE ?", f"generation {gen} reply 0%")[0][0]


def tree(*roots):
    return {(str(root), str(p.relative_to(root)), p.lstat().st_size, p.lstat().st_mtime_ns)
            for root in roots for p in Path(root).rglob("*")}


def reencode(c, entry_index, object_index, change):
    """Replace one delta object by change(records) re-encoded, with its key, bytes and records in the manifest."""
    manifest = json.loads(c.manifest.read_text(encoding="utf-8"))
    size, entry = manifest["database"]["page_size"], manifest["database"]["deltas"][entry_index]
    obj = entry["objects"][object_index]
    body = zlib.decompress((c.objects / obj["key"]).read_bytes()[16:])
    records = [(int.from_bytes(body[i:i + 4], "big"), body[i + 4:i + 4 + size]) for i in range(0, len(body), 4 + size)]
    records = change(records, entry)
    blob = delta_object(size, records)
    obj.update(key=put(c.objects, "deltas", blob, entry["generation"]), bytes=len(blob), records=len(records))
    c.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# -- 2: the checker must first be seen to see --

def test_order_of_proof(chain, tmp_path):
    c = clone(chain, tmp_path)
    gone, changed = plain_row(c, 2), plain_row(c, 3)
    sql(c.copy / "lcm.db", ("DELETE FROM messages WHERE store_id = ?", (gone,)),
        ("UPDATE messages SET content = content || 'x' WHERE store_id = ?", (changed,)))
    code, report = check(c)
    assert code == 1 and report["pass"] is False
    detail = report["checks"]["tables"]["tables"]["messages"]
    assert detail["only_in_origin"] == [gone] and detail["changed"] == [changed]


# -- 1 --

def test_clean_copy(chain, tmp_path):
    for g in chain.gens:
        c = clone(chain, tmp_path / str(g.gen), g.gen)
        for mode in ("origin", "manifest"):
            for objects in (True, False):
                code, report = check(c, mode, objects)
                assert (code, report["pass"], report["mode"]) == (0, True, mode), (g.gen, mode, objects, report)
                assert report["database_read"] == ("rebuilt" if objects else "copy")
                assert report["checks"]["rebuild"]["ran"] is objects
                counts = report["checks"]["unresolved"]["counts"]
                assert counts["copy"] == counts["manifest"] == {"minted": 1, "quoted": 1}
                assert ("origin" in counts) is (mode == "origin")


# -- 3 to 8: the five seeded faults and the undecodable blob --

def test_row_removed(chain, tmp_path):
    c = clone(chain, tmp_path)
    row = plain_row(c)
    sql(c.copy / "lcm.db", ("DELETE FROM messages WHERE store_id = ?", (row,)))
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["tables"]["messages"]["only_in_origin"] == [row]
    code, report = check(c, "manifest")
    assert code == 1 and "table_count" in codes(report, "counts") and "file_sha256" in codes(report, "file_digest")


def test_row_doubled(chain, tmp_path):
    c = clone(chain, tmp_path)
    row = plain_row(c)
    columns = [n for n in contract()[2]["messages"][0] if n != "store_id"]
    listed = ", ".join(columns)
    database = c.copy / "lcm.db"
    sql(database, (f"INSERT INTO messages({listed}) SELECT {listed} FROM messages WHERE store_id = ?", (row,)))
    double = query(database, "SELECT max(store_id) FROM messages")[0][0]
    bindings = query(database, "SELECT uid, lineage_key, kind FROM host_uid_bindings WHERE store_id = ?", row)
    assert double > row and bindings
    rest = "uid, lineage_key, kind, binding_version, proof_kind, created_at, first_check, disagree_seen"
    # The unique canonical index refuses a second canonical binding; a doubled copy is built without it.
    sql(database, "DROP INDEX idx_host_uid_canonical", (
        f"INSERT INTO host_uid_bindings(store_id, {rest}) SELECT ?, {rest} FROM host_uid_bindings WHERE store_id = ?",
        (double, row)))
    code, report = check(c)
    tables = report["checks"]["tables"]["tables"]
    assert code == 1 and tables["messages"]["only_in_copy"] == [double]
    assert sorted(tables["host_uid_bindings"]["only_in_copy"]) == sorted([double, *b] for b in bindings)
    code, report = check(c, "manifest")
    assert code == 1 and "table_count" in codes(report, "counts") and "file_sha256" in codes(report, "file_digest")


def test_content_changed(chain, tmp_path):
    c = clone(chain, tmp_path)
    row = plain_row(c)
    sql(c.copy / "lcm.db", ("UPDATE messages SET content = content || '!' WHERE store_id = ?", (row,)))
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["tables"]["messages"]["changed"] == [row]
    code, report = check(c, "manifest")
    assert code == 1 and "file_sha256" in codes(report, "file_digest") and not failed(report, "counts")


def test_payload_removed(chain, tmp_path):
    c = clone(chain, tmp_path)
    name = sorted(p.name for p in (c.copy / "lcm-large-outputs").iterdir())[0]
    shutil.move(c.copy / "lcm-large-outputs" / name, c.root / "moved")
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)
        assert code == 1 and {"code": "payload_missing", "name": name} in report["checks"]["payload_files"]["failures"]
    code, report = check(c)
    assert {"code": "payload_only_in_origin", "name": name} in report["checks"]["payloads"]["failures"]


@pytest.mark.parametrize("case", ["other_key", "keys_swapped"])
def test_payload_key_sha256(chain, tmp_path, case):
    """A payload entry's key names its file's digest (F2, F5): a key that names other content fails payload_files."""
    c = clone(chain, tmp_path)
    assert check(c, "manifest", objects=False)[0] == 0
    entries = json.loads(c.manifest.read_text(encoding="utf-8"))["payloads"]
    assert len(entries) >= 2 and entries[0]["sha256"] != entries[1]["sha256"]
    if case == "other_key":
        edit(c, lambda m: m["payloads"][0].update(key=f"payloads/{sha(b'other content')}.{99:032x}"))
        names = [entries[0]["name"]]
    else:
        edit(c, lambda m: m["payloads"][0].update(key=entries[1]["key"]) or m["payloads"][1].update(key=entries[0]["key"]))
        names = [entries[0]["name"], entries[1]["name"]]
    for mode, objects in itertools.product(("origin", "manifest"), (False, True)):
        code, report = check(c, mode, objects)
        failing = [n for n, done in report["checks"].items() if done["ran"] and not done["pass"]]
        assert (code, failing) == (1, ["payload_files"]), (mode, objects, report)
        assert report["checks"]["payload_files"]["failures"] == [
            {"code": "payload_key_sha256", "name": n} for n in names], (mode, objects, report)


def ledger_row(database):
    return query(database, "SELECT event_id, lineage_key, generation, position, raw FROM capture_events "
                           "WHERE source_kind = 'line' AND raw IS NOT NULL ORDER BY event_id LIMIT 1")[0]


def test_raw_ledger_blob_cut(chain, tmp_path):
    c = clone(chain, tmp_path)
    event, *key, raw = ledger_row(c.copy / "lcm.db")
    line = zlib.decompress(raw)
    sql(c.copy / "lcm.db", ("UPDATE capture_events SET raw = ? WHERE event_id = ?",
                             (zlib.compress(line[:len(line) // 2]), event)))
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)
        assert code == 1 and {"code": "raw_digest", "key": key} in report["checks"]["ledger"]["failures"]
    assert report["checks"]["ledger"]["counts"]["rows"] > 0
    code, report = check(c)
    assert report["checks"]["tables"]["tables"]["capture_events"]["changed"] == [key]


def test_raw_blob_undecodable(chain, tmp_path):
    c = clone(chain, tmp_path)
    event, *key, raw = ledger_row(c.copy / "lcm.db")
    sql(c.copy / "lcm.db", ("UPDATE capture_events SET raw = ? WHERE event_id = ?", (raw[:len(raw) // 2], event)))
    code, report = check(c)
    assert code == 1 and {"code": "raw_undecodable", "key": key} in report["checks"]["ledger"]["failures"]


# -- 9 to 11: references that name no file --

def test_unresolved_same_on_both_sides(chain, tmp_path):
    c = clone(chain, tmp_path)
    code, report = check(c)
    assert code == 0 and report["checks"]["unresolved"]["pass"]
    assert report["checks"]["unresolved"]["counts"] == {side: {"minted": 1, "quoted": 1}
                                                        for side in ("copy", "manifest", "origin")}


def test_unresolved_one_side_only(chain, tmp_path):
    c = clone(chain, tmp_path / "quoted")
    row = plain_row(c)
    sql(c.copy / "lcm.db", ("UPDATE messages SET content = content || ? WHERE store_id = ?",
                             (" [Externalized payload: q; ref={ref}]", row)))
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)
        assert code == 1 and "{ref}" in json.dumps(report["checks"]["unresolved"]["failures"])
    c = clone(chain, tmp_path / "minted")
    (c.origin / "lcm-large-outputs" / LOST).write_text("{}", encoding="utf-8")
    code, report = check(c)
    unresolved = report["checks"]["unresolved"]
    assert code == 1 and LOST in json.dumps(unresolved["failures"]) and unresolved["counts"]["origin"]["minted"] == 0
    assert {"code": "payload_only_in_origin", "name": LOST} in report["checks"]["payloads"]["failures"]


def test_only_quoted_placeholders(tmp_path):
    quiet = build_chain(tmp_path / "quiet", {"q1": [_entry("user", "u1", f"see {QUOTED}", session="q1")]}, steps=())
    c = clone(quiet, tmp_path / "c", 1)
    assert not list((c.copy / "lcm-large-outputs").iterdir())
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)
        assert code == 0 and report["checks"]["unresolved"]["counts"]["copy"] == {"minted": 0, "quoted": 1}


# -- 12 and 13: the stamp and its exemption --

def test_stamp(chain, tmp_path):
    def stamp(c):
        return json.loads(query(c.copy / "lcm.db", "SELECT value FROM metadata WHERE key = 'import:origin'")[0][0])

    c = clone(chain, tmp_path / "owner")
    code, report = check(c, extra=("--owner-file", c.origin / "capture-owner.json"))
    assert code == 0 and report["checks"]["stamp"]["pass"]
    other = c.root / "other-owner.json"
    other.write_text(json.dumps({"store_uuid": "11111111-2222-4333-8444-555555555555"}), encoding="utf-8")
    code, report = check(c, extra=("--owner-file", other))
    assert code == 1 and codes(report, "stamp") == ["stamp_owner_file"]
    c = clone(chain, tmp_path / "missing")
    sql(c.copy / "lcm.db", "DELETE FROM metadata WHERE key = 'import:origin'")
    assert "stamp_missing" in codes(check(c)[1], "stamp")
    c = clone(chain, tmp_path / "field")
    changed = {**stamp(c), "origin_harness": "other-harness"}
    sql(c.copy / "lcm.db", ("UPDATE metadata SET value = ? WHERE key = 'import:origin'", (json.dumps(changed),)))
    assert "stamp_manifest" in codes(check(c, "manifest")[1], "stamp")
    c = clone(chain, tmp_path / "uuid")
    sql(c.copy / "lcm.db", "INSERT INTO metadata(key, value) VALUES ('store_uuid', 'x')")
    assert "store_uuid_row" in codes(check(c, "manifest")[1], "stamp")


def test_stamp_shape(chain, tmp_path):
    """A stamp that is not a JSON object fails, even when the copy and its manifest agree on it."""
    for name, value in (("null", None), ("list", [1, "two"])):
        c = clone(chain, tmp_path / name)
        database = c.copy / "lcm.db"
        sql(database, ("UPDATE metadata SET value = ? WHERE key = 'import:origin'", (json.dumps(value),)))
        manifest = json.loads(c.manifest.read_text(encoding="utf-8"))
        manifest.update(stamp=value, database=base_block(database.read_bytes(), manifest["generation"], c.objects))
        c.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        for mode in ("origin", "manifest"):
            code, report = check(c, mode)
            failing = [n for n, done in report["checks"].items() if done["ran"] and not done["pass"]]
            assert (code, failing, codes(report, "stamp")) == (1, ["stamp"], ["stamp_shape"]), (name, mode, report)


def test_exemption_is_exact(chain, tmp_path):
    c = clone(chain, tmp_path)
    sql(c.copy / "lcm.db", "INSERT INTO metadata(key, value) VALUES ('import:other', '1')")
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["tables"]["metadata"]["only_in_copy"] == ["import:other"]


# -- 14 and 15: columns by name, optional tables --

def test_columns_by_name(chain, tmp_path):
    c = clone(chain, tmp_path)
    database = c.copy / "lcm.db"
    before = query(database, "SELECT * FROM session_lineage ORDER BY child_session_id")
    assert before
    reordered = "source, created_at, child_role, parent_subagent_id, child_subagent_id, parent_session_id, child_session_id"
    sql(database, "ALTER TABLE session_lineage RENAME TO session_lineage_old",
        "CREATE TABLE session_lineage (source TEXT NOT NULL, created_at REAL, child_role TEXT, parent_subagent_id TEXT, "
        "child_subagent_id TEXT, parent_session_id TEXT NOT NULL, child_session_id TEXT PRIMARY KEY)",
        f"INSERT INTO session_lineage({reordered}) SELECT {reordered} FROM session_lineage_old",
        "DROP TABLE session_lineage_old",
        "CREATE INDEX idx_session_lineage_parent ON session_lineage(parent_session_id)")
    assert query(database, "SELECT name FROM pragma_table_info('session_lineage') WHERE cid = 0") == [("source",)]
    data = database.read_bytes()
    edit(c, lambda m: m.update(database=base_block(data, m["generation"], c.objects)))
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)
        assert code == 0 and report["database_read"] == "rebuilt", report


def test_optional_tables(chain, tmp_path):
    c = clone(chain, tmp_path)
    assert "ledger_meta" not in json.loads(c.manifest.read_text())["tables"]
    code, report = check(c)
    assert code == 0 and report["checks"]["tables"]["pass"]
    sql(c.copy / "lcm.db", "CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
    code, report = check(c)
    assert code == 1 and {"code": "table_presence", "table": "ledger_meta"} in report["checks"]["tables"]["failures"]


def test_key_column_absent_on_both_sides(chain, tmp_path):
    """stamp_observations without its contracted column source on both sides (F6 allows it) is still compared."""
    c = clone(chain, tmp_path)
    table = "CREATE TABLE stamp_observations (lineage TEXT, occ_id INTEGER, stamp TEXT, kind TEXT, view_seq INTEGER)"
    row = "INSERT INTO stamp_observations VALUES ('lineage-1', 1, 'stamp-1', 'seen', 1)"
    for database in (c.copy / "lcm.db", c.origin / "lcm.db"):
        sql(database, table, row)
    manifest = json.loads(c.manifest.read_text(encoding="utf-8"))
    manifest["tables"]["stamp_observations"] = 1
    manifest["database"] = base_block((c.copy / "lcm.db").read_bytes(), manifest["generation"], c.objects)
    c.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    code, report = check(c)
    assert code == 0 and report["checks"]["tables"]["pass"], report
    sql(c.origin / "lcm.db", "INSERT INTO stamp_observations VALUES ('lineage-2', 2, 'stamp-2', 'seen', 2)")
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["failures"] == [
        {"code": "only_in_origin", "table": "stamp_observations", "count": 1}], report
    assert report["checks"]["tables"]["tables"]["stamp_observations"]["only_in_origin"] == [
        ["lineage-2", 2, "stamp-2", "seen", 2]]


def rebase(c, **counts):
    """Remake the manifest's database block for the copy's file as it now is, with any table counts given."""
    edit(c, lambda m: m.update(tables={**m["tables"], **counts},
                               database=base_block((c.copy / "lcm.db").read_bytes(), m["generation"], c.objects)))


def strings(value):
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    return [s for v in value for s in strings(v)] if isinstance(value, list) else [value] if isinstance(value, str) else []


@pytest.mark.parametrize("case", ["messages_copy_only", "messages_both", "metadata_both"])
def test_no_stored_text_without_key_columns(chain, tmp_path, case):
    """With none of a table's key columns on both sides, its rows are no key: codes and counts only, never text."""
    c = clone(chain, tmp_path)
    table, column = ("metadata", "key") if case == "metadata_both" else ("messages", "store_id")
    stamp = json.dumps({**json.loads(query(c.copy / "lcm.db", "SELECT value FROM metadata WHERE key = 'import:origin'")[0][0]),
                        "note": PLANT})
    sql(c.copy / "lcm.db", ("UPDATE metadata SET value = ? WHERE key = 'import:origin'", (stamp,)))
    sql(c.origin / "lcm.db", ("UPDATE messages SET content = content || ' edited' WHERE content LIKE ?", (f"%{PLANT}%",)))
    for database in (c.copy, c.origin) if case.endswith("both") else (c.copy,):
        sql(database / "lcm.db", f"ALTER TABLE {table} RENAME COLUMN {column} TO renamed")
    rebase(c)
    code, report = check(c)  # cli() asserts the planted string is absent from stdout and stderr
    assert code == 1 and failed(report, "tables") and table not in report["checks"]["tables"]["tables"], report
    assert any(f.get("table") == table and f.get("count") for f in report["checks"]["tables"]["failures"]), report
    assert not [s for s in strings(report) if stamp in s or PLANT in s]


def test_no_contracted_columns(chain, tmp_path):
    """A table with none of its contracted columns on both sides is compared by its row count."""
    c = clone(chain, tmp_path)
    for database in (c.copy / "lcm.db", c.origin / "lcm.db"):
        sql(database, "CREATE TABLE ledger_meta (extra TEXT)", "INSERT INTO ledger_meta VALUES ('one')")
    rebase(c, ledger_meta=1)
    code, report = check(c)
    assert code == 0 and report["checks"]["tables"]["pass"], report
    sql(c.origin / "lcm.db", "INSERT INTO ledger_meta VALUES ('two')")
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["failures"] == [
        {"code": "changed", "table": "ledger_meta", "count": 1}], report
    assert "ledger_meta" not in report["checks"]["tables"]["tables"]


def test_one_columns_failure_per_table(chain, tmp_path):
    c = clone(chain, tmp_path)
    sql(c.copy / "lcm.db", "ALTER TABLE capture_events RENAME COLUMN source_kind TO renamed")
    rebase(c)
    code, report = check(c)
    assert code == 1 and report["checks"]["tables"]["failures"] == [{"code": "columns", "table": "capture_events"}]
    assert report["checks"]["tables"]["failure_count"] == 1, report


# -- 16 and 17: refusals, and where the checker writes --

def test_refusals(chain, tmp_path):
    def wal(c):
        with closing(sqlite3.connect(c.origin / "lcm.db")) as conn:
            assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        assert not (c.origin / "lcm.db-wal").exists()

    def manifest(change):
        return lambda c: edit(c, change)

    def delta_under_parts(m):
        obj = m["database"]["deltas"][0]["objects"][0]
        obj["key"] = "parts/" + obj["key"].split("/")[1]

    cases = {
        "copy_is_capture_home": lambda c: shutil.copy2(c.origin / "capture-owner.json", c.copy / "capture-owner.json"),
        "origin_wal_without_sidecars": wal,
        "manifest_unreadable": lambda c: c.manifest.write_bytes(b"{not json"),
        "newer_layout": manifest(lambda m: m.update(layout=2)),
        "replica_newer_contract": manifest(lambda m: m["stamp"]["contract"].update(version=contract()[0] + 1)),
        "bad_name/payload": manifest(lambda m: m["payloads"][0].update(name="a/b.json")),
        "bad_name/part": manifest(lambda m: m["database"]["base"]["parts"][0].update(
            key=m["database"]["base"]["parts"][0]["key"].upper().replace("PARTS/", "parts/"))),
        "bad_name/delta": manifest(delta_under_parts),
    }
    for name, mutate in cases.items():
        c = clone(chain, tmp_path / name.replace("/", "-"))
        mutate(c)
        (c.root / "rebuild").mkdir()
        before = tree(c.root)
        code, report = cli("--copy", c.copy, "--manifest", c.manifest, "--origin", c.origin,
                           "--objects", c.objects, "--rebuild-dir", c.root / "rebuild")
        assert (code, report["refused"]) == (2, name.split("/")[0]), name
        assert tree(c.root) == before, name
    c = clone(chain, tmp_path / "flags")
    (c.root / "rebuild").mkdir()
    for args, refused in (((("--objects", c.objects),), "rebuild_needs_both"),
                          ((("--rebuild-dir", c.root / "rebuild"),), "rebuild_needs_both")):
        before = tree(c.root)
        code, report = cli("--copy", c.copy, "--manifest", c.manifest, *args[0])
        assert (code, report["refused"]) == (2, refused) and tree(c.root) == before
    (c.root / "rebuild" / "here").write_text("", encoding="utf-8")
    before = tree(c.root)
    code, report = cli("--copy", c.copy, "--manifest", c.manifest, "--objects", c.objects,
                       "--rebuild-dir", c.root / "rebuild")
    assert (code, report["refused"]) == (2, "rebuild_dir_not_empty") and tree(c.root) == before


def test_writes_in_one_place(chain, tmp_path):
    c = clone(chain, tmp_path)
    watched = (c.copy, c.origin, c.objects)
    before, everything = tree(*watched), tree(c.root)
    code, report = check(c, objects=False)
    assert code == 0 and tree(*watched) == before and tree(c.root) == everything
    code, report = check(c)
    new = sorted(p[1] for p in tree(c.root) - everything)
    assert code == 0 and tree(*watched) == before
    assert len(new) == 2 and new[1] == new[0] + "/lcm.db", new  # the rebuild folder the test made, and lcm.db in it
    obj = json.loads(c.manifest.read_text())["database"]["deltas"][-1]["objects"][0]["key"]
    shutil.move(c.objects / obj, c.root / "aside")
    before, everything = tree(*watched), tree(c.root)
    code, report = check(c)
    new = sorted(p[1] for p in tree(c.root) - everything)
    assert code == 1 and tree(*watched) == before and len(new) == 2 and new[1].endswith("/lcm.db.partial")


def test_no_stored_text(chain, tmp_path):
    c = clone(chain, tmp_path)
    assert PLANT.encode() in (c.copy / "lcm.db").read_bytes()
    assert any(PLANT.encode() in p.read_bytes() for p in (c.copy / "lcm-large-outputs").iterdir())
    row = query(c.copy / "lcm.db", "SELECT store_id FROM messages WHERE content LIKE ?", f"%{PLANT}%")[0][0]
    sql(c.copy / "lcm.db", ("UPDATE messages SET content = content || ? WHERE store_id = ?", (" " + PLANT, row)))
    for p in (c.copy / "lcm-large-outputs").iterdir():
        p.write_bytes(p.read_bytes() + PLANT.encode())
    for mode in ("origin", "manifest"):
        code, report = check(c, mode)  # cli() asserts the planted string is absent from stdout and stderr
        assert code == 1 and "payload_differs" in codes(report, "payload_files")
    assert report is not None and check(c)[1]["checks"]["tables"]["tables"]["messages"]["changed"] == [row]


# -- 19 to 21: constants, isolation, the shared vectors --

def test_constants(tool):
    version, digest, tables = contract()
    assert (tool.CONTRACT_VERSION, tool.CONTRACT_DIGEST) == (version, digest)
    assert len(tables) == 15
    assert {t: tool.COLUMNS[t] for t in tool.COLUMNS} == {t: names for t, (names, _) in tables.items()}
    assert {t: tool.PRIMARY[t] for t in tool.PRIMARY} == {t: pk for t, (_, pk) in tables.items()}
    assert tool.EXTERNALIZED == externalize._EXTERNALIZED_REF_RE.pattern
    assert tool.INGEST == ingest_content._INGEST_PLACEHOLDER_RE.pattern
    assert tool.MINTED == externalize._GENERATED_PAYLOAD_REF_RE.pattern
    assert tool.OWNER_KEY == cc.OWNER_KEY


def test_isolated_run(chain, tmp_path):
    done = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "--rebuild-dir" in done.stdout
    code, report = check(clone(chain, tmp_path))
    assert code == 0 and report["pass"] is True
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module.split(".")[0])
    assert imported and imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


def file_a():
    data = bytearray(b"".join(bytes([k]) * 512 for k in range(3)))
    data[16:18] = b"\x02\x00"
    return bytes(data)


def file_b():
    a = file_a()
    return a[:512] + b"\xaa" * 512 + a[1024:] + bytes(512)


OBJECT_AB = bytes.fromhex("4c434d585044310a0000020000000002789c636060605c350a463460606060661805231a0000442e5414")
BLOCK_F11 = {
    "page_size": 512, "file_bytes": 2048,
    "file_sha256": "916e79b77b05ac39fbb92433771f87ced54de932e04c3a01042be863e537ac4f",
    "page_list_sha256": "0b4ae118a4c210fabb5d988c90f4d4a3fdfafd6727f7426b7e08bd461147d1f4",
    "base": {"generation": 1, "file_bytes": 1536,
             "file_sha256": "144056a8546425225385ebdb956b7584a8a34f1ebecac3e68c3f73553c8f9a46",
             "reason": "first", "part_size": 1024, "parts": [
                 {"key": "parts/3b2318e81fd1de9fc1e3fbd184bfdca9efdac6ee7d10bcf78cba505b2bfb475b."
                         "00000000000000000000000000000001", "bytes": 1024},
                 {"key": "parts/1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9."
                         "00000000000000000000000000000001", "bytes": 512}]},
    "deltas": [{"generation": 2, "file_pages": 4, "objects": [
        {"key": "deltas/e467bc42aa2753ec64ece09b0f147c8b3f6a14265916145d222328707e7c69ea."
                "00000000000000000000000000000002", "bytes": 42, "records": 2}]}]}


def f11_objects(folder, object_bytes=OBJECT_AB):
    a = file_a()
    for key, blob in zip([p["key"] for p in BLOCK_F11["base"]["parts"]] + [BLOCK_F11["deltas"][0]["objects"][0]["key"]],
                         (a[:1024], a[1024:], object_bytes)):
        (folder / key).parent.mkdir(parents=True, exist_ok=True)
        (folder / key).write_bytes(blob)
    return folder


def test_shared_vectors(tool, tmp_path):
    files = {"a": file_a(), "b": file_b(), "c": bytes(16) + b"\x00\x01" + bytes(65536 - 18)}
    expected = {
        "a": (["b6d17a1a271154e2d74bdf05d114d2f55626bd2898e9c147c7489e3c1c878f75",
               "6caf38d537984e261527b8caef5f990fb91415a1db917198821a79ed28997973",
               "1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9"],
              "5651d68c2542dc58414e77e1816c5af341e54e5b2daf96396af4cdae06813e8a",
              "144056a8546425225385ebdb956b7584a8a34f1ebecac3e68c3f73553c8f9a46", 512),
        "b": (["b6d17a1a271154e2d74bdf05d114d2f55626bd2898e9c147c7489e3c1c878f75",
               "799edf40e8115dc980109a64ff0a7ae2c6b62e20313c4a01f9871d0e189aa7c2",
               "1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9",
               "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560"],
              "0b4ae118a4c210fabb5d988c90f4d4a3fdfafd6727f7426b7e08bd461147d1f4",
              "916e79b77b05ac39fbb92433771f87ced54de932e04c3a01042be863e537ac4f", 512),
        "c": (["0a0c88c08b7fcf477db1ae6d68f48abc6b2ef3e3e7fad9936e518f94e26c2d7b"],
              "b03d47e0fc0d7bb43d78f933243b667f09b8b8a212f7bfa74f7442bf9c90cf0f",
              "0a0c88c08b7fcf477db1ae6d68f48abc6b2ef3e3e7fad9936e518f94e26c2d7b", 65536),
    }
    for name, (digests, listing, whole, size) in expected.items():
        (tmp_path / name).write_bytes(files[name])
        assert [d.hex() for d in tool.page_digests(io.BytesIO(files[name]), size)] == digests, name
        assert tool.scan(tmp_path / name, size) == (len(files[name]), size, whole, listing), name
    assert len(OBJECT_AB) == 42
    assert sha(OBJECT_AB) == "e467bc42aa2753ec64ece09b0f147c8b3f6a14265916145d222328707e7c69ea"
    assert list(tool.records(io.BytesIO(OBJECT_AB), 512)) == [(1, b"\xaa" * 512), (3, bytes(512))]
    (tmp_path / "out").mkdir()
    tool.rebuild(deepcopy.deepcopy(BLOCK_F11), f11_objects(tmp_path / "objects"), tmp_path / "out", 2)
    assert os.listdir(tmp_path / "out") == ["lcm.db"] and (tmp_path / "out/lcm.db").read_bytes() == file_b()


# -- 22 to 24: object refusals, rebuild refusals, the chain's shape --

def refused(tool, code, *call):
    with pytest.raises(tool.Refused) as caught:
        result = call[0](*call[1:])
        list(result) if result is not None else None
    assert caught.value.code == code, (code, caught.value.code)
    return caught.value


def test_object_refusals(tool, tmp_path):
    good = [(1, b"\x01" * 512), (4, b"\x04" * 512)]
    body = b"".join(struct.pack(">I", i) + p for i, p in good)
    stream = zlib.compress(body, 6)

    def decode(blob, size=512):
        return tool.records(io.BytesIO(blob), size)

    assert list(decode(delta_object(512, good))) == good
    cases = [
        ("delta_magic", b"LCMXPD2\n" + delta_object(512, good)[8:]),
        ("delta_magic", delta_object(512, good)[:15]),
        ("delta_page_size", delta_object(1024, [], count=2, compressed=stream)),
        ("delta_stream", delta_object(512, good, count=3)),
        ("delta_stream", delta_object(512, good, count=1)),
        ("delta_stream", delta_object(512, good, compressed=stream[:len(stream) // 2])),
        ("delta_stream", delta_object(512, good, compressed=stream[:-2])),
        ("delta_stream", delta_object(512, good, compressed=b"not a zlib stream at all")),
        ("delta_trailing", delta_object(512, good) + b"\x00"),
        ("delta_order", delta_object(512, [(2, bytes(512)), (2, bytes(512))])),
        ("delta_order", delta_object(512, [(3, bytes(512)), (2, bytes(512))])),
    ]
    for code, blob in cases:
        refused(tool, code, decode, blob)
    assert list(decode(delta_object(512, [], compressed=zlib.compress(b"", 6)))) == []
    changed = bytearray(OBJECT_AB)
    changed[30] ^= 1
    (tmp_path / "out").mkdir()
    error = refused(tool, "object_sha256", tool.rebuild, deepcopy.deepcopy(BLOCK_F11),
                    f11_objects(tmp_path / "objects", bytes(changed)), tmp_path / "out", 2)
    assert error.detail["key"] == BLOCK_F11["deltas"][0]["objects"][0]["key"]


def rebuild_codes(c):
    code, report = check(c, "manifest")
    assert code == 1 and failed(report, "rebuild"), report["checks"]["rebuild"]
    return report["checks"]["rebuild"]


def test_rebuild_refusals(chain, wave, tmp_path):
    counter = itertools.count()

    def fresh(source=chain):
        return clone(source, tmp_path / str(next(counter)))

    def manifest_of(c):
        return json.loads(c.manifest.read_text(encoding="utf-8"))

    # object_missing: a part, an object, a symlink in a part's place
    c = fresh()
    part = manifest_of(c)["database"]["base"]["parts"][1]["key"]
    shutil.move(c.objects / part, c.root / "aside")
    assert rebuild_codes(c)["failures"] == [{"code": "object_missing", "key": part}]
    c = fresh()
    obj = manifest_of(c)["database"]["deltas"][1]["objects"][0]["key"]
    shutil.move(c.objects / obj, c.root / "aside")
    assert rebuild_codes(c)["failures"] == [{"code": "object_missing", "key": obj}]
    c = fresh()
    shutil.move(c.objects / part, c.root / "aside")
    os.symlink(c.root / "aside", c.objects / part)
    assert rebuild_codes(c)["failures"] == [{"code": "object_missing", "key": part}]

    # object_bytes: a part (the sum kept), an object
    c = fresh()
    m = edit(c, lambda m: (m["database"]["base"]["parts"][0].update(bytes=PART_SIZE - 512),
                           m["database"]["base"]["parts"][1].update(bytes=m["database"]["base"]["parts"][1]["bytes"] + 512)))
    assert rebuild_codes(c)["failures"] == [{"code": "object_bytes", "key": m["database"]["base"]["parts"][0]["key"]}]
    c = fresh()
    m = edit(c, lambda m: m["database"]["deltas"][0]["objects"][0].update(bytes=1))
    assert rebuild_codes(c)["failures"] == [{"code": "object_bytes", "key": m["database"]["deltas"][0]["objects"][0]["key"]}]

    # base_digest_mismatch: the base digest changed; two equal-length parts exchanged
    c = fresh()
    edit(c, lambda m: m["database"]["base"].update(file_sha256="0" * 64))
    assert rebuild_codes(c)["codes"] == ["base_digest_mismatch"]
    c = fresh()
    parts = manifest_of(c)["database"]["base"]["parts"]
    assert len(parts) >= 3 and parts[0]["bytes"] == parts[1]["bytes"]
    edit(c, lambda m: m["database"]["base"].update(parts=[parts[1], parts[0], *parts[2:]]))
    assert rebuild_codes(c)["codes"] == ["base_digest_mismatch"]

    # delta_count, delta_order (two objects of one entry exchanged), delta_index
    c = fresh()
    edit(c, lambda m: m["database"]["deltas"][0]["objects"][0].update(records=m["database"]["deltas"][0]["objects"][0]["records"] + 1))
    assert rebuild_codes(c)["codes"] == ["delta_count"]
    c = fresh()
    assert len(manifest_of(c)["database"]["deltas"][0]["objects"]) >= 2
    edit(c, lambda m: m["database"]["deltas"][0].update(objects=m["database"]["deltas"][0]["objects"][::-1]))
    assert rebuild_codes(c)["codes"] == ["delta_order"]
    c = fresh()
    reencode(c, 1, -1, lambda records, entry: records + [(entry["file_pages"], bytes(4096))])
    assert rebuild_codes(c)["codes"] == ["delta_index"]

    # delta_gap: the first entry after the base; an entry after a shorter generation
    for source, entry_index in ((chain, 0), (wave, 1)):
        c = fresh(source)
        m = manifest_of(c)["database"]
        before = m["base"]["file_bytes"] // m["page_size"] if entry_index == 0 else m["deltas"][entry_index - 1]["file_pages"]
        assert m["deltas"][entry_index]["file_pages"] > before
        reencode(c, entry_index, -1, lambda records, entry: records[:-1])
        assert rebuild_codes(c)["failures"] == [{"code": "delta_gap", "generation": m["deltas"][entry_index]["generation"]}]

    # rebuild_digest_mismatch: file_sha256 changed; page_list_sha256 changed
    for field in ("file_sha256", "page_list_sha256"):
        c = fresh()
        edit(c, lambda m: m["database"].update({field: "1" * 64}))
        assert rebuild_codes(c)["failures"] == [{"code": "rebuild_digest_mismatch", "generation": 3}]


def test_chain_shape(chain, tmp_path):
    counter = itertools.count()

    def first(m):
        return m["database"]["deltas"][0]

    breaks = {
        "missing": (3, "bad_database_block", lambda m: m["database"].pop("page_list_sha256")),
        "true": (3, "bad_database_block", lambda m: first(m).update(file_pages=True)),
        "page_size": (3, "bad_database_block", lambda m: m["database"].update(page_size=1000)),
        "whole_pages": (3, "bad_database_block", lambda m: m["database"]["base"].update(
            file_bytes=m["database"]["base"]["file_bytes"] + 1)),
        "sum": (3, "bad_database_block", lambda m: m["database"]["base"]["parts"][0].update(bytes=PART_SIZE - 1)),
        "file_pages": (3, "bad_database_block", lambda m: first(m).update(file_pages=0)),
        "file_bytes": (3, "bad_database_block", lambda m: m["database"].update(
            file_bytes=m["database"]["file_bytes"] + m["database"]["page_size"])),
        "base_digest": (1, "bad_database_block", lambda m: m["database"].update(file_sha256="2" * 64)),
        "repeated": (3, "chain_generation", lambda m: m["database"]["deltas"][1].update(generation=2)),
        "last": (3, "chain_generation", lambda m: m.update(generation=4)),
        "empty": (1, "chain_generation", lambda m: m["database"]["base"].update(generation=2)),
    }
    for name, (gen, expected, change) in breaks.items():
        c = clone(chain, tmp_path / str(next(counter)), gen)
        edit(c, change)
        for mode in ("origin", "manifest"):
            for objects in (False, c.root / "absent"):
                code, report = check(c, mode, objects)
                assert (code, codes(report, "chain")) == (1, [expected]), (name, mode, objects)
                assert report["checks"]["rebuild"]["ran"] is False and report["checks"]["file_digest"]["ran"] is False
                assert not (c.root / "absent").exists()
    allowed = {"reason": lambda m: m["database"]["base"].update(reason="unheard-of"),
               "part_size": lambda m: m["database"]["base"].pop("part_size"),
               "part_bytes": lambda m: m["database"]["base"].update(part_size=PART_SIZE * 2),
               "extra": lambda m: (m["database"].update(novel=1), first(m)["objects"][0].update(novel=[1]))}
    for name, change in allowed.items():
        c = clone(chain, tmp_path / str(next(counter)))
        edit(c, change)
        code, report = check(c, "manifest")
        assert code == 0 and report["database_read"] == "rebuilt", name


# -- 25 to 31: delta chains and their faults --

def test_shorter_then_longer(wave, tmp_path):
    sizes = [len(g.data) for g in wave.gens]
    assert sizes[1] < sizes[0] < sizes[2], sizes
    for g in wave.gens:
        c = clone(wave, tmp_path / str(g.gen), g.gen)
        for mode in ("origin", "manifest"):
            code, report = check(c, mode)
            assert code == 0 and report["database_read"] == "rebuilt", (g.gen, mode, report)


def test_delta_left_out(chain, tmp_path):
    c = clone(chain, tmp_path / "entry")
    edit(c, lambda m: m["database"]["deltas"].pop(0))
    code, report = check(c, "manifest")
    assert code == 1 and codes(report, "chain") == ["chain_generation"]
    c = clone(chain, tmp_path / "object")
    key = json.loads(c.manifest.read_text())["database"]["deltas"][0]["objects"][0]["key"]
    shutil.move(c.objects / key, c.root / "aside")
    code, report = check(c, "manifest")
    assert report["checks"]["rebuild"]["failures"] == [{"code": "object_missing", "key": key}]


def test_deltas_swapped(chain, tmp_path):
    c = clone(chain, tmp_path / "entries")
    edit(c, lambda m: m["database"].update(deltas=m["database"]["deltas"][::-1]))
    code, report = check(c, "manifest")
    # The two generations differ in size, so condition (e) holds as well; F10 step 1 accepts either code.
    assert code == 1 and codes(report, "chain") in (["chain_generation"], ["bad_database_block"])
    c = clone(chain, tmp_path / "objects")

    def swap(m):
        one, two = m["database"]["deltas"]
        one["objects"], two["objects"] = two["objects"], one["objects"]

    edit(c, swap)
    code, report = check(c, "manifest")
    # Generation 3's objects write pages at or past generation 2's page count: this fixture gives delta_index.
    assert code == 1 and report["checks"]["rebuild"]["codes"] == ["delta_index"]


def stale(chain, c):
    """Generation 3's first record carries that page's generation-2 bytes; key, bytes and records recomputed."""
    two = chain.gens[1].data

    def change(records, entry):
        index, _ = records[0]
        size = page_size(two)
        return [(index, two[index * size:(index + 1) * size])] + records[1:]

    reencode(c, 1, 0, change)


def test_stale_page(chain, tmp_path):
    c = clone(chain, tmp_path)
    stale(chain, c)
    code, report = check(c, "manifest")
    assert code == 1 and report["checks"]["rebuild"]["codes"] == ["rebuild_digest_mismatch"]
    assert report["checks"]["file_digest"]["pass"] is True


def test_restore_differs(chain, tmp_path):
    c = clone(chain, tmp_path)
    data = bytearray((c.copy / "lcm.db").read_bytes())
    size = page_size(bytes(data))
    data[2 * size + 100] ^= 0xFF
    (c.copy / "lcm.db").write_bytes(bytes(data))
    code, report = check(c, "manifest")
    assert code == 1 and failed(report, "file_digest")
    assert report["checks"]["rebuild"]["failures"] == [{"code": "rebuild_differs", "page": 2}]


def test_copy_journal_mode(chain, tmp_path):
    """The copy, and the rebuilt file, are in rollback-journal mode: header bytes 18 and 19 are both 1."""
    c = clone(chain, tmp_path)
    data = bytearray((c.copy / "lcm.db").read_bytes())
    assert data[18:20] == b"\x01\x01" and check(c)[0] == 0
    data[18:20] = b"\x02\x02"
    (c.copy / "lcm.db").write_bytes(bytes(data))
    edit(c, lambda m: m.update(database=base_block(bytes(data), m["generation"], c.objects)))
    for mode, objects in (("origin", True), ("manifest", False)):
        code, report = check(c, mode, objects)
        assert code == 1 and codes(report, "file_digest") == ["copy_journal_mode"], (mode, report)
        assert codes(report, "rebuild") == ["copy_journal_mode"] if objects else not report["checks"]["rebuild"]["ran"]


def test_bounded_decode(tool):
    compressor, chunks = zlib.compressobj(6), []
    for _ in range(64):
        chunks.append(compressor.compress(bytes(1 << 20)))
    chunks.append(compressor.flush())
    blob = delta_object(4096, [], count=1, compressed=b"".join(chunks))
    handle = io.BytesIO(blob)
    tracemalloc.start()
    try:
        with pytest.raises(tool.Refused) as caught:
            list(tool.records(handle, 4096))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert caught.value.code == "delta_stream" and peak < 16 * 1024 * 1024, peak


ROW_CHECKS = ("integrity", "payload_files", "unresolved", "stamp", "counts", "ledger", "tables", "payloads")


def test_which_database_was_read(chain, tmp_path):
    c = clone(chain, tmp_path / "good")
    code, report = check(c)
    assert code == 0 and report["database_read"] == "rebuilt"
    assert all(report["checks"][name]["ran"] and report["checks"][name]["pass"] for name in ROW_CHECKS)
    c = clone(chain, tmp_path / "stale")
    stale(chain, c)
    code, report = check(c)
    assert code == 1 and report["pass"] is False and report["database_read"] == "copy"
    assert all(report["checks"][name]["ran"] and report["checks"][name]["pass"] for name in ROW_CHECKS)
