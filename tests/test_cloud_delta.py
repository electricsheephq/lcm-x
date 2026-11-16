"""Packet 2d: literal vectors, independent objects and small synthetic stores."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import struct
import subprocess
import sys
import tracemalloc
import zlib

import pytest

from hermes_lcm import cloud_delta as cd

COPY = "abcdef01-2345-6789-abcd-0123456789ab"
MAGIC = b"LCMXPD1\n"
A = bytearray(b"\0" * 512 + b"\1" * 512 + b"\2" * 512)
A[16:18] = b"\2\0"
A = bytes(A)
B = A[:512] + b"\xaa" * 512 + A[1024:] + b"\0" * 512
C = bytearray(65536)
C[16:18] = b"\0\1"
C = bytes(C)
A_PAGES = [
    "b6d17a1a271154e2d74bdf05d114d2f55626bd2898e9c147c7489e3c1c878f75",
    "6caf38d537984e261527b8caef5f990fb91415a1db917198821a79ed28997973",
    "1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9",
]
B_PAGES = [A_PAGES[0], "799edf40e8115dc980109a64ff0a7ae2c6b62e20313c4a01f9871d0e189aa7c2",
           A_PAGES[2], "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560"]
A_LIST = "5651d68c2542dc58414e77e1816c5af341e54e5b2daf96396af4cdae06813e8a"
B_LIST = "0b4ae118a4c210fabb5d988c90f4d4a3fdfafd6727f7426b7e08bd461147d1f4"
A_SHA = "144056a8546425225385ebdb956b7584a8a34f1ebecac3e68c3f73553c8f9a46"
B_SHA = "916e79b77b05ac39fbb92433771f87ced54de932e04c3a01042be863e537ac4f"
C_SHA = "0a0c88c08b7fcf477db1ae6d68f48abc6b2ef3e3e7fad9936e518f94e26c2d7b"
C_LIST = "b03d47e0fc0d7bb43d78f933243b667f09b8b8a212f7bfa74f7442bf9c90cf0f"
VECTOR = bytes.fromhex("4c434d585044310a0000020000000002789c636060605c350a463460606060661805231a0000442e5414")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def info(data, p=512):
    digests = b"".join(hashlib.sha256(data[i:i+p]).digest() for i in range(0, len(data), p))
    return cd.PageList(p, len(data)//p, digests, len(data), sha(data), sha(digests))


def put(tmp_path, data, name="staged"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def refused(code, call, *args, **kwargs):
    with pytest.raises(cd.DeltaRefused) as exc:
        call(*args, **kwargs)
    assert exc.value.code == code
    return exc.value


def obj(records, p=512, count=None, level=6):
    raw = b"".join(struct.pack(">I", i) + page for i, page in records)
    return MAGIC + struct.pack(">II", p, len(records) if count is None else count) + zlib.compress(raw, level)


def reference(objects, data, kind="deltas", **extra):
    key = f"{kind}/{sha(data)}." + "0" * 31 + "1"
    objects[key] = data
    return {"key": key, "bytes": len(data), **extra}


def base(data=A, part_size=1024):
    objects = {}
    pi = info(data)
    block = {"page_size": 512, "file_bytes": len(data), "file_sha256": sha(data),
             "page_list_sha256": pi.page_list_sha256,
             "base": {"generation": 1, "file_bytes": len(data), "file_sha256": sha(data),
                      "reason": "first", "part_size": part_size,
                      "parts": [reference(objects, data[i:i+part_size], "parts")
                                for i in range(0, len(data), part_size)]}, "deltas": []}
    return block, objects


def append(block, objects, old, new, *, cap=100):
    indexes = [i for i in range(len(new)//512)
               if new[i*512:(i+1)*512] != old[i*512:(i+1)*512] or i*512 >= len(old)]
    refs = []
    for start in range(0, len(indexes), cap):
        group = indexes[start:start+cap]
        refs.append(reference(objects, obj([(i, new[i*512:(i+1)*512]) for i in group]), records=len(group)))
    block["deltas"].append({"generation": len(block["deltas"])+2, "file_pages": len(new)//512, "objects": refs})
    block.update(file_bytes=len(new), file_sha256=sha(new), page_list_sha256=info(new).page_list_sha256)


def worked():
    block, objects = base()
    block.update(file_bytes=2048, file_sha256=B_SHA, page_list_sha256=B_LIST)
    block["deltas"] = [{"generation": 2, "file_pages": 4,
                        "objects": [reference(objects, VECTOR, records=2)]}]
    return block, objects


def real_db(path, p=512):
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA page_size={p}")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)")
    conn.commit()
    return conn


def test_page_size_from_the_header(tmp_path):
    for p in (512, 4096, 32768, 65536):
        data = bytearray(p)
        data[16:18] = (1 if p == 65536 else p).to_bytes(2, "big")
        assert cd.hash_pages(put(tmp_path, data)).page_size == p
    conn = real_db(tmp_path / "real", 65536)
    conn.close()
    assert cd.hash_pages(tmp_path / "real").page_size == 65536
    for p in (0, 3, 256, 1000):
        data = bytearray(A)
        data[16:18] = p.to_bytes(2, "big")
        refused("page_size", cd.hash_pages, put(tmp_path, data))
    refused("page_size", cd.hash_pages, put(tmp_path, A), expected_page_size=4096)
    refused("file_not_whole_pages", cd.hash_pages, put(tmp_path, A+b"x"))
    refused("file_not_whole_pages", cd.hash_pages, put(tmp_path, b""))


def test_the_m3_vector(tmp_path):
    for data, p, pages, digest, listed in (
        (A, 512, A_PAGES, A_SHA, A_LIST), (B, 512, B_PAGES, B_SHA, B_LIST),
        (C, 65536, [C_SHA], C_SHA, C_LIST),
    ):
        got = cd.hash_pages(put(tmp_path, data))
        assert got.page_size == p
        assert got.pages == len(pages)
        assert got.digests == bytes.fromhex("".join(pages))
        assert got.file_bytes == len(data)
        assert got.file_sha256 == digest
        assert got.page_list_sha256 == listed


def test_a_real_file_read_only(tmp_path):
    for p in (512, 4096):
        path = tmp_path / f"real-{p}"
        conn = real_db(path, p)
        conn.executemany("INSERT INTO t(data) VALUES (?)", [("x"*600,)]*12)
        conn.commit()
        count = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.close()
        data, stamp = path.read_bytes(), path.stat().st_mtime_ns
        got = cd.hash_pages(path)
        assert got.pages == count
        assert got == info(data, p)
        if os.geteuid() != 0:
            path.chmod(0o400)
            assert cd.hash_pages(path) == got
            assert cd.encode_delta(path, b"", got, tmp_path / f"objects-{p}")
        assert path.read_bytes() == data and path.stat().st_mtime_ns == stamp


def test_changed_pages(tmp_path):
    old, new = info(A).digests, info(B).digests
    assert cd.changed_pages(old, new) == [1, 3]
    assert cd.changed_pages(old, old) == []
    assert cd.changed_pages(old, old+hashlib.sha256(bytes(512)).digest()) == [3]
    assert cd.changed_pages(old, old+old[:32]) == [3]
    assert cd.changed_pages(old, new[:64]) == [1]
    for a, b in ((b"x"*33, new), (old, b"x"*33)):
        refused("bad_page_list", cd.changed_pages, a, b)


def test_object_bytes(tmp_path):
    path = put(tmp_path, B)
    assert cd.encode_delta(path, info(B).digests, info(B), tmp_path / "idle") == []
    assert not (tmp_path / "idle").exists()
    refs = cd.encode_delta(path, info(A).digests, info(B), tmp_path / "objects")
    assert len(refs) == 1
    ref = refs[0]
    data = ref.path.read_bytes()
    assert ref.path.name == ref.sha256 == sha(data)
    assert ref.bytes == len(data) and ref.records == 2
    assert ref.path.stat().st_mode & 0o777 == 0o600
    assert data[:16].hex() == "4c434d585044310a0000020000000002"
    assert data[16:18] == b"\x78\x9c"
    assert zlib.decompress(data[16:]) == struct.pack(">I", 1)+b"\xaa"*512+struct.pack(">I", 3)+bytes(512)


def test_row_2_stream(tmp_path):
    path = tmp_path / "real"
    conn = real_db(path)
    old = info(path.read_bytes()).digests
    conn.executemany("INSERT INTO t(data) VALUES (?)", [("r"*700,)]*15)
    conn.commit()
    conn.close()
    data = path.read_bytes()
    indexes = [i for i in range(len(data)//512)
               if i*32 >= len(old) or hashlib.sha256(data[i*512:(i+1)*512]).digest() != old[i*32:(i+1)*32]]
    refs = cd.encode_delta(path, old, info(data), tmp_path / "objects", records_per_object=3)
    assert refs
    assert b"".join(zlib.decompress(r.path.read_bytes()[16:]) for r in refs) == b"".join(
        i.to_bytes(4, "big")+data[i*512:(i+1)*512] for i in indexes)


def test_the_decode_vector():
    assert list(cd.read_object(VECTOR, 512)) == [(1, b"\xaa"*512), (3, bytes(512))]


def test_several_objects(tmp_path, monkeypatch):
    old = A[:512] + bytes(512)*7
    new = A[:512] + b"\xaa"*512*8
    refs = cd.encode_delta(put(tmp_path, new), info(old).digests, info(new), tmp_path / "objects", records_per_object=3)
    assert [r.records for r in refs] == [3, 3, 2]
    assert [i for r in refs for i, _ in cd.read_object(r.path.read_bytes(), 512)] == list(range(1, 9))
    block, objects = base(old)
    append(block, objects, old, new, cap=3)
    block["deltas"][0]["objects"] = [reference(objects, r.path.read_bytes(), records=r.records) for r in refs]
    path = tmp_path / "rebuilt"
    cd.rebuild(path, block, 2, objects.__getitem__)
    assert path.read_bytes() == new
    assert [cd.max_records(p) for p in (4096, 65536, 512, 1024)] == [8168, 510, 64900, 32576]
    for cap in (0, cd.max_records(512)+1):
        with pytest.raises(ValueError):
            cd.encode_delta(tmp_path / "staged", info(old).digests, info(new), tmp_path / "objects", records_per_object=cap)
    monkeypatch.setattr(cd, "MAX_OBJECT_BYTES", 16)
    refused("delta_object_too_large", cd.encode_delta, put(tmp_path, B), info(A).digests, info(B), tmp_path / "tiny")


def test_self_check_passes_and_fails(tmp_path):
    old, new = info(A).digests, info(B)
    refs = cd.encode_delta(put(tmp_path, B), old, new, tmp_path / "objects")
    cd.verify_delta([r.path for r in refs], old, new)
    sets = [[(3, bytes(512))], [(0, A[:512]), (1, b"\xaa"*512), (3, bytes(512))],
            [(1, A[512:1024]), (3, bytes(512))], [(1, b"\xaa"*512), (3, bytes(512)), (4, bytes(512))]]
    for records in sets:
        refused("delta_selfcheck", cd.verify_delta, [put(tmp_path, obj(records), "hand")], old, new)
    assert refs
    refs[0].path.write_bytes(obj([(1, bytes(512)), (3, bytes(512))]))
    refused("delta_selfcheck", cd.verify_delta, [r.path for r in refs], old, new)


def test_file_changed_after_hash_pass(tmp_path):
    path = put(tmp_path, B)
    new = cd.hash_pages(path)
    path.write_bytes(B[:512]+bytes(512)+B[1024:])
    refused("delta_selfcheck", cd.encode_delta, path, info(A).digests, new, tmp_path / "objects")
    path.write_bytes(B[:-512])
    refused("delta_selfcheck", cd.encode_delta, path, info(A).digests, new, tmp_path / "short")


def test_object_refusals():
    valid = obj([(1, b"\xaa"*512)])
    cases = [("delta_magic", b"X"+valid[1:]), ("delta_magic", valid[:15]),
             ("delta_page_size", obj([(1, bytes(1024))], p=1024)),
             ("delta_stream", obj([(1, bytes(512))], count=2)),
             ("delta_stream", obj([(1, bytes(512))], count=0)),
             ("delta_stream", valid[:20]), ("delta_stream", valid[:-2]),
             ("delta_stream", valid[:16]+b"no zlib"), ("delta_trailing", valid+b"x"),
             ("delta_order", obj([(1, bytes(512)), (1, bytes(512))])),
             ("delta_order", obj([(2, bytes(512)), (1, bytes(512))]))]
    for code, data in cases:
        refused(code, lambda: list(cd.read_object(data, 512)))
    refused("delta_count", lambda: list(cd.read_object(valid, 512, records=2)))
    assert list(cd.read_object(obj([]), 512)) == []
    assert list(cd.read_object(obj([(1, bytes(512))], level=1), 512)) == [(1, bytes(512))]


def test_bounded_decode():
    compressor = zlib.compressobj()
    chunks = [compressor.compress(bytes(1024*1024)) for _ in range(64)]
    data = MAGIC+struct.pack(">II", 512, 1)+b"".join(chunks)+compressor.flush()
    tracemalloc.start()
    try:
        refused("delta_stream", lambda: list(cd.read_object(data, 512)))
        assert tracemalloc.get_traced_memory()[1] < 16*1024*1024
    finally:
        tracemalloc.stop()


def test_rebuild_worked_example(tmp_path):
    block, objects = worked()
    path = tmp_path / "rebuilt"
    assert cd.rebuild(path, block, 2, objects.__getitem__) == info(B)
    assert path.read_bytes() == B
    block, objects = base()
    assert cd.rebuild(tmp_path / "base", block, 1, objects.__getitem__) == info(A)
    assert (tmp_path / "base").read_bytes() == A


def test_rebuild_real_chain(tmp_path):
    path = tmp_path / "real"
    conn = real_db(path)
    old = path.read_bytes()
    block, objects = base(old)
    snapshots = [old]
    initial = tmp_path / "g1"
    assert cd.rebuild(initial, block, 1, objects.__getitem__) == info(old)
    assert initial.read_bytes() == old
    with sqlite3.connect(initial) as check:
        assert check.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    for action in ("grow", "update", "shrink", "regrow"):
        if action in ("grow", "regrow"):
            conn.executemany("INSERT INTO t(data) VALUES (?)", [("x"*700,)]*(40 if action == "regrow" else 20))
        elif action == "update":
            conn.execute("UPDATE t SET data=? WHERE id=1", ("y"*700,))
        else:
            conn.execute("DELETE FROM t")
            conn.commit()
            conn.execute("VACUUM")
        conn.commit()
        new = path.read_bytes()
        append(block, objects, old, new, cap=3)
        snapshots.append(new)
        out = tmp_path / f"g{len(snapshots)}"
        cd.rebuild(out, block, len(snapshots), objects.__getitem__)
        assert out.read_bytes() == new
        check = sqlite3.connect(out)
        assert check.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        check.close()
        old = new
    conn.close()
    assert len(snapshots[1]) > len(snapshots[0])
    assert len(snapshots[2]) == len(snapshots[1])
    assert len(snapshots[3]) < len(snapshots[2]) < len(snapshots[4])


def test_seeded_chains(tmp_path):
    for seed in range(50):
        rng = random.Random(seed)
        old = A
        block, objects = base(old)
        for step in range(6):
            pages = [old[i:i+512] for i in range(0, len(old), 512)]
            count = rng.randint(1, 12)
            pages = (pages + [bytes(512)]*count)[:count]
            for i in range(1, count):
                if rng.choice((True, False)):
                    pages[i] = rng.randbytes(512)
            new = b"".join(pages)
            append(block, objects, old, new, cap=3)
            path = tmp_path / f"s{seed}-g{step}"
            cd.rebuild(path, block, step+2, objects.__getitem__)
            assert path.read_bytes() == new
            old = new


def test_chain_refusals(tmp_path):
    cases = []
    for kind in ("part", "object"):
        for code in ("object_bytes", "object_sha256", "object_missing"):
            block, objects = worked()
            ref = block["base"]["parts"][0] if kind == "part" else block["deltas"][0]["objects"][0]
            key = ref["key"]
            if code == "object_bytes":
                ref["bytes"] += 1
                if kind == "part":
                    block["base"]["parts"][1]["bytes"] -= 1
            elif code == "object_sha256":
                objects[key] = b"X"+objects[key][1:]
            else:
                del objects[key]
            cases.append((code, block, objects, 2, key))
    block, objects = worked()
    block["deltas"][0]["objects"][0]["records"] += 1
    cases.append(("delta_count", block, objects, 2, None))
    block, objects = base()
    append(block, objects, A, B, cap=1)
    block["deltas"][0]["objects"].reverse()
    cases.append(("delta_order", block, objects, 2, None))
    for records, code in (([(4, bytes(512))], "delta_index"), ([(1, b"\xaa"*512)], "delta_gap")):
        block, objects = worked()
        block["deltas"][0]["objects"] = [reference(objects, obj(records), records=len(records))]
        cases.append((code, block, objects, 2, None))
    block, objects = base()
    append(block, objects, A, A[:512])
    append(block, objects, A[:512], B)
    block["deltas"][-1]["objects"] = [reference(objects, obj([(3, bytes(512))]), records=1)]
    cases.append(("delta_gap", block, objects, 3, None))
    block, objects = worked()
    block["base"]["file_sha256"] = "0"*64
    cases.append(("base_digest_mismatch", block, objects, 2, None))
    block, objects = base(A, 512)
    append(block, objects, A, B)
    block["base"]["parts"][0], block["base"]["parts"][1] = block["base"]["parts"][1], block["base"]["parts"][0]
    cases.append(("base_digest_mismatch", block, objects, 2, None))
    for i, (code, block, objects, generation, key) in enumerate(cases):
        exc = refused(code, cd.rebuild, tmp_path / f"bad{i}", block, generation, objects.__getitem__)
        if code == "object_missing":
            assert exc.detail["key"] == key


def test_shape_refusals_read_nothing(tmp_path):
    block, objects = worked()
    append(block, objects, B, B)
    append(block, objects, B, B)
    cases = []
    for change in (lambda b: b["deltas"].pop(1), lambda b: b["deltas"].reverse(),
                   lambda b: b["deltas"][1].update(generation=2),
                   lambda b: b["deltas"][-1].update(generation=9)):
        bad = deepcopy(block)
        change(bad)
        cases.append(("chain_generation", bad, 4))
    bad, _ = base()
    cases.append(("chain_generation", bad, 2))
    changes = [lambda b: b.pop("page_list_sha256"), lambda b: b.update(file_bytes=True),
               lambda b: b.update(page_size=1000), lambda b: b["base"].update(file_bytes=1537),
               lambda b: b["base"]["parts"][0].update(bytes=1000),
               lambda b: b["deltas"][0].update(file_pages=0), lambda b: b.update(file_bytes=1024),
               lambda b: b.update(file_sha256="X"*64), lambda b: b.update(deltas={}),
               lambda b: b["deltas"][0]["objects"][0].update(records=1.0),
               lambda b: b["base"].update(file_bytes=0)]
    for change in changes:
        bad, _ = worked()
        change(bad)
        cases.append(("bad_database_block", bad, 2))
    bad, _ = base()
    bad["file_sha256"] = "0"*64
    cases.append(("bad_database_block", bad, 1))
    for key in ("parts/"+"A"*64+"."+"0"*32, "parts/../x"):
        bad, _ = worked()
        bad["base"]["parts"][0]["key"] = key
        cases.append(("bad_name", bad, 2))
    bad, _ = worked()
    bad["deltas"][0]["objects"][0]["key"] = bad["base"]["parts"][0]["key"]
    cases.append(("bad_name", bad, 2))
    for i, (code, bad, generation) in enumerate(cases):
        calls = []
        path = tmp_path / f"shape{i}"
        refused(code, cd.rebuild, path, bad, generation, lambda key: calls.append(key))
        assert not calls and not path.exists()
    good, objects = worked()
    good["base"]["reason"] = "future-reason"
    del good["base"]["part_size"]
    good["unknown"] = {"ignored": True}
    good["base"]["parts"][0]["extra"] = "ignored"
    good["deltas"][0]["extra"] = "ignored"
    good["deltas"][0]["objects"][0]["extra"] = "ignored"
    assert cd.rebuild(tmp_path / "good", good, 2, objects.__getitem__) == info(B)
    good["base"]["part_size"] = 512
    assert cd.rebuild(tmp_path / "odd-parts", good, 2, objects.__getitem__) == info(B)


def test_only_whole_file_digest_sees_it(tmp_path):
    for case in ("old-page", "file-hash", "page-list", "header"):
        block, objects = worked()
        if case == "old-page":
            data = obj([(1, A[512:1024]), (3, bytes(512))])
            block["deltas"][0]["objects"] = [reference(objects, data, records=2)]
        elif case == "file-hash":
            block["file_sha256"] = "0"*64
        elif case == "page-list":
            block["page_list_sha256"] = "0"*64
        else:
            data = obj([(0, bytes(512)), (1, b"\xaa"*512), (3, bytes(512))])
            block["deltas"][0]["objects"] = [reference(objects, data, records=3)]
        exc = refused("rebuild_digest_mismatch", cd.rebuild, tmp_path / case, block, 2, objects.__getitem__)
        assert exc.detail["generation"] == 2


def test_rebuild_by_patch(tmp_path):
    block, objects = worked()
    later = B[:512]+b"\xbb"*512+B[1024:]
    append(block, objects, B, later)
    for k, data in ((0, A), (1, B), (2, later)):
        calls = []
        def fetch(key):
            calls.append(key)
            return objects[key]
        path = put(tmp_path, data, f"patch{k}")
        assert cd.rebuild(path, block, 3, fetch, have=k) == info(later)
        assert path.read_bytes() == later
        excluded = [r["key"] for r in block["base"]["parts"]]
        excluded += [r["key"] for e in block["deltas"][:k] for r in e["objects"]]
        assert not set(calls).intersection(excluded)
    refused("start_mismatch", cd.rebuild, put(tmp_path, B[:-512], "short"), block, 3, objects.__getitem__, have=1)
    wrong = bytes(512)+B[512:]
    refused("rebuild_digest_mismatch", cd.rebuild, put(tmp_path, wrong, "wrong"), block, 3, objects.__getitem__, have=1)


def test_rebuild_never_overwrites(tmp_path):
    block, objects = worked()
    path = put(tmp_path, b"keep")
    refused("output_exists", cd.rebuild, path, block, 2, objects.__getitem__)
    assert path.read_bytes() == b"keep"


def test_page_list_files(tmp_path):
    folder = tmp_path / "pages"
    digests = info(A).digests
    cd.save_pending_list(folder, COPY, digests)
    assert cd.load_head_list(folder, COPY, A_LIST, 3) == digests
    assert not (folder / f"{COPY}.pending").exists()
    last = folder / f"{COPY}.last"
    assert last.read_bytes() == digests
    assert sha(last.read_bytes()) == A_LIST
    assert folder.stat().st_mode & 0o777 == 0o700
    assert last.stat().st_mode & 0o777 == 0o600
    cd.save_pending_list(folder, COPY, info(B).digests)
    cd.promote_pending_list(folder, COPY)
    assert last.read_bytes() == info(B).digests
    cd.promote_pending_list(folder, COPY)


def test_no_match_is_none(tmp_path, monkeypatch):
    assert cd.load_head_list(tmp_path / "missing", COPY, A_LIST, 3) is None
    digests = info(A).digests
    for i, (data, count) in enumerate(((b"x"+digests[1:], 3), (digests[:-1], 3),
                                     (b"x"*33, 3), (digests, 4), (info(B).digests, 3))):
        folder = tmp_path / str(i)
        folder.mkdir()
        for suffix in ("last", "pending"):
            put(folder, data, f"{COPY}.{suffix}")
        assert cd.load_head_list(folder, COPY, A_LIST, count) is None
        assert (folder / f"{COPY}.pending").read_bytes() == data
    folder = tmp_path / "links"
    folder.mkdir()
    target = put(tmp_path, digests, "target")
    for suffix in ("last", "pending"):
        (folder / f"{COPY}.{suffix}").symlink_to(target)
    assert cd.load_head_list(folder, COPY, A_LIST, 3) is None
    assert (folder / f"{COPY}.pending").is_symlink()
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    put(unreadable, digests, f"{COPY}.last")
    real_open = os.open
    def denied(path, *args, **kwargs):
        if Path(path).parent == unreadable:
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(cd.os, "open", denied)
        assert cd.load_head_list(unreadable, COPY, A_LIST, 3) is None
    # Both crashes are exercised separately; establish here that a match is usable.
    assert cd.load_head_list(unreadable, COPY, A_LIST, 3) == digests


def test_the_two_crashes(tmp_path, monkeypatch):
    folder = tmp_path / "pages"
    folder.mkdir()
    last, pending = folder / f"{COPY}.last", folder / f"{COPY}.pending"
    last.write_bytes(info(A).digests)
    pending.write_bytes(info(B).digests)
    assert cd.load_head_list(folder, COPY, A_LIST, 3) == info(A).digests
    assert last.read_bytes() == info(A).digests and pending.read_bytes() == info(B).digests
    assert cd.load_head_list(folder, COPY, B_LIST, 4) == info(B).digests
    assert last.read_bytes() == info(B).digests and not pending.exists()
    cd.save_pending_list(folder, COPY, info(A).digests)
    assert last.read_bytes() == info(B).digests
    def failed_replace(*args):
        raise OSError("injected replace failure")
    monkeypatch.setattr(cd.os, "replace", failed_replace)
    with pytest.raises(OSError):
        cd.save_pending_list(folder, COPY, info(C, 65536).digests)
    assert pending.read_bytes() == info(A).digests
    assert last.read_bytes() == info(B).digests


def test_copy_id_checked_first(tmp_path):
    for copy_id in ("../x", COPY.upper(), "0"*32, ""):
        folder = tmp_path / "pages"
        refused("bad_name", cd.load_head_list, folder, copy_id, A_LIST, 3)
        refused("bad_name", cd.save_pending_list, folder, copy_id, info(A).digests)
        refused("bad_name", cd.promote_pending_list, folder, copy_id)
        assert not folder.exists()


def base_args():
    return dict(page_size=512, file_bytes=2048, list_matches=True, forget=False,
                object_missing=False, delta_bytes=0)


def test_base_rule_each_rule_alone():
    block, _ = worked()
    args = base_args()
    assert cd.base_reason(None, **args) == "first"
    assert cd.base_reason(block, **args) is None
    for change, reason in (({"page_size": 4096}, "page_size"), ({"list_matches": False}, "page_list"),
                           ({"forget": True}, "forget"), ({"object_missing": True}, "missing_object"),
                           ({"delta_bytes": 4096}, "size"), ({"max_generations": 2}, "length")):
        assert cd.base_reason(block, **(args | change)) == reason


def test_base_rule_edges():
    block, _ = worked()
    args = base_args()
    assert cd.base_reason(block, **(args | {"delta_bytes": 4096-42})) == "size"
    assert cd.base_reason(block, **(args | {"delta_bytes": 4096-43})) is None
    block["base"]["parts"][0]["bytes"] = 10**20
    assert cd.base_reason(block, **args) is None
    block["deltas"] = [{"objects": []} for _ in range(1022)]
    assert cd.base_reason(block, **args) is None
    block["deltas"].append({"objects": []})
    assert cd.base_reason(block, **args) == "length"
    assert cd.base_reason(block, **args, max_generations=1025) is None
    block, _ = worked()
    assert cd.base_reason(block, **args, size_factor=0) == "size"
    all_set = args | dict(page_size=4096, list_matches=False, forget=True, object_missing=True, delta_bytes=99999)
    assert cd.base_reason(None, **all_set) == "first"
    assert cd.base_reason(block, **(args | {"forget": True, "delta_bytes": 99999})) == "forget"
    del args["forget"]
    with pytest.raises(TypeError):
        cd.base_reason(block, **args)
    del block["deltas"]
    refused("bad_database_block", cd.base_reason, block, **base_args())


def test_store_signature(tmp_path, monkeypatch):
    db = put(tmp_path, A, "lcm.db")
    before = db.stat().st_mtime_ns
    def never_open(*args, **kwargs):
        pytest.fail("store_signature opened a file")
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", never_open)
        patch.setattr(Path, "open", never_open)
        got = cd.store_signature(tmp_path)
    assert got == [[len(A), before], None]
    assert db.read_bytes() == A and db.stat().st_mtime_ns == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lcm.db"]
    wal = put(tmp_path, b"", "lcm.db-wal")
    assert cd.store_signature(tmp_path) != got
    for path in (db, wal):
        old = cd.store_signature(tmp_path)
        path.write_bytes(path.read_bytes()+b"x")
        assert cd.store_signature(tmp_path) != old
        old = cd.store_signature(tmp_path)
        stamp = path.stat().st_mtime_ns
        os.utime(path, ns=(stamp, stamp+1000000000))
        assert cd.store_signature(tmp_path) != old
    got = cd.store_signature(tmp_path)
    assert json.loads(json.dumps(got)) == got
    assert cd.store_signature(tmp_path / "absent") == [None, None]


def skip_args():
    return dict(manual=False, signature=[[1536, 17], None], recorded_signature=[[1536, 17], None],
                decision_pending=False, last_pair=[7, "ab"], pending_pair=[7, "ab"],
                retention_done_generation=7, head_generation=7, last_publish_s=100, now_s=101)


def test_skip_blocker():
    args = skip_args()
    assert cd.skip_blocker(**args) is None
    cases = [("manual", {"manual": True}), ("changed", {"signature": [[1, 17], None]}),
             ("changed", {"recorded_signature": None}), ("changed", {"signature": [None, None]}),
             ("decision_pending", {"decision_pending": True}),
             ("attempt_pending", {"pending_pair": [8, "ab"]}), ("attempt_pending", {"last_pair": None}),
             ("attempt_pending", {"pending_pair": None}),
             ("attempt_pending", {"pending_pair": None, "last_pair": None}),
             ("retention_pending", {"retention_done_generation": 6}),
             ("retention_pending", {"retention_done_generation": None}),
             ("retention_pending", {"head_generation": None}),
             ("ceiling", {"now_s": 86500}), ("ceiling", {"now_s": 99}),
             ("ceiling", {"last_publish_s": None}), ("ceiling", {"ceiling_s": 1})]
    for code, change in cases:
        assert cd.skip_blocker(**(args | change)) == code
    assert cd.skip_blocker(**(args | {"pending_pair": (7, "ab"), "signature": ((1536, 17), None)})) is None
    assert cd.skip_blocker(**(args | {"manual": True, "decision_pending": True})) == "manual"


def test_isolation():
    path = Path(cd.__file__)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name.split(".")[0] for n in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]]
            assert not isinstance(node, ast.ImportFrom) or node.level == 0
            assert set(names) <= sys.stdlib_module_names
            assert not set(names) & {"sqlite3", "time", "socket", "subprocess"}
        assert not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "os" and node.attr == "environ")
    code = "import runpy, sys; m=runpy.run_path(sys.argv[1]); assert m['MAX_OBJECT_BYTES'] == 2**25"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
