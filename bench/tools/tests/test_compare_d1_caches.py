"""Tiny synthetic inputs for the read-only D1 comparison CLI."""
import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from importlib import import_module

import pytest

from benchmarking.d1_cache import D1Cache
from benchmarking.longmemeval import StubEmbedder

compare = import_module("bench.tools.d1-shard-check.compare_d1_caches")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("fault", [None, "identity", "only_serial", "only_other", "vector"])
def test_caches_readonly_seals_and_differences(tmp_path, capsys, fault):
    paths = [tmp_path / name for name in ("serial.sqlite", "other.sqlite")]
    for i, path in enumerate(paths):
        payload = struct.pack("<2d", *((0.0, 1.0) if i and fault == "vector" else (1.0, 0.0)))
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE binding (identity TEXT)")
            db.execute("CREATE TABLE entries (key TEXT, dim INTEGER, payload BLOB, checksum TEXT)")
            db.execute("INSERT INTO binding VALUES (?)", ("other" if i and fault == "identity" else "same",))
            db.execute("INSERT INTO entries VALUES ('k',2,?,?)", (payload, hashlib.sha256(b"k" + payload).hexdigest()))
            if fault == ("only_serial" if i else "only_other"):
                db.execute("DELETE FROM entries")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert compare.main(["caches", *map(str, paths)]) == int(fault is not None)
    out = json.loads(capsys.readouterr().out)
    assert out["seal_equal"] is (fault is None)
    assert out["identity_equal"] is (fault != "identity")
    assert out["only_serial"] == int(fault == "only_serial")
    assert out["only_other"] == int(fault == "only_other")
    assert out["checksum_mismatches"] == int(fault == "vector")
    assert out["max_abs_diff"] == (1.0 if fault == "vector" else 0.0)
    assert out["min_cosine"] == (0.0 if fault == "vector" else 1.0)
    identity, rows, digest = compare.load_cache(paths[0])
    manifest = json.dumps({"identity": identity, "entries": sorted((k, r[2]) for k, r in rows.items())}, separators=(",", ":"))
    assert digest == hashlib.sha256(manifest.encode()).hexdigest()
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def cache_file(path, vector):
    payload = struct.pack(f"<{len(vector)}d", *vector)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE binding (identity TEXT)")
        db.execute("INSERT INTO binding VALUES ('same')")
        db.execute("CREATE TABLE entries (key TEXT, dim INTEGER, payload BLOB, checksum TEXT)")
        db.execute("INSERT INTO entries VALUES ('k',?,?,?)", (len(vector), payload, hashlib.sha256(b"k" + payload).hexdigest()))
    return path


@pytest.mark.parametrize("side", ["serial", "other"])
def test_caches_changed_payload_with_unchanged_checksum(tmp_path, capsys, side):
    paths = {name: cache_file(tmp_path / f"{name}.sqlite", [1.0, 0.0]) for name in ("serial", "other")}
    with closing(sqlite3.connect(paths[side])) as db, db:
        db.execute("UPDATE entries SET payload=?", (struct.pack("<2d", 0.0, 1.0),))
    assert compare.main(["caches", *map(str, paths.values())]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["payload_mismatches"] == 1 and out["checksum_mismatches"] == 0
    assert out["corrupt_serial"] == int(side == "serial")
    assert out["corrupt_other"] == int(side == "other")
    assert out["max_abs_diff"] == 1.0 and out["min_cosine"] == 0.0


def test_caches_dimension_mismatch_skips_vector_metrics(tmp_path, capsys):
    a = cache_file(tmp_path / "serial.sqlite", [1.0, 0.0])
    b = cache_file(tmp_path / "other.sqlite", [0.0, 1.0, 1.0])
    assert compare.main(["caches", str(a), str(b)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["dim_mismatches"] == out["payload_mismatches"] == 1
    assert out["corrupt_serial"] == out["corrupt_other"] == 0
    assert out["max_abs_diff"] == 0.0 and out["min_cosine"] == 1.0


def test_caches_invalid_payload_length_is_corrupt(tmp_path, capsys):
    a = cache_file(tmp_path / "serial.sqlite", [1.0, 0.0])
    b = cache_file(tmp_path / "other.sqlite", [0.0, 1.0])
    with closing(sqlite3.connect(b)) as db, db:
        db.execute("UPDATE entries SET dim=3")
    assert compare.main(["caches", str(a), str(b)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["corrupt_serial"] == 0 and out["corrupt_other"] == 1


def test_caches_d1_writer_checksum_matches(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LCM_D1_BUILD_PARTITION", raising=False)
    monkeypatch.delenv("LCM_D1_DISPATCH_LOG", raising=False)
    path = tmp_path / "cache.sqlite"
    D1Cache(StubEmbedder(), path, options={"dimension": 64}, readonly=False).embed_documents(["unit"])
    assert compare.main(["caches", str(path), str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["corrupt_serial"] == out["corrupt_other"] == 0


@pytest.mark.parametrize("fault", [None, "duplicate_serial_calls", "duplicate_part_calls", "only_serial", "only_parts", "count_mismatches", "hash_mismatches"])
def test_dispatch_union_diagnostics(tmp_path, capsys, fault):
    serial = [{"call": i, "sent": 100, "keys_sha256": "a" * 64} for i in range(2)]
    parts = [dict(row) for row in serial]
    if fault == "duplicate_serial_calls":
        serial.append(serial[0])
    if fault == "duplicate_part_calls":
        parts.append(parts[0])
    if fault == "only_serial":
        parts.pop()
    if fault == "only_parts":
        parts.append({**parts[0], "call": 2})
    if fault in ("count_mismatches", "hash_mismatches"):
        parts[0]["sent" if fault == "count_mismatches" else "keys_sha256"] = 0
    paths = [write_jsonl(tmp_path / "s.jsonl", serial), write_jsonl(tmp_path / "p0.jsonl", parts[:1]), write_jsonl(tmp_path / "p1.jsonl", parts[1:])]
    assert compare.main(["dispatch", *map(str, paths)]) == int(fault is not None)
    assert json.loads(capsys.readouterr().out) == {k: int(k == fault) for k in ("duplicate_serial_calls", "duplicate_part_calls", "only_serial", "only_parts", "count_mismatches", "hash_mismatches")}


@pytest.mark.parametrize("fault", [None, "header", "only_a", "only_b", "ranking", "metadata"])
def test_candidates_header_questions_and_rank_order(tmp_path, capsys, fault):
    header = {"__dump_header__": {"provider": "stub", "top_k": 10}}
    row = {"question_id": "q0", "category": "synthetic", "abstention": False, "gold_sessions": [], "gold_turns": [], "arms": {"lcm_recall": {"sessions_top10": ["s0", "s1"], "turns_top10": [["s0", 0]]}}}
    other = {**row, "arms": {"lcm_recall": {**row["arms"]["lcm_recall"], "sessions_top10": ["s1", "s0"]}}} if fault == "ranking" else {**row, "category": "changed"} if fault == "metadata" else row
    a = write_jsonl(tmp_path / "a.jsonl", [header, row])
    b = write_jsonl(tmp_path / "b.jsonl", [{"__dump_header__": {}} if fault == "header" else header, *([] if fault == "only_a" else [other]), *([{**row, "question_id": "q1"}] if fault == "only_b" else [])])
    assert compare.main(["candidates", str(a), str(b)]) == int(fault is not None)
    out = json.loads(capsys.readouterr().out)
    assert out == {"header_equal": fault != "header", "only_a": int(fault == "only_a"), "only_b": int(fault == "only_b"), "ranked_mismatches": int(fault == "ranking"), "files_equal": fault is None}


@pytest.mark.parametrize("rows,ok", [([{"call": 0, "keys": 3, "mismatches": 0}], True), ([{"call": 0, "keys": 3, "mismatches": 1}], False),
                                     ([{"call": 0, "keys": 3, "mismatches": 0}] * 2, False), ([], False)])
def test_verify_summarises_logs(tmp_path, capsys, rows, ok):
    assert compare.main(["verify", str(write_jsonl(tmp_path / "verify.jsonl", rows))]) == int(not ok)
    out = json.loads(capsys.readouterr().out)
    assert (out["calls"], out["mismatches"]) == (len(rows), sum(row["mismatches"] for row in rows))
