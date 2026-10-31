"""Read-only D1 cache, dispatch, ranked-candidate and verify-log comparisons."""
import argparse
import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from math import sqrt
from pathlib import Path


def load_cache(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity = db.execute("SELECT identity FROM binding").fetchone()[0]
        rows = {k: (d, p, c) for k, d, p, c in db.execute("SELECT key, dim, payload, checksum FROM entries")}
    manifest = json.dumps({"identity": identity, "entries": sorted((k, row[2]) for k, row in rows.items())}, separators=(",", ":"))
    return identity, rows, hashlib.sha256(manifest.encode()).hexdigest()


def caches(serial, other):
    si, s, sh = load_cache(serial)
    oi, o, oh = load_cache(other)
    common = s.keys() & o.keys()
    diff = [k for k in common if s[k][2] != o[k][2]]
    dim_mismatches = sum(s[k][0] != o[k][0] for k in common)
    payload_mismatches = sum(s[k][:2] != o[k][:2] for k in common)
    corrupt_serial, corrupt_other = (
        sum(hashlib.sha256(k.encode() + payload).hexdigest() != checksum or len(payload) != 8 * dim
            for k, (dim, payload, checksum) in rows.items())
        for rows in (s, o)
    )
    worst_abs, worst_cos = 0.0, 1.0
    for k in common:
        if s[k] == o[k] or s[k][0] != o[k][0] or any(len(row[1]) != 8 * row[0] for row in (s[k], o[k])):
            continue
        a, b = (struct.unpack(f"<{row[0]}d", row[1]) for row in (s[k], o[k]))
        worst_abs = max(worst_abs, max((abs(x - y) for x, y in zip(a, b)), default=0.0))
        norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
        worst_cos = min(worst_cos, sum(x * y for x, y in zip(a, b)) / norm if norm else float(a == b))
    out = {"identity_equal": si == oi, "serial_entries": len(s), "other_entries": len(o),
           "only_serial": len(s.keys() - o.keys()), "only_other": len(o.keys() - s.keys()),
           "checksum_mismatches": len(diff), "max_abs_diff": worst_abs, "min_cosine": worst_cos, "seal_equal": sh == oh,
           "dim_mismatches": dim_mismatches, "payload_mismatches": payload_mismatches,
           "corrupt_serial": corrupt_serial, "corrupt_other": corrupt_other}
    return out, si == oi and s.keys() == o.keys() and not (payload_mismatches or diff or corrupt_serial or corrupt_other)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def dispatch(serial, *parts):
    left, right = jsonl(serial), [row for path in parts for row in jsonl(path)]
    s, p = ({row["call"]: row for row in rows} for rows in (left, right))
    common = s.keys() & p.keys()
    out = {"duplicate_serial_calls": len(left) - len(s), "duplicate_part_calls": len(right) - len(p),
           "only_serial": len(s.keys() - p.keys()), "only_parts": len(p.keys() - s.keys()),
           "count_mismatches": sum(s[k]["sent"] != p[k]["sent"] for k in common),
           "hash_mismatches": sum(s[k]["keys_sha256"] != p[k]["keys_sha256"] for k in common)}
    return out, not any(out.values())


def candidates(a, b):
    left, right = jsonl(a), jsonl(b)
    s, p = ({row["question_id"]: row for row in rows[1:]} for rows in (left, right))
    out = {"header_equal": left[0] == right[0], "only_a": len(s.keys() - p.keys()), "only_b": len(p.keys() - s.keys()),
           "ranked_mismatches": sum(s[k]["arms"] != p[k]["arms"] for k in s.keys() & p.keys()),
           "files_equal": Path(a).read_bytes() == Path(b).read_bytes()}
    return out, out["files_equal"]


def verify(*logs):
    rows = [row for path in logs for row in jsonl(path)]
    out = {"calls": len(rows), "duplicate_calls": len(rows) - len({row["call"] for row in rows}),
           "keys": sum(row["keys"] for row in rows), "mismatches": sum(row["mismatches"] for row in rows),
           "mismatched_calls": sorted(row["call"] for row in rows if row["mismatches"])}
    return out, bool(rows) and not (out["duplicate_calls"] or out["mismatches"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("caches", "dispatch", "candidates"):
        command = commands.add_parser(name)
        command.add_argument("serial", type=Path)
        command.add_argument("other", type=Path, nargs="+" if name == "dispatch" else None)
    commands.add_parser("verify").add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "verify":
        out, equal = verify(*args.logs)
    else:
        out, equal = dispatch(args.serial, *args.other) if args.command == "dispatch" else globals()[args.command](args.serial, args.other)
    print(json.dumps(out, separators=(",", ":")))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
