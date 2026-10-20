"""Complete identity-scoped coverage and immutable-preimage audit, read-only SQLite."""
import hashlib
import json
import sqlite3

from .common import dump, load, parser


def digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps([v.hex() if isinstance(v, bytes) else v for v in row],
                            separators=(",", ":")).encode() + b"\n")
    return h.hexdigest()


def connect(db):
    c = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    c.execute("PRAGMA query_only=ON")
    return c


def preimage(c):
    raw = digest(c.execute("SELECT * FROM messages ORDER BY store_id"))
    vectors = {}
    for kind, table, key in (("chunks", "lcm_chunk_vectors", "chunk_id"),
                             ("leaves", "lcm_embedding_vectors", "embedded_id")):
        for ident, identity, vec in c.execute(f"SELECT {key},identity_hash,vec FROM {table}"):
            vectors[f"{kind}/{identity}/{ident}"] = hashlib.sha256(vec).hexdigest()
    return {"raw_digest": raw, "vectors": vectors}


def audit(db, manifest):
    with connect(db) as c:
        before = manifest["preimage"]
        now = preimage(c)
        changed = [k for k, v in before["vectors"].items() if now["vectors"].get(k) != v]
        result = {"raw_unchanged": now["raw_digest"] == before["raw_digest"],
                  "changed_vectors": changed, "classes": {}}
        for kind, table, meta, key, task in (
            ("chunks", "lcm_chunk_vectors", "lcm_chunk_meta", "chunk_id", "chunk"),
            ("leaves", "lcm_embedding_vectors", "lcm_embedding_meta", "embedded_id", "summary")):
            identity = manifest["identities"][task]
            actual = {str(r[0]) for r in c.execute(
                f"SELECT v.{key} FROM {table} v JOIN {meta} m USING ({key},identity_hash) "
                "WHERE v.identity_hash=? AND m.archived=0 AND length(v.vec)=?",
                (identity, manifest["dim"] * 4))}
            expected = set(manifest["expected"][kind])
            missing = sorted(expected - actual)
            result["classes"][kind] = {"covered": len(expected & actual), "expected": len(expected),
                "missing_count": len(missing), "missing_first20": missing[:20]}
            result["classes"][kind]["covered_tokens"] = sum(manifest["expected"][kind][i] for i in expected & actual)
            if kind == "chunks":
                result["covered_rows"] = len({i.split(":")[0] for i in expected} - {i.split(":")[0] for i in missing})
        exists = c.execute("SELECT 1 FROM sqlite_master WHERE name='lcm_embedding_backfill_inflight'").fetchone()
        result["residue"] = dict(c.execute("SELECT state,count(*) FROM lcm_embedding_backfill_inflight GROUP BY state")) if exists else {}
        result["scan"] = list(c.execute("SELECT key,value FROM metadata WHERE key LIKE 'chunk_gapfill_scan:%' OR key LIKE 'chunk_policy_revision_through:%' ORDER BY key"))
        result["integrity"] = c.execute("PRAGMA quick_check").fetchone()[0]
    result["complete"] = (all(v["missing_count"] == 0 for v in result["classes"].values())
        and not result["residue"] and result["raw_unchanged"] and not changed and result["integrity"] == "ok")
    return result


def progress(db, manifest):
    result = {"classes": {}, "raw_unchanged": True, "changed_vectors": [], "integrity": "not_checked"}
    with connect(db) as c:
        for kind, table, meta, key, task in (("chunks", "lcm_chunk_vectors", "lcm_chunk_meta", "chunk_id", "chunk"),
                                           ("leaves", "lcm_embedding_vectors", "lcm_embedding_meta", "embedded_id", "summary")):
            expected = manifest["expected"][kind]
            count, tokens, missing = c.execute(f"SELECT count(v.{key}),coalesce(sum(CASE WHEN v.{key} IS NOT NULL THEN e.value END),0),group_concat(CASE WHEN v.{key} IS NULL THEN e.key END) FROM json_each(?) e LEFT JOIN {meta} m ON m.{key}=e.key AND m.identity_hash=? AND m.archived=0 LEFT JOIN {table} v ON v.{key}=m.{key} AND v.identity_hash=m.identity_hash AND length(v.vec)=?", (json.dumps(expected), manifest["identities"][task], manifest["dim"]*4)).fetchone()
            result["classes"][kind] = dict(covered=count, expected=len(expected), covered_tokens=tokens, missing_count=len(expected)-count, missing_first20=(missing or "").split(",")[:20] if missing else [])
            if kind == "chunks":
                result["covered_rows"] = len({i.split(":")[0] for i in expected}-{i.split(":")[0] for i in (missing or "").split(",")})
        result["scan"] = list(c.execute("SELECT key,value FROM metadata WHERE key LIKE 'chunk_gapfill_scan:%' OR key LIKE 'chunk_policy_revision_through:%' ORDER BY key"))
    result["complete"] = all(v["missing_count"] == 0 for v in result["classes"].values())
    return result


def baseline(db, manifest):
    result = audit(db, manifest)
    with connect(db) as c:
        count = sum(c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("lcm_chunk_vectors", "lcm_embedding_vectors"))
    if (not result["raw_unchanged"] or result["changed_vectors"] or result["residue"] or result["integrity"] != "ok"
            or count != len(manifest["preimage"]["vectors"]) or result["classes"] != manifest["baseline_classes"]):
        raise ValueError("store does not match manifest baseline")
    return result


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--db", type=__import__("pathlib").Path, required=True)
    p.add_argument("--manifest", type=__import__("pathlib").Path, required=True)
    p.add_argument("--out", type=__import__("pathlib").Path, required=True)
    a = p.parse_args()
    dump(a.out, audit(a.db, load(a.manifest)))
