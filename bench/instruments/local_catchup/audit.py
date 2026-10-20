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


def preimage(c, node_bound=None):
    raw = digest(c.execute("SELECT * FROM messages ORDER BY store_id"))
    provenance = {table: digest(c.execute(f"SELECT * FROM {table}" + (f" WHERE node_id<={int(node_bound)}" if node_bound is not None else "") + f" ORDER BY {order}")
                               if c.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() else [])
                  for table, order in (("summary_nodes", "node_id"), ("summary_node_provenance", "node_id"),
                                       ("lcm_rollup_sources", "rollup_id,node_id"))}
    vectors, metadata = {}, {}
    for kind, table, key in (("chunks", "lcm_chunk_vectors", "chunk_id"),
                             ("leaves", "lcm_embedding_vectors", "embedded_id")):
        for ident, identity, vec in c.execute(f"SELECT {key},identity_hash,vec FROM {table}"):
            vectors[f"{kind}/{identity}/{ident}"] = hashlib.sha256(vec).hexdigest()
        meta = "lcm_chunk_meta" if kind == "chunks" else "lcm_embedding_meta"
        cur = c.execute(f"SELECT * FROM {meta}")
        columns = [d[0] for d in cur.description]
        for row in cur:
            fields = dict(zip(columns, row))
            suffix = "/"+fields["embedded_kind"] if kind == "leaves" else ""
            metadata[f"{kind}/{fields['identity_hash']}/{fields[key]}{suffix}"] = digest([row])
    return {"raw_digest": raw, "vectors": vectors, "metadata": metadata, "provenance": provenance}


def audit(db, manifest):
    with connect(db) as c:
        c.execute("BEGIN")
        before = manifest["preimage"]
        now = preimage(c)
        observed = manifest.setdefault("observed_vectors", {}) if now["vectors"] else manifest.get("observed_vectors", {})
        changed = [k for k, v in (observed | before["vectors"]).items() if now["vectors"].get(k) != v]
        for k, v in now["vectors"].items():
            observed.setdefault(k, v)
        changed_meta = [k for k, v in (manifest.get("observed_metadata", {}) | before["metadata"]).items() if now["metadata"].get(k) != v]
        expected_ids = {f"{kind}/{manifest['identities'][task]}/{ident}"
                        for kind, task in (("chunks", "chunk"), ("leaves", "summary"))
                        for ident in manifest["expected"][kind]}
        expected_meta = {k+("/summary" if k.startswith("leaves/") else "") for k in expected_ids}
        unexpected = sorted((now["vectors"].keys() - (before["vectors"].keys() | expected_ids)) |
                            (now["metadata"].keys() - (before["metadata"].keys() | expected_meta)))
        new_vectors = now["vectors"].keys() - before["vectors"].keys()
        new_meta = now["metadata"].keys() - before["metadata"].keys()
        paired_meta = {k.removesuffix("/summary") if k.startswith("leaves/") else k for k in new_meta}
        unpaired = sorted(new_vectors ^ paired_meta)
        result = {"changed_metadata": changed_meta, "unexpected_rows": unexpected, "unpaired_rows": unpaired,
                  "raw_unchanged": now["raw_digest"] == before["raw_digest"] and now["provenance"] == before["provenance"],
                  "changed_vectors": changed, "classes": {}}
        for kind, table, meta, key, task in (
            ("chunks", "lcm_chunk_vectors", "lcm_chunk_meta", "chunk_id", "chunk"),
            ("leaves", "lcm_embedding_vectors", "lcm_embedding_meta", "embedded_id", "summary")):
            identity = manifest["identities"][task]
            fields = "store_id,chunk_index,char_start,char_end,token_estimate" if kind == "chunks" else "source_token_count"
            for row in c.execute(f"SELECT {key},{fields} FROM {meta} WHERE identity_hash=?", (identity,)):
                ident = str(row[0])
                if ident in manifest["expected"][kind] and list(row[1:]) != manifest["expected_metadata"][kind][ident]:
                    changed_meta.append(f"{kind}/{identity}/{ident}")
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
        and not result["residue"] and result["raw_unchanged"] and not changed and not changed_meta and not unexpected and not unpaired and result["integrity"] == "ok")
    return result


def progress(db, manifest, verify=False):
    result = {"classes": {}, "raw_unchanged": True, "changed_vectors": [], "changed_metadata": [], "integrity": "not_checked"}
    with connect(db) as c:
        c.execute("BEGIN")
        if verify:
            now, before = preimage(c), manifest["preimage"]
            result.update(raw_unchanged=now["raw_digest"] == before["raw_digest"] and now["provenance"] == before["provenance"],
                changed_vectors=[k for k, v in (manifest.get("observed_vectors", {}) | before["vectors"]).items() if now["vectors"].get(k) != v],
                changed_metadata=[k for k, v in (manifest.get("observed_metadata", {}) | before["metadata"]).items() if now["metadata"].get(k) != v])
            observed_meta = manifest.setdefault("observed_metadata", {})
            for k in now["metadata"].keys() - before["metadata"].keys():
                observed_meta.setdefault(k, now["metadata"][k])
        observed = manifest.setdefault("observed_vectors", {})
        for kind, table, key in (("chunks", "lcm_chunk_vectors", "chunk_id"), ("leaves", "lcm_embedding_vectors", "embedded_id")):
            known = dict(manifest["preimage"]["vectors"], **observed)
            for ident, identity, vec in c.execute(f"SELECT {key},identity_hash,vec FROM {table} WHERE ?||'/'||identity_hash||'/'||{key} NOT IN (SELECT key FROM json_each(?))", (kind, json.dumps(known))):
                observed[f"{kind}/{identity}/{ident}"] = hashlib.sha256(vec).hexdigest()
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


def vector_counts(db, manifest):
    """Cheap identity counts within the generated corpus; exclude new trace rows/leaves.

    The builder allocates contiguous store/node IDs from 1. Boundary expected-ID
    audits remain authoritative; these counts only track incompleteness in flight.
    """
    counts = {}
    with connect(db) as c:
        for kind, table, meta, key, task, scope, bound in (
            ("chunks", "lcm_chunk_vectors", "lcm_chunk_meta", "chunk_id", "chunk", "m.store_id", manifest["rows"]),
            ("leaves", "lcm_embedding_vectors", "lcm_embedding_meta", "embedded_id", "summary", "CAST(m.embedded_id AS INTEGER)", len(manifest["expected"]["leaves"]))):
            counts[kind] = c.execute(f"SELECT count(*) FROM {table} v JOIN {meta} m USING ({key},identity_hash) "
                f"WHERE v.identity_hash=? AND m.archived=0 AND length(v.vec)=? AND {scope} BETWEEN 1 AND ?",
                (manifest["identities"][task], manifest["dim"]*4, bound)).fetchone()[0]
    return counts


def baseline(db, manifest):
    result = audit(db, manifest)
    with connect(db) as c:
        count = sum(c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("lcm_chunk_vectors", "lcm_embedding_vectors"))
    if (not result["raw_unchanged"] or result["changed_vectors"] or result["changed_metadata"]
            or result["unexpected_rows"] or result["unpaired_rows"] or result["residue"] or result["integrity"] != "ok"
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
