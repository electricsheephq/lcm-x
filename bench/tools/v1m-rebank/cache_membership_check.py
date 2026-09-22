#!/usr/bin/env python3
"""Cache-membership check for the F53 re-bank park record (no provider calls).

The dry run stopped at the first validator block before reporting `already_cached`, so the corpus-identity
bar of RUN-SHEET-V1M-REBANK §4.4 — every RAW prepared-m unit is present in the F53 embedding cache — is
established here directly: the cache key is `sha256(exact UTF-8 unit text)` under (provider, model), the same
digest `corpus_privacy_inventory.py` records. Also derives the exact `would_populate` the dry run would have
reported had no unit blocked: the protected digests of the transformed units that are not in the cache.
Usage: cache_membership_check.py <repo> <prepared-dir> <shards-manifest> <cache.sqlite> <inventory.json> <out.json>
"""
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

repo, prepared_dir, shards, cache_path, inventory_path, out_path = sys.argv[1:7]
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402

lme._ensure_hermes_lcm_package()


def open_readonly(path: str) -> sqlite3.Connection:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.execute("select 1 from embedding_cache limit 1").fetchall()
        return con
    except sqlite3.OperationalError:
        return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


# The run's cache identity (run_shard.sh / prewarm_gate.sh: provider voyage, model voyage-context-3). Selected explicitly —
# never "the largest group" — so a cache holding another model's rows cannot report membership for the wrong identity (PR #416 review).
PROVIDER = "voyage"
MODEL = "voyage-context-3"

t0 = time.time()
con = open_readonly(cache_path)
groups = con.execute("select provider, model, count(*) from embedding_cache group by 1, 2").fetchall()
matching = [g for g in groups if g[0] == PROVIDER and g[1] == MODEL]
if len(matching) != 1:
    sys.exit(f"cache has no ({PROVIDER}, {MODEL}) group; groups present: {groups}")
provider, model, cache_rows = matching[0]
cache_digests = {row[0] for row in con.execute(
    "select content_sha256 from embedding_cache where provider = ? and model = ?", (provider, model))}
con.close()

prepared = lme.load_prepared_dataset(Path(prepared_dir), dataset_label="m")
ids = lme.load_shard_question_ids(Path(shards))
raw_unique: set[str] = set()
raw_in_cache = 0
occurrences = 0
for question in prepared.iter_question_ids(ids):
    for text in lme.iter_ingest_embedding_request_units(question):
        occurrences += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in raw_unique:
            continue
        raw_unique.add(digest)
        if digest in cache_digests:
            raw_in_cache += 1

inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
protected_digests = {entry["protected_sha256"] for entry in inventory["changed"]}
protected_in_cache = sum(1 for d in protected_digests if d in cache_digests)
raw_digests_of_changed = {entry["raw_sha256"] for entry in inventory["changed"]}

result = {
    "cache_groups": [{"provider": g[0], "model": g[1], "rows": g[2]} for g in groups],
    "cache_identity_checked": {"provider": provider, "model": model, "rows": cache_rows},
    "unit_occurrences": occurrences,
    "raw_unique_units": len(raw_unique),
    "raw_unique_units_in_cache": raw_in_cache,
    "raw_unique_units_missing_from_cache": len(raw_unique) - raw_in_cache,
    "cache_rows_not_in_corpus": cache_rows - raw_in_cache,
    "changed_occurrences": len(inventory["changed"]),
    "changed_raw_unique_units": len(raw_digests_of_changed),
    "changed_protected_unique_digests": len(protected_digests),
    "changed_protected_digests_in_cache": protected_in_cache,
    "would_populate_if_unblocked": len(protected_digests) - protected_in_cache,
    "blocked_occurrences": inventory["blocked_units"],
    "elapsed_s": round(time.time() - t0, 1),
}
Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, sort_keys=True))
