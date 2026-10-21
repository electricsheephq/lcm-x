"""Offline standalone core-call calibration, separate from autonomous evidence."""
from contextlib import closing
import sqlite3
from statistics import median
import time

from .common import MODEL, config, dump, isolated, product


def standalone(a):
    root = a.scratch/"calibration"
    root.mkdir(parents=True)
    isolated(root/"home", a.model_cache)
    __import__("os").environ["HOME"] = str(root/"home")
    mod = product(a.lcmx_repo, a.hermes_source)
    db = root/"template.db"
    mod("store").MessageStore(db).close()
    dag = mod("dag").SummaryDAG(db)
    for i in range(32):
        dag.add_node(mod("dag").SummaryNode(session_id="calibration", summary=f"Fixed synthetic leaf {i}.", token_count=8))
    dag.close()
    vectors = mod("vector_store").VectorStore(db, config=config(mod, db))
    vectors.register_profile(MODEL, "fastembed", 384, task="summary")
    vectors.close()
    start = time.perf_counter()
    mod("embedding_provider").resolve_provider(config(mod, db), for_backfill=True)._ensure_local(allow_download=False)
    preparation_s = time.perf_counter()-start
    times = []
    for i in range(5):
        target = root/f"batch-{i}.db"
        with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
        start = time.perf_counter()
        result = mod("command")._embedding_backfill_summary_run(config(mod, target), target,
            apply=True, limit=32, retry_uncertain=False, include_next_hint=False)
        times.append(time.perf_counter()-start)
        if not isinstance(result, dict) or result.get("embedded") != 32:
            raise RuntimeError("standalone calibration did not write 32 vectors")
        target.unlink()
    value = dict(core_seconds=times, standalone_calibration_s=median(times), reference_standalone_s=None, model_preparation_s=preparation_s)
    dump(a.out/"calibration.json", value)
    return value

