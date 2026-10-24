"""Serial warm LOCAL exact-recall contention diagnostic on a synthetic store.

Each latency invocation interleaves class a canaries and class b broad queries.
Busy Python threads contend for the GIL; they do not emulate CPU allocation.
Profile invocations are separate diagnostics, never latency measurements.
"""
import argparse
import collections
import cProfile
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import resource
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--hermes-source', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--busy-threads', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--recall-timeout-s', type=float,
                   help='Harness-only override of recall_query_timeout_s for uncensored exactness')
    a = p.parse_args()
    from bench.instruments.local_catchup.common import product, isolated, config, safe_paths
    paths = (a.repo, a.hermes_source, a.model_cache, a.store, a.home, a.out)
    safe_paths(*paths)
    if any('.hermes' in path.parts for path in paths):
        p.error('live .hermes paths forbidden')
    if a.n < 1 or a.offset < 0:
        p.error('--n must be positive and --offset must be nonnegative')
    if a.recall_timeout_s is not None and (
            not math.isfinite(a.recall_timeout_s) or a.recall_timeout_s <= 0):
        p.error('--recall-timeout-s must be finite and positive')
    if any(path.exists() or path.is_symlink() for path in (a.home, a.out)):
        p.error('--home and --out must not exist')
    if a.home.resolve() == a.out.resolve():
        p.error('--home and --out must be distinct')
    for writable in (a.home.resolve(), a.out.resolve()):
        for other in paths:
            if other in (a.home, a.out) and other.resolve() == writable:
                continue
            resolved = other.resolve()
            if writable.is_relative_to(resolved) or resolved.is_relative_to(writable):
                p.error('--home and --out must be disjoint from all other paths')
    if sys.platform not in ('darwin', 'linux'):
        p.error('peak RSS conversion supports macOS and Linux only')
    code = {}
    for name, repo in (('product', a.repo), ('hermes', a.hermes_source)):
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
        if head.returncode:
            if name != 'hermes':
                raise RuntimeError('product must be a Git checkout')
            stub = repo / 'agent' / 'context_engine.py'
            code[name] = dict(sha=None, contract_sha256=hashlib.sha256(stub.read_bytes()).hexdigest())
        else:
            code[name] = dict(sha=head.stdout.strip(), dirty=bool(subprocess.check_output(
                ['git', '-C', str(repo), 'status', '--porcelain'], text=True)))
    code['instrument_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    isolated(a.home, a.model_cache)
    os.environ.update(HOME=str(a.home), CFFIXED_USER_HOME=str(a.home),
                      ORT_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                      MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for name in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'TMPDIR'):
        target = a.home / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    def deny(*args, **kwargs):
        raise RuntimeError('offline-only diagnostic: network forbidden')
    socket.socket.connect = socket.socket.connect_ex = deny
    socket.socket.sendto = socket.socket.sendmsg = deny
    socket.getaddrinfo = socket.gethostbyname = socket.gethostbyname_ex = deny
    socket.gethostbyaddr = socket.getnameinfo = deny
    mod = product(a.repo, a.hermes_source)
    cfg = config(mod, a.store / 'store.db')
    if a.recall_timeout_s is not None:
        cfg = dataclasses.replace(cfg, recall_query_timeout_s=a.recall_timeout_s)
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    provider = mod('embedding_provider').resolve_provider(cfg)
    ensure = provider._ensure_local
    ensure(**({'allow_download': False} if 'allow_download' in inspect.signature(ensure).parameters else {}))
    tools = mod('tools')
    reader = SimpleNamespace(_config=cfg, _hermes_home=str(a.home),
        _store=mod('store').MessageStore(a.store/'store.db'),
        _dag=mod('dag').SummaryDAG(a.store/'store.db'), current_session_id='i2-recall')
    reader._lcm_embedding_provider_cache = (('fastembed', 'BAAI/bge-small-en-v1.5'), provider)
    manifest = json.loads((a.store/'manifest.json').read_text())
    order = list(range(len(manifest['canaries'])))
    random.Random(1586).shuffle(order)
    broad = ('What turn was acknowledged', 'Recall the filler from a turn', 'Which turn contains a value')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'identity.json').write_text(json.dumps(dict(repo=str(a.repo), store=str(a.store),
        code=code, config=dataclasses.asdict(cfg), ort_threads=mod('embedding_provider')._FASTEMBED_THREADS,
        env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS',
            'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        cpu_affinity=False, compaction=False, offset=a.offset, n_per_class=a.n,
        shuffle_seed=1586, busy_threads=a.busy_threads, profile=a.profile), indent=2))

    profiles = []
    profile_done = []
    spans = []
    sqlite_spans = []
    recording = False
    def timed(owner, name):
        original = getattr(owner, name, None)
        if original is None:
            return
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            finally:
                if recording:
                    meta = {}
                    if name == '_bounded_chunk_candidate_ids' and result is not None:
                        meta['returned_ids'] = len(result)
                    if name == '_load_chunk_matrix' and result is not None:
                        meta['matrix_rows'] = len(result[1])
                    if name == '_scan_ranked' and result is not None:
                        meta.update(scanned=result[1], stopped=result[2])
                    if name == '_binary_fully_synced':
                        meta['eligible'] = result
                    spans.append(dict(name=name, start=start, end=time.perf_counter(), thread=threading.current_thread().name, **meta))
        setattr(owner, name, wrapped)
    if a.profile:
        from contextlib import contextmanager
        core = mod('retrieval_core')
        original_lease = core._vector_store_lease
        @contextmanager
        def lease(entry):
            start = time.perf_counter()
            with original_lease(entry):
                if recording:
                    spans.append(dict(name='pool_lock_acquire',start=start,end=time.perf_counter(),
                                      thread=threading.current_thread().name))
                yield
        core._vector_store_lease = lease
        class Cursor(sqlite3.Cursor):
            def execute(self, sql, parameters=()):
                text = sql.lower()
                self.category = ('chunk BLOB' if 'v.vec' in text and 'lcm_chunk_vectors' in text
                    else 'chunk IDs' if 'select cm.chunk_id' in text
                    else 'other SQL')
                start = time.perf_counter()
                try:
                    return super().execute(sql, parameters)
                finally:
                    if recording:
                        sqlite_spans.append(dict(category=self.category, op='execute', seconds=time.perf_counter()-start))
            def fetchall(self):
                start = time.perf_counter()
                try:
                    rows = super().fetchall()
                    return rows
                finally:
                    if recording:
                        sqlite_spans.append(dict(category=getattr(self,'category','other SQL'),op='fetchall',
                            seconds=time.perf_counter()-start, rows=len(rows) if 'rows' in locals() else None))
            def fetchone(self):
                start = time.perf_counter()
                try:
                    return super().fetchone()
                finally:
                    if recording:
                        sqlite_spans.append(dict(category=getattr(self,'category','other SQL'),op='fetchone',
                            seconds=time.perf_counter()-start))
        class Connection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                return self.cursor(factory=Cursor).execute(sql,parameters)
        original_connect = sqlite3.connect
        def connect(*args, **kwargs):
            kwargs.setdefault('factory',Connection)
            return original_connect(*args, **kwargs)
        sqlite3.connect = connect
        original = tools._run_within_deadline
        profile_guard = threading.Lock()
        def dispatch(fn, *args, **kwargs):
            def profiled():
                if not recording or not profile_guard.acquire(blocking=False):
                    return fn()
                pr = cProfile.Profile()
                done = threading.Event()
                profiles.append(pr)
                profile_done.append(done)
                try:
                    return pr.runcall(fn)
                finally:
                    done.set()
                    profile_guard.release()
            return original(profiled, *args, **kwargs)
        tools._run_within_deadline = dispatch
        for name in ('_lcm_recall_fts_arm','_lcm_recall_summary_arm','_lcm_recall_chunk_arm',
                     'rrf_fuse','apply_prior','_embed_query_with_deadline'):
            timed(tools, name)
        vs = mod('vector_store').VectorStore
        for name in ('_bounded_chunk_candidate_ids','_load_chunk_matrix',
                     '_load_chunk_vectors_for_ids','_numpy_chunk_rows','_scan_ranked',
                     '_binary_fully_synced','_two_stage_rank','knn_chunks','knn'):
            timed(vs, name)

    fts = {}
    original_fts = tools._lcm_recall_fts_arm
    def fts_arm(*args, **kwargs):
        start = time.perf_counter()
        try:
            hits, error = original_fts(*args, **kwargs)
            fts.update(hits=len(hits), error=error)
            return hits, error
        finally:
            fts['seconds'] = time.perf_counter()-start
    tools._lcm_recall_fts_arm = fts_arm
    halt = threading.Event()
    def burn():
        while not halt.is_set():
            sum(range(10000))
    busy = [threading.Thread(target=burn, name=f'diagnostic-gil-load-{i}')
            for i in range(a.busy_threads)]
    try:
        # Warm inference/pool before measurement, record separately.
        start = time.perf_counter()
        warm = json.loads(tools.lcm_recall({'query': manifest['canaries'][order[0]]['value']}, engine=reader))
        (a.out/'warmup.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,payload=warm),indent=2))
        for thread in busy:
            thread.start()
        rows = []
        recording = a.profile
        if a.profile:
            profiles.clear()
        for i in range(a.offset, a.offset+a.n):
            for cls, query in [('a', manifest['canaries'][order[i % len(order)]]['value']), ('b', broad[i % 3])]:
                if a.profile and cls == 'a':
                    continue
                if __import__('shutil').disk_usage(a.store).free < 40*2**30:
                    raise RuntimeError('A1 disk stop')
                fts.clear()
                start = time.perf_counter()
                payload = json.loads(tools.lcm_recall({'query': query}, engine=reader))
                row = dict(i=i, query_class=cls, query=query, seconds=time.perf_counter()-start,
                    timeout=payload.get('timeout',False), error=payload.get('error'),
                    coverage=payload.get('provenance',{}).get('coverage'), fts=dict(fts), payload=payload)
                rows.append(row)
                with (a.out/'recalls.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
        if profiles:
            for done in profile_done:
                if not done.wait(30):
                    raise RuntimeError('profile worker did not drain')
            recording = False
            import pstats
            completed_profiles = [pr for pr in profiles if pr.getstats()]
            stats = pstats.Stats(completed_profiles[0])
            for pr in completed_profiles[1:]:
                stats.add(pstats.Stats(pr))
            stats.dump_stats(str(a.out/'profile.pstats'))
            with (a.out/'profile.txt').open('w') as f:
                stats.stream = f
                stats.sort_stats('cumulative').print_stats(100)
            (a.out/'spans.json').write_text(json.dumps(spans,indent=2))
            (a.out/'sqlite-spans.json').write_text(json.dumps(sqlite_spans,indent=2))
        recording = False
        summary = {}
        for cls in ('a','b'):
            group = [r for r in rows if r['query_class']==cls]
            if not group:
                continue
            lat = sorted(r['seconds'] for r in group)
            summary[cls] = dict(n=len(lat),p50=lat[math.ceil(.5*len(lat))-1],p90=lat[math.ceil(.9*len(lat))-1],
                max=max(lat),timeouts=sum(bool(r['timeout']) for r in group), errors=sum(bool(r['error']) for r in group),
                coverage=dict(collections.Counter(json.dumps(r['coverage'],sort_keys=True) for r in group)),
                fts_none=sum(r['fts'].get('hits',0)==0 for r in group),
                fts_hits=sum(r['fts'].get('hits',0) for r in group))
            fts_lat = sorted(r['fts']['seconds'] for r in group if 'seconds' in r['fts'])
            if fts_lat:
                summary[cls]['fts_latency'] = dict(n=len(fts_lat),
                    p50=fts_lat[math.ceil(.5*len(fts_lat))-1],
                    p90=fts_lat[math.ceil(.9*len(fts_lat))-1], max=max(fts_lat))
        # ru_maxrss is bytes on macOS, KiB on Linux; this is the whole process
        # high-water mark (including preparation and warmup), not a per-query delta.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary['_process'] = dict(peak_rss_bytes=int(rss * (1 if sys.platform == 'darwin' else 1024)),
            peak_rss_native=rss, peak_rss_native_unit='bytes' if sys.platform == 'darwin' else 'KiB',
            busy_threads=a.busy_threads, profile=a.profile)
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
        # Attribution changes allocator/timing behaviour: acceptance ends above.
        halt.set()
        for thread in busy:
            thread.join()
        from bench.instruments.local_catchup.memory_sites import capture
        reader._config = dataclasses.replace(cfg, recall_query_timeout_s=3600)
        capture(a.repo, a.out, lambda: json.loads(tools.lcm_recall(
            {'query': broad[0]}, engine=reader)))
    finally:
        halt.set()
        for thread in busy:
            thread.join()
        reset = getattr(mod('retrieval_core'),'_reset_vector_store_pool',None)
        if reset:
            reset()
        reader._dag.close()
        reader._store.close()


if __name__ == "__main__":
    main()
