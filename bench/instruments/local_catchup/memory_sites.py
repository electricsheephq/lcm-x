"""One untimed allocation pass after each contention run; no corpus values."""
import json
from pathlib import Path
import resource
import sys
import threading
import tracemalloc


def capture(repo, out, query):
    """Record RCA transfer stages separately from the latency/RSS acceptance arm."""
    import psutil
    process = psutil.Process()
    events, seen = [], set()
    source = str(Path(repo) / 'vector_store.py')
    functions = {'_fetch_exact_rows', '_load_chunk_matrix', '_load_vectors_for_ids'}

    def trace(frame, event, arg):
        if frame.f_code.co_filename != source or frame.f_code.co_name not in functions:
            return None
        values = frame.f_locals
        name = frame.f_code.co_name
        if name == '_fetch_exact_rows' and not values.get('vectors', True):
            return None
        group = 'chunk' if name == '_load_chunk_matrix' or 'lcm_chunk_vectors' in values.get('sql', '') else 'summary'
        label = None
        if event == 'line':
            if 'matrix' in values:
                label = 'matrix'
            elif 'decoded' in values and len(values['decoded']) == len(values.get('rows', [])):
                label = 'decoded'
            elif 'raw' in values and 'decoded' not in values:
                label = 'raw'
            elif 'rows' in values and 'raw' not in values:
                label = 'metadata'
            elif 'metadata' in values and 'rows' not in values:
                label = 'aggregate'
        key = (group, name, label)
        if label and key not in seen:
            seen.add(key)
            objects = {}
            for key_name in ('metadata', 'hex_vectors', 'transferred', 'raw', 'rows', 'decoded', 'blobs', 'matrix'):
                value = values.get(key_name)
                if isinstance(value, (bytes, str, list, memoryview)):
                    objects[key_name] = dict(type=type(value).__name__, length=len(value),
                                             shallow_bytes=sys.getsizeof(value))
                elif hasattr(value, 'nbytes'):
                    objects[key_name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                        nbytes=value.nbytes, owns=bool(value.flags.owndata), base_type=type(value.base).__name__)
            events.append(dict(group=group, function=name, stage=label, line=frame.f_lineno,
                rss_bytes=process.memory_info().rss, traced_bytes=tracemalloc.get_traced_memory(),
                objects=objects))
        return trace

    tracemalloc.start(1)
    sys.settrace(trace)
    threading.settrace(trace)
    try:
        payload = query()
    finally:
        sys.settrace(None)
        threading.settrace(None)
    snapshot = tracemalloc.take_snapshot()
    sites = [dict(file=Path(item.traceback[0].filename).name,
                  line=item.traceback[0].lineno, bytes=item.size, count=item.count)
             for item in snapshot.statistics('lineno')[:20]]
    settled = dict(rss_bytes=process.memory_info().rss, traced_bytes=tracemalloc.get_traced_memory())
    tracemalloc.stop()
    result = dict(method='RCA-v1 transfer stages; one separate untimed raised-deadline query',
        acceptance_latency=False, acceptance_rss=False, events=events, settled=settled,
        settled_sites=sites, diagnostic_peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        coverage=payload.get('provenance', {}).get('coverage'), timeout=payload.get('timeout', False))
    (Path(out) / 'memory-attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
