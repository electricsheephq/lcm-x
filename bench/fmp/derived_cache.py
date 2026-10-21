"""Run-owned subscription cache; append-only send receipts, no uncertain replay."""
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
from .common import sha
from .dev_cache import Incomplete, request_identity
from .dev_panel import encoded
from .models import input_bound, parse_spec
def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    try:
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise Incomplete('immutable input changed') from None
    return value
class Cache:
    def __init__(self, root, backend, endpoints, deadline, renderer):
        self.root, self.backend = Path(root), backend
        self.endpoints, self.deadline, self.renderer = endpoints, deadline, renderer
        self.root.mkdir(parents=True, exist_ok=True)
    def call(self, spec, messages, records, *, role, rubric, sources, max_tokens, temperature=None):
        family, _, effort = parse_spec(spec)
        if family not in ('codex', 'glm') or not self.endpoints.get(family):
            raise Incomplete('subscription endpoint identity required')
        endpoint = dict(family=family, adapter='dm0-cache-v1', **self.endpoints[family])
        identity = request_identity(spec, messages, endpoint=endpoint, effort=effort, temperature=temperature,
            max_tokens=max_tokens, tools=None, tool_choice='none', rubric=rubric, renderer=self.renderer, sources=sources)
        key = sha(encoded(identity))
        prefix = self.root / key
        with prefix.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done, pending = prefix.with_suffix('.ok.json'), prefix.with_suffix('.pending.json')
            if done.exists():
                saved = json.loads(done.read_bytes())
                if saved['identity'] != identity or sha(encoded(saved['response'])) != saved['response_sha256']:
                    raise Incomplete('cache identity/digest mismatch')
                records.extend(dict(r, cache='cached', cache_key=key) for r in saved['attempts'])
                return saved['response']
            if pending.exists():
                raise Incomplete('uncertain/error send; explicit disposition required')
            if time.time() >= self.deadline:
                raise Incomplete('deadline')
            if shutil.disk_usage(self.root).free < 60*1024**3:
                raise Incomplete('disk floor')
            if self.backend is None:
                raise Incomplete('backend not activated')
            if hasattr(self.backend, 'context_length') and input_bound(messages) + max_tokens + 20000 > self.backend.context_length(spec):
                raise Incomplete('transport context bound')
            immutable(pending, dict(identity=identity, state='pending', role=role, request=messages))
            calls = []
            try:
                response = self.backend.call(spec, messages, calls, tools=None, max_tokens=max_tokens,
                    tool_choice='none', temperature=temperature)
                if not response.get('content') or not calls or any(r.get('status') != 'ok' for r in calls):
                    raise Incomplete('invalid response')
                if any(r.get('returned_model') not in (None, parse_spec(spec)[1]) for r in calls):
                    raise Incomplete('model identity mismatch')
                attempts = [{k: r.get(k) for k in ('status', 'requested_model', 'returned_model', 'input_tokens',
                    'output_tokens', 'latency_s', 'retries', 'identity_source')} for r in calls]
                immutable(done, dict(identity=identity, response=response, response_sha256=sha(encoded(response)), attempts=attempts))
            except Exception as exc:
                immutable(prefix.with_suffix('.error.json'), dict(state='uncertain', error=type(exc).__name__, attempts=len(calls)))
                raise Incomplete('send incomplete') from None
            records.extend(dict(r, cache='fresh', cache_key=key) for r in attempts)
            return response
def snapshot(path, value=None):
    if value is not None:
        immutable(path, dict(value=value, sha256=sha(encoded(value))))
    saved = json.loads(Path(path).read_bytes())
    if sha(encoded(saved['value'])) != saved['sha256']:
        raise Incomplete('snapshot digest mismatch')
    return saved['value']
