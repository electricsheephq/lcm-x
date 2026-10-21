"""Run-owned subscription cache; append-only send receipts, no uncertain replay."""
import fcntl
import json
import os
import re
import shutil
import time
from pathlib import Path
from .common import sha
from .dev_cache import Incomplete, request_identity
from .dev_panel import encoded
from . import models
from .models import input_bound, parse_spec
class Paused(Incomplete):
    pass
class Excluded(Incomplete):
    pass
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
def backend_call(backend, spec, messages, records, **kwargs):
    """Preserve real harness failures before models.py reduces them to *_failed."""
    target=backend.backend(spec)[0] if isinstance(backend,models.Backends) else backend
    if not isinstance(target,models.Harness):
        return backend.call(spec,messages,records,**kwargs)
    original=target.runner
    def runner(*args,**kw):
        proc=original(*args,**kw)
        errors=[proc.stderr or ''] if proc.returncode else []
        for line in proc.stdout.splitlines():
            try:
                event=json.loads(line)
            except ValueError:
                continue
            if event.get('type') in ('error','turn.failed') or event.get('is_error'):
                errors.append(json.dumps(event))
        reason=' '.join(errors)
        if not models.limited(reason) and not re.search(r'403|forbidden',reason,re.I) and re.search(
                r'\b(?:429|5\d\d)\b|connection|reset|refused|stream|protocol|transport',reason,re.I):
            raise ConnectionError('transient backend transport/status')
        return proc
    target.runner=runner
    try:
        return backend.call(spec,messages,records,**kwargs)
    finally:
        target.runner=original
class Cache:
    def __init__(self, root, backend, endpoints, deadline, renderer):
        self.root, self.backend = Path(root), backend
        self.endpoints, self.deadline, self.renderer = endpoints, deadline, renderer
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.root.glob('*.paused-*.json'):
            resumed=path.with_name(path.name.replace('.paused-','.resumed-'))
            pause=json.loads(path.read_bytes())
            if not resumed.exists():
                immutable(resumed,dict(at=time.time()))
                quota=getattr(backend,'quota',None)
                if quota is not None:
                    quota.state[pause['family']]['stopped']=False
                    models.write_json(quota.path,quota.state)
            self.deadline+=max(0,json.loads(resumed.read_bytes())['at']-pause['at'])
    def send(self, prefix, spec, messages, calls, **kwargs):
        for attempt, delay in enumerate((0,60,180,600)):
            failure=prefix.with_suffix(f'.retry-{attempt}.json')
            if failure.exists():
                continue
            if time.time()+delay >= self.deadline:
                raise Incomplete('deadline')
            if delay:
                time.sleep(delay)
            current=[]
            generation=0
            while prefix.with_suffix(f'.paused-{attempt}-{generation}.json').exists():
                generation+=1
            sending=prefix.with_suffix(f'.send-{attempt}-{generation}.json')
            if sending.exists() or prefix.with_suffix(f'.send-{attempt}.json').exists():
                raise Incomplete('uncertain send; explicit disposition required')
            immutable(sending,dict(state='pending',attempt=attempt))
            try:
                response=backend_call(self.backend,spec,messages,current,**kwargs)
                if any(r.get('status')!='ok' for r in current):
                    raise Incomplete('invalid response')
                calls.extend(current)
                return response
            except Exception as exc:
                reason=' '.join([type(exc).__name__,str(exc),*(str(r.get('error','')) for r in current)])
                if isinstance(exc,models.BudgetStopped) or re.search(r'usage.limit|quota',reason,re.I):
                    immutable(prefix.with_suffix(f'.paused-{attempt}-{generation}.json'),dict(at=time.time(),family=parse_spec(spec)[0]))
                    raise Paused('paused: quota') from None
                if re.search(r'403|forbidden|model_mismatch|tool_violation',reason,re.I) or not re.search(
                        r'timeout|timed out|transport|connection|reset|refused|stream|protocol|429|\b5\d\d\b',reason,re.I) and not isinstance(exc,models.httpx.TransportError):
                    raise
                immutable(failure,dict(state='transient',attempt=attempt,reason='transient transport/status'))
        raise Excluded('transient retries exhausted')
    def call(self, spec, messages, records, *, role, rubric, sources, max_tokens, temperature=None, replay=0):
        family, _, effort = parse_spec(spec)
        if family not in ('codex', 'glm') or not self.endpoints.get(family):
            raise Incomplete('subscription endpoint identity required')
        endpoint = dict(family=family, adapter='dm0-cache-v1', **self.endpoints[family])
        identity = request_identity(spec, messages, endpoint=endpoint, effort=effort, temperature=temperature,
            max_tokens=max_tokens, tools=None, tool_choice='none', rubric=rubric, renderer=self.renderer, sources=sources)
        key = sha(encoded([identity,replay]) if replay else encoded(identity))
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
                if prefix.with_suffix('.excluded.json').exists():
                    raise Excluded('transient retries exhausted')
                if not any(prefix.parent.glob(prefix.name+'.retry-*.json')) and not any(prefix.parent.glob(prefix.name+'.paused-*.json')):
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
                response = self.send(prefix, spec, messages, calls, tools=None, max_tokens=max_tokens,
                    tool_choice='none', temperature=temperature)
                if not response.get('content') or not calls or any(r.get('status') != 'ok' for r in calls):
                    raise Incomplete('invalid response')
                if any((set(r['returned_model'])!={parse_spec(spec)[1]} if family=='glm' and isinstance(r.get('returned_model'),list)
                        else r.get('returned_model') not in (None,parse_spec(spec)[1])) for r in calls):
                    raise Incomplete('model identity mismatch')
                attempts = [{k: r.get(k) for k in ('status', 'requested_model', 'returned_model', 'input_tokens',
                    'output_tokens', 'latency_s', 'retries', 'identity_source')} for r in calls]
                immutable(done, dict(identity=identity, response=response, response_sha256=sha(encoded(response)), attempts=attempts))
            except Exception as exc:
                if isinstance(exc,Paused):
                    raise
                if isinstance(exc,Excluded):
                    immutable(prefix.with_suffix('.excluded.json'),dict(reason='transient retries exhausted'))
                    raise
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
