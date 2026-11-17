"""Cloud publication guard and retention; no host or uploader imports.

The CLI injects the uploader's existing snapshot, head and state primitives into
install(). Decisions are scoped to exact manifest bytes and exact missing items.
"""
import base64
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import zlib

from . import cloud_delta

Refused = cloud_delta.DeltaRefused
DAY = 86400
FORGOTTEN = 'capture:forgotten:'
HEX = r'[0-9a-f]{64}'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _ids(values):
    data, previous = bytearray(), 0
    for value in sorted(set(values)):
        delta, previous = value - previous, value
        while delta >= 128:
            data.append((delta & 127) | 128)
            delta >>= 7
        data.append(delta)
    return base64.b64encode(zlib.compress(data)).decode()


def _decode(value):
    result, number, shift, previous = set(), 0, 0, 0
    for byte in zlib.decompress(base64.b64decode(value, validate=True)):
        number |= (byte & 127) << shift
        if byte < 128:
            previous += number
            result.add(previous)
            number, shift = 0, 0
        else:
            shift += 7
    if shift:
        raise ValueError('truncated ids')
    return result


def _ro(db):
    return sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)


def _sessions(db):
    with closing(_ro(db)) as conn:
        result = {r[0] for r in conn.execute('SELECT session_id FROM messages')}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='capture_events'").fetchone():
            result.update(r[0] for r in conn.execute('SELECT session_id FROM capture_events'))
        return result


def _markers(db):
    with closing(_ro(db)) as conn:
        return {r[0][len(FORGOTTEN):] for r in conn.execute('SELECT key FROM metadata') if r[0].startswith(FORGOTTEN)}


def _state(home, copy_id):
    try:
        value = json.loads((Path(home) / 'cloud/state.json').read_text())
        if not isinstance(value, dict):
            raise ValueError()
        return value if value.get('copy_id') == copy_id else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        raise Refused('state_unreadable') from None


@contextmanager
def _locked(home):
    folder = Path(home) / 'cloud'
    folder.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(folder / 'upload.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused('lock_held') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def hardware_id():
    try:
        if sys.platform == 'darwin':
            data = subprocess.run(['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'],
                                  check=True, capture_output=True, text=True, timeout=5).stdout
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', data)
            value = match[1] if match else ''
        else:
            value = Path('/etc/machine-id').read_text().strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    raise Refused('hardware_id_unavailable')


def publisher(copy_id, hardware_id):
    value = hardware_id() if callable(hardware_id) else hardware_id
    if not isinstance(value, str) or not value.strip():
        raise Refused('hardware_id_unavailable')
    return _sha((copy_id + '\n' + value).encode())


def validate_head(head):
    try:
        for pin in head.get('pins', []):
            valid = (type(pin['generation']) is int and pin['generation'] > 0
                     and re.fullmatch(r'[0-9a-f]{32}', pin['decision'])
                     and all(type(pin[k]) in (int, float) and math.isfinite(pin[k]) for k in ('from', 'until'))
                     and pin['until'] == pin['from'] + 30 * DAY)
            if not valid:
                raise ValueError()
        for item in head.get('superseded', []):
            if not (re.fullmatch(cloud_delta._COPY, item['copy_id'])
                    and type(item['generation']) is int and item['generation'] > 0
                    and re.fullmatch(HEX, item['manifest_sha256'])):
                raise ValueError()
        attempt = head['attempt']
        if (attempt['generation'] != head['generation']
                or not re.fullmatch(r'[0-9a-f]{32}', attempt['token'])):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise Refused('bad_head', generation=head['generation']) from None


def check_chain(head, state, publisher):
    if head is None:
        return
    validate_head(head)
    pair = [head['attempt']['generation'], head['attempt']['token']]
    if head.get('publisher') != publisher or pair not in (state.get('last_pair'), state.get('pending_pair')):
        raise Refused('chain_conflict', generation=head['generation'])


def inherited_items(old_head):
    return [{k: item[k] for k in ('key', 'kind', 'session')}
            for item in old_head.get('forgotten', []) if item['kind'] == 'forgotten']


def returned_sessions(head, db):
    return _sessions(db) & {i['session'] for i in (head or {}).get('forgotten', []) if i['kind'] == 'forgotten'}


def as_delete_rows_present(decisions, staged_db):
    with closing(_ro(staged_db)) as conn:
        present = {r[0] for r in conn.execute('SELECT store_id FROM messages')}
    if any(present & _decode(d['ids']) for d in decisions if d['kind'] == 'as-delete'):
        raise Refused('as_delete_rows_present')


def forget_items_for(home, copy_id, inherited=None, *, live_ro=None):
    home, inherited = Path(home), inherited or []
    def items(head, staged_db):
        decisions = _state(home, copy_id).get('unpublished_decisions', [])
        as_delete_rows_present(decisions, staged_db)
        if head is None and _sessions(staged_db) & {i['session'] for i in inherited}:
            raise Refused('forgotten_session_returned')
        existing = {i['key'] for i in (head or {}).get('forgotten', [])}
        held = (head or {}).get('guard', {})
        held_sessions = set(held.get('sessions', {})) | set(held.get('removed_sessions', []))
        with closing(live_ro(home) if live_ro else _ro(home / 'lcm.db')) as live:
            markers = {r[0][len(FORGOTTEN):] for r in live.execute('SELECT key FROM metadata') if r[0].startswith(FORGOTTEN)}
        candidates = {s: dict(key=s, kind='forgotten', session=s, chain_may_hold=s in held_sessions)
                      for s in markers}
        if head is None:
            candidates.update({i['key']: i | dict(chain_may_hold=True) for i in inherited})
        for d in decisions:
            if d['kind'] == 'as-delete':
                candidates[d['id']] = dict(key=d['id'], kind='as-delete', session=d['session'],
                                           chain_may_hold=d['session'] in held_sessions)
        return [candidates[k] for k in sorted(candidates) if k not in existing]
    return items


def losses(head_guard, staged_guard, *, unresolved=(), forgotten=()):
    """Return content-free exact missing sets. Ledger bypasses the digest fast path."""
    result = {}
    current_sessions = staged_guard['sessions']
    for sid, old in head_guard['sessions'].items():
        if sid in forgotten:
            continue
        new = current_sessions.get(sid, {})
        if old.get('digest') != new.get('digest'):
            current = dict(new.get('messages', []))
            missing = [i for i, digest in old['messages'] if current.get(i) != digest]
            if missing:
                result.setdefault('messages', {})[sid] = _ids(missing)
            for field in ('bindings', 'relations'):
                missing = sorted(set(old[field]) - set(new.get(field, [])))
                if missing:
                    result.setdefault(field, {})[sid] = missing
        ledger = {_json(row[:2]): row for row in new.get('ledger', [])}
        old_messages, new_messages = dict(old['messages']), dict(new.get('messages', []))
        for row in old['ledger']:
            actual = ledger.get(_json(row[:2]))
            valid = actual is not None and actual[2:4] == row[2:4]
            if actual is not None:
                if actual[3] is not None:
                    valid = valid and actual[3] == actual[2]
                elif row[4] is not None:
                    ids = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                    valid = valid and actual[4] == row[4] and all(i in new_messages and new_messages[i] == old_messages.get(i) for i in ids)
            if not valid:
                result.setdefault('ledger', {}).setdefault(sid, []).append(row[:2])
    unresolved = {i['name'] for i in unresolved}
    for name, digest in head_guard.get('payloads', {}).items():
        actual = staged_guard.get('payloads', {}).get(name)
        if (actual is not None and actual != digest) or (actual is None and name in unresolved):
            result.setdefault('payloads', []).append(name)
    if 'payloads' in result:
        result['payloads'].sort()
    return result


def _eligible(decision, head, head_sha):
    return decision.get('generation') == head['generation'] and decision.get('manifest_sha256') == head_sha


def _subtract(loss, decision):
    covered = decision.get('losses', {})
    for field in list(loss):
        if field == 'payloads':
            loss[field] = sorted(set(loss[field]) - set(covered.get(field, [])))
        else:
            for sid in list(loss[field]):
                if decision['kind'] == 'as-delete' and sid != decision['session']:
                    continue
                available = covered.get(field, {}).get(sid, [])
                if field == 'messages':
                    available = decision['ids'] if decision['kind'] == 'as-delete' else available
                    remain = _decode(loss[field][sid]) - (_decode(available) if available else set())
                    if remain:
                        loss[field][sid] = _ids(remain)
                    else:
                        del loss[field][sid]
                else:
                    loss[field][sid] = [v for v in loss[field][sid] if v not in available]
                    if not loss[field][sid]:
                        del loss[field][sid]
        if not loss[field]:
            del loss[field]


def guard(head, db, m, *, decisions=(), bucket_now=None, head_sha=None):
    """Check row rules in priority order, then author cumulative removed sets."""
    old, new = head['guard'], m['guard']
    if returned_sessions(head, db):
        raise Refused('forgotten_session_returned', generation=head['generation'])
    current = {i: (sid, digest) for sid, s in new['sessions'].items() for i, digest in s['messages']}
    previous = {i: (sid, digest) for sid, s in old['sessions'].items() for i, digest in s['messages']}
    removed = _decode(old.get('removed_ids', _ids([])))
    if ((m['sequence']['messages'] or 0) < old['mark'] or removed & current.keys()
            or any(i in current and current[i] != value for i, value in previous.items())):
        raise Refused('store_id_reused', generation=head['generation'])
    loss = losses(old, new, unresolved=m['unresolved_minted'] + m['unresolved_quoted'], forgotten=_markers(db))
    lost_sessions = set().union(*(set(v) for k, v in loss.items() if k != 'payloads'))
    applied = []
    for decision in decisions:
        if decision['kind'] in ('accept-loss', 'as-delete') and _eligible(decision, head, head_sha or _sha(_json(head))):
            _subtract(loss, decision)
            applied.append(decision)
    if loss:
        raise Refused('guard_loss', generation=head['generation'])
    missing = previous.keys() - current.keys()
    new['removed_ids'] = _ids(removed | missing)
    new['removed_sessions'] = sorted(set(old.get('removed_sessions', [])) | {previous[i][0] for i in missing}
                                     | (old['sessions'].keys() - new['sessions'].keys()) | lost_sessions)
    _pins(head, m, decisions, applied, bucket_now)
    return {d['id'] for d in applied} | {d['id'] for d in decisions if d['kind'] == 'release'}


def _pins(head, m, decisions, applied, now):
    released = {d['pin'] for d in decisions if d['kind'] == 'release'}
    pins = [p for p in head.get('pins', []) if p['generation'] not in released and (now is None or p['until'] > now)]
    for d in applied:
        if d['kind'] == 'accept-loss':
            if now is None:
                raise Refused('bucket_date_unavailable')
            pins.append(dict(generation=head['generation'], decision=d['id'], **{'from': now, 'until': now + 30 * DAY}))
    m['pins'] = sorted(pins, key=lambda p: (p['generation'], p['decision']))


def record_decision(home, copy_id, decision, *, update_state):
    """Record an already-reviewed exact loss set; no bucket or live-store mutation."""
    if decision.get('kind') not in ('accept-loss', 'as-delete', 'release') or not re.fullmatch(r'[0-9a-f]{32}', decision.get('id', '')):
        raise Refused('bad_decision')
    with _locked(home):
        decisions = _state(home, copy_id).get('unpublished_decisions', [])
        if any(d['id'] == decision['id'] for d in decisions):
            raise Refused('bad_decision')
        update_state(Path(home), copy_id, unpublished_decisions=decisions + [decision])
    return decision


def withdraw(home, copy_id, decision_id, *, update_state):
    with _locked(home):
        decisions = _state(home, copy_id).get('unpublished_decisions', [])
        if not any(d['id'] == decision_id for d in decisions):
            raise Refused('decision_not_unpublished')
        update_state(Path(home), copy_id, consumed_decisions={decision_id})


def _refs(manifest):
    db = manifest['database']
    return {r['key'] for r in db['base']['parts'] + [r for e in db['deltas'] for r in e['objects']] + manifest['payloads']}


def retention(transport, prefix, manifest, *, uploader):
    """Only the winning after-create hook calls this, under upload.lock."""
    manifests, named = {}, set()
    for key in transport.list(prefix + '/gen/'):
        uploader._name(key, re.escape(prefix) + r'/gen/[1-9][0-9]*\.json')
        m = json.loads(transport.get(key))
        generation = int(key.rsplit('/', 1)[1][:-5])
        if m['generation'] != generation or m['copy_id'] != manifest['copy_id']:
            raise Refused('bad_head')
        cloud_delta._shape(m['database'], generation)
        refs = _refs(m)
        for ref in refs:
            uploader._name(ref, uploader._OBJECT)
        manifests[generation] = (m, refs, key)
        named.update(refs)
    pins = {p['generation'] for p in manifest['pins']}
    cutoff = max((i['base_generation'] or 0 for i in manifest['forgotten']), default=0)
    forgotten = {i['session'] for i in manifest['forgotten'] if i['kind'] == 'forgotten'}
    keep = set()
    for gen, (m, refs, key) in manifests.items():
        pruned = (m['database']['base']['generation'] < cutoff or forgotten & m['guard']['sessions'].keys())
        if gen in pins or gen == manifest['generation'] or (gen >= manifest['generation'] - 2 and not pruned):
            keep.add(gen)
    protected = set().union(*(manifests[g][1] for g in keep))
    failed, done = set(), set()
    def delete(key):
        try:
            transport.delete(key)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
    for gen, (m, refs, key) in sorted(manifests.items()):
        if gen in keep:
            continue
        for ref in sorted(refs - protected - done - failed):
            (done if delete(prefix + '/' + ref) else failed).add(ref)
    # All exclusive keys are attempted before any manifest is removed.
    complete = not failed
    for gen, (m, refs, key) in sorted(manifests.items()):
        if gen not in keep and not refs & failed:
            complete = delete(key) and complete
    now = transport.now()
    for folder in ('parts', 'deltas', 'payloads'):
        for key in transport.list(prefix + '/' + folder + '/'):
            uploader._name(key, re.escape(prefix) + '/' + uploader._OBJECT)
            if key[len(prefix) + 1:] in named:
                continue
            info = transport.stat(key)
            if now is not None and info and now - info['time'] > 7 * DAY:
                complete = delete(key) and complete
    return complete


def install(home, copy_id, transport, prefix, *, uploader, hardware_id=hardware_id):
    """Return keyword callables; the uploader retains the single lock and state merge."""
    home, applied = Path(home), set()
    checked, decisions = {}, []
    def chain_check(head, state, publisher_id, head_bytes):
        check_chain(head, state, publisher_id)
        checked.update(head=head, sha=_sha(head_bytes))
    def prepare_manifest(m):
        if checked.get('head'):
            _pins(checked['head'], m, decisions, [d for d in decisions if d['id'] in applied], transport.now())
    def run_guard(head, db, m):
        applied.clear()
        decisions[:] = uploader._state(home, copy_id).get('unpublished_decisions', [])
        applied.update(guard(head, db, m, decisions=decisions, bucket_now=transport.now(), head_sha=checked['sha']))
    def after_create(m):
        current, data = uploader._head(transport, prefix, copy_id)
        own = current is not None and _sha(data) == _sha(uploader._json(m))
        result = dict(last_pair=[m['generation'], m['attempt']['token']])
        if own and retention(transport, prefix, m, uploader=uploader):
            result['retention_done_generation'] = m['generation']
        consumed = (applied | {i['key'] for i in m['forgotten'] if i['kind'] == 'as-delete'}
                    | {p['decision'] for p in m['pins']}
                    | {p['decision'] for p in (checked.get('head') or {}).get('pins', [])})
        return result | dict(consumed_decisions=consumed)
    return dict(publisher=lambda: publisher(copy_id, hardware_id),
                chain_check=chain_check, guard=run_guard, after_create=after_create, prepare_manifest=prepare_manifest,
                forget_items=forget_items_for(home, copy_id, live_ro=uploader._live_ro))
