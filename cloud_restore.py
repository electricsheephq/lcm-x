"""Rebuild and verify hosted capture generations in fresh folders."""
from contextlib import closing, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unicodedata

from . import cloud_copy, cloud_delta

OWNER_FILE = 'capture-owner.json'
HERMES_MARKERS = ('config.yaml', 'state.db')
SCRIPTS = Path(__file__).resolve().parent / 'scripts'


@contextmanager
def _claim(folder):
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise cloud_copy.Refused('restore_target_busy') from None
        if os.listdir(fd):
            raise cloud_copy.Refused('restore_target_not_empty')
        os.fchmod(fd, 0o700)
        yield fd
    finally:
        os.close(fd)


def _object_path(root, key=''):
    path = Path(root) / key
    for part in (path, *path.parents):
        try:
            if stat.S_ISLNK(part.lstat().st_mode):
                raise cloud_copy.Refused('restore_objects_symlink')
        except FileNotFoundError:
            pass
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise cloud_copy.Refused('restore_objects_symlink')
    return path


def materialize(manifest, stored_bytes, fetch, folder, *, held=None, objects_dir=None, _fd=None):
    """Build privately, then install and roll back only entries created under the claim.

    held is reserved for refresh's private patch input. Part A rebuilds in full.
    """
    folder = Path(folder)
    if _fd is None:
        with _claim(folder) as fd:
            return materialize(manifest, stored_bytes, fetch, folder, held=held, objects_dir=objects_dir, _fd=fd)
    saved, directories, owned, payload_fd = [], set(), [], None
    database = manifest['database']
    refs = {r['key']: r for r in database['base']['parts']}
    refs.update((r['key'], r) for e in database['deltas'] for r in e['objects'])

    def checked_fetch(key):
        data = cloud_delta._fetch(refs[key], fetch)
        if objects_dir is not None:
            path = _object_path(objects_dir, key)
            if not path.parent.exists():
                directories.add(path.parent)
            cloud_copy._mkdir(path.parent)
            try:
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as handle:
                    saved.append(path)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                with os.fdopen(os.open(_object_path(objects_dir, key), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
                    cloud_delta._fetch(refs[key], lambda _: handle.read())
        return data

    def write(name, data, fd):
        with os.fdopen(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd), 'wb') as handle:
            owned.append((fd, name))
            if isinstance(data, bytes):
                handle.write(data)
            else:
                shutil.copyfileobj(data, handle)
            handle.flush()
            os.fsync(handle.fileno())

    try:
        with tempfile.TemporaryDirectory(prefix='restore-build-', dir=folder.parent) as stage:
            path = Path(stage) / 'lcm.db'
            cloud_delta.rebuild(path, database, manifest['generation'], checked_fetch)
            with path.open('rb') as source:
                write('lcm.db', source, _fd)
        os.mkdir(cloud_copy.PAYLOAD_FOLDER, mode=0o700, dir_fd=_fd)
        payload_fd = os.open(cloud_copy.PAYLOAD_FOLDER, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=_fd)
        for row in manifest['payloads']:
            if row['sha256'] != row['key'].split('/')[1][:64]:
                raise cloud_copy.Refused('object_sha256', key=row['key'])
            write(row['name'], cloud_delta._fetch(row, fetch), payload_fd)
        write('manifest.json', stored_bytes, _fd)
    except Exception:
        for fd, name in reversed(owned):
            os.unlink(name, dir_fd=fd)
        if payload_fd is not None:
            os.rmdir(cloud_copy.PAYLOAD_FOLDER, dir_fd=_fd)
        for path in saved:
            _object_path(path.parent, path.name).unlink(missing_ok=True)
        for path in directories:
            path = _object_path(path)
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
        raise
    finally:
        if payload_fd is not None:
            os.close(payload_fd)


def _within(path, parent):
    for ancestor in (path, *path.parents):
        if ancestor.exists() and parent.exists() and os.path.samefile(ancestor, parent):
            return True
    normalized = [Path(unicodedata.normalize('NFD', str(p.resolve())).casefold()) for p in (path, parent)]
    return normalized[0].is_relative_to(normalized[1])


def _safe_path(path):
    path = Path(path).resolve()
    hermes = Path(os.environ.get('HERMES_HOME') or Path.home() / '.hermes').expanduser().resolve()
    if _within(path, hermes) or any(
            (parent / name).exists() for parent in (path, *path.parents)
            for name in (OWNER_FILE, *HERMES_MARKERS)):
        raise cloud_copy.Refused('restore_target_is_home')
    return path


def _target(to):
    to = _safe_path(to)
    if to.exists() and not to.is_dir():
        raise cloud_copy.Refused('restore_target_not_empty')
    to.mkdir(mode=0o700, parents=True, exist_ok=True)
    return to


def _run(script, arguments):
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(SCRIPTS / script), *arguments],
                            capture_output=True, timeout=60, env={'PATH': os.defpath})
    if result.returncode:
        raise ValueError('checker failed')


def _reader(to, manifest, message, reader_argv):
    calls = [('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {},
                             'clientInfo': {'name': 'restore', 'version': '1'}}),
             ('tools/call', {'name': 'lcm_grep', 'arguments': {'query': 'synthetic'}})]
    if manifest['payloads']:
        expand = {'externalized_ref': manifest['payloads'][0]['name']}
    else:
        expand = {'store_id': message[0]} if message else None
    if expand is not None:
        calls.append(('tools/call', {'name': 'lcm_expand', 'arguments': expand}))
    requests = [dict(jsonrpc='2.0', id=n, method=m, params=p) for n, (m, p) in enumerate(calls)]
    with tempfile.TemporaryDirectory(prefix='restore-reader-', dir=to.parent) as home:
        env = dict(PATH=os.defpath, HERMES_HOME=home, HOME=home,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        payloads = Path(home) / cloud_copy.PAYLOAD_FOLDER
        cloud_copy._mkdir(payloads)
        for row in manifest['payloads']:
            source, target = to / cloud_copy.PAYLOAD_FOLDER / row['name'], payloads / row['name']
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        command = list(reader_argv) if reader_argv is not None else [sys.executable, str(SCRIPTS / 'lcm_x_mcp.py')]
        result = subprocess.run(command + ['--store', str(to / 'lcm.db'), '--hermes-home', home,
                                          '--log-dir', str(Path(home) / 'logs')],
                                input=''.join(json.dumps(r) + '\n' for r in requests),
                                text=True, capture_output=True, env=env, timeout=60)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    if result.returncode or len(replies) != len(requests):
        raise ValueError('reader failed')
    for request, reply in zip(requests, replies):
        if reply.get('id') != request['id'] or 'error' in reply or not isinstance(reply.get('result'), dict):
            raise ValueError('reader protocol')
        if request['method'] == 'tools/call':
            body = reply['result']
            if body.get('isError') or not body.get('content'):
                raise ValueError('reader tool failed')
            for block in body['content']:
                if block.get('type') == 'text' and 'error' in json.loads(block['text']):
                    raise ValueError('reader tool failed')


def restore(transport, prefix, copy_id, generation, to, *, origin=None, owner_file=None,
            transcripts=(), objects_dir=None, reader_argv=None):
    """Restore a named generation; four ordered checks retain a failed copy."""
    to = Path(to).resolve()
    if objects_dir is not None:
        objects_dir = _safe_path(_object_path(objects_dir))
        if _within(objects_dir, to) or _within(to, objects_dir):
            raise cloud_copy.Refused('restore_objects_overlap')
    to = _target(to)
    with _claim(to) as fd:
        return _restore(transport, prefix, copy_id, generation, to, origin, owner_file, transcripts, objects_dir, reader_argv, fd)


def _restore(transport, prefix, copy_id, generation, to, origin, owner_file, transcripts, objects_dir, reader_argv, fd):
    manifest, stored = cloud_copy._head(transport, prefix, copy_id, generation=generation)
    try:
        materialize(manifest, stored, lambda key: transport.get(prefix + '/' + key), to,
                    objects_dir=objects_dir, _fd=fd)
    except cloud_copy.Refused as error:
        if 'key' in error.detail:
            error.detail['key'] = prefix + '/' + error.detail['key']
        raise
    check, check3 = 1, 'check3_not_run'
    try:
        with closing(sqlite3.connect((to / 'lcm.db').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
            if conn.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('integrity')
            sessions = {r[0] for r in conn.execute('SELECT session_id FROM messages UNION '
                        'SELECT session_id FROM summary_nodes UNION SELECT session_id FROM capture_events') if r[0]}
            message = conn.execute('SELECT store_id FROM messages ORDER BY store_id LIMIT 1').fetchone()
        check = 2
        arguments = ['--copy', str(to), '--manifest', str(to / 'manifest.json')]
        for flag, value in (('--origin', origin), ('--owner-file', owner_file)):
            if value is not None:
                arguments += [flag, str(value)]
        with tempfile.TemporaryDirectory(prefix='restore-check-', dir=to.parent) as rebuild:
            if objects_dir is not None:
                arguments += ['--objects', str(objects_dir), '--rebuild-dir', rebuild]
            _run('cloud_copy_outside_in.py', arguments)
        check = 3
        transcripts = list(transcripts)
        if transcripts and sessions <= {item.split('=', 1)[0] for item in transcripts}:
            _run('capture_outside_in.py', ['--home', str(to)] +
                 [arg for item in transcripts for arg in ('--transcript', item)])
            check3 = 'done'
        check = 4
        _reader(to, manifest, message, reader_argv)
    except subprocess.TimeoutExpired:
        raise cloud_copy.Refused('restore_check_timeout', check=check, generation=generation) from None
    except Exception:
        raise cloud_copy.Refused('restore_check_failed', check=check, generation=generation) from None
    return dict(code='done', copy_id=copy_id, generation=generation, check3=check3,
                file_bytes=manifest['database']['file_bytes'], payloads=len(manifest['payloads']),
                unresolved_minted=len(manifest['unresolved_minted']),
                unresolved_quoted=len(manifest['unresolved_quoted']))
