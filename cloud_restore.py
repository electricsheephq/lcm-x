"""Rebuild and verify hosted capture generations in fresh folders."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from . import cloud_copy, cloud_delta

OWNER_FILE = 'capture-owner.json'
HERMES_MARKERS = ('config.yaml', 'state.db')
SCRIPTS = Path(__file__).resolve().parent / 'scripts'


def materialize(manifest, stored_bytes, fetch, folder, *, held=None, objects_dir=None):
    """Fill an empty private folder; discard every partial output on refusal.

    held is reserved for refresh's private patch input. Part A rebuilds in full.
    objects_dir holds relative object keys for the independent checker's rebuild.
    """
    folder = Path(folder)
    if any(folder.iterdir()):
        raise cloud_copy.Refused('restore_target_not_empty')
    saved, directories = [], set()
    database = manifest['database']
    refs = {r['key']: r for r in database['base']['parts']}
    refs.update((r['key'], r) for e in database['deltas'] for r in e['objects'])

    def checked_fetch(key):
        data = cloud_delta._fetch(refs[key], fetch)
        if objects_dir is not None:
            path = Path(objects_dir) / key
            if not path.parent.exists():
                directories.add(path.parent)
            cloud_copy._mkdir(path.parent)
            if path.exists():
                if path.read_bytes() != data:
                    raise cloud_copy.Refused('object_sha256', key=key)
            else:
                saved.append(path)
                cloud_copy._write(path, data)
        return data

    try:
        cloud_delta.rebuild(folder / 'lcm.db', database, manifest['generation'], checked_fetch)
        cloud_copy._mkdir(folder / cloud_copy.PAYLOAD_FOLDER)
        for row in manifest['payloads']:
            if row['sha256'] != row['key'].split('/')[1][:64]:
                raise cloud_copy.Refused('object_sha256', key=row['key'])
            data = cloud_delta._fetch(row, fetch)
            cloud_copy._write(folder / cloud_copy.PAYLOAD_FOLDER / row['name'], data)
        cloud_copy._write(folder / 'manifest.json', stored_bytes)
    except Exception:
        for path in folder.iterdir():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        for path in saved:
            path.unlink(missing_ok=True)
        for path in directories:
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
        raise


def _safe_path(path):
    path = Path(path).resolve()
    hermes = Path(os.environ.get('HERMES_HOME') or Path.home() / '.hermes').expanduser().resolve()
    if path.is_relative_to(hermes) or any(
            (parent / name).exists() for parent in (path, *path.parents)
            for name in (OWNER_FILE, *HERMES_MARKERS)):
        raise cloud_copy.Refused('restore_target_is_home')
    return path


def _target(to):
    to = _safe_path(to)
    if to.exists() and (not to.is_dir() or any(to.iterdir())):
        raise cloud_copy.Refused('restore_target_not_empty')
    to.mkdir(mode=0o700, parents=True, exist_ok=True)
    to.chmod(0o700)
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
        objects_dir = _safe_path(objects_dir)
        if objects_dir.is_relative_to(to) or to.is_relative_to(objects_dir):
            raise cloud_copy.Refused('restore_objects_overlap')
    to = _target(to)
    manifest, stored = cloud_copy._head(transport, prefix, copy_id, generation=generation)
    try:
        materialize(manifest, stored, lambda key: transport.get(prefix + '/' + key), to,
                    objects_dir=objects_dir)
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
    except Exception:
        raise cloud_copy.Refused('restore_check_failed', check=check, generation=generation) from None
    return dict(code='done', copy_id=copy_id, generation=generation, check3=check3,
                file_bytes=manifest['database']['file_bytes'], payloads=len(manifest['payloads']),
                unresolved_minted=len(manifest['unresolved_minted']),
                unresolved_quoted=len(manifest['unresolved_quoted']))
