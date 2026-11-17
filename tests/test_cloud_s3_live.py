"""Owner-bucket isolation gate; never enabled by the lane's offline acceptance run."""
import hashlib
import json
import os
import ssl
from pathlib import Path
import uuid

import pytest

from hermes_lcm import cloud_s3 as s3


def exercise(a, b):
    # Both credentials must address the same host and bucket, with verified TLS.
    assert (a.host, a.bucket) == (b.host, b.bucket)
    assert all(t.context.verify_mode == ssl.CERT_REQUIRED and t.context.check_hostname for t in (a, b))
    # Unreferenced payloads are safe to leave behind if this process is interrupted.
    probes = []
    for t in (a, b):
        body = ('LCM-X isolation probe ' + uuid.uuid4().hex).encode()
        key = t.prefix + '/payloads/' + hashlib.sha256(body).hexdigest() + '.' + uuid.uuid4().hex
        probes.append((t, key, body))
    cleanup = []
    try:
        # Both positive controls must pass before any cross-prefix denial counts.
        for t, key, body in probes:
            assert t.now() > 0
            cleanup.append((t, key))
            t.put_new(key, body)
            assert key in t.list(t.prefix + '/payloads/')
            assert t.get(key) == body
            t.delete(key)
            assert t.stat(key) is None
            t.put_new(key, body)
        _, key, body = probes[0]
        for method, target, query in [
            ('GET', '', [('list-type', '2'), ('prefix', a.prefix + '/payloads/'), ('delimiter', '/')]),
            ('GET', '', [('list-type', '2'), ('prefix', a.prefix + '/gen/'), ('delimiter', '/')]),
            ('GET', key, []), ('PUT', key, []), ('DELETE', key, []),
        ]:
            # Deliberately bypass the client's name/prefix validator. The host must deny.
            status, _, _ = b._send(method, target, query, body if method == 'PUT' else b'',
                                    {'If-None-Match': '*'} if method == 'PUT' else {})
            assert status == 403
        assert s3.check_host(a)['conditional_create'] is True
    finally:
        for t, key in cleanup:
            t.delete(key)


def configured(name):
    config = json.loads(Path(os.environ[name]).read_text())
    # Explicit live-test paths still pass the product's complete file-permission gate.
    path = Path(config['credentials_file'])
    return s3.S3Transport(config['endpoint'], config['bucket'], config['region'], config['prefix'],
                           s3._read_credentials(path))


@pytest.mark.skipif(not (os.environ.get('LCMX_S3_LIVE_A') and os.environ.get('LCMX_S3_LIVE_B')),
                    reason='owner bucket and two scoped credentials required (O1)')
def test_live_isolation():
    exercise(configured('LCMX_S3_LIVE_A'), configured('LCMX_S3_LIVE_B'))
