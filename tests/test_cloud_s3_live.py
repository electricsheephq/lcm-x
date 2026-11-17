"""Owner-bucket isolation gate; never enabled by the lane's offline acceptance run."""
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
    # Positive control precedes every cross-prefix request. Use a fresh generation key.
    key = a.prefix + '/gen/' + str(uuid.uuid4().int) + '.json'
    try:
        assert a.now() > 0
        a.put_new(key, b'{}')
        assert key in a.list(a.prefix + '/gen/')
        assert a.get(key) == b'{}'
        a.delete(key)
        assert a.stat(key) is None
        a.put_new(key, b'{}')
        for method, target, query in [
            ('GET', '', [('list-type', '2'), ('prefix', a.prefix + '/gen/'), ('delimiter', '/')]),
            ('GET', key, []), ('PUT', key, []), ('DELETE', key, []),
        ]:
            # Deliberately bypass the client's name/prefix validator. The host must deny.
            status, _, _ = b._send(method, target, query, b'{}' if method == 'PUT' else b'',
                                    {'If-None-Match': '*'} if method == 'PUT' else {})
            assert status == 403
        assert s3.check_host(a)['conditional_create'] is True
    finally:
        a.delete(key)


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
