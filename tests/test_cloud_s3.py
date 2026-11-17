"""Synthetic S3 protocol, signing, publication and credential regressions. No sockets."""
import ast
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from hermes_lcm import cloud_copy as cp, cloud_guard as cg, cloud_s3 as s3
from tests.test_cloud_copy import PREFIX, TOKEN, grow, objects, pub, rebuild, refused
from tests.test_cloud_copy import fixture as capture_fixture

fixture = capture_fixture
SECRET = 'planted-S3-secret-that-must-never-escape'
CREDS = {'access_key_id': 'synthetic-access', 'secret_access_key': SECRET}
KEY = PREFIX + '/gen/1.json'
DATE = 'Fri, 24 May 2013 00:00:00 GMT'
EPOCH = 1369353600


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('socket connection attempted')
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


class Responses:
    """Record only redacted wire requests; signatures never enter test evidence."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body):
        headers = {k.lower(): v for k, v in headers.items()}
        self.calls.append((method, url, {k: v for k, v in headers.items() if k != 'authorization'}, body))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def transport(request, **kwargs):
    return s3.S3Transport(kwargs.pop('endpoint', 's3.example.invalid'), 'bucket', 'us-east-1',
                          PREFIX, CREDS, request=request, clock=lambda: EPOCH, **kwargs)


def response(status=200, body=b'', **headers):
    return status, headers, body


# AWS S3 User Guide, "Signature Calculations for the Authorization Header:
# Transferring Payload in a Single Chunk (AWS Signature Version 4)":
# https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
# Written from memory without network; reviewer must verify against the page.
# Published inputs: examplebucket.s3.amazonaws.com, us-east-1, 20130524,
# AKIAIOSFODNN7EXAMPLE, wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY.
# Canonical URI is supplied directly: transport URLs remain path-style.
@pytest.mark.parametrize('method,path,query,extra,body,signature', [
    ('GET', '/test.txt', [], {'Range': 'bytes=0-9'}, b'',
     'f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41'),
    ('GET', '/', [('lifecycle', '')], {}, b'',
     'fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543'),
    ('GET', '/', [('prefix', 'J'), ('max-keys', '2')], {}, b'',
     '34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7'),
    ('PUT', '/test$file.text', [], {'Date': DATE, 'x-amz-storage-class': 'REDUCED_REDUNDANCY'},
     b'Welcome to Amazon S3.',
     '98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd'),
])
def test_sigv4_aws_examples(method, path, query, extra, body, signature):
    # Compare booleans so failure output cannot record an Authorization signature.
    creds = {'access_key_id': 'AKIAIOSFODNN7EXAMPLE',
             'secret_access_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'}
    headers = s3.sign(method, path, query, {'Host': 'examplebucket.s3.amazonaws.com', **extra},
                      body, creds, 'us-east-1', '20130524T000000Z')
    matches = headers.get('authorization', '').endswith('Signature=' + signature)
    assert matches is True
    assert headers.get('x-amz-content-sha256') == hashlib.sha256(body).hexdigest()


def test_get_and_errors():
    r = Responses(response(body=b'value'), response(404), response(403, b'<Error><Code>AccessDenied</Code></Error>'))
    t = transport(r)
    assert t.get(KEY) == b'value'
    with pytest.raises(FileNotFoundError):
        t.get(KEY)
    with pytest.raises(s3.S3Error) as caught:
        t.get(KEY)
    assert isinstance(caught.value, OSError)
    assert (caught.value.status, caught.value.code) == (403, 'AccessDenied')
    assert [c[0] for c in r.calls] == ['GET'] * 3
    assert r.calls[0][1] == 'https://s3.example.invalid/bucket/' + KEY


def test_stat():
    r = Responses(response(**{'Content-Length': '17', 'Last-Modified': DATE}), response(404), response(403))
    t = transport(r)
    assert t.stat(KEY) == {'bytes': 17, 'time': EPOCH}
    assert t.stat(KEY) is None
    with pytest.raises(s3.S3Error):
        t.stat(KEY)
    assert [c[0] for c in r.calls] == ['HEAD'] * 3


def listing(keys=(), *, token=None, common=False):
    return ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + ''.join('<Contents><Key>' + k + '</Key></Contents>' for k in keys)
            + ('<CommonPrefixes><Prefix>sub/</Prefix></CommonPrefixes>' if common else '')
            + ('<IsTruncated>true</IsTruncated><NextContinuationToken>' + token
               + '</NextContinuationToken>' if token else '<IsTruncated>false</IsTruncated>')
            + '</ListBucketResult>').encode()


def test_list_pagination():
    r = Responses(response(body=listing([PREFIX + '/gen/2.json'], token='a+/=')),
                  response(body=listing([KEY])))
    assert transport(r).list(PREFIX + '/gen/') == [KEY, PREFIX + '/gen/2.json']
    first, second = [parse_qs(urlsplit(c[1]).query) for c in r.calls]
    assert first == {'list-type': ['2'], 'prefix': [PREFIX + '/gen/'], 'delimiter': ['/']}
    assert second == {**first, 'continuation-token': ['a+/=']}


@pytest.mark.parametrize('body', [listing(common=True), listing([PREFIX + '/gen/0.json']),
                                  listing(['foreign/gen/1.json']), listing([PREFIX + '/parts/' + 'a' * 64 + '.' + TOKEN])])
def test_listing_bad_name(body):
    refused('bad_name', transport(Responses(response(body=body))).list, PREFIX + '/gen/')


@pytest.mark.parametrize('method', ['get', 'stat', 'list', 'put_new', 'delete'])
@pytest.mark.parametrize('key', ['foreign/gen/1.json', PREFIX + '/gen/../1.json',
                                  PREFIX + '/probe/' + TOKEN, PREFIX + '/gen/0.json', None])
def test_names_before_request(method, key):
    r = Responses(response())
    args = (key, b'x') if method == 'put_new' else (key,)
    refused('bad_name', getattr(transport(r), method), *args)
    assert r.calls == []


@pytest.mark.parametrize('endpoint', ['http://s3.example.invalid', 'ftp://s3.example.invalid'])
def test_https_only(endpoint):
    refused('tls_required', transport, Responses(), endpoint=endpoint)


@pytest.mark.parametrize('source', ['default', 'environment'])
@pytest.mark.parametrize('entry', ['0123abcd.0', 'abcdef01.9', None, 'ABCDEF01.0', 'abcdef01.10', 'abcdef01.x'])
def test_zero_ca(tmp_path, monkeypatch, source, entry):
    directory = tmp_path / 'certificates'
    directory.mkdir()
    if entry is not None:
        (directory / entry).touch()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, 'create_default_context', lambda: context)
    paths = ssl.get_default_verify_paths()._replace(cafile=None, capath=str(directory) if source == 'default' else None)
    monkeypatch.setattr(ssl, 'get_default_verify_paths', lambda: paths)
    monkeypatch.delenv('SSL_CERT_DIR', raising=False)
    if source == 'environment':
        monkeypatch.setenv('SSL_CERT_DIR', str(directory))
    if entry in ('0123abcd.0', 'abcdef01.9'):
        try:
            result = transport(Responses()).context
        except cp.Refused:
            result = None
        assert result is context
    else:
        refused('tls_no_ca_bundle', transport, Responses())
    # A missing or absent directory must refuse too.
    if source == 'default':
        paths = paths._replace(capath=None)
    else:
        monkeypatch.setenv('SSL_CERT_DIR', str(tmp_path / 'missing'))
    refused('tls_no_ca_bundle', transport, Responses())


@pytest.mark.parametrize('status', [200, 201, 204])
def test_put_created(status):
    r = Responses(response(status))
    transport(r).put_new(KEY, io.BytesIO(b'content'))
    assert len(r.calls) == 1
    method, _, headers, body = r.calls[0]
    assert method == 'PUT' and headers['if-none-match'] == '*' and body == b'content'
    assert headers['x-amz-content-sha256'] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize('status', [412, 409])
@pytest.mark.parametrize('present', [True, False])
def test_put_conflicts(status, present):
    r = Responses(response(status, b'<Error><Code>ConditionalRequestConflict</Code></Error>'),
                  response(200 if present else 404, **{'Content-Length': '1', 'Last-Modified': DATE}))
    t = transport(r)
    with pytest.raises(FileExistsError if status == 412 or present else s3.S3Error):
        t.put_new(KEY, b'x')
    assert [c[0] for c in r.calls] == (['PUT', 'HEAD'] if status == 409 else ['PUT'])


@pytest.mark.parametrize('status', [200, 204, 404, 403])
def test_delete(status):
    r = Responses(response(status))
    if status == 403:
        with pytest.raises(s3.S3Error):
            transport(r).delete(KEY)
    else:
        transport(r).delete(KEY)
    assert len(r.calls) == 1 and r.calls[0][0] == 'DELETE'


@pytest.mark.parametrize('status', [200, 204, 299, 403, 404])
@pytest.mark.parametrize('date', [DATE, None, 'garbage'])
def test_bucket_clock(date, status):
    r = Responses(response(status, **({'Date': date} if date else {})))
    if date == DATE and status != 404:
        try:
            result = transport(r).now()
        except s3.S3Error:
            result = None
        assert result == EPOCH
    else:
        with pytest.raises(s3.S3Error):
            transport(r).now()
    assert len(r.calls) == 1
    assert r.calls[0][:2] == ('HEAD', 'https://s3.example.invalid/bucket')


@pytest.mark.parametrize('method', ['get', 'stat', 'list', 'delete', 'now'])
@pytest.mark.parametrize('failure', [500, 502, 503, 504, ConnectionResetError(SECRET), TimeoutError(SECRET)])
def test_retry_reads_and_delete(monkeypatch, method, failure):
    sleeps = []
    monkeypatch.setattr(s3.time, 'sleep', sleeps.append)
    bad = response(failure) if isinstance(failure, int) else failure
    good = response(body=listing() if method == 'list' else b'x',
                    **{'Content-Length': '1', 'Last-Modified': DATE, 'Date': DATE})
    r = Responses(bad, bad, good)
    getattr(transport(r), method)(*([] if method == 'now' else [PREFIX + '/gen/' if method == 'list' else KEY]))
    assert len(r.calls) == 3 and sleeps == [1, 4]


@pytest.mark.parametrize('failure', [503, TimeoutError(SECRET), ConnectionResetError(SECRET)])
def test_put_never_retries(monkeypatch, failure):
    sleeps = []
    monkeypatch.setattr(s3.time, 'sleep', sleeps.append)
    r = Responses(response(failure) if isinstance(failure, int) else failure, response(412))
    with pytest.raises(s3.S3Error) as caught:
        transport(r).put_new(KEY, b'x')
    assert len(r.calls) == 1 and sleeps == []
    assert SECRET not in str(caught.value) + repr(caught.value)


def test_retry_exhausted(monkeypatch):
    sleeps = []
    monkeypatch.setattr(s3.time, 'sleep', sleeps.append)
    r = Responses(*[response(503, b'<Error><Code>SlowDown</Code></Error>')] * 3)
    with pytest.raises(s3.S3Error) as caught:
        transport(r).get(KEY)
    assert (caught.value.status, caught.value.code) == (503, 'SlowDown')
    assert len(r.calls) == 3 and sleeps == [1, 4]


# Synthetic documented S3 XML shapes, not captured host responses:
# AWS API ErrorResponses, Cloudflare R2 S3 error codes, Backblaze B2 S3
# error handling, and Hetzner Object Storage (Ceph S3 compatibility).
@pytest.mark.parametrize('provider,namespace', [('aws', ''), ('r2', ''), ('b2', ''),
                                                ('hetzner-ceph', ' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"')])
@pytest.mark.parametrize('status,code', [(404, 'NoSuchKey'), (412, 'PreconditionFailed'), (403, 'AccessDenied')])
def test_provider_error_xml(provider, namespace, status, code):
    body = (f'<Error{namespace}><Code>{code}</Code><Message>{SECRET}</Message>'
            '<Resource>/bucket/key</Resource><RequestId>synthetic</RequestId></Error>').encode()
    r = Responses(response(status, body))
    with pytest.raises(s3.S3Error) as caught:
        transport(r)._ensure(*transport(r)._send('GET', KEY))
    assert str(caught.value) == f'{status} {code}'
    assert SECRET not in str(caught.value) + repr(caught.value) + repr(r.calls)


@pytest.mark.parametrize('provider', ['aws', 'r2', 'b2', 'hetzner-ceph'])
def test_provider_list_xml(provider):
    xml = listing([KEY])
    if provider in ('aws', 'r2'):
        xml = xml.replace(b' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"', b'')
    r = Responses(response(body=xml))
    assert transport(r).list(PREFIX + '/gen/') == [KEY]


@pytest.mark.parametrize('body', [b'not XML', b'<Error>',
                                  ('<Error><Code>' + SECRET + '</Code></Error>').encode()])
def test_untrusted_errors_redacted(body):
    r = Responses(response(403, body, **{'Authorization': SECRET}))
    with pytest.raises(s3.S3Error) as caught:
        transport(r).get(KEY)
    assert str(caught.value) == '403 UnknownError'
    assert SECRET not in repr(caught.value) + str(caught.value)


@pytest.mark.parametrize('mode', [0o640, 0o604, 0o660, 0o600, 0o400])
def test_credentials_modes(tmp_path, mode):
    folder = tmp_path / 'cloud'
    folder.mkdir()
    path = folder / 'credentials.json'
    path.write_text(json.dumps(CREDS))
    path.chmod(mode)
    if mode & 0o077:
        refused('credentials_mode', s3.load_credentials, tmp_path)
    else:
        creds = s3.load_credentials(tmp_path)
        assert creds is not None
        assert creds.access_key_id == CREDS['access_key_id'] and creds.secret_access_key == SECRET
        assert all(v not in repr(creds) + str(creds) for v in CREDS.values())


def test_credentials_symlink(tmp_path):
    (tmp_path / 'cloud').mkdir()
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(CREDS))
    source.chmod(0o600)
    (tmp_path / 'cloud/credentials.json').symlink_to(source)
    refused('credentials_mode', s3.load_credentials, tmp_path)


def test_credentials_owner(tmp_path, monkeypatch):
    (tmp_path / 'cloud').mkdir()
    path = tmp_path / 'cloud/credentials.json'
    path.write_text(json.dumps(CREDS))
    path.chmod(0o600)
    monkeypatch.setattr(s3.os, 'getuid', lambda: path.stat().st_uid + 1)
    refused('credentials_mode', s3.load_credentials, tmp_path)


@pytest.mark.parametrize('raw', ['invalid ' + SECRET, json.dumps({'access_key_id': SECRET}),
                                 json.dumps({'access_key_id': SECRET, 'secret_access_key': []})])
def test_credentials_bad_json_safe(tmp_path, raw):
    (tmp_path / 'cloud').mkdir()
    path = tmp_path / 'cloud/credentials.json'
    path.write_text(raw)
    path.chmod(0o600)
    with pytest.raises(cp.Refused) as caught:
        s3.load_credentials(tmp_path)
    assert SECRET not in str(caught.value) + repr(caught.value)


def test_secret_and_signature_not_recorded(tmp_path, caplog, capsys):
    (tmp_path / 'cloud').mkdir()
    path = tmp_path / 'cloud/credentials.json'
    path.write_text(json.dumps(CREDS))
    path.chmod(0o600)
    creds = s3.load_credentials(tmp_path)
    signatures = []
    def wire(method, url, headers, body):
        # The sole unredacted signing boundary; retain a digest only.
        auth = headers['authorization']
        signatures.append(hashlib.sha256(auth.rsplit('=', 1)[1].encode()).hexdigest())
        assert SECRET not in url + repr(headers) + repr(body)
        raise OSError(SECRET + ' Signature=do-not-copy')
    t = s3.S3Transport('s3.example.invalid', 'bucket', 'us-east-1', PREFIX, creds,
                       request=wire, clock=lambda: EPOCH)
    with pytest.raises(s3.S3Error) as caught:
        t.put_new(KEY, b'x')
    out = capsys.readouterr()
    recorded = repr(creds) + str(creds) + repr(t) + repr(caught.value) + str(caught.value) + caplog.text + out.out + out.err
    assert signatures and SECRET not in recorded and CREDS['access_key_id'] not in repr(creds) + str(creds)
    assert 'do-not-copy' not in recorded and 'Signature=' not in recorded


class FakeBucket:
    def __init__(self):
        self.data, self.calls = {}, []
        self.hook = lambda method, key: None

    def __call__(self, method, url, headers, body):
        parsed = urlsplit(url)
        key = unquote(parsed.path).split('/', 2)[2] if parsed.path.count('/') >= 2 else ''
        headers = {k.lower(): v for k, v in headers.items()}
        self.calls.append((method, key, {k: v for k, v in headers.items() if k != 'authorization'}))
        self.hook(method, key)
        if parsed.query:
            prefix = parse_qs(parsed.query)['prefix'][0]
            return response(body=listing(sorted(k for k in self.data if k.startswith(prefix))))
        if method == 'HEAD' and not key:
            return response(Date=DATE)
        if method == 'PUT':
            assert headers['if-none-match'] == '*'
            if key in self.data:
                return response(412, b'<Error><Code>PreconditionFailed</Code></Error>')
            self.data[key] = body
            return response(200)
        if method == 'DELETE':
            self.data.pop(key, None)
            return response(204)
        if key not in self.data:
            return response(404, b'<Error><Code>NoSuchKey</Code></Error>')
        return response(body=self.data[key] if method == 'GET' else b'',
                        **{'Content-Length': str(len(self.data[key])), 'Last-Modified': DATE})


def s3_fixture(fixture):
    fake = FakeBucket()
    t = transport(fake)
    return (fixture[0], t, *fixture[2:]), fake


def test_parity_base_publish(fixture):
    f, fake = s3_fixture(fixture)
    m = pub(f, part_size=8192)
    assert m['generation'] == 1 and m['database']['deltas'] == []
    assert len(m['database']['base']['parts']) > 1
    assert cp._sha(rebuild(f, m, 'restored')) == m['database']['file_sha256']
    assert all(PREFIX + '/' + obj['key'] in fake.data for obj in objects(m))


def test_parity_delta_publish(fixture):
    f, fake = s3_fixture(fixture)
    first = pub(f, part_size=8192)
    grow(f)
    m = pub(f, part_size=8192)
    assert m['generation'] == 2 and len(m['database']['deltas']) == 1
    assert m['database']['base'] == first['database']['base']
    with closing(cp._ro(f[0] / 'lcm.db')) as db:
        count = db.execute('SELECT count(*) FROM messages').fetchone()[0]
    restored = f[0].parent / 'restored'
    rebuild(f, m, 'restored')
    with closing(cp._ro(restored)) as db:
        assert db.execute('SELECT count(*) FROM messages').fetchone()[0] == count
    assert fake.data


def test_parity_chain_conflict(fixture):
    f, fake = s3_fixture(fixture)
    pub(f)
    grow(f)
    def collide(method, key):
        if method == 'PUT' and key == PREFIX + '/gen/2.json':
            fake.data[key] = b'{}'
    fake.hook = collide
    refused('chain_conflict', pub, f)
    assert fake.data[PREFIX + '/gen/2.json'] == b'{}'


def test_parity_resume_partial(fixture):
    f, fake = s3_fixture(fixture)
    stopped = []
    def interrupt(method, key):
        if method == 'PUT' and '/parts/' in key and not stopped:
            stopped.append(key)
        elif method == 'PUT' and '/parts/' in key:
            raise OSError('synthetic interruption')
    fake.hook = interrupt
    with pytest.raises(s3.S3Error):
        pub(f, part_size=8192)
    assert stopped[0] in fake.data
    fake.hook = lambda *args: None
    m = pub(f, part_size=8192)
    assert m['generation'] == 1 and stopped[0] in fake.data
    assert all(PREFIX + '/' + obj['key'] in fake.data for obj in objects(m))


def test_parity_retention_delete(fixture):
    f, fake = s3_fixture(fixture)
    for _ in range(4):
        m = pub(f, part_size=8192)
        grow(f)
    assert KEY in fake.data
    m.update(pins=[], forgotten=[])
    result = cg.retention(f[1], PREFIX, m, uploader=cp)
    assert KEY not in fake.data and PREFIX + '/gen/4.json' in fake.data
    assert result is not False


@pytest.mark.parametrize('second', [412, 200, 409, 403])
def test_host_probe(second):
    r = Responses(response(200), response(second), response(204))
    t = transport(r)
    if second == 412:
        assert s3.check_host(t) == {'conditional_create': True}
    else:
        with pytest.raises(cp.Refused if second == 200 else s3.S3Error) as caught:
            s3.check_host(t)
        if second == 200:
            assert caught.value.code == 'host_no_conditional_create'
    assert [c[0] for c in r.calls] == ['PUT', 'PUT', 'DELETE']
    paths = [unquote(urlsplit(c[1]).path) for c in r.calls]
    assert len(set(paths)) == 1
    assert paths[0].startswith('/bucket/' + PREFIX + '/probe/')
    token = paths[0].rsplit('/', 1)[1]
    assert len(token) == 32 and int(token, 16) >= 0


def test_tls_verified_and_timeouts(monkeypatch):
    seen = []
    class Connection:
        def __init__(self, host, *, timeout, context):
            assert timeout == 30
            assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
            seen.append(host)
            self.sock = self
        def settimeout(self, timeout):
            assert timeout == 30
        def request(self, method, path, body, headers):
            assert method == 'GET' and path == '/bucket/' + KEY
        def getresponse(self):
            return self
        status = 200
        def getheaders(self):
            return []
        def read(self):
            return b'over TLS'
        def close(self):
            pass
    monkeypatch.setattr(s3.http.client, 'HTTPSConnection', Connection)
    # Inject the real HTTP adapter, backed by a socket-free HTTPSConnection double.
    t = transport(lambda *args: None)
    t.request = t._request
    assert t.get(KEY) == b'over TLS'
    assert seen == ['s3.example.invalid']


def test_import_rule_44():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'cloud_s3.py').read_text())
    siblings = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
    assert len(siblings) == 1
    assert siblings[0].level == 1 and siblings[0].module is None
    assert [(n.name, n.asname) for n in siblings[0].names] == [('cloud_copy', '_copy')]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.level == 1 and node.module is None and [n.name for n in node.names] == ['cloud_copy']) or (
                node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)
    # Synthetic package avoids the host plugin __init__, exactly like the cloud CLI.
    code = ('import sys,types; p=types.ModuleType("hermes_lcm"); p.__path__=[sys.argv[1]]; '
            'sys.modules["hermes_lcm"]=p; from hermes_lcm import cloud_s3; '
            'assert callable(cloud_s3.S3Transport)')
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(root)], capture_output=True)
    assert result.returncode == 0
    assert '"cloud_s3.py",' in (root / 'scripts/maintainer_gate.py').read_text()


def test_live_contract_offline():
    from tests.test_cloud_s3_live import exercise
    fake = FakeBucket()
    a = transport(fake)
    attempts = []
    def denied(method, url, headers, body):
        attempts.append(method)
        return response(403, b'<Error><Code>AccessDenied</Code></Error>')
    b = s3.S3Transport('s3.example.invalid', 'bucket', 'us-east-1',
                       'copies/11111111-1111-1111-1111-111111111111',
                       {'access_key_id': 'fixture-key', 'secret_access_key': 'fixture-secret'},
                       request=denied)
    exercise(a, b)
    assert sum(method == 'HEAD' and key == '' for method, key, _ in fake.calls) == 1
    assert attempts == ['GET', 'GET', 'PUT', 'DELETE']
    assert fake.data == {}


@pytest.mark.parametrize('method', ['get', 'stat', 'list', 'put_new', 'delete', 'now'])
@pytest.mark.parametrize('status', [301, 400, 403])
def test_non_success_status(method, status):
    r = Responses(response(status, b'<Error><Code>AccessDenied</Code></Error>', Date=DATE))
    args = [] if method == 'now' else [PREFIX + '/gen/' if method == 'list' else KEY]
    if method == 'put_new':
        args.append(b'x')
    if method == 'now' and status == 403:
        try:
            result = transport(r).now()
        except s3.S3Error:
            result = None
        assert result == EPOCH
    else:
        with pytest.raises(s3.S3Error) as caught:
            getattr(transport(r), method)(*args)
        assert (caught.value.status, caught.value.code) == (status, 'AccessDenied')
    assert len(r.calls) == 1


@pytest.mark.parametrize('body', [b'invalid', b'<Error/>',
    b'<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>',
    b'<ListBucketResult><Contents/></ListBucketResult>',
    b'<?xml version="1.0" encoding="unknown-encoding"?><ListBucketResult/>'])
def test_malformed_listing(body):
    with pytest.raises(Exception) as caught:
        transport(Responses(response(body=body))).list(PREFIX + '/gen/')
    assert isinstance(caught.value, s3.S3Error)


def test_unknown_encoding_error():
    body = b'<?xml version="1.0" encoding="unknown-encoding"?><Error><Code>AccessDenied</Code></Error>'
    with pytest.raises(Exception) as caught:
        transport(Responses(response(403, body))).get(KEY)
    assert isinstance(caught.value, s3.S3Error)
    assert (caught.value.status, caught.value.code) == (403, 'UnknownError')


@pytest.mark.parametrize('field', ['access_key_id', 'secret_access_key'])
@pytest.mark.parametrize('suffix', ['\u00e9', '\t', '\x00', '\x7f'])
def test_credentials_invalid_values(tmp_path, field, suffix):
    secret = SECRET + suffix
    (tmp_path / 'cloud').mkdir()
    path = tmp_path / 'cloud/credentials.json'
    path.write_text(json.dumps({**CREDS, field: secret}))
    path.chmod(0o600)
    try:
        s3.load_credentials(tmp_path)
    except Exception as error:
        caught = error
    else:
        caught = None
    assert isinstance(caught, cp.Refused)
    assert caught.code == 'credentials_mode'
    assert secret not in str(caught) + repr(caught) + repr(caught.args)


def test_repeated_continuation_token():
    r = Responses(response(body=listing(token='same')), response(body=listing(token='same')))
    with pytest.raises(s3.S3Error):
        transport(r).list(PREFIX + '/gen/')
    assert len(r.calls) == 2


@pytest.mark.parametrize('headers', [{}, {'Content-Length': '-1', 'Last-Modified': DATE},
    {'Content-Length': '1', 'Last-Modified': 'invalid'}])
def test_malformed_stat(headers):
    with pytest.raises(s3.S3Error):
        transport(Responses((200, headers, b''))).stat(KEY)


def test_credential_value_repr_masked():
    creds = s3.Credentials(CREDS['access_key_id'], SECRET)
    assert all(v not in str(creds) + repr(creds) for v in CREDS.values())
    assert str(creds) == repr(creds) == 'Credentials(<redacted>)'


def test_credentials_not_regular(tmp_path):
    (tmp_path / 'cloud/credentials.json').mkdir(parents=True, mode=0o700)
    refused('credentials_mode', s3.load_credentials, tmp_path)


def test_host_probe_first_failure():
    r = Responses(response(403, b'<Error><Code>AccessDenied</Code></Error>'), response(204))
    with pytest.raises(s3.S3Error):
        s3.check_host(transport(r))
    assert [c[0] for c in r.calls] == ['PUT', 'DELETE']


def test_live_configured_credentials_file(tmp_path, monkeypatch):
    from tests.test_cloud_s3_live import configured
    credential_file = tmp_path / 'private-credentials.json'
    credential_file.write_text(json.dumps(CREDS))
    credential_file.chmod(0o600)
    config_file = tmp_path / 'host.json'
    config_file.write_text(json.dumps(dict(endpoint='s3.example.invalid', bucket='bucket',
        region='us-east-1', prefix=PREFIX, credentials_file=str(credential_file))))
    monkeypatch.setenv('LCMX_S3_LIVE_A', str(config_file))
    constructor = s3.S3Transport
    fake = FakeBucket()
    def offline(*args):
        return constructor(*args, request=fake, clock=lambda: EPOCH)
    monkeypatch.setattr(s3, 'S3Transport', offline)
    t = configured('LCMX_S3_LIVE_A')
    assert t.credentials.secret_access_key == SECRET
    assert t.context.verify_mode == ssl.CERT_REQUIRED and t.context.check_hostname
    t.put_new(KEY, b'fixture')
    assert t.get(KEY) == b'fixture'
    credential_file.chmod(0o640)
    refused('credentials_mode', configured, 'LCMX_S3_LIVE_A')
