"""Standard-library, TLS-only S3 transport for create-only hosted copies."""
import datetime
from email.utils import parsedate_to_datetime
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import stat
import time
from urllib.parse import quote, urlsplit
import uuid
import xml.etree.ElementTree as ET

from . import cloud_copy as _copy

Refused = _copy.Refused
_RETRY = {500, 502, 503, 504}
# A server-controlled Code must not become an exception-text exfiltration path.
_CODES = set('AccessDenied NoSuchKey NoSuchBucket PreconditionFailed ConditionalRequestConflict '
             'SlowDown InternalError ServiceUnavailable RequestTimeout SignatureDoesNotMatch '
             'InvalidAccessKeyId InvalidToken ExpiredToken InvalidArgument InvalidRequest '
             'AuthorizationHeaderMalformed PermanentRedirect TemporaryRedirect NotImplemented '
             'MethodNotAllowed EntityTooLarge EntityTooSmall BadDigest RequestTimeTooSkewed '
             'InvalidBucketName InvalidObjectState OperationAborted AllAccessDisabled'.split())


class S3Error(OSError):
    """A deliberately small diagnostic, excluding request and exception details."""
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(f'{status} {code}')


class Credentials:
    def __init__(self, access_key_id, secret_access_key):
        self.access_key_id, self.secret_access_key = access_key_id, secret_access_key

    def __repr__(self):
        return 'Credentials(<redacted>)'

    __str__ = __repr__


def load_credentials(home):
    """Read credentials only from the fixed file."""
    return _read_credentials(Path(home) / 'cloud' / 'credentials.json')


def _read_credentials(path):
    """The live gate reuses this permission check for its explicit file paths."""
    def allowed(info):
        return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077
    try:
        before = path.lstat()
        if not allowed(before):
            raise Refused('credentials_mode')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'r', encoding='utf-8') as handle:
            info = os.fstat(handle.fileno())
            if not allowed(info) or (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                raise Refused('credentials_mode')
            raw = json.load(handle)
        values = [raw[k] for k in ('access_key_id', 'secret_access_key')]
        if not all(isinstance(v, str) and v and v.isascii() and v.isprintable() for v in values):
            raise ValueError
        return Credentials(*values)
    except (OSError, ValueError, TypeError, KeyError):
        raise Refused('credentials_mode') from None


def _credentials(value):
    if isinstance(value, Credentials):
        return value
    return Credentials(value['access_key_id'], value['secret_access_key'])


def _query(query):
    return '&'.join(k + '=' + v for k, v in sorted(
        (quote(k, safe='-_.~'), quote(v, safe='-_.~')) for k, v in query))


def sign(method, path, query, headers, body, credentials, region, timestamp):
    """AWS SigV4 S3 single-chunk signing; path is the unescaped object path."""
    credentials = _credentials(credentials)
    if not isinstance(timestamp, str):
        timestamp = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    headers = {k.lower(): ' '.join(v.split()) for k, v in headers.items() if k.lower() != 'authorization'}
    headers['x-amz-date'] = timestamp
    headers['x-amz-content-sha256'] = hashlib.sha256(body).hexdigest()
    names = ';'.join(sorted(headers))
    canonical_headers = ''.join(k + ':' + headers[k] + '\n' for k in sorted(headers))
    canonical = '\n'.join((method, quote(path, safe='/-_.~'), _query(query), canonical_headers,
                            names, headers['x-amz-content-sha256']))
    scope = timestamp[:8] + '/' + region + '/s3/aws4_request'
    to_sign = 'AWS4-HMAC-SHA256\n' + timestamp + '\n' + scope + '\n' + hashlib.sha256(canonical.encode()).hexdigest()
    key = ('AWS4' + credentials.secret_access_key).encode()
    for part in (timestamp[:8], region, 's3', 'aws4_request'):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers['authorization'] = ('AWS4-HMAC-SHA256 Credential=' + credentials.access_key_id + '/' + scope
                                + ', SignedHeaders=' + names + ', Signature=' + signature)
    return headers


def _xml(body):
    try:
        root = ET.fromstring(body)
        for node in root.iter():
            node.tag = node.tag.rsplit('}', 1)[-1]
        return root
    except (ET.ParseError, ValueError, LookupError):
        raise S3Error(0, 'InvalidResponse') from None


def _error(status, body):
    try:
        code = _xml(body).findtext('Code')
    except S3Error:
        code = None
    return S3Error(status, code if code in _CODES else 'UnknownError')


def _date(value):
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            raise ValueError
        return date.timestamp()
    except (ValueError, TypeError, OverflowError):
        raise S3Error(0, 'InvalidResponse') from None


def _ca_directory():
    # OpenSSL loads hashed directory certificates lazily, during a handshake.
    directory = ssl.get_default_verify_paths().capath
    try:
        return bool(directory) and any(re.fullmatch(r'[0-9a-f]{8}\.[0-9]', name) for name in os.listdir(directory))
    except OSError:
        return False


class S3Transport:
    """Path-style S3 adapter. Inject request for offline use; PUT is never retried."""
    def __init__(self, endpoint, bucket, region, prefix, credentials, *, request=None, clock=None):
        parsed = urlsplit(endpoint if '://' in endpoint else 'https://' + endpoint)
        if parsed.scheme != 'https':
            raise Refused('tls_required')
        if not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise Refused('bad_name')
        _copy._name(bucket, r'[A-Za-z0-9][A-Za-z0-9._-]*')
        _copy._name(region, r'[A-Za-z0-9_-]+')
        _copy.check_key(prefix, f'{prefix}/gen/', listing=True)
        self.context = ssl.create_default_context()
        if not self.context.get_ca_certs() and not _ca_directory():
            raise Refused('tls_no_ca_bundle')
        self.host, self.bucket, self.region, self.prefix = parsed.netloc, bucket, region, prefix
        self.credentials = _credentials(credentials)
        self.request, self.clock = request or self._request, clock or time.time

    def _request(self, method, url, headers, body):
        parsed = urlsplit(url)
        conn = http.client.HTTPSConnection(self.host, timeout=30, context=self.context)
        try:
            conn.request(method, parsed.path + ('?' + parsed.query if parsed.query else ''), body, headers)
            if conn.sock is not None:
                conn.sock.settimeout(30)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def _send(self, method, key='', query=(), body=b'', headers=None):
        # Private raw boundary: check_host uses probe keys; live tests bypass prefix validation.
        path = '/' + self.bucket + ('/' + key if key else '')
        url = 'https://' + self.host + quote(path, safe='/-_.~')
        if query:
            url += '?' + _query(query)
        attempts = 1 if method == 'PUT' else 3
        for attempt in range(attempts):
            signed = sign(method, path, query, {'host': self.host, **(headers or {})}, body,
                          self.credentials, self.region, self.clock())
            try:
                status, reply_headers, reply_body = self.request(method, url, signed, body)
            except (ConnectionResetError, TimeoutError, ssl.SSLEOFError, http.client.IncompleteRead):
                if attempt + 1 == attempts:
                    raise S3Error(0, 'ConnectionError') from None
            except (OSError, http.client.HTTPException):
                raise S3Error(0, 'ConnectionError') from None
            else:
                if status not in _RETRY or attempt + 1 == attempts:
                    return status, {k.lower(): v for k, v in reply_headers.items()}, reply_body
            time.sleep((1, 4)[attempt])

    def _ensure(self, status, headers, body):
        if not 200 <= status < 300:
            raise _error(status, body)
        return body

    def get(self, key):
        _copy.check_key(self.prefix, key)
        result = self._send('GET', key)
        if result[0] == 404:
            raise FileNotFoundError(key)
        return self._ensure(*result)

    def stat(self, key):
        _copy.check_key(self.prefix, key)
        result = self._send('HEAD', key)
        if result[0] == 404:
            return None
        self._ensure(*result)
        try:
            size = int(result[1]['content-length'])
            if size < 0:
                raise ValueError
            return {'bytes': size, 'time': _date(result[1].get('last-modified'))}
        except (ValueError, KeyError):
            raise S3Error(0, 'InvalidResponse') from None

    def list(self, prefix):
        _copy.check_key(self.prefix, prefix, listing=True)
        keys, tokens, token = set(), set(), None
        while True:
            query = [('list-type', '2'), ('prefix', prefix), ('delimiter', '/')]
            if token is not None:
                query.append(('continuation-token', token))
            result = self._send('GET', query=query)
            root = _xml(self._ensure(*result))
            if root.tag != 'ListBucketResult':
                raise S3Error(0, 'InvalidResponse')
            if root.find('CommonPrefixes') is not None:
                raise Refused('bad_name')
            for entry in root.findall('Contents'):
                key = entry.findtext('Key')
                if key is None:
                    raise S3Error(0, 'InvalidResponse')
                _copy.check_key(self.prefix, key)
                if not key.startswith(prefix):
                    raise Refused('bad_name')
                keys.add(key)
            truncated = root.findtext('IsTruncated')
            if truncated == 'false':
                return sorted(keys)
            token = root.findtext('NextContinuationToken')
            if truncated != 'true' or not token or token in tokens:
                raise S3Error(0, 'InvalidResponse')
            tokens.add(token)

    def put_new(self, key, data):
        _copy.check_key(self.prefix, key)
        body = data.read() if hasattr(data, 'read') else data
        result = self._send('PUT', key, body=body, headers={'If-None-Match': '*'})
        if result[0] == 412 or (result[0] == 409 and self.stat(key) is not None):
            raise FileExistsError(key)
        self._ensure(*result)

    def delete(self, key):
        _copy.check_key(self.prefix, key)
        result = self._send('DELETE', key)
        if result[0] != 404:
            self._ensure(*result)

    def now(self):
        result = self._send('HEAD')
        if result[0] != 403:
            self._ensure(*result)
        return _date(result[1].get('date'))


def check_host(transport):
    """Prove create-if-absent with an unpublishable, private probe key."""
    key = transport.prefix + '/probe/' + uuid.uuid4().hex
    try:
        transport._ensure(*transport._send('PUT', key, headers={'If-None-Match': '*'}))
        second = transport._send('PUT', key, headers={'If-None-Match': '*'})
        if 200 <= second[0] < 300:
            raise Refused('host_no_conditional_create')
        if second[0] != 412:
            transport._ensure(*second)
        return {'conditional_create': True}
    finally:
        deleted = transport._send('DELETE', key)
        if deleted[0] != 404:
            transport._ensure(*deleted)
