"""Synthetic HTTP responses only: shared retry policy and the live brake patterns."""
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error

import pytest

TRACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK / "harness"))
sys.path.insert(0, str(TRACK / "s2"))
from lanes import LaneError, glm  # noqa: E402
from s2lib import c1_budget, reader, seam  # noqa: E402
from test_d2_calibrated import load  # noqa: E402


def http_error(code=429, body=b"temporary congestion", headers=None):
    return urllib.error.HTTPError(glm.URL, code, "Too Many Requests" if code == 429 else "Bad Request",
                                  headers or {}, io.BytesIO(body))


@pytest.fixture(params=["reader", "lane"])
def client(request, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "synthetic-key")
    if request.param == "reader":
        obj = reader.GLMReader()

        def call():
            text, usage = obj._post([{"role": "user", "content": "synthetic"}])
            return text, obj.readback
    else:
        obj = glm.GLMLane()

        def call():
            reply = obj._call("system", ["synthetic"], 17)
            return reply.text, reply.meta
    return request.param, obj, call


@pytest.fixture
def transport(monkeypatch):
    requests, sleeps, responses = [], [], []

    def urlopen(req, timeout):
        requests.append((req, req.data, dict(req.headers), timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps({"model": glm.MODEL, "choices": [{"message": {"content": "answer"}}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 2}}).encode())

    monkeypatch.setattr(glm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(glm.time, "sleep", sleeps.append)
    return responses, requests, sleeps


def brakes():
    shell = (TRACK / "decision/run_pmw_cell.sh").read_text()
    shell_pattern = re.search(r"grep -q -i -E '([^']+)'", shell)[1]
    pmw = (TRACK / "decision/pmw_read.py").read_text()
    pmw_pattern = re.search(r'env = bool\(re.search\(r"([^"]+)"', pmw)[1]
    return [load("d2_driver").LIMIT, re.compile(shell_pattern, re.I), re.compile(pmw_pattern, re.I),
            c1_budget.LIMIT, seam.LIMIT_RE]


def test_recovers_identical_request_and_records_call(client, transport, capsys):
    kind, obj, call = client
    responses, requests, sleeps = transport
    responses.extend([http_error(), "ok", "ok"])
    text, meta = call()
    assert text == "answer" and meta["throttle_retries"] == 1
    assert sleeps == [30]
    assert requests[0] == requests[1]  # same Request, bytes, headers and timeout
    line = capsys.readouterr().err
    assert line == "GLM throttle: retry 1/3 after 30 s\n"
    assert all(not pattern.search(line) for pattern in brakes())
    # Also execute the shell's actual ERE, rather than only its Python equivalent.
    pattern = brakes()[1].pattern
    assert subprocess.run(["grep", "-i", "-E", pattern], input=line, text=True, capture_output=True).returncode == 1
    _, meta = call()
    assert meta["throttle_retries"] == 0
    if kind == "reader":
        assert obj.readback["throttle_retries_total"] == 1


def test_exhausted_still_brakes(client, transport, capsys):
    kind, _, call = client
    responses, requests, sleeps = transport
    errors = [http_error() for _ in range(4)]
    responses.extend(errors)
    with pytest.raises(urllib.error.HTTPError if kind == "reader" else LaneError) as caught:
        call()
    assert len(requests) == 4 and sleeps == [30, 60, 120]
    assert load("d2_driver").LIMIT.search(str(caught.value))
    lines = capsys.readouterr().err.splitlines()
    assert lines == [f"GLM throttle: retry {n}/3 after {seconds} s" for n, seconds in enumerate(sleeps, 1)]
    assert all(not pattern.search(line) for pattern in brakes() for line in lines)
    if kind == "reader":
        assert caught.value is errors[-1]
        assert caught.value.read() == b"temporary congestion"
    else:
        assert str(caught.value) == "HTTP 429: temporary congestion"


@pytest.mark.parametrize("body", [b"Usage Limit reached", b'{"error":{"message":"QUOTA exhausted"}}',
                                  b"x" * 301 + b"quota"])
def test_quota_is_terminal(client, transport, body, capsys):
    kind, _, call = client
    responses, requests, sleeps = transport
    responses.append(http_error(body=body))
    with pytest.raises(urllib.error.HTTPError if kind == "reader" else LaneError) as caught:
        call()
    assert len(requests) == 1 and sleeps == [] and capsys.readouterr().err == ""
    if kind == "reader":
        assert caught.value.read() == body
    else:
        assert str(caught.value) == "HTTP 429: " + body.decode()[:300]


@pytest.mark.parametrize("header,seconds", [("7", 7), ("999", 120), ("0", 0), ("1.5", 1.5),
                                            ("7.429", 7.429), ("0.000429", 0.000429),
                                            ("invalid", 30), ("-1", 30), ("nan", 30)])
def test_retry_after(client, transport, header, seconds, capsys):
    _, _, call = client
    responses, _, sleeps = transport
    responses.extend([http_error(headers={"Retry-After": header}), "ok"])
    _, meta = call()
    assert sleeps == [seconds] and meta["throttle_retries"] == 1
    line = capsys.readouterr().err
    assert all(not pattern.search(line) for pattern in brakes())


def test_non_throttle_is_not_retried(client, transport):
    kind, _, call = client
    responses, requests, sleeps = transport
    error = http_error(400, b"bad row")
    responses.append(error)
    with pytest.raises(urllib.error.HTTPError if kind == "reader" else LaneError) as caught:
        call()
    assert len(requests) == 1 and sleeps == []
    if kind == "reader":
        assert caught.value is error and caught.value.read() == b"bad row"
    else:
        assert str(caught.value) == "HTTP 400: bad row"


def test_reader_400_compatibility(monkeypatch, transport):
    monkeypatch.setenv("GLM_API_KEY", "synthetic-key")
    obj = reader.GLMReader()
    responses, requests, sleeps = transport
    responses.extend([http_error(400), "ok"])
    text, meta = obj.ask("system", [{"role": "user", "content": "context"}], ["question"], [])
    assert text == "answer" and meta["context_format"] == "flattened (native rejected: HTTP 400)"
    assert meta["usage"] == {"prompt_tokens": 7, "completion_tokens": 2}
    assert len(requests) == 2 and sleeps == []
    first, second = (json.loads(item[1]) for item in requests)
    assert first["messages"][0]["role"] == "system"
    assert second["messages"][0] == {"role": "user", "content": "[SYSTEM]\nsystem\n\n[USER]\ncontext"}
    for payload in (first, second):
        assert payload["max_tokens"] == 8192 and payload["thinking"] == {"type": "disabled"}
