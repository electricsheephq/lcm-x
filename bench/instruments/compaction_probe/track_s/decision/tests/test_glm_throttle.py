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
        return io.BytesIO(json.dumps({"model": glm.MODEL, "choices": [{"message": {"content": response.get("content", "answer") if isinstance(response, dict) else "answer"}}],
                                     "usage": response.get("usage") if isinstance(response, dict) else {"prompt_tokens": 7, "completion_tokens": 2}}).encode())

    monkeypatch.setattr(glm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(glm.time, "sleep", sleeps.append)
    monkeypatch.setattr(glm.time, "monotonic", lambda: 0)
    return responses, requests, sleeps


def brakes():
    shell = (TRACK / "decision/run_pmw_cell.sh").read_text()
    shell_pattern = re.search(r"grep -q -i -E '([^']+)'", shell)[1]
    pmw = (TRACK / "decision/pmw_read.py").read_text()
    pmw_pattern = re.search(r'env = bool\(re.search\(r"([^"]+)"', pmw)[1]
    return [load("d2_driver").LIMIT, re.compile(shell_pattern, re.I), re.compile(pmw_pattern, re.I),
            c1_budget.LIMIT, seam.LIMIT_RE]


def test_recovers_identical_request_and_records_call(client, transport, capsys, monkeypatch):
    kind, obj, call = client
    responses, requests, sleeps = transport
    responses.extend([http_error(), "ok", "ok"])
    now = [0]
    monkeypatch.setattr(glm.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(glm.time, "sleep", lambda delay: (sleeps.append(delay), now.__setitem__(0, now[0] + delay)))
    text, meta = call()
    assert text == "answer" and meta["throttle_retries"] == 1
    assert sleeps == [30]
    assert requests[0][:3] == requests[1][:3]  # same Request, bytes and headers
    assert requests[1][3] == requests[0][3] - 30
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


@pytest.fixture
def clock(monkeypatch, transport):
    now, durations = [0.0], []
    responses, requests, sleeps = transport
    original = glm.urllib.request.urlopen

    def urlopen(req, timeout):
        now[0] += durations.pop(0)
        return original(req, timeout)

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(glm.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(glm.time, "sleep", sleep)
    monkeypatch.setattr(glm.urllib.request, "urlopen", urlopen)
    return now, durations, responses, requests, sleeps


def test_budget_bounds_attempts_and_sleep(client, clock, monkeypatch):
    kind, _, call = client
    now, durations, responses, requests, sleeps = clock
    monkeypatch.setattr(reader if kind == "reader" else glm,
                        "READER_TIMEOUT_S" if kind == "reader" else "CALL_TIMEOUT_S", 40)
    durations.extend([2, 7])
    responses.extend([http_error(), "ok"])
    _, meta = call()
    assert meta["throttle_retries"] == 1
    assert [r[3] for r in requests] == [40, 8]
    assert sleeps == [30] and now[0] == 39 <= 40


@pytest.mark.parametrize("budget", [30, 31])
def test_deadline_skips_retry_and_stays_terminal(client, clock, monkeypatch, budget):
    kind, obj, call = client
    now, durations, responses, requests, sleeps = clock
    monkeypatch.setattr(reader if kind == "reader" else glm,
                        "READER_TIMEOUT_S" if kind == "reader" else "CALL_TIMEOUT_S", budget)
    durations.extend([0] * 4)
    responses.extend([http_error() for _ in range(4)])
    receipts = []
    original = glm._open

    def observed(req, timeout, receipt):
        receipts.append(receipt)
        return original(req, timeout, receipt)

    monkeypatch.setattr(glm, "_open", observed)
    with pytest.raises(urllib.error.HTTPError if kind == "reader" else LaneError) as caught:
        call()
    assert len(requests) == 1 and sleeps == [] and now[0] == 0
    assert receipts[0]["throttle_deadline_stop"] is True
    assert receipts[0]["throttle_retries"] == 0
    assert load("d2_driver").LIMIT.search(str(caught.value))
    if kind == "reader":
        assert obj.readback["throttle_deadline_stop"] is True


@pytest.mark.parametrize("body", [b"usage-limit", b"usage_limit"])
def test_all_quota_separators_are_terminal(client, transport, body):
    kind, _, call = client
    responses, requests, sleeps = transport
    responses.extend([http_error(body=body) for _ in range(4)])
    with pytest.raises(urllib.error.HTTPError if kind == "reader" else LaneError):
        call()
    assert len(requests) == 1 and sleeps == []


def test_reader_fallback_sums_each_http_call(monkeypatch, transport):
    monkeypatch.setenv("GLM_API_KEY", "synthetic-key")
    obj = reader.GLMReader()
    responses, _, _ = transport
    responses.extend([http_error(), http_error(400), http_error(), "ok"])
    _, info = obj.ask("system", [], ["question"], [])
    assert info["throttle_retries"] == 2
    assert info["usage"] == {"prompt_tokens": 7, "completion_tokens": 2}
    assert obj.readback["throttle_retries_total"] == 2


@pytest.mark.parametrize("mode", ["single", "tools", "capped"])
def test_reader_calls_keep_each_ask_count(monkeypatch, transport, mode):
    from types import SimpleNamespace
    monkeypatch.setenv("GLM_API_KEY", "synthetic-key")
    obj = reader.GLMReader()
    responses, _, _ = transport
    tool = {"content": '{"tool":"lcm_grep","args":{}}', "usage": None}
    final = {"content": '{"answer":{"q":"answer"}}' if mode != "single" else '{"q":"answer"}', "usage": None}
    if mode == "single":
        responses.extend([http_error(), final])
        counts, engine = [1], None
    else:
        responses.extend([http_error(), tool, http_error(), http_error(), final])
        counts = [1, 2]
        engine = SimpleNamespace(handle_tool_call=lambda *_: "synthetic result")
        if mode == "capped":
            monkeypatch.setattr(reader, "RUNAWAY_GUARD", 0)
    answers, meta = reader.answer(obj, "system", [], "question", tools_engine=engine, schemas=[])
    assert answers == {"q": "answer"}
    assert [call["throttle_retries"] for call in meta["reader_calls"]] == counts
    assert obj.readback["throttle_retries_total"] == sum(counts)
    assert meta["tool_calls"] == (1 if mode == "tools" else 0)
    assert meta["runaway_guard_hit"] is (mode == "capped")
