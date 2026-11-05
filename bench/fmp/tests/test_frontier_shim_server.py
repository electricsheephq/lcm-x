"""Synthetic requests only; no real lanes, credentials or subprocesses."""
import hashlib
import http.client
import json
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from bench.fmp.models import Backends, BudgetStopped, Quota, ToolViolation
from bench.frontier_shim import server
from bench.frontier_shim.models import MODELS

REQUEST = {"model": "sol-low", "messages": [
    {"role": "system", "content": "synthetic system text"},
    {"role": "user", "content": "synthetic question text"}], "temperature": 0}
ANSWER = "synthetic answer text"
RECORD = {"returned_model": "gpt-6.1-sol", "transcripts": ["events", "rollout"],
          "identity_source": "response", "weekly_used_percent": 24.0}


@pytest.fixture
def lane(tmp_path):
    def call(spec, messages, records):
        records.append({**RECORD, "input_tokens": 123,
                        "output_tokens": 7, "total_tokens": 130})
        return {"role": "assistant", "content": ANSWER}

    backend = SimpleNamespace(call=Mock(side_effect=call), quota=SimpleNamespace(check=Mock()))
    shim = server.Shim(backend, tmp_path / "calls.jsonl")
    return shim, backend


def invoke(shim, request=REQUEST):
    status, headers, payload = shim.handle(server.encode(request))
    return status, headers, json.loads(payload)


def test_shape_and_safe_receipt(lane, monkeypatch, capsys):
    shim, backend = lane
    marker = "sk-synthetic-credential-not-a-real-key"
    monkeypatch.setenv("OPENAI_API_KEY", marker)
    monkeypatch.setenv("GLM_API_KEY", marker)
    monkeypatch.setenv("LCMX_SHIM_CODEX_HOME", marker)
    status, headers, response = invoke(shim)
    assert status == 200 and headers == {}
    assert response["choices"][0]["message"] == {"role": "assistant", "content": ANSWER}
    assert response["usage"] == {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130}
    assert response["model"] == "gpt-6.1-sol"
    args = backend.call.call_args.args
    assert args[:2] == (MODELS["sol-low"], REQUEST["messages"])
    assert not backend.call.call_args.kwargs  # temperature never reaches the reasoning lane
    receipt = json.loads(shim.log.read_text())
    assert receipt["request_sha256"] == hashlib.sha256(server.encode(REQUEST)).hexdigest()
    assert receipt["response_sha256"] == hashlib.sha256(server.encode(response)).hexdigest()
    assert receipt["spec"] == "codex:gpt-6.1-sol@low"
    assert receipt["served_model"] == "gpt-6.1-sol" and receipt["retries"] == 0
    assert receipt["status"] == 200 and receipt["latency_s"] >= 0
    assert receipt["temperature_applied"] is False and receipt["time"].endswith("Z")
    assert receipt["identity_source"] == "response" and receipt["weekly_used_percent"] == 24.0
    captured = capsys.readouterr()
    assert marker not in captured.err and marker not in captured.out
    for text in [ANSWER, marker, *[m["content"] for m in REQUEST["messages"]]]:
        assert text not in shim.log.read_text()


@pytest.mark.parametrize("usage,expected", [
    ({}, {}), ({"input_tokens": None, "output_tokens": None}, {}),
    ({"input_tokens": 0}, {"prompt_tokens": 0}),
    ({"output_tokens": 7}, {"completion_tokens": 7}),
    ({"input_tokens": 12, "output_tokens": 3}, {"prompt_tokens": 12, "completion_tokens": 3}),
])
def test_usage_missing_stays_missing(lane, usage, expected):
    shim, backend = lane

    def call(spec, messages, records):
        records.append({**RECORD, **usage})
        return {"content": ANSWER}

    backend.call.side_effect = call
    assert invoke(shim)[2]["usage"] == expected
    assert json.loads(shim.log.read_text())["usage"] == expected


@pytest.mark.parametrize("change", [
    {"model": "unknown-private-text"}, {"model": []}, {"messages": "text"},
    {"messages": []}, {"messages": [{"role": "assistant", "content": "text"}]},
    {"messages": [{"role": "user", "content": []}]}, {"stream": True},
    {"tools": []}, {"temperature": "zero"}, {"temperature": True},
])
def test_bad_request_is_400_without_dispatch_or_echo(lane, change):
    shim, backend = lane
    assert invoke(shim, {**REQUEST, **change})[0] == 400
    backend.call.assert_not_called()
    assert "unknown-private-text" not in shim.log.read_text()


@pytest.mark.parametrize("raw", [b"{bad-json", b"\xff", b"[]", b"null"])
def test_malformed_json_is_400(lane, raw):
    shim, backend = lane
    assert shim.handle(raw)[0] == 400
    backend.call.assert_not_called()


def test_usage_limit_latches_until_new_shim(lane, tmp_path):
    shim, backend = lane
    backend.call.side_effect = BudgetStopped()
    for _ in range(3):
        status, headers, response = invoke(shim)
        assert status == 429 and headers == {"Retry-After": "60"}
        assert response["error"]["message"] == "codex_stopped_restart_required"
    assert backend.call.call_count == 1 and shim.stopped.is_set()
    backend.call.side_effect = None
    backend.call.return_value = {"content": ANSWER}
    restarted = server.Shim(backend, tmp_path / "new-log.jsonl")
    assert not restarted.stopped.is_set()


def test_fmp_quota_stop_is_429(lane):
    shim, backend = lane
    backend.quota.check.side_effect = BudgetStopped()
    assert invoke(shim)[0] == 429
    backend.quota.check.side_effect = None
    assert invoke(shim)[0] == 429
    backend.call.assert_not_called()


def test_success_crossing_weekly_brake_keeps_answer(lane):
    shim, backend = lane
    backend.quota.check.side_effect = [None, None, BudgetStopped()]
    assert invoke(shim)[0] == 200 and not shim.stopped.is_set()
    assert invoke(shim)[0] == 429 and shim.stopped.is_set()
    assert backend.call.call_count == 1


@pytest.mark.parametrize("exc", [RuntimeError, TimeoutError, OSError, ValueError])
def test_transport_failure_is_503_without_sensitive_exception(lane, exc):
    shim, backend = lane
    backend.call.side_effect = exc("synthetic secret and prompt must not escape")
    assert invoke(shim)[0] == 503
    assert not shim.stopped.is_set()
    assert "synthetic secret" not in shim.log.read_text()


@pytest.mark.parametrize("model,content", [("unexpected", ANSWER), (None, ANSWER), ("gpt-6.1-sol", None)])
def test_invalid_lane_response_is_422(lane, model, content):
    shim, backend = lane
    backend.call.side_effect = lambda spec, messages, records: (
        records.append({**RECORD, "returned_model": model}) or {"content": content})
    assert invoke(shim)[0] == 422
    assert "unexpected" not in shim.log.read_text()


def test_queue_wait_is_bounded_and_slot_not_released(lane):
    shim, backend = lane
    shim.slots = Mock()
    shim.slots.acquire.return_value = False
    assert invoke(shim)[0] == 503
    shim.slots.acquire.assert_called_once_with(timeout=30)
    shim.slots.release.assert_not_called()
    backend.call.assert_not_called()


@pytest.mark.parametrize("cap", [2, 4])
def test_semaphore_caps_running_calls_and_holds_until_return(lane, cap):
    original, backend = lane
    shim = server.Shim(backend, original.log, concurrency=cap)
    entered, release = threading.Barrier(cap + 1), threading.Event()
    lock = threading.Lock()
    active = peak = 0

    def call(spec, messages, records):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            first_wave = backend.call.call_count <= cap
        if first_wave:
            entered.wait(timeout=5)
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        records.append(dict(RECORD))
        return {"content": ANSWER}

    backend.call.side_effect = call
    with ThreadPoolExecutor(max_workers=cap + 1) as pool:
        futures = [pool.submit(invoke, shim) for _ in range(cap)]
        try:
            entered.wait(timeout=5)
            assert not shim.slots.acquire(blocking=False)
            futures.append(pool.submit(invoke, shim))
        finally:
            release.set()
        assert all(f.result(timeout=5)[0] == 200 for f in futures)
    assert peak == cap
    assert len(shim.log.read_text().splitlines()) == cap + 1


def test_queued_call_checks_stop_again(lane):
    shim, backend = lane
    shim.slots = Mock()
    shim.slots.acquire.side_effect = lambda **_: (shim.stopped.set() or True)
    assert invoke(shim)[0] == 429
    backend.call.assert_not_called()
    shim.slots.release.assert_called_once()


def test_draining_refuses_new_and_queued_requests(lane):
    shim, backend = lane
    shim.draining.set()
    assert invoke(shim) == (503, {}, server.error("shutting_down"))
    shim.draining.clear()
    shim.slots = Mock()
    shim.slots.acquire.side_effect = lambda **_: (shim.draining.set() or True)
    assert invoke(shim)[2] == server.error("shutting_down")
    backend.call.assert_not_called()
    shim.slots.release.assert_called_once()


@pytest.mark.parametrize("cap", [0, 5, True])
def test_concurrency_guard(lane, tmp_path, cap):
    with pytest.raises(ValueError):
        server.Shim(lane[1], tmp_path / "log", concurrency=cap)


def test_http_loopback_shape_and_arbitrary_bearer(lane):
    shim, _ = lane
    with server.Server(shim, port=0) as httpd:
        assert httpd.server_address[0] == "127.0.0.1"
        worker = threading.Thread(target=httpd.serve_forever)
        worker.start()
        client = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        try:
            for key in ("dummy-one", "different-dummy"):
                client.request("POST", "/v1/chat/completions", server.encode(REQUEST),
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                response = client.getresponse()
                assert response.status == 200
                assert json.loads(response.read())["choices"][0]["message"]["content"] == ANSWER
            shim.stopped.set()
            client.request("POST", "/v1/chat/completions", server.encode(REQUEST), {"Content-Type": "application/json"})
            response = client.getresponse()
            assert response.status == 429 and response.getheader("Retry-After") == "60"
            response.read()
        finally:
            client.close()
            httpd.shutdown()
            worker.join(timeout=5)
    assert "dummy-one" not in shim.log.read_text()


def post(shim, headers, handle=None):
    if handle:
        shim.handle = handle
    with server.Server(shim, port=0) as httpd:
        worker = threading.Thread(target=httpd.serve_forever)
        worker.start()
        client = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        try:
            client.request("POST", "/v1/chat/completions", server.encode(REQUEST), headers)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()
            httpd.shutdown()
            worker.join(timeout=5)


@pytest.mark.parametrize("extra", [{"Origin": "http://127.0.0.1:1"}, {"Origin": "null"},
                                   {"Host": "rebound.example:8765"}])
def test_browser_or_foreign_host_requests_are_403_without_dispatch(lane, extra):
    shim, backend = lane
    assert post(shim, {"Content-Type": "application/json", **extra}) == (403, server.error("forbidden"))
    backend.call.assert_not_called()
    assert not shim.log.exists()


@pytest.mark.parametrize("headers", [{}, {"Content-Type": "text/plain"},
                                     {"Content-Type": "application/x-www-form-urlencoded"}])
def test_non_json_content_type_is_415_without_dispatch(lane, headers):
    shim, backend = lane
    assert post(shim, headers) == (415, server.error("unsupported_media_type"))
    backend.call.assert_not_called()
    assert not shim.log.exists()


def test_json_content_type_with_charset_is_served(lane):
    assert post(lane[0], {"Content-Type": "application/json; charset=utf-8"})[0] == 200


@pytest.mark.parametrize("exc", [OSError, RuntimeError])
def test_failure_after_the_body_is_read_is_503(lane, exc):
    status, body = post(lane[0], {"Content-Type": "application/json"}, Mock(side_effect=exc("receipt")))
    assert (status, body) == (503, server.error("lane_unavailable"))


def test_relative_root_is_resolved_before_fmp(tmp_path, monkeypatch):
    home = tmp_path / "lease"
    home.mkdir()
    monkeypatch.setenv("LCMX_SHIM_CODEX_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    codex = Mock()
    monkeypatch.setattr(server, "ShimCodex", codex)
    monkeypatch.setattr(server, "Backends", Mock())
    server.build(Path("state"), Path("log.jsonl"))
    quota_root = codex.call_args.args[0]
    assert quota_root.is_absolute() and quota_root.parent == (tmp_path / "state").resolve()


def test_line_separators_inside_answer_text_survive(tmp_path):
    text = "first\u2028second\u2029third\x85fourth"
    home = tmp_path / "lease"
    home.mkdir()

    def runner(cmd, **kwargs):
        rollout = home / "sessions" / "rollout-test-thread.jsonl"
        rollout.parent.mkdir(exist_ok=True)
        rollout.write_text(json.dumps(
            {"type": "response_item", "payload": {"role": "assistant", "model": "gpt-6.1-sol"}}) + "\n")
        events = [{"type": "thread.started", "thread_id": "thread"},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
                  {"type": "turn.completed"}]
        # Codex writes these code points raw inside JSON strings.
        return subprocess.CompletedProcess(cmd, 0, "\n".join(json.dumps(e, ensure_ascii=False) for e in events), "")

    quota = Quota(tmp_path / "quota.json")
    codex = server.ShimCodex(tmp_path / "run", quota, codex_home=home, runner=runner)
    backends = Backends(tmp_path, quota, MODELS.values(), codex=codex)
    try:
        status, _, body = invoke(server.Shim(backends, tmp_path / "log.jsonl"))
        assert status == 200 and body["choices"][0]["message"]["content"] == text
    finally:
        backends.close()


def test_build_requires_home_and_injects_deadline_and_fresh_quota(tmp_path, monkeypatch):
    monkeypatch.delenv("LCMX_SHIM_CODEX_HOME", raising=False)
    root = tmp_path / "state"
    with pytest.raises(ValueError, match="LCMX_SHIM_CODEX_HOME"):
        server.build(root, tmp_path / "log")
    assert not root.exists()
    home = tmp_path / "lease"
    home.mkdir()
    monkeypatch.setenv("LCMX_SHIM_CODEX_HOME", str(home))
    codex, backends = Mock(), Mock()
    monkeypatch.setattr(server, "ShimCodex", codex)
    monkeypatch.setattr(server, "Backends", backends)
    for _ in range(2):
        server.build(root, tmp_path / "log")
    assert codex.call_args.kwargs == {"codex_home": str(home), "include_total_tokens": True, "timeout": (540, 1800)}
    assert backends.call_args.kwargs == {"codex": codex.return_value}
    assert list(backends.call_args.args[2]) == ["codex:gpt-6.1-sol@low"]
    assert len(list(root.iterdir())) == 2
    assert backends.call_args_list[0].args[1].path != backends.call_args_list[1].args[1].path


@pytest.mark.parametrize("failure,status", [
    ("timeout", 503), ("codex_failed", 503), ("model_mismatch", 422),
    ("tool_violation", 422), ("unexpected_failure", 422),
])
def test_recorded_failures_have_retry_appropriate_status(lane, failure, status):
    shim, backend = lane

    def call(spec, messages, records):
        records.append({**RECORD, "error": failure, "status": "tool_violation" if failure == "tool_violation" else "error"})
        raise ToolViolation() if failure == "tool_violation" else RuntimeError("sensitive exception")

    backend.call.side_effect = call
    assert invoke(shim)[0] == status
    assert not shim.stopped.is_set()


@pytest.mark.parametrize("exc", [json.JSONDecodeError("bad", "", 0), UnicodeError("bad")])
@pytest.mark.parametrize("recorded", [True, False])
def test_backend_decode_failure_is_503(lane, exc, recorded):
    shim, backend = lane

    def call(spec, messages, records):
        if recorded:
            records.append(dict(RECORD))
        raise exc

    backend.call.side_effect = call
    assert invoke(shim)[0] == 503


@pytest.mark.parametrize("transcripts", [None, [], ["events"], ["events", None]])
def test_missing_rollout_is_503_with_identity_receipt(lane, transcripts):
    shim, backend = lane

    def call(spec, messages, records):
        records.append({**RECORD, "transcripts": transcripts, "identity_source": "configured"})
        return {"content": ANSWER}

    backend.call.side_effect = call
    status, _, response = invoke(shim)
    assert status == 503 and response["error"]["message"] == "missing_rollout"
    receipt = json.loads(shim.log.read_text())
    assert receipt["identity_source"] == "configured" and receipt["weekly_used_percent"] == 24.0


def test_unrecorded_answer_is_503(lane):
    shim, backend = lane
    backend.call.side_effect = lambda *args: {"content": ANSWER}
    assert invoke(shim)[0] == 503


@pytest.mark.parametrize("recovered,used", [(True, 24.0), (False, 24.0), (True, 95.0)])
def test_reconnect_limit_text_uses_completed_turn_outcome(tmp_path, monkeypatch, recovered, used):
    monkeypatch.setattr(Path, "symlink_to", Mock(side_effect=AssertionError("no symlinks")))
    home = tmp_path / "lease"
    home.mkdir()

    def runner(cmd, **kwargs):
        rollout = home / "sessions" / "rollout-test-thread.jsonl"
        rollout.parent.mkdir(exist_ok=True)
        rollout.write_text("\n".join(map(json.dumps, [
            {"type": "response_item", "payload": {"role": "assistant", "model": "gpt-6.1-sol"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"x": 1},
                "rate_limits": {"primary": {"used_percent": used, "window_minutes": 10080}}}},
        ])) + "\n")
        events = [{"type": "thread.started", "thread_id": "thread"},
                  {"type": "error", "message": "reconnecting: usage limit"}]
        events += ([{"type": "item.completed", "item": {"type": "agent_message", "text": ANSWER}},
                    {"type": "turn.completed"}] if recovered else [
                        {"type": "turn.failed", "error": {"message": "usage limit"}}])
        return subprocess.CompletedProcess(cmd, 1, "\n".join(map(json.dumps, events)), "")

    quota = Quota(tmp_path / "quota.json")
    codex = server.ShimCodex(tmp_path / "run", quota, codex_home=home, runner=runner)
    backends = Backends(tmp_path, quota, MODELS.values(), codex=codex)
    shim = server.Shim(backends, tmp_path / "log.jsonl")
    try:
        assert invoke(shim)[0] == (200 if recovered else 429)
        assert shim.stopped.is_set() is not recovered
        assert quota.state["codex"]["stopped"] == (not recovered or used >= 95)
        # Classification leaves the original FMP evidence unchanged.
        trace = next((tmp_path / "run").rglob("events.jsonl"))
        assert "reconnecting: usage limit" in trace.read_text()
        assert invoke(shim)[0] == (200 if recovered and used < 95 else 429)
    finally:
        backends.close()


def test_handler_sets_socket_timeout():
    handler = server.Handler.__new__(server.Handler)
    handler.request = Mock()
    handler.setup()
    handler.request.settimeout.assert_called_once_with(60)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_shutdown_drains_calls_before_cleanup_and_overrides_ignored_sigint(lane, monkeypatch, signum):
    shim, backend = lane
    backend.close = Mock()
    ready, entered, release, drained = (threading.Event() for _ in range(4))
    actual_server, instance = server.Server, []
    main_thread = threading.current_thread()

    class ObservedServer(actual_server):
        def __init__(self, shim, port):
            super().__init__(shim, port)
            instance.append(self)
            ready.set()

        def shutdown(self):
            assert threading.current_thread() is not main_thread
            super().shutdown()
            drained.set()

    def call(spec, messages, records):
        entered.set()
        assert release.wait(timeout=5)
        records.append(dict(RECORD))
        return {"content": ANSWER}

    def fetch():
        client = http.client.HTTPConnection(*instance[0].server_address, timeout=5)
        try:
            client.request("POST", "/v1/chat/completions", server.encode(REQUEST), {"Content-Type": "application/json"})
            response = client.getresponse()
            response.read()
            return response.status
        finally:
            client.close()

    def drive(pool):
        try:
            assert ready.wait(timeout=5)
            request = pool.submit(fetch)
            assert entered.wait(timeout=5)
            handler = signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)
            assert shim.draining.is_set()
            assert drained.wait(timeout=5)
            backend.close.assert_not_called()
        finally:
            release.set()
            if instance and not drained.is_set():
                instance[0].shutdown()
        return request.result(timeout=5)

    backend.call.side_effect = call
    monkeypatch.setattr(server, "Server", ObservedServer)
    old_int, old_term = signal.signal(signal.SIGINT, signal.SIG_IGN), signal.getsignal(signal.SIGTERM)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            driver = pool.submit(drive, pool)
            server.serve(shim, 0)
            assert driver.result(timeout=5) == 200
        backend.close.assert_called_once()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == old_term
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
