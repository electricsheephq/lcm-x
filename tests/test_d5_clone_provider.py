"""Exercise the provider through HTTP, including SSE and sanitized receipts."""

import hashlib
import json
import http.client
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from bench.instruments.d5_clone.provider import Provider


def request(provider, path, data=None, raw=False, method=None):
    url = provider.url.removesuffix("/v1") + path
    body = json.dumps(data).encode() if data is not None else None
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method=method)
    with build_opener(ProxyHandler({})).open(req, timeout=3) as response:
        content = response.read()
    return content if raw else json.loads(content)


def complete(provider, model="d5-agent", **kwargs):
    return request(provider, "/v1/chat/completions", {"model": model, "messages": [], **kwargs})


@pytest.fixture
def provider():
    instance = Provider({"steps": {"turn": [
        {"text": "scripted reply"},
        {"tool_calls": [{"id": "call-1", "name": "lookup", "arguments": {"key": "value"}}]},
    ]}, "cron": [{"text": "scheduled reply"}]}).start()
    try:
        yield instance
    finally:
        instance.stop()


def test_script_order_misses_and_stats(provider):
    assert complete(provider)["choices"][0]["message"]["content"] == "d5 replay miss"
    request(provider, "/_d5/step", {"step": "turn"})
    reply = complete(provider)
    assert reply["choices"][0]["message"] == {"role": "assistant", "content": "scripted reply"}
    assert reply["usage"]["total_tokens"] > 0
    # Auxiliary calls must never consume an agent script entry.
    complete(provider, "d5-summary")
    reply = complete(provider)
    assert reply["choices"][0]["finish_reason"] == "tool_calls"
    assert reply["choices"][0]["message"]["tool_calls"] == [{
        "id": "call-1", "type": "function", "function": {
            "name": "lookup", "arguments": '{"key":"value"}'}}]
    assert complete(provider)["choices"][0]["message"]["content"] == "d5 replay miss"
    request(provider, "/_d5/end", method="POST")
    complete(provider)
    stats = request(provider, "/_d5/stats")
    assert stats["replay_misses"] == {"none": 2, "turn": 1}
    assert stats["agent_calls_outside_step"] == 2
    assert stats["calls_per_step"] == {"none": 2, "turn": 3}
    assert stats["calls_by_model"] == {"d5-agent": 5, "d5-summary": 1}
    assert [r[2] for r in stats["request_log"] if r[0] == "d5-agent"] == [0, 0, 1, 2, 1]


def test_digest_is_exact_stable_and_stats_have_no_content(provider):
    marker = "unique-request-marker-é"
    messages = [{"role": "user", "content": {"b": 2, "a": marker}},
                {"role": "assistant", "content": None}]

    def digest(value):
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    expected = ("D5 digest v1\nmessages=2\nuser " + digest(messages[0]["content"])[:12]
                + "\nassistant " + digest(None)[:12] + "\nrequest=" + digest(messages))
    other = Provider({}).start()
    try:
        replies = [complete(p, "unlisted", messages=messages) for p in (provider, provider, other)]
        assert replies[0] == replies[1] == replies[2]
        assert replies[0]["choices"][0]["message"]["content"] == expected
        # Token counts: UTF-8 bytes of the canonical JSON / 4, rounded up, for the messages and the reply message.
        usage = replies[0]["usage"]
        sent = len(json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        assert usage["prompt_tokens"] == -(-sent // 4)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        # Compare actual serialized HTTP bytes as well as decoded values.
        body = {"model": "unlisted", "messages": messages}
        assert request(provider, "/v1/chat/completions", body, raw=True) == request(
            other, "/v1/chat/completions", body, raw=True)
    finally:
        other.stop()
    stats = request(provider, "/_d5/stats")
    assert marker not in json.dumps(stats)
    assert "unique-request-marker" not in json.dumps(stats)
    assert stats["unknown_models"] == {"unlisted": 3}
    assert len(stats["request_log"][0]) == 5
    assert stats["request_log"][0][3] == digest({"model": "unlisted", "messages": messages})
    assert stats["request_log"][0][4] == digest(replies[0])


@pytest.mark.parametrize("model", ["d5-summary", "d5-agent"])
def test_sse_assembles_to_nonstream_content(provider, model):
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}]}
    expected = request(provider, "/v1/chat/completions", body)
    stream = request(provider, "/v1/chat/completions", {**body, "stream": True}, raw=True)
    events = [line.removeprefix("data: ") for line in stream.decode().splitlines() if line]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == expected["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" in chunk for chunk in chunks)


def test_streamed_tool_calls(provider):
    request(provider, "/_d5/step", {"step": "turn"})
    complete(provider)
    stream = request(provider, "/v1/chat/completions",
                     {"model": "d5-agent", "stream": True}, raw=True)
    chunk = json.loads(stream.decode().splitlines()[0].removeprefix("data: "))
    call = chunk["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "lookup"
    assert '"finish_reason":"tool_calls"' in stream.decode()


def test_cron_is_independent(provider):
    request(provider, "/_d5/step", {"step": "turn"})
    complete(provider)
    assert complete(provider, "d5-cron")["choices"][0]["message"]["content"] == "scheduled reply"
    assert complete(provider)["choices"][0]["finish_reason"] == "tool_calls"
    request(provider, "/_d5/end", {})
    assert complete(provider, "d5-cron")["choices"][0]["message"]["content"] == "d5 replay miss"
    assert request(provider, "/_d5/stats")["replay_misses"] == {"cron": 1}


def test_models_and_embeddings(provider):
    assert {m["id"] for m in request(provider, "/v1/models")["data"]} == {
        "d5-agent", "d5-cron", "d5-summary"}
    with pytest.raises(HTTPError) as exc:
        request(provider, "/v1/embeddings", {"input": "not served"})
    assert exc.value.code == 404


@pytest.mark.parametrize("host", ["0.0.0.0", "1.1.1.1", "::", "example.invalid", "127.0.0.2"])
def test_refuse_nonloopback(host):
    with pytest.raises(ValueError):
        Provider({}, host=host)


def test_configured_models_and_string_steps_only():
    instance = Provider({"agent_model": "agent", "cron_model": "cron", "models": ["digest"],
                         "steps": {"0": [{"text": "zero"}]}}).start()
    try:
        with pytest.raises(HTTPError) as refused:
            request(instance, "/_d5/step", {"step": 0})  # 0 and "0" must not alias one step.
        assert refused.value.code == 400
        with pytest.raises(HTTPError) as refused:
            request(instance, "/_d5/step", {"step": None})  # Ending a step is /_d5/end, never a null step.
        assert refused.value.code == 400
        request(instance, "/_d5/step", {"step": "0"})
        assert complete(instance, "agent")["choices"][0]["message"]["content"] == "zero"
        complete(instance, "digest")
        assert request(instance, "/_d5/stats")["unknown_models"] == {}
    finally:
        instance.stop()


def test_oversized_and_deeply_nested_bodies_are_refused(provider):
    host, port = provider.server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.putrequest("POST", "/v1/chat/completions")
    connection.putheader("Content-Length", "99999999999")
    connection.endheaders()
    assert connection.getresponse().status == 400
    connection.close()
    nested = "[" * 100000 + "]" * 100000
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("POST", "/v1/chat/completions", body=nested, headers={"Content-Type": "application/json"})
    assert connection.getresponse().status == 400
    connection.close()
    assert complete(provider, "d5-summary")["choices"][0]["message"]["content"].startswith("D5 digest v1")
