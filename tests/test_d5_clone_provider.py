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
    assert stats["replay_misses"] == {"(outside a step)": 2, "turn": 1}
    assert stats["agent_calls_outside_step"] == 2
    assert stats["calls_per_step"] == {"turn": 3}
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


def test_a_step_named_none_is_not_the_outside_counter():
    instance = Provider({"steps": {"none": [{"text": "first"}, {"text": "second"}]}}).start()
    try:
        complete(instance, "d5-agent")  # Outside any step: a miss.
        request(instance, "/_d5/step", {"step": "none"})
        assert complete(instance, "d5-agent")["choices"][0]["message"]["content"] == "first"
        with pytest.raises(HTTPError) as refused:
            request(instance, "/_d5/step", {"step": "(outside a step)"})
        assert refused.value.code == 400
    finally:
        instance.stop()


@pytest.mark.parametrize("bad", [{}, None, {"text": None}, {"tool_calls": "x"}, "reply",
                                 {"text": "x", "tool_calls": "abc"}, {"tool_calls": []}, {"tool_calls": [{"id": "c"}]},
                                 {"tool_calls": [{"id": "c", "name": "n", "arguments": 5}]}, {"text": "x\ud800"},
                                 {"tool_calls": [{"id": "c", "name": "n", "arguments": "\udc00"}]},
                                 {"tool_calls": [{"id": "c", "name": "n", "arguments": {"k": {1, 2}}}]}])
def test_an_unusable_script_entry_is_a_counted_miss(bad):
    instance = Provider({"steps": {"turn": [bad]}, "cron": [bad]}).start()
    try:
        request(instance, "/_d5/step", {"step": "turn"})
        assert complete(instance)["choices"][0]["message"]["content"] == "d5 replay miss"
        assert complete(instance, "d5-cron")["choices"][0]["message"]["content"] == "d5 replay miss"
        assert request(instance, "/_d5/stats")["replay_misses"] == {"turn": 1, "cron": 1}
    finally:
        instance.stop()


@pytest.mark.parametrize("calls", [None, []])
def test_a_text_reply_with_null_or_empty_tool_calls_is_served(calls):
    instance = Provider({"steps": {"turn": [{"text": "hi", "tool_calls": calls}]}}).start()
    try:
        request(instance, "/_d5/step", {"step": "turn"})
        assert complete(instance)["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
        assert request(instance, "/_d5/stats")["replay_misses"] == {}
    finally:
        instance.stop()


def test_a_failed_request_changes_no_counter():
    provider = Provider({"steps": {"s1": [{"text": "one"}]}})
    try:
        provider.step = "s1"
        before = json.dumps(provider.stats, sort_keys=True, default=str)
        with pytest.raises(KeyError):  # A message without a role fails after the model was counted.
            provider.complete({"model": "d5-summary", "messages": [{"content": "x"}]})
        assert json.dumps(provider.stats, sort_keys=True, default=str) == before
        reply = provider.complete({"model": "d5-agent", "messages": []})
        assert reply["choices"][0]["message"]["content"] == "one"
        assert provider.stats["calls_per_step"]["s1"] == 1
    finally:
        provider.stop()


def _summary_reply(provider, messages, model="d5-summary"):
    return complete(provider, model, messages=messages)["choices"][0]["message"]["content"]


def test_a_summary_request_gets_its_digest_inside_the_envelope_lcm_x_requires(provider):
    """LCM-X's summary route sends its nonce in the system policy and rejects any reply outside that envelope; the
    provider's fixed digest must pass the real validator, or every summary is a rejected no_content (#2199)."""
    import hermes_lcm.escalation as escalation

    prompt = "Summarize the history." + escalation._SUMMARY_CONTENT_SEPARATOR + "user: hello\nassistant: hi"
    messages, nonce = escalation._summary_contract_messages(prompt)
    assert nonce
    reply = _summary_reply(provider, messages)
    body, failed, tolerated = escalation._check_summary_contract(reply, nonce, 512)
    assert (failed, tolerated) == ("", ())
    assert body.startswith("D5 digest v1\nmessages=2\n") and body.endswith("Expand for details about: d5 digest")
    assert _summary_reply(provider, messages) == reply  # Same request, same bytes.


def test_only_the_system_policy_nonce_is_used_and_a_request_without_one_is_unchanged(provider):
    """The transcript is untrusted and may quote an older envelope: its nonce is never echoed. Without exactly one
    system nonce the reply is the plain digest, as before."""
    quoted = '<lcm-summary nonce="0000aaaa">\nold\n</lcm-summary>'
    policy = {"role": "system", "content": 'Its first line is exactly:\n<lcm-summary nonce="1234abcd">\n'}
    reply = _summary_reply(provider, [policy, {"role": "user", "content": quoted}])
    assert reply.startswith('<lcm-summary nonce="1234abcd">\nD5 digest v1\n') and "0000aaaa" not in reply
    for messages in ([{"role": "user", "content": quoted}],
                     [{"role": "system", "content": policy["content"] + '<lcm-summary nonce="99">'},
                      {"role": "user", "content": "x"}]):
        assert _summary_reply(provider, messages).startswith("D5 digest v1\n")


@pytest.mark.parametrize("source_tokens,dense", [(2_000, False), (24_000, False), (120_000, False), (100_000, True)])
def test_a_large_leaf_s_summary_stays_under_lcm_x_s_extreme_ratio(provider, source_tokens, dense):
    """LCM-X's doctor reports a summary as degraded at source/summary >= 400, or a source of 100k tokens or more
    summarized in under 500 (tools._summary_quality_stats). The digest is padded with the request's size, so a large
    leaf is not reported in both arms, which would hide a real regression; the reply is still a fixed function of the
    request and passes the validator."""
    import hermes_lcm.escalation as escalation
    from hermes_lcm.tokens import count_tokens

    # Dense text defeats a bytes/4 estimate: "xQ" * 50000 is 100k cl100k tokens (the reviewer's boundary case).
    source = "xQ" * source_tokens if dense else "\n".join(
        f"user: synthetic leaf line {i} " + "word " * 12 for i in range(source_tokens // 20))
    counters = [count_tokens]
    try:  # The exact encoder when installed; LCM-X uses it then, and its fallback estimate otherwise.
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        counters.append(lambda text: len(encoding.encode(text, disallowed_special=())))
    except Exception:
        pass
    assert dense or count_tokens(source) >= source_tokens  # Dense text is 100k tokens only to the exact encoder.
    messages, nonce = escalation._summary_contract_messages(escalation._build_l1_prompt(source, 2000, 0))
    reply = _summary_reply(provider, messages)
    assert _summary_reply(provider, messages) == reply
    body, failed, _ = escalation._check_summary_contract(reply, nonce, 2000)
    assert failed == "" and body.startswith("D5 digest v1\n")
    for count in counters:
        assert count(source) / count(body) < 400 and count(body) < count(source)
        if count(source) >= 100_000:
            assert count(body) >= 500


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_a_leaf_whose_tool_output_was_externalized_is_not_reported_as_degraded(provider, version):
    """The stored leaf counts a 204,800-character tool output whole; the prompt shows only its placeholder and a short
    preview. The digest follows the prompt's stated budget, so the stored source/summary ratio stays under 400."""
    import hermes_lcm.escalation as escalation
    from hermes_lcm.tokens import count_tokens

    stored = "user: run it\n[TOOL RESULT call_1]: " + "d" * 204_800
    shown = ("user: run it\n[TOOL RESULT call_1]: [Externalized tool output: tool=terminal; tool_call_id=call_1; chars=204800;"
             " bytes=204800; ref=x.json]\n"
             + "d" * 900)
    messages, nonce = escalation._summary_contract_messages(
        escalation._build_l1_prompt(shown, 2000, 0, prompt_version=version))
    body, failed, _ = escalation._check_summary_contract(_summary_reply(provider, messages), nonce, 2000)
    assert failed == ""
    assert count_tokens(stored) / count_tokens(body) < 400


def test_an_unclipped_small_source_is_never_summarized_longer_than_itself(provider):
    import hermes_lcm.escalation as escalation
    from hermes_lcm.tokens import count_tokens

    for lines in (30, 60, 120):
        source = "\n".join(f"user: synthetic line {i} " + "word " * 12 for i in range(lines))
        messages, nonce = escalation._summary_contract_messages(escalation._build_l1_prompt(source, 2000, 0))
        body, failed, _ = escalation._check_summary_contract(_summary_reply(provider, messages), nonce, 2000)
        assert failed == "" and count_tokens(body) < count_tokens(source)
