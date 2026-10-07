"""#959: emitted survival marks survive pasted marks and storage-only call shortening."""

import base64
import json
from copy import deepcopy

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), survival_reserve=0.9,
        large_output_externalization_path=str(tmp_path / "externalized")),
        hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S", platform="telegram", conversation_id="conv", context_length=4000)
    return engine


def _call(call_id, arguments):
    return {"id": call_id, "type": "function", "function": {"name": "tool", "arguments": arguments}}


def _pasted_marks():
    mark = survival_fit._PROJECTED.format(role="assistant", tokens=585, store_id=999999,
                                          head=survival_fit._HEAD, tail=survival_fit._TAIL)
    return (mark + "\n") * 4


def test_bounded_call_mark_after_four_preserved_argument_marks_survives_cold_ingest(tmp_path, monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    engine = _engine(tmp_path)
    pasted = json.dumps({"lcm_survival_fit": _pasted_marks()})
    assert len(pasted) <= survival_fit._HEAD
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        _call("quoted", pasted), _call("bounded", "word " * 3000)]}
    user = {"role": "user", "content": "Run both tools.", "timestamp": 9.0}
    results = [{"role": "tool", "tool_call_id": call_id, "content": "done"}
               for call_id in ("quoted", "bounded")]
    view = [user, assistant, *results]
    original = deepcopy(view)
    try:
        engine.ingest(view)
        before = engine._store.get_session_messages("S")
        source_id = before[1]["store_id"]
        projected = engine._survival_projection([assistant], {id(assistant): source_id}, 0)[0]
        assert projected["tool_calls"][0] == assistant["tool_calls"][0]
        assert len(list(survival_fit._PROJECTED_RE.finditer(
            projected["tool_calls"][0]["function"]["arguments"]))) == 4
        notice = json.loads(projected["tool_calls"][1]["function"]["arguments"])["lcm_survival_fit"]
        assert survival_fit._PROJECTED_RE.match(notice).group(3) == str(source_id)
        assert view == original
        engine.shutdown()
        engine = _engine(tmp_path)
        source = engine._survival_projection_source(projected, "assistant", projected["content"])
        assert source is not None and source["store_id"] == source_id
        engine.ingest(json.loads(json.dumps([user, projected, *results])))
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("content", ["", "Calling the tool.", _pasted_marks(),
                                    [{"type": "text", "text": "Calling the tool."}]],
                         ids=["empty", "short", "pasted-marks", "structured"])
def test_ingest_protected_call_gets_mark_and_cold_ingest_repeats_no_group(tmp_path, monkeypatch, content):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    engine = _engine(tmp_path)
    payload = "data:image/png;base64," + base64.b64encode(b"synthetic image bytes " * 1000).decode("ascii")
    arguments = json.dumps({"image": payload})
    assistant = {"role": "assistant", "content": content, "tool_calls": [_call("image", arguments)]}
    user = {"role": "user", "content": "Process the image.", "timestamp": 9.0}
    result = {"role": "tool", "tool_call_id": "image", "content": "done"}
    view = [user, assistant, result]
    original = deepcopy(view)
    try:
        engine.ingest(view)  # The real protection path writes the stored placeholder and payload file.
        before = engine._store.get_session_messages("S")
        row = before[1]
        stored_calls = row["tool_calls"]
        stored_arguments = stored_calls[0]["function"]["arguments"]
        assert "[Externalized LCM ingest payload:" in stored_arguments and payload not in stored_arguments
        assert len(stored_arguments) <= survival_fit._HEAD
        assert list((tmp_path / "externalized").glob("*.json"))
        fitted = engine._survival_fit(view, view, engine._survival_measure(view), "issue_959")
        projected = next(message for message in fitted if message["role"] == "assistant")
        text = normalize_content_value(projected["content"]) or ""
        marker = survival_fit._PROJECTED.format(role="assistant", tokens=survival_fit.count_message_tokens(assistant),
                                                store_id=row["store_id"], head=1200, tail=600)
        stored_text = normalize_content_value(row["content"]) or ""
        assert text == (f"{stored_text}\n...\n{marker}" if stored_text else marker)
        assert projected["tool_calls"] == stored_calls != assistant["tool_calls"]
        assert view == original
        engine.shutdown()
        engine = _engine(tmp_path)
        source = engine._survival_projection_source(projected, "assistant", text)
        assert source is not None and source["store_id"] == row["store_id"]
        engine.ingest(json.loads(json.dumps(fitted)))
        after = engine._store.get_session_messages("S")
        assert after == before
        assert sum(message["role"] == "assistant" for message in after) == 1
        assert [message["tool_call_id"] for message in after if message["role"] == "tool"] == ["image"]
        changed = {**projected, "content": text + " changed"}
        assert engine._survival_projection_source(changed, "assistant", changed["content"]) is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize("content", ["", "short", "word " * 240, "word " * 468, "word " * 1200])
def test_projection_with_unchanged_calls_keeps_main_bytes(tmp_path, content):
    engine = _engine(tmp_path)
    assistant = {"role": "assistant", "content": content, "tool_calls": [_call("small", "{}")],
                 "timestamp": 9.0}
    original = deepcopy(assistant)
    try:
        engine.ingest([assistant])
        row = engine._store.get_session_messages("S")[0]
        tokens = survival_fit.count_message_tokens(assistant)
        marker = survival_fit._PROJECTED.format(role="assistant", tokens=tokens, store_id=row["store_id"],
                                                head=1200, tail=600)
        expected_text = content
        if len(content) > 2400:
            expected_text = f"{content[:1200]}\n...\n{marker}\n...\n{content[-600:]}"
        elif len(content) > 1200:
            expected_text = f"{content[:1200]}\n...\n{marker}"
        fields = engine._survival_projected_fields(row, tokens, 1200, 600)
        assert json.dumps(fields).encode() == json.dumps({"content": expected_text,
                                                        "tool_calls": assistant["tool_calls"]}).encode()
        projected = engine._survival_projection([assistant], {id(assistant): row["store_id"]}, 0)[0]
        expected = {**assistant, "content": expected_text} if len(content) > 1200 else assistant
        assert json.dumps(projected).encode() == json.dumps(expected).encode()
        assert assistant == original
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arguments", [
    '{"lcm_survival_fit": "x", "y": ' + "[" * 100000 + "]" * 100000 + "}",
    "[" * 100000 + '"lcm_survival_fit"' + "]" * 100000,
], ids=["prefixed", "unprefixed"])
def test_deeply_nested_live_arguments_do_not_abort_recognition(tmp_path, arguments):
    engine = _engine(tmp_path)
    try:
        message = {"role": "assistant", "content": "", "tool_calls": [_call("deep", arguments)]}
        assert engine._survival_projection_source(message, "assistant", "") is None
    finally:
        engine.shutdown()
