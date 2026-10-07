"""#916: survival projections preserve text and the newest user's opening."""

import json
from copy import deepcopy

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value


@pytest.fixture
def rough_counts(monkeypatch):
    """Pin the reported 585-token, 2,340-character scenario to chars / 4."""
    def count(message):
        return len(str(message.get("content") or "")) // 4

    def measure(messages):
        return sum(count(message) for message in messages)
    monkeypatch.setattr(survival_fit, "count_message_tokens", count)
    monkeypatch.setattr(survival_fit, "count_messages_tokens", measure)
    monkeypatch.setattr(survival_fit, "_host_estimate", measure)
    return measure


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_reserve=0.9))
    engine.on_session_start("S", platform="telegram", conversation_id="conv", context_length=4000)
    return engine


def _view():
    return [{"role": "system", "content": "system prompt"},
            {"role": "user", "content": "[T08] older request", "timestamp": 8.0},
            {"role": "assistant", "content": "older reply"},
            {"role": "user", "content": ("[T09] " + "word " * 467)[:2340], "timestamp": 9.0}]


def test_newest_user_keeps_opening_under_survival_budget(tmp_path, rough_counts):
    engine = _engine(tmp_path)
    view = _view()
    try:
        assert len(view[-1]["content"]) == 2340 and rough_counts(view[-1:]) == 585
        engine.ingest(view)
        fitted = engine._survival_fit(view, view, rough_counts(view), "issue_916")
        newest = next(message for message in reversed(fitted) if message["role"] == "user")
        assert newest["content"].startswith(view[-1]["content"][:1200])
        assert "[T09]" in newest["content"] and survival_fit._PROJECTED_PREFIX in newest["content"]
        assert fitted != view and rough_counts(fitted) <= engine._survival_fit_budget(view, rough_counts(view))
        assert engine._store.get_session_messages("S")[-1]["content"] == view[-1]["content"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("length", [0, 1, survival_fit._HEAD - 1, survival_fit._HEAD])
def test_short_row_stays_whole_and_projection_moves_on(tmp_path, rough_counts, length):
    engine = _engine(tmp_path)
    short = {"role": "tool", "content": "s" * length, "tool_call_id": "call_short"}
    long = _view()[-1]
    try:
        ids = [engine._store.append("S", message, conversation_id="conv") for message in [short, long]]
        row = engine._store.get(ids[0])
        assert engine._survival_projected_fields(row, 585, survival_fit._HEAD, survival_fit._TAIL)["content"] == short["content"]
        projected = engine._survival_projection([short, long], {id(short): ids[0], id(long): ids[1]}, 0)
        assert projected[0] is short
        assert projected[1]["content"].startswith(long["content"][:1200])
        assert survival_fit._PROJECTED_PREFIX in projected[1]["content"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("cold", [False, True], ids=["same-process", "cold-resume"])
def test_head_plus_mark_re_ingests_without_storing_again(tmp_path, rough_counts, cold):
    engine = _engine(tmp_path)
    view = _view()
    try:
        engine.ingest(view)
        fitted = engine._survival_fit(view, view, rough_counts(view), "issue_916")
        projected = fitted[-1]
        assert projected["content"].startswith(view[-1]["content"][:1200])
        assert survival_fit._PROJECTED_PREFIX in projected["content"]
        before = engine._store.get_session_messages("S")
        if cold:
            engine.on_session_end("S", fitted)
            engine.shutdown()
            engine = _engine(tmp_path)
        source = engine._survival_projection_source(projected, "user", projected["content"])
        assert source is not None and source["store_id"] == before[-1]["store_id"]
        changed = {**projected, "content": projected["content"] + " changed"}
        assert engine._survival_projection_source(changed, "user", changed["content"]) is None
        engine.ingest([*fitted, {"role": "user", "content": "[T10] new request", "timestamp": 10.0}])
        after = engine._store.get_session_messages("S")
        assert after[:-1] == before and after[-1]["content"] == "[T10] new request"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("head,tail,length", [(1200, 600, 1201), (1200, 600, 2400), (16, 7, 32)])
def test_middle_row_uses_mark_parameters(head, tail, length):
    row = {"store_id": 17, "role": "user", "content": "x" * length}
    mark = survival_fit._PROJECTED.format(role="user", tokens=585, store_id=17, head=head, tail=tail)
    fields = survival_fit.SurvivalFitMixin._survival_projected_fields(row, 585, head, tail)
    assert fields["content"] == f"{row['content'][:head]}\n...\n{mark}"


@pytest.mark.parametrize("head,tail,length", [(1200, 600, 2401), (1200, 600, 10000), (16, 7, 33)])
def test_long_row_is_byte_identical_to_8396706d(head, tail, length):
    row = {"store_id": 17, "role": "user", "content": "opening " + "x" * length + " ending"}
    mark = survival_fit._PROJECTED.format(role="user", tokens=585, store_id=17, head=head, tail=tail)
    text = row["content"]
    baseline = f"{text[:head]}\n...\n{mark}\n...\n{text[-tail:]}" if len(text) > 2 * head else mark
    fields = survival_fit.SurvivalFitMixin._survival_projected_fields(row, 585, head, tail)
    assert fields["content"].encode() == baseline.encode()


def _legacy_projection(message, store_id, tokens=585):
    mark = survival_fit._PROJECTED.format(role=message["role"], tokens=tokens, store_id=store_id,
                                          head=survival_fit._HEAD, tail=survival_fit._TAIL)
    return {**message, "content": mark}


def test_legacy_mark_only_cold_resume_does_not_store_again_or_duplicate_followers(tmp_path, rough_counts):
    engine = _engine(tmp_path)
    user = _view()[-1]
    replies = [{"role": "assistant", "content": "first reply"},
               {"role": "assistant", "content": "second reply"}]
    try:
        assert rough_counts([user]) == 585
        engine.ingest([user, *replies])
        before = engine._store.get_session_messages("S")
        projected = _legacy_projection(user, before[0]["store_id"])
        live = [projected, *replies]
        engine.on_session_end("S", live)
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest(live)
        assert engine._store.get_session_messages("S") == before
        source = engine._survival_projection_source(projected, "user", projected["content"])
        assert source is not None and source["store_id"] == before[0]["store_id"]
        followers = engine._survival_projection_followers(live, 0, source, {0: user["timestamp"]})
        assert [(index, row["store_id"]) for index, row in followers] == [
            (1, before[1]["store_id"]), (2, before[2]["store_id"])]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("legacy", [False, True], ids=["current", "legacy"])
def test_changed_projection_is_a_new_message_on_cold_resume(tmp_path, rough_counts, legacy):
    engine = _engine(tmp_path)
    user = _view()[-1]
    try:
        engine.ingest([user])
        before = engine._store.get_session_messages("S")
        store_id = before[0]["store_id"]
        projected = (_legacy_projection(user, store_id) if legacy else
                     engine._survival_projection([user], {id(user): store_id}, 0)[0])
        changed = {**projected, "content": projected["content"] + " changed"}
        engine.shutdown()
        engine = _engine(tmp_path)
        assert engine._survival_projection_source(changed, "user", changed["content"]) is None
        engine.ingest([changed])
        after = engine._store.get_session_messages("S")
        assert after[:-1] == before and after[-1]["content"] == changed["content"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("length", [0, 1, 1200, 2400, 2401])
def test_legacy_form_keeps_exact_tool_call_bounding(length):
    call = {"id": "call", "type": "function", "function": {"name": "tool", "arguments": "x" * 1201}}
    row = {"store_id": 17, "role": "assistant", "content": "x" * length, "tool_calls": [call]}
    current = survival_fit.SurvivalFitMixin._survival_projected_fields(row, 585, 1200, 600)
    legacy = survival_fit.SurvivalFitMixin._survival_projected_fields(row, 585, 1200, 600, legacy=True)
    mark = survival_fit._PROJECTED.format(role="assistant", tokens=585, store_id=17, head=1200, tail=600)
    assert legacy["content"] == (mark if 0 < length <= 2400 else current["content"])
    assert legacy["tool_calls"] == current["tool_calls"]
    assert legacy["tool_calls"][0]["function"]["name"] == "tool"
    assert "lcm_survival_fit" in legacy["tool_calls"][0]["function"]["arguments"]


def test_short_image_content_stays_structured_after_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    engine = _engine(tmp_path)
    content = [{"type": "text", "text": "Describe this image."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    original_bytes = json.dumps(content).encode()
    user = {"role": "user", "content": content, "timestamp": 9.0}
    try:
        assert len(normalize_content_value(content)) <= survival_fit._HEAD
        assert survival_fit.count_message_tokens(user) > 256
        engine.ingest([user])
        before = engine._store.get_session_messages("S")
        fitted = engine._survival_fit([user], [user], engine._survival_measure([user]), "issue_917")
        assert isinstance(fitted[-1]["content"], list)
        assert fitted[-1]["content"] is content
        assert json.dumps(fitted[-1]["content"]).encode() == original_bytes
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


def test_emitted_marker_after_four_pasted_marks_re_ingests_without_new_row(tmp_path, rough_counts, monkeypatch):
    engine = _engine(tmp_path)
    pasted = survival_fit._PROJECTED.format(role="user", tokens=585, store_id=999999, head=1200, tail=600)
    user = {"role": "user", "content": (pasted + "\n") * 4 + "word " * 400, "timestamp": 9.0}
    try:
        assert len((pasted + "\n") * 4) < survival_fit._HEAD
        engine.ingest([user])
        before = engine._store.get_session_messages("S")
        projected = engine._survival_projection([user], {id(user): before[0]["store_id"]}, 0)[0]
        matches = list(survival_fit._PROJECTED_RE.finditer(projected["content"]))
        assert matches[4].start() == survival_fit._HEAD + len("\n...\n")
        engine.shutdown()
        engine = _engine(tmp_path)
        lookups = []
        get = engine._store.get

        def tracked_get(store_id):
            lookups.append(store_id)
            return get(store_id)

        monkeypatch.setattr(engine._store, "get", tracked_get)
        source = engine._survival_projection_source(projected, "user", projected["content"])
        assert source is not None and source["store_id"] == before[0]["store_id"]
        assert lookups == [before[0]["store_id"]]
        engine.ingest([projected])
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


def test_growing_projection_is_skipped_and_next_row_is_projected(tmp_path):
    engine = _engine(tmp_path)
    dense = {"role": "tool", "content": "界" * (survival_fit._HEAD + 1), "tool_call_id": "call_dense"}
    long = {"role": "user", "content": "word " * 3000}
    try:
        ids = [engine._store.append("S", message, conversation_id="conv") for message in [dense, long]]
        tokens = survival_fit.count_message_tokens(dense)
        fields = engine._survival_projected_fields(engine._store.get(ids[0]), tokens,
                                                    survival_fit._HEAD, survival_fit._TAIL)
        candidate = {**dense, **{key: value for key, value in fields.items() if value is not None}}
        assert tokens > 256 and survival_fit.count_message_tokens(candidate) > tokens
        projected = engine._survival_projection([dense, long], {id(dense): ids[0], id(long): ids[1]}, 0)
        assert projected[0] is dense
        assert survival_fit.count_message_tokens(projected[1]) < survival_fit.count_message_tokens(long)
    finally:
        engine.shutdown()


def test_short_structured_content_with_bounded_calls_is_recognised(tmp_path):
    engine = _engine(tmp_path)
    user = {"role": "user", "content": "Run the tool.", "timestamp": 9.0}
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "Calling the tool."}],
                 "tool_calls": [{"id": "call", "type": "function",
                                 "function": {"name": "tool", "arguments": "word " * 3000}}]}
    original = deepcopy(assistant)
    try:
        engine.ingest([user, assistant])
        before = engine._store.get_session_messages("S")
        projected = engine._survival_projection([assistant], {id(assistant): before[-1]["store_id"]}, 0)[0]
        assert projected["content"] is assistant["content"]
        assert projected["tool_calls"] != assistant["tool_calls"]
        assert assistant == original
        source = engine._survival_projection_source(projected, "assistant", normalize_content_value(projected["content"]))
        assert source is not None and source["store_id"] == before[-1]["store_id"]
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest([user, projected])
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


def test_externalized_assistant_projects_to_its_stored_placeholder(tmp_path):
    engine = _engine(tmp_path)
    placeholder = "[Externalized payload: kind=raw_payload; synthetic placeholder]"
    live = {"role": "assistant", "content": "payload " * 2000}
    try:
        store_id = engine._store.append("S", {"role": "assistant", "content": placeholder}, conversation_id="conv")
        assert len(placeholder) <= survival_fit._HEAD
        projected = engine._survival_projection([live], {id(live): store_id}, 0)[0]
        assert projected["content"] == placeholder
        assert survival_fit.count_message_tokens(projected) < survival_fit.count_message_tokens(live)
    finally:
        engine.shutdown()


def test_bounded_call_marker_is_found_after_four_pasted_marks(tmp_path):
    engine = _engine(tmp_path)
    pasted = survival_fit._PROJECTED.format(role="assistant", tokens=585, store_id=999999, head=1200, tail=600)
    user = {"role": "user", "content": "Run the tool.", "timestamp": 9.0}
    assistant = {"role": "assistant", "content": (pasted + "\n") * 4,
                 "tool_calls": [{"id": "call", "type": "function",
                                 "function": {"name": "tool", "arguments": "word " * 3000}}]}
    try:
        engine.ingest([user, assistant])
        before = engine._store.get_session_messages("S")
        projected = engine._survival_projection([assistant], {id(assistant): before[-1]["store_id"]}, 0)[0]
        assert projected["tool_calls"] != assistant["tool_calls"]
        content = normalize_content_value(projected["content"])
        assert len(list(survival_fit._PROJECTED_RE.finditer(content))) == 4
        source = engine._survival_projection_source(projected, "assistant", content)
        assert source is not None and source["store_id"] == before[-1]["store_id"]
    finally:
        engine.shutdown()
