"""#916: survival projections preserve text and the newest user's opening."""

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


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
