"""Round-9 regressions for protecting restored replay pairs."""

import re

import pytest

from tests.invariants.test_p0_host_user_rows import _assistant, _compress, _history, _stamp, _user
from tests.invariants.test_r4_host_user_replay import _todo
from tests.invariants.test_r7_host_user_replay import (
    MARKER, REDACTION, SENSITIVE, _cached_row, open_engine as open_engine,
)
from tests.invariants.test_r8_host_user_replay import API_NOTE, LITERAL, _assert_absent, _cleanup


@pytest.fixture
def cache_hits(monkeypatch):
    def observe(engine):
        hits = []
        keep = engine._keep_host_held_stubs

        def record(*args):
            hits.append(True)
            return keep(*args)

        monkeypatch.setattr(engine, "_keep_host_held_stubs", record)
        return hits

    return observe


def test_marker_sidecar_is_dropped_on_cached_content_mismatch(open_engine, cache_hits):
    tail = _user(LITERAL)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        hits = cache_hits(engine)
        identity = engine._message_replay_identity(tail)
        tail["content"] += "\n\n" + _todo("Synthetic annotation")
        tail["api_content"] = {"note": LITERAL}
        assert engine._message_replay_identity(tail) == identity
        result = _compress(engine, host, force=True)
        assert hits and engine._last_compress_leaves[1] == 1
        uid = tail["message_uid"]
        for followup in (False, True):
            if followup:
                result = _cleanup(engine, host)
            returned, = [row for row in result if row.get("message_uid") == uid]
            assert returned["content"] == tail["content"]
            cached = _cached_row(engine, uid)
            assert cached["content"] == tail["content"]
            assert ("api_content" in returned, "api_content" in cached) == (False, False)
        assert len(hits) >= 2


def test_cleanup_restores_reasoning_with_protected_sidecar(open_engine, cache_hits):
    answer = _assistant("<think>Synthetic reasoning.</think>Visible answer.")
    host = [_user("Synthetic question."), answer, _user("Another question.")]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        hits = cache_hits(engine)
        answer["api_content"] = API_NOTE
        result = _cleanup(engine, host)
        assert hits
        row, = [row for row in result if row.get("message_uid") == answer["message_uid"]]
        assert (row["content"], re.sub(MARKER, "<redacted>", row.get("api_content", ""))) == (
            answer["content"],
            API_NOTE.replace(SENSITIVE, "<redacted>"),
        )
        _assert_absent(engine, result)


@pytest.mark.parametrize("content_form", ["text", "parts"])
def test_clean_sidecar_survives_redaction_of_content(open_engine, cache_hits, content_form):
    tail = _user("Synthetic question.")
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        hits = cache_hits(engine)
        content = tail["content"] + "\n\n" + _todo(f"api_key={SENSITIVE} Keep this text.")
        tail["content"] = content if content_form == "text" else [{"type": "text", "text": content}]
        tail["api_content"] = "Independent synthetic API message."
        result = _compress(engine, host, force=True)
        assert hits and engine._last_compress_leaves[1] == 1
        row, = [row for row in result if row.get("message_uid") == tail["message_uid"]]
        expected = content.replace(SENSITIVE, "<redacted>")
        actual = row["content"]
        if content_form == "parts":
            expected = [{"type": "text", "text": expected}]
            actual = [part | {"text": re.sub(MARKER, "<redacted>", part["text"])} for part in actual]
        else:
            actual = re.sub(MARKER, "<redacted>", actual)
        assert actual == expected
        _assert_absent(engine, result)
        assert row.get("api_content") == tail["api_content"]
