"""Round-8 regressions for redacting restored replay text in place."""

import re

import pytest

from hermes_lcm.reconcile import _strip_volatile_host_scaffolding
from tests import test_issue_904_hidden_only_no_growth as hidden_backlog
from tests.invariants.test_p0_host_user_rows import _assistant, _compress, _history, _stamp, _user
from tests.invariants.test_r4_host_user_replay import _todo
from tests.invariants.test_r7_host_user_replay import (
    MARKER, REDACTION, SENSITIVE, _cached_row, open_engine as open_engine,
)

API_NOTE = f"API note: api_key={SENSITIVE} ends here."
LITERAL = "The manual quotes [LCM sensitive redaction: example]."
PRE_EXISTING = "pre-existing on aaa1ebba and released tags; redactor changes are out of scope (T1-610)"
backlog_engine = hidden_backlog.engine


def _assert_absent(engine, result):
    assert SENSITIVE not in repr(result)
    assert SENSITIVE not in repr(engine._last_active_replay_messages)


def _assert_sidecar(row, api, patterns):
    expected = api if not patterns else api.replace(SENSITIVE, "<redacted>")
    assert re.sub(MARKER, "<redacted>", row["api_content"]) == expected


@pytest.fixture(params=[True, False], ids=["patterns-on", "patterns-off"])
def patterns(request):
    return request.param


def test_shared_replay_redaction_keeps_api_content(open_engine, patterns):
    source = _user(f"api_key={SENSITIVE} keep this.", api_content=API_NOTE)
    with open_engine(**(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        row, = engine._redact_active_replay_messages([source])
        expected = source["content"].replace(SENSITIVE, "<redacted>") if patterns else source["content"]
        assert re.sub(MARKER, "<redacted>", row["content"]) == expected
        assert row["api_content"] == source["api_content"]


@pytest.mark.parametrize("content", ["Current question", ""], ids=["different-text", "sidecar-only"])
def test_restored_user_sidecar_keeps_text(open_engine, patterns, content):
    tail = _user(content)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        engine.ingest(host)
        tail["api_content"] = API_NOTE
        result = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        cached = _cached_row(engine, tail["message_uid"])
        for row in (result[-1], cached):
            assert row["content"] == content
            _assert_sidecar(row, API_NOTE, patterns)
        if patterns:
            _assert_absent(engine, result)


def _cleanup(engine, host):
    engine._preflight_automatic_request = True
    result = _compress(engine, host)
    assert engine._compress_host_rows_consumed == 0
    assert engine._last_compress_leaves[1] == 0
    return result


def test_promoted_assistant_keeps_redacted_reply(open_engine, patterns):
    promoted = dict(_assistant(""), api_content=API_NOTE)
    host = [_user("Question"), _assistant("First"), _assistant("Second"), _user("Another question"), promoted]
    _stamp(host)
    with open_engine(fresh_tail_count=len(host), **(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        engine.ingest(host)
        result = _cleanup(engine, host)
        row, = [row for row in result if row.get("message_uid") == promoted["message_uid"]]
        assert row["content"] == ""
        _assert_sidecar(row, API_NOTE, patterns)
        if patterns:
            _assert_absent(engine, result)


def test_assistant_cleanup_keeps_both_merged_texts(open_engine, patterns):
    first = _assistant("First answer.")
    second = _assistant("Second answer.")
    host = [_user("Question"), first, second]
    _stamp(host)
    with open_engine(**(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        engine.ingest(host)
        first["api_content"] = API_NOTE
        result = _cleanup(engine, host)
        assert first["message_uid"] != second["message_uid"]
        assert [row["content"] for row in result if row["role"] == "assistant"] == [
            "First answer.\nSecond answer.",
        ]
        if patterns:
            _assert_absent(engine, result)


def test_restored_user_text_stays_utf8_after_redaction(open_engine, patterns):
    tail = _user("Current \ud800 question.", api_content=API_NOTE)
    host = [*_history(), tail]
    with open_engine(**(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        result = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        row, = [row for row in result if row.get("message_uid") == tail["message_uid"]]
        assert row["content"] == "Current \ufffd question."
        _assert_sidecar(row, API_NOTE, patterns)
        if patterns:
            assert SENSITIVE not in repr(result)


def test_literal_marker_annotation_keeps_sidecar_in_cache(open_engine, patterns):
    tail = _user(LITERAL)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**(REDACTION | {"sensitive_patterns_enabled": patterns})) as engine:
        engine.ingest(host)
        tail["content"] += "\n\n" + _todo("Synthetic annotation")
        assert _strip_volatile_host_scaffolding(tail["content"]) == LITERAL
        tail["api_content"] = "Independent synthetic API text"
        result = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        assert _cached_row(engine, tail["message_uid"])["api_content"] == tail["api_content"]
        assert result[-1]["api_content"] == tail["api_content"]


@pytest.mark.xfail(strict=True, reason=PRE_EXISTING)
def test_literal_marker_inside_sensitive_keyed_value(open_engine):
    tail = _user(LITERAL)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        tail["api_content"] = {"api_key": SENSITIVE + " [LCM sensitive redaction: example]"}
        result = _compress(engine, host, force=True)
        assert engine._last_compress_leaves[1] == 1
        assert SENSITIVE not in repr(result)


@pytest.mark.xfail(strict=True, reason=PRE_EXISTING)
def test_hidden_only_pass_with_keyed_sidecar(open_engine, backlog_engine):
    engine, _ = backlog_engine
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    host = hidden_backlog._hidden_view(engine)
    host[0]["api_content"] = {"api_key": SENSITIVE}
    frontier = engine._store_complete_frontier()
    result = hidden_backlog._compress(engine, host)
    assert engine._store_complete_frontier() > frontier
    assert engine._compress_host_rows_consumed == 0
    assert SENSITIVE not in repr(result)


@pytest.mark.xfail(strict=True, reason=PRE_EXISTING)
def test_fresh_user_sidecar_on_ordinary_noop(open_engine):
    tail = _user("Fresh question.", api_content={"api_key": SENSITIVE})
    with open_engine(**REDACTION) as engine:
        assert not engine._last_active_replay_messages
        result = _compress(engine, [tail])
        assert engine._last_compression_status == "noop"
        assert result[-1]["content"] == tail["content"]
        assert SENSITIVE not in repr(result)


def test_fresh_user_sidecar_on_ordinary_leaf_pass(open_engine, monkeypatch):
    tail = _user("Fresh question.", api_content={"api_key": SENSITIVE})
    host = [*_history(), tail]
    with open_engine(context_threshold=0.01, **REDACTION) as engine:
        monkeypatch.setattr(engine, "_keep_host_held_stubs", lambda *_: pytest.fail("Expected a cache miss"))
        result = _compress(engine, host)
        assert engine._last_compress_leaves[1] == 1
        assert result[-1]["content"] == tail["content"]
        assert SENSITIVE not in repr(result)
