"""Round-10 regressions for preserving marker spans while redacting keyed values."""

import re

from hermes_lcm.ingest_protection import redact_sensitive_value
from tests.invariants.test_p0_host_user_rows import _assistant, _compress, _history, _stamp, _user
from tests.invariants.test_r4_host_user_replay import _todo
from tests.invariants.test_r7_host_user_replay import (
    MARKER, REDACTION, SENSITIVE, _cached_row, open_engine as open_engine,
)
from tests.invariants.test_r8_host_user_replay import LITERAL, _cleanup
from tests.invariants.test_r9_host_user_replay import cache_hits as cache_hits

LITERAL_MARKER = "[LCM sensitive redaction: example]"
SIDECAR = {"api_key": f"{SENSITIVE} {LITERAL_MARKER}", "note": "Keep this API note."}


def _returned_row(result, host_row):
    row, = [row for row in result if row.get("message_uid") == host_row["message_uid"]]
    return row


def _assert_protected_pair(engine, result, tail):
    returned = _returned_row(result, tail)
    cached = _cached_row(engine, tail["message_uid"])
    assert (repr(result).count(SENSITIVE), repr(engine._last_active_replay_messages).count(SENSITIVE)) == (0, 0)
    expected = SIDECAR | {"api_key": f"<redacted> {LITERAL_MARKER}"}
    for row in (returned, cached):
        api = row["api_content"]
        assert api | {"api_key": re.sub(MARKER, "<redacted>", api["api_key"])} == expected
    return returned, cached


def test_keyed_sidecar_arrives_one_cache_hit_after_annotation(open_engine, cache_hits):
    tail = _user(LITERAL)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        hits = cache_hits(engine)
        identity = engine._message_replay_identity(tail)
        tail["content"] += "\n\n" + _todo("Synthetic annotation")
        tail["api_content"] = {"note": "First API note."}
        assert engine._message_replay_identity(tail) == identity
        first = _compress(engine, host, force=True)
        assert hits and engine._last_compress_leaves[1] == 1
        assert _returned_row(first, tail)["api_content"] == tail["api_content"]
        assert _cached_row(engine, tail["message_uid"])["content"] == tail["content"]

        tail["api_content"] = SIDECAR
        hits.clear()
        second = _cleanup(engine, host)
        assert hits
        returned, cached = _assert_protected_pair(engine, second, tail)
        assert returned["content"] == cached["content"] == tail["content"]


def test_keyed_sidecar_on_fresh_visible_leaf_assistant_tail(open_engine, cache_hits):
    tail = _assistant(f"api_key={SENSITIVE} Keep this answer.")
    tail["api_content"] = SIDECAR
    host = [*_history(), _user("Final synthetic question."), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        hits = cache_hits(engine)
        assert not engine._last_active_replay_messages
        result = _compress(engine, host, force=True)
        assert not hits and engine._last_compress_leaves[1] == 1
        returned, cached = _assert_protected_pair(engine, result, tail)
        for row in (returned, cached):
            assert re.sub(MARKER, "<redacted>", row["content"]) == "api_key=<redacted> Keep this answer."


def test_reverted_quotation_keeps_marker_sidecar(open_engine, cache_hits):
    tail = _user(LITERAL)
    host = [*_history(), tail]
    _stamp(host)
    with open_engine(**REDACTION) as engine:
        engine.ingest(host)
        hits = cache_hits(engine)
        tail["content"] += "\n\n" + _todo("Synthetic annotation")
        tail["api_content"] = {"note": LITERAL}
        _compress(engine, host, force=True)
        assert hits and engine._last_compress_leaves[1] == 1

        tail["content"] = LITERAL
        hits.clear()
        result = _cleanup(engine, host)
        assert hits
        returned = _returned_row(result, tail)
        cached = _cached_row(engine, tail["message_uid"])
        assert returned["content"] == cached["content"] == LITERAL
        assert (returned.get("api_content"), cached.get("api_content")) == (tail["api_content"],) * 2
        assert (repr(result).count(SENSITIVE), repr(engine._last_active_replay_messages).count(SENSITIVE)) == (0, 0)


def test_keyed_redaction_preserves_placeholder_bytes(open_engine):
    with open_engine(**REDACTION) as engine:
        config = engine._config
        original = redact_sensitive_value({"api_key": SENSITIVE}, config)
        placeholder = original["api_key"]
        assert redact_sensitive_value(original, config)["api_key"].encode() == placeholder.encode()

        value = f"{SENSITIVE}\t{placeholder}  {LITERAL_MARKER}\n{SENSITIVE}"
        protected = redact_sensitive_value({"api_key": value}, config)
        expected = f"{placeholder}\t{placeholder}  {LITERAL_MARKER}\n{placeholder}"
        assert protected["api_key"].encode() == expected.encode()
        assert redact_sensitive_value(protected, config)["api_key"].encode() == expected.encode()
