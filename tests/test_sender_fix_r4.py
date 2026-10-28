"""FIX-R4: invalid senders preserve rows; ambiguous occurrences stay NULL."""
from copy import deepcopy

import pytest

from tests.harness import engine_context
from tests.test_sender_capture import (
    assert_capture, begin, capture_module, clean_capture_registry, expected, message,
)

__all__ = ["clean_capture_registry"]


def test_surrogate_sender_does_not_lose_ingest_batch(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        view = [message(), dict(role="assistant", content="synthetic reply")]
        begin(e, view[:1], sender_id=chr(0xD800))
        e.ingest(view)
        assert e._store.connection.execute("SELECT count(*) FROM messages").fetchone()[0] == len(view)
        assert_capture(e, [(None, None)] * len(view))


def test_surrogate_sender_is_refused_at_creation(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        if begin(e, [message()], sender_id=chr(0xD800)) is not None:
            raise AssertionError("surrogate sender was admitted")


def test_valid_non_ascii_sender_is_captured(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message()
        sender = "ünïcode-名前"
        assert begin(e, [current], sender_id=sender) is not None
        e.ingest([current])
        assert_capture(e, [(f"telegram:{sender}", "gateway_principal")])


@pytest.mark.parametrize("shape", ["same-object", "same-uid"])
def test_multiple_occurrence_matches_leave_all_rows_null(tmp_path, shape):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message(**({"message_uid": "a" * 32} if shape == "same-uid" else {}))
        other = deepcopy(current) if shape == "same-uid" else current
        begin(e, [current])
        originals = [(0, current), (1, other)]
        pending, values = capture_module().batch_senders(e, originals)
        e.ingest([current, other])
        assert len(e._store.get_session_messages("S")) == 2
        assert_capture(e, [(None, None), (None, None)])
        assert (pending, values) == (None, None)


def test_single_occurrence_match_still_stamps_exact_row(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message(message_uid="b" * 32)
        old = message(1)
        begin(e, [old, current])
        e.ingest([old, current])
        assert_capture(e, [(None, None), expected()])
