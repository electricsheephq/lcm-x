"""A user-only stopped tail is still mutable at the floor host's retry seam."""
from copy import deepcopy
import hashlib
from unittest.mock import Mock

import pytest
from agent.context_engine import ContextEngine

from tests.harness import make_engine, pin_char_counter


@pytest.mark.parametrize("check", ["callback", "catchup"])
def test_user_only_stop_defers_until_retry_role_repair(tmp_path, monkeypatch, check):
    monkeypatch.setattr(ContextEngine, "on_turn_complete", lambda *a, **k: None, raising=False)
    pin_char_counter(monkeypatch)
    engine = make_engine(tmp_path, window=128000, platform="acp")
    warm = [{"role": "user", "content": "warm question", "timestamp": 1000.0},
            {"role": "assistant", "content": "warm answer", "timestamp": 1001.0}]
    user = {"role": "user", "content": "cancelled request", "timestamp": 1002.0}

    def ledger():
        return [(m["role"], hashlib.sha256(m["content"].encode()).hexdigest())
                for m in engine._store.get_session_messages("S")]

    try:
        engine.ingest(deepcopy(warm))
        before, cursor = ledger(), engine._ingest_cursor
        note = Mock(wraps=engine.note_turn_complete)
        monkeypatch.setattr(engine, "note_turn_complete", note)
        engine.on_turn_complete(deepcopy(warm + [user]), interrupted=True)
        if check == "callback":
            assert ledger() == before
            assert engine._ingest_cursor == cursor
            note.assert_not_called()

        # floor ACP attaches the cancelled prompt to the retry. With no
        # assistant boundary, role repair folds it into the previous user row,
        # retaining that row's timestamp and supplying no merge-prefix marker.
        retry = user["content"] + "\n\nUser correction/guidance after interrupt: retry request"
        merged = dict(user, content=user["content"] + "\n\n" + retry)
        final = warm + [merged, {"role": "assistant", "content": "retry answer", "timestamp": 1004.0}]
        engine.ingest(deepcopy(final))
        engine.ingest(deepcopy(final))
        expected = [(m["role"], hashlib.sha256(m["content"].encode()).hexdigest()) for m in final]
        assert ledger() == expected
    finally:
        engine.shutdown()
