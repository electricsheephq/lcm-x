"""#2344: compress's id()-keyed store-id map ends with the pass.

The map keys are id() of the pass's own working copies. Left behind, a later ingest could meet a placeholder copy
at a reused address, read it as a mapped stored row, and store a carried "message ignored" placeholder."""

import builtins

import pytest

import hermes_lcm.placeholder_ledger as placeholder_ledger
from hermes_lcm import message_patterns
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_lcm_engine import _FakeTimeoutRegexEngine

PLACEHOLDER = "LCM active replay placeholder: message ignored"


@pytest.fixture(autouse=True)
def _timeout_capable_regex_engine(monkeypatch):
    monkeypatch.setattr(message_patterns, "_regex_engine", _FakeTimeoutRegexEngine)  # as TestMessageFiltering does


def _engine(tmp_path):
    return LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=10,
        ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
        sensitive_patterns_enabled=True, sensitive_patterns=["api_key"],
    ))


def _first_pass(engine):
    engine.on_session_start("old-session", platform="telegram", context_length=1000, conversation_id="conv")
    before = dict(engine._compress_run.store_ids_by_message_id)
    result = engine.compress(
        [
            {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
            {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
            {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
        ],
        current_tokens=10_000, force=True,
    )
    assert sum(PLACEHOLDER in str(m.get("content", "")) for m in result) == 2
    return before, result


def test_compress_leaves_no_store_id_map_behind(tmp_path):
    engine = _engine(tmp_path)
    try:
        before, _ = _first_pass(engine)
        assert engine._compress_run.store_ids_by_message_id == before
    finally:
        engine.shutdown()


def test_reused_address_cannot_store_a_carried_ignored_placeholder(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        pass_keys = []
        end_phase = engine._end_pre_leaf_phase

        def record_pass_keys(*args, **kwargs):
            # Also runs mid-pass; its last call is in compress()'s cleanup, before the restore, so the final
            # capture is the map the pass would otherwise leave behind.
            pass_keys[:] = engine._compress_run.store_ids_by_message_id
            return end_phase(*args, **kwargs)

        monkeypatch.setattr(engine, "_end_pre_leaf_phase", record_pass_keys)
        _, result = _first_pass(engine)
        assert pass_keys  # the pass mapped its working copies, so the stand-in below always substitutes

        def reused_id(obj):
            # Deterministic stand-in for CPython reusing a freed working copy's address: a placeholder copy built
            # during the next ingest gets the id() of a working copy the compress pass mapped.
            if isinstance(obj, dict) and PLACEHOLDER in str(obj.get("content", "")):
                return pass_keys[0]
            return builtins.id(obj)

        monkeypatch.setattr(placeholder_ledger, "id", reused_id, raising=False)
        engine.on_session_start("new-session", platform="telegram", context_length=1000, conversation_id="conv",
                                boundary_reason="compression", old_session_id="old-session")
        engine._ingest_messages(result)
        stored = engine._store.get_session_messages("new-session")
        assert all(PLACEHOLDER not in str(row["content"]) for row in stored)
    finally:
        engine.shutdown()


def test_map_is_restored_when_pass_cleanup_raises(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        engine.on_session_start("old-session", platform="telegram", context_length=1000, conversation_id="conv")
        before = engine._compress_run.store_ids_by_message_id

        def broken_cleanup(*_args, **_kwargs):
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(engine, "_end_pre_leaf_phase", broken_cleanup)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.compress(
                [
                    {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                    {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
                    {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
                ],
                current_tokens=10_000, force=True,
            )
        assert engine._compress_run.store_ids_by_message_id is before
    finally:
        monkeypatch.undo()
        engine.shutdown()
