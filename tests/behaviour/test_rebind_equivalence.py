"""A warmed engine rebound to B behaves like a fresh engine on the same durable preimage."""

import json
import logging
import sqlite3
import time

import pytest

from hermes_lcm.tokens import count_messages_tokens
from tests.harness import ScriptedSummarizer, assistant, engine_context, pin_char_counter, prose, user


CONFIG = dict(fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.01,
              large_output_externalization_enabled=False, fresh_tail_pressure_yield_enabled=False)


class RebindDifference(AssertionError):
    """Keep setup failures outside the strict expected behavioral difference."""


def _history(tag, turns=6):
    return [row for i in range(turns) for row in (
        {**user(tag + prose(800, seed=i)), "timestamp": float(2 * i + 1)},
        {**assistant(tag + prose(800, seed=i + 10)), "timestamp": float(2 * i + 2)},
    )]


def _bytes(view):
    return json.dumps(view, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _warning(engine):
    return engine.get_automatic_compaction_status_message(phase="after", default_message="")


def _warm_a(engine, summaries):
    view = _history("A")
    compacted = engine.compress(view, force=True)
    assert summaries.calls and len(compacted) < len(view)
    # Real fitting of stored rows creates warning state without a model outage/circuit change.
    view = compacted + _history("A-fit", turns=3)
    engine.ingest(view)
    fitted = engine._survival_fit(view, view, count_messages_tokens(view), request_cap=500, reason="warm-a")
    assert len(fitted) < len(view)
    assert engine._survival_fit_pending_warning is not None
    engine.record_rejected_compaction()
    assert engine._no_progress_hold_active()
    engine.should_compress_preflight(view)
    assert engine._last_gate_tokens > 0


def _b_turns(engine, caplog):
    trace = [("initial-warning", _warning(engine))]
    view = [user("B start")]
    for turn in range(3):
        caplog.clear()
        view += _history(f"B{turn}", turns=2)
        engine.ingest(view)
        tokens = count_messages_tokens(view)
        preflight = engine.should_compress_preflight(view)
        decision = engine.should_compress(tokens)
        if preflight or decision:
            view = engine.compress(view, current_tokens=tokens)
        fitted = engine._survival_fit(view, view, count_messages_tokens(view),
                                      request_cap=500, reason="b-turn")
        trace.append((preflight, decision, _bytes(view), _bytes(fitted),
                      engine._last_compression_status, _warning(engine), _warning(engine),
                      tuple(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)))
        view = fitted
        engine.note_turn_complete()
    return trace


@pytest.mark.parametrize("boundary", ["session", "conversation", "reset", "profile"])
def test_rebound_b_matches_fresh_b(tmp_path, monkeypatch, caplog, boundary):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    summaries = ScriptedSummarizer().install(monkeypatch)
    with engine_context(tmp_path / "warm", window=20_000, session_id="A", conversation_id="A",
                        **CONFIG) as warmed:
        _warm_a(warmed, summaries)
        # Equal preimages preserve row/node identifiers in emitted citation bytes. The fresh
        # engine inherits no Python state, but sees exactly the same durable A history.
        fresh_path = tmp_path / "fresh"
        fresh_path.mkdir()
        with sqlite3.connect(fresh_path / "lcm.db") as destination:
            warmed._store.connection.backup(destination)
        session = "A" if boundary == "conversation" else "B"
        kwargs = dict(platform="cli", context_length=20_000, conversation_id="B")
        if boundary == "reset":
            warmed.on_session_reset()
        elif boundary == "profile":
            kwargs["hermes_home"] = str(tmp_path / "other-home")
        warmed.on_session_start(session, **kwargs)
        with engine_context(fresh_path, window=20_000, session_id=session, conversation_id="B",
                            **CONFIG) as fresh:
            actual, expected = _b_turns(warmed, caplog), _b_turns(fresh, caplog)
            assert any(row[1] for row in expected[1:]), "B must exercise compaction decisions"
            assert any(row[2] != row[3] for row in expected[1:]), "B must exercise survival fitting"
            assert expected[0] == ("initial-warning", None)
            assert expected[1][5] and all(row[6] is None for row in expected[1:])
            assert all(row[5] is None for row in expected[2:]), "B warns only once"
            if boundary == "conversation":
                # Deliberate policy, not equivalence (test_issue_597_progress_hold.py): a same-session
                # conversation rebind keeps A's hold until B's first turn ends (engine.py:4186 resets only
                # counters). B's first compaction is suppressed; later decisions match, later bytes do not.
                assert (actual[1][:2], expected[1][:2]) == ((False, False), (True, True))
                assert [row[:2] for row in actual[2:]] == [row[:2] for row in expected[2:]]
                return
            if actual != expected:
                fields = ("preflight", "decision", "context", "fit", "status", "warning", "repeat", "logs")
                differences = {turn: [fields[i] for i, (a, b) in enumerate(zip(left, right)) if a != b]
                               for turn, (left, right) in enumerate(zip(actual[1:], expected[1:]), 1)
                               if left != right}
                raise RebindDifference({"initial": (actual[0], expected[0]), "turns": differences,
                                        "first_decisions": (actual[1][:2], expected[1][:2])})
