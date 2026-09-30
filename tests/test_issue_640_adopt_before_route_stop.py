"""#640: on a host retry, the committed-replay adoption (#457) runs before the #628 summary-route stop.

The issue's sequence: a leaf is committed, rejections open every summary route, the host cancels the turn after
the commit and retries with the original input. The retry adopts the committed summary although the circuit is
still open; it writes no new leaf and calls no summariser."""

from __future__ import annotations

from copy import deepcopy

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "issue-640", "issue-640-conv"
HEADER = "Summary (d0, node "


class _Provider:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return f"Stub summary {self.calls} of ordered work.\nExpand for details about: stub", 1


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_path=str(tmp_path / "externalized"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=200_000)
    return engine


def _transcript(prefix_tokens=24_000):
    plain, i = [], 0
    while len(plain) < 40 or count_messages_tokens(plain[:-30]) < prefix_tokens:
        plain.append({"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"owned turn {i} " + " ".join(f"w{i}x{j}" for j in range(120))})
        i += 1
    return plain


def _compress(engine, messages):
    return engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)


def _leaves(engine):
    return [n for n in engine._dag.get_session_nodes(SID) if n.source_type == "messages"]


def _open_every_route_by_rejections(engine) -> None:
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.rejection_threshold):
        breaker.record_rejection(engine._config.summary_model)
    assert not engine._summary_route_available()


def _rows(engine):
    return engine._store._conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()


def test_retry_adopts_the_committed_summary_while_the_circuit_is_open(tmp_path, monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    messages = _transcript()
    original = deepcopy(messages)
    engine = _engine(tmp_path)
    try:
        engine.ingest(messages)
        _compress(engine, messages)  # committed; the host cancels the turn and discards this result
        (leaf,) = _leaves(engine)
        assert provider.calls == 1
        _open_every_route_by_rejections(engine)
        adopted = []
        real = LCMEngine._committed_replay_drops

        def spy(self, working, start):
            result = real(self, working, start)
            adopted.append(len(result[0]))
            return result

        monkeypatch.setattr(LCMEngine, "_committed_replay_drops", spy)
        out = _compress(engine, deepcopy(original))  # the retry with the original input
        assert adopted == [len(leaf.source_ids)] and adopted[0] > 0
        assert engine._last_compression_status == "compacted"
        assert out is not original and len(out) < len(original)
        assert sum(HEADER in str(m.get("content")) for m in out) == 1
        assert f"node {leaf.node_id}" in next(str(m["content"]) for m in out if HEADER in str(m.get("content")))
        assert provider.calls == 1 and _leaves(engine) == [leaf]  # no summariser call, no new leaf
        assert not engine._summary_route_available()  # the circuit is still open
        rows = _rows(engine)
        assert len(rows) == len(set(rows)) == len(original)
    finally:
        engine.shutdown()


def test_with_nothing_committed_the_route_stop_still_writes_nothing(tmp_path, monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    messages = _transcript()
    engine = _engine(tmp_path)
    try:
        engine.ingest(messages)
        _open_every_route_by_rejections(engine)
        out = _compress(engine, messages)
        assert out is messages and provider.calls == 0 and _leaves(engine) == []
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "summary route unavailable"
    finally:
        engine.shutdown()
