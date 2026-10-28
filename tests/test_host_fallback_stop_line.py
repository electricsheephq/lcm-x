"""D5 (v0.27.0-rc4 soak): a bypassed session's host-compressor wait is booked as ``host=``, not ``pre=``.

Field shape (reference agent, 2026-10-13): async-delegation child sessions are auxiliary/stateless, so ``compress()`` delegates to
Hermes' native ContextCompressor. Its summary call (51-125 s) was reported as ``pre=`` on the
``LCM compaction stop: reason=host_fallback leaves=0`` line, which reads as an LCM pre-leaf stall."""
import logging
import re
import threading
import time

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
STOP = re.compile(r"LCM compaction stop: reason=(\S+) leaves=(\d+) progress=\S+ elapsed=([\d.]+)s pre=([\d.]+)s "
                  r"calls=([\d.]+)s finalize=([\d.]+)s backlog_tokens=(\d+)( host=([\d.]+)s)?")


@pytest.fixture(autouse=True)
def _no_host_estimate(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)


class _Clock:
    def __init__(self):
        self._real, self.offset = time.monotonic, 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


class _HostCompressor:
    """Hermes' native ContextCompressor stand-in: one summary call that takes ``seconds``."""

    compression_count = 0

    def __init__(self, advance, seconds, fail=False):
        self.advance, self.seconds, self.fail = advance, seconds, fail

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        self.advance(self.seconds)
        if self.fail:
            raise RuntimeError("summary call timed out")
        self.compression_count += 1
        return messages[:1] + messages[-4:]


def _turns(n):
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(n):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply {i}{PAD}"}]
    return rows


def _engine(tmp_path, session_id):
    eng = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
                                     max_assembly_tokens=100_000, database_path=str(tmp_path / "lcm.db"),
                                     stateless_session_patterns=["child-*"]))
    eng.on_session_start(session_id, platform="telegram", context_length=200_000, conversation_id="conv")
    return eng


def _stops(caplog):
    return [STOP.search(r.getMessage()) for r in caplog.records if r.getMessage().startswith("LCM compaction stop:")]


@pytest.mark.parametrize("fail", [False, True])
def test_host_fallback_wait_is_booked_as_host_not_pre(tmp_path, monkeypatch, caplog, fail):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    eng = _engine(tmp_path, "child-20261013_183428")
    assert eng._bypasses_lcm_context_management()
    monkeypatch.setattr(eng, "_get_host_fallback_compressor",
                        lambda: _HostCompressor(lambda s: setattr(clock, "offset", clock.offset + s), 118.6, fail))
    monkeypatch.setattr(eng, "_sync_host_fallback_compressor", lambda c: None)
    monkeypatch.setattr(eng, "_mirror_host_fallback_state", lambda c: None)
    view = _turns(8)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        out = eng.compress(view, current_tokens=eng.threshold_tokens + 1)
    assert len(out) < len(view)  # the host (or the deterministic trim after its failure) bounded the list
    [m] = _stops(caplog)
    reason, leaves, elapsed, pre, calls, finalize = m.group(1), int(m.group(2)), *map(float, m.group(3, 4, 5, 6))
    assert (reason, leaves) == ("host_fallback", 0)
    assert m.group(9) is not None, "stop line must carry host= for a host-compressor wait"
    host = float(m.group(9))
    assert host == pytest.approx(118.6, abs=0.15)
    assert pre < 1.0, "the host's summary call is not LCM pre-leaf time"
    assert pre + calls + finalize + host == pytest.approx(elapsed, abs=0.15)  # #605 K8: the seconds add up
    eng.shutdown()


def test_lcm_stop_line_has_no_host_field(tmp_path, monkeypatch, caplog):
    import hermes_lcm.engine as lcm_engine
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                        lambda **kw: ("Earlier turns.\nExpand for details about: turns", 1))
    eng = _engine(tmp_path, "S")
    view = _turns(6)
    eng.ingest(view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        eng.compress(view + _turns(2)[1:], current_tokens=eng.threshold_tokens + 1)
    [m] = _stops(caplog)
    assert m.group(1) != "host_fallback" and m.group(8) is None  # every non-host line is byte-format unchanged
    eng.shutdown()


def test_concurrent_host_wait_stays_on_its_own_thread(tmp_path, monkeypatch, caplog):
    """The engine is shared by the parent and its live children (thread-context stateless): a child's host wait
    must not leak into a concurrent foreground compaction's stop line."""
    import hermes_lcm.engine as lcm_engine
    entered, release = threading.Event(), threading.Event()

    def slow_summary(**kw):
        entered.set()
        release.wait(5)
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", slow_summary)
    eng = _engine(tmp_path, "S")
    view = _turns(6)
    eng.ingest(view)

    def host_sleep(seconds):
        time.sleep(seconds)

    monkeypatch.setattr(eng, "_get_host_fallback_compressor", lambda: _HostCompressor(host_sleep, 0.4))
    monkeypatch.setattr(eng, "_sync_host_fallback_compressor", lambda c: None)
    monkeypatch.setattr(eng, "_mirror_host_fallback_state", lambda c: None)
    child_thread = {}

    def child():
        child_thread["id"] = threading.get_ident()
        eng._mark_thread_context_stateless("child-20261013_183508")
        try:
            entered.wait(5)
            eng.compress(_turns(8), current_tokens=eng.threshold_tokens + 1)
        finally:
            eng._clear_thread_context_stateless("child-20261013_183508")
            release.set()

    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        t = threading.Thread(target=child)
        t.start()
        eng.compress(view + _turns(2)[1:], current_tokens=eng.threshold_tokens + 1)
        t.join(10)
    lines = {r.thread: STOP.search(r.getMessage()) for r in caplog.records
             if r.getMessage().startswith("LCM compaction stop:")}
    assert lines[child_thread["id"]].group(9) is not None
    assert float(lines[child_thread["id"]].group(9)) == pytest.approx(0.4, abs=0.2)
    assert lines[threading.get_ident()].group(8) is None, "the foreground line must not carry the child's host wait"
    eng.shutdown()
