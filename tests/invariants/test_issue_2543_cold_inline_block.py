"""#2543 stays open in v0.30.0 (fix planned for v0.30.1): the busy session end of a non-current session.

When the end's read of the store fails as busy, the end's rows are kept as a pending tail and retried later. The
retry compares the rows the host re-sent with the stored rows and skips the rest of the batch on a mismatch, as the
off-current end does at once. A stored row that holds an inline encoded block (here a line-wrapped base64 body, or a
data URI used as an argument key) is read with LCM-X's own escaping in place, so it does not match the host's
unchanged copy, and the final reply after it is not stored. These strict xfails pin that loss on both cold routes of
the compare matrix (`cold-pairwise`, and `cold-counted`, which also takes the tail's absent count); a fix flips them.
The plain control (no inline block) stores the final reply on both routes.
"""

import base64
import copy
import json
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.invariants.test_p1e_cold_tails import fail_end_read


def _engine(tmp_path):
    home = tmp_path / "hermes"
    return LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"),
                                      large_output_externalization_path=str(home / "externalized")),
                     hermes_home=str(home))


def _b64(n):
    return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(n))).decode("ascii")


def _call(shape):
    if shape == "wrapped-base64":
        long = _b64(3420)
        wrapped = "\n".join(long[i:i + 76] for i in range(0, len(long), 76))
        arguments = json.dumps({"path": "a.b64", "content": "hdr\n" + wrapped + "\nend"})
    elif shape == "data-uri-key":
        arguments = json.dumps({"data:image\\/png;base64," + _b64(1200): "v", "b": "x"})
    else:  # plain control
        arguments = json.dumps({"path": "a.txt", "content": "hdr\nplain body\nend"})
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "m1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}]}


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


def _final_rows(tmp_path, monkeypatch, shape, route):
    """Return (FINAL rows stored, the counted route's absent count or None)."""
    engine = _engine(tmp_path)
    try:
        first = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                 {"role": "user", "content": "C"}, _call(shape),
                 {"role": "tool", "tool_call_id": "m1", "content": "ok 1"}]
        first += [{"role": r, "content": t} for r, t in
                  (("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H"))]
        for k, row in enumerate(first):
            row["timestamp"] = float(k + 1)
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest(copy.deepcopy(first))
        engine.on_session_start("foreground", platform="cli", context_length=200_000)
        _require(len(engine._store.get_session_messages("S1")) == len(first), "the first ingest stored every row")
        view = copy.deepcopy(first) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}]
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_range", fail_end_read(engine))
            engine.on_session_end("S1", view)
        _require(engine._pending_session_tails, "the busy read deferred the end's rows")
        tail = engine._pending_session_tails[-1]
        engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
        absent = engine._cold_tail_absent_count(tail) if route == "cold-counted" else None
        return sum(row.get("content") == "FINAL" for row in engine._store.get_session_messages("S1")), absent
    finally:
        engine.shutdown()


@pytest.mark.parametrize("route", ["cold-pairwise", "cold-counted"])
@pytest.mark.parametrize("shape", ["wrapped-base64", "data-uri-key"])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2543: a stored inline encoded block fails the busy end's retried compare")
def test_a_busy_end_stores_the_final_reply_after_an_inline_block(tmp_path, monkeypatch, shape, route):
    finals, absent = _final_rows(tmp_path, monkeypatch, shape, route)
    # Today's counted result: the inline row and the reply read as absent (2). A fix changes it with the reply.
    _require(route != "cold-counted" or absent == (0 if finals == 1 else 2), f"counted absent={absent}")
    assert finals == 1


@pytest.mark.parametrize("route", ["cold-pairwise", "cold-counted"])
def test_a_busy_end_stores_the_final_reply_after_a_plain_call(tmp_path, monkeypatch, route):
    finals, absent = _final_rows(tmp_path, monkeypatch, "plain", route)
    assert finals == 1
    assert absent == (0 if route == "cold-counted" else None)
