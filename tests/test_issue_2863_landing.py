"""#2863: the progress flag is computed once, on the list object the #2855 return boundary hands to the host.

Progress is reported when a pass changes the list by content and the engine's count of it shrank. A pass that lands
just under the trigger can still re-arm the host's per-turn budget, as the host's own compressor does (#2904).
Synthetic data only.
"""

import copy
import json
import random
import types

import pytest

from hermes_lcm import compaction as C
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_2863_host_contract import USER, WINDOW, force_char_estimate, stub_summarizer

pytestmark = pytest.mark.issue(2863)

WORDS = ("the parser test failed because the timestamp check compares local time with utc and the fixture "
         "builds a naive datetime so the assertion sees an offset of one hour in the summer months only").split()


def prose(seed, chars):
    rnd = random.Random(seed)
    out, n = [], 0
    while n < chars:
        word = rnd.choice(WORDS)
        out.append(word)
        n += len(word) + 1
    return " ".join(out)[:chars]


def prose_step(r, chars):
    cid = f"call_{r:03d}_0"
    return [{"role": "assistant", "content": f"Reading (step {r}).",
             "tool_calls": [{"id": cid, "type": "function",
                             "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/m{r}.py"})}}]},
            {"role": "tool", "tool_call_id": cid, "content": prose(r * 10, chars)}]


def fixed_saving(eng, save_chars):
    """compress() runs the engine's own progress report around a pass that cuts save_chars chars from the oldest
    tool row long enough to hold them; pairing is untouched."""
    def op(messages):
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "tool" and len(m.get("content") or "") > save_chars + 50:
                m["content"] = m["content"][save_chars:]
                return out
        return list(messages)
    eng.compress = types.MethodType(C._reports_progress(lambda self, messages, *a, **k: op(messages)), eng)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    force_char_estimate(monkeypatch)
    home = tmp_path / "hermes-home"
    home.mkdir()
    eng = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(home))
    eng.on_session_start("s2863", platform="cli", model="probe-model", context_length=WINDOW)
    eng.update_model("probe-model", WINDOW, provider="custom")
    monkeypatch.setattr(eng, "_summarize_leaf_chunk_with_rescue", stub_summarizer)
    try:
        yield eng
    finally:
        eng.shutdown()


MSGS = [dict(USER)] + prose_step(1, 4_000) + prose_step(2, 500)
SAVE = 2_000


# ---------------------------------------------------------------- one seam, under the #2855 return boundary
def test_the_flag_matches_the_list_the_host_receives_after_the_return_boundary(engine, monkeypatch):
    msgs = copy.deepcopy(MSGS)
    msgs[-1]["api_content"] = msgs[-1]["content"]

    def op(self, messages, *a, **k):
        out = [dict(m) for m in messages]
        out[-1]["content"] = out[-1]["content"][:100]  # keeps the input's api_content: the boundary drops it
        return out
    monkeypatch.setattr(C.CompactionMixin, "compress", C._reports_progress(op))
    out = engine.compress(msgs, current_tokens=10)
    assert "api_content" not in out[-1] and "api_content" in msgs[-1]
    assert engine._last_compression_made_progress is True
    assert C._made_progress(C._progress_view(msgs), out) is True


def test_a_refused_pass_reports_no_progress(engine):
    fixed_saving(engine, SAVE)
    engine.compress(list(MSGS), current_tokens=10)
    assert engine._last_compression_made_progress is True
    del engine.compress  # the engine's own compress(), with the #2855 wrapper
    engine._compaction_closing = True
    try:
        msgs = list(MSGS)
        assert engine.compress(msgs, current_tokens=10) is msgs
        assert engine._last_compression_made_progress is False
    finally:
        engine._compaction_closing = False
