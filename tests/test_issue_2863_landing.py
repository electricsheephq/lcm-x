"""#2863 D6: a pass reports progress to the host only when its projected landing is at or under the trigger.

L = P - (est_in - est_out). P is the pressure the host passed to compress() (else the engine's last real prompt);
est_in and est_out are the engine's counts of the input and the returned list. Progress needs the list changed by
content, smaller by the counter, and L <= (1 - m) x threshold_tokens, m = LCM_PROGRESS_LANDING_MARGIN_RATIO. Without a
positive P the rule is the earlier one (changed and smaller).

The band regression drives the host model of tests/test_issue_2863_host_contract.py (Hermes 3b421578's per-turn tool
loop, rules cited there) on a prompt 100 tokens under a 48,000 trigger, ~199 provider tokens of regrowth per request
and a pass that saves the same each time. The host re-arms its per-turn budget only after a pass that reported
progress (conversation_compression.py 3b421578:3420-3427; turn_usage.py 3b421578:128-162). Synthetic data only.
"""

import copy
import json
import random
import types

import pytest

from hermes_lcm import compaction as C
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_2863_host_contract import (
    USER, WINDOW, Anchor, ModelClock, force_char_estimate, host_rough, real_msg_tokens, stub_summarizer,
)

pytestmark = pytest.mark.issue(2863)

ENV = "LCM_PROGRESS_LANDING_MARGIN_RATIO"
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


def real_tokens(msgs, overhead=0):
    return int(overhead + sum(real_msg_tokens(m) for m in msgs))


def fixed_saving(eng, save_chars):
    """compress() runs the engine's own _counted_compaction around a pass that cuts save_chars chars from the oldest
    tool row long enough to hold them; pairing is untouched."""
    def op(messages):
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "tool" and len(m.get("content") or "") > save_chars + 50:
                m["content"] = m["content"][save_chars:]
                return out
        return list(messages)
    eng.compress = types.MethodType(C._counted_compaction(lambda self, messages, *a, **k: op(messages)), eng)


def saving_of(msgs, save_chars):
    """The engine's own count of what one fixed_saving pass removes from msgs."""
    before = C._progress_view(msgs)
    after = copy.deepcopy(before)
    row = next(m for m in after if m["role"] == "tool" and len(m["content"]) > save_chars + 50)
    row["content"] = row["content"][save_chars:]
    return count_messages_tokens(before) - count_messages_tokens(after)


def margin_engine(tmp_path, monkeypatch, margin=0.0):
    """The host-contract engine; a margin other than 0 is passed to the config (base has no such field)."""
    force_char_estimate(monkeypatch)
    home = tmp_path / "hermes-home"
    home.mkdir()
    extra = {"progress_landing_margin_ratio": margin} if margin else {}
    eng = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), **extra), hermes_home=str(home))
    eng.on_session_start("s2863", platform="cli", model="probe-model", context_length=WINDOW)
    eng.update_model("probe-model", WINDOW, provider="custom")
    monkeypatch.setattr(eng, "_summarize_leaf_chunk_with_rescue", stub_summarizer)
    return eng


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = margin_engine(tmp_path, monkeypatch)
    try:
        yield eng
    finally:
        eng.shutdown()


MSGS = [dict(USER)] + prose_step(1, 4_000) + prose_step(2, 500)
SAVE = 2_000


# ---------------------------------------------------------------- the rule
def test_a_landing_over_the_trigger_reports_no_progress(engine):
    fixed_saving(engine, SAVE)
    thr, saved = engine.threshold_tokens, saving_of(MSGS, SAVE)
    out = engine.compress(list(MSGS), current_tokens=thr + saved + 1)
    assert out != MSGS and count_messages_tokens(out) < count_messages_tokens(MSGS)
    assert engine._last_compression_made_progress is False


@pytest.mark.parametrize("over", [0, -1, -5_000])
def test_a_landing_at_or_under_the_trigger_reports_progress(engine, over):
    fixed_saving(engine, SAVE)
    thr, saved = engine.threshold_tokens, saving_of(MSGS, SAVE)
    engine.compress(list(MSGS), current_tokens=thr + saved + over)
    assert engine._last_compression_made_progress is True


def test_without_current_tokens_the_last_real_prompt_is_the_pressure(engine):
    fixed_saving(engine, SAVE)
    thr, saved = engine.threshold_tokens, saving_of(MSGS, SAVE)
    engine.last_prompt_tokens = thr + saved + 1
    engine.compress(list(MSGS))
    assert engine._last_compression_made_progress is False
    engine.last_prompt_tokens = thr + saved
    engine.compress(list(MSGS), current_tokens=None)
    assert engine._last_compression_made_progress is True


@pytest.mark.parametrize("current, last", [(None, 0), (0, -1), (None, -1)])
def test_without_a_positive_pressure_the_rule_is_changed_and_smaller(engine, current, last):
    fixed_saving(engine, SAVE)
    engine.last_prompt_tokens = last
    engine.compress(list(MSGS), current_tokens=current)
    assert engine._last_compression_made_progress is True
    engine.compress(list(MSGS[:2]), current_tokens=current)  # nothing long enough to cut: returned unchanged
    assert engine._last_compression_made_progress is False


def test_an_unchanged_list_reports_no_progress_at_any_pressure(engine):
    fixed_saving(engine, 10 ** 6)
    engine.compress(list(MSGS), current_tokens=10)
    assert engine._last_compression_made_progress is False


@pytest.mark.parametrize("margin", [0.05, 0.5])
def test_the_margin_moves_the_line_to_its_share_of_the_trigger(tmp_path, monkeypatch, margin):
    eng = margin_engine(tmp_path, monkeypatch, margin)
    try:
        fixed_saving(eng, SAVE)
        line, saved = int((1 - margin) * eng.threshold_tokens), saving_of(MSGS, SAVE)
        eng.compress(list(MSGS), current_tokens=line + saved + 1)
        assert eng._last_compression_made_progress is False
        eng.compress(list(MSGS), current_tokens=line + saved)
        assert eng._last_compression_made_progress is True
    finally:
        eng.shutdown()


# ---------------------------------------------------------------- the margin setting
def test_the_margin_is_unset_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c = LCMConfig.from_env()
    assert LCMConfig().progress_landing_margin_ratio == 0.0 == c.progress_landing_margin_ratio
    assert c.config_sources["progress_landing_margin_ratio"] == "default"
    assert not any(ENV in warning for warning in c.config_source_warnings)


@pytest.mark.parametrize("raw, value", [("0.05", 0.05), ("0.5", 0.5), ("0", 0.0)])
def test_a_margin_in_range_is_read_from_the_env(monkeypatch, raw, value):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.progress_landing_margin_ratio == value
    assert c.config_sources["progress_landing_margin_ratio"] == f"env:{ENV}"
    assert not any(ENV in warning for warning in c.config_source_warnings)


@pytest.mark.parametrize("raw", ["-0.1", "0.6", "1", "nan", "inf", "abc"])
def test_an_invalid_margin_keeps_zero_with_a_warning(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.progress_landing_margin_ratio == 0.0
    assert c.config_sources["progress_landing_margin_ratio"] == "default"
    assert any(ENV in warning for warning in c.config_source_warnings)


def test_a_margin_out_of_range_in_a_built_config_counts_as_zero(tmp_path, monkeypatch):
    eng = margin_engine(tmp_path, monkeypatch, 0.9)
    try:
        fixed_saving(eng, SAVE)
        thr, saved = eng.threshold_tokens, saving_of(MSGS, SAVE)
        eng.compress(list(MSGS), current_tokens=thr + saved)
        assert eng._last_compression_made_progress is True
    finally:
        eng.shutdown()


# ---------------------------------------------------------------- one seam, under the #2855 return boundary
def test_the_flag_matches_the_list_the_host_receives_after_the_return_boundary(engine, monkeypatch):
    msgs = copy.deepcopy(MSGS)
    msgs[-1]["api_content"] = msgs[-1]["content"]

    def op(self, messages, *a, **k):
        out = [dict(m) for m in messages]
        out[-1]["content"] = out[-1]["content"][:100]  # keeps the input's api_content: the boundary drops it
        return out
    monkeypatch.setattr(C.CompactionMixin, "compress", C._counted_compaction(op))
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


# ---------------------------------------------------------------- band regression (host model)
class BandHost:
    """The host loop of tests/test_issue_2863_host_contract.py Host.run with a prefilled start and prose steps."""

    def __init__(self, eng, start, overhead, clock, *, requests=40, step_chars=500, completion=1_586, cap=3):
        self.eng, self.start, self.clock, self.requests, self.step_chars = eng, start, clock, requests, step_chars
        self.overhead, self.rough_overhead, self.completion, self.cap = overhead, int(overhead / 1.2), completion, cap
        self.thr = int(eng.threshold_tokens)
        self.attempts, self.blocked, self.last_pre, self.anchor = 0, False, None, None
        self.commits, self.rows = 0, []

    def compress(self, msgs, pressure):
        eng = self.eng
        before = copy.deepcopy(msgs)
        out = eng.compress(list(msgs), current_tokens=pressure)
        made_progress = bool(getattr(eng, "_last_compression_made_progress", False))  # base has no such attribute
        if out == before or not out:
            return before
        self.commits += 1
        eng.on_session_start("s2863", boundary_reason="compression", old_session_id="s2863", platform="cli")
        eng.last_prompt_tokens = -1
        eng.last_completion_tokens = 0
        eng.awaiting_real_usage_after_compression = True
        self.anchor = None
        if made_progress:
            eng._verify_compaction_cleared_threshold = True
        return list(out)

    def run(self):
        eng, msgs, r = self.eng, copy.deepcopy(self.start), 0
        eng.should_compress_preflight(list(msgs))
        for _guard in range(2_000):
            if r >= self.requests:
                break
            a = self.anchor.tokens(msgs) if self.anchor else None
            P = a if a is not None else host_rough(msgs) + self.rough_overhead
            prev, self.last_pre = self.last_pre, None
            if prev is not None and P >= self.thr and not (prev > 0 and P < prev * 0.95):
                self.blocked = True
            defer = False if a is not None else bool(getattr(eng, "should_defer_preflight_to_real_usage", lambda _t: False)(P))
            if len(msgs) > 1 and self.attempts < self.cap and not self.blocked and not defer and eng.should_compress(P):
                self.attempts += 1
                self.last_pre = P
                msgs = self.compress(msgs, P)
                continue
            r += 1
            self.clock.tick()
            prompt = real_tokens(msgs, self.overhead)
            verify_before = bool(getattr(eng, "_verify_compaction_cleared_threshold", False))
            eng.update_from_response({"prompt_tokens": prompt, "completion_tokens": self.completion,
                                      "total_tokens": prompt + self.completion})
            self.anchor = Anchor(prompt, self.completion, msgs)
            if self.attempts and verify_before and 0 < prompt < self.thr:
                self.attempts, self.blocked, self.last_pre = 0, False, None
            self.rows.append({"req": r, "provider": prompt, "commits": self.commits})
            self.commits = 0
            if r >= self.requests:
                break
            msgs = msgs + prose_step(r, self.step_chars)
            if getattr(eng, "awaiting_real_usage_after_compression", False):
                continue
            P2 = self.anchor.tokens(msgs)
            if P2 is None:
                lp = int(eng.last_prompt_tokens or 0)
                P2 = lp if lp > 0 else 0 if lp == -1 else host_rough(msgs) + self.rough_overhead
            if self.attempts < self.cap and eng.should_compress(P2):
                self.attempts += 1
                msgs = self.compress(msgs, P2)
        return self.rows


def test_band_regression_a_pass_landing_just_under_the_trigger_does_not_rearm_every_request(engine, monkeypatch):
    thr = int(engine.threshold_tokens)
    assert thr == 48_000
    start = [dict(USER)] + [m for g in range(3) for m in prose_step(900 + g, 40_000)]
    overhead = thr - 100 - real_tokens(start)  # request 1's provider prompt is 100 under the trigger
    regrowth = real_tokens(prose_step(1, 500))
    assert 195 <= regrowth <= 205
    fixed_saving(engine, round(regrowth / 0.30))  # the same saving as the regrowth, every pass
    rows = BandHost(engine, start, overhead, ModelClock(monkeypatch)).run()
    per = [x["commits"] for x in rows]
    longest = run = 0
    for c in per:
        run = run + 1 if c else 0
        longest = max(longest, run)
    assert len(rows) == 40
    # Base gives 3 commits and a longest run of 3 here; reporting progress on every pass gave 39 and 39.
    assert (sum(per), longest) == (3, 3), per
    assert max(x["provider"] for x in rows) < WINDOW
