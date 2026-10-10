"""#614: the leaf summary target is configuration (ratio / min / max); unset, every number equals the base.

The target is min(max, max(min, int(source_tokens * ratio))) and the L1 call receives twice the target as
max_tokens. T5 replays a compaction fixture whose (source_tokens, target) pairs and max_tokens values were
captured on the base (v0.24.6, ae0e996de2) before the knobs existed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta"
ACCEPTED = "Earlier turns.\nExpand for details about: turns"

# Captured at the base with the fixture in _compaction_fixture (one leaf per regime: floor, ratio, cap; the
# 7-token replies fall through to L2 and L3).
BASE_LEAF_BUDGETS = [[8633, 2000], [7, 2000], [28758, 5751], [7, 2000], [92008, 12000], [7, 2000]]
# #605 F2: the three 7-token leaves are written verbatim with no call (at the base each made a level 1 call of
# 4000 and a level 2 call of 2000 max tokens, both rejected as not shorter, then took that same level 3).
BASE_MAX_TOKENS = [4000, 11502, 24000]
BASE_LEAF_NODES = [(7, 7), (7, 7), (7, 7), (8633, 12), (28758, 12), (92008, 12)]


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """Same counter for every run (#615): LCM's character estimate, no host estimator."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture(autouse=True)
def _isolated_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    for key in ("LCM_LEAF_TARGET_RATIO", "LCM_LEAF_TARGET_MIN_TOKENS", "LCM_LEAF_TARGET_MAX_TOKENS"):
        monkeypatch.delenv(key, raising=False)


def _target(source_tokens: int, **config) -> int:
    return LCMEngine._leaf_target_tokens(SimpleNamespace(_config=LCMConfig(**config)), source_tokens)


# -- T1: the defaults reproduce today's target -----------------------------------------------------------------

@pytest.mark.parametrize("source_tokens, expected", [(8_000, 2_000), (20_000, 4_000), (100_000, 12_000)])
def test_t1_default_target_matches_the_historical_constants(source_tokens, expected):
    assert _target(source_tokens) == expected
    assert expected == min(12000, max(2000, int(source_tokens * 0.20)))


# -- T2: the knobs move the target -----------------------------------------------------------------------------

def test_t2_knobs_change_the_target():
    fixed = {"leaf_target_ratio": 0.12, "leaf_target_min_tokens": 2400, "leaf_target_max_tokens": 2400}
    assert _target(20_000, **fixed) == 2400
    assert _target(20_000, leaf_target_ratio=0.5, leaf_target_min_tokens=200, leaf_target_max_tokens=50_000) == 10_000


# -- T3: env parsing and validation ----------------------------------------------------------------------------

def test_t3_env_defaults_and_parsing(monkeypatch):
    c = LCMConfig.from_env()
    assert (c.leaf_target_ratio, c.leaf_target_min_tokens, c.leaf_target_max_tokens) == (0.20, 2000, 12_000)
    assert c.config_sources["leaf_target_ratio"] == "default"
    monkeypatch.setenv("LCM_LEAF_TARGET_RATIO", "0.12")
    monkeypatch.setenv("LCM_LEAF_TARGET_MIN_TOKENS", "2400")
    monkeypatch.setenv("LCM_LEAF_TARGET_MAX_TOKENS", "30000")
    c = LCMConfig.from_env()
    assert (c.leaf_target_ratio, c.leaf_target_min_tokens, c.leaf_target_max_tokens) == (0.12, 2400, 30_000)
    assert c.config_sources["leaf_target_max_tokens"] == "env:LCM_LEAF_TARGET_MAX_TOKENS"
    assert c.config_source_warnings == []


@pytest.mark.parametrize("env, field, default", [
    ({"LCM_LEAF_TARGET_RATIO": "0"}, "leaf_target_ratio", 0.20),
    ({"LCM_LEAF_TARGET_RATIO": "1.5"}, "leaf_target_ratio", 0.20),
    ({"LCM_LEAF_TARGET_RATIO": "nan"}, "leaf_target_ratio", 0.20),
    ({"LCM_LEAF_TARGET_RATIO": "abc"}, "leaf_target_ratio", 0.20),
    ({"LCM_LEAF_TARGET_MIN_TOKENS": "0"}, "leaf_target_min_tokens", 2000),
    ({"LCM_LEAF_TARGET_MIN_TOKENS": "3000", "LCM_LEAF_TARGET_MAX_TOKENS": "2500"}, "leaf_target_max_tokens", 12_000),
    ({"LCM_LEAF_TARGET_MIN_TOKENS": "20000"}, "leaf_target_min_tokens", 2000),
])
def test_t3_invalid_values_fall_back_to_the_default_with_a_warning(monkeypatch, env, field, default):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    c = LCMConfig.from_env()
    assert getattr(c, field) == default
    assert c.config_sources[field] == "default"
    assert any(field.upper() in warning for warning in c.config_source_warnings)
    assert c.leaf_target_min_tokens <= c.leaf_target_max_tokens


def test_t3_min_above_max_keeps_a_valid_env_min(monkeypatch):
    monkeypatch.setenv("LCM_LEAF_TARGET_MIN_TOKENS", "3000")
    monkeypatch.setenv("LCM_LEAF_TARGET_MAX_TOKENS", "2500")
    c = LCMConfig.from_env()
    assert (c.leaf_target_min_tokens, c.leaf_target_max_tokens) == (3000, 12_000)


# -- T4: max_tokens follows the target -------------------------------------------------------------------------

def test_t4_l1_max_tokens_is_twice_the_configured_target(tmp_path, monkeypatch):
    calls: list[int] = []

    def chain(prompt, max_tokens, **kwargs):
        calls.append(int(max_tokens))
        return ACCEPTED

    monkeypatch.setattr(escalation, "_invoke_summary_llm_chain", chain)
    engine = LCMEngine(config=LCMConfig(
        leaf_target_min_tokens=2400, leaf_target_max_tokens=2400, database_path=str(tmp_path / "lcm.db")))
    try:
        chunk = [{"role": "user", "content": "[T0] user turn" + PAD * 5_000}]
        _, source_tokens, summary, level, _ = engine._summarize_leaf_chunk_with_rescue(chunk)
        assert source_tokens > 20_000 and (summary, level) == (ACCEPTED, 1)
        assert calls == [4800]
    finally:
        engine.shutdown()


# -- T5: a default compaction produces the base numbers --------------------------------------------------------

def _compaction_fixture(tmp_path: Path, monkeypatch) -> dict:
    budgets: list[list[int]] = []
    max_tokens_seen: list[int] = []
    real = escalation.summarize_with_escalation

    def spy(*args, **kwargs):
        budgets.append([int(kwargs["source_tokens"]), int(kwargs["token_budget"])])
        return real(*args, **kwargs)

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        max_tokens_seen.append(int(max_tokens))
        return ACCEPTED

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", spy)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)

    view = [{"role": "system", "content": "system prompt"}]
    for i, reps in enumerate((1_500, 5_000, 16_000)):
        view.append({"role": "user", "content": f"[T{i}] user turn" + PAD * reps, "timestamp": 10.0 * (i + 1)})
        view.append({"role": "assistant", "content": f"reply to T{i}", "timestamp": 10.0 * (i + 1) + 1})
    view.append({"role": "user", "content": "[tail] latest", "timestamp": 100.0})
    view.append({"role": "assistant", "content": "tail reply", "timestamp": 101.0})

    engine = LCMEngine(config=LCMConfig(
        fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
        threshold_full_sweep_enabled=True, max_assembly_tokens=1_000_000,
        database_path=str(tmp_path / "lcm.db"),
        # #1013 part B: fixture covers the feature-off mechanism
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False,
    ))
    engine.on_session_start("S", platform="telegram", context_length=2_000_000, conversation_id="conv")
    try:
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        leaves = sorted(
            (int(n.source_token_count), int(n.token_count))
            for n in engine._dag.get_session_nodes("S") if int(n.depth) == 0)
    finally:
        engine.shutdown()
    return {"budgets": budgets, "l1_max_tokens": max_tokens_seen, "leaf_nodes": leaves}


def test_t5_default_compaction_reproduces_the_base_targets(tmp_path, monkeypatch):
    got = _compaction_fixture(tmp_path, monkeypatch)
    assert got["budgets"] == BASE_LEAF_BUDGETS
    assert got["l1_max_tokens"] == BASE_MAX_TOKENS
    assert got["leaf_nodes"] == BASE_LEAF_NODES


def test_t5_default_env_config_equals_the_constructor_defaults():
    c = LCMConfig.from_env()
    d = LCMConfig()
    assert (c.leaf_target_ratio, c.leaf_target_min_tokens, c.leaf_target_max_tokens) == (
        d.leaf_target_ratio, d.leaf_target_min_tokens, d.leaf_target_max_tokens)


def test_t6_config_without_the_614_fields_keeps_the_historical_targets():
    """A config object built before the knobs existed (no leaf_target_* attributes) must not raise."""
    current = LCMConfig()
    old = SimpleNamespace(**{
        name: getattr(current, name) for name in vars(current) if not name.startswith("leaf_target_")})
    assert not hasattr(old, "leaf_target_ratio")
    engine = SimpleNamespace(_config=old)
    assert [LCMEngine._leaf_target_tokens(engine, n) for n in (8_000, 20_000, 100_000)] == [
        2_000, 4_000, 12_000]
