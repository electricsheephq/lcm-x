"""Turn-start deferral uses the issue-2863 loop and a real LCM engine."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_lcm import compaction, survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from tests import test_issue_2863_host_contract as model
from tests.test_host_capability import _load_plugin_module

issue_2863_engine = model.engine

ENV = "LCM_TURN_START_DEFER_CEILING"
WINDOW = 1_000_000
TRIGGER = 500_000
CEILING = 700_000


@pytest.mark.parametrize("raw", [None, "0"])
def test_off_matches_base_host_gate_and_request_trace(issue_2863_engine, monkeypatch, raw):
    engine = issue_2863_engine
    if raw is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, raw)
    config = LCMConfig.from_env()
    # Base does not have the field. The same Host inputs capture its unmodified decisions.
    if hasattr(config, "turn_start_defer_ceiling"):
        engine._config.turn_start_defer_ceiling = config.turn_start_defer_ceiling
    assert engine.should_compress(engine.threshold_tokens)
    host = model.Host(engine, model.R2_STEPS, model.ModelClock(monkeypatch))
    host.run(start_pressure=1)
    requests = [(r["req"], r["provider"], r["newest_stub"] is not None) for r in host.requests]
    replay = json.dumps([host.gates, requests, host.verdicts], sort_keys=True)
    assert hashlib.sha256(replay.encode()).hexdigest() == (
        "0ad7cec318d9358a25f68b9812bc6511e4c959070c54c2a237a31c4f61eba4e7"
    )


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    model.force_char_estimate(monkeypatch)
    monkeypatch.setattr(compaction, "_host_has_post_tool_reanchor", lambda: True)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    engines = []

    def create(ceiling=0.70, *, window=WINDOW, **options):
        root = tmp_path / str(len(engines))
        root.mkdir()
        config = LCMConfig(
            database_path=str(root / "lcm.db"), context_threshold=0.5,
            turn_start_defer_ceiling=ceiling, **options,
        )
        eng = LCMEngine(config=config, hermes_home=str(root))
        engines.append(eng)
        eng.on_session_start("session", platform="cli", context_length=window)
        eng.update_model("probe-model", window, provider="custom")
        monkeypatch.setattr(eng, "_summarize_leaf_chunk_with_rescue", model.stub_summarizer)
        assert eng.threshold_tokens == min(window // 2, options.get("max_assembly_tokens", window))
        return eng

    yield create
    for eng in engines:
        eng.shutdown()


def history():
    return [
        {"role": role, "content": (f"Durable turn {i} {role} details. " * 500).strip()}
        for i in range(6)
        for role in ("user", "assistant")
    ]


def pressure(messages):
    return int(model.OVERHEAD + sum(model.real_msg_tokens(m) for m in messages))


def test_short_turns_commit_at_the_very_next_start(make_engine, monkeypatch):
    eng = make_engine()
    clock = model.ModelClock(monkeypatch)
    monkeypatch.setattr(model, "OVERHEAD", 477_000)
    monkeypatch.setattr(model, "ROUGH_OVERHEAD", 420_000)
    messages = history()
    start_pressure = TRIGGER - 1
    for turn in range(4):
        messages += [{"role": "user", "content": f"Turn {turn}. " + "New context. " * 6000}]
        host = model.Host(eng, ["S"], clock)
        requests = host.run(messages, start_pressure=start_pressure)
        assert len(requests) == 2
        assert host.gates[0] == ("turn_start", start_pressure, turn > 0)
        assert [v[1] for v in host.verdicts] == (["committed"] if turn > 0 else [])
        if turn > 0:
            assert host.verdicts[0][2] and eng._dag.get_session_nodes(eng._session_id)
        post_tool = [(tokens, verdict) for phase, tokens, verdict in host.gates if phase == "post_tool"]
        assert len(post_tool) == 1
        assert TRIGGER < post_tool[0][0] < CEILING
        assert post_tool[0][1] is False
        assert requests[-1]["provider"] == pressure(host.messages)
        messages = host.messages + [{"role": "assistant", "content": f"Completed turn {turn}."}]
        start_pressure = pressure(messages)
        eng.note_turn_complete()


def test_long_tool_turn_defers_until_ceiling_then_commits_and_fits(make_engine, monkeypatch):
    eng = make_engine()
    clock = model.ModelClock(monkeypatch)
    monkeypatch.setattr(model, "OVERHEAD", 575_000)
    monkeypatch.setattr(model, "ROUGH_OVERHEAD", 400_000)
    step_rows = model.step_rows

    def large_second_step(number, kind):
        rows = step_rows(number, kind)
        if number == 2:
            rows[-1]["content"] = "Synthetic tool output. " * 24_000
        return rows

    monkeypatch.setattr(model, "step_rows", large_second_step)
    host = model.Host(eng, ["S", "S"], clock)
    requests = host.run(history() + [dict(model.USER)], start_pressure=TRIGGER - 1)
    mid_gates = [(tokens, verdict) for phase, tokens, verdict in host.gates if phase == "post_tool"]
    assert len(mid_gates) == 2
    assert TRIGGER < mid_gates[0][0] < CEILING and mid_gates[0][1] is False
    assert mid_gates[1][0] >= CEILING and mid_gates[1][1] is True
    assert host.verdicts[0][:2] == (3, "committed")
    assert len(requests) == 3
    assert all(r["provider"] < WINDOW for r in requests)
    assert all(not p["pairing"] and not p["conservation"] for p in host.passes)


@pytest.mark.parametrize("boundary", ["restart", "interrupt", "switch", "reset", "complete", "unknown_response"])
def test_turn_phase_fails_open(make_engine, monkeypatch, boundary):
    eng = make_engine()
    # Keep the interruption exercise outside the stopped-tail ingest path.
    if boundary == "interrupt":
        monkeypatch.setattr(eng, "note_turn_complete", lambda: None)
    eng.update_from_response({"stop_reason": "tool_use"})
    assert eng.should_compress(TRIGGER) is False
    if boundary == "restart":
        eng = make_engine()
    elif boundary == "interrupt":
        eng.on_turn_complete([], interrupted=True)
    elif boundary == "switch":
        eng.on_session_start("another", platform="cli", context_length=WINDOW)
    elif boundary == "reset":
        eng.on_session_reset()
    elif boundary == "complete":
        eng.note_turn_complete()
    else:
        eng.update_from_response({"prompt_tokens": TRIGGER})
    assert eng.should_compress(TRIGGER) is True


def test_compression_rotation_keeps_phase_and_equal_ceiling_passes(make_engine):
    eng = make_engine()
    eng.update_from_response({"stop_reason": "tool_use"})
    eng.on_session_start("rotated", boundary_reason="compression", old_session_id="session", platform="cli")
    assert eng.should_compress(CEILING - 1) is False
    assert eng.should_compress(CEILING) is True


@pytest.mark.parametrize("setting", ["0", "0.70"])
def test_pinned_host_response_hook_tracks_only_its_turn(tmp_path, monkeypatch, setting):
    plugin = _load_plugin_module("turn_start_defer_hook")
    monkeypatch.setenv(ENV, setting)
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.50")
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "hook.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")

    current = SimpleNamespace(turn_id="turn-1", closed=False)
    relay = SimpleNamespace(current_turn=lambda: current)
    monkeypatch.setitem(sys.modules, "agent.relay_runtime", relay)

    ctx = SimpleNamespace(hooks={})
    ctx.register_hook = lambda name, hook: ctx.hooks.__setitem__(name, hook)
    ctx.register_context_engine = lambda engine: setattr(ctx, "engine", engine)
    plugin.register(ctx)
    eng = ctx.engine.clone_for_agent()
    try:
        eng.on_session_start("hook-session", platform="cli", context_length=WINDOW)
        eng.update_from_response({"prompt_tokens": TRIGGER})
        if setting == "0":
            assert "post_api_request" not in ctx.hooks
            assert eng.should_compress(TRIGGER) is True
            return

        hook = ctx.hooks["post_api_request"]
        hook(session_id="foreign", turn_id=current.turn_id, finish_reason="tool_calls")
        assert eng.should_compress(TRIGGER) is True
        hook(session_id="hook-session", turn_id=current.turn_id, finish_reason="tool_calls")
        assert eng.should_compress(TRIGGER) is False
        assert eng.last_prompt_tokens == TRIGGER

        # Simulate an early abort without either turn-complete callback or a new response.
        current.turn_id = "turn-2"
        assert eng.should_compress(TRIGGER) is True
        hook(session_id="hook-session", turn_id="turn-1", finish_reason="tool_calls")
        assert eng.should_compress(TRIGGER) is True
    finally:
        eng.shutdown()
        ctx.engine.shutdown()


def test_forced_overflow_matches_off_below_the_defer_ceiling(make_engine):
    cap = 105_000
    messages = history() + [{"role": "user", "content": "Newest unique input. " * 13_500}]
    assert model.count_messages_tokens(messages) > cap
    results = []
    for setting in (0, 0.70):
        eng = make_engine(setting, max_assembly_tokens=cap)
        eng.update_from_response({"stop_reason": "tool_use"})
        assert eng.should_compress(cap) is True
        result = eng.compress(messages, current_tokens=cap)
        assert model.count_messages_tokens(result) <= cap
        results.append(result)
    assert results[0] == results[1]


def test_survival_fit_matches_off_above_the_defer_ceiling(make_engine):
    window = 64_000
    messages = [{"role": "user", "content": "Newest unique input. " * 13_000}]
    observed = model.count_messages_tokens(messages)
    assert observed > window
    results = []
    for setting in (0, 0.70):
        eng = make_engine(setting, window=window, large_output_externalization_enabled=False)
        eng.update_from_response({"stop_reason": "tool_use"})
        assert eng.should_compress(observed) is True
        result = eng.compress(messages, current_tokens=observed)
        assert eng._last_survival_fit
        assert model.count_messages_tokens(result) < window
        results.append(result)
    assert results[0] == results[1]


@pytest.mark.parametrize(("raw", "reserve"), [
    ("", "0.25"), ("oops", "0.25"), ("nan", "0.25"), ("inf", "0.25"), ("-inf", "0.25"),
    ("-0.1", "0.25"), ("0.75", "0.25"), ("1", "0.25"), ("0.30", "0.70"),
])
def test_invalid_env_ceiling_is_disabled_with_one_warning(monkeypatch, raw, reserve):
    monkeypatch.setenv(ENV, raw)
    monkeypatch.setenv("LCM_SURVIVAL_RESERVE", reserve)
    cfg = LCMConfig.from_env()
    assert cfg.turn_start_defer_ceiling == 0
    warnings = [w for w in cfg.config_source_warnings if ENV in w]
    assert len(warnings) == 1
    assert repr(raw) in warnings[0]


def test_valid_env_ceiling_tracks_source_and_reserve(monkeypatch):
    monkeypatch.setenv(ENV, "0.70")
    monkeypatch.setenv("LCM_SURVIVAL_RESERVE", "0.25")
    cfg = LCMConfig.from_env()
    assert cfg.turn_start_defer_ceiling == 0.70
    assert cfg.config_sources["turn_start_defer_ceiling"] == f"env:{ENV}"
    assert not any(ENV in w for w in cfg.config_source_warnings)


@pytest.mark.parametrize("value", [True, None, "invalid", 0.85])
def test_invalid_direct_ceiling_is_disabled_with_one_warning(value):
    cfg = LCMConfig(turn_start_defer_ceiling=value)
    assert cfg.turn_start_defer_ceiling == 0
    assert len(cfg.config_source_warnings) == 1
    assert f"turn_start_defer_ceiling={value!r}" in cfg.config_source_warnings[0]
