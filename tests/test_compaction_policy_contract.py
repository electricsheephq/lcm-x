"""Pin policy results and observable setter ordering before extraction."""

import logging
import random
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


FIELDS = (
    "raw_context_length", "context_length", "effective_context_length_cap",
    "effective_context_length_reason", "_context_length_source",
    "context_threshold", "_context_threshold_source", "_context_threshold_autoraised",
    "threshold_percent", "threshold_tokens",
)
SOURCES = ("manual_or_default", "env:LCM_CONTEXT_THRESHOLD",
           "config_yaml:lcm.context_threshold", "config_yaml:compression.threshold (floored)")
ROUTES = ((None, None), ("unknown", "custom"), ("glm-5.2-1M", "custom"),
          ("gpt-5.5", "openai-codex"), ("gpt-5.5", "openai"),
          ("gpt-5.6", "openai-codex"), ("openai/gpt-5.5-sol", " OPENAI-CODEX "))


class TracedEngine(LCMEngine):
    def __setattr__(self, name, value):
        if name in FIELDS and "events" in self.__dict__:
            self.events.append(("write", name, value))
        super().__setattr__(name, value)


def make_engine(ratio=0.35, source=SOURCES[0], enabled=True, maximum=0, reserve=0):
    engine = TracedEngine.__new__(TracedEngine)
    engine._config = LCMConfig(
        context_threshold=ratio, config_sources={"context_threshold": source},
        codex_gpt55_autoraise_enabled=enabled, max_assembly_tokens=maximum,
        reserve_tokens_floor=reserve,
        model_thresholds={"glm-5.2": 0.7, "glm-5.2-1M": 0.25},
    )
    engine.model, engine.provider = "gpt-5.5", "openai-codex"
    for field in FIELDS:
        setattr(engine, field, "untouched")
    engine.events = []
    return engine


def expected(engine, raw, model, provider, absolute):
    config = engine._config
    try:
        window = int(raw)
    except (TypeError, ValueError):
        return None
    window = max(0, window)
    model = engine.model if model is None else model
    provider = engine.provider if provider is None else provider
    ratio = float(config.context_threshold)
    source = config.config_sources.get("context_threshold", "manual_or_default")
    notice = None
    cap = 272_000 if provider.strip().lower() == "openai-codex" and "gpt-5." in model else None
    effective = min(window, cap) if cap else window
    capped = cap if cap and window > cap else None
    if "glm-5.2-1M" in model:
        ratio, source = 0.25, "model_thresholds:glm-5.2-1M"
    elif "glm-5.2" in model:
        ratio, source = 0.7, "model_thresholds:glm-5.2"
    elif ("gpt-5.5" in model and provider.strip().lower() == "openai-codex"
          and config.codex_gpt55_autoraise_enabled and ratio < 0.85
          and source not in SOURCES[1:3]):
        ratio, source = 0.85, "codex_gpt55_autoraise"
    if source.startswith("model_thresholds:") or source == "codex_gpt55_autoraise":
        notice = {"from": float(config.context_threshold), "to": ratio}
    caps = [c for c in (config.max_assembly_tokens,
                       effective - config.reserve_tokens_floor
                       if effective > 0 and config.reserve_tokens_floor > 0 else 0) if c > 0]
    tokens = int(effective * ratio)
    if caps:
        tokens = min(tokens, min(caps)) if tokens > 0 else min(caps)
    try:
        absolute = int(absolute or 0)
    except ValueError:
        absolute = 0
    disable = window > 0 and absolute > 0
    tokens = absolute if disable else tokens if window else 0
    return dict(zip(FIELDS, (window, effective, capped,
                            "codex_oauth_context_cap" if capped else "", "contract",
                            ratio, source, notice, ratio, tokens))), disable


@pytest.mark.parametrize("raw", [None, "abc", 0, -1, 1, 100, 272_000, 400_000])
@pytest.mark.parametrize("absolute", [None, "", "0", "abc", "130000"])
def test_setter_characterization(raw, absolute, monkeypatch, caplog):
    rng = random.Random(553)
    caplog.set_level(logging.DEBUG, logger="hermes_lcm.engine")
    for source in SOURCES:
        for model, provider in ROUTES:
            engine = make_engine(rng.choice([0.35, 0.85, 0.95]), source,
                                 rng.choice([False, True]), rng.choice([0, 80]),
                                 rng.choice([0, 10, 100, 500_000]))
            if absolute is None:
                monkeypatch.delenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", raising=False)
            else:
                monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", absolute)
            result = expected(engine, raw, model, provider, absolute)
            initial_enabled = engine._config.codex_gpt55_autoraise_enabled
            caplog.clear()
            actual = engine._set_context_length(raw, source="contract", model=model, provider=provider)
            records = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
            if result is None:
                assert actual is False
                assert all(getattr(engine, f) == "untouched" for f in FIELDS)
                assert engine.events == []
                assert records == [("hermes_lcm.engine", "DEBUG",
                                    f"LCM ignored invalid contract context_length: {raw!r}")]
            else:
                values, disable = result
                assert actual is True
                assert {f: getattr(engine, f) for f in FIELDS} == values
                names = list(FIELDS)
                if int(raw) <= 0:
                    names.insert(5, names.pop())
                    wanted_logs = [("hermes_lcm.engine", "DEBUG",
                                    f"LCM cleared non-positive contract context_length: {raw!r}")]
                else:
                    wanted_logs = []
                    if engine._config.reserve_tokens_floor >= values["context_length"]:
                        wanted_logs = [("hermes_lcm.engine", "WARNING",
                                        f"LCM reserve_tokens_floor={engine._config.reserve_tokens_floor} "
                                        "disables reserve-based assembly cap because "
                                        f"context_length={values['context_length']}")]
                    if disable:
                        names.append("threshold_tokens")
                assert [event[1] for event in engine.events] == names
                assert records == wanted_logs
                assert engine._config.codex_gpt55_autoraise_enabled == (initial_enabled and not disable)


def test_absolute_override_is_live_and_autoraise_mutation_sticks(monkeypatch):
    engine = make_engine()
    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "130000")
    assert engine._set_context_length(400_000, source="contract")
    assert (engine.context_threshold, engine.threshold_tokens) == (0.85, 130_000)
    monkeypatch.delenv("LCM_ABSOLUTE_THRESHOLD_TOKENS")
    assert engine._set_context_length(400_000, source="contract")
    assert (engine.context_threshold, engine.threshold_tokens) == (0.35, 95_200)
    assert engine._config.codex_gpt55_autoraise_enabled is False


def test_warning_and_env_read_keep_their_place_in_attribute_writes(monkeypatch):
    import hermes_lcm.engine as module

    engine = make_engine(reserve=500_000)
    original_get = module.os.environ.get

    def get(key, default=None):
        if key == "LCM_ABSOLUTE_THRESHOLD_TOKENS":
            engine.events.append(("env", key))
            return "130000"
        return original_get(key, default)

    monkeypatch.setattr(module.os.environ, "get", get)
    monkeypatch.setattr(module.logger, "warning", lambda *args: engine.events.append(("warning",)))
    class Config(SimpleNamespace):
        def __setattr__(self, name, value):
            engine.events.append(("config", name, value))
            super().__setattr__(name, value)
    engine._config = Config(**vars(engine._config))
    assert engine._set_context_length(400_000, source="contract")
    assert [event[0] for event in engine.events] == ["write"] * 9 + [
        "warning", "write", "env", "write", "config",
    ]
    engine.events.clear()
    assert engine._set_context_length(0, source="contract")
    assert all(event[0] == "write" for event in engine.events)


def test_assembly_shell_warns_once_per_call(caplog):
    engine = make_engine(maximum=50, reserve=100)
    engine.context_length = 100
    for _ in range(3):
        assert engine._effective_assembly_token_cap() == 50
    assert len(caplog.records) == 3
    engine.context_length = 0
    assert engine._effective_assembly_token_cap() == 50
    assert len(caplog.records) == 3
