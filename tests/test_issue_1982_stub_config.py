"""Replay thresholds accept YAML and environment overrides with safe floors."""

import pytest

import hermes_lcm.config as config


@pytest.mark.parametrize(("raw", "first", "aged"), [
    (None, 10_000, 2_000), (500, 500, 500), (3_000, 3_000, 3_000),
    (0, 10_000, 0), (499, 10_000, 2_000), (-1, 10_000, 2_000),
    (True, 10_000, 2_000), ("invalid", 10_000, 2_000),
])
@pytest.mark.parametrize("source", ["yaml", "env"])
def test_stub_threshold_sources_and_floors(monkeypatch, raw, first, aged, source):
    names = ["large_output_active_replay_stub_threshold_tokens", "large_output_active_replay_stub_aged_threshold_tokens"]
    monkeypatch.setattr(config, "lcm_env", lambda: config.os.environ)
    values = {} if raw is None else dict.fromkeys(names, raw)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {"lcm": values})
    for name in names:
        key = "LCM_" + name.upper()
        monkeypatch.delenv(key, raising=False)
        if source == "env" and raw is not None:
            monkeypatch.setenv(key, str(raw))
    cfg = config.LCMConfig.from_env()
    assert [getattr(cfg, name) for name in names] == [first, aged]
    assert not set(names) & set(cfg.ignored_config_yaml_lcm_keys)
    assert bool(cfg.config_source_warnings) is (raw in (499, -1, True, "invalid") or raw == 0)
    for name in names:
        monkeypatch.setenv("LCM_" + name.upper(), "4000")
    cfg = config.LCMConfig.from_env()
    assert [getattr(cfg, name) for name in names] == [4_000, 4_000]
