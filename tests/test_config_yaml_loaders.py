"""#1916: config values agree with YAML even on minimal Hermes installs."""

from types import SimpleNamespace

import pytest

import hermes_lcm.config as config


@pytest.fixture
def yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(r'''lcm:
  query: "### Query: "
  model: "provider/model#v2"
  single: 'single # text'
  quoted_bool: "true"
  quoted_float: "1.10"
  quoted_zero: "0"
  enabled: true
  count: 7
  ratio: 0.25
  items: [a, b, "c d", "e,f", 'g#h', true, 2, 0.5, "0"] # list note
  quoted_comment: "keep # this" # discard this
  plain_comment: query text # discard this
  escaped: "say \"hi\"\nnext\tline\\end"
  apostrophe: 'it''s # intact'
  empty_list: []
  nested:
    model_thresholds:
      "": 0.5
      valid: 0.4
''')
    return path


def _typed(value):
    """Unlike ordinary equality, preserve bool/int and nested scalar types."""
    kind = type(value).__name__
    if isinstance(value, dict):
        value = [(_typed(key), _typed(item)) for key, item in value.items()]
    elif isinstance(value, list):
        value = [_typed(item) for item in value]
    else:
        return type(value).__name__, value
    return kind, value


@pytest.fixture(params=["pyyaml", "ruamel", "line"])
def selected_loader(request, monkeypatch):
    loader = request.param
    monkeypatch.setattr(config, "yaml", pytest.importorskip("yaml") if loader == "pyyaml" else None)
    monkeypatch.setattr(
        config, "ruamel_yaml", pytest.importorskip("ruamel.yaml") if loader == "ruamel" else None
    )
    return loader


@pytest.mark.parametrize("loader", ["pyyaml", "ruamel", "line"])
def test_shared_fixture_loader_parity(yaml_config, monkeypatch, loader):
    # Each optional loader skips independently; the line-parser case always runs.
    if loader == "pyyaml":
        monkeypatch.setattr(config, "yaml", pytest.importorskip("yaml"))
    else:
        monkeypatch.setattr(config, "yaml", None)
        ruamel = pytest.importorskip("ruamel.yaml") if loader == "ruamel" else None
        monkeypatch.setattr(config, "ruamel_yaml", ruamel, raising=False)
    assert _typed(config._load_hermes_config_yaml(yaml_config)) == _typed({
        "lcm": {
            "query": "### Query: ", "model": "provider/model#v2",
            "single": "single # text", "quoted_bool": "true",
            "quoted_float": "1.10", "quoted_zero": "0", "enabled": True,
            "count": 7, "ratio": 0.25,
            "items": ["a", "b", "c d", "e,f", "g#h", True, 2, 0.5, "0"],
            "quoted_comment": "keep # this", "plain_comment": "query text",
            "escaped": 'say "hi"\nnext\tline\\end', "apostrophe": "it's # intact",
            "empty_list": [], "nested": {"model_thresholds": {"": 0.5, "valid": 0.4}},
        }
    })


def test_ruamel_safe_loader_precedes_line_parser(yaml_config, monkeypatch):
    ruamel = pytest.importorskip("ruamel.yaml")
    calls = []

    def safe_loader(**kwargs):
        calls.append(kwargs)
        return ruamel.YAML(**kwargs)

    # Block scalars are outside the line parser's subset and prove routing.
    yaml_config.write_text('lcm:\n  query: |\n    query # intact\n')
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", SimpleNamespace(YAML=safe_loader), raising=False)
    assert config._load_hermes_config_yaml(yaml_config) == {"lcm": {"query": "query # intact\n"}}
    assert calls == [{"typ": "safe", "pure": True}]


@pytest.mark.parametrize("loader", ["pyyaml", "ruamel"])
def test_loader_error_returns_empty_mapping(yaml_config, monkeypatch, loader):
    def fail(*args, **kwargs):
        raise ValueError("invalid YAML")

    monkeypatch.setattr(config, "yaml", SimpleNamespace(safe_load=fail) if loader == "pyyaml" else None)
    monkeypatch.setattr(config, "ruamel_yaml", SimpleNamespace(YAML=fail), raising=False)
    assert config._load_hermes_config_yaml(yaml_config) == {}



def test_ruamel_with_clib_accepts_url_flow_map(yaml_config, monkeypatch):
    ruamel = pytest.importorskip("ruamel.yaml")
    pytest.importorskip("_ruamel_yaml")
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", ruamel)
    monkeypatch.setattr(config, "_hermes_config_path", lambda: yaml_config)
    yaml_config.write_text(
        "custom_providers: [{name: local, base_url: http://localhost:11434/v1}]\n"
        "compression:\n  codex_gpt55_autoraise: false\n"
        "lcm:\n  context_threshold: 0.4\n"
    )
    assert config._hermes_codex_gpt55_autoraise_with_source(True) == (
        False, "config_yaml:compression.codex_gpt55_autoraise"
    )
    assert config._hermes_compression_threshold_with_source(0.75) == (
        0.4, "config_yaml:lcm.context_threshold"
    )


@pytest.mark.parametrize("text", [
    "lcm:\n  - a # see: x\n",
    "lcm:\n  notes: one\n    three # see: docs\n",
    "--- # doc: 1\nlcm:\n  notes: one\n",
])
def test_comments_do_not_invent_ignored_keys(yaml_config, monkeypatch, text):
    pyyaml = pytest.importorskip("yaml")
    yaml_config.write_text(text)
    expected = config._ignored_lcm_config_yaml_keys(pyyaml.safe_load(text))
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", None)
    loaded = config._load_hermes_config_yaml(yaml_config)
    assert config._ignored_lcm_config_yaml_keys(loaded) == expected
    assert set(loaded) == {"lcm"}


@pytest.mark.parametrize("token,enabled", [
    ("off", False), ("no", False), ("Off", False), ("false", False),
    ("yes", True), ("on", True), ("true", True),
])
def test_compression_enabled_consumer(yaml_config, monkeypatch, selected_loader, token, enabled):
    yaml_config.write_text(f"compression:\n  enabled: {token}\n  threshold: 0.9\n")
    monkeypatch.setattr(config, "_hermes_config_path", lambda: yaml_config)
    assert config._hermes_compression_threshold_with_source(0.75) == (
        (0.9, "config_yaml:compression.threshold") if enabled else (0.75, "default")
    )


@pytest.mark.parametrize("token", ["off", "no", "false"])
def test_autoraise_disabled_consumer(yaml_config, monkeypatch, selected_loader, token):
    yaml_config.write_text(f"compression:\n  codex_gpt55_autoraise: {token}\n")
    monkeypatch.setattr(config, "_hermes_config_path", lambda: yaml_config)
    assert config._hermes_codex_gpt55_autoraise_with_source(True) == (
        False, "config_yaml:compression.codex_gpt55_autoraise"
    )


@pytest.mark.parametrize("text,expected", [
    ("items: [model#v2, other]\n", {"items": ["model#v2", "other"]}),
    ("summary_model: foo#bar\n", {"summary_model": "foo#bar"}),
    ("x: a #c\n", {"x": "a"}),
    ("y: [a, b] # c\n", {"y": ["a", "b"]}),
    ("z: http://h/#frag\n", {"z": "http://h/#frag"}),
])
def test_unquoted_hash_comment_boundary(yaml_config, selected_loader, text, expected):
    yaml_config.write_text(text)
    assert _typed(config._load_hermes_config_yaml(yaml_config)) == _typed(expected)
    # Keep the forced-line cases runnable even without either optional library.
    try:
        import yaml
    except ImportError:
        return
    assert _typed(expected) == _typed(yaml.safe_load(text))



def test_recursive_alias_preserves_other_settings(yaml_config, selected_loader):
    yaml_config.write_text("extra: &loop [*loop]\nlcm: {context_threshold: 0.35}\n")
    loaded = config._load_hermes_config_yaml(yaml_config)
    assert type(loaded) is dict
    if selected_loader != "line":
        assert loaded["lcm"]["context_threshold"] == 0.35
        assert type(loaded["extra"]) is list
        assert loaded["extra"][0] is loaded["extra"]


@pytest.mark.parametrize("anchor", ["[one, two]", "{key: value}"])
def test_ruamel_shared_alias_identity(yaml_config, monkeypatch, anchor):
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", pytest.importorskip("ruamel.yaml"))
    yaml_config.write_text(f"first: &shared {anchor}\nsecond: *shared\n")
    loaded = config._load_hermes_config_yaml(yaml_config)
    assert loaded["first"] is loaded["second"]
    assert type(loaded["first"]) is (list if anchor.startswith("[") else dict)


def test_ruamel_alias_doubling_normalizes_within_budget(yaml_config, monkeypatch):
    from time import perf_counter

    ruamel = pytest.importorskip("ruamel.yaml")
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", ruamel)
    text = "level0: &a0 [leaf]\n" + "".join(
        f"level{i}: &a{i} [*a{i-1}, *a{i-1}]\n" for i in range(1, 7)
    )
    yaml_config.write_text(text)
    raw = ruamel.YAML(typ="safe", pure=True).load(text)
    started = perf_counter()
    normalized = config._plain_yaml_value(raw)
    elapsed = perf_counter() - started
    assert elapsed < 0.1
    loaded = config._load_hermes_config_yaml(yaml_config)
    for document in (normalized, loaded):
        for level in range(1, 7):
            assert document[f"level{level}"][0] is document[f"level{level-1}"]
            assert document[f"level{level}"][1] is document[f"level{level-1}"]
