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


@pytest.mark.parametrize("loader", ["pyyaml", "ruamel", "line"])
def test_shared_fixture_loader_parity(yaml_config, monkeypatch, loader):
    # Each optional loader skips independently; the line-parser case always runs.
    if loader == "pyyaml":
        monkeypatch.setattr(config, "yaml", pytest.importorskip("yaml"))
    else:
        monkeypatch.setattr(config, "yaml", None)
        ruamel = pytest.importorskip("ruamel.yaml") if loader == "ruamel" else None
        monkeypatch.setattr(config, "ruamel_yaml", ruamel, raising=False)
    assert config._load_hermes_config_yaml(yaml_config) == {
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
    }


def test_ruamel_safe_loader_precedes_line_parser(yaml_config, monkeypatch):
    ruamel = pytest.importorskip("ruamel.yaml")
    calls = []

    def safe_loader(*, typ):
        calls.append(typ)
        return ruamel.YAML(typ=typ)

    # Block scalars are outside the line parser's subset and prove routing.
    yaml_config.write_text('lcm:\n  query: |\n    query # intact\n')
    monkeypatch.setattr(config, "yaml", None)
    monkeypatch.setattr(config, "ruamel_yaml", SimpleNamespace(YAML=safe_loader), raising=False)
    assert config._load_hermes_config_yaml(yaml_config) == {"lcm": {"query": "query # intact\n"}}
    assert calls == ["safe"]


@pytest.mark.parametrize("loader", ["pyyaml", "ruamel"])
def test_loader_error_returns_empty_mapping(yaml_config, monkeypatch, loader):
    def fail(*args, **kwargs):
        raise ValueError("invalid YAML")

    monkeypatch.setattr(config, "yaml", SimpleNamespace(safe_load=fail) if loader == "pyyaml" else None)
    monkeypatch.setattr(config, "ruamel_yaml", SimpleNamespace(YAML=fail), raising=False)
    assert config._load_hermes_config_yaml(yaml_config) == {}
