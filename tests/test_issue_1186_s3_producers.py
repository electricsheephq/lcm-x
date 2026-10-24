"""Release producers must measure and record the selected tree's configuration."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import config as config_module

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"


@pytest.fixture
def bridge_module():
    original = sys.stdout
    try:
        spec = importlib.util.spec_from_file_location("s3_bridge", BRIDGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.stdout = original
    return module


@pytest.mark.parametrize("enabled", [False, True])
def test_bridge_reads_tree_default_and_records_it(bridge_module, tmp_path, monkeypatch, enabled):
    original = config_module.LCMConfig
    def tree_config(**kwargs):
        return original(**{"embeddings_enabled": enabled, "embedding_provider": "fastembed",
                           "embedding_model": "tree-specific-model", **kwargs})
    monkeypatch.setattr(config_module, "LCMConfig", tree_config)
    monkeypatch.setenv("HERMES_LCM_REPO", str(ROOT))
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path))
    monkeypatch.delenv("HERMES_MB_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_MB_MODEL", raising=False)
    bridge = bridge_module.Bridge()
    calls = []
    bridge._resolve_harness_provider = lambda provider, model: calls.append((provider, model)) or SimpleNamespace(dim=2, model_id=model)
    result = bridge.initialize({})
    assert result["embeddings_enabled"] is enabled
    assert result["model"] == "tree-specific-model"
    assert calls == ([("fastembed", "tree-specific-model")] if enabled else [])
    assert json.loads((tmp_path / "effective-config.json").read_text())["model"] == result["model"]
    assert bridge._config(tmp_path / "store.db").embeddings_enabled is enabled
    if not enabled:
        response = bridge.ingest({"containerTag": "fixture", "session": {
            "sessionId": "old-session", "messages": [{"role": "user", "content": "cobalt memory"}],
        }})
        assert response["ok"]
        result = bridge.search({"containerTag": "fixture", "query": "cobalt"})
        assert result["results"]


def test_qa_caller_does_not_invent_an_explicit_arm():
    source = (BRIDGE.parents[1] / "index.ts").read_text()
    assert 'HERMES_MB_PROVIDER: process.env.HERMES_MB_PROVIDER || "fastembed"' not in source


@pytest.mark.parametrize("enabled", [True, False])
def test_longmemeval_shipped_default_records_effective_model(tmp_path, monkeypatch, enabled):
    import benchmarking.longmemeval as lme
    original = config_module.LCMConfig
    monkeypatch.setattr(config_module, "LCMConfig", lambda: original(
        embeddings_enabled=enabled, embedding_provider="fastembed", embedding_model=""))
    resolved = []
    original_resolve = lme.resolve_harness_providers
    def fake_provider(provider, model, **kwargs):
        resolved.append((provider, model))
        return original_resolve("stub", "", **kwargs)
    monkeypatch.setattr(lme, "resolve_harness_providers", fake_provider)
    report = lme.run_harness([], provider_name="stub", model="", shipped_default=True,
                             tmp_dir=tmp_path, reuse_db_template=False)
    assert report["provider"] == "fastembed"
    assert report["model"] == "BAAI/bge-small-en-v1.5"
    assert report["embeddings_enabled"] is enabled
    assert resolved == ([("fastembed", "BAAI/bge-small-en-v1.5")] if enabled else [])


def test_shipped_default_does_not_override_explicit_pair(tmp_path):
    import benchmarking.longmemeval as lme
    with pytest.raises(ValueError, match="cannot override explicit"):
        lme.run_harness([], provider_name="voyage", model="explicit-cloud", shipped_default=True, tmp_dir=tmp_path)
