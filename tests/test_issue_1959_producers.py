"""Producer arm identity and option admission, with no model execution."""
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_lcm import config as config_module
from tests.test_issue_1186_s3_producers import ROOT, _older_tree_config
from tests.test_issue_1186_s3_producers import bridge_module as bridge_module


def _load(relative):
    spec = importlib.util.spec_from_file_location("producer_1959", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("provider,model", [("", ""), ("", "fixture-model"), ("off", ""),
                                           ("fastembed", "fixture-model"), ("voyage", "fixture-model")])
def test_d1_bridge_accepts_pre_1938_signature(bridge_module, tmp_path, monkeypatch, provider, model):
    monkeypatch.setattr(config_module, "LCMConfig", _older_tree_config())
    monkeypatch.delattr(config_module, "effective_embedding_model")
    monkeypatch.setenv("HERMES_LCM_REPO", str(ROOT))
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MB_PROVIDER", provider)
    monkeypatch.setenv("HERMES_MB_MODEL", model)
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture")
    bridge = bridge_module.Bridge()
    bridge._resolve_harness_provider = lambda p, m: SimpleNamespace(dim=2, model_id=m)
    assert bridge.initialize({})["ok"]
    config = bridge._config(tmp_path / "lcm.db")
    assert config.embeddings_enabled is bool(model)
    assert not hasattr(config, "embeddings_enabled_explicit")


@pytest.mark.parametrize("producer", ["replay", "steady_state", "stress", "lifecycle", "rebind", "sweep", "fuzz", "stubbing"])
def test_deterministic_producers_pin_off(producer, tmp_path, monkeypatch):
    from benchmarking.replay import _ensure_hermes_lcm_package
    from benchmarking.types import LCMPolicy

    _ensure_hermes_lcm_package()
    from hermes_lcm import engine

    class Captured(Exception):
        pass

    configs = []

    def capture(*, config, **kwargs):
        configs.append(config)
        raise Captured

    monkeypatch.setattr(engine, "LCMEngine", capture)
    if producer == "replay":
        from benchmarking.replay import _config_from_policy
        configs.append(_config_from_policy(LCMPolicy("fixture", 400, .2, 1, 20), tmp_path / "lcm.db"))
    else:
        with pytest.raises(Captured):
            if producer == "steady_state":
                from benchmarking.steady_state import SteadyStateCase, _build_engine
                _build_engine(SteadyStateCase("fixture"), tmp_path)
            elif producer in {"stress", "lifecycle", "rebind"}:
                from benchmarking import stress
                run = stress.StressRun(output_dir=tmp_path, tier=stress.TIERS["smoke"])
                if producer == "stress":
                    run.make_engine("fixture")
                elif producer == "lifecycle":
                    stress._case_lifecycle_soak_and_profile_rebinds(run)
                else:
                    stress._profile_rebind_probe(run, "fixture")
            elif producer == "sweep":
                _load("scripts/benchmark_threshold_full_sweep.py")._run_policy(
                    tmp_path, name="fixture", full_sweep=False, messages=[],
                    fresh_tail_messages=1, leaf_chunk_tokens=20)
            elif producer == "stubbing":
                _load("benchmarks/benchmark_active_tool_stubbing.py")._run_mode(
                    tmp_path, [], enabled=True, stub_threshold_tokens=20)
            else:
                _load("bench/instruments/occurrence_fuzz/driver.py").run(
                    {}, tmp_path / "fuzz", config_module, engine, SimpleNamespace(bad=[]))
    assert len(configs) == 1
    assert configs[0].embeddings_enabled is False


@pytest.mark.parametrize("provider", ["off", "none"])
def test_bridge_off_record_has_no_provider_or_model(bridge_module, tmp_path, monkeypatch, provider):
    monkeypatch.setenv("HERMES_LCM_REPO", str(ROOT))
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MB_PROVIDER", provider)
    monkeypatch.setenv("HERMES_MB_MODEL", "ignored-local-model")
    bridge = bridge_module.Bridge()
    bridge._resolve_harness_provider = lambda *a: pytest.fail("OFF resolved an embedder")
    record = bridge.initialize({})
    assert record["embeddings_enabled"] is False
    assert record["provider"] == record["model"] == ""
    assert json.loads((tmp_path / "effective-config.json").read_text()) == record


@pytest.mark.parametrize("provider", ["fastembed", "fast-embed", "voyage"])
@pytest.mark.parametrize("entry", ["resolver", "harness"])
def test_explicit_provider_requires_model(provider, entry, tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    # No model construction/download even on the failing base.
    from hermes_lcm import embedding_provider
    monkeypatch.setattr(embedding_provider, "FastembedProvider", lambda model, **kw: SimpleNamespace(
        dim=2, model_id=model, provider_id="fastembed"))
    resolve = lme.resolve_harness_providers
    monkeypatch.setattr(lme, "resolve_harness_providers", lambda *a, **kw: resolve("stub", "", **kw))
    with pytest.raises(ValueError, match="--model is required for --provider"):
        if entry == "resolver":
            lme.resolve_harness_provider(provider, "", warmup=False, use_embed_cache=False)
        else:
            lme.run_harness([], provider_name=provider, model="", tmp_dir=tmp_path)


@pytest.mark.parametrize("option,value", [("--emb", "on"), ("--prov", "fastembed"), ("--mod", "fixture")])
def test_cli_rejects_abbreviated_embedding_options(option, value, tmp_path, capsys):
    from tests.conftest import load_cli

    cli = load_cli()
    args = ["run", "--dataset", str(tmp_path / "unused"), "--output", str(tmp_path / "out"),
            "--shipped-default", option, value]
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(args)
    assert exc.value.code == 2
    assert "unrecognized arguments:" in capsys.readouterr().err


def test_track_s_default_arm_records_effective_model(monkeypatch, tmp_path):
    # Load by path: the instrument normally imports siblings via its script directory.
    directory = ROOT / "bench/instruments/compaction_probe/track_s/s2"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv("TRACK_S_OUT", str(tmp_path))
    monkeypatch.setenv("TRACK_S_MATERIAL", str(tmp_path / "unused"))
    sys.modules.pop("producer_1959", None)
    module = _load(str(directory.relative_to(ROOT) / "run_s_lcmx.py"))
    record = module.config_dict(config_module.LCMConfig())
    assert record["embeddings_enabled"] is True
    assert record["embedding_provider"] == "fastembed"
    assert record["embedding_model"] == "BAAI/bge-small-en-v1.5"


@pytest.mark.parametrize("json_output", [True, False])
def test_live_recall_eval_records_resolved_configuration(monkeypatch, capsys, json_output):
    module = _load("scripts/eval_retrieval_recall.py")
    monkeypatch.setattr(module, "_load_live_provider", lambda: SimpleNamespace(
        provider_id="fastembed", model_id="resolved-local-model"))
    monkeypatch.setattr(module, "evaluate", lambda provider: {})
    monkeypatch.setattr(module, "_markdown", lambda metrics: "fixture metrics")
    monkeypatch.setattr(sys, "argv", ["eval", "--live-provider", *(["--json"] if json_output else [])])
    assert module.main() == 0
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["effective_config"] == {
            "embeddings_enabled": True, "embedding_provider": "fastembed",
            "embedding_model": "resolved-local-model"}
    else:
        assert "fastembed" in output and "resolved-local-model" in output
