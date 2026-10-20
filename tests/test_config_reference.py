"""Pin the config metadata consumed by the configuration reference."""

import pytest

from hermes_lcm import config


def test_reference_scalar_specs_match_constructor_defaults():
    defaults = config.LCMConfig()
    assert len({spec.env_key for spec in config.ENV_FIELD_SPECS}) == len(config.ENV_FIELD_SPECS)
    for spec in config.ENV_FIELD_SPECS:
        value = getattr(defaults, spec.name)
        # Privacy's None means automatic provider-dependent protection.
        assert value is None or type(value) is spec.py_type


def test_reference_diagnostic_ranges():
    assert config._ENV_RANGES == {
        "context_threshold": (0.0, 1.0, False),
        "survival_reserve": (0.0, 0.9, True),
        "leaf_chunk_tokens": (1, None, True),
        "summary_timeout_ms": (0, None, False),
    }


def test_reference_diagnostic_choices():
    assert {name: row[0] for name, row in config._ENV_CHOICES.items()} == {
        "embedding_provider": (
            "voyage", "voyageai", "ollama", "fastembed", "fast-embed",
            "openai-compatible", "openai", "siliconflow",
        ),
        "preanswer_evidence_mode": ("off", "legacy_selective", "requirements_v1"),
        "embedding_storage_dtype": ("float32", "int8"),
    }


@pytest.mark.parametrize("key", (
    "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED",
))
def test_reference_deprecation_warning_bytes(monkeypatch, key):
    for name in tuple(config.os.environ):
        if name.startswith("LCM_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    monkeypatch.setenv(key, "true")
    assert config.LCMConfig.from_env().config_source_warnings == [
        f"deprecated: {key} is set; this experiment is removed in v0.29.0 (#1246). "
        "Until then it behaves as before."
    ]


@pytest.fixture
def reference():
    from scripts import gen_config_reference

    return gen_config_reference


def test_committed_reference_is_current(reference):
    document = reference.REFERENCE.read_text(encoding="utf-8")
    assert document == reference.update_reference(document), (
        "Configuration reference is stale; run python scripts/gen_config_reference.py"
    )


def test_generator_uses_metadata_and_constructor_defaults(monkeypatch, reference):
    from types import SimpleNamespace

    specs = (
        SimpleNamespace(name="limit", env_key="LCM_LIMIT", py_type=float),
        SimpleNamespace(name="mode", env_key="LCM_MODE", py_type=str),
        SimpleNamespace(name="privacy", env_key="LCM_PRIVACY", py_type=bool),
        SimpleNamespace(name="enabled", env_key="LCM_ENABLED", py_type=bool),
    )
    monkeypatch.setattr(reference, "ENV_FIELD_SPECS", specs)
    monkeypatch.setattr(reference, "LCMConfig", lambda: SimpleNamespace(
        limit=0.5, mode="", privacy=None, enabled=False,
    ))
    monkeypatch.setattr(reference, "_ENV_RANGES", {"limit": (0, 1, False)})
    monkeypatch.setattr(reference, "_ENV_CHOICES", {"mode": (("a", "b"), str.lower, "fallback")})
    monkeypatch.setattr(reference, "ENV_FIELD_DEPRECATIONS", {"enabled": ("v1", "v2")})
    assert reference.render_reference().splitlines()[2:] == [
        "| `LCM_LIMIT` | `float` | `0.5` | `> 0`, `<= 1` | — |",
        '| `LCM_MODE` | `str` | `""` | `"a"`, `"b"` | — |',
        "| `LCM_PRIVACY` | `bool` | `null` | — | — |",
        "| `LCM_ENABLED` | `bool` | `false` | — | Deprecated v1; removed v2 |",
    ]


def test_generator_ignores_environment_and_host_config(monkeypatch, reference):
    import sys

    expected = reference.render_reference()
    module = sys.modules[reference.LCMConfig.__module__]

    def forbidden(*args, **kwargs):
        pytest.fail("reference generation must not read host configuration")

    monkeypatch.setattr(module.LCMConfig, "from_env", forbidden)
    monkeypatch.setattr(module, "_load_hermes_config_yaml", forbidden)
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "999")
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-placeholder")
    assert reference.render_reference() == expected
    assert "synthetic-placeholder" not in expected


def test_generator_replaces_only_marked_bytes_and_is_idempotent(reference, monkeypatch, tmp_path):
    document = "# Before\n\n" + reference.START + "\nold\n" + reference.END + "\n\n# After\n"
    path = tmp_path / "configuration.md"
    path.write_text(document, encoding="utf-8")
    monkeypatch.setattr(reference, "REFERENCE", path)
    reference.main()
    generated = path.read_text(encoding="utf-8")
    assert generated == (
        "# Before\n\n" + reference.START + "\n" + reference.render_reference()
        + reference.END + "\n\n# After\n"
    )
    reference.main()
    assert path.read_text(encoding="utf-8") == generated


@pytest.mark.parametrize("document", (
    "no markers", "{start}", "{end}", "{end}\n{start}",
    "{start}\n{start}\n{end}", "{start}\n{end}\n{end}",
))
def test_generator_refuses_missing_duplicate_or_reversed_markers(reference, document):
    with pytest.raises(ValueError):
        reference.update_reference(document.format(start=reference.START, end=reference.END))


@pytest.fixture
def choice_plugin():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", Path(__file__).resolve().parents[1] / "__init__.py",
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_lcm"
    spec.loader.exec_module(module)
    return module


def _provider_choice_row(monkeypatch, spelling):
    """Exercise the decision sites with inert storage and no provider dispatch."""
    from types import SimpleNamespace

    from hermes_lcm import command, embedding_maintenance as maintenance
    from hermes_lcm import embedding_provider as providers, tools

    cfg = config.LCMConfig(
        embeddings_enabled=True, embedding_provider=spelling,
        embedding_model="choice-model", embedding_base_url="http://localhost:8080/v1",
    )
    for key in ("VOYAGE_API_KEY", "LCM_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda _: None)
    try:
        resolved = type(providers.resolve_provider(cfg)).__name__
    except providers.ProviderUnavailable as exc:
        resolved = type(exc).__name__
    probe = providers.probe_provider_availability(cfg)
    mapped = []

    def capture_provider(provider):
        mapped.append(provider)
        raise LookupError("captured before storage access")

    with monkeypatch.context() as capture:
        capture.setattr(tools, "embedding_provider_requires_privacy", capture_provider)
        try:
            tools._lcm_recall_has_usable_vector_corpus(
                None, run_summary=True, run_chunk=True, provider_name=spelling,
                model_name="choice-model", excluded_session_ids=set(), deadline=float("inf"),
            )
        except LookupError:
            pass
    health_mapped = []
    with monkeypatch.context() as health:
        # Isolate spelling resolution from the separately owned privacy policy.
        health.setattr(tools, "embedding_provider_requires_privacy",
                       lambda value: config.embedding_provider_id(value) in ("voyage", "openai-compatible"))
        health.setenv("VOYAGE_API_KEY", "synthetic-placeholder")
        health.setenv("LCM_EMBEDDING_API_KEY", "synthetic-placeholder")
        health.setattr(providers.importlib.util, "find_spec", lambda _: object())
        health.setattr(tools, "_active_embedding_profile_exists", lambda _: False)
        health.setattr(tools, "embedding_privacy_revision", lambda _: "choice-revision")
        health.setattr(tools.sqlite3, "connect", lambda *a, **kw: SimpleNamespace(close=lambda: None))
        health.setattr(tools, "_lcm_store_has_live_chunk_vectors", lambda _: False)

        def capture_health(conn, *, provider, model, task):
            health_mapped.append(provider)
            return "choice-revision"

        health.setattr(tools, "_lcm_embedding_profile_revision", capture_health)
        health_result = tools._embedding_provider_health_check(SimpleNamespace(
            _config=cfg, _store=SimpleNamespace(db_path="unused"),
        ))
    auto = maintenance._automatic_chunks_allowed(cfg, command)
    monkeypatch.setattr(command, "_embedding_read_connection", lambda _: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(command, "_embedding_current_profile", lambda _: None)
    monkeypatch.setattr(command, "_chunk_current_profile", lambda _: None)
    monkeypatch.setattr(command, "_embedding_register_profiles", lambda *a, **kw: "local registration")
    local = maintenance._run_pass("unused", cfg, None)
    # Drive the nested contextualized estimate through the real dry-run path.
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a: (1, [], None))
    monkeypatch.setattr(command, "embedding_provider_requires_privacy", lambda _: False)
    monkeypatch.setattr(command, "_prepare_embedding_provider_documents",
                        lambda *a, **kw: ([("row", "text", 5)], None, 0, 0, None))
    monkeypatch.setattr(command, "_chunk_context_estimates", lambda *a: (11, 12, 13))
    monkeypatch.setattr(command, "_embedding_batch_estimate", lambda *a: 17)
    result = command._chunk_backfill_run(
        cfg, "unused", apply=False, limit=1, retry_uncertain=False, policy="conversational",
    )
    assert isinstance(result, dict)
    return {
        "resolver": resolved, "offline": probe,
        "mapped_id": mapped[0] if mapped else None, "automatic_chunks": auto,
        "health_mapped_ids": health_mapped, "health_check": health_result,
        "local_registration": local, "chunk_model": providers.default_chunk_model(spelling, "choice-model"),
        "chunk_estimate": [result[key] for key in ("estimated_tokens", "estimated_cost_tokens", "estimated_batches")],
    }


def test_choice_constants_feed_reference():
    assert config._ENV_CHOICES["embedding_provider"][0] == tuple(
        spelling for spellings in config.EMBEDDING_PROVIDER_SPELLINGS.values() for spelling in spellings
    )
    assert config._ENV_CHOICES["preanswer_evidence_mode"][0] == config.PREANSWER_EVIDENCE_MODES
    assert config._ENV_CHOICES["embedding_storage_dtype"][0] == config.EMBEDDING_STORAGE_DTYPES


@pytest.mark.parametrize("provider", ("voyage", "ollama", "fastembed", "openai-compatible"))
def test_provider_consumers_follow_choice_constant(monkeypatch, provider):
    with monkeypatch.context() as patch:
        expected = _provider_choice_row(patch, provider)
    spelling = "new-choice"
    monkeypatch.setitem(config.EMBEDDING_PROVIDER_SPELLINGS, provider,
                        config.EMBEDDING_PROVIDER_SPELLINGS[provider] + (spelling,))
    actual = _provider_choice_row(monkeypatch, spelling)
    # The probe deliberately reports the configured spelling, not the provider id.
    expected["offline"]["provider"] = spelling
    expected["health_check"]["detail"]["provider"] = spelling
    assert actual == expected


def test_preanswer_consumer_follows_choice_constant(monkeypatch, choice_plugin):
    from types import SimpleNamespace

    monkeypatch.setattr(config, "PREANSWER_EVIDENCE_MODES", config.PREANSWER_EVIDENCE_MODES + ("new-mode",))
    assert choice_plugin._effective_preanswer_mode(SimpleNamespace(
        preanswer_evidence_enabled=True, preanswer_evidence_mode=" NEW-MODE ",
    )) == "new-mode"
    assert choice_plugin._effective_preanswer_mode(SimpleNamespace(
        preanswer_evidence_enabled=True, preanswer_evidence_mode="requirements_v1",
    )) == "requirements_v1"


def test_dtype_consumers_follow_choice_constant(monkeypatch):
    from types import SimpleNamespace

    from hermes_lcm import command

    monkeypatch.setattr(config, "EMBEDDING_STORAGE_DTYPES", config.EMBEDDING_STORAGE_DTYPES + ("new-dtype",))
    assert command._resolve_storage_dtype(SimpleNamespace(embedding_storage_dtype=" NEW-DTYPE ")) == "new-dtype"
    assert command._resolve_storage_dtype(None, "NEW-DTYPE") == "new-dtype"
    result = command._embedding_backfill_options(["--dtype", "NEW-DTYPE"])
    assert isinstance(result, tuple) and result[-1] == "new-dtype"


@pytest.mark.parametrize("threshold", (1.5, 0.97, 0, -1, 0.2))
def test_doctor_context_threshold_reproduction(monkeypatch, tmp_path, threshold):
    import json

    from hermes_lcm import tools
    from tests.harness import engine_context, pin_char_counter

    pin_char_counter(monkeypatch)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", str(threshold))
    cfg = config.LCMConfig.from_env()
    with engine_context(tmp_path, context_threshold=cfg.context_threshold,
                        config_source_warnings=cfg.config_source_warnings) as engine:
        payload = json.loads(tools.lcm_doctor({}, engine=engine))
        assert engine._config.context_threshold == threshold
        warnings = next(check["detail"] for check in payload["checks"]
                        if check["check"] == "config_validation")
        threshold_lines = [line for line in warnings if "context_threshold" in line.lower()]
        assert threshold_lines == (cfg.config_source_warnings if threshold > 1 or threshold <= 0 else [
            "runtime context_threshold > 0.95 leaves very little headroom" if threshold > 0.95
            else "runtime context_threshold < 0.3 triggers compaction very early",
        ])
        print("THRESHOLD_CHECKS=" + json.dumps([
            check for check in payload["checks"]
            if isinstance(check["detail"], (str, list))
            and "context_threshold" in json.dumps(check["detail"]).lower()
        ], sort_keys=True))
