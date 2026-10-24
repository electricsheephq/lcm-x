"""r4: truthful CPU-limited readiness guidance and proactive budget parity."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
import json
import sqlite3

import pytest

from hermes_lcm import tools
from hermes_lcm.diagnostics import doctor_guidance_for_check
from tests.test_issue_1186_r3 import default_engine  # noqa: F401


@pytest.mark.parametrize("cpus", [1, 8])
@pytest.mark.parametrize("coverage_error", [False, True])
def test_not_prepared_guidance_with_real_readiness(default_engine, monkeypatch, cpus, coverage_error):
    from hermes_lcm import command, embedding_maintenance

    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: None)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": "test", "available": True, "detail": "configured",
    })
    if coverage_error:
        def unreadable(*args):
            raise sqlite3.OperationalError("synthetic coverage read failure")
        monkeypatch.setattr(command, "_embedding_read_connection", unreadable)
    check = tools._embedding_provider_health_check(default_engine)
    assert check["detail"]["readiness"].startswith("model: not prepared (run /lcm embed warmup);")
    guidance = doctor_guidance_for_check(check)
    if cpus == 8:
        assert check["status"] == "pass" and guidance is None
    else:
        assert check["status"] == "warn"
        assert not any(marker in guidance["operator_action"] for marker in (
            "configured but not runnable", "hermes-lcm-x-local-embeddings",
        ))
        if coverage_error:
            assert guidance["action"] == "inspect"
            assert "coverage" in guidance["operator_action"] and "SQLite" in guidance["operator_action"]
        else:
            assert guidance["action"] == "safe/ignore"
            assert "expected on this host" in guidance["rationale"]
            assert "LCM_EMBEDDINGS_ENABLED=true" in guidance["operator_action"]
            assert "more than 2 effective CPUs" in guidance["operator_action"]


@pytest.mark.parametrize("model,available,action,remedy", [
    ("present", True, "safe/ignore", "LCM_EMBEDDINGS_ENABLED=true"),
    ("preparing", True, "inspect", "prepar"),
    ("unavailable(download failed)", True, "inspect", "warmup"),
    ("unavailable(missing fastembed)", False, "inspect", "hermes-lcm-x-local-embeddings"),
])
def test_cpu_limited_readiness_remedies(default_engine, monkeypatch, model, available, action, remedy):
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": "test", "available": available, "detail": model,
    })
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *args: f"model: {model}; coverage: 0/0 summaries")
    guidance = doctor_guidance_for_check(tools._embedding_provider_health_check(default_engine))
    if available and not model.startswith("unavailable("):
        assert not any(marker in guidance["operator_action"] for marker in (
            "configured but not runnable", "hermes-lcm-x-local-embeddings",
        ))
    assert guidance["action"] == action and remedy in guidance["operator_action"]
    if not available:
        assert "configured but not runnable" in guidance["operator_action"]


@pytest.mark.parametrize("override", ["", " ", "voyage", " fastembed "])
@pytest.mark.parametrize("cpus", [1, 8])
def test_prefix_budget_matches_off_unless_semantic_active(default_engine, monkeypatch, override, cpus):
    engine = default_engine
    config = engine._config
    config.proactive_recall_enabled = True
    config.proactive_recall_budget_tokens = 500
    config.proactive_recall_provider = override
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    monkeypatch.setattr(engine, "_survival_ceiling", lambda: 2000)
    monkeypatch.setattr(engine, "_survival_assembly_overhead", lambda *args: 0)

    def budget():
        return engine._resolve_survival_prefix_budget(
            None, None, [], [], [], "user", ["summary " * 3000], [(0, 1)], [], None, None,
        )[0]

    config.embeddings_enabled = False
    off = budget()
    assert off == 2000
    config.embeddings_enabled = True
    active = budget()
    assert active == off - (500 if cpus > 2 or override.strip() else 0)
    if cpus <= 2 and not override.strip():
        # Assert the assembled bytes as well as the selected numerical budget.
        engine.on_session_start("synthetic-r4", context_length=2000)
        tail = [{"role": "user", "content": "current question"}]
        on_bytes = json.dumps(engine._assemble_context(None, tail, persist=False), sort_keys=True)
        config.embeddings_enabled = False
        assert on_bytes == json.dumps(engine._assemble_context(None, tail, persist=False), sort_keys=True)
