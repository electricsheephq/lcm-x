"""#672: lcm_doctor must disclose a configured-but-unavailable embedding provider.

Before this change `lcm_doctor` emitted no embedding-related check at all, so an
operator who enabled semantic embeddings and then lost the provider (e.g. the
optional ``fastembed`` dependency dropped when a Hermes update rebuilt the
virtualenv) saw every check ``pass`` while `lcm_recall` reported
``degraded=true`` and `lcm_grep` ``mode=semantic`` silently fell back to
full-text.

These tests pin the new ``embedding_provider_health`` check, its offline-safety
contract, and its operator guidance. They fail on the base SHA (9bbfb8f) because
the check does not exist there.
"""

from __future__ import annotations

import json

import pytest

import hermes_lcm.embedding_provider as embedding_provider
from hermes_lcm.config import LCMConfig
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import embedding_privacy_revision
from hermes_lcm.vector_store import VectorStore


def probe_provider_availability(config):
    """Resolve the probe lazily.

    Imported at call time, not module import time, so that on the base SHA the
    doctor-behavior tests below fail on their own assertions (proving the
    observable gap) instead of the whole module failing to collect.
    """
    return embedding_provider.probe_provider_availability(config)


def _make_engine(tmp_path, db_name="lcm_672.db", **config_kwargs):
    config = LCMConfig(database_path=str(tmp_path / db_name), **config_kwargs)
    engine = LCMEngine(config=config)
    engine._session_id = "test-session"
    engine.context_length = 200000
    return engine


def _doctor_check(engine, name="embedding_provider_health"):
    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    checks = {check["check"]: check for check in payload["checks"]}
    return payload, checks.get(name)


# --------------------------------------------------------------------------
# The check exists at all (this alone fails on the base SHA).
# --------------------------------------------------------------------------


def test_doctor_emits_an_embedding_provider_health_check(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _, check = _doctor_check(engine)
        assert check is not None, (
            "lcm_doctor must emit an embedding_provider_health check so a dead "
            "embedding provider is detectable"
        )
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# The reported production failure: enabled + provider missing -> warn.
# --------------------------------------------------------------------------


def test_enabled_but_missing_optional_dependency_warns(tmp_path, monkeypatch):
    """The exact observed state: fastembed configured, dependency absent."""
    monkeypatch.setattr(
        embedding_provider.importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "fastembed" else object(),
    )
    engine = _make_engine(
        tmp_path,
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    try:
        payload, check = _doctor_check(engine)

        assert check is not None
        assert check["status"] == "warn"
        assert payload["overall"] != "healthy"

        detail = check["detail"]
        assert detail["embeddings_enabled"] is True
        assert detail["available"] is False
        assert detail["provider"] == "fastembed"
        assert "fastembed" in detail["reason"].lower()
        # The operator must be told what actually broke, not just that something did.
        assert "full-text" in detail["impact"]
    finally:
        engine.shutdown()


def test_warning_carries_actionable_operator_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embedding_provider.importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "fastembed" else object(),
    )
    engine = _make_engine(
        tmp_path,
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    try:
        payload, check = _doctor_check(engine)
        guidance = {item["check"]: item for item in payload["guidance"]}

        assert "embedding_provider_health" in guidance
        item = guidance["embedding_provider_health"]
        assert item["warning_only"] is True
        assert "fastembed" in item["operator_action"]

        # Also exercise the pure function directly.
        standalone = doctor_guidance_for_check(check)
        assert standalone is not None
        assert standalone["check"] == "embedding_provider_health"
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# Default configuration stays quiet (backward compatibility).
# --------------------------------------------------------------------------


def test_disabled_embeddings_pass_without_noise(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        payload, check = _doctor_check(engine)

        assert check["status"] == "pass"
        assert "disabled" in check["detail"]
        # A default install must not start warning because of this check.
        assert payload["overall"] == "healthy"
    finally:
        engine.shutdown()


def test_enabled_and_available_provider_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embedding_provider.importlib.util, "find_spec", lambda name, *a, **kw: object()
    )
    engine = _make_engine(
        tmp_path,
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    try:
        _, check = _doctor_check(engine)
        assert check["status"] == "pass"
        assert check["detail"]["available"] is True
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# Offline-safety contract: the probe must not touch the network.
# --------------------------------------------------------------------------


def test_probe_performs_no_network_or_model_load(tmp_path, monkeypatch):
    """lcm_doctor is diagnostic; the check must never download or dial out."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("embedding_provider_health performed network I/O")

    monkeypatch.setattr(embedding_provider.urllib.request, "urlopen", explode)
    monkeypatch.setattr(
        embedding_provider, "_load_fastembed", explode
    )  # importing/constructing the model is also forbidden

    monkeypatch.setenv("VOYAGE_API_KEY", "not-a-real-key")
    engine = _make_engine(
        tmp_path,
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-3",
    )
    try:
        vectors = VectorStore(engine._store.db_path, config=engine._config)
        try:
            vectors.register_profile(
                "voyage-3", "voyage", 2,
                revision=embedding_privacy_revision(engine._config),
            )
        finally:
            vectors.close()
        _, check = _doctor_check(engine)
        assert check["status"] == "pass"
        # Credential presence is not reachability, and the check must say so.
        assert check["detail"]["reachability_probed"] is False
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# probe_provider_availability unit coverage.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expect_available, needle",
    [
        ({"embedding_provider": "", "embedding_model": ""}, False, "unset"),
        ({"embedding_provider": "fastembed", "embedding_model": ""}, False, "LCM_EMBEDDING_MODEL"),
        ({"embedding_provider": "", "embedding_model": "x"}, False, "LCM_EMBEDDING_PROVIDER"),
        ({"embedding_provider": "nope", "embedding_model": "x"}, False, "unsupported"),
        ({"embedding_provider": "ollama", "embedding_model": "x"}, True, "not probed"),
    ],
)
def test_probe_classifies_configuration(kwargs, expect_available, needle):
    probe = probe_provider_availability(LCMConfig(**kwargs))
    assert probe["available"] is expect_available
    assert needle.lower() in probe["detail"].lower()


def test_probe_reports_missing_voyage_credentials(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    probe = probe_provider_availability(
        LCMConfig(embedding_provider="voyage", embedding_model="voyage-3")
    )
    assert probe["available"] is False
    assert "VOYAGE_API_KEY" in probe["detail"]


def test_disabled_embeddings_warning_gets_its_own_guidance():
    check = {
        "check": "embedding_provider_health",
        "status": "warn",
        "detail": {"embeddings_enabled": False, "active_embedding_profile": True, "available": False},
    }
    guidance = doctor_guidance_for_check(check)
    assert guidance is not None
    text = str(guidance)
    assert "LCM_EMBEDDINGS_ENABLED" in text
    assert "pip install" not in text and "reinstall" not in text
