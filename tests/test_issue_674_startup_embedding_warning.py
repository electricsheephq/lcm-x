"""#674: plugin load must say so when semantics is configured but cannot run.

The provider import is lazy, so before this change the plugin logged a plain
"LCM plugin loaded -- lossless context management active" success line and then
stayed silent until the first semantic query, which degrades to full-text rather
than raising. Combined with the (now fixed) doctor blindness in #672, a real host
ran for 14 days with semantic recall dead and nothing anywhere said a word.

These tests pin the startup warning, its silence in the healthy/disabled cases,
and its offline-safety and never-raise contracts.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

import hermes_lcm.embedding_provider as embedding_provider
from hermes_lcm.config import LCMConfig


def _load_plugin_module():
    """Load the plugin's __init__.py so its load-time helpers are testable.

    tests/conftest.py registers the ``hermes_lcm`` package without executing
    ``__init__.py``, so the helper is not reachable via a plain import. Module
    level there is only imports plus a logger -- ``register(ctx)`` is a
    function and is never called here -- so executing it is side-effect free.
    """
    path = Path(__file__).resolve().parent.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_lcm_plugin_under_test", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_lcm"
    spec.loader.exec_module(module)
    return module


_plugin = _load_plugin_module()


def _warn_if_embedding_provider_unavailable(config):
    """Resolve the load-time helper at call time.

    Looked up per call, not at import, so on the base SHA each test fails on
    its own assertion about startup behaviour instead of the module failing to
    collect.
    """
    helper = getattr(_plugin, "_warn_if_embedding_provider_unavailable", None)
    if helper is None:
        return None  # base behaviour: plugin load says nothing at all
    return helper(config)


@pytest.fixture(autouse=True)
def _capture_plugin_logger(caplog):
    """The helper logs on the plugin module's own logger."""
    caplog.set_level(logging.DEBUG, logger=_plugin.logger.name)
    _plugin.logger.propagate = True


def _warn_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_warns_when_enabled_but_dependency_missing(caplog, monkeypatch):
    """The exact production state: fastembed configured, dependency gone."""
    monkeypatch.setattr(
        embedding_provider.importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "fastembed" else object(),
    )
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(config)

    warnings = _warn_records(caplog)
    assert len(warnings) == 1, "an unavailable configured provider must warn exactly once"

    message = warnings[0].getMessage()
    # Name the fault, the blast radius, and the fix.
    assert "fastembed" in message.lower()
    assert "degraded to full-text" in message
    assert "nothing is lost" in message.lower()


def test_silent_when_embeddings_disabled(caplog):
    """The default install must not gain a new warning."""
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(LCMConfig())

    assert _warn_records(caplog) == []


def test_silent_when_provider_is_available(caplog, monkeypatch):
    monkeypatch.setattr(
        embedding_provider.importlib.util, "find_spec", lambda name, *a, **kw: object()
    )
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(config)

    assert _warn_records(caplog) == []


def test_startup_probe_does_no_network_or_model_load(caplog, monkeypatch):
    """Plugin load must never download, dial out, or warm a model up."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("startup embedding probe performed network I/O")

    monkeypatch.setattr(embedding_provider.urllib.request, "urlopen", explode)
    monkeypatch.setattr(embedding_provider, "_load_fastembed", explode)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-4-lite",
    )

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(config)

    # Missing credentials are a real unavailability, so this warns -- the point
    # is that it got there without touching the network.
    assert len(_warn_records(caplog)) == 1


def test_probe_failure_never_breaks_plugin_load(caplog, monkeypatch):
    """A diagnostic must not be able to take the plugin down."""

    def boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(embedding_provider, "probe_provider_availability", boom)
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(config)  # must not raise

    assert _warn_records(caplog) == []


@pytest.mark.parametrize(
    "provider, model",
    [
        ("", ""),
        ("fastembed", ""),
        ("nope", "some-model"),
    ],
)
def test_warns_on_incomplete_or_unsupported_configuration(
    caplog, provider, model
):
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider=provider,
        embedding_model=model,
    )

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _warn_if_embedding_provider_unavailable(config)

    assert len(_warn_records(caplog)) == 1
