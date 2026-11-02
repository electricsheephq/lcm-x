"""Tests for issue #2126: default Ollama base URL uses 127.0.0.1 to avoid IPv6 connect delay on Windows."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_provider import (
    OllamaProvider,
    resolve_provider,
    _is_loopback_url,
)
from hermes_lcm.embedding_maintenance import _automatic_chunks_allowed
import hermes_lcm.command as command_mod


def test_ollama_base_url_default_is_ipv4_loopback(monkeypatch):
    """The default Ollama base URL must be http://127.0.0.1:11434 rather than localhost."""
    monkeypatch.delenv("LCM_OLLAMA_BASE_URL", raising=False)
    config = LCMConfig()
    assert config.ollama_base_url == "http://127.0.0.1:11434"

    from_env_config = LCMConfig.from_env()
    assert from_env_config.ollama_base_url == "http://127.0.0.1:11434"


def test_ollama_provider_init_default_base_url():
    """OllamaProvider defaults to http://127.0.0.1:11434 when base_url is not specified or empty."""
    provider = OllamaProvider("nomic-embed-text")
    assert provider.base_url == "http://127.0.0.1:11434"

    provider_empty = OllamaProvider("nomic-embed-text", base_url="")
    assert provider_empty.base_url == "http://127.0.0.1:11434"


def test_resolve_provider_uses_default_ollama_base_url(monkeypatch):
    """resolve_provider creates OllamaProvider with 127.0.0.1 by default."""
    monkeypatch.delenv("LCM_OLLAMA_BASE_URL", raising=False)
    config = LCMConfig(embedding_provider="ollama", embedding_model="nomic-embed-text")
    provider = resolve_provider(config=config)
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://127.0.0.1:11434"


def test_ollama_base_url_explicit_override_preserved(monkeypatch):
    """Users can still explicitly configure a custom or IPv6 base URL."""
    monkeypatch.setenv("LCM_OLLAMA_BASE_URL", "http://[::1]:11434")
    config = LCMConfig.from_env()
    assert config.ollama_base_url == "http://[::1]:11434"

    provider = OllamaProvider("nomic-embed-text", config=config, base_url=config.ollama_base_url)
    assert provider.base_url == "http://[::1]:11434"


def test_ollama_base_url_loopback_classification():
    """127.0.0.1:11434 and localhost:11434 are recognized as loopback."""
    assert _is_loopback_url("http://127.0.0.1:11434") is True
    assert _is_loopback_url("http://localhost:11434") is True
    config_default = LCMConfig(embedding_provider="ollama")
    assert _automatic_chunks_allowed(config_default, command_mod) is True
