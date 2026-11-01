"""Which embedding providers count as local, and when new chunks embed automatically.

Characterization for the store -> command import cycle break: these pin the current
policy before it moves, so the move can be checked against them.
"""
from types import SimpleNamespace

import pytest

from hermes_lcm import command, embedding_maintenance
from hermes_lcm.config import LCMConfig


@pytest.mark.parametrize(("name", "local"), [
    ("fastembed", True), ("ollama", True), (" Ollama ", True), ("FASTEMBED", True),
    ("fast-embed", False), ("voyage", False), ("openai", False), ("", False), (None, False),
])
def test_local_embedding_provider_names(name, local):
    assert command._is_local_embedding_provider(name) is local


@pytest.mark.parametrize(("provider", "base_url", "allowed"), [
    ("fastembed", None, True),
    ("fast-embed", None, True),
    ("ollama", None, True),
    ("ollama", "http://localhost:11434", True),
    ("ollama", "http://127.0.0.1:11434/", True),
    ("ollama", "http://[::1]:11434", True),
    ("ollama", "", True),
    ("ollama", "http://192.0.2.1:11434", False),
    ("ollama", "http://ollama.example:11434", False),
    ("voyage", None, False),
    ("openai", None, False),
])
def test_automatic_chunks_allowed(provider, base_url, allowed):
    options = {} if base_url is None else {"ollama_base_url": base_url}
    config = LCMConfig(embedding_provider=provider, **options)
    assert embedding_maintenance._automatic_chunks_allowed(config, command) is allowed


def test_automatic_chunks_allowed_empty_provider():
    # LCMConfig resolves an empty provider to fastembed, unless only a model is set.
    assert embedding_maintenance._automatic_chunks_allowed(LCMConfig(embedding_provider=""), command)
    model_only = LCMConfig(embedding_provider="", embedding_model="model-a")
    assert not embedding_maintenance._automatic_chunks_allowed(model_only, command)


@pytest.mark.parametrize(("config", "allowed"), [
    (SimpleNamespace(), False),
    (SimpleNamespace(embedding_provider=None), False),
    (SimpleNamespace(embedding_provider="ollama"), True),
    (SimpleNamespace(embedding_provider="ollama", ollama_base_url="http://192.0.2.1:11434"), False),
])
def test_automatic_chunks_allowed_attribute_defaults(config, allowed):
    assert embedding_maintenance._automatic_chunks_allowed(config, command) is allowed
