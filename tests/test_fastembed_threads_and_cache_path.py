from pathlib import Path

import pytest

import hermes_lcm.embedding_provider as provider_mod


def test_construct_passes_threads_cache_and_local_only(monkeypatch):
    calls = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeTextEmbedding)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", " /x/y ")
    provider = provider_mod.FastembedProvider("test-model")

    assert isinstance(provider._construct(allow_download=False), FakeTextEmbedding)
    assert calls == [{
        "model_name": "test-model",
        "cache_dir": "/x/y",
        "threads": 2,
        "local_files_only": True,
    }]


@pytest.mark.parametrize("env_value", ["/x/y", " /x/y ", None, "", " \t "])
def test_cache_path_environment_and_default(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    else:
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", env_value)

    provider = provider_mod.FastembedProvider("test-model")
    expected = Path("/x/y") if env_value and env_value.strip() else Path.home() / ".cache" / "fastembed"
    assert provider.cache_dir == expected


@pytest.mark.parametrize("cache_dir", ["/explicit/cache", Path("/explicit/cache")])
def test_explicit_cache_path_wins(monkeypatch, cache_dir):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/x/y")

    provider = provider_mod.FastembedProvider("test-model", cache_dir=cache_dir)

    assert provider.cache_dir == Path("/explicit/cache")
