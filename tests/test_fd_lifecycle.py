"""Descriptor ownership guards; cyclic GC must not rescue a missed close."""

import gc
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import benchmarking.longmemeval as lme
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.conftest import _MAINTENANCE_SCHEDULERS, _wait_scheduler_idle


@pytest.fixture
def fd_count():
    fd_dir = next((p for p in (Path('/proc/self/fd'), Path('/dev/fd')) if p.is_dir()), None)
    if fd_dir is None:
        pytest.skip('no descriptor directory on this platform')
    return lambda: len(os.listdir(fd_dir))


@pytest.mark.parametrize('operation', ['initialize', 'miss', 'hit', 'lookup', 'corrupt-read'])
def test_embedding_cache_closes_each_operation(tmp_path, fd_count, operation):
    path = tmp_path / 'embedding-cache.db'
    provider = lme.StubEmbedder()
    cache = lme.ContentHashEmbeddingCache(provider, path)
    cache.embed_documents(['warm'])
    if operation == 'corrupt-read':
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE embedding_cache SET vector_f64_le = x'00'")

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = fd_count()
        if operation == 'initialize':
            lme.ContentHashEmbeddingCache(provider, path)
        elif operation == 'miss':
            cache.embed_documents(['cold'])
        elif operation == 'hit':
            cache.embed_documents(['warm'])
        elif operation == 'lookup':
            assert cache.cached_count(['warm']) == 1
        else:
            with pytest.raises(ValueError, match='payload is corrupt'):
                cache.embed_documents(['warm'])
        after = fd_count()
        assert after <= before, f'{operation} left {after - before} descriptors open'
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_ingest_compress_shutdown_releases_file_descriptors(tmp_path, monkeypatch, fd_count):
    monkeypatch.setattr(engine_module, 'summarize_with_escalation',
                        lambda **kwargs: ('A deterministic lifecycle summary.', 1))
    config = LCMConfig(
        database_path=str(tmp_path / 'lifecycle.db'),
        fresh_tail_count=2,
        leaf_chunk_tokens=100,
    )
    messages = [
        {'role': role, 'content': f'turn {turn} {role}: ' + 'alpha beta gamma delta ' * 40}
        for turn in range(8) for role in ('user', 'assistant')
    ]
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = fd_count()
        engine = LCMEngine(config=config, hermes_home=str(tmp_path))
        try:
            assert fd_count() > before  # The observer sees this engine's open files.
            engine.on_session_start('fd-lifecycle', platform='cli', context_length=200_000)
            engine.ingest(messages)
            engine.compress(messages, current_tokens=100_000, force=True)
            assert engine._last_compression_status == 'compacted'
            # Default-on maintenance owns separate connections until its jobs finish.
            for name in _MAINTENANCE_SCHEDULERS:
                scheduler = getattr(engine_module, name, None)
                if scheduler is not None:
                    assert _wait_scheduler_idle(scheduler, timeout=30.0), name
        finally:
            engine.shutdown()
        # Retain the engine so neither refcounting nor cyclic GC can hide a missed close.
        after = fd_count()
        assert after <= before, f'engine lifecycle left {after - before} descriptors open'
    finally:
        if was_enabled:
            gc.enable()
