"""Pilot lifetime must not depend on SQLite cyclic garbage collection."""
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import psutil
import pytest

from bench.instruments.local_catchup import measurement

REPO = Path(__file__).resolve().parents[4]


def arguments(tmp_path):
    cache = tmp_path/"cache"
    cache.mkdir(mode=0o555)
    return SimpleNamespace(scratch=tmp_path/"scratch", out=tmp_path/"out", model_cache=cache,
        user_home=str(tmp_path), blocks=1, samples=1, queries=1, rows=8, shape="short", baseline="empty", seed=1586,
        lcmx_repo=REPO, hermes_source=Path(os.environ.get("INSTRUMENT_HERMES_SOURCE", REPO)))


def track_connections(monkeypatch):
    # Retain objects so GC cannot hide an unclosed connection; never fake build.
    opened, original = [], sqlite3.connect
    def connect(*args, **kwargs):
        c = original(*args, **kwargs)
        opened.append(c)
        return c
    monkeypatch.setattr(sqlite3, "connect", connect)
    return opened


@pytest.mark.usefixtures("fake_store_disk_headroom")
def test_prepare_closes_pilot_before_safety(tmp_path, monkeypatch):
    a = arguments(tmp_path)
    opened = track_connections(monkeypatch)
    for name in ("HOME", "CFFIXED_USER_HOME", "HERMES_HOME", "LCM_HERMES_BASE_DIR",
                 "FASTEMBED_CACHE_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "ORT_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    try:
        measurement.prepare(a)
        a.home, a.db, a.root = a.scratch/"builder-home", a.scratch/"baseline.db", a.out
        foreign = [f.path for f in psutil.Process().open_files()
                   if Path(f.path).is_relative_to(a.scratch)
                   and not Path(f.path).is_relative_to(a.home)
                   and Path(f.path).name not in ("baseline.db", "baseline.db-wal", "baseline.db-shm", "baseline.db-journal")]
        assert foreign == [], foreign
        assert measurement.safety(a)["violations"] == 0
        assert not list(a.scratch.glob("pilot.db*"))
        for c in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                c.execute("SELECT 1")
    finally:
        for c in opened:
            c.close()


def test_build_closes_resources_on_failure(tmp_path, monkeypatch):
    from bench.instruments.local_catchup import common, corpus
    monkeypatch.setenv("HERMES_HOME", str(tmp_path/"home"))
    mod = common.product(REPO, Path(os.environ.get("INSTRUMENT_HERMES_SOURCE", REPO)))
    opened = track_connections(monkeypatch)
    def fail(_):
        raise RuntimeError("injected corpus failure")
    monkeypatch.setattr(mod("tokens"), "count_tokens", fail)
    try:
        with pytest.raises(RuntimeError, match="injected corpus failure"):
            corpus.build(REPO, REPO, tmp_path/"failed.db", 8)
        for c in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                c.execute("SELECT 1")
    finally:
        for c in opened:
            c.close()


@pytest.mark.usefixtures("fake_store_disk_headroom")
def test_prepare_rejects_open_pilot_before_removing_it(tmp_path, monkeypatch):
    a = arguments(tmp_path)
    for name in ("HOME", "CFFIXED_USER_HOME", "HERMES_HOME", "LCM_HERMES_BASE_DIR",
                 "FASTEMBED_CACHE_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "ORT_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    original, held = measurement.build, []
    def build(*args, **kwargs):
        manifest = original(*args, **kwargs)
        held.append(sqlite3.connect(args[2]))
        return manifest
    monkeypatch.setattr(measurement, "build", build)
    try:
        with pytest.raises(RuntimeError, match="pilot store handles remain open"):
            measurement.prepare(a)
        assert (a.scratch/"pilot.db").is_file()
        assert not (a.scratch/"baseline.db").exists()
    finally:
        for c in held:
            c.close()
