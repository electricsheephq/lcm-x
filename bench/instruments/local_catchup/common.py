"""Isolation and product loading; never execute the plugin registration entrypoint."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types

MODEL = "BAAI/bge-small-en-v1.5"


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--lcmx-repo", type=Path, required=True)
    p.add_argument("--hermes-source", type=Path, required=True)
    return p


def product(repo, host):
    repo = repo.resolve()
    existing = sys.modules.get("hermes_lcm")
    if existing and list(existing.__path__) != [str(repo)]:
        raise RuntimeError("one product checkout per interpreter")
    sys.path.insert(0, str(host.resolve()))
    sys.path.insert(0, str(repo))
    if not existing:
        pkg = types.ModuleType("hermes_lcm")
        pkg.__path__ = [str(repo)]
        sys.modules["hermes_lcm"] = pkg
    return lambda name: importlib.import_module("hermes_lcm." + name)


def isolated(home, cache):
    home = home.resolve()
    if home.exists() or ".hermes" in home.parts:
        raise ValueError("home must be new and must not be a live .hermes profile")
    home.mkdir(parents=True)
    for k in list(os.environ):
        if k.startswith("LCM_"):
            del os.environ[k]
    os.environ.update(HERMES_HOME=str(home), LCM_HERMES_BASE_DIR=str(home.parent),
                      FASTEMBED_CACHE_PATH=str(cache.resolve()), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                      TOKENIZERS_PARALLELISM="false")


def config(mod, db, enabled=True):
    return mod("config").LCMConfig(database_path=str(db.resolve()),
        embeddings_enabled=enabled, embedding_provider="fastembed", embedding_model=MODEL,
        temporal_rollups_enabled=False, embedding_max_batch_items=32)


def guards(mod, events):
    """Observe real product calls, fail closed on network or foreground backfill/prepare."""
    def deny(*a, **kw):
        append(events, {"violation": "network_attempt", "thread": threading.current_thread().name})
        raise RuntimeError("offline instrument denies network")
    socket.socket.connect = socket.socket.connect_ex = deny
    provider = mod("embedding_provider")
    if provider._FASTEMBED_THREADS > 2:
        raise RuntimeError("product ORT thread count exceeds two")
    original = provider.FastembedProvider._construct
    def construct(self, *, allow_download):
        foreground = threading.current_thread() is threading.main_thread()
        append(events, {"event": "prepare", "foreground": foreground,
                        "download": allow_download, "at": time.monotonic()})
        if allow_download or foreground:
            raise RuntimeError("foreground/download preparation forbidden")
        started = time.monotonic()
        model = original(self, allow_download=allow_download)
        append(events, {"event": "prepared", "seconds": time.monotonic()-started})
        return model
    provider.FastembedProvider._construct = construct
    command = mod("command")
    for name in ("_embedding_backfill_summary_run", "_chunk_backfill_run"):
        original_run = getattr(command, name)
        def run(*a, _name=name, _run=original_run, **kw):
            foreground = threading.current_thread() is threading.main_thread()
            append(events, {"event": _name, "foreground": foreground})
            if foreground:
                raise RuntimeError("foreground backfill forbidden")
            return _run(*a, **kw)
        setattr(command, name, run)


def immutable_cache(cache):
    if not cache.is_dir() or any(p.stat().st_mode & 0o222 for p in [cache, *cache.rglob("*")]):
        raise ValueError("model cache must exist and have no write bits (including its contents)")
