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
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


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


def safe_paths(*paths):
    if any(".hermes" in p.resolve().parts for p in paths):
        raise ValueError("live .hermes paths forbidden")


def foreign_files(pid, home, db, cache, user_home=None, out=None, scratch=None):
    """Open files that reach another profile's data (the declaration's cross-profile stop).

    That is any path under a `.hermes` directory, or under the real user home (other profiles,
    sibling cells, live stores) outside the worker's own home, store, model cache and Python
    runtime. OS-managed per-user caches outside the home (e.g. macOS's URL cache) are not profiles.
    """
    import psutil
    roots = [p.resolve() for p in (home, cache, Path(sys.prefix), Path(sys.base_prefix), out, scratch) if p is not None]
    store = [db.resolve(), *[Path(str(db.resolve())+s) for s in ("-wal", "-shm", "-journal")]]
    user_home = (user_home or Path.home()).resolve()
    found = []
    for f in psutil.Process(pid).open_files():
        path = Path(f.path).resolve()
        if ".hermes" not in path.parts and (path in store or any(path.is_relative_to(r) for r in roots)):
            continue
        if ".hermes" in path.parts or path.is_relative_to(user_home):
            found.append(f.path)
    return found


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
        embeddings_enabled=enabled, embedding_provider="fastembed", embedding_model=MODEL)


def guards(mod, events, generation=0):
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
        append(events, {"event": "prepared", "generation": generation, "seconds": time.monotonic()-started,
                        "foreground": foreground, "at": time.perf_counter()})
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
            start = time.perf_counter()
            result = _run(*a, **kw)
            append(events, {"event": _name, "foreground": foreground, "at": start,
                            "end": time.perf_counter(), "embedded": result.get("embedded", 0) if isinstance(result, dict) else 0})
            return result
        setattr(command, name, run)


def immutable_cache(cache):
    if not cache.is_dir() or any(p.stat().st_mode & 0o222 for p in [cache, *cache.rglob("*")]):
        raise ValueError("model cache must exist and have no write bits (including its contents)")
