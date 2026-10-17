"""SQLite-version A/B stress probe for the CI 'malformed' / FTS vtable failures (#1084, #1112, #1352).

Diagnostic only; skipped unless LCM_SQLITE_AB=1. One process drives the real
LCMEngine: foreground ingest and forced compaction from this thread, the
product-default temporal-rollup worker, and incremental embedding with a
deterministic stub provider, with periodic engine restarts (shutdown
checkpoints while background passes may still run). After every cycle a fresh
connection runs PRAGMA integrity_check and the FTS5 'integrity-check' command on
every FTS5 table. Output: one line 'SQLITE_AB_RESULT {json}'.

Env: LCM_SQLITE_AB_CYCLES (200), LCM_SQLITE_AB_SEED (1), LCM_SQLITE_AB_KEEP_DIR
(copy the failing DB dir incl. -wal/-shm), LCM_SQLITE_AB_WAL_AUTOCKPT (optional
override of the product's wal_autocheckpoint=500; recorded in the result).
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sqlite3
import tempfile
import time
import traceback
from pathlib import Path

import pytest

if os.environ.get("LCM_SQLITE_AB") != "1":
    pytest.skip("SQLite A/B probe runs only with LCM_SQLITE_AB=1", allow_module_level=True)

import hermes_lcm.command as command_mod
import hermes_lcm.db_bootstrap as db_bootstrap
import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm import engine as engine_module
from hermes_lcm import escalation, tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore
from tests.conftest import _MAINTENANCE_SCHEDULERS, _wait_scheduler_idle

SESSION, CONVERSATION = "ab-session", "ab-conversation"
WORDS = ("alpha", "zebra", "ledger", "migration", "βeta", "orchid", "kernel", "harbor",
         "quartz", "falcon", "delta", "ember", "lantern", "meadow", "nickel", "violet")
CORRUPTION_MARKERS = ("malformed", "vtable constructor failed", "corrupt", "not a database",
                      "fts5: ", "missing from index", "wrong # of entries", "never used",
                      "invalid page", "btreeinitpage")


class StubProvider:
    """Deterministic offline embedding provider (same shape as the #1014 test fake)."""

    def __init__(self, provider_id, model_id):
        self.provider_id, self.model_id, self.dim = provider_id, model_id, 2
        self.last_skipped_documents, self.last_usage_tokens = [], 1

    def embed_document_batches(self, texts, *, before_dispatch):
        indexes = tuple(range(len(texts)))
        before_dispatch(indexes)
        yield EmbeddedDocumentBatch(indexes, tuple((1.0, float(len(t) % 7)) for t in texts))

    def embed_query(self, text):
        return [1.0, 0.0]


class Capture(logging.Handler):
    """Background passes swallow exceptions into logs (rollups at DEBUG); keep the DB ones."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.corrupt, self.locked = [], 0

    def emit(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        text = f"{record.getMessage()} {exc!r}" if exc else record.getMessage()
        if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
            self.locked += 1
        elif "walresetbug" in text:
            return  # the product's version warning, not an error
        elif (isinstance(exc, sqlite3.DatabaseError) or any(m in text.lower() for m in CORRUPTION_MARKERS[:3])):
            if isinstance(exc, sqlite3.DatabaseError) or "sqlite" in text.lower() or "database" in text.lower():
                site = ""
                if record.exc_info and record.exc_info[2] is not None:
                    frames = traceback.extract_tb(record.exc_info[2])
                    site = " <- ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in frames[-3:][::-1])
                self.corrupt.append(f"log[{record.name}:{record.levelname}] {text[:400]} @ {site}")


def is_corruption(exc):
    return isinstance(exc, sqlite3.DatabaseError) and not (
        isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower())


def site_of(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    return " <- ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in frames[-3:][::-1])


def check_integrity(db_path):
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        problems = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall() if r[0] != "ok"]
        fts = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts5%'")]
        for name in fts:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                conn.execute(f"INSERT INTO {quoted}({quoted}) VALUES('integrity-check')")
                conn.commit()
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    continue  # a busy writer is not corruption; next cycle re-checks
                problems.append(f"{name}: {exc!r}")
            except sqlite3.DatabaseError as exc:
                problems.append(f"{name}: {exc!r}")
        return problems, fts
    finally:
        conn.close()


def test_sqlite_ab_probe(monkeypatch):
    cycles = int(os.environ.get("LCM_SQLITE_AB_CYCLES", "200"))
    seed = int(os.environ.get("LCM_SQLITE_AB_SEED", "1"))
    keep_dir = os.environ.get("LCM_SQLITE_AB_KEEP_DIR")
    autockpt = os.environ.get("LCM_SQLITE_AB_WAL_AUTOCKPT")
    rng = random.Random(seed)
    work = Path(tempfile.mkdtemp(prefix="lcm-sqlite-ab-"))
    db_path = str(work / "lcm.db")

    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(escalation, "_call_llm_for_summary",
                        lambda *a, **k: "Earlier turns retained about ledger and zebra.\nExpand for details about: turns")
    monkeypatch.setattr(command_mod, "resolve_provider", lambda config, **_k: StubProvider(
        config.embedding_provider, config.embedding_model))
    monkeypatch.setattr(provider_mod, "probe_provider_availability",
                        lambda _config: {"available": True, "detail": "ab stub"})
    knobs = {"fresh_tail_count": 4, "leaf_chunk_tokens": 64, "embeddings_enabled": True,
             "embedding_provider": "ollama(stub)", "rollup_pass_after_each_compaction": True,
             "restart_every_cycles": 4, "wal_autocheckpoint": int(autockpt) if autockpt else 500}
    if autockpt:
        original = db_bootstrap.configure_connection

        def configure(conn):
            original(conn)
            conn.execute(f"PRAGMA wal_autocheckpoint={int(autockpt)}")

        monkeypatch.setattr(db_bootstrap, "configure_connection", configure)
        for name, module in list(__import__("sys").modules.items()):
            if name.startswith("hermes_lcm.") and getattr(module, "configure_connection", None) is original:
                monkeypatch.setattr(module, "configure_connection", configure)

    capture = Capture()
    lcm_logger = logging.getLogger("hermes_lcm")
    old_level = lcm_logger.level
    lcm_logger.setLevel(logging.DEBUG)
    lcm_logger.addHandler(capture)

    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=64,
                       embeddings_enabled=True, embedding_provider="ollama", embedding_model="ab-model")

    def start():
        engine = LCMEngine(config=config, hermes_home=str(work / "home"))
        engine.on_session_start(SESSION, platform="cli", conversation_id=CONVERSATION, context_length=200_000)
        return engine

    result = {"sqlite_version": sqlite3.sqlite_version, "cycles_run": 0, "first_failure_cycle": None,
              "error": None, "integrity": "ok", "seed": seed, "sqlite_module": sqlite3.__name__,
              "product_sqlite_version": db_bootstrap.sqlite3.sqlite_version,
              "wal_reset_affected": db_bootstrap.sqlite_wal_reset_affected(), "knobs": knobs}
    engine = None
    host, clock = [], time.time() - 40 * 86400
    ops = {"ingest": 0, "compress": 0, "restart": 0, "locked": 0}
    started = time.monotonic()
    try:
        engine = start()
        result["observed_wal_autocheckpoint"] = {
            "store": engine._store.connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
            "dag": engine._dag.connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0]}
        store = VectorStore(db_path, config=config)
        try:
            store.register_profile("ab-model", "ollama", 2)
            store.register_profile("ab-model", "ollama", 2, task="chunk")
        finally:
            store.close()
        for cycle in range(1, cycles + 1):
            try:
                if cycle % knobs["restart_every_cycles"] == 0:
                    engine.shutdown()  # no drain: background passes may still be writing
                    engine = start()
                    ops["restart"] += 1
                else:
                    engine.on_session_start(SESSION, platform="cli", conversation_id=CONVERSATION,
                                            context_length=200_000)
                for turn in range(rng.randint(3, 8)):
                    clock += rng.uniform(600, 9000)  # spans many UTC days, so rollups have work
                    text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 40)))
                    host.append({"role": "user", "content": f"c{cycle} t{turn} {text}", "timestamp": clock})
                    if rng.random() < 0.3:
                        call = {"id": f"call-{cycle}-{turn}", "type": "function",
                                "function": {"name": "lookup", "arguments": json.dumps({"q": text[:20]})}}
                        host.append({"role": "assistant", "content": "calling", "tool_calls": [call],
                                     "timestamp": clock + 1})
                        host.append({"role": "tool", "content": f"result {text[::-1]}", "tool_call_id": call["id"],
                                     "timestamp": clock + 2})
                    host.append({"role": "assistant", "content": f"answer {text} {cycle}", "timestamp": clock + 3})
                    engine.ingest(host)
                    ops["ingest"] += 1
                    if rng.random() < 0.45:
                        host = engine.compress(host, current_tokens=engine.threshold_tokens + 1, force=True)
                        ops["compress"] += 1
                        engine._schedule_rollup_maintenance(SESSION)
            except sqlite3.DatabaseError as exc:
                if not is_corruption(exc):
                    ops["locked"] += 1
                else:
                    result.update(first_failure_cycle=cycle, error=f"op: {exc!r} @ {site_of(exc)}")
            result["cycles_run"] = cycle
            if result["error"] is None and capture.corrupt:
                result.update(first_failure_cycle=cycle, error=capture.corrupt[0])
            problems, fts_tables = check_integrity(db_path)
            result["fts_tables"] = fts_tables
            if problems:
                result["integrity"] = "failed"
                result["integrity_problems"] = problems[:5]
                if result["error"] is None:
                    result.update(first_failure_cycle=cycle, error=f"integrity: {problems[0]}")
            if result["error"] is not None:
                break
    except Exception as exc:  # setup or non-DB failure: report, then fail the test below
        if result["error"] is None:
            result.update(first_failure_cycle=result["cycles_run"] + 1,
                          error=f"{'op' if is_corruption(exc) else 'harness'}: {exc!r} @ {site_of(exc)}")
    finally:
        try:
            if engine is not None:
                engine.shutdown()
            for name in _MAINTENANCE_SCHEDULERS:
                scheduler = getattr(engine_module, name, None)
                if scheduler is not None:
                    _wait_scheduler_idle(scheduler, 60.0)
            if result["error"] is None:
                problems, _ = check_integrity(db_path)
                if problems:
                    result.update(integrity="failed", integrity_problems=problems[:5],
                                  first_failure_cycle=result["cycles_run"], error=f"final integrity: {problems[0]}")
            conn = sqlite3.connect(db_path)
            try:
                def count(sql):
                    try:
                        return conn.execute(sql).fetchone()[0]
                    except sqlite3.DatabaseError as exc:
                        return repr(exc)
                result["workload"] = dict(ops, seconds=round(time.monotonic() - started, 1),
                    messages=count("SELECT count(*) FROM messages"),
                    summary_nodes=count("SELECT count(*) FROM summary_nodes"),
                    rollups=count("SELECT count(*) FROM lcm_rollups"),
                    rollups_ready=count("SELECT count(*) FROM lcm_rollups WHERE status='ready'"),
                    summary_vectors=count("SELECT count(*) FROM lcm_embedding_meta"),
                    chunk_vectors=count("SELECT count(*) FROM lcm_chunk_meta"),
                    log_lock_errors=capture.locked)
            finally:
                conn.close()
        finally:
            lcm_logger.removeHandler(capture)
            lcm_logger.setLevel(old_level)
            if result["error"] is not None and keep_dir:
                dest = Path(keep_dir) / f"{work.name}-seed{seed}-v{sqlite3.sqlite_version}"
                shutil.copytree(work, dest, dirs_exist_ok=True)
                result["kept"] = str(dest)
            shutil.rmtree(work, ignore_errors=True)
            print("SQLITE_AB_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    assert result["error"] is None and result["integrity"] == "ok", result
