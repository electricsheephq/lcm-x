"""One isolated engine process. Only the provider is stubbed; prep/publication are real."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import sqlite3
import sys
import threading
import time
import types

VERSION = "bgprep-v1"
TURNS = 64
SETTINGS = dict(
    fresh_tail_count=2, leaf_chunk_tokens=2000, context_threshold=.75,
    threshold_full_sweep_enabled=True, large_output_externalization_enabled=False,
    large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
    embeddings_enabled=False, l3_truncate_tokens=2, incremental_max_depth=0,
    background_leaf_prep_max_calls=64, summary_spend_max_calls=0,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(row, *, timestamp=True):
    value = {"role": row["role"], "content": row.get("content") or ""}
    if timestamp:
        value["timestamp"] = row.get("timestamp")
    return json.dumps(value, sort_keys=True)


def workload(seed):
    """64 ordinary user/reply pairs, ~100k tokens: repeated crossings after first publication."""
    rng = random.Random(seed)
    return [[{"role": role, "content": f"[seed={seed};turn={turn};role={role}] " + " ".join(
        rng.choice(("alpha", "beta", "gamma", "delta", "epsilon")) for _ in range(400)),
        "timestamp": float(turn)} for role in ("user", "assistant")] for turn in range(1, TURNS + 1)]


def run_cell(args, root, arm, seed):
    # Every cell is a new subprocess. No live profile config or inherited LCM settings.
    for key in list(os.environ):
        if key.startswith("LCM_"):
            del os.environ[key]
    home = root / "home"
    home.mkdir()
    os.environ.update(HERMES_HOME=str(home), LCM_HERMES_BASE_DIR=str(root))
    if not args.real_route:
        def denied(*a, **kw):
            raise RuntimeError("network denied in deterministic stub cell")
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
    sys.path.insert(0, str(args.hermes_source))
    pkg = types.ModuleType("hermes_lcm")
    pkg.__path__ = [str(args.product_source)]
    sys.modules["hermes_lcm"] = pkg  # avoid plugin registration, exactly as repository tests do
    # Pin the deterministic slice-1 token estimator; no model downloads or host model guessing.
    tokens = importlib.import_module("hermes_lcm.tokens")
    tokens.count_tokens = lambda text: 0 if not text else tokens._fallback_token_estimate(text)
    engine_module, config, prep, escalation = [importlib.import_module("hermes_lcm." + n)
        for n in ("engine", "config", "leaf_prep", "escalation")]
    importlib.import_module("hermes_lcm.survival_fit")._host_estimate = lambda messages: None
    counts, calls, lock = Counter(), [], threading.Lock()

    class ObservedPrep(prep._LeafPrep):
        def enqueue(self, job):
            counts["enqueue_attempts"] += 1
            previous = self._worker
            result = super().enqueue(job)
            counts["worker_constructions"] += self._worker is not previous
            return result

        def _run(self):
            counts["worker_starts"] += 1
            return super()._run()

        def promoted(self, *a, **kw):
            counts["promotions"] += 1
            return super().promoted(*a, **kw)

    state = prep._LEAF_PREP = ObservedPrep()
    original_ready = prep.ready_chain

    def ready(*a, **kw):
        counts["promotion_lookups"] += 1
        return original_ready(*a, **kw)

    prep.ready_chain = ready
    original_provider = escalation._invoke_summary_llm

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        worker = threading.current_thread().name == "lcm-leaf-prep"
        start = time.monotonic()
        call = dict(worker=worker, prompt=prompt)
        with lock:
            counts["worker_summary_calls" if worker else "foreground_summary_calls"] += 1
            calls.append(call)
        try:
            result = (original_provider(prompt, max_tokens, model=model, timeout=timeout, reasoning_effort=reasoning_effort)
                      if args.real_route else "Earlier turns " + digest(prompt)[:16] + ".\nExpand for details about: turns")
            call["result"] = result
            return result
        except Exception as exc:
            call["exception_type"] = type(exc).__name__  # never copy provider auth/error bodies
            raise
        finally:
            call["wall_s"] = time.monotonic() - start

    escalation._invoke_summary_llm = provider
    cfg = config.LCMConfig(database_path=str(home / "lcm.db"), background_leaf_prep_enabled=arm == "on",
                           summary_model=args.summary_model, **SETTINGS)
    dump(root / "config.json", asdict(cfg))
    engine = engine_module.LCMEngine(config=cfg, hermes_home=str(home))
    engine.on_session_start("bgprep", platform="eval", conversation_id="bgprep", context_length=60000)
    def counted_leaf(operation):
        def invoke(*a, **kw):
            counts["leaf_compactions"] += 1  # one leaf operation, not one provider retry
            return operation(*a, **kw)
        return invoke

    engine._summarize_leaf_chunk_with_rescue = counted_leaf(engine._summarize_leaf_chunk_with_rescue)
    prepared_summary = engine._prepared_leaf_summary
    engine._prepared_leaf_summary = lambda leaf: counted_leaf(prepared_summary(leaf))
    turns = workload(seed)
    dump(root / "workload.json", turns)
    probes, events, expected, view, previous_end = [], [], [], [], None

    def activity():
        return {k: int(counts[k] if k != "jobs_enqueued" else state.counters[k]) for k in (
            "worker_starts", "worker_constructions", "enqueue_attempts", "jobs_enqueued",
            "worker_summary_calls", "foreground_summary_calls", "promotions", "promotion_lookups")}

    def probe(turn, phase):
        with sqlite3.connect(f"file:{home / 'lcm.db'}?mode=ro", uri=True) as db:
            rows = [dict(store_id=i, identity=identity(dict(role=r, content=c, timestamp=t)),
                         active_identity=identity(dict(role=r, content=c), timestamp=False)) for i, r, c, t in
                    db.execute("SELECT store_id, role, content, observed_at FROM messages ORDER BY store_id")]
            leaves = [json.loads(r[0]) for r in db.execute(
                "SELECT source_ids FROM summary_nodes WHERE depth=0 AND source_type='messages' ORDER BY node_id")]
        frontier = engine._lifecycle.get_by_conversation("bgprep").current_frontier_store_id
        live = []
        for msg in view:
            remainder = engine._generated_context_carrier_remainder(msg)
            live.append(identity({**msg, "content": remainder} if remainder is not None else msg, timestamp=False))
        p = dict(turn=turn, phase=phase, expected=list(expected), rows=rows, leaf_sources=leaves,
                 frontier=frontier, active_view=live, activity=activity())
        probes.append(p)
        # Persist observations before any scorer reads them, even if a later turn fails.
        with (root / "probes.jsonl").open("a") as out:
            out.write(json.dumps(p) + "\n")

    started = time.monotonic()
    try:
        for turn, (user, assistant) in enumerate(turns, 1):
            if previous_end is not None:
                remaining = args.gap_seconds - (time.monotonic() - previous_end)
                if remaining > 0:
                    time.sleep(remaining)  # actual wall time, never token/time substitution
            actual_gap = time.monotonic() - previous_end if previous_end is not None else None
            view.append(user)
            expected.append(identity(user))
            start = time.monotonic()
            engine.ingest(view)
            cur = tokens.count_messages_tokens(view)
            crossing = cur >= engine.threshold_tokens
            before = {n.node_id for n in engine._dag.get_session_nodes("bgprep", limit=100000)}
            promoted = counts["promotions"]
            ran = counts["leaf_compactions"]
            if crossing:
                view = engine.compress(view, current_tokens=cur)
            nodes = engine._dag.get_session_nodes("bgprep", limit=100000)
            events.append(dict(turn=turn, crossing=crossing, tokens=cur, threshold=engine.threshold_tokens,
                               leaves=sum(n.depth == 0 and n.node_id not in before for n in nodes),
                               leaf_compactions=counts["leaf_compactions"] - ran,
                               promoted=counts["promotions"] - promoted, visible_wait_s=time.monotonic() - start,
                               gap_s=actual_gap, status=engine.last_compression_status))
            probe(turn, "pre-response")
            view.append(assistant)
            expected.append(identity(assistant))
            engine.ingest(view)
            engine.last_prompt_tokens = tokens.count_messages_tokens(view)
            engine.note_turn_complete()
            engine._prepare_leaves_after_turn(view)
            previous_end = time.monotonic()
            if args.gap_seconds == 0 and not state.wait_idle(30):
                raise RuntimeError("zero-gap wiring synchronization timed out")
            probe(turn, "post-turn")
            print(f"{arm} seed={seed} turn={turn}/{TURNS} crossing={crossing}", flush=True)
        if not state.wait_idle(180):
            raise RuntimeError("worker teardown timeout")
        probe(TURNS, "drained")
        with sqlite3.connect(f"file:{home / 'lcm.db'}?mode=ro", uri=True) as src, sqlite3.connect(root / "snapshot.db") as dst:
            src.backup(dst)
        dump(root / "raw.json", dict(arm=arm, seed=seed, workload_sha=digest(turns), probes=probes, events=events,
                                     calls=calls, wall_s=time.monotonic() - started,
                                     threshold=engine.threshold_tokens))
    finally:
        engine.shutdown()
