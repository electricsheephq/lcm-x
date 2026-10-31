"""Paired synthetic admission timings, with preparation outside measured gates."""
import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import textwrap
import time
import types


def refuse_profile(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        value, forbidden = os.fsdecode(args[0]), str(Path.home() / ".hermes")
        if value == forbidden or value.startswith(forbidden + "/"):
            raise RuntimeError("forbidden live-profile access")


def load(name, root):
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    engine = importlib.import_module(name + ".engine")
    config = importlib.import_module(name + ".config")
    tokens = importlib.import_module(name + ".tokens")
    tokens._encoder_ready, tokens._encoder = True, None
    tokens._get_encoder = lambda: None
    importlib.import_module(name + ".survival_fit")._host_estimate = lambda messages: None
    importlib.import_module(name + ".db_bootstrap").dispatch_background_stem_backfill = lambda *a, **kw: None
    engine.summarize_with_escalation = lambda **kw: (_ for _ in ()).throw(AssertionError("model call"))
    return engine, config


def make(modules, path, count, seed=True):
    path.mkdir(parents=True, exist_ok=True)
    engine_mod, config_mod = modules
    config = config_mod.LCMConfig(database_path=str(path / "lcm.db"), embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
        fresh_tail_pressure_yield_enabled=False, context_threshold=0.5,
        fresh_tail_count=2, leaf_chunk_tokens=400)
    engine = engine_mod.LCMEngine(config=config, hermes_home=str(path / "home"))
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    if seed:
        conn = engine._store.connection
        conn.executemany("INSERT INTO messages(session_id,conversation_id,role,content,timestamp,token_estimate) "
                         "VALUES('S','conv','user',?,?,8)",
                         ((f"synthetic immutable source {i}", float(i + 1)) for i in range(count - 1)))
        conn.commit()
    replay = [{"role": "user", "content": "synthetic active replay", "timestamp": 1.0}]
    # Genuine current-ingest proof is preparation, outside every measured gate.
    engine.ingest(replay)
    assert engine._consecutive_ingest_failures == 0
    assert engine._store.get_session_count("S") == count
    return engine


def measure(fn):
    begin = time.perf_counter_ns()
    value = fn()
    return (time.perf_counter_ns() - begin) / 1e6, value


def stats(values):
    return dict(n=len(values), p50_ms=statistics.median(values),
                p95_ms=sorted(values)[int((len(values) - 1) * 0.95)])


def main():
    p = argparse.ArgumentParser()
    for flag in ("base", "candidate", "scratch", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    sys.addaudithook(refuse_profile)
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(args.scratch / "synthetic-home")
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    # Use the repository's exact dependency-free CI host contract.
    workflow = (args.candidate / ".github/workflows/ci.yml").read_text()
    stub = workflow.split("cat > agent/context_engine.py <<'STUB'", 1)[1].split("\n          STUB", 1)[0]
    stub_root = args.scratch / "host-stub"
    (stub_root / "agent").mkdir(parents=True, exist_ok=True)
    (stub_root / "agent/__init__.py").write_text("")
    (stub_root / "agent/context_engine.py").write_text(textwrap.dedent(stub).lstrip())
    sys.path.insert(0, str(stub_root))
    baseline, candidate = load("baseline_lcm", args.base), load("candidate_lcm", args.candidate)
    key = "compaction_no_progress:v1:S"
    samples, summaries = [], []
    for count in (10_000, 100_000):
        pair = [make(baseline, args.scratch / f"base-{count}", count),
                make(candidate, args.scratch / f"candidate-{count}", count)]
        try:
            for mode in ("absent", "mismatch"):
                series = [[], []]
                if mode == "mismatch":
                    now = time.time()
                    record = dict(version=1, fingerprint="0" * 64, reason="no_progress",
                                  recorded_at=now, expires_at=now + 600)
                    for engine in pair:
                        engine._store.write_metadata_json([key], json.dumps(record))
                changes = [engine._store.connection.total_changes for engine in pair]
                for i in range(220):
                    for j in ([0, 1] if i % 2 == 0 else [1, 0]):
                        elapsed, result = measure(lambda: pair[j].should_compress(110_000))
                        assert result is True
                        if i >= 20:
                            series[j].append(elapsed)
                            samples.append(dict(rows=count, mode=mode, pair=i-20, arm=j, ms=elapsed))
                assert [engine._store.connection.total_changes for engine in pair] == changes
                if mode == "mismatch":
                    assert pair[1]._store.read_metadata_json(key) == record
                base, cand = stats(series[0]), stats(series[1])
                limit = max(base["p50_ms"] * 1.10, base["p50_ms"] + 0.05)
                summaries.append(dict(rows=count, mode=mode, base=base, candidate=cand,
                                      limit_ms=limit, passed=cand["p50_ms"] <= limit))
            held = pair[1]
            # Synthetic completed-proof setup is outside measured admission.
            held._start_no_progress_hold("no_progress", persist=True)
            statements = []
            held._store.connection.set_trace_callback(statements.append)
            held._dag._conn.set_trace_callback(statements.append)
            held._lifecycle._conn.set_trace_callback(statements.append)
            values = []
            for i in range(200):
                elapsed, result = measure(lambda: held.should_compress(110_000))
                assert result is False
                values.append(elapsed)
                samples.append(dict(rows=count, mode="held", pair=i, arm=1, ms=elapsed))
            held._store.connection.set_trace_callback(None)
            held._dag._conn.set_trace_callback(None)
            held._lifecycle._conn.set_trace_callback(None)
            sql = sorted(set(statements))
            assert not any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "BEGIN", "COMMIT")) for s in sql)
            plans = []
            for query in sql:
                if query.lstrip().upper().startswith("SELECT"):
                    plan = [list(row) for row in held._store.connection.execute("EXPLAIN QUERY PLAN " + query)]
                    assert not any("SCAN messages" in str(row) or "SCAN summary_nodes" in str(row)
                                   for row in plan), plan
                    plans.append(dict(query=query, plan=plan))
            summaries.append(dict(rows=count, mode="held", candidate=stats(values), model_calls=0,
                                  metadata_writes=0, full_history_scans=0, query_plans=plans))
            cold = []
            for i in range(20):
                rebuilt = make(candidate, args.scratch / f"candidate-{count}", count, seed=False)
                try:
                    changes = rebuilt._store.connection.total_changes
                    elapsed, result = measure(lambda: rebuilt.should_compress(110_000))
                    assert result is False
                    assert rebuilt._store.connection.total_changes == changes
                    cold.append(elapsed)
                    samples.append(dict(rows=count, mode="rebuilt-held", pair=i, arm=1, ms=elapsed))
                finally:
                    rebuilt.shutdown()
            summaries.append(dict(rows=count, mode="rebuilt-held", candidate=stats(cold),
                                  model_calls=0, metadata_writes=0))
        finally:
            for engine in pair:
                engine.shutdown()
    (args.out / "samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in samples))
    result = dict(protocol="1880-admission-v2-steady", warmup=20, paired_n=200,
                  baseline_sha="10407e12e33c1d7b09fe0e05141153555aec5d62", cells=summaries,
                  passed=all(row.get("passed", True) for row in summaries),
                  boundary="Gate only; seed/ingest/rebuild excluded; synthetic data; no runtime latency claim")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
