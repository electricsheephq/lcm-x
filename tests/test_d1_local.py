"""D1 instrument proofs: deterministic fakes only, no FastEmbed/model downloads."""
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarking.longmemeval as lme
from benchmarking.d1_cache import D1Cache, seal

OPTIONS = {"dimension": 64, "normalization": "fake-unit", "mode": "document"}


class Fake(lme.StubEmbedder):
    def __init__(self):
        super().__init__()
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return super().embed_documents(texts)


def questions():
    return [lme.parse_question({
        "question_id": f"q{i}", "question_type": "single-session-user",
        "question": f"What is the code for project {i}?", "answer": f"code{i}",
        "question_date": "2023-01-01", "haystack_dates": ["2023-01-01"] * 3,
        "haystack_session_ids": [f"s{j}" for j in range(3)],
        "haystack_sessions": [[{"role": "user", "content": f"project {i} code{i}", "has_answer": True}],
                              [{"role": "user", "content": "Shared distractor about lunch"}],
                              [{"role": "assistant", "content": "Shared distractor about travel"}]],
        "answer_session_ids": ["s0"],
    }) for i in range(10)]


def test_ten_question_equivalence(tmp_path, monkeypatch):
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    raw = Fake()
    path = tmp_path / "cache.sqlite"
    builder = D1Cache(raw, path, options=OPTIONS, readonly=False)
    report = lme.prewarm_embedding_cache(questions(), builder)
    assert report["unique_request_units"] == len(set(raw.documents))
    assert len(raw.documents) == len(set(raw.documents))  # pre-embed each text once
    manifest_hash = seal(path)
    before = path.read_bytes()
    cached_raw = Fake()
    cached = D1Cache(cached_raw, path, options=OPTIONS)
    baseline_dir, cached_dir = tmp_path / "baseline", tmp_path / "cached"
    baseline_dir.mkdir()
    cached_dir.mkdir()
    proof = []
    for question in questions():
        baseline = lme.evaluate_question(question, Fake(), provider_name="stub", tmp_dir=baseline_dir, include_rankings=True, embeddings_enabled=True)
        actual = lme.evaluate_question(question, cached, provider_name="stub", tmp_dir=cached_dir, include_rankings=True, embeddings_enabled=True)
        assert actual["_candidate_rankings"] == baseline["_candidate_rankings"]
        for arm in lme.ARMS:
            for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10", "turn"):
                assert actual[arm][metric] == baseline[arm][metric], (question.question_id, arm, metric)
        assert actual["embed_cache"]["hits"] > 0
        assert actual["embed_cache"]["misses"] == 0
        assert actual["embedding"]["wall_ms"] > 0
        assert actual["embedding"]["manifest_sha256"] == manifest_hash
        record = lme._question_checkpoint_record(question, actual)
        assert "embedding" not in record
        assert "embedding" not in record["arms"]
        proof.append({"question_id": question.question_id,
                      "baseline": {arm: {key: value for key, value in baseline[arm].items() if key != "latency_ms"} for arm in lme.ARMS},
                      "cached": {arm: {key: value for key, value in actual[arm].items() if key != "latency_ms"} for arm in lme.ARMS},
                      "rankings": actual["_candidate_rankings"]})
    assert not cached_raw.documents
    assert path.read_bytes() == before
    if os.environ.get("D1_EQUIVALENCE_REPORT"):
        Path(os.environ["D1_EQUIVALENCE_REPORT"]).write_text(json.dumps(proof, indent=2) + "\n")


def test_corrupt_entry_detected_recomputed_and_counted(tmp_path):
    path = tmp_path / "cache.sqlite"
    builder = D1Cache(Fake(), path, options=OPTIONS, readonly=False)
    expected = builder.embed_documents(["alpha", "beta"])
    seal(path)
    path.chmod(0o644)  # deliberate mutation of a test-owned fixture
    with sqlite3.connect(path) as db:
        db.execute("UPDATE entries SET payload=zeroblob(length(payload)) WHERE key=?", (builder.content_sha256("alpha"),))
    path.chmod(0o444)
    before = path.read_bytes()
    raw = Fake()
    cached = D1Cache(raw, path, options=OPTIONS)
    assert cached.embed_documents(["alpha", "beta", "a miss"])[:2] == expected
    assert raw.documents == ["alpha", "a miss"]
    assert (cached.hits, cached.misses, cached.corruptions) == (1, 2, 1)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["model", "dimension", "normalization", "mode", "product_sha", "role"])
def test_model_and_options_cannot_cross(tmp_path, change):
    path = tmp_path / "cache.sqlite"
    D1Cache(Fake(), path, options=OPTIONS, readonly=False).embed_documents(["alpha"])
    seal(path)
    raw, options = Fake(), dict(OPTIONS)
    if change == "model":
        raw.model_id = "different-model"
    else:
        options[change] = "different"
    with pytest.raises(ValueError, match="model/options mismatch"):
        D1Cache(raw, path, options=options)
    assert not raw.documents


def test_readers_share_sealed_cache(tmp_path):
    path = tmp_path / "cache.sqlite"
    expected = D1Cache(Fake(), path, options=OPTIONS, readonly=False).embed_documents(["alpha"])
    seal(path)
    def read(_):
        return D1Cache(Fake(), path, options=OPTIONS).embed_documents(["alpha"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(read, range(6))) == [expected] * 6


def driver():
    spec = importlib.util.spec_from_file_location("d1_driver", Path(__file__).resolve().parents[1] / "scripts/lcm_d1_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", ["exit", "torn", "timeout", "older", "budget", "pretimeout",
                                  "timeout_lookup", "timeout_permission", "pretimeout_lookup", "pretimeout_permission"])
def test_parallel_driver_receipts_and_failure_isolation(tmp_path, monkeypatch, fault):
    cli = driver()
    if "_" in fault:
        fault, exception = fault.split("_")
        killpg = cli.os.killpg
        def raced_kill(pid, sig):
            killpg(pid, sig)  # Model a worker disappearing just before kill.
            if sig == signal.SIGKILL:
                raise {"lookup": ProcessLookupError, "permission": PermissionError}[exception]()
        monkeypatch.setattr(cli.os, "killpg", raced_kill)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli, "pin", lambda tree: ("b" if str(tree).endswith("candidate-tree") else "a") * 40)
    shards = tmp_path / "shards"
    qid = 0
    for k, size in enumerate([85] * 5 + [75]):
        directory = shards / f"shard-{k}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"questions": [{"question_id": f"q{i}"} for i in range(qid, qid + size)]}))
        qid += size
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [{"question_id": f"q{i}"} for i in range(500)]}))
    aa = tmp_path / "aa"
    aa.mkdir()
    (aa / "manifest.json").write_text(json.dumps({"questions": [{"question_id": f"q{i}"} for i in range(100)]}))
    lock = threading.Lock()
    joined = threading.Condition(lock)
    active, peak, roots, caps = 0, 0, [], []

    real_run_child = cli.run_child

    def fake_run(command, *, timeout, cwd, env, stdout, stderr, interrupt):
        nonlocal active, peak
        cache = Path(env[lme.EMBED_CACHE_ENV])
        if "prewarm-cache" in command:
            assert env["LCM_D1_CACHE_MODE"] == "build"
            options = {**OPTIONS, "product_sha": env["LCM_D1_PRODUCT_SHA"], "role": env["LCM_D1_ROLE"]}
            D1Cache(Fake(), cache, options=options, readonly=False).embed_documents(["shared"])
            Path(env["LCM_D1_THREAD_REPORT"]).write_text(json.dumps({
                "thread_cap_applied": fault != "older", "effective_intra_op_threads": 4 if fault == "budget" else (None if fault == "older" else 2),
                "reason": "unavailable" if fault == "older" else None,
            }))
            if fault == "pretimeout":
                return real_run_child([sys.executable, "-c", "import time; time.sleep(60)"], timeout=timeout, stdout=stdout, stderr=stderr)
            return {"exit_code": 0, "timed_out": False}
        assert env["LCM_D1_CACHE_MODE"] == "read"
        if fault == "timeout" and "shard-1" in env["TMPDIR"]:
            return real_run_child([sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"], timeout=timeout, stdout=stdout, stderr=stderr)
        with joined:
            active += 1
            peak = max(peak, active)
            roots.append((env["HERMES_HOME"], env["TMPDIR"]))
            caps.append(int(env["LCM_D1_ORT_THREADS"]))
            joined.notify_all()
            # Hold each worker until three have overlapped once, so the peak never depends on CI timing.
            joined.wait_for(lambda: peak >= 3, timeout=10)
        out = Path(command[command.index("--output") + 1])
        out.mkdir(parents=True)
        row = {"question_id": "q0", "embed_cache": {"hits": 2, "misses": 1}}
        torn = '{"question_id":' if fault == "torn" and out.name == "shard-1" else ""
        (out / "per_question_checkpoint.jsonl").write_text(json.dumps(row) + "\n" + torn)
        (out / "embedding-metrics.jsonl").write_text(json.dumps({
            "question_id": "q0", "wall_ms": 4, "hits": 2, "misses": 1, "corruptions": 1,
        }) + "\n")
        with lock:
            active -= 1
        return {"exit_code": int(fault == "exit" and out.name == "shard-1"), "timed_out": False}

    monkeypatch.setattr(cli, "run_child", fake_run)
    out = tmp_path / "output"
    assert cli.main(["--product", "base=base-tree", "--product", "candidate=candidate-tree",
                     "--prepared-dir", str(tmp_path), "--shards-dir", str(shards),
                     "--aprime-dir", str(aa), "--output", str(out), "--scratch", str(tmp_path / "scratch"), "--worker-timeout", "0.05"]) == int(fault != "older")
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    receipt = json.loads((out / "d1-receipt.json").read_text())
    if os.environ.get("D1_DRIVER_REPORT_DIR"):
        Path(os.environ["D1_DRIVER_REPORT_DIR"], f"driver-{fault}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["fresh_output_guard_already_existed"] is True
    if fault in {"budget", "pretimeout"}:
        assert not receipt["arms"]
        assert not receipt["thread_budget_ok"] if fault == "budget" else all(row["timed_out"] and row["exit_code"] == 124 for row in receipt["preembed"])
        return
    if fault == "older":
        assert all(not role["thread_cap_applied"] and role["effective_intra_op_threads"] is None for role in receipt["role_threads"].values())
    assert len(receipt["arms"]) == 16
    assert {(arm["role"], arm["arm"]) for arm in receipt["arms"]} == {
        (role, name) for role in ("base", "candidate")
        for name in [*(f"shard-{i}" for i in range(6)), "aa-A", "aa-Aprime"]
    }
    assert sum(arm["exit_code"] for arm in receipt["arms"]) == (2 if fault == "exit" else 248 if fault == "timeout" else 0)
    assert sum(bool(arm["parse_error"]) for arm in receipt["arms"]) == (2 if fault == "torn" else 0)
    assert peak == 3
    assert peak * max(caps) <= receipt["thread_budget"] == 8
    assert len(set(roots)) == len(roots) == (14 if fault == "timeout" else 16)
    assert receipt["products"] == {"base": "a" * 40, "candidate": "b" * 40}
    assert len(set(receipt["cache_manifests"].values())) == 2
    for arm in receipt["arms"]:
        assert arm["product_sha"] == receipt["products"][arm["role"]]
        assert arm["cache_manifest_sha256"] == receipt["cache_manifests"][arm["role"]]
        assert arm["cache_manifest_sha256"] == hashlib.sha256((out / arm["role"] / "embeddings.manifest.json").read_bytes()).hexdigest()
        if arm["timed_out"]:
            assert arm["exit_code"] == 124 and not arm["parse_error"]
            continue
        if arm["parse_error"]:
            assert "JSONDecodeError" in arm["parse_error"]
            continue
        assert arm["cache_hit_rate"] == 2 / 3
        assert arm["cache_corruptions"] == 1
        assert arm["embedding_wall_ms"] == 4


def test_harness_cache_env_shim(tmp_path, monkeypatch):
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "build")
    monkeypatch.setenv("LCM_D1_PRODUCT_SHA", "a" * 40)
    monkeypatch.setenv("LCM_D1_ROLE", "base")
    builder = lme.resolve_harness_provider("stub", "")
    assert isinstance(builder, D1Cache)
    expected = builder.embed_documents(["exact text"])
    seal(builder.cache_path)
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    assert lme.resolve_harness_provider("stub", "").embed_documents(["exact text"]) == expected
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "typo")
    with pytest.raises(ValueError, match="mode must be"):
        lme.resolve_harness_provider("stub", "")


def test_role_bound_caches_score_with_their_own_vectors(tmp_path, monkeypatch):
    class Candidate(Fake):
        def embed_documents(self, texts):
            return [[-x for x in vector] for vector in super().embed_documents(texts)]

    question = questions()[0]
    rankings, vectors, keys = {}, {}, {}
    for role, sha, factory in (("base", "a" * 40, Fake), ("candidate", "b" * 40, Candidate)):
        path = tmp_path / role / "cache.sqlite"
        path.parent.mkdir()
        monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(path))
        monkeypatch.setenv("LCM_D1_CACHE_MODE", "build")
        monkeypatch.setenv("LCM_D1_PRODUCT_SHA", sha)
        monkeypatch.setenv("LCM_D1_ROLE", role)
        monkeypatch.setattr(lme, "StubEmbedder", factory)
        builder = lme.resolve_harness_provider("stub", "")
        assert builder.options["product_sha"] == sha
        assert builder.options["role"] == role
        lme.prewarm_embedding_cache([question], builder)
        vectors[role] = builder.embed_documents(["shared text"])
        keys[role] = builder.content_sha256("shared text")
        seal(path)
        baseline = lme.evaluate_question(question, factory(), provider_name="stub", tmp_dir=tmp_path / f"{role}-raw", include_rankings=True, embeddings_enabled=True)
        monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
        cached = lme.resolve_harness_provider("stub", "")
        actual = lme.evaluate_question(question, cached, provider_name="stub", tmp_dir=tmp_path / f"{role}-cached", include_rankings=True, embeddings_enabled=True)
        assert actual["_candidate_rankings"] == baseline["_candidate_rankings"]
        for arm in lme.ARMS:
            for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10", "turn"):
                assert actual[arm][metric] == baseline[arm][metric]
        assert cached.embed_documents(["shared text"]) == vectors[role]
        assert cached.hits > 0 and cached.misses == 0
        assert not cached._provider.documents
        rankings[role] = actual["_candidate_rankings"]
    assert keys["base"] != keys["candidate"]
    assert vectors["base"] != vectors["candidate"]
    assert rankings["base"] != rankings["candidate"]


@pytest.mark.parametrize("dirty", [False, True])
def test_pin_resolves_head_and_refuses_dirty_tree(tmp_path, monkeypatch, dirty):
    cli = driver()
    calls = []

    def git(command, **kwargs):
        assert kwargs["timeout"] == 60
        calls.append(command)
        if command[-1] == "--porcelain":
            return b" M embedding_provider.py\n" if dirty else b""
        assert command[-2:] == ["rev-parse", "HEAD"]
        return "a" * 40 + "\n"

    monkeypatch.setattr(cli.subprocess, "check_output", git)
    if dirty:
        with pytest.raises(ValueError, match="clean product"):
            cli.pin(tmp_path)
        assert len(calls) == 1
    else:
        assert cli.pin(tmp_path) == "a" * 40
        assert calls[-1] == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]


def test_cached_aa_checkpoint_compatible_with_release_scorer(tmp_path, monkeypatch):
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LCM_D1_PRODUCT_SHA", "a" * 40)
    monkeypatch.setenv("LCM_D1_ROLE", "base")
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "build")
    builder = lme.resolve_harness_provider("stub", "")
    lme.prewarm_embedding_cache(questions()[:2], builder)
    manifest = seal(builder.cache_path)
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    evaluate = lme.evaluate_question
    wall_ms = 10

    def timed(*args, **kwargs):
        scored = evaluate(*args, **kwargs)
        scored["embedding"]["wall_ms"] = wall_ms  # deterministic, different arm timings
        return scored

    monkeypatch.setattr(lme, "evaluate_question", timed)
    runs, timings = [], []
    for name in ("A", "Aprime"):
        checkpoint = tmp_path / name / "per_question_checkpoint.jsonl"
        lme.run_harness(questions()[:2], provider_name="stub", model="", tmp_dir=tmp_path / f"{name}-db",
                        checkpoint_path=checkpoint, reuse_db_template=False)
        runs.append({r["question_id"]: r for r in map(json.loads, checkpoint.read_text().splitlines()) if "question_id" in r})
        metrics = list(map(json.loads, checkpoint.with_name("embedding-metrics.jsonl").read_text().splitlines()))
        assert {r["question_id"] for r in metrics} == set(runs[-1])
        assert all(r["hits"] > 0 and r["misses"] == 0 and r["manifest_sha256"] == manifest for r in metrics)
        timings.append([r["wall_ms"] for r in metrics])
        wall_ms += 10

    # Minimal comparison from the v0.29.0-rc2 release kit's u-off/aa.py,
    # VOLATILE/strip and disc_rows: only latency_ms and ingest_ms are volatile.
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in {"latency_ms", "ingest_ms"}}
        return value

    a, b = runs
    discordant = sorted(q for q in set(a) | set(b) if strip(a.get(q)) != strip(b.get(q)))
    assert timings == [[10, 10], [20, 20]]
    assert discordant == []
    assert all("embedding" not in r and "embedding" not in r["arms"] for run in runs for r in run.values())


def test_fastembed_thread_cap_without_loading_model(tmp_path, monkeypatch):
    lme._ensure_hermes_lcm_package()
    from hermes_lcm import embedding_provider as ep, embedding_maintenance as em
    class FakeFastembed(Fake):
        def __init__(self, model, **kwargs):
            super().__init__()
            self.model_id = model
        def _cache_key(self):
            return self.model_id, "fake-cache", 4
        def warmup(self):
            self._cache_key()
            return self.embed_query("warmup")
    monkeypatch.setattr(ep, "FastembedProvider", FakeFastembed)
    monkeypatch.setattr(em, "note_local_model_present", lambda _: None)
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    monkeypatch.setenv("LCM_D1_ORT_THREADS", "2")
    report = tmp_path / "threads.json"
    monkeypatch.setenv("LCM_D1_THREAD_REPORT", str(report))
    provider = lme.resolve_harness_provider("fastembed", "fake-model")
    assert provider._cache_key()[2] == 2
    assert json.loads(report.read_text()) == {"thread_cap_applied": True, "effective_intra_op_threads": 2, "reason": None}


def test_missing_sealed_row_is_corruption(tmp_path):
    path = tmp_path / "cache.sqlite"
    builder = D1Cache(Fake(), path, options=OPTIONS, readonly=False)
    expected = builder.embed_documents(["alpha"])
    seal(path)
    path.chmod(0o644)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM entries WHERE key=?", (builder.content_sha256("alpha"),))
    path.chmod(0o444)
    before = path.read_bytes()
    raw = Fake()
    cached = D1Cache(raw, path, options=OPTIONS)
    assert cached.embed_documents(["alpha", "ordinary miss"])[:1] == expected
    assert (cached.hits, cached.misses, cached.corruptions) == (0, 2, 1)
    assert raw.documents == ["alpha", "ordinary miss"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["mismatch", "duplicate", "aprime", "output"])
def test_driver_refuses_inputs_before_workers(tmp_path, monkeypatch, capsys, fault):
    cli = driver()
    monkeypatch.setattr(cli, "pin", lambda _: "a" * 40)
    def manifest(directory, ids):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"questions": [{"question_id": q} for q in ids]}))
    ids = [f"q{i}" for i in range(500)]
    manifest(tmp_path / "prepared", ids)
    for k in range(6):
        selected = ids[k::6]
        if k == 0 and fault in {"mismatch", "duplicate"}:
            selected[0] = "other" if fault == "mismatch" else "q1"
        manifest(tmp_path / "shards" / f"shard-{k}", selected)
    manifest(tmp_path / "aa", ids[:99] + (["other"] if fault == "aprime" else [ids[99]]))
    out = tmp_path / "out"
    if fault == "output":
        out.mkdir()
        (out / "existing").write_text("preserve")
    monkeypatch.setattr(cli, "run_child", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--product", "base=base", "--product", "candidate=candidate",
                  "--prepared-dir", str(tmp_path / "prepared"), "--shards-dir", str(tmp_path / "shards"),
                  "--aprime-dir", str(tmp_path / "aa"), "--output", str(out), "--scratch", str(tmp_path / "scratch")])
    error = capsys.readouterr().err
    assert {"mismatch": "partition", "duplicate": "duplicate question", "aprime": "subset", "output": "fresh directory"}[fault] in error
    if fault == "output":
        assert (out / "existing").read_text() == "preserve"


@pytest.mark.parametrize("effective", [None, 1])
def test_older_provider_proceeds_without_hook(tmp_path, monkeypatch, effective):
    lme._ensure_hermes_lcm_package()
    from hermes_lcm import embedding_provider as ep, embedding_maintenance as em
    class OlderProvider(Fake):
        def __init__(self, model, **kwargs):
            super().__init__()
        def warmup(self):
            if effective is not None:
                self._model = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(get_session_options=lambda: SimpleNamespace(intra_op_num_threads=effective))))
            return self.embed_query("warmup")
    monkeypatch.setattr(ep, "FastembedProvider", OlderProvider)
    monkeypatch.setattr(em, "note_local_model_present", lambda _: None)
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    monkeypatch.setenv("LCM_D1_ORT_THREADS", "2")
    report = tmp_path / "threads.json"
    monkeypatch.setenv("LCM_D1_THREAD_REPORT", str(report))
    assert lme.resolve_harness_provider("fastembed", "fake-model").dim == 64
    receipt = json.loads(report.read_text())
    assert receipt["thread_cap_applied"] is False
    assert receipt["effective_intra_op_threads"] == effective
    assert bool(receipt["reason"]) == (effective is None)


def wait_until(predicate, seconds=8):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.025)
    assert predicate(), "condition did not become true before deadline"


def pid_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_timeout_kills_grandchild(tmp_path):
    cli = driver()
    grand_pid = tmp_path / "grand.pid"
    parent_pid = tmp_path / "parent.pid"
    grand_code = f"import os,time; from pathlib import Path; Path({str(grand_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    code = (f"import os,subprocess,sys,time; from pathlib import Path; Path({str(parent_pid)!r}).write_text(str(os.getpid())); "
            f"subprocess.Popen([sys.executable, '-c', {grand_code!r}]); time.sleep(60)")
    try:
        result = cli.run_child([sys.executable, "-c", code], timeout=1)
        assert result == {"exit_code": 124, "timed_out": True}
        assert grand_pid.exists(), "grandchild must have started before timeout"
        wait_until(lambda: pid_gone(int(grand_pid.read_text())), seconds=3)
    finally:
        if parent_pid.exists():
            try:
                os.killpg(int(parent_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def driver_args(root):
    ids = [f"q{i}" for i in range(500)]
    def manifest(directory, selected):
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"questions": [{"question_id": q} for q in selected]}))
    manifest(root / "prepared", ids)
    manifest(root / "aa", ids[:100])
    for k in range(6):
        manifest(root / "shards" / f"shard-{k}", ids[k::6])
    return ["--product", "base=base", "--product", "candidate=candidate",
            "--prepared-dir", str(root / "prepared"), "--shards-dir", str(root / "shards"),
            "--aprime-dir", str(root / "aa"), "--output", str(root / "out"), "--scratch", str(root / "scratch")]


@pytest.mark.parametrize("phase", ["build", "read"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_driver_interrupt_cleans_groups_and_writes_partial_receipt(tmp_path, phase, signum):
    argv = driver_args(tmp_path)
    # A descendant ignores the forwarded signal, exercising bounded escalation
    # even when the worker leader exits immediately.
    grand = "import os,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
    sleeper = ("import json,os,subprocess,sys,time; from pathlib import Path; "
               f"p=subprocess.Popen([sys.executable,'-c',{grand!r}],stdout=subprocess.PIPE,text=True); "
               "g=int(p.stdout.readline()); Path(sys.argv[1]).write_text(json.dumps([os.getpid(),g])); time.sleep(60)")
    runner = f'''
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("driver", {str(Path(__file__).resolve().parents[1] / 'scripts/lcm_d1_local.py')!r})
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
cli.pin=lambda _: "a"*40
import benchmarking.d1_cache
benchmarking.d1_cache.seal=lambda _: "fake-manifest"
run=cli.run_child
def fake(command, **kw):
    env=kw["env"]
    if env["LCM_D1_CACHE_MODE"] == "build" and {phase!r} == "read":
        Path(env["LCM_D1_THREAD_REPORT"]).write_text(json.dumps({{"effective_intra_op_threads":2}}))
        return {{"exit_code":0,"timed_out":False}}
    ready=Path(env["TMPDIR"]).with_suffix(".ready")
    kw.pop("cwd")
    return run([sys.executable,"-c",{sleeper!r},str(ready)], **kw)
cli.run_child=fake
raise SystemExit(cli.main({argv!r}))
'''
    process = subprocess.Popen([sys.executable, "-c", runner], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    expected = 1 if phase == "build" else 3
    try:
        wait_until(lambda: len(list((tmp_path / "scratch").glob("*/*.ready"))) == expected)
        ready = list((tmp_path / "scratch").glob("*/*.ready"))
        wait_until(lambda: all(p.stat().st_size for p in ready))
        groups = [json.loads(p.read_text()) for p in ready]
        process.send_signal(signum)
        assert process.wait(timeout=10) == 128 + signum
        receipt = json.loads((tmp_path / "out/d1-receipt.json").read_text())
        assert receipt["interrupted"] is True
        rows = receipt["preembed"] if phase == "build" else receipt["arms"]
        assert any(row["exit_code"] == 128 + signum for row in rows)
        for leader, grandchild in groups:
            wait_until(lambda: pid_gone(leader) and pid_gone(grandchild))
            with pytest.raises(ProcessLookupError):
                os.killpg(leader, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for ready in (tmp_path / "scratch").glob("*/*.ready"):
            if ready.stat().st_size:
                try:
                    os.killpg(json.loads(ready.read_text())[0], signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_interrupt_after_leader_exit_cleans_grandchild_and_receipt(tmp_path, monkeypatch):
    cli = driver()
    argv = driver_args(tmp_path)
    monkeypatch.setattr(cli, "pin", lambda _: "a" * 40)
    grand_pid = tmp_path / "grand.pid"
    grand = ("import os,signal,time; from pathlib import Path; "
             "signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             f"Path({str(grand_pid)!r}).write_text(str(os.getpid())); print('ready',flush=True); time.sleep(60)")
    worker = ("import subprocess,sys; "
              f"p=subprocess.Popen([sys.executable,'-c',{grand!r}],stdout=subprocess.PIPE,text=True); "
              "assert p.stdout.readline().strip() == 'ready'")
    original_wait = subprocess.Popen.wait
    original_run = cli.run_child
    interrupt = None
    leaders = []

    def wait_then_interrupt(child, *args, **kwargs):
        result = original_wait(child, *args, **kwargs)
        if not leaders:
            assert result == 0 and pid_gone(child.pid)
            assert grand_pid.exists() and not pid_gone(int(grand_pid.read_text()))
            leaders.append(child.pid)
            interrupt[0] = signal.SIGINT
        return result

    def fake_run(command, **kwargs):
        nonlocal interrupt
        interrupt = kwargs["interrupt"]
        Path(kwargs["env"]["LCM_D1_THREAD_REPORT"]).write_text(json.dumps({"effective_intra_op_threads": 2}))
        kwargs.pop("cwd")
        return original_run([sys.executable, "-c", worker], **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", wait_then_interrupt)
    monkeypatch.setattr(cli, "run_child", fake_run)
    try:
        result = cli.main(argv)
        wait_until(lambda: pid_gone(int(grand_pid.read_text())), seconds=3)
        assert result == 130
        receipt = json.loads((tmp_path / "out/d1-receipt.json").read_text())
        assert receipt["interrupted"] is True
        assert receipt["preembed"] == [{"role": "base", "arm": "preembed", "exit_code": 130, "timed_out": False}]
    finally:
        for leader in leaders:
            cli.signal_group(leader, signal.SIGKILL)
        if grand_pid.exists():
            wait_until(lambda: pid_gone(int(grand_pid.read_text())))


@pytest.mark.parametrize("exists,insensitive", [(False, False), (True, False), (True, True)])
def test_distinct_normalized_sidecar_paths_are_accepted(tmp_path, monkeypatch, exists, insensitive):
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    output = tmp_path / "caf\u00e9"
    output.mkdir()
    sidecar = output / "embedding-metrics.jsonl"
    dump = tmp_path / "cafe\u0301" / sidecar.name
    if exists:
        sidecar.write_text("preserve\n")
        # APFS cannot host distinct NFC/NFD entries: simulate their distinct
        # filesystem identities, including an authoritative False on samefile.
        path_exists = Path.exists
        monkeypatch.setattr(Path, "exists", lambda path: True if path == dump else path_exists(path))
        monkeypatch.setattr(lme.os.path, "samefile", lambda a, b: False)
    monkeypatch.setattr(lme, "_d1_normalization_insensitive", lambda parent: insensitive, raising=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    lme._validate_d1_sidecar_alias(output / "per_question_checkpoint.jsonl", dump)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("alias", ["exact", "symlink", "dotdot", "case", "hardlink", "nfd"])
@pytest.mark.parametrize("exists", [False, True])
def test_cli_refuses_metrics_sidecar_alias_before_open(tmp_path, monkeypatch, alias, exists):
    spec = importlib.util.spec_from_file_location("lme_cli", Path(__file__).resolve().parents[1] / "scripts/lcm_longmemeval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / ("caf\u00e9" if alias == "nfd" else "output")
    output.mkdir()
    sidecar = output / "embedding-metrics.jsonl"
    if exists:
        sidecar.write_text("preserve\n")
    dump = sidecar
    if alias == "symlink":
        dump = tmp_path / "alias.jsonl"
        dump.symlink_to(sidecar)
    elif alias == "dotdot":
        (output / "nested").mkdir()
        dump = output / "nested/../embedding-metrics.jsonl"
    elif alias == "case":
        if not output.with_name("OUTPUT").exists():
            pytest.skip("case-sensitive volume")
        dump = output / "EMBEDDING-METRICS.JSONL"
    elif alias == "hardlink":
        if not exists:
            pytest.skip("hardlink requires existing target")
        dump = tmp_path / "hardlink.jsonl"
        os.link(sidecar, dump)
    elif alias == "nfd":
        if not (tmp_path / "cafe\u0301").exists():
            pytest.skip("normalization-sensitive volume")
        dump = tmp_path / "cafe\u0301" / "embedding-metrics.jsonl"
    dataset = tmp_path / lme.DATASET_COORDS["s"]["file"]
    dataset.write_text("[]")
    monkeypatch.setattr(cli, "load_questions_with_sha256", lambda *a, **k: (questions()[:1], "a" * 64))
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="must not alias embedding-metrics.jsonl"):
        cli.main(["run", "--dataset", str(dataset), "--output", str(output),
                  "--allow-external-output", "--dump-candidates", str(dump)])
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("crash", ["midwrite", "before_checkpoint"])
def test_read_mode_resume_reconciles_metrics(tmp_path, monkeypatch, crash):
    monkeypatch.setenv("LCM_D1_PRODUCT_SHA", "a" * 40)
    monkeypatch.setenv("LCM_D1_ROLE", "base")
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "build")
    builder = lme.resolve_harness_provider("stub", "")
    lme.prewarm_embedding_cache(questions()[:2], builder)
    seal(builder.cache_path)
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    checkpoint = tmp_path / "run/per_question_checkpoint.jsonl"
    sidecar = checkpoint.with_name("embedding-metrics.jsonl")
    write = lme._write_checkpoint_record
    def crash_write(file, record):
        if record.get("question_id") == "q1":
            if crash == "midwrite" and Path(file.name) == sidecar:
                file.write('{"question_id": "q1", "wall_ms":')
                file.flush()
                raise RuntimeError("injected midwrite crash")
            if crash == "before_checkpoint" and Path(file.name) == checkpoint:
                raise RuntimeError("injected before_checkpoint crash")
        write(file, record)
    monkeypatch.setattr(lme, "_write_checkpoint_record", crash_write)
    args = dict(provider_name="stub", model="", tmp_dir=tmp_path / "db", checkpoint_path=checkpoint, reuse_db_template=False)
    with pytest.raises(RuntimeError, match="injected"):
        lme.run_harness(questions()[:2], **args)
    if crash == "midwrite":
        assert sidecar.read_text().endswith('"wall_ms":')
    else:
        assert len(sidecar.read_text().splitlines()) == 2
    monkeypatch.setattr(lme, "_write_checkpoint_record", write)
    lme.run_harness(questions()[:2], resume=True, **args)
    rows = list(map(json.loads, sidecar.read_text().splitlines()))
    checkpoint_ids = [r["question_id"] for r in map(json.loads, checkpoint.read_text().splitlines()) if "question_id" in r]
    assert [row["question_id"] for row in rows] == checkpoint_ids == ["q0", "q1"]
    # Completed resumes reconcile too, without constructing a provider.
    with sidecar.open("a") as file:
        file.write('{"question_id":')
    lme.run_harness(questions()[:2], resume=True, **args)
    assert list(map(json.loads, sidecar.read_text().splitlines())) == rows


def test_reconcile_keeps_original_bytes_and_file_mode(tmp_path):
    sidecar = tmp_path / "embedding-metrics.jsonl"
    sidecar.write_bytes(b'{"question_id":"q0","wall_ms":4}\n{"question_id": "q1", "wall_ms": 6}\n'
                        b'{"question_id":"qx","wall_ms":5}\n{"question_id":"q0","wall_ms":7}\n{"question_id":')
    sidecar.chmod(0o640)
    lme._reconcile_d1_metrics(sidecar, {"q0", "q1"})
    # The newest row per checkpointed question survives byte for byte; un-checkpointed and torn rows go.
    assert sidecar.read_bytes() == b'{"question_id":"q0","wall_ms":7}\n{"question_id": "q1", "wall_ms": 6}\n'
    assert sidecar.stat().st_mode & 0o777 == 0o640
