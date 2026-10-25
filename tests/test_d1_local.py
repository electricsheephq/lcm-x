"""D1 instrument proofs: deterministic fakes only, no FastEmbed/model downloads."""
import hashlib
import importlib.util
import json
import os
import sqlite3
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
        assert record["embedding"] == actual["embedding"]
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


@pytest.mark.parametrize("change", ["model", "dimension", "normalization", "mode"])
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


def test_parallel_driver_receipts_and_failure_isolation(tmp_path, monkeypatch):
    cli = driver()
    monkeypatch.setattr(cli, "pin", lambda _: "a" * 40)
    shards = tmp_path / "shards"
    qid = 0
    for k, size in enumerate([85] * 5 + [75]):
        directory = shards / f"shard-{k}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"questions": [{"question_id": f"q{i}"} for i in range(qid, qid + size)]}))
        qid += size
    aa = tmp_path / "aa"
    aa.mkdir()
    (aa / "manifest.json").write_text(json.dumps({"questions": [{"question_id": f"q{i}"} for i in range(100)]}))
    lock = threading.Lock()
    active, peak, roots, caps = 0, 0, [], []

    def fake_run(command, *, cwd, env, stdout, stderr):
        nonlocal active, peak
        cache = Path(env[lme.EMBED_CACHE_ENV])
        if "prewarm-cache" in command:
            assert env["LCM_D1_CACHE_MODE"] == "build"
            D1Cache(Fake(), cache, options=OPTIONS, readonly=False).embed_documents(["shared"])
            return SimpleNamespace(returncode=0)
        assert env["LCM_D1_CACHE_MODE"] == "read"
        with lock:
            active += 1
            peak = max(peak, active)
            roots.append((env["HERMES_HOME"], env["TMPDIR"]))
            caps.append(int(env["LCM_D1_ORT_THREADS"]))
        time.sleep(0.015)
        out = Path(command[command.index("--output") + 1])
        out.mkdir(parents=True)
        row = {"question_id": "q0", "embed_cache": {"hits": 2, "misses": 1},
               "embedding": {"wall_ms": 4, "corruptions": 1}}
        (out / "per_question_checkpoint.jsonl").write_text(json.dumps(row) + "\n")
        with lock:
            active -= 1
        return SimpleNamespace(returncode=int(out.name == "shard-1"))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    out = tmp_path / "output"
    assert cli.main(["--product", "base=base-tree", "--product", "candidate=candidate-tree",
                     "--prepared-dir", str(tmp_path), "--shards-dir", str(shards),
                     "--aprime-dir", str(aa), "--output", str(out), "--scratch", str(tmp_path / "scratch")]) == 1
    receipt = json.loads((out / "d1-receipt.json").read_text())
    assert len(receipt["arms"]) == 16
    assert sum(arm["exit_code"] for arm in receipt["arms"]) == 2
    assert peak == 3
    assert peak * max(caps) <= receipt["thread_budget"] == 8
    assert len(set(roots)) == len(roots) == 16
    for arm in receipt["arms"]:
        assert arm["cache_hit_rate"] == 2 / 3
        assert arm["cache_corruptions"] == 1
        assert arm["embedding_wall_ms"] == 4
        assert arm["cache_manifest_sha256"] == hashlib.sha256((out / "embeddings.manifest.json").read_bytes()).hexdigest()


def test_harness_cache_env_shim(tmp_path, monkeypatch):
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "build")
    builder = lme.resolve_harness_provider("stub", "")
    assert isinstance(builder, D1Cache)
    expected = builder.embed_documents(["exact text"])
    seal(builder.cache_path)
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "read")
    assert lme.resolve_harness_provider("stub", "").embed_documents(["exact text"]) == expected
    monkeypatch.setenv("LCM_D1_CACHE_MODE", "typo")
    with pytest.raises(ValueError, match="mode must be"):
        lme.resolve_harness_provider("stub", "")


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
            return self.embed_query("warmup")
    monkeypatch.setattr(ep, "FastembedProvider", FakeFastembed)
    monkeypatch.setattr(em, "note_local_model_present", lambda _: None)
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    monkeypatch.setenv("LCM_D1_ORT_THREADS", "2")
    provider = lme.resolve_harness_provider("fastembed", "fake-model")
    assert provider._cache_key()[2] == 2
