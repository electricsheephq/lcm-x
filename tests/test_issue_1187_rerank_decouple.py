"""#1187: independent rerank selection, cached-only local inference and telemetry."""

import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import command, embedding_provider as ep, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.vector_store import VectorStore


LOCAL_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def entries(n=3):
    return [{"hit": {"session_id": f"s{i}", "snippet": f"kanban snippet {i}"},
             "rrf_score": 0.02 / (i + 1), "_final_score": 0.02 / (i + 1)}
            for i in range(n)]


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.01)


def run_rerank(config, ordered=None, provider=None, **kwargs):
    return tools._lcm_recall_rerank(
        provider, "kanban", entries() if ordered is None else ordered,
        window=kwargs.pop("window", 2), deadline=kwargs.pop("deadline", time.monotonic() + 2),
        config=config, **kwargs,
    )


@pytest.fixture
def local(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], payloads=[], logits=[-1.0, 1.0], absent=False, delay=0.0,
                            download_delay=0.0, download_error=None)

    class Encoder:
        def __init__(self, **kwargs):
            state.calls.append(kwargs)
            if state.absent and kwargs["local_files_only"]:
                raise FileNotFoundError("synthetic empty cache")
            if not kwargs["local_files_only"]:
                time.sleep(state.download_delay)
                if state.download_error is not None:
                    raise state.download_error

        def rerank(self, query, documents):
            state.payloads.append((query, list(documents)))
            time.sleep(state.delay)
            return [state.score(document) for document in documents] if hasattr(state, "score") else state.logits[:len(documents)]

    module = ModuleType("fastembed.rerank.cross_encoder")
    module.TextCrossEncoder = Encoder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "model-cache"))
    monkeypatch.setattr(ep, "_LOCAL_RERANKER", None, raising=False)
    monkeypatch.setattr(ep, "_RERANK_WARNED", set(), raising=False)
    return state


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "recall.db"), embeddings_enabled=False,
                       rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}")
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    instance._session_id = "current"
    instance.context_length = 200000
    yield instance
    instance.shutdown()


def test_r1_voyage_bare_preserves_order_status_scores(monkeypatch):
    config = LCMConfig(rerank_enabled=True, embedding_privacy_enabled=False)
    incoming = entries()
    observed = []

    class Voyage:
        provider_id = "voyage"

        def rerank(self, query, documents, **kwargs):
            observed.append((query, documents, kwargs))
            return [(1, 0.9), (0, 0.3)]

    monkeypatch.setattr(tools, "resolve_provider", lambda _: pytest.fail("must reuse Voyage"))
    output, status, scores = run_rerank(config, incoming, Voyage())
    assert output == [incoming[1], incoming[0], incoming[2]]
    assert status == "applied"
    assert scores == [[1, "s1", 0.9], [0, "s0", 0.3]]
    assert observed[0][:2] == ("kanban", ["kanban snippet 0", "kanban snippet 1"])
    assert observed[0][2]["model"] == "rerank-2.5-lite"
    assert observed[0][2]["top_k"] == 2


def test_disabled_resolves_nothing(monkeypatch, local):
    monkeypatch.setattr(tools, "resolve_provider", lambda _: pytest.fail("disabled resolution"))
    config = LCMConfig(rerank_model=f"fastembed:{LOCAL_MODEL}")
    incoming = entries()
    assert run_rerank(config, incoming) == (incoming, "disabled", [])
    assert local.calls == []


@pytest.mark.parametrize("embedding", ["fastembed", "off"])
def test_r2_voyage_uses_separate_engine_cache(monkeypatch, embedding):
    config = LCMConfig(rerank_enabled=True, rerank_model="voyage:rerank-2.5-lite",
                       embeddings_enabled=embedding != "off", embedding_provider="fastembed",
                       embedding_model="embedding-model", embedding_privacy_enabled=False)
    embedding_provider = SimpleNamespace(provider_id="fastembed")
    engine = SimpleNamespace(_config=config, _lcm_embedding_provider_cache=("sentinel", embedding_provider))
    cache = engine._lcm_embedding_provider_cache
    calls = []
    voyage = SimpleNamespace(provider_id="voyage", rerank=lambda *a, **k: [(1, 0.9), (0, 0.1)])

    def resolve(candidate):
        calls.append(candidate)
        assert candidate.embedding_provider == "voyage"
        return voyage

    monkeypatch.setattr(tools, "resolve_provider", resolve)
    for _ in range(2):
        output, status, _ = run_rerank(config, provider=embedding_provider if embedding != "off" else None,
                                      engine=engine)
        assert status == "applied" and output[0]["hit"]["session_id"] == "s1"
    assert len(calls) == 1
    assert engine._lcm_embedding_provider_cache is cache
    assert engine._lcm_rerank_provider_cache is voyage
    assert config.embedding_provider == "fastembed" and config.embedding_model == "embedding-model"


def test_r2_cloud_privacy_uses_rerank_provider(monkeypatch):
    config = LCMConfig(rerank_enabled=True, rerank_model="voyage:rerank-2.5-lite")
    outbound = []
    privacy_providers = []
    monkeypatch.setattr(tools, "embedding_provider_requires_privacy",
                        lambda provider: privacy_providers.append(provider) or provider == "voyage")
    monkeypatch.setattr(tools, "protect_embedding_text", lambda text, *a, **k: ("protected:" + text, "r", True))
    monkeypatch.setattr(tools, "validate_embedding_privacy_dispatch", lambda *a, **k: None)
    voyage = SimpleNamespace(provider_id="voyage", rerank=lambda q, d, **k: outbound.append((q, d)) or [(0, 1.0)])
    monkeypatch.setattr(tools, "resolve_provider", lambda _: voyage)
    assert run_rerank(config, provider=SimpleNamespace(provider_id="fastembed"))[1] == "applied"
    assert privacy_providers == ["voyage"]
    assert outbound[0][0] == "protected:kanban"
    assert all(text.startswith("protected:") for text in outbound[0][1])


@pytest.mark.parametrize("embedding", ["fastembed", "off"])
def test_r3_local_sigmoid_order(local, embedding):
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}",
                       embeddings_enabled=embedding != "off")
    incoming = entries()
    output, status, scores = run_rerank(config, incoming,
                                      SimpleNamespace(provider_id=embedding) if embedding != "off" else None)
    assert status == "applied" and output == [incoming[1], incoming[0], incoming[2]]
    assert [row[:2] for row in scores] == [[1, "s1"], [0, "s0"]]
    assert [row[2] for row in scores] == pytest.approx([1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])
    assert local.payloads == [("kanban", ["kanban snippet 0", "kanban snippet 1"])]
    assert local.calls == [{"model_name": LOCAL_MODEL, "cache_dir": os.environ["FASTEMBED_CACHE_PATH"],
                            "threads": ep._FASTEMBED_THREADS, "local_files_only": True}]
    assert output[0]["_final_score"] == incoming[1]["_final_score"]


def test_r3_fts_only_recall(local, engine):
    for i in range(2):
        engine._store.append(f"session-{i}", {"role": "user", "content": f"kanban detail {i}"})
    args = {"query": "kanban", "limit": 2, "scope_bias": 0}
    engine._config.rerank_enabled = False
    before = json.loads(tools.lcm_recall(args, engine=engine))
    engine._config.rerank_enabled = True
    after = json.loads(tools.lcm_recall(args, engine=engine))
    assert after["provenance"]["rerank"] == "applied"
    assert after["provenance"]["arms_run"] == ["fts"]
    assert [hit["store_id"] for hit in after["hits"]] == [hit["store_id"] for hit in reversed(before["hits"])]
    assert len(after["provenance"]["rerank_scores"]) == 2


def test_r4_absent_skips_once_without_download(local, caplog):
    local.absent = True
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}")
    incoming = entries()
    for _ in range(2):
        assert run_rerank(config, incoming) == (incoming, "skipped: rerank model not prepared", [])
    warnings = [r.message for r in caplog.records if "rerank model not prepared" in r.message]
    assert len(warnings) == 1 and "/lcm embed warmup" in warnings[0]
    assert local.calls and all(call["local_files_only"] for call in local.calls)
    assert local.payloads == []


def test_r5_slow_local_preserves_incoming_deadline(local):
    local.delay = 0.15
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}")
    incoming = entries()
    started = time.monotonic()
    output, status, scores = run_rerank(config, incoming, deadline=started + 0.04)
    elapsed = time.monotonic() - started
    assert output is incoming and status.startswith("skipped:") and scores == []
    assert elapsed < 0.12
    time.sleep(0.17)  # Let the bounded worker finish before the fake is removed.


def test_r6_fts_anchor_follows_local_rerank(local, engine, monkeypatch):
    engine._config.embeddings_enabled = True
    engine._config.embedding_provider, engine._config.embedding_model = "mock", "mock-model"
    provider = SimpleNamespace(provider_id="mock", model_id="mock-model", embed_query=lambda _: [1.0, 0.0])
    monkeypatch.setattr(tools, "resolve_provider", lambda _: provider)
    anchor_id = engine._store.append("fts-session", {"role": "user", "content": "kanban anchor detail"})
    local.score = lambda document: -4.0 if "anchor" in document else 4.0
    vectors = VectorStore(engine._store.db_path, config=engine._config)
    try:
        vectors.register_profile("mock-model", "mock", 2)
        identity = vectors.capture_identity("mock-model", provider="mock")
        for i in range(4):
            node_id = engine._dag.add_node(SummaryNode(
                session_id=f"summary-{i}", depth=0, summary=f"kanban summary {i}",
                token_count=4, source_ids=[], source_type="messages", created_at=time.time(),
            ))
            vectors.record_embedding(str(node_id), "summary", "mock-model", [1.0, 0.0], identity=identity)
    finally:
        vectors.close()
    args = {"query": "kanban", "limit": 4, "scope_bias": 0}
    payload = json.loads(tools.lcm_recall(args, engine=engine))
    assert payload["provenance"]["rerank"] == "applied"
    assert payload["provenance"]["fts_anchor"]["fired"] is True
    assert payload["provenance"]["fts_anchor"]["delivered"] is True
    assert any(hit.get("store_id") == anchor_id for hit in payload["hits"])
    proactive = json.loads(tools.lcm_recall(args, engine=engine, fts_anchor=False))
    assert not any(hit.get("store_id") == anchor_id for hit in proactive["hits"])
    assert all(hit["score"] < 0.1 for hit in proactive["hits"])


def test_r7_unknown_warns_once(local, caplog):
    config = LCMConfig(rerank_enabled=True, rerank_model="other:model")
    incoming = entries()
    for _ in range(2):
        assert run_rerank(config, incoming) == (incoming, "skipped: unknown rerank model", [])
    warnings = [r.message for r in caplog.records if "unknown rerank model" in r.message]
    assert len(warnings) == 1 and "warmup" not in warnings[0]  # Warmup cannot fix an unknown prefix.
    assert local.calls == []


@pytest.mark.parametrize(("logits", "expected", "status"), [
    ([0.0, 0.1], "s0", "applied: rank1-held"),
    ([-2.0, 2.0], "s1", "applied"),
])
def test_r8_sigmoid_margin(local, logits, expected, status):
    local.logits = logits
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}", rerank_margin=0.3)
    output, actual, scores = run_rerank(config)
    assert actual == status and output[0]["hit"]["session_id"] == expected
    assert all(0 <= row[2] <= 1 for row in scores)


def test_r9_warmup_prepares_local_with_embeddings_off(local, engine):
    local.absent = True
    text = command.handle_lcm_command("embed warmup", engine)
    assert f"rerank: fastembed:{LOCAL_MODEL} ready" in text
    assert local.calls and local.calls[-1]["local_files_only"] is False
    assert run_rerank(engine._config)[1] == "applied"
    assert len(local.calls) == 1  # Same process-wide instance used by recall.


@pytest.mark.parametrize(("enabled", "model", "prepared", "key", "expected"), [
    (False, f"fastembed:{LOCAL_MODEL}", False, False, "disabled"),
    (True, f"fastembed:{LOCAL_MODEL}", False, False, "not prepared"),
    (True, f"fastembed:{LOCAL_MODEL}", True, False, f"fastembed:{LOCAL_MODEL} ready"),
    (True, "other:model", False, False, "unknown"),
    (True, "voyage:rerank-2.5-lite", False, False, "not prepared"),
    (True, "voyage:rerank-2.5-lite", False, True, "voyage:rerank-2.5-lite ready"),
])
def test_r9_doctor_states(local, engine, monkeypatch, enabled, model, prepared, key, expected):
    engine._config.rerank_enabled, engine._config.rerank_model = enabled, model
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    if key:
        monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")
    if prepared:
        command.handle_lcm_command("embed warmup", engine)
    before = len(local.calls)
    assert tools._embedding_provider_health_check(engine)["rerank"] == expected
    text = command.handle_lcm_command("doctor", engine)
    assert f"rerank: {expected}" in text.splitlines()
    assert len(local.calls) == before  # Doctor does not construct or download models.


def test_r9_disabled_warmup_does_not_prepare(local, engine):
    engine._config.rerank_enabled = False
    command.handle_lcm_command("embed warmup", engine)
    assert local.calls == []


def test_local_single_instance_concurrent(local):
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}")
    results = []
    workers = [threading.Thread(target=lambda: results.append(run_rerank(config)[1])) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert results == ["applied"] * 4 and len(local.calls) == 1


def test_r10_cached_model_latency():
    """Opt-in receipt: no download, no corpus, 20 measurements of 50 x 300 chars."""
    receipt = os.environ.get("LCMX_1187_LATENCY_RECEIPT")
    if not receipt:
        pytest.skip("run separately with LCMX_1187_LATENCY_RECEIPT for the cached-model probe")
    path = Path(receipt)
    try:
        reranker = ep.local_reranker(LOCAL_MODEL)
        reranker._ensure_local()
    except (AttributeError, ep.ProviderNotWarmedUp) as exc:
        reason = f"SKIP: MiniLM-L6 is not cached/prepared ({exc}); no model download attempted.\n"
        path.write_text(reason)
        pytest.skip(reason)
    documents = [(f"Synthetic project {i}: kanban planning and release tasks. " * 8)[:300] for i in range(50)]
    elapsed = []
    for _ in range(20):
        started = time.perf_counter()
        scores = reranker.rerank("Which project has kanban tasks?", documents, top_k=50, timeout=30)
        elapsed.append((time.perf_counter() - started) * 1000)
        assert len(scores) == 50 and all(0 <= score <= 1 for _, score in scores)
    p50 = statistics.median(elapsed)
    p90 = sorted(elapsed)[math.ceil(len(elapsed) * 0.9) - 1]
    path.write_text(f"model | snippets | runs | p50 ms | p90 ms\n{LOCAL_MODEL} | 50 x 300 chars | 20 | {p50:.2f} | {p90:.2f}\n")


@pytest.mark.parametrize("model", ["rerank-2.5-lite", "voyage:rerank-2.5-lite"])
@pytest.mark.parametrize("degraded", [None, SimpleNamespace(provider_id="fastembed")],
                         ids=["embedding-failed-or-timed-out", "proactive-fastembed-override"])
def test_r1_degraded_voyage_embedding_keeps_base_skip(monkeypatch, model, degraded):
    config = LCMConfig(rerank_enabled=True, rerank_model=model, embeddings_enabled=True,
                       embedding_provider="voyage", embedding_model="voyage-4-lite",
                       embedding_privacy_enabled=False)
    monkeypatch.setattr(tools, "resolve_provider", lambda _: pytest.fail("no rerank-only Voyage provider"))
    engine = SimpleNamespace(_config=config)
    incoming = entries()
    assert run_rerank(config, incoming, degraded, engine=engine) == (
        incoming, "skipped: rerank requires the voyage provider", [])
    assert not hasattr(engine, "_lcm_rerank_provider_cache")


def test_r3_slow_inference_blocks_no_state_reader(local, engine):
    local.delay = 2.0
    config = engine._config
    first = threading.Thread(target=run_rerank, args=(config,), kwargs={"deadline": time.monotonic() + 5})
    first.start()
    observed = []

    def readers():
        ep.local_reranker(LOCAL_MODEL)
        ep.warn_rerank_once("synthetic reason")
        observed.append(ep.rerank_readiness(config))
        observed.append(tools._embedding_provider_health_check(engine)["rerank"])
        observed.append(run_rerank(config, deadline=time.monotonic() + 0.3)[1])

    reader = threading.Thread(target=readers)
    try:
        wait_for(lambda: local.payloads)
        reader.start()
        reader.join(1.5)
        assert not reader.is_alive() and first.is_alive()
        assert observed[:2] == [f"fastembed:{LOCAL_MODEL} ready"] * 2
        # The second recall resolved its reranker; only its inference waited on the first.
        assert "lcm-recall-rerank exceeded" in observed[2] or observed[2] == "skipped: rerank deadline exceeded"
    finally:
        first.join()
        if reader.ident is not None:
            reader.join()


def test_r3_warmup_download_blocks_neither_recall_nor_doctor(local, engine):
    local.absent, local.download_delay = True, 2.0
    warm = threading.Thread(target=command.handle_lcm_command, args=("embed warmup", engine))
    warm.start()
    observed = []

    def readers():
        observed.append(ep.rerank_readiness(engine._config))
        observed.append(tools._embedding_provider_health_check(engine)["rerank"])
        observed.append(run_rerank(engine._config)[1])

    reader = threading.Thread(target=readers)
    try:
        wait_for(lambda: any(not call["local_files_only"] for call in local.calls))
        reader.start()
        reader.join(1.5)
        assert not reader.is_alive() and warm.is_alive()
        assert observed == ["not prepared", "not prepared", "skipped: rerank model not prepared"]
    finally:
        warm.join()
        if reader.ident is not None:
            reader.join()


@pytest.mark.parametrize(("value", "expected"), [
    (f"fastembed:  {LOCAL_MODEL} ", ("fastembed", LOCAL_MODEL)),
    (" voyage: rerank-2.5 ", ("voyage", "rerank-2.5")),
])
def test_r6_whitespace_after_prefix(value, expected):
    assert ep.rerank_selection(SimpleNamespace(rerank_model=value)) == expected


@pytest.mark.parametrize(("failure", "cause"), [
    ("unknown", "is not supported in TextCrossEncoder"),
    ("fastembed-missing", "FastEmbed is not installed"),
    ("network", "synthetic network unreachable"),
])
def test_r7_failed_rerank_warmup_still_warms_embeddings(local, engine, monkeypatch, failure, cause):
    warmed = []
    monkeypatch.setattr(command, "_embedding_register_profiles",
                        lambda *a, **k: warmed.append(k) or "LCM embedding warmup\nstatus: ok")
    if failure == "fastembed-missing":
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
    elif failure == "unknown":
        local.download_error = ValueError(f"Model {LOCAL_MODEL} is not supported in TextCrossEncoder.")
    else:
        local.download_error = ConnectionError("synthetic network unreachable")
    text = command.handle_lcm_command("embed warmup", engine)
    assert warmed == [{"allow_download": True}]
    assert text.startswith("LCM embedding warmup\nstatus: ok\n")
    line = text.splitlines()[-1]
    assert line.startswith("rerank: error:") and cause in line and "not prepared" not in line


def test_r7_cached_files_report_cached_without_loading(local, engine):
    onnx = (Path(os.environ["FASTEMBED_CACHE_PATH"]) / "models--Xenova--ms-marco-MiniLM-L-6-v2"
            / "snapshots" / "synthetic-revision" / "onnx" / "model.onnx")
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"synthetic")
    assert tools._embedding_provider_health_check(engine)["rerank"] == f"fastembed:{LOCAL_MODEL} cached"
    assert f"rerank: fastembed:{LOCAL_MODEL} cached" in command.handle_lcm_command("doctor", engine).splitlines()
    assert local.calls == []


def test_slot_exhaustion_names_rerank_stage(local, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(tools, "_lcm_semantic_worker_slots", slots)
    config = LCMConfig(rerank_enabled=True, rerank_model=f"fastembed:{LOCAL_MODEL}")
    assert run_rerank(config)[1] == "skipped: lcm-recall-rerank worker capacity is exhausted"


def test_unsupported_local_model_has_no_warmup_hint(local, monkeypatch, caplog):
    def unsupported(**kwargs):
        raise ValueError("Model unsupported is not supported in TextCrossEncoder.")
    monkeypatch.setattr(sys.modules["fastembed.rerank.cross_encoder"], "TextCrossEncoder", unsupported)
    config = LCMConfig(rerank_enabled=True, rerank_model="fastembed:unsupported")
    assert "not supported" in run_rerank(config)[1]
    assert "warmup" not in caplog.text


def test_concurrent_warmups_construct_once(local):
    from concurrent.futures import ThreadPoolExecutor

    local.download_delay = 0.1
    reranker = ep.local_reranker(LOCAL_MODEL)
    barrier = threading.Barrier(2)

    def warm():
        barrier.wait(timeout=2)
        return reranker.warmup()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(warm), pool.submit(warm)
        assert first.result(timeout=2) is second.result(timeout=2)
    assert len(local.calls) == 1


@pytest.mark.parametrize("profile_key", [False, True])
def test_rerank_readiness_uses_profile_key(monkeypatch, profile_key):
    from hermes_lcm import config as config_module

    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-process-key")
    monkeypatch.setattr(config_module, "lcm_env",
                        lambda: {"VOYAGE_API_KEY": "synthetic-profile-key"} if profile_key else {})
    config = LCMConfig(rerank_enabled=True, rerank_model="voyage:rerank-2.5-lite")
    expected = "voyage:rerank-2.5-lite ready" if profile_key else "not prepared"
    assert ep.rerank_readiness(config) == expected
