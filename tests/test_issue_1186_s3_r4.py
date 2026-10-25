"""R4: explicit cloud consent and nonblocking local readiness on real tool paths."""
import io
import json
import socket
import threading
from dataclasses import asdict

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.embedding_provider import FastembedProvider, ProviderUnavailable, resolve_provider
from hermes_lcm.engine import LCMEngine

MODEL = "BAAI/bge-small-en-v1.5"
MARK = "cobalt synthetic query"
REAL_PREPARE = maintenance.start_local_model_prepare


@pytest.fixture
def transport(monkeypatch):
    """Exercise both HTTP clients while making socket/DNS egress impossible."""
    sent, sockets = [], []

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json"}

    def record(request, *args, **kwargs):
        payload = json.loads(request.data)
        texts = payload.get("input", [])
        texts = [texts] if isinstance(texts, str) else texts
        if "inputs" in payload:
            texts = [text for group in payload["inputs"] for text in group]
            body = {"data": [{"index": i, "data": [
                {"index": j, "embedding": [1.0, 0.0]} for j in range(len(group))]}
                for i, group in enumerate(payload["inputs"])]}
        elif "/api/embed" in request.full_url:
            body = {"embeddings": [[1.0, 0.0] for text in texts]}
        else:
            body = {"data": [{"index": i, "embedding": [1.0, 0.0]}
                             for i, text in enumerate(texts)]}
        sent.append({"url": request.full_url, "carries_mark": any(MARK in text for text in texts)})
        return Response(json.dumps(body).encode())

    def refuse(*args, **kwargs):
        sockets.append("refused")
        raise OSError("synthetic regression refuses network")

    monkeypatch.setattr("urllib.request.urlopen", record)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda self, request, *a, **k: record(request))
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return sent, sockets


def _engine(tmp_path, config):
    config.database_path = str(tmp_path / "lcm.db")
    config.temporal_rollups_enabled = False
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def _seed(engine, session="old"):
    target = engine._store.append(session, {"role": "user", "content": MARK})
    engine._dag.add_node(SummaryNode(session_id=session, depth=0, summary=MARK,
                                    created_at=1.0, latest_at=1.0))
    return target


@pytest.mark.parametrize("provider,model", [("voyage", "voyage-3"), ("ollama", "nomic-embed-text")])
@pytest.mark.parametrize("flag", [None, "false", "true"])
@pytest.mark.parametrize("loader", ["env", "constructor"])
def test_proactive_override_requires_explicit_consent(tmp_path, monkeypatch, transport, provider, model, flag, loader):
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture-key")
    warm = LCMConfig(embeddings_enabled=True, embedding_provider=provider, embedding_model=model)
    engine = _engine(tmp_path, warm)
    try:
        assert isinstance(command._embedding_register_profiles(warm, warm.database_path, allow_download=False), dict)
        _seed(engine)
        assert "error" not in str(maintenance.run_incremental_embedding_pass(warm.database_path, warm))
        monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv("LCM_EMBEDDING_MODEL", model)
        monkeypatch.setenv("LCM_PROACTIVE_RECALL_ENABLED", "true")
        monkeypatch.setenv("LCM_PROACTIVE_RECALL_PROVIDER", provider)
        if flag is None:
            monkeypatch.delenv("LCM_EMBEDDINGS_ENABLED", raising=False)
        else:
            monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", flag)
        engine._config = LCMConfig.from_env()
        if loader == "constructor":
            settings = {} if flag is None else {"embeddings_enabled": flag == "true"}
            engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=model,
                                       proactive_recall_enabled=True, proactive_recall_provider=provider, **settings)
        sent, sockets = transport
        sent.clear()
        engine._assemble_context({"role": "system", "content": "fixture"},
                                 [{"role": "user", "content": MARK}])
        print(json.dumps({"provider": provider, "loader": loader, "flag": flag, "sent": sent, "sockets": sockets}))
        assert sockets == []
        assert any(row["carries_mark"] for row in sent) is (flag == "true")
        if flag != "true":
            assert sent == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("provider", [None, "", "fastembed", "fast-embed", "voyage", "openai-compatible", "ollama", "siliconflow"])
@pytest.mark.parametrize("enable", [None, False, True])
def test_constructor_provider_scoped_default(provider, enable):
    kwargs = {} if provider is None else {"embedding_provider": provider}
    if enable is not None:
        kwargs["embeddings_enabled"] = enable
    config = LCMConfig(**kwargs)
    expected = enable if enable is not None else provider in {None, "", "fastembed", "fast-embed"}
    assert config.embeddings_enabled is expected
    assert isinstance(config.embedding_provider, str)
    assert config.embeddings_enabled_explicit is True  # Manual CPU policy stays explicit.
    assert asdict(config)["embeddings_enabled"] is expected


def test_constructor_model_only_retains_base_meaning():
    config = LCMConfig(embedding_model="fixture-model")
    assert config.embedding_provider == "" and config.embeddings_enabled is False
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        resolve_provider(config)


def test_manual_local_cpu_policy_is_unchanged(monkeypatch):
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    manual = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    assert not tools.default_semantic_cpu_limited(manual)
    assert manual.embeddings_enabled_explicit
    loaded = LCMConfig.from_env()
    assert tools.default_semantic_cpu_limited(loaded)
    assert not loaded.embeddings_enabled_explicit


@pytest.mark.parametrize("provider", ["voyage", "openai-compatible", "ollama", "siliconflow"])
@pytest.mark.parametrize("enable", [None, False, True])
def test_constructor_egress_matrix(tmp_path, monkeypatch, transport, provider, enable):
    for key in ("VOYAGE_API_KEY", "LCM_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"):
        monkeypatch.setenv(key, "synthetic-fixture-key")
    settings = {"embedding_provider": provider, "embedding_model": "fixture-model",
                "embedding_base_url": "https://fixture.invalid/v1"}
    warm = LCMConfig(**settings, embeddings_enabled=True)
    engine = _engine(tmp_path, warm)
    try:
        assert isinstance(command._embedding_register_profiles(warm, warm.database_path, allow_download=False), dict)
        _seed(engine)
        settings["database_path"] = warm.database_path
        if enable is not None:
            settings["embeddings_enabled"] = enable
        engine._config = LCMConfig(**settings, temporal_rollups_enabled=False)
        sent, sockets = transport
        rows = {}
        paths = {
            "recall": lambda: tools.lcm_recall({"query": MARK}, engine=engine),
            "grep_semantic": lambda: tools.lcm_grep({"query": MARK, "mode": "semantic"}, engine=engine),
            "grep_hybrid": lambda: tools.lcm_grep({"query": MARK, "mode": "hybrid"}, engine=engine),
            "maintenance": lambda: maintenance.run_incremental_embedding_pass(warm.database_path, engine._config),
            "session": lambda: engine.on_session_start("current", platform="cli", context_length=40_000),
        }
        for name, call in paths.items():
            sent.clear()
            call()
            if name == "session":
                engine.drain_embedding_maintenance(timeout=2)
            rows[name] = list(sent)
            if enable is not True:
                assert sent == [], (name, sent)
        assert sockets == []
        if enable is True:
            assert any(row["carries_mark"] for values in rows.values() for row in values)
        print(json.dumps({"provider": provider, "enable": enable, "paths": rows, "sockets": sockets}))
    finally:
        engine.shutdown()


def test_prepare_start_failure_degrades_and_can_retry(tmp_path, monkeypatch, transport):
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    config.stateless_session_patterns = ["ephemeral-*"]
    engine = _engine(tmp_path, config)
    target = _seed(engine)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True})
    real_start = threading.Thread.start
    starts = []

    def start(thread):
        if thread.name == "lcm-embedding-prepare":
            starts.append(thread.name)
            raise RuntimeError("synthetic prepare thread unavailable")
        return real_start(thread)

    try:
        engine.on_session_start("ephemeral-1", platform="cli", context_length=40_000)
        assert maintenance.local_model_state(config) is None
        with monkeypatch.context() as patch:
            patch.setattr(maintenance, "start_local_model_prepare", REAL_PREPARE)
            patch.setattr(threading.Thread, "start", start)
            result = json.loads(engine.handle_tool_call("lcm_recall", {"query": "cobalt"}))
        assert "error" not in result and result["degraded"] is True
        assert "prepar" in result["degraded_reason"]
        assert any(hit.get("store_id") == target for hit in result["hits"])
        assert starts == ["lcm-embedding-prepare"]
        assert maintenance.local_model_state(config) is None
        assert maintenance._PREPARE_THREAD is None
        retry = maintenance._PREPARE_RETRY[maintenance._local_model_key(config)]
        assert retry[0] == 1 and retry[1] > maintenance.time.monotonic()
        # R5 keeps the retry backoff; a later trigger after it expires can start.
        monkeypatch.setattr(maintenance.time, "monotonic", lambda: retry[1] + 1)
        with monkeypatch.context() as patch:
            patch.setattr(maintenance, "_prepare_local_model", lambda *args: None)
            assert REAL_PREPARE(config.database_path, config) is True
            maintenance._PREPARE_THREAD.join(2)
            assert not maintenance._PREPARE_THREAD.is_alive()
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_stateless_grep_schedules_slow_model_off_turn(tmp_path, monkeypatch, transport, mode):
    monkeypatch.setattr(tools, "probe_provider_availability", lambda _c: {"available": True})
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed", embedding_query_timeout_s=0.25)
    config.stateless_session_patterns = ["ephemeral-*"]
    engine = _engine(tmp_path, config)
    _seed(engine, "ephemeral-1")
    release = threading.Event()
    calls = []

    class FakeModel:
        def embed(self, texts):
            return [[1.0, 0.0] for text in texts]
        query_embed = embed

    def slow_model(self, *, allow_download):
        calls.append((threading.current_thread().name, allow_download))
        assert release.wait(2), "slow-model gate was not released"
        return FakeModel()

    monkeypatch.setattr(FastembedProvider, "_construct_uncached", slow_model)
    try:
        engine.on_session_start("ephemeral-1", platform="cli", context_length=40_000)
        assert engine._session_stateless and maintenance.local_model_state(config) is None
        with monkeypatch.context() as patch:
            patch.setattr(maintenance, "start_local_model_prepare", REAL_PREPARE)
            result = json.loads(tools.lcm_grep({"query": "cobalt", "mode": mode}, engine=engine))
            release.set()
            thread = maintenance._PREPARE_THREAD
            if thread is not None:
                thread.join(2)
        print(json.dumps({"mode": mode, "result": result, "constructs": calls}))
        assert "error" not in result
        assert result["degraded_to_fts"] is True
        assert "preparing" in result["degraded_reason"]
        assert result["results"]
        assert calls and all(name == "lcm-embedding-prepare" and not download for name, download in calls)
    finally:
        release.set()
        engine.shutdown()


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_provider_is_unset_local(monkeypatch, raw):
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", raw)
    config = LCMConfig.from_env()
    assert config.embedding_provider == "fastembed" and config.embeddings_enabled is True


@pytest.mark.parametrize("state,available,prefix", [
    (None, False, "local semantic recall unavailable;"),
    (("preparing", ""), True, "local semantic recall preparing;"),
    (("unavailable", "fixture uncached"), True, "local semantic recall unavailable;"),
])
def test_unready_recall_has_single_readiness_sentence(tmp_path, monkeypatch, transport, state, available, prefix):
    engine = _engine(tmp_path, LCMConfig(embeddings_enabled=True, embedding_provider="fastembed"))
    try:
        _seed(engine)
        monkeypatch.setattr(maintenance, "local_model_state", lambda config: state)
        monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": available, "detail": "fixture missing"})
        result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=engine))
        assert result["degraded_reason"].startswith(prefix)
        assert "preparing: " not in result["degraded_reason"]
        assert result["provenance"]["coverage"]["fts"] == "ok"
    finally:
        engine.shutdown()


def test_held_cloud_profile_preserved_and_explained(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture-key")
    warm = LCMConfig(embeddings_enabled=True, embedding_provider="voyage", embedding_model="voyage-3")
    engine = _engine(tmp_path, warm)
    try:
        assert isinstance(command._embedding_register_profiles(warm, warm.database_path, allow_download=False), dict)
        target = _seed(engine)
        assert "error" not in maintenance.run_incremental_embedding_pass(warm.database_path, warm)
        conn = engine._dag.connection
        tables = ("lcm_embedding_profile", "lcm_embedding_meta", "lcm_embedding_vectors", "messages", "summary_nodes")
        before = {table: list(conn.execute(f"SELECT * FROM {table}")) for table in tables}
        assert before["lcm_embedding_vectors"], "the preservation fixture must contain real stored vectors"
        engine._config = LCMConfig.from_env()
        sent, sockets = transport
        sent.clear()
        for _ in range(2):
            status = maintenance.run_incremental_embedding_pass(warm.database_path, engine._config)
            assert status == "profile_held:voyage"
        result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=engine))
        assert result["degraded"] and "profile belongs to voyage" in result["degraded_reason"]
        assert "catching up" not in result["degraded_reason"]
        assert any(hit.get("store_id") == target for hit in result["hits"])
        doctor = json.dumps(tools._embedding_provider_health_check(engine))
        assert "voyage" in doctor and "LCM_EMBEDDINGS_ENABLED=true" in doctor
        assert "/lcm embed warmup" in doctor
        # Another database may have prepared this process-wide local model,
        # while this database still holds its explicit cloud identity.
        monkeypatch.setattr(maintenance, "local_model_state", lambda config: ("present", ""))
        monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
            "available": True, "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "detail": "configured"})
        text = command.handle_lcm_command("doctor", engine)
        line = next(line for line in text.splitlines() if line.startswith("embedding_readiness:"))
        assert line.startswith("embedding_readiness: Model ready; summaries 0/1; chunks ")
        assert line.count("summaries") == 1 and "maintenance" not in line
        assert "status: issues-found" in text and "profile belongs to voyage" in text
        assert {table: list(conn.execute(f"SELECT * FROM {table}")) for table in tables} == before
        assert sent == [] and sockets == []
    finally:
        engine.shutdown()
