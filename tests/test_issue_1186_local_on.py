"""#1186: local semantic recall is on by default and built in.

T1-T7 of the #1186 spec; T8-T10 of its round-1 fix. The FastEmbed model is a fake behind the real
``FastembedProvider`` (``_load_fastembed``), so no test downloads anything.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_maintenance as maintenance_mod
import hermes_lcm.embedding_provider as provider_mod
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "BAAI/bge-small-en-v1.5"
SESSION = "local-on-session"
# conftest replaces the prepare with a no-op; tests that exercise it put the real one back.
REAL_START = getattr(maintenance_mod, "start_local_model_prepare", None)


@dataclasses.dataclass
class _Before(LCMConfig):
    """The pre-#1186 defaults, to show the flip moves nothing an overlay sets."""

    embeddings_enabled: bool = False
    embedding_provider: str = ""
    embedding_model: str = ""


def _clear_lcm_env(monkeypatch, tmp_path):
    import os

    for key in list(os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _snapshot(config, *, ignore=()) -> str:
    fields = {k: v for k, v in dataclasses.asdict(config).items() if k not in ignore}
    return json.dumps(fields, sort_keys=True, default=str)


# With embeddings off nothing reads the provider or model; an off overlay that leaves them
# unset now carries the (inert) local defaults.
INERT_WHEN_OFF = ("embedding_provider", "embedding_model", "embedding_model_auto")


class _FakeFastembed:
    """Stands in for ``fastembed.TextEmbedding``; ``download`` is the network fetch."""

    def __init__(self, *, cached=False, download=None):
        self.cached = cached
        self.download = download
        self.downloads = 0
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.documents: list[str] = []
        fake = self

        class TextEmbedding:
            def __init__(self, model_name, cache_dir, threads, local_files_only):
                if not fake.cached:
                    if local_files_only:
                        raise ValueError("model is not in the local cache")
                    with fake.lock:
                        fake.downloads += 1
                    if fake.download is not None:
                        fake.download()
                    fake.cached = True

            def query_embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed(self, texts):
                with fake.lock:
                    fake.active += 1
                    fake.max_active = max(fake.max_active, fake.active)
                    fake.documents.extend(texts)
                try:
                    time.sleep(0.002)
                    return [[1.0, 0.0] for _ in texts]
                finally:
                    with fake.lock:
                        fake.active -= 1

        self.cls = TextEmbedding


@pytest.fixture
def fastembed(monkeypatch):
    def install(**kwargs):
        fake = _FakeFastembed(**kwargs)
        monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: fake.cls)
        # find_spec("fastembed") is the only availability signal; the CI host has no fastembed.
        available = {"available": True, "provider": "fastembed", "model": MODEL, "probed": True,
                     "detail": "fastembed is importable"}
        monkeypatch.setattr(provider_mod, "probe_provider_availability", lambda _config: dict(available))
        monkeypatch.setattr(lcm_tools, "probe_provider_availability", lambda _config: dict(available))
        return fake

    return install


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.setattr(maintenance_mod, "start_local_model_prepare", REAL_START, raising=False)
    monkeypatch.setattr(maintenance_mod, "_FAILURE_WARNED", False)
    yield
    _join_prepare()


def _join_prepare(timeout=10.0) -> None:
    for thread in threading.enumerate():
        if thread.name == "lcm-embedding-prepare":
            thread.join(timeout)
            assert not thread.is_alive()


def _engine(tmp_path, home, name="lcm.db"):
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / name)
    return LCMEngine(config=config, hermes_home=str(home))


def _recall(engine, query):
    started = time.monotonic()
    result = json.loads(engine.handle_tool_call("lcm_recall", {"query": query, "scope_bias": 0.0}))
    return result, time.monotonic() - started


def _profiles(db_path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return sorted(row[0] for row in conn.execute("SELECT task FROM lcm_embedding_profile"))
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _summary_vectors(db_path) -> dict[str, bytes]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            str(embedded_id): bytes(vector)
            for embedded_id, vector in conn.execute(
                "SELECT m.embedded_id, v.vec FROM lcm_embedding_meta AS m "
                "JOIN lcm_embedding_vectors AS v "
                "ON v.embedded_id = m.embedded_id AND v.identity_hash = m.identity_hash "
                "WHERE m.embedded_kind = 'summary'"
            )
        }
    finally:
        conn.close()


def test_t1_fresh_config_has_local_recall_on(tmp_path, monkeypatch):
    _clear_lcm_env(monkeypatch, tmp_path)

    config = LCMConfig.from_env()

    assert config.embeddings_enabled is True
    assert config.embedding_provider == "fastembed"
    assert config.embedding_model == MODEL


# The managed overlays that set the keys: the fleet's local keys and a cloud overlay; plus the
# #1013 deployed configuration fixture (tests/test_issue_1016_nontool_externalization_floor.py).
OVERLAYS = {
    "fleet-local": {"LCM_EMBEDDINGS_ENABLED": "true", "LCM_EMBEDDING_PROVIDER": "fastembed",
                    "LCM_EMBEDDING_MODEL": MODEL},
    "voyage": {"LCM_EMBEDDINGS_ENABLED": "true", "LCM_EMBEDDING_PROVIDER": "voyage",
               "LCM_EMBEDDING_MODEL": "voyage-4-lite"},
}


def test_t2_explicit_off_and_managed_overlays_are_unchanged(tmp_path, monkeypatch):
    home = _clear_lcm_env(monkeypatch, tmp_path)
    # The flip changes the no-overlay default ...
    assert _snapshot(LCMConfig.from_env()) != _snapshot(_Before.from_env())

    # ... and nothing an overlay sets explicitly.
    for name, overlay in OVERLAYS.items():
        for key, value in overlay.items():
            monkeypatch.setenv(key, value)
        try:
            assert _snapshot(LCMConfig.from_env()) == _snapshot(_Before.from_env()), name
        finally:
            for key in overlay:
                monkeypatch.delenv(key)

    deployed = dict(
        database_path=str(tmp_path / "lcm.db"), context_threshold=0.75, fresh_tail_count=24,
        fresh_tail_max_tokens=24_000, leaf_chunk_tokens=8_000, threshold_full_sweep_enabled=True,
        large_output_externalization_enabled=True, large_output_externalization_threshold_chars=12_000,
        large_output_active_replay_stubbing_enabled=True, temporal_rollups_enabled=True,
        embeddings_enabled=False, summary_model="stub-model",
    )
    assert LCMConfig(**deployed).embeddings_enabled is False
    assert _snapshot(LCMConfig(**deployed), ignore=INERT_WHEN_OFF) == _snapshot(
        _Before(**deployed), ignore=INERT_WHEN_OFF
    )

    # Explicit off: disabled, and a bound session schedules no embedding work.
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is False
    started = []
    monkeypatch.setattr(maintenance_mod, "start_local_model_prepare", lambda *a, **k: started.append(a))
    config.database_path = str(tmp_path / "off.db")
    engine = LCMEngine(config=config, hermes_home=str(home))
    try:
        engine.on_session_start(SESSION, platform="cli", context_length=200_000)
        assert started == []
        assert maintenance_mod.incremental_embedding_state(config.database_path)[0] is None
    finally:
        engine.shutdown()


def test_t3_recall_never_waits_on_the_background_download(tmp_path, monkeypatch, fastembed, prepare):
    home = _clear_lcm_env(monkeypatch, tmp_path)
    release = threading.Event()
    fake = fastembed(download=lambda: release.wait(10.0))  # a slow download, held open
    engine = _engine(tmp_path, home)
    try:
        engine._store.append("earlier-session", {"role": "user", "content": "the zanzibar ferry leaves at noon"})
        engine.on_session_start(SESSION, platform="cli", context_length=200_000)
        assert maintenance_mod.local_model_state(engine._config) == ("preparing", "")

        result, elapsed = _recall(engine, "zanzibar ferry")

        assert elapsed < engine._config.recall_query_timeout_s
        assert any("zanzibar" in json.dumps(hit) for hit in result["hits"])
        assert result["degraded"] is True
        assert result["provenance"]["coverage"]["fts"] == "ok"
        assert all(result["provenance"]["coverage"].get(arm) != "ok" for arm in ("summary", "chunk"))

        # One prepare per process: a second bind and a second engine start none.
        engine.on_session_start(SESSION + "-2", platform="cli", context_length=200_000)
        other = _engine(tmp_path, home, "other.db")
        other.on_session_start(SESSION, platform="cli", context_length=200_000)
        other.shutdown()
        release.set()
        _join_prepare()
        assert fake.downloads == 1
        assert maintenance_mod.local_model_state(engine._config) == ("present", "")
        assert _profiles(engine._config.database_path) == ["chunk", "summary"]
    finally:
        release.set()
        engine.shutdown()


def test_t4_offline_warns_once_serves_full_text_and_does_not_retry(
    tmp_path, monkeypatch, fastembed, prepare, caplog,
):
    home = _clear_lcm_env(monkeypatch, tmp_path)

    def offline():
        raise OSError("network is unreachable")

    fake = fastembed(download=offline)
    engine = _engine(tmp_path, home)
    try:
        with caplog.at_level(logging.DEBUG, logger=maintenance_mod.__name__):
            engine._store.append("earlier-session", {"role": "user", "content": "the zanzibar ferry leaves at noon"})
            engine.on_session_start(SESSION, platform="cli", context_length=200_000)
            _join_prepare()
            assert engine.drain_embedding_maintenance(timeout=30)
            for index in range(3):  # later triggers and starts: no retry within the process
                engine.on_session_start(f"{SESSION}-{index}", platform="cli", context_length=200_000)
                assert engine.drain_embedding_maintenance(timeout=30)
            _join_prepare()
            result, _elapsed = _recall(engine, "zanzibar ferry")

        warnings = [r.getMessage() for r in caplog.records
                    if r.name == maintenance_mod.__name__ and r.levelno >= logging.WARNING]
        print(f"T4: warnings={warnings}; downloads={fake.downloads}")
        assert len(warnings) == 1 and "/lcm doctor" in warnings[0] and "network is unreachable" in warnings[0]
        assert fake.downloads == 1
        state = maintenance_mod.local_model_state(engine._config)
        assert state[0] == "unavailable" and "network is unreachable" in state[1]
        assert any("zanzibar" in json.dumps(hit) for hit in result["hits"])
        assert result["degraded"] is True
    finally:
        engine.shutdown()


def test_t5_catch_up_drains_across_triggers_without_reembedding(tmp_path, monkeypatch, fastembed):
    home = _clear_lcm_env(monkeypatch, tmp_path)
    for key, value in OVERLAYS["fleet-local"].items():  # explicit, so the drain is the only variable
        monkeypatch.setenv(key, value)
    fake = fastembed(cached=True)
    engine = _engine(tmp_path, home)
    db_path = engine._config.database_path
    total = 100
    try:
        node_ids = [
            str(engine._dag.add_node(SummaryNode(
                session_id=SESSION, depth=0, summary=f"existing summary number {index}",
                created_at=1.0, latest_at=1.0,
            )))
            for index in range(total)
        ]
        # A trigger whose budget allows one batch embeds one batch and leaves the rest
        # (a controlled clock: each batch takes 1 s against a 0.5 s budget).
        clock = [0.0]
        real_summary_run = command_mod._embedding_backfill_summary_run

        def one_second_batch(*args, **kwargs):
            clock[0] += 1.0
            return real_summary_run(*args, **kwargs)

        monkeypatch.setattr(maintenance_mod, "_CATCH_UP_BUDGET_S", 0.5, raising=False)
        monkeypatch.setattr(maintenance_mod, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
        monkeypatch.setattr(command_mod, "_embedding_backfill_summary_run", one_second_batch)
        first = maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        monkeypatch.setattr(maintenance_mod, "time", time)
        monkeypatch.setattr(command_mod, "_embedding_backfill_summary_run", real_summary_run)
        before = _summary_vectors(db_path)
        assert 0 < len(before) < total, first

        # The next trigger resumes and drains the remainder within its budget.
        monkeypatch.setattr(maintenance_mod, "_CATCH_UP_BUDGET_S", 20.0, raising=False)
        second = maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        after = _summary_vectors(db_path)
        print(f"T5: first={first}; second={second}; max_active={fake.max_active}")
        assert sorted(after) == sorted(node_ids)
        assert {key: after[key] for key in before} == before  # existing vectors untouched

        third = maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        assert third.startswith("summaries=") and third.split()[0].endswith(":0"), third
        summary_docs = [doc for doc in fake.documents if doc.startswith("existing summary number")]
        assert sorted(summary_docs) == sorted(f"existing summary number {i}" for i in range(total))
        assert 1 <= fake.max_active <= provider_mod._LOCAL_EMBED_MAX_WORKERS
    finally:
        engine.shutdown()


def _manifest_dependencies() -> tuple[set[str], list[str]]:
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    keys = set(re.findall(r"^([A-Za-z_]+):", manifest, re.MULTILINE))
    block = re.search(r"^pip_dependencies:\n((?:  - .*\n)+)", manifest, re.MULTILINE)
    specs = [] if block is None else [
        line.strip()[2:].strip().strip('"').strip("'") for line in block.group(1).splitlines()
    ]
    return keys, specs


def test_t6_plugin_yaml_declares_fastembed_and_is_a_member():
    from packaging.requirements import Requirement

    keys, specs = _manifest_dependencies()

    assert specs == ["fastembed==0.8.0"]
    requirement = Requirement(specs[0])
    assert requirement.name == "fastembed" and not requirement.url
    # The fallback file admits the declared version.
    semantic = [line.strip() for line in (REPO_ROOT / "requirements-semantic.txt").read_text().splitlines()
                if line.strip().startswith("fastembed")]
    assert Requirement(semantic[0]).specifier.contains("0.8.0")
    # PythonDeclaration.is_member (Hermes pm/plugin_declarations.py): not external and
    # (a pyproject or index requirements). No pyproject: membership comes from the manifest.
    external = "python_runtime" in keys
    has_pyproject = (REPO_ROOT / "pyproject.toml").exists()
    assert not external and not has_pyproject
    assert (has_pyproject or bool(specs)) is True


@pytest.mark.parametrize("state, status", [
    (("present", ""), "pass"),
    (("preparing", ""), "pass"),
    (("unavailable", "network is unreachable"), "warn"),
])
def test_t7_doctor_readiness_line(tmp_path, monkeypatch, fastembed, state, status):
    home = _clear_lcm_env(monkeypatch, tmp_path)
    fastembed(cached=True)
    engine = _engine(tmp_path, home)
    db_path = engine._config.database_path
    try:
        for index in range(2):
            engine._dag.add_node(SummaryNode(session_id=SESSION, depth=0, summary=f"covered {index}",
                                             created_at=1.0, latest_at=1.0))
        maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        engine._dag.add_node(SummaryNode(session_id=SESSION, depth=0, summary="not yet covered",
                                         created_at=1.0, latest_at=1.0))
        maintenance_mod._PREPARE[MODEL] = state

        check = lcm_tools._embedding_provider_health_check(engine)

        expected_model = state[0] if state[0] != "unavailable" else f"unavailable({state[1]})"
        print(f"T7 {state[0]}: {check['status']} {check['detail']['readiness']}")
        assert check["detail"]["readiness"] == f"model: {expected_model}; coverage: 2/3 summaries"
        assert check["status"] == status
    finally:
        engine.shutdown()


# Round 1 (T8-T10): the default model pairs with fastembed only; the exact pin line; the eval arms stay explicit.
@pytest.mark.parametrize("env, provider, model", [
    ({"LCM_EMBEDDING_PROVIDER": "voyage"}, "voyage", ""),
    ({"LCM_EMBEDDING_PROVIDER": "fastembed"}, "fastembed", MODEL),
    ({"LCM_EMBEDDING_PROVIDER": "fast-embed"}, "fast-embed", MODEL),
    ({}, "fastembed", MODEL),
    ({"LCM_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"}, "fastembed", "BAAI/bge-base-en-v1.5"),
])
def test_t8_default_model_pairs_with_fastembed_only(tmp_path, monkeypatch, env, provider, model):
    _clear_lcm_env(monkeypatch, tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    kwargs = {"LCM_EMBEDDING_PROVIDER": "embedding_provider", "LCM_EMBEDDING_MODEL": "embedding_model"}

    for config in (LCMConfig.from_env(), LCMConfig(**{kwargs[key]: value for key, value in env.items()})):
        assert (config.embedding_provider, config.embedding_model) == (provider, model)

    if not model:
        # Any other provider with no model keeps the pre-#1186 missing-model path and message.
        with pytest.raises(provider_mod.ProviderUnavailable, match="must both be set"):
            provider_mod.resolve_provider(LCMConfig.from_env())


def test_t9_plugin_yaml_carries_the_exact_pin_line():
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")

    assert "  - fastembed==0.8.0" in manifest.splitlines()
    assert _manifest_dependencies()[1] == ["fastembed==0.8.0"]
    try:
        import yaml
    except ImportError:  # the CI venv has no PyYAML; Hermes parses the manifest with it
        return
    assert yaml.safe_load(manifest)["pip_dependencies"] == ["fastembed==0.8.0"]


@pytest.mark.parametrize("env", [
    {},
    {"LCM_EMBEDDING_PROVIDER": "fastembed"},
    {"LCM_EMBEDDING_MODEL": MODEL},
])
def test_t10_eval_live_provider_requires_both_keys(tmp_path, monkeypatch, env):
    import importlib.util

    _clear_lcm_env(monkeypatch, tmp_path)
    monkeypatch.setenv("LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # The config default is on, so without the guard from_env would hand the eval a semantic arm.
    assert LCMConfig.from_env().embedding_model == MODEL
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval_recall_t10", REPO_ROOT / "scripts" / "eval_retrieval_recall.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    with pytest.raises(RuntimeError, match=r"^LCM_EMBEDDING_PROVIDER and LCM_EMBEDDING_MODEL are required$"):
        script._load_live_provider()


# Round 2 (T11-T14): cross-model review findings.
def test_t11_auto_paired_model_follows_provider_overrides(tmp_path, monkeypatch):
    _clear_lcm_env(monkeypatch, tmp_path)
    given = "BAAI/bge-base-en-v1.5"
    resolved = []
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda config: resolved.append(config) or object())

    # The proactive-recall override (tools.py): copy.copy, then assign the provider.
    for config, model in ((LCMConfig.from_env(), ""), (LCMConfig(embedding_model=given), given)):
        lcm_tools._resolve_recall_provider(SimpleNamespace(_config=config), provider_override="voyage")
        assert (resolved[-1].embedding_provider, resolved[-1].embedding_model) == ("voyage", model)
        assert config.embedding_provider == "fastembed"  # the engine's own config is untouched

    # dataclasses.replace: an auto-paired model follows the provider there and back ...
    ollama = dataclasses.replace(LCMConfig(), embedding_provider="ollama")
    assert ollama.embedding_model == ""
    assert dataclasses.replace(ollama, embedding_provider="fastembed").embedding_model == MODEL
    assert dataclasses.replace(LCMConfig.from_env(), embedding_provider="voyage").embedding_model == ""
    # ... and a given model, from code or the environment, never changes.
    assert dataclasses.replace(LCMConfig(embedding_model=given), embedding_provider="ollama").embedding_model == given
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", MODEL)
    from_env = LCMConfig.from_env()
    assert dataclasses.replace(from_env, embedding_provider="openai-compatible").embedding_model == MODEL
    replaced = dataclasses.replace(LCMConfig(), embedding_model=given)
    assert dataclasses.replace(replaced, embedding_provider="ollama").embedding_model == given


def test_t12_one_prepare_thread_per_process(tmp_path, monkeypatch, prepare):
    lock, gate = threading.Lock(), threading.Event()
    active, peak = [0], [0]

    def register(config, db_path, *, allow_download):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        gate.wait(5)
        with lock:
            active[0] -= 1
        return {}

    monkeypatch.setattr(command_mod, "_embedding_register_profiles", register)
    db_path = str(tmp_path / "lcm.db")
    first, second = LCMConfig(embedding_model=MODEL), LCMConfig(embedding_model="BAAI/bge-base-en-v1.5")

    assert maintenance_mod.start_local_model_prepare(db_path, first) is True
    admitted = maintenance_mod.start_local_model_prepare(db_path, second)
    time.sleep(0.2)  # time for a second thread, if one started, to enter the download
    alive = [t for t in threading.enumerate() if t.name == "lcm-embedding-prepare"]
    gate.set()
    _join_prepare()
    assert admitted is False and len(alive) == 1 and peak[0] == 1
    assert maintenance_mod.local_model_state(first) == ("present", "")
    assert maintenance_mod.local_model_state(second) is None

    # The other model is admitted by a later trigger, once the first prepare has finished.
    assert maintenance_mod.start_local_model_prepare(db_path, second) is True
    _join_prepare()
    assert maintenance_mod.local_model_state(second) == ("present", "") and peak[0] == 1


def test_t13_no_batch_starts_after_the_catch_up_budget(tmp_path, monkeypatch):
    clock, starts = [0.0], []

    def corpus(name):
        def run(*_args, **_kwargs):
            starts.append((name, clock[0]))
            clock[0] += 15.0  # each batch takes 15 s against the 20 s budget
            return {"status": "partial", "embedded": 1, "remaining": 10, "stop_reason": "", "error": None,
                    "failed": [], "selected": 1, "privacy_withheld": 0, "skipped": []}
        return run

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(maintenance_mod, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    monkeypatch.setattr(command_mod, "_embedding_read_connection", lambda _path: _Conn())
    monkeypatch.setattr(command_mod, "_embedding_current_profile", lambda _conn: {"identity_hash": "x"})
    monkeypatch.setattr(command_mod, "_chunk_current_profile", lambda _conn: {"identity_hash": "y"})
    monkeypatch.setattr(provider_mod, "probe_provider_availability", lambda _config: {"available": True})
    monkeypatch.setattr(command_mod, "_embedding_backfill_summary_run", corpus("summaries"))
    monkeypatch.setattr(command_mod, "_chunk_backfill_run", corpus("chunks"))

    outcome = maintenance_mod.run_incremental_embedding_pass(str(tmp_path / "lcm.db"), LCMConfig())

    assert starts == [("summaries", 0.0), ("summaries", 15.0)], (outcome, starts)
    assert outcome == "summaries=partial:2 chunks=deferred"


@pytest.mark.parametrize("state, expected", [
    (("present", ""), "present"),
    (("preparing", ""), "preparing"),
    (("unavailable", "network is unreachable"), "unavailable(network is unreachable)"),
])
def test_t14_doctor_text_shows_readiness(tmp_path, monkeypatch, fastembed, state, expected):
    from hermes_lcm.command import handle_lcm_command

    home = _clear_lcm_env(monkeypatch, tmp_path)
    fastembed(cached=True)
    engine = _engine(tmp_path, home)
    db_path = engine._config.database_path
    try:
        engine._session_id = SESSION
        engine.context_length = 200_000
        engine._dag.add_node(SummaryNode(session_id=SESSION, depth=0, summary="covered", created_at=1.0, latest_at=1.0))
        maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        engine._dag.add_node(SummaryNode(session_id=SESSION, depth=0, summary="not yet covered",
                                         created_at=1.0, latest_at=1.0))
        maintenance_mod._PREPARE[MODEL] = state

        text = handle_lcm_command("doctor", engine)

        assert f"embedding_readiness: model: {expected}; coverage: 1/2 summaries" in text.splitlines()
    finally:
        engine.shutdown()
