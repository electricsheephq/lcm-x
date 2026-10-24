"""Default-only small-host recall; all stores and cgroups are synthetic."""
# ruff: noqa: F811 -- imported pytest fixture is injected by name
from dataclasses import dataclass
import importlib
import json
from pathlib import Path

import pytest

from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from tests.test_lcm_recall import recall_engine  # noqa: F401


@dataclass
class DefaultOnConfig(LCMConfig):
    """Keep an explicit local-model fixture alongside the shipped default."""
    embeddings_enabled: bool = True
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@pytest.mark.parametrize("version,files,affinity,expected", [
    (2, {"cpu.max": "max 100000"}, None, 16),
    (2, {}, None, 16),
    (2, {"cpu.max": "broken"}, None, 16),
    (2, {"cpu.max": "100000 0"}, None, 16),
    (2, {"cpu.max": "0 100000"}, None, 16),
    (2, {"cpu.max": "800000 100000", "parent/cpu.max": "150000 100000",
         "parent/child/cpu.max": "300000 100000"}, 8, 1.5),
    (2, {"parent/child/cpu.max": "50000 100000"}, 4, 0.5),
    (2, {"cpu.max": "max 100000"}, 2, 2),
    (2, {"cpu.max": "600000 100000"}, 3, 3),
    (1, {"cpu.cfs_quota_us": "-1", "cpu.cfs_period_us": "100000"}, None, 16),
    (1, {"parent/cpu.cfs_quota_us": "125000", "parent/cpu.cfs_period_us": "100000",
         "parent/child/cpu.cfs_quota_us": "400000",
         "parent/child/cpu.cfs_period_us": "100000"}, 8, 1.25),
    (1, {"cpu.cfs_quota_us": "bad", "cpu.cfs_period_us": "100000"}, 4, 4),
])
def test_effective_cpus(monkeypatch, tmp_path, version, files, affinity, expected):
    capacity = importlib.import_module("hermes_lcm.cpu_capacity")
    mount = tmp_path / "cpu"
    for name, value in files.items():
        target = mount / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    proc = {
        "/proc/self/cgroup": "0::/tenant/parent/child" if version == 2 else "5:cpu,cpuacct:/tenant/parent/child",
        "/proc/self/mountinfo": f"1 0 0:1 /tenant {mount} rw - "
                                + ("cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,cpu,cpuacct"),
    }
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: proc[str(p)] if str(p) in proc else read_text(p, *a, **k))
    def get_affinity(pid):
        if affinity is None:
            raise OSError("unavailable")
        return set(range(affinity))
    monkeypatch.setattr(capacity.os, "sched_getaffinity", get_affinity, raising=False)
    monkeypatch.setattr(capacity.os, "cpu_count", lambda: 16)
    capacity.effective_cpus.cache_clear()
    try:
        assert capacity.effective_cpus() == expected
        monkeypatch.setattr(capacity.os, "cpu_count", lambda: 99)
        assert capacity.effective_cpus() == expected  # process cache
    finally:
        capacity.effective_cpus.cache_clear()


def test_effective_cpus_without_proc_or_affinity(monkeypatch):
    capacity = importlib.import_module("hermes_lcm.cpu_capacity")
    monkeypatch.delattr(capacity.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(capacity.os, "cpu_count", lambda: None)
    capacity.effective_cpus.cache_clear()
    try:
        assert capacity.effective_cpus() == 1
    finally:
        capacity.effective_cpus.cache_clear()


@pytest.mark.parametrize("raw", [None, "true", "false"])
def test_provenance(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", raw)
    config = DefaultOnConfig.from_env()
    assert config.embeddings_enabled_explicit is (raw is not None)
    assert config.embeddings_enabled is (raw != "false")
    assert LCMConfig(embeddings_enabled=True).embeddings_enabled_explicit
    assert LCMConfig().embeddings_enabled is True


@pytest.mark.parametrize("cpus", [0.5, 1, 2])
@pytest.mark.parametrize("include", ["all", "summaries", "verbatim"])
def test_default_small_host_fts(recall_engine, monkeypatch, cpus, include):
    engine = recall_engine
    engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus, raising=False)
    for name in ("_resolve_recall_provider", "_lcm_recall_has_usable_vector_corpus",
                 "_lcm_recall_summary_arm", "_lcm_recall_chunk_arm"):
        monkeypatch.setattr(tools, name, lambda *a, **k: pytest.fail("semantic work on small host"))
    wanted = engine._store.append("keep", {"role": "user", "content": "cobalt evidence"})
    engine._store.append("exclude", {"role": "user", "content": "cobalt private"})
    result = json.loads(tools.lcm_recall({"query": "cobalt", "include": include,
        "detail": "answer_ready", "seen_refs": [], "exclude_session_ids": ["exclude"]}, engine=engine))
    assert result["degraded"] is True
    assert result["degraded_reason"] == "cpu_limited"
    assert result["provenance"]["arms_run"] == ["fts"]
    assert result["provenance"]["coverage"]["fts"] == "ok"
    for arm in ({"all": ["summary", "chunk"], "summaries": ["summary"], "verbatim": ["chunk"]}[include]):
        assert result["provenance"]["coverage"][arm] == "disabled"
    assert [hit["store_id"] for hit in result["hits"]] == [wanted]
    assert result["hits"][0]["exact_ref"]
    assert engine._config.embeddings_enabled is True  # catch-up remains enabled


@pytest.mark.parametrize("explicit,cpus", [(True, 1), (False, 4)])
def test_explicit_or_normal_host_keeps_semantic(recall_engine, monkeypatch, explicit, cpus):
    config = DefaultOnConfig.from_env()
    config.embeddings_enabled_explicit = explicit
    recall_engine._config = config
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus, raising=False)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True, "detail": "synthetic backend"})
    called = []
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **k: called.append(True))
    result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=recall_engine))
    assert called
    assert "cpu_limited" not in result.get("degraded_reason", "")


def test_small_host_fts_failure_is_honest(recall_engine, monkeypatch):
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: 2, raising=False)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **k: ([], {"error": "unavailable"}))
    result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=recall_engine))
    assert result["provenance"]["coverage"]["fts"] == "none"
    assert "cpu_limited" in result["degraded_reason"]
    assert "full-text arm unavailable" in result["degraded_reason"]


def test_doctor_small_host(recall_engine, monkeypatch):
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1.5, raising=False)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda *a: {"provider": "fastembed", "available": True})
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *a: "model: present; coverage: 0/0 summaries")
    check = tools._embedding_provider_health_check(recall_engine)
    assert check["status"] == "warn"
    assert check["detail"]["cpu_limit_reason"] == "semantic recall limited on this host (1.5 effective CPUs); full-text recall is active"


@pytest.mark.parametrize("raw", ["true", "false"])
def test_explicit_environment_is_not_cpu_limited(monkeypatch, raw):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", raw)
    monkeypatch.setattr(tools, "effective_cpus", lambda: pytest.fail("explicit config must not probe CPU"))
    assert tools.default_semantic_cpu_limited(DefaultOnConfig.from_env()) is False


def test_explicit_cloud_provider_is_not_cpu_limited(monkeypatch):
    config = DefaultOnConfig.from_env()
    config.embedding_provider = "voyage"
    monkeypatch.setattr(tools, "effective_cpus", lambda: pytest.fail("cloud must not probe CPU"))
    assert tools.default_semantic_cpu_limited(config) is False


@pytest.mark.parametrize("provider", ["FASTEMBED", " fastembed ", "FAST-EMBED", " fast-embed "])
def test_local_provider_spellings_share_policy(recall_engine, monkeypatch, provider):
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", provider)
    config = DefaultOnConfig.from_env()
    recall_engine._config = config
    monkeypatch.setattr(tools, "effective_cpus", lambda: 2)
    assert tools.default_semantic_cpu_limited(config) is True
    assert tools._embedding_provider_health_check(recall_engine)["detail"]["cpu_limit_reason"] == (
        "semantic recall limited on this host (2 effective CPUs); full-text recall is active"
    )


@pytest.mark.parametrize("override", ["voyage", "ollama", "fastembed", " voyage ", "", " ", "\t"])
def test_request_provider_override_preserves_explicit_choice(recall_engine, monkeypatch, override):
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: 2)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True, "detail": "synthetic backend"})
    called = []
    monkeypatch.setattr(tools, "_resolve_recall_provider",
                        lambda *a, **k: called.append(k["provider_override"]))
    result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=recall_engine,
                                        provider_override=override))
    if override.strip():
        assert called == [override.strip()]
        assert "cpu_limited" not in result.get("degraded_reason", "")
    else:
        assert not called
        assert "cpu_limited" in result["degraded_reason"]
