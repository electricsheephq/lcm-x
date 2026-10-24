"""r3: CPU fallback, diagnostic completeness and proactive OFF parity."""
# ruff: noqa: F811 -- imported pytest fixture is injected by name
import importlib
import json
from pathlib import Path

import pytest

from hermes_lcm import tools
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.engine import LCMEngine
from tests.test_issue_1186_smallhost import DefaultOnConfig
from tests.test_lcm_recall import recall_engine  # noqa: F401


@pytest.mark.parametrize("broken", ["cgroup", "mountinfo", "quota"])
@pytest.mark.parametrize("affinity", [None, 4])
def test_undecodable_proc_falls_back(monkeypatch, tmp_path, broken, affinity):
    capacity = importlib.import_module("hermes_lcm.cpu_capacity")
    mount = tmp_path / "cpu"
    mount.mkdir()
    paths = {"cgroup": tmp_path / "cgroup", "mountinfo": tmp_path / "mountinfo", "quota": mount / "cpu.max"}
    paths["cgroup"].write_text("0::/tenant")
    paths["mountinfo"].write_text(f"1 0 0:1 /tenant {mount} rw - cgroup2 cgroup rw")
    paths["quota"].write_text("max 100000")
    paths[broken].write_bytes(b"bad\xff")
    read_text = Path.read_text
    def read(path, *args, **kwargs):
        target = paths.get(str(path).removeprefix("/proc/self/"), path)
        return read_text(target, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    def get_affinity(_pid):
        if affinity is None:
            raise OSError("unavailable")
        return set(range(affinity))
    monkeypatch.setattr(capacity.os, "sched_getaffinity", get_affinity, raising=False)
    monkeypatch.setattr(capacity.os, "cpu_count", lambda: 16)
    capacity.effective_cpus.cache_clear()
    try:
        assert capacity.effective_cpus() == (affinity or 16)
    finally:
        capacity.effective_cpus.cache_clear()


@pytest.mark.parametrize("host,expected", [(2, 2), (None, 8)])
def test_quota_without_affinity_respects_known_host(monkeypatch, host, expected):
    capacity = importlib.import_module("hermes_lcm.cpu_capacity")
    files = {"/proc/self/cgroup": "0::/", "/proc/self/mountinfo": "1 0 0:1 / /cpu rw - cgroup2 cgroup rw",
             "/cpu/cpu.max": "800000 100000"}
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    monkeypatch.delattr(capacity.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(capacity.os, "cpu_count", lambda: host)
    capacity.effective_cpus.cache_clear()
    try:
        assert capacity.effective_cpus() == expected
    finally:
        capacity.effective_cpus.cache_clear()


@pytest.mark.parametrize("state", ["healthy", "missing", "unavailable", "preparing"])
def test_cpu_limited_doctor_keeps_provider_and_readiness(recall_engine, monkeypatch, state):
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    available = state != "missing"
    reason = "FastEmbed is not installed" if not available else "provider configured"
    probe = {"provider": "fastembed", "model": "test", "available": available, "detail": reason}
    called = []
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: called.append("probe") or probe)
    model = {"healthy": "present", "missing": "unavailable(FastEmbed is not installed)",
             "unavailable": "unavailable(download failed)", "preparing": "preparing"}[state]
    readiness = f"model: {model}; coverage: 0/1 summaries"
    monkeypatch.setattr(tools, "_embedding_readiness", lambda *args: called.append("readiness") or readiness)
    check = tools._embedding_provider_health_check(recall_engine)
    assert called == ["probe", "readiness"]
    assert check["status"] == "warn"
    detail = check["detail"]
    assert detail["available"] is available and detail["reason"] == reason
    assert detail["readiness"] == readiness
    assert detail["cpu_limited"] is True and detail["effective_cpus"] == 1
    assert "semantic recall limited on this host (1 effective CPUs); full-text recall is active" in detail["cpu_limit_reason"]
    guidance = doctor_guidance_for_check(check)
    if state == "healthy":
        assert guidance["action"] == "safe/ignore"
        assert "expected on this host" in guidance["rationale"]
        assert "set LCM_EMBEDDINGS_ENABLED=true" in guidance["operator_action"]
        assert "more than 2 effective CPUs" in guidance["operator_action"]
        assert "reinstall" not in guidance["operator_action"]
    else:
        assert guidance["action"] == "inspect"
        assert "warmup" in guidance["operator_action"]


@pytest.fixture
def default_engine(tmp_path):
    config = DefaultOnConfig.from_env()
    config.database_path = str(tmp_path / "default.db")
    config.temporal_rollups_enabled = False
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "synthetic-home"))
    yield engine
    engine.shutdown()


@pytest.mark.parametrize("override", ["", " ", "voyage", " fastembed "])
def test_cpu_limited_proactive_matches_off_unless_overridden(default_engine, monkeypatch, override):
    engine = default_engine
    engine._config.proactive_recall_enabled = True
    engine._config.proactive_recall_provider = override
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    called = []
    def recall(*args, **kwargs):
        called.append(kwargs["provider_override"])
        return json.dumps({"hits": [{"snippet": "older cobalt memory", "score": 0.5}]})
    monkeypatch.setattr(tools, "lcm_recall", recall)
    message = engine._build_proactive_recall_message([{"role": "user", "content": "cobalt"}], "user", set())
    if override.strip():
        assert called and message is not None
        assert "older cobalt memory" in message["content"]
    else:
        assert message is None and called == []


@pytest.mark.parametrize("cpus", [2.01, 2.5, 3])
def test_default_normal_host_retains_semantic(recall_engine, monkeypatch, cpus):
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    called = []
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *args, **kwargs: called.append(True))
    result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=recall_engine))
    assert called
    assert "cpu_limited" not in result.get("degraded_reason", "")


def test_cpu_limited_default_still_arms_maintenance(default_engine, monkeypatch):
    engine_mod = importlib.import_module("hermes_lcm.engine")
    maintenance = importlib.import_module("hermes_lcm.embedding_maintenance")
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    assert tools.default_semantic_cpu_limited(default_engine._config)
    scheduled, prepared = [], []
    monkeypatch.setattr(maintenance, "start_local_model_prepare", lambda *args, **kwargs: prepared.append(args))
    monkeypatch.setattr(engine_mod._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
                        lambda *args, **kwargs: scheduled.append((args, kwargs)))
    default_engine._schedule_embedding_maintenance()
    assert prepared and scheduled
    assert scheduled[0][0][0][1] == "embeddings"
    assert default_engine._config.embeddings_enabled is True


def test_cpu_limited_slash_doctor_reports_missing_fastembed(default_engine, monkeypatch):
    import importlib.util
    from hermes_lcm.command import handle_lcm_command
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "fastembed" else find_spec(name, *a))
    check = tools._embedding_provider_health_check(default_engine)
    assert isinstance(check["detail"], dict)
    assert check["detail"]["available"] is False
    assert "FastEmbed is not installed" in check["detail"]["reason"]
    assert check["detail"]["cpu_limited"] is True
    assert "model: unavailable(" in check["detail"]["readiness"]
    text = handle_lcm_command("doctor", default_engine)
    assert "FastEmbed is not installed" in text
    assert "semantic recall limited on this host (1 effective CPUs); full-text recall is active" in text
    assert "embedding_readiness: model: unavailable(" in text
    assert "pip install fastembed" in text
