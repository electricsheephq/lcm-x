from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _run_in_subprocess(
    tmp_path, *, sabotage="", run_kwargs, sys_path_prepend=None, env_extra=None
):
    """Execute phase_a.run() in a child interpreter.

    Full-matrix runs register worktree modules whose provider singletons hold
    file descriptors and can shadow the conftest's hermes_lcm registration —
    in-process execution contaminates the shared pytest process (CI's low-FD
    canary caught this), so every run() goes through a child python.
    """
    worktree = Path(__file__).resolve().parents[1]
    runner = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    out = tmp_path / "sub-receipt"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        + (
            f"sys.path.insert(0, {str(sys_path_prepend)!r})\n"
            if sys_path_prepend is not None
            else ""
        )
        + f"spec = importlib.util.spec_from_file_location('phase_a_sub', {str(runner)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        + sabotage
        + f"run_kwargs = {run_kwargs!r}\n"
        "if run_kwargs.get('hermes_repo') is not None:\n"
        "    run_kwargs['hermes_repo'] = Path(run_kwargs['hermes_repo'])\n"
        f"code, receipt = mod.run(Path({str(worktree)!r}), Path({str(out)!r}), **run_kwargs)\n"
        "print('RECEIPT::' + str(receipt))\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LCM_", "HERMES_"))}
    for key in ("VOYAGE_API_KEY", "LCM_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"):
        env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        [sys.executable, str(wrapper)], capture_output=True, text=True, env=env,
        cwd=str(worktree), timeout=600,
    )
    receipt_path = None
    for line in completed.stdout.splitlines():
        if line.startswith("RECEIPT::"):
            receipt_path = Path(line.split("::", 1)[1])
    assert receipt_path is not None, completed.stdout + completed.stderr
    return (
        completed.returncode,
        receipt_path.read_text(encoding="utf-8"),
        completed.stderr,
    )


def _runner_module():
    worktree = Path(__file__).resolve().parents[1]
    path = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    spec = importlib.util.spec_from_file_location("phase_a_tool_matrix_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doctor_scenario(phase_a, monkeypatch, checks):
    monkeypatch.setattr(phase_a.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(phase_a, "_payload", lambda *_args: {
        "checks": checks, "overall": "warnings",
    })
    context = {
        "finance": {"content": "taxi costs $60; train costs $20", "store_id": "fixture"},
        "atlas": {},
    }
    return phase_a._run_matrix(
        object(), ["lcm_doctor"], context, patterns=False, posture="local", missing=[],
    )[0]


def test_local_doctor_accepts_only_preparing_warning(monkeypatch):
    checks = [
        {"check": "inactive_process", "status": "pass"},
        {"check": "embedding_provider_health", "status": "warn", "detail": {
            "readiness": "model: preparing; background initialization",
            "impact": "lcm_recall keeps answering from full text",
        }},
    ]
    assert _doctor_scenario(_runner_module(), monkeypatch, checks)[2] == "PASS"


@pytest.mark.parametrize("name,readiness,impact", [
    ("embedding_provider_health", "model: failed; initialization", "full text"),
    ("embedding_provider_health", "model: preparing; initialization", "unavailable"),
    ("embedding_provider_health", "", "full text"),
    ("another_check", "model: preparing; initialization", "full text"),
])
def test_local_doctor_rejects_other_warnings(monkeypatch, name, readiness, impact):
    checks = [
        {"check": "inactive_process", "status": "pass"},
        {"check": name, "status": "warn", "detail": {
            "readiness": readiness, "impact": impact,
        }},
    ]
    assert _doctor_scenario(_runner_module(), monkeypatch, checks)[2] == "FAIL"


@pytest.mark.parametrize("status", ["warn", "pass"])
def test_prepared_doctor_requires_pass_after_warmup_and_backfill(monkeypatch, status):
    phase_a = _runner_module()
    engine, calls = object(), []
    monkeypatch.setattr(phase_a.importlib.util, "find_spec", lambda _name: object())

    def command(text, target):
        assert target is engine
        calls.append(text)
        return "status: ready" if text == "embed warmup" else "status: complete"

    def doctor(target, tool, args):
        assert target is engine and tool == "lcm_doctor" and args == {}
        assert calls == ["embed warmup", "embed backfill --apply --limit 50"]
        return {"checks": [{"check": "embedding_provider_health", "status": status}],
                "overall": "healthy" if status == "pass" else "warnings"}

    monkeypatch.setattr(phase_a, "_payload", doctor)
    row = phase_a._prepared_doctor({"command": SimpleNamespace(handle_lcm_command=command)}, engine)
    assert row[:3] == ("local", "lcm_doctor (prepared)", "PASS" if status == "pass" else "FAIL")
    assert calls == ["embed warmup", "embed backfill --apply --limit 50"]


def _fake_hermes_repo(root: Path, *, expected_surface: bool = True) -> Path:
    repo = root / "hermes-agent"
    agent = repo / "agent"
    agent.mkdir(parents=True)
    (agent / "__init__.py").write_text("", encoding="utf-8")
    surface = (
        "    def select_context(self, *args, **kwargs):\n        return None\n"
        "    def on_turn_complete(self, *args, **kwargs):\n        return None\n"
        if expected_surface
        else ""
    )
    (agent / "context_engine.py").write_text(
        "class ContextEngine:\n"
        "    def get_status(self):\n        return {}\n"
        + surface,
        encoding="utf-8",
    )
    return repo


def _clean_hermes_git_sabotage() -> str:
    return (
        "mod._hermes_git_identity = lambda _p: "
        "{'head': 'hermes123', 'dirty': False, 'error': None}\n"
    )


def test_dev_posture_covers_every_tool_despite_disabled_tools_env(tmp_path):
    worktree = Path(__file__).resolve().parents[1]
    runner = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    env = os.environ.copy()
    env["LCM_DISABLED_TOOLS"] = "lcm_recall"
    # Pin the provider: a parent-environment LCM_GAUNTLET_CLOUD_PROVIDER would
    # change which credential gate (and SKIP wording) the receipt asserts.
    env["LCM_GAUNTLET_CLOUD_PROVIDER"] = "voyage"
    env.pop("VOYAGE_API_KEY", None)
    env.pop("LCM_EMBEDDING_API_KEY", None)
    env.pop("SILICONFLOW_API_KEY", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--worktree",
            str(worktree),
            "--out",
            str(tmp_path),
            "--no-expect-365-fixed",
        ],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    receipt = (tmp_path / "PHASE-A-RECEIPT.md").read_text(encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr + receipt
    assert "DEV RUN — identity unbound" in receipt
    assert "Hermes checkout provenance: `Hermes checkout not configured`" in receipt
    # Probe in the runner's import environment, without pytest's in-process host modules.
    host_probe = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path[0] = {str(runner.parent)!r}\n"
         "try:\n"
         "    from agent.context_engine import ContextEngine\n"
         "except ModuleNotFoundError as exc:\n"
         "    if exc.name not in {'agent', 'agent.context_engine'}: raise\n"
         "    print('engine=stubbed-context-base (NOT the hermes surface)')\n"
         "else:\n"
         "    print('engine=hermes-context-engine (')\n"],
        cwd=worktree, env=env, text=True, capture_output=True, timeout=10,
    )
    assert host_probe.returncode == 0, host_probe.stderr
    assert host_probe.stdout.strip() in receipt
    assert "LCM_DISABLED_TOOLS" in receipt
    assert "Registry coverage: `COMPLETE`" in receipt
    assert "Missing matrix rows: `none`" in receipt
    assert "Unexpected scenario rows: `none`" in receipt
    assert receipt.count("| local | `lcm_") == 17
    assert "| local | `lcm_doctor` | PASS |" in receipt
    prepared_result = "PASS" if importlib.util.find_spec("fastembed") else "SKIP"
    assert f"| local | `lcm_doctor (prepared)` | {prepared_result} |" in receipt
    assert "| cloud-default | `lcm_recall` | SKIP | SKIP: VOYAGE_API_KEY is absent |" in receipt
    for battery in (
        "planted-secret",
        "opt-out",
        "durable-redaction",
        "misconfiguration",
    ):
        assert f"| {battery} | SKIP | SKIP: VOYAGE_API_KEY is absent |" in receipt
    assert "rerank transform covered by product regressions, not a live battery" in receipt
    assert "Exit verdict: `PASS`" in receipt


def test_release_mode_blocks_the_same_cloud_skip(tmp_path):
    hermes_repo = _fake_hermes_repo(tmp_path)
    sabotage = (
        "mod._git_identity = lambda _p: "
        "{'head': 'abc123', 'tree': 'tree123', 'tag': 'v0.0.0-rc1', 'dirty': False}\n"
        + _clean_hermes_git_sabotage()
    )
    code, receipt, stderr = _run_in_subprocess(
        tmp_path,
        sabotage=sabotage,
        run_kwargs={
            "release": True,
            "rc_tag": "v0.0.0-rc1",
            "hermes_repo": str(hermes_repo),
            "expect_365_fixed": False,
        },
    )
    module_path = (hermes_repo / "agent/context_engine.py").resolve()
    assert code != 0, stderr + receipt
    assert "| cloud-default | `lcm_recall` | SKIP |" in receipt
    assert f"Hermes ContextEngine module __file__: `{module_path}`" in receipt
    assert (
        f"Hermes ContextEngine module sha256: "
        f"`{hashlib.sha256(module_path.read_bytes()).hexdigest()}`"
    ) in receipt
    assert "Hermes checkout HEAD SHA: `hermes123`" in receipt
    assert "Hermes checkout dirty state: `clean`" in receipt
    assert "Exit verdict: `BLOCKED`" in receipt


@pytest.mark.parametrize("ambient", [None, "outside-hermes"])
@pytest.mark.parametrize("abort", [False, True])
def test_scrub_pins_fresh_synthetic_hermes_home_and_restores_environment(
    tmp_path, monkeypatch, ambient, abort
):
    phase_a = _runner_module()
    out = tmp_path / "out"
    if ambient is None:
        monkeypatch.delenv("HERMES_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ambient))
    original = dict(os.environ)
    homes = []
    for _ in range(2):
        try:
            with phase_a._scrubbed_environment(tmp_path, out):
                home = Path(os.environ["HERMES_HOME"])
                assert home.is_dir() and home.is_relative_to(out)
                assert not list(home.iterdir())
                assert str(home) != original.get("HERMES_HOME")
                homes.append(home)
                if abort:
                    raise RuntimeError("synthetic interruption")
        except RuntimeError as exc:
            assert abort and str(exc) == "synthetic interruption"
        assert dict(os.environ) == original
    assert homes[0] != homes[1]


def test_home_resolver_outside_out_blocks_before_engine_construction(tmp_path):
    outside = tmp_path / "outside-hermes"
    sabotage = (
        "from types import ModuleType\n"
        "host = ModuleType('hermes_constants')\n"
        f"host.get_hermes_home = lambda: Path({str(outside)!r})\n"
        "sys.modules['hermes_constants'] = host\n"
        "def forbidden_engine(*args, **kwargs):\n"
        "    raise AssertionError('engine must not be constructed')\n"
        "mod._engine = forbidden_engine\n"
    )
    code, receipt, stderr = _run_in_subprocess(
        tmp_path, sabotage=sabotage, run_kwargs={}
    )
    assert code == 1, stderr + receipt
    assert "Exit verdict: `BLOCKED`" in receipt
    assert "HERMES_HOME" in receipt and str(outside) in receipt
    assert "| local |" not in receipt and "| cloud-default |" not in receipt
    assert "engine must not be constructed" not in receipt


@pytest.mark.parametrize("drained", [True, False])
def test_run_quiesces_maintenance_inside_synthetic_home_before_cleanup(
    tmp_path, monkeypatch, drained
):
    phase_a = _runner_module()
    worktree = Path(__file__).resolve().parents[1]
    out = tmp_path / "out"
    calls = []
    original = dict(os.environ)

    def quiesce(name, timeout):
        assert timeout > 0
        assert Path(os.environ["HERMES_HOME"]).is_relative_to(out)
        calls.append(name)
        return drained

    engine = SimpleNamespace(**{
        name: SimpleNamespace(quiesce=lambda timeout, name=name: quiesce(name, timeout))
        for name in ("_ROLLUP_MAINTENANCE_SCHEDULER", "_EMBEDDING_MAINTENANCE_SCHEDULER")
    })

    def load(*args, **kwargs):
        def cleanup():
            assert Path(os.environ["HERMES_HOME"]).is_relative_to(out)
            calls.append("cleanup")
        return {"engine": engine}, "fake-host", None, cleanup

    monkeypatch.setattr(phase_a, "_load", load)
    monkeypatch.setattr(phase_a.importlib, "import_module", lambda _name: SimpleNamespace(
        get_hermes_home=lambda: tmp_path / "outside-hermes"
    ))
    code, receipt = phase_a.run(worktree, out)
    text = receipt.read_text()
    assert code == 1 and "Exit verdict: `BLOCKED`" in text
    assert calls == ["_ROLLUP_MAINTENANCE_SCHEDULER", "_EMBEDDING_MAINTENANCE_SCHEDULER", "cleanup"]
    assert ("HERMES_HOME maintenance did not quiesce" in text) is (not drained)
    assert dict(os.environ) == original


def test_scrub_preserves_cloud_environment_allowlist(tmp_path, monkeypatch):
    phase_a = _runner_module()
    allowed = {
        "LCM_GAUNTLET_CLOUD_PROVIDER": "openai-compatible",
        "LCM_GAUNTLET_CLOUD_MODEL": "test-model",
        "LCM_EMBEDDING_BASE_URL": "https://example.invalid/v1",
        "LCM_EMBEDDING_API_KEY": "placeholder-key",
        "LCM_EMBEDDING_API_KEY_ENV": "TEST_EMBEDDING_KEY",
        "VOYAGE_API_KEY": "placeholder-voyage-key",
    }
    for key, value in allowed.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LCM_DISABLED_TOOLS", "lcm_recall")

    with phase_a._scrubbed_environment(tmp_path, tmp_path / "out") as scrubbed:
        assert {key: os.environ.get(key) for key in allowed} == allowed
        assert "LCM_DISABLED_TOOLS" not in os.environ
        assert "LCM_DISABLED_TOOLS" in scrubbed


def test_hermes_git_identity_is_read_from_the_selected_checkout(
    tmp_path, monkeypatch
):
    phase_a = _runner_module()
    calls = []
    outputs = {
        ("rev-parse", "--show-toplevel"): str(tmp_path),
        ("rev-parse", "HEAD"): "hermes123",
        ("status", "--porcelain", "--untracked-files=all"): "",
    }

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, outputs[tuple(args[3:])], "")

    monkeypatch.setattr(phase_a.subprocess, "run", fake_run)
    assert phase_a._hermes_git_identity(tmp_path) == {
        "head": "hermes123", "dirty": False, "error": None,
    }
    assert all(call[:3] == ["git", "-C", str(tmp_path)] for call in calls)


@pytest.mark.parametrize("provider", ["fastembed", "unknown-provider"])
def test_cloud_posture_blocks_non_product_cloud_provider(
    tmp_path, monkeypatch, provider
):
    phase_a = _runner_module()
    worktree = Path(__file__).resolve().parents[1]
    mod, _surface, _identity, world_cleanup = phase_a._load(
        worktree, hermes_repo=_fake_hermes_repo(tmp_path / "loader")
    )
    request_cleanup = world_cleanup  # restore the module world when done
    monkeypatch.setenv("LCM_GAUNTLET_CLOUD_PROVIDER", provider)

    records, batteries = phase_a._cloud_rows(
        mod,
        tmp_path,
        ["lcm_status"],
        [],
        expect_365_fixed=False,
    )

    assert records[0][2] == "BLOCKED"
    assert all(row[1] == "BLOCKED" for row in batteries)
    assert [row[0] for row in batteries] == [
        "planted-secret",
        "opt-out",
        "durable-redaction",
        "misconfiguration",
    ]
    assert f"provider {provider!r} is not in the product cloud set" in records[0][3]
    assert (
        phase_a._verdict(
            records,
            batteries,
            release=False,
            coverage_failed=False,
            preflight=[],
        )
        == "BLOCKED"
    )
    request_cleanup()


def test_known_leak_blocks_release_but_stays_loud_but_ok_in_dev():
    phase_a = _runner_module()
    batteries = [
        ("planted-secret", "KNOWN-LEAK-ON-BASE", "synthetic known leak")
    ]

    def verdict(release):
        return phase_a._verdict(
            [], batteries, release=release, coverage_failed=False, preflight=[]
        )

    assert verdict(False) == "PASS"
    assert verdict(True) == "BLOCKED"
    assert (
        phase_a._verdict(
            [("cloud", "synthetic", "KNOWN-LEAK-ON-BASE", "synthetic")],
            [],
            release=True,
            coverage_failed=False,
            preflight=[],
        )
        == "BLOCKED"
    )


@pytest.mark.parametrize(
    ("identity", "tag", "expected"),
    [
        ({"dirty": False, "tag": "v1.2.3-rc1"}, "v1.2.3-rc1", []),
        ({"dirty": False, "tag": "v1.2.3"}, "v1.2.3", ["RC-suffixed"]),
        ({"dirty": True, "tag": "v1.2.3-rc1"}, "v1.2.3-rc1", ["dirty"]),
        ({"dirty": False, "tag": "v1.2.3-rc2"}, "v1.2.3-rc1", ["exact tag"]),
        (
            {"dirty": False, "tag": "release-1.2.3"},
            "release-1.2.3",
            ["RC-suffixed"],
        ),
    ],
)
def test_release_identity_requires_clean_exact_rc_tag(identity, tag, expected):
    failures = _runner_module()._release_identity_failures(identity, tag)
    assert all(
        any(fragment in failure for failure in failures) for fragment in expected
    )
    assert bool(failures) is bool(expected)


def test_release_mode_rejects_ambient_hermes_module(tmp_path):
    ambient = tmp_path / "ambient"
    ambient_agent = ambient / "agent"
    ambient_agent.mkdir(parents=True)
    (ambient_agent / "__init__.py").write_text("", encoding="utf-8")
    (ambient_agent / "context_engine.py").write_text(
        "class ContextEngine:\n    def get_status(self):\n        return {}\n",
        encoding="utf-8",
    )
    hermes_repo = tmp_path / "claimed-hermes"
    hermes_repo.mkdir()
    sabotage = (
        "mod._git_identity = lambda _p: "
        "{'head': 'abc123', 'tree': 'tree123', 'tag': 'v1.2.3-rc1', 'dirty': False}\n"
        + _clean_hermes_git_sabotage()
    )
    code, receipt, stderr = _run_in_subprocess(
        tmp_path,
        sabotage=sabotage,
        run_kwargs={
            "release": True,
            "rc_tag": "v1.2.3-rc1",
            "hermes_repo": str(hermes_repo),
        },
        sys_path_prepend=ambient,
    )

    assert code != 0, stderr + receipt
    assert "agent.context_engine imported from" in receipt
    assert "not --hermes-repo" in receipt
    assert "Exit verdict: `BLOCKED`" in receipt


@pytest.mark.parametrize(
    ("hermes_identity", "expected"),
    [
        ({"head": "hermes123", "dirty": True, "error": None}, "is dirty"),
        (
            {"head": "unavailable", "dirty": None, "error": "not a git checkout"},
            "not a git checkout",
        ),
    ],
)
def test_release_mode_requires_clean_hermes_git_checkout(
    tmp_path, hermes_identity, expected
):
    hermes_repo = _fake_hermes_repo(tmp_path)
    sabotage = (
        "mod._git_identity = lambda _p: "
        "{'head': 'abc123', 'tree': 'tree123', 'tag': 'v1.2.3-rc1', 'dirty': False}\n"
        f"mod._hermes_git_identity = lambda _p: {hermes_identity!r}\n"
    )
    code, receipt, stderr = _run_in_subprocess(
        tmp_path,
        sabotage=sabotage,
        run_kwargs={
            "release": True,
            "rc_tag": "v1.2.3-rc1",
            "hermes_repo": str(hermes_repo),
        },
    )

    assert code != 0, stderr + receipt
    assert expected in receipt
    assert "Exit verdict: `BLOCKED`" in receipt


def test_release_mode_rejects_hermes_module_missing_surface(tmp_path):
    phase_a = _runner_module()
    hermes_repo = _fake_hermes_repo(tmp_path, expected_surface=False)

    with pytest.raises(RuntimeError, match="Hermes ContextEngine surface invalid") as exc:
        phase_a._load(
            Path(__file__).resolve().parents[1],
            hermes_repo=hermes_repo,
            release=True,
        )

    assert "select_context (assemble marker)" in str(exc.value)
    assert "on_turn_complete (ingest marker)" in str(exc.value)


def test_privacy_shapes_include_365_compositions_and_chunk_bearing_turn():
    phase_a = _runner_module()
    fixtures = phase_a._privacy_fixtures()
    content = "\n".join(item["content"] for item in fixtures)
    assert "password: -----BEGIN PRIVATE KEY-----\n" in content
    assert "passphrase=-----BEGIN PRIVATE KEY-----\n" in content
    assert 'password="-----BEGIN PRIVATE KEY-----\n' in content
    assert any(
        item["kind"] == "chunk" and len(item["content"]) > 2_000 for item in fixtures
    )


def test_contextual_chunk_dispatch_is_captured():
    phase_a = _runner_module()

    class ContextualProvider:
        def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
            yield from ()

    outbound = []
    capture = phase_a._Capture(ContextualProvider(), outbound)
    assert list(capture.embed_chunk_group_batches([[(0, "first"), (1, "second")]])) == []
    assert outbound == ["first", "second"]


@pytest.mark.parametrize(
    ("privacy_setting", "expected"),
    [(None, None), (False, False), (True, True)],
)
def test_engine_threads_embedding_privacy_setting(tmp_path, privacy_setting, expected):
    phase_a = _runner_module()
    captured = {}

    class Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Engine:
        def __init__(self, *, config, hermes_home):
            self.config = config
            self.hermes_home = hermes_home

    mod = {
        "config": type("ConfigModule", (), {"LCMConfig": Config}),
        "engine": type("EngineModule", (), {"LCMEngine": Engine}),
    }
    phase_a._engine(
        mod,
        tmp_path,
        "cloud",
        patterns=False,
        embedding_privacy=privacy_setting,
    )

    assert captured["sensitive_patterns_enabled"] is False
    assert captured["embedding_privacy_enabled"] is expected
    # The all-tools matrix must expose all default-dormant tools.
    assert captured["assertions_enabled"] is True
    assert captured["adaptive_retrieval_enabled"] is True
    assert captured["entities_enabled"] is True
    assert len(phase_a.SCENARIOS - {"lcm_doctor (prepared)"}) == 16
    assert "lcm_doctor (prepared)" in phase_a.SCENARIOS


def test_noop_scenario_is_a_runner_failure(tmp_path):
    sabotage = (
        "_orig = mod._scenario\n"
        "def _noop(engine, tool, context, *, patterns, posture):\n"
        "    if tool == 'lcm_status':\n"
        "        return None\n"
        "    return _orig(engine, tool, context, patterns=patterns, posture=posture)\n"
        "mod._scenario = _noop\n"
    )
    code, receipt, stderr = _run_in_subprocess(
        tmp_path,
        sabotage=sabotage,
        run_kwargs={"expect_365_fixed": False},
    )
    assert code != 0, stderr + receipt
    assert "| local | `lcm_status` | FAIL |" in receipt
    assert "scenario returned no postcondition evidence" in receipt
    assert "Exit verdict: `FAIL`" in receipt
