from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

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


def test_dev_posture_covers_every_tool_despite_disabled_tools_env(tmp_path):
    worktree = Path(__file__).resolve().parents[1]
    runner = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    env = os.environ.copy()
    env["LCM_DISABLED_TOOLS"] = "lcm_recall"
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
        timeout=60,
    )
    receipt = (tmp_path / "PHASE-A-RECEIPT.md").read_text(encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr + receipt
    assert "DEV RUN — identity unbound" in receipt
    assert "engine=stubbed-context-base (NOT the hermes surface)" in receipt
    assert "LCM_DISABLED_TOOLS" in receipt
    assert "Registry coverage: `COMPLETE`" in receipt
    assert "Missing matrix rows: `none`" in receipt
    assert "Unexpected scenario rows: `none`" in receipt
    assert receipt.count("| local | `lcm_") == 15
    assert "| cloud | `lcm_recall` | SKIP | SKIP: VOYAGE_API_KEY is absent |" in receipt
    assert "Exit verdict: `PASS`" in receipt


def test_release_mode_blocks_the_same_cloud_skip(tmp_path):
    hermes_repo = _fake_hermes_repo(tmp_path)
    sabotage = (
        "mod._git_identity = lambda _p: "
        "{'head': 'abc123', 'tree': 'tree123', 'tag': 'v0.0.0-rc1', 'dirty': False}\n"
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
    assert "| cloud | `lcm_recall` | SKIP |" in receipt
    assert f"Hermes ContextEngine module __file__: `{module_path}`" in receipt
    assert (
        f"Hermes ContextEngine module sha256: "
        f"`{hashlib.sha256(module_path.read_bytes()).hexdigest()}`"
    ) in receipt
    assert "Exit verdict: `BLOCKED`" in receipt


@pytest.mark.parametrize("provider", ["fastembed", "unknown-provider"])
def test_cloud_posture_blocks_non_product_cloud_provider(
    tmp_path, monkeypatch, provider
):
    phase_a = _runner_module()
    worktree = Path(__file__).resolve().parents[1]
    mod, _surface, _identity = phase_a._load(
        worktree, hermes_repo=_fake_hermes_repo(tmp_path / "loader")
    )
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
        ({"dirty": False, "tag": "v1.2.3"}, "v1.2.3", []),
        ({"dirty": True, "tag": "v1.2.3-rc1"}, "v1.2.3-rc1", ["dirty"]),
        ({"dirty": False, "tag": "v1.2.3-rc2"}, "v1.2.3-rc1", ["exact tag"]),
        (
            {"dirty": False, "tag": "release-1.2.3"},
            "release-1.2.3",
            ["must match"],
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


def test_noop_scenario_is_a_runner_failure(tmp_path):
    sabotage = (
        "_orig = mod._scenario\n"
        "def _noop(engine, tool, context, *, patterns):\n"
        "    if tool == 'lcm_status':\n"
        "        return None\n"
        "    return _orig(engine, tool, context, patterns=patterns)\n"
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
