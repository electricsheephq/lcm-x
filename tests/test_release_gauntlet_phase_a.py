from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _runner_module():
    worktree = Path(__file__).resolve().parents[1]
    path = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    spec = importlib.util.spec_from_file_location("phase_a_tool_matrix_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_hermes_repo(root: Path) -> Path:
    repo = root / "hermes-agent"
    agent = repo / "agent"
    agent.mkdir(parents=True)
    (agent / "__init__.py").write_text("", encoding="utf-8")
    (agent / "context_engine.py").write_text(
        "class ContextEngine:\n    def get_status(self):\n        return {}\n",
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


def test_release_mode_blocks_the_same_cloud_skip(tmp_path, monkeypatch):
    phase_a = _runner_module()
    worktree = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("LCM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr(
        phase_a,
        "_git_identity",
        lambda _path: {
            "head": "abc123",
            "tree": "tree123",
            "tag": "v0.0.0-rc1",
            "dirty": False,
        },
    )
    code, receipt_path = phase_a.run(
        worktree,
        tmp_path / "receipt",
        release=True,
        rc_tag="v0.0.0-rc1",
        hermes_repo=_fake_hermes_repo(tmp_path),
        expect_365_fixed=False,
    )
    receipt = receipt_path.read_text(encoding="utf-8")
    assert code != 0
    assert "| cloud | `lcm_recall` | SKIP |" in receipt
    assert "Exit verdict: `BLOCKED`" in receipt


@pytest.mark.parametrize(
    ("identity", "tag", "expected"),
    [
        ({"dirty": False, "tag": "v1.2.3-rc1"}, "v1.2.3-rc1", []),
        ({"dirty": True, "tag": "v1.2.3-rc1"}, "v1.2.3-rc1", ["dirty"]),
        ({"dirty": False, "tag": "v1.2.3-rc2"}, "v1.2.3-rc1", ["exact tag"]),
    ],
)
def test_release_identity_requires_clean_exact_rc_tag(identity, tag, expected):
    failures = _runner_module()._release_identity_failures(identity, tag)
    assert all(
        any(fragment in failure for failure in failures) for fragment in expected
    )
    assert bool(failures) is bool(expected)


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


def test_noop_scenario_is_a_runner_failure(tmp_path, monkeypatch):
    phase_a = _runner_module()
    original = phase_a._scenario

    def no_op_status(engine, tool, context, *, patterns):
        if tool == "lcm_status":
            return None
        return original(engine, tool, context, patterns=patterns)

    monkeypatch.setattr(phase_a, "_scenario", no_op_status)
    code, receipt_path = phase_a.run(
        Path(__file__).resolve().parents[1],
        tmp_path,
        expect_365_fixed=False,
    )
    receipt = receipt_path.read_text(encoding="utf-8")
    assert code != 0
    assert "| local | `lcm_status` | FAIL |" in receipt
    assert "scenario returned no postcondition evidence" in receipt
    assert "Exit verdict: `FAIL`" in receipt
