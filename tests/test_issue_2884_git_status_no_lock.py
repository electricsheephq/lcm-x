"""#2884: the plugin's git identity probe must not take the index lock.

`_git_runtime_identity` runs `git status --porcelain` in the plugin's own checkout with a 1 s timeout. Plain
`git status` refreshes the index and takes `index.lock`; a probe killed at the timeout left an empty lock behind and
the next git write in that checkout failed. The probe now runs git with `GIT_OPTIONAL_LOCKS=0` in its environment
(the `--no-optional-locks` flag is unknown to git older than 2.15, which ignores the variable instead).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_lcm import runtime_identity


def _fake_run_recorder(calls):
    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("env")))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return fake_run


def _no_optional_locks(argv, env):
    return "--no-optional-locks" not in argv and bool(env) and env.get("GIT_OPTIONAL_LOCKS") == "0"


def test_status_probe_runs_without_optional_locks(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls = []
    monkeypatch.setattr(runtime_identity.subprocess, "run", _fake_run_recorder(calls))

    runtime_identity._git_runtime_identity(tmp_path)

    status_calls = [(argv, env) for argv, env in calls if "status" in argv]
    assert status_calls, calls
    for argv, env in status_calls:
        assert _no_optional_locks(argv, env), argv
    for argv, env in calls:
        assert _no_optional_locks(argv, env), argv
        assert env.get("PATH") == os.environ.get("PATH")
        assert env.get("HOME") == os.environ.get("HOME")


_ISOLATED_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True, env=_ISOLATED_ENV)


@pytest.fixture
def dirty_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init", "-q", "-b", "main")
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"installed git does not support `git init -b`: {exc.stderr.strip()}")
    (repo / "tracked.txt").write_text("one\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    _git(repo, "config", "remote.origin.url", "https://example.invalid/fixture.git")
    (repo / "tracked.txt").write_text("two\n")
    (repo / "untracked.txt").write_text("new\n")
    return repo


def test_probe_classification_unchanged_in_dirty_repo(dirty_repo):
    head = _git(dirty_repo, "rev-parse", "HEAD").stdout.strip()

    identity = runtime_identity._git_runtime_identity(dirty_repo)

    assert identity == {
        "plugin_git_commit": head,
        "plugin_git_branch": "main",
        "plugin_git_dirty": True,
        "plugin_git_remote": "https://example.invalid/fixture.git",
    }


def test_probe_classification_clean_repo(dirty_repo):
    _git(dirty_repo, "checkout", "-q", "--", "tracked.txt")
    (dirty_repo / "untracked.txt").unlink()

    assert runtime_identity._git_runtime_identity(dirty_repo)["plugin_git_dirty"] is False


def test_probe_leaves_no_index_lock(dirty_repo):
    runtime_identity._git_runtime_identity(dirty_repo)

    assert not (Path(dirty_repo) / ".git" / "index.lock").exists()
