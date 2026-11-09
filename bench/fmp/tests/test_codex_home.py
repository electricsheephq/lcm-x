"""CLI leased-home admission and routing; synthetic calls and mocked auth only."""
import fcntl
import json
import stat
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import models, run


@pytest.fixture
def cli(tmp_path, monkeypatch):
    question = SimpleNamespace(qid="synthetic", store_id="store", qtype="fact",
                               question="Synthetic?", gold="yes", dataset="lme-s",
                               session_map={}, question_date="2025-01-01")
    store = SimpleNamespace(build=Mock(), generation="synthetic",
                            engine=lambda _: nullcontext((None, None)))
    monkeypatch.setattr(run, "commit", lambda _: "synthetic-commit")
    monkeypatch.setattr(run, "load_lcm", lambda _: None)
    monkeypatch.setattr(run.datasets, "load", lambda *a: ([question], "dataset", "ids", {}))
    monkeypatch.setattr(run, "Stores", Mock(return_value=store))
    monkeypatch.setattr(run, "versions", lambda: {})
    monkeypatch.setattr(run.delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run.subprocess, "run", Mock(side_effect=AssertionError("no real child")))
    (tmp_path / "lease").mkdir(exist_ok=True)
    clients, environments = [], []

    def runner(cmd, **kwargs):
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "yes"}},
            {"type": "turn.completed", "usage": {"output_tokens": 1}},
        ])), "")

    def backends(*args, **kwargs):
        client = models.Backends(*args, **kwargs)
        clients.append(client)
        client.codex.runner = runner
        return client

    built = Mock(side_effect=backends)
    monkeypatch.setattr(run, "Backends", built)

    def answer(q, mode, client, spec, **kwargs):
        reply = client.call(spec, [{"role": "user", "content": q.question}], kwargs["records"])
        return {"status": "ok", "hypothesis": reply["content"]}

    def judge(row, q, client, specs, agreement_model=None):
        for spec in [*specs, *([agreement_model] if agreement_model else [])]:
            client.call(spec, [{"role": "user", "content": q.question}], [])
        return {**row, "status": "ok"}

    monkeypatch.setattr(run.delivery, "answer", answer)
    monkeypatch.setattr(run.judges, "judge_row", judge)
    monkeypatch.setattr(run, "canary", lambda args, client, *a:
                        client.call(args.reader, [{"role": "user", "content": "Synthetic?"}], []))

    def args(command="answer", *extra):
        return run.parser().parse_args([
            command, "--split-label", "synthetic", "--data", str(tmp_path / "synthetic.json"),
            "--store-root", str(tmp_path / "stores"), "--output", str(tmp_path / "output"),
            "--agreement-judge", "none", "--workers", "1", *extra])

    return SimpleNamespace(args=args, built=built, clients=clients, environments=environments)


@pytest.mark.parametrize("command", ["canary", "answer", "judge"])
def test_t1_leased_home_reaches_codex_without_symlinks(tmp_path, monkeypatch, cli, command):
    symlink = Mock(side_effect=AssertionError("no symlinks"))
    monkeypatch.setattr(Path, "symlink_to", symlink)
    home = tmp_path / "lease"
    home.mkdir(exist_ok=True)
    args = cli.args(command, "--codex-home", str(home))
    if command == "judge":
        run.execute(cli.args("answer", "--codex-home", str(home)))
    run.execute(args)
    assert cli.clients and all(client.codex.home == home for client in cli.clients)
    assert cli.environments and all(env["CODEX_HOME"] == str(home) for env in cli.environments)
    symlink.assert_not_called()
    assert stat.S_IMODE((home / "holder.lock").stat().st_mode) == 0o600
    if command != "canary":
        manifest = json.loads((args.output / "run-manifest.json").read_text())
        assert manifest["harness"]["codex_home_mode"] == "leased"
        assert str(home) not in json.dumps(manifest)


@pytest.mark.parametrize("invalid", ["relative", "missing", "file", "workers"])
def test_t2_unsafe_options_rejected_before_backends(tmp_path, cli, invalid):
    home = tmp_path
    extra = []
    error = "--codex-home must name an existing absolute directory"
    if invalid == "relative":
        home = Path("relative")
    elif invalid == "missing":
        home = tmp_path / "missing"
    elif invalid == "file":
        home = tmp_path / "file"
        home.touch()
    else:
        extra = ["--workers", "2"]
        error = "--codex-home requires --workers 1"
    with pytest.raises(ValueError, match=f"^{error}$"):
        run.execute(cli.args("answer", "--codex-home", str(home), *extra))
    cli.built.assert_not_called()
    assert not cli.environments


@pytest.mark.parametrize("command", ["canary", "answer", "judge"])
def test_t3_busy_home_rejected_before_backends(tmp_path, cli, command):
    with (tmp_path / "lease" / "holder.lock").open("w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="^codex home busy$"):
            run.execute(cli.args(command, "--codex-home", str(tmp_path / "lease")))
    cli.built.assert_not_called()
    assert not cli.environments


def test_t4_unset_preserves_symlink_home_and_records_mode(monkeypatch, cli):
    symlink = Mock()
    monkeypatch.setattr(Path, "symlink_to", symlink)
    args = cli.args()
    run.execute(args)
    backend = cli.clients[0].codex
    assert backend.home == backend.scratch / "codex-home"
    symlink.assert_called_once_with(Path.home() / ".codex" / "auth.json")
    assert cli.environments[0]["CODEX_HOME"] == str(backend.home)
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert manifest["harness"]["codex_home_mode"] == "symlink"


def test_lock_held_through_construction_calls_and_close(tmp_path, monkeypatch, cli):
    lock = tmp_path / "lease" / "holder.lock"

    def check_lock():
        with lock.open("r+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    build, call, close = cli.built.side_effect, models.Codex.call, models.Codex.close
    observed = []

    def checked_build(*args, **kwargs):
        check_lock()
        observed.append("build")
        return build(*args, **kwargs)

    def checked_call(*args, **kwargs):
        check_lock()
        observed.append("call")
        return call(*args, **kwargs)

    def checked_close(*args, **kwargs):
        check_lock()
        observed.append("close")
        return close(*args, **kwargs)

    cli.built.side_effect = checked_build
    monkeypatch.setattr(models.Codex, "call", checked_call)
    monkeypatch.setattr(models.Codex, "close", checked_close)
    run.execute(cli.args("canary", "--codex-home", str(tmp_path / "lease")))
    assert observed == ["build", "call", "close"]
    with lock.open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("failure", ["construction", "command"])
def test_lock_released_on_failure(tmp_path, monkeypatch, cli, failure):
    error = Mock(side_effect=RuntimeError("synthetic failure"))
    if failure == "construction":
        # Fail before allocating a scratch backend.
        monkeypatch.setattr(run, "Codex", error)
    else:
        monkeypatch.setattr(run, "canary", error)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run.execute(cli.args("canary", "--codex-home", str(tmp_path / "lease")))
    with (tmp_path / "lease" / "holder.lock").open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(not client.codex.scratch.exists() for client in cli.clients)


def test_judge_updates_home_mode_and_routes_all_codex_specs(tmp_path, monkeypatch, cli):
    monkeypatch.setattr(Path, "symlink_to", Mock())
    run.execute(cli.args())
    args = cli.args("judge", "--codex-home", str(tmp_path / "lease"),
                    "--judge", models.CODEX_DEFAULT_JUDGE, models.CODEX_DEFAULT_READER,
                    "--agreement-judge", models.CODEX_DEFAULT_JUDGE, "--agreement-fraction", "1")
    run.execute(args)
    assert len(cli.environments) == 4
    assert all(env["CODEX_HOME"] == str(tmp_path / "lease") for env in cli.environments[1:])
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert manifest["harness"]["codex_home_mode"] == "symlink"  # the answer run's mode is kept
    assert manifest["harness"]["judge_codex_home_mode"] == "leased"


def test_output_or_store_root_inside_home_rejected(tmp_path, cli):
    home = tmp_path / "lease"
    home.mkdir(exist_ok=True)
    for flag in ("--output", "--store-root"):
        args = cli.args("answer", "--codex-home", str(home))
        setattr(args, flag[2:].replace("-", "_"), home / "inside")
        with pytest.raises(ValueError, match="^--output and --store-root must be outside --codex-home$"):
            run.execute(args)
    cli.built.assert_not_called()
    assert not (home / "holder.lock").exists()


def test_sigterm_unwinds_inside_lease_and_handler_restored(tmp_path, monkeypatch, cli):
    import os
    import signal
    before = signal.getsignal(signal.SIGTERM)

    def terminate(*a, **k):
        assert signal.getsignal(signal.SIGTERM) is run._terminate
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not unwind")

    monkeypatch.setattr(run, "canary", terminate)
    with pytest.raises(SystemExit) as exc:
        run.execute(cli.args("canary", "--codex-home", str(tmp_path / "lease")))
    assert exc.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == before
    with (tmp_path / "lease" / "holder.lock").open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_score_and_build_stores_ignore_codex_home_options(tmp_path, monkeypatch, cli):
    for command in ("score", "build-stores"):
        args = cli.args(command, "--codex-home", "relative", "--workers", "4")
        calls = []
        monkeypatch.setattr(run, "_execute", lambda a: calls.append(a.command))
        run.execute(args)
        assert calls == [command]
    assert not (tmp_path / "holder.lock").exists()


@pytest.mark.parametrize("entry", ["AGENTS.md", "AGENTS.override.md", "skills/custom"])
def test_home_with_instructions_or_skills_refused(tmp_path, cli, entry):
    home = tmp_path / "lease"
    (home / "skills" / ".system").mkdir(parents=True)
    target = home / entry
    if entry.startswith("skills/"):
        target.mkdir()
    else:
        target.write_text("synthetic")
    with pytest.raises(ValueError, match="^--codex-home holds instructions or skills a reader would load: "):
        run.execute(cli.args("answer", "--codex-home", str(home)))
    cli.built.assert_not_called()


def test_home_with_only_system_skills_allowed(tmp_path, cli):
    home = tmp_path / "lease"
    (home / "skills" / ".system").mkdir(parents=True)
    run.execute(cli.args("canary", "--codex-home", str(home)))
    assert cli.built.called


def test_rejected_output_inside_home_gets_no_failure_record(tmp_path, monkeypatch, cli):
    import sys
    home = tmp_path / "lease"
    monkeypatch.setattr(sys, "argv", ["run.py", "answer", "--split-label", "synthetic", "--data", str(tmp_path / "synthetic.json"),
                                      "--store-root", str(tmp_path / "stores"), "--output", str(home / "inside"),
                                      "--agreement-judge", "none", "--workers", "1", "--codex-home", str(home)])
    assert run.main() == 1
    assert sorted(p.name for p in home.iterdir()) == []


def test_sigterm_drain_starts_no_further_model_call(tmp_path, monkeypatch, cli):
    import os
    import signal
    import threading
    calls, started = [], []
    monkeypatch.setattr(models, "STOPPING", threading.Event(), raising=False)  # never leak a set event
    questions = [SimpleNamespace(qid=f"q{i}", store_id="store", qtype="fact", question="Synthetic?", gold="yes",
                                 dataset="lme-s", session_map={}, question_date="2025-01-01") for i in range(3)]
    monkeypatch.setattr(run.datasets, "load", lambda *a: (questions, "dataset", "ids", {}))
    run.execute(cli.args())
    handled = threading.Event()
    terminate = run._terminate

    def observed(signum, frame):
        handled.set()
        terminate(signum, frame)

    def judge(row, q, client, specs, agreement_model=None):
        started.append(q.qid)
        for spec in specs:  # like judges.judge: a failed judge call does not stop the next judge
            try:
                client.call(spec, [{"role": "user", "content": q.question}], [])
            except RuntimeError:
                pass
        return {**row, "status": "ok"}

    def runner(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            assert handled.wait(5)  # the main thread is unwinding before this call returns
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "yes"}},
            {"type": "turn.completed", "usage": {"output_tokens": 1}},
        ])), "")

    original = models.Backends

    def backends(*a, **kw):
        client = original(*a, **kw)
        client.codex.runner = runner
        return client

    monkeypatch.setattr(run, "_terminate", observed)
    monkeypatch.setattr(run.judges, "judge_row", judge)
    monkeypatch.setattr(run, "Backends", backends)
    args = cli.args("judge", "--codex-home", str(tmp_path / "lease"),
                    "--judge", models.CODEX_DEFAULT_JUDGE, models.CODEX_DEFAULT_READER)
    with pytest.raises(SystemExit):
        run.execute(args)
    assert started == ["q0"]  # queued questions never start
    assert len(calls) == 1  # the second judge of the in-flight question never starts
    with (tmp_path / "lease" / "holder.lock").open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
