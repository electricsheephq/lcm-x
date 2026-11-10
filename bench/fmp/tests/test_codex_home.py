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

REAL_RUN = subprocess.run  # the cli fixture replaces subprocess.run; the reaping probes need a real child


@pytest.fixture(autouse=True)
def fresh_stop_flag(monkeypatch):
    # The real SIGTERM handler sets the module-global flag; never let it reach a later test.
    import threading
    monkeypatch.setattr(models, "STOPPING", threading.Event(), raising=False)


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


@pytest.mark.parametrize("sender", ["main", "other_thread"])
def test_sigterm_drain_starts_no_further_model_call(tmp_path, monkeypatch, cli, sender):
    # A leased run (one worker) runs questions on the main thread, so the handler runs before any later call can
    # start. "other_thread" sends the signal from another thread while the main thread is inside the model call.
    import os
    import signal
    import threading
    calls, started = [], []
    questions = [SimpleNamespace(qid=f"q{i}", store_id="store", qtype="fact", question="Synthetic?", gold="yes",
                                 dataset="lme-s", session_map={}, question_date="2025-01-01") for i in range(3)]
    monkeypatch.setattr(run.datasets, "load", lambda *a: (questions, "dataset", "ids", {}))
    run.execute(cli.args())

    def judge(row, q, client, specs, agreement_model=None):
        started.append(q.qid)
        for spec in specs:  # like judges.judge: a failed judge call does not stop the next judge
            try:
                client.call(spec, [{"role": "user", "content": q.question}], [])
            except RuntimeError:
                pass
        return {**row, "status": "ok"}

    def runner(cmd, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        if len(calls) == 1:
            if sender == "main":
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                sent = threading.Event()
                threading.Thread(target=lambda: (os.kill(os.getpid(), signal.SIGTERM), sent.set())).start()
                assert sent.wait(5)
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "yes"}},
            {"type": "turn.completed", "usage": {"output_tokens": 1}},
        ])), "")

    original = models.Backends

    def backends(*a, **kw):
        client = original(*a, **kw)
        client.codex.runner = runner
        return client

    monkeypatch.setattr(run.judges, "judge_row", judge)
    monkeypatch.setattr(run, "Backends", backends)
    args = cli.args("judge", "--codex-home", str(tmp_path / "lease"),
                    "--judge", models.CODEX_DEFAULT_JUDGE, models.CODEX_DEFAULT_READER)
    with pytest.raises(SystemExit):
        run.execute(args)
    assert calls == [True]  # one model call, made on the main thread; nothing after SIGTERM
    assert started == ["q0"]  # no queued question starts
    with (tmp_path / "lease" / "holder.lock").open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cli_keeps_fixed_canary_reason_only(tmp_path, monkeypatch, capsys):
    import sys
    output = tmp_path / "output"
    base = ["run.py", "canary", "--split-label", "synthetic", "--data", str(tmp_path / "synthetic.json"),
            "--store-root", str(tmp_path / "stores"), "--output", str(output), "--agreement-judge", "none"]
    for text, shown in [(run.CANARY_NO_MCP, True), (run.CANARY_FAILED, True),
                        ("canary_failed: private corpus text", False), ("private corpus text", False)]:
        monkeypatch.setattr(run, "execute", Mock(side_effect=RuntimeError(text)))
        monkeypatch.setattr(sys, "argv", base)
        assert run.main() == 1
        err = capsys.readouterr().err
        failure = json.loads((output / "failure.json").read_text())
        assert (text in err) is shown and (failure.get("reason") == text) is shown
        assert "private corpus text" not in err + json.dumps(failure)


def test_launcher_forwards_sigterm_to_python(tmp_path):
    import os
    import signal
    import time
    marker = tmp_path / "got-sigterm"
    fake = tmp_path / "python"
    fake.write_text("#!/bin/bash\n"
                    "if [[ \"$1\" == -c ]]; then exit 0; fi\n"
                    f"trap 'touch {marker}; exit 0' TERM\n"
                    f"echo $$ > {tmp_path / 'started'}\n"
                    "while true; do sleep 0.05; done\n")
    fake.chmod(0o700)
    script = Path(run.__file__).with_name("run.sh")
    proc = subprocess.Popen(["bash", str(script), "answer"], env={**os.environ, "FMP_PYTHON": str(fake)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "started").exists()
        proc.send_signal(signal.SIGTERM)  # to the launcher's own PID
        proc.wait(timeout=10)
        assert marker.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        started = tmp_path / "started"
        if not marker.exists() and started.exists() and started.read_text().strip():
            try:  # never leave the fake reader looping when the launcher dropped the signal
                os.kill(int(started.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_one_worker_main_thread_matches_pool_outputs(tmp_path, monkeypatch, cli):
    questions = [SimpleNamespace(qid=f"q{i}", store_id="store", qtype="fact", question=f"Synthetic {i}?", gold="yes",
                                 dataset="lme-s", session_map={}, question_date="2025-01-01") for i in range(4)]
    monkeypatch.setattr(run.datasets, "load", lambda *a: (questions, "dataset", "ids", {}))
    outputs = {}
    for workers in ("1", "2"):  # "2" takes the ThreadPoolExecutor path the one-worker run used before
        for command in ("answer", "judge"):
            args = cli.args(command, "--workers", workers)
            args.output = tmp_path / f"out-{workers}"
            run.execute(args)
        outputs[workers] = {name: [json.loads(line) for line in (tmp_path / f"out-{workers}" / name).read_text().splitlines()]
                            for name in ("answers.jsonl", "judged.jsonl")}
    volatile = {"latency_s", "at", "started_at", "ended_at", "elapsed_s", "call_dir", "transcripts", "wall_s"}

    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in volatile}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    assert [r["qid"] for r in outputs["1"]["answers.jsonl"]] == ["q0", "q1", "q2", "q3"]
    assert stable(outputs["1"]) == stable(outputs["2"])


def _real_child_drain(tmp_path, monkeypatch, cli, child_code, on_child):
    """Run one leased judge whose first model call is a real child process; return (child pid, alive at release)."""
    import threading
    pid_file = tmp_path / "child.pid"
    calls, alive_at_release = [], []

    sent = threading.Event()

    def send(pid_file):
        on_child(pid_file)
        sent.set()

    def runner(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            threading.Thread(target=send, args=(pid_file,), daemon=True).start()
            kwargs.pop("stdin", None)
            kwargs.pop("cwd", None)
            REAL_RUN([__import__("sys").executable, "-c", child_code.format(pid=pid_file)], **kwargs)
            assert sent.wait(10)  # a child that exits at once: the signal still lands inside this call
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "yes"}},
            {"type": "turn.completed", "usage": {"output_tokens": 1}}])), "")

    original, lock = models.Backends, run.codex_home_lock

    def backends(*a, **kw):
        client = original(*a, **kw)
        client.codex.runner = runner
        return client

    import contextlib
    import os

    @contextlib.contextmanager
    def watched(home):
        with lock(home):
            try:
                yield
            finally:  # runs before the real lock context releases the flock
                if not pid_file.exists():
                    alive_at_release.append(None)  # the child never started: report that, not a read error
                else:
                    try:
                        os.kill(int(pid_file.read_text()), 0)
                        alive_at_release.append(True)
                    except ProcessLookupError:
                        alive_at_release.append(False)

    run.execute(cli.args())  # the unleased answer run that the judge run reads, before any probe is armed
    monkeypatch.setattr(run, "Backends", backends)
    monkeypatch.setattr(run, "codex_home_lock", watched)
    args = cli.args("judge", "--codex-home", str(tmp_path / "lease"),
                    "--judge", models.CODEX_DEFAULT_JUDGE, models.CODEX_DEFAULT_READER)
    with pytest.raises(SystemExit):
        run.execute(args)
    return calls, alive_at_release


def _when_child_writes_pid(action):
    def on_child(pid_file):
        import time
        deadline = time.monotonic() + 10
        while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.01)
        if pid_file.exists() and pid_file.read_text().strip():  # never signal after a failed start: the guard is gone
            action()
    return on_child


@pytest.mark.parametrize("case", ["running_child", "child_already_exited", "second_signal_while_unwinding"])
def test_sigterm_probes_leave_no_live_child_and_start_no_call(tmp_path, monkeypatch, cli, case):
    import os
    import signal
    import time
    term = lambda: os.kill(os.getpid(), signal.SIGTERM)  # noqa: E731
    if case == "running_child":
        code = "import os,time; open(r'{pid}','w').write(str(os.getpid())); time.sleep(30)"
    elif case == "child_already_exited":
        code = "import os; open(r'{pid}','w').write(str(os.getpid()))"  # exits at once; the signal lands after the reap
    else:
        code = "import os,time; open(r'{pid}','w').write(str(os.getpid())); time.sleep(30)"
        kill = subprocess.Popen.kill

        def kill_with_second_signal(self):
            term()  # a second SIGTERM lands while subprocess.run is killing the child
            time.sleep(0.05)
            kill(self)

        monkeypatch.setattr(subprocess.Popen, "kill", kill_with_second_signal)
    started = time.monotonic()
    previous = signal.signal(signal.SIGTERM, lambda *a: None)  # a stray late signal must never end pytest
    try:
        calls, alive = _real_child_drain(tmp_path, monkeypatch, cli, code, _when_child_writes_pid(term))
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert len(calls) == 1  # no model call starts after the signal
    assert alive == [False]  # the child is gone before the lease is released
    assert time.monotonic() - started < 10  # the child was killed, not waited out


def test_sigterm_between_questions_starts_no_call(tmp_path, monkeypatch, cli):
    import os
    import signal
    calls, started = [], []
    questions = [SimpleNamespace(qid=f"q{i}", store_id="store", qtype="fact", question="Synthetic?", gold="yes",
                                 dataset="lme-s", session_map={}, question_date="2025-01-01") for i in range(3)]
    monkeypatch.setattr(run.datasets, "load", lambda *a: (questions, "dataset", "ids", {}))
    run.execute(cli.args())

    def judge(row, q, client, specs, agreement_model=None):
        started.append(q.qid)
        for spec in specs:
            client.call(spec, [{"role": "user", "content": q.question}], [])
        os.kill(os.getpid(), signal.SIGTERM)  # after this question's calls, with no child running
        return {**row, "status": "ok"}

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "yes"}},
            {"type": "turn.completed", "usage": {"output_tokens": 1}}])), "")

    original = models.Backends

    def backends(*a, **kw):
        client = original(*a, **kw)
        client.codex.runner = runner
        return client

    monkeypatch.setattr(run.judges, "judge_row", judge)
    monkeypatch.setattr(run, "Backends", backends)
    args = cli.args("judge", "--codex-home", str(tmp_path / "lease"),
                    "--judge", models.CODEX_DEFAULT_JUDGE, models.CODEX_DEFAULT_READER)
    with pytest.raises(SystemExit):
        run.execute(args)
    assert started == ["q0"] and len(calls) == 2  # q0's two judges ran; nothing after the signal
    with (tmp_path / "lease" / "holder.lock").open("r+") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
