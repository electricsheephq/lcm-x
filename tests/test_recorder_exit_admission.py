"""Checkpoint admission closes before the recorder's exit drain marker."""

import faulthandler
import importlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.test_recorder import Lab, REPO, _recorder, needs312


def _exit_drain_scenario(work):
    def offline(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("exit regression forbids network")

    sys.addaudithook(offline)
    faulthandler.dump_traceback_later(15)
    mp = pytest.MonkeyPatch()
    lab = Lab(work, mp)
    home = lab.home()
    mp.setenv("HERMES_HOME", str(home))
    module, ctx = lab.load(home)
    engine, rec = ctx.offered, _recorder()
    assert rec is not None and rec.stopped is None
    rm = importlib.import_module(module.__name__ + ".recorder")
    ms = importlib.import_module(module.__name__ + ".maintenance_scheduler")
    engine.on_session_start("s", platform="cli", conversation_id="c")
    history = [{"role": role, "content": content, "timestamp": 1760000001.0 + n}
               for n, (role, content) in enumerate([
                   ("user", "repeat"), ("assistant", "answer"),
                   ("user", "again"), ("assistant", "done")])]
    ctx.hooks["post_llm_call"][0](session_id="s", platform="cli", conversation_history=history[:2])
    # The exit-path recipe: a busy SQLite write leaves a real tail for the later atexit root.
    with sqlite3.connect(home / "lcm.db", isolation_level=None) as locker:
        locker.execute("BEGIN IMMEDIATE")
        engine.on_session_end("s", history)
        locker.rollback()
    assert engine._pending_session_tails and engine in ms._PENDING_TAIL_ENGINES
    writer = rec.writer
    assert writer.drain(5)
    backups = writer.backups
    skipped = rec.counters["recorder.checkpoint_skipped"]
    mp.setattr(rm, "clock", lambda: rec.started + 24 * 3600 + 1)
    marker, window_done = threading.Event(), threading.Event()
    process, offer = writer._process, writer.offer
    observations, errors, sync_roots = {}, [], []
    main = threading.current_thread()

    def process_gate(item):
        if item[0] == "sync" and threading.current_thread() is writer:
            assert writer.io._is_owned()
            marker.set()
            assert window_done.wait(1.5), "window root exceeded the real exit drain budget"
        return process(item)

    def offer_observed(item, nbytes=256, force=False):
        if (writer.sync and threading.current_thread() is main and item[0] == "event"
                and item[1].get("name") == "_flush_pending_tails_at_exit"):
            assert rec.lock._is_owned()
            sync_roots.append(item[1]["name"])
        return offer(item, nbytes, force)

    def window_root():
        try:
            assert marker.wait(5), "exit drain never reached the marker"
            assert rec.inflight == 0
            engine.get_tool_schemas()
            assert rec.inflight == 0
            # Capture before releasing the marker: the writer cannot consume a queued backup or warning yet.
            observations.update(backups=writer.backups,
                                skipped=rec.counters["recorder.checkpoint_skipped"],
                                exit_warning=("LCM recorder: checkpoint +%dh skipped: %s", 24, "exit")
                                in rec.warnings)
        except BaseException as exc:
            errors.append(exc)
        finally:
            window_done.set()

    mp.setattr(writer, "_process", process_gate)
    mp.setattr(writer, "offer", offer_observed)
    daemon = threading.Thread(target=window_root, name="exit-window-root", daemon=True)
    daemon.start()
    rec._atexit()  # Real admission barrier, drain marker and synchronous transition on main.
    daemon.join(5)
    assert not daemon.is_alive() and not errors, errors
    assert writer.sync and rec.stopped is None
    assert observations["backups"] == backups, observations
    assert observations["skipped"] == skipped + 1, observations
    assert observations["exit_warning"], observations
    ms._flush_pending_tails_at_exit()  # A recorded root must emit synchronously and return on main.
    assert sync_roots == ["_flush_pending_tails_at_exit"]
    assert not engine._pending_session_tails
    with sqlite3.connect(home / "lcm.db") as conn:
        assert [row[0] for row in conn.execute("SELECT content FROM messages ORDER BY store_id")] == [
            "repeat", "answer", "again", "done"]
    assert rec.counters["recorder.failed"] == 0
    print("EXIT_ADMISSION_OK: no backup admitted; exit skip counted and warned; sync tail flush returned", flush=True)


@needs312
def test_exit_drain_skips_new_checkpoint(tmp_path):
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, PYTHONPATH=path, HERMES_HOME=str(tmp_path / "home"), TMPDIR=str(tmp_path))
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=25)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "EXIT_ADMISSION_OK" in done.stdout


if __name__ == "__main__":
    try:
        _exit_drain_scenario(Path(sys.argv[1]))
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(2)
    os._exit(0)  # This spent subprocess must not run unrelated atexit hooks or fixture teardown.
