"""Cheap gates: 96-row drain/restart, missing-model offline bind, 16-process screen."""
import json
import multiprocessing
import os
import sqlite3
import threading
import time

from .catchup import arguments, run
from .common import config, dump, foreground_request, guards, isolated, product, safe_paths
from .corpus import build


def cold_side(a, enabled):
    home, db = a.home/str(enabled), a.home/f"{enabled}.db"
    isolated(home, a.home/"absent-model-cache")
    os.environ.update(HOME=str(home), CFFIXED_USER_HOME=str(home))
    with sqlite3.connect(a.db) as source, sqlite3.connect(db) as dest:
        source.backup(dest)
    mod = product(a.lcmx_repo, a.hermes_source)
    events = a.out/f"events-{enabled}.jsonl"
    events.touch()
    guards(mod, events)
    manifest = json.loads(a.manifest.read_text())
    engine = mod("engine").LCMEngine(config=config(mod, db, enabled), hermes_home=str(home))
    try:
        start = time.monotonic()
        engine.on_session_start(manifest["sessions"][0], platform="cli", context_length=200000)
        bind = (time.monotonic()-start)*1000
        start = time.monotonic()
        with foreground_request():
            recall = {"payload": json.loads(mod("tools").lcm_grep(
                {"query": manifest["canaries"][0]["id"], "mode": "hybrid"}, engine=engine)),
                "elapsed_ms": (time.monotonic()-start)*1000}
    finally:
        engine.shutdown()
    dump(a.out/f"side-{enabled}.json", dict(bind_ms=bind, recall=recall,
        events=[json.loads(s) for s in events.read_text().splitlines()],
        foreground_threads=[t.name for t in threading.enumerate()]))


def cold(a):
    safe_paths(a.home, a.db)
    a.out.mkdir(parents=True, exist_ok=False)
    a.home.mkdir(parents=True, exist_ok=False)
    bind, recalls, events, threads = {}, {}, [], {}
    for enabled in (False, True):
        process = multiprocessing.get_context("spawn").Process(target=cold_side, args=(a, enabled))
        process.start()
        process.join()
        if process.exitcode:
            raise RuntimeError(f"cold side {enabled} exited: {process.exitcode}")
        side = json.loads((a.out/f"side-{enabled}.json").read_text())
        bind[str(enabled)], recalls[str(enabled)] = side["bind_ms"], side["recall"]
        events.extend(side["events"])
        threads[str(enabled)] = side["foreground_threads"]
    payload = recalls["True"]["payload"]
    passed = (bind["True"]-bind["False"] <= 500 and payload.get("degraded_to_fts") is True
        and payload.get("total_results", 0) > 0 and not any(e.get("foreground") or e.get("violation") for e in events))
    dump(a.out/"summary.json", {"state": "PASS" if passed else "FAIL", "bind_ms": bind,
        "first_bind_added_ms": bind["True"]-bind["False"], "recalls": recalls, "events": events,
        "foreground_threads": threads})


if __name__ == "__main__":
    # The same explicit arguments keep probe and production-cell isolation identical.
    import sys
    mode = sys.argv.pop(1)
    if mode == "screen16":
        from .rig import main
        sys.argv.append("--screen16")
        main()
    else:
        a = arguments()
        safe_paths(a.home, a.db)
        if mode == "drain96":
            a.drain96 = True
            if a.db.exists():
                raise ValueError("drain96 needs a new DB")
            a.db.parent.mkdir(parents=True, exist_ok=True)
            dump(a.manifest, build(a.lcmx_repo, a.hermes_source, a.db, 96, "mixed", "upgrade"))
            print(run(a)["state"])
        elif mode == "cold":
            cold(a)
        else:
            raise ValueError("choose drain96, cold or screen16")
