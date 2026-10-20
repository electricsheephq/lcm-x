"""Cheap gates: 96-row drain/restart, missing-model offline bind, 16-process screen."""
import json
import threading
import time

from .catchup import arguments, run
from .common import config, dump, guards, isolated, product
from .corpus import build


def cold(a):
    a.out.mkdir(parents=True, exist_ok=False)
    absent = a.home.parent/"absent-model-cache"
    absent.mkdir(parents=True)
    isolated(a.home, absent)
    mod = product(a.lcmx_repo, a.hermes_source)
    guards(mod, a.out/"events.jsonl")
    bind, recalls = {}, {}
    manifest = json.loads(a.manifest.read_text())
    for enabled in (False, True):
        engine = mod("engine").LCMEngine(config=config(mod, a.db, enabled), hermes_home=str(a.home))
        start = time.monotonic()
        engine.on_session_start(manifest["sessions"][0], platform="cli", context_length=200000)
        bind[str(enabled)] = (time.monotonic()-start)*1000
        start = time.monotonic()
        recalls[str(enabled)] = {"payload": json.loads(mod("tools").lcm_grep(
            {"query": manifest["canaries"][0]["id"], "mode": "hybrid"}, engine=engine)),
            "elapsed_ms": (time.monotonic()-start)*1000}
        engine.shutdown()
    events = [json.loads(s) for s in (a.out/"events.jsonl").read_text().splitlines()]
    payload = recalls["True"]["payload"]
    passed = (bind["True"]-bind["False"] <= 500 and payload.get("degraded_to_fts") is True
        and payload.get("total_results", 0) > 0 and not any(e.get("foreground") or e.get("download") or e.get("violation") for e in events))
    dump(a.out/"summary.json", {"state": "PASS" if passed else "FAIL", "bind_ms": bind,
        "first_bind_added_ms": bind["True"]-bind["False"], "recalls": recalls, "events": events,
        "foreground_threads": [t.name for t in threading.enumerate()]})


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
        if mode == "drain96":
            if a.db.exists():
                raise ValueError("drain96 needs a new DB")
            a.db.parent.mkdir(parents=True, exist_ok=True)
            dump(a.manifest, build(a.lcmx_repo, a.hermes_source, a.db, 96, "mixed", "upgrade"))
            print(run(a)["state"])
        elif mode == "cold":
            cold(a)
        else:
            raise ValueError("choose drain96, cold or screen16")
