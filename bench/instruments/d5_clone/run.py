"""D5 clone-run CLI (bench/specs/RELEASE-READINESS-V1.md, "D5 clone run"): orchestration only.

Order: free space, P1 pins, P2 snapshot, the plan (on a clone), P3-P7 (any failure stops the run before either arm),
the rc arm then the GA arm on APFS clones of the verified snapshot, T0/H8 captures around each rc-arm cold bind with
that bind's replay right after its H8 (then its captures are removed), the bars, then the receipt. Every external
effect is a LocalHost method, so a test can run the whole flow with a fake host and a fake edition. The runner never
opens the pinned snapshot itself, only its clones; P2 hashes its files and checks SQLite on scratch clones.

The rules module (``rules_module`` in the run config, a pinned file in the instrument directory, imported only after
P1) provides ``identity(row)``, ``credits`` (the two host-rewrite credits, each ``(row, host_row) -> bool``) and
``covered(session_id)`` for planning. The edition steps are argv templates in the config: ``constants_argv`` writes
``{out}/constants.json`` (with ``range_new_rows``) and ``replay_argv`` writes ``{out}/receipt.json``.
"""

import argparse
import hashlib
import importlib.util
import json
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from . import bars, capture, gateway, pins, plan as planner, provider as prov, sandbox, snapshot

SCHEMA = "d5-clone-receipt/1"
NEED_FREE = 40 * 10**9
PREFLIGHTS = ("P3", "P4", "P5", "P6", "P7")


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LocalHost:
    """The real effects. ``config`` is the run declaration; ``run_dir`` holds every clone, capture and log."""

    def __init__(self, config, run_dir):
        self.c, self.run_dir, self._rules = config, Path(run_dir), None

    def _dir(self, *parts):
        path = self.run_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def free_bytes(self):
        return shutil.disk_usage(self._dir()).free

    def pins(self):
        args, kwargs = (self.c["manifest"], self.c["instrument_dir"]), {"interpreter": self.c.get("interpreter")}
        try:
            pins.require(*args, **kwargs)  # SystemExit(2) on any mismatch, before anything is imported.
        except SystemExit:
            return pins.verify(*args, **kwargs)
        manifest = json.loads(Path(self.c["manifest"]).read_text(encoding="utf-8"))
        return {**pins.verify(*args, **kwargs), "manifest_sha256": capture._sha256(self.c["manifest"]),
                "functions_sha256": _sha_json(manifest["functions"])}

    def snapshot(self):
        return snapshot.verify(self.c["snapshot_pin"], self.c["snapshot_dir"], self._dir("scratch"))

    def clone(self, name):
        dest = self._dir(name) / "home"
        capture.clone_tree(self.c["snapshot_dir"], dest)
        return dest

    def rules(self):
        if self._rules is None:
            sys.dont_write_bytecode = True  # Bytecode beside a pinned file would fail the pins.
            spec = importlib.util.spec_from_file_location(
                "d5_clone_rules", Path(self.c["instrument_dir"]) / self.c["rules_module"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._rules = {"identity": module.identity, "credits": tuple(module.credits), "covered": module.covered}
        return self._rules

    def plan(self, copy):
        chosen = planner.choose_sessions(copy / "host-0.db", copy / "store.db", self.rules()["covered"])
        return planner.build_plan(chosen, nonce=self.c["nonce"])

    def rows(self, path):
        out = {}
        for kind, names, order in (("store", ("store.db", "lcm.db"), "store_id"), ("host", ("host-0.db", "state.db"), "id")):
            db = next(Path(path) / n for n in names if (Path(path) / n).is_file())
            with closing(capture._readonly(db)) as con:
                con.row_factory = sqlite3.Row
                out[kind] = [dict(r, _dir=str(path)) for r in con.execute(f"SELECT * FROM messages ORDER BY {order}")]
        return out

    def preflight(self, name, plan):
        if name == "P5":  # Egress: run live under the sandbox profile.
            server = prov.Provider(plan["provider"]).start()
            try:
                return sandbox.positive_control(server.url + "/models")
            finally:
                server.stop()
        path = self.c.get("preflights", {}).get(name)  # The others: a recorded preflight receipt, by sha256.
        if not path:
            return {"verdict": "MISSING"}
        return {"verdict": json.loads(Path(path).read_text(encoding="utf-8")).get("verdict"),
                "sha256": capture._sha256(path)}

    def arm(self, name, home, plan):
        tree = self.c[name]
        for old, new in (("store.db", "lcm.db"), ("host-0.db", "state.db")):
            for suffix in ("", "-wal", "-shm"):
                if (home / (old + suffix)).exists():
                    (home / (old + suffix)).rename(home / (new + suffix))
        server = prov.Provider(plan["provider"]).start()
        routing = gateway.edit_copy_routing(home / "state.db", plan["conversations"], sessions_dir=home / "sessions",
                                            run_dir=self.run_dir, provider_url=server.url)
        gateway.write_config(home, gateway.build_config(
            self.c["settings"], provider_url=server.url, home=home, aux=gateway.aux_tasks(tree["host_src"]),
            context_length=self.c["context_length"], stored_route_providers=routing["stored_route_providers"]))
        env, _ = gateway.build_env(self.c["lcm_env"], home=home, provider_url=server.url, host_bin=tree["host_bin"],
                                   port_file=home.parent / "d5-local.port", lcm_tree=tree["lcm_tree"])
        port = server.server.server_address[1]
        return {"name": name, "home": home, "provider": server, "port": port, "tree": tree,
                "gateway": gateway.Gateway(tree["host_bin"], home, env, home.parent, provider_port=port),
                "payloads": sum(p.is_file() for p in (home / "lcm-large-outputs").rglob("*")),
                "chats": {c["name"]: c["chat_id"] for c in plan["conversations"]}}

    def start(self, ctx):
        ctx["gateway"].start(self.c.get("start_timeout", 300))

    def stop(self, ctx):
        ctx["gateway"].stop()

    def turn(self, ctx, step):
        with ctx["provider"].lock:
            ctx["provider"].step = step["id"]
        try:
            return ctx["gateway"].turn(ctx["chats"][step["conversation"]], step["id"], step["user_text"]).get(
                "completed") is True
        except OSError:
            return False
        finally:
            with ctx["provider"].lock:
                ctx["provider"].step = None

    def capture(self, ctx, kind, step, extras=None):
        out = self._dir(ctx["name"], "captures") / f"{step['id']}-{kind}"
        home = ctx["home"]
        return out, capture.capture(kind, out, store_db=home / "lcm.db", host_db=home / "state.db",
                                    payload_dir=home / "lcm-large-outputs", extras=extras)

    def replay(self, ctx, step, t0, h8):
        """The edition on both trees, under the sandbox: (receipt, error, range_new_rows)."""
        out = self._dir(ctx["name"], "replays", step["id"])
        values = {"python": self.c["interpreter"], "instrument_dir": self.c["instrument_dir"], "t0": t0, "h8": h8,
                  "out": out, "rc_tree": self.c["rc"]["lcm_tree"], "ga_tree": self.c["ga"]["lcm_tree"]}
        for key in ("constants_argv", "replay_argv"):
            argv = sandbox.wrap([a.format(**values) for a in self.c[key]], ports=[ctx["port"]], socket_roots=[out])
            env = {"PATH": "/usr/bin:/bin", "HOME": str(out), "PYTHONDONTWRITEBYTECODE": "1"}
            if subprocess.run(["/usr/bin/nice", "-n", "19", *argv], cwd=out, env=env, capture_output=True).returncode:
                return None, True, None
        try:
            constants = json.loads((out / "constants.json").read_text(encoding="utf-8"))
            return json.loads((out / "receipt.json").read_text(encoding="utf-8")), False, constants.get("range_new_rows")
        except (OSError, ValueError):
            return None, True, None

    def remove(self, path):
        capture.remove_capture(path, receipt_written=True)

    def egress(self, ctx):
        return gateway.nonlocal_connections(ctx["gateway"].pids())

    def wait(self, seconds):
        time.sleep(max(0, seconds))

    def logs(self, ctx):
        files = sorted(ctx["home"].parent.glob("gateway-*.out")) + sorted((ctx["home"] / "logs").glob("*.log"))
        return [line for f in files for line in f.read_text(encoding="utf-8", errors="replace").splitlines()]

    def roots(self, ctx):
        return [str(ctx["home"] / "plugins" / "hermes-lcm-x"), str(Path(ctx["tree"]["lcm_tree"]).resolve())]

    def checks(self, ctx):
        """quick_check and integrity on both stores. Doctor and schema checks are not collected here yet, so bar 5
        reads them as not shown (INCONCLUSIVE) until they are."""
        out = []
        for name in ("lcm.db", "state.db"):
            with closing(capture._readonly(ctx["home"] / name)) as db:
                for kind, pragma in (("quick_check", "quick_check"), ("integrity", "integrity_check")):
                    ok = db.execute(f"PRAGMA {pragma}").fetchall() == [("ok",)]
                    out.append({"kind": kind, "check": name, "status": "ok" if ok else "fail"})
        return out

    def facts(self, ctx):
        """Workload observations this host can count. Cron firing, published compactions and the two host rewrites
        are not observed here yet, so bar 6 reads them as missing (INCONCLUSIVE) until they are."""
        now = sum(p.is_file() for p in (ctx["home"] / "lcm-large-outputs").rglob("*"))
        return {"payload_files_created": now - ctx["payloads"]}

    def stats(self, ctx):
        with ctx["provider"].lock:
            stats = ctx["provider"].stats
            return {**{k: dict(v) for k, v in stats.items() if isinstance(v, dict)},
                    "agent_calls_outside_step": stats["agent_calls_outside_step"], "requests": len(stats["request_log"])}

    def close(self, ctx):
        ctx["provider"].stop()


def run_arm(host, name, home, plan, rules, extras):
    ctx = host.arm(name, home, plan)
    sid = {c["name"]: c["session_id"] for c in plan["conversations"]}
    binds, restarts, egress, tool_turns, started = [], [], [], 0, time.monotonic()
    try:
        for epoch in plan["epochs"]:
            if epoch["restart_before"]:
                host.stop(ctx)
            host.start(ctx)
            begun, cold = time.monotonic(), []
            for step in epoch["steps"]:
                bind = name == "rc" and step["kind"] == "cold"
                if bind:
                    t0_dir, t0 = host.capture(ctx, "T0", step, extras)
                    session = sid[step["conversation"]]
                    t0_keys = bars.keys([r for r in host.rows(t0_dir)["store"] if r["session_id"] == session],
                                        rules["identity"])
                ok = host.turn(ctx, step)
                cold += [step["conversation"]] if ok and step["kind"] == "cold" else []
                tool_turns += bool(ok and step["tools"])
                if bind:
                    h8_dir, h8 = host.capture(ctx, "H8", step)
                    receipt, error, range_new = host.replay(ctx, step, t0_dir, h8_dir)
                    binds.append({"bind": step["id"], "session_id": session, "t0_max": t0["store_max_store_id"],
                                  "h8_max": h8["store_max_store_id"], "t0_keys": t0_keys,
                                  "t0_capture_sha256": t0["capture_hashes_sha256"],
                                  "h8_capture_sha256": h8["capture_hashes_sha256"],
                                  "bar2": bars.bar2_bind(receipt, range_new_rows=range_new, error=error)})
                    if receipt is not None:  # Kept for diagnosis when the replay left no receipt (r2.6).
                        host.remove(t0_dir)
                        host.remove(h8_dir)
                egress.append(host.egress(ctx))
            host.wait(epoch["min_seconds"] - (time.monotonic() - begun))
            if epoch["restart_before"]:
                restarts.append({"epoch": epoch["epoch"], "cold_binds": cold})
    finally:
        host.stop(ctx)
    stats = host.stats(ctx)
    host.close(ctx)
    workload = {"conversation_names": list(sid), "restarts": restarts, "tool_heavy_turns": tool_turns,
                "large_conversation_rows": max(c["lcm_rows"] for c in plan["conversations"]),
                "replay_misses": sum(stats.get("replay_misses", {}).values()), **host.facts(ctx)}
    return {"binds": binds, "workload": workload, "egress": {"samples": len(egress), "non_local": sum(egress)},
            "provider": stats, "logs": host.logs(ctx), "roots": host.roots(ctx), "checks": host.checks(ctx),
            "rows": host.rows(home), "wall_s": round(time.monotonic() - started, 3)}


def run(host, *, rc_id, prior=(), instrument_fix=None):
    """The whole run; returns the receipt (counts, ids, sha256s and verdicts only)."""
    started = time.monotonic()
    prior = [p for p in prior if p.get("rc") == rc_id and "verdict" in p]  # A stopped run is not an attempt.
    prior_verdicts = [p["verdict"]["raw"] for p in prior]
    receipt = {"schema": SCHEMA, "rc": rc_id, "attempt": len(prior_verdicts) + 1, "instrument_fix": instrument_fix}
    refusal = bars.may_run(prior_verdicts, instrument_fix)
    receipt["preflights"] = steps = {"free_space": {"verdict": "PASS" if host.free_bytes() >= NEED_FREE else "FAIL"}}
    if refusal is None and steps["free_space"]["verdict"] == "PASS":
        steps["P1"] = receipt["pins"] = host.pins()
        if instrument_fix and any(p.get("pins", {}).get("functions_sha256") != steps["P1"]["functions_sha256"]
                                  for p in prior):  # Spec: a fix never changes the rule code on this rc.
            refusal = "the instrument fix changed rule-function code; that applies from the next rc"
    for name in ("P2", "plan", *PREFLIGHTS):
        if refusal or any(s.get("verdict") != "PASS" for s in steps.values()):
            break
        if name == "P2":
            steps["P2"] = receipt["snapshot"] = host.snapshot()
        elif name == "plan":
            plan = host.plan(host.clone("plan"))
            receipt["plan"] = {"sha256": planner.plan_sha256(plan), "declaration": planner.declaration(plan)}
            steps["plan"] = {"verdict": "PASS"}
        else:
            steps[name] = host.preflight(name, plan)
    stopped = refusal or next((k for k, s in steps.items() if s.get("verdict") != "PASS"), None)
    if stopped or len(steps) != 4 + len(PREFLIGHTS):
        return {**receipt, "stopped_at": stopped, "wall_s": {"total": round(time.monotonic() - started, 3)}}
    rules, base = host.rules(), host.run_dir / "plan" / "home"
    arms = {name: run_arm(host, name, host.clone(name), plan, rules, {"RC3/store.db": base / "store.db"})
            for name in ("rc", "ga")}
    ga_rerun = None  # Bar 5: a failed check in the GA arm re-runs that arm once (release manager, 2026-11-07).
    if bars.failed_checks(arms["ga"]["checks"]):
        ga_rerun = run_arm(host, "ga", host.clone("ga-rerun"), plan, rules, {"RC3/store.db": base / "store.db"})
    snap = host.rows(base)
    tops = {k: max((r[c] for r in snap[k]), default=0) for k, c in (("store", "store_id"), ("host", "id"))}
    new = {a: [r for r in arms[a]["rows"]["store"] if r["store_id"] > tops["store"]] for a in arms}
    binds = arms["rc"]["binds"]
    results = {
        "bar1": bars.bar1(new["rc"], [r for r in arms["rc"]["rows"]["host"] if r["id"] > tops["host"]],
                          snap["store"] + snap["host"], new["ga"], binds,
                          {b["bind"]: b["bar2"]["verdict"] for b in binds}, rules["identity"], rules["credits"]),
        "bar2": bars.bar2([b["bar2"] for b in binds]),
        "bar3": bars.bar3(snap["store"], arms["rc"]["rows"]["store"], rules["identity"]),
        "bar4": bars.bar4(arms["rc"]["logs"], arms["ga"]["logs"], arms["rc"]["roots"], arms["ga"]["roots"]),
        "bar5": bars.bar5(arms["rc"]["checks"], arms["ga"]["checks"], ga_rerun and ga_rerun["checks"]),
        "bar6": bars.bar6({a: arms[a]["workload"] for a in arms})}
    receipt.update(
        arms={a: {k: arms[a][k] for k in ("workload", "egress", "provider")} for a in arms},
        ga_rerun=ga_rerun and {k: ga_rerun[k] for k in ("workload", "egress", "provider", "wall_s")},
        binds=[{k: v for k, v in b.items() if k != "t0_keys"} for b in binds], bars=results,
        verdict=bars.verdict(results, prior_verdicts),
        wall_s={**{a: arms[a]["wall_s"] for a in arms}, "total": round(time.monotonic() - started, 3)})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="run declaration (JSON)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--instrument-dir", required=True)
    parser.add_argument("--rc-id", required=True, help="the rc under test (tag or commit)")
    parser.add_argument("--prior-receipt", action="append", default=[], help="an earlier receipt for this rc")
    parser.add_argument("--instrument-fix", help="the reviewed instrument fix, after an instrument defect")
    args = parser.parse_args(argv)
    config = {**json.loads(Path(args.config).read_text(encoding="utf-8")), "instrument_dir": args.instrument_dir}
    prior = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.prior_receipt]
    receipt = run(LocalHost(config, args.run_dir), rc_id=args.rc_id, prior=prior, instrument_fix=args.instrument_fix)
    with (Path(args.run_dir) / "d5-clone-receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, sort_keys=True, indent=2)
    print(json.dumps({"verdict": receipt.get("verdict"), "stopped_at": receipt.get("stopped_at")}))
    return 0 if "verdict" in receipt else 2


if __name__ == "__main__":
    sys.exit(main())
