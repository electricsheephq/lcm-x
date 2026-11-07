"""D5 clone-run preflights P3, P4, P6 and P7 (#2199): each writes the receipt the runner reads by its verdict.

Every effect goes through the runner's LocalHost seams, on a fresh clone per preflight and tree. Receipts carry
counts, ids, sha256s and verdicts only. A preflight that cannot run (a crash, a gateway that will not start, a turn
that does not complete) is INCONCLUSIVE, never PASS.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import ProxyHandler, Request, build_opener

from . import bars, capture, plan as planner, provider as prov, run

SCHEMA = "d5-clone-preflight/1"
ANCHOR = "identity-anchor recognised"
DIGEST12 = re.compile(r"_([0-9a-f]{12})_[0-9a-f]+\.json$")  # The payload filename's content digest prefix.


class Inconclusive(Exception):
    """The preflight could not run to its checks."""


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def _failed(checks):
    return [name for name, bad in checks if bad]


def _steps(plan, epoch=None):
    return [s for e in plan["epochs"] if epoch in (None, e["epoch"]) for s in e["steps"] if s["conversation"] == "L"]


def _turn(host, ctx, step):
    if host.turn(ctx, step) is not True:
        raise Inconclusive(f"turn {step['id']} not completed")


def _session(host, tree, clone, plan, steps, *, until=None, restart_before=()):
    """Arm a fresh clone, run the steps (restarting the gateway before those named), stop; returns the context."""
    ctx = host.arm(tree, host.clone(clone), plan)
    host.pause_cron(ctx)  # As run_arm: no snapshot job fires.
    try:
        host.start(ctx)
        for step in steps:
            if step["id"] in restart_before:
                host.stop(ctx)
                host.start(ctx)
            _turn(host, ctx, step)
            if until and until(ctx):
                break
    finally:
        host.stop(ctx)
        host.close(ctx)
    return ctx


def _rows(ctx, db, sql, args=()):
    return run._rows(Path(ctx["home"]) / db, sql, args)


def _tops(ctx):
    return {k: _rows(ctx, db, f"SELECT max({c}) FROM messages")[0][0] or 0
            for k, db, c in (("store", "lcm.db", "store_id"), ("host", "state.db", "id"))}


def _tool_rows(ctx, plan, step):
    """(host contents, LCM-X contents) for the step's scripted tool call, stored after the arm's marks."""
    call = plan["provider"]["steps"][step["id"]][0]["tool_calls"][0]["id"]
    return tuple([t for t, in _rows(ctx, db, f"SELECT content FROM messages WHERE role = 'tool' AND tool_call_id = ?"
                                    f" AND {col} > ?", (call, ctx["marks"][mark]))]
                 for db, col, mark in (("state.db", "id", "host_id"), ("lcm.db", "store_id", "store_id")))


def p3(host, plan):
    reasons, facts, saved = [], {}, host.c
    server = prov.Provider(plan["provider"]).start()
    try:
        for model in ("d5-summary", "d5-aux"):
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"d5 preflight {model}"}]})
            replies = [build_opener(ProxyHandler({})).open(Request(
                server.url + "/chat/completions", data=body.encode(), headers={"Content-Type": "application/json"}),
                timeout=30).read() for _ in range(2)]
            facts[f"{model}_reply_sha256"] = [_sha(r) for r in replies]
            reasons += _failed([(f"{model}: the same request gave different bytes", replies[0] != replies[1])])
    finally:
        server.stop()
    host.c = {**saved, "context_length": saved["p3_context_length"]}  # Only P3's clones cross the threshold early.
    try:
        for tree in ("rc", "ga"):
            ctx = _session(host, tree, f"pf-P3-{tree}", plan, _steps(plan)[:saved.get("p3_max_turns", 6)],
                           until=lambda c: host.facts(c)["compactions_published"] >= 1)
            stats, published = host.stats(ctx), host.facts(ctx)["compactions_published"]
            ran = [s for s in plan["provider"]["steps"] if s in stats["calls_per_step"]]
            nodes = [s for s, in _rows(ctx, "lcm.db", "SELECT summary FROM summary_nodes WHERE node_id > ?",
                                       (ctx["marks"]["node_id"],))]
            f = facts[tree] = {
                "steps_run": len(ran), "compactions_published": published, "new_nodes": len(nodes),
                "fixed_rule_nodes": sum("D5 digest v1" in s for s in nodes),
                "summary_calls": stats["calls_by_model"].get("d5-summary", 0),
                "outside_step": stats["agent_calls_outside_step"], "replay_misses": sum(stats["replay_misses"].values()),
                "unknown_models": sum(stats["unknown_models"].values()),
                "off_script": [s for s in ran if stats["calls_per_step"][s] != len(plan["provider"]["steps"][s])]}
            reasons += [f"{tree}: {k}" for k in _failed([
                ("no published compaction", published < 1), ("no d5-summary call", f["summary_calls"] < 1),
                ("a new summary node not from the fixed rule", not nodes or f["fixed_rule_nodes"] != len(nodes)),
                ("agent calls outside a step", f["outside_step"]), ("replay misses", f["replay_misses"]),
                ("unknown models", f["unknown_models"]), ("step calls != planned entries", f["off_script"])])]
    finally:
        host.c = saved
    return reasons, [], facts


def p4(host, plan):
    step, reasons, facts = next(s for s in _steps(plan, 1) if s["tools"] == "large"), [], {}
    for tree in ("rc", "ga"):
        ctx = _session(host, tree, f"pf-P4-{tree}", plan, [step])
        hosts, lcms = _tool_rows(ctx, plan, step)
        refs = [m[1] for t in hosts + lcms for m in run.EXTERNALIZED_REF.finditer(t or "")]
        path = Path(ctx["home"]) / "lcm-large-outputs" / (refs[0] if refs else "-")
        payload = json.loads(path.read_bytes()) if refs and path.is_file() else {}
        f = facts[tree] = {
            "rows": [len(hosts), len(lcms)], "refs": len(refs), "files": host.facts(ctx)["payload_files_created"],
            "stub_masked_sha256": [_sha(run.EXTERNALIZED_REF.sub(lambda m: m[0].replace(m[1], "<ref>"), t or ""))
                                   for t in hosts + lcms],
            "content_sha256": _sha(payload.get("content")), "content_chars": payload.get("content_chars"),
            "content_bytes": payload.get("content_bytes"), "digest12": [m[1] for m in map(DIGEST12.search, refs) if m]}
        reasons += _failed([(f"{tree}: not one tool row per store, one reference and one payload file",
                             (f["rows"], f["refs"], f["files"]) != ([1, 1], 1, 1))])
    return reasons + [f"{k} differs between the trees" for k in ("stub_masked_sha256", "content_sha256",
                      "content_chars", "content_bytes", "digest12") if facts["rc"][k] != facts["ga"][k]], [], facts


def _find(value, key):
    """Every value under *key* anywhere in a JSON value."""
    if isinstance(value, dict):
        return [v for k, v in value.items() if k == key] + [x for v in value.values() for x in _find(v, key)]
    return [x for v in value for x in _find(v, key)] if isinstance(value, list) else []


def p6(host, plan):
    cold, warm = _steps(plan, 0)[:2]
    ctx = host.arm("rc", host.clone("pf-P6-rc"), plan)
    host.pause_cron(ctx)
    sid, ids = ctx["sessions"]["L"], ctx["session_ids"]

    def anchors():
        return sum(ANCHOR in line and f"session={sid}" in line for line in host.logs(ctx))

    try:
        host.start(ctx)
        before = _tops(ctx)
        t0_dir, _ = host.capture(ctx, "T0", cold, {"RC3/store.db": host.run_dir / "pf-plan" / "home" / "store.db"})
        _turn(host, ctx, cold)
        h8_dir, _ = host.capture(ctx, "H8", cold)
        receipt, error, range_new = host.replay(ctx, cold, t0_dir, h8_dir)  # As run_arm: right after its H8.
        bind, cold_anchors, mid = bars.bar2_bind(receipt, range_new_rows=range_new, error=error), anchors(), _tops(ctx)
        # The edition's conversation_id is the T0 host session row's session_key (p_s6r_clone.py:1148).
        keys = {k for k, in run._rows(Path(t0_dir) / "host-0.db", "SELECT session_key FROM sessions WHERE id = ?", (sid,))}
        if receipt is not None:  # As run_arm: the captures go once the replay left its receipt.
            host.remove(t0_dir)
            host.remove(h8_dir)
        _turn(host, ctx, warm)
        warm_anchors = anchors() - cold_anchors
        ended = _rows(ctx, "state.db", "SELECT count(*) FROM sessions WHERE end_reason IS NOT NULL"
                      f" AND id IN ({','.join('?' * len(ids))})", ids)[0][0]
    finally:
        host.stop(ctx)
        host.close(ctx)

    def new(db, col, mark, session=True):
        return _rows(ctx, db, f"SELECT count(*) FROM messages WHERE {col} > ?" + " AND session_id = ?" * session,
                     (mark, sid)[:1 + session])[0][0]

    import yaml
    enabled = ((yaml.safe_load((Path(ctx["home"]) / "config.yaml").read_text()) or {}).get("plugins") or {}).get("enabled")
    live = {c for c, in _rows(ctx, "lcm.db", "SELECT DISTINCT conversation_id FROM messages WHERE session_id = ?"
                              " AND store_id > ? AND store_id <= ?", (sid, before["store"], mid["store"]))}
    replayed = {str(c) for c in _find(receipt or {}, "conversation_id")} | {str(k) for k in keys if k}
    f = {"bind": bind, "cold_anchors": cold_anchors, "warm_anchors": warm_anchors, "ended_sessions": ended,
         "warm_rows": [new("lcm.db", "store_id", mid["store"]), new("state.db", "id", mid["host"])],
         "host_rows": [new("state.db", "id", before["host"], False), new("state.db", "id", before["host"])],
         "live_conversation_sha256": sorted(map(_sha, live)), "receipt_conversation_sha256": sorted(map(_sha, replayed))}
    return _failed([
        ("bind validity is not PASS", not error and bind["validity"] != "PASS"),
        ("range_new_rows is not 0", not error and range_new != 0),
        ("no anchor line at the cold bind", cold_anchors < 1), ("an anchor line after the warm turn", warm_anchors),
        ("the warm turn stored LCM-X rows beyond the host's", len(set(f["warm_rows"])) != 1),
        ("a chosen session ended", ended), ("d5-local is not enabled", "d5-local" not in (enabled or ())),
        ("new host rows outside L's session", len(set(f["host_rows"])) != 1 or f["host_rows"][0] < 2),
        ("the live and replayed conversation_id differ", replayed and live != replayed)]), \
        _failed([("the edition replay errored (no receipt to judge)", error),
                 ("neither the T0 session row nor the edition receipt gives a conversation_id to compare", not replayed)]), f


def p7(host, plan):
    rewrite, large = _steps(plan, 0)[1], next(s for s in _steps(plan, 1) if s["tools"] == "large")
    cold = _steps(plan, 2)[0]
    ctx = _session(host, "rc", "pf-P7-rc", plan, [rewrite, large, cold], restart_before=(cold["id"],))
    users = [t for t, in _rows(ctx, "state.db", "SELECT content FROM messages WHERE role = 'user' AND session_id = ?"
                               " AND id > ? ORDER BY id", (ctx["sessions"]["L"], ctx["marks"]["host_id"]))]
    workload, (hosts, lcms) = host.facts(ctx), _tool_rows(ctx, plan, large)
    swap = len(hosts) == len(lcms) == 1 and run._swap_payload(ctx["home"], hosts[0]) == lcms[0]
    f = {**{k: workload[k] for k in ("whitespace_rewrites", "payload_swaps", "payload_files_created")},
         "stored_user_sha256": _sha(users[0]) if users else None, "stripped_sha256": _sha(rewrite["user_text"].strip()),
         "tool_rows": [len(hosts), len(lcms)], "host_row_is_the_stub": swap}
    return _failed([
        ("the host did not store the stripped form", f["stored_user_sha256"] != f["stripped_sha256"]),
        ("no whitespace rewrite", f["whitespace_rewrites"] < 1), ("no payload swap", f["payload_swaps"] < 1),
        ("no payload file", f["payload_files_created"] < 1),
        ("the host row differs from the LCM-X row by more than the stub", not swap)]), [], f


PREFLIGHTS = {"P3": p3, "P4": p4, "P6": p6, "P7": p7}


def produce(host, names):
    """Write <run_dir>/<name>.json for each preflight; returns {name: verdict}."""
    common = {"snapshot_pin_sha256": capture._sha256(host.c["snapshot_pin"]), "plan_sha256": None,
              "trees": {t: host.c[t].get("commit") or _sha(Path(host.c[t]["lcm_tree"]).resolve()) for t in ("rc", "ga")}}
    failed, verdicts = None, {}
    try:
        if host.pins().get("verdict") != "PASS":  # The rules module is imported for planning only after P1.
            raise Inconclusive("P1 pins did not pass")
        plan = host.plan(host.clone("pf-plan"))
        common["plan_sha256"] = planner.plan_sha256(plan)
    except Exception as exc:
        failed = f"no plan: {type(exc).__name__}: {str(exc)[:160]}"
    for name in names:
        try:
            if failed:
                raise Inconclusive(failed)
            reasons, unsure, facts = PREFLIGHTS[name](host, plan)
            verdict, reasons = "FAIL" if reasons else "INCONCLUSIVE" if unsure else "PASS", reasons + unsure
        except Exception as exc:  # Never a PASS: the checks did not all run.
            verdict, reasons, facts = "INCONCLUSIVE", [f"{type(exc).__name__}: {str(exc)[:160]}"], {}
        with (Path(host.run_dir) / f"{name}.json").open("x", encoding="utf-8") as out:
            json.dump({"schema": SCHEMA, "name": name, "verdict": verdict, "reasons": reasons, **common,
                       "facts": facts}, out, sort_keys=True, indent=2, default=str)
        verdicts[name] = verdict
    return verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="run declaration (JSON)")
    parser.add_argument("--run-dir", required=True, help="a new directory")
    parser.add_argument("--instrument-dir", required=True)
    parser.add_argument("--only", default=",".join(PREFLIGHTS))
    args = parser.parse_args(argv)
    config = {**json.loads(Path(args.config).read_text(encoding="utf-8")), "instrument_dir": args.instrument_dir}
    Path(args.run_dir).mkdir(parents=True)
    verdicts = produce(run.LocalHost(config, args.run_dir), [n for n in args.only.split(",") if n])
    print(json.dumps(verdicts))
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
