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
import time
from urllib.request import ProxyHandler, Request, build_opener

from . import bars, capture, plan as planner, provider as prov, run

SCHEMA = run.PREFLIGHT_SCHEMA  # The runner accepts a receipt only in this schema (ruling R5-5).
ANCHOR = "identity-anchor recognised"
HYGIENE = "Session hygiene"  # The host's own gateway session-hygiene compression (gateway/run_turn.py).
DIGEST12 = re.compile(r"_([0-9a-f]{12})_[0-9a-f]+\.json$")  # The payload filename's content digest prefix.


class Inconclusive(Exception):
    """The preflight could not run to its checks. Its message is this module's own fixed text."""


def _count(value):
    """A payload's size field as a receipt may carry it: an int, else None (never the field's text)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reason(exc):
    """A receipt reason for an exception: this module's own message, else the exception's type only. Another
    exception's text can quote its input (a parser error quotes the document), and receipts carry no content."""
    return str(exc)[:200] if isinstance(exc, Inconclusive) else type(exc).__name__


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def _failed(checks):
    return [name for name, bad in checks if bad]


def _steps(plan, epoch=None, conversation="L"):
    return [s for e in plan["epochs"] if epoch in (None, e["epoch"]) for s in e["steps"]
            if s["conversation"] == conversation]


def _turn(host, ctx, step):
    if host.turn(ctx, step) is not True:
        raise Inconclusive(f"turn {step['id']} not completed")


def _session(host, tree, clone, plan, steps, *, until=None, restart_before=(), settle=0):
    """Arm a fresh clone, run the steps (restarting the gateway before those named), stop; returns the context. With
    *settle*, a gateway whose steps all ran without *until* holding stays up for at most that many seconds while it
    does (a compression the host finishes in the background); ctx["settle_s"] is the time it waited."""
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
        else:
            started = time.monotonic()
            while settle and until and not until(ctx) and time.monotonic() - started < settle:
                time.sleep(min(5.0, settle / 10))
            ctx["settle_s"] = round(time.monotonic() - started, 1)
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
    """(host contents, LCM-X contents) for the step's scripted tool call, stored after the arm's marks. Host rows are the
    active ones: an in-place compaction keeps the original row, inactive, beside its active copy."""
    call = plan["provider"]["steps"][step["id"]][0]["tool_calls"][0]["id"]
    return tuple([t for t, in _rows(ctx, db, f"SELECT content FROM messages WHERE role = 'tool' AND tool_call_id = ?"
                                    f" AND {col} > ?{extra}", (call, ctx["marks"][mark]))]
                 for db, col, mark, extra in (("state.db", "id", "host_id", " AND active = 1"),
                                              ("lcm.db", "store_id", "store_id", "")))


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
    host.c = {**saved, "context_length": saved["p3_context_length"]}  # Only P3 crosses the threshold early: the declared context_length (1,000,000) keeps the others below it.
    # P3 drives S2, not L. Over the small context, L's ~560k-token history also starts the host's own session-hygiene
    # compression, which holds the compression lock in the background, so LCM-X's compaction is skipped and whether
    # one publishes depended on load (run 4 yes, run 5 no). S2's ~65k-token history crosses LCM-X's threshold and
    # compresses quickly. A compaction still running after the last turn gets a bounded wait; a miss is INCONCLUSIVE.
    unsure, wait = [], saved.get("p3_publish_wait", 300)
    try:
        for tree in ("rc", "ga"):
            ctx = _session(host, tree, f"pf-P3-{tree}", plan,
                           _steps(plan, conversation="S2")[:saved.get("p3_max_turns", 6)],
                           until=lambda c: host.facts(c)["compactions_published"] >= 1, settle=wait)
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
                "unknown_model_sha256_12": sorted(_sha(m)[:12] for m in stats["unknown_models"]),  # Never the names.
                "off_script": [s for s in ran if stats["calls_per_step"][s] != len(plan["provider"]["steps"][s])],
                "publish_wait_s": ctx.get("settle_s", 0),
                # Recorded, not judged: S2 at P3's small context may also start the host's own hygiene compression.
                "hygiene_lines": len({line for line in host.logs(ctx)
                                      if HYGIENE in line and "skipping compression" not in line})}
            unsure += [f"{tree}: no compaction published within {wait} s of the last turn"] if published < 1 else []
            reasons += [f"{tree}: {k}" for k in _failed([
                ("no step ran", not ran),  # A compaction with no turn behind it shows nothing about the turns.
                ("no d5-summary call", published >= 1 and f["summary_calls"] < 1),
                ("a new summary node not from the fixed rule",
                 published >= 1 and (not nodes or f["fixed_rule_nodes"] != len(nodes))),
                ("agent calls outside a step", f["outside_step"]), ("replay misses", f["replay_misses"]),
                ("unknown models", f["unknown_models"]), ("step calls != planned entries", f["off_script"])])]
    finally:
        host.c = saved
    return reasons, unsure, facts


def _home_paths(text, home):
    """*text* with each absolute path under this arm's own home masked. The host's terminal tool truncates a large
    output and names the file that keeps it in full, under the arm's home and with a timestamp, so the same output
    differs between the trees only there."""
    for root in sorted({str(home), str(Path(home).resolve())}, key=len, reverse=True):
        # Beneath it, over path characters only: a separator, an escape or any text after the path stays compared.
        text = re.sub(re.escape(root) + r"/[A-Za-z0-9._/+@-]*", "<home-path>", text)
    return text


def p4(host, plan):
    step, reasons, facts = next(s for s in _steps(plan, 1) if s["tools"] == "large"), [], {}
    for tree in ("rc", "ga"):
        ctx = _session(host, tree, f"pf-P4-{tree}", plan, [step])
        hosts, lcms = _tool_rows(ctx, plan, step)
        refs = [m[1] for t in hosts + lcms for m in run.EXTERNALIZED_REF.finditer(t or "")]
        path = Path(ctx["home"]) / "lcm-large-outputs" / (refs[0] if refs else "-")
        payload = json.loads(path.read_bytes()) if refs and path.is_file() else {}
        content = payload.get("content")
        f = facts[tree] = {
            "rows": [len(hosts), len(lcms)], "refs": len(refs), "files": host.facts(ctx)["payload_files_created"],
            "stub_masked_sha256": [_sha(_home_paths(run.EXTERNALIZED_REF.sub(
                lambda m: re.sub(r"(;\s*(?:chars|bytes)=)[0-9]+(?=;)", r"\1<size>",
                                 m[0].replace(m[1], "<ref>")), t or ""), ctx["home"])) for t in hosts + lcms],
            "content_sha256": _sha(payload.get("content")),
            "content_masked_sha256": _sha(_home_paths(payload["content"], ctx["home"]))
            if isinstance(payload.get("content"), str) else None, "content_chars": _count(payload.get("content_chars")),
            "content_bytes": _count(payload.get("content_bytes")), "digest12": [m[1] for m in map(DIGEST12.search, refs) if m]}
        reasons += _failed([(f"{tree}: not one tool row per store, one reference and one payload file",
                             (f["rows"], f["refs"], f["files"]) != ([1, 1], 1, 1)),
                            (f"{tree}: the payload file holds no content and sizes",  # Two empty payloads compare equal.
                             not payload.get("content") or not all(isinstance(f[k], int) and f[k] > 0
                                                                   for k in ("content_chars", "content_bytes"))),
                            (f"{tree}: the reference's digest is not its own content's hash",  # LCM-X names the
                             f["digest12"] != [f["content_sha256"][:12]])])  # payload by the content it holds.
        for field, actual in (("content_chars", len(content) if isinstance(content, str) else None),
                              ("content_bytes", len(content.encode("utf-8")) if isinstance(content, str) else None)):
            if actual is None or f[field] != actual:
                reasons.append(f"{tree}: {field} does not match its own content")
        for text in hosts + lcms:
            for stub in run.EXTERNALIZED_REF.finditer(text or ""):
                for field in ("chars", "bytes"):
                    sizes = re.findall(r";\s*" + field + r"=([^;]*)(?=;)", stub[0])
                    if sizes != [str(f[f"content_{field}"])]:
                        reasons.append(f"{tree}: stub {field} does not match payload content_{field}")
    # Equal masked hashes imply equal masked character/UTF-8 lengths. Raw sizes are validated within each tree only.
    return reasons + [f"{k} differs between the trees" for k in ("stub_masked_sha256", "content_masked_sha256")
                      if facts["rc"][k] != facts["ga"][k]], [], facts


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

    def lines():  # Distinct records written since the arm (the host writes each LCM-X line to more than one log file).
        return set(host.logs(ctx)) - old

    def anchors():
        return len({line for line in lines() if ANCHOR in line and f"session={sid}" in line})

    old = set(host.logs(ctx))  # A clone that carries older log files never counts their lines.
    try:
        host.start(ctx)
        before = _tops(ctx)
        t0_dir, t0 = host.capture(ctx, "T0", cold, {"RC3/store.db": host.run_dir / "pf-plan" / "home" / "store.db"})
        _turn(host, ctx, cold)
        durable, waited = host.wait_cold_ingest(ctx, cold, t0)  # As run_arm: H8 only once the cold ingest is durable.
        if durable:
            h8_dir, _ = host.capture(ctx, "H8", cold)
            receipt, error, range_new = host.replay(ctx, cold, t0_dir, h8_dir)  # As run_arm: right after its H8.
        else:
            receipt, error, range_new = None, True, None
        bind, cold_anchors, mid = bars.bar2_bind(receipt, range_new_rows=range_new, error=error), anchors(), _tops(ctx)
        # The edition's conversation_id is the T0 host session row's session_key (p_s6r_clone.py:1148).
        keys = {k for k, in run._rows(Path(t0_dir) / "host-0.db", "SELECT session_key FROM sessions WHERE id = ?", (sid,))}
        if receipt is not None:  # As run_arm: the captures go once the replay left its receipt.
            host.remove(t0_dir)
            host.remove(h8_dir)
        _turn(host, ctx, warm)
        warm_anchors = anchors() - cold_anchors
        # Any hygiene line but a skip: its usual lines name no session, and P6 drives only L.
        hygiene = len({line for line in lines() if HYGIENE in line and "skipping compression" not in line})
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
    live = {str(c) for c, in _rows(ctx, "lcm.db", "SELECT DISTINCT conversation_id FROM messages WHERE session_id = ?"
                                   " AND store_id > ? AND store_id <= ?", (sid, before["store"], mid["store"])) if c}
    replayed = {str(c) for c in _find(receipt or {}, "conversation_id") if c} | {str(k) for k in keys if k}
    f = {"bind": bind, "cold_anchors": cold_anchors, "warm_anchors": warm_anchors, "ended_sessions": ended,
         "warm_rows": [new("lcm.db", "store_id", mid["store"]), new("state.db", "id", mid["host"])],
         "host_rows": [new("state.db", "id", before["host"], False), new("state.db", "id", before["host"])],
         "live_conversation_sha256": sorted(map(_sha, live)), "receipt_conversation_sha256": sorted(map(_sha, replayed)),
         "cold_ingest_durable": durable, "cold_ingest_wait_s": waited, "hygiene_lines": hygiene,
         "replay_diagnostics": ctx.get("replay_diagnostics") if durable else None}
    fails = _failed([
        ("bind validity is not PASS", not error and bind["validity"] != "PASS"),
        ("range_new_rows is not 0", not error and range_new != 0),
        ("no anchor line at the cold bind", cold_anchors < 1), ("an anchor line after the warm turn", warm_anchors),
        ("the warm turn stored LCM-X rows beyond the host's", len(set(f["warm_rows"])) != 1),
        ("the warm turn stored fewer than two rows", min(f["warm_rows"]) < 2),  # Its user row and its reply.
        ("a chosen session ended", ended), ("d5-local is not enabled", "d5-local" not in (enabled or ())),
        ("new host rows outside L's session", len(set(f["host_rows"])) != 1 or f["host_rows"][0] < 2),
        ("the live and replayed conversation_id differ", replayed and live and live != replayed)])
    unsure = _failed([("the cold turn's ingest was not durable within the wait (no H8, no replay)", not durable),
                 (run.replay_error(f["replay_diagnostics"]), durable and error),
                 ("neither the T0 session row nor the edition receipt gives a conversation_id to compare", not replayed),
                 ("the cold bind's LCM-X rows carry no conversation_id to compare", not live)])
    if hygiene:  # The host's own compression rewrote L during the bind: what P6 saw is not the turn's own bind.
        kept = [r for r in fails if r == "d5-local is not enabled"]  # Configuration: no compression causes it.
        return kept, [f"host session hygiene ran on L ({hygiene} lines); the checks are not judged"] + [
            r for r in fails if r not in kept] + unsure, f
    return fails, unsure, f


def p7(host, plan):
    rewrite, large = _steps(plan, 0)[1], next(s for s in _steps(plan, 1) if s["tools"] == "large")
    cold = _steps(plan, 2)[0]
    ctx = _session(host, "rc", "pf-P7-rc", plan, [rewrite, large, cold], restart_before=(cold["id"],))
    users = [t for t, in _rows(ctx, "state.db", "SELECT content FROM messages WHERE role = 'user' AND session_id = ?"
                               " AND id > ? ORDER BY id", (ctx["sessions"]["L"], ctx["marks"]["host_id"]))]
    workload, (hosts, lcms) = host.facts(ctx), _tool_rows(ctx, plan, large)
    swap = len(hosts) == len(lcms) == 1 and run._swap_payload(ctx["home"], hosts[0]) == lcms[0]
    f = {**{k: workload[k] for k in ("whitespace_rewrites", "payload_swaps", "payload_files_created")},
         # Every new user row, not the first: an in-place compaction also writes active copies of older user rows.
         "stripped_stored": rewrite["user_text"].strip() in users, "raw_stored": rewrite["user_text"] in users,
         "new_user_rows": len(users),
         "tool_rows": [len(hosts), len(lcms)], "host_row_is_the_stub": swap}
    return _failed([
        ("the host did not store the stripped form", not f["stripped_stored"]),
        ("the host stored the unstripped form", f["raw_stored"]),
        ("no whitespace rewrite", f["whitespace_rewrites"] < 1), ("no payload swap", f["payload_swaps"] < 1),
        ("no payload file", f["payload_files_created"] < 1),
        ("the host row differs from the LCM-X row by more than the stub", not swap)]), [], f


PREFLIGHTS = {"P3": p3, "P4": p4, "P6": p6, "P7": p7}


def produce(host, names):
    """Write <run_dir>/<name>.json for each preflight; returns {name: verdict}."""
    common = {"snapshot_pin_sha256": capture._sha256(host.c["snapshot_pin"]), "plan_sha256": None,
              "trees": {t: run.tree_id(host.c, t) for t in ("rc", "ga")}, "config_sha256": run.config_identity(host.c),
              **run.instrument_identity()}
    failed, verdicts = None, {}
    try:
        if host.pins().get("verdict") != "PASS":  # The rules module is imported for planning only after P1.
            raise Inconclusive("P1 pins did not pass")
        # P2 before any clone, as in the runner: a symlink or unpinned file in the snapshot never reaches a clone,
        # and a receipt never binds a snapshot pin that was not verified.
        if host.snapshot().get("verdict") != "PASS":
            raise Inconclusive("P2 snapshot did not pass")
        if capture._sha256(host.c["snapshot_pin"]) != common["snapshot_pin_sha256"]:  # Bind the pin P2 verified.
            raise Inconclusive("the snapshot pin changed while P2 ran")
        plan = host.plan(host.clone("pf-plan"))
        common["plan_sha256"] = planner.plan_sha256(plan)
        common["plan_conversations"] = [  # Counts and the session id's sha256 12 only (#2471).
            {**d, "session_sha256_12": hashlib.sha256(c["session_id"].encode()).hexdigest()[:12]}
            for d, c in zip(planner.declaration(plan)["conversations"], plan["conversations"])]
        with (Path(host.run_dir) / "plan-counts.json").open("x", encoding="utf-8") as out:  # Readable before any check.
            json.dump(common["plan_conversations"], out, sort_keys=True, indent=2)
    except Exception as exc:
        failed = f"no plan: {_reason(exc)}"
    for name in names:
        try:
            if failed:
                raise Inconclusive(failed)
            if None in common["trees"].values():
                raise Inconclusive("a tree has no declared 40-hex commit, so no receipt could bind it")
            if name not in run.TREE_BOUND_PREFLIGHTS and common["trees"]["rc"] != common["trees"]["ga"]:
                # Ruling R5-5: P3, P6 and P7 run the previous GA in both arms; the runner reads them that way.
                raise Inconclusive("this check runs the previous GA in both arms, and the rc tree is another tree")
            saved = host.c.get("rc")
            if name not in run.TREE_BOUND_PREFLIGHTS:  # R5-5: the previous GA's own configuration runs in both arms.
                host.c["rc"] = host.c["ga"]
            try:
                reasons, unsure, facts = PREFLIGHTS[name](host, plan)
            finally:
                host.c["rc"] = saved
            verdict, reasons = "FAIL" if reasons else "INCONCLUSIVE" if unsure else "PASS", reasons + unsure
        except Exception as exc:  # Never a PASS: the checks did not all run.
            verdict, reasons, facts = "INCONCLUSIVE", [_reason(exc)], {}
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
    Path(run.outside_package(args.run_dir)).mkdir(parents=True)
    verdicts = produce(run.LocalHost(config, args.run_dir), [n for n in args.only.split(",") if n])
    print(json.dumps(verdicts))
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
