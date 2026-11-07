"""The P3, P4, P6 and P7 preflight producer on a fake host: real stores and a real provider, no gateway."""

from contextlib import closing
import json
import sqlite3
import sys
import types

import pytest

from bench.instruments.d5_clone import plan as planner, preflights as pf, provider as prov

MARKER = "synthetic-marker-5c1e"
ANCHOR = "LCM identity-anchor recognised 1 replayed rows: session=sess-l cursor=0 incoming=2"


class FakeHost:
    """The LocalHost seams the producer uses. Each turn stores the host and LCM-X rows a working arm would."""

    def __init__(self, root, fault=None):
        self.run_dir, self.fault, self.files, self.removed = root, fault, 0, []
        (root / "pin.json").write_text("{}")
        self.c = {"snapshot_pin": str(root / "pin.json"), "context_length": 200000, "p3_context_length": 64000,
                  "rc": {"lcm_tree": str(root), "commit": "a" * 40}, "ga": {"lcm_tree": str(root), "commit": "a" * 40}}

    def pins(self):
        return {"verdict": "PASS"}

    def plan(self, copy):
        return planner.build_plan([dict(session_id=s, name=n, chat_id=f"d5-{n}", lcm_rows=1, host_rows=1)
                                   for n, s in (("L", "sess-l"), ("S1", "sess-a"), ("S2", "sess-b"))], nonce="t")

    def clone(self, name):
        (self.run_dir / name / "home").mkdir(parents=True)
        return self.run_dir / name / "home"

    def arm(self, tree, home, plan):
        with closing(sqlite3.connect(home / "lcm.db")) as lcm, closing(sqlite3.connect(home / "state.db")) as host:
            lcm.executescript("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id, role, content,"
                              " tool_call_id, conversation_id); CREATE TABLE summary_nodes (node_id INTEGER PRIMARY"
                              " KEY, session_id, summary);")
            host.executescript("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id, role, content,"
                               " tool_call_id); CREATE TABLE sessions (id, end_reason);"
                               " INSERT INTO sessions VALUES ('sess-l', NULL);")
        (home / "lcm-large-outputs").mkdir()
        (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["hermes-lcm-x", "d5-local"]}}))  # JSON is YAML.
        return {"name": tree, "home": home, "plan": plan, "log": [], "sessions": {"L": "sess-l"},
                "session_ids": ["sess-l"], "marks": {"store_id": 0, "host_id": 0, "node_id": 0},
                "facts": dict.fromkeys(("compactions_published", "whitespace_rewrites", "payload_swaps"), 0),
                "stats": {"calls_per_step": {}, "calls_by_model": {}, "replay_misses": {}, "unknown_models": {},
                          "agent_calls_outside_step": 0}}

    def turn(self, ctx, step):
        entries, (sid, home) = ctx["plan"]["provider"]["steps"][step["id"]], ("sess-l", ctx["home"])
        text = step["user_text"] if self.fault == "P7" else step["user_text"].strip()
        host = [("user", text, None), ("assistant", MARKER, None)]
        lcm = [("user", step["user_text"], None), ("assistant", MARKER, None)]
        if step["tools"]:
            self.files += 1
            content = MARKER * (2 if self.fault == "P4" and ctx["name"] == "ga" else 1)
            name, call = f"20261107_000000_t_{'b' * 12}_{self.files:x}.json", entries[0]["tool_calls"][0]["id"]
            (home / "lcm-large-outputs" / name).write_text(json.dumps({"content": content, "content_chars": 1}))
            host.append(("tool", f"[Externalized tool output: {MARKER}; ref={name}]", call))
            lcm.append(("tool", content, call))
            ctx["facts"]["payload_swaps"] += 1
        with closing(sqlite3.connect(home / "lcm.db")) as db, db:
            db.executemany("INSERT INTO messages (session_id, role, content, tool_call_id, conversation_id) VALUES"
                           " (?, ?, ?, ?, 'conv')", [(sid, *r) for r in lcm])
            db.execute("INSERT INTO summary_nodes (session_id, summary) VALUES (?, 'D5 digest v1')", (sid,))
        with closing(sqlite3.connect(home / "state.db")) as db, db:
            db.executemany("INSERT INTO messages (session_id, role, content, tool_call_id) VALUES (?, ?, ?, ?)",
                           [(sid, *r) for r in host])
        ctx["log"] += [ANCHOR] if step["kind"] == "cold" or self.fault == "P6" else []
        ctx["facts"]["compactions_published"] += 1
        ctx["facts"]["whitespace_rewrites"] += bool(step["rewrite"])
        ctx["stats"]["calls_per_step"][step["id"]] = len(entries)
        ctx["stats"]["calls_by_model"]["d5-summary"] = 1
        return self.fault != "crash"

    def capture(self, ctx, kind, step, extras=None):
        return self.run_dir / kind, {}

    def replay(self, ctx, step, t0, h8):
        if self.fault == "replay-error":
            return None, True, None
        conversation = {} if self.fault == "no-conversation" else {"payload": {"conversation_id": "conv"}}
        return {"checks": {"validity": {"verdict": "PASS"}}, **conversation}, False, 0

    def remove(self, path):
        self.removed.append(path.name)

    def facts(self, ctx):
        return {**ctx["facts"], "payload_files_created": sum(1 for _ in (ctx["home"] / "lcm-large-outputs").iterdir())}

    def stats(self, ctx):
        return ctx["stats"]

    def logs(self, ctx):
        return ctx["log"]

    def pause_cron(self, ctx):
        pass

    start = stop = close = pause_cron


@pytest.mark.parametrize("fault", [None, "P3", "P4", "P6", "P7", "crash", "no-conversation", "replay-error"])
def test_each_preflight_fails_on_its_own_fault(tmp_path, monkeypatch, fault):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))  # Not in every test venv.
    if fault == "P3":  # A provider whose second reply differs from the first.
        complete = prov.Provider._complete
        monkeypatch.setattr(prov.Provider, "_complete", lambda self, request: {
            **complete(self, request), "created": len(self.stats["request_log"])})
    host = FakeHost(tmp_path, fault)
    verdicts = pf.produce(host, list(pf.PREFLIGHTS))
    assert host.removed == ([] if fault in ("crash", "replay-error") else ["T0", "H8"])
    expected = dict.fromkeys(pf.PREFLIGHTS, "INCONCLUSIVE" if fault == "crash" else "PASS")
    expected.update({fault: "FAIL"} if fault in pf.PREFLIGHTS else {"P6": "INCONCLUSIVE"} if fault else {})
    assert verdicts == expected
    for name in pf.PREFLIGHTS:
        text = (tmp_path / f"{name}.json").read_text()
        receipt = json.loads(text)
        assert MARKER not in text and receipt["schema"] == pf.SCHEMA and receipt["trees"]["rc"] == "a" * 40
        assert receipt["plan_sha256"] and bool(receipt["reasons"]) == (receipt["verdict"] != "PASS")


def test_no_plan_is_inconclusive_and_receipts_are_never_overwritten(tmp_path):
    host = FakeHost(tmp_path)
    host.pins = lambda: {"verdict": "FAIL"}
    assert pf.produce(host, ["P3", "P7"]) == {"P3": "INCONCLUSIVE", "P7": "INCONCLUSIVE"}
    assert "P1 pins did not pass" in json.loads((tmp_path / "P3.json").read_text())["reasons"][0]
    with pytest.raises(FileExistsError):
        pf.produce(host, ["P3"])
