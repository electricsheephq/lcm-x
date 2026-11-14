"""The P3, P4, P6 and P7 preflight producer on a fake host: real stores and a real provider, no gateway."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
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
        self.p3_conversations = []
        self.c = {"snapshot_pin": str(root / "pin.json"), "context_length": 200000, "p3_context_length": 64000,
                  "p3_publish_wait": 0.3,
                  "rc": {"lcm_tree": str(root), "commit": "a" * 40}, "ga": {"lcm_tree": str(root), "commit": "a" * 40}}

    def pins(self):
        return {"verdict": "PASS"}

    def snapshot(self):
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
                               " tool_call_id, active DEFAULT 1); CREATE TABLE sessions (id, end_reason);"
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
        if self.fault == "user-copy-first" and "pf-P7" in str(home):  # In P7: an in-place compaction's active copy of an older user row comes first.
            host.insert(0, ("user", "older user row copy", None))
        lcm = [("user", step["user_text"], None), ("assistant", MARKER, None)]
        if self.fault == "warm-unstored" and step["kind"] == "warm" and "pf-P6" in str(home):  # Completed, stored nothing.
            host, lcm = [], []
        if "pf-P3" in str(home):
            self.p3_conversations.append(step["conversation"])
        if step["tools"]:
            self.files += 1
            content = MARKER * (2 if self.fault == "P4" and ctx["name"] == "ga" else 1)
            if self.fault == "home-paths":  # The terminal tool's truncation result, as the pinned host writes it.
                saved = f"{home}/tool_output/terminal-{ctx['name']}-1762556{len(ctx['name'])}-4242.txt"
                content = json.dumps({"output": MARKER + "\n\n... [OUTPUT TRUNCATED - 204,800 chars] ...",
                                      "exit_code": 0, "full_output_path": saved,
                                      "truncation_note": f"Full output saved to {saved}"})
            digest = "b" * 12 if self.fault == "wrong-digest" else hashlib.sha256(content.encode()).hexdigest()[:12]
            name, call = f"20261107_000000_t_{digest}_{self.files:x}.json", entries[0]["tool_calls"][0]["id"]
            (home / "lcm-large-outputs" / name).write_text(json.dumps(
                {} if self.fault == "empty-payload" else {"content": content, "content_chars": len(content),
                                                        "content_bytes": len(content.encode("utf-8"))}))
            host.append(("tool", f"[Externalized tool output: {MARKER}; chars={len(content)}; "
                         f"bytes={len(content.encode('utf-8'))}; ref={name}]", call))
            lcm.append(("tool", content, call))
            ctx["facts"]["payload_swaps"] += 1
        with closing(sqlite3.connect(home / "lcm.db")) as db, db:
            db.executemany("INSERT INTO messages (session_id, role, content, tool_call_id, conversation_id) VALUES"
                           " (?, ?, ?, ?, 'conv')", [(sid, *r) for r in lcm])
            db.execute("INSERT INTO summary_nodes (session_id, summary) VALUES (?, 'D5 digest v1')", (sid,))
        with closing(sqlite3.connect(home / "state.db")) as db, db:
            db.executemany("INSERT INTO messages (session_id, role, content, tool_call_id) VALUES (?, ?, ?, ?)",
                           [(sid, *r) for r in host])
            if self.fault == "inactive-copy":  # An in-place compaction: the original stays, inactive, beside it.
                db.executemany("INSERT INTO messages (session_id, role, content, tool_call_id, active) VALUES"
                               " (?, ?, ?, ?, 0)", [(sid, *r) for r in host if r[0] == "tool"])
        stamp = f"{step['id']} "  # A distinct record per turn; the host writes each one to two log files.
        ctx["log"] += [stamp + ANCHOR] * 2 if step["kind"] == "cold" or self.fault == "P6" else []
        if self.fault == "hygiene":
            ctx["log"] += [f"{stamp}INFO gateway.run: Session hygiene compression for session sess-l started"] * 2
        if self.fault == "hygiene-plain":  # The host's usual lines name no session (gateway/run_turn.py).
            ctx["log"] += [f"{stamp}INFO gateway.run: Session hygiene: 40 messages, ~560,339 tokens (85%) —"
                           " auto-compressing", f"{stamp}INFO gateway.run: Session hygiene: compressed 40 → 6 msgs"]
        if self.fault == "hygiene-skip":  # A skip compresses nothing: P6 still judges.
            ctx["log"] += [f"{stamp}INFO gateway.run: Session hygiene: skipping compression for sess-l; cooldown"]
        ctx["facts"]["compactions_published"] += self.fault not in ("no-publish", "late-publish")
        ctx["facts"]["whitespace_rewrites"] += bool(step["rewrite"])
        ctx["stats"]["calls_per_step"][step["id"]] = len(entries)
        ctx["stats"]["calls_by_model"]["d5-summary"] = 1
        return self.fault != "crash"

    def capture(self, ctx, kind, step, extras=None):
        return self.run_dir / kind, {}

    def wait_cold_ingest(self, ctx, step, t0):
        return self.fault != "not-durable", 0.0

    def replay(self, ctx, step, t0, h8):
        if self.fault == "replay-error":
            return None, True, None
        conversation = {} if self.fault == "no-conversation" else {"payload": {"conversation_id": "conv"}}
        return {"checks": {"validity": {"verdict": "PASS"}}, **conversation}, False, 0

    def remove(self, path):
        self.removed.append(path.name)

    def facts(self, ctx):
        if self.fault == "late-publish" and "pf-P3" in str(ctx["home"]):  # The compaction lands after the last turn.
            ctx["polls"] = ctx.get("polls", 0) + 1
            ctx["facts"]["compactions_published"] = int(ctx["polls"] > 6)  # After the six turns' checks: the first wait.
        return {**ctx["facts"], "payload_files_created": sum(1 for _ in (ctx["home"] / "lcm-large-outputs").iterdir())}

    def stats(self, ctx):
        return ctx["stats"]

    def logs(self, ctx):
        return ctx["log"]

    def pause_cron(self, ctx):
        pass

    start = stop = close = pause_cron


@pytest.mark.parametrize("fault", [None, "P3", "P4", "P6", "P7", "crash", "no-conversation", "replay-error",
                                   "not-durable", "inactive-copy",
                                   "user-copy-first", "hygiene", "empty-payload", "warm-unstored",
                                   "hygiene-plain", "hygiene-skip", "no-publish", "late-publish", "home-paths",
                                   "wrong-digest"])
def test_each_preflight_fails_on_its_own_fault(tmp_path, monkeypatch, fault):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))  # Not in every test venv.
    if fault == "P3":  # A provider whose second reply differs from the first.
        complete = prov.Provider._complete
        monkeypatch.setattr(prov.Provider, "_complete", lambda self, request: {
            **complete(self, request), "created": len(self.stats["request_log"])})
    host = FakeHost(tmp_path, fault)
    verdicts = pf.produce(host, list(pf.PREFLIGHTS))
    assert host.removed == ([] if fault in ("crash", "replay-error", "not-durable") else ["T0", "H8"])
    expected = dict.fromkeys(pf.PREFLIGHTS, "INCONCLUSIVE" if fault == "crash" else "PASS")
    fails = {"empty-payload": ("P4", "P7"), "warm-unstored": ("P6",), "wrong-digest": ("P4",)}  # Two empty payloads or no warm rows compare equal.
    passing = ("inactive-copy", "user-copy-first", "hygiene-skip", "late-publish", "home-paths")
    expected.update({fault: "FAIL"} if fault in pf.PREFLIGHTS else dict.fromkeys(fails[fault], "FAIL") if fault in fails
                    else {"P3": "INCONCLUSIVE"} if fault == "no-publish"
                    else {"P6": "INCONCLUSIVE"} if fault and fault not in passing else {})
    assert verdicts == expected
    p6 = json.loads((tmp_path / "P6.json").read_text())["facts"]
    if fault != "crash":  # P3 drives S2 only: L's history starts the host's own hygiene compression (run 5).
        assert set(host.p3_conversations) == {"S2"}
    if fault is None:  # One distinct anchor record per turn, though each is written twice.
        assert (p6["cold_anchors"], p6["warm_anchors"], p6["hygiene_lines"]) == (1, 0, 0)
    if fault != "crash":  # The plan's counts are written before any check runs (#2471): counts and hashes only.
        counts = (tmp_path / "plan-counts.json").read_text()
        assert MARKER not in counts and [set(c) for c in json.loads(counts)] == [
            {"name", "lcm_rows", "host_rows", "flagged_summary_nodes", "session_sha256_12"}] * 3
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


def test_an_exception_s_text_never_enters_a_receipt(tmp_path, monkeypatch):
    """An exception's message can quote its input (a parser error quotes the document): only its type is kept."""
    def leaks(host, plan):
        raise ValueError(f"could not parse: {MARKER}")

    monkeypatch.setitem(pf.PREFLIGHTS, "P3", leaks)
    assert pf.produce(FakeHost(tmp_path), ["P3"]) == {"P3": "INCONCLUSIVE"}
    text = (tmp_path / "P3.json").read_text()
    assert MARKER not in text and json.loads(text)["reasons"] == ["ValueError"]


def test_only_paths_beneath_the_arm_s_home_are_masked(tmp_path):
    """A sibling such as <home>-backup is outside the home: masking it could hide a real difference between the trees."""
    home = tmp_path / "pf-P4-rc" / "home"
    text = f'{{"full_output_path": "{home}/tool_output/a-1.txt", "other": "{home}-backup/lost.txt", "bare": "{home}"}}'
    masked = pf._home_paths(text, home)
    assert masked.count("<home-path>") == 1 and f"{home}-backup/lost.txt" in masked and f'"bare": "{home}"' in masked
    escaped = json.dumps({"output": f"{home}/a.txt\nRC-TEXT more"})  # The JSON escape ends the path, not the text.
    assert pf._home_paths(escaped, home) == json.dumps({"output": "<home-path>\nRC-TEXT more"})
    assert pf._home_paths(f"{home}/o.txt;exit=1", home) != pf._home_paths(f"{home}/o.txt;exit=0", home)
    assert pf._home_paths(f"{home}/a/b.txt", home) == pf._home_paths(f"{home}/a/c.txt", home)


MODEL = "unlisted-model-9d2b"


class Tweaked(FakeHost):
    """FakeHost plus one targeted change: tweak(host, ctx, step) after each turn, ctx["over"] over the workload facts,
    host.reply as the edition replay's result, and *seed* log lines already present before the arm."""

    def __init__(self, root, tweak, fault=None, seed=()):
        super().__init__(root, fault)
        self.tweak, self.seed, self.reply, self.armed = tweak, list(seed), None, []

    def arm(self, tree, home, plan):
        self.armed.append((home.parent.name, self.c["context_length"]))
        return {**super().arm(tree, home, plan), "log": list(self.seed)}

    def turn(self, ctx, step):
        done = super().turn(ctx, step)
        self.tweak(self, ctx, step)
        return done

    def replay(self, ctx, step, t0, h8):
        return self.reply or super().replay(ctx, step, t0, h8)

    def facts(self, ctx):
        return {**super().facts(ctx), **ctx.get("over", {})}


def _sql(ctx, db, sql, *args):
    with closing(sqlite3.connect(ctx["home"] / db)) as con, con:
        con.execute(sql, args)


def _payload(ctx, **changes):
    """Rewrite the arm's one payload file with *changes*, renamed by its content's digest, and repoint the host row."""
    old = next((ctx["home"] / "lcm-large-outputs").iterdir())
    data = {**json.loads(old.read_text()), **changes}
    new = old.with_name(old.name.replace(pf.DIGEST12.search(old.name)[1],
                                         hashlib.sha256(data["content"].encode()).hexdigest()[:12]))
    old.rename(new)
    new.write_text(json.dumps(data))
    _sql(ctx, "state.db", "UPDATE messages SET content = replace(content, ?, ?)", old.name, new.name)


def _receipt(validity="PASS", conversation="conv", range_new=0):
    return {"checks": {"validity": {"verdict": validity}}, "payload": {"conversation_id": conversation}}, False, range_new


def _disable(ctx):
    (ctx["home"] / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["hermes-lcm-x"]}}))


def _rc(ctx):
    return ctx["name"] == "rc"


def _noop(host, ctx, step):
    pass


def _case(name, tweak, verdict, reasons, id, **kw):
    return pytest.param(name, tweak, verdict, reasons, kw, id=id)


HYGIENE_NOTE = "host session hygiene ran on L (2 lines); the checks are not judged"
NO_LIVE = "the cold bind's LCM-X rows carry no conversation_id to compare"
NO_REPLAYED = "neither the T0 session row nor the edition receipt gives a conversation_id to compare"
P4_ROWS, P4_EMPTY = "not one tool row per store, one reference and one payload file", "the payload file holds no content and sizes"
P4_DIGEST = "the reference's digest is not its own content's hash"
CASES = [
    _case("P3", _noop, "FAIL", [f"{m}: the same request gave different bytes" for m in ("d5-summary", "d5-aux")],
          "P3-replies-vary", vary=True),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"]["calls_per_step"].clear(), "FAIL", ["rc: no step ran"], "P3-no-step"),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"]["calls_by_model"].clear(), "FAIL", ["rc: no d5-summary call"],
          "P3-no-summary-call"),
    _case("P3", lambda h, c, s: _rc(c) and _sql(c, "lcm.db", "INSERT INTO summary_nodes (summary) VALUES ('free text')"),
          "FAIL", ["rc: a new summary node not from the fixed rule"], "P3-free-node"),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"].update(agent_calls_outside_step=1), "FAIL",
          ["rc: agent calls outside a step"], "P3-outside-step"),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"].update(replay_misses={"x": 1}), "FAIL", ["rc: replay misses"],
          "P3-replay-miss"),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"].update(unknown_models={MODEL: 1}), "FAIL", ["rc: unknown models"],
          "P3-unknown-model"),
    _case("P3", lambda h, c, s: _rc(c) and c["stats"]["calls_per_step"].update({s["id"]: 9}), "FAIL",
          ["rc: step calls != planned entries"], "P3-off-script"),
    _case("P4", lambda h, c, s: _rc(c) and (c["home"] / "lcm-large-outputs" / "extra.json").write_text("{}"), "FAIL",
          [f"rc: {P4_ROWS}"], "P4-second-file"),
    _case("P4", lambda h, c, s: _payload(c, content_chars=0), "FAIL",
          [reason for tree in ("rc", "ga") for reason in (f"{tree}: {P4_EMPTY}",
           f"{tree}: content_chars does not match its own content", f"{tree}: stub chars does not match payload content_chars")],
          "P4-zero-chars"),  # Raw sizes must remain positive, though they are not compared between trees.
    _case("P4", _noop, "FAIL", [f"rc: {P4_DIGEST}", f"ga: {P4_DIGEST}"], "P4-wrong-digest", fault="wrong-digest"),
    _case("P4", lambda h, c, s: _rc(c) and _payload(c, content_chars=2), "FAIL",
          ["rc: content_chars does not match its own content", "rc: stub chars does not match payload content_chars"],
          "P4-chars-differ"),
    _case("P4", lambda h, c, s: _rc(c) and _payload(c, content_bytes=2), "FAIL",
          ["rc: content_bytes does not match its own content", "rc: stub bytes does not match payload content_bytes"],
          "P4-bytes-differ"),
    _case("P4", lambda h, c, s: _rc(c) and _sql(c, "state.db", "UPDATE messages SET content ="
                                                 " replace(content, 'output: ', 'output: other ')"),
          "FAIL", ["stub_masked_sha256 differs between the trees"], "P4-stub-differs"),
    _case("P4", lambda h, c, s: _rc(c) and _payload(c, content=MARKER.upper()), "FAIL",  # Same length, its own digest.
          ["content_masked_sha256 differs between the trees"], "P4-content-differs"),
    _case("P6", lambda h, c, s: setattr(h, "reply", _receipt(validity="FAIL")), "FAIL", ["bind validity is not PASS"],
          "P6-validity"),
    _case("P6", lambda h, c, s: setattr(h, "reply", _receipt(range_new=1)), "FAIL", ["range_new_rows is not 0"],
          "P6-range-new"),
    _case("P6", lambda h, c, s: c.update(log=[x for x in c["log"] if not x.startswith(s["id"] + " ")]), "FAIL",
          ["no anchor line at the cold bind"], "P6-no-cold-anchor"),
    _case("P6", lambda h, c, s: c.update(log=[x for x in c["log"] if not x.startswith(s["id"] + " ")]), "FAIL",
          ["no anchor line at the cold bind"], "P6-anchor-before-arm", seed=[f"old {ANCHOR}"] * 2),
    _case("P6", _noop, "FAIL", ["an anchor line after the warm turn"], "P6-warm-anchor", fault="P6"),
    _case("P6", lambda h, c, s: s["kind"] == "warm" and _sql(c, "lcm.db", "INSERT INTO messages (session_id, role)"
                                                              " VALUES ('sess-l', 'assistant')"),
          "FAIL", ["the warm turn stored LCM-X rows beyond the host's"], "P6-warm-extra-lcm"),
    _case("P6", lambda h, c, s: s["kind"] == "warm" and [_sql(c, db, "DELETE FROM messages WHERE rowid ="
                                                              " (SELECT max(rowid) FROM messages)")
                                                         for db in ("lcm.db", "state.db")],
          "FAIL", ["the warm turn stored fewer than two rows"], "P6-warm-one-row"),
    _case("P6", lambda h, c, s: _sql(c, "state.db", "UPDATE sessions SET end_reason = 'x'"), "FAIL",
          ["a chosen session ended"], "P6-session-ended"),
    _case("P6", lambda h, c, s: _disable(c), "FAIL", ["d5-local is not enabled"], "P6-d5-local-off"),
    _case("P6", lambda h, c, s: s["kind"] == "warm" and _sql(c, "state.db", "INSERT INTO messages (session_id, role)"
                                                              " VALUES ('sess-x', 'user')"),
          "FAIL", ["new host rows outside L's session"], "P6-host-row-elsewhere"),
    _case("P6", lambda h, c, s: setattr(h, "reply", _receipt(conversation="other")), "FAIL",
          ["the live and replayed conversation_id differ"], "P6-conversation-differs"),
    _case("P6", lambda h, c, s: _sql(c, "lcm.db", "UPDATE messages SET conversation_id = ''"), "INCONCLUSIVE",
          [NO_LIVE], "P6-live-id-empty"),
    _case("P6", lambda h, c, s: [_sql(c, "lcm.db", "UPDATE messages SET conversation_id = ''"),
                                 setattr(h, "reply", _receipt(conversation=""))], "INCONCLUSIVE",
          [NO_REPLAYED, NO_LIVE], "P6-both-ids-empty"),
    _case("P6", lambda h, c, s: _disable(c), "FAIL", ["d5-local is not enabled", HYGIENE_NOTE], "P6-hygiene-d5-local-off",
          fault="hygiene"),
    _case("P7", lambda h, c, s: s["rewrite"] and _sql(c, "state.db", "UPDATE messages SET content = 'other'"
                                                       " WHERE role = 'user' AND content = ?", s["user_text"].strip()),
          "FAIL", ["the host did not store the stripped form"], "P7-stripped-missing"),
    _case("P7", lambda h, c, s: s["rewrite"] and _sql(c, "state.db", "INSERT INTO messages (session_id, role, content)"
                                                       " VALUES ('sess-l', 'user', ?)", s["user_text"]),
          "FAIL", ["the host stored the unstripped form"], "P7-raw-stored"),
    _case("P7", lambda h, c, s: c.update(over={"whitespace_rewrites": 0}), "FAIL", ["no whitespace rewrite"],
          "P7-no-rewrite"),
    _case("P7", lambda h, c, s: c.update(over={"payload_swaps": 0}), "FAIL", ["no payload swap"], "P7-no-swap"),
    _case("P7", lambda h, c, s: c.update(over={"payload_files_created": 0}), "FAIL", ["no payload file"], "P7-no-file"),
    _case("P7", lambda h, c, s: s["tools"] and _sql(c, "lcm.db", "UPDATE messages SET content = content || 'x'"
                                                     " WHERE role = 'tool'"),
          "FAIL", ["the host row differs from the LCM-X row by more than the stub"], "P7-not-the-stub"),
]


@pytest.mark.parametrize("name, tweak, verdict, reasons, kw", CASES)
def test_each_check_fires_alone(tmp_path, monkeypatch, name, tweak, verdict, reasons, kw):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    if kw.pop("vary", False):  # A provider whose second reply differs from the first.
        complete = prov.Provider._complete
        monkeypatch.setattr(prov.Provider, "_complete", lambda self, request: {
            **complete(self, request), "created": len(self.stats["request_log"])})
    assert pf.produce(Tweaked(tmp_path, tweak, **kw), [name]) == {name: verdict}
    text = (tmp_path / f"{name}.json").read_text()
    receipt = json.loads(text)
    assert receipt["reasons"] == reasons and MODEL not in text
    if name == "P3":  # Unknown models are kept as 12-hex sha256s, never by name.
        unknown = [hashlib.sha256(MODEL.encode()).hexdigest()[:12]] if reasons == ["rc: unknown models"] else []
        assert receipt["facts"]["rc"]["unknown_model_sha256_12"] == unknown


@pytest.mark.parametrize("raises", [False, True])
def test_p3_context_length_stays_in_p3(tmp_path, monkeypatch, raises):
    """host.c is restored after P3, even when a turn raises midway, and P3's small context never reaches P4, P6 or P7."""
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))

    def boom(host, ctx, step):
        if raises and "pf-P3" in str(ctx["home"]):
            raise RuntimeError("turn failed")

    host = Tweaked(tmp_path, boom)
    saved = dict(host.c)
    assert pf.produce(host, list(pf.PREFLIGHTS)) == {**dict.fromkeys(pf.PREFLIGHTS, "PASS"),
                                                     "P3": "INCONCLUSIVE" if raises else "PASS"}
    assert host.c == saved and {n[:5] for n, _ in host.armed} == {"pf-P3", "pf-P4", "pf-P6", "pf-P7"}
    assert all((n.startswith("pf-P3")) == (length == 64000) for n, length in host.armed)
    assert json.loads((tmp_path / "P3.json").read_text())["reasons"] == (["RuntimeError"] if raises else [])


def test_a_produced_receipt_is_accepted_by_the_runner_for_the_same_run(tmp_path, monkeypatch):
    """Ruling R5-5 end to end: the producer records the plan, the snapshot pin, the instrument and the trees, and the
    runner's receipt check accepts what the producer wrote for the same declaration."""
    from bench.instruments.d5_clone import run as runner
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = FakeHost(tmp_path)
    assert pf.produce(host, ["P3", "P4"]) == {"P3": "PASS", "P4": "PASS"}
    receipt = json.loads((tmp_path / "P3.json").read_text())
    assert receipt["instrument_sha256"] == runner.instrument_identity()["instrument_sha256"]
    local = runner.LocalHost({**host.c, "preflights": {n: str(tmp_path / f"{n}.json") for n in ("P3", "P4")}}, tmp_path)
    plan = host.plan(None)
    assert [local.preflight(n, plan)["verdict"] for n in ("P3", "P4")] == ["PASS", "PASS"]
    assert receipt["config_sha256"] == runner.config_identity(host.c)
    other = runner.LocalHost({**local.c, "settings": {"changed": True}}, tmp_path)  # Another declared setting.
    assert [other.preflight(n, plan)["verdict"] for n in ("P3", "P4")] == ["MISSING", "MISSING"]


def test_p3_p6_p7_refuse_a_declaration_whose_rc_tree_is_not_the_previous_ga(tmp_path, monkeypatch):
    """Ruling R5-5: P3, P6 and P7 run the previous GA in both arms, and the runner reads their receipts that way. With
    another rc tree they run nothing and are INCONCLUSIVE; P4, bound to the run's two trees, still runs."""
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = Tweaked(tmp_path, lambda host, ctx, step: None)
    host.c["rc"] = {**host.c["rc"], "commit": "b" * 40}
    assert pf.produce(host, list(pf.PREFLIGHTS)) == {"P3": "INCONCLUSIVE", "P4": "PASS", "P6": "INCONCLUSIVE",
                                                     "P7": "INCONCLUSIVE"}
    assert {n[:5] for n, _ in host.armed} == {"pf-P4"}
    for name in ("P3", "P6", "P7"):
        assert json.loads((tmp_path / f"{name}.json").read_text())["reasons"] == [
            "this check runs the previous GA in both arms, and the rc tree is another tree"]


def test_no_preflight_runs_for_a_tree_without_a_declared_commit(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = Tweaked(tmp_path, lambda host, ctx, step: None)
    host.c["ga"] = {"lcm_tree": host.c["ga"]["lcm_tree"]}  # No commit: a path alone is no identity.
    assert pf.produce(host, list(pf.PREFLIGHTS)) == dict.fromkeys(pf.PREFLIGHTS, "INCONCLUSIVE")
    assert host.armed == []
    assert json.loads((tmp_path / "P4.json").read_text())["reasons"] == [
        "a tree has no declared 40-hex commit, so no receipt could bind it"]


@pytest.mark.parametrize("snap", [{"verdict": "FAIL", "reasons": ["a symlink in the snapshot"]}, {}, "raises"])
def test_no_clone_or_preflight_runs_unless_the_snapshot_verifies(tmp_path, monkeypatch, snap):
    """P2 before any clone: a snapshot that fails verification (a symlinked config.yaml, an unpinned file) is never
    cloned, planned or armed, so write_config never follows a copied link; every receipt is INCONCLUSIVE."""
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = Tweaked(tmp_path, lambda host, ctx, step: None)

    def snapshot():
        if snap == "raises":
            raise RuntimeError("verify failed")
        return snap
    host.snapshot, cloned = snapshot, []
    host.clone = lambda name: cloned.append(name)
    assert pf.produce(host, list(pf.PREFLIGHTS)) == dict.fromkeys(pf.PREFLIGHTS, "INCONCLUSIVE")
    assert cloned == [] and host.armed == [] and not (tmp_path / "plan-counts.json").exists()
    reason = json.loads((tmp_path / "P3.json").read_text())["reasons"][0]
    assert reason == ("no plan: RuntimeError" if snap == "raises" else "no plan: P2 snapshot did not pass")


@pytest.mark.parametrize("value, kept", [(12, 12), ("twelve chars", None), (True, None), (None, None), (1.5, None)])
def test_p4_never_copies_a_payload_size_field_as_text(value, kept):
    assert pf._count(value) == kept


def test_p4_writes_a_text_size_field_as_null_and_fails(tmp_path, monkeypatch):
    """Through P4 itself: a payload whose size fields hold text gives a FAIL whose receipt never carries that text."""
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    def tweak(h, c, s):
        return _payload(c, content_chars="PRIVATE-SIZE-TEXT", content_bytes="PRIVATE-SIZE-TEXT")
    assert pf.produce(Tweaked(tmp_path, tweak), ["P4"]) == {"P4": "FAIL"}
    text = (tmp_path / "P4.json").read_text()
    assert "PRIVATE-SIZE-TEXT" not in text
    assert {json.loads(text)["facts"][t][k] for t in ("rc", "ga") for k in ("content_chars", "content_bytes")} == {None}


def test_the_producer_entry_point_refuses_a_run_dir_inside_the_package(tmp_path):
    (tmp_path / "decl.json").write_text("{}")
    inside = pf.run.PACKAGE / f"refused-{tmp_path.name}"
    with pytest.raises(ValueError, match="outside the instrument package"):
        pf.main(["--config", str(tmp_path / "decl.json"), "--run-dir", str(inside), "--instrument-dir", str(tmp_path)])
    assert not inside.exists()


def test_no_receipt_binds_a_snapshot_pin_that_changed_while_p2_ran(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = Tweaked(tmp_path, lambda host, ctx, step: None)

    def snapshot():
        Path(host.c["snapshot_pin"]).write_text('{"changed": true}')
        return {"verdict": "PASS"}
    host.snapshot = snapshot
    assert pf.produce(host, ["P3", "P4"]) == {"P3": "INCONCLUSIVE", "P4": "INCONCLUSIVE"}
    assert host.armed == [] and json.loads((tmp_path / "P4.json").read_text())["reasons"] == [
        "no plan: the snapshot pin changed while P2 ran"]


def test_p3_p6_p7_run_the_previous_ga_configuration_in_both_arms(tmp_path, monkeypatch):
    """R5-5: with one declared commit but two extracted trees, P3, P6 and P7 still run the GA's own configuration in
    both arms (never the rc entry's path); P4 runs each arm's own; the declaration is unchanged afterwards."""
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    host = Tweaked(tmp_path, lambda host, ctx, step: None)
    host.c["rc"] = {**host.c["ga"], "lcm_tree": str(tmp_path / "rc-tree")}
    saved, seen = dict(host.c), []
    real_arm = host.arm

    def arm(tree, home, plan):
        seen.append((home.parent.name[:5], tree, host.c[tree]["lcm_tree"]))
        return real_arm(tree, home, plan)
    host.arm = arm
    pf.produce(host, list(pf.PREFLIGHTS))
    assert host.c == saved
    ga_tree, rc_tree = host.c["ga"]["lcm_tree"], host.c["rc"]["lcm_tree"]
    assert {path for name, _, path in seen if name != "pf-P4"} == {ga_tree}
    assert {(tree, path) for name, tree, path in seen if name == "pf-P4"} == {("rc", rc_tree), ("ga", ga_tree)}


def test_2647_p4_compares_unequal_spill_path_widths_after_masking(tmp_path):
    class SpillHost(FakeHost):
        def turn(self, ctx, step):
            super().turn(ctx, step)
            home = ctx["home"]
            (payload_path,) = (home / "lcm-large-outputs").iterdir()
            spill = "out-1-2-ab.log" if ctx["name"] == "rc" else "out-1762556000-42422-abcdef.log"
            content = json.dumps({"output": MARKER + "é", "full_output_path": f"{home}/tool_output/{spill}"}, ensure_ascii=False)
            if self.fault == "content" and ctx["name"] == "ga":
                content += "changed"
            digest = hashlib.sha256(content.encode()).hexdigest()[:12]
            name = f"20261107_000000_t_{digest}_1.json"
            payload_path.unlink()
            (home / "lcm-large-outputs" / name).write_text(json.dumps({
                "content": content, "content_chars": len(content), "content_bytes": len(content.encode())}))
            stub = (f'[Externalized tool output: tool=terminal; chars={len(content)}; bytes={len(content.encode())}; '
                    f'read it with lcm_expand(externalized_ref="{name}"); ref={name}]')
            if ctx["name"] == "ga":
                if self.fault == "stub-kind":
                    stub = stub.replace("Externalized", "GC'd externalized")
                elif self.fault == "stub-metadata":
                    stub = stub.replace("tool=terminal", "tool=other")
            for db, text in (("state.db", stub), ("lcm.db", content)):
                with closing(sqlite3.connect(home / db)) as conn, conn:
                    conn.execute("UPDATE messages SET content=? WHERE role='tool'", (text,))
            return True

    for fault in (None, "content", "stub-kind", "stub-metadata"):
        root = tmp_path / str(fault)
        root.mkdir()
        host = SpillHost(root, fault)
        fails, unsure, facts = pf.p4(host, host.plan(None))
        assert not unsure
        if fault is None:
            assert not fails
            for key in ("content_chars", "content_bytes", "content_sha256"):
                assert facts["rc"][key] != facts["ga"][key]  # Raw facts survive, without being compared.
            assert facts["rc"]["content_masked_sha256"] == facts["ga"]["content_masked_sha256"]
            assert facts["rc"]["stub_masked_sha256"] == facts["ga"]["stub_masked_sha256"]
        else:
            assert ("content_masked_sha256 differs between the trees" if fault == "content"
                    else "stub_masked_sha256 differs between the trees") in fails


@pytest.mark.parametrize("step, crash", [("constants", True), ("replay", True), ("constants", False)])
def test_2647_p6_records_the_failed_step_and_private_cause(tmp_path, monkeypatch, step, crash):
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=json.loads))
    monkeypatch.setattr(pf.run.sandbox, "wrap", lambda argv, **kw: argv)
    seen = []

    def fake_run(argv, **kwargs):
        current = "constants" if argv[-1] == "gen" else "replay"
        seen.append(current)
        out = Path(kwargs["cwd"])
        if current == "constants":
            (out / "constants.json").write_text(json.dumps({"range_new_rows": 0,
                                                           "bind_verdict": "OK" if crash else "NOT SHOWN"}))
        failed = current == step
        stderr = (f'Traceback (most recent call last):\n  File "/private/secret/driver.py", line 8, in main\n'
                  f'    secret({MARKER!r})\n  File "/private/secret/subprocess.py", line 9, in run\n'
                  f'    secret({MARKER!r})\nsubprocess.CalledProcessError: {MARKER}\n').encode() if failed and crash else b""
        return types.SimpleNamespace(returncode=int(failed), stderr=stderr, stdout=MARKER.encode())

    monkeypatch.setattr(pf.run.subprocess, "run", fake_run)
    host = FakeHost(tmp_path)
    config = {**host.c, "interpreter": "python3", "instrument_dir": "kit", "constants_argv": ["gen"],
              "replay_argv": ["replay"]}
    local = pf.run.LocalHost(config, tmp_path)
    host.replay = lambda ctx, s, t0, h8: local.replay(ctx, s, t0, h8)  # Same ctx carries only safe diagnostics.
    arm = host.arm
    host.arm = lambda *args: {**arm(*args), "port": 1}
    fails, unsure, facts = pf.p6(host, host.plan(None))
    assert not fails
    diag = facts["replay_diagnostics"]
    expected = {"exit_code": 1, "exception_type": "subprocess.CalledProcessError" if crash else None,
                "frame_file": "subprocess.py" if crash else None, "frame_function": "run" if crash else None}
    if not crash:
        expected["bind_verdict"] = "NOT SHOWN"
    assert diag[step] == expected
    if step == "constants":
        assert diag["replay"] is None
    else:
        assert diag["constants"]["exit_code"] == 0
    assert seen == (["constants"] if step == "constants" else ["constants", "replay"])
    assert any(step in reason and ("CalledProcessError" if crash else "NOT SHOWN") in reason for reason in unsure)
    text = json.dumps({"facts": facts, "reasons": unsure})
    assert MARKER not in text and "/private/secret" not in text


@pytest.mark.parametrize("tree", ["rc", "ga"])
@pytest.mark.parametrize("field", ["content_chars", "content_bytes", "chars", "bytes"])
def test_2647_r2_p4_checks_each_trees_own_sizes(tmp_path, tree, field):
    def corrupt(host, ctx, step):
        if ctx["name"] != tree:
            return
        if field.startswith("content_"):
            _payload(ctx, **{field: 2})
        else:
            _sql(ctx, "state.db", "UPDATE messages SET content = replace(content, ?, ?)",
                 f"{field}={len(MARKER)};", f"{field}=2;")

    host = Tweaked(tmp_path, corrupt)
    fails, unsure, _ = pf.p4(host, host.plan(None))
    assert not unsure
    reason = (f"{field} does not match its own content" if field.startswith("content_") else
              f"stub {field} does not match payload content_{field}")
    assert f"{tree}: {reason}" in fails
