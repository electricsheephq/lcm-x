"""Synthetic session generator adapted from the issue #2890 attack reproductions."""
import base64
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from textwrap import wrap

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import host_call

SESSION = "lineage"  # the oracle helpers read this session id
PAD = " the quick brown fox jumps over"
WINDOWS = [16, 24, 32, 48, 64, 96, 128, 200, 1024]
B64 = base64.b64encode(bytes((i * 131 + 7) % 256 for i in range(3600))).decode("ascii")


class Sim:
    def __init__(self, rng, root, window, cfg, edits, helpers):
        self.rng, self.root, self.window, self.cfg = rng, root, window, cfg
        self.edits = edits
        self.content_digest, self.covered_digests, self.nodes, self.stored = helpers
        self.view = [{"role": "system", "content": "You are a helpful agent."}] if rng.random() < 0.5 else []
        self.ts = 1_800_000_000.0
        self.n = 0
        self.events = []
        self.engine = None
        self.start()

    def start(self):
        cfg = LCMConfig(database_path=str(self.root / "lcm.db"), embeddings_enabled=False,
                        temporal_rollups_enabled=False, **self.cfg)
        self.engine = LCMEngine(config=cfg, hermes_home=str(self.root))
        self.engine.on_session_start(SESSION, platform="cli", context_length=self.window * 1024, conversation_id="conv")
        eng = self.engine
        consume, publish, compress = eng._leaf_consumption, eng._publish_leaf_node, eng.compress
        warn = eng._warn_unattributed_leaf

        def record_consumption(*args):
            result = consume(*args)
            self.selected_leaf |= bool(args[0] or result[3])
            self.pending = [self.content_digest(eng, row) for row in result[2]]
            return result

        def record_publication(node, *args, **kwargs):
            try:
                result = publish(node, *args, **kwargs)
            except Exception as error:
                if "source coverage is not contiguous" in str(error):
                    self.publication_error = True
                raise
            self.leaf_count += 1
            self.consumed.extend((digest, set(node.source_ids)) for digest in self.pending)
            return result

        def record_compression(*args, **kwargs):
            result = compress(*args, **kwargs)
            self.status = eng._last_compression_status
            self.noop = eng._last_compression_noop_reason or ""
            self.refused |= self.status in {"error", "held"} or any(
                marker in self.noop for marker in ("lacks raw store lineage", "contiguous stored prefix", "retained")
            )
            self.publication_error |= "publication could not prove" in self.noop
            return result

        def record_refusal(rows):
            self.lineage_refusal = True
            return warn(rows)

        eng._leaf_consumption = record_consumption
        eng._publish_leaf_node = record_publication
        eng.compress = record_compression
        eng._warn_unattributed_leaf = record_refusal
        if claim := getattr(eng, "_tool_result_leaf_claim", None):
            def record_claim(message, caller=None, **kwargs):
                store_id = claim(message, caller, **kwargs)
                if store_id is not None:
                    matches = eng._store.find_session_tool_rows(
                        SESSION, message["tool_call_id"], conversation_id="conv", tool_name="",
                    )
                    (self.weak_matches if len(matches) == 1 else self.group_matches).add(store_id)
                return store_id

            eng._tool_result_leaf_claim = record_claim

    def restart(self):
        self.engine.shutdown()
        self.start()
        self.events.append(("restart", self.n))

    def row(self, **fields):
        self.ts += 2
        r = {**fields, "timestamp": self.ts}
        self.view.append(r)
        return r

    def turn(self):
        rng = self.rng
        self.n += 1
        n = self.n
        self.row(role="user", content=f"question {n}" + PAD * rng.randint(1, 30))
        if rng.random() < 0.55:
            k = rng.choice([1, 1, 1, 2, 3])
            calls = []
            for i in range(k):
                if rng.random() < 0.08:
                    body = "\n".join(wrap(B64, 76))
                else:
                    body = f"contents {n}.{i}" + PAD * rng.randint(0, 40)
                calls.append({"id": f"call_{n}_{i}", "type": "function",
                              "function": {"name": "write_file",
                                           "arguments": json.dumps({"path": f"/srv/f{n}_{i}", "content": body})}})
            self.row(role="assistant", content="" if rng.random() < 0.5 else f"calling {n}", tool_calls=calls)
            for c in calls:
                style = rng.random()
                names = ({"name": "write_file", "tool_name": "write_file"} if style < 0.6 else
                         {"name": "write_file"} if style < 0.85 else {"tool_name": "write_file"})
                self.row(role="tool", tool_call_id=c["id"], content=json.dumps({"ok": True, "n": n}) + PAD * rng.randint(0, 60),
                         **names)
        self.row(role="assistant", content=f"reply {n}" + PAD * rng.randint(1, 120))
        self.engine.ingest(deepcopy(self.view))

    def mutate(self):
        rng = self.rng
        kind = rng.choice(["drop_toolname", "edit", "edit", "argsfmt", "restart", "drop_toolname"])
        if not self.edits:
            return
        body = [i for i, r in enumerate(self.view) if r.get("role") != "system"]
        if not body:
            return
        if kind == "restart":
            self.restart()
            return
        if kind == "drop_toolname":
            cand = [i for i in body if self.view[i]["role"] == "tool" and (self.view[i].get("name") or self.view[i].get("tool_name"))]
        elif kind == "edit":
            cand = [i for i in body if self.view[i]["role"] in ("user", "assistant") and isinstance(self.view[i].get("content"), str)
                    and "Summary (d" not in self.view[i]["content"] and "[LCM" not in self.view[i]["content"]]
        else:
            cand = [i for i in body if self.view[i].get("tool_calls")]
        if not cand:
            return
        i = rng.choice(cand)
        r = dict(self.view[i])
        if kind == "drop_toolname":
            r.pop("name", None)
            r.pop("tool_name", None)
        elif kind == "edit":
            r["content"] = (r.get("content") or "") + " (edited)"
        else:
            r["tool_calls"] = [dict(c, function=dict(c["function"], arguments=json.dumps(json.loads(c["function"]["arguments"]), indent=1)))
                               for c in r["tool_calls"]]
        self.view[i] = r
        self.events.append((kind, self.n, i, self.content_digest(self.engine, r)))
        if rng.random() < 0.5:  # host re-flushes the mutated history now (a copy may be stored after later rows)
            self.engine.ingest(deepcopy(self.view))
            self.events.append(("ingest", self.n))

    def compress(self, step):
        rng = self.rng
        before = json.loads(json.dumps(self.view))
        inject = None
        if rng.random() < 0.15 and len(before) > 4 and self.edits:  # a transient host row never ingested
            j = rng.randrange(1, len(before) - 1)
            inject = j
            before.insert(j, {"role": "assistant", "content": ""} if rng.random() < 0.5 else
                          {"role": "user", "content": "[host note] continue"})
        events, self.events = self.events, []
        tok0 = count_messages_tokens(before)
        kind = "forced" if rng.random() < 0.5 else "auto"
        record = self.run_pass(before, kind, tok0, step, [*events, ["pass", step, kind, inject]])
        if rng.random() < 0.7:
            self.engine.on_session_start(SESSION, old_session_id=SESSION, boundary_reason="compression",
                                         platform="cli", conversation_id="conv")
        return record

    def run_pass(self, before, kind, tok0, step, events):
        eng = self.engine
        self.pending, self.consumed, self.leaf_count = [], [], 0
        self.weak_matches, self.group_matches = set(), set()
        self.status, self.noop, self.refused, self.publication_error = "idle", "", False, False
        self.selected_leaf = self.lineage_refusal = False
        before_state = None
        if os.environ.get("LCMX_2890_SWEEP"):
            from tests.test_issue_2890_lineage_guard import _digest

            before_state = _digest([before, self.stored(eng), [asdict(node) for node in self.nodes(eng)]])
        if kind == "forced":
            returned = eng.compress(before, current_tokens=tok0, focus_topic=None, force=True)
        else:
            returned = host_call(eng, before).view
        stored = self.stored(eng)
        present = self.covered_digests(eng, stored, returned)
        by_digest = {}
        for row in stored:
            by_digest.setdefault(self.content_digest(eng, row), []).append(row["store_id"])
        missing = {digest: ids for digest, ids in by_digest.items() if digest not in present}
        raw = {self.content_digest(eng, row) for row in returned}
        named = {store_id for node in self.nodes(eng) if node.source_type == "messages" for store_id in node.source_ids}
        lineage_lost = {digest: by_digest[digest] for digest, _ in self.consumed
                        if digest in by_digest and digest not in raw and not named.intersection(by_digest[digest])}
        published = {store_id for _, sources in self.consumed for store_id in sources}
        needs_compaction = self.selected_leaf or tok0 >= eng.threshold_tokens
        refused = self.refused or (needs_compaction and not self.leaf_count and self.status in {"idle", "noop"})
        rec = dict(step=step, events=events,
                   status=self.status, needs_compaction=needs_compaction,
                   noop=self.noop, refused=refused, lineage_refusal=self.lineage_refusal,
                   publication_error=self.publication_error,
                   leaves=self.leaf_count, lineage_lost=lineage_lost, missing=missing, covered=sorted(present),
                   weak_sources=sorted(self.weak_matches & published), group_sources=sorted(self.group_matches & published),
                   view=[len(before), len(returned)], before_state=before_state)
        self.view = returned
        return rec


def simulate(seed, tmp_path, monkeypatch, summary, helpers, *, edits, drain=4, grow=2, stop_after=None):
    rng = random.Random(seed)
    outage = rng.random() < 0.15

    def summarize(*args, **kwargs):
        text = summary(*args, **kwargs)
        return None if outage else text

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summarize)
    clock = [1_800_000_000.0]

    def now():
        clock[0] += 0.5
        return clock[0]

    monkeypatch.setattr(time, "time", now)
    window = rng.choice(WINDOWS)
    cfg = dict(fresh_tail_count=rng.choice([2, 4, 6, 8, 12, 24]),
               leaf_chunk_tokens=rng.choice([400, 800, 2000, 6000, 24000]))
    if rng.random() < 0.3:
        cfg["large_output_externalization_enabled"] = False
    sim = Sim(rng, tmp_path, window, cfg, edits, helpers)
    passes = []
    try:
        for step in range(rng.randint(25, 45)):
            sim.turn()
            if rng.random() < 0.25:
                sim.mutate()
            if rng.random() < 0.3:
                passes.append(sim.compress(step))
                if len(passes) == stop_after:
                    return passes
        for extra in range(drain):
            if extra < grow:
                sim.turn()
            passes.append(sim.compress(f"drain{extra}"))
            if len(passes) == stop_after:
                return passes
    finally:
        sim.engine.shutdown()
    return passes


def save_session(seed, passes):
    if output := os.environ.get("LCMX_2890_SESSIONS"):
        (Path(output) / f"{seed}.json").write_text(json.dumps(passes) + "\n")
