"""#2856 PR2 counting pins on the probe's pass shape (synthetic; counts only, no timing).

A compacted session (300 rows, one pass), grown to N=1,800 rows in its child session, three ordinary turns, then one
automatic pass at half the list's tokens (a stand-in summariser that honours its timeout). Counted on whichever tree
runs it (it imports no PR2 symbol): C3 token-cache misses per ordinary turn, C5 map builds that page rows from SQL,
metadata reads by key per pass (recorded only), and every ``write_generation`` read with the memo slot that made it
(the price of a hit). Evidence goes to LCMX_FIXEDWORK_EVIDENCE when set.
"""
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import hermes_lcm.engine as lcm_engine
import hermes_lcm.reconcile as reconcile
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from tests.invariants.test_fixedwork_identity_memo_count import Rows, one_turn

N, S0, TURNS = 1_800, 300, 3
MEMO = hasattr(reconcile, "_store_generation_memo")


def _grow(rows, gen, n):
    while len(rows) < n:
        rows += gen.exchange()
    return rows


def _summariser(**kwargs):
    assert kwargs.get("timeout") is None or kwargs["timeout"] > 0
    return "Earlier turns: the user asked questions and tools ran.\nExpand for details about: turns", 1


def _compact(engine, history, fraction):
    tokens_now = tokens.count_messages_tokens(history)
    keep, engine.threshold_tokens = engine.threshold_tokens, int(tokens_now * fraction)
    try:
        return engine.compress(deepcopy(history), current_tokens=tokens_now)
    finally:
        engine.threshold_tokens = keep


class Counts:
    def __init__(self):
        self.c = Counter()

    def install(self, monkeypatch):
        c, depth = self.c, [0]
        owner = next(k for k in LCMEngine.__mro__ if "_get_store_id_map_for_messages" in k.__dict__)
        build = owner.__dict__["_get_store_id_map_for_messages"]

        def counted_build(self, *a, **k):
            c["map_builds"] += 1
            depth.append(0)
            try:
                return build(self, *a, **k)
            finally:
                c["map_builds_paging_sql"] += depth.pop() > 0
        monkeypatch.setattr(owner, "_get_store_id_map_for_messages", counted_build)
        for name in ("get_session_messages_after", "get_range"):
            original = MessageStore.__dict__[name]

            def paging(self, *a, original=original, **k):
                caller = sys._getframe(1).f_code.co_name
                if caller in ("_page_store_id_map_candidates", "_get_store_id_map_for_messages") and len(depth) > 1:
                    depth[-1] += 1
                    c["map_paging_selects"] += 1
                return original(self, *a, **k)
            monkeypatch.setattr(MessageStore, name, paging)
        read = MessageStore.__dict__["read_metadata_json"]

        def read_metadata(self, key):
            kind = str(key).split(":", 1)[0]
            c["metadata_read:" + kind] += 1
            value = read(self, key)
            c["metadata_read_found:" + kind] += value is not None
            return value
        monkeypatch.setattr(MessageStore, "read_metadata_json", read_metadata)
        generation = MessageStore.__dict__["write_generation"]

        def write_generation(self):
            frame = sys._getframe(1)
            slot = frame.f_locals.get("slot") if frame.f_code.co_name == "_store_generation_memo" else None
            c[f"write_generation:{slot or frame.f_code.co_name}"] += 1
            return generation(self)
        monkeypatch.setattr(MessageStore, "write_generation", write_generation)
        if MEMO:
            memo = reconcile._store_generation_memo

            def counted_memo(owner, slot, key, load, copy):
                ran = []
                c[f"memo_calls:{slot}"] += 1
                out = memo(owner, slot, key, lambda: ran.append(1) or load(), copy)
                c[f"memo_loads:{slot}"] += bool(ran)
                return out
            monkeypatch.setattr(reconcile, "_store_generation_memo", counted_memo)

    def take(self):
        out = dict(self.c)
        self.c.clear()
        return out


def _state_db(path):
    import sqlite3
    conn = sqlite3.connect(path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, NULL, NULL)",
                     [("S", None, "compression"), ("S2", "S", None)])
    conn.commit()
    conn.close()


def test_c3_c5_g_counts_on_the_pass_shape(tmp_path, monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summariser)
    _state_db(tmp_path)
    counts = Counts()
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), foreground_soft_seconds=1e6,
                                        embeddings_enabled=False, temporal_rollups_enabled=False),
                       hermes_home=str(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=50_000_000, conversation_id="conv")
    record = dict(tree="candidate" if MEMO else "base", n=N, turns=[])
    try:
        gen = Rows()
        h0 = _grow([{"role": "system", "content": "[r0] system prompt" + " lorem" * 100}], gen, S0)
        engine.should_compress_preflight(deepcopy(h0))
        engine.ingest(deepcopy(h0))
        engine.note_turn_complete()
        out = _compact(engine, h0, 0.25)
        engine.on_session_end("S", h0)
        engine.on_session_start("S2", platform="telegram", context_length=50_000_000, conversation_id="conv",
                                boundary_reason="compression", old_session_id="S")
        history = _grow(list(out), gen, N)
        engine.should_compress_preflight(deepcopy(history))
        engine.ingest(deepcopy(history))
        engine.note_turn_complete()
        counts.install(monkeypatch)
        for _ in range(TURNS):
            before = tokens._count_tokens_cached.cache_info()
            old = len(history)
            history, new = one_turn(engine, history, gen)
            after = tokens._count_tokens_cached.cache_info()
            strings = {s for row in history[old:] for s in (
                str(row.get("content") or ""),
                *(str(call["function"].get(field) or "") for call in row.get("tool_calls") or [] for field in ("name", "arguments")))}
            record["turns"].append(dict(rows=len(history), new_rows=new, new_strings=len(strings),
                                        token_hits=after.hits - before.hits, token_misses=after.misses - before.misses,
                                        **counts.take()))
        returned = _compact(engine, history, 0.5)
        record["pass"] = dict(rows_in=len(history), rows_out=len(returned), status=engine._last_compression_status,
                              **counts.take())
    finally:
        engine.shutdown()
    out_dir = os.environ.get("LCMX_FIXEDWORK_EVIDENCE")
    if out_dir:
        Path(out_dir, f"pr2_counts_{record['tree']}.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    p = record["pass"]
    assert p["status"] == "compacted", p
    assert p["map_builds"] >= 20, p  # the shape rebuilds the map many times per pass
    for turn in record["turns"][1:]:  # C3: the second and later warm walks
        if MEMO:
            assert turn["token_misses"] <= turn["new_strings"] + 16, record["turns"]
        else:  # discriminating: the 2,048-entry cache misses about once per string of the walk
            assert turn["token_misses"] > 10 * (turn["new_strings"] + 16), record["turns"]
    if MEMO:  # C5: at most one load per write generation of the pass
        assert p["map_builds_paging_sql"] <= p["map_builds"] // 2, p
    else:
        assert p["map_builds_paging_sql"] == p["map_builds"], p
