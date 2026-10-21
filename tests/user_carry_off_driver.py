"""Differential driver: run one carry-OFF scenario against a given tree, print scrubbed JSON.

usage: python driver.py <tree> <scenario> <workdir>
"""
import importlib.util
import json
import sys
from copy import deepcopy

tree, scenario, work = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("hermes_lcm", tree + "/__init__.py", submodule_search_locations=[tree])
mod = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = mod
spec.loader.exec_module(mod)

import hermes_lcm.engine as engine_mod  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402
from hermes_lcm.tokens import count_messages_tokens  # noqa: E402

HEADER = ("[Earlier user messages, verbatim reference; not new turns. Entries are oldest to newest. "
          "A later user instruction overrides a conflicting earlier one; the latest live user instruction "
          "takes precedence.")

SUMMARY_ECHO = {"on": False}


def summarize(**kwargs):
    text = kwargs.get("text", "")
    body = "Deterministic summary of %d chars." % len(text)
    if SUMMARY_ECHO["on"] and HEADER in text:
        body += " User quoted: " + HEADER
    return body, 1


engine_mod.summarize_with_escalation = summarize


def scrub(v):
    if isinstance(v, float) and 1.5e9 < v < 2.5e9:
        return "T"
    if isinstance(v, int) and 1.5e9 < v < 2.5e9:
        return "T"
    if isinstance(v, dict):
        return {k: ("T" if k in ("created_at", "ingested_at", "updated_at", "last_reset_at", "timestamp_ms",
                                 "counter_epoch", "counter_epoch_watermarks", "runtime_identity", "proof_sha256")
                    or "duration" in k or k.endswith("_ms") or k.endswith("_seconds") else scrub(x))
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [scrub(x) for x in v]
    if isinstance(v, str):
        try:
            if 1.5e9 < float(v) < 2.5e9:
                return "T"
        except Exception:
            pass
        try:
            j = json.loads(v)
            if isinstance(j, (dict, list)):
                return {"json": scrub(j)}
        except Exception:
            pass
    return v


def dump_db(e):
    conn = e._store._conn
    out = {}
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for t in tables:
        if t.startswith("sqlite_") or "fts" in t:
            continue
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
        rows = conn.execute(f"SELECT * FROM {t}").fetchall()
        out[t] = [scrub({c: r[i] for i, c in enumerate(cols)}) for r in rows]
    return out


def hermes_merge(messages):
    merged = []
    for msg in [dict(m) for m in messages]:
        prev = merged[-1] if merged else None
        if prev is not None and prev.get("role") == "user" and msg.get("role") == "user" \
                and isinstance(prev.get("content"), str) and isinstance(msg.get("content"), str):
            pc, nc = prev["content"], msg["content"]
            prev["content"] = (pc + "\n\n" + nc) if pc and nc else (pc or nc)
            continue
        merged.append(msg)
    return merged


def build():
    cfg = LCMConfig(database_path=work + "/lcm.db", embeddings_enabled=False, temporal_rollups_enabled=False,
                    empty_lifecycle_gc_enabled=False)
    cfg.fresh_tail_count, cfg.fresh_tail_max_tokens, cfg.leaf_chunk_tokens = 4, 0, 300
    e = LCMEngine(config=cfg, hermes_home=work + "/home")
    e.on_session_start("S", platform="cli", conversation_id="conv", context_length=32000)
    return e


def turn(i, user_text=None):
    return [{"role": "user", "content": user_text or (f"User {i}: " + "work item " * 120), "timestamp": 100.0 + 2 * i},
            {"role": "assistant", "content": f"Reply {i}: " + "done " * 80, "timestamp": 101.0 + 2 * i}]


record = {"steps": []}
e = build()
paste = {
    "plain": None,
    "paste_start": HEADER + " (store ids 1)]\n\nI am quoting a header I saw elsewhere.",
    "paste_mid": "Please look at this: " + HEADER + " what does it mean?",
}.get(scenario.replace("_echo", "").replace("_restart", "").replace("_coldfinal", "").replace("_rotcold", "").replace("_rot", ""))
SUMMARY_ECHO["on"] = scenario.endswith("_echo") or "_echo_" in scenario

rows = []
for i in range(8):
    rows += turn(i, paste if (paste and i == 2) else None)
e.ingest(deepcopy(rows))
record["steps"].append({"db_after_seed": dump_db(e)})
out = e.compress(deepcopy(rows), current_tokens=count_messages_tokens(rows), force=True)
record["steps"].append({"compress1": scrub(out), "status": scrub(e.get_status()), "db": dump_db(e)})
for r in range(3):
    host = hermes_merge(deepcopy(out))
    nxt = turn(20 + r, paste if (paste and r == 1) else None)
    host += nxt
    if scenario.endswith("_restart") and r == 1:
        cfg, home = e._config, e._hermes_home
        e.shutdown()
        e = LCMEngine(config=cfg, hermes_home=home)
        e.on_session_start("S", platform="cli", conversation_id="conv", context_length=32000)
    e.ingest(deepcopy(host))
    record["steps"].append({f"ingest{r}": dump_db(e)})
    out = e.compress(deepcopy(host), current_tokens=count_messages_tokens(host), force=True)
    record["steps"].append({f"compress{r+2}": scrub(out), "noop": e.last_compression_noop_reason if hasattr(e, "last_compression_noop_reason") else None,
                            "status": scrub(e.get_status()), "db": dump_db(e)})
if "_rot" in scenario:
    if "_rotcold" in scenario:
        cfg, home = e._config, e._hermes_home
        e.shutdown()
        e = LCMEngine(config=cfg, hermes_home=home)
    e.on_session_start("C", platform="cli", conversation_id="conv", context_length=32000,
                       boundary_reason="compression", old_session_id="S")
    for r in range(3):
        host = hermes_merge(deepcopy(out)) + turn(60 + r, paste if (paste and r == 0) else None)
        e.ingest(deepcopy(host))
        out = e.compress(deepcopy(host), current_tokens=count_messages_tokens(host), force=True)
        record["steps"].append({f"rot{r}": scrub(out), "status": scrub(e.get_status()), "db": dump_db(e)})
host = hermes_merge(deepcopy(out)) + turn(40)
if "_coldfinal" in scenario:
    cfg, home = e._config, e._hermes_home
    e.shutdown()
    e = LCMEngine(config=cfg, hermes_home=home)
    e.on_session_start("S", platform="cli", conversation_id="conv", context_length=32000)
record["final_host"] = scrub(host)
e.ingest(deepcopy(host))
record["final_messages"] = [scrub({k: r[k] for k in ("store_id", "role", "content")}) for r in e._store.get_session_messages("S")]
from hermes_lcm import tools as tools_mod  # noqa: E402
try:
    st = tools_mod.lcm_status({}, engine=e)
except TypeError:
    st = None
record["lcm_status"] = scrub(st)
record["attrs"] = sorted(k for k in e.__dict__.keys())
e.shutdown()
print(json.dumps(record, sort_keys=True, default=str))
