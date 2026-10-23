"""Reuse the real-host probes for the approved attribution shapes; no live home."""
import json
import os
import subprocess

import pytest

from tests.test_real_plugin_manager_compaction import _PROBE as COMMIT_PROBE
from tests.test_real_turn_loop_acp_override import HERMES, _PROBE as TURN_PROBE

pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")


def run_probe(tmp_path, source, **extra):
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm-x").symlink_to(os.path.dirname(os.path.dirname(__file__)))
    (home / "config.yaml").write_text("plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n")
    python, src = HERMES
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "HERMES_HOME": str(home),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "OPENROUTER_API_KEY": "test-key", "PROBE_IN_PLACE": "1",
           "PROBE_TRAILING": "0", "PROBE_TURNS": "1", "PROBE_REPEAT": "40", "PROBE_REAL_USAGE": "1",
           "LCM_FRESH_TAIL_COUNT": "6", "LCM_LEAF_CHUNK_TOKENS": "12000", **extra}
    completed = subprocess.run([python, "-c", source], cwd=src or None, env=env,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.mark.parametrize("count", [245, 1400])
def test_real_commit_attribution(tmp_path, count):
    source = COMMIT_PROBE.split("host, t, statuses, sids, repairs, adopted =")[0]
    source = "import socket\nsocket.socket.connect = lambda *a: (_ for _ in ()).throw(OSError('offline'))\n" + source
    source += f'''
engine._config.embeddings_enabled = False
engine._config.temporal_rollups_enabled = False
host = [{{"role": "user" if i % 2 == 0 else "assistant", "content": f"Synthetic row {{i}}: " +
         "alpha beta gamma delta " * 40, "timestamp": 1760000000.0 + i}} for i in range({count})]
engine.ingest(host)
token = engine._diagnostic_engine_token
calls = []
original_end = engine._finalize_foreground_session_end_unlocked
def end(sid, messages):
    assert engine._diagnostic_engine_token == token
    assert engine._compress_commit_proof is not None
    original_ingest = engine._ingest_messages
    def ingest(*a, **k):
        calls.append(1)
        return original_ingest(*a, **k)
    engine._ingest_messages = ingest
    try:
        return original_end(sid, messages)
    finally:
        engine._ingest_messages = original_ingest
engine._finalize_foreground_session_end_unlocked = end
compressed, _ = agent._compress_context(host, "sys", approx_tokens=100000, force=True)
lines = [line for line in buf.getvalue().splitlines() if "LCM attribution phase=session_end" in line]
assert len(compressed) < len(host)
assert len(lines) == 1
fields = dict(item.split("=", 1) for item in lines[0].split() if "=" in item)
print(json.dumps({{"fields": fields, "token": token, "calls": len(calls),
                  "commit_logged": buf.getvalue().count("as a compaction commit"),
                  "stored": engine._store.get_session_count("S0")}}))
engine.shutdown()
'''
    result = run_probe(tmp_path, source)
    assert result["commit_logged"] == 1
    assert result["fields"]["reason"] == "fast"
    assert result["fields"]["engine_token"] == str(result["token"])
    assert result["calls"] == 0
    assert result["stored"] == count


def test_cold_23_replay_ordering(tmp_path):
    seed = '''
seed = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Synthetic seed {i}: " +
         "alpha beta gamma delta " * 40, "timestamp": 1760000000.0 + i} for i in range(1400)]
engine._config.embeddings_enabled = False
engine._config.temporal_rollups_enabled = False
engine.ingest(seed)
import copy
config, cls, cid = copy.deepcopy(engine._config), type(engine), engine.current_conversation_id
engine.shutdown()
engine = cls(config=config, hermes_home=str(home))
engine.on_session_start("S0", platform="acp", conversation_id=cid)
agent.context_compressor = engine
history = copy.deepcopy(seed[-23:])
events = []
class Observe(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if "recognised 23 replayed rows" in message: events.append("replay")
        if "LCM attribution phase=preflight" in message: events.append("preflight")
        if message.startswith("Turn ended:"): events.append("Turn ended")
        if "LCM attribution phase=post_llm_call" in message: events.append("post_llm_call")
logging.getLogger().addHandler(Observe())
sys.modules[type(engine).__module__].logger.setLevel(logging.DEBUG)
'''
    source = TURN_PROBE.replace("history = []", seed)
    # This timestamp/order marker runs INSIDE the real provider callback, never during response construction.
    source = source.replace('sent = sum(len(str(m.get("content") or ""))',
                            'events.append("API callback")\n            sent = sum(len(str(m.get("content") or ""))')
    source = source.split('db = sqlite3.connect(str(home / "lcm.db"))')[0]
    source += '''
rows = engine._store.get_session_messages("S0")
print(json.dumps({"events": events, "stored": len(rows), "distinct": len({r["content"] for r in rows}),
                  "fresh_user": sum(r["role"] == "user" and "[T01] user turn 1:" in r["content"] for r in rows),
                  "fresh_reply": sum(r["content"] == "reply to T01: noted item 1." for r in rows)}))
engine.shutdown()
'''
    result = run_probe(tmp_path, source)
    events = result["events"]
    assert events.index("replay") < events.index("preflight") < events.index("API callback")
    assert events.index("API callback") < events.index("Turn ended") < events.index("post_llm_call")
    assert result["stored"] == result["distinct"] == 1402
    assert result["fresh_user"] == result["fresh_reply"] == 1
