import gc,json,os,sys,time
from pathlib import Path
repo=Path(sys.argv[1]);sys.path.insert(0,str(repo))
import tests.conftest
from tests.phase0_hygiene import product_env_keys
for key in product_env_keys(repo):os.environ.pop(key,None)
from hermes_lcm.engine import LCMEngine
from hermes_lcm.config import LCMConfig
from tests.test_host_uid_emit_carry import _ContainsPattern
home=Path(sys.argv[2]);home.mkdir(exist_ok=True)
config=LCMConfig(database_path=":memory:",embeddings_enabled=False,temporal_rollups_enabled=False,
 empty_lifecycle_gc_enabled=False,large_output_externalization_enabled=False,
 large_output_externalization_path=str(home/"payloads"))
for line in sys.stdin:
 q=json.loads(line); n=q["n"]; sample=q["sample"]
 e=LCMEngine(config=config,hermes_home=str(home));e.on_session_start("bench1822",platform="cli",context_length=1000000)
 if q["pattern"]:e._compiled_ignore_message_patterns=[_ContainsPattern("IGNORE THIS ROW")]
 messages=[dict(role="user" if i%2==0 else "assistant",content=f"Message {i}: ordinary migration content.",timestamp=float(100+i)) for i in range(n-2 if q["path"]=="rewind" else n)]
 e._ingest_messages(messages)
 if q["path"]=="rewind":
  messages[0]["content"]="Revised migration prompt."
  messages.extend([dict(role="user",content="now deploy it",timestamp=float(n+200)),dict(role="assistant",content="Deploying.",timestamp=float(n+201))])
 assert len(messages)==n
 gc.collect();gc.disable()
 start=time.perf_counter_ns();e._ingest_messages(messages);elapsed=time.perf_counter_ns()-start
 gc.enable();e.shutdown()
 print(json.dumps(dict(ns=elapsed)),flush=True)
