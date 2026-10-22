import json, statistics
from pathlib import Path
root=Path(__file__).parent
rows=[json.loads(line) for line in (root/"cost-200.jsonl").read_text().splitlines()]
assert len(rows)==4800
summary=[]
for n in (200,2000):
 for path,pattern in (("rewind",True),("rewind",False),("steady",True),("steady",False)):
  arms={arm:[r["ns"]/1e6 for r in rows if (r["n"],r["path"],r["pattern"],r["arm"])==(n,path,pattern,arm)] for arm in ("base","candidate","pre1628")}
  assert all(len(v)==200 for v in arms.values())
  med={arm:statistics.median(v) for arm,v in arms.items()}
  if path=="rewind" and pattern:limit=med["pre1628"]+max(.1*med["pre1628"],.5)
  elif path=="rewind":limit=med["base"]
  else:limit=med["base"]+max(.1*med["base"],.05)
  entry=dict(n=n,path=path,pattern=pattern,median_ms=med,limit_ms=limit,pass_gate=med["candidate"]<=limit,
    relative_to_base_pct=100*(med["candidate"]/med["base"]-1),relative_to_pre1628_pct=100*(med["candidate"]/med["pre1628"]-1),
    iqr_ms={arm:statistics.quantiles(v,n=4)[2]-statistics.quantiles(v,n=4)[0] for arm,v in arms.items()})
  summary.append(entry)
(root/"cost-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
for entry in summary:print(json.dumps(entry))
assert all(r["pass_gate"] for r in summary), "cost gate failed"
