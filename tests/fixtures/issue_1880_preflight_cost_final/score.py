import json
from pathlib import Path
import statistics
rows=[json.loads(x) for x in Path(__file__).with_name("pairs.jsonl").read_text().splitlines()]
for count in (10000,100000):
    pairs=[r for r in rows if r["rows"]==count]
    a=statistics.median(r["baseline_ms"] for r in pairs)
    b=statistics.median(r["candidate_ms"] for r in pairs)
    delta=statistics.median(r["candidate_ms"]-r["baseline_ms"] for r in pairs)
    limit=max(a*.1,.05)
    print(json.dumps(dict(rows=count,n=len(pairs),baseline_ms=a,candidate_ms=b,paired_delta_ms=delta,limit_ms=limit,passed=b-a<=limit and delta<=limit)))
