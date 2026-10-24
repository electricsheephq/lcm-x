"""Re-score the stopped P1 paired-prefix diagnostic; never infer latency gates."""
import json
import math
from pathlib import Path
root = Path(__file__).parent
rows = [json.loads(line) for line in (root/'partial-recalls.jsonl').read_text().splitlines()]
arms = {arm: [row for row in rows if row['arm'] == arm] for arm in ('base', 'candidate')}
assert len(arms['base']) == 200 and len(arms['candidate']) == 36
matched = 0
differences = []
for index, (old, new) in enumerate(zip(arms['base'], arms['candidate'])):
    assert (old['i'], old['query_class'], old['query']) == (new['i'], new['query_class'], new['query'])
    def ids(row):
        return [(h['kind'], h['session_id'], h['node_id'], h['store_id'], h['chunk_span']) for h in row['hits']]
    same_ids = ids(old) == ids(new)
    deltas = [abs(x['score']-y['score']) for x, y in zip(old['hits'], new['hits'])] if same_ids else []
    same_scores = same_ids and all(math.isfinite(d) and d <= 4*2**-23 for d in deltas)
    if same_ids and same_scores:
        matched += 1
    else:
        differences.append(dict(paired_row=index, i=new['i'], query_class=new['query_class'],
            base_timeout=old['timeout'], candidate_timeout=new['timeout'],
            base_coverage=old['coverage'], candidate_coverage=new['coverage']))
summary = dict(status='EXACTNESS_STOP', base_completed=len(arms['base']), candidate_completed=len(arms['candidate']),
    paired_queries=len(arms['candidate']), matched=matched, differences=differences,
    C1='NOT RUN', C2='NOT RUN; not PENDING-QUIET', C3='NOT RUN', memory='INCOMPLETE',
    score_tolerance=4*2**-23, representative_failure='E-B1 query index3/classa: base full chunks, candidate timeout/chunk none',
    caveat='Partial preflight under load; no completed-cell latency comparison or accuracy estimate.')
print(json.dumps(summary, indent=2))
