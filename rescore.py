"""Rescore immutable #1749 timing snapshots; no generated corpus needed."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).parent
rows = json.loads((root/'SAMPLES.json').read_text())
summary = {}
for phase in ('base', 'product'):
    summary[phase] = {}
    for site in ('sessions', 'vector.chunk', 'vector.summary'):
        groups = [[r for r in rows if r['phase']==phase and r['site']==site and r['busy']==b]
                  for b in (False, True)]
        assert all(len(g)==5 for g in groups)
        entry = {}
        for label, group in zip(('idle', 'busy'), groups):
            values = [r['wall_s'] for r in group]
            entry[label] = dict(median=median(values), min=min(values), max=max(values), n=len(group),
                                rows=[r['rows'] for r in group], callbacks=[r['callbacks'] for r in group],
                                timeouts=sum(r['timeout'] for r in group),
                                exceeded_deadline=sum(r['exceeded_deadline'] for r in group))
        entry['busy_idle_ratio'] = entry['busy']['median']/entry['idle']['median']
        if phase=='base':
            entry['decision'] = ('not reproduced' if entry['busy_idle_ratio']<=2 and
                                  entry['busy']['median']<=.25 else 'fix')
        digests = {r['payload_sha256'] for g in groups for r in g if not r['timeout']}
        assert len(digests)==1
        entry['payload_sha256'] = digests.pop()
        summary[phase][site] = entry
for site in summary['base']:
    assert summary['base'][site]['payload_sha256']==summary['product'][site]['payload_sha256']
print(json.dumps(summary, indent=2, sort_keys=True))
