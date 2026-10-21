"""Score fixed snapshots; no store, host or product imports required."""
import json
from pathlib import Path
from statistics import median
root=Path(__file__).parent
rows=json.loads((root/'SAMPLES.json').read_text())
summary={}
for phase in ('base','product'):
    summary[phase]={}
    for site in ('sessions','vector.chunk','vector.summary'):
        entry={}
        for busy,label in ((False,'idle'),(True,'busy')):
            group=[r for r in rows if r['phase']==phase and r['site']==site and r['busy']==busy]
            assert len(group)==5
            values=[r['wall_s'] for r in group]
            entry[label]=dict(median=median(values),min=min(values),max=max(values),n=5,
                rows=[r['rows'] for r in group],callbacks=[r['callbacks'] for r in group],
                timeouts=sum(r['timeout'] for r in group),exceeded_deadline=sum(r['exceeded_deadline'] for r in group))
        entry['busy_idle_ratio']=entry['busy']['median']/entry['idle']['median']
        digests={r['payload_sha256'] for r in rows if r['phase']==phase and r['site']==site and not r['timeout']}
        assert len(digests)==1
        entry['payload_sha256']=digests.pop()
        summary[phase][site]=entry
for site in summary['base']:
    assert summary['base'][site]['payload_sha256']==summary['product'][site]['payload_sha256']
limits=json.loads((root/'LIMITS.json').read_text())
summary['limits']={}
for shape in ('250k-48','1m-128','1m-384'):
    entry={}
    for mode in ('speed','memory','deadline'):
        groups={phase:[r for r in limits if r['shape']==shape and r['mode']==mode and r['phase']==phase] for phase in ('base','product')}
        if not groups['base']:continue
        if mode=='speed':
            assert all(len(g)==4 for g in groups.values())
            values={p:median(r['runs'][0]['elapsed'] for r in g) for p,g in groups.items()}
            ratio=values['product']/values['base']
            entry[mode]=dict(values,ratio=ratio,passed=ratio<=1.1)
        elif mode=='memory':
            values={p:g[0]['delta_mb'] for p,g in groups.items()}
            ratio=values['product']/values['base']
            entry[mode]=dict(values,ratio=ratio,passed=ratio<=1.25)
        else:
            values={p:sum(r['runs'][0]['result']=='ok' and r['runs'][0]['n']==1000000 for r in g) for p,g in groups.items()}
            entry[mode]=dict(values,passed=values['base']==values['product']==4)
    summary['limits'][shape]=entry
print(json.dumps(summary,indent=2,sort_keys=True))
