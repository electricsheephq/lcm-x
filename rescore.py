import json, statistics
from pathlib import Path
s=json.loads((Path(__file__).parent/'SUMMARY.json').read_text())
for name,row in s['steady_cost'].items():
    base=statistics.median(row['samples_ms']['base'])
    fixed=statistics.median(row['samples_ms']['head'])
    assert fixed <= max(base*1.10,base+0.05), (name,base,fixed)
for metric,samples in [('pass_ms_median','pass_samples_ms'),('status_ms_median','status_samples_ms')]:
    base=statistics.median(s['liveness_cost']['base'][samples])
    fixed=statistics.median(s['liveness_cost']['fixed'][samples])
    assert fixed <= max(base*1.5,base+1), (metric,base,fixed)
for row in s['liveness'].values():
    assert row['marker_equals_cursor_first_at_call'] is not None
    assert all(row[k]==0 for k in ['width_violations','ro_violations','final_oracle_missing','calls_holes_missing_but_remaining_zero','calls_holes_missing_but_status_zero'])
for row in s['status'].values():
    assert all(row[k]==0 for k in ['steady_pass_scans','steady_status_scans','steady_ro_violations','oracle_missing'])
print('PASS: preserved timing samples, liveness and read-only summaries')
print('Oracle controls require the complete raw bundle; the accepted ruling is recorded in SUMMARY.json.')
