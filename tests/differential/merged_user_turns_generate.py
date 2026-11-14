"""Freeze BASELINE_REF only, then deterministically derive #2707 declarations.

Run as LCM_DIFFERENTIAL=1 python -m tests.differential.merged_user_turns_generate SCRATCH.
No candidate tree or candidate output is read. SCRATCH/evidence.json retains the
host, committed-owner, storage, fault and protection evidence used by eligibility.
"""
from dataclasses import asdict
import json
from pathlib import Path
import pprint
import sys

import tests.conftest  # noqa: F401 - standalone package aliases
from . import paired
from .merged_user_turns_expected import EXPECTED
from .record_contract import declaration_stages, declarations
from .test_paired_merged_user_turns import CASES, _execute_case


def generate(directory):
    tree = paired.materialize(paired.BASELINE_REF, directory / 'baseline')
    versions = {name: paired._load(tree, '_extras_baseline_' + name) for name in ('main', 'candidate')}
    evidence, declared, dispositions, retrieval, absent = {}, {}, {}, {}, {}
    for case in CASES:
        _, traces = _execute_case(case, directory / str(case), versions)
        trace = traces['main']
        stages = declaration_stages(trace)
        declared[case] = declarations(case, stages, paired.BASELINE_REF)
        absent[case] = [(e['capture_view'], e['predecessor_ordinal'], e['expected']['content'],
                         'injected persistent writer fault') for e in declared[case] if e['due_stage'] is None]
        retrieval[case] = dict(kind='retrieval-missing', label='#2707 proven',
                               values=[s['retrieval_missing'] for s in stages])
        evidence[case] = dict(stages=stages, faults=[asdict(f) for f in trace.faults])
        dispositions[case] = []
        for stage, kind, value, view, count, label in EXPECTED.get(case, ()):
            if label != '#2707 proven':
                continue
            matches = [e for e in declared[case] if e['capture_view'] == view and e['stage'] <= stage
                       and e['expected']['content'] == value]
            capture = next((c for c in stages[stage - 1]['captures'] if c['view'] == view), None)
            reason = 'declared' if matches else ('ambiguous-watch' if not capture else
                     'already-stored' if any(i in stages[stage - 1]['committed_before'] for i in capture['ids'][:-1])
                     else 'copied-head' if not capture['head_present'] else
                     'already-merged-input' if capture['prefix'] else 'other:no-committed-owner')
            dispositions[case].append((stage, kind, value, view, count, reason))
    output = Path(__file__).with_name('merged_user_turns_extras.py')
    (directory / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    output.write_text('"""Generated declarations; regenerate with merged_user_turns_generate.py.\n\n'
                      'Audit stages and faults go to SCRATCH/evidence.json outside the repository.\n"""\n'
                      + '\n'.join(name + ' = ' + pprint.pformat(value, width=120, sort_dicts=False)
                                  for name, value in [('DECLARED', declared), ('DISPOSITIONS', dispositions),
                                                      ('RETRIEVAL_MISSING', retrieval), ('FAULT_ABSENT', absent)]) + '\n')
    print(json.dumps(dict(records=sum(map(len, declared.values())), pins=dict(__import__('collections').Counter(
        {reason: sum(row[4] for rows in dispositions.values() for row in rows if row[5] == reason)
         for reason in {row[5] for rows in dispositions.values() for row in rows}})))))


if __name__ == '__main__':
    generate(Path(sys.argv[1]))
