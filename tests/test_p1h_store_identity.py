"""#2293: every table stays identical to the archived base, including ledger rows.

Set LCMX_P1H_BASE_TREE to a read-only archive of the pinned comparison tree.
Each tree runs in its own process, sequentially, sharing only synthetic host identity.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_host_uid_shadow import _state_db


def test_all_table_identity(tmp_path):
    base = os.environ.get('LCMX_P1H_BASE_TREE')
    if not base:
        pytest.skip('requires explicit archived comparison tree')
    tree = Path(__file__).resolve().parents[1]
    host = tmp_path / 'synthetic-host'
    host.mkdir()
    _state_db(host, [('S', None, None)])
    results = []
    for name, source in (('base', Path(base)), ('candidate', tree)):
        output = tmp_path / (name + '.json')
        run = subprocess.run([sys.executable, str(tree / 'tests/p1h_store_dump.py'), str(source),
                              str(tmp_path / name), str(host), str(output)],
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stdout + run.stderr
        results.append(json.loads(output.read_text()))
        if evidence := os.environ.get('LCMX_P1H_IDENTITY_EVIDENCE'):
            Path(evidence, 'store-identity-' + name + '.json').write_bytes(output.read_bytes())
    assert results[0] == results[1]
    # Red control: even one surplus row in a ledger table breaks the comparison.
    changed = json.loads(json.dumps(results[1]))
    changed['shadow/new-2000'][0]['occurrence_bindings']['rows'].append(['fabricated surplus'])
    assert results[0] != changed
