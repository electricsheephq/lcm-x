"""Offline, pinned baseline versus working-tree engine comparisons.

LCMX_DIFF_BASELINE_REF overrides the pin; LCMX_DIFF_CANDIDATE_TREE selects an
archived positive control. Missing git/history is an explicit pytest skip.
"""
from collections import Counter
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

BASELINE_REF = "7d614f5fe22eb7cf7d8955230c51619916a1fed3"
ROOT = Path(__file__).resolve().parents[2]


def _load(tree, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(tree) / '__init__.py', submodule_search_locations=[str(tree)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def archive(ref, destination):
    """Use local history only; neither fetching nor a checkout is needed."""
    try:
        packed = subprocess.run(['git', 'archive', ref], cwd=ROOT,
                                check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode(errors='replace').strip() if isinstance(
            exc, subprocess.CalledProcessError) else str(exc)
        pytest.skip(f'differential baseline {ref!r} unavailable: {detail}')
    with tarfile.open(fileobj=io.BytesIO(packed)) as source:
        source.extractall(destination, filter='data')
    return destination


@pytest.fixture(scope='module')
def versions(tmp_path_factory):
    ref = os.environ.get('LCMX_DIFF_BASELINE_REF', BASELINE_REF)
    tree = archive(ref, tmp_path_factory.mktemp('paired-baseline'))
    return {
        'main': _load(tree, '_paired_baseline'),
        'candidate': _load(Path(os.environ.get('LCMX_DIFF_CANDIDATE_TREE', ROOT)),
                           '_paired_candidate'),
    }


def normalise_rows(rows, preserved=None):
    """Replace only explicitly proven exact tuples; never remove a row."""
    preserved = preserved or {}
    return [preserved.get(tuple(row), tuple(row)) for row in rows]


def compare_rows(baseline, candidate):
    """Return multiset LOSS/EXTRA and common-row ORDER differences for one ingest.

    Call for every ingest, including intermediate states. Counter subtraction
    retains duplicate multiplicity, and common-row order uses occurrence ranks.
    """
    baseline, candidate = normalise_rows(baseline), normalise_rows(candidate)
    left, right = Counter(baseline), Counter(candidate)
    common = left & right

    def shared(rows):
        seen = Counter()
        result = []
        for row in rows:
            seen[row] += 1
            if seen[row] <= common[row]:
                result.append(row)
        return result

    return {
        'LOSS': list((left - right).elements()),
        'EXTRA': list((right - left).elements()),
        'ORDER': shared(baseline) != shared(candidate),
    }


def compare_stages(results, record_property):
    """Record every ingest, even when an earlier ingest already diverged."""
    import json

    baseline, candidate = results['main'], results['candidate']
    counts = {name: [len(rows) for rows in stages] for name, stages in results.items()}
    differences = []
    for index in range(max(len(baseline), len(candidate))):
        rows = compare_rows(baseline[index] if index < len(baseline) else [],
                            candidate[index] if index < len(candidate) else [])
        if any(rows.values()) or index >= min(len(baseline), len(candidate)):
            differences.append(dict(ingest=index + 1, **rows))
    record_property('row_counts', json.dumps(counts))
    record_property('differences', json.dumps(differences))
    return differences
