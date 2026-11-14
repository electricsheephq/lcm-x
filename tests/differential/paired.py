"""Offline, pinned baseline versus working-tree engine comparisons.

LCMX_DIFF_BASELINE_REF overrides the pin; LCMX_DIFF_CANDIDATE_TREE selects an
archived positive control. Missing history fails in CI and skips locally.
"""
from collections import Counter
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from .trees import PrerequisiteUnavailable, materialize, resolve, skip_or_fail

BASELINE_REF = "7d614f5fe22eb7cf7d8955230c51619916a1fed3"
ROOT = Path(__file__).resolve().parents[2]


def _load(tree, name):
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        name, Path(tree) / '__init__.py', submodule_search_locations=[str(tree)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def versions(tmp_path_factory):
    ref = os.environ.get('LCMX_DIFF_BASELINE_REF', BASELINE_REF)
    try:
        tree = materialize(resolve(ref), tmp_path_factory.mktemp('paired-baseline'))
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    return {
        'main': _load(tree, '_paired_baseline'),
        'candidate': _load(Path(os.environ.get('LCMX_DIFF_CANDIDATE_TREE', ROOT)),
                           '_paired_candidate'),
    }


def normalise_rows(rows, preserved=None, reference_rows=()):
    """Replace only explicitly proven exact tuples; never remove a row."""
    preserved = preserved or {}
    result = []
    for row in rows:
        row = tuple(row)
        if row and isinstance(row[0], tuple):
            fields = dict(row)
            short = tuple(fields[key] for key in ('role', 'content', 'observed_at', 'session_id', 'conversation_id'))
            target = preserved.get(row, preserved.get(short))
            if target is not None:
                if target and isinstance(target[0], tuple):
                    target = dict(target)
                    if {k: v for k, v in fields.items() if k not in ('content', 'observed_at')} != {
                            k: v for k, v in target.items() if k not in ('content', 'observed_at')}:
                        result.append(row)
                        continue
                    fields.update(content=target['content'], observed_at=target['observed_at'])
                else:
                    proposed = dict(fields, content=target[1], observed_at=target[2])
                    exact = tuple(sorted(proposed.items()))
                    if exact not in reference_rows:
                        result.append(row)
                        continue
                    assert (short[0], *short[3:]) == (target[0], *target[3:])
                    fields = proposed
                row = tuple(sorted(fields.items()))
        else:
            target = preserved.get(row, row)
            if (row[0], *row[3:]) == (target[0], *target[3:]):
                row = target
        result.append(row)
    return result


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
