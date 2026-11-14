"""The reusable oracle must expose loss, additions, multiplicity and order."""
import subprocess

import pytest

from .paired import archive, compare_rows, normalise_rows
from .session import Trace

ROW = ('user', 'one', 1., 'parent', 'conv')
OTHER = ('user', 'two', 2., 'parent', 'conv')


@pytest.mark.parametrize('baseline,candidate,loss,extra,order', [
    ([ROW], [ROW], [], [], False),
    ([ROW], [], [ROW], [], False),
    ([], [ROW], [], [ROW], False),
    ([ROW, ROW], [ROW], [ROW], [], False),
    ([ROW], [ROW, ROW], [], [ROW], False),
    ([ROW, OTHER], [OTHER, ROW], [], [], True),
    ([ROW, ROW, OTHER], [ROW, OTHER, ROW], [], [], True),
    ([ROW, OTHER], [OTHER, ROW, ROW], [], [ROW], True),
])
def test_compare_rows(baseline, candidate, loss, extra, order):
    assert compare_rows(baseline, candidate) == dict(LOSS=loss, EXTRA=extra, ORDER=order)


def test_normalisation_is_exact_and_keeps_multiplicity():
    assert normalise_rows([ROW, ROW, OTHER], {ROW: OTHER}) == [OTHER, OTHER, OTHER]


@pytest.mark.parametrize('error', [FileNotFoundError('git unavailable'),
                                 subprocess.CalledProcessError(128, 'git', stderr=b'unknown ref')])
def test_unavailable_archive_skips_with_reason(monkeypatch, tmp_path, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, 'run', unavailable)
    with pytest.raises(pytest.skip.Exception, match='differential baseline.*unavailable'):
        archive('missing-ref', tmp_path)


def test_absolute_oracle_uses_raw_rows_and_retains_missing_queued_text():
    trace = Trace()
    trace.ingests.append(dict(texts=['one', 'queued'], rows=[ROW, ROW]))
    assert trace.violations() == [dict(ingest=1, missing=['queued'], duplicates=[ROW])]
