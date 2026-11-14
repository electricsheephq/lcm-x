"""The reusable oracle must expose loss, additions, multiplicity and order."""

import pytest

from .paired import compare_rows, normalise_rows

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
