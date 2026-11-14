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


def test_diagnostics_keep_repeated_properties_and_safe_names(tmp_path, monkeypatch):
    import json
    import re
    from .paired import write_diagnostics

    monkeypatch.setenv('LCM_DIFFERENTIAL_OUT', str(tmp_path))
    properties = [('differences', '[]'), ('differences', '[1]'), ('attempts', 0)]
    path = write_diagnostics('../case/1', 'test/name[param]', properties)
    assert path.parent == tmp_path / 'paired'
    assert re.fullmatch(r'[A-Za-z0-9_-]+\.json', path.name)
    assert json.loads(path.read_text()) == dict(case='../case/1', test='test/name[param]',
                                              properties=[list(item) for item in properties])


def test_diagnostics_write_nothing_without_output(tmp_path, monkeypatch):
    from .paired import write_diagnostics

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('LCM_DIFFERENTIAL_OUT', raising=False)
    assert write_diagnostics(1, 'test_absolute', [('baseline_pin', '{}')]) is None
    assert not list(tmp_path.iterdir())
