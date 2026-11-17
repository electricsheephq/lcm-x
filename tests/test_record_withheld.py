"""Non-differential checks for canonical expansion and capture attribution."""
from copy import deepcopy

import pytest

from tests.differential.record_contract import judge
from tests.differential.test_record_withheld import observation, refusal_observation, row


@pytest.mark.parametrize('companion', ['alone', 'record', 'witness'])
@pytest.mark.parametrize('correct', [False, True])
def test_every_canonical_witness_expansion_is_checked(companion, correct):
    observed, entries = observation([row(2, 'needle\n\nreply')], witness=True)
    witness = dict(row(2, 'needle\n\nreply'), kind='canonical')
    hit = dict(witness, expanded=witness)
    checked = dict(hit, content='snippet', expanded=witness if correct else dict(witness, content='WRONG'))
    observed['hits'] = [checked, *dict(alone=[], record=observed['hits'], witness=[hit])[companion]]
    assert judge(observed, entries) == ([] if correct else [(0, 0, 'needle')])


@pytest.mark.parametrize('wrong_capture', [False, True])
def test_refusal_count_belongs_to_its_capture(wrong_capture):
    observed, entries = refusal_observation()
    observed['refused_captures'] = {int(wrong_capture): dict(backward_overlap=1)}
    if wrong_capture:
        with pytest.raises(AssertionError, match='backward_overlap refusal count differs'):
            judge(observed, entries)
    else:
        assert judge(observed, entries) == []
    # One capture refuses its whole run, rather than counting every predecessor.
    entries.append(deepcopy(entries[0]))
    entries[1]['predecessor_ordinal'] = 1
    observed['refused_captures'] = {0: dict(backward_overlap=1)}
    assert judge(observed, entries) == []
