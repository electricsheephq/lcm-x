"""Ungated absolute-oracle regressions; no engine or baseline is needed."""
import pytest

from .paired import normalise_rows
from .session import Trace, duplicate_rows, freeze, missing_messages


def stored(text, bindings=(), **fields):
    return freeze(dict(role='user', content=text, observed_at=10., source='cli', session_id='parent',
                       conversation_id='conv', bindings=bindings, **fields))


def host(text, stamp, uid=None):
    return dict(role='user', content=text, timestamp=stamp, message_uid=uid)


@pytest.mark.parametrize('messages,rows,expected', [
    ([host('R', 10, 'uid-A'), host('R', 30, 'uid-B')], [stored('R')], {'R': 1}),
    ([host('R', 10)], [stored('R-2707403-1')], {'R': 1}),
    ([host('R', 10, 'uid-A'), host('P2\n\nR', 30, 'uid-B')], [stored('P2\n\nR')], {'R': 1}),
    ([host('P1', 10), host('R', 30)], [stored('P1\n\nR')], {}),
])
def test_ingest_segment_consumption(messages, rows, expected):
    trace = Trace()
    trace.observe(messages, ingest=True)
    assert missing_messages(trace.messages, rows) == expected


def test_pre_only_segments_and_multiplicity():
    trace = Trace()
    trace.observe([host('P1', 10), host('R', 30)])
    assert missing_messages(trace.messages, [stored('P1\n\nR')]) == {}
    trace.observe([host('R', 40)])
    assert missing_messages(trace.messages, [stored('P1\n\nR')]) == {'R': 1}


@pytest.mark.parametrize('bindings,expected', [
    ([(('uid-A', 'canonical', 'parent'),), (('uid-B', 'canonical', 'parent'),)], []),
    ([(('uid-A', 'canonical', 'parent'),), ()], [2]),
    ([(), ()], [2]),
])
def test_duplicate_identity(bindings, expected):
    duplicates = duplicate_rows([stored('R', binding) for binding in bindings])
    assert [count for _, count in duplicates] == expected
    assert all('bindings' not in dict(identity) for identity, _ in duplicates)


def test_uid_and_unlabelled_snapshot_are_one_message():
    trace = Trace()
    rows = [host('R', 10), host('R', 10, 'uid-A'), host('R', 10, 'uid-B')]
    trace.observe(rows, ingest=True)
    rows[0]['content'] = 'mutated'
    assert len(trace.messages) == 2
    assert missing_messages(trace.messages, [stored('R')]) == {'R': 1}


def test_user_message_only_identity_uses_call_ordinal():
    trace = Trace()
    trace.capture([host('R', 10)], 'R')
    trace.capture([], 'R')
    trace.capture([], 'R')
    assert len(trace.messages) == 3
    assert missing_messages(trace.messages, [stored('R')]) == {'R': 2}


def test_preservation_only_changes_content_and_observed_at():
    old = stored('P\n\nR')
    new = freeze(dict(dict(old), content='R', observed_at=20.))
    assert normalise_rows([old, old], {old: new}) == [new, new]
    for field, value in [('source', 'other'), ('observed_at_source', 'other'),
                         ('bindings', (('uid-B', 'canonical', 'parent'),))]:
        changed = freeze(dict(dict(new), **{field: value}))
        assert normalise_rows([old], {old: changed}) == [old]
