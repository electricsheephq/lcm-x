"""Ungated absolute-oracle regressions; no engine or baseline is needed."""
import pytest
from types import SimpleNamespace

from .paired import normalise_rows
from .session import FaultRun, Trace, assert_faults, audit_fault, duplicate_rows, fault_counts, freeze, missing_messages


def stored(text, bindings=(), **fields):
    return freeze(dict(role='user', content=text, observed_at=10., source='cli', session_id='parent',
                       conversation_id='conv', bindings=bindings, **fields))


def host(text, stamp, uid=None):
    return dict(role='user', content=text, timestamp=stamp, message_uid=uid)


@pytest.mark.parametrize('messages,rows,expected,ingest', [
    ([host('R', 10, 'uid-A'), host('R', 30, 'uid-B')], [stored('R')], {'R': 1}, True),
    ([host('R', 10)], [stored('R-2707403-1')], {'R': 1}, True),
    ([host('R', 10, 'uid-A'), host('P2\n\nR', 30, 'uid-B')], [stored('P2\n\nR')], {'R': 1}, True),
    ([host('P1', 10), host('R', 30)], [stored('P1\n\nR')], {}, False),
])
def test_view_segment_consumption(messages, rows, expected, ingest):
    trace = Trace()
    trace.observe(messages, ingest=ingest)
    assert missing_messages(trace.views, rows) == {(0, text): count for text, count in expected.items()}


def test_b4_alias_rewrite_views_do_not_accumulate_identity():
    # Seed 2707408, stop evidence: the initial alias rows retain both R's.
    trace = Trace()
    trace.observe([host('P1\n\nR', 10, 'uid-A'), host('P2\n\nR', 30, 'uid-B')], ingest=True)
    trace.observe([host('R', 20, 'uid-new-2707408-0'), host('R', 40, 'uid-B'),
                   host('R-2707408-0', 110)], ingest=True)
    assert missing_messages(trace.views, [stored('P1\n\nR'), stored('P2\n\nR'),
                                          stored('R-2707408-0')]) == {}


def test_consumption_resets_between_views_but_not_within_one_view():
    trace = Trace()
    trace.observe([host('R', 10)])
    trace.observe([host('R', 40)])
    rows = [stored('R')]
    assert missing_messages(trace.views, rows) == {}
    trace.observe([host('R', 10), host('R', 40)])
    assert missing_messages(trace.views, rows) == {(2, 'R'): 1}


@pytest.mark.parametrize('bindings,expected', [
    ([(('uid-A', 'canonical', 'parent'),), (('uid-B', 'canonical', 'parent'),)], []),
    ([(('uid-A', 'canonical', 'parent'),), ()], []),
    ([(), ()], [1]),
])
def test_duplicate_full_identity_includes_bindings(bindings, expected):
    duplicates = duplicate_rows([stored('R', binding) for binding in bindings])
    assert [count for _, count in duplicates] == expected
    assert all('bindings' in dict(identity) for identity, _ in duplicates)


@pytest.mark.parametrize('multiplicity,expected', [(2, []), (1, [1])])
def test_duplicate_host_multiplicity(multiplicity, expected):
    trace = Trace()
    for _ in range(2):
        trace.observe([host('R', 10)] * multiplicity, ingest=True)
    duplicates = duplicate_rows([stored('R'), stored('R')], trace.views)
    assert [count for _, count in duplicates] == expected


def test_snapshot_and_user_message_are_per_call():
    trace = Trace()
    rows = [host('R', 10)]
    trace.capture(rows, 'R')
    rows[0]['content'] = 'changed'
    trace.capture([], 'R')
    trace.capture([], 'R')
    assert [len(view['rows']) for view in trace.views] == [1, 1, 1]
    assert missing_messages(trace.views, [stored('R')]) == {}


def test_preservation_only_changes_content_and_observed_at():
    old = stored('P\n\nR')
    new = freeze(dict(dict(old), content='R', observed_at=20.))
    assert normalise_rows([old, old], {old: new}) == [new, new]
    for field, value in [('source', 'other'), ('observed_at_source', 'other'),
                         ('bindings', (('uid-B', 'canonical', 'parent'),))]:
        changed = freeze(dict(dict(new), **{field: value}))
        assert normalise_rows([old], {old: changed}) == [old]


def test_short_preservation_requires_matching_full_reference_identity():
    old = stored('P\n\nR')
    new = freeze(dict(dict(old), content='R', observed_at=20.))
    proof = {('user', 'P\n\nR', 10., 'parent', 'conv'): ('user', 'R', 20., 'parent', 'conv')}
    assert normalise_rows([old], proof, [new]) == [new]
    changed = freeze(dict(dict(old), source='other'))
    assert normalise_rows([changed], proof, [new]) == [changed]


@pytest.mark.parametrize('role,expected', [('user', 1), ('assistant', 2)])
def test_only_user_history_suppresses_user_message(role, expected):
    trace = Trace()
    trace.observe([dict(host('R', 10), role=role)], user_message='R')
    assert len(trace.views[0]['rows']) == expected


def fault_trace(present=True, called=0, raised=0, stubbed=False):
    trace = Trace(stub_faults=stubbed)
    fault = audit_fault(SimpleNamespace(_paired_trace=trace), 'evidence', present)
    fault.called, fault.raised = called, raised
    return trace


@pytest.mark.parametrize('present,expected', [(False, 'inert'), (True, 'unreached')])
def test_fault_absent_or_unreached_passes_case(present, expected):
    trace = fault_trace(present)
    assert_faults(123, {'main': trace})
    counts = fault_counts(trace.faults)['evidence']
    assert counts == dict(requested=1, inert=int(not present), unreached=int(present), called=0, raised=0)
    assert counts[expected] == 1


def test_fault_called_without_raise_fails_its_registration():
    trace = fault_trace(called=1, raised=1)
    # A firing of the same kind at another stage must not hide the misfire.
    trace.ingests.append({})
    fault = audit_fault(SimpleNamespace(_paired_trace=trace), 'evidence', True)
    fault.called = 1
    with pytest.raises(AssertionError, match='123: candidate evidence stage 2: fault called but never raised'):
        assert_faults(123, {'candidate': trace})


def test_full_fault_run_fails_floor_for_unreached_present_kind():
    run = FaultRun()
    run.record(1, {'main': fault_trace()})
    result = run.finish([1], [1])
    assert result['evaluated'] and result['reason'] is None
    assert result['errors'] == ['main: evidence present but never raised across full case set']


def test_partial_fault_run_skips_floor_with_reason():
    run = FaultRun()
    run.record(1, {'main': fault_trace()})
    result = run.finish([1, 2], [1])
    assert not result['evaluated'] and not result['errors']
    assert result['reason'] == 'partial case set selected (1/2); floor skipped'


@pytest.mark.parametrize('order', [(1, 2), (2, 1), (1, 2, 1)])
def test_floor_is_per_tree_order_independent_and_does_not_double_count(order):
    run = FaultRun()
    for case in order:
        run.record(case, {'main': fault_trace(called=int(case == 2), raised=int(case == 2)),
                          'candidate': fault_trace()})
    result = run.finish([1, 2], [1, 2])
    assert result['errors'] == ['candidate: evidence present but never raised across full case set']
    assert result['totals']['main']['evidence']['requested'] == 2
    assert result['totals']['main']['evidence']['raised'] == 1


def test_stubbed_faults_are_excluded_from_case_rule_and_floor():
    trace = fault_trace(called=1, stubbed=True)
    assert_faults(1, {'main': trace})
    run = FaultRun()
    run.record(1, {'main': trace})
    result = run.finish([1], [1])
    assert result['evaluated'] and not result['errors']
    assert result['totals'] == {}
