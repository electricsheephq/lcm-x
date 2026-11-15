"""Revision 3d: independent judge boundaries and hostile footer observations."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from .record_contract import check_stage, expected_record, judge, observe, projection
from .session import freeze


def row(sid, text, session='parent', conversation='conv', role='user'):
    return dict(store_id=sid, role=role, content=text, session_id=session, conversation_id=conversation)


def observation(later, outcome='listed', text='needle', witness=False):
    entry = dict(owner_store_id=1, predecessor_ordinal=0, captured_uid=None, capture_view=0, due_stage=1,
                 expected=dict(role='user', content=text, observed_at=10., observed_at_source='host_message_timestamp'))
    record = expected_record(entry)
    canonical = sorted([row(1, 'reply'), *later], key=lambda r: r['store_id'])
    reference = dict(tool='lcm_expand', args=dict(queued_user_ref=[1, 0]))
    # As stored: a witnessed owner's record is superseded by the witness row (fake_records.py sets it).
    exact = dict(record, kind='record', superseded_by=2 if witness else None)
    visible = dict(exact, reference=reference, expanded=exact)
    snapshot = dict(canonical=list(map(freeze, canonical)),
                    records=[freeze(dict(record, superseded_by=2 if witness else None))])
    listing = [dict(r, kind='record' if 'owner_store_id' in r else 'canonical')
               for raw in projection(snapshot) if (r := dict(raw))]
    pages = [visible if r['kind'] == 'record' else r for r in listing]
    withheld = []
    if outcome != 'listed':
        pages = [r for r in pages if r['kind'] != 'record']
        withheld = [dict(owner_store_id=1, ordinal=0, state=outcome, reference=reference, expanded=exact)]
    counts = {state: int(state == outcome) for state in ('overlap', 'unscanned')}
    return dict(pages=pages, hits=[visible] if outcome == 'listed' and text.strip() else [],
                withheld=withheld, withheld_counts=[counts], listing_projection=listing,
                baseline_canonical=snapshot['canonical'],
                baseline_canonical_rows=[(r['store_id'], r['role'], r['content']) for r in canonical],
                summary_sources=[], baseline_summary_sources=[], reload_witness={1: 2} if witness else {}), [entry]


@pytest.mark.parametrize('count,hit,allowed', [
    (64, True, {'overlap'}), (65, True, {'overlap', 'unscanned'}),
    (64, False, {'listed'}), (65, False, {'listed', 'unscanned'}),
])
def test_user_row_floor(count, hit, allowed):
    later = [row(i + 3, 'needle' if hit and i == count - 1 else 'other') for i in range(count)]
    # Tool and assistant rows never consume the user-row floor.
    later = [row(2, 'needle', role='tool'), *later, row(100, 'needle', role='assistant')]
    assert_outcomes(later, allowed)


def test_overlap_inside_floor_with_rows_beyond():
    later = [row(i + 2, 'needle' if i == 0 else 'other') for i in range(65)]
    assert_outcomes(later, {'overlap'})


def test_cumulative_utf8_byte_floor():
    suffix = '\n\nneedle'
    texts = ['é' * 300000, 'é' * ((600000 - len(suffix)) // 2) + suffix]
    assert [len(text.encode('utf-8')) for text in texts] == [600000, 600000]
    assert_outcomes([row(i + 2, text) for i, text in enumerate(texts)], {'overlap', 'unscanned'})


@pytest.mark.parametrize('size,hit,allowed', [
    (1048576, True, {'overlap'}), (1048577, True, {'overlap', 'unscanned'}),
    (1048576, False, {'listed'}), (1048577, False, {'listed', 'unscanned'}),
])
def test_utf8_byte_floor(size, hit, allowed):
    suffix = '\n\nneedle' if hit else ''
    # Multibyte content ensures this boundary measures UTF-8 bytes, not characters.
    remaining = size - len(suffix)
    text = 'é' * (remaining // 2) + 'x' * (remaining % 2) + suffix
    assert len(text.encode('utf-8')) == size
    assert_outcomes([row(2, text)], allowed)


def assert_outcomes(later, allowed, **kwargs):
    messages = dict(listed='withheld locator missing', overlap='withheld overlap not proven',
                    unscanned='unscanned inside floor')
    for outcome in ('listed', 'overlap', 'unscanned'):
        observed, entries = observation(later, outcome, **kwargs)
        if outcome in allowed:
            assert judge(observed, entries) == []
        else:
            with pytest.raises(AssertionError, match=messages[outcome]):
                judge(observed, entries)


@pytest.mark.parametrize('session,conversation,allowed', [
    ('parent', 'other', {'overlap'}), ('other', 'conv', {'listed'}),
    ('other', 'other', {'listed'}),
])
def test_session_scope(session, conversation, allowed):
    assert_outcomes([row(2, 'needle', session, conversation)], allowed)


def test_owner_scope_comes_from_canonical_snapshot():
    observed, entries = observation([row(2, 'needle', 'other', 'other')], 'overlap')
    observed['baseline_canonical'][0] = freeze(row(1, 'reply', 'other', 'owner-conversation'))
    assert judge(observed, entries) == []


@pytest.mark.parametrize('text,later,allowed', [
    ('a\n\nb', 'z\n\na\n\nb\n\nc', {'overlap'}),
    ('a\n\nb', 'a\n\nx\n\nb', {'listed'}),
    ('needle', 'prefixneedlesuffix', {'listed'}),
    (' needle ', 'needle', {'overlap'}),
    (' \t ', '', {'listed'}),
    # A short id: the 1 MiB value as its id makes CI's collect-only listing take over 20 minutes.
    pytest.param(' \t ', 'x' * 1048577, {'listed'}, id='blank-text-1mib-later'),
])
def test_segment_predicate_and_empty_text(text, later, allowed):
    assert_outcomes([row(2, later)], allowed, text=text)


@pytest.mark.parametrize('text', [' \t ', 'needle'])
@pytest.mark.parametrize('field,value', [('content', 'wrong'), ('observed_at', -1.), ('session_id', 'wrong')])
def test_listing_reference_is_exact(text, field, value):
    observed, entries = observation([], text=text)
    assert judge(observed, entries) == []
    observed['pages'][0]['expanded'] = dict(observed['pages'][0]['expanded'], **{field: value})
    assert judge(observed, entries) == [(0, 0, text)]


@pytest.mark.parametrize('field,value', [('observed_at', -1.), ('session_id', 'wrong')])
def test_search_reference_is_exact(field, value):
    observed, entries = observation([])
    assert judge(observed, entries) == []
    hit = observed['hits'][0]
    observed['hits'] = [dict(hit, expanded=dict(hit['expanded'], **{field: value}))]
    assert judge(observed, entries) == [(0, 0, 'needle')]


def test_witnessed_search_reference_matches_witness():
    observed, entries = observation([row(2, 'needle\n\nreply')], witness=True)
    hit = observed['hits'][0]
    observed['hits'] = [dict(hit, expanded=dict(hit['expanded'], superseded_by=2))]
    assert judge(observed, entries) == []


@pytest.mark.parametrize('field,value', [
    ('superseded_by', None), ('superseded_by', 3), ('observed_at', -1.), ('session_id', 'wrong'),
])
def test_witnessed_search_reference_is_exact(field, value):
    observed, entries = observation([row(2, 'needle\n\nreply')], witness=True)
    hit = observed['hits'][0]
    hit = dict(hit, expanded=dict(hit['expanded'], superseded_by=2))
    observed['hits'] = [hit]
    assert judge(observed, entries) == []
    observed['hits'] = [dict(hit, expanded=dict(hit['expanded'], **{field: value}))]
    assert judge(observed, entries) == [(0, 0, 'needle')]


def test_witness_precedes_overlap():
    observed, entries = observation([row(2, 'needle\n\nreply')], witness=True)
    assert judge(observed, entries) == []


def test_whitespace_turn_raw_witness(monkeypatch, tmp_path):
    from . import test_paired_merged_user_turns as merged
    from .session import Trace

    pending = [dict(role='user', content=text, timestamp=10. + i)
               for i, text in enumerate([' \t ', 'P1', 'P2', 'R'])]
    traces = []
    def opened(*_):
        trace = Trace()
        traces.append(trace)
        def capture(**kwargs):
            trace.capture(kwargs['conversation_history'], kwargs['user_message'])
        engine = SimpleNamespace(_paired_trace=trace, ingest=lambda *_: None, shutdown=lambda: None)
        return SimpleNamespace(engine=engine, hooks=dict(pre_llm_call=capture))
    # Exercise the fixture's actual witness builder without opening a product store.
    monkeypatch.setattr(merged, '_wide_reload_plan', lambda _: (1, pending, None))
    monkeypatch.setattr(merged.base, '_open', opened)
    merged._wide_reload_case(2708039, tmp_path, monkeypatch, dict(main=None, candidate=None), lambda *_: None)
    _, reloaded, joined = traces[0].reloads[0]
    assert reloaded['content'] == 'P1\n\nP2\n\nR'
    observed, template = observation([row(2, reloaded['content'])], text=' \t ')
    entries = [dict(template[0], predecessor_ordinal=i, expected=dict(template[0]['expected'], content=text))
               for i, text in enumerate([' \t ', 'P1', 'P2'])]
    records = [expected_record(entry) for entry in entries]
    snapshot = dict(canonical=observed['baseline_canonical'], records=list(map(freeze, records)))
    witness = {1: 2} if reloaded['content'] == joined else {}
    baseline = dict(canonical=snapshot['canonical'], reload_witness=witness)
    observed['reload_witness'] = witness
    observed['listing_projection'] = [dict(r, kind='record' if 'owner_store_id' in r else 'canonical')
                                      for raw in projection(snapshot) if (r := dict(raw))]
    observed['withheld'] = [dict(owner_store_id=1, ordinal=r['ordinal'], state='overlap',
        reference=dict(tool='lcm_expand', args=dict(queued_user_ref=[1, r['ordinal']])),
        expanded=dict(r, kind='record')) for r in records[1:]]
    observed['withheld_counts'] = [dict(overlap=2, unscanned=0)]
    check_stage(snapshot, baseline, entries)
    assert judge(observed, entries) == []

    superseded = dict(snapshot, records=[freeze(dict(r, superseded_by=2 if r['ordinal'] else None))
                                        for r in records])
    with pytest.raises(AssertionError, match='superseded without occurrence witness'):
        check_stage(superseded, baseline, entries)


def test_withheld_search_hit_expands_to_record():
    observed, entries = observation([row(2, 'needle')], 'overlap')
    locator = observed['withheld'][0]
    observed['hits'] = [dict(kind='canonical', role='user', content='needle', reference=locator['reference'],
                             expanded=locator['expanded'])]
    with pytest.raises(AssertionError, match='withheld record searchable'):
        judge(observed, entries)


def test_observe_footer_cursor_expansions_and_raw_order():
    calls, queries, expansions = [], [], []
    def record(ordinal):
        return dict(owner_store_id=1, ordinal=ordinal, role='user', content='needle',
                    reference=dict(tool='lcm_expand', args=dict(ordinal=ordinal)))
    def load(args, **_):
        calls.append(args)
        second = args.get('withheld_cursor') == 1
        return json.dumps(dict(messages=[] if second else [record(9), record(0)], next_cursor=None,
            withheld=dict(counts=dict(overlap=2, unscanned=0), locators=[dict(record(0), state='overlap')],
                          next_cursor=None if second else dict(withheld_cursor=1))))
    def grep(args, **_):
        queries.append(args['query'])
        return json.dumps(dict(results=[record(9), record(0)]))
    def expand(args, **_):
        expansions.append(args['ordinal'])
        return json.dumps(record(args['ordinal']))
    engine = SimpleNamespace(_dag=SimpleNamespace(_conn=SimpleNamespace(execute=lambda _: [])),
        _paired_trace=SimpleNamespace(record_captures=[dict(rows=[dict(content=' \t '),
            dict(content='needle'), dict(content='reply')])]))
    observed = observe(engine, SimpleNamespace(lcm_load_session=load, lcm_grep=grep, lcm_expand=expand))
    assert len(calls) == 2 and calls[1]['record_cursor'] is None
    assert [r['ordinal'] for r in observed['pages']] == [9, 0]
    assert [r['ordinal'] for r in observed['hits']] == [9, 0]
    assert [r['ordinal'] for r in observed['withheld']] == [0, 0]
    assert observed['withheld_counts'] == [dict(overlap=2, unscanned=0)] * 2
    assert expansions == [9, 0, 0, 0, 9, 0] and queries == ['"needle"']


@pytest.mark.parametrize('mutation,message', [
    ('listed', 'withheld record listed'), ('searchable', 'withheld record searchable'),
    ('missing', 'withheld locator missing'), ('expansion', 'withheld expansion differs'),
    ('counts', 'withheld counts differ from locators'), ('page-counts', 'withheld counts disagree across pages'),
    ('duplicate', 'duplicate withheld locator'), ('no-overlap', 'withheld overlap not proven'),
    ('inside-floor', 'unscanned inside floor'), ('superseded', 'withheld locator not active or superseded'),
])
def test_broken_withheld_judgments(mutation, message, record_property):
    observed, entries = observation([row(2, 'needle')], 'overlap')
    locator = observed['withheld'][0]
    if mutation in ('listed', 'searchable'):
        visible, _ = observation([row(2, 'needle')])
        observed['pages' if mutation == 'listed' else 'hits'] = visible['pages' if mutation == 'listed' else 'hits']
    elif mutation == 'missing':
        observed['withheld'], observed['withheld_counts'] = [], [dict(overlap=0, unscanned=0)]
    elif mutation == 'expansion':
        locator['expanded'] = dict(locator['expanded'], observed_at=-1.)
    elif mutation == 'counts':
        observed['withheld_counts'] = [dict(overlap=0, unscanned=1)]
    elif mutation == 'page-counts':
        observed['withheld_counts'].append(dict(overlap=0, unscanned=0))
    elif mutation == 'duplicate':
        observed['withheld'].append(deepcopy(locator))
    elif mutation == 'no-overlap':
        observed, entries = observation([row(2, 'unrelated')], 'overlap')
    elif mutation == 'inside-floor':
        observed, entries = observation([], 'unscanned')
    else:
        observed, entries = observation([row(2, 'needle\n\nreply')], 'overlap', witness=True)
    with pytest.raises(AssertionError, match=message) as caught:
        judge(observed, entries)
    record_property('mutation_failure', str(caught.value))
