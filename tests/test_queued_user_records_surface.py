"""The records arm may add bounded siblings, never alter canonical tool values."""
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm import expand_tools, queued_user_rows as q, store as sm, tools
from hermes_lcm.dag import SummaryNode
from tests.harness import engine_context
from tests.test_queued_user_rows import folded, records, user


CALLS = [
    ('lcm_load_session', dict(session_id='S')),
    ('lcm_load_session', dict(session_id='S', after_store_id=1)),
    ('lcm_load_session', dict(session_id='S', anchor_store_id=1)),
    ('lcm_load_session', dict(session_id='S', roles=['assistant'])),
    *[('lcm_grep', dict(query='lost', sort=sort)) for sort in ('relevance', 'recency', 'hybrid')],
    ('lcm_grep', dict(query='lost', time_from=0, time_to=11)),
    ('lcm_recall', dict(query='lost')),
    ('lcm_expand', dict(store_id=1)),
    ('lcm_expand', dict(node_id=1)),
    ('lcm_expand', dict(node_id=987654321)),
]
ALLOWED = {'queued_user_records', 'queued_user_records_withheld', 'queued_user_record_hits',
           'queued_user_records_status', 'queued_user_records_omitted'}


@pytest.mark.parametrize('state', ['absent', 'present', 'empty', 'corrupt', 'locked', 'journal'])
@pytest.mark.parametrize('name,args', CALLS)
def test_canonical_response_unchanged(tmp_path, name, args, state):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._dag.add_node(SummaryNode(session_id='S', depth=0, summary='summary',
            token_count=1, source_token_count=1, source_ids=[1], source_type='messages',
            created_at=1., earliest_at=1., latest_at=1.))
        path = sm.records_path(e._store)
        lock = None
        if state == 'absent':
            path.rename(path.with_suffix('.saved'))
        elif state == 'empty':
            with sqlite3.connect(path) as side:
                side.execute('DELETE FROM queued_user_rows')
        elif state == 'corrupt':
            path.write_bytes(b'not a sqlite database')
        elif state == 'locked':
            lock = sqlite3.connect(path)
            lock.execute('BEGIN EXCLUSIVE')
        elif state == 'journal':
            Path(str(path) + '-journal').write_bytes(b'\xd9\xd5\x05\xf9\x20\xa1\x63\xd7' + b'\0' * 1024)
        handler = getattr(tools, name)
        original = expand_tools.lcm_expand.__wrapped__ if name == 'lcm_expand' else handler.__wrapped__
        try:
            expected = original(args, engine=e)
            actual = handler(args, engine=e)
            base, candidate = json.loads(expected), json.loads(actual)
            assert {k: candidate[k] for k in base} == base
            assert set(candidate) - set(base) <= ALLOWED
            if state in ('absent', 'empty'):
                assert actual == expected
            elif state in ('corrupt', 'locked', 'journal') and 'error' not in base:
                assert candidate['queued_user_records_status']['available'] is False
                if state == 'journal':
                    assert candidate['queued_user_records_status']['reason'] == 'recovery_pending'
                    assert Path(str(path) + '-journal').exists()
        finally:
            if lock:
                lock.close()


def test_owner_page_and_expand_reference(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        record, = records(e)
        page = json.loads(tools.lcm_load_session(dict(session_id='S', limit=1), engine=e))
        visible, = page['queued_user_records']
        assert visible['reference'] == dict(tool='lcm_expand', args=dict(store_id=record['owner_store_id']))
        expanded = json.loads(tools.lcm_expand(visible['reference']['args'], engine=e))
        assert expanded['queued_user_records'][0]['content'] == 'lost'
        assert expanded['queued_user_records'][0]['state'] == 'visible'
        later = json.loads(tools.lcm_load_session(dict(session_id='S', after_store_id=1), engine=e))
        assert not (set(later) & ALLOWED)
        parsed_owner = json.loads(tools.lcm_expand(dict(store_id=str(record['owner_store_id'])), engine=e))
        assert parsed_owner['queued_user_records'] == expanded['queued_user_records']


def test_provider_cache_reused_with_records(tmp_path, monkeypatch):
    from tests.test_embedding_search_modes import MockProvider
    # Use the same provider-cache assertion as the existing semantic regression.
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._config.embeddings_enabled = True
        e._config.embedding_provider = 'ollama'
        e._config.embedding_model = 'mock-model'
        provider, calls = MockProvider(), []
        def resolve(*args, **kwargs):
            calls.append(True)
            return provider
        monkeypatch.setattr(tools, 'resolve_provider', resolve)
        for _ in range(2):
            tools.lcm_grep(dict(query='lost', mode='semantic'), engine=e)
        assert len(calls) == 1
        assert provider.queries == ['lost', 'lost']


@pytest.mark.parametrize('name', ['lcm_grep', 'lcm_recall'])
def test_search_bounds_and_omissions_preserve_canonical(tmp_path, monkeypatch, name):
    from hermes_lcm import queued_user_rows as q
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e, tuple('needle-' + str(i) + '界' * 1000 for i in range(30)) + ('newest',))
        while e._queued_entries:
            q.drain(e)
        constant = '_LCM_GREP_RESPONSE_CHAR_CAP' if name == 'lcm_grep' else '_LCM_RECALL_RESPONSE_CHAR_CAP'
        monkeypatch.setattr(tools, constant, 4000)
        args = dict(query='needle')
        handler = getattr(tools, name)
        base = json.loads(handler.__wrapped__(args, engine=e))
        raw = handler(args, engine=e)
        actual = json.loads(raw)
        assert {k: actual[k] for k in base} == base
        hits = actual.get('queued_user_record_hits', [])
        assert len(raw) <= 4000
        assert len(hits) + actual['queued_user_records_omitted'] == 30
        assert [r['ordinal'] for r in hits] == list(reversed(range(30)))[:len(hits)]
        for hit in hits:
            assert len(hit['content']) == len(hit['snippet']) == tools._LCM_RECALL_SNIPPET_CHARS
            assert hit['content_truncated'] is True


def test_load_and_expand_use_canonical_slicing(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        text = 'needle ' * 1000
        folded(e, (text, 'newest'))
        page = json.loads(tools.lcm_load_session(dict(session_id='S', max_content_chars=17), engine=e))
        record, = page['queued_user_records']
        assert tools._slice_loaded_content(text, 17).items() <= record.items()
        expanded = json.loads(tools.lcm_expand(dict(store_id=1, max_tokens=10, content_offset=7), engine=e))
        record, = expanded['queued_user_records']
        assert tools._slice_content_for_response(text, 10, 7).items() <= record.items()


@pytest.mark.parametrize('filters,expected', [
    ({}, True), ({'role': 'user'}, True), ({'role': 'assistant'}, False),
    ({'conversation_id': ' conv '}, True), ({'conversation_id': 'other'}, False),
    ({'time_from': 9, 'time_to': 10}, True), ({'time_from': 11}, False),
    ({'session_scope': 'session', 'session_id': 'S'}, True),
    ({'session_scope': 'session', 'session_id': 'other'}, False),
    ({'source': 'telegram'}, False), ({'content_scope': 'both'}, False),
    ({'include': 'summaries'}, False),
])
def test_search_filters_do_not_leak_records(tmp_path, filters, expected):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        reply = json.loads(tools.lcm_grep(dict(query='lost', **filters), engine=e))
        assert bool(reply.get('queued_user_record_hits')) is expected


@pytest.mark.parametrize('name', ['lcm_grep', 'lcm_recall'])
@pytest.mark.parametrize('exclusions', [dict(exclude_current_session=True), dict(exclude_session_ids=[' S '])])
def test_record_session_exclusions(tmp_path, name, exclusions):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        reply = json.loads(getattr(tools, name)(dict(query='lost', **exclusions), engine=e))
        assert not reply.get('queued_user_record_hits')


def test_p1_expand_node_with_null_store_id(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._dag.add_node(SummaryNode(session_id='S', depth=0, summary='summary', token_count=1,
            source_token_count=1, source_ids=[1], source_type='messages', created_at=1., earliest_at=1., latest_at=1.))
        assert records(e)
        args = dict(node_id=1, store_id=None)
        base = json.loads(expand_tools.lcm_expand.__wrapped__(args, engine=e))
        assert 'error' not in base
        assert json.loads(tools.lcm_expand(args, engine=e)) == base


def iso_folded(e, stamp='2026-11-16T10:00:00Z'):
    run = [user('lost', stamp, 'T-0'), user('newest', '2026-11-16T10:00:01Z', 'T-1')]
    q.capture(e, dict(turn_id='T', session_id='S', conversation_id='conv',
                      conversation_history=run, user_message='newest'))
    run[0]['content'] = 'newest'
    e.ingest([run[0], dict(role='assistant', content='answer', timestamp=50.)])


def legacy_iso_stamp(e):
    with sqlite3.connect(sm.records_path(e._store)) as side:
        side.execute('UPDATE queued_user_rows SET observed_at=? WHERE captured_uid=?',
                     ('2026-11-16T10:00:00Z', 'T-0'))


@pytest.mark.parametrize('legacy', [False, True])
def test_p1_grep_time_filter_iso_record(tmp_path, legacy):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        iso_folded(e)
        if legacy:
            legacy_iso_stamp(e)
        args = dict(query='lost', time_from=0, time_to='2026-11-16T10:00:01Z')
        base = json.loads(tools.lcm_grep.__wrapped__(args, engine=e))
        got = json.loads(tools.lcm_grep(args, engine=e))
        assert {k: got[k] for k in base} == base
        hit, = got['queued_user_record_hits']
        assert hit['content'] == 'lost' and hit['observed_at'] == 1794823200.
        assert isinstance(hit['observed_at'], (int, float))
        assert 'queued_user_records_status' not in got
        if not legacy:
            assert records(e)[0]['observed_at'] == hit['observed_at']
        for handler, call in [(tools.lcm_load_session, dict(session_id='S')),
                              (tools.lcm_expand, dict(store_id=hit['owner_store_id']))]:
            assert json.loads(handler(call, engine=e))['queued_user_records'][0]['observed_at'] == hit['observed_at']


@pytest.mark.parametrize('legacy', [False, True])
def test_p1_mixed_stamp_sort(tmp_path, legacy):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        iso_folded(e)
        if legacy:
            legacy_iso_stamp(e)
        folded(e, texts=('lost again', 'second'), turn='U')
        args = dict(query='lost')
        base = json.loads(tools.lcm_grep.__wrapped__(args, engine=e))
        got = json.loads(tools.lcm_grep(args, engine=e))
        assert {k: got[k] for k in base} == base
        hits = got['queued_user_record_hits']
        assert [hit['content'] for hit in hits] == ['lost', 'lost again']
        assert all(isinstance(hit['observed_at'], (int, float)) for hit in hits)
        assert hits[0]['observed_at'] == 1794823200.
        assert 'queued_user_records_status' not in got


def test_unusable_observed_at_keeps_record(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        iso_folded(e, '2026-11-16T10:00:00')
        record, = records(e)
        assert record['content'] == 'lost' and record['observed_at'] is None
        assert record['observed_at_source'] == 'host_message_timestamp'
        got = json.loads(tools.lcm_grep(dict(query='lost'), engine=e))
        assert got['queued_user_record_hits'][0]['observed_at'] is None
        assert 'queued_user_records_status' not in got
        filtered = json.loads(tools.lcm_grep(dict(query='lost', time_from=1), engine=e))
        assert not filtered.get('queued_user_record_hits')


@pytest.mark.parametrize('name,step,args', [
    ('lcm_load_session', 'listing', dict(session_id='S')),
    ('lcm_expand', 'decide', dict(store_id=1)),
    ('lcm_grep', 'search_records', dict(query='lost')),
    ('lcm_recall', 'search_records', dict(query='lost')),
])
@pytest.mark.parametrize('stage', ['records', 'bounds', 'cap'])
def test_unexpected_records_exception_preserves_main(tmp_path, monkeypatch, caplog, name, step, args, stage):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        handler = getattr(tools, name)
        original = expand_tools.lcm_expand.__wrapped__ if name == 'lcm_expand' else handler.__wrapped__
        base = json.loads(original(args, engine=e))
        def fail(*args, **kwargs):
            raise RuntimeError('private record content must never be logged')
        if stage == 'records':
            monkeypatch.setattr(q, step, fail)
        elif stage == 'bounds':
            slicer = '_slice_content_for_response' if name == 'lcm_expand' else '_slice_loaded_content'
            original_slice = getattr(tools, slicer)
            def slice_records(content, *args, **kwargs):
                return fail() if content == 'lost' else original_slice(content, *args, **kwargs)
            monkeypatch.setattr(tools, slicer, slice_records)
        else:
            if name not in ('lcm_grep', 'lcm_recall'):
                # Exercise final serialization for tools without a response cap.
                target = 'queued_user_records'
            else:
                target = 'queued_user_record_hits'
            original_dumps = json.dumps
            def dump_records(value, *args, **kwargs):
                return fail() if isinstance(value, dict) and target in value else original_dumps(value, *args, **kwargs)
            monkeypatch.setattr(json, 'dumps', dump_records)
        got = json.loads(handler(args, engine=e))
        assert got == dict(base, queued_user_records_status=dict(available=False, reason='read_failed'))
        warnings = [r.getMessage() for r in caplog.records if r.name == q.__name__ and r.levelname == 'WARNING']
        assert warnings == ['queued records read failed: RuntimeError']
        assert 'private record content' not in caplog.text


def test_non_ascii_hits_use_unescaped_response_cap(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e, ('needle 界' * 20, 'needle 語' * 20, 'newest'))
        args = dict(query='needle')
        unbounded = json.loads(tools.lcm_grep(args, engine=e))
        hits = unbounded['queued_user_record_hits']
        assert len(hits) == 2
        cap = len(json.dumps(unbounded, ensure_ascii=False)) + 1
        assert cap < len(json.dumps(unbounded))
        monkeypatch.setattr(tools, '_LCM_GREP_RESPONSE_CHAR_CAP', cap)
        got = json.loads(tools.lcm_grep(args, engine=e))
        assert got['queued_user_record_hits'] == hits
        assert 'queued_user_records_omitted' not in got
        assert len(json.dumps(got, ensure_ascii=False)) <= cap


@pytest.mark.parametrize('raw', ['not json', '[]', 'null', '123', '"text"'])
def test_non_object_main_response_is_returned_unchanged(tmp_path, raw):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        handler = q.surface('grep')(lambda args, **kwargs: raw)
        assert handler(dict(query='lost'), engine=e) == raw
