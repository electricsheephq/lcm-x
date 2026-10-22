"""E1a: cited entity reads, immutable database and default-path byte guards."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from hermes_lcm import tools
from hermes_lcm.assertion_store import AssertionCandidate
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

BASE = Path(__file__).with_name('fixtures') / 'e1a_default_bytes.json'


def make_engine(tmp_path, *, entities=True, assertions=True):
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'),
                       embeddings_enabled=False, assertions_enabled=assertions)
    # Also constructs on origin/main for the baseline/red record.
    config.entities_enabled = entities
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / 'home'))
    engine.on_session_start('one', conversation_id='conv-one')
    return engine


def seed(engine):
    ids = []
    for session, content, observed in [
        ('one', 'Alex approved the release budget.', 1700000000),
        ('one', 'Lex likes tea.', None),
        ('two', 'Alex moved the release to Friday.', 1700000200),
        ('two', 'Blair alone manages the unrelated budget.', None),
    ]:
        sid = engine._store.append(session, {'role': 'user', 'content': content})
        engine._store.connection.execute(
            'UPDATE messages SET timestamp=?, observed_at=?, observed_at_source=? WHERE store_id=?',
            (1700000100 + len(ids) * 10, observed, 'host' if observed else None, sid))
        ids.append(sid)
    engine._store.connection.commit()
    if engine._assertions is not None:
        for sid, subject in [(ids[0], 'person: ＡＬＥＸ! '), (ids[1], 'user:lex'),
                             (ids[2], 'project:alex'), (ids[0], 'event:alex'),
                             (ids[0], 'conversation:alex'), (ids[0], 'future:alex')]:
            snapshot = engine._assertions.snapshot_source(sid)
            # One source publication may contain several namespace candidates.
            existing = getattr(engine, '_e1a_candidates', {})
            existing.setdefault(sid, []).append(AssertionCandidate(
                0, len(snapshot.content), subject, 'note', value_text=snapshot.content))
            engine._e1a_candidates = existing
        for sid, candidates in engine._e1a_candidates.items():
            engine._assertions.publish_source(engine._assertions.snapshot_source(sid), candidates)
    return ids


@pytest.fixture
def seeded(tmp_path):
    engine = make_engine(tmp_path)
    ids = seed(engine)
    try:
        yield engine, ids
    finally:
        engine.shutdown()


def call(engine, action='lookup', **args):
    return json.loads(tools.lcm_memory({'action': action, **args}, engine=engine))


def digest(engine):
    return hashlib.sha256('\n'.join(engine._store.connection.iterdump()).encode()).hexdigest()


def test_lookup_card_grouping_time_and_citations(seeded):
    engine, ids = seeded
    card = call(engine, name='Alex', aliases=['Lex', 'Alex'])
    assert card['names_searched'] == ['Alex', 'Lex']
    assert card['mention_count'] == 3 and card['truncated'] is False
    assert card['first_seen'] == {'time': '2023-11-14T22:13:20Z',
                                  'time_source': 'event_time', 'store_id': ids[0]}
    assert card['last_seen']['store_id'] == ids[2]
    assert {s['session_id']: s['count'] for s in card['sessions']} == {'one': 2, 'two': 1}
    snippets = [x for s in card['sessions'] for x in s['snippets']]
    assert {s['store_id'] for s in snippets} == set(ids[:3])
    fallback = next(s for s in snippets if s['store_id'] == ids[1])
    assert fallback['time_source'] == 'write_timestamp'
    assert all(len(s['snippets']) <= 3 for s in card['sessions'])
    assert {f['subject_key'] for f in card['facts']} == {
        'person: ＡＬＥＸ!', 'user:lex', 'project:alex'}
    assert all(f['store_id'] == f['source_store_id'] and f['observed_at'] for f in card['facts'])
    assert card['ambiguous'] is True


def test_question_is_ranked_only_within_matching_rows(seeded):
    engine, ids = seeded
    card = call(engine, name='Alex', aliases=['Lex'], question='budget')
    assert [m['store_id'] for m in card['mentions']] == [ids[0]]
    assert all(m['store_id'] != ids[3] for m in card['mentions'])
    expected = engine._store.search('release', sort='relevance', matching_names=['Alex'], limit=5)
    assert [m['store_id'] for m in call(engine, name='Alex', question='release')['mentions']] == [
        r['store_id'] for r in expected]
    # The question's best hit must survive a card limit that omitted its row.
    for name in ('Zephyr', '東京'):
        engine._store.append('one', {'role': 'user', 'content': name})
        target = engine._store.append('two', {'role': 'user', 'content': name + ' approved Ganymede.'})
        answer = call(engine, name=name, question='Ganymede', limit=1)
        assert answer['truncated'] is True
        assert [m['store_id'] for m in answer['mentions']] == [target]


def test_list_and_database_are_read_only(seeded):
    engine, ids = seeded
    before = digest(engine)
    call(engine, name='Alex', aliases=['Lex'], question='release')
    result = call(engine, 'list')
    assert digest(engine) == before
    assert {e['namespace'] for e in result['entities']} == {
        'person', 'user', 'project', 'event', 'conversation', 'future'}
    assert all(e['count'] == 1 and e['store_id'] in ids for e in result['entities'])
    assert len(call(engine, 'list', limit=1)['entities']) == 1


def test_assertions_off_and_cached_disabled_call(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        seed(engine)
        card = call(engine, name='Alex')
        assert card['facts'] == [] and card['facts_status'] == 'assertions_disabled'
        assert card['ambiguous'] is False
        assert call(engine, 'list') == {'entities': [], 'status': 'no_entity_registry_until_e2'}
        engine._config.entities_enabled = False
        assert call(engine, name='Alex')['status'] == 'disabled'
        assert 'lcm_memory' not in {s['name'] for s in engine.get_tool_schemas()}
        assert json.loads(engine.handle_tool_call('lcm_memory', {'action': 'list'}))['status'] == 'disabled'
    finally:
        engine.shutdown()


def test_caps_deduplication_and_validation(seeded):
    engine, ids = seeded
    for i in range(60):
        engine._store.append('many', {'role': 'user', 'content': f'Alex Lex message {i}'})
    card = call(engine, name='Alex', aliases=['Lex'], limit=500)
    assert card['mention_count'] == 50 and card['truncated'] is True
    assert call(engine, name='Alex')['mention_count'] == 20
    assert call(engine, name='missing')['first_seen'] is None
    assert call(engine, name='Alex', aliases=['x'] * 9)['error']
    assert call(engine, name='Alex', aliases='Lex')['error']
    for args in ({'name': ''}, {'name': 'Alex', 'limit': 0}, {'name': 'Alex', 'limit': True}):
        assert call(engine, **args)['error']
    assert call(engine, 'remember', name='Alex')['error']
    single = call(engine, name='Lex')
    assert single['ambiguous'] is False
    assert len({f['store_id'] for f in single['facts']}) == 1


def test_sql_cost_is_bounded_without_per_row_queries(seeded):
    engine, _ = seeded
    for i in range(100):
        engine._store.append('many', {'role': 'user', 'content': f'Alex Lex release {i}'})
    statements = []
    engine._store.connection.set_trace_callback(statements.append)
    engine._assertions.connection.set_trace_callback(statements.append)
    try:
        call(engine, name='Alex', aliases=['Lex'], question='release', limit=50)
    finally:
        engine._store.connection.set_trace_callback(None)
        engine._assertions.connection.set_trace_callback(None)
    selects = [s for s in statements if s.lstrip().upper().startswith(('SELECT', 'WITH'))]
    assert len(selects) <= 8, selects
    assert not any('WHERE store_id =' in s for s in selects)


def default_bytes(engine):
    recall = tools.lcm_recall({'query': 'Alex', 'include': 'verbatim'}, engine=engine)
    assembled = engine._assemble_context({'role': 'system', 'content': 'Fixture system'},
        [{'role': 'user', 'content': 'Fixture tail'}], persist=False)
    return {'schemas': json.dumps(engine.get_tool_schemas()), 'recall': recall,
            'context': json.dumps(assembled)}


def test_default_and_unused_enabled_bytes_match_origin_main(tmp_path):
    engine = make_engine(tmp_path, entities=False, assertions=False)
    try:
        seed(engine)
        baseline = default_bytes(engine)
        if os.environ.get('E1A_CAPTURE_BASELINE'):
            BASE.write_text(json.dumps(baseline, indent=2) + '\n')
            return
        assert baseline == json.loads(BASE.read_text())
        engine._config.entities_enabled = True
        enabled = default_bytes(engine)
        assert enabled['context'] == baseline['context']
        assert enabled['recall'] == baseline['recall']
        schemas = json.loads(enabled['schemas'])
        memory = schemas.pop()
        assert memory['name'] == 'lcm_memory'
        assert json.dumps(schemas) == baseline['schemas']
        assert len(json.dumps(memory).encode()) > 0
    finally:
        engine.shutdown()
