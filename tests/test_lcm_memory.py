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


# fix-r1f transport/snapshot regressions: RED on the frozen fix-r1e candidate.
@pytest.mark.parametrize('name', ['alex', 'Ａｌｅｘ'])
def test_r1f_native_metadata_transport(tmp_path, name):
    engine = make_engine(tmp_path, assertions=False)
    try:
        sid = engine._store.append('one', {'role': 'user', 'content': name})
        statements, metadata = [], {'rows': []}
        engine._store.connection.set_trace_callback(statements.append)
        try:
            rows = engine._store.search('"' + name + '"', sort='relevance', _memory=metadata)
        finally:
            engine._store.connection.set_trace_callback(None)
        assert [r['store_id'] for r in rows] == [sid]
        assert [r['store_id'] for r in metadata['rows']] == [sid]
        selects = [s for s in statements if s.lstrip().upper().startswith(('SELECT', 'WITH'))]
        assert len(selects) == 2, selects
        native = [s for s in selects if not any(
            word in s.lower() for word in ('json_object', 'printf', 'json_group_array'))]
        match = [s for s in native if 'LIMIT' not in s.upper()]
        assert len(match) == 1, selects
        assert 'store_id' in match[0] and 'observed_at' in match[0]
    finally:
        engine.shutdown()


def _assert_snapshot_released(store):
    from threading import Thread

    assert not store.connection.in_transaction
    acquired = []

    def probe():
        held = store._write_lock.acquire(blocking=False)
        acquired.append(held)
        if held:
            store._write_lock.release()

    thread = Thread(target=probe)
    thread.start()
    thread.join(timeout=1)
    assert acquired == [True]


@pytest.mark.parametrize('name', ['alex', 'Ａｌｅｘ'])
def test_r1f_missing_display_metadata_returns_error_and_releases(tmp_path, monkeypatch, name):
    engine = make_engine(tmp_path, assertions=False)
    try:
        engine._store.append('one', {'role': 'user', 'content': name})
        original = engine._store.search

        def drop(query, **kwargs):
            rows = original(query, **kwargs)
            if '_memory' in kwargs and rows:
                displayed = rows[0]['store_id']
                kwargs['_memory']['rows'] = [r for r in kwargs['_memory']['rows']
                                             if r['store_id'] != displayed]
            return rows

        monkeypatch.setattr(engine._store, 'search', drop)
        assert call(engine, name=name) == {'error': 'Memory lookup metadata is inconsistent'}
        _assert_snapshot_released(engine._store)
    finally:
        engine.shutdown()


def test_r1f_exception_releases_snapshot_and_lock(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, assertions=False)
    try:
        state = []

        class Interrupted(BaseException):
            pass

        def interrupt(*args, **kwargs):
            state.append((engine._store.connection.in_transaction,
                          engine._store._write_lock._is_owned()))
            raise Interrupted()

        monkeypatch.setattr(engine._store, 'search', interrupt)
        with pytest.raises(Interrupted):
            call(engine, name='alex')
        _assert_snapshot_released(engine._store)
        assert state == [(True, True)]
    finally:
        engine.shutdown()


def test_r1f_snapshot_covers_forms_question_and_carried_read(tmp_path, monkeypatch):
    import sqlite3
    from hermes_lcm import store as store_module

    engine = make_engine(tmp_path, assertions=False)
    writer = None
    try:
        engine._store.append('one', {'role': 'user', 'content': 'alex release'})
        blair = engine._store.append('two', {'role': 'user', 'content': 'blair release'})
        engine._store._ensure_identity_anchor_schema()
        conn = engine._store.connection
        conn.execute('UPDATE messages SET timestamp=1700000000, observed_at=NULL')
        conn.commit()
        expected = call(engine, name='alex', aliases=['blair'], question='release')
        writer = sqlite3.connect(str(tmp_path / 'lcm.db'))
        original = store_module._fetch_ranked_search_rows
        writes = []

        def concurrent_write(*args, **kwargs):
            rows = original(*args, **kwargs)
            if not writes:
                writer.execute("UPDATE messages SET content='gone', timestamp=1800000000 WHERE store_id=?",
                               (blair,))
                writer.execute("INSERT INTO message_relations(store_id,kind,related_store_id,observed_at,created_at) "
                               "VALUES (?, 'carried_stamp', 1, 1690000000, 1)", (blair,))
                writer.commit()
                writes.append(True)
            return rows

        monkeypatch.setattr(store_module, '_fetch_ranked_search_rows', concurrent_write)
        assert call(engine, name='alex', aliases=['blair'], question='release') == expected
        assert writes == [True]
        _assert_snapshot_released(engine._store)
        assert call(engine, name='blair')['mention_count'] == 0
    finally:
        if writer is not None:
            writer.close()
        engine.shutdown()


# Control: an enclosing transaction belongs to the caller on success and error.
def test_r1f_existing_transaction_is_retained(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, assertions=False)
    try:
        engine._store.append('one', {'role': 'user', 'content': 'alex'})
        conn = engine._store.connection
        conn.execute('BEGIN')
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            assert call(engine, name='alex')['mention_count'] == 1

            def interrupt(*args, **kwargs):
                raise ValueError('lookup interrupted')

            monkeypatch.setattr(engine._store, 'search', interrupt)
            with pytest.raises(ValueError, match='lookup interrupted'):
                call(engine, name='alex')
        finally:
            conn.set_trace_callback(None)
        assert conn.in_transaction
        assert not engine._store._write_lock._is_owned()
        assert not any(s.lstrip().upper().startswith(('BEGIN', 'COMMIT', 'ROLLBACK')) for s in statements)
        conn.rollback()
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
    assert card['mention_count'] == 63 and card['truncated'] is True
    assert call(engine, name='Alex')['mention_count'] == 62
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
    assert len(selects) <= 12, selects  # Prior bound 8 + two metadata reads + carried read/cold probe.
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


# fix-r1: the review's four reproduced behaviors and its bounded guards only.
def mention_ids(card):
    return sorted(m['store_id'] for session in card['sessions'] for m in session['snippets'])


def test_first_last_cover_all_500_mentions(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = []
        for i in range(500):
            sid = engine._store.append('many', {'role': 'user', 'content': 'Alex release'})
            engine._store.connection.execute(
                'UPDATE messages SET timestamp=?, observed_at=? WHERE store_id=?',
                (1700001000 + i, 1700000000 + i, sid))
            ids.append(sid)
        engine._store.connection.commit()
        card = call(engine, name='Alex', limit=20)
        assert card['truncated'] and card['mention_count'] == 500
        assert (card['first_seen']['time'], card['last_seen']['time']) == (
            '2023-11-14T22:13:20Z', '2023-11-14T22:21:39Z')
        assert (card['first_seen']['store_id'], card['last_seen']['store_id']) == (ids[0], ids[-1])
    finally:
        engine.shutdown()


def test_punctuation_strip_reaches_fixed_point(seeded):
    engine, _ = seeded
    expected = call(engine, name='Alex')['facts']
    assert expected
    assert call(engine, name='alex . .')['facts'] == expected


def test_carried_stamp_card_times(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        previous = engine._store.append('one', {'role': 'user', 'content': 'Previous turn.',
                                               'timestamp': 1690000000})
        carried = engine._store.append('one', {'role': 'user', 'content': 'Quinn carried turn.',
                                              'timestamp': 1690000000})
        engine._store._ensure_identity_anchor_schema()
        engine._store.connection.execute(
            "INSERT INTO message_relations(store_id,kind,related_store_id,observed_at,created_at) "
            "VALUES (?, 'carried_stamp', ?, 1690000000, 1)", (carried, previous))
        engine._store.connection.commit()
        assert engine._store.get_carried_stamp_ids([carried]) == {carried}
        card = call(engine, name='Quinn')
        entries = [card['first_seen'], card['last_seen']]
        for session in card['sessions']:
            entries.extend([session['first_seen'], session['last_seen'], *session['snippets']])
        assert all(entry.get('time_uncertain') == 'carried' for entry in entries)
    finally:
        engine.shutdown()


def test_d3_spellings_preserve_raw_and_normalised_mentions(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        for session, text in [('one', 'Alex approved the release budget.'),
                              ('one', 'I told ALEX about it.'), ('two', 'alex. said hi'),
                              ('two', 'Blair manages the budget.'),
                              ('two', 'Ａｌｅｘ wrote in fullwidth.')]:
            engine._store.append(session, {'role': 'user', 'content': text})
        # Pinned d10777eb raw IDs; the normalized form finds ASCII rows only.
        raw_ids = {'Ａｌｅｘ': {5}, 'Alex': {1, 2, 3}, 'ALEX': {1, 2, 3},
                   ' alex. ': {1, 2, 3}}
        normalized = set(mention_ids(call(engine, name='alex')))
        assert normalized == {1, 2, 3}
        for value, baseline in raw_ids.items():
            actual = set(mention_ids(call(engine, name=value)))
            assert actual >= normalized | baseline, (value, actual)
        for value in ('Straße', 'ﬁle'):
            sid = engine._store.append('raw', {'role': 'user', 'content': value})
            assert sid in mention_ids(call(engine, name=value))
    finally:
        engine.shutdown()


def test_already_normalised_mentions_guard(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = seed(engine)
        engine._store.append('two', {'role': 'user', 'content': 'Ａｌｅｘ wrote in fullwidth.'})
        assert mention_ids(call(engine, name='alex')) == [ids[0], ids[2]]
    finally:
        engine.shutdown()



def _n2_base_store():
    """Execute the exact d10777eb methods, without requiring CI git history."""
    from hermes_lcm import store

    raw = (BASE.parent / 'e1a_d10777eb_search.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '7b4f18720f511d13217daf776b0051534e1489830f3c7a7281b3057a03ab4931'
    frozen = json.loads(raw)
    assert frozen['commit'] == 'd10777ebfd7454ec97cf08267ad9d3b567550b40'
    namespace = dict(vars(store))
    for source in frozen['methods'].values():
        exec(compile(source, '<d10777eb/store.py>', 'exec'), namespace)
    return type('D10777ebSearch', (store.MessageStore,),
                {name: namespace[name] for name in frozen['methods']})


def _n2_stages(records):
    counts = dict.fromkeys(('mention', 'metadata', 'carried', 'probe', 'assertions', 'question'), 0)
    for stage, sql in records:
        text = ' '.join(sql.split())
        if text.startswith('SELECT m.store_id,m.session_id,m.timestamp,m.observed_at FROM ') or text.startswith(
                'SELECT store_id,session_id,timestamp,observed_at,content FROM messages WHERE '):
            assert stage == 'mention', (stage, text)
            key = 'metadata'
        elif text.startswith('WITH ranked AS NOT MATERIALIZED (') or (
                text.startswith('SELECT store_id, session_id, source, role, content,') and ' LIMIT ' in text):
            assert stage in ('mention', 'question'), (stage, text)
            key = stage
        elif text.startswith('SELECT store_id FROM ') and "WHERE kind='carried_stamp'" in text and 'json_each(' in text:
            assert stage == 'lookup', (stage, text)
            key = 'carried'
        elif text == "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'":
            assert stage == 'lookup', (stage, text)
            key = 'probe'
        elif text.startswith('WITH active AS (SELECT a.*, m.content AS current_content,'):
            assert stage == 'assertions', (stage, text)
            key = 'assertions'
        else:
            raise AssertionError(('unclassified statement', stage, text))
        counts[key] += 1
    return counts


def _n2_assert(records, expected):
    actual = _n2_stages(records)
    assert actual == expected, (actual, expected)
    return actual


def test_sql_cost_max_input_and_mention_growth(tmp_path, monkeypatch):
    from hermes_lcm import memory_tools

    engine = make_engine(tmp_path)
    store = engine._store
    native, baseline = type(store), _n2_base_store()
    try:
        for name, count in [('quillon', 10), ('marisol', 100), ('thaddeus', 500)]:
            for _ in range(count):
                store.append('one', {'role': 'user', 'content': f'{name} budget'})
        cases = [{'name': name} for name in ('quillon', 'marisol', 'thaddeus')]
        for n in (1, 9):
            names = ['thaddeus', 'thad', 'marisol', 'quillon', 'zzz1', 'zzz2', 'zzz3', 'zzz4', 'zzz5'][:n]
            for differing in (False, True):
                values = [''.join(chr(ord(c) + 0xFEE0) for c in v) for v in names] if differing else names
                cases.append({'name': values[0], 'aliases': values[1:], 'question': 'budget', 'limit': 50})
        receipts = []
        for args in cases:
            records, controls, searches = [], [], []
            active = ['lookup']
            original = store.search

            def trace(sql, origin=None):
                if sql.lstrip().upper().startswith(('SELECT', 'WITH')):
                    records.append((origin or active[0], sql))
                elif sql in ('BEGIN', 'COMMIT', 'ROLLBACK'):
                    controls.append(sql)

            def search(query, **options):
                stage = 'mention' if '_memory' in options else 'question'
                searches.append((stage, query, dict(options)))
                active[0] = stage
                try:
                    return original(query, **options)
                finally:
                    active[0] = 'lookup'

            store._memory_relation_connections = {}
            for state in ('cold', 'warm'):
                records.clear()
                controls.clear()
                searches.clear()
                store.connection.set_trace_callback(trace)
                engine._assertions.connection.set_trace_callback(lambda sql: trace(sql, 'assertions'))
                with monkeypatch.context() as patch:
                    patch.setattr(store, 'search', search)
                    try:
                        assert 'error' not in call(engine, **args)
                    finally:
                        store.connection.set_trace_callback(None)
                        engine._assertions.connection.set_trace_callback(None)
                head_records, head_controls = list(records), list(controls)
                assert head_controls == ['BEGIN', 'COMMIT']
                base_counts = {'mention': 0, 'question': 0}
                base_traces = {}
                store.__class__ = baseline
                try:
                    for stage, query, options in searches:
                        options.pop('_memory', None)
                        captured = []
                        store.connection.set_trace_callback(lambda sql: captured.append(sql) if sql.lstrip().upper().startswith(('SELECT', 'WITH')) else None)
                        try:
                            store.search(query, **options)
                        finally:
                            store.connection.set_trace_callback(None)
                        base_counts[stage] += len(captured)
                        base_traces.setdefault(stage, []).extend(captured)
                finally:
                    store.__class__ = native
                # Ranked statements, including inherited paging, equal the base at identical inputs.
                for stage in ('mention', 'question'):
                    ranked = [sql for label, sql in head_records if label == stage and
                              _n2_stages([(label, sql)])[stage]]
                    assert [' '.join(sql.split()) for sql in ranked] == [
                        ' '.join(sql.split()) for sql in base_traces.get(stage, [])], stage
                expected = {'mention': base_counts['mention'],
                            'metadata': sum(stage == 'mention' for stage, _, _ in searches),
                            'carried': 1, 'probe': int(state == 'cold'),
                            'assertions': 1, 'question': base_counts['question']}
                actual = _n2_assert(head_records, expected)
                if args.get('question'):
                    n = 1 + len(args['aliases'])
                    differing = args['name'] != memory_tools._name(args['name'])
                    warm_total = {(1, False): 8, (1, True): 10, (9, False): 24, (9, True): 42}[n, differing]
                    assert actual['mention'] == n * (2 if differing else 1)
                    assert len(head_records) == warm_total + (state == 'cold')
                receipts.append({'args': args, 'state': state, 'stages': actual,
                                 'controls': head_controls, 'base_ranked': base_counts,
                                 'inherited_mention_pages': base_counts['mention'] - expected['metadata']})
                metadata_index = next(i for i, item in enumerate(head_records) if _n2_stages([item])['metadata'])
                # Stage-2 +1/-1 mutants both fail the same classifier/assertion.
                for mutant in (head_records + [head_records[metadata_index]],
                               head_records[:metadata_index] + head_records[metadata_index + 1:]):
                    with pytest.raises(AssertionError):
                        _n2_assert(mutant, expected)
                with pytest.raises(AssertionError, match='unclassified statement'):
                    _n2_assert(head_records + [('lookup', 'SELECT 8675309')], expected)
        print('N2-S receipt:', json.dumps(receipts))
    finally:
        store.__class__ = native
        engine.shutdown()


def test_fresh_env_identity_and_schema_size(tmp_path, monkeypatch):
    outputs = []
    for flag in (None, '1'):
        if flag is None:
            monkeypatch.delenv('LCM_ENTITIES_ENABLED', raising=False)
        else:
            monkeypatch.setenv('LCM_ENTITIES_ENABLED', flag)
        config = LCMConfig.from_env()
        assert config.entities_enabled is (flag == '1')
        config.database_path = str(tmp_path / str(flag) / 'lcm.db')
        config.embeddings_enabled = False
        config.assertions_enabled = False
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / str(flag) / 'home'))
        engine.on_session_start('one', conversation_id='conv-one')
        try:
            seed(engine)
            output = default_bytes(engine)
            output['grep'] = tools.lcm_grep({'query': 'Alex', 'session_scope': 'all'}, engine=engine)
            output['expand'] = tools.lcm_expand({'store_id': 1}, engine=engine)
            schemas = json.loads(output.pop('schemas'))
            if flag == '1':
                memory = schemas.pop()
                assert memory['name'] == 'lcm_memory'
                assert len(json.dumps(memory).encode()) == 863
            output['schemas'] = json.dumps(schemas)
            outputs.append(output)
        finally:
            engine.shutdown()
    assert outputs[0] == outputs[1]
    assert {k: outputs[0][k] for k in ('schemas', 'recall', 'context')} == json.loads(BASE.read_text())


@pytest.mark.parametrize('value', ['Straße', 'ﬁle'])
def test_raw_forms_keep_matches_after_normalisation(tmp_path, value):
    engine = make_engine(tmp_path, assertions=False)
    try:
        raw = engine._store.append('raw', {'role': 'user', 'content': value})
        normalized = 'strasse' if value == 'Straße' else 'file'
        norm = engine._store.append('norm', {'role': 'user', 'content': normalized})
        assert set(mention_ids(call(engine, name=value))) == {raw, norm}
        assert mention_ids(call(engine, name=normalized)) == [norm]
    finally:
        engine.shutdown()


def test_union_dedup_order_limit_and_full_aggregates(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = []
        for i, text in enumerate(['Ａｌｅｘ', 'alex', 'Ａｌｅｘ alex', 'alex', 'Ａｌｅｘ', 'alex']):
            sid = engine._store.append('union', {'role': 'user', 'content': text})
            engine._store.connection.execute(
                'UPDATE messages SET timestamp=?, observed_at=? WHERE store_id=?',
                (1700000000 + i, 1700000000 + i, sid))
            ids.append(sid)
        # A session entirely outside the selected snippets must still be counted.
        extra = engine._store.append('other', {'role': 'user', 'content': 'alex'})
        engine._store.connection.execute(
            'UPDATE messages SET timestamp=?, observed_at=? WHERE store_id=?',
            (1700000100, 1700000100, extra))
        engine._store.connection.commit()
        hits = {}
        for route, value in enumerate(('alex', 'Ａｌｅｘ')):
            for row in engine._store.search('"' + value + '"', limit=3, sort='relevance'):
                key = (route, row.get('search_rank') or 0, row['store_id'])
                if row['store_id'] not in hits or key < hits[row['store_id']][0]:
                    hits[row['store_id']] = (key, row)
        # Amendment 4: FTS first, best rank within a route, then store_id.
        expected = [row for _, row in sorted(hits.values())]
        card = call(engine, name='Ａｌｅｘ', aliases=['Ａｌｅｘ', 'alex'], limit=2)
        assert card['mention_count'] == 7
        assert card['truncated'] is True
        assert sorted(mention_ids(card)) == sorted(r['store_id'] for r in expected[:2])
        assert {s['session_id']: s['count'] for s in card['sessions']} == {'union': 6, 'other': 1}
        assert card['first_seen']['store_id'] == ids[0]
        assert card['last_seen']['store_id'] == extra
        union = next(s for s in card['sessions'] if s['session_id'] == 'union')
        assert union['first_seen']['store_id'] == ids[0]
        assert union['last_seen']['store_id'] == ids[-1]
        assert call(engine, name='Ａｌｅｘ', aliases=['alex'], limit=2) == card
    finally:
        engine.shutdown()


def test_union_rank_ties_break_by_store_id(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = [engine._store.append('one', {'role': 'user', 'content': 'alex'})
               for _ in range(3)]
        engine._store.connection.execute('UPDATE messages SET timestamp=1700000000')
        engine._store.connection.commit()
        card = call(engine, name='alex', aliases=['alex', 'alex'], limit=2)
        assert [m['store_id'] for m in card['sessions'][0]['snippets']] == ids[:2]
    finally:
        engine.shutdown()


def test_equal_forms_preserve_base_counts_order_and_unaffected_aggregates(tmp_path):
    from hermes_lcm import memory_tools
    import inspect

    engine = make_engine(tmp_path, assertions=False)
    try:
        for text in ('alex', 'alex', '東京', 'missing nothing'):
            engine._store.append('one', {'role': 'user', 'content': text})
        engine._store.connection.execute(
            'UPDATE messages SET timestamp=1700000000, observed_at=NULL')
        engine._store.connection.commit()
        # Amendment 2: base one + one batched read + cold schema probe.
        # IDs below are in d10777eb card order.
        for case, (value, limit, ids) in enumerate([('alex', 20, [1, 2]), ('alex', 1, [1]),
                                  ('東京', 20, [3]), ('absent', 20, [])]):
            assert memory_tools._name(value) == value
            statements = []
            engine._store.connection.set_trace_callback(statements.append)
            try:
                card = call(engine, name=value, aliases=[value], limit=limit)
            finally:
                engine._store.connection.set_trace_callback(None)
            assert [s for s in statements if s in ('BEGIN', 'COMMIT')] == ['BEGIN', 'COMMIT']
            statements = [s for s in statements if s.lstrip().upper().startswith(('SELECT', 'WITH'))]
            assert len(statements) == (4 if case == 0 else 3)
            assert sum('json_each' in q for q in statements) == 1
            assert sum('sqlite_master' in q for q in statements) == (case == 0)
            actual_ids = [m['store_id'] for s in card['sessions'] for m in s['snippets']]
            assert actual_ids == ids
            single = []
            engine._store.connection.set_trace_callback(single.append)
            try:
                options = {'limit': limit + 1, 'sort': 'relevance'}
                # The head's optional aggregate is part of its single-form statement.
                if '_memory' in inspect.signature(engine._store._search_fts).parameters:
                    options['_memory'] = {'rows': [], 'ranked': False}
                engine._store.search('"' + value + '"', **options)
            finally:
                engine._store.connection.set_trace_callback(None)
            assert statements[:2] == [s for s in single if s.lstrip().upper().startswith(('SELECT', 'WITH'))]
            if (value, limit) != ('alex', 1):
                def stamp(sid):
                    return {'time': '2023-11-14T22:13:20Z',
                            'time_source': 'write_timestamp', 'store_id': sid}
                expected = {'mention_count': len(ids), 'truncated': False,
                            'first_seen': stamp(ids[0]) if ids else None,
                            'last_seen': stamp(ids[-1]) if ids else None}
                assert {key: card[key] for key in expected} == expected
                assert all('time_uncertain' not in entry for entry in
                           [card['first_seen'], card['last_seen']] if entry is not None)
    finally:
        engine.shutdown()


def test_union_question_uses_same_deduplicated_forms(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = [engine._store.append('one', {'role': 'user', 'content': value + ' release'})
               for value in ('Ａｌｅｘ', 'alex')]
        original = engine._store.search
        seen = []
        def track(query, **kwargs):
            seen.append((query, kwargs))
            return original(query, **kwargs)
        monkeypatch.setattr(engine._store, 'search', track)
        card = call(engine, name='Ａｌｅｘ', aliases=['Ａｌｅｘ', 'alex'], question='release')
        assert set(m['store_id'] for m in card['mentions']) == set(ids)
        assert seen[-1][1]['matching_names'] == ['Ａｌｅｘ', 'alex']
        assert [query for query, _ in seen[:-1]] == ['"Ａｌｅｘ"', '"alex"']
    finally:
        engine.shutdown()


def _traced_lookup(engine, **args):
    statements = []
    engine._store.connection.set_trace_callback(statements.append)
    try:
        card = tools.lcm_memory({'action': 'lookup', **args}, engine=engine)
    finally:
        engine._store.connection.set_trace_callback(None)
    assert [s for s in statements if s in ('BEGIN', 'COMMIT')] == ['BEGIN', 'COMMIT']
    return card, [s for s in statements if s.lstrip().upper().startswith(('SELECT', 'WITH'))]


@pytest.mark.parametrize('name', ['Alex', 'alex'])
def test_regression_ascii_skip_is_byte_identical_and_one_search(tmp_path, monkeypatch, name):
    from hermes_lcm import memory_tools

    engine = make_engine(tmp_path, assertions=False)
    try:
        for text in ('Alex release', 'alex release', 'ALEX said hi', 'Ａｌｅｘ fullwidth'):
            engine._store.append('one', {'role': 'user', 'content': text})
        call(engine, name=name)
        card, statements = _traced_lookup(engine, name=name, limit=2)
        monkeypatch.setattr(memory_tools, '_ASCII_CASE_FOLD', {}, raising=False)
        both, both_statements = _traced_lookup(engine, name=name, limit=2)
        assert card == both
        assert len(statements) == 3
        assert sum('MATCH' in sql for sql in statements) == 2
        assert len(both_statements) == (5 if name == 'Alex' else 3)
    finally:
        engine.shutdown()


@pytest.mark.parametrize('value,normalized', [('Straße', 'strasse'), ('ﬁle', 'file'), ('Ａｌｅｘ', 'alex')])
def test_control_keep_raw_and_fullwidth_run_both(tmp_path, monkeypatch, value, normalized):
    engine = make_engine(tmp_path, assertions=False)
    try:
        ids = [engine._store.append('one', {'role': 'user', 'content': text})
               for text in (value, normalized)]
        original, seen = engine._store.search, []

        def track(query, **kwargs):
            seen.append(query)
            return original(query, **kwargs)

        monkeypatch.setattr(engine._store, 'search', track)
        assert set(mention_ids(call(engine, name=value))) == set(ids)
        assert seen == ['"' + value + '"', '"' + normalized + '"']
    finally:
        engine.shutdown()


def test_regression_multiform_keeps_fts_highlighted_snippets(tmp_path):
    engine = make_engine(tmp_path, assertions=False)
    try:
        for text in ('alex release', 'blair alex release', 'blair plan'):
            engine._store.append('one', {'role': 'user', 'content': text})
        expected = {}
        for name in ('alex', 'blair'):
            for row in engine._store.search('"' + name + '"', limit=21, sort='relevance'):
                key = (row['search_rank'], row['store_id'])
                if row['store_id'] not in expected or key < expected[row['store_id']][0]:
                    expected[row['store_id']] = (key, row['snippet'])
        card = call(engine, name='alex', aliases=['blair'])
        for snippet in card['sessions'][0]['snippets']:
            assert snippet['snippet'] == expected[snippet['store_id']][1]
            assert '>>>' in snippet['snippet']
    finally:
        engine.shutdown()


def test_regression_route_first_best_rank_then_id(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, assertions=False)
    try:
        def search(query, **kwargs):
            source = {'"alex"': [(2, -0.01), (1, -0.01)], '"blair"': [(1, -0.02)],
                      '"Ａｌｅｘ"': [(3, -99)]}[query]
            rows = [{'store_id': sid, 'session_id': 'one', 'timestamp': 1700000000,
                     'observed_at': None, 'content': 'raw', 'snippet': '>>>hit<<<',
                     'search_rank': rank} for sid, rank in source]
            kwargs['_memory']['rows'] = rows
            kwargs['_memory']['like'] = query == '"Ａｌｅｘ"'
            return rows

        monkeypatch.setattr(engine._store, 'search', search)
        card = call(engine, name='alex', aliases=['blair', 'Ａｌｅｘ'])
        assert [s['store_id'] for s in card['sessions'][0]['snippets']] == [1, 2, 3]
    finally:
        engine.shutdown()


@pytest.mark.parametrize('name', ['alex', 'Ａｌｅｘ'])
def test_regression_exact_time_facts_against_plain_sql(tmp_path, name):
    # Adapted from variantB edge_keys.py: exact SQL rows are the oracle.
    from hermes_lcm import memory_tools

    engine = make_engine(tmp_path, assertions=False)
    try:
        values = [None, 1700000000.9999995, '2023-11-14T22:13:20Z',
                  '2023-11-14T22:13:20.9999996+00:00', '2023-11-14 22:13:20',
                  '   ', 'garbage', -5.0, 0, 1e12, 253402300799.5,
                  253402300798.25, '1690000050.5']
        conn = engine._store.connection
        for i, observed in enumerate(values):
            # Reverse IDs so JSON-rounded ties select the wrong extrema too.
            for offset in (0.9999998, 0.9999993):
                sid = engine._store.append('edge' + str(i), {'role': 'user', 'content': name})
                conn.execute('UPDATE messages SET timestamp=?, observed_at=? WHERE store_id=?',
                             (1700000000 + offset, observed, sid))
        engine._store._ensure_identity_anchor_schema()
        carried = {2, 4, sid}
        for store_id in carried:
            conn.execute("INSERT INTO message_relations(store_id,kind,related_store_id,observed_at,created_at) "
                         "VALUES (?, 'carried_stamp', 1, 1690000000, 1)", (store_id,))
        conn.commit()
        columns = ('store_id', 'session_id', 'timestamp', 'observed_at')
        rows = [dict(zip(columns, row)) for row in conn.execute(
            'SELECT store_id,session_id,timestamp,observed_at FROM messages')]
        for row in rows:
            if row['store_id'] in carried:
                row['time_uncertain'] = 'carried'

        def key(row):
            event = tools._normalize_observed_at(row['observed_at'])
            return event if event is not None else row['timestamp'], row['store_id']

        def bounds(group):
            return memory_tools._time(min(group, key=key)), memory_tools._time(max(group, key=key))

        card = call(engine, name=name, limit=1)
        assert card['mention_count'] == len(rows) and card['truncated'] is True
        assert (card['first_seen'], card['last_seen']) == bounds(rows)
        for session in card['sessions']:
            group = [r for r in rows if r['session_id'] == session['session_id']]
            assert session['count'] == len(group)
            assert (session['first_seen'], session['last_seen']) == bounds(group)
            for snippet in session['snippets']:
                exact = next(r for r in rows if r['store_id'] == snippet['store_id'])
                assert {k: snippet[k] for k in memory_tools._time(exact)} == memory_tools._time(exact)
    finally:
        engine.shutdown()
