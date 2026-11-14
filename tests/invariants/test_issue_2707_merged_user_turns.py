"""I1/I2: #2707 Design B, paired with exact main and a real v0.28.0 reader.

Shapes port Phase B probe F: Q, queued P1..Pn/R, host repair, live-list
recreation, next turn, transcript reload, rollback and re-upgrade. No host I/O.
"""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

MAIN = '647cecaac513827c2709ce7ec285838fdee479ba'
ROOT = Path(__file__).resolve().parents[2]


def _load(tree, name):
    spec = importlib.util.spec_from_file_location(name, tree / '__init__.py',
                                                submodule_search_locations=[str(tree)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def versions(tmp_path_factory):
    result = {'candidate': _load(ROOT, '_issue2707_candidate')}
    for name, ref in [('main', MAIN), ('rollback', 'v0.28.0'), ('previous_ga', 'v0.30.0')]:
        tree = tmp_path_factory.mktemp('issue2707-' + name)
        files = subprocess.check_output(['git', 'ls-tree', '--name-only', ref], cwd=ROOT,
                                        text=True).splitlines()
        files = [f for f in files if f.endswith('.py')]
        archive = subprocess.check_output(['git', 'archive', ref, '--', *files], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as packed:
            packed.extractall(tree, filter='data')
        result[name] = _load(tree, '_issue2707_' + name)
    return result


def _row(text, stamp, role='user'):
    return {'role': role, 'content': text, 'timestamp': stamp}


def _open(module, directory, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(directory / 'home'))
    monkeypatch.setenv('LCM_DATABASE_PATH', str(directory / 'lcm.db'))
    monkeypatch.setenv('LCM_EMBEDDINGS_ENABLED', '0')
    monkeypatch.setenv('LCM_BACKGROUND_LEAF_PREP_ENABLED', '0')
    ctx = SimpleNamespace(hooks={})
    ctx.register_context_engine = lambda engine: setattr(ctx, 'engine', engine)
    ctx.register_hook = lambda name, callback: ctx.hooks.__setitem__(name, callback)
    module.register(ctx)
    engine = ctx.engine
    monkeypatch.setattr(sys.modules[engine.__module__], 'summarize_with_escalation',
                        lambda **_kwargs: ('Earlier turns.', 1))
    engine.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
    return ctx


def _rows(engine):
    return engine._store.get_session_messages('parent')


def _identities(engine):
    return [(r['role'], r['content'], r['observed_at'], r['session_id'], r['conversation_id'])
            for r in _rows(engine)]


def _ordered_coverage(rows, texts):
    """Joined rows count every sent occurrence, including X/Y/X, in order."""
    stream = '\n\n'.join(r[1] for r in rows)
    offset = 0
    for text in texts:
        index = stream.find(text, offset)
        if index < 0:
            return False
        offset = index + len(text)
    return True


def _scenario(module, rollback, directory, monkeypatch, texts, stamp, prefix, fault):
    ctx = _open(module, directory, monkeypatch)
    engine = ctx.engine
    q = _row('Q', 9., 'assistant')
    queued = [_row(text, float(10 * (i + 1))) for i, text in enumerate(texts)]
    a = _row('A', float(10 * len(texts) + 1), 'assistant')
    transcript = [q, _row('\n\n'.join(texts), 10.), a]
    stages = {}
    try:
        engine.ingest([q])
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                  user_message=texts[-1], conversation_history=[q, *queued])
        assert _identities(engine) == [('assistant', 'Q', 9., 'parent', 'conv')]
        head = queued[0]
        if prefix:
            head['_merged_turn_prefix'] = '\n\n'.join(texts[:-1]) + '\n\n'
            head['content'] = '\n\n'.join(texts)
        else:
            head['content'] = texts[-1]
        if stamp:
            head['timestamp'] = queued[-1]['timestamp']
        live = [q, head, a]
        if fault:
            def boom(*_args, **_kwargs):
                raise RuntimeError('injected anchor pre-match fault')
            monkeypatch.setattr(engine, '_identity_anchor_prematch', boom)
        engine.ingest(live)
        stages['warm'] = _identities(engine)
        engine.ingest(live)
        stages['warm-replay'] = _identities(engine)
        engine.on_session_end('parent', live)
    finally:
        engine.shutdown()
    ctx = _open(module, directory, monkeypatch)
    engine = ctx.engine
    try:
        if fault:
            monkeypatch.setattr(engine, '_identity_anchor_prematch', boom)
        engine.ingest(deepcopy(live))
        stages['recreate-live'] = _identities(engine)
        n, b = _row('N', 100.), _row('B', 101., 'assistant')
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='N',
                                  conversation_history=[*live, n])
        live += [n, b]
        transcript += [n, b]
        engine.ingest(live)
        stages['next-turn'] = _identities(engine)
    finally:
        engine.shutdown()
    # A fresh process cannot carry the capture: new engine and fresh transcript
    # objects use the durable head stamp that probe F's transcript view restores.
    for label, version in [('restart-transcript', module), ('rollback', rollback), ('re-upgrade', module)]:
        ctx = _open(version, directory, monkeypatch)
        engine = ctx.engine
        try:
            if fault:
                monkeypatch.setattr(engine, '_identity_anchor_prematch', boom)
            engine.ingest(deepcopy(transcript))
            stages[label] = _identities(engine)
            if label == 'rollback':
                engine.ingest(deepcopy(live))
                stages['rollback-newest'] = _identities(engine)
                turn = [_row('M', 110.), _row('C', 111., 'assistant')]
                live += turn
                transcript += turn
                engine.ingest(deepcopy(transcript))
                stages['rollback-turn'] = _identities(engine)
            if label == 're-upgrade':
                engine.ingest(deepcopy(live))
                stages['re-upgrade-newest'] = _identities(engine)
                turn = [_row('E', 120.), _row('F', 121., 'assistant')]
                live += turn
                transcript += turn
                engine.ingest(deepcopy(transcript))
                stages['re-upgrade-turn'] = _identities(engine)
            engine.on_session_end('parent', transcript)
        finally:
            engine.shutdown()
    return stages


@pytest.mark.parametrize('texts', [('P1', 'R'), ('P1', 'P2', 'R'), ('X', 'Y', 'X'),
                                 ('P1', 'P2', 'P3', 'R')],
                         ids=['two', 'three', 'repeated-1632', 'four-probe-F'])
@pytest.mark.parametrize('stamp', [False, True], ids=['persist-unset', 'gateway-event'])
@pytest.mark.parametrize('prefix', [False, True], ids=['whole-replace', 'prefix-keeping'])
@pytest.mark.parametrize('anchor', ['0', '1'], ids=['anchor-off', 'anchor-on'])
@pytest.mark.parametrize('fault', [False, True], ids=['normal', 'prematch-fault'])
def test_matrix(tmp_path, monkeypatch, versions, texts, stamp, prefix, anchor, fault):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', anchor)
    monkeypatch.setenv('LCM_INGEST_LEDGER', 'shadow')
    before = _scenario(versions['main'], versions['rollback'], tmp_path / 'main', monkeypatch,
                       texts, stamp, prefix, fault)
    after = _scenario(versions['candidate'], versions['rollback'], tmp_path / 'candidate', monkeypatch,
                      texts, stamp, prefix, fault)
    for stage, rows in after.items():
        base = before[stage]
        # Every existing host row stays reachable; a preserved composite admits
        # its newest-only host form. Coverage is occurrence-ordered, not set-only.
        wanted = ['Q', *(texts if anchor == '1' or prefix else [texts[-1]]), 'A']
        if stage not in ('warm', 'warm-replay', 'recreate-live'):
            wanted += ['N', 'B']
        if stage in ('rollback-turn', 're-upgrade', 're-upgrade-newest', 're-upgrade-turn'):
            wanted += ['M', 'C']
        if stage == 're-upgrade-turn':
            wanted += ['E', 'F']
        assert _ordered_coverage(rows, wanted), (stage, rows, wanted)
        assert len(rows) <= len(base), (stage, len(rows), len(base))
        assert all(r[3:] == ('parent', 'conv') for r in rows)
        if prefix or anchor == '0':
            assert rows == base, (stage, rows, base)
        elif stage in ('warm', 'warm-replay'):
            assert rows == [('assistant', 'Q', 9., 'parent', 'conv'),
                            ('user', '\n\n'.join(texts), 10., 'parent', 'conv'),
                            ('assistant', 'A', float(10 * len(texts) + 1), 'parent', 'conv')]
        # Existing host rows never disappear relative to main, including the
        # assistant duplicates main's ambiguous fault fallback can retain.
        for role, text, _stamp, session, conversation in base:
            assert any(r[0] == role and text in r[1] and r[3:] == (session, conversation) for r in rows)


def test_red_two_whole_replace(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    stages = _scenario(versions['candidate'], versions['rollback'], tmp_path, monkeypatch,
                       ('earlier-message', 'newest-message'), False, False, False)
    warm = stages['warm']
    kept = sum(any(text in row[1] for row in warm) for text in ('earlier-message', 'newest-message'))
    assert _ordered_coverage(warm, ['earlier-message', 'newest-message']), (
        f'queued_kept={kept}/2 queued_lost={2 - kept} host_rows={len(warm)} extra_rows={len(warm) - 3}', warm)
    assert len(warm) == 3


def test_alias_format_and_hash(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    ctx = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = ctx.engine
    try:
        first, newest = _row('earlier', 10.), _row('newest', 20.)
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                  conversation_history=[first, newest])
        first['content'], first['timestamp'] = 'newest', 20.
        engine.ingest([first, _row('reply', 21., 'assistant')])
        rows = _rows(engine)
        assert len(rows) == 2
        joined = 'earlier\n\nnewest'
        assert (rows[0]['content'], rows[0]['observed_at']) == (joined, 10.)
        payload = engine._store.read_metadata_json('host_rewrite_identity:' + str(rows[0]['store_id']))
        assert payload == {'version': 1, 'content': 'newest',
                           'stored_sha256': hashlib.sha256(joined.encode()).hexdigest()}
        assert engine._host_rewrite_override_content(rows[0]) == 'newest'
        assert engine._store.get_message_relations([rows[0]['store_id']], 'alt_stamp')[0]['observed_at'] == 20.
        # A stale alias cannot confer identity on unrelated durable content.
        assert engine._host_rewrite_override_content({**rows[0], 'content': 'different'}) is None
    finally:
        engine.shutdown()
    old = _open(versions['rollback'], tmp_path, monkeypatch).engine
    try:
        row = _rows(old)[0]
        assert old._host_rewrite_override_content(row) == 'newest'
        assert old._host_rewrite_override_content({**row, 'content': 'different'}) is None
    finally:
        old.shutdown()


@pytest.mark.parametrize('guard', ['anchor-off', 'missing-payload', 'head-absent', 'newest-present',
                                   'prefix-marker', 'absorbed-marker', 'different-content',
                                   'multimodal', 'steer', 'summary-carrier'])
def test_overwrite_proof_guards(tmp_path, monkeypatch, versions, guard):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '0' if guard == 'anchor-off' else '1')
    ctx = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = ctx.engine
    try:
        first, newest = _row('earlier', 10.), _row('newest', 20.)
        if guard == 'multimodal':
            first['content'] = [{'type': 'text', 'text': 'earlier'}]
        elif guard == 'steer':
            first['display_kind'] = 'steer'
        elif guard == 'summary-carrier':
            first['content'] = 'Earlier turns have been compacted.\nsummary'
            # The floor host's carrier guard is application-owned. Install the
            # same predicate under the stub host for this stop-rule check.
            from types import ModuleType
            compressor = ModuleType('agent.context_compressor')
            compressor.split_user_originated_turn = lambda row: (
                'summary' if row['content'].startswith('Earlier turns have been compacted.') else None, None)
            monkeypatch.setitem(sys.modules, 'agent.context_compressor', compressor)
        payload = dict(session_id='parent', conversation_id='conv', user_message='newest',
                       conversation_history=[first, newest])
        if guard == 'missing-payload':
            del payload['user_message']
        ctx.hooks['pre_llm_call'](**payload)
        first['content'] = 'different' if guard == 'different-content' else 'newest'
        if guard == 'prefix-marker':
            first['_merged_turn_prefix'] = ''  # presence, even if empty, vetoes B
        elif guard == 'absorbed-marker':
            first['_absorbed_message_uids'] = []
        live = [first]
        if guard == 'head-absent':
            live = [dict(first)]
        elif guard == 'newest-present':
            live.append(newest)
        live.append(_row('reply', 21., 'assistant'))
        engine.ingest(live)
        assert [r['content'] for r in _rows(engine)] == [r['content'] for r in live]
        assert not engine._store._conn.execute(
            "SELECT key FROM metadata WHERE key LIKE 'host_rewrite_identity:%'").fetchall()
    finally:
        engine.shutdown()


@pytest.mark.parametrize('after_write', [False, True], ids=['before-alias', 'after-alias'])
def test_alias_fault_rolls_back_row_and_alias_then_keeps_main(tmp_path, monkeypatch, versions, after_write):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    ctx = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = ctx.engine
    try:
        first, newest = _row('earlier', 10.), _row('newest', 20.)
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                  conversation_history=[first, newest])
        first['content'], first['timestamp'] = 'newest', 20.
        original = engine._write_host_rewrite_identity
        def fail(store_id, payload):
            if after_write:
                original(store_id, payload)
            raise RuntimeError('injected queued-user alias fault')
        monkeypatch.setattr(engine, '_write_host_rewrite_identity', fail)
        live = [first, _row('reply', 21., 'assistant')]
        engine.ingest(live)
        assert _identities(engine) == [('user', 'newest', 20., 'parent', 'conv'),
                                      ('assistant', 'reply', 21., 'parent', 'conv')]
        assert not engine._store._conn.execute(
            "SELECT key FROM metadata WHERE key LIKE 'host_rewrite_identity:%'").fetchall()
        assert not engine._store.get_message_relations([r['store_id'] for r in _rows(engine)], 'alt_stamp')
        assert engine._ingest_failure_count == 0
    finally:
        engine.shutdown()


def test_different_reload_join_adds_at_most_mains_single_row(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results = {}
    for label in ('main', 'candidate'):
        ctx = _open(versions[label], tmp_path / label, monkeypatch)
        engine = ctx.engine
        try:
            first, newest = _row('earlier', 10.), _row('newest', 20.)
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                      conversation_history=[first, newest])
            first['content'] = 'newest'
            reply = _row('reply', 21., 'assistant')
            engine.ingest([first, reply])
            before = _identities(engine)
            engine.ingest([_row('earlier \n\nnewest', 10.), deepcopy(reply)])
            after = _identities(engine)
            assert all(row in after for row in before)
            assert any(r[1] == 'earlier \n\nnewest' for r in after)
            results[label] = len(after) - len(before)
        finally:
            engine.shutdown()
    assert results['candidate'] <= results['main'] <= 1, results


def test_warm_newest_time_cache_then_head_time_alias(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    ctx = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = ctx.engine
    first, newest, reply = _row('earlier', 10.), _row('newest', 20.), _row('reply', 21., 'assistant')
    try:
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                  conversation_history=[first, newest])
        first['content'], first['timestamp'] = 'newest', 20.
        engine.ingest([first, reply])
    finally:
        engine.shutdown()
    engine = _open(versions['candidate'], tmp_path, monkeypatch).engine
    try:
        engine.ingest([deepcopy(first), deepcopy(reply)])
        expected = _identities(engine)
        assert len(expected) == 2
        engine.ingest([_row('newest', 10.), deepcopy(reply)])
        assert _identities(engine) == expected
        assert engine._last_active_replay_messages[0]['timestamp'] == 10.
        engine.ingest([deepcopy(first), deepcopy(reply)])
        assert _identities(engine) == expected
    finally:
        engine.shutdown()


@pytest.mark.parametrize('form', ['joined', 'newest'])
@pytest.mark.parametrize('reader', ['head', 'durable-post-proof', 'mapper'])
def test_both_forms_reach_every_reader(tmp_path, monkeypatch, versions, form, reader):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    engine_context = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = engine_context.engine
    try:
        q, first, newest = _row('Q', 9., 'assistant'), _row('earlier', 10.), _row('newest', 20.)
        reply = _row('reply', 21., 'assistant')
        engine.ingest([q])
        engine_context.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                            conversation_history=[q, first, newest])
        first['content'] = 'newest'
        engine.ingest([q, first, reply])
        live = [q, _row('earlier\n\nnewest' if form == 'joined' else 'newest', 10.), reply]
        if reader == 'head':
            assert engine._cursor_from_host_rewrite_head([*live, _row('next', 100.)], 3) == 3
        elif reader == 'mapper':
            mapped = engine._get_store_id_map_for_messages(live)
            assert [mapped[id(row)] for row in live] == [row['store_id'] for row in _rows(engine)]
        else:
            reconcile = sys.modules[versions['candidate'].__name__ + '.reconcile']
            digest = reconcile._commit_proof_identity_digest(reconcile._proof_user_identity(
                engine._message_replay_identity(q, strip_carrier=False)))
            payload = {'version': 3, 'hermes_home': str(engine._hermes_home), 'session_id': 'parent',
                       'conversation_id': 'conv', 'created_at': 1700001000.,
                       'effective_sha256': [digest], 'last_store_id': _rows(engine)[0]['store_id'],
                       'carry_ranges': []}
            engine._store.write_metadata_json(['compaction_commit_proof:parent'], json.dumps(payload))
            assert engine._cursor_from_durable_commit_proof(live) == 3
    finally:
        engine.shutdown()


@pytest.mark.parametrize('stamp', [False, True], ids=['persist-unset', 'gateway-event'])
@pytest.mark.parametrize('prefix', [False, True], ids=['whole-replace', 'prefix-keeping'])
@pytest.mark.parametrize('anchor', ['0', '1'], ids=['anchor-off', 'anchor-on'])
def test_previous_ga_sibling_row(tmp_path, monkeypatch, versions, stamp, prefix, anchor):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', anchor)
    texts = ('P1', 'P2', 'R')
    before = _scenario(versions['previous_ga'], versions['rollback'], tmp_path / 'ga', monkeypatch,
                       texts, stamp, prefix, False)
    after = _scenario(versions['candidate'], versions['rollback'], tmp_path / 'candidate', monkeypatch,
                      texts, stamp, prefix, False)
    for stage, rows in after.items():
        assert len(rows) <= len(before[stage]), (stage, rows, before[stage])
        for role, text, _stamp, session, conversation in before[stage]:
            assert any(row[0] == role and text in row[1] and row[3:] == (session, conversation) for row in rows)
        if prefix or anchor == '0':
            assert rows == before[stage]


@pytest.mark.parametrize('fault', [False, True], ids=['normal', 'prematch-fault'])
def test_uid_head_keeps_its_slot_across_live_recreate(tmp_path, monkeypatch, versions, fault):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results = {}
    for name in ('main', 'candidate'):
        ctx = _open(versions[name], tmp_path / name, monkeypatch)
        engine = ctx.engine
        first, newest = _row('earlier', 10.), _row('newest', 20.)
        first['message_uid'], newest['message_uid'] = '1' * 32, '2' * 32
        reply = _row('reply', 21., 'assistant')
        try:
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                      conversation_history=[first, newest])
            first['content'], first['timestamp'] = 'newest', 20.
            engine.ingest([first, reply])
        finally:
            engine.shutdown()
        engine = _open(versions[name], tmp_path / name, monkeypatch).engine
        try:
            if fault:
                def boom(*_args, **_kwargs):
                    raise RuntimeError('injected pre-match failure')
                monkeypatch.setattr(engine, '_identity_anchor_prematch', boom)
            engine.ingest([deepcopy(first), deepcopy(reply)])
            engine.ingest([dict(first, timestamp=10.), deepcopy(reply)])
            results[name] = _identities(engine)
        finally:
            engine.shutdown()
    assert len(results['candidate']) <= len(results['main']), results
    assert _ordered_coverage(results['candidate'], ['earlier', 'newest'])


def test_changed_uid_is_a_new_occurrence_even_through_alias(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results = {}
    for name in ('main', 'candidate'):
        ctx = _open(versions[name], tmp_path / name, monkeypatch)
        engine = ctx.engine
        try:
            first, newest, reply = _row('earlier', 10.), _row('newest', 20.), _row('reply', 21., 'assistant')
            first['message_uid'], newest['message_uid'] = 'uid-first', 'uid-newest'
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                      conversation_history=[first, newest])
            first['content'], first['timestamp'] = 'newest', 20.
            engine.ingest([first, reply])
            new = dict(first, message_uid='uid-new-occurrence')
            engine.ingest([new, deepcopy(reply)])
            results[name] = _identities(engine)
        finally:
            engine.shutdown()
    assert len(results['candidate']) == len(results['main']) == 3, results
    assert [row[1] for row in results['candidate']] == ['earlier\n\nnewest', 'reply', 'newest']


def test_clean_payload_text_and_sender_capture_disabled(tmp_path, monkeypatch, versions):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    monkeypatch.setenv('LCM_SENDER_CAPTURE', '0')
    ctx = _open(versions['candidate'], tmp_path, monkeypatch)
    engine = ctx.engine
    try:
        first, newest = _row('earlier\n', 10.), _row('newest\n', 20.)
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='newest',
                                  conversation_history=[first, newest])
        assert _rows(engine) == []
        first['content'] = 'newest'
        engine.ingest([first, _row('reply', 21., 'assistant')])
        assert _identities(engine) == [('user', 'earlier\n\n\nnewest', 10., 'parent', 'conv'),
                                      ('assistant', 'reply', 21., 'parent', 'conv')]
    finally:
        engine.shutdown()
