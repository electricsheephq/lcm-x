"""#2751: a committed warm retry must retire the cached pre-restamp occurrence."""
import copy
from collections import Counter

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _open, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _Probe


@pytest.mark.parametrize('final', [False, True], ids=['no-FINAL', 'FINAL'])
@pytest.mark.parametrize('restamp,busy,clean_ends', [
    (True, True, 2),
    (False, True, 2), (True, False, 2), (True, True, 1),
], ids=['restamp-busy-two-ends', 'no-restamp', 'no-busy', 'one-clean-end'])
def test_warm_retry_restamped_user_stored_once(tmp_path, monkeypatch, restamp, busy, clean_ends, final):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        if restamp:
            view[0]['timestamp'] += 1000
        if busy:
            probe = _Probe(monkeypatch)
            with monkeypatch.context() as patch:
                probe.busy_end(patch, engine, 'warm', view)
            if restamp and not any(t['kind'] == 'warm' for t in engine._pending_session_tails):
                pytest.fail('setup: the restamped occurrence must be frozen as a warm tail')
        if final:
            view.append(dict(role='assistant', content='FINAL', timestamp=1006.0))
        for _ in range(clean_ends):
            engine.on_session_end('parent', copy.deepcopy(view))
        rows = engine._store.get_session_messages('parent')
        stamps = [r['observed_at'] for r in rows if r['content'] == 'question-0']
        assert stamps == ([1.0, 1001.0] if restamp else [1.0]), stamps
        assert [r['role'] for r in rows[:4]] == ['user', 'assistant', 'assistant', 'tool']
        assert sum(r['content'] == 'FINAL' for r in rows) == int(final)
        # Every row exactly once: a duplicated reply, call or result row fails here too.
        expected = [('user', 1.0), ('assistant', 2.0), ('assistant', 3.0), ('tool', 4.0)]
        expected += [('user', 1001.0)] * restamp + [('assistant', 1006.0)] * final
        assert sorted((r['role'], r['observed_at']) for r in rows) == sorted(expected)
    finally:
        engine.shutdown()


@pytest.mark.parametrize('proof_metadata', ['empty', 'full'])
def test_cached_restamp_forced_durable_proof(tmp_path, monkeypatch, proof_metadata, record_property):
    """A durable proof must preserve cached evidence, including unchanged siblings."""
    def occurrences(engine):
        return sorted((row['role'], row['content'], row['observed_at'])
                      for row in engine._store.get_session_messages('parent'))

    clean_root = tmp_path / 'clean'
    clean_root.mkdir()
    with monkeypatch.context() as patch:
        clean = _open(clean_root, patch, None)
        try:
            view = _turn(0, 'plain')
            clean.ingest(copy.deepcopy(view))
            for stamp in (1001., 2001.):
                view[0]['timestamp'] = stamp
                clean.on_session_end('parent', copy.deepcopy(view))
            expected = occurrences(clean)
        finally:
            clean.shutdown()

    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] = 1001.
        with monkeypatch.context() as patch:
            _Probe(monkeypatch).busy_end(patch, engine, 'warm', view)
        assert any(t['kind'] == 'warm' for t in engine._pending_session_tails)
        proof = {'output_metadata': []}
        if proof_metadata == 'full':
            engine._prepare_compress_commit_proof(proof, copy.deepcopy(view))
            assert proof['output_metadata'][0]['timestamp'] == 1001.
            assert proof['output_metadata'][0]['identity_sha256']
        monkeypatch.setattr(engine, '_durable_commit_proof_payload', lambda *a, **k: proof)
        monkeypatch.setattr(engine, '_cursor_from_durable_commit_proof', lambda *a, **k: 4)
        view[0]['timestamp'] = 2001.
        record_property('before', repr(occurrences(engine)))
        engine.on_session_end('parent', copy.deepcopy(view))
        actual = occurrences(engine)
        record_property('occurrences', repr(actual))
        record_property('clean_occurrences', repr(expected))
        assert actual == expected, {'stored': actual, 'clean': expected}
    finally:
        engine.shutdown()


@pytest.mark.parametrize('proof_metadata', ['empty', 'full'])
def test_cached_restamp_unknown_pruned_tool_sibling(tmp_path, monkeypatch, proof_metadata, record_property):
    """#1540 must not inherit cached skip evidence discarded by the durable proof."""
    from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

    def run(root, busy):
        root.mkdir()
        with monkeypatch.context() as patch:
            engine = _open(root, patch, None)
            try:
                turn = _turn(0, 'plain', uids=False)
                view = [turn[0], turn[2], turn[3]]
                view[2].pop('timestamp')
                view[2]['content'] = 'original tool output ' * 11
                engine.ingest(copy.deepcopy(view))
                view[0]['timestamp'] = 1001.
                if busy:
                    with patch.context() as fault:
                        _Probe(patch).busy_end(fault, engine, 'warm', view)
                    assert any(t['kind'] == 'warm' and t.get('cache_update')
                               for t in engine._pending_session_tails)
                    proof = {'output_metadata': []}
                    if proof_metadata == 'full':
                        engine._prepare_compress_commit_proof(proof, copy.deepcopy(view))
                        assert proof['output_metadata'][0]['timestamp'] == 1001.
                        assert proof['output_metadata'][0]['identity_sha256']
                    with engine._exclusive_lifecycle('end'):
                        engine._retry_pending_session_tails()
                    assert not engine._pending_session_tails
                    assert engine._last_active_replay_messages == view
                    assert (engine._deferred_active_replay_cache_generation
                            == engine._active_replay_cache_generation)
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
                view[0]['timestamp'] = 2001.
                view[2]['content'] = _HERMES_PRUNED_TOOL_PLACEHOLDER
                for n in (1, 2):
                    view.extend(_turn(n, 'plain', uids=False)[2:])
                if busy:
                    patch.setattr(engine, '_durable_commit_proof_payload',
                                  lambda *a, **k: proof)
                    patch.setattr(engine, '_cursor_from_durable_commit_proof', lambda *a, **k: len(view))
                engine.on_session_end('parent', copy.deepcopy(view))
                assert not engine._pending_session_tails
                return sorted((row['role'], row['content'], row['observed_at'])
                              for row in engine._store.get_session_messages('parent'))
            finally:
                engine.shutdown()

    expected = run(tmp_path / 'clean', False)
    actual = run(tmp_path / 'busy', True)
    record_property('occurrences', repr(actual))
    record_property('clean_occurrences', repr(expected))
    stored, clean = Counter(actual), Counter(expected)
    # Known #436 partial-segment re-store on the forced durable branch (identity_anchor.py:1598-1599); #2781.
    assert clean <= stored, {'missing': clean - stored, 'stored': actual, 'clean': expected}
    assert stored - clean == Counter({('assistant', '', 3.0): 1}), stored - clean


@pytest.mark.parametrize('clean_ends', [1, 2])
@pytest.mark.parametrize('route', ['end', 'ingest'])
@pytest.mark.parametrize('which', [0, 1])
def test_restamp_after_clear(tmp_path, monkeypatch, clean_ends, route, which):
    """The clearing end has no FINAL: both cache vectors must be refilled together."""
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] = 1001.
        probe = _Probe(monkeypatch)
        with monkeypatch.context() as patch:
            probe.busy_end(patch, engine, 'warm', view)
        assert any(t['kind'] == 'warm' for t in engine._pending_session_tails)
        for _ in range(clean_ends):
            engine.on_session_end('parent', copy.deepcopy(view))
        before = engine._store.get_session_messages('parent')
        assert [r['observed_at'] for r in before if r['content'] == 'question-0'] == [1., 1001.]
        view[which]['timestamp'] += 2000
        if route == 'end':
            engine.on_session_end('parent', copy.deepcopy(view))
        else:
            engine.ingest(copy.deepcopy(view))
        rows = engine._store.get_session_messages('parent')
        assert rows[:len(before)] == before  # Earlier stamps and all other columns stay unchanged.
        for index, content in enumerate(('question-0', 'reply-0')):
            expected = [1., 1001.] if index == 0 else [2.]
            if index == which:
                expected.append(view[which]['timestamp'])
            assert [r['observed_at'] for r in rows if r['content'] == content] == expected
        assert len(rows) == 6
    finally:
        engine.shutdown()


def test_second_restamp_without_final(tmp_path, monkeypatch):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] = 1001.
        probe = _Probe(monkeypatch)
        with monkeypatch.context() as patch:
            probe.busy_end(patch, engine, 'warm', view)
        for _ in range(2):
            engine.on_session_end('parent', copy.deepcopy(view))
        view[0]['timestamp'] = 2001.
        for _ in range(2):
            engine.on_session_end('parent', copy.deepcopy(view))
        rows = engine._store.get_session_messages('parent')
        assert [r['observed_at'] for r in rows if r['content'] == 'question-0'] == [1., 1001., 2001.]
        assert len(rows) == 6
    finally:
        engine.shutdown()


@pytest.mark.parametrize('second', ['user', 'reply', 'none'])
@pytest.mark.parametrize('when', ['pending', 'after-first-end'])
@pytest.mark.parametrize('clean_ends', [1, 3])
def test_pending_retry_restamp_matrix(tmp_path, monkeypatch, second, when, clean_ends, record_property):
    """Every host occurrence, including one arriving before the retry, is committed once."""
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] = 1001.
        probe = _Probe(monkeypatch)
        with monkeypatch.context() as patch:
            probe.busy_end(patch, engine, 'warm', view)
        assert any(t['kind'] == 'warm' for t in engine._pending_session_tails)
        if when == 'after-first-end':
            engine.on_session_end('parent', copy.deepcopy(view))
        expected = [(r['content'], r['timestamp']) for r in _turn(0, 'plain')]
        expected.append(('question-0', 1001.))
        if second != 'none':
            row, stamp = (0, 2001.) if second == 'user' else (1, 2002.)
            view[row]['timestamp'] = stamp
            expected.append((view[row]['content'], stamp))
        for _ in range(clean_ends):
            engine.on_session_end('parent', copy.deepcopy(view))
        actual = [(r['content'], r['observed_at']) for r in engine._store.get_session_messages('parent')]
        record_property('occurrences', repr(sorted(actual)))
        assert sorted(actual) == sorted(expected), actual
    finally:
        engine.shutdown()


def test_pending_retry_newer_cache_guard(tmp_path, monkeypatch):
    """An ordinary successful ingest supersedes the frozen cache before its tail drains."""
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] = 1001.
        probe = _Probe(monkeypatch)
        with monkeypatch.context() as patch:
            probe.busy_end(patch, engine, 'warm', view)
        assert any(t['kind'] == 'warm' for t in engine._pending_session_tails)
        # The successful view excludes the pending 1001 occurrence and records a newer one.
        view[0]['timestamp'] = 2001.
        engine.ingest(copy.deepcopy(view))
        assert engine._pending_session_tails
        newer_cache = copy.deepcopy((engine._last_active_replay_source_identities, engine._last_active_replay_messages))
        drained_caches = []
        retry = engine._retry_pending_session_tails

        def observe_retry(**kwargs):
            retry(**kwargs)
            drained_caches.append(copy.deepcopy((
                engine._last_active_replay_source_identities, engine._last_active_replay_messages)))

        monkeypatch.setattr(engine, '_retry_pending_session_tails', observe_retry)
        for _ in range(3):
            engine.on_session_end('parent', copy.deepcopy(view))
        rows = engine._store.get_session_messages('parent')
        stamps = sorted(r['observed_at'] for r in rows if r['content'] == 'question-0')
        assert drained_caches[0] == newer_cache, 'retry replaced the newer successful cache with the frozen view'
        assert stamps == [1., 1001., 2001.], stamps
        assert len(rows) == 6, [(r['content'], r['observed_at']) for r in rows]
    finally:
        engine.shutdown()


def _run_chained_warm_tails(root, monkeypatch, shape, drain, followup, *, busy, ordinary=False):
    root.mkdir()
    with monkeypatch.context() as patch:
        engine = _open(root, patch, None)
        try:
            view = _turn(0, 'plain', uids=shape.startswith('uid-'))
            if shape.endswith('user'):
                view = view[:1]
            engine.ingest(copy.deepcopy(view))
            probe = _Probe(patch)
            frozen = []
            for stamp in (1001., 2001.):
                view[0]['timestamp'] = stamp
                if ordinary and stamp == 2001.:
                    engine.ingest(copy.deepcopy(view))
                    newer_cache = copy.deepcopy(engine._last_active_replay_messages)
                if busy:
                    with patch.context() as fault:
                        probe.busy_end(fault, engine, 'warm', view)
                    frozen.append([t['kind'] for t in engine._pending_session_tails])
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
            if busy:
                expected_frozen = [['warm'], ['warm']] if ordinary or shape == 'uid-turn' else [
                    ['warm'], ['warm', 'warm']]
                assert frozen == expected_frozen
            if drain == 'end':
                engine.on_session_end('parent', copy.deepcopy(view))
            else:
                engine._retry_pending_session_tails()
            assert not engine._pending_session_tails
            if ordinary:
                assert engine._last_active_replay_messages == newer_cache
            if followup == 'ingest':
                engine.ingest(copy.deepcopy(view))
            elif followup.startswith('end'):
                for _ in range(int(followup[3:])):
                    engine.on_session_end('parent', copy.deepcopy(view))
            return sorted((r['content'], r['observed_at'])
                          for r in engine._store.get_session_messages('parent'))
        finally:
            engine.shutdown()


@pytest.mark.parametrize('shape', ['uid-user', 'nouid-user', 'nouid-turn', 'uid-turn'])
@pytest.mark.parametrize('drain', ['end', 'retry'])
@pytest.mark.parametrize('followup', ['none', 'ingest', 'end1', 'end3'])
def test_chained_warm_tail_cache_updates(tmp_path, monkeypatch, shape, drain, followup, record_property):
    """Chained frozen views keep main, with the single-tail uid turn as a control."""
    expected = _run_chained_warm_tails(tmp_path / 'clean', monkeypatch, shape, drain, followup, busy=False)
    actual = _run_chained_warm_tails(tmp_path / 'busy', monkeypatch, shape, drain, followup, busy=True)
    record_property('clean_occurrences', repr(expected))
    record_property('occurrences', repr(actual))
    if shape == 'nouid-turn' and (drain, followup) in {
            ('end', 'end1'), ('end', 'end3'), ('end', 'ingest'), ('retry', 'end3')}:
        # #2751: the chained case keeps main's duplicate; tracked open on #2751
        expected = [('', 3.), ('question-0', 1.), ('question-0', 1001.),
                    ('question-0', 2001.), ('question-0', 2001.), ('reply-0', 2.), ('result-0', 4.)]
    if (shape, drain, followup) == ('uid-turn', 'retry', 'none'):
        # #2778: the single-tail control cannot reach clean; pin main's two-busy-base.json list.
        expected = [('', 3.), ('question-0', 1.), ('question-0', 1001.), ('reply-0', 2.), ('result-0', 4.)]
    assert actual == expected, {'stored': actual, 'clean': expected}


@pytest.mark.parametrize('shape', ['uid-user', 'nouid-user', 'nouid-turn'])
def test_chained_warm_tails_ordinary_ingest_wins(tmp_path, monkeypatch, shape):
    """An ordinary ingest between busy ends supersedes the first frozen view."""
    expected = _run_chained_warm_tails(tmp_path / 'clean', monkeypatch, shape, 'retry', 'end3',
                                       busy=False, ordinary=True)
    actual = _run_chained_warm_tails(tmp_path / 'busy', monkeypatch, shape, 'retry', 'end3',
                                     busy=True, ordinary=True)
    assert actual == expected, {'stored': actual, 'clean': expected}


def test_chained_tails_retire_both_frozen_views(tmp_path, monkeypatch):
    """A non-busy failure can leave the first tail pending while the second commits."""
    # Main's rows for this sequence, also equal to the clean run.
    expected = [('question-0', 1.), ('question-0', 1001.), ('question-0', 2001.)]
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain')[:1]
        engine.ingest(copy.deepcopy(view))
        ordinary_view = copy.deepcopy(engine._last_active_replay_messages)
        probe = _Probe(monkeypatch)
        for stamp in (1001., 2001.):
            view[0]['timestamp'] = stamp
            with monkeypatch.context() as patch:
                probe.busy_end(patch, engine, 'warm', view)
        assert len(engine._pending_session_tails) == 2
        older = engine._pending_session_tails[0]
        match = engine._match_pending_tail
        with monkeypatch.context() as patch:
            def fail_older(tail):
                if tail is older:
                    raise RuntimeError('first tail temporarily unavailable')
                return match(tail)
            patch.setattr(engine, '_match_pending_tail', fail_older)
            engine._retry_pending_session_tails()
        assert engine._pending_session_tails == [older]
        assert engine._last_active_replay_messages == ordinary_view
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
        assert engine._last_active_replay_messages == ordinary_view
        actual = sorted((r['content'], r['observed_at']) for r in engine._store.get_session_messages('parent'))
        assert actual == expected
    finally:
        engine.shutdown()


@pytest.mark.parametrize('second_change', ['reply2002', 'none', 'user2001'])
@pytest.mark.parametrize('recover', [False, True])
@pytest.mark.parametrize('uids', [True, False])
def test_chained_tails_failing_older_keep_main(tmp_path, monkeypatch, second_change, recover, uids):
    # Chained tails keep main's behaviour. Main's missing rows here are the older tail
    # still pending (recover=False) or #2778.
    expected = {
        'uids=True/recover=False/reply2002': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-1', 21.0), ('reply-0', 2.0),
            ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=True/recover=False/none': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-1', 21.0), ('reply-0', 2.0),
            ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=True/recover=False/user2001': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-1', 21.0), ('reply-0', 2.0),
            ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=True/recover=True/reply2002': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 0),
        'uids=True/recover=True/none': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 0),
        'uids=True/recover=True/user2001': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 0),
        'uids=False/recover=False/reply2002': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-1', 21.0), ('reply-0', 2.0),
            ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=False/recover=False/none': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=False/recover=False/user2001': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 2001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 1),
        'uids=False/recover=True/reply2002': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-1', 21.0),
            ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0), ('result-1', 24.0),
        ], 0),
        'uids=False/recover=True/none': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 1001.0),
            ('question-1', 21.0), ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0),
            ('result-1', 24.0),
        ], 0),
        'uids=False/recover=True/user2001': ([
            ('', 3.0), ('', 23.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
            ('question-1', 21.0), ('reply-0', 2.0), ('reply-1', 22.0), ('result-0', 4.0),
            ('result-1', 24.0),
        ], 0),
    }
    rows, pending_left = expected[f'uids={uids}/recover={recover}/{second_change}']
    e = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0, 'plain', uids=uids)
        e.ingest(copy.deepcopy(view))
        probe = _Probe(monkeypatch)
        view[0]['timestamp'] = 1001.
        with monkeypatch.context() as p:
            probe.busy_end(p, e, 'warm', view)
        view = view + _turn(1, 'plain', uids=uids)
        with monkeypatch.context() as p:
            probe.busy_end(p, e, 'warm', view)
        older = e._pending_session_tails[0] if e._pending_session_tails else None
        match = e._match_pending_tail

        def fail_older(tail):
            if tail is older:
                raise RuntimeError('probe: older tail unavailable')
            return match(tail)
        monkeypatch.setattr(e, '_match_pending_tail', fail_older)
        e._retry_pending_session_tails()
        if second_change == 'reply2002':
            view[1]['timestamp'] = 2002.
        elif second_change == 'user2001':
            view[0]['timestamp'] = 2001.
        for _ in range(2):
            e.on_session_end('parent', copy.deepcopy(view))
        if recover:
            monkeypatch.setattr(e, '_match_pending_tail', match)
            e.on_session_end('parent', copy.deepcopy(view))
        pend = len(e._pending_session_tails)
        occ = sorted((r['content'][:10], r['observed_at']) for r in e._store.get_session_messages('parent'))
        assert occ == rows
        assert pend == pending_left
    finally:
        e.shutdown()


def test_lone_tail_after_retired_chain_applies(tmp_path, monkeypatch):
    def run(root, busy):
        root.mkdir()
        with monkeypatch.context() as patch:
            engine = _open(root, patch, None)
            try:
                view = _turn(0, 'plain')[:1]
                engine.ingest(copy.deepcopy(view))
                probe = _Probe(patch)
                for stamp in (1001., 2001.):
                    view[0]['timestamp'] = stamp
                    if busy:
                        with patch.context() as fault:
                            probe.busy_end(fault, engine, 'warm', view)
                    else:
                        engine.on_session_end('parent', copy.deepcopy(view))
                if busy:
                    assert len(engine._pending_session_tails) == 2
                    engine._retry_pending_session_tails()
                    assert not engine._pending_session_tails
                view[0]['timestamp'] = 3001.
                if busy:
                    with patch.context() as fault:
                        probe.busy_end(fault, engine, 'warm', view)
                    assert len(engine._pending_session_tails) == 1
                    engine._retry_pending_session_tails()
                    assert not engine._pending_session_tails
                    assert (engine._deferred_active_replay_cache_generation
                            == engine._active_replay_cache_generation)
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
                for _ in range(3):
                    engine.on_session_end('parent', copy.deepcopy(view))
                return sorted((r['content'], r['observed_at'])
                              for r in engine._store.get_session_messages('parent'))
            finally:
                engine.shutdown()

    assert run(tmp_path / 'busy', True) == run(tmp_path / 'clean', False)


def _run_retired_chain_after_lone_retry(root, monkeypatch, drain, followup, *, busy=True):
    root.mkdir()
    with monkeypatch.context() as patch:
        engine = _open(root, patch, None)
        sibling = None
        try:
            view = _turn(0, 'plain', uids=False)[:1]
            engine.ingest(copy.deepcopy(view))
            probe = _Probe(patch)
            view[0]['timestamp'] = 1001.
            if busy:
                with patch.context() as fault:
                    probe.busy_end(fault, engine, 'warm', view)
                assert [t['kind'] for t in engine._pending_session_tails] == ['warm']
                if drain == 'sibling':
                    sibling = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
                    sibling.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
                elif drain == 'retry':
                    engine._retry_pending_session_tails()
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
                assert not engine._pending_session_tails
            else:
                engine.on_session_end('parent', copy.deepcopy(view))
            for stamp in (2001., 3001.):
                view[0]['timestamp'] = stamp
                if busy:
                    with patch.context() as fault:
                        probe.busy_end(fault, engine, 'warm', view)
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
            if busy:
                assert [t['kind'] for t in engine._pending_session_tails] == ['warm', 'warm']
            engine.on_session_end('parent', copy.deepcopy(view))
            assert not engine._pending_session_tails
            if followup == 'end':
                engine.on_session_end('parent', copy.deepcopy(view))
            elif followup == 'ingest':
                engine.ingest(copy.deepcopy(view))
            return sorted((r['content'], r['observed_at'])
                          for r in engine._store.get_session_messages('parent'))
        finally:
            if sibling is not None:
                sibling.shutdown()
            engine.shutdown()


def test_retired_chain_after_lone_retry_keeps_main(tmp_path, monkeypatch):
    # Main 4f75302e's exact stored occurrences for the review bot's sequence.
    expected = [('question-0', 1.), ('question-0', 1001.), ('question-0', 2001.), ('question-0', 3001.)]
    assert _run_retired_chain_after_lone_retry(tmp_path / 'busy', monkeypatch, 'sibling', 'none') == expected


@pytest.mark.parametrize('drain', ['sibling', 'retry', 'end'])
@pytest.mark.parametrize('followup', ['end', 'ingest'])
def test_retired_chain_after_lone_retry_variants(tmp_path, monkeypatch, drain, followup):
    # Main 4f75302e's stored list, independently observed for each of the six cells.
    expected = [('question-0', 1.), ('question-0', 1001.), ('question-0', 2001.), ('question-0', 3001.)]
    assert _run_retired_chain_after_lone_retry(tmp_path / 'busy', monkeypatch, drain, followup) == expected



def _run_rolled_back_chain(root, monkeypatch, shape, step5, ends, fault, *, faults=True):
    """The speculative second warm tail rolls back before a successful busy retry."""
    import sqlite3

    root.mkdir()
    with monkeypatch.context() as patch:
        engine = _open(root, patch, None)
        try:
            view = _turn(0, 'plain')
            if shape == 'user':
                view = view[:1]
            engine.ingest(copy.deepcopy(view))
            probe = _Probe(patch)
            for stamp in (1001., 2001.):
                view[0]['timestamp'] = stamp
                if faults:
                    with patch.context() as injected:
                        probe.busy_end(injected, engine, 'warm', view)
                else:
                    engine.on_session_end('parent', copy.deepcopy(view))
                if stamp == 1001.:
                    engine._retry_pending_session_tails()
            pending = list(engine._pending_session_tails)
            if step5 == 'user3001':
                view[0]['timestamp'] = 3001.
            elif step5 == 'reply3002':
                if shape == 'turn':
                    view[1]['timestamp'] = 3002.
                else:
                    view.append(dict(role='assistant', content='late-reply', timestamp=3002., message_uid='late'))
            match, append = engine._match_pending_tail, engine._append_ingest_batch
            hits = []

            def fail_pending(tail):
                if pending and tail is pending[0]:
                    error = sqlite3.OperationalError if fault == 'busy' else RuntimeError
                    raise error('database is locked' if fault == 'busy' else 'probe: T2 unavailable')
                return match(tail)

            def busy_once(batch):
                if not hits:
                    hits.append(True)
                    raise sqlite3.OperationalError('database is locked')
                return append(batch)

            with patch.context() as injected:
                if faults:
                    injected.setattr(engine, '_match_pending_tail', fail_pending)
                    injected.setattr(engine, '_append_ingest_batch', busy_once)
                engine.on_session_end('parent', copy.deepcopy(view))
            for _ in range(ends):
                engine.on_session_end('parent', copy.deepcopy(view))
            return sorted((r['content'], r['observed_at'])
                          for r in engine._store.get_session_messages('parent'))
        finally:
            engine.shutdown()


@pytest.mark.parametrize('shape', ['user', 'turn'])
@pytest.mark.parametrize('step5', ['user3001', 'reply3002', 'same'])
@pytest.mark.parametrize('ends', [1, 3])
@pytest.mark.parametrize('fault', ['busy', 'error'])
def test_rolled_back_chain_keeps_main(tmp_path, monkeypatch, shape, step5, ends, fault):
    # Each literal pair was independently run on main 4f75302e, with faults and without.
    # Main and fault-free differ for turn/user3001 (one or three ends, either fault):
    # main loses question-0@2001; the single-tail oracle permits either bounding count.
    expected = {'turn/reply3002/1/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)],
                               [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)]),
     'turn/reply3002/1/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                 ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)],
                                [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                 ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)]),
     'turn/reply3002/3/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)],
                               [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)]),
     'turn/reply3002/3/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                 ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)],
                                [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                 ('reply-0', 2.0), ('reply-0', 3002.0), ('result-0', 4.0)]),
     'turn/same/1/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                           ('reply-0', 2.0), ('result-0', 4.0)],
                          [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                           ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/same/1/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                            ('reply-0', 2.0), ('result-0', 4.0)],
                           [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                            ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/same/3/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                           ('reply-0', 2.0), ('result-0', 4.0)],
                          [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                           ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/same/3/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                            ('reply-0', 2.0), ('result-0', 4.0)],
                           [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                            ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/user3001/1/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 3001.0),
                               ('reply-0', 2.0), ('result-0', 4.0)],
                              [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0), ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/user3001/1/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 3001.0),
                                ('reply-0', 2.0), ('result-0', 4.0)],
                               [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0), ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/user3001/3/busy': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 3001.0),
                               ('reply-0', 2.0), ('result-0', 4.0)],
                              [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0), ('reply-0', 2.0), ('result-0', 4.0)]),
     'turn/user3001/3/error': ([('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 3001.0),
                                ('reply-0', 2.0), ('result-0', 4.0)],
                               [('', 3.0), ('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0), ('reply-0', 2.0), ('result-0', 4.0)]),
     'user/reply3002/1/busy': ([('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                ('question-0', 2001.0)],
                               [('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                ('question-0', 2001.0)]),
     'user/reply3002/1/error': ([('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                 ('question-0', 2001.0)],
                                [('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                 ('question-0', 2001.0)]),
     'user/reply3002/3/busy': ([('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                ('question-0', 2001.0)],
                               [('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                ('question-0', 2001.0)]),
     'user/reply3002/3/error': ([('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                 ('question-0', 2001.0)],
                                [('late-reply', 3002.0), ('question-0', 1.0), ('question-0', 1001.0),
                                 ('question-0', 2001.0)]),
     'user/same/1/busy': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)],
                          [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)]),
     'user/same/1/error': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)],
                           [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)]),
     'user/same/3/busy': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)],
                          [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)]),
     'user/same/3/error': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)],
                           [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0)]),
     'user/user3001/1/busy': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0)],
                              [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0)]),
     'user/user3001/1/error': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0)],
                               [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0)]),
     'user/user3001/3/busy': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0)],
                              [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                               ('question-0', 3001.0)]),
     'user/user3001/3/error': ([('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0)],
                               [('question-0', 1.0), ('question-0', 1001.0), ('question-0', 2001.0),
                                ('question-0', 3001.0)])}
    main, clean = map(Counter, expected[f'{shape}/{step5}/{ends}/{fault}'])
    actual = Counter(_run_rolled_back_chain(tmp_path / 'busy', monkeypatch, shape, step5, ends, fault))
    for row in main.keys() | clean.keys() | actual.keys():
        assert min(main[row], clean[row]) <= actual[row] <= max(main[row], clean[row]), (row, actual, main, clean)
