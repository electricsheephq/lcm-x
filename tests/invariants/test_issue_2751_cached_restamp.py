"""#2751: a committed warm retry must retire the cached pre-restamp occurrence."""
import copy

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
