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
