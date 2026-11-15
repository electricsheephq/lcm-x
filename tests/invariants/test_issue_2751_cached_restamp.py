"""#2751: a committed warm retry must retire the cached pre-restamp occurrence."""
import copy

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _open, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _Probe


@pytest.mark.parametrize('restamp,busy,clean_ends', [
    (True, True, 2),
    (False, True, 2), (True, False, 2), (True, True, 1),
], ids=['restamp-busy-two-ends', 'no-restamp', 'no-busy', 'one-clean-end'])
def test_warm_retry_restamped_user_stored_once(tmp_path, monkeypatch, restamp, busy, clean_ends):
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
        view.append(dict(role='assistant', content='FINAL', timestamp=1006.0))
        for _ in range(clean_ends):
            engine.on_session_end('parent', copy.deepcopy(view))
        rows = engine._store.get_session_messages('parent')
        stamps = [r['observed_at'] for r in rows if r['content'] == 'question-0']
        assert stamps == ([1.0, 1001.0] if restamp else [1.0]), stamps
        assert [r['role'] for r in rows[:4]] == ['user', 'assistant', 'assistant', 'tool']
        assert sum(r['content'] == 'FINAL' for r in rows) == 1
    finally:
        engine.shutdown()
