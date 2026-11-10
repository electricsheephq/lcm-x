"""#2558: a seeded differential for the session-end row compare.

(1) Every generated pair WITHOUT a bare saved-output marker stays off the new code: with the bare-marker identity made
to raise, every result is unchanged. This is a structural check that the new identity is unreachable there; it is not
an independent oracle for the shared compare, which main's own tests cover. (2) Bare marker against
bare marker: the pair is one row exactly when its texts are byte-identical once ingest placeholders are replaced by
their payloads, a reference written here apart from the product code.
"""

import json
import os
import random

import pytest

from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_2521_spillover_upgrade_replay import _engine, _home

ALIAS = '[Externalized LCM ingest payload: 4 chars; ref=leaf.json]'  # the fixture's leaf.json holds "LEAF"


@pytest.fixture
def engine(tmp_path):
    engine = _engine(tmp_path)
    storage = _home(tmp_path) / 'externalized'
    storage.mkdir(parents=True, exist_ok=True)
    (storage / 'leaf.json').write_text(json.dumps({
        'kind': 'ingest_payload', 'content': 'LEAF', 'session_id': 'S1', 'field_path': 'content[0].text'}))
    engine.on_session_start('S1', platform='cli', context_length=200_000, conversation_id='c')
    try:
        yield engine
    finally:
        engine.shutdown()


def _main_equal(engine, a, b, monkeypatch):
    with monkeypatch.context() as patch:
        def raises(*args, **kwargs):
            raise RuntimeError('main compare')
        patch.setattr(engine, '_bare_marker_identity', raises)
        return engine._session_end_rows_equal(a, b, session_id='S1')


TEXTS = ['plain', 'plain ', 'LEAF', ALIAS, 'a\r\nb', 'a\nb', '{"k": 1}', '{"k":1}', '[{"type": "text", "text": "x"}]', '']


def _row(rng):
    role = rng.choice(['user', 'assistant', 'tool'])
    row = {'role': role, 'content': rng.choice(TEXTS)}
    if role == 'tool':
        row['tool_call_id'] = rng.choice(['c1', 'c2'])
    if role == 'assistant' and rng.random() < 0.4:
        row['tool_calls'] = [{'id': rng.choice(['c1', 'c2']), 'type': 'function',
                              'function': {'name': 'f', 'arguments': rng.choice(TEXTS[:8])}}]
    return row


def _bare(rng, home):
    path = rng.choice([f'/tmp/hermes-results/{rng.choice(["c1", "c2"])}.txt',
                       f'{home}/sessions/spillover/{rng.choice(["c1", "c2"])}.txt'])
    body = rng.choice(['x' * 300, 'y' * 300, 'data:image/png;base64,' + 'A' * 280])
    preview = rng.choice([None, body[:30], body[:29], ALIAS, 'LEAF', 'other preview'])
    text = _marker(path, body, count=rng.choice([None, 999]), preview=preview)
    return {'role': 'tool', 'tool_call_id': 'c1', 'content': text.replace('\n', '\r\n') if rng.random() < 0.2 else text}


def test_seeded_rows_equal_differential(engine, monkeypatch):  # noqa: F811
    seed, count = int(os.environ.get('LCMX_2558_SEED', '2558')), int(os.environ.get('LCMX_2558_PAIRS', '1500'))
    rng, home, markers = random.Random(seed), str(engine._hermes_home), 0
    for index in range(count):
        if rng.random() < 0.5:
            a, b = _row(rng), _row(rng) if rng.random() < 0.5 else None
            b = b or json.loads(json.dumps(a))
            head = engine._session_end_rows_equal(a, b, session_id='S1')
            assert head == _main_equal(engine, a, b, monkeypatch), (seed, index)
        else:
            a, b = _bare(rng, home), _bare(rng, home)
            if rng.random() < 0.3:
                b = json.loads(json.dumps(a))
            markers += 1
            head = engine._session_end_rows_equal(a, b, session_id='S1')
            reference = a['content'].replace(ALIAS, 'LEAF') == b['content'].replace(ALIAS, 'LEAF')
            assert head == reference, (seed, index)
    assert markers
