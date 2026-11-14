"""#2703 (I1/I6): characterize recovered v0.29.2 temporary-result replay.

Fixture setup and eight-turn replay ported from the S5/S7 release probes.
The oracle restores fixture payloads independently of the replay identity code.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _marker

SESSION, CONVERSATION = 'u1-session', 'u1-conv'
PAD = ' alpha beta gamma delta' * 30
BODY = 'SPILLOVER_RECOVERY_NEEDLE:\n' + 'é文abcdef\n' * 600
EXT = re.compile(r'\[Externalized (?:LCM ingest payload|tool output|payload):[^\]]*?;\s*ref=([^;\]\s]+)\]')


def _restore(content, storage):
    for _ in range(8):
        expanded = EXT.sub(lambda m: json.loads((storage / Path(m[1]).name).read_text())['content'], content or '')
        if expanded == content:
            return expanded
        content = expanded
    return content


def _logical(row, storage, marker, body):
    content = _restore(row.get('content') or '', storage)
    # Bare saved marker and its recovered durable body represent the SAME host slot.
    if content in (marker, body):
        content = '@C1_RESULT@'
    calls = row.get('tool_calls') or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    return (row['role'], content, row.get('tool_call_id') or '', json.dumps(calls, sort_keys=True),
            (row.get('tool_name') or row.get('name') or '') if row['role'] == 'tool' else '')


def _seed_shape(db, home, temp, host, shape):
    name = shape
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures'
    def sub(text):
        return text.replace('@HERMES_HOME@', str(home)).replace('@HOST_TEMP@', str(temp))
    with sqlite3.connect(db) as conn:
        conn.executescript(sub((fixtures / f'v0292_{name}_store.sql').read_text()))
    storage = home / 'externalized'
    storage.mkdir()
    for payload in (fixtures / f'v0292_{name}_payloads').glob('*.json'):
        seeded = json.loads(sub(payload.read_text()))
        # Row 3 is rewritten to the host call below; its payload must agree.
        seeded['tool_call_id'] = host[2]['tool_call_id']
        (storage / payload.name).write_text(json.dumps(seeded))
    with sqlite3.connect(db) as conn:
        stored_marker = conn.execute('SELECT content FROM messages WHERE store_id=3').fetchone()[0]
        for i, row in enumerate(host, 1):
            # Preserve the genuine v0.29.2 recovered body and provenance in row 3.
            content = stored_marker if i == 3 else row['content']
            values = (row['role'], content, row.get('tool_call_id'),
                      json.dumps(row['tool_calls']) if row.get('tool_calls') else None,
                      'dump' if row['role'] == 'tool' else None, float(1762700000 + i),
                      float(1762700000 + i), i)
            if i <= 4:
                conn.execute('UPDATE messages SET role=?,content=?,tool_call_id=?,tool_calls=?,tool_name=?,timestamp=?,observed_at=? WHERE store_id=?', values)
            else:
                conn.execute('INSERT INTO messages(role,content,tool_call_id,tool_calls,tool_name,timestamp,observed_at,store_id,session_id,conversation_id,source,observed_at_source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                             (*values, SESSION, CONVERSATION, 'cli', 'host_message_timestamp'))
        for table in ('messages_fts', 'messages_fts_stem'):
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
    return stored_marker


@pytest.fixture
def replay_setup(tmp_path, monkeypatch):
    def setup(shape, variant, nofile=False):
        home, temp = tmp_path / 'hermes', tmp_path / 'host-temp'
        home.mkdir()
        temp.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(temp))
        saved = temp / 'hermes-results/call_spill.txt'
        saved.parent.mkdir(parents=True)
        body = ('data:application/octet-stream;base64,' + 'QUJD' * 80 + '\n'
                if shape == 'temp_data_uri' else '') + BODY
        saved.write_bytes(body.encode())
        marker = _marker(saved, body, preview=body[:400] if shape == 'temp_data_uri' else body[:30])
        host = [
            {'role': 'user', 'content': 'U1' + PAD},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'dump', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': marker, 'tool_call_id': 'c1', 'tool_name': 'dump'},
            {'role': 'user', 'content': 'U2' + PAD},
            {'role': 'assistant', 'content': 'A2' + PAD},
        ]
        for i, row in enumerate(host, 1):
            row['timestamp'] = float(1762700000 + i)
        db, storage = home / 'lcm.db', home / 'externalized'
        stored_marker = _seed_shape(db, home, temp, host, shape)
        assert stored_marker != marker and _restore(stored_marker, storage) == body
        if nofile:
            saved.rename(saved.with_suffix('.gone'))
        monkeypatch.setattr(engine_module, 'summarize_with_escalation',
                            lambda *args, **kwargs: ('Earlier turns.\nExpand for details about: turns', 1))
        engine = LCMEngine(config=LCMConfig(
            database_path=str(db), large_output_externalization_path=str(storage),
            fresh_tail_count=8, fresh_tail_max_tokens=0, fresh_tail_pressure_yield_enabled=False,
            leaf_chunk_tokens=400, dynamic_leaf_chunk_max=400, context_threshold=0.001,
            threshold_full_sweep_enabled=False, temporal_rollups_enabled=False,
            embeddings_enabled=False,  # Replay/count probes own every store write.
        ), hermes_home=str(home))
        engines.append(engine)
        for _ in range(2):
            engine.on_session_start(SESSION, platform='cli', context_length=200_000,
                                    conversation_id=CONVERSATION)
        if variant == 'V2':
            node = SummaryNode(session_id=SESSION, summary='U1 requested dump c1.', token_count=6,
                               source_token_count=200, source_ids=[1, 2])
            engine._dag.add_node(node, before_commit=lambda conn, node_id:
                engine._lifecycle.stage_compaction_publication(conn, CONVERSATION, SESSION, node_id, 0, [1, 2]))
            engine._last_compacted_store_id = 2
        assert engine._store_complete_frontier() == (2 if variant == 'V2' else 0)
        return engine, host, storage, marker, body

    engines = []
    yield setup
    for engine in engines:
        engine.shutdown()


def _replay(setup):
    engine, host, storage, marker, body = setup
    def logical(row):
        return _logical(row, storage, marker, body)

    def summary_carrier():
        parts = [engine_module._summary_part_text(n) for n in engine._summary_frontier_nodes()]
        return [{'role': 'user', 'content': '\n\n---\n\n'.join(parts)}] if parts else []

    carrier, records = summary_carrier(), []
    for turn in range(1, 9):
        for role in ('user', 'assistant'):
            host.append({'role': role, 'content': f'turn {turn} {role}' + PAD,
                         'timestamp': float(1762700000 + len(host) + 1)})
        view = deepcopy(carrier + host)
        engine._get_store_id_map_for_messages(view)
        engine.ingest(view)
        engine._get_store_id_map_for_messages(view)
        engine._fresh_tail_boundary(view)
        with engine._compress_run.preserve_store_ids():
            engine._store_complete_backlog(view, engine._leading_anchor_count(view))
        engine.compress(view, current_tokens=10_000, force=True)
        rows = engine._store.get_session_messages(SESSION, limit=100_000)
        actual = Counter(logical(row) for row in rows)
        expected = Counter(logical(row) for row in host)
        missing, extra = expected - actual, actual - expected
        records.append((len(rows), sum(missing.values()), sum(extra.values())))
        carrier = summary_carrier()
        engine.note_turn_complete()
    return records


@pytest.mark.parametrize('shape', ['temp_plain_lf', 'temp_data_uri'])
@pytest.mark.parametrize('variant', ['V1', 'V2'])
def test_saved_file_present_stores_each_row_once(shape, variant, replay_setup):
    records = _replay(replay_setup(shape, variant))
    assert [missing for _, missing, _ in records] == [0] * 8, records
    assert [extra for _, _, extra in records] == [0] * 8, records


@pytest.mark.parametrize('shape', ['temp_plain_lf', 'temp_data_uri'])
@pytest.mark.parametrize('variant', ['V1', 'V2'])
def test_saved_file_removed_preserves_every_row(shape, variant, replay_setup):
    # Always runs separately so loss cannot hide inside the expected duplicate failure.
    records = _replay(replay_setup(shape, variant, nofile=True))
    assert [missing for _, missing, _ in records] == [0] * 8, records


@pytest.mark.xfail(
    strict=True,
    reason="#2703: a replay after the saved temporary file is gone stores the tool result again",
)
@pytest.mark.parametrize('shape', ['temp_plain_lf', 'temp_data_uri'])
@pytest.mark.parametrize('variant', ['V1', 'V2'])
def test_saved_file_removed_stores_each_row_once(shape, variant, replay_setup):
    records = _replay(replay_setup(shape, variant, nofile=True))
    assert [extra for _, _, extra in records] == [0] * 8, records
