"""#2703 (I1/I6): recovered v0.29.2 temporary results survive replay once.

Fixture setup and eight-turn replay ported from the S5/S7 release probes.
The oracle restores fixture payloads independently of the replay identity code.
"""
from collections import Counter
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.reconcile as reconcile_module
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


def _replay(setup, *, logical=None, check_rows=None):
    engine, host, storage, marker, body = setup
    if logical is None:
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
        # Each turn must preserve all host rows, even on the best-effort fault path.
        assert not missing, (turn, records)
        if check_rows is not None:
            check_rows(rows)
        carrier = summary_carrier()
        engine.note_turn_complete()
    return records


@pytest.mark.parametrize('shape', ['temp_plain_lf', 'temp_data_uri'])
@pytest.mark.parametrize('variant', ['V1', 'V2'])
@pytest.mark.parametrize('nofile', [False, True], ids=['present', 'removed'])
def test_recovered_temp_marker_stored_once(shape, variant, nofile, replay_setup):
    records = _replay(replay_setup(shape, variant, nofile))
    print(f'REPLAY {shape} {variant} nofile={nofile}: {records}')
    if nofile:
        # a file-removed replay of a recovered temporary marker still stores the tool rows again, as on main;
        # tracked separately
        # Per-turn counts from round-1 red-2703-7.log, main 7828784f.
        main_counts = {
            ('temp_plain_lf', 'V1'): [9, 11, 13, 15, 17, 20, 22, 24],
            ('temp_data_uri', 'V1'): [9, 11, 13, 15, 17, 20, 23, 25],
            ('temp_plain_lf', 'V2'): [9, 11, 14, 16, 19, 21, 23, 25],
            ('temp_data_uri', 'V2'): [9, 11, 14, 16, 18, 20, 22, 24],
        }
        assert [missing for _, missing, _ in records] == [0] * 8, records
        assert [count for count, _, _ in records] == main_counts[shape, variant], records
        return
    # Retain all eight observations in a red proof instead of stopping on turn one.
    assert [extra for _, _, extra in records] == [0] * 8, records


def test_payload_load_fault_keeps_main_counts(replay_setup, monkeypatch):
    setup = _keep_setup(replay_setup)
    engine = setup[0]
    original = reconcile_module.load_externalized_payload
    faults = []

    def fail_candidate(ref, **kwargs):
        # Limit the injected fault to stored identities evaluated by the new check.
        if any(frame.function == 'legacy_recovered_keep' for frame in inspect.stack()):
            faults.append(ref)
            raise OSError('injected candidate payload read failure')
        return original(ref, **kwargs)

    monkeypatch.setattr(reconcile_module, 'load_externalized_payload', fail_candidate)
    records = _replay(setup)
    print(f'FAULT records={records} injected={len(faults)}')
    assert records == [(9, 0, 2), (11, 0, 2), (13, 0, 2), (15, 0, 2),
                       (18, 0, 3), (21, 0, 4), (23, 0, 4), (25, 0, 4)]
    # On main there is no candidate check, so the same control still pins its counts.
    assert not hasattr(engine, '_legacy_recovered_source_identities') or faults


def test_candidate_payload_loaded_once_per_pass(replay_setup, monkeypatch):
    engine, host, storage, marker, body = _keep_setup(replay_setup)
    original = reconcile_module.load_externalized_payload
    loads = Counter()

    def counted(ref, **kwargs):
        loads[(engine._store.write_generation(), ref)] += 1
        return original(ref, **kwargs)

    # Background embeddings are disabled in this fixture. Reproduce a foreign
    # commit halfway through the reads instead of relying on worker timing.
    assert engine.drain_embedding_maintenance(timeout=10)
    monkeypatch.setattr(reconcile_module, 'load_externalized_payload', counted)
    for read in range(40):
        if read == 20:
            with sqlite3.connect(engine._config.database_path) as writer:
                writer.execute("INSERT INTO metadata(key,value) VALUES ('issue2703_count_probe','1')")
        identity = engine._message_replay_identity(host[2])
        assert identity[1] == body
    assert loads and max(loads.values()) == 1, loads
    assert len({generation for generation, _ in loads}) == 2, loads
    assert len({ref for _, ref in loads}) == 1, loads
    print(f'CANDIDATE_PAYLOAD_LOADS {dict(loads)} over 40 identity reads; one foreign commit')

    # A full replay makes writes between passes. Within each store generation,
    # the new check must read each candidate payload at most once, however often
    # reconciliation asks for that identity.
    loads.clear()

    def counted_candidate(ref, **kwargs):
        if any(frame.function == 'legacy_recovered_keep' for frame in inspect.stack()):
            loads[(engine._store.write_generation(), ref)] += 1
        return original(ref, **kwargs)

    monkeypatch.setattr(reconcile_module, 'load_externalized_payload', counted_candidate)
    records = _replay((engine, host, storage, marker, body))
    assert [extra for _, _, extra in records] == [0] * 8
    assert loads and max(loads.values()) == 1, loads
    print(f'REPLAY_CANDIDATE_PAYLOAD_LOADS generations={len(loads)} max={max(loads.values())}')


def _literal_reference_setup(replay_setup):
    engine, host, storage, _, stored_body = replay_setup('temp_plain_lf', 'V1')
    stored = engine._store.get_session_messages(SESSION, limit=100_000)[2]
    ref = EXT.fullmatch(stored['content'])[1]
    payload = reconcile_module.load_externalized_payload(
        ref, config=engine._config, hermes_home=engine._hermes_home)
    assert payload['content'] == stored_body
    body = stored_body + f'\nDISTINCT_BODY_B\n[Externalized tool output: tool=dump; ref={ref}]'
    path = Path(tempfile.tempdir) / 'hermes-results/call_distinct.txt'
    path.write_text(body)
    marker = _marker(path, body)
    assert str(path) not in stored['content'] and body != stored_body
    host[2]['content'] = marker
    return engine, host, storage, marker, body


def _replay_literal_reference(setup):
    _, _, storage, marker, body = setup

    def content_of(row):
        content = row.get('content') or ''
        # Expand only the store's whole placeholder once. B's embedded reference
        # is literal text, so recursively expanding it would hide the lost row.
        match = EXT.fullmatch(content)
        return json.loads((storage / match[1]).read_text())['content'] if match else content

    def logical(row):
        calls = row.get('tool_calls') or []
        if isinstance(calls, str):
            calls = json.loads(calls)
        content = body if row.get('content') == marker else content_of(row)
        return (row['role'], content, row.get('tool_call_id') or '',
                json.dumps(calls, sort_keys=True),
                (row.get('tool_name') or row.get('name') or '') if row['role'] == 'tool' else '')

    def check_rows(rows):
        assert any(row['role'] == 'tool' and content_of(row) == body for row in rows), 'B missing'

    return _replay(setup, logical=logical, check_rows=check_rows)


def test_distinct_recovered_body_with_literal_reference(replay_setup):
    records = _replay_literal_reference(_literal_reference_setup(replay_setup))
    print(f'LITERAL_REFERENCE records={records}')
    # Main 7828784f, round-3 main-2703-r3-3.log, turns 1 through 8.
    assert [count for count, _, _ in records] == [9, 11, 13, 15, 18, 21, 23, 25], records


def test_stored_body_with_literal_reference(replay_setup):
    setup = _keep_setup(replay_setup)
    engine, _, _, _, body = setup
    stored = engine._store.get_session_messages(SESSION, limit=100_000)[2]
    ref = EXT.fullmatch(stored['content'])[1]
    payload = reconcile_module.load_externalized_payload(
        ref, config=engine._config, hermes_home=engine._hermes_home)
    assert payload['content'] == body
    assert not EXT.search(body)
    content = f'DISTINCT_STORED_C\n[Externalized tool output: tool=dump; ref={ref}]'
    with sqlite3.connect(engine._config.database_path) as conn:
        conn.execute('UPDATE messages SET content=? WHERE store_id=?',
                     (content, stored['store_id']))
        for table in ('messages_fts', 'messages_fts_stem'):
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
    records = _replay_literal_reference(setup)
    print(f'STORED_LITERAL_REFERENCE records={records}')
    # Main 7828784f, round-4 main-2703-r4-2.log, turns 1 through 8.
    assert [count for count, _, _ in records] == [9, 11, 13, 15, 18, 21, 23, 25], records


def test_distinct_recovered_body_with_literal_reference_load_fault(replay_setup, monkeypatch):
    setup = _literal_reference_setup(replay_setup)
    original = reconcile_module.load_externalized_payload
    faults = []
    inspected_roles = []

    def fail_load(ref, **kwargs):
        # Fault the post-keep load of B's literal ref, leaving main's existing
        # durable-content loads and the recursive candidate comparison readable.
        caller = inspect.currentframe().f_back
        state = caller.f_locals
        required = {'stored_row', 'msg', 'content', 'durable_content', 'stale_durable_content'}
        if (caller.f_code.co_name == '_message_replay_identity'
                and required <= state.keys()
                and state['stored_row'] is False
                and state['msg'].get('role') == 'tool'
                and isinstance(state['content'], str)
                and state['durable_content'] is None
                and state['stale_durable_content'] is None):
            inspected_roles.append(ref)
        host_body = (not state.get('stored_row', True)
                     and state.get('msg', {}).get('content') == setup[3]
                     and state.get('content') == setup[4]
                     and state.get('durable_content') is None
                     and state.get('stale_durable_content') is None)
        del caller
        if host_body:
            faults.append(ref)
            raise OSError('injected literal-reference payload read failure')
        return original(ref, **kwargs)

    monkeypatch.setattr(reconcile_module, 'load_externalized_payload', fail_load)
    # Exercise the predicate against a real host marker frame. A reference in
    # the marker itself reaches the existing load without the guarded keep.
    path = Path(tempfile.tempdir) / 'hermes-results/call_distinct.txt'
    control_marker = _marker(path, setup[4], preview=setup[4])
    setup[0]._message_replay_identity({**setup[1][2], 'content': control_marker})
    assert inspected_roles, 'fault predicate never inspected the expected host identity locals'
    records = _replay_literal_reference(setup)
    print(f'LITERAL_REFERENCE_FAULT records={records} injected={len(faults)} '
          f'inspected_host_frames={len(inspected_roles)}')
    assert [count for count, _, _ in records] == [9, 11, 13, 15, 18, 21, 23, 25], records
    assert not faults, 'the post-keep literal-reference load must be unreachable'


def _keep_setup(replay_setup):
    engine, host, storage, _, body = replay_setup('temp_plain_lf', 'V1')
    # A different live path has no persisted-marker provenance in the payload.
    # This reaches the keep even now that the seeded row/payload call ids agree.
    path = Path(tempfile.tempdir) / 'hermes-results/call_alias.txt'
    path.write_text(body)
    marker = _marker(path, body, preview=body[:30])
    host[2]['content'] = marker
    return engine, host, storage, marker, body


def _rewrite_candidate(setup, *, content=None, payload_changes=None, missing_provenance=False,
                       append_pruned=False):
    engine, _, storage, _, _ = setup
    stored = engine._store.get_session_messages(SESSION, limit=100_000)[2]
    ref = EXT.fullmatch(stored['content'])[1]
    if payload_changes is not None or missing_provenance:
        path = storage / ref
        payload = json.loads(path.read_text())
        payload.update(payload_changes or {})
        if missing_provenance:
            for field in ('tool_call_id', 'session_id', 'tool_name'):
                payload.pop(field, None)
        path.write_text(json.dumps(payload))
    if content is not None or append_pruned:
        with sqlite3.connect(engine._config.database_path) as conn:
            if content is not None:
                conn.execute('UPDATE messages SET content=? WHERE store_id=?',
                             (content, stored['store_id']))
            if append_pruned:
                conn.execute("""INSERT INTO messages
                    (role,content,tool_call_id,tool_name,timestamp,observed_at,
                     session_id,conversation_id,source,observed_at_source)
                    VALUES ('tool',?,'c1','dump',1762700006,1762700006,?,?,'cli','host_message_timestamp')""",
                    (reconcile_module._HERMES_PRUNED_TOOL_PLACEHOLDER, SESSION, CONVERSATION))
            for table in ('messages_fts', 'messages_fts_stem'):
                conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
    return ref


def test_pruned_stored_alias_cannot_admit_recovered_body(replay_setup):
    setup = _keep_setup(replay_setup)
    stored = setup[0]._store.get_session_messages(SESSION, limit=100_000)[2]
    ref = EXT.fullmatch(stored['content'])[1]
    content = 'DISTINCT_EARLIER_C ' * 20 + f'[Externalized tool output: tool=dump; ref={ref}]'
    assert len(content) > 200
    _rewrite_candidate(setup, content=content, append_pruned=True)
    records = _replay_literal_reference(setup)
    print(f'A1_PRUNED_ALIAS records={records}')
    assert [count for count, _, _ in records] == MAIN_A1_COUNTS, records


@pytest.mark.parametrize('provenance', ['another-call', 'absent'])
def test_whole_placeholder_requires_payload_provenance(replay_setup, provenance):
    setup = _keep_setup(replay_setup)
    _rewrite_candidate(setup, payload_changes={'tool_call_id': 'another-call'},
                       missing_provenance=provenance == 'absent')
    records = _replay_literal_reference(setup)
    print(f'A2_WHOLE_PLACEHOLDER {provenance} records={records}')
    assert [count for count, _, _ in records] == MAIN_A2_COUNTS, records


@pytest.mark.parametrize('field', ['tool_call_id', 'session_id', 'tool_name', 'content'])
def test_candidate_provenance_and_content_negatives(replay_setup, field):
    setup = _keep_setup(replay_setup)
    value = setup[4][:-1] + 'X' if field == 'content' else 'another-' + field
    _rewrite_candidate(setup, payload_changes={field: value})
    records = _replay_literal_reference(setup)
    print(f'PROVENANCE_NEGATIVE {field} records={records}')
    assert [count for count, _, _ in records] == MAIN_NEGATIVE_COUNTS[field], records


@pytest.mark.parametrize('stored_tool', [None, ''], ids=['null-tool', 'empty-tool'])
def test_unnamed_candidate_requires_host_tool_provenance(replay_setup, stored_tool):
    setup = _keep_setup(replay_setup)
    engine, host, _, _, body = setup
    _rewrite_candidate(setup, payload_changes={'tool_name': 'another-tool'})
    with sqlite3.connect(engine._config.database_path) as conn:
        conn.execute('UPDATE messages SET tool_name=? WHERE store_id=3', (stored_tool,))
    assert host[2]['tool_name'] == 'dump'
    assert engine._message_replay_identity(host[2])[1] != body
    records = _replay_literal_reference(setup)
    print(f'UNNAMED_TOOL_PROVENANCE {stored_tool!r} records={records}')
    # Main 7828784f, round-5c main-negative.log, with no stored tool name.
    assert [count for count, _, _ in records] == [9, 11, 13, 15, 17, 20, 22, 24], records


def test_unnamed_candidate_requires_equal_content(replay_setup):
    setup = _keep_setup(replay_setup)
    engine, host, _, _, body = setup
    _rewrite_candidate(setup, payload_changes={'content': body[:-1] + 'X'})
    with sqlite3.connect(engine._config.database_path) as conn:
        conn.execute('UPDATE messages SET tool_name=NULL WHERE store_id=3')
    host[2].pop('tool_name')
    assert not host[2].get('name')
    assert engine._message_replay_identity(host[2])[1] != body
    records = _replay_literal_reference(setup)
    print(f'UNNAMED_CONTENT_MISMATCH records={records}')
    assert [count for count, _, _ in records] == [9, 11, 13, 15, 17, 20, 22, 24], records


# Eight-turn engine replay on main 7828784f, main-2703-r5b-4.log.
MAIN_A1_COUNTS = [10, 12, 14, 16, 19, 22, 24, 26]
MAIN_A2_COUNTS = [9, 11, 13, 15, 18, 21, 23, 25]
MAIN_NEGATIVE_COUNTS = {
    'tool_call_id': MAIN_A2_COUNTS,
    'session_id': MAIN_A2_COUNTS,
    'tool_name': MAIN_A2_COUNTS,
    'content': [9, 11, 13, 15, 17, 20, 22, 24],
}


@pytest.mark.parametrize('warm', [False, True], ids=['cold-cache', 'warm-cache'])
def test_rolled_back_candidate_cannot_keep_host_row(replay_setup, monkeypatch, warm):
    setup = _keep_setup(replay_setup)
    engine, host, _, _, body = setup
    stored = engine._store.get_session_messages(SESSION, limit=100_000)[2]
    _rewrite_candidate(setup, content='DISTINCT_DURABLE_C ' * 20)
    assert engine.drain_embedding_maintenance(timeout=10)
    names = ('_legacy_recovered_source_generation', '_legacy_recovered_source_identities',
             '_legacy_recovered_source_store')
    if warm:
        assert engine._message_replay_identity(host[2])[1] != body
    before = {name: deepcopy(getattr(engine, name)) if name.endswith('_identities')
              else getattr(engine, name) for name in names if hasattr(engine, name)}
    inside_generation = []

    def rolled_back(*args, **kwargs):
        assert engine._store._conn.in_transaction
        engine._store._conn.execute("""INSERT INTO messages
            (role,content,tool_call_id,tool_name,timestamp,observed_at,
             session_id,conversation_id,source,observed_at_source)
            VALUES ('tool',?,'c1','dump',1762700006,1762700006,?,?,'cli','host_message_timestamp')""",
            (stored['content'], SESSION, CONVERSATION))
        assert engine._message_replay_identity(host[2])[1] == body
        assert engine._legacy_recovered_source_identities
        inside_generation.append(engine._store.write_generation())
        raise OSError('injected ingest rollback after candidate cached')

    with monkeypatch.context() as patch:
        patch.setattr(engine, '_ingest_messages_transaction', rolled_back)
        failures = engine._ingest_failure_count
        engine.ingest(deepcopy(host))
        assert engine._ingest_failure_count == failures + 1
        assert 'injected ingest rollback' in engine._last_ingest_error
    # SQLite total_changes includes rolled-back writes: generation alone cannot
    # invalidate this cache. The engine transaction must restore it explicitly.
    assert inside_generation == [engine._store.write_generation()]
    for name in names:
        if name in before:
            assert getattr(engine, name) == before[name]
        else:
            assert not hasattr(engine, name), name
    assert engine._store.get_session_messages(SESSION, limit=100_000)[2]['content'] != stored['content']
    assert len(engine._store.get_session_messages(SESSION, limit=100_000)) == 5
    assert engine._message_replay_identity(host[2])[1] != body
    records = _replay_literal_reference(setup)
    assert [count for count, _, _ in records] == MAIN_NEGATIVE_COUNTS['content'], records
    print(f'ROLLBACK warm={warm} records={records}')
