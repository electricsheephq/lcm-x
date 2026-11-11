"""#2624: strict proof, candidate scope, atomicity, and foreground replay safety."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_lcm import aux_session, engine as engine_module, host_row_recovery as recovery
from hermes_lcm import ingest_ledger, maintenance_scheduler as schedulers
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from tests.test_issue_519_replaced_tail import C, P, _Rotation

def pair(stamps=True, uids=False):
    return [dict(role=role, content=text, **({"timestamp": 1900000000 + i} if stamps else {}),
                 **({"message_uid": "stranded-uid-" + str(i)} if uids else {}))
            for i, (role, text) in enumerate((("user", "stranded question 2624"),
                                             ("assistant", "stranded reply 2624")))]

def tools(results=2, reply=True):
    return [pair()[0], {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-2624-" + str(i), "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}} for i in range(2)]},
        *[{"role": "tool", "content": "tool payload 2624 " + str(i),
           "tool_call_id": "call-2624-" + str(i)} for i in range(results)],
        *([pair()[1]] if reply else [])]

class Host:
    def __init__(self, lists, tip=P):
        self.lists, self.tip, self.calls = lists, tip, []

    def get_messages_as_conversation(self, session_id, *, include_ancestors=True,
                                     include_inactive=False, repair_alternation=True,
                                     include_row_ids=False, include_compacted=False):
        self.calls.append(("read", session_id))
        return copy.deepcopy(self.lists[session_id])

    def get_compression_tip(self, session_id):
        self.calls.append(("tip", session_id))
        return self.tip

@pytest.fixture
def case(tmp_path, monkeypatch, caplog):
    made = []
    yield _make_case(tmp_path, monkeypatch, caplog, made)
    for engine in made:
        engine._close_storage()

def _make_case(tmp_path, monkeypatch, caplog, live_engines, recover=True):
    from tests import test_issue_519_replaced_tail as rotation_module
    config = rotation_module._config
    if recover:  # the control reuses the wrapped config
        monkeypatch.setattr(rotation_module, '_config', lambda path, **kw: config(
            path, temporal_rollups_enabled=False, embeddings_enabled=False, **kw))
    r = _Rotation(tmp_path, monkeypatch)
    (home := tmp_path / "home").mkdir(exist_ok=True)
    state = home / "state.db"
    with sqlite3.connect(state) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, ended_at REAL, "
                     "end_reason TEXT, model_config TEXT, source TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?)", [
            (P, None, 1, "compression", None, "acp"), (C, P, None, None, None, "acp")])
    jobs = []
    live_engines.append(r.engine)
    r.engine._config.host_row_recovery_enabled = recover
    if recover:
        monkeypatch.setattr(recovery._HOST_ROW_RECOVERY_SCHEDULER, "schedule",
                            lambda key, job: jobs.append(job) or True)
        monkeypatch.setattr(recovery, "_SEEN", set())
        monkeypatch.setattr(recovery, "_WARNED", set())
    prefix = r.restored()
    host = Host({C: prefix + pair()})
    def start(sid=P, conversation=False):
        live = LCMEngine(config=copy.deepcopy(r.engine._config), hermes_home=str(home))
        live_engines.append(live)
        live._host_session_db = host
        if sid:
            live.bind_session_state(host, sid)
            live.on_session_start(sid, platform="acp", context_length=200_000,
                                  **({"conversation_id": P} if conversation else {}))
        return live

    def run(job=None):
        from tests.test_issue_2624_host_row_recovery_review import _run_in_recovery_thread
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            _run_in_recovery_thread(job or jobs[-1])
        for text in [record.getMessage() for record in caplog.records if record.name.startswith("hermes_lcm")]:
            assert C not in text and all(m["content"] not in text for m in host.lists.get(C, []) if m.get("content"))
        return [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING and rec.name == recovery.__name__]

    def control():  # R6-48: the same store and host on their own paths, recovery off
        return _make_case(tmp_path / "control", monkeypatch, caplog, live_engines, recover=False)

    r.engine._close_storage()
    live = start()
    return type("Case", (), dict(r=r, live=live, host=host, prefix=prefix, jobs=jobs, state=state,
                                 start=staticmethod(start), run=staticmethod(run), control=staticmethod(control)))()

def ident(rs, conv=None):
    return [(r['session_id'], r['conversation_id'], r['role'], r['content']) for r in rs if conv in (None, r['conversation_id'])]

def rows(case, sid=C):
    return case.live._store.get_session_messages(sid)

def assert_suffix(case, suffix, sid=C):
    stored = rows(case, sid)
    assert [(m["role"], m["content"]) for m in stored] == [(m["role"], m["content"]) for m in suffix]
    assert all(m["session_id"] == sid and m["conversation_id"] == P for m in stored)
    assert all(case.live._store.read_metadata_json("host_row_recovered:" + str(row["store_id"]))
               == {"version": 1, "issue": 2624} for row in stored)
    return stored

@pytest.mark.parametrize("suffix", [pair()[1:], pair(), tools(), tools(reply=False),
                                    tools(results=1, reply=False), tools(results=0, reply=False), []],
                         ids=["reply", "pair", "complete", "tool-loop", "partial", "unanswered", "empty"])
def test_suffix_shapes_and_double_worker(case, suffix):
    case.host.lists[C] = case.prefix + suffix
    before = len(rows(case, P))
    assert not case.run()
    assert_suffix(case, suffix)
    assert not case.run()  # bypass once-per-process set: the proof alone prevents duplication
    assert_suffix(case, suffix)
    recovery._SEEN.clear()
    case.start()
    assert not case.run()  # a fresh live engine and detached store connection also write nothing
    assert_suffix(case, suffix)
    assert len(rows(case, P)) == before

@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("root_more", [False, True])
@pytest.mark.parametrize("conversation", [True, False])
@pytest.mark.parametrize("shape", ["uids", "stamps", "neither"])
def test_rebind_never_binds_child_rows_from_root(case, monkeypatch, mode, root_more, conversation, shape):
    monkeypatch.setenv("LCM_INGEST_LEDGER", mode)
    stranded = pair(stamps=shape != "neither", uids=shape == "uids")
    case.host.lists[C] = case.prefix + stranded
    case.run()
    recovered = assert_suffix(case, stranded)
    def sql_rows(table, engine=case.live):
        return ingest_ledger._rows(engine._store.connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
    def bindings():
        return sql_rows('occurrence_bindings') if case.live._store._ingest_ledger_tables_exist() else []
    def ingest_checked(engine, messages):
        prior = {b['occ_id'] for b in bindings()}
        engine.ingest(messages)
        for b in bindings():
            if b['occ_id'] not in prior and b['store_id'] is not None:
                assert engine._store.get(b['store_id'])['conversation_id'] == engine._conversation_id
    root = [dict(role=r, content=r + ' root-only 2624', timestamp=2000000010 + i) for i, r in enumerate(('user', 'assistant'))]
    if root_more:
        ingest_checked(case.live, root)
        assert not {b['store_id'] for b in bindings()} & {m['store_id'] for m in recovered}
    before = sql_rows('messages')
    child = case.start(C, conversation=conversation)
    ingest_checked(child, case.host.lists[C])
    after = sql_rows('messages')
    assert all(row in after for row in before)
    # The hand-off row is LCM-X's own summary scaffold; this ingest skips it (main stores it, #2630).
    real = [m for m in case.host.lists[C] if not child._is_replayed_context_scaffold_message(m)]
    assert len(real) == len(case.host.lists[C]) - 1
    if conversation:
        assert len(after) == len(before)
    else:  # R6-47 / #2630: accepted surplus, counted separately in each conversation.
        assert [(m['role'], m['content']) for m in after if m['conversation_id'] == C] == [(m['role'], m['content']) for m in real]
        for m in stranded:
            assert Counter(r['conversation_id'] for r in after if (r['role'], r['content']) == (m['role'], m['content'])) == {P: 1, C: 1}
    assert all(b['disposition'] == 'migrated' for b in bindings() if b['store_id'] in {m['store_id'] for m in recovered})
    assert len(bindings()) == len({b['store_id'] for b in bindings()})
    ingest_checked(case.start(C, conversation=conversation), case.host.lists[C])
    again = sql_rows('messages')
    if conversation:
        assert again == after
    else:  # R6-48 (d') and (f): identities against the same sequence with recovery off
        ctl = case.control()
        ctl.host.lists[C] = ctl.prefix + stranded
        if root_more:
            ctl.live.ingest(root)
        n_ctl = len(sql_rows('messages', ctl.live))
        ctl.start(C).ingest(ctl.host.lists[C])
        ctl_after = sql_rows('messages', ctl.live)
        ctl.start(C).ingest(ctl.host.lists[C])
        ctl_again = sql_rows('messages', ctl.live)
        handoff = [(C, C, m['role'], m['content']) for m in case.host.lists[C] if m not in real]
        assert len(handoff) == 1 and ident(ctl_after, C).count(handoff[0]) == 1
        assert [r for r in ident(ctl_after, C) if r != handoff[0]] == ident(after, C)  # (f)
        assert ident(ctl_after[:n_ctl]) == [r for r in ident(after[:len(before)]) if r[0] != C or r[1] != P]
        assert ident(again[len(after):]) == ident(ctl_again[len(ctl_after):])  # (d')
        for m in stranded:
            key = (m['role'], m['content'])
            assert Counter(r[1] for r in ident(again) if r[2:] == key) == Counter(
                r[1] for r in ident(ctl_again) if r[2:] == key) + Counter({P: 1})
    case.start()
    assert len(case.jobs) == 2  # root once, child once, including the second child bind

@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("order", ["recovery", "foreground"])
@pytest.mark.parametrize("shape", ["uids", "stamps", "neither"])
def test_resent_pair_survives_both_orders_and_cold_restart(case, monkeypatch, mode, order, shape):
    monkeypatch.setenv("LCM_INGEST_LEDGER", mode)
    stranded = pair(stamps=shape != "neither", uids=shape == "uids")
    case.host.lists[C] = case.prefix + stranded
    resent = [{**row, **({"timestamp": row["timestamp"] + 100} if "timestamp" in row else {}),
               **({"message_uid": "resent-uid-" + str(i)} if "message_uid" in row else {})} for i, row in enumerate(stranded)]
    root_prefix = [{"role": "user", "content": "root-history question 2624", "timestamp": 1800000000},
                   {"role": "assistant", "content": "root-history reply 2624", "timestamp": 1800000001}]
    root_host = [case.live._store.to_openai_msg(m) for m in rows(case, P)] + root_prefix
    case.live.ingest(root_host)
    n = len(rows(case, P))
    if order == "recovery":
        case.run()
    root_host += resent
    case.live.ingest(root_host)
    assert len(rows(case, P)) == n + 2
    if order == "foreground":
        case.run()
    assert_suffix(case, stranded)
    case.live.ingest(root_host)
    assert len(rows(case, P)) == n + 2
    cold = case.start()
    new = [{"role": "user", "content": "new cold question 2624"},
           {"role": "assistant", "content": "new cold reply 2624"}]
    cold.ingest(root_host + new)
    assert len(rows(case, P)) == n + 4
    if mode == "shadow":
        assert not cold._store.connection.execute(
            "SELECT store_id FROM occurrence_bindings GROUP BY store_id HAVING count(*)>1").fetchall()

def proof_update(case, **changes):
    payload = case.live._store.read_metadata_json('compaction_commit_proof:' + C)
    case.live._store.write_metadata_json(['compaction_commit_proof:' + C], json.dumps({**payload, **changes}))

@pytest.mark.parametrize('kind', ['missing', 'native', 'empty', 'merged', 'tail', 'reader',
                                 'handle', 'signature', 'orphan', 'double', 'ignored', 'reset'])
def test_no_proof_matrix_warns_once_without_writes(case, monkeypatch, kind):
    if kind == 'missing':
        case.live._store.append(C, {'role': 'assistant', 'content': 'already held 2624'}, 5, conversation_id=P)
        case.live._store.connection.execute('DELETE FROM metadata WHERE key=?', ('compaction_commit_proof:' + C,))
        case.live._store.connection.commit()
    elif kind == 'native':
        proof_update(case, native=True)
    elif kind == 'empty':
        proof_update(case, effective_sha256=[], effective_sha256_v4=[])
    elif kind == 'merged':
        case.host.lists[C][len(case.prefix) - 1]['content'] += '\n\n' + pair()[0]['content']
    elif kind == 'tail':
        case.host.lists[C] = pair() + case.host.lists[C]
    elif kind == 'reader':
        monkeypatch.setattr(case.host, 'lists', {})
    elif kind == 'handle':
        monkeypatch.setattr(case.host, 'get_messages_as_conversation', None)
    elif kind == 'signature':
        monkeypatch.setattr(case.host, 'get_messages_as_conversation', lambda wrong: [])
    elif kind == 'orphan':
        case.host.lists[C] += [{'role': 'tool', 'content': 'orphan 2624', 'tool_call_id': 'absent'}]
    elif kind == 'double':
        case.host.lists[C] = case.prefix + tools() + [tools()[2]]
    elif kind == 'ignored':
        # Exercise recovery's gate without the optional regex dependency in the hoststub runtime.
        monkeypatch.setattr(LCMEngine, '_matches_ignore_message_patterns',
                            lambda det, m, **kw: m.get('content') == pair()[0]['content'])
    else:
        payload = case.r.proof
        with case.live._store.atomic():
            case.live._store.connection.execute('UPDATE lcm_lifecycle_state SET last_reset_at=? WHERE conversation_id=?',
                                               (payload['created_at'], P))
    before = rows(case)
    warnings = case.run()
    assert rows(case) == before
    assert len(warnings) == 1 and warnings[0].startswith('LCM host-row recovery unproven: candidate=')
    assert hashlib.sha256(C.encode()).hexdigest()[:12] in warnings[0]
    assert C not in warnings[0] and not case.run() and rows(case) == before
    assert not case.live._store.connection.execute("SELECT 1 FROM metadata WHERE key LIKE 'host_row_recovered:%'").fetchall()

@pytest.mark.parametrize('change', ['reset', 'replace'])
def test_r3_change_after_proof_before_transaction(case, monkeypatch, change):
    original, calls = recovery._strict_proof_cursor, []
    def paused(det, messages):
        cursor = original(det, messages)
        calls.append(cursor)
        assert cursor == len(case.prefix)
        if len(calls) == 1:
            with case.live._store.atomic(immediate=True):
                if change == 'reset':
                    case.live._store.connection.execute(
                        'UPDATE lcm_lifecycle_state SET last_reset_at=? WHERE conversation_id=?',
                        (case.r.proof['created_at'] + 1, P))
                else:
                    proof_update(case, race_nonce=1)
        return cursor
    monkeypatch.setattr(recovery, '_strict_proof_cursor', paused)
    before = recovery.host_row_recovery_counts()['raced']
    assert len(case.run()) == 1 and not rows(case)
    assert recovery.host_row_recovery_counts()['raced'] == before + 1
    assert len(calls) == 1  # the raw/reset guard aborts before the in-transaction proof

@pytest.mark.parametrize('phase', ['first_insert', 'last_insert'])
def test_atomic_rows_and_marks_rollback_then_clean_retry(case, monkeypatch, phase):
    append = MessageStore._append_protected_batch
    seen = []
    def failing(store, *args, **kwargs):
        if phase == 'last_insert':
            ids = append(store, *args, **kwargs)
            seen.extend(ids)
            assert len(ids) == 2 and store.connection.in_transaction
            assert all(store.read_metadata_json('host_row_recovered:' + str(i)) for i in ids)
            raise RuntimeError('injected interruption after rows and marks')
        metadata = kwargs['metadata_factory']
        def hook(msg, store_id):
            result = metadata(msg, store_id)
            seen.append(store_id)
            assert result and store.connection.in_transaction
            raise RuntimeError('injected transaction interruption')
        kwargs['metadata_factory'] = hook
        return append(store, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(MessageStore, '_append_protected_batch', failing)
        assert len(case.run()) == 1
    assert seen and not rows(case)
    assert not case.live._store.connection.execute("SELECT 1 FROM metadata WHERE key LIKE 'host_row_recovered:%'").fetchall()
    assert not case.run()
    assert_suffix(case, pair())

@pytest.mark.parametrize('order', ['foreground_commits_first', 'recovery_commits_first'])
def test_foreground_during_recovery_both_commit_orders(case, monkeypatch, order):
    ready, resume = threading.Event(), threading.Event()
    inserted, commit, attempted = threading.Event(), threading.Event(), threading.Event()
    original = recovery._strict_proof_cursor
    def paused(det, messages):
        cursor = original(det, messages)
        if not det._store.connection.in_transaction:
            ready.set()
            assert resume.wait(15)
        return cursor
    monkeypatch.setattr(recovery, '_strict_proof_cursor', paused)
    append = MessageStore._append_protected_batch
    def held(store, sid, *args, **kwargs):
        ids = append(store, sid, *args, **kwargs)
        if sid == C and order == 'recovery_commits_first':
            inserted.set()
            assert commit.wait(15)
        return ids
    monkeypatch.setattr(MessageStore, '_append_protected_batch', held)
    foreground = [case.live._store.to_openai_msg(m) for m in rows(case, P)] + pair(stamps=False)
    before = len(rows(case, P))
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(case.jobs[0])
        assert ready.wait(15)
        if order == 'foreground_commits_first':
            case.live.ingest(foreground)
        resume.set()
        if order == 'recovery_commits_first':
            assert inserted.wait(15)
            def ingest_live():
                attempted.set()
                case.live.ingest(foreground)
            concurrent = pool.submit(ingest_live)
            assert attempted.wait(15)
            commit.set()
        worker.result(timeout=20)
        if order == 'recovery_commits_first':
            concurrent.result(timeout=20)
    assert_suffix(case, pair())
    assert len(rows(case, P)) == before + 2
    case.live.ingest(foreground)
    assert len(rows(case, P)) == before + 2

def test_two_connections_race_after_same_proof(case, monkeypatch):
    original, barrier, connections = recovery._strict_proof_cursor, threading.Barrier(2), set()
    def paused(det, messages):
        cursor = original(det, messages)
        if not det._store.connection.in_transaction:
            connections.add(id(det._store.connection))
            barrier.wait(timeout=15)
        return cursor
    monkeypatch.setattr(recovery, '_strict_proof_cursor', paused)
    before = recovery.host_row_recovery_counts()['raced']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(case.jobs[0]) for _ in range(2)]
        for future in futures:
            future.result(timeout=25)
    assert len(connections) == 2
    assert_suffix(case, pair())
    assert recovery.host_row_recovery_counts()['raced'] == before + 1

def shared_snapshot():
    from hermes_lcm import engine_registry
    modules = (aux_session, ingest_ledger, engine_registry)
    state = {(module.__name__, name): copy.copy(value) for module in modules for name, value in vars(module).items()
             if name.startswith('_') and isinstance(value, (dict, set))}
    state['tails'] = set(schedulers._PENDING_TAIL_ENGINES)
    for name, scheduler in vars(schedulers).items():
        if isinstance(scheduler, schedulers._RollupMaintenanceScheduler) and scheduler is not recovery._HOST_ROW_RECOVERY_SCHEDULER:
            state[name] = {k: copy.copy(v) for k, v in vars(scheduler).items() if isinstance(v, (dict, set, list)) or k == '_jobs'}
    state['registry'] = (dict(engine_registry._ACTIVE_ENGINES_BY_SESSION_ID),
                         dict(engine_registry._ACTIVE_ENGINES_BY_CONVERSATION_ID))
    return state

def test_detached_scope_registries_exit_and_connections(case, monkeypatch):
    instances, calls = [], []
    constructor, close = LCMEngine.__init__, LCMEngine._close_storage
    def track(det, *args, **kwargs):
        constructor(det, *args, **kwargs)
        instances.append(det)
    def closing(det):
        calls.append(det)
        close(det)
    monkeypatch.setattr(LCMEngine, '__init__', track)
    monkeypatch.setattr(LCMEngine, '_close_storage', closing)
    def forbidden(*args, **kwargs):
        pytest.fail('detached engine entered a forbidden lifecycle method')
    for name in ('on_session_start', 'bind_session_state', 'ingest', 'compress', 'on_session_end', 'shutdown'):
        monkeypatch.setattr(LCMEngine, name, forbidden)
    connections = (case.live._store.connection, case.live._dag._conn, case.live._lifecycle._conn)
    live = dict(case.live.__dict__)
    shared = shared_snapshot()
    flags = {k: v for k, v in vars(engine_module).items() if k.endswith('_WARNING_LOGGED')}
    assert len(flags) == 3
    assert not case.run()
    assert len(instances) == 1 and calls == instances and shared_snapshot() == shared
    assert case.live.__dict__.keys() == live.keys()
    assert all(case.live.__dict__[key] is value for key, value in live.items())
    assert {k: getattr(engine_module, k) for k in flags} == flags
    for attr, flag in [('native_recovery', '_NATIVE_RECOVERY_WARNING_LOGGED'),
                       ('cache_friendly_condensation_enabled', '_CACHE_FRIENDLY_CONDENSATION_WARNING_LOGGED'),
                       ('large_output_transcript_gc_enabled', '_LARGE_OUTPUT_TRANSCRIPT_GC_WARNING_LOGGED')]:
        assert not getattr(case.live._config, attr) or flags[flag]
    before = rows(case)
    schedulers._flush_pending_tails_at_exit()
    assert rows(case) == before and calls == instances
    assert all(conn.execute('SELECT 1').fetchone()[0] == 1 for conn in connections)

@pytest.mark.parametrize('kind', ['branch', 'delegate', 'reset', 'tool', 'no_provenance', 'tip', 'fixed_tip', 'depth2', 'two_leaves'])
def test_candidate_walk_exclusions_and_depth(case, kind):
    with sqlite3.connect(case.state) as conn:
        if kind in ('branch', 'delegate', 'reset'):
            marker = {'branch': '_branched_from', 'delegate': '_delegate_from', 'reset': '_reset_from'}[kind]
            conn.execute('UPDATE sessions SET model_config=? WHERE id=?', (json.dumps({marker: P}), C))
        elif kind == 'tool':
            conn.execute("UPDATE sessions SET source='tool' WHERE id=?", (C,))
        elif kind == 'no_provenance':
            case.live._store.connection.execute('DELETE FROM metadata WHERE key=?', ('compaction_commit_proof:' + C,))
            case.live._store.connection.commit()
        elif kind in ('tip', 'fixed_tip'):
            case.host.tip = C
            if kind == 'fixed_tip':
                case.start(C, conversation=True)
        else:
            other = 'leaf-two-2624'
            conn.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?)', (other, P, None, None, None, 'acp'))
            if kind == 'depth2':
                conn.execute("UPDATE sessions SET ended_at=2,end_reason='compression' WHERE id=?", (other,))
                conn.execute('UPDATE sessions SET parent_session_id=? WHERE id=?', (other, C))
            else:
                payload = {**case.r.proof, 'session_id': other}
                payload['emissions'] = [{**d, 'scope': {**d['scope'], 'session_id': other}}
                                        for d in payload['emissions']]
                case.live._store.write_metadata_json(['compaction_commit_proof:' + other], json.dumps(payload))
                case.host.lists[other] = copy.deepcopy(case.host.lists[C])
    assert not case.run()
    if kind in ('depth2', 'two_leaves'):
        assert_suffix(case, pair())
        if kind == 'two_leaves':
            assert_suffix(case, pair(), sid=other)
    else:
        assert not rows(case) and not any(call[0] == 'read' for call in case.host.calls)

@pytest.mark.parametrize('kind', ['boundary', 'bind', 'turn_rebind', 'seen', 'off', 'stateless', 'ignored', 'side_channel', 'binding_mismatch'])
def test_enqueue_gates_and_switch(case, monkeypatch, kind):
    case.jobs.clear()
    case.host.calls.clear()
    if kind != 'seen':
        recovery._SEEN.clear()
    if kind == 'boundary':
        case.live.on_session_start(P, boundary_reason='compression')
    elif kind == 'bind':
        case.live.bind_session_state(case.host, P)
    elif kind == 'turn_rebind':
        case.live._on_session_start_unlocked(P)
    elif kind == 'seen':
        case.live.on_session_start(P)
    else:
        if kind == 'off':
            case.live._config.host_row_recovery_enabled = False
        elif kind == 'stateless':
            case.live._session_stateless = True
        elif kind == 'ignored':
            case.live._session_ignored = True
        elif kind == 'side_channel':
            monkeypatch.setattr(type(case.live), 'side_channel_active', property(lambda self: True))
        case.live._enqueue_host_row_recovery(C if kind == 'binding_mismatch' else P, {})
    assert not case.jobs and not case.host.calls and not rows(case)

@pytest.mark.parametrize('foreground_first', [False, True, 'bound_only'])
def test_auxiliary_binding_never_enqueues(case, caplog, foreground_first):
    from tests.engine_helpers import TestSessionRollover
    live = case.live if foreground_first else case.start(sid=None)
    if foreground_first is not True:
        case.jobs.clear()
        recovery._SEEN.clear()
    child = TestSessionRollover.HostAgentFrame('aux-child-2624', P, Path(live._hermes_home))
    child.on_session_start(live)
    assert live._session_id == (P if foreground_first else '')
    assert live._thread_context_has_auxiliary_session(child.session_id)
    assert any('is a live child of bound session' in r.getMessage() for r in caplog.records)
    assert len(case.jobs) == int(foreground_first is True)
    live.on_session_start(P)
    assert len(case.jobs) == 1 and case.jobs[0].__closure__[0].cell_contents[6] == P

def test_bound_default_proof_helpers_unchanged(case):
    live = case.live
    messages = [live._store.to_openai_msg(m) for m in rows(case, P)]
    before = (live._durable_commit_proof_payload(), live._cursor_from_durable_commit_proof(messages))
    assert live._session_id == case.host.tip == P
    case.run(case.jobs[0])
    assert_suffix(case, pair())
    assert (live._durable_commit_proof_payload(), live._cursor_from_durable_commit_proof(messages)) == before

def test_log_hygiene_and_counter_receipt(case, caplog):
    before = recovery.host_row_recovery_counts()
    case.run()
    after = recovery.host_row_recovery_counts()
    assert {k: after[k] - before[k] for k in ('candidates', 'recovered_candidates', 'recovered_rows')} == {
        'candidates': 1, 'recovered_candidates': 1, 'recovered_rows': 2}
    records = [r for r in caplog.records if r.name.startswith('hermes_lcm')]
    assert sum(r.levelno == logging.INFO and 'host-row recovery stored:' in r.getMessage() for r in records) == 1
    for record in records:
        assert not record.exc_info
        assert all(value not in record.getMessage() for value in (C, str(case.state), str(case.live._db_path)))

def test_r2_protected_rows_equal_child_normal_ingest(case, monkeypatch, tmp_path):
    from hermes_lcm.ingest_protection import load_externalized_payload, extract_externalized_ref
    from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
    from tests.test_issue_10_persisted_output_spillover import _write_output
    marker = _write_output(Path(case.live._hermes_home) / 'cache/spillover/call-2624-1.txt')
    suffix = tools(results=0, reply=False)
    suffix[1]['tool_calls'].append({'id': 'call-2624-2', 'function': {'name': 'read_file', 'arguments': '{}'}})
    suffix += [{'role': 'tool', 'tool_call_id': 'call-2624-' + str(i), 'content': text}
               for i, text in enumerate([_HERMES_PRUNED_TOOL_PLACEHOLDER, marker, 'bounded payload line\n' * 2500])]
    suffix += pair()[1:]
    case.host.lists[C] = case.prefix + suffix
    baseline_path = tmp_path / 'normal-child.db'
    with sqlite3.connect(baseline_path) as conn:
        case.live._store.connection.backup(conn)
    config = copy.deepcopy(case.live._config)
    config.database_path = str(baseline_path)
    normal = LCMEngine(config=config, hermes_home=case.live._hermes_home)
    try:
        normal.on_session_start(C, conversation_id=P, platform='acp', context_length=200_000)
        normal.ingest(case.host.lists[C])
        expected = normal._store.get_session_messages(C)
        original, scopes = recovery._strict_proof_cursor, []
        def scoped(det, messages):
            scopes.append((det._session_id, det._conversation_id, det._hermes_home))
            assert (det._session_id, det._conversation_id) == (C, P)
            return original(det, messages)
        monkeypatch.setattr(recovery, '_strict_proof_cursor', scoped)
        assert not case.run(case.jobs[0])
        protected = rows(case)
        fields = ('role', 'content', 'tool_calls', 'tool_call_id', 'session_id', 'conversation_id')
        assert [tuple(r.get(k) for k in fields) for r in protected] == [tuple(r.get(k) for k in fields) for r in expected]
        assert len(scopes) == 2 and case.live._session_id == P
        for row in protected:
            if row['role'] == 'tool' and row['tool_call_id'] == 'call-2624-2':
                assert (ref := extract_externalized_ref(row['content']))
                payload = load_externalized_payload(ref, config=config, hermes_home=case.live._hermes_home)
                assert payload['session_id'] == C and payload['content'] == suffix[-2]['content']
        child = case.start(C, conversation=True)
        n = len(rows(case))
        child.ingest(case.host.lists[C])
        assert len(rows(case)) == n
    finally:
        normal._close_storage()

def test_recovery_mark_identity_scorers_and_inspector(case, tmp_path):
    from bench.instruments.reliability.scorers import dupes, multiset
    from scripts import store_wedge_inspect as inspect_store
    case.run()
    recovered = rows(case)
    engine = case.start(C, conversation=True)
    keys = ['host_row_recovered:' + str(row['store_id']) for row in recovered]
    def identity():
        return [(engine._message_replay_identity(row, stored_row=True),
                 tuple(row.get(k) for k in ('store_id', 'session_id', 'conversation_id', 'role', 'content', 'tool_calls', 'tool_call_id')))
                for row in rows(case)]
    marked = identity()
    connection = case.live._store.connection
    sql_rows = [(r['store_id'], r['session_id'], r['role'], r['content']) for r in recovered]
    assert multiset.score([(m['role'], m['content']) for m in pair()], sql_rows)['verdict'] == 'PASS'
    assert dupes.count(case.live._db_path)['key_based_extra_rows_all_time'] == 0
    reads = []
    connection.set_trace_callback(reads.append)
    def metadata_read():
        with pytest.raises(inspect_store.InspectionRefused, match='unverified_ancestry'):
            inspect_store._view_probe(connection, recovered, set(), 0, C, {}, None, recovered)
    try:
        metadata_read()
        with case.live._store.atomic():
            connection.executemany('DELETE FROM metadata WHERE key=?', [(key,) for key in keys])
        assert identity() == marked
        metadata_read()
        assert sum('SELECT key,value FROM metadata' in query for query in reads) == 2
    finally:
        connection.set_trace_callback(None)
        case.live._store.write_metadata_json(keys, json.dumps({'version': 1, 'issue': 2624}))
    assert identity() == marked

@pytest.mark.parametrize('kind', ['descendants', 'tip_missing', 'tip_cycle'])
def test_walk_limits_fail_closed(case, monkeypatch, kind):
    if kind == 'descendants':
        monkeypatch.setattr(recovery, '_MAX_SESSIONS', 1)
    elif kind == 'tip_missing':
        case.host.tip = 'missing-tip-2624'
    else:
        case.host.tip = C
        with sqlite3.connect(case.state) as conn:
            conn.execute('UPDATE sessions SET parent_session_id=? WHERE id=?', (C, C))
    before = recovery.host_row_recovery_counts()['walk_capped']
    assert len(case.run()) == 1 and not rows(case)
    assert recovery.host_row_recovery_counts()['walk_capped'] == before + 1
    assert not any(kind == 'read' for kind, sid in case.host.calls)

def test_unproven_retry_next_start_has_no_durable_recovery_state(case, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(case.host, 'lists', {})
        assert len(case.run()) == 1
    assert not rows(case)
    assert not case.live._store.connection.execute("SELECT 1 FROM metadata WHERE key LIKE 'host_row_recovered:%'").fetchall()
    recovery._SEEN.clear()  # a new process has no in-memory once keys
    case.start()
    case.run()
    assert_suffix(case, pair())

def test_constructor_warn_once_flags_already_set_by_live_engine(case, monkeypatch):
    flags = {'native_recovery': '_NATIVE_RECOVERY_WARNING_LOGGED',
             'cache_friendly_condensation_enabled': '_CACHE_FRIENDLY_CONDENSATION_WARNING_LOGGED',
             'large_output_transcript_gc_enabled': '_LARGE_OUTPUT_TRANSCRIPT_GC_WARNING_LOGGED'}
    for config_name, global_name in flags.items():
        monkeypatch.setattr(engine_module, global_name, False)
        setattr(case.r.engine._config, config_name, True)
    live = case.start()
    assert all(getattr(engine_module, name) for name in flags.values())
    recovery._SEEN.clear()
    live._enqueue_host_row_recovery(P, {})
    assert not case.run()
    assert all(getattr(engine_module, name) for name in flags.values())
    assert_suffix(case, pair())

@pytest.mark.parametrize('malformed', [None, {'role': 'system', 'content': 'unexpected 2624'},
    {'role': 'assistant', 'tool_calls': {}}, {'role': 'assistant', 'tool_calls': [{'id': ''}]},
    {'role': 'tool', 'tool_call_id': ''}])
def test_suffix_malformed_no_write(case, malformed):
    case.host.lists[C] = case.prefix + pair() + [malformed]
    case.jobs[0]()
    assert not rows(case)

def test_host_database_is_opened_readonly_and_handle_uses_defaults(case, monkeypatch):
    before = case.state.read_bytes()
    original, opened = sqlite3.connect, []
    def connect(path, *args, **kwargs):
        if 'state.db' in str(path):
            opened.append((str(path), kwargs))
            assert str(path).endswith('?mode=ro') and kwargs.get('uri') is True
        return original(path, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    case.run()
    assert opened and case.state.read_bytes() == before and case.host.calls == [('tip', P), ('tip', P), ('read', C), ('tip', P)]
    assert_suffix(case, pair())
@pytest.fixture(autouse=True)
def audit_all_plugin_logs(caplog, tmp_path):
    violations = set()
    class Audit(logging.Handler):
        def emit(self, record):
            if record.name != recovery.__name__ and record.threadName != 'lcm-host-row-recovery':
                return
            text = record.getMessage()
            if record.exc_info or any(value in text for value in (
                    C, 'leaf-two-2624', str(tmp_path), 'state.db', 'lcm.db', pair()[0]['content'],
                    pair()[1]['content'], 'tool payload 2624')):
                violations.add((record.name, record.funcName))
    logger = logging.getLogger('hermes_lcm')
    audit = Audit()
    caplog.set_level(logging.DEBUG, logger='hermes_lcm')
    logger.addHandler(audit)
    yield violations
    logger.removeHandler(audit)
    assert not violations, sorted(violations)

def test_all_file_plugin_debug_logs_are_private(case, caplog, audit_all_plugin_logs):
    case.run()
    assert any(r.threadName == 'lcm-host-row-recovery' and 'host-row recovery stored:' in r.getMessage() for r in caplog.records)
    assert not audit_all_plugin_logs, sorted(audit_all_plugin_logs)
