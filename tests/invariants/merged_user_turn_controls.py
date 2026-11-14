"""Unchanged historical host traces from the #2707 round-four experiment.

Private Design-B preservation assertions are capability checks: main has no
preservation seam. All ordered row comparisons and host inputs remain active.
"""
from copy import deepcopy
import hashlib
import json
import sqlite3
import sys

from tests.differential.paired import compare_stages as _compare_round2
from tests.differential.session import _capture, _identities, _open, _row, _rows


def test_round2_existing_alias_fallback_matches_main(tmp_path, monkeypatch, versions,
                                                   record_property, seed_alias):
    """Both reviewers' old-alias/cache shapes, followed by one transient alias failure."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    monkeypatch.setenv('LCM_INGEST_LEDGER', 'shadow')
    results = {}
    for name in ('main', 'candidate'):
        directory = tmp_path / name
        ctx = _open(versions[name], directory, monkeypatch)
        engine = ctx.engine
        q, reply = _row('Q', 9., 'assistant'), _row('A', 21., 'assistant')
        try:
            if seed_alias:
                engine.ingest([q, _row('P1\n\nR', 10.), reply])
                sid = _rows(engine)[1]['store_id']
                engine._store.write_metadata_json([f'host_rewrite_identity:{sid}'], json.dumps({
                    'version': 1, 'content': 'R',
                    'stored_sha256': hashlib.sha256(b'P1\n\nR').hexdigest()}))
                engine._store.add_message_relations([[(sid, 'alt_stamp', None, None, 20.)]])
                live = [q, _row('R', 20.), reply]
            else:
                engine.ingest([q])
                live = _capture(ctx, [q], 'P1', 'R', 10., 20.)
                live[-1] = reply
                engine.ingest(live)
        finally:
            engine.shutdown()
        ctx = _open(versions[name], directory, monkeypatch)
        engine = ctx.engine
        try:
            live = deepcopy(live)
            engine.ingest(live)
            if not seed_alias:
                engine.ingest(live)  # Claude's second warm replay after recreation.
            previous = _identities(engine)
            live = _capture(ctx, live)
            calls = []
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                original = engine._store_merged_user_aliases

                def once(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        raise sqlite3.OperationalError('database is locked')
                    return original(*args, **kwargs)

                monkeypatch.setattr(engine, '_store_merged_user_aliases', once)
            engine.ingest(live)
            stored = _identities(engine)
            assert stored[:len(previous)] == previous
            assert engine._ingest_failure_count == 0
            # The successful first turn intentionally preserved P1 only on the
            # candidate. Compare that existing slot in main's host form; the
            # failed turn and every additional row must be identical to main.
            if not seed_alias:
                stored = [(role, 'R', 20., session, conv) if text == 'P1\n\nR'
                          else (role, text, stamp, session, conv)
                          for role, text, stamp, session, conv in stored]
            results[name] = [stored]
        finally:
            engine.shutdown()
    _compare_round2(results, record_property)


def test_round2_fallback_then_again_matches_main(tmp_path, monkeypatch, versions,
                                               record_property, failure):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results, attempts = {}, []
    for name in ('main', 'candidate'):
        ctx = _open(versions[name], tmp_path / name, monkeypatch)
        engine = ctx.engine
        try:
            if failure == 'redaction':
                engine._config.sensitive_patterns_enabled = True
                engine._config.sensitive_patterns = ['password_assignment']
            q = _row('Q', 9., 'assistant')
            engine.ingest([q])
            live = _capture(ctx, [q], 'password=fake-test-value' if failure == 'redaction' else 'P1',
                            'R', 10., 20.)
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                original = engine._store_merged_user_aliases

                def fail(*args, **kwargs):
                    attempts.append(1)
                    if failure == 'persistent' or failure == 'transient' and len(attempts) == 1:
                        raise sqlite3.OperationalError('database is locked')
                    return original(*args, **kwargs)

                monkeypatch.setattr(engine, '_store_merged_user_aliases', fail)
            stages = []
            engine.ingest(live)
            stages.append(_identities(engine))
            engine.ingest(live)
            stages.append(_identities(engine))
            engine.ingest([*live, _row('N', 60.), _row('C', 61., 'assistant')])
            stages.append(_identities(engine))
            results[name] = stages
        finally:
            engine.shutdown()
    record_property('preservation_attempts', len(attempts))
    _compare_round2(results, record_property)
    assert len(attempts) == (1 if hasattr(engine, '_store_merged_user_aliases') else 0)


def test_round3_watched_capture_stays_retired_after_eviction(tmp_path, monkeypatch, versions, record_property):
    """S1: a stored head must never activate its skipped capture on a later turn."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results = {}
    for name in ('main', 'candidate'):
        ctx = _open(versions[name], tmp_path / name, monkeypatch)
        engine = ctx.engine
        try:
            q, head = _row('Q', 9., 'assistant'), _row('P1', 10.)
            engine.ingest([q, head])
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message='R',
                                      conversation_history=[q, head, _row('P2', 15.), _row('R', 20.)])
            head.update(content='R', timestamp=20.)
            live = [q, head, _row('A', 21., 'assistant')]
            engine.ingest(live)
            stages = [_identities(engine)]
            for turn in range(9):
                live.extend([_row(f'N{turn}', 30. + turn * 2),
                             _row(f'A{turn}', 31. + turn * 2, 'assistant')])
                engine.ingest(live)
                stages.append(_identities(engine))
            engine.ingest(live)
            results[name] = [*stages, _identities(engine)]
        finally:
            engine.shutdown()
    record_property('stored_rows', json.dumps(results))
    _compare_round2(results, record_property)
    assert len(results['candidate'][-1]) == 22


def _seed_round3_aliases(engine):
    for row in _rows(engine):
        if row['role'] == 'user':
            sid = row['store_id']
            engine._store.write_metadata_json([f'host_rewrite_identity:{sid}'], json.dumps({
                'version': 1, 'content': 'R',
                'stored_sha256': hashlib.sha256(row['content'].encode()).hexdigest()}))
            engine._store.add_message_relations([[(sid, 'alt_stamp', None, None, row['observed_at'] + 10.)]])


def _fail_round3_preservation_once(engine, monkeypatch):
    calls, original = [], engine._store_merged_user_aliases

    def once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('one-shot preservation failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, '_store_merged_user_aliases', once)
    return calls


def test_round3_rewritten_prefix_drops_false_fallback_marks(tmp_path, monkeypatch, versions, record_property):
    """S2: UID B at a shifted position is replay, not a replacement of UID A."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results = {}
    for name in ('main', 'candidate'):
        directory = tmp_path / name
        engine = _open(versions[name], directory, monkeypatch).engine
        try:
            initial = [_row('Q', 1., 'assistant'), dict(_row('P1\n\nR', 10.), message_uid='uid-A'),
                       _row('A', 21., 'assistant'), dict(_row('P2\n\nR', 30.), message_uid='uid-B'),
                       _row('B', 41., 'assistant')]
            engine.ingest(initial)
            _seed_round3_aliases(engine)
        finally:
            engine.shutdown()
        ctx = _open(versions[name], directory, monkeypatch)
        engine = ctx.engine
        try:
            live = [initial[0], dict(initial[1], content='R', timestamp=20.), initial[2],
                    dict(initial[3], content='R', timestamp=40.), initial[4]]
            engine.ingest(live)
            incoming = _capture(ctx, [_row('summary', 1., 'assistant'), live[3], live[4]])
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                calls = _fail_round3_preservation_once(engine, monkeypatch)
            engine.ingest(incoming)
            results[name] = [_identities(engine)]
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                assert len(calls) == 1
            assert engine._ingest_failure_count == 0
        finally:
            engine.shutdown()
    record_property('stored_rows', json.dumps(results))
    _compare_round2(results, record_property)
    assert len(results['candidate'][0]) == 8


def test_round3_new_uid_keeps_host_time_through_fallback(tmp_path, monkeypatch, versions, record_property):
    """S3: the alias's stored stamp belongs to replay, not a proven new UID."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results, cache_stamps, projected_stamps, fallback_stamps = {}, {}, {}, {}
    for name in ('main', 'candidate'):
        directory = tmp_path / name
        engine = _open(versions[name], directory, monkeypatch).engine
        try:
            engine.ingest([dict(_row('P1\n\nR', 10.), message_uid='uid-A'), _row('A', 21., 'assistant')])
            _seed_round3_aliases(engine)
        finally:
            engine.shutdown()
        ctx = _open(versions[name], directory, monkeypatch)
        engine = ctx.engine
        try:
            live = [dict(_row('R', 20.), message_uid='uid-A'), _row('A', 21., 'assistant')]
            engine.ingest(live)
            incoming = _capture(ctx, [dict(live[0], message_uid='uid-NEW'), live[1]])
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                sender = sys.modules[engine.__module__.rsplit('.', 1)[0] + '.sender_capture']
                projection = sender.merged_user_projection(engine, incoming)
                projected_stamps[name], fallback_stamps[name] = projection[0][0]['timestamp'], projection[3][0]['timestamp']
                calls = _fail_round3_preservation_once(engine, monkeypatch)
            engine.ingest(incoming)
            results[name] = [_identities(engine)]
            cache_stamps[name] = engine._last_active_replay_messages[0]['timestamp']
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                assert len(calls) == 1
            assert incoming[0]['timestamp'] == 20.
            assert engine._ingest_failure_count == 0
        finally:
            engine.shutdown()
    record_property('stored_rows', json.dumps(results))
    record_property('cache_stamps', json.dumps(cache_stamps))
    _compare_round2(results, record_property)
    assert len(results['candidate'][0]) == 5
    assert [row[2] for row in results['candidate'][0] if row[1] == 'R'] == [20.]
    expected = {'candidate': 20.} if hasattr(engine, '_store_merged_user_aliases') else {}
    assert projected_stamps == fallback_stamps == expected
    assert cache_stamps == {'main': 20., 'candidate': 20.}


def test_round2_fallback_has_own_evidence_retry(tmp_path, monkeypatch, versions, record_property):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    monkeypatch.setenv('LCM_INGEST_LEDGER', 'shadow')
    results, faults = {}, {}
    for name in ('main', 'candidate'):
        ctx = _open(versions[name], tmp_path / name, monkeypatch)
        engine = ctx.engine
        try:
            initial = [_row('U1', 1.), _row('A1', 2., 'assistant'),
                       _row('U2', 3.), _row('A2', 4., 'assistant')]
            engine.ingest(initial)
            live = _capture(ctx, [*initial[:3], _row('I', 3.5), initial[3]])
            fallen_back, calls = [], []
            if name == 'candidate' and hasattr(engine, '_store_merged_user_aliases'):
                def fail_alias(*_args, **_kwargs):
                    fallen_back.append(1)
                    raise ValueError('injected preservation failure')

                monkeypatch.setattr(engine, '_store_merged_user_aliases', fail_alias)
            original = engine._identity_anchor_prematch

            def prematch(*args, **kwargs):
                if (name == 'main' or not hasattr(engine, '_store_merged_user_aliases') or fallen_back) and not calls:
                    calls.append(1)
                    raise RuntimeError('one-shot fallback pre-match failure')
                return original(*args, **kwargs)

            monkeypatch.setattr(engine, '_identity_anchor_prematch', prematch)
            engine.ingest(live)
            first_run = _identities(engine)
            engine.ingest(live)
            results[name] = [first_run, _identities(engine)]
            faults[name] = len(calls)
        finally:
            engine.shutdown()
    assert faults == {'main': 1, 'candidate': 1}
    _compare_round2(results, record_property)
    assert [r[1] for r in results['candidate'][0]].count('I') == 1
    assert [r[1] for r in results['candidate'][0]].count('A2') == 1
