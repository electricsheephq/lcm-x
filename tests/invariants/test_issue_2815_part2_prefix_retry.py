"""#2815 part 2 / T1-211: bounded escape-frame prefix retries (I1/I2/I9/I11).

Synthetic rows only. The provider and token estimator are offline; publication,
identity, persistence, leaf selection and recursive store walks stay real.
"""

from collections import Counter
import hashlib
import json
import logging

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm import store_complete
from hermes_lcm.store_complete import StoreCompleteMixin
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import make_engine, pin_char_counter


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    prompts = []

    def summarize(**kwargs):
        prompts.append(len(kwargs['text']))
        return 'Earlier turns.\nExpand for details about: synthetic turns', 1

    monkeypatch.setattr(engine_module, 'summarize_with_escalation', summarize)
    return prompts


def row(n):
    return {'role': 'user', 'content': f'synthetic row {n:03d} ' + ' abcdefghij' * 12,
            'timestamp': float(n)}


def build(path, ctx=1_000_000, **overrides):
    settings = dict(fresh_tail_count=3, leaf_chunk_tokens=378, dynamic_leaf_chunk_max=378,
                    dynamic_leaf_chunk_enabled=False, context_threshold=0.001,
                    threshold_full_sweep_enabled=False, temporal_rollups_enabled=False,
                    empty_lifecycle_gc_enabled=False, fresh_tail_pressure_yield_enabled=False)
    settings.update(overrides)
    return make_engine(path, window=ctx, conversation_id='conv', **settings)


def rows(engine):
    return engine._store.get_session_messages('S', limit=100_000)


def digest(engine):
    data = [{k: r.get(k) for k in ('store_id', 'session_id', 'conversation_id', 'role',
                                 'content', 'tool_calls', 'tool_call_id')} for r in rows(engine)]
    return len(data), hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def spy(monkeypatch):
    calls = []
    original = StoreCompleteMixin._store_complete_input

    def counted(self, *args, **kwargs):
        record = {'budget': args[7], 'flags': {k: v for k, v in kwargs.items() if k not in ('scanned', 'copy_lookup')}, 'chunk': len(args[0]), 'claim_ids': list(args[1].values()),
                  'mapped_ids': list(args[2].values())}
        calls.append(record)
        out = original(self, *args, **kwargs)
        record['out'] = [sid for _m, ids in out or () for sid in ids]
        record['excluded'] = list(self._store_complete_excluded)
        record['copy_proofs'] = dict(self._store_complete_copy_exclusion_proofs)
        record['tokens'] = count_messages_tokens([m for m, _ids in out or ()])
        return out

    monkeypatch.setattr(StoreCompleteMixin, '_store_complete_input', counted)
    return calls


def field(engine, monkeypatch, copy=False, blocker=True):
    monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 40)
    stored = [row(n) for n in range(1, 65)]
    engine.ingest(stored)
    ids = [int(r['store_id']) for r in rows(engine)]
    chunk = [*stored[19:24], *stored[49:52], dict(stored[0])]
    claims = {id(chunk[8]): [ids[0]]}
    if copy:
        chunk.append(dict(stored[4]))
        claims[id(chunk[9])] = [ids[4]]
    outside = [dict(stored[29])] if blocker else []
    view = [*chunk, *outside, *stored[-3:]]
    mapping = {id(m): ids[i] for m, i in zip(chunk[:8], [*range(19, 24), *range(49, 52)])}
    return stored, ids, chunk, claims, mapping, view


def walk(engine, case, **kwargs):
    _stored, _ids, chunk, claims, mapping, view = case
    return engine._store_complete_input(chunk, claims, mapping, view, chunk, 0, [],
                                        count_messages_tokens(chunk), allow_retained_escape=True, **kwargs)


def covered(out, mapping):
    return [sid for m, ids in out or () for sid in [*ids, *([mapping[id(m)]] if id(m) in mapping else [])]]


def assert_leaf(engine, case, out):
    _stored, ids, chunk, claims, mapping, _view = case
    assert out, 'the escape-frame hole still returns no leaf'
    sources = covered(out, mapping)
    assert len(sources) == len(set(sources))
    assert min(sources) == ids[0]
    exclusions = engine._store_complete_excluded
    ignored = [int(r['store_id']) for r in rows(engine)
               if engine._matches_ignore_message_patterns(r, stored_row=True)]
    counts = Counter([*sources, *exclusions, *ignored])
    assert all(counts[s] == 1 for s in ids if ids[0] <= s <= max(sources)), counts
    escapes = {s for s, *_ in engine._store_complete_escaped}
    assert set(sources) & {s for claim in claims.values() for s in claim} <= escapes
    assert engine._store_complete_cut
    assert not any(any(m is c for c in chunk) for m, _ in out)
    assert count_messages_tokens([m for m, _ in out]) <= max(engine._config.leaf_chunk_tokens,
                                                           count_messages_tokens(chunk))
    engine._compress_run.store_ids_by_message_id = mapping
    summary, source_map, anchors = engine._leaf_summary_input(out, chunk)
    consumption = engine._leaf_consumption(chunk, summary, source_map, out, anchors, set())
    assert consumption[1:3] == ([], [])
    assert consumption[3] == consumption[4] == sorted(sources)


def test_n1_field_chain_q(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='hermes_lcm')
    engine = build(tmp_path)
    try:
        case = field(engine, monkeypatch)
        calls = spy(monkeypatch)
        out = walk(engine, case)
        # Budget first, then the shown blocker, then escape: the field's chain Q.
        assert [c['flags'].get('escape_retained', False) for c in calls[:3]] == [False, False, True]
        assert calls[1]['budget'] > calls[0]['budget']
        assert_leaf(engine, case, out)
        assert len(calls) <= 7
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        assert retries and all(c['budget'] == calls[0]['budget'] for c in retries)
        assert 'prefix_retry=True' in caplog.text
        assert f'retry_budget={calls[0]["budget"]}' in caplog.text
    finally:
        engine.shutdown()


def test_n2_copy_claim_escapes(tmp_path, monkeypatch):
    engine = build(tmp_path)
    try:
        case = field(engine, monkeypatch, copy=True)
        calls = spy(monkeypatch)
        out = walk(engine, case)
        assert_leaf(engine, case, out)
        assert (case[1][4], 'user', 'shown') in engine._store_complete_escaped
        assert len(calls) <= 7
    finally:
        engine.shutdown()


def test_n3_top_blocked_before_emission(tmp_path, monkeypatch):
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 2)
    engine = build(tmp_path)
    try:
        stored = [row(i) for i in range(1, 8)]
        engine.ingest(stored)
        ids = [int(r['store_id']) for r in rows(engine)]
        chunk = [stored[3], dict(stored[0])]
        mapping = {id(stored[3]): ids[3], id(stored[1]): ids[1]}
        case = (stored, ids, chunk, {id(chunk[1]): [ids[0]]}, mapping, [*chunk, stored[1]])
        calls = spy(monkeypatch)
        out = walk(engine, case)
        assert calls[1]['flags'].get('escape_retained') is True
        assert_leaf(engine, case, out)
        assert len(calls) <= 7
        assert any(c['flags'].get('prefix_retry') is False for c in calls)
    finally:
        engine.shutdown()


def test_n4_direct_lift_retry_budget(tmp_path, monkeypatch):
    engine = build(tmp_path)
    try:
        case = field(engine, monkeypatch, blocker=False)
        calls = spy(monkeypatch)
        out = walk(engine, case)
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        assert retries, 'fixture did not reach the direct lifted retry'
        assert retries[0]['budget'] == calls[0]['budget'], 'retry inherited the lifted budget'
        assert_leaf(engine, case, out)
        assert len(calls) <= 7
    finally:
        engine.shutdown()


def node_sources(engine):
    found = engine._store.connection.execute(
        "SELECT node_id, source_ids FROM summary_nodes WHERE source_type='messages' ORDER BY node_id").fetchall()
    result = {r[0]: json.loads(r[1]) for r in found}
    ids = [sid for sources in result.values() for sid in sources]
    assert len(ids) == len(set(ids)), 'a source id appears in two nodes'
    return result


def test_n5_restart_control(tmp_path, monkeypatch, offline):
    paths = [tmp_path / 'fixed', tmp_path / 'control']
    for p in paths:
        p.mkdir()
    engines = [build(p) for p in paths]
    table = []
    try:
        case = field(engines[0], monkeypatch)
        stored, _ids, chunk, _claims, _mapping, view = case
        # A later mapped row stops real chunk extension before the shown
        # blocker. The blocker stays eligible once the prefix has drained.
        view.insert(len(chunk), stored[59])
        view[-3:-3] = stored[52:59]
        engines[1].ingest(stored)
        views = [list(view), list(view)]
        previous, prior_nodes = 0, {}
        calls = spy(monkeypatch)
        for turn in range(6):
            for index, engine in enumerate(engines):
                engine._retained_escape_enabled = index == 0
                engine.ingest(views[index])
                engine._clear_persisted_no_progress_hold()
                engine._no_progress_hold = None
            start, prompt_start = len(calls), len(offline)
            views[0] = engines[0].compress(views[0], current_tokens=10_000, force=True)
            frontier = engines[0]._store_complete_frontier()
            assert frontier > previous, f'pass {turn + 1}: frontier did not rise'
            sources = node_sources(engines[0])
            assert all(sources[k] == v for k, v in prior_nodes.items()), 'an older node changed'
            fixed_calls = calls[start:]
            prompt_characters = offline[prompt_start:]
            views[1] = engines[1].compress(views[1], current_tokens=10_000, force=True)
            assert digest(engines[0]) == digest(engines[1]), 'N5 stored-row digest differs from control'
            node_sources(engines[1])
            new = {k: v for k, v in sources.items() if k not in prior_nodes}
            retry = any(c['flags'].get('prefix_retry') is False for c in fixed_calls)
            tokens = max((c['tokens'] for c in fixed_calls if c.get('out')), default=0)
            lifted = any(c['budget'] > fixed_calls[0]['budget'] for c in fixed_calls)
            source_tokens = [engines[0]._dag.get_node(k).source_token_count for k in new]
            table.append({'pass': turn + 1, 'frontier': frontier, 'leaf_ids': list(new.values()),
                          'leaf_source_tokens': source_tokens, 'leaf_input_tokens': tokens,
                          'summariser_input_characters': prompt_characters,
                          'frame': 'retry' if retry else 'lift' if lifted else 'ordinary',
                          'frame_budgets': [c['budget'] for c in fixed_calls]})
            print('N5_PASS ' + json.dumps(table[-1], sort_keys=True))
            if retry:
                assert tokens <= fixed_calls[0]['budget']
            previous, prior_nodes = frontier, sources
            for engine in engines:
                engine.shutdown()
            engines = [build(p) for p in paths]
        assert previous > case[1][51], 'passes did not cross the copies behind the holder'
    finally:
        for engine in engines:
            engine.shutdown()


def clear_holds(engine):
    engine._clear_persisted_no_progress_hold()
    engine._no_progress_hold = None


def composite_stop(engine, monkeypatch):
    # The #2032 composite is R2-witnessed by the real identity path, never
    # presented to the retry as a falsely matching single-row occurrence.
    stored = [row(i) for i in range(1, 5)]
    call = {'role': 'assistant', 'content': '', 'timestamp': 5.0, 'tool_calls': [
        {'id': 'call', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}
    result = {'role': 'tool', 'tool_call_id': 'call', 'content': 'synthetic result', 'timestamp': 6.0}
    engine.ingest([*stored, call, result])
    composite = dict(stored[0], content=stored[0]['content'] + '\n\n' + stored[1]['content'])
    chunk = [stored[3], call, composite]
    view = [*chunk, stored[2], result]
    engine._ingest_cursor = len(view)
    monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 4)
    return view, chunk


def test_p1_composite_guard_stays_retained_and_paced(tmp_path, monkeypatch):
    engine = build(tmp_path, leaf_chunk_tokens=150, fresh_tail_count=2)
    try:
        view, _chunk = composite_stop(engine, monkeypatch)
        before, signatures = digest(engine), []
        calls = spy(monkeypatch)
        for _ in range(2):
            clear_holds(engine)
            engine.compress(view, current_tokens=10_000)
            assert engine._last_compress_leaves[1] == 0
            assert engine._last_compression_noop_reason.startswith(
                'leaf would end before an unresolved retained occurrence')
            signatures.append((engine._store_complete_frontier(), engine._store_complete_blocked[0],
                               engine._store_complete_hole))
            assert digest(engine) == before
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        if any('retry_allowed' in c['flags'] for c in calls):
            assert retries and all(c['out'] == [1, 2] for c in retries)
            assert not engine._store_complete_escaped
        assert signatures[0] == signatures[1] == (0, 3, 1)
        assert engine._retained_stop_hold_active()
        witness = engine._store.get_message_relations([1], 'composite')
        assert {r['related_store_id'] for r in witness} == {1, 2}
    finally:
        engine.shutdown()


@pytest.mark.parametrize('escape', [False, True], ids=['inspect_allow_false', 'direct_escape'])
def test_p3_permission(tmp_path, monkeypatch, escape):
    engine = build(tmp_path)
    try:
        case = field(engine, monkeypatch)
        _stored, _ids, chunk, claims, mapping, view = case
        calls = spy(monkeypatch)
        out = engine._store_complete_input(chunk, claims, mapping, view, chunk, 0, [],
                                            count_messages_tokens(chunk), escape_retained=escape,
                                            allow_retained_escape=False)
        assert out == []
        assert engine._store_complete_escaped == []
        assert not any(c['flags'].get('prefix_retry') is False for c in calls)
        assert all(c['flags'].get('escape_retained', False) == escape for c in calls)
    finally:
        engine.shutdown()


@pytest.mark.parametrize('ctx', [32_000, 0])
@pytest.mark.parametrize('shape', ['field', 'copy', 'blocked_first', 'direct'])
def test_p4_recursion_bound(tmp_path, monkeypatch, ctx, shape):
    engine = build(tmp_path, ctx=ctx)
    try:
        if shape == 'blocked_first':
            stored = [row(i) for i in range(1, 8)]
            engine.ingest(stored)
            chunk = [stored[3], dict(stored[0])]
            case = (stored, list(range(1, 8)), chunk, {id(chunk[1]): [1]},
                    {id(stored[3]): 4, id(stored[1]): 2}, [*chunk, stored[1]])
            monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 2)
        else:
            case = field(engine, monkeypatch, copy=shape == 'copy', blocker=shape != 'direct')
        calls = spy(monkeypatch)
        walk(engine, case)
        # Top, lift/escape, escape/lift, retry, retry-escape, with one possible
        # B-GROUP-1 and one copy-proof re-entry each. No recursion-limit changes.
        assert len(calls) <= 7
        assert sum(c['flags'].get('scale', 1) > 1 for c in calls) <= 1
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        assert len(retries) <= 2
    finally:
        engine.shutdown()


def test_p6_kill_switch(tmp_path, monkeypatch):
    engine = build(tmp_path)
    engine._retained_escape_enabled = False
    try:
        case = field(engine, monkeypatch)
        before = digest(engine)
        calls = spy(monkeypatch)
        assert walk(engine, case) == []
        assert len(calls) == 1
        assert calls[0]['budget'] == 378
        assert engine._store_complete_empty_reason == 'leaf input cannot form a contiguous stored prefix'
        assert engine._store_complete_escaped == []
        assert digest(engine) == before
    finally:
        engine.shutdown()


def test_p7_prepared_consumption_restores_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_lcm.compaction import _CompressionState, _LeafState

    engine = build(tmp_path)
    try:
        case = field(engine, monkeypatch)
        _stored, _ids, chunk, _claims, mapping, view = case
        engine._compress_run.store_ids_by_message_id = mapping
        fields = ('excluded', 'cut', 'blocked', 'empty_reason', 'escaped', 'order', 'hole')
        sentinels = ([999], True, (999, 'user', 'shown', 999), 'sentinel',
                     [(999, 'user', 'shown')], {999: 0}, 999)
        for name, value in zip(fields, sentinels):
            setattr(engine, '_store_complete_' + name, value)
        calls = spy(monkeypatch)
        consumption = engine._prepared_chunk_consumption(chunk, view, set(), set())
        assert tuple(getattr(engine, '_store_complete_' + name) for name in fields) == sentinels
        if not any(c['flags'].get('prefix_retry') is False for c in calls):
            # This is a new reachable path after part 2; on the base it refuses.
            assert consumption is None
            return
        assert consumption == (list(range(1, 10)),) * 2
        state = _CompressionState(working_messages=view)
        leaf = _LeafState(to_compact=chunk)
        leaf.prepared = SimpleNamespace(covered_store_ids=tuple(consumption[0]))
        rejected = []
        monkeypatch.setattr(engine, '_reject_prepared_chain', lambda *_a: rejected.append(True))
        action, result = engine._prepare_leaf_input(state, leaf, set(), lambda *_a: 0)
        assert (action, result) == ('next', None)
        assert not rejected
        assert tuple(engine._leaf_consumption(chunk, leaf.summary_input_chunk, leaf.sources,
                                              leaf.anchored_input, leaf.anchor_claims, set())[3:]) == consumption
    finally:
        engine.shutdown()


def test_p2_copy_exclusion_cut_claim_guard(tmp_path, monkeypatch):
    from hermes_lcm.dag import SummaryNode

    engine = build(tmp_path)
    try:
        # E1: an older node owns an identical assistant call. A composite-like
        # holder claims its second occurrence but does not show its own form.
        call = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'copy', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}
        stored = [dict(call, timestamp=1.0), dict(call, timestamp=2.0),
                  *[row(i) for i in range(3, 65)]]
        engine.ingest(stored)
        engine._dag.add_node(SummaryNode(session_id='S', summary='synthetic prior leaf',
                                        source_ids=[1], token_count=5, created_at=1.0))
        engine._last_compacted_store_id = 1
        monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 40)
        holder = {'role': 'user', 'content': 'synthetic composite carrier', 'timestamp': 2.0}
        chunk = [*stored[19:24], *stored[49:52], holder]
        mapping = {id(m): i for m, i in zip(chunk[:8], [*range(20, 25), *range(50, 53)])}
        view = [*chunk, dict(stored[29])]
        calls = spy(monkeypatch)
        out = engine._store_complete_input(chunk, {id(holder): [2]}, mapping, view, chunk, 1, [],
                                            count_messages_tokens(chunk), allow_retained_escape=True)
        assert out == []
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        if not retries:  # base never enters this escape-frame branch
            assert len(calls) == 3
            return
        assert any(c['copy_proofs'].get(2) == 1 and 2 in c['excluded'] for c in retries)
        assert engine._store_complete_copy_exclusion_proofs == {}
        assert engine._store_complete_excluded == []
        assert engine._store_complete_escaped == []
        assert not engine._store_complete_cut
    finally:
        engine.shutdown()


def memo_overflow(engine):
    stored = [row(i) for i in range(1, 301)]
    engine.ingest(stored)
    # Exercise the actual memo writer in batches, with every host copy retained.
    copies = [dict(m) for m in stored]
    for start in range(0, 300, 50):
        engine._record_retained_escape([(sid, 'user', 'shown')
                                        for sid in range(start + 1, start + 51)])
    return copies


def test_p5_memo_keeps_highest_256_ids(tmp_path):
    engine = build(tmp_path)
    try:
        copies = memo_overflow(engine)
        assert len(copies) == 300
        assert set(engine._retained_escape_memo().values()) == set(range(45, 301))
    finally:
        engine.shutdown()


def test_p5_evicted_copy_refuses_without_lineage(tmp_path):
    from hermes_lcm.compaction import _CompressionState, _LeafState
    from hermes_lcm.dag import SummaryNode

    engine = build(tmp_path)
    try:
        copies = memo_overflow(engine)
        # All originals already have a single owner; no hidden backlog remains.
        engine._dag.add_node(SummaryNode(session_id='S', summary='synthetic covered originals',
                                        source_ids=list(range(1, 301)), token_count=5, created_at=1.0))
        engine._last_compacted_store_id = 300
        before, before_nodes = digest(engine), node_sources(engine)
        assert not engine._identity_anchor_covered_view(copies[0], {}, 300)
        assert engine._identity_anchor_covered_view(copies[-1], {}, 300)
        assert not engine._store_complete_backlog([copies[0]], 0)
        # Direct preparation is the addendum's allowed refusal probe. No
        # hand-made chunk is sent to publication or patched selection seam.
        for _ in range(3):
            state = _CompressionState(working_messages=[copies[0]])
            leaf = _LeafState(to_compact=[copies[0]])
            action, result = engine._prepare_leaf_input(state, leaf, set(), lambda *_a: 0)
            assert (action, result) == ('break', None)
            assert state.noop_reason == 'selected leaf chunk lacks raw store lineage'
            assert not engine._retained_stop_hold_active()
            assert digest(engine) == before
            assert node_sources(engine) == before_nodes
        print('P5: memo=256; recorded escapes=300; direct refusals=3; retained pacing=False')
    finally:
        engine.shutdown()


def test_p8_prefix_rehole_retained_and_paced(tmp_path, monkeypatch):
    from hermes_lcm.identity_anchor import _composite_relation

    engine = build(tmp_path, fresh_tail_count=4)
    try:
        stored, _ids, chunk, _claims, _mapping, view = field(engine, monkeypatch)
        composite = dict(stored[0], content=stored[0]['content'] + '\n\n' + stored[19]['content'])
        engine._store.add_message_relations([_composite_relation([rows(engine)[0], rows(engine)[19]], 1.0)])
        chunk[0] = composite
        view[0] = composite
        chunk[8] = dict(stored[1])
        view[8] = chunk[8]
        engine._ingest_cursor = len(view)
        before, signatures = digest(engine), []
        calls = spy(monkeypatch)
        for _ in range(2):
            clear_holds(engine)
            engine.compress(view, current_tokens=10_000)
            assert engine._last_compress_leaves[1] == 0
            assert engine._last_compression_noop_reason.startswith(
                'leaf would end before an unresolved retained occurrence')
            signatures.append((engine._store_complete_frontier(), engine._store_complete_blocked[0],
                               engine._store_complete_hole))
            assert digest(engine) == before
        retries = [c for c in calls if c['flags'].get('prefix_retry') is False]
        if any('retry_allowed' in c['flags'] for c in calls):
            assert retries and all(c['out'] == [] for c in retries)
            assert all(c['budget'] == calls[0]['budget'] for c in retries)
            assert any(c['claim_ids'] == [[1, 20]] for c in retries)
        print('P8: signatures=' + str(signatures) + '; retry frames=' + str(len(retries)))
        assert signatures[0] == signatures[1]
        assert engine._retained_stop_hold_active()
    finally:
        engine.shutdown()



def test_r3_scan_extension_forwards_prefix_retry(tmp_path, monkeypatch):
    engine = build(tmp_path)
    try:
        call = {'role': 'assistant', 'content': '', 'timestamp': 2.0, 'tool_calls': [
            {'id': name, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
            for name in ('first', 'second')]}
        stored = [row(1), call, *[
            {'role': 'tool', 'tool_call_id': name, 'content': 'synthetic result', 'timestamp': float(i)}
            for i, name in enumerate(('first', 'second'), 3)], *[row(i) for i in range(5, 65)]]
        engine.ingest(stored)
        monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 3)
        holder = dict(stored[0])
        chunk = [stored[49], holder]
        mapping, claims = {id(chunk[0]): 50}, {id(holder): [1]}
        budget = count_messages_tokens(chunk)
        calls = []
        original = engine._store_complete_input

        def counted(*args, **kwargs):
            calls.append((args[7], kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, '_store_complete_input', counted)
        # Enter a lifted escape frame with the original caller's retry permission
        # and budget. The first tool group straddles both sides of the scan cap.
        out = engine._store_complete_input(chunk, claims, mapping, chunk, chunk, 0, [], 400_000,
                                          escape_retained=True, allow_retained_escape=False,
                                          lift_allowed=False, retry_allowed=True, retry_budget=budget)
        extended = [(b, kw) for b, kw in calls if kw.get('scale') == store_complete._SCAN_EXTEND]
        assert extended, 'fixture did not reach the scan-extension recursion'
        frame_budget, flags = extended[0]
        assert frame_budget > budget
        assert flags.get('escape_retained') is True
        assert flags.get('retry_allowed') is True, 'scan extension dropped the inherited retry permission'
        assert flags.get('retry_budget') == budget, 'scan extension reset the inherited retry budget'
        retry_budget, retry_flags = calls[calls.index(extended[0]) + 1]
        assert retry_flags.get('prefix_retry') is False, 'the extended frame did not retry its retained hole'
        assert retry_flags.get('retry_allowed') is False
        assert retry_budget == min(frame_budget, budget)
        assert engine._store_complete_hole == 1
        assert covered(out, mapping) == [1]
        assert count_messages_tokens([m for m, _ in out]) <= budget
        assert engine._store_complete_cut
    finally:
        engine.shutdown()


def test_r3_copy_rerun_forwards_prefix_retry(tmp_path, monkeypatch):
    from hermes_lcm.dag import SummaryNode

    engine = build(tmp_path)
    try:
        call = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'copy', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}
        stored = [dict(call, timestamp=1.0), *[row(i) for i in range(2, 21)],
                  dict(call, timestamp=21.0), *[row(i) for i in range(22, 65)]]
        engine.ingest(stored)
        engine._dag.add_node(SummaryNode(session_id='S', summary='synthetic prior leaf',
                                        source_ids=[1], token_count=5, created_at=1.0))
        engine._last_compacted_store_id = 1
        monkeypatch.setattr(store_complete, '_SCAN_LIMIT', 40)
        holder = dict(stored[1])
        chunk = [stored[49], holder]
        mapping, claims = {id(chunk[0]): 50}, {id(holder): [2]}
        budget = count_messages_tokens(chunk)
        calls = []
        original = engine._store_complete_input

        def counted(*args, **kwargs):
            calls.append((args[7], kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, '_store_complete_input', counted)
        # The small first frame stops before the E1 copy at 21; the lifted walk
        # reaches it, then the retained claim at 2 empties the whole leaf.
        out = engine._store_complete_input(chunk, claims, mapping, chunk, chunk, 1, [], budget,
                                          allow_retained_escape=True)
        reruns = [(b, kw) for b, kw in calls if kw.get('copy_lookup') == ({}, set())]
        assert reruns, 'fixture did not reach the copy-exclusion rerun'
        frame_budget, flags = reruns[0]
        assert frame_budget > budget
        assert flags.get('retry_allowed') is True, 'copy rerun dropped the inherited retry permission'
        assert flags.get('retry_budget') == budget, 'copy rerun reset the inherited retry budget'
        retry_budget, retry_flags = calls[calls.index(reruns[0]) + 1]
        assert retry_flags.get('prefix_retry') is False, 'the copy-exclusion rerun did not retry its retained hole'
        assert retry_flags.get('retry_allowed') is False
        assert retry_budget == min(frame_budget, budget)
        assert engine._store_complete_hole == 2
        assert covered(out, mapping) == [2, 3]
        assert count_messages_tokens([m for m, _ in out]) <= budget
        assert engine._store_complete_cut
    finally:
        engine.shutdown()
