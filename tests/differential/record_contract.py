"""Independent #2707 record eligibility, occurrence projection and recall contract.

Writer faults count and raise only on calls carrying at least one record. Empty
calls pass through uncounted. Once raises on the first counted call after install;
persistent raises on every counted call while installed.

Only observe(engine), the tool/locator adapter, may be adapted by a candidate PR.
judge(observations, entries) contains frozen semantics and must not be changed.
Observation rows have role, content and kind: canonical has store_id; record has
owner_store_id and ordinal. Search hits also have reference={tool, args} and an
expanded normalized row. Observations include pages, hits, reload_witness and
summary_sources; record references are compared with summary source locators.
"""
from collections import Counter
import json
import sys

from .session import freeze


def declaration_stages(trace):
    """Use the same baseline stage evidence for generation and live checks."""
    return [{k: v for k, v in stage.items() if k != 'preserved'} for stage in trace.ingests]


def writer_fault_stages(case):
    """Frozen planned raising stages, independent of candidate/audit output."""
    from .test_paired_merged_user_turns import RANDOM_SEEDS, _plan
    if case in RANDOM_SEEDS:
        plan = _plan(case)
        return {turn + 2: step['fault'] for turn, step in enumerate(plan['steps'][:plan['ingests'] - 1])
                if not (turn == 0 and plan['cache'] != 'cold')
                and step['fault'] in ('preservation-once', 'persistent')}
    return {2707101: {3: 'preservation-once'}, 2707102: {2: 'preservation-once'},
            2707103: {2: 'persistent', 3: 'persistent'}, 2707202: {3: 'preservation-once'},
            2707203: {3: 'preservation-once'}}.get(case, {})


def writer_fault(engine, kind, patch, calls):
    """Install the counted-call rule on the declared writer seam."""
    from .session import audit_fault
    present = hasattr(engine, '_store_queued_user_rows')
    audit = audit_fault(engine, kind, present)
    if audit.stubbed or not present:
        return audit.stubbed
    original = engine._store_queued_user_rows
    plan = writer_fault_stages(engine._paired_trace.case)
    assert plan.get(audit.stage) == kind, 'writer fault differs from frozen plan'
    def wrapped(rows, *args, **kwargs):
        if rows:
            calls.append(1)
            audit.called += 1
            if kind == 'persistent' or len(calls) == 1:
                audit.raised += 1
                raise ValueError('injected writer failure')
        return original(rows, *args, **kwargs)
    patch.setattr(engine, '_store_queued_user_rows', wrapped)
    return False


def declarations(case, stages, baseline_sha):
    result, retired = [], set()
    faults = writer_fault_stages(case)
    for stage, snapshot in enumerate(stages, 1):
        assert not snapshot['protection']['sensitive_patterns_enabled'], 'protection needs independent redaction model'
        for capture in snapshot['captures']:
            view = capture['view']
            if view in retired:
                continue
            ids = capture['ids']
            before = snapshot['committed_before']
            # Current fixtures never commit a middle predecessor at the owner handoff.
            if capture['kind'] != 'proven' or any(i in before for i in ids[:-1]):
                retired.add(view)
                continue
            if not (capture['head_present'] and not capture['newest_present'] and capture['overwritten']
                    and not capture['prefix'] and '\n\n'.join(r['content'] for r in capture['rows']) != capture['clean']):
                continue
            owner = snapshot['committed'].get(ids[0])
            if owner is None:
                continue
            retired.add(view)
            for ordinal, row in enumerate(capture['rows'][:-1]):
                result.append(dict(issue=2707, baseline_sha=baseline_sha, case=case, stage=stage,
                    due_stage=next((i for i in range(stage, len(stages) + 1) if i not in faults), None), capture_view=view,
                    predecessor_ordinal=ordinal, owner_occurrence=ids[0], owner_store_id=owner,
                    captured_uid=row.get('message_uid'), expected=dict(role='user', content=row['content'],
                    observed_at=row['timestamp'], observed_at_source='host_message_timestamp'), multiplicity=1))
    return result


def expected_record(entry):
    return dict(entry['expected'], owner_store_id=entry['owner_store_id'], ordinal=entry['predecessor_ordinal'],
                session_id='parent', conversation_id='conv', captured_uid=entry['captured_uid'], superseded_by=None)


def active(entries, stage):
    return [entry for entry in entries if entry['due_stage'] is not None and entry['due_stage'] <= stage]


def projection(snapshot):
    """Stable owner/ordinal ordering; each physical record supplies one segment."""
    records = [dict(row) for row in snapshot['records'] if dict(row)['superseded_by'] is None]
    return [freeze(item) for row in snapshot['canonical'] for item in (
        *sorted((r for r in records if r['owner_store_id'] == dict(row)['store_id']), key=lambda r: r['ordinal']),
        dict(row))]


def check_stage(snapshot, baseline, entries):
    assert snapshot['canonical'] == baseline['canonical'], 'canonical row changed'
    # Supersession needs a same-turn reload witness, not equal bytes from another turn.
    witness = baseline.get('reload_witness', {})
    assert len(set(witness.values())) == len(witness), 'reload consumed by multiple owners'
    records = [dict(row) for row in snapshot['records']]
    for row in records:
        if row['superseded_by'] is not None:
            assert witness.get(row['owner_store_id']) == row['superseded_by'], 'superseded without occurrence witness'
            row['superseded_by'] = None
    assert Counter(map(freeze, records)) == Counter(freeze(expected_record(e)) for e in entries), 'declared record multiset'
    assert len(projection(snapshot)) == len(snapshot['canonical']) + sum(
        dict(row)['superseded_by'] is None for row in snapshot['records']), 'record owner absent'


def observe(engine):
    """Tool calls and locator translation only; candidate PRs may adapt this adapter."""
    tools = getattr(engine, '_contract_tools', sys.modules[engine.__module__.rsplit('.', 1)[0] + '.tools'])
    def call(name, args):
        reply = json.loads(getattr(tools, name)(args, engine=engine))
        assert 'error' not in reply, f'{name}: {reply.get("error")}'
        return reply
    def normalized(row):
        return dict(row, kind='record' if 'owner_store_id' in row else 'canonical')
    pages, cursor, seen = [], None, set()
    while True:
        args = dict(session_id='parent', limit=1, include_exact_ref=True)
        if isinstance(cursor, dict):
            args.update(cursor)
        elif cursor is not None:
            assert isinstance(cursor, int), 'unnamed record cursor'
            args['after_store_id'] = cursor
        page = call('lcm_load_session', args)
        pages.extend(normalized(row) for row in page.get('messages', []))
        cursor = page.get('next_cursor')
        if cursor is None:
            break
        assert freeze(cursor) not in seen, 'pagination stalled'
        seen.add(freeze(cursor))
    hits = []
    texts = {row['content'] for capture in engine._paired_trace.record_captures for row in capture['rows'][:-1]}
    for text in sorted(texts):
        found = call('lcm_grep', dict(query='"' + text + '"', mode='full_text', limit=100))
        for row in found.get('results', []):
            reference = row.get('reference') or dict(tool='lcm_expand', args=dict(store_id=row['store_id']))
            expanded = normalized(call(reference['tool'], reference['args']))
            hits.append(dict(normalized(dict(row, content=row.get('content', expanded.get('content')))),
                             reference=reference, expanded=expanded))
    sources = [sid for (raw,) in engine._dag._conn.execute('SELECT source_ids FROM summary_nodes')
               for sid in json.loads(raw)]
    return dict(pages=pages, hits=hits, summary_sources=sources)


def judge(observations, entries):
    """Frozen semantic contract; candidate PRs must not change this function."""
    pages, failures = observations['pages'], []
    witness = observations.get('reload_witness', {})
    for entry in entries:
        expected = expected_record(entry)
        key = (expected['owner_store_id'], expected['ordinal'])
        def matches(row, span=False):
            if span and witness.get(key[0]) is not None and row.get('store_id') == witness[key[0]]:
                parts = row.get('content', '').split('\n\n')
                return row.get('role') == 'user' and len(parts) > key[1] and parts[key[1]] == expected['content']
            return row['kind'] == 'record' and (row.get('owner_store_id'), row.get('ordinal')) == key \
                and row.get('role') == 'user' and row.get('content') == expected['content']
        found = [row for row in observations['hits'] if matches(row, span=True)]
        if sum(matches(row, span=True) for row in pages) != 1 or not any(matches(r['expanded']) for r in found):
            failures.append((entry['capture_view'], entry['predecessor_ordinal'], expected['content']))
    ordered = [(r['owner_store_id'], r['ordinal']) for r in pages if r['kind'] == 'record']
    assert ordered == sorted(ordered), 'record listing owner order'
    for index, row in enumerate(pages):
        if row['kind'] == 'record':
            following = next((r for r in pages[index + 1:] if r['kind'] == 'canonical'), {})
            preceding = next((r for r in reversed(pages[:index]) if r['kind'] == 'canonical'), {})
            assert following.get('store_id') == row['owner_store_id'] \
                and preceding.get('store_id', -1) < row['owner_store_id'], 'record owner placement'
    locators = [hit['reference']['args'] for hit in observations['hits'] if hit['kind'] == 'record']
    assert not any(freeze(source) in {freeze(ref), *(freeze(v) for v in ref.values())}
                   for source in observations['summary_sources'] for ref in locators), 'record summary source'
    return failures


# Explicit accepted (case, kind) exemptions to the main-raised/candidate-inert rule.
FAULT_EXEMPTIONS = set()


FAULT_CONTROLS = {2707102: {'preservation-once'}, 2707202: {'preservation-once'},
                  2707203: {'preservation-once'}, 2707405: {'persistent'}, 2707407: {'preservation-once'}}


def mandatory_floor(result, cases):
    assert result['evaluated'], 'floor skipped'
    for case, kinds in FAULT_CONTROLS.items():
        for kind in kinds:
            counts = cases.get(case, {}).get('candidate', {}).get(kind, {})
            assert counts.get('requested') and not counts.get('inert') and counts.get('called') and counts.get('raised'), \
                f'{case}: declared fault {kind} never fired'
