"""Independent #2707 record eligibility, occurrence projection and recall contract.

Writer faults are ingest-scoped and never inspect arguments: every writer entry
in the first faulted ingest raises for once; every ingest raises for persistent.
Entries are counted per ingest regardless of batching, retries or call shape.

Frozen except the accepted round-3 refusal and canonical-expansion amendments:
check_stage, active, expected_record, projection,
mandatory_floor, judge(), overlap(), FLOOR_USER_ROWS=64, FLOOR_BYTES=1048576,
the records-v1 block in test_paired_merged_user_turns.py,
and the snapshot shape (session.py: canonical_rows, queued_rows, Trace.stored,
RECORD_COLUMNS, zero-based ordinal and host_message_timestamp provenance).
The frozen listing_projection input uses that stage's projection in records-v1
and no records in main; judge requires exact record content, multiplicity and
positions. check_stage independently validates the stored projection's records.
Only observe()'s tool-call adapters may change, keeping the same observations.
Observation rows have role, content and kind: canonical has store_id; record has
owner_store_id and ordinal. Search hits also have reference={tool, args} and an
expanded normalized row. Observations include pages, hits, reload_witness and
summary_sources, baseline_summary_sources and baseline_canonical_rows for equality,
baseline_canonical (the full canonical snapshot, including session/conversation),
withheld (owner_store_id, ordinal, state, reference, expanded; duplicates kept),
withheld_counts (each footer page's per-state counts), and listing_projection.
Witnesses take precedence. overlap() uses the owner's canonical session, stripped
nonempty record segments, later user rows and the longest prefix inside both
floors. judge applies Revision 3d's outcome table; projection itself is unchanged.
The witness is the captured texts joined oldest first with "\n\n", with no
per-text strip and empty texts kept. A turn with a member whose text differs
from its stripped form, a whitespace-only one included, has no witness
(Revision 3f).
"""
from collections import Counter
import json

from .session import freeze


FLOOR_USER_ROWS = 64
FLOOR_BYTES = 1048576


def reload_witness(trace):
    """3h: the floor host supplies no positive absorbed-UID evidence."""
    return {}


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
    """Install the ingest-scoped rule on the declared writer seam."""
    from .session import audit_fault
    present = hasattr(engine, '_store_queued_user_rows')
    audit = audit_fault(engine, kind, present)
    if audit.stubbed or not present:
        return audit.stubbed
    original = engine._store_queued_user_rows
    plan = writer_fault_stages(engine._paired_trace.case)
    assert plan.get(audit.stage) == kind, 'writer fault differs from frozen plan'
    def wrapped(*args, **kwargs):
        stage = len(engine._paired_trace.ingests) + 1
        if kind == 'persistent' or stage == audit.stage:
            calls.append(stage)
            audit.called += 1
            audit.raised += 1
            raise ValueError('injected writer failure')
        return original(*args, **kwargs)
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


def earlier_spans(record, canonical):
    """Exact substring evidence, independent of the product's overlap predicate."""
    rows = [dict(row) for row in canonical]
    owner = next((row for row in rows if row['store_id'] == record['owner_store_id']), None)
    return [row for row in rows if owner is not None and row['store_id'] < owner['store_id']
            and row['session_id'] == owner['session_id'] and row['role'] == 'user'
            and record['content'] in row['content']]


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
    # The judge separately requires counted refusal and retrieval for these spans.
    stored_keys = {(r['owner_store_id'], r['ordinal']) for r in records}
    required = [expected_record(e) for e in entries if
                (e['owner_store_id'], e['predecessor_ordinal']) in stored_keys
                or not earlier_spans(expected_record(e), snapshot['canonical'])]
    assert Counter(map(freeze, records)) == Counter(map(freeze, required)), 'declared record multiset'
    assert len(projection(snapshot)) == len(snapshot['canonical']) + sum(
        dict(row)['superseded_by'] is None for row in snapshot['records']), 'record owner absent'


def overlap(record, canonical):
    """Return (overlap inside F, overlap anywhere, rows beyond F), independently."""
    rows = [dict(row) for row in canonical]
    owner = next(row for row in rows if row['store_id'] == record['owner_store_id'])
    text = record['content'].strip()
    inside, anywhere, beyond, used = False, False, False, 0
    for index, row in enumerate(r for r in rows if r['store_id'] > owner['store_id']
                                and r['session_id'] == owner['session_id'] and r['role'] == 'user'):
        used += len(row['content'].encode('utf-8'))
        beyond |= index >= FLOOR_USER_ROWS or used > FLOOR_BYTES
        parts, needle = row['content'].split('\n\n'), text.split('\n\n')
        hit = bool(text) and any(parts[i:i + len(needle)] == needle for i in range(len(parts)))
        anywhere |= hit
        inside |= hit and not beyond
    return inside, anywhere, beyond


def observe(engine, tools):
    """Tool calls and locator translation only; candidate PRs may adapt this adapter."""
    def call(name, args):
        reply = json.loads(getattr(tools, name)(args, engine=engine))
        assert 'error' not in reply, f'{name}: {reply.get("error")}'
        return reply
    def normalized(row):
        return dict(row, kind='record' if 'owner_store_id' in row else 'canonical')
    def expanded(row):
        reference = row['reference']
        return dict(normalized(row), expanded=normalized(call(reference['tool'], reference['args'])))
    pages, withheld, counts = [], [], []
    cursor, footer_cursor, seen, footer_seen = {}, {}, set(), set()
    while True:
        args = dict(session_id='parent', limit=1, include_exact_ref=True)
        if isinstance(cursor, dict):
            args.update(cursor)
        elif cursor is not None:
            assert isinstance(cursor, int), 'unnamed record cursor'
            args['after_store_id'] = cursor
        else:
            args['record_cursor'] = None
        args.update(footer_cursor if footer_cursor is not None else dict(withheld_cursor=None))
        page = call('lcm_load_session', args)
        pages.extend(expanded(row) if 'owner_store_id' in row else normalized(row)
                     for row in page.get('messages', []))
        footer = page.get('withheld', {})
        if footer:
            counts.append(footer['counts'])
            withheld.extend(expanded(row) for row in footer['locators'])
        cursor, footer_cursor = page.get('next_cursor'), footer.get('next_cursor')
        for value, visited in ((cursor, seen), (footer_cursor, footer_seen)):
            if value is not None:
                assert freeze(value) not in visited, 'pagination stalled'
                visited.add(freeze(value))
        if cursor is None and footer_cursor is None:
            break
    hits = []
    texts = {row['content'] for capture in engine._paired_trace.record_captures for row in capture['rows'][:-1]}
    for text in sorted(texts):
        if not text.strip():
            continue
        found = call('lcm_grep', dict(query='"' + text + '"', mode='full_text', limit=100))
        for row in found.get('results', []):
            reference = row.get('reference') or dict(tool='lcm_expand', args=dict(store_id=row['store_id']))
            result = normalized(call(reference['tool'], reference['args']))
            hits.append(dict(normalized(dict(row, content=row.get('content', result.get('content')))),
                             reference=reference, expanded=result))
    sources = [sid for (raw,) in engine._dag._conn.execute('SELECT source_ids FROM summary_nodes')
               for sid in json.loads(raw)]
    result = dict(pages=pages, hits=hits, summary_sources=sources, withheld=withheld, withheld_counts=counts)
    if hasattr(engine._paired_trace, 'baseline_stages'):
        current = dict(getattr(engine, '_queued_counts', {}))
        previous = getattr(engine, '_contract_refusal_previous', {})
        by_capture = engine._paired_trace.__dict__.setdefault('_contract_refusals', {})
        stage = engine._paired_trace.ingests[-1]
        fresh = [c for c in stage['captures'] if c['head_present'] and c['overwritten']
                 and not c['newest_present'] and c['ids'][0] in stage['committed']
                 and c['ids'][0] not in stage['committed_before']]
        delta = current.get('backward_overlap', 0) - previous.get('backward_overlap', 0)
        if fresh:
            assert len(fresh) == 1 or delta == 0, 'ambiguous refusal capture'
            for capture in fresh:
                by_capture[capture['view']] = dict(backward_overlap=delta)
        engine._contract_refusal_previous = current
        result['refused_captures'] = dict(by_capture)
    return result


def judge(observations, entries):
    """Semantic contract, including the accepted round-3 refusal amendment."""
    pages, failures = observations['pages'], []
    assert [(r['store_id'], r['role'], r['content']) for r in pages if r['kind'] == 'canonical'] \
        == observations['baseline_canonical_rows'], \
        'canonical listing changed'
    assert observations['summary_sources'] == observations['baseline_summary_sources'], 'summary sources changed'
    witness = observations.get('reload_witness', {})
    locators = observations['withheld']
    keys = [(r['owner_store_id'], r['ordinal']) for r in locators]
    assert len(keys) == len(set(keys)), 'duplicate withheld locator'
    counts = dict(overlap=0, unscanned=0)
    counts.update(Counter(r['state'] for r in locators))
    reported = observations['withheld_counts']
    if reported:
        assert all(c == reported[0] for c in reported), 'withheld counts disagree across pages'
        assert len(locators) == sum(reported[0].values()), 'withheld footer incomplete'
        assert counts == reported[0], 'withheld counts differ from locators'
    assert not locators or observations['withheld_counts'], 'withheld counts missing'
    withheld = dict(zip(keys, locators))
    eligible = {(e['owner_store_id'], e['predecessor_ordinal']) for e in entries
                if e['due_stage'] is not None and e['owner_store_id'] not in witness}
    assert set(withheld) <= eligible, 'withheld locator not active or superseded'
    declared = {(r['owner_store_id'], r['ordinal']) for r in map(expected_record, entries)}
    refused = {}
    if 'refused_captures' in observations:
        stored = {(r.get('owner_store_id'), r.get('ordinal')) for r in observations['listing_projection']}
        refused = {(e['owner_store_id'], e['predecessor_ordinal']): earlier_spans(
            expected_record(e), observations['baseline_canonical']) for e in entries
            if e['due_stage'] is not None and (e['owner_store_id'], e['predecessor_ordinal']) not in stored
            and earlier_spans(expected_record(e), observations['baseline_canonical'])}
        checked_views = {e['capture_view'] for e in entries
                         if (e['owner_store_id'], e['predecessor_ordinal']) in stored
                         or (e['owner_store_id'], e['predecessor_ordinal']) in refused}
        for view in checked_views:
            expected_count = int(any(e['capture_view'] == view and
                (e['owner_store_id'], e['predecessor_ordinal']) in refused for e in entries))
            assert observations['refused_captures'].get(view, {}).get('backward_overlap', 0) == expected_count, \
                'backward_overlap refusal count differs'
    for entry in entries:
        expected = expected_record(entry)
        key = (expected['owner_store_id'], expected['ordinal'])
        def exact(row, **overrides):
            return all(k in row and row[k] == value for k, value in dict(expected, **overrides).items())
        if key in refused:
            fields = ('store_id', 'session_id', 'conversation_id', 'role', 'content')
            expansions = ([hit['expanded'] for hit in observations['hits']] if expected['content'].strip()
                          else [row for row in pages if row['kind'] == 'canonical'])
            assert any(all(expanded.get(f) == prior.get(f) for f in fields)
                       for prior in refused[key] for expanded in expansions), 'refused text not retrievable'
            continue
        locator = withheld.get(key)
        # Main's empty physical projection retains its historical recall misses.
        if key in eligible and any((r.get('owner_store_id'), r.get('ordinal')) == key
                                   for r in observations['listing_projection']):
            inside, anywhere, beyond = overlap(expected, observations['baseline_canonical'])
            if locator:
                assert locator['state'] in ('overlap', 'unscanned'), 'unknown withheld state'
                if locator['state'] == 'overlap':
                    assert anywhere, 'withheld overlap not proven'
                else:
                    assert beyond and not inside and expected['content'].strip(), 'unscanned inside floor'
                assert not any((r.get('owner_store_id'), r.get('ordinal')) == key for r in pages), \
                    'withheld record listed'
                assert not any((r.get('owner_store_id'), r.get('ordinal')) == key
                               for hit in observations['hits'] for r in (hit, hit['expanded'])), \
                    'withheld record searchable'
                assert exact(locator['expanded']), 'withheld expansion differs'
                continue
            assert not anywhere, 'withheld locator missing'
        def matches(row, span=False):
            if span and witness.get(key[0]) is not None and row.get('store_id') == witness[key[0]]:
                parts = row.get('content', '').split('\n\n')
                return row.get('role') == 'user' and len(parts) > key[1] and parts[key[1]] == expected['content']
            return row['kind'] == 'record' and (row.get('owner_store_id'), row.get('ordinal')) == key \
                and row.get('role') == 'user' and row.get('content') == expected['content']
        def witness_matches(row):
            baseline = next((dict(r) for r in observations['baseline_canonical']
                             if dict(r)['store_id'] == witness.get(key[0])), {})
            return matches(row, span=True) and all(row.get(field) == baseline.get(field)
                for field in ('store_id', 'session_id', 'conversation_id', 'role', 'content'))
        found = [row for row in observations['hits'] if matches(row, span=True) and not (
            row.get('store_id') == witness.get(key[0]) and row['expanded']['kind'] == 'record'
            and row['expanded'].get('owner_store_id') == key[0] and row['expanded'].get('ordinal') != key[1]
            and (key[0], row['expanded'].get('ordinal')) in declared)]
        record_hits = [hit for hit in observations['hits']
                       if any(row['kind'] == 'record'
                              and (row.get('owner_store_id'), row.get('ordinal')) == key
                              for row in (hit, hit['expanded']))]
        canonical_hits = [hit for hit in observations['hits']
                          if witness.get(key[0]) is not None and hit.get('store_id') == witness[key[0]]
                          and hit['expanded']['kind'] == 'canonical']
        found.extend(hit for hit in canonical_hits if hit not in found and witness_matches(hit['expanded']))
        retrieved = (any(exact(r['expanded']) for r in pages if matches(r) and 'expanded' in r)
                     if not expected['content'].strip() else bool(found) and all(
                         exact(r['expanded'], superseded_by=witness.get(key[0]))
                         if r['expanded']['kind'] == 'record' else witness_matches(r['expanded']) for r in found))
        retrieved = retrieved and all(
            (r['expanded'].get('owner_store_id'), r['expanded'].get('ordinal')) == key
            and exact(r['expanded'], superseded_by=witness.get(key[0]))
            if r['expanded']['kind'] == 'record' else witness_matches(r['expanded']) for r in record_hits)
        retrieved = retrieved and all(witness_matches(hit['expanded']) for hit in canonical_hits)
        listing_exact = all(exact(r['expanded']) for r in pages if matches(r))
        if sum(matches(row, span=True) for row in pages) != 1 or not retrieved or not listing_exact:
            failures.append((entry['capture_view'], entry['predecessor_ordinal'], expected['content']))
    ordered = [(r['owner_store_id'], r['ordinal']) for r in pages if r['kind'] == 'record']
    assert ordered == sorted(ordered), 'record listing owner order'
    for index, row in enumerate(pages):
        if row['kind'] == 'record':
            following = next((r for r in pages[index + 1:] if r['kind'] == 'canonical'), {})
            preceding = next((r for r in reversed(pages[:index]) if r['kind'] == 'canonical'), {})
            assert following.get('store_id') == row['owner_store_id'] \
                and preceding.get('store_id', -1) < row['owner_store_id'], 'record owner placement'
    def listed_records(rows):
        return [(index, row['owner_store_id'], row['ordinal'], row['role'], row['content'])
                for index, row in enumerate(rows) if 'owner_store_id' in row]
    listing = [r for r in observations['listing_projection']
               if (r.get('owner_store_id'), r.get('ordinal')) not in withheld]
    assert listed_records(pages) == listed_records(listing), \
        'listed records differ from projection'
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
