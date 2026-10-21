"""Seven #1730 regressions; fabricated inputs and fake transports only."""
import json
from dataclasses import asdict

import pytest

from bench.fmp.tests.test_derived import (
    Fake, bundle, cache, cached, delivery, encoded, extract, item, qid, request_identity,
    run, score, sha, window, wire,
)


def repeats():
    b = bundle()
    qs = [qid(f'r{i}') for i in range(44)]
    for q in qs:
        q.question += q.qid
    b.update(questions=[asdict(q) for q in qs], repeat_ids=[q.qid for q in qs], controls={},
             panel=[b['panel'][0] | dict(id=f'r{i}') for i in range(886)],
             baselines={q.qid: run.baseline_messages(q, 'HIT') for q in qs})
    for q in qs:
        a = b['baselines'][q.qid]
        identity = request_identity(run.READER, a, endpoint={}, effort='medium', temperature=None,
            max_tokens=32000, tools=None, tool_choice='none', rubric='fmp-single', renderer={}, sources={})
        response = dict(content='historical')
        b['controls'][q.qid] = dict(identity=identity, status='ok', response=response,
                                  response_sha256=sha(encoded(response)))
    return b


@pytest.mark.parametrize('phase', ['drift', 'main'])
def test_1_deadline_settles_and_scores(tmp_path, monkeypatch, phase):
    clock = [1000.]
    monkeypatch.setattr(cache.time, 'time', lambda: clock[0])
    b = repeats() if phase == 'drift' else bundle()
    b['deadline'] = 1010.
    # The synthetic freeze avoids live provenance; execute still exercises real drift.
    if phase == 'drift':
        b['panel'] = b['panel'][:44]
    run.freeze(tmp_path, b)
    if phase == 'drift':
        monkeypatch.setattr(run, 'inputs', lambda root: b | dict(synthetic=False))
    class Expiring(Fake):
        def call(self, *args, **kw):
            response = super().call(*args, **kw)
            if kw['max_tokens'] == 4096:
                clock[0] = 1020.
            return response
    backend = Expiring()
    report = run.execute(tmp_path, backend)
    assert report['reasons'] == ['deadline expired']
    evaluation = cache.snapshot(tmp_path/'evaluation-inputs.json')
    rows = cache.snapshot(tmp_path/'rows.json')
    assert score.evaluate(evaluation['entries'], rows, **evaluation['parameters']) == report
    assert cache.snapshot(tmp_path/'report.json') == report
    calls = backend.calls
    assert run.execute(tmp_path, backend) == report and backend.calls == calls
    monkeypatch.setattr('sys.argv', ['derived', 'score', '--product', 'unused', '--source',
        str(tmp_path/'inputs.json'), '--run', str(tmp_path)])
    assert run.main() == 0 and (tmp_path/'RESULT-MANIFEST.json').exists()


def test_2_cross_chunk_quote_is_extractable():
    value = asdict(qid())
    limit = 7000*4-1024
    quote = 'I prefer tea with a slice of lemon'
    value['sessions'][0]['turns'][0]['content'] = 'λ'*(limit-8) + quote + '尾'*9000
    view = extract.source_view(value)
    windows = extract.windows(view, source_tokens=7000, context_tokens=100)
    assert windows == extract.windows(view, source_tokens=7000, context_tokens=100)
    assert all(len(r['content']) <= limit for w in windows for r in w['rows'])
    admitted = [i for w in windows for i in extract.validate(wire(w, [item(quote=quote)]), w)['items']]
    assert admitted, 'quote crossing the old hard boundary must survive admission'
    assert {i['sources'][0]['start'] for i in admitted} == {limit-8}


def test_3_typed_items_reject_locally_and_repair(tmp_path):
    w = window()
    fields = ['local_id', 'kind', 'subject_key', 'subject_resolution', 'predicate_key', 'object_value',
              'value_text', 'polarity', 'scope_key', 'event_at', 'valid_from', 'valid_to', 'explicitness']
    for field in fields:
        for bad in ([], {}, 17, True):
            got = extract.validate(wire(w, [item(), item('bad') | {field: bad}]), w)
            assert len(got['items']) == 1 and len(got['rejected']) == 1, (field, bad)
            assert delivery.render(got['items'], [], 'drink')[0]
    class Repair(Fake):
        def call(self, *args, **kw):
            response = super().call(*args, **kw)
            if self.calls == 1:
                data = json.loads(response['content'])
                data[0]['items'] += [item('bad') | dict(predicate_key=[])]
                response['content'] = json.dumps(data)
            return response
    backend = Repair()
    result = extract.execute([extract.source_view(asdict(qid()))], cached(tmp_path/'responses', backend), run.READER, tmp_path)
    assert backend.calls == 2, 'malformed typed item must receive the existing repair attempt'
    assert len(result['items']) == 1 and result['ledger'][0]['state'] == 'partial'
    # The repair omits 'bad'; its unreplaced rejection must remain visible to the 5% cap.
    assert result['ledger'][0]['rejected_count'] == 1
    assert delivery.render(result['items'], [], 'drink')[0]


def test_4_drift_exclusions_use_repeat_population(tmp_path, monkeypatch):
    b = repeats()
    assert run.drift(b, cached(tmp_path/'responses'), tmp_path, [dict(id='outside')]*50)
    other = tmp_path/'two-lost'
    class LostRepeats(Fake):
        def call(self, spec, messages, *args, **kw):
            if messages in (b['baselines']['r0'], b['baselines']['r1']):
                raise ConnectionError('transport reset')
            return super().call(spec, messages, *args, **kw)
    monkeypatch.setattr(cache.time, 'sleep', lambda seconds: None)
    c = cached(other/'responses', LostRepeats())
    c.deadline += 3600
    assert not run.drift(b, c, other), \
        'two lost repeats exceed 3% of 44 even with an 886-item panel'
    exclusions = [dict(id='r0')]
    assert not run.drift(b, c, other, exclusions)
    assert [e['id'] for e in exclusions] == ['r0', 'r1']


def test_5_drift_checkpoint_crash_resumes(tmp_path, monkeypatch):
    b = repeats()
    backend = Fake()
    original = run.snapshot
    def crash(path, value=None):
        if path.name == 'drift-admission.json' and value is not None:
            raise KeyboardInterrupt('crash between drift checkpoints')
        return original(path, value)
    monkeypatch.setattr(run, 'snapshot', crash)
    with pytest.raises(KeyboardInterrupt):
        run.drift(b, cached(tmp_path/'responses', backend), tmp_path)
    assert (tmp_path/'drift.json').exists() and not (tmp_path/'drift-admission.json').exists()
    sends = backend.calls
    monkeypatch.setattr(run, 'snapshot', original)
    assert run.drift(b, cached(tmp_path/'responses', backend), tmp_path)
    assert backend.calls == sends
    assert run.drift(b, cached(tmp_path/'responses', backend), tmp_path)


def test_6_extraction_exclusions_settle_before_calls(tmp_path, monkeypatch):
    b = bundle()
    b['questions'][1]['sessions'][0]['turns'][0]['content'] = 'I prefer coffee'
    run.freeze(tmp_path, b)
    w = window()
    good = extract.validate(wire(w, [item()]), w)['items']
    lost = extract.source_view(b['questions'][1])['haystack_hash']
    cache.snapshot(tmp_path/'extraction.json', dict(items=good, relations=[], ledger=[
        dict(source=w['haystack_hash'], state='succeeded', extracted_count=1),
        dict(source=lost, state='excluded')]))
    monkeypatch.setattr(run, 'inputs', lambda root: b | dict(synthetic=False))
    drift_calls = []
    monkeypatch.setattr(run, 'drift', lambda *args: drift_calls.append(args) or True)
    backend = Fake()
    report = run.execute(tmp_path, backend)
    assert not drift_calls and backend.calls == 0, 'excess extraction exclusions must spend zero calls'
    assert report['verdict'] == 'PARK-INCOMPLETE' and report['exclusions'] == 1
    assert cache.snapshot(tmp_path/'rows.json') == []
    assert run.execute(tmp_path, backend) == report and backend.calls == 0


@pytest.mark.parametrize('interim', [False, True])
def test_7_card_uses_each_scored_denominator(tmp_path, monkeypatch, interim):
    b = bundle()
    entries = [b['panel'][0] | dict(id=str(i), cluster=str(i)) for i in range(100)]
    b['panel'] = entries
    b['baselines'] = {e['id']: b['baselines']['q1'] for e in entries}
    b['questions'] = [b['questions'][0] | dict(qid=e['id']) for e in entries]
    run.freeze(tmp_path, b)
    scored = entries[:80] if interim else entries
    rows = [dict(id=e['id'], arm=a, judge=j, correct=True)
            for e in scored for a in run.ARMS for j in run.JUDGES]
    rows[-1]['correct'] = None
    parameters = dict(complete=not interim, exclusions=[dict(id='0')])
    report = score.evaluate(scored, rows, **parameters)
    assert report['judges']
    if interim:
        assert report['verdict'] == 'PARK-FUTILITY'
    cache.snapshot(tmp_path/'evaluation-inputs.json', dict(entries=scored, parameters=parameters))
    cache.snapshot(tmp_path/'rows.json', rows)
    cache.snapshot(tmp_path/'report.json', report)
    cache.snapshot(tmp_path/'extraction.json', {})
    (tmp_path/'responses').mkdir()
    monkeypatch.setattr('sys.argv', ['derived', 'score', '--product', 'unused', '--source',
        str(tmp_path/'inputs.json'), '--run', str(tmp_path)])
    assert run.main() == 0
    card = json.loads((tmp_path/'RESULT-MANIFEST.json').read_text())
    assert {h['n'] for h in card['headline']} == {t['overall']['n'] for t in report['judges'].values()}
    assert len(card['headline']) == len(run.JUDGES)
    assert 'Panel length: 100' in (tmp_path/'RESULT.md').read_text()


def extraction_panel(tmp_path, monkeypatch, n_other=6, excluded=('x0',)):
    """Reviewer fixture: 44 repeats plus a tail, with independent source histories."""
    b = repeats()
    qs = [qid(f'r{i}') for i in range(44)] + [qid(f'x{i}') for i in range(n_other)]
    for q in qs:
        q.question += q.qid
        q.sessions[0]['turns'][0]['content'] += ' ' + q.qid
    b.update(questions=[asdict(q) for q in qs],
        panel=[b['panel'][0] | dict(id=q.qid, cluster=q.qid, target=q.qid.startswith('r')) for q in qs],
        baselines={q.qid: run.baseline_messages(q, 'HIT') for q in qs},
        w1_endpoint={}, w1_renderer={}, w1_sources={f'r{i}': {} for i in range(44)})
    for q in qs[:44]:
        b['controls'][q.qid]['identity'] = request_identity(run.READER, b['baselines'][q.qid],
            endpoint={}, effort='medium', temperature=None, max_tokens=32000, tools=None,
            tool_choice='none', rubric='fmp-single', renderer={}, sources={})
    run.freeze(tmp_path, b)
    good = extract.validate(wire(window(), [item()]), window())['items']
    cache.snapshot(tmp_path/'extraction.json', dict(items=good, relations=[], ledger=[
        dict(source=extract.source_view(q)['haystack_hash'],
             state='excluded' if q['qid'] in excluded else 'succeeded', extracted_count=1)
        for q in b['questions']]))
    monkeypatch.setattr(run, 'inputs', lambda root: b | dict(synthetic=False))
    return b


def test_fix_round1_resume_deduplicates_extraction_exclusions(tmp_path, monkeypatch):
    extraction_panel(tmp_path, monkeypatch)
    class QuotaAfterInterim(Fake):
        def call(self, *args, **kw):
            if (tmp_path/'interim.json').exists():
                raise RuntimeError('usage limit reached')
            return super().call(*args, **kw)
    assert run.execute(tmp_path, QuotaAfterInterim())['status'] == 'PAUSED'
    assert (tmp_path/'interim.json').exists()
    report = run.execute(tmp_path, Fake())
    assert (tmp_path/'report.json').exists(), report
    assert cache.snapshot(tmp_path/'report.json') == report
    ids = [e['id'] for e in report['exclusion_details']]
    assert ids == ['x0'] and len(ids) == len(set(ids))
    admission = cache.snapshot(tmp_path/'drift-admission.json')
    assert admission['exclusions'] == []


class BatchRepair(Fake):
    def __init__(self, repair):
        super().__init__()
        self.repair = repair
    def call(self, *args, **kw):
        response = super().call(*args, **kw)
        if kw['max_tokens'] == 16384:
            data = json.loads(response['content'])
            good = [item(str(i)) | dict(scope_key=str(i)) for i in range(40)]
            bad = item('bad') | dict(value_text='unquoted')
            data[0]['items'] = good + [bad] if self.calls == 1 else [bad]
            response['content'] = 'not json' if self.calls == 2 and self.repair == 'invalid' else json.dumps(data)
        return response


def test_fix_round1_invalid_repair_keeps_valid_items(tmp_path):
    run.freeze(tmp_path, bundle())
    report = run.execute(tmp_path, BatchRepair('invalid'))
    extracted = cache.snapshot(tmp_path/'extraction.json')
    assert len(extracted['items']) == 40
    assert 'terminal_reason' not in extracted
    assert extracted['ledger'][0]['state'] == 'partial'
    assert report['rejection_rate'] == pytest.approx(1/41)
    assert report['judges'], report


def test_fix_round1_repair_rejection_denominator(tmp_path):
    run.freeze(tmp_path, bundle())
    report = run.execute(tmp_path, BatchRepair('still-bad'))
    extracted = cache.snapshot(tmp_path/'extraction.json')
    assert len(extracted['items']) == 40
    ledger = extracted['ledger'][0]
    assert ledger['extracted_count'] == 41 and ledger['rejected_count'] == 1
    assert report['rejection_rate'] == pytest.approx(1/41)
    assert report['judges'], report


def test_fix_round1_extraction_lost_repeats_skip_drift(tmp_path, monkeypatch):
    b = extraction_panel(tmp_path, monkeypatch, n_other=56, excluded=('r0', 'r1'))
    backend = Fake()
    report = run.execute(tmp_path, backend)
    lost = {encoded(b['baselines'][i]) for i in ('r0', 'r1')}
    pending = [json.loads(p.read_bytes()) for p in (tmp_path/'responses').glob('*.pending.json')]
    drift_calls = [p for p in pending if 'repeat' in p['identity']['retrieved_sources']]
    assert all(p['identity']['retrieved_sources']['repeat'] not in ('r0', 'r1') for p in drift_calls)
    assert not any(encoded(p['request']) in lost for p in pending)
    admission = cache.snapshot(tmp_path/'drift-admission.json')
    assert admission['valid'] is True and admission['extraction_skipped'] == 2
    assert admission['exclusions'] == []
    assert report['exclusions'] == 2 and report['judges'], report
