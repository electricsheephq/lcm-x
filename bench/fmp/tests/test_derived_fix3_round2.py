"""Final #1738 repair/deadline regressions; fabricated responses only."""
import json

import pytest

from bench.fmp.tests.test_derived import Fake, bundle, cache, cached, extract, item, run, score, window, wire


class RepairCache:
    def __init__(self, failure=None):
        self.messages = []
        self.failure = failure

    def call(self, spec, messages, calls, **kw):
        self.messages.append(messages)
        w = json.loads(messages[1]['content'])['windows'][0]
        if len(self.messages) == 1:
            items = [item(), item('bad') | dict(value_text=5, scope_key='evening')]
        else:
            if self.failure:
                raise self.failure('repair failed')
            items = [item() | dict(predicate_key='favorite_drink'),
                     item('bad') | dict(scope_key='evening')]
        return dict(content=json.dumps([wire(w, items)]))


def test_repair_only_replaces_rejected_local_items(tmp_path):
    transport = RepairCache()
    result = extract.execute([extract.source_view(bundle()['questions'][0])], transport, run.READER, tmp_path)
    assert len(result['items']) == 2
    assert {i['predicate_key'] for i in result['items']} == {'drink'}
    repair = json.loads(transport.messages[1][1]['content'])['repairs'][0]
    assert repair['window_id'] == '0'
    assert repair['items'][0]['local_id'] == 'bad'
    assert repair['items'][0]['item'] == 1
    assert 'only' in transport.messages[1][0]['content'].lower()
    assert result['ledger'][0]['extracted_count'] == 2
    assert result['ledger'][0]['rejected_count'] == 0


@pytest.mark.parametrize('failure', [cache.Excluded, cache.Deadline, cache.Incomplete, ValueError, TypeError, KeyError])
def test_failed_repair_keeps_admitted_sibling(tmp_path, failure):
    transport = RepairCache(failure)
    result = extract.execute([extract.source_view(bundle()['questions'][0])], transport, run.READER, tmp_path)
    assert len(result['items']) == 1
    assert result['items'][0]['predicate_key'] == 'drink'
    if failure in (cache.Excluded, cache.Deadline):
        assert cache.snapshot(tmp_path/'extraction.json')['items'] == result['items']


def test_paused_repair_resumes_with_admitted_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.time, 'sleep', lambda seconds: None)
    class PauseThenExclude(Fake):
        def call(self, *args, **kw):
            if self.calls:
                if not getattr(self, 'paused', False):
                    self.paused = True
                    raise RuntimeError('usage limit reached')
                raise ConnectionError('transport reset')
            response = super().call(*args, **kw)
            w = json.loads(response['content'])[0]
            w['items'].append(item('bad') | dict(value_text=5))
            response['content'] = json.dumps([w])
            return response
    backend = PauseThenExclude()
    view = extract.source_view(bundle()['questions'][0])
    c = cached(tmp_path/'responses', backend)
    c.deadline += 3600
    with pytest.raises(cache.Paused):
        extract.execute([view], c, run.READER, tmp_path)
    c = cached(tmp_path/'responses', backend)
    c.deadline += 3600
    result = extract.execute([view], c, run.READER, tmp_path)
    expected = extract.validate(wire(window(), [item()]), window())['items']
    assert result['items'] == expected
    assert cache.snapshot(tmp_path/'extraction.json')['items'] == expected
    assert backend.calls == 1, 'the successful first-pass response must be reused'


def test_main_deadline_keeps_drift_and_completed_prefix(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(cache.time, 'time', lambda: clock[0])
    b = bundle()
    b['questions'] = [b['questions'][0] | dict(qid=f'q{i}', question=f'Which drink {i}?') for i in range(4)]
    b['panel'] = [b['panel'][0] | dict(id=q['qid'], cluster=q['qid']) for q in b['questions']]
    b['baselines'] = {q['qid']: run.baseline_messages(run.datasets.Question(**q), 'HIT') for q in b['questions']}
    b['deadline'] = 1010.
    run.freeze(tmp_path, b)
    monkeypatch.setattr(run, 'inputs', lambda root: b | dict(synthetic=False))
    def completed_drift(*args):
        cache.snapshot(tmp_path/'drift-admission.json', dict(valid=True, exclusions=[]))
        return True
    monkeypatch.setattr(run, 'drift', completed_drift)
    class Expiring(Fake):
        judges = 0
        def call(self, *args, **kw):
            response = super().call(*args, **kw)
            if kw['max_tokens'] == 32000:
                response['content'] = f'tea answer {self.calls}'
            if kw['max_tokens'] == 4096:
                self.judges += 1
                if self.judges == 9:  # Two complete items, then one partial item.
                    clock[0] = 1020.
            return response
    report = run.execute(tmp_path, Expiring())
    evaluation = cache.snapshot(tmp_path/'evaluation-inputs.json')
    assert evaluation['parameters']['drift_valid'] is True
    assert len(evaluation['entries']) == 2
    rows = cache.snapshot(tmp_path/'rows.json')
    assert {r['id'] for r in rows} == {'q0', 'q1'}
    assert report['judges'] and all(t['overall']['n'] == 2 for t in report['judges'].values())
    assert score.evaluate(evaluation['entries'], rows, **evaluation['parameters']) == report
    run.card(tmp_path)
    assert 'Panel length: 4; evaluated prefix length: 2' in (tmp_path/'RESULT.md').read_text()
    assert run.execute(tmp_path, Fake()) == report
